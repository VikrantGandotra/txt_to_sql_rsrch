"""Compile execution-verified trajectory and logical-validity demonstrations.

Compile explicitly with ``python src/dspy_optimizer.py``. Importing this module
does not configure a model or make requests. The resulting JSON contains DSPy
predictor state and demonstrations, without an attached language-model client.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from time import perf_counter
from contextlib import closing, contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dspy

from src.bird_evaluator import _execute_result_set
from src.dspy_modules import COMPILED_MODEL_PATH, TextToSQLProgram, create_dspy_lm, provider_model_id
from src.dspy_telemetry import capture_lm_responses, record_stage_metrics, summarize_query_metrics
from src.dspy_verifier import (
    COMPILED_VERIFIER_PATH, VERIFIER_MANIFEST_KEY, VERIFIER_MODEL, VERIFIER_VERSION,
    TrajectoryVerifier, sanitize_trajectory,
)


TRAINING_SCHEMA = """CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER NOT NULL,
    city TEXT NOT NULL
);"""


def build_trainset(db_path: str | Path) -> list[dspy.Example]:
    """Create a new fixture database and four examples; refuse existing files.

    Only question and schema_ddl are model inputs. Gold SQL and the local
    database path are labels used exclusively by the execution metric.
    """
    fixture_path = Path(db_path).resolve()
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    with fixture_path.open("xb"):
        pass
    try:
        with closing(sqlite3.connect(fixture_path)) as connection:
            connection.execute(TRAINING_SCHEMA)
            connection.executemany(
                "INSERT INTO users (id, name, age, city) VALUES (?, ?, ?, ?)",
                [
                    (1, "Ada", 36, "London"),
                    (2, "Linus", 55, "Helsinki"),
                    (3, "Grace", 85, "New York"),
                    (4, "Edsger", 72, "Amsterdam"),
                ],
            )
            connection.commit()
    except BaseException:
        fixture_path.unlink(missing_ok=True)
        raise

    questions_and_sql = [
        (
            "What is the name of the oldest user?",
            "SELECT name FROM users ORDER BY age DESC LIMIT 1;",
        ),
        (
            "List the names of users older than 50.",
            "SELECT name FROM users WHERE age > 50;",
        ),
        (
            "How many users live in London?",
            "SELECT COUNT(*) FROM users WHERE city = 'London';",
        ),
        (
            "What is the average age of all users?",
            "SELECT AVG(age) FROM users;",
        ),
    ]
    return [
        dspy.Example(
            question=question,
            schema_ddl=TRAINING_SCHEMA,
            sql_query=sql,
            db_path=str(fixture_path),
        ).with_inputs("question", "schema_ddl")
        for question, sql in questions_and_sql
    ]


def execution_metric(example, prediction, trace=None) -> bool:
    """Accept only read-only predictions matching complete gold result sets."""
    try:
        db_path = Path(example.db_path).resolve()
        if getattr(prediction, "action", "final_answer") != "final_answer":
            return False
        predicted_rows = _execute_result_set(db_path, prediction.sql_query, 30.0)
        gold_rows = _execute_result_set(db_path, example.sql_query, 30.0)
        return predicted_rows == gold_rows
    except (AttributeError, sqlite3.Error, OSError, TypeError, ValueError):
        return False


def build_bird_trainset(train_json: str | Path, db_dir: str | Path, limit: int | None = None):
    """Load only a designated training split; gold SQL never becomes a predictor input."""
    from src.bird_evaluator import _database_path
    from src.graph_pruner import DatabaseGraphPruner

    path = Path(train_json)
    if any(part in path.stem.lower() for part in ("dev", "test", "valid")):
        raise ValueError("Compilation requires a training split, never dev/test/validation data")
    examples = []
    # BIRD's train.json can omit question_id; assign stable dataset positions.
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("BIRD training JSON must contain a list")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("Training limit must be a positive integer")
    rows = rows[:limit] if limit is not None else rows
    pruners = {}
    for index, row in enumerate(rows):
        db_path = _database_path(str(db_dir), row["db_id"])
        if db_path not in pruners:
            pruners[db_path] = DatabaseGraphPruner(str(db_path))
        question = row["question"]
        evidence = row.get("evidence", "") or ""
        _, ddl = pruners[db_path].prune(question, evidence)
        model_question = question + (f"\n\nEvidence: {evidence}" if evidence else "")
        examples.append(dspy.Example(
            question=model_question, schema_ddl=ddl, sql_query=row["SQL"],
            db_path=str(db_path), question_id=str(row.get("question_id", index)),
            raw_question=question, evidence=evidence,
        ).with_inputs("question", "schema_ddl"))
    if not examples:
        raise ValueError("Training split is empty")
    return examples


class TrajectoryTrainingProgram(TextToSQLProgram):
    """Run the same five-iteration graph during compilation as during inference.

    Database lookup is hidden module state, not an input field, prompt or demo.
    BootstrapFewShot captures calls to the inherited two predictors, including
    successful probes and the subsequent observations supplied to the generator.
    """

    def __init__(self, trainset):
        super().__init__()
        self._training_databases = {
            (item.question, item.schema_ddl): item.db_path for item in trainset
        }

    def forward(self, question: str, schema_ddl: str):
        from src.sql_agent_graph import SQLAgentGraph

        agent = SQLAgentGraph(dspy_program=self)
        final = agent.run(
            question=question, evidence="", schema_ddl=schema_ddl,
            db_path=self._training_databases[(question, schema_ddl)], max_turns=5,
        )
        return dspy.Prediction(
            logical_plan=final.get("plan"), action=final.get("current_action"),
            sql_query=final.get("final_sql"), history=final.get("history", []),
        )


def _save_compiled(compiled, destination: Path, manifest_key: str, manifest: dict):
    """Atomic credential-free predictor state and provenance manifest."""
    compiled.set_lm(None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.stem}.", suffix=".json", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        compiled.save(str(temporary_path), save_program=False)
        state = json.loads(temporary_path.read_text(encoding="utf-8"))
        state[manifest_key] = manifest
        temporary_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _destination(output_path):
    destination = Path(output_path).resolve()
    if destination.suffix.lower() != ".json":
        raise ValueError("Compiled DSPy state must be saved to a .json file")
    if destination.is_dir():
        raise IsADirectoryError(destination)
    return destination


def _protect_inputs(destination, *input_paths):
    outputs = {destination, destination.with_suffix(".compilation.metrics.json")}
    if any(Path(path).resolve() in outputs for path in input_paths if path is not None):
        raise ValueError("Compilation outputs must not overwrite training inputs")


@contextmanager
def _compilation_accounting(destination, stage_name):
    """Keep paid calls measurable even when BootstrapFewShot fails."""
    started = perf_counter()
    failure = None
    with capture_lm_responses(include_nested=True) as responses:
        try:
            yield responses
            for record in responses:
                raw = record.get("response")
                actual = raw.get("model") if isinstance(raw, dict) else getattr(raw, "model", None)
                if provider_model_id(str(actual)) != provider_model_id(VERIFIER_MODEL):
                    raise ValueError(f"Compilation received a different or unspecified provider model: {actual}")
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            raise
        finally:
            elapsed = perf_counter() - started
            row = record_stage_metrics(stage_name, 0, responses, elapsed, failure)
            metrics = summarize_query_metrics([row], turn=0, latency_seconds=elapsed)
            metrics["error"] = failure
            report_path = destination.with_suffix(".compilation.metrics.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _provenance(trainset, training_source: str):
    # Hash labels and schemas; do not serialize gold SQL/DB paths into metadata.
    digest = hashlib.sha256(json.dumps([
        {"question": row.question, "schema_ddl": row.schema_ddl, "gold": row.sql_query}
        for row in trainset
    ], sort_keys=True).encode("utf-8")).hexdigest()
    return {"training_source": training_source, "training_examples": len(trainset),
            "training_sha256": digest, "model_name": VERIFIER_MODEL}


def compile_program(
    output_path: str | Path = COMPILED_MODEL_PATH,
    *,
    lm=None,
    train_json: str | Path | None = None,
    db_dir: str | Path | None = None,
    limit: int | None = None,
) -> TextToSQLProgram:
    """Compile and atomically save predictor state, returning the program.

    Pass an injected DSPy LM for offline testing. Otherwise the configured live
    model is used. A failed compile preserves any existing output artifact.
    """
    destination = _destination(output_path)
    _protect_inputs(destination, train_json)
    if (train_json is None) != (db_dir is None):
        raise ValueError("Supply both train_json and db_dir")
    active_lm = lm if lm is not None else create_dspy_lm(formal=True)
    started = perf_counter()
    with TemporaryDirectory(prefix="mars_dspy_training_") as directory:
        trainset = (build_bird_trainset(train_json, db_dir, limit) if train_json is not None
                    else build_trainset(Path(directory) / "training.sqlite"))
        optimizer = dspy.BootstrapFewShot(
            metric=execution_metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=0,
            max_rounds=1,
            max_errors=4,
        )
        with _compilation_accounting(destination, "compilation") as responses, dspy.context(
            lm=active_lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False),
        ):
            compiled = optimizer.compile(TrajectoryTrainingProgram(trainset), trainset=trainset)

    if not compiled.decomposer.demos or not compiled.generator.demos:
        raise RuntimeError(
            "Compilation produced no execution-verified demonstrations for both "
            "predictors; no compiled model was saved."
        )
    elapsed = perf_counter() - started
    stage = record_stage_metrics("compilation", 0, responses, elapsed)
    manifest = {**_provenance(trainset, str(train_json) if train_json else "synthetic_smoke_only"),
                "version": 2, "action_contract": "probe_final_answer", "max_iterations": 5,
                "compilation_metrics": summarize_query_metrics([stage], turn=0, latency_seconds=elapsed)}
    _save_compiled(compiled, destination, "mars_generator_manifest", manifest)
    return compiled


def verifier_metric(example, prediction, trace=None) -> bool:
    """EX-derived training labels stay outside the verifier's input signature."""
    try:
        score = float(prediction.score)
        return 0 <= score <= 1 and (score >= 0.5) == bool(example.correct)
    except (TypeError, ValueError, AttributeError):
        return False


def build_verifier_trainset(trainset, training_trajectories=None):
    """Label training trajectories by EX without embedding labels/reference SQL.

    Real BIRD compilation requires trajectories generated on that training split.
    The synthetic fixture alone supplies known SQL for a compiler smoke test.
    """
    examples = []
    if training_trajectories is None:
        records = [
            (item, {"final_sql": sql, "history": []})
            for item in trainset
            for sql in (item.sql_query, "SELECT NULL;")
        ]
    else:
        rows = json.loads(Path(training_trajectories).read_text(encoding="utf-8"))
        if isinstance(rows, dict) and isinstance(rows.get("queries"), list):
            rows = [
                {**candidate, "question_id": query["question_id"]}
                for query in rows["queries"] for candidate in query.get("candidates", [])
            ]
        if not isinstance(rows, list):
            raise ValueError("Training trajectories must be a JSON list")
        by_id = {str(item.question_id): item for item in trainset}
        records = []
        for row in rows:
            question_id = str(row["question_id"])
            if question_id not in by_id:
                raise ValueError(f"Trajectory question_id {question_id} is outside the training split")
            records.append((by_id[question_id], row))
    for example, candidate in records:
        sql = candidate.get("final_sql")
        if not isinstance(sql, str) or not sql.strip():
            continue
        sanitized = sanitize_trajectory(candidate)
        try:
            predicted = _execute_result_set(Path(example.db_path), sql, 30.0)
            gold = _execute_result_set(Path(example.db_path), example.sql_query, 30.0)
            correct = predicted == gold
            # Recompute final observation rather than trusting training-file labels.
            sample_rows = [[value.hex() if isinstance(value, bytes) else value for value in row]
                           for row in sorted(predicted, key=repr)[:50]]
            sanitized["execution"] = {"success": True, "row_count": len(predicted), "sample_rows": sample_rows}
        except (sqlite3.Error, OSError, TypeError, ValueError):
            # Runtime selection likewise considers only executable final answers.
            continue
        examples.append(dspy.Example(
            question=getattr(example, "raw_question", example.question),
            evidence=getattr(example, "evidence", ""), schema_ddl=example.schema_ddl,
            trajectory=json.dumps(sanitized, ensure_ascii=False), correct=correct,
        ).with_inputs("question", "evidence", "schema_ddl", "trajectory"))
    if {bool(item.correct) for item in examples} != {True, False}:
        raise ValueError("Verifier training requires both correct and incorrect executable trajectories")
    return examples


def compile_verifier(
    output_path=COMPILED_VERIFIER_PATH, *, lm=None, train_json=None, db_dir=None,
    training_trajectories=None, limit=None,
):
    destination = _destination(output_path)
    _protect_inputs(destination, train_json, training_trajectories)
    if (train_json is None) != (db_dir is None):
        raise ValueError("Supply both train_json and db_dir")
    if train_json is not None and training_trajectories is None:
        raise ValueError("Real BIRD verifier compilation requires --training-trajectories from the training split")
    active_lm = lm if lm is not None else create_dspy_lm(formal=True)
    started = perf_counter()
    with TemporaryDirectory(prefix="mars_dspy_verifier_training_") as directory:
        trainset = (build_bird_trainset(train_json, db_dir, limit) if train_json is not None
                    else build_trainset(Path(directory) / "training.sqlite"))
        labeled = build_verifier_trainset(trainset, training_trajectories)
        optimizer = dspy.BootstrapFewShot(metric=verifier_metric, max_bootstrapped_demos=8,
                                         max_labeled_demos=0, max_rounds=1, max_errors=4)
        with _compilation_accounting(destination, "verifier_compilation") as responses, dspy.context(
            lm=active_lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False),
        ):
            compiled = optimizer.compile(TrajectoryVerifier(), trainset=labeled)
    if not compiled.judge.demos or {float(demo.score) >= 0.5 for demo in compiled.judge.demos} != {True, False}:
        raise RuntimeError("Verifier compilation must learn demonstrations of both correct and incorrect trajectories")
    elapsed = perf_counter() - started
    stage = record_stage_metrics("verifier_compilation", 0, responses, elapsed)
    manifest = {**_provenance(trainset, str(train_json) if train_json else "synthetic_smoke_only"),
                "version": VERIFIER_VERSION, "score_kind": "elicited_validity_rating",
                "compilation_metrics": summarize_query_metrics([stage], turn=0, latency_seconds=elapsed)}
    _save_compiled(compiled, destination, VERIFIER_MANIFEST_KEY, manifest)
    return compiled


def collect_training_trajectories(
    train_json, db_dir, output_path=None, *, limit=32, num_candidates=8,
    max_workers=8, prompt_mode="compiled_dspy", compiled_model_path=None,
):
    """Generate and retain training rollouts with stable IDs for verifier labels.

    Gold SQL is used by the batch scorer after generation only. Formal selection
    remains self-consistency during collection; no verifier is trained yet.
    """
    from src.bird_evaluator import run_bird_evaluation

    source = Path(train_json).resolve()
    if any(part in source.stem.lower() for part in ("dev", "test", "valid")):
        raise ValueError("Trajectory collection requires a training split")
    destination = _destination(output_path or COMPILED_MODEL_PATH.parent / "artifacts" / "training" / "predict.json")
    protected = {source, COMPILED_MODEL_PATH.resolve(), COMPILED_VERIFIER_PATH.resolve()}
    if compiled_model_path is not None:
        protected.add(Path(compiled_model_path).resolve())
    if {destination, destination.with_suffix(".metrics.json")} & protected:
        raise ValueError("Training trajectory outputs must not overwrite training data or compiled artifacts")
    raw = source.read_bytes()
    rows = json.loads(raw)
    if not isinstance(rows, list):
        raise ValueError("BIRD training JSON must contain a list")
    normalized = [{**row, "question_id": row.get("question_id", index)} for index, row in enumerate(rows)]
    with TemporaryDirectory(prefix="mars_training_split_") as directory:
        normalized_path = Path(directory) / "normalized_train.json"
        normalized_path.write_text(json.dumps(normalized, ensure_ascii=False), encoding="utf-8")
        run_bird_evaluation(
            str(normalized_path), str(db_dir), str(destination), limit,
            num_candidates=num_candidates, max_workers=max_workers,
            selection_mode="self_consistency", prompt_mode=prompt_mode,
            compiled_model_path=str(compiled_model_path) if compiled_model_path else None,
            temperature=0.8, top_p=0.7, max_turns=5, execution_timeout=30.0,
        )
    report_path = destination.with_suffix(".metrics.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.setdefault("manifest", {}).update(
        training_source=str(source), training_source_sha256=hashlib.sha256(raw).hexdigest(),
        split_role="training_only", normalized_question_ids=True,
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generator", "collect", "verifier", "both"), default="generator")
    parser.add_argument("--train-json", type=Path, help="BIRD training split only; omission uses the synthetic smoke fixture")
    parser.add_argument("--db-dir", type=Path, help="BIRD training database directory")
    parser.add_argument("--training-trajectories", type=Path, help="List of training question_id/final_sql/history records for verifier compilation")
    parser.add_argument("--limit", type=int, help="Bound training examples")
    parser.add_argument("--candidates", type=int, default=8, help="Training collection rollouts per question")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prompt-mode", choices=("compiled_dspy", "uncompiled_dspy"), default="compiled_dspy")
    parser.add_argument("--compiled-model", type=Path, default=None)
    parser.add_argument("--verifier-output", type=Path, default=COMPILED_VERIFIER_PATH)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Generator artifact destination, or prediction JSON destination in collect mode",
    )
    args = parser.parse_args()
    if args.mode == "collect":
        if args.train_json is None or args.db_dir is None:
            parser.error("collect mode requires --train-json and --db-dir")
        report = collect_training_trajectories(
            args.train_json, args.db_dir, args.output, limit=args.limit if args.limit is not None else 32,
            num_candidates=args.candidates, max_workers=args.workers,
            prompt_mode=args.prompt_mode, compiled_model_path=args.compiled_model,
        )
        print(f"Saved training trajectories for verifier compilation: {report}")
        return
    if args.train_json is None:
        print("Using four synthetic questions for a compiler smoke test; this is not BIRD training.")
    if args.mode in ("generator", "both"):
        output = args.output or COMPILED_MODEL_PATH
        compiled = compile_program(output, train_json=args.train_json, db_dir=args.db_dir, limit=args.limit)
        print(f"Saved generator to {output}: {len(compiled.decomposer.demos)} plan demos, {len(compiled.generator.demos)} action demos.")
    if args.mode in ("verifier", "both"):
        verifier = compile_verifier(args.verifier_output, train_json=args.train_json, db_dir=args.db_dir,
                                    training_trajectories=args.training_trajectories, limit=args.limit)
        print(f"Saved logical-validity verifier to {args.verifier_output}: {len(verifier.judge.demos)} demos.")


if __name__ == "__main__":
    main()
