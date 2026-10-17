"""Live Best-of-8 probe/observe verification using BIRD Mini-Dev question 28.

The question and gold SQL are from the BIRD Team's SQLite Mini-Dev split:
https://huggingface.co/datasets/birdsql/bird_mini_dev/blob/main/data/mini_dev_sqlite-00000-of-00001.json
The small SQLite database below is a synthetic fixture, not BIRD's database.
This checks architecture, not the published benchmark's execution accuracy.
"""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from src.bird_evaluator import run_bird_evaluation
from src.sql_agent_graph import build_sql_graph


QUESTION = {
    "question_id": 28,
    "db_id": "california_schools",
    "question": (
        "Consider the average difference between K-12 enrollment and 15-17 "
        "enrollment of schools that are locally funded, list the names and DOC "
        "type of schools which has a difference above this average."
    ),
    "evidence": (
        "Difference between K-12 enrollment and 15-17 enrollment can be "
        "computed by `Enrollment (K-12)` - `Enrollment (Ages 5-17)`"
    ),
    "SQL": (
        "SELECT T2.School, T2.DOC FROM frpm AS T1 INNER JOIN schools AS T2 "
        "ON T1.CDSCode = T2.CDSCode WHERE T2.FundingType = 'Locally funded' "
        "AND (T1.`Enrollment (K-12)` - T1.`Enrollment (Ages 5-17)`) > "
        "(SELECT AVG(T3.`Enrollment (K-12)` - T3.`Enrollment (Ages 5-17)`) "
        "FROM frpm AS T3 INNER JOIN schools AS T4 ON T3.CDSCode = T4.CDSCode "
        "WHERE T4.FundingType = 'Locally funded')"
    ),
}


def _make_fixture(root: Path) -> tuple[Path, Path]:
    db_root = root / "dev_databases"
    db_dir = db_root / QUESTION["db_id"]
    db_dir.mkdir(parents=True)
    with closing(sqlite3.connect(db_dir / "california_schools.sqlite")) as db:
        db.executescript('''
            CREATE TABLE schools (
                CDSCode TEXT PRIMARY KEY, School TEXT NOT NULL,
                DOC INTEGER NOT NULL, FundingType TEXT NOT NULL
            );
            CREATE TABLE frpm (
                CDSCode TEXT PRIMARY KEY,
                "Enrollment (K-12)" INTEGER NOT NULL,
                "Enrollment (Ages 5-17)" INTEGER NOT NULL,
                FOREIGN KEY (CDSCode) REFERENCES schools(CDSCode)
            );
            INSERT INTO schools VALUES
                ('A', 'Oak', 52, 'Locally funded'),
                ('B', 'Pine', 54, 'Locally funded'),
                ('C', 'Cedar', 54, 'Locally funded'),
                ('D', 'Elm', 31, 'State funded');
            INSERT INTO frpm VALUES
                ('A', 100, 20), ('B', 120, 20),
                ('C', 200, 20), ('D', 500, 20);
        ''')
        db.commit()
    dev_json = root / "dev.json"
    dev_json.write_text(json.dumps([QUESTION], indent=2), encoding="utf-8")
    return dev_json, db_root


def _scripted_graphs():
    """Replace only model transport; execute the real graph, SQLite, and voter."""
    created = 0

    def factory(**configuration):
        nonlocal created
        index = created
        created += 1
        calls = 0

        def create(**request):
            nonlocal calls
            calls += 1
            if calls == 1:
                payload = {
                    "target_tables": ["schools", "frpm"],
                    "target_columns": ["School", "DOC", "Enrollment (K-12)",
                                       "Enrollment (Ages 5-17)"],
                    "join_logic": "schools.CDSCode = frpm.CDSCode",
                    "filter_logic": "schools.FundingType = 'Locally funded'",
                    "reasoning": "Compute the local-school average before selecting schools above it.",
                }
            elif calls == 2:
                payload = {
                    "action": "probe", "rationale": "Observe the average local funding threshold.",
                    "sql_query": (
                        "SELECT AVG(f.`Enrollment (K-12)` - f.`Enrollment (Ages 5-17)`) "
                        "AS threshold FROM frpm f JOIN schools s ON f.CDSCode = s.CDSCode "
                        "WHERE s.FundingType = 'Locally funded'"
                    ),
                }
            elif calls == 3:
                if "120.0" not in json.dumps(request["messages"]):
                    raise AssertionError("The probe's exact SQLite observation did not reach the generator")
                payload = {
                    "action": "final_answer", "rationale": "Apply the observed threshold.",
                    "sql_query": QUESTION["SQL"] if index < 6 else (
                        "SELECT School, DOC FROM schools WHERE FundingType = 'Locally funded'"
                    ),
                }
            else:
                raise AssertionError("Unexpected model call")
            return SimpleNamespace(
                model="scripted-model", usage={"prompt_tokens": 10, "completion_tokens": 3,
                                                "total_tokens": 13},
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))],
            )

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return build_sql_graph(client=client, model_name="scripted-model", verbose=True)

    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-mode", choices=("uncompiled_dspy", "compiled_dspy", "baseline"),
                        default="uncompiled_dspy")
    parser.add_argument("--compiled-model", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline-scripted", action="store_true",
                        help="Prove graph/SQLite/voting mechanics with scripted model transport")
    args = parser.parse_args()
    artifact_dir = Path(__file__).resolve().parent / "artifacts" / f"architecture_{uuid4().hex}"
    artifact_dir.mkdir(parents=True)
    prediction = artifact_dir / "predict.json"
    print(f"BIRD Mini-Dev question 28: {QUESTION['question']}", flush=True)
    print("Fixture threshold: (80 + 100 + 180) / 3 = 120; expected row: ('Cedar', 54)", flush=True)
    print("Running eight scripted rollouts through the real graph and SQLite."
          if args.offline_scripted else
          "Running eight live OpenRouter rollouts; model behavior may or may not include probes.", flush=True)
    with TemporaryDirectory(prefix="mars_architecture_") as directory:
        dev_json, db_root = _make_fixture(Path(directory))
        def evaluate():
            run_bird_evaluation(
                str(dev_json), str(db_root), str(prediction),
                num_candidates=8, max_workers=8, max_turns=5,
                selection_mode="self_consistency", temperature=0.8, top_p=0.7,
                prompt_mode=args.prompt_mode, compiled_model_path=args.compiled_model,
                seed=args.seed, verbose=True,
            )
        if args.offline_scripted:
            with patch("src.bird_evaluator.build_sql_graph", side_effect=_scripted_graphs()):
                evaluate()
        else:
            evaluate()
    report = json.loads(prediction.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    query = report["queries"][0]
    probes = sum(step.get("action") == "probe" for candidate in query["candidates"]
                 for step in candidate["history"])
    print(f"Probe actions observed: {probes}", flush=True)
    print(f"EX on synthetic fixture: {report['execution_accuracy']:.2f}%", flush=True)
    print(f"Selected prediction: {json.loads(prediction.read_text(encoding='utf-8'))['28']}", flush=True)
    print(f"Full trajectory report: {prediction.with_suffix('.metrics.json')}", flush=True)
    if probes == 0:
        raise SystemExit("No exploratory probe occurred; this run did not demonstrate the multi-turn loop.")


if __name__ == "__main__":
    main()
