"""Generate BIRD predictions and score SQLite execution accuracy (EX).

Run from the project root, for example::

    python -m src.bird_evaluator --dev-json data/dev.json \
        --db-dir data/dev_databases --output predict.json --limit 10

Prediction generation uses the configured live model. Scoring only executes SQL.
EX follows BIRD's reference evaluator: equality of sets of complete result rows,
ignoring row order and duplicate multiplicity, but preserving column order.
Reference: https://github.com/bird-bench/mini_dev/blob/main/evaluation/evaluation_ex.py
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import logging
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any

from src.agent_state import AgentState
from src.evaluation_checkpoint import EvaluationCheckpoint, atomic_write_json
from src.graph_pruner import DatabaseGraphPruner
from src.sql_agent_graph import build_sql_graph

BIRD_SEPARATOR = "\t----- bird -----\t"
FORMAL_MODEL = "openrouter/qwen/qwen-2.5-7b-instruct"
logger = logging.getLogger(__name__)


def _load_items(dev_json_path: str, limit: int | None) -> list[dict[str, Any]]:
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
    ):
        raise ValueError("limit must be a non-negative integer or None")
    with open(dev_json_path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("BIRD dev.json must contain a list of question records")
    if limit is not None:
        data = data[:limit]
    question_ids = set()
    for item in data:
        if not isinstance(item, dict) or item.get("question_id") is None:
            raise ValueError("Each BIRD record must have a question_id")
        question_id = str(item["question_id"])
        if question_id in question_ids:
            raise ValueError(f"Duplicate question_id: {question_id}")
        question_ids.add(question_id)
    return data


def _database_path(db_dir: str, db_id: str) -> Path:
    if (
        not isinstance(db_id, str) or not db_id or db_id in {".", ".."}
        or "/" in db_id or "\\" in db_id or ":" in db_id
    ):
        raise ValueError("db_id must name one database directory")
    root = Path(db_dir).resolve()
    path = (root / db_id / f"{db_id}.sqlite").resolve()
    if not path.is_relative_to(root):
        raise ValueError("Database path must stay inside db_dir")
    # Validate before pruning and model initialization.
    if not path.is_file():
        raise FileNotFoundError(f"Database not found: {path}")
    return path


def run_bird_evaluation(
    dev_json_path: str,
    db_dir: str,
    output_path: str,
    limit: int | None = None,
    *,
    metrics_path: str | None = None,
    num_candidates: int = 1,
    selection_mode: str = "self_consistency",
    max_turns: int = 5,
    max_workers: int = 8,
    temperature: float = 0.8,
    top_p: float = 0.7,
    seed: int = 0,
    prompt_mode: str = "uncompiled_dspy",
    compiled_model_path: str | None = None,
    verifier_path: str | None = None,
    verification_samples: int = 1,
    execution_timeout: float = 30.0,
    verbose: bool = False,
    resume: bool = False,
):
    """Run independent trajectories, select without labels, then measure EX.

    The library default is one trajectory for live integration checks. Formal
    ``run_ablation.py`` runs eight in parallel. Missing final answers are JSON
    null, never executable placeholders. Judge-based selection modes also
    save self-consistency predictions from the SAME candidate pool.

    Each completed question is scored and durably committed to a checkpoint
    journal. Pass resume=True with identical inputs/configuration to skip
    committed IDs; failed attempts remain recorded and are not retried.
    """
    for name, value in (("num_candidates", num_candidates), ("max_turns", max_turns),
                        ("max_workers", max_workers), ("verification_samples", verification_samples)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if selection_mode not in {"self_consistency", "dspy_verifier", "hybrid_cascade"}:
        raise ValueError("Unknown selection_mode")
    if prompt_mode not in {"uncompiled_dspy", "compiled_dspy", "baseline"}:
        raise ValueError("Unknown prompt_mode")
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be between 0 (exclusive) and 1")
    if not math.isfinite(execution_timeout) or execution_timeout <= 0:
        raise ValueError("execution_timeout must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    data = _load_items(dev_json_path, limit)
    destination = Path(output_path)
    metrics_destination = Path(metrics_path) if metrics_path else destination.with_suffix(".metrics.json")
    baseline_destination = destination.with_suffix(".self_consistency.json")
    checkpoint_destination = destination.with_suffix(".checkpoint.jsonl")
    outputs = [destination.resolve(), metrics_destination.resolve()]
    if selection_mode != "self_consistency":
        outputs.append(baseline_destination.resolve())
    derived_outputs = list(outputs)
    outputs.extend([checkpoint_destination.resolve(),
                    checkpoint_destination.with_suffix(".jsonl.lock").resolve()])
    protected = {Path(dev_json_path).resolve()}
    for artifact in (compiled_model_path, verifier_path):
        if artifact is not None:
            protected.add(Path(artifact).resolve())
    database_root = Path(db_dir).resolve()
    database_paths = {}
    for item in data:
        db_id = item.get("db_id")
        if isinstance(db_id, str) and db_id and db_id not in {".", ".."} and not any(
            character in db_id for character in ("/", "\\", ":")
        ):
            database_paths[db_id] = (database_root / db_id / f"{db_id}.sqlite").resolve()
            protected.add(database_paths[db_id])
    if len(set(outputs)) != len(outputs) or set(outputs) & protected:
        raise ValueError("Prediction and metrics outputs must be distinct from datasets, databases and compiled artifacts")
    predictions: dict[str, str | None] = {}
    baseline_predictions: dict[str, str | None] = {}
    query_metrics: list[dict[str, Any]] = []
    apps = None
    verifier = None
    if selection_mode in {"dspy_verifier", "hybrid_cascade"} and data:
        from src.dspy_verifier import COMPILED_VERIFIER_PATH
        verifier_path = str(verifier_path or COMPILED_VERIFIER_PATH)

    def select_rating(*args, **kwargs):
        # One compiled rating judge per run, loaded only when selection needs it.
        nonlocal verifier
        from src.dspy_verifier import load_compiled_verifier, select_with_dspy
        if verifier is None:
            verifier = load_compiled_verifier(verifier_path)
        return select_with_dspy(*args, verifier=verifier, artifact_path=verifier_path, **kwargs)
    if prompt_mode == "compiled_dspy":
        from src.dspy_modules import COMPILED_MODEL_PATH
        compiled_model_path = str(compiled_model_path or COMPILED_MODEL_PATH)
        if data and not Path(compiled_model_path).is_file():
            raise FileNotFoundError(f"Compile the trajectory program first: {compiled_model_path}")
    # Include default artifact paths, which may have been resolved after the
    # initial output validation. Reject collisions before any model calls.
    for artifact in (compiled_model_path, verifier_path):
        if artifact is not None and Path(artifact).resolve() in outputs:
            raise ValueError("Outputs must not overwrite compiled DSPy artifacts")
    pruners: dict[Path, DatabaseGraphPruner] = {}
    manifest = {
        "model_name": FORMAL_MODEL, "cross_model_fallback": False,
        "num_candidates": num_candidates, "max_turns": max_turns,
        "max_workers": min(max_workers, num_candidates), "temperature": temperature,
        "top_p": top_p, "top_k": None, "seed": seed, "prompt_mode": prompt_mode,
        "selection_mode": selection_mode, "verification_samples": verification_samples,
        "judge_score_kind": "elicited_validity_rating" if selection_mode != "self_consistency" else None,
        "execution_timeout_seconds": execution_timeout,
        "ex_timeout_scope": "prediction_and_gold_combined",
        "dataset_sha256": _file_hash(dev_json_path),
        "compiled_generator_sha256": _file_hash(compiled_model_path) if data else None,
        "compiled_verifier_sha256": _file_hash(verifier_path) if data else None,
        "question_ids": [str(item["question_id"]) for item in data],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dependencies": _dependency_versions(),
        "independent_plans": True,
        "database_root": str(database_root),
        "database_files": {name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                           if path.is_file() else None for name, path in sorted(database_paths.items())},
        "source_sha256": {name: _file_hash(Path(__file__).parent / name) for name in (
            "bird_evaluator.py", "evaluation_checkpoint.py", "agent_state.py", "graph_pruner.py",
            "sql_agent_graph.py", "dspy_modules.py", "dspy_telemetry.py",
            "dspy_verifier.py", "generative_verifier.py")},
    }

    results = []
    baseline_results = []
    with EvaluationCheckpoint(checkpoint_destination, manifest, resume=resume,
                              outputs=derived_outputs) as checkpoint:
        manifest = checkpoint.manifest
        resumed_questions = len(checkpoint.records)
        if resume:
            print(f"Resuming: skipping {resumed_questions}/{len(data)} committed questions", flush=True)
        for question_index, item in enumerate(data):
            question_id = str(item["question_id"])
            if question_id in checkpoint.records:
                saved = checkpoint.records[question_id]
                predictions[question_id] = saved["prediction"]
                baseline_predictions[question_id] = saved["baseline_prediction"]
                query_metrics.append(saved["metrics"])
                results.append(saved["result"])
                if saved["baseline_result"] is not None:
                    baseline_results.append(saved["baseline_result"])
                continue
            if selection_mode == "dspy_verifier" and verifier is None:
                from src.dspy_verifier import load_compiled_verifier
                verifier = load_compiled_verifier(verifier_path)
            db_id = item.get("db_id", "")
            started = perf_counter()
            pruning_latency = None
            metrics = _unknown_query_metrics()
            error_message = None
            sql = None
            baseline_sql = None
            candidates = []
            selection = ({"mode": "hybrid_cascade", "selected_candidate_index": None,
                          "cascade_tie_breaker_engaged": False}
                         if selection_mode == "hybrid_cascade" else None)
            baseline_selection = None
            baseline_metrics = None
            selection_started = False
            try:
                db_path = _database_path(db_dir, db_id)
                question = item["question"]
                evidence = item.get("evidence") or ""
                if db_path not in pruners:
                    pruners[db_path] = DatabaseGraphPruner(str(db_path))
                _, ddl = pruners[db_path].prune(question, evidence)
                pruning_latency = perf_counter() - started
                state: AgentState = {
                    "question_id": question_id,
                    "db_id": db_id,
                    "question": question,
                    "evidence": evidence,
                    "schema_ddl": ddl,
                    "db_path": str(db_path),
                    "plan": None,
                    "current_sql": None,
                    "observation": None,
                    "history": [],
                    "turn": 0,
                    "max_turns": max_turns,
                    "is_valid": False,
                    "final_sql": None,
                }
                # Construct independently configured programs on the calling thread;
                # worker threads must never mutate DSPy's global settings.
                if apps is None:
                    apps = [build_sql_graph(
                        model_name=FORMAL_MODEL, prompt_mode=prompt_mode,
                        compiled_model_path=compiled_model_path, temperature=temperature,
                        top_p=top_p, formal=True, execution_timeout=execution_timeout,
                        verbose=verbose,
                    ) for _ in range(num_candidates)]
                states = []
                for candidate_index in range(num_candidates):
                    candidate_state = deepcopy(state)
                    candidate_state["candidate_index"] = candidate_index
                    candidate_state["rollout_seed"] = seed + question_index * num_candidates + candidate_index
                    states.append(candidate_state)
                if verbose:
                    print(f"[question {question_id}] initialized {num_candidates} isolated rollouts "
                          f"(workers={min(max_workers, num_candidates)}, seeds="
                          f"{[state['rollout_seed'] for state in states]})", flush=True)
                inference_started = perf_counter()
                if num_candidates == 1:
                    candidates = [_run_candidate(apps[0], states[0])]
                else:
                    with ThreadPoolExecutor(max_workers=min(max_workers, num_candidates)) as pool:
                        # map preserves candidate order despite out-of-order completion.
                        candidates = list(pool.map(_run_candidate, apps, states))
                if verbose:
                    print(f"[question {question_id}] final candidate SQL queries:", flush=True)
                    for candidate in candidates:
                        print(f"  #{candidate['candidate_index'] + 1}: {candidate['final_sql']!r} "
                              f"turns={candidate['turn']} reason={candidate['termination_reason']} "
                              f"error={candidate['error']!r}", flush=True)
                _execute_candidates(candidates, db_path, execution_timeout)
                baseline_selection = select_self_consistency(candidates)
                if verbose:
                    print(f"[question {question_id}] self-consistency execution results:", flush=True)
                    for candidate in candidates:
                        execution = candidate.get("execution") or {}
                        print(f"  #{candidate['candidate_index'] + 1}: "
                              f"rows={execution.get('sample_rows')} "
                              f"row_count={execution.get('row_count')} "
                              f"truncated={execution.get('truncated', False)} "
                              f"hash={execution.get('result_hash')} "
                              f"error={execution.get('error') or candidate.get('error')}", flush=True)
                    print(f"[question {question_id}] vote groups (hash -> zero-based indices): "
                          f"{json.dumps(baseline_selection['groups'], sort_keys=True)}", flush=True)
                    print(f"[question {question_id}] self-consistency winner: "
                          f"{None if baseline_selection['selected_candidate_index'] is None else baseline_selection['selected_candidate_index'] + 1}", flush=True)
                baseline_sql = _selected_sql(candidates, baseline_selection)
                baseline_metrics = _aggregate_candidate_metrics(candidates, baseline_selection, perf_counter() - inference_started)
                baseline_metrics.update({"end_to_end_latency_seconds": perf_counter() - started,
                                         "fallback_used": not bool(baseline_sql)})
                if selection_mode == "dspy_verifier":
                    selection_started = True
                    selection = select_rating(
                        question, evidence, ddl, candidates,
                        verification_samples=verification_samples,
                    )
                elif selection_mode == "hybrid_cascade":
                    selection = select_hybrid_cascade(
                        question, evidence, ddl, candidates,
                        verification_samples=verification_samples, judge=select_rating,
                    )
                else:
                    selection = baseline_selection
                if verbose and selection_mode != "self_consistency":
                    print(f"[question {question_id}] {selection_mode} selection: "
                          f"{json.dumps({key: value for key, value in selection.items() if key != 'metrics'}, sort_keys=True)}",
                          flush=True)
                sql = _selected_sql(candidates, selection)
                metrics = _aggregate_candidate_metrics(candidates, selection, perf_counter() - inference_started)
                if not sql:
                    errors = [candidate["error"] for candidate in candidates if candidate.get("error")]
                    errors.extend(str(error) for error in selection.get("errors", []))
                    error_message = "; ".join(errors) or "No eligible final answer was selected"
            except Exception as error:
                # Configuration errors must not masquerade as a dataset-wide model
                # failure (in particular a stale compiled DSPy signature).
                if apps is None and isinstance(error, (ValueError, FileNotFoundError)) and pruning_latency is not None:
                    raise
                logger.error("Error on question %s: %s", question_id, error)
                if candidates:
                    selection = (getattr(error, "mars_selection", None) or selection
                                 or {"mode": selection_mode, "selected_candidate_index": None})
                    if getattr(error, "mars_metrics", None):
                        selection["metrics"] = error.mars_metrics
                    elif selection_started or selection.get("cascade_tie_breaker_engaged"):
                        selection["metrics"] = _unknown_query_metrics()
                    metrics = _aggregate_candidate_metrics(candidates, selection, perf_counter() - inference_started)
                else:
                    metrics.update(getattr(error, "mars_metrics", None) or {})
                error_message = f"{type(error).__name__}: {error}"
            elapsed = perf_counter() - started
            metrics.update({
                "question_id": question_id,
                "db_id": db_id,
                "fallback_used": not bool(sql),  # Legacy field: now means failure, not substitute SQL.
                "generation_failed": not any(candidate.get("final_sql") for candidate in candidates),
                "selection_failed": bool(candidates) and not bool(sql),
                "error": error_message,
                "pruning_latency_seconds": pruning_latency if pruning_latency is not None else elapsed,
                "end_to_end_latency_seconds": elapsed,
                "candidates": candidates,
                "selection": selection,
                "self_consistency_selection": baseline_selection,
                "self_consistency_metrics": baseline_metrics,
            })
            query_metrics.append(metrics)
            predictions[question_id] = f"{sql}{BIRD_SEPARATOR}{db_id}" if sql else None
            baseline_predictions[question_id] = f"{baseline_sql}{BIRD_SEPARATOR}{db_id}" if baseline_sql else None
            result = _score_prediction(item, db_dir, predictions[question_id], execution_timeout)
            baseline_result = (_score_prediction(item, db_dir, baseline_predictions[question_id], execution_timeout)
                               if selection_mode != "self_consistency" else None)
            metrics.update(execution_correct=result["correct"], execution_error=result["error"])
            results.append(result)
            if baseline_result is not None:
                baseline_results.append(baseline_result)
            checkpoint.append({"question_id": question_id, "prediction": predictions[question_id],
                               "baseline_prediction": baseline_predictions[question_id], "metrics": metrics,
                               "result": result, "baseline_result": baseline_result})
            print(f"Checkpointed question {question_id} ({question_index + 1}/{len(data)})", flush=True)

        atomic_write_json(destination, predictions)
        report = {
            "schema_version": 3,
            "manifest": manifest,
            "predictions_path": str(destination.resolve()),
            "checkpoint_path": str(checkpoint_destination.resolve()),
            "resumed_questions": resumed_questions,
            **_score_summary(results),
            "queries": query_metrics,
            "totals": _sum_query_metrics(query_metrics),
        }
        if selection_mode != "self_consistency":
            atomic_write_json(baseline_destination, baseline_predictions)
            report["self_consistency"] = {
                "predictions_path": str(baseline_destination.resolve()),
                "totals": _sum_query_metrics([row.get("self_consistency_metrics") or _unknown_query_metrics()
                                              for row in query_metrics]),
                **_score_summary(baseline_results),
            }
        atomic_write_json(metrics_destination, report)
        return output_path


def _file_hash(path) -> str | None:
    if path is None:
        return None
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _dependency_versions():
    versions = {}
    for package in ("dspy", "langgraph", "langfuse", "litellm", "networkx", "openai"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _run_candidate(app, state):
    """One independent rollout; transport failures cannot cancel sibling rollouts."""
    metrics = _unknown_query_metrics()
    final_state = {}
    error = None
    started = perf_counter()
    try:
        final_state = app.invoke(state)
        metrics.update(final_state.get("metrics") or {})
    except Exception as failure:
        final_state = getattr(failure, "mars_state", None) or {}
        metrics.update(getattr(failure, "mars_metrics", None) or {})
        error = f"{type(failure).__name__}: {failure}"
    sql = (final_state.get("final_sql") or "").strip() or None
    if sql is None and error is None:
        error = "Trajectory did not produce an executable explicit final_answer"
    return {
        "candidate_index": state["candidate_index"], "rollout_seed": state["rollout_seed"],
        "final_sql": sql, "history": final_state.get("history", []),
        "logical_plan": str(final_state.get("plan") or ""),
        "turn": final_state.get("turn", metrics.get("turn")),
        "error": error, "metrics": metrics, "wall_latency_seconds": perf_counter() - started,
        "termination_reason": final_state.get("termination_reason") or (
            "error" if error else "final_answer"
        ),
    }


def _result_fingerprint(rows: set[tuple]) -> str:
    """Stable hash of complete BIRD-equivalent sets, including blobs and NULL.

    Numeric normalization follows Python/SQLite row equality (1 == 1.0).
    Values are type-tagged so blobs, text and NULL never collide.
    """
    def cell(value):
        if value is None:
            return ["null"]
        if isinstance(value, bytes):
            return ["blob", value.hex()]
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                return ["number", str(value)]
            numerator, denominator = value.as_integer_ratio()
            return ["number", str(numerator), str(denominator)]
        return ["text", value]
    encoded = sorted(json.dumps([cell(value) for value in row], ensure_ascii=False,
                                separators=(",", ":")) for row in rows)
    return hashlib.sha256(json.dumps(encoded, ensure_ascii=False).encode("utf-8")).hexdigest()


def _execute_candidates(candidates, db_path, timeout):
    cache = {}
    for candidate in candidates:
        sql = candidate["final_sql"]
        if not sql:
            continue
        if sql not in cache:
            try:
                rows = _execute_result_set(db_path, sql, timeout)
                preview = sorted(rows, key=repr)[:50]
                cache[sql] = {
                    "result_hash": _result_fingerprint(rows), "row_count": len(rows),
                    "sample_rows": [[value.hex() if isinstance(value, bytes) else
                                     str(value) if isinstance(value, float) and not math.isfinite(value) else value for value in row]
                                    for row in preview],
                    "truncated": len(rows) > 50, "error": None,
                    "row_count_semantics": "distinct_rows_for_BIRD_EX",
                }
            except (sqlite3.Error, OSError, ValueError, TypeError) as error:
                cache[sql] = {"error": f"{type(error).__name__}: {error}"}
        candidate["execution"] = deepcopy(cache[sql])
        if candidate["execution"]["error"]:
            candidate["error"] = candidate["execution"]["error"]
            candidate["final_sql"] = None


def select_self_consistency(candidates):
    """Zero-model-cost vote over full execution results; earliest index breaks ties."""
    groups = {}
    for candidate in candidates:
        execution = candidate.get("execution") or {}
        if candidate.get("final_sql") and not execution.get("error") and execution.get("result_hash"):
            groups.setdefault(execution["result_hash"], []).append(candidate["candidate_index"])
    winner = min(groups.values(), key=lambda indices: (-len(indices), min(indices))) if groups else []
    return {"mode": "self_consistency", "selected_candidate_index": min(winner) if winner else None,
            "groups": groups, "extra_llm_calls": 0}


def select_hybrid_cascade(
    question, evidence, schema_ddl, candidates, *, verification_samples=1, judge=None,
):
    """Use DSPy validity ratings only for tied top execution-result groups.

    A unique top group wins, even without an absolute majority of all rollouts
    (e.g. 3-2-2-1). Execution eligibility and hashing are identical to the
    self-consistency baseline. Candidate indices always refer to the original
    rollout pool, including when only a subset is sent to the judge.
    """
    selection = dict(select_self_consistency(candidates), mode="hybrid_cascade",
                     cascade_tie_breaker_engaged=False, tied_candidate_indices=[],
                     selection_source="self_consistency")
    groups = selection["groups"]
    if not groups:
        return selection
    highest_count = max(len(indices) for indices in groups.values())
    top_groups = [indices for indices in groups.values() if len(indices) == highest_count]
    if len(top_groups) == 1:
        return selection

    tied_indices = {index for indices in top_groups for index in indices}
    tied_candidates = [candidate for candidate in candidates
                       if candidate["candidate_index"] in tied_indices]
    selection.update(selected_candidate_index=None, cascade_tie_breaker_engaged=True,
                     tied_candidate_indices=sorted(tied_indices),
                     selection_source="dspy_verifier", score_kind="elicited_validity_rating", extra_llm_calls=None)
    try:
        if judge is None:
            from src.dspy_verifier import select_with_dspy
            judge = select_with_dspy
        judged = judge(question, evidence, schema_ddl, tied_candidates,
                       verification_samples=verification_samples)
        selected_index = judged.get("selected_candidate_index")
        if selected_index is not None and selected_index not in tied_indices:
            failure = ValueError("DSPy verifier selected a candidate outside the tied top groups")
            failure.mars_metrics = judged.get("metrics")
            raise failure
    except Exception as error:
        # Retain routing evidence and billed/unknown selector usage if setup,
        # provider transport, or telemetry fails after entering the judge path.
        error.mars_selection = selection
        raise
    return dict(judged, mode="hybrid_cascade", groups=groups,
                cascade_tie_breaker_engaged=True,
                tied_candidate_indices=selection["tied_candidate_indices"],
                selection_source="dspy_verifier",
                extra_llm_calls=(judged.get("metrics") or {}).get("llm_call_count"))


def _selected_sql(candidates, selection):
    index = (selection or {}).get("selected_candidate_index")
    return next((candidate["final_sql"] for candidate in candidates
                 if candidate["candidate_index"] == index), None)


def _aggregate_candidate_metrics(candidates, selection, elapsed):
    measurements = [candidate["metrics"] for candidate in candidates]
    selection_metrics = (selection or {}).get("metrics")
    if selection_metrics:
        measurements.append(dict(selection_metrics, turn=0, sql_attempts=0,
                                 sql_execution_count=0, self_correction_cycles=0))
    aggregate = _sum_query_metrics(measurements)
    aggregate.pop("fallback_count", None)
    aggregate["stages"] = [dict(stage, candidate_index=candidate["candidate_index"])
                           for candidate in candidates for stage in candidate["metrics"].get("stages", [])]
    aggregate["stages"].extend((selection_metrics or {}).get("stages", []))
    aggregate["generation_cost_usd"] = _sum_query_metrics([candidate["metrics"] for candidate in candidates])["cost_usd"]
    aggregate["selection_cost_usd"] = selection_metrics.get("cost_usd") if selection_metrics else 0.0
    aggregate["latency_seconds"] = elapsed
    aggregate["trajectory_count"] = len(candidates)
    aggregate["selected_candidate_index"] = (selection or {}).get("selected_candidate_index")
    aggregate["selected_turn"] = next((candidate["turn"] for candidate in candidates
                                      if candidate["candidate_index"] == aggregate["selected_candidate_index"]), None)
    if len(candidates) == 1:
        for key in ("trace_id", "trace_url"):
            if key in candidates[0]["metrics"]:
                aggregate[key] = candidates[0]["metrics"][key]
    return aggregate


def _unknown_query_metrics() -> dict[str, Any]:
    return {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "cost_usd": None,
        "observed_prompt_tokens": None,
        "observed_completion_tokens": None,
        "observed_total_tokens": None,
        "observed_cost_usd": None,
        "llm_latency_seconds": None,
        "latency_seconds": None,
        "turn": None,
        "sql_attempts": None,
        "sql_execution_count": None,
        "self_correction_cycles": None,
        "accounting_complete": False,
        "stages": [],
    }


def _sum_query_metrics(queries: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum complete fields only; never disguise missing accounting as zero."""
    fields = (
        "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd",
        "llm_latency_seconds", "latency_seconds", "end_to_end_latency_seconds",
        "turn", "sql_attempts", "sql_execution_count", "self_correction_cycles",
        "observed_prompt_tokens", "observed_completion_tokens",
        "observed_total_tokens", "observed_cost_usd",
    )
    totals = {}
    for field in fields:
        values = [query.get(field) for query in queries]
        if field.startswith("observed_"):
            known_values = [value for value in values if value is not None]
            totals[field] = sum(known_values) if known_values or not queries else None
        else:
            totals[field] = sum(values) if all(value is not None for value in values) else None
    totals["accounting_complete"] = all(query["accounting_complete"] for query in queries)
    totals["fallback_count"] = sum(bool(query.get("fallback_used")) for query in queries)
    return totals


def _execute_result_set(db_path: Path, sql: str, timeout_seconds: float, *, deadline: float | None = None) -> set[tuple]:
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL query must be a non-empty string")
    if deadline is None:
        deadline = monotonic() + timeout_seconds
    if monotonic() >= deadline:
        raise TimeoutError("SQL execution exceeded its time budget")
    uri = db_path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=timeout_seconds)) as connection:
        connection.execute("PRAGMA query_only = ON")
        read_actions = {
            sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
            sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE,
        }
        connection.set_authorizer(
            lambda action, *_: (
                sqlite3.SQLITE_OK if action in read_actions else sqlite3.SQLITE_DENY
            )
        )
        connection.set_progress_handler(lambda: monotonic() >= deadline, 1000)
        cursor = connection.execute(sql)
        if cursor.description is None:
            raise ValueError("SQL query did not return a result set")
        rows = set()
        while batch := cursor.fetchmany(1000):
            if monotonic() >= deadline:
                raise TimeoutError("SQL execution exceeded its time budget")
            rows.update(batch)
        if monotonic() >= deadline:
            raise TimeoutError("SQL execution exceeded its time budget")
        return rows


def _score_prediction(item, db_dir, prediction, timeout_seconds):
    matched = False
    error_message = None
    try:
        if not isinstance(prediction, str) or BIRD_SEPARATOR not in prediction:
            raise ValueError("Missing or malformed prediction")
        sql, predicted_db_id = prediction.rsplit(BIRD_SEPARATOR, 1)
        if predicted_db_id != item["db_id"]:
            raise ValueError("Prediction db_id does not match the dataset record")
        db_path = _database_path(db_dir, item["db_id"])
        deadline = monotonic() + timeout_seconds
        predicted_rows = _execute_result_set(db_path, sql, timeout_seconds, deadline=deadline)
        gold_rows = _execute_result_set(db_path, item["SQL"], timeout_seconds, deadline=deadline)
        matched = predicted_rows == gold_rows
    except (sqlite3.Error, OSError, ValueError, KeyError, TypeError) as error:
        error_message = f"{type(error).__name__}: {error}"
    return {"question_id": str(item["question_id"]), "correct": matched, "error": error_message}


def _score_summary(results):
    correct = sum(int(result["correct"]) for result in results)
    total = len(results)
    return {"correct": correct, "total": total,
            "execution_accuracy": 100.0 * correct / total if total else 0.0,
            "results": results}


def compute_execution_accuracy(
    dev_json_path: str,
    db_dir: str,
    predictions_path: str,
    limit: int | None = None,
    *,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Return correct, total, execution_accuracy (0-100), and per-item results.

    Match predictions to question_id, and execute both queries on the gold
    record's database. Missing/malformed predictions, SQL errors, and timeouts
    count as incorrect and remain in the denominator. Use the same limit as
    generation when evaluating a subset. No selected records yields EX = 0.0.
    """
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a positive finite number")
    data = _load_items(dev_json_path, limit)
    with open(predictions_path, encoding="utf-8") as handle:
        predictions = json.load(handle)
    if not isinstance(predictions, dict):
        raise ValueError("predict.json must contain an object keyed by question_id")

    return _score_summary([_score_prediction(item, db_dir, predictions.get(str(item["question_id"])),
                                            timeout_seconds) for item in data])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-json", required=True, help="BIRD dev.json path")
    parser.add_argument("--db-dir", required=True, help="BIRD dev_databases directory")
    parser.add_argument("--output", default="predict.json", help="Prediction JSON path")
    parser.add_argument("--metrics-output", default=None, help="Metrics JSON path (default: <output stem>.metrics.json)")
    parser.add_argument("--limit", type=int, default=None, help="First N questions")
    parser.add_argument("--score-only", action="store_true", help="Score existing predictions")
    parser.add_argument("--resume", action="store_true", help="Resume a matching question checkpoint")
    args = parser.parse_args()
    if not args.score_only:
        run_bird_evaluation(
            args.dev_json, args.db_dir, args.output, args.limit,
            metrics_path=args.metrics_output,
            resume=args.resume,
        )
        print(f"Predictions saved to {args.output}")
        metrics_path = Path(args.metrics_output) if args.metrics_output else Path(args.output).with_suffix(".metrics.json")
        print(f"Metrics saved to {metrics_path}")
        with metrics_path.open(encoding="utf-8") as handle:
            score = json.load(handle)
    else:
        score = compute_execution_accuracy(args.dev_json, args.db_dir, args.output, args.limit)
    print(f"Execution Accuracy (EX): {score['execution_accuracy']:.2f}% "
          f"({score['correct']}/{score['total']})")


if __name__ == "__main__":
    main()
