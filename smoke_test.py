"""Run and strictly validate six live three-question ablation configurations.

Uses real OpenRouter calls through run_ablation.py. Stop on the first failed
process or validation; semantic EX is reported rather than used as a gate.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from uuid import uuid4


ROOT = Path(__file__).resolve().parent
SEPARATOR = "\t----- bird -----\t"
FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd")


def _require(condition, message):
    if not condition:
        raise ValueError(f"Smoke validation failed: {message}")


def _accounting(metrics, label):
    _require(metrics.get("accounting_complete") is True, f"{label}: incomplete accounting")
    for field in FIELDS:
        value = metrics.get(field)
        _require(isinstance(value, (int, float)) and not isinstance(value, bool)
                 and math.isfinite(value) and value >= 0, f"{label}: missing/invalid {field}")
        if field != "cost_usd":
            _require(float(value).is_integer(), f"{label}: fractional {field}")


def validate_run(output, items, *, prompt_mode, selection_mode, candidates, dataset_sha256,
                 compiled_sha256=None):
    output = Path(output)
    predictions = json.loads(output.read_text(encoding="utf-8"))
    report = json.loads(output.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    ids = [str(item["question_id"]) for item in items]
    _require(isinstance(predictions, dict) and set(predictions) == set(ids), "missing/extra prediction IDs")
    queries = report.get("queries", [])
    _require(report.get("total") == 3 and len(queries) == 3, "expected exactly three evaluated questions")
    _require([row.get("question_id") for row in queries] == ids, "metrics IDs/order do not match dataset")
    manifest = report.get("manifest", {})
    for field, value in {"prompt_mode": prompt_mode, "selection_mode": selection_mode,
                         "question_ids": ids, "num_candidates": candidates,
                         "dataset_sha256": dataset_sha256,
                         "compiled_generator_sha256": compiled_sha256}.items():
        _require(manifest.get(field) == value, f"wrong manifest {field}")
    _accounting(report.get("totals", {}), "totals")
    for item, query in zip(items, queries):
        question_id = str(item["question_id"])
        prediction = predictions[question_id]
        _require(isinstance(prediction, str) and SEPARATOR in prediction,
                 f"question {question_id}: missing final SQL")
        sql, db_id = prediction.rsplit(SEPARATOR, 1)
        _require(bool(sql.strip()) and db_id == item["db_id"], f"question {question_id}: malformed prediction")
        for flag in ("generation_failed", "selection_failed", "fallback_used"):
            _require(query.get(flag) is False, f"question {question_id}: {flag}")
        _require(not query.get("error") and not query.get("execution_error"),
                 f"question {question_id}: {query.get('error') or query.get('execution_error')}")
        selection = query.get("selection") or {}
        _require(not selection.get("errors"), f"question {question_id}: judge errors")
        index = selection.get("selected_candidate_index")
        rollouts = query.get("candidates", [])
        _require(len(rollouts) == candidates, f"question {question_id}: wrong trajectory count")
        selected = next((row for row in rollouts if row.get("candidate_index") == index), None)
        _require(selected is not None and selected.get("final_sql") == sql,
                 f"question {question_id}: selected SQL/index mismatch")
        execution = selected.get("execution") or {}
        _require(not execution.get("error") and bool(execution.get("result_hash")),
                 f"question {question_id}: selected SQL did not execute")
        if selection_mode == "hybrid_cascade":
            _require(isinstance(selection.get("cascade_tie_breaker_engaged"), bool),
                     f"question {question_id}: missing cascade routing flag")
        _accounting(query, f"question {question_id}")
    for field in FIELDS:
        total = sum(query[field] for query in queries)
        _require(math.isclose(total, report["totals"][field], rel_tol=1e-9, abs_tol=1e-12),
                 f"totals {field} does not match question measurements")
    return {"prompt_mode": prompt_mode, "selection_mode": selection_mode,
            "execution_accuracy": report["execution_accuracy"],
            "total_tokens": report["totals"]["total_tokens"],
            "cost_usd": report["totals"]["cost_usd"], "output": str(output)}


def run_matrix(*, dev_json=ROOT / "data/dev.json", db_dir=ROOT / "data/dev_databases",
               output_root=ROOT / "artifacts/smoke_test", compiled_model=ROOT / "compiled_dspy_model.json",
               verifier_selection="dspy_verifier", verifier_model=None,
               candidates=8, workers=8, seed=0, timeout_seconds=1800, runner=None):
    """Fresh process/output per configuration; all exceptions halt the matrix."""
    dev_json, db_dir, compiled_model = map(lambda path: Path(path).resolve(),
                                         (dev_json, db_dir, compiled_model))
    items = json.loads(dev_json.read_text(encoding="utf-8"))[:3]
    _require(len(items) == 3 and len({str(row["question_id"]) for row in items}) == 3,
             "dataset must supply three distinct question IDs")
    for item in items:
        db_id = item["db_id"]
        _require(isinstance(db_id, str) and db_id not in {"", ".", ".."}
                 and not any(char in db_id for char in ("/", "\\", ":")), "invalid database ID")
        _require((db_dir / db_id / f"{db_id}.sqlite").is_file(), f"database missing: {db_id}")
    _require(candidates > 0 and workers > 0 and timeout_seconds > 0, "budgets must be positive")
    _require(verifier_selection == "dspy_verifier", "only the DSPy rating verifier is supported")
    # Check compiled artifacts before spending on even the uncompiled runs.
    artifact = json.loads(compiled_model.read_text(encoding="utf-8"))
    manifest = artifact.get("mars_generator_manifest", {})
    _require(manifest.get("version") == 2 and manifest.get("action_contract") == "probe_final_answer"
             and manifest.get("max_iterations") == 5, "incompatible compiled generator manifest")
    for stage in ("decomposer", "generator"):
        _require(bool(artifact.get(stage, {}).get("demos")), f"compiled generator lacks {stage} demos")
    verifier_model = Path(verifier_model or ROOT / "compiled_dspy_verifier.json").resolve()
    from src.dspy_verifier import validate_compiled_verifier_artifact
    validate_compiled_verifier_artifact(verifier_model)
    dataset_hash = hashlib.sha256(dev_json.read_bytes()).hexdigest()
    compiled_hash = hashlib.sha256(compiled_model.read_bytes()).hexdigest()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:12]
    run_root = Path(output_root).resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    print(f"Live matrix smoke test artifacts: {run_root}", flush=True)
    runner = runner or subprocess.run
    results = []
    for prompt_label, prompt_mode in (("uncompiled", "uncompiled_dspy"), ("compiled", "compiled_dspy")):
        for selection_label, selection_mode in (("self_consistency", "self_consistency"),
                                                ("verifier", verifier_selection), ("hybrid", "hybrid_cascade")):
            name = f"{prompt_label}_{selection_label}"
            directory = run_root / name
            directory.mkdir()
            output = directory / "predict.json"
            command = [sys.executable, "-u", str(ROOT / "run_ablation.py"),
                       "--dev-json", str(dev_json), "--db-dir", str(db_dir), "--limit", "3",
                       "--prompt-mode", prompt_mode, "--selection", selection_mode,
                       "--candidates", str(candidates), "--workers", str(workers),
                       "--max-turns", "5", "--temperature", "0.8", "--top-p", "0.7",
                       "--seed", str(seed), "--verification-samples", "1", "--output", str(output)]
            if prompt_mode == "compiled_dspy":
                command.extend(["--compiled-model", str(compiled_model)])
            if selection_mode in {"dspy_verifier", "hybrid_cascade"}:
                command.extend(["--verifier-model", str(verifier_model)])
            log_path = directory / "run.log"
            print(f"[{len(results) + 1}/6] Running {name}; log: {log_path}", flush=True)
            (directory / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
            with log_path.open("w", encoding="utf-8") as log:
                runner(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       timeout=timeout_seconds, check=True)
            try:
                result = validate_run(output, items, prompt_mode=prompt_mode, selection_mode=selection_mode,
                                      candidates=candidates, dataset_sha256=dataset_hash,
                                      compiled_sha256=compiled_hash if prompt_mode == "compiled_dspy" else None)
            except Exception as error:
                raise RuntimeError(f"{name} failed validation; inspect {directory}: {error}") from error
            results.append(result)
            print(f"PASS {name}: EX={result['execution_accuracy']:.2f}% "
                  f"tokens={result['total_tokens']} cost_usd={result['cost_usd']}", flush=True)
    (run_root / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("Matrix Smoke Test Fully Passed", flush=True)
    return run_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-json", type=Path, default=ROOT / "data/dev.json")
    parser.add_argument("--db-dir", type=Path, default=ROOT / "data/dev_databases")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/smoke_test")
    parser.add_argument("--compiled-model", type=Path, default=ROOT / "compiled_dspy_model.json")
    parser.add_argument("--verifier-selection", choices=("dspy_verifier",), default="dspy_verifier",
                        help="Standalone and hybrid judges both use DSPy validity ratings")
    parser.add_argument("--verifier-model", type=Path)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    args = vars(parser.parse_args(argv))
    run_matrix(**args)


if __name__ == "__main__":
    main()
