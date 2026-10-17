"""Run one real model query and retain predictions plus exact accounting."""

from contextlib import closing
import argparse
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from uuid import uuid4

from src.bird_evaluator import run_bird_evaluation


def _display(value):
    return "unknown (provider accounting unavailable)" if value is None else value


def run_live_trace(*, prompt_mode="uncompiled_dspy"):
    artifact_dir = Path(__file__).resolve().parent / "artifacts" / f"live_trace_{uuid4().hex}"
    artifact_dir.mkdir(parents=True)
    predict_path = artifact_dir / "predict.json"
    metrics_path = artifact_dir / "predict.metrics.json"

    with TemporaryDirectory(prefix="mars_live_trace_") as temporary_directory:
        test_dir = Path(temporary_directory)
        db_root = test_dir / "dev_databases"
        db_dir = db_root / "mock_db"
        db_dir.mkdir(parents=True)
        with closing(sqlite3.connect(db_dir / "mock_db.sqlite")) as connection:
            connection.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);")
            connection.execute("INSERT INTO users VALUES (1, 'Alice', 30), (2, 'Bob', 25), (3, 'Charlie', 35);")
            connection.commit()
        dev_json_path = test_dir / "dev.json"
        dev_json_path.write_text(json.dumps([{
            "question_id": 999,
            "db_id": "mock_db",
            "question": "What is the name of the oldest user?",
            "evidence": "Sort by age descending.",
            "SQL": "SELECT name FROM users ORDER BY age DESC LIMIT 1;",
        }]), encoding="utf-8")

        print("Sending live query to OpenRouter...")
        run_bird_evaluation(str(dev_json_path), str(db_root), str(predict_path), prompt_mode=prompt_mode)

    predictions = json.loads(predict_path.read_text(encoding="utf-8"))
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = report["queries"][0]
    print("\nPrediction for question 999:")
    print(predictions["999"])
    for stage in metrics["stages"]:
        print(
            f"{stage.get('stage', stage.get('name', 'model'))} "
            f"(attempt {stage.get('attempt', stage.get('turn', '?'))}): "
            f"prompt={_display(stage.get('prompt_tokens'))}, "
            f"completion={_display(stage.get('completion_tokens'))}, "
            f"total={_display(stage.get('total_tokens'))}, "
            f"cost_usd={_display(stage.get('cost_usd'))}, "
            f"latency_seconds={_display(stage.get('latency_seconds'))}"
        )
    print(
        f"Query totals: prompt={_display(metrics['prompt_tokens'])}, "
        f"completion={_display(metrics['completion_tokens'])}, "
        f"total={_display(metrics['total_tokens'])}, "
        f"cost_usd={_display(metrics['cost_usd'])}, "
        f"llm_latency_seconds={_display(metrics['llm_latency_seconds'])}, "
        f"latency_seconds={_display(metrics['latency_seconds'])}"
    )
    print(
        f"Generator iterations: {_display(metrics['turn'])}; "
        f"SQL executions: {_display(metrics.get('sql_execution_count'))}; "
        f"self-correction cycles: {_display(metrics['self_correction_cycles'])}"
    )
    print(f"Execution Accuracy (EX): {report['execution_accuracy']:.2f}% ({report['correct']}/{report['total']})")
    print(f"Predictions: {predict_path}")
    print(f"Exact metrics: {metrics_path}")
    if metrics.get("trace_url"):
        print(f"Langfuse trace: {metrics['trace_url']}")
    elif metrics.get("trace_id"):
        print(f"Langfuse trace ID: {metrics['trace_id']}")
    if metrics["fallback_used"] or report["execution_accuracy"] != 100.0:
        raise RuntimeError(f"Live query failed verification: {metrics.get('error') or metrics.get('execution_error') or 'incorrect SQL result'}")
    print("Success! The generated SQL returns Charlie.")
    if not metrics["accounting_complete"]:
        print("Accounting is incomplete; inspect null values and stage errors in the metrics report.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-mode", choices=("uncompiled_dspy", "compiled_dspy", "baseline"),
                        default="uncompiled_dspy")
    run_live_trace(prompt_mode=parser.parse_args().prompt_mode)
