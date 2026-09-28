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
import json
import logging
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from time import monotonic
from typing import Any

from src.agent_state import AgentState
from src.graph_pruner import DatabaseGraphPruner
from src.sql_agent_graph import build_sql_graph

BIRD_SEPARATOR = "\t----- bird -----\t"
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
    # The pruner opens SQLite normally, which would otherwise create missing DBs.
    if not path.is_file():
        raise FileNotFoundError(f"Database not found: {path}")
    return path


def run_bird_evaluation(
    dev_json_path: str,
    db_dir: str,
    output_path: str,
    limit: int | None = None,
):
    """Prune and generate queries, saving BIRD-formatted predictions in order.

    An item failure or a run without final_sql emits SELECT 1; as specified by
    the harness contract. Gold SQL is never included in the graph's input.
    A limit of zero writes an empty prediction object without building a graph.
    """
    data = _load_items(dev_json_path, limit)
    predictions: dict[str, str] = {}
    app = build_sql_graph() if data else None
    pruners: dict[Path, DatabaseGraphPruner] = {}

    for item in data:
        question_id = str(item["question_id"])
        db_id = item.get("db_id", "")
        sql = "SELECT 1;"
        try:
            db_path = _database_path(db_dir, db_id)
            question = item["question"]
            evidence = item.get("evidence") or ""
            if db_path not in pruners:
                pruners[db_path] = DatabaseGraphPruner(str(db_path))
            _, ddl = pruners[db_path].prune(question, evidence)
            state: AgentState = {
                "question": question,
                "evidence": evidence,
                "schema_ddl": ddl,
                "db_path": str(db_path),
                "plan": None,
                "current_sql": None,
                "observation": None,
                "history": [],
                "turn": 0,
                "max_turns": 3,
                "is_valid": False,
                "final_sql": None,
            }
            final_state = app.invoke(state)
            sql = (final_state.get("final_sql") or "SELECT 1;").strip() or "SELECT 1;"
        except Exception as error:
            logger.error("Error on question %s: %s", question_id, error)
        predictions[question_id] = f"{sql}{BIRD_SEPARATOR}{db_id}"

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return output_path


def _execute_result_set(db_path: Path, sql: str, timeout_seconds: float) -> set[tuple]:
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL query must be a non-empty string")
    uri = db_path.as_uri() + "?mode=ro"
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
        deadline = monotonic() + timeout_seconds
        connection.set_progress_handler(lambda: monotonic() >= deadline, 1000)
        cursor = connection.execute(sql)
        if cursor.description is None:
            raise ValueError("SQL query did not return a result set")
        rows = set()
        while batch := cursor.fetchmany(1000):
            if monotonic() >= deadline:
                raise TimeoutError("SQL execution exceeded its time budget")
            rows.update(batch)
        return rows


def compute_execution_accuracy(
    dev_json_path: str,
    db_dir: str,
    predictions_path: str,
    limit: int | None = None,
    *,
    timeout_seconds: float = 5.0,
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

    results = []
    correct = 0
    for item in data:
        question_id = str(item["question_id"])
        matched = False
        error_message = None
        try:
            prediction = predictions.get(question_id)
            if not isinstance(prediction, str) or BIRD_SEPARATOR not in prediction:
                raise ValueError("Missing or malformed prediction")
            sql, predicted_db_id = prediction.rsplit(BIRD_SEPARATOR, 1)
            if predicted_db_id != item["db_id"]:
                raise ValueError("Prediction db_id does not match the dataset record")
            db_path = _database_path(db_dir, item["db_id"])
            predicted_rows = _execute_result_set(db_path, sql, timeout_seconds)
            gold_rows = _execute_result_set(db_path, item["SQL"], timeout_seconds)
            matched = predicted_rows == gold_rows
        except (sqlite3.Error, OSError, ValueError, KeyError, TypeError) as error:
            error_message = f"{type(error).__name__}: {error}"
        correct += int(matched)
        results.append({
            "question_id": question_id,
            "correct": matched,
            "error": error_message,
        })

    total = len(data)
    return {
        "correct": correct,
        "total": total,
        "execution_accuracy": 100.0 * correct / total if total else 0.0,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-json", required=True, help="BIRD dev.json path")
    parser.add_argument("--db-dir", required=True, help="BIRD dev_databases directory")
    parser.add_argument("--output", default="predict.json", help="Prediction JSON path")
    parser.add_argument("--limit", type=int, default=None, help="First N questions")
    parser.add_argument("--score-only", action="store_true", help="Score existing predictions")
    args = parser.parse_args()
    if not args.score_only:
        run_bird_evaluation(args.dev_json, args.db_dir, args.output, args.limit)
        print(f"Predictions saved to {args.output}")
    score = compute_execution_accuracy(args.dev_json, args.db_dir, args.output, args.limit)
    print(f"Execution Accuracy (EX): {score['execution_accuracy']:.2f}% "
          f"({score['correct']}/{score['total']})")


if __name__ == "__main__":
    main()
