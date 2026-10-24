"""Offline BIRD pipeline and execution accuracy checks.

Run with ``python -m tests.test_bird_evaluator`` from the project directory.
Only model transport is scripted; pruning, LangGraph, and SQLite run normally.
"""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.bird_evaluator import compute_execution_accuracy, run_bird_evaluation
from src.sql_agent_graph import build_sql_graph


BIRD_SEPARATOR = "\t----- bird -----\t"
NAMES_SQL = (
    "SELECT student_name FROM students WHERE student_name != 'Émile' ORDER BY student_id"
)
COUNT_SQL = "SELECT COUNT(*) FROM students WHERE age >= 21"
PLAN = {
    "target_tables": ["students"],
    "target_columns": ["students.student_name", "students.age"],
    "join_logic": None,
    "filter_logic": None,
    "reasoning": "Project the requested names or count rows using the requested age filter.",
}


def generated_sql(sql, action="final_answer"):
    return {"action": action, "rationale": "Use the students schema and the requested filter.", "sql_query": sql}


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        if index >= len(self.responses):
            raise AssertionError("Unexpected extra model request")
        response = self.responses[index]
        if isinstance(response, Exception):
            raise response
        message = SimpleNamespace(content=json.dumps(response))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class BirdEvaluatorIntegrationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="mars_bird_evaluator_")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.db_dir = self.root / "dev_databases"
        self.db_path = self.db_dir / "school" / "school.sqlite"
        self.db_path.parent.mkdir(parents=True)
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.executescript(
                """
                CREATE TABLE students (
                    student_id INTEGER PRIMARY KEY,
                    student_name TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    school_id INTEGER REFERENCES schools(school_id)
                );
                -- Deliberately create the FK parent after the child table.
                CREATE TABLE schools (school_id INTEGER PRIMARY KEY, school_name TEXT);
                INSERT INTO students (student_id, student_name, age)
                    VALUES (1, 'Ada', 22), (2, 'Zoë', 24), (3, 'Linus', 19);
                """
            )
        self.items = [
            {
                "question_id": 7,
                "db_id": "school",
                "question": "List students' student_name values except Émile, including Zoë.",
                "evidence": "student_name is the student's displayed name.",
                "SQL": NAMES_SQL,
                "difficulty": "simple",
            },
            {
                "question_id": 42,
                "db_id": "school",
                "question": "How many students have age at least 21?",
                "SQL": COUNT_SQL,
                "difficulty": "simple",
            },
        ]
        self.dev_path = self.root / "dev.json"
        self.output_path = self.root / "predictions.json"
        self.write_dataset()

    def write_dataset(self):
        self.dev_path.write_text(json.dumps(self.items, ensure_ascii=False), encoding="utf-8")

    def correct_predictions(self):
        return {
            str(item["question_id"]): item["SQL"] + BIRD_SEPARATOR + item["db_id"]
            for item in self.items
        }

    def write_predictions(self, predictions):
        self.output_path.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")

    def evaluate(self, *, limit=None):
        return compute_execution_accuracy(
            str(self.dev_path), str(self.db_dir), str(self.output_path), limit=limit
        )

    def run_pipeline(self, responses, *, limit=None):
        # Each subtest is a fresh experiment; evaluation outputs are protected.
        self.pipeline_runs = getattr(self, "pipeline_runs", 0) + 1
        if self.pipeline_runs > 1:
            self.output_path = self.root / f"predictions_{self.pipeline_runs}.json"
        client = ScriptedClient(responses)
        graph = build_sql_graph(client=client, model_name="test-model")
        with patch("src.bird_evaluator.build_sql_graph", return_value=graph):
            output = run_bird_evaluation(
                str(self.dev_path), str(self.db_dir), str(self.output_path), limit=limit
            )
        self.assertEqual(output, str(self.output_path))
        predictions = json.loads(self.output_path.read_text(encoding="utf-8"))
        return predictions, client

    def assert_score(self, report, total, correct):
        self.assertEqual(report["total"], total)
        self.assertEqual(report["correct"], correct)
        self.assertEqual(report["execution_accuracy"], correct / total * 100 if total else 0.0)
        self.assertEqual(len(report["results"]), total)

    def assert_fallback(self, prediction, db_id):
        self.assertIsNone(prediction)

    def test_full_pipeline_preserves_question_ids_and_scores_correct_sql(self):
        predictions, client = self.run_pipeline(
            [PLAN, generated_sql(NAMES_SQL), PLAN, generated_sql(COUNT_SQL)]
        )
        self.assertEqual(predictions, self.correct_predictions())
        self.assertEqual(list(predictions), ["7", "42"])
        self.assertEqual(len(client.calls), 4)
        prompts = json.dumps([call["messages"] for call in client.calls], ensure_ascii=False)
        self.assertIn("Zoë", prompts)
        self.assertIn(self.items[0]["evidence"], prompts)
        self.assertIn("CREATE TABLE", prompts)
        self.assertNotIn(NAMES_SQL, prompts)
        self.assertNotIn(COUNT_SQL, prompts)
        report = self.evaluate()
        self.assert_score(report, 2, 2)
        self.assertEqual({row["question_id"] for row in report["results"]}, {"7", "42"})
        self.assertTrue(all(row["correct"] and row["error"] is None for row in report["results"]))

    def test_wrong_executable_query_is_incorrect(self):
        predictions = self.correct_predictions()
        predictions["7"] = "SELECT student_name FROM students WHERE age < 21" + BIRD_SEPARATOR + "school"
        self.write_predictions(predictions)
        report = self.evaluate()
        self.assert_score(report, 2, 1)
        self.assertFalse(report["results"][0]["correct"])
        self.assertTrue(report["results"][1]["correct"])

    def test_missing_and_malformed_predictions_stay_in_denominator(self):
        for invalid_prediction in (None, "SELECT 1", "SELECT 1" + BIRD_SEPARATOR + "wrong_db"):
            with self.subTest(prediction=invalid_prediction):
                predictions = self.correct_predictions()
                if invalid_prediction is None:
                    del predictions["7"]
                else:
                    predictions["7"] = invalid_prediction
                self.write_predictions(predictions)
                report = self.evaluate()
                self.assert_score(report, 2, 1)
                self.assertFalse(report["results"][0]["correct"])
                self.assertTrue(report["results"][0]["error"])

    def test_limit_selects_same_items_for_generation_and_scoring(self):
        for limit in (1, 0):
            with self.subTest(limit=limit):
                responses = [PLAN, generated_sql(NAMES_SQL)] if limit else []
                predictions, client = self.run_pipeline(responses, limit=limit)
                expected = {"7": self.correct_predictions()["7"]} if limit else {}
                self.assertEqual(predictions, expected)
                self.assertEqual(len(client.calls), 2 * limit)
                self.assert_score(self.evaluate(limit=limit), limit, limit)

    def test_exception_or_exhaustion_emits_null_and_next_item_still_runs(self):
        failures = [
            [RuntimeError("Scripted model transport failure")],
            [PLAN] + [generated_sql("SELECT nonexistent_column FROM students", "probe")] * 5,
        ]
        for failed_responses in failures:
            with self.subTest(failure=type(failed_responses[0]).__name__):
                responses = failed_responses + [PLAN, generated_sql(COUNT_SQL)]
                predictions, client = self.run_pipeline(responses)
                self.assert_fallback(predictions["7"], "school")
                self.assertEqual(predictions["42"], self.correct_predictions()["42"])
                self.assertEqual(len(client.calls), len(responses))
                self.assert_score(self.evaluate(), 2, 1)

    def test_missing_database_does_not_get_created_and_next_item_runs(self):
        self.items[0]["db_id"] = "missing_school"
        self.write_dataset()
        missing_path = self.db_dir / "missing_school" / "missing_school.sqlite"
        missing_path.parent.mkdir()

        predictions, client = self.run_pipeline([PLAN, generated_sql(COUNT_SQL)])
        self.assertFalse(missing_path.exists())
        self.assert_fallback(predictions["7"], "missing_school")
        self.assertEqual(predictions["42"], self.correct_predictions()["42"])
        self.assertEqual(len(client.calls), 2)
        report = self.evaluate()
        self.assert_score(report, 2, 1)
        self.assertTrue(report["results"][0]["error"])
        self.assertFalse(missing_path.exists())

    def test_execution_scoring_rejects_mutation_without_changing_database(self):
        predictions = self.correct_predictions()
        predictions["7"] = "DELETE FROM students" + BIRD_SEPARATOR + "school"
        self.write_predictions(predictions)
        report = self.evaluate()
        self.assert_score(report, 2, 1)
        self.assertTrue(report["results"][0]["error"])
        with closing(sqlite3.connect(self.db_path)) as connection:
            rows = connection.execute("SELECT student_name FROM students ORDER BY student_id").fetchall()
        self.assertEqual(rows, [("Ada",), ("Zoë",), ("Linus",)])

    def test_execution_accuracy_ignores_row_order_and_duplicate_multiplicity(self):
        predictions = self.correct_predictions()
        predictions["7"] = (
            "SELECT student_name FROM students UNION ALL SELECT student_name FROM students "
            "ORDER BY student_name DESC" + BIRD_SEPARATOR + "school"
        )
        self.write_predictions(predictions)
        self.assert_score(self.evaluate(), 2, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
