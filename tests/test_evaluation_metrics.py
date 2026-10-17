"""Exercise persisted accounting and EX using temporary SQLite fixtures."""

from contextlib import closing, redirect_stdout
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("DSPY_CACHEDIR", str(Path(__file__).resolve().parents[1] / ".dspy_cache"))

from src.bird_evaluator import BIRD_SEPARATOR, compute_execution_accuracy, run_bird_evaluation


class EvaluationMetricsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="mars_evaluation_metrics_")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.db_dir = self.root / "dev_databases"
        db_path = self.db_dir / "users" / "users.sqlite"
        db_path.parent.mkdir(parents=True)
        with closing(sqlite3.connect(db_path)) as connection:
            connection.executescript(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);"
                "INSERT INTO users VALUES (1, 'Alice', 30), (2, 'Charlie', 35);"
            )
        self.items = [
            {"question_id": 7, "db_id": "users", "question": "Who is the oldest user?",
             "SQL": "SELECT name FROM users ORDER BY age DESC LIMIT 1"},
            {"question_id": 42, "db_id": "users", "question": "How many users are there?",
             "SQL": "SELECT COUNT(*) FROM users"},
        ]
        self.dev_path = self.root / "dev.json"
        self.output_path = self.root / "predict.json"
        self.metrics_path = self.output_path.with_suffix(".metrics.json")
        self.write_dataset()

    def write_dataset(self):
        self.dev_path.write_text(json.dumps(self.items), encoding="utf-8")

    def measured(self, *, prompt=721, completion=24, cost=0.0001414):
        return {
            "prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion, "cost_usd": cost,
            "observed_prompt_tokens": prompt, "observed_completion_tokens": completion,
            "observed_total_tokens": prompt + completion, "observed_cost_usd": cost,
            "llm_latency_seconds": 1.2, "latency_seconds": 1.3,
            "turn": 1, "sql_attempts": 1, "self_correction_cycles": 0,
            "accounting_complete": True,
            "stages": [{"stage": "generator", "turn": 1, "prompt_tokens": prompt,
                        "completion_tokens": completion, "cost_usd": cost}],
            "trace_id": "example-trace-id",
        }

    def run_with_responses(self, responses, *, limit=None, metrics_path=None):
        invoke = Mock(side_effect=responses)
        with patch("src.bird_evaluator.build_sql_graph", return_value=SimpleNamespace(invoke=invoke)) as factory:
            returned = run_bird_evaluation(
                str(self.dev_path), str(self.db_dir), str(self.output_path), limit,
                metrics_path=str(metrics_path) if metrics_path else None,
            )
        self.assertEqual(returned, str(self.output_path))
        report = json.loads((metrics_path or self.metrics_path).read_text(encoding="utf-8"))
        return report, invoke, factory

    def test_reports_exact_metrics_per_query_and_total_with_execution_accuracy(self):
        report, invoke, _ = self.run_with_responses([
            {"final_sql": item["SQL"], "metrics": self.measured()} for item in self.items
        ])
        self.assertEqual(report["execution_accuracy"], 100.0)
        self.assertEqual((report["correct"], report["total"]), (2, 2))
        self.assertEqual([query["question_id"] for query in report["queries"]], ["7", "42"])
        first = report["queries"][0]
        self.assertEqual(first["prompt_tokens"], 721)
        self.assertEqual(first["completion_tokens"], 24)
        self.assertEqual(first["trace_id"], "example-trace-id")
        self.assertFalse(first["fallback_used"])
        self.assertTrue(first["execution_correct"])
        self.assertGreaterEqual(first["end_to_end_latency_seconds"], first["pruning_latency_seconds"])
        self.assertEqual(report["totals"]["prompt_tokens"], 1442)
        self.assertEqual(report["totals"]["completion_tokens"], 48)
        self.assertEqual(report["totals"]["total_tokens"], 1490)
        self.assertAlmostEqual(report["totals"]["cost_usd"], 0.0002828)
        self.assertTrue(report["totals"]["accounting_complete"])
        predictions = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(predictions["7"], self.items[0]["SQL"] + BIRD_SEPARATOR + "users")
        self.assertNotIn("SQL", invoke.call_args_list[0].args[0])

    def test_errors_preserve_partial_accounting_and_remain_in_ex_denominator(self):
        failure = RuntimeError("Provider request failed on retry")
        failure.mars_metrics = {
            **self.measured(), "prompt_tokens": None, "completion_tokens": None,
            "total_tokens": None, "cost_usd": None, "accounting_complete": False,
            "self_correction_cycles": 1,
        }
        report, _, _ = self.run_with_responses([
            failure, {"final_sql": self.items[1]["SQL"], "metrics": self.measured()},
        ])
        self.assertEqual((report["correct"], report["total"]), (1, 2))
        self.assertEqual(report["execution_accuracy"], 50.0)
        failed = report["queries"][0]
        self.assertTrue(failed["fallback_used"])
        self.assertIn("Provider request failed", failed["error"])
        self.assertEqual(failed["observed_prompt_tokens"], 721)
        self.assertEqual(failed["self_correction_cycles"], 1)
        self.assertIsNone(report["totals"]["prompt_tokens"])
        self.assertIsNone(report["totals"]["cost_usd"])
        self.assertEqual(report["totals"]["observed_prompt_tokens"], 1442)
        self.assertEqual(report["totals"]["fallback_count"], 1)

    def test_missing_database_retains_unknown_metrics_without_creating_database(self):
        self.items[0]["db_id"] = "missing"
        self.write_dataset()
        report, invoke, _ = self.run_with_responses([
            {"final_sql": self.items[1]["SQL"], "metrics": self.measured()},
        ])
        self.assertEqual(invoke.call_count, 1)
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["execution_accuracy"], 50.0)
        self.assertIsNone(report["queries"][0]["prompt_tokens"])
        self.assertIsNone(report["queries"][0]["turn"])
        self.assertFalse(report["queries"][0]["accounting_complete"])
        self.assertEqual(report["totals"]["observed_prompt_tokens"], 721)
        self.assertIsNone(report["totals"]["prompt_tokens"])
        self.assertFalse((self.db_dir / "missing" / "missing.sqlite").exists())

    def test_empty_limit_writes_empty_complete_report_without_model_initialization(self):
        report, invoke, factory = self.run_with_responses([], limit=0)
        factory.assert_not_called()
        invoke.assert_not_called()
        self.assertEqual(report["queries"], [])
        self.assertEqual(report["total"], 0)
        self.assertEqual(report["execution_accuracy"], 0.0)
        self.assertEqual(report["totals"]["prompt_tokens"], 0)
        self.assertEqual(report["totals"]["cost_usd"], 0)
        self.assertTrue(report["totals"]["accounting_complete"])

    def test_subset_custom_path_and_failure_cannot_match_select_one_gold(self):
        self.items[0]["SQL"] = "SELECT 1"
        self.write_dataset()
        custom_path = self.root / "reports" / "costs.json"
        report, invoke, _ = self.run_with_responses([
            {"final_sql": None, "metrics": self.measured()},
        ], limit=1, metrics_path=custom_path)
        self.assertEqual(invoke.call_count, 1)
        self.assertFalse(self.metrics_path.exists())
        self.assertEqual((report["correct"], report["total"]), (0, 1))
        self.assertTrue(report["queries"][0]["fallback_used"])
        self.assertFalse(report["queries"][0]["execution_correct"])
        self.assertEqual(report["totals"]["fallback_count"], 1)

    def test_real_dspy_graph_predictions_are_still_scored_by_sql_execution(self):
        import dspy
        from dspy.utils.dummies import DummyLM
        from src.dspy_modules import TextToSQLProgram, configure_dspy
        from src.dspy_telemetry import OpenRouterLM
        from src.sql_agent_graph import build_sql_graph

        self.addCleanup(dspy.configure, lm=dspy.settings.lm, adapter=dspy.settings.adapter)
        scripted = DummyLM([
            {"logical_plan": "Return the oldest user's name."},
            {"action": "final_answer", "rationale": "Return the requested name.", "sql_query": self.items[0]["SQL"]},
            {"logical_plan": "Count all users."},
            {"action": "final_answer", "rationale": "Intentionally wrong executable answer.", "sql_query": "SELECT 99"},
        ])
        lm = OpenRouterLM(model="openai/test-model", cache=False)
        configure_dspy(lm=lm)

        def respond(prompt=None, messages=None, **kwargs):
            response = scripted.forward(prompt=prompt, messages=messages, **kwargs)
            response.usage.update(prompt_tokens=12, completion_tokens=4, total_tokens=16, cost=0.0001)
            return response

        graph = build_sql_graph(dspy_program=TextToSQLProgram())
        with (
            patch("src.bird_evaluator.build_sql_graph", return_value=graph),
            patch.object(dspy.LM, "forward", side_effect=respond),
        ):
            run_bird_evaluation(str(self.dev_path), str(self.db_dir), str(self.output_path))
        report = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(report["execution_accuracy"], 50.0)
        self.assertEqual([query["execution_correct"] for query in report["queries"]], [True, False])
        self.assertFalse(any(query["fallback_used"] for query in report["queries"]))
        self.assertTrue(report["totals"]["accounting_complete"])
        self.assertEqual(report["totals"]["prompt_tokens"], 48)
        self.assertEqual(report["totals"]["completion_tokens"], 16)
        self.assertAlmostEqual(report["totals"]["cost_usd"], 0.0004)
        self.assertEqual([query["turn"] for query in report["queries"]], [1, 1])
        separate_score = compute_execution_accuracy(str(self.dev_path), str(self.db_dir), str(self.output_path))
        self.assertEqual(report["results"], separate_score["results"])

    def test_live_script_retains_accounting_and_verifies_answer_before_success(self):
        import live_test

        output = io.StringIO()
        invoke = Mock(return_value={
            "final_sql": "SELECT name FROM users ORDER BY age DESC LIMIT 1",
            "metrics": self.measured(),
        })
        with (
            patch.object(live_test, "__file__", str(self.root / "live_test.py")),
            patch("src.bird_evaluator.build_sql_graph", return_value=SimpleNamespace(invoke=invoke)),
            redirect_stdout(output),
        ):
            live_test.run_live_trace()
        metrics_files = list((self.root / "artifacts").glob("live_trace_*/predict.metrics.json"))
        self.assertEqual(len(metrics_files), 1)
        report = json.loads(metrics_files[0].read_text(encoding="utf-8"))
        self.assertEqual(report["execution_accuracy"], 100.0)
        self.assertEqual(report["queries"][0]["question_id"], "999")
        self.assertIn("prompt=721", output.getvalue())
        self.assertIn("completion=24", output.getvalue())
        self.assertIn("Success! The generated SQL returns Charlie.", output.getvalue())
        self.assertFalse(Path(invoke.call_args.args[0]["db_path"]).exists())

    def test_live_script_raises_for_wrong_sql_and_preserves_failure_report(self):
        import live_test

        output = io.StringIO()
        invoke = Mock(return_value={"final_sql": "SELECT 1", "metrics": self.measured()})
        with (
            patch.object(live_test, "__file__", str(self.root / "live_test.py")),
            patch("src.bird_evaluator.build_sql_graph", return_value=SimpleNamespace(invoke=invoke)),
            redirect_stdout(output),
            self.assertRaisesRegex(RuntimeError, "incorrect SQL result"),
        ):
            live_test.run_live_trace()
        self.assertNotIn("Success!", output.getvalue())
        metrics_files = list((self.root / "artifacts").glob("live_trace_*/predict.metrics.json"))
        self.assertEqual(len(metrics_files), 1)
        report = json.loads(metrics_files[0].read_text(encoding="utf-8"))
        self.assertEqual(report["execution_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
