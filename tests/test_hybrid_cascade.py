"""Offline checks for execution consensus followed by a tie-only DSPy rating judge."""

from contextlib import closing, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("DSPY_CACHEDIR", str(Path(__file__).resolve().parents[1] / ".dspy_cache"))

from src import bird_evaluator
import run_ablation


def measured(prompt=10, cost=0.001, *, calls=1):
    return {
        "prompt_tokens": prompt, "completion_tokens": 2, "total_tokens": prompt + 2,
        "cost_usd": cost, "observed_prompt_tokens": prompt,
        "observed_completion_tokens": 2, "observed_total_tokens": prompt + 2,
        "observed_cost_usd": cost, "accounting_complete": True,
        "llm_latency_seconds": 0.1, "latency_seconds": 0.1, "turn": 1,
        "sql_attempts": 1, "sql_execution_count": 1, "self_correction_cycles": 0,
        "llm_call_count": calls, "stages": [],
    }


def candidate(index, result_hash, *, sql="SELECT 1", error=None):
    return {
        "candidate_index": index, "final_sql": sql,
        "execution": {"result_hash": result_hash, "error": error},
        "history": [], "metrics": measured(),
    }


class HybridCascadeRoutingTests(unittest.TestCase):
    def select(self, candidates, judge):
        return bird_evaluator.select_hybrid_cascade(
            "Which user is oldest?", "Sort age descending.", "CREATE TABLE users(age INTEGER);",
            candidates, judge=judge,
        )

    def test_unique_top_group_needs_no_judge_even_without_absolute_majority(self):
        candidates = [candidate(index, group) for index, group in enumerate("AAABBCCD")]
        judge = Mock(side_effect=AssertionError("A unique top group must not call the judge"))
        selection = self.select(candidates, judge)
        judge.assert_not_called()
        self.assertEqual(selection["mode"], "hybrid_cascade")
        self.assertEqual(selection["selected_candidate_index"], 0)
        self.assertFalse(selection["cascade_tie_breaker_engaged"])
        self.assertEqual(selection["selection_source"], "self_consistency")
        self.assertEqual(selection["extra_llm_calls"], 0)
        self.assertEqual(sorted(map(len, selection["groups"].values())), [1, 2, 2, 3])

    def test_tie_passes_only_top_group_members_with_original_indices_and_order(self):
        candidates = [
            candidate(20, "A"), candidate(9, "C"), candidate(3, "B"),
            candidate(11, "A"), candidate(6, "B"), candidate(25, "D"),
            candidate(7, "A", error="SQLite failed"), candidate(4, "A", sql=None),
        ]
        judge_metrics = measured(20, 0.002, calls=4)
        judge = Mock(return_value={
            "mode": "dspy_verifier", "selected_candidate_index": 11,
            "score_kind": "elicited_validity_rating", "metrics": judge_metrics,
        })
        selection = self.select(candidates, judge)
        judged = judge.call_args.args[3]
        self.assertEqual([item["candidate_index"] for item in judged], [20, 3, 11, 6])
        self.assertEqual(judge.call_args.kwargs["verification_samples"], 1)
        self.assertEqual(selection["selected_candidate_index"], 11)
        self.assertEqual(set(selection["tied_candidate_indices"]), {20, 3, 11, 6})
        self.assertTrue(selection["cascade_tie_breaker_engaged"])
        self.assertEqual(selection["selection_source"], "dspy_verifier")
        self.assertEqual(selection["mode"], "hybrid_cascade")
        self.assertEqual(selection["metrics"], judge_metrics)
        self.assertEqual(selection["extra_llm_calls"], 4)

    def test_all_singleton_groups_are_judged_and_empty_execution_results_can_vote(self):
        candidates = [candidate(8, "empty"), candidate(2, "nonempty"), candidate(5, "other")]
        candidates[0]["execution"].update(row_count=0, sample_rows=[])
        judge = Mock(return_value={"selected_candidate_index": 8, "metrics": measured(calls=3)})
        selection = self.select(candidates, judge)
        self.assertEqual(judge.call_args.args[3], candidates)
        self.assertEqual(selection["selected_candidate_index"], 8)
        self.assertTrue(selection["cascade_tie_breaker_engaged"])

    def test_no_eligible_candidates_returns_no_selection_without_judge(self):
        judge = Mock()
        for candidates in (
            [],
            [candidate(0, "A", sql=None), candidate(1, "B", error="failed"),
             {"candidate_index": 2, "final_sql": "SELECT 1"}],
        ):
            with self.subTest(candidates=candidates):
                selection = self.select(candidates, judge)
                self.assertIsNone(selection["selected_candidate_index"])
                self.assertFalse(selection["cascade_tie_breaker_engaged"])
                self.assertEqual(selection["groups"], {})
        judge.assert_not_called()

    def test_judge_failure_retains_cascade_routing_metadata_on_exception(self):
        failure = RuntimeError("Invalid DSPy rating")
        failure.mars_metrics = measured(20, 0.002)
        with self.assertRaises(RuntimeError) as caught:
            self.select([candidate(3, "A"), candidate(8, "B")], Mock(side_effect=failure))
        self.assertIs(caught.exception, failure)
        routing = caught.exception.mars_selection
        self.assertEqual(routing["mode"], "hybrid_cascade")
        self.assertTrue(routing["cascade_tie_breaker_engaged"])
        self.assertIsNone(routing["selected_candidate_index"])
        self.assertEqual(set(routing["tied_candidate_indices"]), {3, 8})
        self.assertEqual(caught.exception.mars_metrics["cost_usd"], 0.002)

    def test_judge_cannot_select_candidate_outside_tied_top_groups(self):
        candidates = [candidate(index, group) for index, group in enumerate("AABBC")]
        judge = Mock(return_value={"selected_candidate_index": 4, "metrics": measured()})
        with self.assertRaises((ValueError, RuntimeError)) as caught:
            self.select(candidates, judge)
        self.assertTrue(caught.exception.mars_selection["cascade_tie_breaker_engaged"])
        self.assertIsNone(caught.exception.mars_selection["selected_candidate_index"])


class HybridCascadeBatchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="mars_hybrid_")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.db_dir = self.root / "databases"
        self.db = self.db_dir / "users" / "users.sqlite"
        self.db.parent.mkdir(parents=True)
        with closing(sqlite3.connect(self.db)) as connection:
            connection.executescript(
                "CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT, age INTEGER);"
                "INSERT INTO users VALUES(1,'Alice',30),(2,'Bob',25),"
                "(3,'Charlie',35),(4,'Dora',10);"
            )
        self.dev = self.root / "dev.json"
        self.dev.write_text(json.dumps([{
            "question_id": 999, "db_id": "users", "question": "Which user is oldest?",
            "evidence": "Sort age descending.",
            "SQL": "SELECT name FROM users ORDER BY age DESC LIMIT 1 /* GOLD_ONLY_SENTINEL */",
        }]), encoding="utf-8")
        self.output = self.root / "predict.json"

    def graph_for(self, names):
        def invoke(state):
            self.assertNotIn("SQL", state)
            return {
                "final_sql": f"SELECT name FROM users WHERE name='{names[state['candidate_index']]}'",
                "metrics": measured(), "turn": 1, "history": [],
            }
        return SimpleNamespace(invoke=invoke)

    def run_evaluation(self, names, judge, **kwargs):
        artifact = self.root / "judge.json"
        artifact.write_text("{}", encoding="utf-8")
        kwargs.setdefault("verifier_path", str(artifact))
        with patch("src.bird_evaluator.build_sql_graph", return_value=self.graph_for(names)), \
             patch("src.dspy_verifier.load_compiled_verifier", return_value=object()), \
             patch("src.dspy_verifier.select_with_dspy", side_effect=judge):
            bird_evaluator.run_bird_evaluation(
                str(self.dev), str(self.db_dir), str(self.output), num_candidates=len(names),
                selection_mode="hybrid_cascade", **kwargs,
            )
        return json.loads(self.output.with_suffix(".metrics.json").read_text(encoding="utf-8"))

    def test_batch_tie_logs_mode_routing_paired_baseline_and_all_candidate_cost(self):
        def judge(question, evidence, schema, candidates, **kwargs):
            self.assertEqual([item["candidate_index"] for item in candidates], list(range(6)))
            serialized = json.dumps(candidates)
            for label in ("GOLD_ONLY_SENTINEL", "gold_sql", "execution_correct"):
                self.assertNotIn(label, serialized)
            return {
                "mode": "dspy_verifier", "selected_candidate_index": 3,
                "score_kind": "elicited_validity_rating", "metrics": measured(20, 0.002, calls=6),
            }
        report = self.run_evaluation(["Alice"] * 3 + ["Charlie"] * 3 + ["Bob"] * 2, judge)
        query = report["queries"][0]
        self.assertEqual(report["manifest"]["selection_mode"], "hybrid_cascade")
        self.assertEqual(query["selection"]["mode"], "hybrid_cascade")
        self.assertTrue(query["selection"]["cascade_tie_breaker_engaged"])
        self.assertEqual(query["selected_candidate_index"], 3)
        self.assertEqual(report["execution_accuracy"], 100)
        self.assertEqual(report["self_consistency"]["execution_accuracy"], 0)
        self.assertEqual(report["self_consistency"]["totals"]["prompt_tokens"], 80)
        self.assertEqual(report["totals"]["prompt_tokens"], 100)
        self.assertAlmostEqual(report["totals"]["cost_usd"], 0.010)
        self.assertAlmostEqual(query["generation_cost_usd"], 0.008)
        self.assertAlmostEqual(query["selection_cost_usd"], 0.002)
        self.assertEqual(query["turn"], 8)
        self.assertEqual(len(query["candidates"]), 8)

    def test_batch_unique_top_group_uses_no_selector_tokens_or_cost(self):
        judge = Mock(side_effect=AssertionError("No judge should be initialized"))
        report = self.run_evaluation(["Charlie"] * 3 + ["Alice"] * 2 + ["Bob"] * 2 + ["Dora"], judge)
        judge.assert_not_called()
        query = report["queries"][0]
        self.assertFalse(query["selection"]["cascade_tie_breaker_engaged"])
        self.assertEqual(report["execution_accuracy"], 100)
        self.assertEqual(report["totals"]["prompt_tokens"], 80)
        self.assertAlmostEqual(report["totals"]["cost_usd"], 0.008)
        self.assertEqual(query["selection_cost_usd"], 0)
        self.assertEqual(report["self_consistency"]["execution_accuracy"], 100)

    def test_failed_tie_judge_preserves_engaged_flag_and_unknown_accounting(self):
        report = self.run_evaluation(["Alice", "Charlie"], Mock(side_effect=RuntimeError("Judge transport failed")))
        query = report["queries"][0]
        self.assertTrue(query["selection"]["cascade_tie_breaker_engaged"])
        self.assertEqual(query["selection"]["mode"], "hybrid_cascade")
        self.assertTrue(query["selection_failed"])
        self.assertIsNone(json.loads(self.output.read_text(encoding="utf-8"))["999"])
        self.assertIsNone(report["totals"]["prompt_tokens"])
        self.assertEqual(report["totals"]["observed_prompt_tokens"], 20)
        self.assertAlmostEqual(report["totals"]["observed_cost_usd"], 0.002)
        self.assertFalse(report["totals"]["accounting_complete"])

    def test_invalid_rating_does_not_fall_back_to_consensus_and_keeps_billed_cost(self):
        judge = Mock(return_value={
            "mode": "dspy_verifier", "selected_candidate_index": None,
            "errors": [{"candidate_index": 1, "error": "Invalid DSPy rating"}],
            "metrics": measured(20, 0.002, calls=2),
        })
        report = self.run_evaluation(["Charlie", "Alice"], judge)
        query = report["queries"][0]
        self.assertTrue(query["selection"]["cascade_tie_breaker_engaged"])
        self.assertTrue(query["selection_failed"])
        self.assertIsNone(query["selected_candidate_index"])
        self.assertIsNone(json.loads(self.output.read_text(encoding="utf-8"))["999"])
        self.assertEqual(report["execution_accuracy"], 0)
        self.assertEqual(report["self_consistency"]["execution_accuracy"], 100)
        self.assertIn("Invalid DSPy rating", query["error"])
        self.assertEqual(report["totals"]["prompt_tokens"], 40)
        self.assertAlmostEqual(report["totals"]["cost_usd"], 0.004)
        self.assertTrue(report["totals"]["accounting_complete"])

    def test_cli_accepts_hybrid_and_routes_configuration_to_evaluator(self):
        def evaluate(*args, **kwargs):
            self.assertEqual(kwargs["selection_mode"], "hybrid_cascade")
            self.assertEqual(kwargs["num_candidates"], 8)
            self.assertEqual(kwargs["verification_samples"], 2)
            Path(args[2]).with_suffix(".metrics.json").write_text(json.dumps({
                "execution_accuracy": 100, "correct": 1, "total": 1,
                "totals": {"total_tokens": 10, "cost_usd": 0.001, "accounting_complete": True},
                "queries": [{"final_sql": "SELECT name FROM users"}],
            }), encoding="utf-8")
        with patch("run_ablation.run_bird_evaluation", side_effect=evaluate) as evaluate_mock, \
             redirect_stdout(StringIO()) as output:
            run_ablation.main([
                "--dev-json", str(self.dev), "--db-dir", str(self.db_dir),
                "--selection", "hybrid_cascade", "--verification-samples", "2",
                "--output", str(self.output), "--prompt-mode", "uncompiled_dspy",
            ])
        evaluate_mock.assert_called_once()
        self.assertIn("hybrid_cascade: EX=100.00%", output.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
