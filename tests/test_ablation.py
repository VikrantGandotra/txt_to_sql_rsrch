"""Parallel rollout, label-free selection and complete compute accounting checks."""

from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from threading import Barrier, Lock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("DSPY_CACHEDIR", str(Path(__file__).resolve().parents[1] / ".dspy_cache"))

from src.bird_evaluator import (
    BIRD_SEPARATOR, _execute_candidates, _result_fingerprint,
    compute_execution_accuracy, run_bird_evaluation, select_self_consistency,
)
from src.graph_pruner import DatabaseGraphPruner
from src.sql_agent_graph import build_sql_graph


def measured(prompt=10, cost=0.001):
    return {
        "prompt_tokens": prompt, "completion_tokens": 2, "total_tokens": prompt + 2,
        "cost_usd": cost, "observed_prompt_tokens": prompt,
        "observed_completion_tokens": 2, "observed_total_tokens": prompt + 2,
        "observed_cost_usd": cost, "accounting_complete": True,
        "llm_latency_seconds": 0.1, "latency_seconds": 0.1, "turn": 1,
        "sql_attempts": 1, "sql_execution_count": 1, "self_correction_cycles": 0,
        "stages": [],
    }


class AblationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="mars_ablation_")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.db_dir = self.root / "databases"
        self.db = self.db_dir / "users" / "users.sqlite"
        self.db.parent.mkdir(parents=True)
        with closing(sqlite3.connect(self.db)) as connection:
            connection.executescript("CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT, age INTEGER);"
                                     "INSERT INTO users VALUES(1,'Alice',30),(2,'Charlie',35);")
        self.items = [{"question_id": 999, "db_id": "users",
                       "question": "What is the name of the oldest user?", "evidence": "Sort age descending.",
                       "SQL": "SELECT name FROM users ORDER BY age DESC LIMIT 1"}]
        self.dev = self.root / "dev.json"
        self.dev.write_text(json.dumps(self.items), encoding="utf-8")
        self.output = self.root / "predict.json"

    def run_evaluation(self, **kwargs):
        run_bird_evaluation(str(self.dev), str(self.db_dir), str(self.output), **kwargs)
        return json.loads(self.output.with_suffix(".metrics.json").read_text(encoding="utf-8"))

    def test_eight_real_graphs_run_in_parallel_with_independent_probe_memory(self):
        barrier = Barrier(8, timeout=20)
        lock = Lock()
        client_calls = {}
        def factory(**configuration):
            self.assertEqual(configuration["model_name"], "openrouter/qwen/qwen-2.5-7b-instruct")
            self.assertEqual((configuration["temperature"], configuration["top_p"]), (0.8, 0.7))
            self.assertTrue(configuration["formal"])
            index = len(client_calls)
            calls = client_calls[index] = []
            def create(**request):
                with lock:
                    calls.append(request)
                if len(calls) == 1:
                    barrier.wait()  # Sequential orchestration would time out here.
                    payload = {"target_tables": ["users"], "target_columns": ["age", "name"],
                               "reasoning": f"Independent plan {index}"}
                elif len(calls) == 2:
                    payload = {"action": "probe", "rationale": "Inspect an age threshold.",
                               "sql_query": f"SELECT AVG(age) AS threshold_{index} FROM users"}
                else:
                    self.assertEqual(len(calls), 3)
                    prompt = json.dumps(request["messages"])
                    self.assertIn(f"threshold_{index}", prompt)
                    self.assertIn("32.5", prompt)
                    for sibling in set(range(8)) - {index}:
                        self.assertNotIn(f"threshold_{sibling}", prompt)
                    payload = {"action": "final_answer", "rationale": "Submit the oldest name.",
                               "sql_query": self.items[0]["SQL"]}
                return SimpleNamespace(model="test-model", usage={"prompt_tokens": 10, "completion_tokens": 2,
                                       "total_tokens": 12, "cost": 0.001},
                                       choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            return build_sql_graph(client=client, model_name="test-model")
        original_prune = DatabaseGraphPruner.prune
        with patch("src.bird_evaluator.build_sql_graph", side_effect=factory), patch.object(
            DatabaseGraphPruner, "prune", autospec=True, side_effect=original_prune,
        ) as prune:
            report = self.run_evaluation(num_candidates=8)
        self.assertEqual(prune.call_count, 1)
        self.assertEqual(report["execution_accuracy"], 100)
        query = report["queries"][0]
        self.assertEqual(query["selected_candidate_index"], 0)
        self.assertEqual([row["candidate_index"] for row in query["candidates"]], list(range(8)))
        self.assertEqual({row["rollout_seed"] for row in query["candidates"]}, set(range(8)))
        self.assertEqual([row["turn"] for row in query["candidates"]], [2] * 8)
        self.assertEqual(query["self_correction_cycles"], 0)
        self.assertEqual(query["turn"], 16)
        self.assertEqual(query["prompt_tokens"], 240)
        self.assertAlmostEqual(query["cost_usd"], 0.024)
        self.assertEqual(query["selection"]["extra_llm_calls"], 0)

    def test_vote_uses_complete_results_and_handles_empty_blobs_and_numeric_equivalence(self):
        candidates = [
            {"candidate_index": 0, "final_sql": "SELECT 1"},
            {"candidate_index": 1, "final_sql": "SELECT 1.0 UNION ALL SELECT 1"},
            {"candidate_index": 2, "final_sql": "SELECT 2"},
            {"candidate_index": 3, "final_sql": "SELECT name FROM users WHERE 0"},
            {"candidate_index": 4, "final_sql": "SELECT age FROM users WHERE 0"},
            {"candidate_index": 5, "final_sql": "DELETE FROM users"},
        ]
        _execute_candidates(candidates, self.db, 30)
        selection = select_self_consistency(candidates)
        self.assertEqual(selection["selected_candidate_index"], 0)  # 2-to-2 tie: earliest.
        self.assertIsNone(candidates[5]["final_sql"])
        self.assertEqual(candidates[3]["execution"]["row_count"], 0)
        self.assertEqual(_result_fingerprint({(1,), (1.0,)}), _result_fingerprint({(1.0,)}))
        self.assertNotEqual(_result_fingerprint({(b"ab",)}), _result_fingerprint({("6162",)}))
        prefix = {(i,) for i in range(60)}
        self.assertNotEqual(_result_fingerprint(prefix), _result_fingerprint(prefix | {(99,)}))

    def test_dspy_selection_reports_same_pool_baseline_and_all_candidate_cost(self):
        states = []
        def invoke(state):
            states.append(state)
            self.assertNotIn("SQL", state)
            index = state["candidate_index"]
            sql = "SELECT name FROM users WHERE name='Alice'" if index < 5 else self.items[0]["SQL"]
            return {"final_sql": sql, "metrics": measured(), "turn": 1, "history": []}
        def select(question, evidence, schema, candidates, **kwargs):
            self.assertEqual(len(candidates), 8)
            for candidate in candidates:
                self.assertNotIn("execution_correct", candidate)
                self.assertNotIn("gold_sql", candidate)
            return {"selected_candidate_index": 5, "mode": "dspy_verifier", "metrics": measured(20, 0.002)}
        artifact = self.root / "judge.json"
        artifact.write_text("{}", encoding="utf-8")
        with patch("src.bird_evaluator.build_sql_graph", return_value=SimpleNamespace(invoke=invoke)), \
             patch("src.dspy_verifier.load_compiled_verifier", return_value=object()), \
             patch("src.dspy_verifier.select_with_dspy", side_effect=select):
            report = self.run_evaluation(num_candidates=8, selection_mode="dspy_verifier", verifier_path=str(artifact))
        self.assertEqual(len(states), 8)
        self.assertEqual(report["execution_accuracy"], 100)
        self.assertEqual(report["self_consistency"]["execution_accuracy"], 0)
        self.assertEqual(report["self_consistency"]["totals"]["prompt_tokens"], 80)
        self.assertAlmostEqual(report["self_consistency"]["totals"]["cost_usd"], 0.008)
        self.assertEqual(report["totals"]["prompt_tokens"], 100)
        self.assertAlmostEqual(report["totals"]["cost_usd"], 0.010)
        self.assertEqual(report["queries"][0]["turn"], 8)

    def test_failed_rollout_and_unexpected_selector_failure_retain_billed_usage(self):
        failure = RuntimeError("Transport failed after a probe")
        failure.mars_metrics = measured()
        failure.mars_state = {"history": [{"action": "probe", "sql": "SELECT 1", "observation": "{}"}], "turn": 1}
        invoke = Mock(side_effect=[failure, {"final_sql": "SELECT 1", "metrics": measured(), "turn": 1}])
        artifact = self.root / "judge.json"
        artifact.write_text("{}", encoding="utf-8")
        with patch("src.bird_evaluator.build_sql_graph", return_value=SimpleNamespace(invoke=invoke)), \
             patch("src.dspy_verifier.load_compiled_verifier", return_value=object()), \
             patch("src.dspy_verifier.select_with_dspy", side_effect=RuntimeError("Tracing failed")):
            report = self.run_evaluation(num_candidates=2, selection_mode="dspy_verifier", verifier_path=str(artifact))
        self.assertIsNone(report["totals"]["prompt_tokens"])
        self.assertEqual(report["totals"]["observed_prompt_tokens"], 20)
        self.assertEqual(report["queries"][0]["candidates"][0]["history"][0]["action"], "probe")
        self.assertIsNone(json.loads(self.output.read_text())["999"])
        self.assertEqual(report["execution_accuracy"], 0)

    def test_ex_shares_one_thirty_second_deadline_between_prediction_and_gold(self):
        self.output.write_text(json.dumps({"999": self.items[0]["SQL"] + BIRD_SEPARATOR + "users"}))
        with patch("src.bird_evaluator._execute_result_set", return_value={("Charlie",)}) as execute:
            score = compute_execution_accuracy(str(self.dev), str(self.db_dir), str(self.output))
        first, second = execute.call_args_list
        self.assertEqual(first.args[2], 30)
        self.assertEqual(first.kwargs["deadline"], second.kwargs["deadline"])
        self.assertEqual(score["execution_accuracy"], 100)

    def test_outputs_cannot_overwrite_dataset_database_or_compiled_artifacts(self):
        protected = [self.dev, self.db]
        for target in protected:
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "distinct"):
                run_bird_evaluation(str(self.dev), str(self.db_dir), str(target), limit=1)
        artifact = self.root / "compiled.json"
        artifact.write_text("preserve", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact"):
            run_bird_evaluation(str(self.dev), str(self.db_dir), str(artifact), limit=1,
                                compiled_model_path=str(artifact), prompt_mode="compiled_dspy")
        self.assertEqual(artifact.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main(verbosity=2)
