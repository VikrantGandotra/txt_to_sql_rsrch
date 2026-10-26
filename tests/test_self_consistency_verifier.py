"""Execution grouping followed by unconditional representative rating selection."""

from contextlib import closing, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import run_ablation
from src import bird_evaluator as evaluator
from tests.test_hybrid_cascade import measured, candidate


class ConsensusVerifierTests(unittest.TestCase):
    def judge(self, scores):
        return Mock(return_value={"selected_candidate_index": min(scores, key=lambda i: -scores[i]),
            "scores": [{"candidate_index": i, "score": score} for i, score in scores.items()],
            "errors": [], "metrics": dict(measured(20, 0.002, calls=len(scores)), lm_call_count=len(scores))})

    def select(self, candidates, judge):
        return evaluator.select_self_consistency_verifier("Q", "E", "DDL", candidates, judge=judge)

    def test_judges_all_groups_once_and_rating_can_overrule_majority(self):
        candidates = [candidate(i, group) for i, group in enumerate("AAAAABBC")]
        judge = self.judge({0: 0.1, 5: 0.9, 7: 0.5})
        result = self.select(candidates, judge)
        self.assertEqual([c["candidate_index"] for c in judge.call_args.args[3]], [0, 5, 7])
        self.assertEqual(result["selected_candidate_index"], 5)
        self.assertEqual(result["extra_llm_calls"], 3)
        self.assertEqual(result["mode"], "self_consistency_verifier")
        self.assertEqual(judge.call_args.args[:3], ("Q", "E", "DDL"))

    def test_equal_ratings_use_votes_then_original_candidate_index(self):
        candidates = [candidate(9, "A"), candidate(7, "B"), candidate(3, "B")]
        result = self.select(candidates, self.judge({9: 0.8, 3: 0.8}))
        self.assertEqual(result["selected_candidate_index"], 3)
        result = self.select([candidate(9, "A"), candidate(3, "B")], self.judge({9: 0.8, 3: 0.8}))
        self.assertEqual(result["selected_candidate_index"], 3)

    def test_unanimous_pool_still_judged_and_empty_pool_never_calls_judge(self):
        judge = self.judge({0: 0.5})
        self.assertTrue(self.select([candidate(0, "A"), candidate(1, "A")], judge)["judge_engaged"])
        judge.assert_called_once()
        judge.reset_mock()
        result = self.select([candidate(0, "A", sql=None)], judge)
        judge.assert_not_called()
        self.assertFalse(result["judge_engaged"])
        self.assertIsNone(result["selected_candidate_index"])

    def test_partial_failed_ranking_has_no_consensus_fallback(self):
        judge = Mock(return_value={"selected_candidate_index": None, "scores": [],
            "errors": [{"error": "Context limit"}], "metrics": measured()})
        result = self.select([candidate(0, "A")], judge)
        self.assertIsNone(result["selected_candidate_index"])
        self.assertEqual(result["errors"][0]["error"], "Context limit")

    def test_invalid_ranking_retains_accounting_and_group_diagnostics(self):
        judge = Mock(return_value={"selected_candidate_index": 0, "scores": [], "errors": [], "metrics": measured()})
        with self.assertRaises(ValueError) as caught:
            self.select([candidate(0, "A")], judge)
        self.assertEqual(caught.exception.mars_metrics["prompt_tokens"], 10)
        self.assertEqual(caught.exception.mars_selection["groups"], {"A": [0]})

    def test_both_prompt_modes_persist_all_six_dimensions_and_resume_without_rejudging(self):
        for prompt_mode in ("compiled_dspy", "uncompiled_dspy"):
            with self.subTest(prompt_mode=prompt_mode), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); db=root/"databases/users/users.sqlite"; db.parent.mkdir(parents=True)
                with closing(sqlite3.connect(db)) as connection:
                    connection.executescript("CREATE TABLE users(name TEXT); INSERT INTO users VALUES('Alice'),('Charlie');")
                dev=root/"dev.json"; output=root/"predict.json"; artifact=root/"compiled.json"
                dev.write_text(json.dumps([{"question_id": 7, "db_id": "users", "question": "Find Charlie",
                    "SQL": "SELECT name FROM users WHERE name='Charlie'"}]))
                artifact.write_text("{}")
                def invoke(state):
                    name="Alice" if state["candidate_index"]<6 else "Charlie"
                    return {"final_sql": f"SELECT name FROM users WHERE name='{name}'", "turn": 1,
                            "termination_reason": "final_answer", "metrics": measured()}
                judge=self.judge({0: 0.1, 6: 0.9})
                def verify(question,evidence,schema,candidates,**kwargs):
                    self.assertEqual([c["candidate_index"] for c in candidates], [0,6])
                    for forbidden in ('"gold_sql"', '"execution_correct"', '"correct"'):
                        self.assertNotIn(forbidden, json.dumps(candidates))
                    return judge(question,evidence,schema,candidates,**kwargs)
                with patch("src.bird_evaluator.build_sql_graph", return_value=SimpleNamespace(invoke=invoke)), \
                     patch("src.dspy_verifier.load_compiled_verifier", return_value=object()), \
                     patch("src.dspy_verifier.select_with_dspy", side_effect=verify), redirect_stdout(StringIO()):
                    kwargs=dict(num_candidates=8,selection_mode="self_consistency_verifier",prompt_mode=prompt_mode,
                                verifier_path=str(artifact),compiled_model_path=str(artifact) if prompt_mode=="compiled_dspy" else None)
                    evaluator.run_bird_evaluation(str(dev),str(root/"databases"),str(output),**kwargs)
                    before=output.with_suffix(".checkpoint.jsonl").read_bytes()
                    evaluator.run_bird_evaluation(str(dev),str(root/"databases"),str(output),resume=True,**kwargs)
                    self.assertEqual(output.with_suffix(".checkpoint.jsonl").read_bytes(),before)
                judge.assert_called_once()
                report=json.loads(output.with_suffix(".metrics.json").read_text()); q=report["queries"][0]
                self.assertEqual(report["execution_accuracy"],100)
                self.assertEqual(report["self_consistency"]["execution_accuracy"],0)
                self.assertEqual(q["final_sql"],"SELECT name FROM users WHERE name='Charlie'")
                self.assertEqual(q["status"],"correct")
                self.assertEqual(q["configuration"]["selection_mode"],"self_consistency_verifier")
                self.assertEqual(q["configuration"]["prompt_mode"],prompt_mode)
                self.assertEqual(q["total_rollout_turns"],8)
                self.assertEqual(q["selected_trajectory_turns"],1)
                self.assertEqual(report["totals"]["prompt_tokens"],100)
                self.assertEqual(report["totals"]["completion_tokens"],18)
                self.assertEqual(report["totals"]["total_tokens"],118)
                self.assertAlmostEqual(report["totals"]["cost_usd"],0.01)
                self.assertTrue(report["totals"]["accounting_complete"])

    def test_cli_accepts_new_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"predict.json"
            def evaluate(*args,**kwargs):
                self.assertEqual(kwargs["selection_mode"],"self_consistency_verifier")
                output.with_suffix(".metrics.json").write_text(json.dumps({"execution_accuracy":100,
                    "correct":1,"total":1,"queries":[{"final_sql":"SELECT 1"}],
                    "totals":{"total_tokens":1,"cost_usd":0,"accounting_complete":True}}))
            with patch("run_ablation.run_bird_evaluation",side_effect=evaluate),redirect_stdout(StringIO()):
                run_ablation.main(["--dev-json","unused","--db-dir","unused","--selection",
                    "self_consistency_verifier","--output",str(output)])


if __name__ == "__main__":
    unittest.main()
