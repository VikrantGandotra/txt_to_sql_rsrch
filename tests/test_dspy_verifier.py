"""Offline evidence for independent, label-free DSPy trajectory selection."""

import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("DSPY_CACHEDIR", str(Path(__file__).resolve().parents[1] / ".dspy_cache"))

import dspy
from dspy.utils.dummies import DummyLM

from src.dspy_telemetry import OpenRouterLM
from src.dspy_verifier import (
    TrajectoryVerifier, VERIFIER_MODEL, load_compiled_verifier,
    sanitize_trajectory, select_with_dspy,
)


class DSPyVerifierTests(unittest.TestCase):
    def candidates(self):
        return [
            {"candidate_index": index, "final_sql": f"SELECT {index}",
             "history": [{"action": "probe", "sql": "SELECT missing", "rationale": "Inspect data",
                          "observation": json.dumps({"is_valid": False, "error_message": "no such column",
                                                     "sample_rows": [], "gold_sql": "SECRET_REFERENCE"}),
                          "correct": True}],
             "execution": {"row_count": 1, "sample_rows": [[index]], "correct": True},
             "gold_sql": "SECRET_REFERENCE", "execution_accuracy": 100}
            for index in (4, 1)
        ]

    def test_score_average_earliest_tie_and_nested_label_exclusion(self):
        calls = []
        answers = iter([0.8, 0.6, 0.7, 0.7])

        def judge(**inputs):
            calls.append(inputs)
            return SimpleNamespace(score=next(answers), rationale="Assess answer semantics")

        result = select_with_dspy("Count users", "", "CREATE TABLE users(id INT)", self.candidates(),
                                  verifier=judge, verification_samples=2)
        self.assertEqual(result["selected_candidate_index"], 1)
        self.assertEqual(len(result["metrics"]["stages"]), 4)
        for call in calls:
            self.assertEqual(set(call), {"question", "evidence", "schema_ddl", "trajectory"})
            self.assertNotIn("SECRET_REFERENCE", call["trajectory"])
            self.assertNotIn('"correct"', call["trajectory"])
            self.assertIn("no such column", call["trajectory"])
        self.assertEqual(result["score_kind"], "elicited_validity_rating")

    def test_partial_ranking_is_failure_not_a_biased_selection(self):
        answers = iter([SimpleNamespace(score=0.9, rationale="Seems correct"), ValueError("parse failure")])

        def judge(**inputs):
            answer = next(answers)
            if isinstance(answer, Exception):
                raise answer
            return answer

        result = select_with_dspy("Question", "", "schema", self.candidates(), verifier=judge)
        self.assertIsNone(result["selected_candidate_index"])
        self.assertEqual(len(result["metrics"]["stages"]), 2)
        self.assertIn("parse failure", result["errors"][0]["error"])
        self.assertIsNone(result["scores"][1]["score"])

    def test_invalid_ratings_are_rejected(self):
        for score in (-0.1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(score=score):
                result = select_with_dspy("Q", "", "schema", self.candidates()[:1],
                                          verifier=lambda **_: SimpleNamespace(score=score, rationale="R"))
                self.assertIsNone(result["selected_candidate_index"])
                self.assertEqual(len(result["errors"]), 1)

    def test_empty_candidate_pool_has_known_zero_selection_cost(self):
        judge = Mock()
        result = select_with_dspy("Q", "", "schema", [], verifier=judge)
        judge.assert_not_called()
        self.assertIsNone(result["selected_candidate_index"])
        self.assertEqual(result["metrics"]["cost_usd"], 0)
        self.assertEqual(result["metrics"]["total_tokens"], 0)
        self.assertTrue(result["metrics"]["accounting_complete"])

    def test_manifest_required_before_network_configuration(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            TrajectoryVerifier().save(str(path), save_program=False)
            with patch("src.dspy_modules.create_dspy_lm") as create:
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    load_compiled_verifier(path)
            create.assert_not_called()

    def test_provider_usage_cost_and_parse_failure_are_retained(self):
        lm = OpenRouterLM(model=VERIFIER_MODEL, cache=False)
        verifier = TrajectoryVerifier()
        verifier.set_lm(lm)
        scripted = DummyLM([{"rationale": "Conditions match", "score": 0.9},
                            {"rationale": "Incomplete output"}])
        observations = []

        @contextmanager
        def start_observation(**fields):
            observations.append(fields)
            yield SimpleNamespace(trace_id="dspy-verifier-trace", id=f"dspy-observation-{len(observations)}",
                                  update=lambda **updates: fields.update(updates))

        client = SimpleNamespace(start_as_current_observation=start_observation, flush=Mock())

        def response(prompt=None, messages=None, **kwargs):
            result = scripted.forward(prompt=prompt, messages=messages, **kwargs)
            result.model = VERIFIER_MODEL
            result.usage.update(prompt_tokens=20, completion_tokens=5, total_tokens=25, cost=0.0001)
            return result

        with patch.object(dspy.LM, "forward", side_effect=response):
            result = select_with_dspy("Q", "", "schema", self.candidates(),
                                      verifier=verifier, langfuse_client=client)
        self.assertIsNone(result["selected_candidate_index"])
        self.assertEqual(result["metrics"]["prompt_tokens"], 40)
        self.assertEqual(result["metrics"]["completion_tokens"], 10)
        self.assertAlmostEqual(result["metrics"]["cost_usd"], 0.0002)
        self.assertEqual(observations[1]["usage_details"], {"input": 20, "output": 5, "total": 25})
        for index, stage in enumerate(result["metrics"]["stages"], start=1):
            self.assertEqual(stage["trace_id"], "dspy-verifier-trace")
            self.assertEqual(stage["observation_id"], f"dspy-observation-{index}")
        client.flush.assert_called_once()

    def test_formal_model_mismatch_retains_charge_but_cannot_win(self):
        lm = OpenRouterLM(model=VERIFIER_MODEL, cache=False)
        verifier = TrajectoryVerifier()
        verifier.set_lm(lm)
        verifier._mars_formal = True
        scripted = DummyLM([{"rationale": "Claim correctness", "score": 1.0}])

        def response(prompt=None, messages=None, **kwargs):
            result = scripted.forward(prompt=prompt, messages=messages, **kwargs)
            result.model = "qwen/qwen-2.5-coder-32b-instruct"
            result.usage.update(prompt_tokens=20, completion_tokens=5, total_tokens=25, cost=0.0001)
            return result

        with patch.object(dspy.LM, "forward", side_effect=response):
            result = select_with_dspy("Q", "", "schema", self.candidates()[:1], verifier=verifier)
        self.assertIsNone(result["selected_candidate_index"])
        self.assertIn("different model", result["errors"][0]["error"])
        self.assertAlmostEqual(result["metrics"]["cost_usd"], 0.0001)

    def test_rating_calls_do_not_request_probability_parameters(self):
        lm = OpenRouterLM(model=VERIFIER_MODEL, cache=False)
        verifier = TrajectoryVerifier()
        verifier.set_lm(lm)
        scripted = DummyLM([{"rationale": "Wrong filter", "score": 0.2},
                            {"rationale": "Correct conditions", "score": 0.9}])
        requests = []

        def response(prompt=None, messages=None, **kwargs):
            requests.append(kwargs)
            result = scripted.forward(prompt=prompt, messages=messages, **kwargs)
            result.model = VERIFIER_MODEL
            result.usage.update(prompt_tokens=20, completion_tokens=5, total_tokens=25, cost=0.0001)
            return result

        with patch.object(dspy.LM, "forward", side_effect=response):
            result = select_with_dspy("Q", "", "schema", self.candidates(), verifier=verifier)
        self.assertEqual(result["selected_candidate_index"], 1)
        self.assertEqual(result["score_kind"], "elicited_validity_rating")
        self.assertEqual(result["metrics"]["total_tokens"], 50)
        for request in requests:
            self.assertNotIn("logprobs", request)
            self.assertNotIn("top_logprobs", request)

    def test_tracing_failure_retains_billed_rating_responses(self):
        lm = OpenRouterLM(model=VERIFIER_MODEL, cache=False)
        verifier = TrajectoryVerifier()
        verifier.set_lm(lm)
        scripted = DummyLM([{"rationale": "Correct conditions", "score": 0.9}])

        def response(prompt=None, messages=None, **kwargs):
            result = scripted.forward(prompt=prompt, messages=messages, **kwargs)
            result.usage.update(prompt_tokens=20, completion_tokens=5, total_tokens=25, cost=0.0001)
            return result

        @contextmanager
        def observation(**fields):
            yield SimpleNamespace(update=Mock(side_effect=RuntimeError("trace update failed")))

        client = SimpleNamespace(start_as_current_observation=observation, flush=Mock())
        with patch.object(dspy.LM, "forward", side_effect=response):
            with self.assertRaisesRegex(RuntimeError, "trace update failed") as caught:
                select_with_dspy("Q", "", "schema", self.candidates()[:1],
                                 verifier=verifier, langfuse_client=client)
        self.assertEqual(caught.exception.mars_metrics["total_tokens"], 25)
        self.assertAlmostEqual(caught.exception.mars_metrics["cost_usd"], 0.0001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
