"""Offline tests for measured next-token confidence, without model API calls."""

import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

os.environ.setdefault("DSPY_CACHEDIR", str(Path(__file__).resolve().parents[1] / ".dspy_cache"))

from src.dspy_modules import FORMAL_MODEL, provider_model_id
from src.generative_verifier import select_with_generative


def response(yes=0.6, no=0.3, *, model=None):
    return {
        "id": "response-test", "model": model or provider_model_id(FORMAL_MODEL),
        "usage": {"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21, "cost": 0.0001},
        "choices": [{"logprobs": {"content": [{
            "token": "Yes", "logprob": math.log(yes), "top_logprobs": [
                {"token": "Yes", "logprob": math.log(yes)},
                {"token": "No", "logprob": math.log(no)},
                {"token": "Other", "logprob": math.log(0.01)},
            ],
        }]}}],
    }


class GenerativeVerifierTests(unittest.TestCase):
    def candidates(self):
        return [{"candidate_index": index, "final_sql": f"SELECT {index}",
                 "gold_sql": "DO_NOT_LEAK", "correct": True, "execution_accuracy": 100,
                 "execution": {"sample_rows": [[index]], "correct": True, "result_hash": f"hash-{index}"},
                 "history": [{"sql": "SELECT missing", "action": "probe", "correct": False,
                              "observation": json.dumps({"error_message": "no such column",
                                                         "gold_sql": "DO_NOT_LEAK"})}]}
                for index in (5, 2)]

    def client(self, responses):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=Mock(side_effect=responses))))

    def test_probability_request_allowlist_original_indices_and_tie(self):
        client = self.client([response(), response()])
        result = select_with_generative("Q", "E", "schema", self.candidates(), client=client)
        self.assertEqual(result["selected_candidate_index"], 2)
        self.assertAlmostEqual(result["scores"][0]["score"], 2 / 3)
        self.assertEqual(result["score_kind"], "next_token_yes_probability")
        for call in client.chat.completions.create.call_args_list:
            args = call.kwargs
            self.assertEqual(args["model"], provider_model_id(FORMAL_MODEL))
            self.assertTrue(args["logprobs"])
            self.assertEqual(args["top_logprobs"], 20)
            self.assertEqual(args["max_tokens"], 1)
            self.assertEqual(args["temperature"], 1)
            self.assertEqual(args["top_p"], 1)
            self.assertTrue(args["extra_body"]["provider"]["require_parameters"])
            text = args["messages"][1]["content"]
            self.assertNotIn("DO_NOT_LEAK", text)
            self.assertNotIn('"correct"', text)
            self.assertNotIn("execution_accuracy", text)
            self.assertIn("no such column", text)

    def test_multiple_samples_average_and_account_for_every_call(self):
        client = self.client([response(0.8, 0.2), response(0.2, 0.8),
                              response(0.9, 0.1), response(0.7, 0.3)])
        result = select_with_generative("Q", "", "schema", self.candidates(), client=client,
                                        verification_samples=2)
        self.assertEqual(result["selected_candidate_index"], 2)
        self.assertAlmostEqual(result["scores"][0]["score"], 0.5)
        self.assertAlmostEqual(result["scores"][1]["score"], 0.8)
        self.assertEqual(result["metrics"]["total_tokens"], 84)
        self.assertAlmostEqual(result["metrics"]["cost_usd"], 0.0004)

    def test_missing_logprobs_reject_partial_ranking_but_retain_charge(self):
        missing = response()
        missing["choices"][0]["logprobs"] = None
        result = select_with_generative("Q", "", "schema", self.candidates(),
                                        client=self.client([response(), missing]))
        self.assertIsNone(result["selected_candidate_index"])
        self.assertIsNone(result["scores"][1]["score"])
        self.assertIn("omitted", result["errors"][0]["error"])
        self.assertEqual(result["metrics"]["total_tokens"], 42)
        self.assertAlmostEqual(result["metrics"]["cost_usd"], 0.0002)

    def test_missing_no_probability_is_not_assumed_zero(self):
        missing = response()
        missing["choices"][0]["logprobs"]["content"][0]["top_logprobs"].pop(1)
        result = select_with_generative("Q", "", "schema", self.candidates()[:1],
                                        client=self.client([missing]))
        self.assertIsNone(result["selected_candidate_index"])
        self.assertIn("requires usable Yes and No", result["errors"][0]["error"])

    def test_variants_are_grouped_and_duplicate_sample_is_counted_once(self):
        variations = response(0.2, 0.3)
        first = variations["choices"][0]["logprobs"]["content"][0]
        first["top_logprobs"].append({"token": " YES", "logprob": math.log(0.1)})
        result = select_with_generative("Q", "", "schema", self.candidates()[:1],
                                        client=self.client([variations]))
        self.assertAlmostEqual(result["scores"][0]["score"], 0.5)

    def test_model_mismatch_cannot_win_and_retains_accounting(self):
        result = select_with_generative("Q", "", "schema", self.candidates()[:1],
                                        client=self.client([response(model="qwen/32b")]))
        self.assertIsNone(result["selected_candidate_index"])
        self.assertIn("different model", result["errors"][0]["error"])
        self.assertEqual(result["metrics"]["total_tokens"], 21)

    def test_empty_pool_has_zero_cost_and_no_network_calls(self):
        client = self.client([])
        result = select_with_generative("Q", "", "schema", [], client=client)
        client.chat.completions.create.assert_not_called()
        self.assertEqual(result["metrics"]["cost_usd"], 0)
        self.assertEqual(result["metrics"]["total_tokens"], 0)
        self.assertIsNone(result["selected_candidate_index"])

    def test_failed_or_unexecuted_queries_are_excluded(self):
        candidates = self.candidates()
        candidates[0]["execution"]["error"] = "timeout"
        candidates[1]["execution"].pop("result_hash")
        client = self.client([])
        result = select_with_generative("Q", "", "schema", candidates, client=client)
        client.chat.completions.create.assert_not_called()
        self.assertEqual(result["scores"], [])
        self.assertIsNone(result["selected_candidate_index"])

    def test_langfuse_receives_exact_provider_accounting(self):
        observations = []

        @contextmanager
        def start_observation(**fields):
            observations.append(fields)
            yield SimpleNamespace(trace_id="generative-verifier-trace", id="generative-observation-1",
                                  update=lambda **updates: fields.update(updates))

        langfuse = SimpleNamespace(start_as_current_observation=start_observation, flush=Mock())
        result = select_with_generative("Q", "", "schema", self.candidates()[:1],
                                        client=self.client([response()]), langfuse_client=langfuse)
        self.assertEqual(result["selected_candidate_index"], 5)
        self.assertEqual(observations[0]["usage_details"], {"input": 20, "output": 1, "total": 21})
        self.assertEqual(observations[0]["cost_details"], {"total": 0.0001})
        self.assertEqual(result["metrics"]["stages"][0]["trace_id"], "generative-verifier-trace")
        self.assertEqual(result["metrics"]["stages"][0]["observation_id"], "generative-observation-1")
        langfuse.flush.assert_called_once()

    def test_later_tracing_failures_preserve_completed_and_current_billing(self):
        for fail_at, expected_calls in (("create", 1), ("enter", 1), ("update", 2), ("exit", 2)):
            with self.subTest(fail_at=fail_at):
                observation_count = 0

                def start_observation(**fields):
                    nonlocal observation_count
                    observation_count += 1
                    current = observation_count
                    if current == 2 and fail_at == "create":
                        raise RuntimeError("trace creation failed")

                    @contextmanager
                    def observation():
                        if current == 2 and fail_at == "enter":
                            raise RuntimeError("trace enter failed")

                        def update(**updates):
                            if current == 2 and fail_at == "update":
                                raise RuntimeError("trace update failed")

                        yield SimpleNamespace(update=update)
                        if current == 2 and fail_at == "exit":
                            raise RuntimeError("trace exit failed")

                    return observation()

                langfuse = SimpleNamespace(start_as_current_observation=start_observation, flush=Mock())
                client = self.client([response(), response()])
                with self.assertRaisesRegex(RuntimeError, "trace") as raised:
                    select_with_generative("Q", "", "schema", self.candidates(),
                                            client=client, langfuse_client=langfuse)
                metrics = raised.exception.mars_metrics
                self.assertEqual(metrics["total_tokens"], 21 * expected_calls)
                self.assertAlmostEqual(metrics["cost_usd"], 0.0001 * expected_calls)
                self.assertEqual(metrics["lm_call_count"], expected_calls)
                self.assertEqual([stage["candidate_index"] for stage in metrics["stages"]],
                                 [5, 2][:expected_calls])


if __name__ == "__main__":
    unittest.main(verbosity=2)
