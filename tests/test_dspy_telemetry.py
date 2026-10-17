"""Offline checks for provider accounting at the real DSPy LM boundary."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("DSPY_CACHEDIR", str(Path(__file__).resolve().parents[1] / ".dspy_cache"))

import dspy
from openai.types import CompletionUsage

from src.dspy_telemetry import (
    OpenRouterLM,
    capture_lm_responses,
    capture_query_metrics,
    generation_telemetry,
    make_response_record,
    record_stage_metrics,
    summarize_query_metrics,
)


MODEL = "qwen/qwen-2.5-coder-32b-instruct"


def response(text="[[ ## answer ## ]]\nCharlie\n\n[[ ## completed ## ]]", **usage):
    return SimpleNamespace(
        model=MODEL,
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=CompletionUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18, **usage),
        _hidden_params={"response_cost": 9.99},
    )


class DSPyTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.lm = OpenRouterLM("openrouter/qwen/qwen-2.5-7b-instruct", cache=False)

    def test_raw_provider_cost_and_token_breakdowns_survive_real_dspy_predictor(self):
        provider_response = response(
            cost=0.000123,
            cost_details={"upstream_inference_cost": 0.99},
            prompt_tokens_details={"cached_tokens": 4},
            completion_tokens_details={"reasoning_tokens": 2},
        )
        with patch.object(dspy.LM, "forward", return_value=provider_response):
            with dspy.context(lm=self.lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
                with capture_lm_responses() as records:
                    prediction = dspy.Predict("question -> answer")(question="Who is oldest?")
        self.assertEqual(prediction.answer, "Charlie")
        self.assertIs(records[0]["response"], provider_response)
        measured = generation_telemetry(records)
        self.assertEqual(measured["usage_details"], {"input": 11, "output": 7, "total": 18})
        self.assertEqual(measured["cost_details"], {"total": 0.000123})
        self.assertEqual(measured["model"], MODEL)
        self.assertEqual(measured["metadata"]["cost_source"], "openrouter")
        self.assertEqual(measured["metadata"]["provider_usage"][0]["prompt_tokens_details"]["cached_tokens"], 4)
        self.assertEqual(measured["metadata"]["lm_call_count"], 1)
        self.assertGreaterEqual(measured["metadata"]["llm_latency_seconds"], 0)

    def test_missing_partial_and_invalid_costs_are_unknown_but_zero_is_measured(self):
        for cost in (None, -1, float("nan"), float("inf"), True, "0.1"):
            with self.subTest(cost=cost):
                record = {"usage": {"prompt_tokens": 1, "completion_tokens": 2, "cost": cost}}
                measured = generation_telemetry([record])
                self.assertNotIn("cost_details", measured)
                self.assertEqual(measured["metadata"]["cost_source"], "unavailable")
        measured = generation_telemetry([{"usage": {"cost": 0, "prompt_tokens": 0, "completion_tokens": 0}}])
        self.assertEqual(measured["cost_details"], {"total": 0})
        self.assertEqual(measured["usage_details"], {"input": 0, "output": 0, "total": 0})
        partial = generation_telemetry([{"usage": {"cost": 0.1}}, {"usage": {}}])
        self.assertNotIn("cost_details", partial)
        self.assertNotIn("usage_details", partial)

    def test_counts_and_cost_sum_all_responses_without_inventing_missing_tokens(self):
        measured = generation_telemetry([
            {"usage": {"input_tokens": 2, "output_tokens": 3, "cost": 0.1}, "model": MODEL, "latency_seconds": 1.2},
            {"usage": {"prompt_tokens": 5, "completion_tokens": 7, "cost": 0.2}, "model": MODEL, "latency_seconds": 0.3},
        ])
        self.assertEqual(measured["usage_details"], {"input": 7, "output": 10, "total": 17})
        self.assertAlmostEqual(measured["cost_details"]["total"], 0.3)
        self.assertEqual(measured["metadata"]["lm_call_count"], 2)
        self.assertEqual(measured["metadata"]["llm_latency_seconds"], 1.5)
        measured = generation_telemetry([{"usage": {"prompt_tokens": 0}}, {"usage": {"prompt_tokens": 3}}])
        self.assertEqual(measured["usage_details"], {"input": 3})

    def test_capture_does_not_depend_on_disabled_or_full_dspy_history(self):
        with patch.object(dspy.LM, "forward", return_value=response(cost=0.1)):
            with dspy.context(disable_history=True):
                with capture_lm_responses() as records:
                    self.lm("hello")
            self.assertEqual(self.lm.history, [])
            self.assertEqual(len(records), 1)
            with dspy.context(max_history_size=1):
                self.lm("first")
                with capture_lm_responses() as records:
                    self.lm("second")
                    self.lm("third")
            self.assertEqual(len(self.lm.history), 1)
            self.assertEqual(len(records), 2)
            self.assertEqual(generation_telemetry(records)["cost_details"], {"total": 0.2})

    def test_paid_response_is_captured_even_when_adapter_cannot_parse_output(self):
        with patch.object(dspy.LM, "forward", return_value=response(text="unstructured answer", cost=0.25)):
            with dspy.context(lm=self.lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
                with capture_lm_responses() as records:
                    with self.assertRaises(dspy.utils.exceptions.AdapterParseError):
                        dspy.Predict("question -> answer")(question="Who is oldest?")
        self.assertEqual(len(records), 1)
        self.assertEqual(generation_telemetry(records)["cost_details"], {"total": 0.25})

    def test_async_response_capture_and_nested_context_isolation(self):
        async def invoke():
            with capture_lm_responses() as records:
                await self.lm.aforward(prompt="hello")
            return records

        with patch.object(dspy.LM, "aforward", new=AsyncMock(return_value=response(cost=0.3))):
            records = asyncio.run(invoke())
        self.assertEqual(generation_telemetry(records)["cost_details"], {"total": 0.3})
        with patch.object(dspy.LM, "forward", return_value=response(cost=0.1)):
            with capture_lm_responses() as outer:
                self.lm.forward(prompt="outer")
                with capture_lm_responses() as inner:
                    self.lm.forward(prompt="inner")
                self.lm.forward(prompt="outer again")
        self.assertEqual(len(outer), 2)
        self.assertEqual(len(inner), 1)
        self.lm.history.clear()

    def test_compiler_can_capture_nested_graph_responses_once_including_failure(self):
        with patch.object(dspy.LM, "forward", return_value=response(cost=0.1)):
            with capture_lm_responses(include_nested=True) as compilation:
                self.lm.forward(prompt="teacher")
                with capture_lm_responses() as graph_stage:
                    self.lm.forward(prompt="decompose")
                    with capture_lm_responses() as nested_stage:
                        self.lm.forward(prompt="probe")
                try:
                    with capture_lm_responses() as failed_stage:
                        self.lm.forward(prompt="paid malformed response")
                        raise ValueError("parse failure")
                except ValueError:
                    pass
        self.assertEqual(len(compilation), 4)
        self.assertEqual(len(graph_stage), 1)
        self.assertEqual(len(nested_stage), 1)
        self.assertEqual(len(failed_stage), 1)
        self.assertAlmostEqual(generation_telemetry(compilation)["cost_details"]["total"], 0.4)

    def test_stage_report_retains_each_provider_call_without_request_contents(self):
        first_response = response(cost=0.1)
        first_response.id = "gen-first"
        first_response.api_key = "must-not-be-serialized"
        first = make_response_record(first_response, "configured-model", 0.4)
        second = make_response_record({
            "id": "gen-second",
            "model": "different-served-model",
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "cost": 0.2},
        }, "configured-model", 0.6)
        row = record_stage_metrics("generator", 2, [first, second], 1.2)
        self.assertEqual(row["prompt_tokens"], 31)
        self.assertEqual(row["completion_tokens"], 15)
        self.assertEqual(row["total_tokens"], 46)
        self.assertAlmostEqual(row["cost_usd"], 0.3)
        self.assertEqual(row["model"], None)
        self.assertEqual(row["lm_call_count"], 2)
        self.assertEqual(row["llm_latency_seconds"], 1.0)
        self.assertEqual(row["calls"][0]["response_id"], "gen-first")
        self.assertEqual(row["calls"][0]["model"], MODEL)
        self.assertEqual(row["calls"][1]["model"], "different-served-model")
        self.assertNotIn("must-not-be-serialized", json.dumps(row))
        self.assertNotIn("response", row["calls"][0])

    def test_query_report_distinguishes_unknown_empty_accounting_from_measured_zero(self):
        empty = summarize_query_metrics([], turn=0, latency_seconds=0.1)
        unknown_stage = record_stage_metrics("decomposer", 0, [], 0.1, error="No provider response")
        unknown = summarize_query_metrics([unknown_stage], turn=0, latency_seconds=0.1)
        for summary in (empty, unknown):
            self.assertFalse(summary["accounting_complete"])
            for field in ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"):
                self.assertIsNone(summary[field])
                self.assertIsNone(summary[f"observed_{field}"])
        record = make_response_record({
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0},
        }, MODEL, 0.1)
        row = record_stage_metrics("generator", 1, [record], 0.2)
        zero = summarize_query_metrics([row], turn=1, latency_seconds=0.3)
        self.assertTrue(zero["accounting_complete"])
        self.assertEqual(zero["sql_attempts"], 1)
        self.assertEqual(zero["self_correction_cycles"], 0)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"):
            self.assertEqual(zero[field], 0)
            self.assertEqual(zero[f"observed_{field}"], 0)

    def test_partial_failure_retains_billed_subtotals_and_counts_unexecuted_repair(self):
        with capture_query_metrics() as stages:
            record_stage_metrics("decomposer", 0, [make_response_record(response(cost=0.1), MODEL, 0.2)], 0.3)
            record_stage_metrics("generator", 1, [make_response_record(response(cost=0.2), MODEL, 0.4)], 0.5)
            record_stage_metrics("generator", 2, [], 0.6, error="Provider unavailable")
        summary = summarize_query_metrics(stages, turn=1, latency_seconds=1.5)
        self.assertEqual(summary["sql_attempts"], 1)
        self.assertEqual(summary["self_correction_cycles"], 1)
        self.assertEqual(summary["lm_call_count"], 2)
        self.assertFalse(summary["accounting_complete"])
        self.assertIsNone(summary["cost_usd"])
        self.assertAlmostEqual(summary["observed_cost_usd"], 0.3)
        self.assertIsNone(summary["prompt_tokens"])
        self.assertEqual(summary["observed_prompt_tokens"], 22)
        self.assertEqual(summary["observed_completion_tokens"], 14)
        self.assertEqual(summary["observed_total_tokens"], 36)
        self.assertEqual(summary["stages"][-1]["error"], "Provider unavailable")

    def test_partial_accounting_within_one_stage_retains_known_individual_calls(self):
        known = make_response_record(response(cost=0.1), MODEL, 0.2)
        unknown = make_response_record({"usage": {}}, MODEL, 0.3)
        row = record_stage_metrics("generator", 1, [known, unknown], 0.6)
        summary = summarize_query_metrics([row], turn=1, latency_seconds=0.7)
        self.assertIsNone(summary["cost_usd"])
        self.assertEqual(summary["observed_cost_usd"], 0.1)
        self.assertEqual(summary["observed_total_tokens"], 18)
        self.assertEqual(summary["lm_call_count"], 2)

    def test_query_contexts_are_isolated_across_nested_and_concurrent_runs(self):
        with capture_query_metrics() as outer:
            record_stage_metrics("decomposer", 0, [], 0.1)
            with capture_query_metrics() as inner:
                record_stage_metrics("generator", 1, [], 0.2)
            record_stage_metrics("generator", 1, [], 0.3)
        self.assertEqual([stage["stage"] for stage in outer], ["decomposer", "generator"])
        self.assertEqual([stage["stage"] for stage in inner], ["generator"])
        record_stage_metrics("generator", 2, [], 0.4)
        self.assertEqual(len(outer), 2)

        async def collect(turn):
            with capture_query_metrics() as stages:
                await asyncio.sleep(0)
                record_stage_metrics("generator", turn, [], 0.1)
            return stages

        async def concurrent():
            return await asyncio.gather(collect(1), collect(2))

        first, second = asyncio.run(concurrent())
        self.assertEqual([stage["turn"] for stage in first], [1])
        self.assertEqual([stage["turn"] for stage in second], [2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
