"""Offline checks of actual Langfuse/OpenTelemetry exports from the DSPy path."""

from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

os.environ.setdefault("DSPY_CACHEDIR", str(Path(__file__).resolve().parents[1] / ".dspy_cache"))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import dspy
from dspy.dsp.utils.utils import dotdict
from dspy.utils.exceptions import AdapterParseError
import httpx
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.dspy_modules import TextToSQLProgram
from src.dspy_telemetry import OpenRouterLM
from src.sql_agent_graph import SQLAgentGraph, build_sql_graph


ACTUAL_MODEL = "qwen/qwen-2.5-coder-32b-instruct"
PLAN = "Order users by age descending and return the first name."
SQL = "SELECT name FROM users ORDER BY age DESC LIMIT 1;"
OBSERVATION = "langfuse.observation."
TRACE = "langfuse.trace."


def formatted_field(name, value, action="final_answer"):
    if name == "sql_query":
        return (f"[[ ## action ## ]]\n{action}\n\n[[ ## rationale ## ]]\nUse observations.\n\n"
                f"[[ ## sql_query ## ]]\n{value}\n\n[[ ## completed ## ]]")
    return f"[[ ## {name} ## ]]\n{value}\n\n[[ ## completed ## ]]"


class LangfuseExportIntegrationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="mars_langfuse_export_")
        self.addCleanup(directory.cleanup)
        self.db_path = Path(directory.name) / "users.sqlite"
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.executescript(
                "CREATE TABLE users (name TEXT, age INTEGER);"
                "INSERT INTO users VALUES ('Ada', 36), ('Grace', 85);"
            )
        self.http_requests = []

        def reject_network(request):
            self.http_requests.append(str(request.url))
            raise AssertionError("The in-memory export must not make HTTP requests")

        http_client = httpx.Client(transport=httpx.MockTransport(reject_network))
        self.addCleanup(http_client.close)
        self.exporter = InMemorySpanExporter()
        provider = TracerProvider()
        self.addCleanup(provider.shutdown)
        self.langfuse = Langfuse(
            public_key=f"pk-offline-{uuid4().hex}",
            secret_key="sk-offline-placeholder",
            base_url="https://offline.invalid",
            httpx_client=http_client,
            tracer_provider=provider,
            span_exporter=self.exporter,
            sample_rate=1.0,
            flush_at=100,
            flush_interval=60.0,
        )
        self.addCleanup(self.langfuse.shutdown)

    def state(self):
        return {
            "question": "What is the name of the oldest user?",
            "evidence": "Oldest means highest age.",
            "schema_ddl": "CREATE TABLE users (name TEXT, age INTEGER);",
            "db_path": str(self.db_path),
            "question_id": 42,
            "db_id": "users",
            "plan": None,
            "current_sql": None,
            "observation": None,
            "history": [],
            "turn": 0,
            "max_turns": 3,
            "is_valid": False,
            "final_sql": None,
        }

    def invoke(self, texts, *, disable_history=False, expect_parse_error=False, via_factory=False):
        lm = OpenRouterLM(
            model="openrouter/qwen/qwen-2.5-7b-instruct",
            cache=False,
            temperature=0.0,
            max_tokens=2048,
            num_retries=0,
        )
        program = TextToSQLProgram()
        program.set_lm(lm)
        agent = SQLAgentGraph(dspy_program=program)
        agent._langfuse_client = self.langfuse
        runnable = agent
        if via_factory:
            # Inject only the configured SDK client; execute the real factory's
            # runnable so the BIRD entry point must also trace and flush.
            with patch("src.sql_agent_graph.SQLAgentGraph", return_value=agent):
                runnable = build_sql_graph(dspy_program=program)
        responses = [
            dotdict(
                choices=[dotdict(
                    message=dotdict(content=text, role="assistant", tool_calls=None),
                    finish_reason="stop",
                )],
                model=ACTUAL_MODEL,
                usage=dotdict(
                    prompt_tokens=11 * (index + 1),
                    completion_tokens=7 * (index + 1),
                    total_tokens=18 * (index + 1),
                    prompt_tokens_details={"cached_tokens": 3},
                    completion_tokens_details={"reasoning_tokens": 2},
                    cost=0.001 * (index + 1),
                ),
                # This estimate must not replace the provider's actual charge.
                _hidden_params={"response_cost": 9.99},
            )
            for index, text in enumerate(texts)
        ]
        result = None
        with dspy.context(
            adapter=dspy.ChatAdapter(use_json_adapter_fallback=False),
            disable_history=disable_history,
        ), patch("dspy.clients.lm.litellm_completion", side_effect=responses) as completion:
            if expect_parse_error:
                with self.assertRaises(AdapterParseError) as caught:
                    runnable.invoke(self.state(), {"recursion_limit": 9})
                self.last_failure = caught.exception
            else:
                result = runnable.invoke(self.state(), {"recursion_limit": 9})
        self.assertEqual(completion.call_count, len(texts))
        self.assertEqual(self.http_requests, [])
        # invoke must flush; the test deliberately does not call Langfuse.flush.
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), len(texts) + 1)
        return result, lm, spans

    def assert_query_trace(self, spans, expected_names):
        root = next(span for span in spans if span.name == "mars_sql_query")
        children = sorted(
            (span for span in spans if span is not root), key=lambda span: span.start_time
        )
        self.assertEqual([span.name for span in children], expected_names)
        self.assertIsNone(root.parent)
        for span in children:
            self.assertEqual(span.context.trace_id, root.context.trace_id)
            self.assertEqual(span.parent.span_id, root.context.span_id)
            self.assertEqual(span.attributes[TRACE + "metadata.question_id"], "42")
        for span in spans:
            self.assertGreater(span.end_time - span.start_time, 0)
            self.assertGreater(float(span.attributes[OBSERVATION + "metadata.latency_seconds"]), 0)
        self.assertEqual(root.attributes[TRACE + "name"], "mars_sql_query")
        return root, children

    def assert_generation_accounting(self, span, call_number):
        attributes = span.attributes
        self.assertEqual(attributes[OBSERVATION + "model.name"], ACTUAL_MODEL)
        self.assertEqual(json.loads(attributes[OBSERVATION + "usage_details"]), {
            "input": 11 * call_number,
            "output": 7 * call_number,
            "total": 18 * call_number,
        })
        self.assertEqual(json.loads(attributes[OBSERVATION + "cost_details"]), {
            "total": 0.001 * call_number,
        })
        self.assertEqual(attributes[OBSERVATION + "metadata.cost_source"], "openrouter")
        self.assertGreater(float(attributes[OBSERVATION + "metadata.llm_latency_seconds"]), 0)
        usage = json.loads(attributes[OBSERVATION + "metadata.provider_usage"])[0]
        self.assertEqual(usage["prompt_tokens_details"], {"cached_tokens": 3})
        self.assertEqual(usage["completion_tokens_details"], {"reasoning_tokens": 2})

    def test_real_exports_include_usage_cost_and_trace_when_history_is_disabled(self):
        result, lm, spans = self.invoke([
            formatted_field("logical_plan", PLAN), formatted_field("sql_query", SQL),
        ], disable_history=True, via_factory=True)
        self.assertTrue(result["is_valid"])
        self.assertEqual(lm.history, [])
        metrics = result["metrics"]
        self.assertEqual(metrics["prompt_tokens"], 33)
        self.assertEqual(metrics["completion_tokens"], 21)
        self.assertEqual(metrics["total_tokens"], 54)
        self.assertAlmostEqual(metrics["cost_usd"], 0.003)
        self.assertTrue(metrics["accounting_complete"])
        self.assertEqual([stage["prompt_tokens"] for stage in metrics["stages"]], [11, 22])
        root, children = self.assert_query_trace(spans, ["mars_decomposer", "mars_generator"])
        self.assertEqual(metrics["trace_id"], format(root.context.trace_id, "032x"))
        self.assertEqual(root.attributes[TRACE + "metadata.prompt_tokens"], "33")
        self.assertEqual(root.attributes[OBSERVATION + "metadata.prompt_tokens"], 33)
        self.assertNotIn(OBSERVATION + "cost_details", root.attributes)
        self.assertNotIn(OBSERVATION + "usage_details", root.attributes)
        for number, child in enumerate(children, start=1):
            self.assert_generation_accounting(child, number)
        self.assertEqual(root.attributes[TRACE + "metadata.turn"], "1")
        self.assertEqual(root.attributes[TRACE + "metadata.self_correction_cycles"], "0")
        self.assertEqual(root.attributes[OBSERVATION + "metadata.turn"], 1)
        self.assertEqual(json.loads(root.attributes[OBSERVATION + "output"])["final_sql"], SQL)
        self.assertEqual(json.loads(children[0].attributes[OBSERVATION + "output"]), {"logical_plan": PLAN})
        self.assertEqual(json.loads(children[1].attributes[OBSERVATION + "output"]), {
            "action": "final_answer", "rationale": "Use observations.", "sql_query": SQL,
        })

    def test_repair_exports_all_three_billed_generations_in_one_query_trace(self):
        result, _, spans = self.invoke([
            formatted_field("logical_plan", PLAN),
            formatted_field("sql_query", "SELECT missing_column FROM users", action="probe"),
            formatted_field("sql_query", SQL),
        ])
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["turn"], 2)
        self.assertEqual(result["self_correction_cycles"], 1)
        self.assertEqual(result["metrics"]["prompt_tokens"], 66)
        self.assertEqual(result["metrics"]["completion_tokens"], 42)
        self.assertAlmostEqual(result["metrics"]["cost_usd"], 0.006)
        root, children = self.assert_query_trace(
            spans, ["mars_decomposer", "mars_generator", "mars_generator"]
        )
        for number, child in enumerate(children, start=1):
            self.assert_generation_accounting(child, number)
        total_cost = sum(json.loads(span.attributes[OBSERVATION + "cost_details"])["total"] for span in children)
        self.assertAlmostEqual(total_cost, 0.006)
        self.assertEqual(root.attributes[TRACE + "metadata.turn"], "2")
        self.assertEqual(root.attributes[TRACE + "metadata.self_correction_cycles"], "1")
        self.assertEqual(root.attributes[OBSERVATION + "metadata.self_correction_cycles"], 1)

    def test_parse_failure_still_exports_paid_generation_and_flushes_query(self):
        _, _, spans = self.invoke([
            formatted_field("logical_plan", PLAN), "This response has no named SQL field.",
        ], expect_parse_error=True)
        self.assertEqual(self.last_failure.mars_metrics["prompt_tokens"], 33)
        self.assertAlmostEqual(self.last_failure.mars_metrics["cost_usd"], 0.003)
        root, children = self.assert_query_trace(spans, ["mars_decomposer", "mars_generator"])
        for number, child in enumerate(children, start=1):
            self.assert_generation_accounting(child, number)
        self.assertEqual(root.attributes[TRACE + "metadata.turn"], "0")
        self.assertFalse(json.loads(root.attributes[OBSERVATION + "output"])["is_valid"])
        self.assertIsNone(json.loads(root.attributes[OBSERVATION + "output"])["final_sql"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
