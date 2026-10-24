"""Offline routing and exact OpenAI usage persistence for self-hosted vLLM."""

from contextlib import closing, contextmanager
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4
from pydantic import SecretStr

import dspy
from dspy.utils.dummies import DummyLM
import httpx
from openai import OpenAI

from src.bird_evaluator import run_bird_evaluation
from src.config import Settings, settings
from src.dspy_modules import FORMAL_MODEL, TextToSQLProgram, create_dspy_lm, provider_model_id
from src.dspy_telemetry import generation_telemetry, make_response_record, record_stage_metrics, summarize_query_metrics
from src.llm_endpoint import resolve_llm_endpoint
from src.sql_agent_graph import SQLAgentGraph
from src.telemetry_errors import redact_text


class VLLMEndpointTests(unittest.TestCase):
    def setUp(self):
        self.api_key = uuid4().hex
        self.key_patch = patch.object(settings, "vllm_api_key", SecretStr(self.api_key))
        self.key_patch.start()
        self.addCleanup(self.key_patch.stop)

    def test_endpoint_switch_uses_dynamic_key_and_native_checkpoint_name(self):
        endpoint = resolve_llm_endpoint(FORMAL_MODEL, settings=SimpleNamespace(vllm_base_url="https://pod.example/", vllm_api_key=SecretStr(self.api_key)))
        self.assertEqual(endpoint.base_url, "https://pod.example/v1")
        self.assertEqual(endpoint.api_key, self.api_key)
        self.assertNotIn(self.api_key, repr(endpoint))
        self.assertEqual(endpoint.model, "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(provider_model_id(endpoint.model), provider_model_id(FORMAL_MODEL))
        self.assertNotIn("api_key", endpoint.manifest())
        with patch.object(settings, "vllm_base_url", "https://pod.example/v1"), patch("src.dspy_modules.OpenRouterLM") as constructor:
            create_dspy_lm(formal=True)
            self.assertEqual(constructor.call_args.kwargs["model"], "openai/Qwen/Qwen2.5-7B-Instruct")
            self.assertEqual(constructor.call_args.kwargs["api_key"], self.api_key)
            self.assertEqual(constructor.call_args.kwargs["api_base"], "https://pod.example/v1")
            self.assertEqual(constructor.call_args.kwargs["telemetry_backend"], "vllm")

    def test_blank_url_preserves_openrouter_and_rejects_urls_with_credentials(self):
        endpoint = resolve_llm_endpoint(FORMAL_MODEL, settings=SimpleNamespace(vllm_base_url="", openrouter_api_key="test-key"))
        self.assertEqual(endpoint.backend, "openrouter")
        self.assertEqual(endpoint.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(endpoint.api_key, "test-key")
        for url in ("ftp://pod.example", "https://user:secret@pod.example", "https://pod.example?token=secret"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                resolve_llm_endpoint(FORMAL_MODEL, settings=SimpleNamespace(vllm_base_url=url))

    def test_vllm_requires_key_and_https_without_provider_fallback(self):
        for key in (None, "", "   ", "key\nvalue"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "VLLM_API_KEY"):
                resolve_llm_endpoint(FORMAL_MODEL, settings=SimpleNamespace(vllm_base_url="https://pod.example/v1", vllm_api_key=key))
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            resolve_llm_endpoint(FORMAL_MODEL, settings=SimpleNamespace(vllm_base_url="http://pod.example", vllm_api_key=self.api_key))

    def test_dotenv_key_loading_environment_precedence_and_redaction(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            os.environ.pop("VLLM_API_KEY", None)
            os.environ.pop("VLLM_BASE_URL", None)
            env = Path(directory) / ".env"
            env.write_text(f"VLLM_BASE_URL=https://pod.example/v1\nVLLM_API_KEY={self.api_key}\n")
            kwargs = dict(_env_file=env, openrouter_api_key="", langfuse_public_key=uuid4().hex,
                          langfuse_secret_key=uuid4().hex, langfuse_base_url="https://telemetry.example")
            loaded = Settings(**kwargs)
            self.assertEqual(resolve_llm_endpoint(FORMAL_MODEL, settings=loaded).api_key, self.api_key)
            self.assertNotIn(self.api_key, repr(loaded))
            override = uuid4().hex
            os.environ["VLLM_API_KEY"] = override
            self.assertEqual(Settings(**kwargs).vllm_api_key.get_secret_value(), override)
        self.assertNotIn(self.api_key, redact_text("Server echoed " + self.api_key))

    def test_vllm_usage_without_cost_keeps_exact_counts_and_declares_cost_scope(self):
        record = make_response_record({"model": "Qwen/Qwen2.5-7B-Instruct", "usage": {
            "prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}, FORMAL_MODEL, 0.1, backend="vllm")
        telemetry = generation_telemetry([record])
        self.assertEqual(telemetry["usage_details"], {"input": 11, "output": 7, "total": 18})
        self.assertEqual(telemetry["cost_details"], {"total": 0})
        self.assertEqual(telemetry["metadata"]["cost_source"], "self_hosted_api_fee_excludes_gpu_rental")
        record["usage"] = {}
        summary = summarize_query_metrics([record_stage_metrics("generator", 1, [record], 0.1)], turn=1, latency_seconds=0.1)
        self.assertFalse(summary["accounting_complete"])
        self.assertIsNone(summary["total_tokens"])

    def test_raw_graph_http_request_uses_custom_url_and_captures_standard_usage(self):
        requests = []
        answers = [{"target_tables": ["users"], "target_columns": ["name"],
                    "join_logic": None, "filter_logic": None, "reasoning": "List names."},
                   {"action": "final_answer", "rationale": "List names.", "sql_query": "SELECT name FROM users"}]
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"id": "mock", "object": "chat.completion", "created": 1,
                "model": "Qwen/Qwen2.5-7B-Instruct", "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": json.dumps(answers[len(requests)-1])},
                    "finish_reason": "stop"}], "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}})
        with tempfile.TemporaryDirectory() as directory, httpx.Client(transport=httpx.MockTransport(handler)) as transport:
            db = Path(directory) / "users.sqlite"
            with closing(sqlite3.connect(db)) as connection:
                connection.executescript("CREATE TABLE users(name TEXT); INSERT INTO users VALUES('Ada');")
            def constructor(**kwargs):
                return OpenAI(**kwargs, http_client=transport)
            with patch.object(settings, "vllm_base_url", "https://pod.example/v1"), \
                 patch("src.sql_agent_graph.SQLAgentGraph._initialize_langfuse"), \
                 patch("langfuse.openai.OpenAI", side_effect=constructor):
                agent = SQLAgentGraph(prompt_mode="baseline", formal=True)
                result = agent.run("List users", "", "CREATE TABLE users(name TEXT)", str(db))
            self.assertEqual(result["metrics"]["prompt_tokens"], 22)
            self.assertEqual(result["metrics"]["completion_tokens"], 14)
            self.assertEqual(result["metrics"]["total_tokens"], 36)
            self.assertTrue(result["metrics"]["accounting_complete"])
            self.assertEqual(result["final_sql"], "SELECT name FROM users")
            for request in requests:
                self.assertEqual(str(request.url), "https://pod.example/v1/chat/completions")
                self.assertEqual(request.headers["Authorization"], f"Bearer {self.api_key}")
                self.assertEqual(json.loads(request.content)["model"], "Qwen/Qwen2.5-7B-Instruct")

    def test_dspy_graph_saves_exact_vllm_usage_to_json_checkpoint_and_trace(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(settings, "vllm_base_url", "https://pod.example/v1"):
            root = Path(directory); db = root / "databases/users/users.sqlite"; db.parent.mkdir(parents=True)
            with closing(sqlite3.connect(db)) as connection:
                connection.executescript("CREATE TABLE users(name TEXT); INSERT INTO users VALUES('Ada');")
            dev = root / "dev.json"; output = root / "predict.json"
            dev.write_text(json.dumps([{"question_id": 7, "db_id": "users", "question": "List users", "SQL": "SELECT name FROM users"}]))
            scripted = DummyLM([{"logical_plan": "List names."},
                {"action": "final_answer", "rationale": "List names.", "sql_query": "SELECT name FROM users"}])
            lm = create_dspy_lm(formal=True)
            def provider(**kwargs):
                response = scripted.forward(**kwargs)
                response.model = "Qwen/Qwen2.5-7B-Instruct"
                response.usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
                return response
            observations = []
            @contextmanager
            def trace(**kwargs):
                observations.append(kwargs)
                yield SimpleNamespace(update=lambda **updates: kwargs.update(updates))
            def build(**kwargs):
                program = TextToSQLProgram(); program.set_lm(lm)
                agent = SQLAgentGraph(dspy_program=program, formal=True)
                agent._langfuse_client = SimpleNamespace(start_as_current_observation=trace, flush=lambda: None)
                return agent
            with patch("src.bird_evaluator.build_sql_graph", side_effect=build), patch.object(dspy.LM, "forward", side_effect=provider):
                run_bird_evaluation(str(dev), str(root / "databases"), str(output))
            report = json.loads(output.with_suffix(".metrics.json").read_text())
            self.assertEqual(report["execution_accuracy"], 100)
            self.assertEqual(report["totals"]["prompt_tokens"], 22)
            self.assertEqual(report["totals"]["completion_tokens"], 14)
            self.assertEqual(report["totals"]["total_tokens"], 36)
            self.assertTrue(report["totals"]["accounting_complete"])
            self.assertEqual(report["queries"][0]["configuration"]["llm_backend"], "vllm")
            saved = json.loads(output.with_suffix(".checkpoint.jsonl").read_text().splitlines()[1])["record"]
            self.assertEqual(saved["metrics"]["total_tokens"], 36)
            self.assertEqual([o["usage_details"]["total"] for o in observations if o["as_type"] == "generation"], [18, 18])
