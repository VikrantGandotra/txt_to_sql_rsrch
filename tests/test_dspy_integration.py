"""Offline DSPy compilation and LangGraph integration checks."""

from contextlib import closing, contextmanager
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("DSPY_CACHEDIR", str(Path(__file__).resolve().parents[1] / ".dspy_cache"))

import dspy
from dspy.utils.dummies import DummyLM

from src.dspy_modules import TextToSQLProgram, configure_dspy, create_dspy_lm, FORMAL_MODEL, provider_model_id
from src.dspy_telemetry import OpenRouterLM
from src.dspy_optimizer import build_trainset, compile_program, execution_metric
from src.sql_agent_graph import SQLAgentGraph


DEMO_PLAN = "TRAINING_DEMO: Use the requested user columns, filters, ordering, and aggregation."


def generation(sql, action="final_answer"):
    return {"action": action, "rationale": "Use the available observations.", "sql_query": sql}


class DSPyIntegrationTests(unittest.TestCase):
    def test_native_openrouter_route_has_one_prefix_and_preserves_7b_pin(self):
        from litellm import get_llm_provider

        for configured in (FORMAL_MODEL, provider_model_id(FORMAL_MODEL)):
            with self.subTest(model=configured), patch("src.dspy_modules.OpenRouterLM") as constructor:
                create_dspy_lm(model_name=configured, formal=True)
                request = constructor.call_args.kwargs
                self.assertEqual(request["model"], FORMAL_MODEL)
                model, provider, _, _ = get_llm_provider(
                    model=request["model"], api_base=request["api_base"], api_key="offline-key"
                )
                self.assertEqual(provider, "openrouter")
                self.assertEqual(model, provider_model_id(FORMAL_MODEL))

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="mars_dspy_")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.db_path = self.root / "training.sqlite"
        self.artifact_path = self.root / "compiled_text_to_sql.json"
        self.examples = build_trainset(str(self.db_path))
        self.addCleanup(dspy.configure, lm=dspy.settings.lm, adapter=dspy.settings.adapter)

    def compile_artifact(self):
        answers = []
        for example in self.examples:
            answers.extend([
                {"logical_plan": DEMO_PLAN},
                generation(example.sql_query),
            ])
        lm = DummyLM(answers)
        program = compile_program(str(self.artifact_path), lm=lm)
        return program, lm

    def run_agent(self, agent):
        example = self.examples[0]
        return agent.run(
            question=example.question,
            evidence="Oldest refers to the highest age.",
            schema_ddl=example.schema_ddl,
            db_path=str(self.db_path),
            max_turns=3,
        )

    def test_real_bootstrap_compiler_saves_and_reloads_demonstrations(self):
        self.assertEqual(len(self.examples), 4)
        for example in self.examples:
            self.assertEqual(set(example.inputs()), {"question", "schema_ddl"})
        compiled, lm = self.compile_artifact()
        self.assertEqual(len(lm.history), 8)
        self.assertTrue(self.artifact_path.is_file())
        loaded = TextToSQLProgram()
        loaded.load(str(self.artifact_path))
        for stage in ("decomposer", "generator"):
            compiled_demos = getattr(compiled, stage).demos
            loaded_demos = getattr(loaded, stage).demos
            self.assertEqual(len(compiled_demos), 4)
            self.assertEqual(len(loaded_demos), 4)
            self.assertEqual(
                [dict(demo) for demo in compiled_demos],
                [dict(demo) for demo in loaded_demos],
            )
        runtime_lm = DummyLM([
            {"logical_plan": "Order by age descending and return the first name."},
            generation(self.examples[0].sql_query),
        ])
        with dspy.context(lm=runtime_lm):
            prediction = loaded(**self.examples[0].inputs())
        self.assertIsInstance(prediction, dspy.Prediction)
        self.assertEqual(prediction.sql_query, self.examples[0].sql_query)
        self.assertTrue(prediction.logical_plan)
        self.assertIn(DEMO_PLAN, json.dumps(runtime_lm.history[0]["messages"]))

    def test_compiler_rejects_zero_successful_traces_without_replacing_artifact(self):
        prior_artifact = "previous compiled artifact"
        self.artifact_path.write_text(prior_artifact, encoding="utf-8")
        answers = [
            {"logical_plan": "Attempt a query with an invalid column."},
            generation("SELECT missing_column FROM users"),
        ] * 4
        with self.assertRaises(RuntimeError):
            compile_program(str(self.artifact_path), lm=DummyLM(answers))
        self.assertEqual(self.artifact_path.read_text(encoding="utf-8"), prior_artifact)

    def test_execution_metric_rejects_wrong_error_and_mutating_sql(self):
        example = self.examples[0]
        self.assertTrue(execution_metric(example, dspy.Prediction(sql_query=example.sql_query)))
        for sql in (
            "SELECT name FROM users ORDER BY age ASC LIMIT 1",
            "SELECT missing_column FROM users",
            "DELETE FROM users",
        ):
            with self.subTest(sql=sql):
                self.assertFalse(execution_metric(example, dspy.Prediction(sql_query=sql)))
        with closing(sqlite3.connect(self.db_path)) as connection:
            rows = connection.execute("SELECT name FROM users ORDER BY id").fetchall()
        self.assertEqual(rows, [("Ada",), ("Linus",), ("Grace",), ("Edsger",)])
        missing_path = self.root / "missing.sqlite"
        missing_example = example.copy(db_path=str(missing_path))
        self.assertFalse(execution_metric(missing_example, dspy.Prediction(sql_query=example.sql_query)))
        self.assertFalse(missing_path.exists())

    def test_injected_dspy_program_repairs_sql_with_database_feedback(self):
        invalid_sql = "SELECT missing_column FROM users"
        logical_plan = "Return the name of the oldest user."
        lm = DummyLM([
            {"logical_plan": logical_plan},
            generation(invalid_sql, "probe"),
            generation(self.examples[0].sql_query),
        ])
        self.assertIs(configure_dspy(lm=lm), lm)
        agent = SQLAgentGraph(dspy_program=TextToSQLProgram())
        result = self.run_agent(agent)

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["plan"], logical_plan)
        self.assertEqual(result["turn"], 2)
        self.assertEqual(result["final_sql"], self.examples[0].sql_query)
        self.assertEqual(len(result["history"]), 2)
        self.assertEqual(len(lm.history), 3)
        repair_prompt = json.dumps(lm.history[2]["messages"])
        self.assertIn(invalid_sql, repair_prompt)
        self.assertIn("no such column", repair_prompt)
        self.assertIn("Oldest refers to the highest age.", repair_prompt)
        self.assertEqual(json.loads(result["observation"])["sample_rows"], [["Grace"]])

    def test_graph_explicitly_loads_compiled_artifact_and_uses_its_demos(self):
        self.compile_artifact()
        runtime_lm = DummyLM([
            {"logical_plan": "Sort users by age descending and return the first name."},
            generation(self.examples[0].sql_query),
        ])
        with patch(
            "src.sql_agent_graph.create_dspy_lm", return_value=runtime_lm,
        ) as configure, patch.object(SQLAgentGraph, "_initialize_langfuse"):
            agent = SQLAgentGraph(compiled_model_path=str(self.artifact_path), prompt_mode="compiled_dspy")
            result = self.run_agent(agent)
        configure.assert_called_once()
        self.assertTrue(result["is_valid"])
        self.assertIsInstance(result["plan"], str)
        self.assertEqual(result["final_sql"], self.examples[0].sql_query)
        self.assertEqual(len(runtime_lm.history), 2)
        self.assertIn(DEMO_PLAN, json.dumps(runtime_lm.history[0]["messages"]))

    def test_dspy_langfuse_generations_record_outputs_usage_and_actual_model(self):
        fallback_model = "qwen/qwen-2.5-coder-32b-instruct"

        observations = []

        @contextmanager
        def start_observation(**fields):
            observations.append(fields)
            yield SimpleNamespace(update=lambda **updates: fields.update(updates))

        telemetry = SimpleNamespace(start_as_current_observation=start_observation, flush=Mock())
        logical_plan = "Order users by age descending and select the first name."
        scripted = DummyLM([
            {"logical_plan": logical_plan},
            generation(self.examples[0].sql_query),
        ])
        lm = OpenRouterLM(model="openrouter/qwen/qwen-2.5-7b-instruct", cache=False)

        def provider_response(prompt=None, messages=None, **kwargs):
            response = scripted.forward(prompt=prompt, messages=messages, **kwargs)
            call_number = len(lm.history) + 1
            response.model = fallback_model
            response.usage.update(
                prompt_tokens=11 * call_number,
                completion_tokens=7 * call_number,
                total_tokens=18 * call_number,
                cost=0.001 * call_number,
            )
            return response

        program = TextToSQLProgram()
        program.set_lm(lm)
        agent = SQLAgentGraph(dspy_program=program, compiled_model_path=self.artifact_path)
        agent._langfuse_client = telemetry
        with patch.object(dspy.LM, "forward", side_effect=provider_response):
            result = self.run_agent(agent)

        self.assertTrue(result["is_valid"])
        root = observations[0]
        self.assertEqual(root["name"], "mars_sql_query")
        self.assertEqual(root["metadata"]["turn"], 1)
        self.assertEqual(root["metadata"]["self_correction_cycles"], 0)
        self.assertGreater(root["metadata"]["latency_seconds"], 0)
        generations = observations[1:]
        self.assertEqual([item["name"] for item in generations], ["mars_decomposer", "mars_generator"])
        self.assertEqual(generations[0]["output"], {"logical_plan": logical_plan})
        self.assertEqual(generations[1]["output"], generation(self.examples[0].sql_query))
        self.assertEqual(generations[1]["input"]["logical_plan"], logical_plan)
        for call_number, observation in enumerate(generations, start=1):
            self.assertEqual(observation["as_type"], "generation")
            self.assertEqual(observation["model"], fallback_model)
            self.assertEqual(observation["usage_details"], {
                "input": 11 * call_number,
                "output": 7 * call_number,
                "total": 18 * call_number,
            })
            self.assertAlmostEqual(observation["cost_details"]["total"], 0.001 * call_number)
            self.assertIn(self.examples[0].question, observation["input"]["question"])
            self.assertEqual(observation["input"]["schema_ddl"], self.examples[0].schema_ddl)
            self.assertEqual(observation["metadata"]["prompt_source"], "dspy")
            self.assertEqual(observation["metadata"]["artifact"], str(self.artifact_path))
        telemetry.flush.assert_called_once_with()

    def test_explicit_raw_client_bypasses_compiled_artifact(self):
        self.artifact_path.write_text("not valid JSON", encoding="utf-8")
        responses = iter([
            {
                "target_tables": ["users"],
                "target_columns": ["users.name", "users.age"],
                "join_logic": None,
                "filter_logic": None,
                "reasoning": "Order users by age descending and return the first name.",
            },
            generation(self.examples[0].sql_query),
        ])
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content=json.dumps(next(responses)))
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch("src.sql_agent_graph.create_dspy_lm") as configure:
            uncompiled = SQLAgentGraph(compiled_model_path=str(self.root / "absent.json"), prompt_mode="baseline")
            self.assertIsNone(uncompiled.dspy_program)
            agent = SQLAgentGraph(
                client=client,
                model_name="test-model",
                compiled_model_path=str(self.artifact_path),
            )
            result = self.run_agent(agent)
        configure.assert_not_called()
        self.assertEqual(len(calls), 2)
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["final_sql"], self.examples[0].sql_query)

    def test_compiled_mode_requires_compatible_artifact(self):
        with self.assertRaises(FileNotFoundError):
            SQLAgentGraph(prompt_mode="compiled_dspy", compiled_model_path=self.artifact_path)
        self.artifact_path.write_text(json.dumps({"generator": {"signature": {"fields": [{}] * 4}}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "recompile"):
            SQLAgentGraph(prompt_mode="compiled_dspy", compiled_model_path=self.artifact_path)

    def test_dspy_successful_probe_is_observed_before_final_submission(self):
        program = TextToSQLProgram()
        lm = DummyLM([
            {"logical_plan": "Inspect ages then answer."},
            generation("SELECT MAX(age) AS oldest_age FROM users", "probe"),
            generation(self.examples[0].sql_query),
        ])
        program.set_lm(lm)
        result = self.run_agent(SQLAgentGraph(dspy_program=program))
        self.assertEqual(result["turn"], 2)
        self.assertEqual(result["self_correction_cycles"], 0)
        self.assertEqual(result["final_sql"], self.examples[0].sql_query)
        prompt = json.dumps(lm.history[-1]["messages"])
        self.assertIn("oldest_age", prompt)
        self.assertIn("85", prompt)
        self.assertIn("Iteration 2 of 3", prompt)

    def test_parallel_dspy_programs_keep_models_and_histories_independent(self):
        queries = ["SELECT MIN(age) FROM users", "SELECT MAX(age) FROM users"]
        agents = []
        language_models = []
        for index, sql in enumerate(queries):
            program = TextToSQLProgram()
            lm = DummyLM([
                {"logical_plan": f"Independent plan {index}"},
                generation(sql),
            ])
            program.set_lm(lm)
            language_models.append(lm)
            agents.append(SQLAgentGraph(dspy_program=program))
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(self.run_agent, agents))
        self.assertEqual([result["final_sql"] for result in results], queries)
        self.assertEqual([result["plan"] for result in results], ["Independent plan 0", "Independent plan 1"])
        self.assertEqual([len(lm.history) for lm in language_models], [2, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
