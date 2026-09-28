"""Offline integration checks: run with ``python -m tests.test_agent_graph``."""

import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest

import httpx
from openai import OpenAI
from pydantic import ValidationError

from src.agent_state import DecompositionPlan, VerificationResult
from src.graph_pruner import DatabaseGraphPruner
from src.sql_agent_graph import SQLAgentGraph, should_continue


QUESTION = "List student_name values for students taking the Database Systems course."
EVIDENCE = "course_name refers to courses; enrollments connects students and courses."
JOIN_SQL = """SELECT s.student_name
FROM students AS s
JOIN enrollments AS e ON s.student_id = e.student_id
JOIN courses AS c ON e.course_id = c.course_id
WHERE c.course_name = 'Database Systems'
ORDER BY s.student_name"""
PLAN = {
    "target_tables": ["students", "enrollments", "courses"],
    "target_columns": ["students.student_name", "courses.course_name"],
    "join_logic": (
        "students.student_id = enrollments.student_id AND "
        "enrollments.course_id = courses.course_id"
    ),
    "filter_logic": "courses.course_name = 'Database Systems'",
    "reasoning": "Join students through enrollments, filter the course, and project names.",
}


class ScriptedClient:
    """Only replace model transport; the state graph and SQLite execute normally."""

    def __init__(self, sql_candidates):
        self.responses = [PLAN] + [
            {"thought": "Use the schema and previous database observation.", "sql_query": sql}
            for sql in sql_candidates
        ]
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        if index >= len(self.responses):
            raise AssertionError("The graph requested an unexpected extra model response.")
        message = SimpleNamespace(content=json.dumps(self.responses[index]))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class SQLAgentGraphIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory(prefix="mars_agent_graph_")
        self.addCleanup(temporary_directory.cleanup)
        self.db_path = Path(temporary_directory.name) / "school.db"
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.executescript(
                """
                CREATE TABLE students (
                    student_id INTEGER PRIMARY KEY,
                    student_name TEXT NOT NULL
                );
                CREATE TABLE enrollments (
                    enrollment_id INTEGER PRIMARY KEY,
                    student_id INTEGER,
                    course_id INTEGER
                );
                CREATE TABLE courses (
                    course_id INTEGER PRIMARY KEY,
                    course_name TEXT NOT NULL
                );
                INSERT INTO students VALUES (1, 'Ada'), (2, 'Grace'), (3, 'Linus');
                INSERT INTO courses VALUES (1, 'Database Systems'), (2, 'Networks');
                INSERT INTO enrollments VALUES (1, 1, 1), (2, 2, 1), (3, 3, 2);
                """
            )
        self.active_tables, self.schema_ddl = DatabaseGraphPruner(str(self.db_path)).prune(
            QUESTION, EVIDENCE
        )

    def make_agent(self, sql_candidates, *, expect_non_empty=True):
        client = ScriptedClient(sql_candidates)
        agent = SQLAgentGraph(
            client=client, model_name="test-model", expect_non_empty=expect_non_empty
        )
        return agent, client

    def run_agent(self, agent, **overrides):
        arguments = {
            "question": QUESTION,
            "evidence": EVIDENCE,
            "schema_ddl": self.schema_ddl,
            "db_path": str(self.db_path),
            "max_turns": 3,
        }
        arguments.update(overrides)
        return agent.run(**arguments)

    def observation(self, state):
        return VerificationResult.model_validate_json(state["observation"])

    def test_pruned_schema_and_structured_model_calls_produce_valid_join(self):
        agent, client = self.make_agent([JOIN_SQL])
        result = self.run_agent(agent)

        self.assertEqual(self.active_tables, {"students", "enrollments", "courses"})
        self.assertTrue({"decomposer", "generator", "verifier"}.issubset(
            agent.graph.get_graph().nodes
        ))
        self.assertIsInstance(result["plan"], DecompositionPlan)
        self.assertEqual(result["plan"].target_tables, PLAN["target_tables"])
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["final_sql"], JOIN_SQL)
        self.assertEqual(result["turn"], 1)
        self.assertEqual(result["history"], [{
            "sql": JOIN_SQL, "observation": result["observation"]
        }])
        observation = self.observation(result)
        self.assertEqual(observation.row_count, 2)
        self.assertEqual(observation.sample_rows, [["Ada"], ["Grace"]])
        self.assertIsNone(observation.error_message)
        self.assertEqual(should_continue(result), "end")
        self.assertEqual(len(client.calls), 2)
        for call in client.calls:
            self.assertEqual(call["model"], "test-model")
            self.assertEqual(call["temperature"], 0.0)
            self.assertEqual(call["response_format"]["type"], "json_schema")
            structured_format = call["response_format"]["json_schema"]
            self.assertTrue(structured_format["strict"])
            schema = structured_format["schema"]
            self.assertEqual(set(schema["required"]), set(schema["properties"]))
            self.assertFalse(schema["additionalProperties"])
            for property_schema in schema["properties"].values():
                self.assertNotIn("default", property_schema)
        decomposition_prompt = json.dumps(client.calls[0]["messages"])
        self.assertIn(QUESTION, decomposition_prompt)
        self.assertIn(EVIDENCE, decomposition_prompt)
        self.assertIn("enrollments", decomposition_prompt)

    def test_openrouter_fallback_is_serialized_for_both_model_requests(self):
        primary_model = "qwen/qwen2.5-coder-7b-instruct"
        fallback_model = "qwen/qwen-2.5-coder-32b-instruct"
        request_bodies = []
        responses = [PLAN, {"thought": "Join the requested course to its students.", "sql_query": JOIN_SQL}]

        def respond(request):
            request_bodies.append(json.loads(request.content))
            content = responses[len(request_bodies) - 1]
            return httpx.Response(200, json={
                "id": f"offline-fallback-{len(request_bodies)}",
                "object": "chat.completion",
                "created": 0,
                "model": fallback_model,
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps(content)},
                }],
            })

        with OpenAI(
            api_key="offline-test-key",
            base_url="https://openrouter.ai/api/v1",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        ) as client:
            agent = SQLAgentGraph(client=client, model_name=primary_model)
            result = self.run_agent(agent)

        self.assertEqual(len(request_bodies), 2)
        required_fields = [
            ("target_tables", "target_columns", "reasoning"),
            ("thought", "sql_query"),
        ]
        for body, fields in zip(request_bodies, required_fields):
            self.assertEqual(body["model"], primary_model)
            self.assertEqual(body["models"], [fallback_model])
            self.assertNotIn("extra_body", body)
            self.assertEqual(body["max_tokens"], 2048)
            self.assertEqual(body["response_format"], {"type": "json_object"})
            schema_instruction = body["messages"][0]["content"]
            self.assertIn('"properties"', schema_instruction)
            for field in fields:
                self.assertIn(f'"{field}"', schema_instruction)
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["final_sql"], JOIN_SQL)
        self.assertEqual(self.observation(result).sample_rows, [["Ada"], ["Grace"]])

    def test_sql_error_is_observed_and_sent_to_the_repair_turn(self):
        invalid_sql = "SELECT imaginary_column FROM students"
        agent, client = self.make_agent([invalid_sql, JOIN_SQL])
        result = self.run_agent(agent)

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["final_sql"], JOIN_SQL)
        self.assertEqual(result["turn"], 2)
        self.assertEqual(len(result["history"]), 2)
        first = result["history"][0]
        self.assertEqual(first["sql"], invalid_sql)
        failed = VerificationResult.model_validate_json(first["observation"])
        self.assertFalse(failed.is_valid)
        self.assertIn("no such column", failed.error_message.lower())
        self.assertEqual(failed.row_count, 0)
        repair_prompt = json.dumps(client.calls[2]["messages"])
        self.assertIn(invalid_sql, repair_prompt)
        self.assertIn("no such column", repair_prompt)
        self.assertEqual(len(client.calls), 3)

    def test_empty_result_triggers_repair_and_preserves_observation(self):
        empty_sql = "SELECT student_name FROM students WHERE student_id < 0"
        agent, client = self.make_agent([empty_sql, JOIN_SQL])
        result = self.run_agent(agent)

        self.assertEqual(result["turn"], 2)
        self.assertTrue(result["is_valid"])
        empty = VerificationResult.model_validate_json(result["history"][0]["observation"])
        self.assertFalse(empty.is_valid)
        self.assertEqual(empty.row_count, 0)
        self.assertEqual(empty.sample_rows, [])
        repair_prompt = json.dumps(client.calls[2]["messages"])
        self.assertIn(empty_sql, repair_prompt)
        self.assertIn("row_count", repair_prompt)

    def test_failed_queries_stop_at_turn_limit_without_final_sql(self):
        invalid_sql = "SELECT missing_column FROM students"
        for limit in (1, 3, 15):
            with self.subTest(max_turns=limit):
                agent, client = self.make_agent([invalid_sql] * limit)
                result = self.run_agent(agent, max_turns=limit)
                self.assertFalse(result["is_valid"])
                self.assertIsNone(result["final_sql"])
                self.assertEqual(result["turn"], limit)
                self.assertEqual(len(result["history"]), limit)
                self.assertEqual(len(client.calls), limit + 1)
                self.assertEqual(should_continue(result), "end")
                retry_state = dict(result, turn=limit - 1)
                self.assertEqual(should_continue(retry_state), "generator")

    def test_explicitly_allowed_empty_results_and_zero_counts_are_valid(self):
        cases = [
            ("SELECT student_name FROM students WHERE student_id < 0", False, 0, []),
            ("SELECT COUNT(*) FROM students WHERE student_id < 0", True, 1, [[0]]),
        ]
        for sql, expect_non_empty, row_count, rows in cases:
            with self.subTest(sql=sql):
                agent, client = self.make_agent([sql], expect_non_empty=expect_non_empty)
                result = self.run_agent(agent)
                self.assertTrue(result["is_valid"])
                self.assertEqual(result["final_sql"], sql)
                self.assertEqual(result["turn"], 1)
                observation = self.observation(result)
                self.assertEqual(observation.row_count, row_count)
                self.assertEqual(observation.sample_rows, rows)
                self.assertEqual(len(client.calls), 2)

    def test_verifier_rejects_database_mutation_and_preserves_rows(self):
        agent, _ = self.make_agent(["DELETE FROM students"])
        result = self.run_agent(agent, max_turns=1)

        self.assertFalse(result["is_valid"])
        self.assertIsNone(result["final_sql"])
        self.assertTrue(self.observation(result).error_message)
        with closing(sqlite3.connect(self.db_path)) as connection:
            rows = connection.execute(
                "SELECT student_name FROM students ORDER BY student_id"
            ).fetchall()
        self.assertEqual(rows, [("Ada",), ("Grace",), ("Linus",)])

    def test_verifier_counts_all_rows_but_keeps_five_json_serializable_samples(self):
        sql = """WITH RECURSIVE numbers(n) AS (
            SELECT 1 UNION ALL SELECT n + 1 FROM numbers WHERE n < 1005
        ) SELECT n, X'00FF' FROM numbers ORDER BY n"""
        agent, _ = self.make_agent([sql])
        result = self.run_agent(agent)

        self.assertTrue(result["is_valid"])
        observation = self.observation(result)
        self.assertEqual(observation.row_count, 1005)
        self.assertEqual(observation.sample_rows, [[n, "00ff"] for n in range(1, 6)])

    def test_attach_cannot_create_another_database(self):
        attached_path = self.db_path.with_name("attached.db")
        sql_path = attached_path.as_posix().replace("'", "''")
        agent, _ = self.make_agent([f"ATTACH DATABASE '{sql_path}' AS extra"])
        result = self.run_agent(agent, max_turns=1)

        self.assertFalse(result["is_valid"])
        self.assertIsNone(result["final_sql"])
        self.assertTrue(self.observation(result).error_message)
        self.assertFalse(attached_path.exists())

    def test_invalid_model_output_propagates_instead_of_becoming_sql_failure(self):
        for model_name in ("test-model", "qwen/qwen2.5-coder-7b-instruct"):
            with self.subTest(model=model_name):
                client = ScriptedClient([JOIN_SQL])
                agent = SQLAgentGraph(client=client, model_name=model_name)
                client.responses[1] = {"thought": "Missing the required SQL field."}
                with self.assertRaises(ValidationError):
                    self.run_agent(agent)
                self.assertEqual(len(client.calls), 2)

    def test_missing_database_is_not_created(self):
        missing_path = self.db_path.with_name("does_not_exist.db")
        agent, _ = self.make_agent(["SELECT 1"])
        try:
            result = self.run_agent(agent, db_path=str(missing_path), max_turns=1)
        except FileNotFoundError:
            pass
        else:
            self.assertFalse(result["is_valid"])
            self.assertIsNone(result["final_sql"])
            self.assertTrue(self.observation(result).error_message)
        self.assertFalse(missing_path.exists())

    def test_invalid_turn_limits_fail_before_model_calls(self):
        for invalid_limit in (0, -1, 1.5, True):
            with self.subTest(max_turns=invalid_limit):
                agent, client = self.make_agent([JOIN_SQL])
                with self.assertRaises((TypeError, ValueError)):
                    self.run_agent(agent, max_turns=invalid_limit)
                self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
