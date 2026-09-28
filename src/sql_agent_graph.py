"""Milestone 3: a Decomposer -> Generator -> Verifier self-correction graph."""

import json
import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from time import monotonic
from typing import Any, Literal, TypeVar

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from src.agent_state import (
    AgentState,
    DecompositionPlan,
    GeneratedSQL,
    VerificationResult,
)

StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)
logger = logging.getLogger(__name__)
QWEN_CODER_FALLBACK = "qwen/qwen-2.5-coder-32b-instruct"


def should_continue(state: AgentState) -> Literal["generator", "end"]:
    """Retry failed verification only while another execution turn remains."""
    if state["is_valid"] or state["turn"] >= state["max_turns"]:
        return "end"
    return "generator"


class SQLAgentGraph:
    """Generate and execute SQLite queries using the existing OpenRouter setup.

    A supplied client must expose ``chat.completions.create``; supply its model
    name as well. Otherwise settings and the traced client are loaded lazily.
    ``run`` returns the full state, including each SQL/observation pair. A failed
    run keeps its last candidate in ``current_sql`` and leaves ``final_sql`` None.
    SQLite execution has a five-second time budget. Validity means execution
    succeeded and met the configured empty-result rule;
    it is not a guarantee of semantic correctness against the user's question.
    """

    def __init__(
        self,
        client: Any = None,
        model_name: str | None = None,
        *,
        expect_non_empty: bool = True,
    ):
        if client is not None and not model_name:
            raise ValueError("model_name is required when supplying a client")
        self.client = client
        self.model_name = model_name
        self.expect_non_empty = expect_non_empty
        self._langfuse_client = None
        self._langfuse_public_key = None

        workflow = StateGraph(AgentState)
        workflow.add_node("decomposer", self.decomposer)
        workflow.add_node("generator", self.generator)
        workflow.add_node("verifier", self.verifier)
        workflow.add_edge(START, "decomposer")
        workflow.add_edge("decomposer", "generator")
        workflow.add_edge("generator", "verifier")
        workflow.add_conditional_edges(
            "verifier", should_continue, {"generator": "generator", "end": END}
        )
        self.graph = workflow.compile()

    def _structured_output(
        self,
        output_type: type[StructuredOutput],
        messages: list[dict[str, str]],
        name: str,
    ) -> StructuredOutput:
        if self.client is None:
            from src.config import settings
            from langfuse import Langfuse
            from langfuse.openai import OpenAI

            self._langfuse_public_key = settings.langfuse_public_key
            self._langfuse_client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                base_url=settings.langfuse_base_url,
            )
            self.client = OpenAI(
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                timeout=60.0,
                max_retries=2,
                default_headers={
                    "HTTP-Referer": "https://github.com/mars-ablation",
                    "X-Title": "MARS-SQL Serverless Ablation",
                },
            )
            self.model_name = self.model_name or settings.model_name

        # Both model-output schemas are flat objects. Strict structured outputs
        # require every property, including nullable ones, and forbid extra keys.
        schema = output_type.model_json_schema()
        schema["required"] = list(schema["properties"])
        schema["additionalProperties"] = False
        for prop in schema["properties"].values():
            prop.pop("default", None)
        # Some fallback providers ignore json_schema. Keep the output contract
        # in the prompt as well, without modifying the caller's messages.
        schema_instruction = (
            "\n\nReturn only a valid JSON object matching this JSON Schema. "
            "Include every required field. Use JSON null for unused nullable "
            "fields. Do not include Markdown fences, comments, or other text.\n"
            + json.dumps(schema)
        )
        request_messages = [
            {**messages[0], "content": messages[0]["content"] + schema_instruction},
            *messages[1:],
        ]
        request = {
            "model": self.model_name,
            # OpenRouter takes fallback IDs in an array, not in the model string.
            "extra_body": {"models": [QWEN_CODER_FALLBACK]},
            "messages": request_messages,
            "temperature": 0.0,
            # Bound output instead of reserving the provider's full token limit.
            "max_tokens": 2048,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": output_type.__name__,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        if self.model_name in {"qwen/qwen2.5-coder-7b-instruct", QWEN_CODER_FALLBACK}:
            # These Qwen routes support JSON mode rather than schema enforcement.
            # model_validate_json below still validates every returned field.
            request["response_format"] = {"type": "json_object"}
        if self._langfuse_client is not None:
            request["name"] = name
            request["langfuse_public_key"] = self._langfuse_public_key
        response = self.client.chat.completions.create(**request)
        if not response.choices:
            raise ValueError(f"{name} returned no completion choices")
        message = response.choices[0].message
        if getattr(message, "refusal", None):
            raise ValueError(f"{name} refused the request: {message.refusal}")
        if not message.content:
            raise ValueError(f"{name} returned no structured output")
        return output_type.model_validate_json(message.content)

    def decomposer(self, state: AgentState) -> dict[str, Any]:
        """Plan the joins, filters, aggregation and projection before synthesis."""
        self._validate_max_turns(state["max_turns"])
        plan = self._structured_output(
            DecompositionPlan,
            [
                {
                    "role": "system",
                    "content": (
                        "Plan a SQLite query using only the supplied schema. Identify "
                        "tables, columns, explicit join conditions, filters, and any "
                        "aggregation, grouping, ordering or limits in your reasoning. "
                        "Use the question and evidence to resolve intent. Return the "
                        "requested JSON plan; use null for unused join/filter logic."
                    ),
                },
                {"role": "user", "content": self._task_context(state)},
            ],
            "mars_decomposer",
        )
        return {"plan": plan}

    def generator(self, state: AgentState) -> dict[str, Any]:
        """Synthesize a candidate using the plan and all previous observations."""
        if state["plan"] is None:
            raise ValueError("The generator requires a decomposition plan")
        candidate = self._structured_output(
            GeneratedSQL,
            [
                {
                    "role": "system",
                    "content": (
                        "Generate one read-only SQLite SELECT query (CTEs allowed) "
                        "using the supplied schema and plan. On a retry, repair the "
                        "previous query using the database observations. Preserve the "
                        "question's meaning; do not remove required filters just to "
                        "get rows. Return JSON with thought and sql_query; the query "
                        "must contain executable SQL without Markdown fences."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{self._task_context(state)}\n\n"
                        f"Plan:\n{state['plan'].model_dump_json()}\n\n"
                        f"Attempt: {state['turn'] + 1} of {state['max_turns']}\n"
                        f"Previous attempts:\n{json.dumps(state['history'])}\n"
                        f"Latest observation:\n{state['observation'] or 'No execution yet.'}"
                    ),
                },
            ],
            "mars_generator",
        )
        return {
            "current_sql": candidate.sql_query.strip(),
            "is_valid": False,
            "final_sql": None,
        }

    def verifier(self, state: AgentState) -> dict[str, Any]:
        """Execute without modifying the database; count rows and retain five."""
        sql = state["current_sql"] or ""
        try:
            if not sql.strip():
                raise sqlite3.OperationalError("The generated SQL query is empty")
            # mode=ro also prevents silently creating a database for a bad path.
            uri = Path(state["db_path"]).resolve().as_uri() + "?mode=ro"
            with closing(sqlite3.connect(uri, uri=True, timeout=5.0)) as connection:
                connection.execute("PRAGMA query_only = ON")
                read_actions = {
                    sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
                    sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE,
                }
                # Restrict statements as well as writes: ATTACH can create a file
                # even when the main database was opened in read-only mode.
                connection.set_authorizer(
                    lambda action, *_: (
                        sqlite3.SQLITE_OK if action in read_actions else sqlite3.SQLITE_DENY
                    )
                )
                deadline = monotonic() + 5.0
                connection.set_progress_handler(lambda: monotonic() >= deadline, 1000)
                cursor = connection.execute(sql)
                if cursor.description is None:
                    raise sqlite3.OperationalError("The query did not return a result set")
                row_count = 0
                sample_rows = []
                while rows := cursor.fetchmany(1000):
                    row_count += len(rows)
                    for row in rows[: max(0, 5 - len(sample_rows))]:
                        # BLOBs need an explicit representation to keep JSON valid.
                        sample_rows.append(
                            [value.hex() if isinstance(value, bytes) else value for value in row]
                        )
            empty_error = self.expect_non_empty and row_count == 0
            result = VerificationResult(
                is_valid=not empty_error,
                error_message=(
                    "Query returned an empty result set; records were expected. "
                    "Check joins and filters against the question and evidence."
                    if empty_error else None
                ),
                row_count=row_count,
                sample_rows=sample_rows,
            )
        except sqlite3.Error as error:
            result = VerificationResult(
                is_valid=False,
                error_message=f"{type(error).__name__}: {error}",
            )

        observation = result.model_dump_json()
        return {
            "observation": observation,
            "history": state["history"] + [{"sql": sql, "observation": observation}],
            "turn": state["turn"] + 1,
            "is_valid": result.is_valid,
            "final_sql": sql if result.is_valid else None,
        }

    @staticmethod
    def _task_context(state: AgentState) -> str:
        return (
            f"Question: {state['question']}\nEvidence: {state['evidence']}\n"
            f"Schema DDL:\n{state['schema_ddl']}"
        )

    @staticmethod
    def _validate_max_turns(max_turns: int) -> None:
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")

    def run(
        self,
        question: str,
        evidence: str,
        schema_ddl: str,
        db_path: str,
        max_turns: int = 3,
    ) -> AgentState:
        """Run a fresh graph; set expect_non_empty=False for legitimate empty sets."""
        self._validate_max_turns(max_turns)
        state: AgentState = {
            "question": question,
            "evidence": evidence,
            "schema_ddl": schema_ddl,
            "db_path": db_path,
            "plan": None,
            "current_sql": None,
            "observation": None,
            "history": [],
            "turn": 0,
            "max_turns": max_turns,
            "is_valid": False,
            "final_sql": None,
        }
        try:
            # One decomposition plus two nodes per attempt, with room for END.
            return self.graph.invoke(state, {"recursion_limit": 2 * max_turns + 3})
        finally:
            if self._langfuse_client is not None:
                try:
                    self._langfuse_client.flush()
                except Exception:
                    # Telemetry failure must not hide the SQL result or its error.
                    logger.warning("Could not flush SQL agent traces", exc_info=True)


def build_sql_graph(
    client: Any = None,
    model_name: str | None = None,
    *,
    expect_non_empty: bool = True,
):
    """Build a compiled graph for callers that supply a complete AgentState."""
    return SQLAgentGraph(
        client=client,
        model_name=model_name,
        expect_non_empty=expect_non_empty,
    ).graph
