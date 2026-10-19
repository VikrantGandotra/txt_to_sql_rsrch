"""Explicit probe/observe/final-answer trajectories with measured model calls."""

import json
import logging
import sqlite3
from threading import Lock
import dspy
from contextlib import closing, nullcontext
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any, Literal, TypeVar

from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig, RunnableLambda
from pydantic import BaseModel

from src.agent_state import (
    AgentState,
    DecompositionPlan,
    GeneratedSQL,
    VerificationResult,
)
from src.dspy_modules import (
    COMPILED_MODEL_PATH,
    FORMAL_MODEL,
    TextToSQLProgram,
    create_dspy_lm,
    provider_model_id,
)
from src.dspy_telemetry import (
    capture_lm_responses,
    capture_query_metrics,
    generation_telemetry,
    make_response_record,
    observation_identifiers,
    record_stage_metrics,
    summarize_query_metrics,
)

StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)
logger = logging.getLogger(__name__)
_verbose_print_lock = Lock()


def should_continue(state: AgentState) -> Literal["generator", "end"]:
    """Probes always feed observations back; only final submission or budget ends."""
    if state.get("is_final", False) or state["turn"] >= state["max_turns"]:
        return "end"
    return "generator"


class SQLAgentGraph:
    """Generate and execute SQLite queries using the existing OpenRouter setup.

    A supplied client must expose ``chat.completions.create``; supply its model
    name as well. Otherwise settings and the traced client are loaded lazily.
    ``run`` returns the full state, including each SQL/observation pair. A failed
    run keeps its last candidate in ``current_sql`` and leaves ``final_sql`` None.
    Choose compiled_dspy, uncompiled_dspy or baseline explicitly; artifacts never
    switch the runtime implicitly. An injected client uses the baseline API path.
    SQLite execution has a configurable time budget. Validity means execution
    succeeded and met the configured empty-result rule;
    it is not a guarantee of semantic correctness against the user's question.
    """

    def __init__(
        self,
        client: Any = None,
        model_name: str | None = None,
        *,
        expect_non_empty: bool = False,
        compiled_model_path: str | Path | None = None,
        dspy_program: TextToSQLProgram | None = None,
        prompt_mode: Literal["compiled_dspy", "uncompiled_dspy", "baseline"] = "uncompiled_dspy",
        temperature: float = 0.8,
        top_p: float = 0.7,
        formal: bool = False,
        execution_timeout: float = 30.0,
        verbose: bool = False,
    ):
        if client is not None and not model_name:
            raise ValueError("model_name is required when supplying a client")
        if client is not None and dspy_program is not None:
            raise ValueError("Supply either a client or a DSPy program, not both")
        if prompt_mode not in {"compiled_dspy", "uncompiled_dspy", "baseline"}:
            raise ValueError("Unknown prompt_mode")
        if formal and model_name is not None and provider_model_id(model_name) != provider_model_id(FORMAL_MODEL):
            raise ValueError(f"Formal evaluation requires model {FORMAL_MODEL}")
        if execution_timeout <= 0:
            raise ValueError("execution_timeout must be positive")
        self.client = client
        self.model_name = FORMAL_MODEL if formal else model_name
        self.prompt_mode = "baseline" if client is not None else prompt_mode
        self.temperature = temperature
        self.top_p = top_p
        self.formal = formal
        self.execution_timeout = execution_timeout
        self.verbose = verbose
        self.expect_non_empty = expect_non_empty
        self._langfuse_client = None
        self._langfuse_public_key = None
        self.dspy_program = dspy_program
        self.compiled_model_path = (
            Path(compiled_model_path) if compiled_model_path is not None else COMPILED_MODEL_PATH
        )
        if client is None and self.dspy_program is None and prompt_mode != "baseline":
            self.dspy_program = TextToSQLProgram()
            if prompt_mode == "compiled_dspy":
                if not self.compiled_model_path.is_file():
                    raise FileNotFoundError(f"Compiled DSPy artifact not found: {self.compiled_model_path}")
                if formal:
                    artifact = json.loads(self.compiled_model_path.read_text(encoding="utf-8"))
                    manifest = artifact.get("mars_generator_manifest", {})
                    if (manifest.get("version") != 2 or
                        manifest.get("action_contract") != "probe_final_answer" or
                        provider_model_id(manifest.get("model_name", "")) != provider_model_id(FORMAL_MODEL) or
                        manifest.get("max_iterations") != 5):
                        raise ValueError("Formal DSPy generation requires a compatible 7B trajectory compilation manifest; recompile")
                self.dspy_program.load(str(self.compiled_model_path))
                if not self.dspy_program.decomposer.demos or not self.dspy_program.generator.demos:
                    raise ValueError("Compiled DSPy mode requires demonstrations for both predictors")
            lm = create_dspy_lm(model_name=self.model_name, temperature=temperature, top_p=top_p, formal=formal)
            self.dspy_program.set_lm(lm)
            self._initialize_langfuse()

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

    def _initialize_langfuse(self) -> None:
        from src.config import settings
        from langfuse import Langfuse

        self._langfuse_public_key = settings.langfuse_public_key
        self._langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_base_url,
        )

    def _call_dspy(self, stage: str, state: AgentState, **inputs):
        predictor = getattr(self.dspy_program, stage)
        metadata = {
            "prompt_source": "dspy",
            "artifact": str(self.compiled_model_path),
            "turn": state["turn"] + (stage == "generator"),
            "candidate_index": state.get("candidate_index"),
        }
        started = perf_counter()
        trace = (
            self._langfuse_client.start_as_current_observation(
                as_type="generation", name=f"mars_{stage}", input=inputs, metadata=metadata,
            ) if self._langfuse_client is not None else nullcontext(None)
        )
        with trace as observation:
            # Capture provider responses before DSPy's adapter parses text. Paid
            # calls remain measurable even when parsing fails or history is off.
            with capture_lm_responses() as responses:
                failure = None
                try:
                    config = {"seed": state["rollout_seed"]} if "rollout_seed" in state else {}
                    with dspy.context(adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
                        prediction = predictor(**inputs, config=config)
                    if self.formal:
                        if not responses:
                            raise ValueError("Formal DSPy call did not expose provider model metadata")
                        for response in responses:
                            raw = response["response"]
                            actual = raw.get("model") if isinstance(raw, dict) else getattr(raw, "model", None)
                            self._validate_actual_model(actual)
                    if observation is not None:
                        observation.update(output=dict(prediction))
                    return prediction
                except Exception as error:
                    failure = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    stage_metrics = record_stage_metrics(
                        stage, metadata["turn"], responses, perf_counter() - started, failure,
                    )
                    stage_metrics.update(observation_identifiers(observation))
                    details = generation_telemetry(responses)
                    details["metadata"] = {
                        **metadata,
                        **details.get("metadata", {}),
                        **{key: stage_metrics[key] for key in (
                            "prompt_tokens", "completion_tokens", "total_tokens",
                            "cost_usd", "latency_seconds",
                        )},
                    }
                    if observation is not None:
                        observation.update(**details)

    @staticmethod
    def _dspy_question(state: AgentState, *, include_feedback: bool = False) -> str:
        question = state["question"]
        if state["evidence"]:
            question += f"\n\nEvidence: {state['evidence']}"
        if include_feedback and (state["history"] or state["observation"]):
            question += (
                f"\n\nPrevious SQL attempts: {json.dumps(state['history'])}"
                f"\nLatest execution observation: {state['observation']}"
                "\nUse these <observation> records to refine the plan and answer the original question."
            )
        if include_feedback:
            question += (
                f"\nIteration {state['turn'] + 1} of {state['max_turns']}; "
                f"{state['max_turns'] - state['turn']} decisions remaining. "
                "Use action=probe for exploration or action=final_answer with the final <solution> SQL. "
                "On the last iteration submit final_answer."
            )
        return question

    def _validate_actual_model(self, actual_model: str | None) -> None:
        if self.formal and provider_model_id(actual_model or "") != provider_model_id(FORMAL_MODEL):
            raise ValueError(f"Formal evaluation received model {actual_model!r}; required {FORMAL_MODEL}")

    def _structured_output(
        self,
        output_type: type[StructuredOutput],
        messages: list[dict[str, str]],
        name: str,
        *,
        turn: int = 0,
        rollout_seed: int | None = None,
    ) -> StructuredOutput:
        if self.client is None:
            from src.config import settings
            from langfuse.openai import OpenAI

            if self._langfuse_client is None:
                self._initialize_langfuse()
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
        # Some compatible providers ignore json_schema. Keep the output contract
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
            "model": provider_model_id(self.model_name),
            "messages": request_messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
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
        if rollout_seed is not None:
            request["seed"] = rollout_seed
        if provider_model_id(self.model_name) == provider_model_id(FORMAL_MODEL):
            # These Qwen routes support JSON mode rather than schema enforcement.
            # model_validate_json below still validates every returned field.
            request["response_format"] = {"type": "json_object"}
        if self._langfuse_client is not None:
            request["name"] = name
            request["langfuse_public_key"] = self._langfuse_public_key
        response = None
        failure = None
        started = perf_counter()
        try:
            response = self.client.chat.completions.create(**request)
            self._validate_actual_model(getattr(response, "model", None))
            if not response.choices:
                raise ValueError(f"{name} returned no completion choices")
            message = response.choices[0].message
            if getattr(message, "refusal", None):
                raise ValueError(f"{name} refused the request: {message.refusal}")
            if not message.content:
                raise ValueError(f"{name} returned no structured output")
            return output_type.model_validate_json(message.content)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            raise
        finally:
            elapsed = perf_counter() - started
            responses = [make_response_record(response, self.model_name, elapsed)] if response is not None else []
            record_stage_metrics(name.removeprefix("mars_"), turn, responses, elapsed, failure)

    def decomposer(self, state: AgentState) -> dict[str, Any]:
        """Plan the joins, filters, aggregation and projection before synthesis."""
        self._validate_max_turns(state["max_turns"])
        if self.dspy_program is not None:
            result = self._call_dspy(
                "decomposer",
                state,
                question=self._dspy_question(state),
                schema_ddl=state["schema_ddl"],
            )
            if not isinstance(result.logical_plan, str) or not result.logical_plan.strip():
                raise ValueError("DSPy decomposer returned an empty logical plan")
            return {"plan": result.logical_plan}
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
            turn=state["turn"],
            rollout_seed=state.get("rollout_seed"),
        )
        return {"plan": plan}

    def generator(self, state: AgentState) -> dict[str, Any]:
        """Synthesize a candidate using the plan and all previous observations."""
        if state["plan"] is None:
            raise ValueError("The generator requires a decomposition plan")
        if self.dspy_program is not None:
            plan = state["plan"]
            result = self._call_dspy(
                "generator",
                state,
                question=self._dspy_question(state, include_feedback=True),
                schema_ddl=state["schema_ddl"],
                logical_plan=plan if isinstance(plan, str) else plan.reasoning,
            )
            candidate = GeneratedSQL.model_validate(dict(result))
        else:
            candidate = self._structured_output(
                GeneratedSQL,
                [
                    {
                        "role": "system",
                        "content": (
                            "Generate one read-only SQLite SELECT query (CTEs allowed) "
                            "using the supplied schema and plan. Choose action=probe to "
                            "explore; its successful rows or errors return as observations. "
                            "Choose action=final_answer only to submit the final solution. "
                            "Use previous observations to revise the plan if needed. Preserve the "
                            "question's meaning; do not remove required filters just to "
                            "get rows. On the last iteration submit final_answer. "
                            "Return JSON with action, rationale and sql_query; the query "
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
                            f"<observation>\n{state['observation'] or 'No execution yet.'}\n</observation>"
                        ),
                    },
                ],
                "mars_generator",
                turn=state["turn"] + 1,
                rollout_seed=state.get("rollout_seed"),
            )
        prior_error = bool(state["observation"] and not json.loads(state["observation"])["is_valid"])
        return {
            "current_sql": candidate.sql_query.strip(),
            "current_action": candidate.action,
            "rationale": candidate.rationale,
            "is_final": candidate.action == "final_answer",
            "turn": state["turn"] + 1,
            "self_correction_cycles": state.get("self_correction_cycles", 0) + int(prior_error),
            "is_valid": False,
            "final_sql": None,
        }

    def verifier(self, state: AgentState) -> dict[str, Any]:
        """Execute and observe each action; only explicit final submissions become answers."""
        sql = state["current_sql"] or ""
        try:
            if not sql.strip():
                raise sqlite3.OperationalError("The generated SQL query is empty")
            # mode=ro also prevents silently creating a database for a bad path.
            uri = Path(state["db_path"]).resolve().as_uri() + "?mode=ro"
            with closing(sqlite3.connect(uri, uri=True, timeout=self.execution_timeout)) as connection:
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
                deadline = monotonic() + self.execution_timeout
                connection.set_progress_handler(lambda: monotonic() >= deadline, 1000)
                cursor = connection.execute(sql)
                if cursor.description is None:
                    raise sqlite3.OperationalError("The query did not return a result set")
                row_count = 0
                columns = [column[0] for column in cursor.description]
                sample_rows = []
                while rows := cursor.fetchmany(1000):
                    row_count += len(rows)
                    for row in rows[: max(0, 50 - len(sample_rows))]:
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
                columns=columns,
            )
        except sqlite3.Error as error:
            result = VerificationResult(
                is_valid=False,
                error_message=f"{type(error).__name__}: {error}",
            )

        observation = result.model_dump_json()
        if self.verbose:
            label = f"[rollout {state.get('candidate_index', 0) + 1} turn {state['turn']}]"
            route = "END" if state.get("is_final", False) or state["turn"] >= state["max_turns"] else "generator"
            with _verbose_print_lock:
                print(f"{label} action={state['current_action']} SQL: {sql}\n"
                      f"{label} SQLite observation fed to model: {observation}\n"
                      f"{label} route={route} "
                      f"(explicit_final={state.get('is_final', False)}, valid={result.is_valid})", flush=True)
        return {
            "observation": observation,
            "history": state["history"] + [{
                "action": state["current_action"], "rationale": state["rationale"],
                "sql": sql, "observation": observation,
            }],
            "sql_execution_count": state.get("sql_execution_count", 0) + 1,
            "is_valid": result.is_valid,
            "final_sql": sql if state.get("is_final", False) and result.is_valid else None,
            "termination_reason": (
                "final_answer" if state.get("is_final", False) and result.is_valid
                else "invalid_final_answer" if state.get("is_final", False)
                else "budget_exhausted" if state["turn"] >= state["max_turns"]
                else "continuing"
            ),
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

    def invoke(self, state: AgentState, config: RunnableConfig | None = None) -> AgentState:
        """Trace one complete query, including repairs, then export its metrics."""
        self._validate_max_turns(state["max_turns"])
        if self._langfuse_client is None and self.client is None and self.dspy_program is None:
            self._initialize_langfuse()
        final_state = dict(state)
        metadata = {
            "prompt_source": "dspy" if self.dspy_program is not None else "baseline",
            "db_id": state.get("db_id") or Path(state["db_path"]).stem,
            "max_turns": state["max_turns"],
            "prompt_mode": self.prompt_mode,
            "model": self.model_name,
            "candidate_index": state.get("candidate_index"),
            "rollout_seed": state.get("rollout_seed"),
        }
        if "question_id" in state:
            metadata["question_id"] = state["question_id"]
        trace = (
            self._langfuse_client.start_as_current_observation(
                name="mars_sql_query", as_type="span",
                input={"question": state["question"], "evidence": state["evidence"]},
                metadata=metadata,
            )
            if self._langfuse_client is not None else nullcontext(None)
        )
        started = perf_counter()
        try:
            with trace as observation, capture_query_metrics() as stage_metrics:
                if observation is not None:
                    from langfuse import propagate_attributes

                    attributes = propagate_attributes(
                        trace_name="mars_sql_query",
                        metadata={key: str(value) for key, value in metadata.items()},
                    )
                else:
                    attributes = nullcontext()
                with attributes:
                    failure = None
                    try:
                        # Keep the last completed state even when a later model
                        # call fails. This also works for the BIRD .invoke path.
                        for snapshot in self.graph.stream(state, config=config, stream_mode="values"):
                            final_state = dict(snapshot)
                        return final_state
                    except Exception as error:
                        failure = error
                        raise
                    finally:
                        corrections = final_state.get("self_correction_cycles", 0)
                        if (failure is not None and stage_metrics and
                            stage_metrics[-1]["stage"] == "generator" and
                            stage_metrics[-1]["turn"] > final_state["turn"] and
                            final_state.get("observation") and
                            not json.loads(final_state["observation"])["is_valid"]):
                            corrections += 1
                        query_metrics = summarize_query_metrics(
                            stage_metrics, turn=final_state["turn"],
                            latency_seconds=perf_counter() - started,
                        )
                        query_metrics.update({
                            "turn": final_state["turn"],
                            "sql_execution_count": final_state.get("sql_execution_count", 0),
                            "self_correction_cycles": corrections,
                            "termination_reason": "model_error" if failure else final_state.get("termination_reason"),
                            "is_valid": final_state["is_valid"],
                            "trace_id": getattr(observation, "trace_id", None),
                        })
                        final_state["metrics"] = query_metrics
                        final_state["self_correction_cycles"] = query_metrics["self_correction_cycles"]
                        final_state["latency_seconds"] = query_metrics["latency_seconds"]
                        if failure is not None:
                            # Preserve the original exception type while making
                            # paid calls before failure available to the batch report.
                            failure.mars_metrics = query_metrics
                            final_state["termination_reason"] = "model_error"
                            failure.mars_state = dict(final_state)
                        if observation is not None:
                            metrics = {key: value for key, value in query_metrics.items() if key != "stages"}
                            observation.update(
                                output={"final_sql": final_state["final_sql"], **metrics},
                                metadata={**metadata, **metrics},
                            )
                            # Keep totals as query metadata, not billed generation
                            # usage: copying them onto a parent generation would
                            # double-count its children's tokens and cost.
                            # The v4 SDK sets trace metadata via propagation;
                            # updating only observation metadata is insufficient.
                            with propagate_attributes(
                                metadata={key: str(value) for key, value in metrics.items() if value is not None}
                            ):
                                pass
        finally:
            if self._langfuse_client is not None:
                try:
                    self._langfuse_client.flush()
                except Exception:
                    logger.warning("Could not flush SQL agent traces", exc_info=True)

    def run(
        self,
        question: str,
        evidence: str,
        schema_ddl: str,
        db_path: str,
        max_turns: int = 5,
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
            "sql_execution_count": 0,
            "self_correction_cycles": 0,
            "is_final": False,
            "max_turns": max_turns,
            "is_valid": False,
            "final_sql": None,
        }
        # One decomposition plus two nodes per attempt, with room for END.
        return self.invoke(state, {"recursion_limit": 2 * max_turns + 3})


def build_sql_graph(
    client: Any = None,
    model_name: str | None = None,
    *,
    expect_non_empty: bool = False,
    compiled_model_path: str | Path | None = None,
    dspy_program: TextToSQLProgram | None = None,
    prompt_mode: Literal["compiled_dspy", "uncompiled_dspy", "baseline"] = "uncompiled_dspy",
    temperature: float = 0.8,
    top_p: float = 0.7,
    formal: bool = False,
    execution_timeout: float = 30.0,
    verbose: bool = False,
):
    """Build a traced graph runnable accepting a complete AgentState."""
    agent = SQLAgentGraph(
        client=client,
        model_name=model_name,
        expect_non_empty=expect_non_empty,
        compiled_model_path=compiled_model_path,
        dspy_program=dspy_program,
        prompt_mode=prompt_mode, temperature=temperature, top_p=top_p,
        formal=formal, execution_timeout=execution_timeout,
        verbose=verbose,
    )
    return RunnableLambda(agent.invoke, name="mars_sql_query")
