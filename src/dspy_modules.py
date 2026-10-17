"""DSPy prompt program shared by compilation and the LangGraph runtime."""

import json
from pathlib import Path
from typing import Literal

import dspy

from src.dspy_telemetry import OpenRouterLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPILED_MODEL_PATH = PROJECT_ROOT / "compiled_dspy_model.json"
FORMAL_MODEL = "openrouter/qwen/qwen-2.5-7b-instruct"


def provider_model_id(model: str) -> str:
    """Remove transport prefixes before sending or comparing provider IDs."""
    for prefix in ("openrouter/", "openai/"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


class DecomposerSignature(dspy.Signature):
    """Plan a SQLite query using only the supplied schema and question.

    Identify the relevant tables, projected columns, joins, filters, grouping,
    aggregation, ordering and limits. Use any evidence included in the question.
    """

    question: str = dspy.InputField(desc="User question and any supporting evidence.")
    schema_ddl: str = dspy.InputField(desc="Available SQLite table and column definitions.")
    logical_plan: str = dspy.OutputField(desc="A concise, complete execution plan.")


class GeneratorSignature(dspy.Signature):
    """Choose one read-only SQLite action, including CTEs when needed.

    Follow the logical plan and schema. If the question includes observations
    from previous actions, use the rows or repair errors without changing the
    user's intent. Set action=probe to execute exploratory SQL and continue with
    its observation. Set action=final_answer ONLY to submit the final solution.
    A successful probe does not finish the task. The last available iteration
    must submit the final answer. Return executable SQL without Markdown fences.
    """

    question: str = dspy.InputField(desc="Question, evidence, and any execution feedback.")
    schema_ddl: str = dspy.InputField(desc="Available SQLite table and column definitions.")
    logical_plan: str = dspy.InputField(desc="Plan produced by the decomposer.")
    action: Literal["probe", "final_answer"] = dspy.OutputField(desc="Explicit exploratory action or final submission.")
    rationale: str = dspy.OutputField(desc="Concise justification using the question and observations.")
    sql_query: str = dspy.OutputField(desc="Executable SQLite query; the solution when action=final_answer.")


class TextToSQLProgram(dspy.Module):
    """Two predictors whose demonstrations are learned by BootstrapFewShot.

    LangGraph invokes these same predictors separately to retain its verifier
    and correction loop; compilation invokes the complete forward chain.
    """

    def __init__(self):
        super().__init__()
        self.decomposer = dspy.Predict(DecomposerSignature)
        self.generator = dspy.Predict(GeneratorSignature)

    def forward(self, question: str, schema_ddl: str):
        decomposition = self.decomposer(question=question, schema_ddl=schema_ddl)
        generation = self.generator(
            question=question,
            schema_ddl=schema_ddl,
            logical_plan=decomposition.logical_plan,
        )
        return dspy.Prediction(
            logical_plan=decomposition.logical_plan, action=generation.action,
            rationale=generation.rationale, sql_query=generation.sql_query
        )

    def load(self, path, *args, **kwargs):
        """Reject legacy SQL-only demonstrations instead of changing semantics silently."""
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        generator = artifact.get("generator", {})
        fields = generator.get("signature", {}).get("fields", [])
        if len(fields) != len(GeneratorSignature.fields) or any(
            demo.get("action") not in {"probe", "final_answer"} or "rationale" not in demo
            for demo in generator.get("demos", [])
        ):
            raise ValueError("Compiled DSPy artifact uses incompatible trajectory signatures; recompile with python src/dspy_optimizer.py")
        return super().load(path, *args, **kwargs)


def create_dspy_lm(lm=None, *, model_name: str | None = None,
                   temperature: float = 0.8, top_p: float = 0.7, formal: bool = False):
    """Create an independent OpenRouter LM without mutating global DSPy settings.

    The openrouter/ prefix selects LiteLLM's native OpenRouter transport; it is not
    part of the model ID sent to OpenRouter. Inject an LM for offline tests.
    """
    if lm is None:
        from src.config import settings
        selected_model = model_name or (FORMAL_MODEL if formal else settings.model_name)
        if formal and provider_model_id(selected_model) != provider_model_id(FORMAL_MODEL):
            raise ValueError(f"Formal evaluation requires model {FORMAL_MODEL}")
        lm = OpenRouterLM(
            model=f"openrouter/{provider_model_id(selected_model)}",
            api_key=settings.openrouter_api_key,
            api_base="https://openrouter.ai/api/v1",
            temperature=temperature,
            top_p=top_p,
            max_tokens=2048,
            num_retries=2,
            timeout=60.0,
            cache=False,
        )
    return lm


def configure_dspy(lm=None, **kwargs):
    """Configure the main-thread compiler; runtime workers use create_dspy_lm."""
    lm = create_dspy_lm(lm, **kwargs)
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False))
    return lm
