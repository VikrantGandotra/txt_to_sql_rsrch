"""Capture provider accounting before DSPy parses or discards LM responses."""

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from math import isfinite
from time import perf_counter
from typing import Any

import dspy


_response_records: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "mars_dspy_response_records", default=None
)
_query_stages: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "mars_query_stage_metrics", default=None
)
_response_observers: ContextVar[tuple] = ContextVar("mars_response_observers", default=())


@contextmanager
def capture_lm_responses(*, include_nested: bool = False):
    """Yield records for provider responses in this execution context only.

    A record contains the raw ``response``, normalized ``usage`` dictionary,
    actual ``model``, provider-reported ``cost`` (or None), and
    ``latency_seconds``. Capturing does not depend on DSPy's optional history.
    """
    records: list[dict[str, Any]] = []
    token = _response_records.set(records)
    observers_token = _response_observers.set(
        (*_response_observers.get(), records) if include_nested else _response_observers.get()
    )
    try:
        yield records
    finally:
        _response_observers.reset(observers_token)
        _response_records.reset(token)


@contextmanager
def capture_query_metrics():
    """Collect stage accounting for one query without sharing concurrent runs."""
    stages: list[dict[str, Any]] = []
    token = _query_stages.set(stages)
    try:
        yield stages
    finally:
        _query_stages.reset(token)


def _get(value: Any, key: str, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def observation_identifiers(observation: Any) -> dict[str, str]:
    """Link local accounting to its Langfuse generation without SDK objects.

    Tracing can be absent or replaced with a test double. Retain only actual
    nonempty string IDs so either case stays JSON serializable.
    """
    identifiers = {}
    for field, attribute in (("trace_id", "trace_id"), ("observation_id", "id")):
        value = getattr(observation, attribute, None)
        if isinstance(value, str) and value.strip():
            identifiers[field] = value
    return identifiers


def _normalise(value: Any):
    if hasattr(value, "model_dump"):
        value = value.model_dump(exclude_none=True)
    if isinstance(value, Mapping):
        return {key: _normalise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    return value


def _usage_dict(value: Any) -> dict[str, Any]:
    value = _normalise(value)
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    # Also accept simple response objects used by compatible providers.
    return {
        name: _normalise(getattr(value, name))
        for name in (
            "prompt_tokens", "completion_tokens", "total_tokens", "input_tokens",
            "output_tokens", "prompt_tokens_details", "completion_tokens_details",
            "input_tokens_details", "output_tokens_details", "cost", "cost_details",
        )
        if getattr(value, name, None) is not None
    }


def _nonnegative_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not isfinite(value) or value < 0:
        return None
    return float(value)


def _token_count(value: Any) -> int | None:
    number = _nonnegative_number(value)
    return int(number) if number is not None and number.is_integer() else None


def make_response_record(response: Any, configured_model: str, elapsed: float, *, backend="openrouter") -> dict[str, Any]:
    """Normalize a provider response for both DSPy and direct API accounting."""
    usage = _usage_dict(_get(response, "usage"))
    return {
        "response": response,
        "usage": usage,
        "model": _get(response, "model") or configured_model,
        "cost": _nonnegative_number(usage.get("cost")),
        "backend": backend,
        "latency_seconds": elapsed,
    }


def _record_response(response: Any, configured_model: str, elapsed: float, *, backend="openrouter") -> None:
    records = _response_records.get()
    record = make_response_record(response, configured_model, elapsed, backend=backend)
    if records is not None:
        records.append(record)
    for observer in _response_observers.get():
        if observer is not records:
            observer.append(record)


class OpenRouterLM(dspy.LM):
    """DSPy LM that retains OpenRouter accounting before adapter parsing.

    ``forward`` receives the provider response, while DSPy callbacks and
    predictors usually receive only generated text. Capture at this boundary
    also accounts for a paid response whose text cannot be parsed by DSPy.
    """

    forward_contract = "legacy"

    def __init__(self, *args, telemetry_backend="openrouter", **kwargs):
        super().__init__(*args, **kwargs)
        self.telemetry_backend = telemetry_backend

    def forward(self, prompt=None, messages=None, **kwargs):
        started = perf_counter()
        response = super().forward(prompt=prompt, messages=messages, **kwargs)
        _record_response(response, self.model, perf_counter() - started, backend=self.telemetry_backend)
        return response

    async def aforward(self, prompt=None, messages=None, **kwargs):
        started = perf_counter()
        response = await super().aforward(prompt=prompt, messages=messages, **kwargs)
        _record_response(response, self.model, perf_counter() - started, backend=self.telemetry_backend)
        return response


def generation_telemetry(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return Langfuse generation updates using measured provider accounting.

    Missing measurements remain unknown; zero is preserved. Cached/reasoning
    tokens are already included in provider totals, so their breakdown stays
    in metadata instead of being added to the billable token totals again.
    """
    usages = [_usage_dict(record.get("usage")) for record in records]
    metadata: dict[str, Any] = {
        "lm_call_count": len(records),
        "llm_latency_seconds": sum(record.get("latency_seconds", 0.0) for record in records),
        "cost_source": "unavailable",
        "provider_usage": usages,
    }
    update: dict[str, Any] = {"metadata": metadata}
    if not records:
        return update

    models = list(dict.fromkeys(record.get("model") for record in records if record.get("model")))
    if len(models) == 1:
        update["model"] = models[0]
    elif models:
        # An adapter can make multiple calls, possibly served by different
        # fallback models. Keep this fact explicit rather than mislabeling it.
        metadata["response_models"] = models

    counts = []
    for usage in usages:
        prompt = _token_count(usage.get("prompt_tokens"))
        if prompt is None:
            prompt = _token_count(usage.get("input_tokens"))
        completion = _token_count(usage.get("completion_tokens"))
        if completion is None:
            completion = _token_count(usage.get("output_tokens"))
        total = _token_count(usage.get("total_tokens"))
        if total is None and prompt is not None and completion is not None:
            total = prompt + completion
        counts.append({"input": prompt, "output": completion, "total": total})
    usage_details = {
        key: sum(count[key] for count in counts)
        for key in ("input", "output", "total")
        if all(count[key] is not None for count in counts)
    }
    if usage_details:
        update["usage_details"] = usage_details

    # OpenRouter supplies an API charge. Self-hosted vLLM has no per-request
    # API fee; its zero explicitly excludes GPU rental and operating costs.
    # Never substitute LiteLLM's public-model price estimate for either basis.
    costs = [0.0 if record.get("backend") == "vllm" else _nonnegative_number(usage.get("cost"))
             for record, usage in zip(records, usages)]
    if all(cost is not None for cost in costs):
        update["cost_details"] = {"total": sum(costs)}
        metadata["cost_source"] = ("self_hosted_api_fee_excludes_gpu_rental"
                                   if all(record.get("backend") == "vllm" for record in records) else "openrouter")
    return update


def _accounting_fields(telemetry: dict[str, Any]) -> dict[str, Any]:
    usage = telemetry.get("usage_details", {})
    return {
        "prompt_tokens": usage.get("input"),
        "completion_tokens": usage.get("output"),
        "total_tokens": usage.get("total"),
        "cost_usd": telemetry.get("cost_details", {}).get("total"),
    }


def record_stage_metrics(
    stage: str,
    turn: int,
    records: list[dict[str, Any]],
    latency_seconds: float,
    error: str | None = None,
) -> dict[str, Any]:
    """Retain a node's measured accounting, including responses before failure.

    The returned row is JSON serializable and contains no prompts, request
    arguments, credentials, or raw SDK objects. Individual call measurements
    remain available when a stage's complete bill is unknown.
    """
    telemetry = generation_telemetry(records)
    metadata = telemetry["metadata"]
    calls = []
    for record in records:
        call_telemetry = generation_telemetry([record])
        calls.append({
            "response_id": _get(record.get("response"), "id"),
            "model": call_telemetry.get("model"),
            **_accounting_fields(call_telemetry),
            "latency_seconds": record.get("latency_seconds", 0.0),
        })
    row = {
        "stage": stage,
        "turn": turn,
        **_accounting_fields(telemetry),
        "model": telemetry.get("model"),
        "lm_call_count": metadata["lm_call_count"],
        "llm_latency_seconds": metadata["llm_latency_seconds"],
        "latency_seconds": latency_seconds,
        "error": error,
        "calls": calls,
        "cost_source": metadata["cost_source"],
    }
    stages = _query_stages.get()
    if stages is not None:
        stages.append(row)
    return row


def summarize_query_metrics(
    stages: list[dict[str, Any]], *, turn: int, latency_seconds: float
) -> dict[str, Any]:
    """Summarize a query without presenting incomplete accounting as a total.

    ``observed_*`` preserves known provider charges/tokens when any stage or
    call lacks accounting. Empty measurement sets remain unknown, rather
    than being described as free requests.
    """
    summary: dict[str, Any] = {
        "stages": list(stages),
        "sql_attempts": turn,
        "self_correction_cycles": max(
            (stage["turn"] - 1 for stage in stages if stage["stage"] == "generator"),
            default=max(turn - 1, 0),
        ),
        "lm_call_count": sum(stage["lm_call_count"] for stage in stages),
        "llm_latency_seconds": sum(stage["llm_latency_seconds"] for stage in stages),
        "latency_seconds": latency_seconds,
    }
    fields = ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd")
    for field in fields:
        measurements = [stage.get(field) for stage in stages]
        summary[field] = (
            sum(measurements)
            if measurements and all(value is not None for value in measurements)
            else None
        )
        observed = []
        for stage in stages:
            # Individual calls retain measurements even when this stage has
            # other calls with missing accounting and its total is unknown.
            calls = stage.get("calls") or [stage]
            observed.extend(call[field] for call in calls if call.get(field) is not None)
        summary[f"observed_{field}"] = sum(observed) if observed else None
    summary["accounting_complete"] = all(summary[field] is not None for field in fields)
    return summary
