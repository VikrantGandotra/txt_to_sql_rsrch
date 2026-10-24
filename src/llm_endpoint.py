"""One transport configuration for DSPy and OpenAI-compatible graph calls."""

from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit


def canonical_model_id(model):
    for prefix in ("openrouter/", "openai/", "hosted_vllm/"):
        if model.startswith(prefix):
            model = model[len(prefix):]
            break
    # Same pinned checkpoint, using its Hugging Face name on vLLM.
    if model == "Qwen/Qwen2.5-7B-Instruct":
        return "qwen/qwen-2.5-7b-instruct"
    return model


@dataclass(frozen=True)
class LLMEndpoint:
    backend: str
    base_url: str
    model: str
    api_key: str = field(repr=False)

    def manifest(self):
        return {"llm_backend": self.backend, "llm_base_url": self.base_url,
                "served_model": self.model,
                "cost_scope": "api_usage_fee_excludes_gpu_rental" if self.backend == "vllm" else "provider_api_charge"}


def resolve_llm_endpoint(model, *, settings=None):
    if settings is None:
        from src.config import settings
    base_url = (getattr(settings, "vllm_base_url", "") or "").strip()
    canonical = canonical_model_id(model)
    if base_url:
        parsed = urlsplit(base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("VLLM_BASE_URL must be an HTTP(S) API URL without credentials, query parameters or fragments")
        if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Remote VLLM_BASE_URL must use HTTPS to protect VLLM_API_KEY in transit")
        key = getattr(settings, "vllm_api_key", "")
        if hasattr(key, "get_secret_value"):
            key = key.get_secret_value()
        if not isinstance(key, str) or not key.strip():
            raise ValueError("VLLM_API_KEY is required when VLLM_BASE_URL is set")
        if any(character.isspace() for character in key):
            raise ValueError("VLLM_API_KEY must not contain whitespace")
        path = parsed.path.rstrip("/")
        if not path.endswith("/v1"):
            path += "/v1"
        base_url = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
        served_model = "Qwen/Qwen2.5-7B-Instruct" if canonical == "qwen/qwen-2.5-7b-instruct" else canonical
        return LLMEndpoint("vllm", base_url, served_model, key)
    key = getattr(settings, "openrouter_api_key", "")
    if not key:
        raise ValueError("Set VLLM_BASE_URL or provide OPENROUTER_API_KEY")
    return LLMEndpoint("openrouter", "https://openrouter.ai/api/v1", canonical, key)
