"""Exception diagnostics without locals, credentials, or authorization headers."""

import os
import re
import sys
import traceback


def _known_secrets():
    secrets = [value for key, value in os.environ.items()
               if any(word in key.lower() for word in ("api_key", "secret", "token", "password"))
               and len(value) >= 6]
    # Settings may read .env directly rather than populate os.environ. Do not
    # import config here: inspecting diagnostics must work without credentials.
    settings = getattr(sys.modules.get("src.config"), "settings", None)
    if settings is not None:
        for name in ("openrouter_api_key", "langfuse_public_key", "langfuse_secret_key"):
            value = getattr(settings, name, None)
            if isinstance(value, str) and value:
                secrets.append(value)
    return sorted(secrets, key=len, reverse=True)


def redact_text(text, *, secrets=None):
    text = str(text)
    for secret in _known_secrets() if secrets is None else secrets:
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"\b(?:sk|pk)-(?:or-v1-|lf-)?[A-Za-z0-9_-]{8,}", "[REDACTED]", text)
    text = re.sub(r"(?i)\bBearer\s+[^\s\"',}]+", "Bearer [REDACTED]", text)
    return text


def redact_payload(value, *, secrets=None):
    """Redact diagnostic strings at the persistence boundary, including stages."""
    if secrets is None:
        secrets = _known_secrets()
    if isinstance(value, dict):
        return {key: redact_payload(item, secrets=secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_payload(item, secrets=secrets) for item in value]
    return redact_text(value, secrets=secrets) if isinstance(value, str) else value


def exception_details(error, phase):
    return {
        "phase": phase,
        "type": type(error).__name__,
        "message": redact_text(error),
        # format_exception includes chained exceptions, but never frame locals.
        "traceback": redact_text("".join(traceback.format_exception(error))),
    }
