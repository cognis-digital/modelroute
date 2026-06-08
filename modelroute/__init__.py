"""MODELROUTE - Local model router / proxy across Ollama, vLLM and cloud.

Standard-library-only, zero-install. Resolves a model alias to a
cost/latency-aware fallback chain across local (Ollama, vLLM) and cloud
(OpenAI, Anthropic) backends, translates an OpenAI-style chat request into
each provider's wire format, and dispatches with automatic fallback.
"""

from .core import (
    DEFAULT_PROVIDERS,
    STRATEGIES,
    Candidate,
    Model,
    Provider,
    RouteError,
    build_request,
    dispatch,
    estimate_tokens,
    list_models,
    list_providers,
    messages_tokens,
    resolve,
)

TOOL_NAME = "modelroute"
TOOL_VERSION = "1.0.0"

__all__ = [
    "TOOL_NAME",
    "TOOL_VERSION",
    "DEFAULT_PROVIDERS",
    "STRATEGIES",
    "Candidate",
    "Model",
    "Provider",
    "RouteError",
    "build_request",
    "dispatch",
    "estimate_tokens",
    "list_models",
    "list_providers",
    "messages_tokens",
    "resolve",
]
