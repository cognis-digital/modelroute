"""MODELROUTE core engine.

A local model router / proxy across Ollama, vLLM and cloud providers
(OpenAI, Anthropic) with cost/latency-aware fallback chains.

The engine is fully offline and deterministic: it resolves a model alias
to an ordered list of provider candidates, builds the concrete HTTP
request each provider would receive (translating a single OpenAI-style
chat request into Ollama / vLLM / Anthropic wire formats), and estimates
cost. Actually firing the request is a thin, separately-injectable step
so the planning logic can be tested without any network access.

Standard library only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

# Latency tiers: lower is faster/closer. Local engines beat remote.
TIER_LOCAL = 0
TIER_REGIONAL = 1
TIER_CLOUD = 2


@dataclass(frozen=True)
class Model:
    """A concrete model served by a provider."""

    name: str  # provider-native model id, e.g. "llama3.1:8b"
    aliases: Tuple[str, ...]  # routable aliases, e.g. ("llama3", "fast")
    ctx: int  # context window in tokens
    in_cost: float  # USD per 1M input tokens (0 for local)
    out_cost: float  # USD per 1M output tokens (0 for local)


@dataclass(frozen=True)
class Provider:
    """An inference backend."""

    name: str
    kind: str  # "ollama" | "vllm" | "openai" | "anthropic"
    base_url: str
    tier: int
    models: Tuple[Model, ...]
    requires_key: bool = False

    def find(self, alias: str) -> Optional[Model]:
        alias = alias.lower()
        for m in self.models:
            if m.name.lower() == alias or alias in (a.lower() for a in m.aliases):
                return m
        return None


# Default registry. Local-first, then regional self-host, then cloud.
DEFAULT_PROVIDERS: Tuple[Provider, ...] = (
    Provider(
        name="ollama-local",
        kind="ollama",
        base_url="http://localhost:11434",
        tier=TIER_LOCAL,
        models=(
            Model("llama3.1:8b", ("llama3", "fast", "default"), 8192, 0.0, 0.0),
            Model("qwen2.5:7b", ("qwen", "coder"), 32768, 0.0, 0.0),
            Model("mistral:7b", ("mistral",), 8192, 0.0, 0.0),
        ),
    ),
    Provider(
        name="vllm-host",
        kind="vllm",
        base_url="http://localhost:8000",
        tier=TIER_REGIONAL,
        models=(
            Model(
                "meta-llama/Llama-3.1-8B-Instruct",
                ("llama3", "fast"),
                131072,
                0.0,
                0.0,
            ),
            Model("Qwen/Qwen2.5-7B-Instruct", ("qwen", "coder"), 32768, 0.0, 0.0),
        ),
    ),
    Provider(
        name="openai",
        kind="openai",
        base_url="https://api.openai.com/v1",
        tier=TIER_CLOUD,
        requires_key=True,
        models=(
            Model("gpt-4o-mini", ("fast", "default", "smart"), 128000, 0.15, 0.60),
            Model("gpt-4o", ("smart", "big"), 128000, 2.50, 10.00),
        ),
    ),
    Provider(
        name="anthropic",
        kind="anthropic",
        base_url="https://api.anthropic.com/v1",
        tier=TIER_CLOUD,
        requires_key=True,
        models=(
            Model(
                "claude-3-5-haiku-20241022", ("fast", "haiku"), 200000, 0.80, 4.00
            ),
            Model(
                "claude-3-5-sonnet-20241022",
                ("smart", "big", "sonnet"),
                200000,
                3.00,
                15.00,
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

STRATEGIES = ("local-first", "cheapest", "fastest", "quality")


@dataclass
class Candidate:
    provider: Provider
    model: Model

    @property
    def is_local(self) -> bool:
        return self.provider.kind in ("ollama", "vllm")

    def est_cost(self, in_tok: int, out_tok: int) -> float:
        return (in_tok / 1_000_000) * self.model.in_cost + (
            out_tok / 1_000_000
        ) * self.model.out_cost

    def to_dict(self, in_tok: int = 0, out_tok: int = 0) -> dict:
        return {
            "provider": self.provider.name,
            "kind": self.provider.kind,
            "base_url": self.provider.base_url,
            "model": self.model.name,
            "tier": self.provider.tier,
            "ctx": self.model.ctx,
            "requires_key": self.provider.requires_key,
            "is_local": self.is_local,
            "est_cost_usd": round(self.est_cost(in_tok, out_tok), 6),
        }


class RouteError(Exception):
    pass


def _matches(providers: Tuple[Provider, ...], alias: str) -> List[Candidate]:
    out: List[Candidate] = []
    for p in providers:
        m = p.find(alias)
        if m is not None:
            out.append(Candidate(p, m))
    return out


def _order(
    cands: List[Candidate], strategy: str, in_tok: int, out_tok: int
) -> List[Candidate]:
    if strategy == "local-first":
        # locals first (cheapest local cost ties broken by tier), then by cost.
        def _lf_key(c: Candidate):
            return (not c.is_local, c.provider.tier, c.est_cost(in_tok, out_tok))

        return sorted(cands, key=_lf_key)
    if strategy == "cheapest":
        return sorted(
            cands, key=lambda c: (c.est_cost(in_tok, out_tok), c.provider.tier)
        )
    if strategy == "fastest":
        return sorted(
            cands, key=lambda c: (c.provider.tier, c.est_cost(in_tok, out_tok))
        )
    if strategy == "quality":
        # Higher output cost is a (rough) proxy for capability; locals last.
        return sorted(
            cands, key=lambda c: (c.is_local, -c.model.out_cost, c.provider.tier)
        )
    raise RouteError(f"unknown strategy: {strategy}")


def resolve(
    alias: str,
    strategy: str = "local-first",
    in_tok: int = 0,
    out_tok: int = 0,
    providers: Tuple[Provider, ...] = DEFAULT_PROVIDERS,
    have_keys: bool = False,
) -> List[Candidate]:
    """Resolve an alias to an ordered fallback chain of candidates."""
    if strategy not in STRATEGIES:
        raise RouteError(f"unknown strategy: {strategy!r} (choose from {STRATEGIES})")
    cands = _matches(providers, alias)
    if not have_keys:
        cands = [c for c in cands if not c.provider.requires_key]
    if not cands:
        raise RouteError(f"no provider serves alias {alias!r} (have_keys={have_keys})")
    return _order(cands, strategy, in_tok, out_tok)


# ---------------------------------------------------------------------------
# Request translation (OpenAI-style chat -> provider wire format)
# ---------------------------------------------------------------------------


def build_request(
    cand: Candidate,
    messages: List[dict],
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> dict:
    """Translate a normalized chat request into a provider-specific HTTP call.

    Returns a dict describing method/url/headers/body so a caller can fire it
    with urllib (no network performed here).

    Raises RouteError for invalid inputs (empty messages, bad types, out-of-range
    numeric parameters) so callers always get a clean error, never a TypeError.
    """
    if not isinstance(messages, list) or len(messages) == 0:
        raise RouteError("messages must be a non-empty list")
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise RouteError(f"messages[{i}] must be a dict, got {type(msg).__name__}")
        if "role" not in msg:
            raise RouteError(f"messages[{i}] is missing required key 'role'")
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise RouteError(f"max_tokens must be a positive integer, got {max_tokens!r}")
    if not isinstance(temperature, (int, float)) or not (0.0 <= temperature <= 2.0):
        raise RouteError(
            f"temperature must be a float in [0.0, 2.0], got {temperature!r}"
        )
    kind = cand.provider.kind
    base = cand.provider.base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}

    if kind in ("openai", "vllm"):
        url = f"{base}/chat/completions"
        if kind == "openai":
            headers["Authorization"] = "Bearer $OPENAI_API_KEY"
        body = {
            "model": cand.model.name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    elif kind == "ollama":
        url = f"{base}/api/chat"
        body = {
            "model": cand.model.name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
    elif kind == "anthropic":
        url = f"{base}/messages"
        headers["x-api-key"] = "$ANTHROPIC_API_KEY"
        headers["anthropic-version"] = "2023-06-01"
        system = " ".join(m["content"] for m in messages if m.get("role") == "system")
        chat = [m for m in messages if m.get("role") != "system"]
        body = {
            "model": cand.model.name,
            "messages": chat,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            body["system"] = system
    else:  # pragma: no cover - registry is closed
        raise RouteError(f"unknown provider kind: {kind}")

    return {"method": "POST", "url": url, "headers": headers, "body": body}


# ---------------------------------------------------------------------------
# Token estimation (offline heuristic) + dispatch with fallback
# ---------------------------------------------------------------------------

_WORD = re.compile(r"\w+|[^\w\s]")


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~chars/4 with a word floor)."""
    if not text:
        return 0
    words = len(_WORD.findall(text))
    return max(words, len(text) // 4)


def messages_tokens(messages: List[dict]) -> int:
    """Estimate total tokens for a message list.

    Non-string content (None, int, list, etc.) is treated as empty string.
    Missing or empty messages list returns 0.
    """
    if not messages:
        return 0
    total = 0
    for m in messages:
        raw = m.get("content", "") if isinstance(m, dict) else ""
        content = raw if isinstance(raw, str) else ""
        total += estimate_tokens(content) + 3
    return total


def dispatch(
    chain: List[Candidate],
    request_builder: Callable[[Candidate], dict],
    sender: Callable[[dict], dict],
) -> dict:
    """Try each candidate in order; return first success, else raise.

    `sender(req) -> dict` performs the actual call. It is injected so the
    planning path is testable offline. A sender may raise to signal failure;
    the next candidate is attempted (fallback).

    Raises RouteError immediately when chain is empty.
    """
    if not chain:
        raise RouteError("dispatch called with an empty candidate chain")
    errors: List[str] = []
    for cand in chain:
        req = request_builder(cand)
        try:
            resp = sender(req)
            return {
                "provider": cand.provider.name,
                "model": cand.model.name,
                "attempts": len(errors) + 1,
                "fell_back": bool(errors),
                "errors": errors,
                "response": resp,
            }
        except Exception as exc:  # noqa: BLE001 - fallback on any failure
            errors.append(f"{cand.provider.name}: {exc}")
    raise RouteError("all providers failed: " + " | ".join(errors))


def list_models(providers: Tuple[Provider, ...] = DEFAULT_PROVIDERS) -> List[dict]:
    rows = []
    for p in providers:
        for m in p.models:
            rows.append(
                {
                    "provider": p.name,
                    "kind": p.kind,
                    "model": m.name,
                    "aliases": list(m.aliases),
                    "ctx": m.ctx,
                    "in_cost": m.in_cost,
                    "out_cost": m.out_cost,
                }
            )
    return rows


def list_providers(providers: Tuple[Provider, ...] = DEFAULT_PROVIDERS) -> List[dict]:
    return [
        {
            "provider": p.name,
            "kind": p.kind,
            "base_url": p.base_url,
            "tier": p.tier,
            "requires_key": p.requires_key,
            "n_models": len(p.models),
        }
        for p in providers
    ]
