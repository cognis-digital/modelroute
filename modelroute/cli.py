"""Command-line interface for MODELROUTE."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import TOOL_NAME, TOOL_VERSION
from .core import (
    STRATEGIES,
    RouteError,
    build_request,
    dispatch,
    estimate_tokens,
    list_models,
    list_providers,
    messages_tokens,
    resolve,
)


def _emit(payload, fmt: str, rows: Optional[List[tuple]] = None) -> None:
    if fmt == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if rows:
        widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
        for r in rows:
            print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))
        return
    print(json.dumps(payload, indent=2, sort_keys=True))


def _load_messages(args: argparse.Namespace) -> List[dict]:
    """Build a chat message list from --prompt/--system/--messages-file/stdin."""
    if getattr(args, "messages_file", None):
        with open(args.messages_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "messages" in data:
            data = data["messages"]
        if not isinstance(data, list):
            raise RouteError("messages file must be a JSON list or {messages:[...]}")
        return data
    msgs: List[dict] = []
    if getattr(args, "system", None):
        msgs.append({"role": "system", "content": args.system})
    prompt = getattr(args, "prompt", None)
    if prompt is None and not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            prompt = piped
    if prompt:
        msgs.append({"role": "user", "content": prompt})
    if not msgs:
        msgs = [{"role": "user", "content": "Hello"}]
    return msgs


def _cmd_route(args: argparse.Namespace) -> int:
    messages = _load_messages(args)
    in_tok = messages_tokens(messages)
    out_tok = args.max_tokens
    chain = resolve(
        args.model,
        strategy=args.strategy,
        in_tok=in_tok,
        out_tok=out_tok,
        have_keys=args.have_keys,
    )
    chosen = chain[0]
    request = build_request(chosen, messages, max_tokens=args.max_tokens,
                            temperature=args.temperature)
    payload = {
        "alias": args.model,
        "strategy": args.strategy,
        "in_tokens": in_tok,
        "max_tokens": out_tok,
        "chosen": chosen.to_dict(in_tok, out_tok),
        "fallback_chain": [c.to_dict(in_tok, out_tok) for c in chain],
        "request": request,
    }
    rows = [("rank", "provider", "kind", "model", "local", "est_cost_usd")]
    for i, c in enumerate(chain):
        d = c.to_dict(in_tok, out_tok)
        rows.append((i, d["provider"], d["kind"], d["model"],
                     d["is_local"], f"{d['est_cost_usd']:.6f}"))
    _emit(payload, args.format, rows)
    return 0


def _cmd_providers(args: argparse.Namespace) -> int:
    data = list_providers()
    rows = [("provider", "kind", "tier", "key", "models", "base_url")]
    for p in data:
        rows.append((p["provider"], p["kind"], p["tier"],
                     p["requires_key"], p["n_models"], p["base_url"]))
    _emit({"providers": data}, args.format, rows)
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    data = list_models()
    if args.alias:
        data = [m for m in data if args.alias.lower() in
                [a.lower() for a in m["aliases"]] or m["model"].lower() == args.alias.lower()]
    rows = [("provider", "model", "aliases", "ctx", "in$/1M", "out$/1M")]
    for m in data:
        rows.append((m["provider"], m["model"], ",".join(m["aliases"]),
                     m["ctx"], m["in_cost"], m["out_cost"]))
    _emit({"models": data}, args.format, rows)
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    """Plan a route and simulate dispatch with a list of failing providers."""
    messages = _load_messages(args)
    in_tok = messages_tokens(messages)
    chain = resolve(args.model, strategy=args.strategy, in_tok=in_tok,
                    out_tok=args.max_tokens, have_keys=args.have_keys)
    failing = set(x.strip() for x in (args.fail or "").split(",") if x.strip())

    def sender(req: dict) -> dict:
        # Identify provider by url; fail those named in --fail.
        for c in chain:
            if c.provider.base_url.rstrip("/") in req["url"] and \
                    c.model.name == req["body"]["model"]:
                if c.provider.name in failing:
                    raise RouteError(f"simulated outage on {c.provider.name}")
                return {"ok": True, "echo_model": req["body"]["model"]}
        raise RouteError("no matching candidate")

    result = dispatch(chain, lambda c: build_request(c, messages,
                      max_tokens=args.max_tokens, temperature=args.temperature),
                      sender)
    _emit(result, args.format)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=TOOL_NAME,
                                description="Local model router/proxy with fallback.")
    p.add_argument("--version", action="version",
                   version=f"{TOOL_NAME} {TOOL_VERSION}")
    p.add_argument("--format", choices=("table", "json"), default="table")
    sub = p.add_subparsers(dest="cmd")

    def add_chat_args(sp):
        sp.add_argument("model", help="model alias or native id (e.g. 'fast', 'llama3')")
        sp.add_argument("-s", "--strategy", choices=STRATEGIES, default="local-first")
        sp.add_argument("-p", "--prompt", help="user prompt text")
        sp.add_argument("--system", help="system prompt")
        sp.add_argument("--messages-file", help="JSON file with chat messages")
        sp.add_argument("--max-tokens", type=int, default=512)
        sp.add_argument("--temperature", type=float, default=0.7)
        sp.add_argument("--have-keys", action="store_true",
                        help="include cloud providers that require API keys")

    sp_route = sub.add_parser("route", help="resolve alias to a fallback chain + request plan")
    add_chat_args(sp_route)
    sp_route.set_defaults(func=_cmd_route)

    sp_sim = sub.add_parser("simulate", help="route + dispatch with simulated outages")
    add_chat_args(sp_sim)
    sp_sim.add_argument("--fail", help="comma-separated provider names to fail")
    sp_sim.set_defaults(func=_cmd_simulate)

    sp_prov = sub.add_parser("providers", help="list configured providers")
    sp_prov.set_defaults(func=_cmd_providers)

    sp_mod = sub.add_parser("models", help="list models (optionally filter by alias)")
    sp_mod.add_argument("alias", nargs="?", help="filter to this alias/model id")
    sp_mod.set_defaults(func=_cmd_models)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 2
    try:
        return args.func(args)
    except RouteError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
