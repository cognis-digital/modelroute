"""Smoke tests for MODELROUTE (offline, no network)."""

import json
import os
import sys
import unittest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from modelroute import (  # noqa: E402
    TOOL_NAME,
    TOOL_VERSION,
    RouteError,
    build_request,
    dispatch,
    estimate_tokens,
    messages_tokens,
    resolve,
)
from modelroute.cli import main  # noqa: E402

MSGS = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "Hi there, explain caching."},
]


class TestRouting(unittest.TestCase):
    def test_meta(self):
        self.assertEqual(TOOL_NAME, "modelroute")
        self.assertTrue(TOOL_VERSION)

    def test_local_first_orders_locals_ahead(self):
        chain = resolve("fast", strategy="local-first", have_keys=True)
        self.assertTrue(chain[0].is_local)
        # cloud candidates still present somewhere in the chain
        self.assertTrue(any(not c.is_local for c in chain))

    def test_no_keys_drops_cloud(self):
        chain = resolve("fast", strategy="local-first", have_keys=False)
        self.assertTrue(all(c.is_local for c in chain))

    def test_cheapest_puts_zero_cost_first(self):
        chain = resolve("fast", strategy="cheapest", in_tok=1000,
                        out_tok=1000, have_keys=True)
        self.assertEqual(chain[0].est_cost(1000, 1000), 0.0)

    def test_quality_prefers_cloud(self):
        chain = resolve("smart", strategy="quality", have_keys=True)
        self.assertFalse(chain[0].is_local)

    def test_unknown_alias_raises(self):
        with self.assertRaises(RouteError):
            resolve("does-not-exist", have_keys=True)

    def test_bad_strategy_raises(self):
        with self.assertRaises(RouteError):
            resolve("fast", strategy="nope", have_keys=True)


class TestRequestTranslation(unittest.TestCase):
    def test_ollama_shape(self):
        c = resolve("fast", strategy="local-first")[0]
        req = build_request(c, MSGS, max_tokens=128, temperature=0.2)
        self.assertTrue(req["url"].endswith("/api/chat"))
        self.assertEqual(req["body"]["options"]["num_predict"], 128)
        self.assertFalse(req["body"]["stream"])

    def test_anthropic_splits_system(self):
        c = [x for x in resolve("smart", have_keys=True)
             if x.provider.kind == "anthropic"][0]
        req = build_request(c, MSGS)
        self.assertIn("x-api-key", req["headers"])
        self.assertEqual(req["body"]["system"], "You are terse.")
        self.assertTrue(all(m["role"] != "system" for m in req["body"]["messages"]))

    def test_openai_bearer(self):
        c = [x for x in resolve("smart", have_keys=True)
             if x.provider.kind == "openai"][0]
        req = build_request(c, MSGS)
        self.assertIn("Authorization", req["headers"])
        self.assertTrue(req["url"].endswith("/chat/completions"))


class TestTokens(unittest.TestCase):
    def test_estimate(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertGreater(estimate_tokens("hello world foo"), 0)

    def test_messages_tokens_positive(self):
        self.assertGreater(messages_tokens(MSGS), 0)


class TestDispatchFallback(unittest.TestCase):
    def test_fallback_on_failure(self):
        chain = resolve("fast", strategy="local-first")
        calls = {"n": 0}

        def sender(req):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("primary down")
            return {"ok": True}

        result = dispatch(chain, lambda c: build_request(c, MSGS), sender)
        self.assertTrue(result["fell_back"])
        self.assertEqual(result["attempts"], 2)

    def test_all_fail_raises(self):
        chain = resolve("fast", strategy="local-first")

        def sender(req):
            raise RuntimeError("down")

        with self.assertRaises(RouteError):
            dispatch(chain, lambda c: build_request(c, MSGS), sender)


class TestCLI(unittest.TestCase):
    def _capture(self, argv):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def test_route_json(self):
        code, out = self._capture(["--format", "json", "route", "fast",
                                   "-p", "hello"])
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertIn("fallback_chain", data)
        self.assertTrue(data["chosen"]["is_local"])

    def test_providers_table(self):
        code, out = self._capture(["providers"])
        self.assertEqual(code, 0)
        self.assertIn("ollama-local", out)

    def test_models_filter(self):
        code, out = self._capture(["--format", "json", "models", "smart"])
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertTrue(len(data["models"]) >= 1)

    def test_unknown_alias_nonzero(self):
        code, _ = self._capture(["route", "zzz", "-p", "hi"])
        self.assertEqual(code, 1)

    def test_no_subcommand_returns_2(self):
        code, _ = self._capture([])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
