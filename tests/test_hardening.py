"""Hardening tests: edge cases, invalid inputs, and error paths (offline)."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelroute import RouteError, build_request, dispatch, messages_tokens, resolve
from modelroute.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _local_candidate():
    """Return a local (Ollama) candidate for 'fast' alias."""
    return [c for c in resolve("fast") if c.provider.kind == "ollama"][0]


def _capture(argv):
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        code = main(argv)
    return code, stdout_buf.getvalue(), stderr_buf.getvalue()


# ---------------------------------------------------------------------------
# build_request input validation
# ---------------------------------------------------------------------------

class TestBuildRequestValidation(unittest.TestCase):
    def setUp(self):
        self.cand = _local_candidate()

    def test_empty_messages_raises(self):
        with self.assertRaises(RouteError) as ctx:
            build_request(self.cand, [])
        self.assertIn("non-empty", str(ctx.exception))

    def test_none_messages_raises(self):
        with self.assertRaises(RouteError):
            build_request(self.cand, None)  # type: ignore[arg-type]

    def test_non_list_messages_raises(self):
        with self.assertRaises(RouteError):
            build_request(self.cand, "hello")  # type: ignore[arg-type]

    def test_message_missing_role_raises(self):
        with self.assertRaises(RouteError) as ctx:
            build_request(self.cand, [{"content": "hi"}])
        self.assertIn("role", str(ctx.exception))

    def test_message_not_dict_raises(self):
        with self.assertRaises(RouteError) as ctx:
            build_request(self.cand, ["not a dict"])
        self.assertIn("dict", str(ctx.exception))

    def test_zero_max_tokens_raises(self):
        msgs = [{"role": "user", "content": "hi"}]
        with self.assertRaises(RouteError) as ctx:
            build_request(self.cand, msgs, max_tokens=0)
        self.assertIn("max_tokens", str(ctx.exception))

    def test_negative_max_tokens_raises(self):
        msgs = [{"role": "user", "content": "hi"}]
        with self.assertRaises(RouteError):
            build_request(self.cand, msgs, max_tokens=-1)

    def test_temperature_out_of_range_raises(self):
        msgs = [{"role": "user", "content": "hi"}]
        with self.assertRaises(RouteError) as ctx:
            build_request(self.cand, msgs, temperature=3.5)
        self.assertIn("temperature", str(ctx.exception))

    def test_negative_temperature_raises(self):
        msgs = [{"role": "user", "content": "hi"}]
        with self.assertRaises(RouteError):
            build_request(self.cand, msgs, temperature=-0.1)

    def test_valid_boundary_temperature(self):
        msgs = [{"role": "user", "content": "hi"}]
        # 0.0 and 2.0 are both valid
        req = build_request(self.cand, msgs, temperature=0.0)
        self.assertIsInstance(req, dict)
        req2 = build_request(self.cand, msgs, temperature=2.0)
        self.assertIsInstance(req2, dict)


# ---------------------------------------------------------------------------
# messages_tokens edge cases
# ---------------------------------------------------------------------------

class TestMessagesTokensEdgeCases(unittest.TestCase):
    def test_empty_list_returns_zero(self):
        self.assertEqual(messages_tokens([]), 0)

    def test_none_content_handled(self):
        # content=None should not raise; treated as empty string
        result = messages_tokens([{"role": "user", "content": None}])
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 0)

    def test_missing_content_key(self):
        result = messages_tokens([{"role": "user"}])
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 0)

    def test_integer_content_handled(self):
        # Non-string content should not raise
        result = messages_tokens([{"role": "user", "content": 42}])
        self.assertIsInstance(result, int)


# ---------------------------------------------------------------------------
# dispatch edge cases
# ---------------------------------------------------------------------------

class TestDispatchEdgeCases(unittest.TestCase):
    def test_empty_chain_raises(self):
        with self.assertRaises(RouteError) as ctx:
            dispatch([], lambda c: {}, lambda r: {})
        self.assertIn("empty", str(ctx.exception))


# ---------------------------------------------------------------------------
# CLI hardening: bad arguments return non-zero + write to stderr
# ---------------------------------------------------------------------------

class TestCLIHardening(unittest.TestCase):
    def test_missing_messages_file_returns_nonzero(self):
        code, out, err = _capture(
            ["route", "fast", "--messages-file", "/nonexistent/path/no.json"]
        )
        self.assertNotEqual(code, 0)
        self.assertIn("error", err)

    def test_invalid_json_messages_file_returns_nonzero(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False) as fh:
            fh.write("not valid json }{")
            tmp = fh.name
        try:
            code, out, err = _capture(["route", "fast", "--messages-file", tmp])
            self.assertNotEqual(code, 0)
            self.assertIn("error", err)
        finally:
            os.unlink(tmp)

    def test_zero_max_tokens_returns_nonzero(self):
        code, out, err = _capture(["route", "fast", "-p", "hi", "--max-tokens", "0"])
        self.assertNotEqual(code, 0)
        self.assertIn("error", err)

    def test_negative_max_tokens_returns_nonzero(self):
        code, out, err = _capture(["route", "fast", "-p", "hi", "--max-tokens", "-5"])
        self.assertNotEqual(code, 0)
        self.assertIn("error", err)

    def test_out_of_range_temperature_returns_nonzero(self):
        code, out, err = _capture(
            ["route", "fast", "-p", "hi", "--temperature", "5.0"]
        )
        self.assertNotEqual(code, 0)
        self.assertIn("error", err)

    def test_valid_messages_file_works(self):
        msgs = [{"role": "user", "content": "hello"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False) as fh:
            json.dump(msgs, fh)
            tmp = fh.name
        try:
            code, out, err = _capture(
                ["--format", "json", "route", "fast", "--messages-file", tmp]
            )
            self.assertEqual(code, 0)
            data = json.loads(out)
            self.assertIn("fallback_chain", data)
        finally:
            os.unlink(tmp)

    def test_empty_messages_file_returns_nonzero(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False) as fh:
            json.dump([], fh)
            tmp = fh.name
        try:
            code, out, err = _capture(["route", "fast", "--messages-file", tmp])
            self.assertNotEqual(code, 0)
            self.assertIn("error", err)
        finally:
            os.unlink(tmp)


if __name__ == "__main__":
    unittest.main()
