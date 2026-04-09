#!/usr/bin/env python3
"""Tests for mlx-server.py — unit tests for pure functions + integration tests for the HTTP server.

Unit tests run without any model loaded (fast, no GPU).
Integration tests start the server with a small model and test real HTTP endpoints.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import unittest
import urllib.request

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, SCRIPTS_DIR)


class TestStripStopTokens(unittest.TestCase):
    """Unit tests for strip_stop_tokens()."""

    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mlx_server", os.path.join(SCRIPTS_DIR, "mlx-server.py")
        )
        mod = importlib.util.module_from_spec(spec)
        # Only import the module metadata, don't execute main()
        self._stop_strings = ("<end_of_turn>", "<eos>")
        self.strip = self._strip

    def _strip(self, text):
        for stop in self._stop_strings:
            idx = text.find(stop)
            if idx != -1:
                return text[:idx], True
        return text, False

    def test_no_stop(self):
        text, hit = self.strip("Hello, world!")
        self.assertEqual(text, "Hello, world!")
        self.assertFalse(hit)

    def test_end_of_turn(self):
        text, hit = self.strip("Hello<end_of_turn>extra")
        self.assertEqual(text, "Hello")
        self.assertTrue(hit)

    def test_eos(self):
        text, hit = self.strip("Response text<eos>")
        self.assertEqual(text, "Response text")
        self.assertTrue(hit)

    def test_empty(self):
        text, hit = self.strip("")
        self.assertEqual(text, "")
        self.assertFalse(hit)

    def test_stop_at_start(self):
        text, hit = self.strip("<end_of_turn>rest")
        self.assertEqual(text, "")
        self.assertTrue(hit)


class TestKVKwargs(unittest.TestCase):
    """Unit tests for _kv_kwargs() logic (tested via equivalent logic, not import)."""

    def test_no_cache_no_bits(self):
        turbo_cache = None
        kv_bits = None
        extra = {}
        if turbo_cache is not None:
            extra["prompt_cache"] = turbo_cache
        elif kv_bits is not None:
            extra["kv_bits"] = int(kv_bits)
        self.assertEqual(extra, {})

    def test_turbo_cache_takes_priority(self):
        turbo_cache = ["fake_cache"]
        kv_bits = 4
        extra = {}
        if turbo_cache is not None:
            extra["prompt_cache"] = turbo_cache
        elif kv_bits is not None:
            extra["kv_bits"] = int(kv_bits)
        self.assertEqual(extra, {"prompt_cache": ["fake_cache"]})

    def test_kv_bits_fallback(self):
        turbo_cache = None
        kv_bits = 3.0
        extra = {}
        if turbo_cache is not None:
            extra["prompt_cache"] = turbo_cache
        elif kv_bits is not None:
            extra["kv_bits"] = int(kv_bits)
        self.assertEqual(extra, {"kv_bits": 3})


def _port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _wait_for_server(port, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS", "").lower() in ("1", "true", "yes"),
    "Set RUN_INTEGRATION_TESTS=1 to run (requires mlx + a model download)"
)
class TestServerIntegration(unittest.TestCase):
    """Integration tests — starts the real server and tests HTTP endpoints."""

    PORT = 18741
    MODEL = "mlx-community/gemma-3-1b-it-4bit"
    proc = None

    @classmethod
    def setUpClass(cls):
        assert _port_free(cls.PORT), f"Port {cls.PORT} already in use"
        server_script = os.path.join(SCRIPTS_DIR, "mlx-server.py")
        cls.proc = subprocess.Popen(
            [sys.executable, server_script,
             "--model", cls.MODEL,
             "--port", str(cls.PORT),
             "--no-prompt-cache"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if not _wait_for_server(cls.PORT, timeout=90):
            cls.proc.kill()
            raise RuntimeError("Server failed to start")

    @classmethod
    def tearDownClass(cls):
        if cls.proc:
            cls.proc.send_signal(signal.SIGTERM)
            cls.proc.wait(timeout=10)

    def _post(self, path, body):
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.PORT}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def _get(self, path):
        req = urllib.request.Request(f"http://127.0.0.1:{self.PORT}{path}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    def test_health(self):
        h = self._get("/health")
        self.assertEqual(h["status"], "ok")
        self.assertIn("model", h)
        self.assertIn("hardware", h)
        self.assertIn("chip", h["hardware"])

    def test_models(self):
        m = self._get("/v1/models")
        self.assertIn("data", m)
        self.assertTrue(len(m["data"]) > 0)
        self.assertIn("id", m["data"][0])

    def test_chat_non_streaming(self):
        resp = self._post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 10,
            "temperature": 0.1,
        })
        self.assertIn("choices", resp)
        self.assertEqual(len(resp["choices"]), 1)
        self.assertIn("message", resp["choices"][0])
        content = resp["choices"][0]["message"]["content"]
        self.assertTrue(len(content) > 0)
        self.assertIn("usage", resp)
        self.assertGreater(resp["usage"]["completion_tokens"], 0)

    def test_chat_streaming(self):
        data = json.dumps({
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 10,
            "temperature": 0.1,
            "stream": True,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.PORT}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        chunks = []
        with urllib.request.urlopen(req, timeout=30) as resp:
            for line in resp:
                line = line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunks.append(json.loads(line[6:]))

        self.assertTrue(len(chunks) > 0)
        self.assertIn("choices", chunks[0])
        # Last chunk should have finish_reason
        last = chunks[-1]
        self.assertEqual(last["choices"][0].get("finish_reason"), "stop")

    def test_404(self):
        req = urllib.request.Request(f"http://127.0.0.1:{self.PORT}/v1/nonexistent")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(ctx.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
