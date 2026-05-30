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


class TestNoThinkInjection(unittest.TestCase):
    """Unit tests for _maybe_inject_no_think_instruction().

    Pins the contract that suppresses Gemma 4 thinking-mode degeneration
    documented in `m3_live_path_extractor_strip.md`.

    Module load policy: mlx-server.py's top-level imports are stdlib only —
    `mlx_lm` and `mlx_vlm` are imported inside functions we do not call. So
    importing the module here does NOT trigger model loading.
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mlx_server_for_test", os.path.join(SCRIPTS_DIR, "mlx-server.py")
        )
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def setUp(self):
        # Save + clear env so tests are deterministic regardless of host env.
        self._prev = os.environ.pop("GEMMA_DISABLE_THINKING", None)

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("GEMMA_DISABLE_THINKING", None)
        else:
            os.environ["GEMMA_DISABLE_THINKING"] = self._prev

    # --- _no_think_instruction() returns None when env unset ---
    def test_instruction_returns_none_when_env_unset(self):
        self.assertIsNone(self.mod._no_think_instruction())

    def test_instruction_returns_none_when_env_zero(self):
        os.environ["GEMMA_DISABLE_THINKING"] = "0"
        self.assertIsNone(self.mod._no_think_instruction())

    def test_instruction_returns_string_when_env_set_one(self):
        os.environ["GEMMA_DISABLE_THINKING"] = "1"
        text = self.mod._no_think_instruction()
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 10)

    def test_instruction_returns_string_when_env_set_true(self):
        os.environ["GEMMA_DISABLE_THINKING"] = "true"
        self.assertIsNotNone(self.mod._no_think_instruction())

    # --- _maybe_inject_no_think_instruction() passthrough when disabled ---
    def test_injection_passthrough_when_env_unset(self):
        msgs = [{"role": "user", "content": "hi"}]
        out = self.mod._maybe_inject_no_think_instruction(msgs)
        # Should be the same list (passthrough — env unset)
        self.assertEqual(out, msgs)

    def test_injection_does_not_mutate_input(self):
        os.environ["GEMMA_DISABLE_THINKING"] = "1"
        msgs = [{"role": "system", "content": "Be Seth."},
                {"role": "user", "content": "hi"}]
        original = [dict(m) for m in msgs]  # deep-copy
        _ = self.mod._maybe_inject_no_think_instruction(msgs)
        # Original list and dicts unchanged
        self.assertEqual(msgs, original)

    # --- _maybe_inject_no_think_instruction() behavior when enabled ---
    def test_injection_appends_to_existing_system_message(self):
        os.environ["GEMMA_DISABLE_THINKING"] = "1"
        msgs = [{"role": "system", "content": "Be Seth."},
                {"role": "user", "content": "hi"}]
        out = self.mod._maybe_inject_no_think_instruction(msgs)
        self.assertEqual(len(out), 2)  # no new message added
        self.assertEqual(out[0]["role"], "system")
        self.assertIn("Be Seth.", out[0]["content"])
        # No-think instruction is present in the merged content
        self.assertIn("final response", out[0]["content"].lower())

    def test_injection_inserts_system_message_when_none_exists(self):
        os.environ["GEMMA_DISABLE_THINKING"] = "1"
        msgs = [{"role": "user", "content": "hi"}]
        out = self.mod._maybe_inject_no_think_instruction(msgs)
        self.assertEqual(len(out), 2)  # one new system message at head
        self.assertEqual(out[0]["role"], "system")
        self.assertEqual(out[1], {"role": "user", "content": "hi"})

    def test_injection_appends_only_to_first_system_message(self):
        # Multi-system-message conversation — only the FIRST one is augmented.
        # Gemma's chat template usually merges adjacent system messages anyway.
        os.environ["GEMMA_DISABLE_THINKING"] = "1"
        msgs = [{"role": "system", "content": "First system."},
                {"role": "system", "content": "Second system."},
                {"role": "user", "content": "hi"}]
        out = self.mod._maybe_inject_no_think_instruction(msgs)
        self.assertEqual(len(out), 3)
        self.assertIn("First system.", out[0]["content"])
        self.assertIn("final response", out[0]["content"].lower())
        # Second system message untouched
        self.assertEqual(out[1]["content"], "Second system.")


class TestThinkingHeadroom(unittest.TestCase):
    """Unit tests for _thinking_headroom_tokens().

    Pins the fix for the live M3 production bug documented in
    `m3_live_path_extractor_strip.md`: the seth-lora-v4-repair adapter emits
    ~150-200 thinking tokens before the visible reply even with the no-think
    instruction active, so generation must be granted budget headroom on top
    of the caller's max_tokens or the visible reply is starved to empty.

    Module load is import-safe (top-level imports are stdlib only); see
    TestNoThinkInjection's docstring.
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mlx_server_headroom_test", os.path.join(SCRIPTS_DIR, "mlx-server.py")
        )
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def setUp(self):
        self._prev = os.environ.pop("GEMMA_THINKING_HEADROOM_TOKENS", None)

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("GEMMA_THINKING_HEADROOM_TOKENS", None)
        else:
            os.environ["GEMMA_THINKING_HEADROOM_TOKENS"] = self._prev

    def test_default_is_512_when_env_unset(self):
        # Default must be generous enough to cover the adapter's ~150-200
        # thinking tokens plus the visible reply (the live bug had 0 headroom).
        self.assertEqual(self.mod._thinking_headroom_tokens(), 512)

    def test_custom_positive_value_honored(self):
        os.environ["GEMMA_THINKING_HEADROOM_TOKENS"] = "256"
        self.assertEqual(self.mod._thinking_headroom_tokens(), 256)

    def test_zero_is_honored(self):
        # 0 is a valid explicit choice (operator opting out of headroom).
        os.environ["GEMMA_THINKING_HEADROOM_TOKENS"] = "0"
        self.assertEqual(self.mod._thinking_headroom_tokens(), 0)

    def test_negative_falls_back_to_512(self):
        os.environ["GEMMA_THINKING_HEADROOM_TOKENS"] = "-100"
        self.assertEqual(self.mod._thinking_headroom_tokens(), 512)

    def test_non_integer_falls_back_to_512(self):
        os.environ["GEMMA_THINKING_HEADROOM_TOKENS"] = "lots"
        self.assertEqual(self.mod._thinking_headroom_tokens(), 512)

    def test_whitespace_only_falls_back_to_512(self):
        os.environ["GEMMA_THINKING_HEADROOM_TOKENS"] = "   "
        self.assertEqual(self.mod._thinking_headroom_tokens(), 512)

    def test_headroom_makes_internal_budget_exceed_caller_max(self):
        # The contract the bug fix depends on: internal generation budget is
        # strictly greater than the caller's max_tokens so the visible reply
        # survives the thinking block. (Mirrors _handle_non_stream's
        # internal_max = max_tokens + _thinking_headroom_tokens().)
        caller_max = 80
        internal_max = caller_max + self.mod._thinking_headroom_tokens()
        self.assertGreater(internal_max, caller_max)
        self.assertEqual(internal_max, 80 + 512)


class TestStructuredOutputDetection(unittest.TestCase):
    """Unit tests for _system_prompt_requests_structured_output().

    Pins the detector that distinguishes structured-output callers (feed
    research reports, JSON proposers) from casual chat. Drives which no-think
    instruction variant is injected — see the 2026-05-28 runaway diagnosis in
    `m3_live_path_extractor_strip.md`.

    Module load is import-safe; see TestNoThinkInjection's docstring.
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mlx_server_structdetect_test", os.path.join(SCRIPTS_DIR, "mlx-server.py")
        )
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_casual_chat_is_not_structured(self):
        msgs = [{"role": "system", "content": "You are Seth. Be brief and natural."},
                {"role": "user", "content": "hey, you around?"}]
        self.assertFalse(self.mod._system_prompt_requests_structured_output(msgs))

    def test_empty_messages_is_not_structured(self):
        self.assertFalse(self.mod._system_prompt_requests_structured_output([]))
        self.assertFalse(self.mod._system_prompt_requests_structured_output(None))

    def test_json_request_in_system_is_structured(self):
        msgs = [{"role": "system",
                 "content": "Decide whether to propose. Respond in JSON only."},
                {"role": "user", "content": "..."}]
        self.assertTrue(self.mod._system_prompt_requests_structured_output(msgs))

    def test_single_json_object_request_is_structured(self):
        msgs = [{"role": "system",
                 "content": "Return a single JSON object with fields: decision, reason."}]
        self.assertTrue(self.mod._system_prompt_requests_structured_output(msgs))

    def test_feed_research_report_format_is_structured(self):
        # The actual runaway trigger: feed-research agent report format.
        msgs = [{"role": "system",
                 "content": "Review these feed items. For each, output in the "
                            "following format: Source, Finding, Relevance, "
                            "Priority, Suggested Action."},
                {"role": "user", "content": "Twitter: ..."}]
        self.assertTrue(self.mod._system_prompt_requests_structured_output(msgs))

    def test_marker_in_user_message_is_structured(self):
        # Detector scans user messages too, not just system.
        msgs = [{"role": "user",
                 "content": "Summarize as JSON object with keys a, b, c."}]
        self.assertTrue(self.mod._system_prompt_requests_structured_output(msgs))

    def test_assistant_message_markers_ignored(self):
        # Only system+user drive the decision — a prior assistant turn that
        # happened to mention "json" must NOT flip a casual conversation.
        msgs = [{"role": "system", "content": "You are Seth."},
                {"role": "assistant", "content": "Here is some json: {}"},
                {"role": "user", "content": "cool, thanks"}]
        self.assertFalse(self.mod._system_prompt_requests_structured_output(msgs))

    def test_multimodal_text_parts_scanned(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "respond in json please"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]}]
        self.assertTrue(self.mod._system_prompt_requests_structured_output(msgs))


class TestNoThinkStructuredVariant(unittest.TestCase):
    """Unit tests for the structured no-think instruction variant selection.

    Pins the runaway fix (2026-05-28): a structured caller prompt must receive
    the STRUCTURED instruction (which drops the contradictory "one short
    message / no bullet lists" clauses) while casual chat keeps the casual
    instruction. Both variants retain the universal "no deliberation" core.
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mlx_server_structvariant_test", os.path.join(SCRIPTS_DIR, "mlx-server.py")
        )
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def setUp(self):
        self._prev = os.environ.pop("GEMMA_DISABLE_THINKING", None)
        os.environ["GEMMA_DISABLE_THINKING"] = "1"

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("GEMMA_DISABLE_THINKING", None)
        else:
            os.environ["GEMMA_DISABLE_THINKING"] = self._prev

    def test_casual_prompt_gets_casual_instruction(self):
        msgs = [{"role": "system", "content": "You are Seth."},
                {"role": "user", "content": "hey, you around?"}]
        out = self.mod._maybe_inject_no_think_instruction(msgs)
        merged = out[0]["content"].lower()
        # Casual variant contains the "one short message" clause.
        self.assertIn("one short message", merged)

    def test_structured_prompt_gets_structured_instruction(self):
        msgs = [{"role": "system",
                 "content": "Review feed items. Output in the following format: "
                            "Source, Finding, Suggested Action."},
                {"role": "user", "content": "Reddit: ..."}]
        out = self.mod._maybe_inject_no_think_instruction(msgs)
        merged = out[0]["content"].lower()
        # Structured variant must NOT contain the contradictory clauses.
        self.assertNotIn("one short message", merged)
        self.assertNotIn("markdown bullet lists", merged)
        # But MUST retain the universal no-deliberation core.
        self.assertIn("deliberation", merged)
        self.assertIn("format the prompt specifies", merged)

    def test_both_variants_suppress_deliberation(self):
        # The universal core ("no internal deliberation / thought process") is
        # what protects working JSON callers like init_proposer; it must be
        # present in BOTH variants.
        self.assertIn("deliberation", self.mod._NO_THINK_INSTRUCTION.lower())
        self.assertIn("deliberation", self.mod._NO_THINK_INSTRUCTION_STRUCTURED.lower())
        self.assertIn("thought process", self.mod._NO_THINK_INSTRUCTION.lower())
        self.assertIn("thought process", self.mod._NO_THINK_INSTRUCTION_STRUCTURED.lower())

    def test_structured_variant_drops_contradictory_clauses(self):
        struct = self.mod._NO_THINK_INSTRUCTION_STRUCTURED.lower()
        self.assertNotIn("one short message", struct)
        self.assertNotIn("markdown bullet lists", struct)


class TestPureDeliberationGuard(unittest.TestCase):
    """Unit tests for _is_pure_deliberation().

    Pins the runaway guard (2026-05-28): when the model never closes its
    thought channel and never writes a reply line, the salvage heuristic emits
    garbage. The guard returns True so the caller can return empty instead.
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mlx_server_pdelib_test", os.path.join(SCRIPTS_DIR, "mlx-server.py")
        )
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_empty_is_not_deliberation(self):
        self.assertFalse(self.mod._is_pure_deliberation(""))
        self.assertFalse(self.mod._is_pure_deliberation(None))

    def test_clean_reply_is_not_deliberation(self):
        self.assertFalse(self.mod._is_pure_deliberation(
            "Yeah, just chilling at home, what's up?"))

    def test_unclosed_thought_channel_is_deliberation(self):
        raw = ("<|channel>thought\nThe user asked about feeds. Let me consider "
               "the format requested versus the no-think instruction...")
        self.assertTrue(self.mod._is_pure_deliberation(raw))

    def test_closed_thought_channel_with_reply_is_not_deliberation(self):
        raw = ("<|channel>thought\nThinking...<channel|>Here is the answer.")
        self.assertFalse(self.mod._is_pure_deliberation(raw))

    def test_pure_bullet_list_is_deliberation(self):
        raw = ("*   User asked X.\n"
               "*   Candidate reply A: \"Sure thing\"\n"
               "*   Candidate reply B: \"On it\"\n"
               "*   Evaluating which fits the persona...")
        self.assertTrue(self.mod._is_pure_deliberation(raw))

    def test_bullets_with_final_reply_line_is_not_deliberation(self):
        raw = ("*   Candidate A: \"Sure\"\n"
               "*   Candidate B: \"On it\"\n"
               "\n"
               "On it, leaving now.")
        self.assertFalse(self.mod._is_pure_deliberation(raw))

    def test_plain_prose_without_bullets_is_not_deliberation(self):
        # No thought marker, no bullets — even if rambly, it is a real answer.
        raw = ("I think the best approach here is to wait until tomorrow and "
               "then revisit the plan with fresh eyes.")
        self.assertFalse(self.mod._is_pure_deliberation(raw))


class TestFinalizeGeneration(unittest.TestCase):
    """Unit tests for finalize_generation() — the SHARED finalize logic used
    by both the non-stream (generate_response) and buffered-stream
    (_handle_stream_buffered) paths.

    Pins the (clean_text, is_runaway) contract so the two paths are provably
    identical: thought-stripping, the runaway guard, and the salvage heuristic
    all live in one place. (2026-05-28 streaming-guard arc.)
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mlx_server_finalize_test", os.path.join(SCRIPTS_DIR, "mlx-server.py")
        )
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_empty_returns_empty_not_runaway(self):
        clean, runaway = self.mod.finalize_generation("")
        self.assertEqual(clean, "")
        self.assertFalse(runaway)

    def test_none_returns_empty_not_runaway(self):
        clean, runaway = self.mod.finalize_generation(None)
        self.assertEqual(clean, "")
        self.assertFalse(runaway)

    def test_clean_reply_passes_through(self):
        clean, runaway = self.mod.finalize_generation(
            "Yeah, just chilling at home, what's up?")
        self.assertIn("chilling", clean.lower())
        self.assertFalse(runaway)

    def test_unclosed_thought_channel_is_runaway_empty(self):
        raw = ("<|channel>thought\nThe user asked about feeds. Let me weigh the "
               "format request against the no-think instruction and consider...")
        clean, runaway = self.mod.finalize_generation(raw)
        self.assertEqual(clean, "")
        self.assertTrue(runaway)

    def test_runaway_with_quoted_candidate_salvages_nonempty(self):
        # 2026-05-30: a runaway (unclosed thought) where the model quoted
        # candidate replies must NOT return empty — salvage the last candidate so
        # the chat turn is never empty (HU_SALVAGE_RUNAWAY default on). The real
        # fix is the reply-first/ORPO retrain; this guarantees non-empty until then.
        os.environ.pop("HU_SALVAGE_RUNAWAY", None)  # default-on
        raw = ('<|channel>thought\nThe user said "hey whats up". Candidates: '
               '"not much, you?" or "just chilling, what about you?"')
        clean, runaway = self.mod.finalize_generation(raw)
        self.assertTrue(clean.strip())          # NON-empty
        self.assertIn("you", clean.lower())     # a real candidate reply
        self.assertFalse(runaway)               # resolved by salvage

    def test_runaway_no_candidate_still_empty(self):
        # No quoted candidate to salvage -> still empty (nothing usable).
        os.environ.pop("HU_SALVAGE_RUNAWAY", None)
        raw = ("<|channel>thought\nThe user asked something; let me weigh the "
               "options carefully before deciding how to respond.")
        clean, runaway = self.mod.finalize_generation(raw)
        self.assertEqual(clean, "")
        self.assertTrue(runaway)

    def test_runaway_salvage_disabled_returns_empty(self):
        # HU_SALVAGE_RUNAWAY=0 restores strict empty-on-runaway even with candidates.
        os.environ["HU_SALVAGE_RUNAWAY"] = "0"
        try:
            raw = '<|channel>thought\nOptions: "not much, you?"'
            clean, runaway = self.mod.finalize_generation(raw)
            self.assertEqual(clean, "")
            self.assertTrue(runaway)
        finally:
            os.environ.pop("HU_SALVAGE_RUNAWAY", None)

    def test_markerless_bullets_salvage_not_runaway(self):
        # Markdown-bullet deliberation with NO channel markers: the
        # runaway-empty guard is INTENTIONALLY Case-1-only (unclosed channel),
        # because strip_thought_channels handles marker-free text and a
        # legitimate "respond in bullets" reply must NOT be nuked to empty.
        # So finalize salvages the last line and reports is_runaway=False —
        # identical to the non-stream path. The streaming guard's value is
        # parity + no token-by-token leak, not nuking bullet output.
        raw = ("*   User asked X.\n"
               "*   Candidate reply A: \"Sure thing\"\n"
               "*   Candidate reply B: \"On it\"\n"
               "*   Evaluating which fits the persona...")
        clean, runaway = self.mod.finalize_generation(raw)
        self.assertNotEqual(clean, "")
        self.assertFalse(runaway)

    def test_closed_thought_channel_keeps_reply(self):
        raw = "<|channel>thought\nthinking hard<channel|>The answer is 42."
        clean, runaway = self.mod.finalize_generation(raw)
        self.assertIn("42", clean)
        self.assertFalse(runaway)


class TestStreamShouldBuffer(unittest.TestCase):
    """Unit tests for _stream_should_buffer() — the env gate that decides
    whether the SSE path buffers + strips (default for the deliberating
    seth-lora-v4-repair model) or yields raw incremental chunks.
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mlx_server_sbuf_test", os.path.join(SCRIPTS_DIR, "mlx-server.py")
        )
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def setUp(self):
        self._saved = {
            k: os.environ.get(k)
            for k in ("HU_STREAM_BUFFER_STRIP", "GEMMA_DISABLE_THINKING")
        }

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_override_force_on(self):
        for val in ("1", "true", "yes"):
            os.environ["HU_STREAM_BUFFER_STRIP"] = val
            os.environ.pop("GEMMA_DISABLE_THINKING", None)
            self.assertTrue(self.mod._stream_should_buffer(), val)

    def test_override_force_off_beats_disable_thinking(self):
        for val in ("0", "false", "no"):
            os.environ["HU_STREAM_BUFFER_STRIP"] = val
            os.environ["GEMMA_DISABLE_THINKING"] = "1"
            self.assertFalse(self.mod._stream_should_buffer(), val)

    def test_default_on_when_disable_thinking_set(self):
        os.environ.pop("HU_STREAM_BUFFER_STRIP", None)
        os.environ["GEMMA_DISABLE_THINKING"] = "1"
        self.assertTrue(self.mod._stream_should_buffer())

    def test_default_off_when_disable_thinking_unset(self):
        os.environ.pop("HU_STREAM_BUFFER_STRIP", None)
        os.environ.pop("GEMMA_DISABLE_THINKING", None)
        self.assertFalse(self.mod._stream_should_buffer())


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


class TestStreamShouldBufferPrecedence(unittest.TestCase):
    """Unit tests for the per-request streaming-buffer override (Option B).

    Pins the precedence the gemma-realtime server uses to decide, per request,
    whether the /v1/chat/completions streaming path buffers+strips (clean, but
    TTFT == total) or yields raw token chunks (low TTFT, but leaks markerless
    deliberation). The whole point of the per-request `stream_strip` flag is to
    let h-uman's model_router ask for incremental streaming on casual/reflexive
    turns while analytical/structured turns keep the clean buffered default —
    WITHOUT touching the server's global env. See docs plan
    `2026-05-29-realtime-streaming-sota` (Option B) + scripts/eval_streaming_smoke.py.

    Precedence (highest first): per-request flag > HU_STREAM_BUFFER_STRIP env >
    model-deliberates default. Tested against the REAL production functions
    (_resolve_should_buffer / _request_stream_strip / _stream_should_buffer).

    Module load is import-safe; see TestThinkingHeadroom's docstring.
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mlx_server_bufferprec_test", os.path.join(SCRIPTS_DIR, "mlx-server.py")
        )
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def setUp(self):
        self._prev_env = os.environ.pop("HU_STREAM_BUFFER_STRIP", None)

    def tearDown(self):
        if self._prev_env is None:
            os.environ.pop("HU_STREAM_BUFFER_STRIP", None)
        else:
            os.environ["HU_STREAM_BUFFER_STRIP"] = self._prev_env

    # --- _resolve_should_buffer: the pure precedence matrix ----------------

    def test_request_true_beats_everything(self):
        # Per-request stream_strip=True forces buffering even if env says "0"
        # and the model doesn't deliberate.
        self.assertTrue(self.mod._resolve_should_buffer(True, "0", False))

    def test_request_false_beats_everything(self):
        # Per-request stream_strip=False forces incremental even if env says "1"
        # and the model deliberates (the casual-turn realtime case).
        self.assertFalse(self.mod._resolve_should_buffer(False, "1", True))

    def test_env_used_when_request_unspecified(self):
        # req_flag None => fall through to env.
        self.assertTrue(self.mod._resolve_should_buffer(None, "yes", False))
        self.assertFalse(self.mod._resolve_should_buffer(None, "no", True))

    def test_model_default_when_request_and_env_unspecified(self):
        # No request flag, no env => the model-deliberates default decides.
        self.assertTrue(self.mod._resolve_should_buffer(None, "", True))
        self.assertFalse(self.mod._resolve_should_buffer(None, "", False))

    def test_env_blank_or_garbage_falls_through_to_model_default(self):
        for junk in ("", "   ", "maybe", "2"):
            self.assertTrue(self.mod._resolve_should_buffer(None, junk, True), junk)
            self.assertFalse(self.mod._resolve_should_buffer(None, junk, False), junk)

    # --- _request_stream_strip: tolerant extraction ------------------------

    def test_extract_real_bools(self):
        self.assertIs(self.mod._request_stream_strip({"stream_strip": True}), True)
        self.assertIs(self.mod._request_stream_strip({"stream_strip": False}), False)

    def test_extract_missing_is_none(self):
        self.assertIsNone(self.mod._request_stream_strip({"messages": []}))
        self.assertIsNone(self.mod._request_stream_strip({}))

    def test_extract_non_bool_is_ignored(self):
        # A malformed field must never silently flip global policy — it falls
        # through (None) to env/model default. 1/0 are NOT treated as bools.
        for bad in ({"stream_strip": 1}, {"stream_strip": "true"},
                    {"stream_strip": None}, {"stream_strip": 0}):
            self.assertIsNone(self.mod._request_stream_strip(bad), bad)

    def test_extract_non_dict_is_none(self):
        self.assertIsNone(self.mod._request_stream_strip(None))
        self.assertIsNone(self.mod._request_stream_strip("not a dict"))

    # --- _stream_should_buffer(req): end-to-end with env control -----------

    def test_request_override_wins_over_env(self):
        # The contract the h-uman router relies on: a per-turn stream_strip=False
        # forces incremental streaming regardless of the server's global env.
        os.environ["HU_STREAM_BUFFER_STRIP"] = "1"
        self.assertFalse(self.mod._stream_should_buffer({"stream_strip": False}))
        self.assertTrue(self.mod._stream_should_buffer({"stream_strip": True}))

    def test_no_request_flag_uses_env(self):
        os.environ["HU_STREAM_BUFFER_STRIP"] = "0"
        self.assertFalse(self.mod._stream_should_buffer({"messages": []}))
        os.environ["HU_STREAM_BUFFER_STRIP"] = "1"
        self.assertTrue(self.mod._stream_should_buffer({"messages": []}))

    def test_none_request_reproduces_legacy(self):
        # Passing req=None must behave exactly like the env/default-only path.
        os.environ["HU_STREAM_BUFFER_STRIP"] = "1"
        self.assertEqual(
            self.mod._stream_should_buffer(None),
            self.mod._stream_should_buffer({}),
        )


class TestAdapterServingStatus(unittest.TestCase):
    """_load_with_adapter must FAIL-LOUD (not silently serve base) when a configured
    adapter is missing, and record tensors_loaded_global so /health's adapter_applied
    reflects reality. Regression guard for the fail-silent base-fallback that masked an
    inactive persona fine-tune (active_adapter reported a path while serving base)."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mlx_server_adapter_status", os.path.join(SCRIPTS_DIR, "mlx-server.py")
        )
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_missing_adapter_records_zero_and_serves_base(self):
        import tempfile
        sentinel_model, sentinel_tok = object(), object()
        loader = lambda name: (sentinel_model, sentinel_tok)
        self.mod.tensors_loaded_global = 999  # poison: prove it is reset, not left stale
        with tempfile.TemporaryDirectory() as d:  # no adapters.safetensors inside
            model, _tok = self.mod._load_with_adapter(loader, "base/model", d)
        self.assertIs(model, sentinel_model)                 # base weights served
        self.assertEqual(self.mod.tensors_loaded_global, 0)  # fail-loud observability
        self.assertFalse(self.mod.tensors_loaded_global > 0)  # -> adapter_applied == False

    def test_adapter_applied_records_tensor_count(self):
        import tempfile
        import os as _os
        import mlx.core as mx
        captured = {}

        class _FakeModel:
            def load_weights(self, adapters, strict=False):
                captured["n"] = len(adapters)

        loader = lambda name: (_FakeModel(), object())
        with tempfile.TemporaryDirectory() as d:
            mx.save_safetensors(
                _os.path.join(d, "adapters.safetensors"),
                {"a": mx.zeros((2, 2)), "b": mx.zeros((2, 2))},
            )
            self.mod.tensors_loaded_global = 0
            self.mod._load_with_adapter(loader, "base/model", d)
        self.assertEqual(captured["n"], 2)                   # load_weights got 2 tensors
        self.assertEqual(self.mod.tensors_loaded_global, 2)  # -> adapter_applied == True


if __name__ == "__main__":
    unittest.main()
