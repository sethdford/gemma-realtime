#!/usr/bin/env python3
"""
OpenAI-compatible API server for local MLX model inference.
Serves Gemma 4 models (31B, E4B, E2B) on http://127.0.0.1:8741/v1

Usage:
    python3 scripts/mlx-server.py [--model mlx-community/gemma-4-31b-it-4bit] [--port 8741]
    python3 scripts/mlx-server.py --model mlx-community/gemma-4-e4b-it-4bit --realtime
    python3 scripts/mlx-server.py --speculative-draft ~/.human/training-data/adapters/seth-lora-e2b

Features:
    - Uses mlx_lm for text (fast path, no vision overhead) with mlx_vlm fallback for multimodal
    - Speculative decoding: E2B draft model proposes tokens, target verifies in parallel (~2x speedup)
    - TurboQuant KV cache compression (4.6x smaller, ~0.98x FP16 speed)
    - Cross-turn prompt caching (system prompt KV reused across requests via cache trim)
    - Optional IOSurface staging via libgemma_hw (see GEMMA_IOSURFACE_KV_* env vars, /health)
    - Apple Silicon hardware detection (M5 TensorOps, Neural Accelerators, Metal version)
    - Real-time voice mode: optimized for low TTFT with aggressive KV compression
    - PLE-safe model validation for Gemma 4 (warns about broken quantizations)

The server exposes:
    POST /v1/chat/completions  — OpenAI-compatible chat endpoint (supports stream:true)
    GET  /v1/models            — list available models
    GET  /health               — health check (includes hardware info + tok/s stats)

Environment (see guides/08-inference-sota-roadmap.md):
  MLX_SPECULATIVE_TOKENS, GEMMA_SPECULATIVE_TOKENS — draft tokens per speculative step (default 4)
  GEMMA_KV_ASYMMETRIC=1 — enable asymmetric KV (same as --kv-asymmetric)
  GEMMA_IOSURFACE_KV, GEMMA_IOSURFACE_KV_BYTES — IOSurface staging for /health
  MLX_MODEL, MLX_PORT, MLX_ADAPTER_PATH — server defaults
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import persona_steering as ps  # noqa: E402  (sibling module; activation steering)

DEFAULT_MODEL = "mlx-community/gemma-4-31b-it-4bit"
DEFAULT_PORT = 8741
HUMAN_CONFIG = os.path.expanduser("~/.human/config.json")


def _install_mlx_lm_patches():
    """Monkey-patch mlx_lm to fix two known issues against current Gemma 4:

    1. RotatingKVCache.to_quantized — stock mlx_lm 0.31.x raises NotImplementedError.
       We provide a conversion: pack the current dense K/V into a QuantizedKVCache.
       The result loses rotating-window semantics (becomes unbounded), but for chat
       contexts well below the model's sliding window this is invisible. Saves ~4x
       bytes per token during decode once the cache is quantized.

    2. speculative_generate_step — model forward may return a LanguageModelOutput
       dataclass (instead of an mx.array) for VLM-wrapped models. The function
       does `logits[:, -n_predict:, :]` which fails. We patch the call site by
       wrapping `model()` and `draft_model()` so their outputs are always mx.arrays.
    """
    import mlx.core as mx
    from mlx_lm.models import cache as _cache_mod
    from mlx_lm.models.cache import RotatingKVCache, QuantizedKVCache

    if not getattr(RotatingKVCache, "_hu_quant_patched", False):
        def _rotating_to_quantized(self, group_size: int = 64, bits: int = 4):
            qc = QuantizedKVCache(group_size=group_size, bits=bits)
            if self.keys is None or self.offset == 0:
                return qc
            keys = self._temporal_order(self.keys)
            values = self._temporal_order(self.values)
            keys = keys[..., : self.offset, :]
            values = values[..., : self.offset, :]
            qc.update_and_fetch(keys, values)
            return qc

        RotatingKVCache.to_quantized = _rotating_to_quantized
        RotatingKVCache._hu_quant_patched = True
        print("  [patch] RotatingKVCache.to_quantized — installed", flush=True)

    try:
        import importlib
        _gen_mod = importlib.import_module("mlx_lm.generate")
    except ImportError:
        return

    if not getattr(_gen_mod, "_hu_spec_patched", False):
        _orig_spec = getattr(_gen_mod, "speculative_generate_step", None)
        if _orig_spec is None:
            print(f"  [patch] mlx_lm.generate has no speculative_generate_step "
                  f"(symbols: {[n for n in dir(_gen_mod) if 'spec' in n.lower()]})",
                  flush=True)
            return

        import functools

        class _UnwrapWrapper:
            """Forwards everything to the wrapped module but unwraps the call result."""
            def __init__(self, inner):
                object.__setattr__(self, "_inner", inner)

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def __setattr__(self, name, value):
                setattr(self._inner, name, value)

            def __call__(self, *args, **kwargs):
                out = self._inner(*args, **kwargs)
                if hasattr(out, "logits") and not isinstance(out, mx.array):
                    return out.logits
                return out

        @functools.wraps(_orig_spec)
        def _patched_spec(prompt, model, draft_model, **kw):
            wm = _UnwrapWrapper(model) if not isinstance(model, _UnwrapWrapper) else model
            wd = _UnwrapWrapper(draft_model) if not isinstance(draft_model, _UnwrapWrapper) else draft_model
            yield from _orig_spec(prompt, wm, wd, **kw)

        _gen_mod.speculative_generate_step = _patched_spec
        _gen_mod._hu_spec_patched = True
        print("  [patch] speculative_generate_step — installed (LanguageModelOutput unwrap)", flush=True)


_install_mlx_lm_patches()


def _load_human_config():
    """Read ~/.human/config.json for h-uman integration defaults."""
    if not os.path.isfile(HUMAN_CONFIG):
        return {}
    try:
        with open(HUMAN_CONFIG) as f:
            cfg = json.load(f)
        mlx = cfg.get("mlx_local", {})
        defaults = {}
        if mlx.get("model"):
            defaults["model"] = mlx["model"]
        if mlx.get("adapter_path"):
            defaults["adapter_path"] = os.path.expanduser(mlx["adapter_path"])
        if mlx.get("port"):
            defaults["port"] = int(mlx["port"])
        if mlx.get("realtime"):
            defaults["realtime"] = True
        if mlx.get("kv_bits"):
            defaults["kv_bits"] = float(mlx["kv_bits"])
        if mlx.get("kv_asymmetric"):
            defaults["kv_asymmetric"] = True
        if mlx.get("speculative_draft"):
            defaults["speculative_draft"] = mlx["speculative_draft"]
        if mlx.get("speculative_draft_adapter"):
            defaults["speculative_draft_adapter"] = os.path.expanduser(
                mlx["speculative_draft_adapter"]
            )
        if mlx.get("speculative_tokens") is not None:
            try:
                defaults["speculative_tokens"] = max(1, int(mlx["speculative_tokens"]))
            except (TypeError, ValueError):
                pass
        if "prompt_cache" in mlx:
            defaults["prompt_cache"] = bool(mlx["prompt_cache"])
        if "prompt_cache_lcp" in mlx:
            defaults["prompt_cache_lcp"] = str(mlx["prompt_cache_lcp"])
        if mlx.get("prompt_cache_slots") is not None:
            defaults["prompt_cache_slots"] = mlx["prompt_cache_slots"]
        if "iosurface_kv" in mlx:
            defaults["iosurface_kv"] = bool(mlx["iosurface_kv"])
        if mlx.get("iosurface_kv_bytes") is not None:
            try:
                defaults["iosurface_kv_bytes"] = int(mlx["iosurface_kv_bytes"])
            except (TypeError, ValueError):
                pass
        return defaults
    except Exception:
        return {}


def _apply_mlx_local_env_from_hc(hc: dict) -> None:
    """Set GEMMA_IOSURFACE_* from ~/.human mlx_local before IOSurface init."""
    if not hc:
        return
    if hc.get("iosurface_kv") is False:
        os.environ["GEMMA_IOSURFACE_KV"] = "0"
    elif hc.get("iosurface_kv") is True:
        os.environ.setdefault("GEMMA_IOSURFACE_KV", "1")
    if hc.get("iosurface_kv_bytes") is not None:
        try:
            os.environ["GEMMA_IOSURFACE_KV_BYTES"] = str(int(hc["iosurface_kv_bytes"]))
        except (TypeError, ValueError):
            pass


def _speculative_tokens_default(hc: dict) -> int:
    for key in ("MLX_SPECULATIVE_TOKENS", "GEMMA_SPECULATIVE_TOKENS"):
        raw = os.environ.get(key, "").strip()
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                break
    if hc.get("speculative_tokens") is not None:
        try:
            return max(1, int(hc["speculative_tokens"]))
        except (TypeError, ValueError):
            pass
    return 4

# ── Hardware Detection ─────────────────────────────────────────────

def detect_apple_silicon():
    """Detect Apple Silicon capabilities for inference optimization."""
    info = {
        "chip": "unknown",
        "gpu_cores": 0,
        "neural_engine": False,
        "metal_version": "unknown",
        "unified_memory_gb": 0,
        "has_tensor_ops": False,
        "has_neural_accelerators": False,
    }
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info["chip"] = result.stdout.strip()

        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            displays = data.get("SPDisplaysDataType", [])
            for d in displays:
                metal_support = d.get("spmetal_supported", d.get("metal_support", ""))
                if metal_support:
                    info["metal_version"] = metal_support

        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info["unified_memory_gb"] = int(result.stdout.strip()) // (1024 ** 3)

        chip = info["chip"].lower()
        if "m5" in chip:
            info["has_tensor_ops"] = True
            info["has_neural_accelerators"] = True
            info["gpu_cores"] = 16 if "max" not in chip and "pro" not in chip else 40
        elif "m4" in chip:
            info["has_tensor_ops"] = False
            info["has_neural_accelerators"] = False
            info["gpu_cores"] = 10 if "max" not in chip and "pro" not in chip else 40
        elif "m3" in chip:
            info["gpu_cores"] = 10 if "max" not in chip and "pro" not in chip else 40
        elif "m2" in chip:
            info["gpu_cores"] = 10 if "max" not in chip and "pro" not in chip else 38
        elif "m1" in chip:
            info["gpu_cores"] = 8 if "max" not in chip and "pro" not in chip else 32

        info["neural_engine"] = any(x in chip for x in ("m1", "m2", "m3", "m4", "m5"))
    except Exception:
        pass
    return info


model = None
processor = None
config = None
model_lock = Lock()
model_id = None
use_lm_path = False  # True = mlx_lm (fast text), False = mlx_vlm (multimodal)

draft_model = None
draft_processor = None
speculative_enabled = False
speculative_draft_tokens = 4

kv_bits = None
kv_quant_scheme = "uniform"
turbo_cache = None
iosurface_kv_manager = None

prompt_cache_state = None
_cached_system_hash = None
_cached_system_ntokens = 0

lm_prompt_cache = None
lm_cache_quantized_start = 0
lm_quantized_kv_group_size = 64

# Token-level LCP prompt-cache reuse (see prompt_cache_lcp.py for why the
# whole-system-hash scheme above hits 0% on h-uman's per-turn prompts).
# Modes: off (legacy only) / shadow (legacy + measurement log) / live
# (trim to longest common token prefix, prefill only the suffix).
#
# Multi-slot pool (2026-07-25): :8741 serves interleaved callers — the
# daemon's ~4K-token persona turns mixed with 162-547-token probes — so a
# single slot only ever matches the chat-template preamble (6-9 tokens
# measured in shadow). N small slots let the big-head slot survive the
# probe traffic; lm_prompt_cache always points at the ACTIVE slot's cache
# list so _kv_kwargs/health/legacy paths need no knowledge of the pool.
lcp_mode = "off"
lm_cache_slot_count = 2   # config mlx_local.prompt_cache_slots, clamped 1..4
lm_cache_slots = None     # [{"cache": list, "prev_tokens": list|None, "last_used": int}]
_lcp_shadow_slots = None  # shadow-mode simulation of the pool (token lists only)
_lcp_pending_tokens = None  # this request's tokens; promoted at prefill-done
_lcp_pending_slot = None    # slot index the pending tokens belong to
_lcp_request_counter = 0    # monotonic clock for slot LRU
_lcp_last = {}            # last request's stats, surfaced via /health

STOP_STRINGS = ("<end_of_turn>", "<eos>")
THINKING_TOKENS = frozenset({"|", "|\n", "\n|", "\n"})
_THOUGHT_CHANNEL_RE = None

adapter_path_global = None
tensors_loaded_global = 0  # Number of LoRA tensors applied by the most recent successful load/swap
hw_info = {}
perf_stats = {"total_tokens": 0, "total_time": 0.0, "requests": 0}
# Set in main() for /health (inference_tuning)
server_realtime_mode = False


def _iosurface_http_health():
    try:
        import native_hw as nh
        return nh.health_payload_for_http(iosurface_kv_manager)
    except Exception as e:
        return {"iosurface_probe_error": str(e)}


def _init_iosurface_kv_staging():
    global iosurface_kv_manager
    if platform.system() != "Darwin":
        return
    if os.environ.get("GEMMA_IOSURFACE_KV", "1") == "0":
        return
    nbytes = int(os.environ.get("GEMMA_IOSURFACE_KV_BYTES", "0"))
    if nbytes <= 0:
        return
    try:
        from hw_accel import IOSurfaceKVManager

        iosurface_kv_manager = IOSurfaceKVManager()
        iosurface_kv_manager.allocate_kv_surface("mlx_staging", nbytes)
        print(
            f"  IOSurface KV staging: {nbytes} bytes (GEMMA_IOSURFACE_KV_BYTES)",
            flush=True,
        )
    except Exception as e:
        print(f"  IOSurface KV staging skipped: {e}", flush=True)


# PLE-safe model IDs — these quantize Gemma 4 correctly (skip ScaledLinear/PLE layers)
PLE_SAFE_MODELS = {
    "FakeRockert543/gemma-4-e4b-it-MLX-4bit",
    "FakeRockert543/gemma-4-e4b-it-MLX-8bit",
    "FakeRockert543/gemma-4-e4b-it-MLX-bf16",
    "FakeRockert543/gemma-4-e2b-it-MLX-4bit",
    "FakeRockert543/gemma-4-e2b-it-MLX-8bit",
}

BROKEN_MODELS = {
    "mlx-community/gemma-4-e4b-it-4bit",
    "mlx-community/gemma-4-e2b-it-4bit",
    "unsloth/gemma-4-e4b-it-4bit",
}


def _check_ple_safety(model_name):
    """Warn if using a broken Gemma 4 quantization that corrupts PLE layers."""
    if model_name in BROKEN_MODELS:
        print(f"\n  WARNING: {model_name} has BROKEN PLE quantization!", flush=True)
        print(f"  Gemma 4's ScaledLinear layers are incorrectly quantized in this model.", flush=True)
        print(f"  This causes degraded output quality. Use a PLE-safe model instead:", flush=True)
        print(f"    --model FakeRockert543/gemma-4-e4b-it-MLX-4bit", flush=True)
        print(f"  See: https://github.com/FakeRocket543/mlx-gemma4\n", flush=True)
    elif model_name in PLE_SAFE_MODELS:
        print(f"  PLE-safe model confirmed: {model_name}", flush=True)


def _load_with_adapter(load_fn, model_name, adapter_path):
    """Load a model and apply LoRA adapter weights.

    Fail-loud invariant: when ``adapter_path`` is configured but the adapter cannot
    be applied (no adapters.safetensors), the server must NOT silently fall back to
    base weights. That silent fallback masked an inactive persona fine-tune for
    weeks — /health reported a configured ``adapter`` path while the model was
    actually serving base. We warn prominently and record
    ``tensors_loaded_global = 0`` so the state is observable via /health
    (``adapter_applied``).
    """
    global tensors_loaded_global
    import mlx.core as mx
    from pathlib import Path

    model, tokenizer = load_fn(model_name)
    adapter_file = Path(adapter_path) / "adapters.safetensors"
    if adapter_file.exists():
        adapters = list(mx.load(str(adapter_file)).items())
        model.load_weights(adapters, strict=False)
        tensors_loaded_global = len(adapters)
        print(f"  Applied {len(adapters)} LoRA weight tensors from {adapter_file}", flush=True)
    else:
        tensors_loaded_global = 0
        print(
            f"  WARNING: adapter configured ({adapter_path}) but {adapter_file.name} "
            f"NOT FOUND at {adapter_file} — serving BASE weights. The persona fine-tune "
            f"is NOT active. Check ~/.human/config.json mlx_local.adapter_path.",
            flush=True,
        )
    return model, tokenizer


def _apply_adapter_weights(adapter_path):
    """Apply LoRA weights from <adapter_path>/adapters.safetensors to the live model.

    Caller MUST hold model_lock. Returns the number of tensors applied.
    Raises FileNotFoundError if adapters.safetensors is missing.
    Raises any model.load_weights error (caller decides how to recover).
    """
    global tensors_loaded_global
    import mlx.core as mx
    from pathlib import Path

    if model is None:
        raise RuntimeError("model not loaded; cannot apply adapter")
    adapter_file = Path(adapter_path) / "adapters.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(str(adapter_file))
    adapters = list(mx.load(str(adapter_file)).items())
    model.load_weights(adapters, strict=False)
    tensors_loaded_global = len(adapters)
    return len(adapters)


def load_model(model_name, adapter_path=None):
    global model, processor, config, model_id, adapter_path_global, use_lm_path

    _check_ple_safety(model_name)
    adapter_path_global = adapter_path
    label = model_name
    if adapter_path:
        label += f" + LoRA adapter ({adapter_path})"

    # Try mlx_lm first (fast text path, no vision overhead, ~40% faster)
    try:
        from mlx_lm import load as lm_load
        print(f"Loading {label} via mlx_lm (fast text path)...", flush=True)
        t0 = time.time()

        if adapter_path:
            model, processor = _load_with_adapter(lm_load, model_name, adapter_path)
        else:
            model, processor = lm_load(model_name)

        config = None
        use_lm_path = True
        model_id = model_name.split("/")[-1] if "/" in model_name else model_name
        elapsed = time.time() - t0
        adapter_tag = " (with LoRA adapter)" if adapter_path else ""
        print(f"Model loaded in {elapsed:.1f}s{adapter_tag} via mlx_lm — ready to serve", flush=True)
        return
    except Exception as e:
        print(f"  mlx_lm load failed ({e}), falling back to mlx_vlm...", flush=True)

    # Fallback: mlx_vlm (supports vision/audio but has numpy sync overhead)
    from mlx_vlm import load as vlm_load
    from mlx_vlm.utils import load_config as vlm_load_config

    print(f"Loading {label} via mlx_vlm (multimodal path)...", flush=True)
    t0 = time.time()

    if adapter_path:
        model, processor = _load_with_adapter(vlm_load, model_name, adapter_path)
    else:
        model, processor = vlm_load(model_name)

    config = vlm_load_config(model_name)
    use_lm_path = False
    model_id = model_name.split("/")[-1] if "/" in model_name else model_name
    elapsed = time.time() - t0
    adapter_tag = " (with LoRA adapter)" if adapter_path else ""
    print(f"Model loaded in {elapsed:.1f}s{adapter_tag} via mlx_vlm — ready to serve", flush=True)


def load_draft_model(draft_model_name, draft_adapter_path=None):
    """Load a smaller draft model for speculative decoding."""
    global draft_model, draft_processor, speculative_enabled

    _check_ple_safety(draft_model_name)
    label = draft_model_name
    if draft_adapter_path:
        label += f" + LoRA adapter ({draft_adapter_path})"
    print(f"Loading draft model for speculative decoding: {label}...", flush=True)
    t0 = time.time()

    try:
        from mlx_lm import load as lm_load
        if draft_adapter_path:
            draft_model, draft_processor = _load_with_adapter(lm_load, draft_model_name, draft_adapter_path)
        else:
            draft_model, draft_processor = lm_load(draft_model_name)
    except Exception:
        from mlx_vlm import load as vlm_load
        if draft_adapter_path:
            draft_model, draft_processor = _load_with_adapter(vlm_load, draft_model_name, draft_adapter_path)
        else:
            draft_model, draft_processor = vlm_load(draft_model_name)

    speculative_enabled = True
    elapsed = time.time() - t0
    print(f"Draft model loaded in {elapsed:.1f}s — speculative decoding enabled", flush=True)


def _extract_content(content):
    """Extract text and image data from a message content field.
    Content can be a string or an array of parts (OpenAI vision format)."""
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return str(content) if content else "", []

    text_parts = []
    images = []
    for part in content:
        ptype = part.get("type", "")
        if ptype == "text":
            text_parts.append(part.get("text", ""))
        elif ptype == "image_url":
            url = part.get("image_url", {}).get("url", "")
            if url.startswith("data:"):
                import base64
                from io import BytesIO
                try:
                    header, b64data = url.split(",", 1)
                    raw = base64.b64decode(b64data)
                    from PIL import Image
                    img = Image.open(BytesIO(raw)).convert("RGB")
                    images.append(img)
                except Exception as e:
                    text_parts.append(f"[Image decode failed: {e}]")
            elif url.startswith("http"):
                text_parts.append(f"[Image URL: {url}]")
    return " ".join(text_parts), images


def _has_images(messages):
    """Quick check whether any message contains image data."""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    return True
    return False


_NO_THINK_INSTRUCTION = (
    # Wording chosen 2026-05-27 from the chip-discussion options (detailed,
    # explicit). Names the four failure modes observed empirically in the
    # markdown-thinking salvage logs: candidate replies, internal deliberation,
    # bullet lists, evaluation parentheticals. Tune if the LoRA's voice
    # fidelity drifts; pin any change with the TestNoThinkInjection suite +
    # a live probe via the verification recipe in
    # ~/.claude/projects/.../memory/m3_live_path_extractor_strip.md.
    "Output only the final response. "
    "Do not include candidate replies, internal deliberation, markdown bullet lists, "
    "evaluation parentheticals, or thought process. "
    "Reply directly in one short message."
)


# Structured-output variant of the no-think instruction.
#
# Why this exists (runaway diagnosis 2026-05-28, m3_live_path_extractor_strip.md
# "runaway" arc): the casual _NO_THINK_INSTRUCTION above contains TWO clauses
# that CONTRADICT a structured-output caller prompt (e.g. the feed-research
# agent that asks for a "Source / Finding / Relevance / Priority /
# Suggested-Action" report, or any caller requesting JSON):
#   - "markdown bullet lists" forbidden  → conflicts with a report/list format
#   - "Reply directly in one short message" → conflicts with a multi-field report
# When both the structured request and the casual no-think clause are present,
# the model spends its entire budget deliberating about which instruction to
# obey (ground truth: mlx-server-empty-extract.log 15:09:20, the model writes
# "the system prompt requested a specific format ... However, the *last*
# instruction says ... Reply directly in one short message") and never closes
# its thought channel → runaway → empty/garbage salvage.
#
# The fix keeps the UNIVERSAL "no deliberation / no thought process" core
# (which protects every prompt, including working JSON callers like
# init_proposer) but DROPS the two clauses that fight a structured format.
_NO_THINK_INSTRUCTION_STRUCTURED = (
    "Do not include candidate replies, internal deliberation, "
    "evaluation parentheticals, or thought process. "
    "Produce the requested output directly in the format the prompt specifies."
)


# Markers that indicate a system/user prompt is asking for STRUCTURED output
# (JSON, a multi-field report, a named format) rather than a casual chat reply.
# Matched case-insensitively as substrings. Kept deliberately specific to avoid
# false-positives on casual prompts that merely mention the word "format".
_STRUCTURED_OUTPUT_MARKERS = (
    "output format",
    "respond in json",
    "reply in json",
    "return json",
    "as json",
    "json object",
    "single json",
    "valid json",
    "json only",
    "in the following format",
    "use the following format",
    "respond in the format",
    "following structure",
    "schema:",
    "fields:",
    "format:",
    "suggested action",  # feed-research report field
    "suggested-action",
)


def _system_prompt_requests_structured_output(messages):
    """Return True if any system OR user message asks for structured output.

    Pure function (no I/O, no globals) — directly unit-testable. Drives the
    choice between the casual and structured no-think instruction variants so
    a structured caller prompt isn't sabotaged by the casual "one short
    message / no bullet lists" clauses (the runaway root cause).
    """
    if not messages:
        return False
    for msg in messages:
        role = msg.get("role")
        if role not in ("system", "user"):
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            # Multimodal content parts — concatenate any text parts.
            text = " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        elif isinstance(content, str):
            text = content
        else:
            text = ""
        lowered = text.lower()
        if any(marker in lowered for marker in _STRUCTURED_OUTPUT_MARKERS):
            return True
    return False


def _no_think_requested():
    """True when the operator asked for thinking suppression, ANY base family.

    Deliberately distinct from `_no_think_instruction()`, which returns the
    Gemma-WORDED instruction TEXT and is therefore correctly gated to Gemma
    bases. The suppression MECHANISM differs by family:
      - Gemma 4:  system-instruction text  +  the enable_thinking template flag
      - GLM-4.5:  the enable_thinking template flag ALONE (no text exists for it)

    Deriving "should we suppress?" from "is there Gemma text to inject?"
    conflates the two. The 2026-07-26 family gate (c738d10) correctly stopped
    injecting Gemma wording into GLM prompts by returning None — but the same
    predicate also fed `skip_thinking_primer`, so GLM silently stopped
    receiving `enable_thinking=False` and resumed deliberating. Measured on
    production 2026-07-27: 106 and 103 completion tokens for one-line replies
    (~8 visible tokens). The deliberation never leaked — strip_thought_channels
    caught it, see TestGlmThinkBlockStripping — it was pure latency.

    Use THIS for "do we suppress thinking?"; use `_no_think_instruction()`
    only to decide whether to inject the Gemma text.
    """
    return os.environ.get("GEMMA_DISABLE_THINKING", "").strip().lower() in ("1", "true", "yes")


def _no_think_instruction():
    """Return the system-instruction text that suppresses Gemma 4 thinking mode.

    Gemma 4's default behavior is to emit chain-of-thought as markdown bullets
    before producing a final reply. For h-uman's LoRA-fine-tuned model, the
    final reply is often missing — the model burns its budget on internal
    deliberation and the strip extractor returns empty. This instruction tells
    the model to skip deliberation and reply directly.

    The exact wording is a design decision because it affects model quality:
      - Too aggressive ("DO NOT THINK") may confuse the model or harm voice
        fidelity learned by the LoRA.
      - Too soft ("try to skip thinking") may not work — Gemma 4's training
        defaults to thinking.
      - Just right: an unambiguous instruction that suppresses chain-of-thought
        without contradicting the system prompt that established persona.

    Wording lives in `_NO_THINK_INSTRUCTION` above. Returns None if env var
    `GEMMA_DISABLE_THINKING` is unset (no-op, preserves current behavior).
    """
    if not _no_think_requested():
        return None
    # Model-family gate (2026-07-26). The wording below is Gemma-specific -- it
    # names Gemma 4's markdown-bullet deliberation and was tuned against the
    # seth-lora-v4-repair adapter. The env var, however, is set unconditionally
    # in ~/Library/LaunchAgents/ai.human.mlx-server.plist, so after production
    # flipped to GLM-4.5-Air-4bit on 2026-07-26 this instruction was being
    # prepended to every GLM turn: a Gemma-worded directive aimed at a model
    # with an entirely different thinking mechanism (GLM uses the
    # enable_thinking chat-template flag, handled separately below), injected
    # into a system prompt already 78% over the 16 KB cap and losing its tail.
    #
    # Applies to gemma bases only. HU_NO_THINK_ANY_MODEL=1 restores the old
    # unconditional behavior if a future non-gemma base turns out to want it.
    if os.environ.get("HU_NO_THINK_ANY_MODEL", "").strip().lower() in ("1", "true", "yes"):
        return _NO_THINK_INSTRUCTION
    if not _is_gemma_base(model_id):
        return None
    return _NO_THINK_INSTRUCTION


def _is_gemma_base(name):
    """True when `name` identifies a Gemma base.

    Word-ish match rather than a bare substring: a repo id such as
    "org/not-gemma-clone" should not be treated as Gemma just because the
    letters appear (~/.claude/rules/substring-classifier-pitfalls.md). Checking
    for the 'gemma' token bounded by a non-alphanumeric or string edge is
    enough for the mlx-community naming scheme.
    """
    if not name:
        return False
    low = str(name).casefold()
    idx = 0
    while True:
        i = low.find("gemma", idx)
        if i < 0:
            return False
        before_ok = i == 0 or not low[i - 1].isalnum()
        after = i + 5
        after_ok = after == len(low) or not low[after].isalnum()
        if before_ok and after_ok:
            return True
        idx = i + 1


def _thinking_headroom_tokens():
    """Extra generation budget added on top of the caller's max_tokens to make
    room for Gemma 4 / seth-lora-v4-repair chain-of-thought deliberation that
    precedes the visible reply.

    Why this exists (live M3 production bug, m3_live_path_extractor_strip.md):
    the adapter emits ~150-200 tokens of markdown-bullet deliberation BEFORE
    the visible reply, even when the GEMMA_DISABLE_THINKING no-think
    instruction is active (the adapter was trained to deliberate and largely
    ignores the instruction). If generation is capped at the caller's
    max_tokens (e.g. 80), the thought channel consumes the entire budget and
    the visible reply is starved to empty — the strip extractor then salvages
    a garbage fragment. Empirically, max_tokens=80 degenerated every casual
    prompt to a single word; with ~512 headroom the same prompts return
    coherent in-voice replies (finish_reason=stop, thinking stripped).

    Headroom is a CAP, not a target: a completed reply emits its stop token
    and finishes naturally well below the cap, so a generous headroom is
    nearly free for well-behaved completions and only costs latency on
    runaway (never-stopping) generations.

    Tunable via GEMMA_THINKING_HEADROOM_TOKENS. Default 512. Non-integer or
    negative values fall back to 512.
    """
    raw = os.environ.get("GEMMA_THINKING_HEADROOM_TOKENS", "").strip()
    if not raw:
        return 512
    try:
        val = int(raw)
    except ValueError:
        return 512
    return val if val >= 0 else 512


def _repetition_penalty():
    """Repetition penalty for the logits processor — suppresses the ultra-short-
    prompt runaway (m3_live_path_extractor_strip.md). With NO penalty, minimal
    prompts like "yo" occasionally collapse into a token-repetition loop that
    burns the entire thinking-headroom budget (observed 592 generated tokens)
    and yields an empty reply after strip (the _is_pure_deliberation runaway
    guard correctly returns "" rather than garbage). A mild penalty (default
    1.1) breaks the degenerate loop so the model commits to a real reply.

    Tunable via GEMMA_REPETITION_PENALTY. Default 1.1. 1.0 disables. Non-numeric
    or non-positive values fall back to 1.1.
    """
    raw = os.environ.get("GEMMA_REPETITION_PENALTY", "").strip()
    if not raw:
        return 1.1
    try:
        val = float(raw)
    except ValueError:
        return 1.1
    return val if val > 0 else 1.1


def _repetition_logits_processors():
    """Build the logits_processors list applying the repetition penalty, or
    None when disabled (penalty == 1.0). In this mlx_lm version repetition
    penalty is applied via make_logits_processors, NOT make_sampler (the
    sampler signature has no repetition_penalty arg). Context window tunable
    via GEMMA_REPETITION_CONTEXT (default 20)."""
    penalty = _repetition_penalty()
    if abs(penalty - 1.0) < 1e-9:
        return None
    from mlx_lm.sample_utils import make_logits_processors
    raw = os.environ.get("GEMMA_REPETITION_CONTEXT", "").strip()
    try:
        ctx = int(raw) if raw else 20
    except ValueError:
        ctx = 20
    return make_logits_processors(repetition_penalty=penalty,
                                  repetition_context_size=ctx)


def _maybe_inject_no_think_instruction(messages):
    """If thinking is disabled, append the no-think instruction to the system
    message (or insert a new system message if none exists).

    Picks the STRUCTURED variant of the instruction when the caller prompt
    requests structured output (JSON / a named report format) — otherwise the
    casual variant. This prevents the runaway where the casual "one short
    message / no bullet lists" clauses contradict a structured request and the
    model burns its whole budget deliberating about which to obey.

    Returns a new messages list — does NOT mutate the input. This preserves
    OpenAI request idempotency and makes the function safe to call from both
    streaming and non-streaming paths.
    """
    instruction = _no_think_instruction()
    if not instruction:
        return messages
    if instruction is _NO_THINK_INSTRUCTION and \
            _system_prompt_requests_structured_output(messages):
        instruction = _NO_THINK_INSTRUCTION_STRUCTURED
    out = []
    appended = False
    for msg in messages:
        if not appended and msg.get("role") == "system":
            existing = msg.get("content", "")
            if isinstance(existing, str) and existing.strip():
                merged = f"{existing.rstrip()}\n\n{instruction}"
            else:
                merged = instruction
            out.append({**msg, "content": merged})
            appended = True
        else:
            out.append(msg)
    if not appended:
        # No system message in the conversation — insert one at the head so
        # Gemma's chat template renders it before the first user turn.
        out.insert(0, {"role": "system", "content": instruction})
    return out


def _thinking_suppressed_value():
    """The `enable_thinking` value that SUPPRESSES an auto-primed thought block.

    The polarity is inverted on Gemma 4 relative to every other family:
      - Gemma 4: `enable_thinking=False` force-prepends the thought channel, so
        `True` is what suppresses it (see prepare_prompt_lm's docstring).
      - GLM-4.5, Qwen3, and the rest: the flag reads literally — `False`
        suppresses, `True` asks the model to reason out loud.

    Sending Gemma's polarity to GLM-4.5-Air made every reply open with a
    `<think>` block and burn its token budget deliberating (measured on the
    :8745 rehearsal, 2026-07-25 — 106 tokens at 1.3 tok/s, reply never reached).
    """
    return "gemma" in (model_id or "").lower()


def prepare_prompt_lm(messages, skip_thinking_primer=None):
    """Format messages using mlx_lm's native chat template (fast text path).

    Gemma 4's chat template has confusing semantics around `enable_thinking`:
      - `enable_thinking=False` (DEFAULT): template force-prepends
        `<|channel>thought\\n<channel|>` after the model turn, priming the
        model into thinking mode REGARDLESS of any system instruction.
      - `enable_thinking=True`: template skips the auto-prime, letting the
        model decide based on the prompt.

    When `GEMMA_DISABLE_THINKING` is set we ALSO pass `enable_thinking=True`
    so the template doesn't undo our system-level instruction by mechanically
    opening a thinking block. The system instruction alone is insufficient
    (the template runs after the system message and overrides it).

    `skip_thinking_primer` lets a CALLER force the primer-skip per request,
    independent of the global no-think config. This is the reply-first lever
    for streamed casual tiers (Task-0 spike 2026-05-29): the seth-lora-v4-repair
    model emits a short, in-voice, reply-FIRST response when NOT primed, but the
    template's forced `<|channel>thought` opener makes it front-load ~150 tokens
    of deliberation and emit the reply LAST (streaming_beneficial:false). Passing
    skip_thinking_primer=True drops the opener so the reply streams first.
      - None  (default): preserve legacy behavior — skip iff a no-think
                          instruction is configured.
      - True / False:    explicit override from the caller.
    """
    if skip_thinking_primer is None:
        # `_no_think_requested()`, NOT `_no_think_instruction() is not None`:
        # the template flag is the suppression mechanism for EVERY family,
        # while the instruction text is Gemma-only. See _no_think_requested().
        skip_thinking_primer = _no_think_requested()
    messages = _maybe_inject_no_think_instruction(messages)
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if skip_thinking_primer:
        # Template-level suppression of the thought-channel opener.
        # See module docstring + `_maybe_inject_no_think_instruction()`.
        template_kwargs["enable_thinking"] = _thinking_suppressed_value()
    if hasattr(processor, "apply_chat_template"):
        prompt = processor.apply_chat_template(messages, **template_kwargs)
    else:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            text, _ = _extract_content(msg.get("content", ""))
            parts.append(f"<start_of_turn>{role}\n{text}<end_of_turn>")
        parts.append("<start_of_turn>model\n")
        prompt = "\n".join(parts)
    return prompt


def prepare_prompt_vlm(messages):
    """Format messages using mlx_vlm's template (multimodal path)."""
    from mlx_vlm.prompt_utils import apply_chat_template

    messages = _maybe_inject_no_think_instruction(messages)
    system_parts = []
    conversation = []
    all_images = []
    for msg in messages:
        role = msg.get("role", "user")
        text, images = _extract_content(msg.get("content", ""))
        all_images.extend(images)
        if role == "system":
            system_parts.append(text)
        else:
            conversation.append({"role": role, "content": text})

    if system_parts:
        system_text = "\n".join(system_parts)
        if conversation:
            first_content = conversation[0].get("content", "")
            prompt_text = f"System: {system_text}\n\n{first_content}"
        else:
            prompt_text = f"System: {system_text}"
    elif conversation:
        prompt_text = conversation[-1].get("content", "")
    else:
        prompt_text = ""

    return apply_chat_template(processor, config, prompt_text, num_images=len(all_images)), all_images


def strip_stop_tokens(text):
    for stop in STOP_STRINGS:
        idx = text.find(stop)
        if idx != -1:
            return text[:idx], True
    return text, False


# Deliberation residue that must never be emitted as a reply. These are the
# openers of a model TALKING ABOUT the reply rather than making it.
_DELIB_OPENER_RE = re.compile(
    r"^\s*(?:i\s+(?:should|will|could|need|think\s+i)|reply\s+should|response\s+[ab]\b"
    r"|the\s+user|user\s+(?:asked|says|wants)|persona\b|candidate\b|option\s+[ab]\b"
    r"|draft\b|let\s+me\s+(?:craft|write|think))",
    re.I)

# A tail that begins by CLOSING someone else's sentence — "concern. What ..." —
# i.e. a lowercase word immediately terminated by .!? then more text. A genuine
# casual reply can start lowercase ("yeah should be"), so lowercase alone is not
# the signal; the signal is the sentence boundary sitting one word in.
_FRAGMENT_HEAD_RE = re.compile(r"^[a-z][a-z'’-]*[.!?](?:\s|$)")


def _looks_like_deliberation_residue(text):
    """True when a salvage heuristic produced something that is not a reply.

    Returning empty is SAFE: the daemon retries degenerate output via a slim
    request and then falls back to cloud (doctor: response_pipeline). Emitting
    is not — on 2026-07-26 10:57 this exact shape reached Seth's real-estate
    agent mid-negotiation as two messages, "concern." then "What specific aspect
    of their decision would you like to discuss?" """
    if not text:
        return False
    return bool(_DELIB_OPENER_RE.match(text) or _FRAGMENT_HEAD_RE.match(text))


def _extract_reply_from_body(text):
    """From a fragment that may have parentheticals, label prefixes, or
    surrounding quotes, extract the cleanest version of the actual reply.

    Returns "" when the salvage yields deliberation residue rather than a reply.
    """
    text = text.strip()
    if not text:
        return text
    # Parenthetical evaluation followed by the reply — take what's after `)`.
    # gemma-4 thinking emits exactly: `"Yeah!" (Classic, fits constraint).Yeah!`
    #
    # REQUIRE that documented shape (a QUOTED candidate before the paren) before
    # splitting on `)`. Unconditionally taking rsplit(")")[1] treats any
    # parenthetical in ordinary deliberation prose as the reply boundary, which
    # is how "...the litigation (which she raised twice) concern. What specific
    # aspect..." became the 74-byte fragment sent to a real contact on
    # 2026-07-26. The quote requirement keeps the rule to the case it was
    # written for.
    if ")" in text and re.search(r'["“][^"”]+["”]\s*\(', text):
        tail = text.rsplit(")", 1)[1].strip()
        tail = re.sub(r"^[.,;:!?\s]+", "", tail).strip()
        if tail:
            text = tail
    # "Label: quoted" pattern — extract the quoted value
    label_match = re.match(r'^[A-Za-z][^:]{0,30}:\s*[\"“]([^\"”]+)[\"”]', text)
    if label_match:
        return label_match.group(1)
    # Strip surrounding straight or smart quotes
    for a, b in (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")):
        if text.startswith(a) and text.endswith(b) and len(text) >= 2:
            text = text[1:-1]
            break
    text = text.strip()
    # Final gate: never hand back deliberation residue or a sentence tail.
    # Empty routes to the daemon's retry + cloud fallback; emitting routes to a
    # real human's phone.
    if _looks_like_deliberation_residue(text):
        print(f"[thought-strip] refusing deliberation residue: {text[:60]!r}", flush=True)
        return ""
    return text


def strip_thought_channels(text):
    """Strip model thought blocks from output.

    Handles THREE families. GLM's `<think>` form was added 2026-07-26 when
    production flipped to GLM-4.5-Air: before that, the ONLY thing preventing
    GLM deliberation from reaching recipients was the `enable_thinking`
    polarity in _thinking_suppressed_value(), i.e. a single point of failure
    guarding a failure mode with precedent (2026-07-11: the daemon sent raw
    deliberation to real contacts). This function is the universal output
    funnel, so scrubbing here is defence in depth behind that flag, not a
    replacement for it.

    0. GLM `<think> ... </think>` (GLM-4.5-Air and siblings). Three shapes,
       stripped in this order:
         a. closed pair                  — `<think>A</think>B` -> `B`
         b. orphan CLOSER, no opener     — `A</think>B`        -> `B`
            (the chat template can PRIME the opener, so generation begins
            already inside the thought and only the closer is emitted)
         c. orphan OPENER, never closed  — `<think>A`          -> ``
            (truncated mid-thought, e.g. max_tokens exhausted)
       Shape (c) legitimately yields empty; that is the correct outcome and
       the established contract of this function (see _is_pure_deliberation
       and the empty-result salvage guard) — returning the deliberation text
       instead IS the leak.

    1. Documented channel-marker format:
       `<|channel>thought ... <channel|> actual_response`
       Both closed and unclosed (mid-thought) variants stripped.

    2. ACTUAL format emitted by gemma-4-31b-seth-v3-fused AND stock
       gemma-4-31b-it-4bit: the model does NOT emit channel markers.
       It emits markdown-bullet thinking followed by a final unmarked
       reply on its own line (or inside the last bullet with a
       parenthetical evaluation).  Example:

           *   User: "hey, you around?"
               *   Persona: Seth (brief, natural).
               *   "Possible reply A"
               *   "Possible reply B"

           "Final actual reply"

       OR all-in-one-bullet form:

           *   "Yeah, what's up?" (Classic, fits constraint).Yeah, what's up?

       The reply IS in the text; the prior extractor returned the full
       thinking transcript verbatim because no `<channel|>` markers were
       present.  This branch detects the markdown pattern and extracts.
    """
    import re
    # GLM `<think>` family — closed pair, then orphan closer (template-primed
    # opener), then orphan opener (truncated mid-thought). Order matters: a
    # complete pair must be consumed as a pair before the orphan rules run.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Gemma channel-marker family.
    text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|channel>thought.*$", "", text, flags=re.DOTALL)
    text = text.strip()
    if not text:
        return text

    lines = text.split("\n")
    bullet_re = re.compile(r"^\s*\*\s")
    bullet_indices = [i for i, l in enumerate(lines) if bullet_re.match(l)]
    if not bullet_indices:
        return text  # no markdown bullets — already clean

    last_bullet_idx = bullet_indices[-1]
    tail_lines = [l.strip() for l in lines[last_bullet_idx + 1:]
                  if l.strip() and not bullet_re.match(l)]
    if tail_lines:
        return _extract_reply_from_body(tail_lines[-1])
    last_bullet_body = re.sub(r"^\s*\*\s*", "", lines[last_bullet_idx])
    return _extract_reply_from_body(last_bullet_body)


def _is_pure_deliberation(raw):
    """Return True if `raw` is an unclosed/never-resolved thought block with no
    extractable final reply — i.e. a RUNAWAY generation, not a real answer.

    Pure function (no I/O) — directly unit-testable. Used as a guard before the
    salvage heuristic in generate_response: when the model never closed its
    thought channel and never wrote a reply line, salvaging "the last quoted
    string" produces garbage (a candidate reply the model was still evaluating,
    or a fragment of its own deliberation). It is strictly better to return
    empty (so the caller sees a clean failure / can retry) than to emit a
    confident-looking garbage fragment.

    Detects the runaway shape observed in mlx-server-empty-extract.log:
      - opens with a `<|channel>thought` marker that is NEVER closed by
        `<channel|>`, OR
      - is entirely markdown-bullet deliberation with no non-bullet reply line
    AND strip_thought_channels() already returned empty for it.
    """
    if not raw:
        return False
    import re
    text = raw.strip()

    # Case 1: explicit thought-channel marker opened but never closed.
    if "<|channel>thought" in text and "<channel|>" not in text:
        return True

    # Case 2: markdown-bullet deliberation with no final reply line.
    # (Only meaningful when there ARE bullets — a plain unbulleted answer is
    # never pure deliberation.)
    lines = text.split("\n")
    bullet_re = re.compile(r"^\s*\*\s")
    bullet_lines = [l for l in lines if bullet_re.match(l)]
    if not bullet_lines:
        return False
    non_bullet_reply = [
        l.strip() for l in lines
        if l.strip() and not bullet_re.match(l)
    ]
    # If every non-bullet line is itself empty/whitespace, the model produced
    # only a bullet-list of candidate replies and never committed to one.
    return len(non_bullet_reply) == 0


def _sys_span_cover_min():
    """Minimum fraction of a reply that must be verbatim system-prompt text
    before the echo-guard calls it a leak. Override with
    HU_ECHO_GUARD_COVER_MIN (0.0 = old span-only behavior, 1.0 = exact-dump
    only). Clamped to [0.0, 1.0]; a malformed value falls back to the default."""
    raw = os.environ.get("HU_ECHO_GUARD_COVER_MIN", "").strip()
    if not raw:
        return 0.5
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 0.5


_SYS_SPAN_COVER_MIN = _sys_span_cover_min()


def _is_input_echo(reply, messages, detail=None):
    """Return True if `reply` merely echoes the INPUT rather than answering it —
    a prompt echo, a prior-history echo, or a system-prompt leak.

    Pure function (no I/O) — directly unit-testable. Used as a guard in
    finalize_generation: when generation truncates mid/post-deliberation, the
    strip extractor (or the runaway salvage) can surface a fragment that is
    actually the user's own message, a previous turn, or a span of the system
    prompt rather than a committed reply. Emitting that is worse than empty
    (h-uman's fallback handles an empty turn). Observed live 2026-06-04 at tight
    caps: reply == "did you eat yet?" (prompt echo) and "Their recent messages
    are short; match t..." (system-prompt leak).

    Detection (all on a casefold + whitespace-normalized basis):
      - reply == last USER message, but ONLY when that message is >=3 words, so a
        genuine "ok"/"yeah" answer to "ok"/"yeah" is NOT a false-positive;
      - reply exactly equals any PRIOR (non-final) turn of >=2 words, or is
        contained in one when the reply itself is >=3 words (history echo, e.g.
        the model parroting its own earlier "starving lol");
      - reply shares a >=6-word contiguous span with any SYSTEM message
        (system-prompt leak).
    Returns False on empty/malformed input.
    """
    if not reply or not messages:
        return False

    def _norm(s):
        return " ".join(str(s or "").split()).casefold()

    def _content(m):
        c = m.get("content", "") if isinstance(m, dict) else ""
        if isinstance(c, list):  # multimodal parts -> join text fragments
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        return c

    r = _norm(reply)
    if not r:
        return False
    r_words = r.split()

    # Diagnostics only — records WHICH rule fired and how much of the reply it
    # covered. The True/False verdict is unchanged; `detail` defaults to None so
    # every existing caller behaves exactly as before. Added 2026-07-26: the
    # guard was blanking 9/12 research-agent replies and the 80-char log prefix
    # could not distinguish "guard too strict" from "model regurgitating input".
    def _note(rule, covered=None):
        if detail is None:
            return
        detail["rule"] = rule
        detail["reply_words"] = len(r_words)
        if covered is not None:
            detail["covered_words"] = covered
            detail["covered_frac"] = round(covered / max(1, len(r_words)), 3)

    sys_texts, user_texts, other_texts = [], [], []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "")
        text = _content(m)
        if role == "system":
            sys_texts.append(text)
        elif role == "user":
            user_texts.append(text)
        else:
            other_texts.append(text)

    # 1. Prompt echo: reply == last user message (only if that message is >=3 words).
    if user_texts:
        last_user = _norm(user_texts[-1])
        if len(last_user.split()) >= 3 and r == last_user:
            _note("prompt_echo", covered=len(r_words))
            return True

    # 2. History echo: reply repeats a prior turn (all user turns but the last,
    #    plus every assistant/other turn). Exact match needs >=2 words; the
    #    contained-in case needs the reply itself to be >=3 words.
    for t in user_texts[:-1] + other_texts:
        tn = _norm(t)
        tn_words = tn.split()
        if len(tn_words) >= 2 and r == tn:
            _note("history_echo_exact", covered=len(r_words))
            return True
        if len(r_words) >= 3 and len(tn_words) >= 3 and r in tn:
            _note("history_echo_contained", covered=len(r_words))
            return True

    # 3. System-prompt leak: reply shares a >=6-word contiguous span with a
    #    system message.
    # A >=SPAN-word overlap with the system prompt is necessary but NOT
    # sufficient to call a reply a leak. Measured live 2026-07-26 on GLM-4.5-Air
    # with a 29 KB system prompt: legitimate replies scored frac 0.165 and 0.244
    # (a 79-word research finding sharing 13 words; a 41-word outreach plan
    # sharing 10) while a verbatim system-prompt dump scores 1.0. Span-only
    # gating blanked 9 of 12 completions, including real Seth-voice replies
    # ("Idk yet, she hasn't mentioned it"). Requiring the overlap to cover a
    # MAJORITY of the reply separates "quotes its source" from "is the source".
    # Tunable without a code change; default is the measured-safe value.
    SPAN = 6
    if len(r_words) >= SPAN:
        for st in sys_texts:
            sw = _norm(st).split()
            if len(sw) < SPAN:
                continue
            sys_spans = {tuple(sw[i:i + SPAN]) for i in range(len(sw) - SPAN + 1)}
            hit_starts = [j for j in range(len(r_words) - SPAN + 1)
                          if tuple(r_words[j:j + SPAN]) in sys_spans]
            if hit_starts:
                # Coverage = how many of the reply's words sit inside a matching
                # span. A leak is near-100% verbatim; a legitimate reply that
                # merely QUOTES its source is a minority fraction.
                covered = set()
                for j in hit_starts:
                    covered.update(range(j, j + SPAN))
                frac = len(covered) / max(1, len(r_words))
                if frac >= _SYS_SPAN_COVER_MIN:
                    _note("system_span", covered=len(covered))
                    return True
                # Below threshold: a real reply that happens to share a phrase
                # with the (up to 29 KB) system prompt. Record for observability
                # but SEND it.
                _note("system_span_below_threshold", covered=len(covered))

    return False


def finalize_generation(full, messages=None):
    """Strip thought blocks from a fully-accumulated raw generation.

    SHARED by the non-streaming path (generate_response) and the buffered
    streaming path (_handle_stream_buffered) so BOTH apply identical
    thought-stripping and runaway handling — the markdown-bullet reply can
    only be extracted once the full output is known, so streaming must buffer
    to match.

    Returns (clean_text, is_runaway):
      - clean_text: visible reply with thought blocks removed (possibly
        salvaged from raw if the extractor came up empty but the output was
        NOT pure deliberation)
      - is_runaway: True when the raw output was pure deliberation (unclosed
        thought channel or all-bullets-no-reply); callers emit empty rather
        than the salvage fragment.

    Honors GEMMA_DUMP_EMPTY_EXTRACT for raw-output diagnostics. Side effects
    are limited to that optional diagnostic write + stdout logging, so the
    (text, runaway) decision is deterministic and unit-testable.
    """
    full = (full or "").strip()
    if not full:
        return "", False
    import re as _re

    def _guard(text):
        # Reject input-echo / system-prompt-leak as runaway: when `messages` is
        # supplied and the candidate merely echoes the prompt/history or leaks a
        # system-prompt span, return empty (better empty than garbage — same
        # philosophy as the _is_pure_deliberation guard below). messages=None
        # preserves legacy behavior + existing tests.
        _echo_detail = {}
        if text and messages is not None and _is_input_echo(text, messages, _echo_detail):
            try:
                print(f"  [echo-guard] reply echoes input / leaks system prompt; "
                      f"returning empty: rule={_echo_detail.get('rule', '?')} "
                      f"reply_words={_echo_detail.get('reply_words', '?')} "
                      f"covered={_echo_detail.get('covered_words', '?')} "
                      f"frac={_echo_detail.get('covered_frac', '?')} "
                      f"| {text[:240]!r}", flush=True)
            except Exception:
                pass
            return "", True
        if _echo_detail.get("rule") == "system_span_below_threshold":
            # A reply that shares a phrase with the system prompt but is mostly
            # original — SENT. Logged (counts only, no reply text) so the
            # threshold stays monitorable: a drift toward frac~0.5 here is the
            # signal to revisit HU_ECHO_GUARD_COVER_MIN.
            try:
                print(f"  [echo-guard] near-miss SENT: overlap "
                      f"{_echo_detail.get('covered_words')}/"
                      f"{_echo_detail.get('reply_words')} words "
                      f"frac={_echo_detail.get('covered_frac')} "
                      f"< {_SYS_SPAN_COVER_MIN}", flush=True)
            except Exception:
                pass
        return text, False

    stripped = strip_thought_channels(full)
    if stripped:
        return _guard(stripped)

    # Extractor came up empty — diagnose, then decide runaway vs salvage.
    # DIAGNOSTIC: dump full raw output when GEMMA_DUMP_EMPTY_EXTRACT is set.
    _dump_flag = os.environ.get("GEMMA_DUMP_EMPTY_EXTRACT", "").strip()
    if _dump_flag and _dump_flag.lower() not in ("0", "false", "no"):
        _dump_path = (_dump_flag if "/" in _dump_flag
                      else os.path.expanduser(
                          "~/.human/logs/mlx-server-empty-extract.log"))
        try:
            os.makedirs(os.path.dirname(_dump_path), exist_ok=True)
            with open(_dump_path, "a", encoding="utf-8") as _f:
                _f.write(f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%S')} "
                         f"len={len(full)} =====\n")
                _f.write(full)
                _f.write("\n===== end =====\n")
        except Exception:
            pass  # diagnostic must never break request handling

    # RUNAWAY GUARD: pure deliberation (unclosed thought channel or a
    # bullet-list of candidates with no resolved reply) -> return empty so the
    # caller sees a clean failure instead of a confident-looking garbage
    # fragment. See m3_live_path_extractor_strip.md "runaway" arc.
    if _is_pure_deliberation(full):
        # The model deliberated without committing to a final reply. The prior
        # behavior returned empty ("better empty than garbage"), but for a chat
        # persona an empty turn is the worst outcome — and gemma-4 emits its
        # candidate replies in QUOTES while deliberating, so the last quoted
        # candidate is a plausible in-voice reply, not raw thought. Salvage it
        # so the surface is never empty. Gated by HU_SALVAGE_RUNAWAY (default
        # on; set 0 to restore strict empty-on-runaway). The real fix is the
        # reply-first/ORPO retrain that stops the deliberation at the source;
        # this guarantees a non-empty turn until then.
        salvage_runaway = os.environ.get("HU_SALVAGE_RUNAWAY", "1").strip().lower() \
            not in ("0", "false", "no")
        runaway_quoted = _re.findall(r'"([^"]{1,200})"', full)
        if salvage_runaway and runaway_quoted and runaway_quoted[-1].strip():
            picked = runaway_quoted[-1].strip()
            try:
                print(f"  [runaway-salvage] no committed reply; salvaged candidate "
                      f"({len(full)} chars): {picked[:80]!r}", flush=True)
            except Exception:
                pass
            return _guard(picked)  # resolved -> caller emits the salvaged reply (unless echo)
        try:
            print(f"  [runaway-detected] pure deliberation, no extractable candidate "
                  f"({len(full)} chars); returning empty", flush=True)
        except Exception:
            pass
        return "", True

    # Salvage: last quoted string (gemma-4 emits candidates in quotes), else
    # the last non-empty line with surrounding bullets stripped.
    quoted = _re.findall(r'"([^"]{1,200})"', full)
    if quoted:
        salvaged = quoted[-1]
    else:
        salvaged = ""
        for line in reversed(full.split("\n")):
            cleaned_line = _re.sub(r"^\s*\*\s*", "", line).strip()
            if cleaned_line:
                salvaged = cleaned_line
                break
    try:
        print(f"  [strip-fallback] extractor empty; salvaged from raw "
              f"({len(full)} chars): {salvaged[:80]!r}", flush=True)
    except Exception:
        pass
    return _guard(salvaged)


def _resolve_should_buffer(req_flag, env_value, no_think_active):
    """Pure decision: should the streaming endpoint buffer+strip for this request?

    Buffering accumulates the whole generation and applies the non-stream
    thought-stripping before emitting ONE clean chunk (correct output, but
    TTFT == total). Not-buffering yields raw token chunks incrementally (low
    TTFT, but leaks markerless deliberation for models like seth-lora-v4-repair).

    Inputs, in precedence order (highest first):
      1. req_flag      — per-request `stream_strip` override. True = buffer/clean,
                         False = incremental/raw, None = "not specified" (fall through).
      2. env_value     — raw HU_STREAM_BUFFER_STRIP string ("1"/"0"/"true"/... or "").
      3. no_think_active — bool: a no-think instruction is configured (the model
                         deliberates), so default to buffering for clean output.

    This lets h-uman's model_router request incremental streaming per turn-type
    (e.g. casual/reflexive turns → stream_strip=false for realtime feel) while
    analytical/structured turns keep the clean buffered default — WITHOUT changing
    the server's global env. Per-request beats env beats model-default.
    """
    if req_flag is True:
        return True
    if req_flag is False:
        return False
    ev = (env_value or "").strip().lower()
    if ev in ("1", "true", "yes"):
        return True
    if ev in ("0", "false", "no"):
        return False
    return bool(no_think_active)


def _request_stream_strip(req):
    """Extract the per-request `stream_strip` override from a request body.

    Returns True/False when the field is a real bool, else None ("not specified").
    Non-bool values (ints, strings, missing) are ignored so a malformed field
    can never silently flip the global buffering policy — it just falls through
    to the env/model default.
    """
    if not isinstance(req, dict):
        return None
    v = req.get("stream_strip")
    return v if isinstance(v, bool) else None


def _stream_should_buffer(req=None):
    """Whether the streaming endpoint should buffer the full generation and
    apply non-stream thought-stripping before emitting, instead of yielding
    raw token chunks.

    Required for the seth-lora-v4-repair model: it deliberates in markdown
    bullets with NO channel markers, so the incremental strip-mode filter
    leaks the deliberation to the client (the reply can only be extracted
    once the full output is known). Confirmed live 2026-05-28: a feed-research
    structured prompt streamed bullet-deliberation while the non-stream path
    returned a clean report for the same prompt.

    Precedence: per-request `stream_strip` (body field) > HU_STREAM_BUFFER_STRIP
    env > model-deliberates default (GEMMA_DISABLE_THINKING / no-think config).
    Passing req=None reproduces the legacy env-or-default behavior exactly.
    """
    return _resolve_should_buffer(
        _request_stream_strip(req),
        os.environ.get("HU_STREAM_BUFFER_STRIP", ""),
        _no_think_instruction() is not None,
    )


def _resolve_skip_thinking_primer(req_flag, no_think_active):
    """Pure decision: should we skip Gemma's forced `<|channel>thought` primer?

    Skipping the primer (passing enable_thinking=True to the chat template)
    lets the seth-lora-v4-repair model reply FIRST instead of front-loading
    ~150 tokens of deliberation, which is what makes incremental streaming
    actually win first-token latency (streaming_beneficial:true). Proven by
    the Task-0 spike (2026-05-29): unprimed v4-repair emits short reply-first
    in-voice output with no channel markers.

    Inputs, in precedence order (highest first):
      1. req_flag        — per-request `stream_strip` override.
                           False = casual/incremental tier (reply-first wanted)
                                   → skip the primer.
                           True  = analytical/buffered tier (think-first kept)
                                   → fall through to the global default.
                           None  = "not specified" → fall through.
      2. no_think_active — bool: a no-think instruction is configured, so the
                           legacy default already skips the primer.

    Only `stream_strip=false` (casual) opts into reply-first; True/None preserve
    the EXACT current global behavior so heavy tiers and the non-stream path are
    untouched. Per-request beats model-default.
    """
    if req_flag is False:
        return True
    return bool(no_think_active)


def _request_skip_thinking_primer(req):
    """Whether to skip the thinking-primer for this streaming request.

    Casual streamed tiers (model_router sends stream_strip=false) get reply-first
    generation; every other request keeps the server's global primer behavior.
    Passing req=None reproduces the legacy default exactly.
    """
    return _resolve_skip_thinking_primer(
        _request_stream_strip(req),
        _no_think_requested(),
    )


_thinking_mode = False
_thinking_count = 0


class StreamThoughtFilter:
    """Streaming filter that strips only the `<|channel>thought` and `<channel|>`
    *markers* — keeping their content — because Gemma 4 routinely emits the user's
    answer *inside* an unclosed thought block instead of after the close tag.

    Two filter modes (selected by HU_THOUGHT_FILTER_MODE):
      - "strip" (default): drop only the literal OPEN/CLOSE markers; emit everything else
      - "discard": drop everything between OPEN..CLOSE (legacy behavior, may produce empty)
      - "off": passthrough, emit raw

    Mode "strip" is correct for current Gemma 4 deployments where answers live
    inside thought blocks. Mode "discard" is correct for well-trained models that
    keep answers strictly after `<channel|>`.
    """

    OPEN = "<|channel>thought"
    CLOSE = "<channel|>"

    # GLM `<think>` markers (production base since 2026-07-26). These are NOT
    # handled like OPEN/CLOSE: "strip" mode deliberately KEEPS Gemma thought
    # content because Gemma puts the answer inside the block, whereas GLM emits
    # `<think>reasoning</think>answer` — keeping the content there would stream
    # the reasoning verbatim. So GLM markers trigger hold-and-extract (fail
    # closed) rather than marker removal; see _feed_strip and flush.
    GLM_OPEN = "<think>"
    GLM_CLOSE = "</think>"

    def __init__(self, mode: str = "strip"):
        self.buf = ""
        self.mode = mode
        # Bullet-deliberation hold state. seth-lora-v4-repair deliberates in
        # bare markdown bullets that carry NO channel markers, so the marker
        # strip below can't catch them and they leak token-by-token to the
        # client (confirmed live 2026-05-28). When the FIRST non-marker content
        # of a response is a markdown bullet, we switch to hold-and-strip: keep
        # buffering, emit nothing, and let flush() run the full
        # strip_thought_channels extractor on the complete text. Clean
        # reply-first responses (no leading bullet) are unaffected and still
        # stream incrementally.
        self._emitted_any = False
        self._holding_bullets = False
        # Set when a GLM `<think>`/`</think>` marker is seen. Same hold-and-
        # extract contract as _holding_bullets: emit nothing, let flush() run
        # the full strip_thought_channels extractor on the complete text.
        self._holding_thought = False

    def _must_hold(self):
        """Emit nothing until flush() — the complete text is required to
        extract the real reply without leaking deliberation."""
        return self._holding_bullets or self._holding_thought

    def feed(self, chunk):
        if not chunk:
            return ""
        if self.mode == "off":
            return chunk
        self.buf += chunk
        if self.mode == "discard":
            return self._feed_discard()
        return self._feed_strip()

    def _max_marker_len(self):
        return max(len(self.OPEN), len(self.CLOSE),
                   len(self.GLM_OPEN), len(self.GLM_CLOSE))

    def _tail_could_be_marker_prefix(self):
        m = self._max_marker_len()
        markers = (self.OPEN, self.CLOSE, self.GLM_OPEN, self.GLM_CLOSE)
        for n in range(min(len(self.buf), m - 1), 0, -1):
            tail = self.buf[-n:]
            # GLM markers included so a partially-arrived "<thi" is retained
            # rather than emitted as literal text ahead of the hold decision.
            if any(mk.startswith(tail) for mk in markers):
                return n
        return 0

    def _feed_strip(self):
        # Once committed to holding deliberation, emit nothing until flush() —
        # the full text is needed to extract the real reply.
        if self._must_hold():
            return ""
        # GLM think block: either marker is sufficient to commit to holding.
        # The closer alone is the template-primed case (generation begins
        # already inside the thought, so only `</think>` is ever emitted).
        if self.GLM_OPEN in self.buf or self.GLM_CLOSE in self.buf:
            self._holding_thought = True
            return ""
        # Before the first emit, decide whether this response opens with
        # markdown-bullet deliberation. Defer the decision until we have a few
        # post-marker chars (or a newline) so a lone leading "*" token isn't
        # misjudged mid-stream.
        if not self._emitted_any:
            probe = self.buf.replace(self.OPEN, "").replace(self.CLOSE, "").lstrip()
            if probe and (len(probe) >= 4 or "\n" in self.buf):
                # A markdown bullet is "*" followed by whitespace (or another
                # "*"). "*emphasis*" replies start with "*"+letter and are NOT
                # held. probe is already lstrip'd, so index 0 is the first
                # non-space char.
                if probe[0] == "*" and (len(probe) < 2 or probe[1] in " \t\n*"):
                    self._holding_bullets = True
                    return ""
            elif probe:
                # Not enough leading content to classify yet — keep buffering.
                return ""
        keep = self._tail_could_be_marker_prefix()
        emit_end = len(self.buf) - keep if keep > 0 else len(self.buf)
        if emit_end <= 0:
            return ""
        emit = self.buf[:emit_end]
        self.buf = self.buf[emit_end:]
        emit = emit.replace(self.OPEN, "").replace(self.CLOSE, "")
        if emit.strip():
            self._emitted_any = True
        return emit

    def _feed_discard(self):
        out = []
        in_thought = getattr(self, "_in_thought", False)
        while self.buf:
            if in_thought:
                idx = self.buf.find(self.CLOSE)
                if idx == -1:
                    if len(self.buf) > len(self.CLOSE):
                        self.buf = self.buf[-len(self.CLOSE):]
                    self._in_thought = True
                    return "".join(out)
                self.buf = self.buf[idx + len(self.CLOSE):]
                in_thought = False
                continue
            idx = self.buf.find(self.OPEN)
            if idx == -1:
                keep = 0
                for n in range(min(len(self.buf), len(self.OPEN) - 1), 0, -1):
                    if self.OPEN.startswith(self.buf[-n:]):
                        keep = n
                        break
                if keep > 0:
                    out.append(self.buf[:-keep])
                    self.buf = self.buf[-keep:]
                else:
                    out.append(self.buf)
                    self.buf = ""
                self._in_thought = False
                return "".join(out)
            out.append(self.buf[:idx])
            self.buf = self.buf[idx + len(self.OPEN):]
            in_thought = True
        self._in_thought = in_thought
        return "".join(out)

    def flush(self):
        if self.mode == "off":
            out, self.buf = self.buf, ""
            return out
        if self.mode == "discard" and getattr(self, "_in_thought", False):
            self.buf = ""
            return ""
        out = self.buf
        self.buf = ""
        if self._must_hold():
            # Deliberation (bullet or GLM think block) was detected and held;
            # extract the real reply from the complete text using the same logic
            # the non-stream and buffered-stream paths use, so nothing leaks.
            return strip_thought_channels(out)
        return out.replace(self.OPEN, "").replace(self.CLOSE, "")


_stream_filter = None


def reset_stream_filter():
    global _stream_filter
    mode = os.environ.get("HU_THOUGHT_FILTER_MODE", "strip").strip().lower()
    if mode not in ("strip", "discard", "off"):
        mode = "strip"
    _stream_filter = StreamThoughtFilter(mode=mode)


def stream_filter_feed(text):
    global _stream_filter
    if _stream_filter is None:
        _stream_filter = StreamThoughtFilter()
    return _stream_filter.feed(text)


def stream_filter_flush():
    global _stream_filter
    if _stream_filter is None:
        return ""
    return _stream_filter.flush()


def filter_thinking(text):
    """Legacy single-token filter retained for back-compat in non-stream call sites.
    For streaming, use stream_filter_feed/reset/flush instead."""
    global _thinking_mode, _thinking_count
    stripped = text.strip()
    if stripped in ("|", "") and _thinking_count < 500:
        _thinking_mode = True
        _thinking_count += 1
        return ""
    if _thinking_mode:
        _thinking_mode = False
    return text


def reset_thinking_filter():
    global _thinking_mode, _thinking_count
    _thinking_mode = False
    _thinking_count = 0


def _init_turbo_cache():
    """Initialize TurboQuant+ KV cache for the loaded model."""
    global turbo_cache
    if kv_bits is None or model is None:
        return None

    bits = int(kv_bits) if kv_bits == int(kv_bits) else 4

    try:
        from mlx.nn.layers.turbo_kv_cache import make_turbo_cache
        key_bits = 16 if kv_quant_scheme == "asymmetric" else bits
        turbo_cache = make_turbo_cache(model, bits=bits, key_bits=key_bits)
        mode = f"K=FP16 V={bits}b asymmetric" if key_bits == 16 else f"{bits}-bit symmetric"
        print(f"  TurboQuant+ KV cache initialized: {mode}", flush=True)
        return turbo_cache
    except ImportError:
        pass

    try:
        from mlx.nn.layers.turbo_kv_cache import TurboKVCache
        n_layers = len(model.model.layers) if hasattr(model, "model") else 32
        key_bits = 16 if kv_quant_scheme == "asymmetric" else bits
        turbo_cache = [TurboKVCache(bits=bits, key_bits=key_bits) for _ in range(n_layers)]
        print(f"  TurboQuant+ KV cache initialized: K={key_bits}b V={bits}b ({n_layers} layers)", flush=True)
        return turbo_cache
    except ImportError:
        print("  TurboQuant+ not available — install with:", flush=True)
        print("    pip install git+https://github.com/TheTom/mlx.git@feature/turboquant-plus", flush=True)
        return None


def _compact_turbo_cache():
    """Compress prefill FP16 data to TurboQuant packed storage."""
    if turbo_cache is None:
        return
    try:
        from mlx.nn.layers.turbo_kv_cache import compact_turbo_cache
        compact_turbo_cache(turbo_cache)
    except ImportError:
        pass


lm_cache_supports_quant = False


def _init_lm_prompt_cache():
    """Build a persistent mlx_lm prompt cache for cross-turn reuse.

    Used as the fallback path when TurboQuant+ is not installed. Combined
    with kv_bits/quantized_kv_start, mlx_lm auto-quantizes the cache to
    QuantizedKVCache after `quantized_kv_start` tokens of prefill — but
    only for cache types that implement to_quantized. RotatingKVCache
    (Gemma 4 sliding-window attention) raises NotImplementedError on
    quantization, so we detect that and disable kv_bits for that path.
    """
    global lm_prompt_cache, lm_cache_supports_quant, lm_cache_slots
    if turbo_cache is not None or model is None or use_lm_path is False:
        return None
    try:
        from mlx_lm.models.cache import make_prompt_cache
        lm_prompt_cache = make_prompt_cache(model)

        first = lm_prompt_cache[0] if lm_prompt_cache else None
        cache_type = type(first).__name__ if first is not None else "?"
        if first is not None and hasattr(first, "to_quantized"):
            try:
                _ = first.to_quantized(group_size=64, bits=4)
                lm_cache_supports_quant = True
                first.reset() if hasattr(first, "reset") else None
            except (NotImplementedError, Exception):
                lm_cache_supports_quant = False
            lm_prompt_cache = make_prompt_cache(model)

        tag = f" (kv_bits={int(kv_bits)})" if kv_bits is not None and lm_cache_supports_quant else ""
        if kv_bits is not None and not lm_cache_supports_quant:
            tag = f" (kv_bits requested but {cache_type} unsupported in stock mlx_lm — install TurboQuant+)"
        lm_cache_slots = None  # rebuild the slot pool around the fresh cache
        slots = _ensure_cache_slots() or []
        print(f"  Stock mlx_lm prompt cache initialized: {len(lm_prompt_cache)} layers of {cache_type}{tag}"
              f" x {len(slots)} slot(s)", flush=True)
        return lm_prompt_cache
    except Exception as e:
        print(f"  Stock mlx_lm prompt cache init failed: {e}", flush=True)
        return None


def _kv_kwargs(use_lm_cache=True):
    """Build KV cache kwargs for generate_step (works for both mlx_lm and mlx_vlm).

    Precedence:
      1. TurboQuant+ cache (turbo_cache list) — design-intent path.
      2. Stock mlx_lm prompt cache + kv_bits quantization — fallback path.
      3. Plain kv_bits (no cache reuse) — last resort for vlm path.
    """
    extra = {}
    if turbo_cache is not None:
        extra["prompt_cache"] = turbo_cache
    elif use_lm_cache and lm_prompt_cache is not None:
        extra["prompt_cache"] = lm_prompt_cache
        if kv_bits is not None and lm_cache_supports_quant:
            extra["kv_bits"] = int(kv_bits)
            extra["kv_group_size"] = lm_quantized_kv_group_size
            extra["quantized_kv_start"] = lm_cache_quantized_start
    elif kv_bits is not None:
        extra["kv_bits"] = int(kv_bits)
    return extra


def _cache_offset(cache_list):
    """Read the current token offset from any prompt cache list."""
    if not cache_list:
        return 0
    first = cache_list[0]
    if hasattr(first, "offset"):
        return int(first.offset)
    return int(getattr(first, "size", 0) or 0)


def _trim_cache(cache_list, n):
    """Trim n tokens from the tail of a prompt cache list. Returns tokens actually trimmed."""
    if not cache_list or n <= 0:
        return 0
    try:
        from mlx_lm.models.cache import trim_prompt_cache, can_trim_prompt_cache
        if can_trim_prompt_cache(cache_list):
            trimmed = trim_prompt_cache(cache_list, n)
            return int(trimmed) if isinstance(trimmed, (int, float)) else n
    except Exception:
        pass
    trimmed = 0
    for layer_cache in cache_list:
        if hasattr(layer_cache, "trim"):
            try:
                layer_cache.trim(n)
                trimmed = n
            except Exception:
                pass
    return trimmed


def _active_prompt_cache():
    """Return whichever prompt-cache list is live for reuse, or None."""
    if turbo_cache is not None:
        return turbo_cache
    if lm_prompt_cache is not None:
        return lm_prompt_cache
    return None


def _fresh_lm_cache():
    """A new empty prompt-cache list matching the model's layer structure.

    Falls back to cloning the layer-cache types of the existing list when no
    model is loaded (wiring tests) — KVCache and friends are no-arg there.
    Returns None if neither route works.
    """
    if model is not None:
        try:
            from mlx_lm.models.cache import make_prompt_cache
            return make_prompt_cache(model)
        except Exception:
            return None
    if lm_prompt_cache:
        try:
            return [type(c)() for c in lm_prompt_cache]
        except Exception:
            return None
    return None


def _ensure_cache_slots():
    """Adopt lm_prompt_cache into the slot pool (idempotent).

    Slot 0 wraps the current lm_prompt_cache; the remaining slots get fresh
    empty caches. Empty caches hold no tensors, so an idle pool costs
    nothing — memory is only pinned by slots that have actually prefilled.
    """
    global lm_cache_slots
    if lm_prompt_cache is None:
        return None
    if lm_cache_slots:
        return lm_cache_slots
    slots = [{"cache": lm_prompt_cache, "prev_tokens": None, "last_used": 0}]
    while len(slots) < max(1, int(lm_cache_slot_count)):
        fresh = _fresh_lm_cache()
        if fresh is None:
            break
        slots.append({"cache": fresh, "prev_tokens": None, "last_used": 0})
    lm_cache_slots = slots
    return lm_cache_slots


def _lcp_active_slot():
    """The slot whose cache lm_prompt_cache currently points at, or None."""
    if not lm_cache_slots:
        return None
    for slot in lm_cache_slots:
        if slot["cache"] is lm_prompt_cache:
            return slot
    return None


def _reinit_cache_slot(idx):
    """Fail-safe: rebuild one slot's cache after a trim we can't trust."""
    global lm_prompt_cache
    slot = lm_cache_slots[idx]
    was_active = slot["cache"] is lm_prompt_cache
    fresh = _fresh_lm_cache()
    if fresh is not None:
        slot["cache"] = fresh
    else:
        # Can't rebuild (no model, clone failed) — hard-reset what's there.
        _trim_cache(slot["cache"], _cache_offset(slot["cache"]))
    slot["prev_tokens"] = None
    if was_active:
        lm_prompt_cache = slot["cache"]


def _ensure_shadow_slots():
    """Shadow mode simulates the pool with token lists only — no KV, no
    mutation of the real cache (which the legacy path keeps managing)."""
    global _lcp_shadow_slots
    n = max(1, int(lm_cache_slot_count))
    if not _lcp_shadow_slots or len(_lcp_shadow_slots) != n:
        _lcp_shadow_slots = [{"prev_tokens": None, "last_used": 0} for _ in range(n)]
    return _lcp_shadow_slots


def _invalidate_cross_turn_caches(reason):
    """Drop ALL cross-turn cache state — KV contents and bookkeeping.

    Must run after any weight mutation (LoRA adapter swap or revert): every
    cached KV entry was computed under the OLD weights, so reusing it — via
    the legacy system-hash trim or live LCP slot reuse — would blend stale
    activations into new-weight generation. Found live 2026-07-25: the swap
    endpoint changed weights while the legacy system-boundary cache and the
    LCP slots kept their KV.
    """
    global _cached_system_hash, _cached_system_ntokens
    global _lcp_shadow_slots, _lcp_pending_tokens, _lcp_pending_slot

    _cached_system_hash = None
    _cached_system_ntokens = 0
    _lcp_shadow_slots = None
    _lcp_pending_tokens = None
    _lcp_pending_slot = None
    if lm_cache_slots:
        for idx, slot in enumerate(lm_cache_slots):
            n = _cache_offset(slot["cache"])
            if n > 0 and _trim_cache(slot["cache"], n) != n:
                _reinit_cache_slot(idx)  # trim untrustworthy — rebuild the slot
            slot["prev_tokens"] = None
    elif lm_prompt_cache is not None:
        _trim_cache(lm_prompt_cache, _cache_offset(lm_prompt_cache))
    if turbo_cache is not None:
        _trim_cache(turbo_cache, _cache_offset(turbo_cache))
    print(f"  [cache] cross-turn caches invalidated ({reason})", flush=True)


def _prepare_cache_for_request(messages):
    """Trim cache back to system prompt boundary for cross-turn reuse.

    If the system prompt hasn't changed, we keep its KV state and only
    re-process the new user/assistant tokens. If it changed (or this is
    the first request), the cache is fully reset.
    """
    global _cached_system_hash, _cached_system_ntokens

    # Any legacy-managed request invalidates the LCP bookkeeping for the
    # slot it mutates: after this function runs, that slot's prev_tokens no
    # longer describes its cache's token prefix. Only the ACTIVE slot is
    # touched here — the others keep their KV state and bookkeeping.
    _slot = _lcp_active_slot()
    if _slot is not None:
        _slot["prev_tokens"] = None

    cache_list = _active_prompt_cache()
    if cache_list is None:
        return

    system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
    sys_hash = hash(tuple(system_parts)) if system_parts else None

    cache_size = _cache_offset(cache_list)

    if sys_hash is not None and sys_hash == _cached_system_hash and _cached_system_ntokens > 0:
        trim_amount = cache_size - _cached_system_ntokens
        if trim_amount > 0:
            _trim_cache(cache_list, trim_amount)
        return

    if cache_size > 0:
        _trim_cache(cache_list, cache_size)
    _cached_system_hash = sys_hash
    _cached_system_ntokens = 0


def _record_system_cache_boundary(messages):
    """After prefill, record how many tokens the system prompt occupies."""
    global _cached_system_ntokens

    if _active_prompt_cache() is None or _cached_system_ntokens > 0:
        return

    system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
    if not system_parts or processor is None:
        return

    sys_text = "\n".join(system_parts)
    try:
        sys_tokens = processor.encode(sys_text, add_special_tokens=False)
        _cached_system_ntokens = len(sys_tokens) + 6
    except Exception:
        pass


def _encode_like_stream_generate(text):
    """Tokenize exactly the way mlx_lm.stream_generate tokenizes str prompts:
    add special tokens only when the text doesn't already start with BOS.
    Keeping this identical is what makes passing token ids in live mode a
    no-op on model input."""
    bos = getattr(processor, "bos_token", None)
    add_special = bos is None or not text.startswith(bos)
    return list(processor.encode(text, add_special_tokens=add_special))


def _plan_cache_for_prompt(messages, prompt_text, allow_lcp=True):
    """Per-request cache management. Returns the payload for stream_generate:
    the prompt string (off/shadow — legacy hash path manages the cache) or a
    token-id list (live — chosen slot trimmed to the LCP, suffix to prefill).

    Live mode applies only on the stock lm_prompt_cache (turbo cache trim
    semantics are unproven) and when the caller allows it (non-speculative
    paths). Everything else falls back to legacy, keeping today's behavior.

    Slot routing (both shadow and live): choose_slot picks the slot with
    the best LCP; below the floor it recycles an empty/LRU slot instead of
    evicting the best one, so the daemon's big-head slot survives
    interleaved small-probe traffic.
    """
    global lm_prompt_cache, _lcp_pending_tokens, _lcp_pending_slot
    global _lcp_last, _lcp_request_counter

    _lcp_pending_tokens = None
    _lcp_pending_slot = None
    live_ok = (
        lcp_mode == "live" and allow_lcp
        and turbo_cache is None and lm_prompt_cache is not None
    )

    if lcp_mode == "off" or (lcp_mode == "live" and not live_ok):
        _prepare_cache_for_request(messages)
        return prompt_text

    if lcp_mode == "shadow":
        # Measure what the multi-slot pool would reuse, then run the legacy
        # path unchanged. The simulation lives entirely in token lists, so
        # legacy cache mutation can't corrupt it.
        try:
            from prompt_cache_lcp import choose_slot
            tokens = _encode_like_stream_generate(prompt_text)
            sim = _ensure_shadow_slots()
            slot_idx, reuse = choose_slot(
                [s["prev_tokens"] for s in sim],
                [s["last_used"] for s in sim], tokens)
            _lcp_request_counter += 1
            _lcp_last = {"mode": "shadow", "would_reuse": reuse,
                         "prompt_tokens": len(tokens),
                         "slot": slot_idx, "slots": len(sim),
                         "slot_hit": reuse > 0}
            print(f"  [lcp shadow] would reuse {reuse}/{len(tokens)} prompt toks "
                  f"(slot {slot_idx}, best-of-{len(sim)}, "
                  f"{'hit' if reuse > 0 else 'miss'})", flush=True)
            _prepare_cache_for_request(messages)  # legacy manages the real cache
            sim[slot_idx] = {"prev_tokens": tokens,
                             "last_used": _lcp_request_counter}
        except Exception as e:
            print(f"  [lcp shadow] measurement failed: {e}", flush=True)
            _prepare_cache_for_request(messages)
        return prompt_text

    # live — route the request to the best slot, or recycle an empty/LRU one
    from prompt_cache_lcp import (choose_slot, plan_reuse, common_prefix_len,
                                  DEFAULT_SLOT_FLOOR)
    tokens = _encode_like_stream_generate(prompt_text)
    slots = _ensure_cache_slots()
    slot_idx, planned = choose_slot(
        [s["prev_tokens"] for s in slots],
        [s["last_used"] for s in slots], tokens)
    slot = slots[slot_idx]
    lm_prompt_cache = slot["cache"]  # _kv_kwargs routes this into stream_generate
    cache_size = _cache_offset(slot["cache"])
    if planned > 0:
        reuse, trim_amount = plan_reuse(slot["prev_tokens"], tokens, cache_size)
    else:
        # EVICTION instrumentation (2026-07-27): live token-weighted reuse is 34%
        # vs 60% in shadow, and the misses are LONG prompts (miss-median 1199 toks
        # vs shadow's 232). choose_slot evicts by min(last_used) — pure LRU, size-
        # INDIFFERENT — so any size dependence must come from traffic pattern, not
        # from an eviction policy that prefers big entries. Log what we'd need to
        # tell those apart: what died, how stale it was, and the whole pool, so a
        # reader can check whether the victim was the OLDEST (LRU as designed) or
        # happened to be the LARGEST (would mean the model above is wrong).
        if cache_size > 0:
            _ev_prev = len(slot["prev_tokens"] or ())
            _ev_age = _lcp_request_counter - slot["last_used"]
            _pool = ",".join(
                "%d:%dt/%da" % (i, len(s["prev_tokens"] or ()),
                                _lcp_request_counter - s["last_used"])
                for i, s in enumerate(slots))
            # Best prefix ANY slot offered. If this sits just under the floor the
            # miss is a near-miss (a floor-tuning lever); if it's ~0 the prompt
            # genuinely shares nothing and no slot count would have helped.
            _best = max((common_prefix_len(s["prev_tokens"] or (), tokens)
                         for s in slots), default=0)
            print(f"  [lcp evict] slot {slot_idx}: dropping {cache_size} kv pos "
                  f"({_ev_prev} prev toks), idle {_ev_age} reqs; "
                  f"incoming {len(tokens)} toks; best_lcp {_best}/floor "
                  f"{DEFAULT_SLOT_FLOOR}; pool[{_pool}]", flush=True)
        reuse, trim_amount = 0, cache_size  # recycle the slot as a fresh cache
    if trim_amount > 0:
        trimmed = _trim_cache(slot["cache"], trim_amount)
        if trimmed != trim_amount:
            # Fail-safe: partial/failed trim would leave KV state that does
            # not match our bookkeeping — rebuild THIS slot and prefill all.
            print(f"  [lcp live] trim {trim_amount} honored as {trimmed} — "
                  f"reinitializing slot {slot_idx}", flush=True)
            _reinit_cache_slot(slot_idx)
            reuse = 0
    _lcp_request_counter += 1
    slot["prev_tokens"] = None  # invalid until prefill completes
    slot["last_used"] = _lcp_request_counter
    _lcp_pending_tokens = tokens
    _lcp_pending_slot = slot_idx
    _lcp_last = {"mode": "live", "reused": reuse, "prompt_tokens": len(tokens),
                 "slot": slot_idx, "slots": len(slots)}
    print(f"  [lcp live] reusing {reuse}/{len(tokens)} prompt toks "
          f"(slot {slot_idx}, best-of-{len(slots)}), "
          f"prefilling {len(tokens) - reuse}", flush=True)
    return tokens[reuse:]


def _after_prefill(messages):
    """Post-prefill bookkeeping for whichever cache scheme handled this
    request: promote pending LCP tokens to their slot, or record the legacy
    system boundary."""
    global _lcp_pending_tokens, _lcp_pending_slot
    if _lcp_pending_tokens is not None:
        if (lm_cache_slots and _lcp_pending_slot is not None
                and 0 <= _lcp_pending_slot < len(lm_cache_slots)):
            lm_cache_slots[_lcp_pending_slot]["prev_tokens"] = _lcp_pending_tokens
        _lcp_pending_tokens = None
        _lcp_pending_slot = None
        return
    _record_system_cache_boundary(messages)


def generate_response(messages, max_tokens=256, temperature=0.7, skip_thinking_primer=None):
    """Non-streaming: generate the full response at once.

    `skip_thinking_primer` mirrors stream_response: it is forwarded to
    prepare_prompt_lm so the non-stream `/v1/chat/completions` path makes the
    SAME primer decision the streaming path does, instead of always relying on
    prepare_prompt_lm's bare default. This matters for a server with no-think
    OFF where a casual caller sends stream_strip=false — that request now drops
    Gemma 4's forced `<|channel>thought` opener here too.

    NOTE (verified live 2026-06-01, seth-lora-v4-repair-20260525): dropping the
    primer does NOT by itself cut tokens for this adapter — it was trained to
    deliberate and emits ~150-300 tokens of markdown-bullet reasoning regardless
    (see _thinking_headroom_tokens). Under GEMMA_DISABLE_THINKING the primer is
    already skipped on both paths, so this arg is a no-op there; the non-stream
    token cost is the adapter's deliberation + the intentional thinking headroom,
    not a missing flag. Kept for path consistency + the no-think-off case above.

    None = legacy default (skip iff a no-think instruction is configured);
    True/False = explicit per-request override. See _request_skip_thinking_primer.
    """
    has_imgs = _has_images(messages)

    if use_lm_path and not has_imgs:
        from mlx_lm import stream_generate as lm_stream_generate
        from mlx_lm.sample_utils import make_sampler
        prompt = prepare_prompt_lm(messages, skip_thinking_primer)
        prompt = _plan_cache_for_prompt(messages, prompt)
        extra = _kv_kwargs()
        extra["sampler"] = make_sampler(temp=temperature)
        _rp = _repetition_logits_processors()
        if _rp:
            extra["logits_processors"] = _rp
        text_parts = []
        prompt_toks_out = 0
        gen_toks_out = 0
        prefill_done = False
        for resp in lm_stream_generate(
            model, processor, prompt=prompt,
            max_tokens=max_tokens,
            **extra,
        ):
            if not prefill_done:
                _compact_turbo_cache()
                _after_prefill(messages)
                prefill_done = True
            prompt_toks_out = getattr(resp, "prompt_tokens", prompt_toks_out)
            gen_toks_out = getattr(resp, "generation_tokens", gen_toks_out)
            t = resp.text or ""
            cleaned, hit_stop = strip_stop_tokens(t)
            text_parts.append(cleaned)
            if hit_stop:
                break
        full = "".join(text_parts).strip()
        # Shared finalize: strip thought channels, dump-on-empty diagnostics,
        # runaway guard, and salvage heuristic all live in finalize_generation
        # so the buffered streaming path (_handle_stream_buffered) applies the
        # IDENTICAL logic. See m3_live_path_extractor_strip.md "runaway" arc.
        stripped, _runaway = finalize_generation(full, messages)
        return stripped, prompt_toks_out, gen_toks_out

    from mlx_vlm import generate as vlm_generate
    formatted, images = prepare_prompt_vlm(messages)
    extra = _kv_kwargs()
    if images:
        extra["images"] = images
    result = vlm_generate(
        model, processor, formatted,
        max_tokens=max_tokens, temperature=temperature, verbose=False,
        **extra,
    )
    text, _ = strip_stop_tokens(result.text if hasattr(result, "text") else result)
    prompt_toks = getattr(result, "prompt_tokens", 0)
    gen_toks = getattr(result, "generation_tokens", 0)
    return strip_thought_channels(text), prompt_toks, gen_toks


def stream_response(messages, max_tokens=256, temperature=0.7, skip_thinking_primer=None):
    """Streaming generator: yield (text_chunk, prompt_toks, gen_toks) per token.

    Fast path: mlx_lm.stream_generate (no vision overhead, no numpy sync)
    Fallback: mlx_vlm.stream_generate (multimodal, slower)

    When speculative decoding is enabled, uses the draft model to propose
    multiple tokens that the target model verifies in parallel.

    `skip_thinking_primer` is forwarded to prepare_prompt_lm so streamed casual
    tiers (stream_strip=false) generate reply-first. None = legacy default.
    """
    has_imgs = _has_images(messages)
    prompt_toks = 0
    gen_toks = 0

    # Speculative decoding path — mlx_lm.stream_generate supports draft_model natively
    if speculative_enabled and draft_model is not None and not has_imgs:
        try:
            from mlx_lm import stream_generate as lm_stream_generate
            from mlx_lm.sample_utils import make_sampler
            _prepare_cache_for_request(messages)
            prompt = prepare_prompt_lm(messages, skip_thinking_primer)
            extra = _kv_kwargs()
            extra["sampler"] = make_sampler(temp=temperature)
            _rp = _repetition_logits_processors()
            if _rp:
                extra["logits_processors"] = _rp

            import inspect
            sig = inspect.signature(lm_stream_generate)
            if "num_draft_tokens" in sig.parameters:
                extra["num_draft_tokens"] = speculative_draft_tokens

            prefill_done = False
            for resp in lm_stream_generate(
                model=model,
                tokenizer=processor,
                prompt=prompt,
                max_tokens=max_tokens,
                draft_model=draft_model,
                **extra,
            ):
                if not prefill_done:
                    _compact_turbo_cache()
                    _record_system_cache_boundary(messages)
                    prefill_done = True
                prompt_toks = getattr(resp, "prompt_tokens", prompt_toks)
                gen_toks = getattr(resp, "generation_tokens", gen_toks)
                text = resp.text or ""
                cleaned, hit_stop = strip_stop_tokens(text)
                if cleaned:
                    yield cleaned, prompt_toks, gen_toks
                if hit_stop:
                    return
                finish = getattr(resp, "finish_reason", None)
                if finish in ("stop", "length"):
                    return
            return
        except (ImportError, AttributeError):
            pass
        except (IndexError, RuntimeError) as e:
            # Gemma 4's sliding-window cache is incompatible with some draft models
            print(f"  [spec] Speculative decoding failed ({type(e).__name__}: {e}), falling back to standard generation", flush=True)

    # Fast text path via mlx_lm
    if use_lm_path and not has_imgs:
        from mlx_lm import stream_generate as lm_stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = prepare_prompt_lm(messages, skip_thinking_primer)
        prompt = _plan_cache_for_prompt(messages, prompt)
        extra = _kv_kwargs()
        extra["sampler"] = make_sampler(temp=temperature)
        _rp = _repetition_logits_processors()
        if _rp:
            extra["logits_processors"] = _rp

        prefill_done = False
        for resp in lm_stream_generate(
            model, processor, prompt=prompt,
            max_tokens=max_tokens,
            **extra,
        ):
            if not prefill_done:
                _compact_turbo_cache()
                _after_prefill(messages)
                prefill_done = True
            prompt_toks = getattr(resp, "prompt_tokens", prompt_toks)
            gen_toks = getattr(resp, "generation_tokens", gen_toks)
            text = resp.text or ""
            cleaned, hit_stop = strip_stop_tokens(text)
            if cleaned:
                yield cleaned, prompt_toks, gen_toks
            if hit_stop:
                return
            finish = getattr(resp, "finish_reason", None)
            if finish in ("stop", "length"):
                return
        return

    # Multimodal fallback via mlx_vlm
    from mlx_vlm import stream_generate as vlm_stream_generate
    formatted, images = prepare_prompt_vlm(messages)
    extra = _kv_kwargs()
    if images:
        extra["images"] = images

    for chunk in vlm_stream_generate(
        model, processor, formatted,
        max_tokens=max_tokens, temperature=temperature,
        **extra,
    ):
        prompt_toks = getattr(chunk, "prompt_tokens", prompt_toks)
        gen_toks = getattr(chunk, "generation_tokens", gen_toks)
        text = chunk.text or ""

        cleaned, hit_stop = strip_stop_tokens(text)
        if cleaned:
            yield cleaned, prompt_toks, gen_toks
        if hit_stop:
            return

    return


class ChatHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {args[0]}", flush=True)

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            health = {"status": "ok", "model": model_id, "engine": "mlx_lm" if use_lm_path else "mlx_vlm"}
            if kv_bits is not None:
                health["kv_bits"] = kv_bits
                health["kv_quant_scheme"] = kv_quant_scheme
                health["turboquant_plus"] = turbo_cache is not None
            health["lm_prompt_cache_active"] = lm_prompt_cache is not None
            health["prompt_cache_lcp"] = lcp_mode
            health["prompt_cache_lcp_slots"] = lm_cache_slot_count
            if _lcp_last:
                health["prompt_cache_lcp_last"] = dict(_lcp_last)
            if lm_cache_slots:
                health["prompt_cache_lcp_slot_state"] = [
                    {"prev_tokens": len(s["prev_tokens"] or []),
                     "offset": _cache_offset(s["cache"]),
                     "last_used": s["last_used"],
                     "active": s["cache"] is lm_prompt_cache}
                    for s in lm_cache_slots]
            if _cached_system_ntokens > 0 and _active_prompt_cache() is not None:
                health["cached_system_tokens"] = _cached_system_ntokens
                health["prompt_cache"] = "active"
            elif prompt_cache_state is not None and prompt_cache_state.token_ids is not None:
                health["cached_tokens"] = len(prompt_cache_state.token_ids)
            if adapter_path_global:
                health["adapter"] = adapter_path_global
            # Always include active_adapter (null when no adapter is applied) for the
            # adapter-swap state surface — consumed by the C side personalization loop.
            health["active_adapter"] = adapter_path_global
            # Observability: distinguish "adapter configured" (a path) from "adapter
            # actually applied" (weights loaded). adapter_applied=False with a non-null
            # adapter path means the server silently fell back to BASE weights — the
            # exact fail-silent that masked an inactive persona fine-tune. tensors_loaded
            # is the LoRA tensor count from the most recent load/swap.
            health["tensors_loaded"] = tensors_loaded_global
            health["adapter_applied"] = tensors_loaded_global > 0
            if speculative_enabled:
                health["speculative_decoding"] = True
                health["draft_tokens"] = speculative_draft_tokens
            if hw_info:
                health["hardware"] = hw_info
            if perf_stats["requests"] > 0:
                avg_tps = perf_stats["total_tokens"] / perf_stats["total_time"] if perf_stats["total_time"] > 0 else 0
                health["avg_tok_per_sec"] = round(avg_tps, 1)
                health["total_requests"] = perf_stats["requests"]
            health["inference_tuning"] = {
                "realtime_voice_mode": server_realtime_mode,
                "speculative_draft_tokens": speculative_draft_tokens,
                "speculative_active": speculative_enabled,
                "kv_bits": kv_bits,
                "kv_scheme": kv_quant_scheme,
                "turboquant_active": turbo_cache is not None,
            }
            iosurf = _iosurface_http_health()
            if iosurf:
                health["iosurface"] = iosurf
            self._send_json(200, health)
            return

        if self.path == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [{
                    "id": model_id,
                    "object": "model",
                    "owned_by": "local-mlx",
                }]
            })
            return

        if self.path == "/v1/adapters/current":
            self._send_json(200, {
                "adapter_path": adapter_path_global,
                "tensors_loaded": tensors_loaded_global if adapter_path_global else 0,
            })
            return

        self._send_json(404, {"error": "not found"})

    def _handle_stream(self, req, resp_id, t0):
        messages = req.get("messages", [])
        max_tokens = req.get("max_tokens", 256)
        temperature = req.get("temperature", 0.7)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        # RUNAWAY GUARD for the streaming path (2026-05-28): the
        # seth-lora-v4-repair model deliberates in markdown bullets with NO
        # channel markers, so the incremental strip-mode filter cannot tell
        # deliberation from the reply and leaks it to the client. The reply is
        # only extractable once the full output is known, so when buffering is
        # enabled we accumulate the whole generation and apply the IDENTICAL
        # finalize_generation logic the non-stream path uses, then emit the
        # clean reply as a single SSE content chunk. See _stream_should_buffer.
        if _stream_should_buffer(req):
            self._handle_stream_buffered(messages, resp_id, t0, max_tokens, temperature)
            return

        full_text = []
        prompt_toks = 0
        gen_toks = 0
        first_token_time = None
        reset_thinking_filter()
        reset_stream_filter()

        debug_stream = os.environ.get("HU_DEBUG_STREAM", "").strip() in ("1", "true", "yes")
        debug_log = []
        # Casual streamed tiers (stream_strip=false) skip Gemma's forced thinking
        # primer so v4-repair replies FIRST → real first-token latency win
        # (streaming_beneficial:true). Heavy/buffered tiers took the early return
        # above and are unaffected. See _request_skip_thinking_primer (Task-0 spike).
        skip_primer = _request_skip_thinking_primer(req)
        with model_lock:
            for text, pt, gt in stream_response(messages, max_tokens, temperature,
                                                skip_thinking_primer=skip_primer):
                prompt_toks = pt
                gen_toks = gt

                if debug_stream:
                    debug_log.append(("raw", text))
                filtered = stream_filter_feed(text)
                if not filtered:
                    continue

                if first_token_time is None:
                    first_token_time = time.time()
                full_text.append(filtered)

                chunk = {
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": filtered},
                        "finish_reason": None,
                    }],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

        if debug_stream:
            preview = "".join(t for _, t in debug_log[:60])
            print(f"  [debug-stream] {len(debug_log)} raw chunks, "
                  f"first 60 concatenated: {preview!r}", flush=True)
        # Flush any answer content stuck in the thought-filter tail buffer.
        tail = stream_filter_flush()
        if tail:
            full_text.append(tail)
            chunk = {
                "id": resp_id, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": model_id,
                "choices": [{"index": 0, "delta": {"content": tail}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        done_chunk = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_toks,
                "completion_tokens": gen_toks,
                "total_tokens": prompt_toks + gen_toks,
            },
        }
        self.wfile.write(f"data: {json.dumps(done_chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

        elapsed = time.time() - t0
        ttft = (first_token_time - t0) if first_token_time else elapsed
        combined = "".join(full_text)
        preview = combined[:60].replace("\n", " ")
        tps = gen_toks / elapsed if elapsed > 0 else 0
        cache_tag = f" [TQ{kv_bits}b]" if kv_bits is not None else ""
        spec_tag = " [spec]" if speculative_enabled else ""
        reused = ""
        if _cached_system_ntokens > 0:
            reused = f" [cache:{_cached_system_ntokens} sys toks]"
        elif prompt_cache_state is not None and prompt_cache_state.token_ids is not None:
            reused = f" [cache:{len(prompt_cache_state.token_ids)} toks]"

        perf_stats["total_tokens"] += gen_toks
        perf_stats["total_time"] += elapsed
        perf_stats["requests"] += 1

        print(f"  -> {gen_toks} tokens in {elapsed:.1f}s ({tps:.1f} tok/s, TTFT {ttft:.2f}s){cache_tag}{spec_tag}{reused} | {preview}...", flush=True)

    def _handle_stream_buffered(self, messages, resp_id, t0, max_tokens, temperature):
        """Buffered streaming: accumulate the full generation, apply the SAME
        finalize_generation logic as the non-stream path, then emit the clean
        reply as a single SSE content chunk + done chunk.

        Used when _stream_should_buffer() is true (default for the
        deliberating seth-lora-v4-repair model). The SSE response headers have
        already been sent by _handle_stream; this method owns the body.

        Like _handle_non_stream, grant thinking headroom: the model deliberates
        ~150-200 tokens before the visible reply, so capping at the caller's
        max_tokens starves the reply to empty. Headroom is a CAP (well-behaved
        completions stop early) — see _thinking_headroom_tokens().
        """
        internal_max = max_tokens + _thinking_headroom_tokens()

        raw_parts = []
        prompt_toks = 0
        gen_toks = 0
        first_token_time = None
        with model_lock:
            for text, pt, gt in stream_response(messages, internal_max, temperature):
                prompt_toks = pt
                gen_toks = gt
                if text and first_token_time is None:
                    first_token_time = time.time()
                raw_parts.append(text)

        full = "".join(raw_parts).strip()
        clean, is_runaway = finalize_generation(full, messages)

        if clean:
            content_chunk = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": {"content": clean},
                    "finish_reason": None,
                }],
            }
            self.wfile.write(f"data: {json.dumps(content_chunk)}\n\n".encode())
            self.wfile.flush()

        done_chunk = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_toks,
                "completion_tokens": gen_toks,
                "total_tokens": prompt_toks + gen_toks,
            },
        }
        self.wfile.write(f"data: {json.dumps(done_chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

        elapsed = time.time() - t0
        ttft = (first_token_time - t0) if first_token_time else elapsed
        tps = gen_toks / elapsed if elapsed > 0 else 0
        perf_stats["total_tokens"] += gen_toks
        perf_stats["total_time"] += elapsed
        perf_stats["requests"] += 1
        cache_tag = f" [TQ{kv_bits}b]" if kv_bits is not None else ""
        spec_tag = " [spec]" if speculative_enabled else ""
        if is_runaway:
            tag = " [buffered->runaway-empty]"
        else:
            tag = " [buffered]"
        preview = clean[:60].replace("\n", " ")
        print(f"  -> {gen_toks} tokens in {elapsed:.1f}s ({tps:.1f} tok/s, "
              f"TTFT {ttft:.2f}s){cache_tag}{spec_tag}{tag} | {preview}...",
              flush=True)

    def _handle_non_stream(self, req, resp_id, t0):
        messages = req.get("messages", [])
        max_tokens = req.get("max_tokens", 256)
        temperature = req.get("temperature", 0.7)
        # Gemma 4 (and especially the seth-lora-v4-repair adapter) emits
        # ~150-200 tokens of chain-of-thought deliberation BEFORE the visible
        # reply. The adapter was trained to deliberate and largely IGNORES the
        # GEMMA_DISABLE_THINKING no-think instruction, so we cannot assume the
        # thinking block is absent in no-think mode. If we cap generation at the
        # caller's max_tokens, the thought channel consumes the entire budget
        # and the visible reply is starved to empty — the strip extractor then
        # salvages a garbage fragment ('nights', 'watch'). This was the live
        # M3 production bug (see m3_live_path_extractor_strip.md): with
        # max_tokens=80 every casual prompt degenerated; with ~512 the same
        # prompts returned coherent in-voice replies. So ALWAYS grant thinking
        # headroom. The thought channel is stripped post-generation, so the
        # caller still receives only the short reply they asked for.
        #
        # Headroom is a CAP, not a target: a completed reply emits its stop
        # token and finishes naturally well below the cap, so a generous
        # headroom is nearly free for well-behaved completions and only costs
        # latency on runaway (never-stopping) generations. See
        # _thinking_headroom_tokens() for the full rationale + GEMMA_THINKING_
        # HEADROOM_TOKENS tunable.
        internal_max = max_tokens + _thinking_headroom_tokens()

        # Make the SAME primer decision the streaming path makes (line ~1857)
        # rather than leaning on prepare_prompt_lm's bare default: casual tiers
        # (stream_strip=false) and no-think config drop Gemma's forced
        # `<|channel>thought` opener. Under no-think this is already the default
        # on both paths (no-op); its value is a no-think-OFF server honoring a
        # casual caller's stream_strip=false. Thinking-enabled callers unchanged.
        skip_primer = _request_skip_thinking_primer(req)
        with model_lock:
            text, prompt_toks, gen_toks = generate_response(
                messages, internal_max, temperature,
                skip_thinking_primer=skip_primer,
            )

        elapsed = time.time() - t0
        self._send_json(200, {
            "id": resp_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_toks,
                "completion_tokens": gen_toks,
                "total_tokens": prompt_toks + gen_toks,
            },
        })

        preview = text[:60].replace("\n", " ")
        tps = gen_toks / elapsed if elapsed > 0 else 0
        perf_stats["total_tokens"] += gen_toks
        perf_stats["total_time"] += elapsed
        perf_stats["requests"] += 1
        cache_tag = f" [TQ{kv_bits}b]" if kv_bits is not None else ""
        print(f"  -> {gen_toks} tokens in {elapsed:.1f}s ({tps:.1f} tok/s){cache_tag} | {preview}...", flush=True)

    def _handle_adapter_swap(self):
        """POST /v1/adapters/swap — hot-swap LoRA adapter weights on the live model.

        Body: {"adapter_path": "/abs/path/to/dir-containing-adapters.safetensors"}
        Serializes against /v1/chat/completions via model_lock. On failure mid-swap,
        attempts to revert to the previously-active adapter.
        """
        global adapter_path_global, tensors_loaded_global

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            req = json.loads(body.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "invalid JSON"})
            return

        adapter_path_raw = req.get("adapter_path") if isinstance(req, dict) else None
        if not isinstance(adapter_path_raw, str) or not adapter_path_raw.strip():
            self._send_json(400, {"error": "adapter_path must be a non-empty string"})
            return

        from pathlib import Path
        resolved = Path(os.path.expanduser(adapter_path_raw)).resolve()
        if not resolved.is_dir():
            self._send_json(404, {"error": "adapter directory not found", "path": str(resolved)})
            return
        adapter_file = resolved / "adapters.safetensors"
        if not adapter_file.is_file():
            self._send_json(404, {
                "error": "adapters.safetensors missing in directory",
                "path": str(adapter_file),
            })
            return

        old_adapter = adapter_path_global
        new_adapter = str(resolved)
        # Serialize against inference. Block; adapter swap is rare.
        # The finally below runs on every path (success, revert, revert-
        # failure): once _apply_adapter_weights has been attempted, cached
        # KV can no longer be trusted to match the live weights.
        with model_lock:
            try:
                n_tensors = _apply_adapter_weights(new_adapter)
                adapter_path_global = new_adapter
                print(f"[swap] from {old_adapter} to {new_adapter}: applied {n_tensors} tensors", flush=True)
                self._send_json(200, {
                    "status": "ok",
                    "adapter_path": new_adapter,
                    "tensors_loaded": n_tensors,
                })
                return
            except Exception as e:
                err = str(e)
                print(f"[swap] FAILED applying {new_adapter}: {err}", flush=True)
                # Model may be in a partially-applied state. Try to revert.
                if old_adapter is not None:
                    try:
                        n_reverted = _apply_adapter_weights(old_adapter)
                        adapter_path_global = old_adapter
                        print(f"[swap] reverted to {old_adapter} ({n_reverted} tensors)", flush=True)
                    except Exception as revert_err:
                        print(f"[swap] REVERT FAILED: {revert_err}", flush=True)
                else:
                    # No prior adapter to revert to. Clear the marker and tensor count;
                    # the underlying base weights remain whatever load_weights left them as.
                    adapter_path_global = None
                    tensors_loaded_global = 0
                self._send_json(500, {
                    "error": "adapter load failed",
                    "message": err,
                    "reverted_to": old_adapter,
                })
                return
            finally:
                _invalidate_cross_turn_caches("adapter swap")

    def do_POST(self):
        if self.path == "/v1/adapters/swap":
            self._handle_adapter_swap()
            return

        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            req = json.loads(body.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "invalid JSON"})
            return

        t0 = time.time()
        resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        # Activation steering: arm per-request trait coefficients from the
        # optional "steering" field, then clear after. The server is
        # single-threaded (HTTPServer), so requests are fully serialized — no
        # cross-request leakage. No "steering" field => set_active({}) => the
        # layer hook is a strict no-op (byte-identical to unsteered).
        ps.set_active(req.get("steering"))
        try:
            if req.get("stream", False):
                self._handle_stream(req, resp_id, t0)
            else:
                self._handle_non_stream(req, resp_id, t0)
        finally:
            ps.clear_active()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def main():
    global kv_bits, kv_quant_scheme, prompt_cache_state, hw_info, speculative_draft_tokens
    global server_realtime_mode, lcp_mode, lm_cache_slot_count

    hc = _load_human_config()
    _apply_mlx_local_env_from_hc(hc)
    _spec_tokens_default = _speculative_tokens_default(hc)

    parser = argparse.ArgumentParser(
        description="MLX OpenAI-compatible model server with speculative decoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard serving (reads ~/.human/config.json for defaults)
  %(prog)s

  # Real-time voice mode (E4B + aggressive KV compression)
  %(prog)s --model mlx-community/gemma-4-e4b-it-4bit --realtime

  # Speculative decoding (E4B target + E2B draft = ~2x speedup)
  %(prog)s --model mlx-community/gemma-4-e4b-it-4bit \\
    --speculative-draft mlx-community/gemma-4-e2b-it-4bit

  # Fine-tuned with LoRA adapters on both target and draft
  %(prog)s --model mlx-community/gemma-4-e4b-it-4bit \\
    --adapter-path ~/.human/adapters/persona \\
    --speculative-draft mlx-community/gemma-4-e2b-it-4bit \\
    --speculative-draft-adapter ~/.human/adapters/draft
""",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MLX_MODEL", hc.get("model", DEFAULT_MODEL)),
    )
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("MLX_PORT", hc.get("port", DEFAULT_PORT))),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--adapter-path",
        default=os.environ.get("MLX_ADAPTER_PATH", hc.get("adapter_path")),
        help="Path to LoRA adapter directory (e.g. from finetune-gemma.py).",
    )
    parser.add_argument(
        "--kv-bits", type=float, default=hc.get("kv_bits"),
        help="KV cache quantization bits. Use 3 for TurboQuant 3-bit (4.6x compression).",
    )
    parser.add_argument(
        "--kv-asymmetric", action="store_true",
        default=hc.get("kv_asymmetric", False),
        help="Asymmetric KV: keep keys at FP16, compress only values (recommended for Q4_K_M models).",
    )
    parser.add_argument(
        "--no-prompt-cache", action="store_true",
        help="Disable cross-turn prompt cache reuse.",
    )
    parser.add_argument(
        "--prompt-cache-lcp",
        default=os.environ.get("MLX_PROMPT_CACHE_LCP",
                               hc.get("prompt_cache_lcp", "shadow")),
        help="Token-LCP cross-turn cache reuse: off (legacy hash only), "
             "shadow (legacy + log what LCP would reuse; default), live "
             "(trim to common prefix, prefill only the suffix).",
    )
    parser.add_argument(
        "--prompt-cache-slots", type=int,
        default=os.environ.get("MLX_PROMPT_CACHE_SLOTS",
                               hc.get("prompt_cache_slots", 2)),
        help="LCP cache slot-pool size (1-4, default 2). Each slot pins real "
             "KV memory once prefilled, so keep this small; 2 covers the "
             "big-daemon-plus-small-probes traffic on one port.",
    )
    parser.add_argument(
        "--speculative-draft", default=hc.get("speculative_draft"),
        help="Draft model for speculative decoding (e.g. mlx-community/gemma-4-e2b-it-4bit "
             "or a path to a LoRA adapter dir with adapters.safetensors).",
    )
    parser.add_argument(
        "--speculative-draft-adapter", default=hc.get("speculative_draft_adapter"),
        help="LoRA adapter path for the draft model.",
    )
    parser.add_argument(
        "--speculative-tokens", type=int, default=_spec_tokens_default,
        help="Draft tokens per speculative step (default: 4, or MLX_SPECULATIVE_TOKENS / GEMMA_SPECULATIVE_TOKENS).",
    )
    parser.add_argument(
        "--realtime", action="store_true",
        default=hc.get("realtime", False),
        help="Real-time voice mode: auto-enable TurboQuant 4-bit KV, aggressive caching, "
             "and optimized generation for lowest TTFT.",
    )
    args = parser.parse_args()
    if os.environ.get("GEMMA_KV_ASYMMETRIC", "").strip().lower() in ("1", "true", "yes"):
        args.kv_asymmetric = True
    if hc.get("prompt_cache") is False:
        args.no_prompt_cache = True
    server_realtime_mode = bool(args.realtime)
    speculative_draft_tokens = args.speculative_tokens

    print(f"\n{'='*60}", flush=True)
    print(f"  MLX Inference Server", flush=True)
    print(f"{'='*60}", flush=True)
    if hc:
        print(f"  Config:  {HUMAN_CONFIG}", flush=True)

    hw_info = detect_apple_silicon()
    print(f"  Hardware: {hw_info['chip']}", flush=True)
    print(f"  Memory:   {hw_info['unified_memory_gb']} GB unified", flush=True)
    if hw_info["has_tensor_ops"]:
        print(f"  TensorOps: ENABLED (M5 Neural Accelerators in GPU cores)", flush=True)
    if hw_info["has_neural_accelerators"]:
        print(f"  Neural Accelerators: ENABLED (per-GPU-core matrix multiply)", flush=True)
    print(f"{'='*60}\n", flush=True)

    if args.realtime:
        if args.kv_bits is None:
            args.kv_bits = 4.0
        print("Real-time voice mode enabled:", flush=True)
        print(f"  - TurboQuant+ {int(args.kv_bits)}-bit KV cache (3.8x compression, +0.23% PPL)", flush=True)
        print("  - Optimized for lowest TTFT + best quality/compression tradeoff", flush=True)
        print("", flush=True)

    if args.kv_bits is not None:
        kv_bits = args.kv_bits
        kv_quant_scheme = "turboquant"
        if getattr(args, "kv_asymmetric", False):
            kv_quant_scheme = "asymmetric"
        try:
            from mlx.nn.layers.turbo_kv_cache import TurboKVCache
            bits = int(kv_bits) if kv_bits == int(kv_bits) else 4
            compression = {2: "6.4x", 3: "4.6x", 4: "3.8x"}.get(bits, f"{bits}b")
            mode = "asymmetric (K=FP16, V=turbo)" if kv_quant_scheme == "asymmetric" else "symmetric"
            print(f"TurboQuant+ detected: {bits}-bit KV cache ({compression} compression, {mode})", flush=True)
        except ImportError:
            kv_quant_scheme = "uniform"
            print(f"KV quantization: {kv_bits}-bit (TurboQuant+ not installed)", flush=True)
            print(f"  Install: pip install git+https://github.com/TheTom/mlx.git@feature/turboquant-plus", flush=True)

    if not args.no_prompt_cache:
        try:
            from mlx_vlm.generate import PromptCacheState
            prompt_cache_state = PromptCacheState()
            print("Prompt cache: cross-turn system prompt caching enabled (trim-and-reuse)", flush=True)
        except ImportError:
            print("Prompt cache: not available (mlx_vlm.generate.PromptCacheState missing)", flush=True)

    from prompt_cache_lcp import parse_mode as _lcp_parse_mode, parse_slots as _lcp_parse_slots
    lcp_mode = "off" if args.no_prompt_cache else _lcp_parse_mode(args.prompt_cache_lcp)
    lm_cache_slot_count = _lcp_parse_slots(args.prompt_cache_slots)
    print(f"Prompt cache LCP mode: {lcp_mode}"
          + (f" ({lm_cache_slot_count} slots, token-prefix reuse ACTIVE)" if lcp_mode == "live" else
             f" ({lm_cache_slot_count} slots simulated, measurement only)" if lcp_mode == "shadow" else ""),
          flush=True)

    load_model(args.model, adapter_path=args.adapter_path)

    # Phase 1: install activation steering if trait vectors are present. Probe-
    # gated (declines on shape/version mismatch) and default-off — no behavior
    # change until a request carries a "steering" field. Never breaks generation.
    try:
        # Pass the base model id: vectors are only valid for the architecture
        # they were extracted from, so they live in a per-model subdirectory.
        _steer_vecs = ps.load_vectors(
            os.path.expanduser("~/.human/persona_vectors"), model_id=args.model)
        if use_lm_path and _steer_vecs:
            ps.install_steering(ps.get_layers(model), _steer_vecs,
                                model_id=args.model)
        elif _steer_vecs and not use_lm_path:
            print("[steering] vectors present but vlm path active; steering off", flush=True)
    except Exception as _steer_ex:  # noqa: BLE001
        print(f"[steering] install skipped ({_steer_ex}); running unsteered", flush=True)

    _init_iosurface_kv_staging()

    if kv_bits is not None:
        _init_turbo_cache()

    if turbo_cache is None and use_lm_path and not args.no_prompt_cache:
        _init_lm_prompt_cache()

    if args.speculative_draft:
        draft_name = args.speculative_draft
        from pathlib import Path
        if Path(draft_name).is_dir() and (Path(draft_name) / "adapters.safetensors").exists():
            load_draft_model("mlx-community/gemma-4-e2b-it-4bit", draft_adapter_path=draft_name)
        else:
            load_draft_model(draft_name, draft_adapter_path=args.speculative_draft_adapter)

    class MLXHTTPServer(HTTPServer):
        allow_reuse_address = True
        allow_reuse_port = True

    server = MLXHTTPServer((args.host, args.port), ChatHandler)
    tq_label = "TurboQuant+" if turbo_cache is not None else "quantized"
    kv_info = f", KV={kv_bits}b {tq_label}" if kv_bits else ""
    cache_info = ", prompt-cache=on" if (turbo_cache is not None or prompt_cache_state) else ""
    adapter_info = f", adapter={args.adapter_path}" if args.adapter_path else ""
    spec_info = f", speculative={args.speculative_draft}" if args.speculative_draft else ""
    engine_tag = "mlx_lm" if use_lm_path else "mlx_vlm"
    print(f"\nServing on http://{args.host}:{args.port}/v1/chat/completions")
    print(f"Model: {args.model} ({model_id}{kv_info}{cache_info}{adapter_info}{spec_info})")
    print(f"Engine: {engine_tag} {'(fast text)' if use_lm_path else '(multimodal)'}")
    print(f"Health: http://{args.host}:{args.port}/health\n", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
