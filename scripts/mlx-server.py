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
import subprocess
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Lock

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
    """Load a model and apply LoRA adapter weights."""
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
    if os.environ.get("GEMMA_DISABLE_THINKING", "").strip().lower() not in ("1", "true", "yes"):
        return None
    return _NO_THINK_INSTRUCTION


def _maybe_inject_no_think_instruction(messages):
    """If thinking is disabled, append the no-think instruction to the system
    message (or insert a new system message if none exists).

    Returns a new messages list — does NOT mutate the input. This preserves
    OpenAI request idempotency and makes the function safe to call from both
    streaming and non-streaming paths.
    """
    instruction = _no_think_instruction()
    if not instruction:
        return messages
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


def prepare_prompt_lm(messages):
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
    """
    messages = _maybe_inject_no_think_instruction(messages)
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if _no_think_instruction() is not None:
        # Template-level suppression of the thought-channel opener.
        # See module docstring + `_maybe_inject_no_think_instruction()`.
        template_kwargs["enable_thinking"] = True
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


def _extract_reply_from_body(text):
    """From a fragment that may have parentheticals, label prefixes, or
    surrounding quotes, extract the cleanest version of the actual reply."""
    import re
    text = text.strip()
    if not text:
        return text
    # Parenthetical evaluation followed by the reply — take what's after `)`.
    # gemma-4 thinking often emits: `"Yeah!" (Classic, fits constraint).Yeah!`
    if ")" in text:
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
    return text.strip()


def strip_thought_channels(text):
    """Strip Gemma 4 thought blocks from output.

    Handles TWO formats observed empirically (2026-05-25 audit):

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

    def __init__(self, mode: str = "strip"):
        self.buf = ""
        self.mode = mode

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
        return max(len(self.OPEN), len(self.CLOSE))

    def _tail_could_be_marker_prefix(self):
        m = self._max_marker_len()
        for n in range(min(len(self.buf), m - 1), 0, -1):
            tail = self.buf[-n:]
            if self.OPEN.startswith(tail) or self.CLOSE.startswith(tail):
                return n
        return 0

    def _feed_strip(self):
        keep = self._tail_could_be_marker_prefix()
        emit_end = len(self.buf) - keep if keep > 0 else len(self.buf)
        if emit_end <= 0:
            return ""
        emit = self.buf[:emit_end]
        self.buf = self.buf[emit_end:]
        emit = emit.replace(self.OPEN, "").replace(self.CLOSE, "")
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
    global lm_prompt_cache, lm_cache_supports_quant
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
        print(f"  Stock mlx_lm prompt cache initialized: {len(lm_prompt_cache)} layers of {cache_type}{tag}", flush=True)
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


def _prepare_cache_for_request(messages):
    """Trim cache back to system prompt boundary for cross-turn reuse.

    If the system prompt hasn't changed, we keep its KV state and only
    re-process the new user/assistant tokens. If it changed (or this is
    the first request), the cache is fully reset.
    """
    global _cached_system_hash, _cached_system_ntokens

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


def generate_response(messages, max_tokens=256, temperature=0.7):
    """Non-streaming: generate the full response at once."""
    has_imgs = _has_images(messages)

    if use_lm_path and not has_imgs:
        from mlx_lm import stream_generate as lm_stream_generate
        from mlx_lm.sample_utils import make_sampler
        _prepare_cache_for_request(messages)
        prompt = prepare_prompt_lm(messages)
        extra = _kv_kwargs()
        extra["sampler"] = make_sampler(temp=temperature)
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
                _record_system_cache_boundary(messages)
                prefill_done = True
            prompt_toks_out = getattr(resp, "prompt_tokens", prompt_toks_out)
            gen_toks_out = getattr(resp, "generation_tokens", gen_toks_out)
            t = resp.text or ""
            cleaned, hit_stop = strip_stop_tokens(t)
            text_parts.append(cleaned)
            if hit_stop:
                break
        full = "".join(text_parts).strip()
        stripped = strip_thought_channels(full)
        # 2026-05-25: when the extractor strips everything (model emitted
        # markdown bullets with no extractable final reply), log the raw
        # output for diagnosis AND fall back to a salvage-from-raw heuristic
        # so the daemon doesn't get empty content and silently fail.
        if not stripped and full:
            import re as _re
            # Salvage: extract the LAST quoted string in the raw output —
            # gemma-4 thinking often emits candidate replies inside quotes.
            quoted = _re.findall(r'"([^"]{1,200})"', full)
            if quoted:
                stripped = quoted[-1]
            else:
                # Last-resort: last non-empty line with surrounding bullets stripped
                for line in reversed(full.split("\n")):
                    cleaned_line = _re.sub(r"^\s*\*\s*", "", line).strip()
                    if cleaned_line:
                        stripped = cleaned_line
                        break
            # Truncate raw for log readability; one-line label per call.
            try:
                print(f"  [strip-fallback] extractor empty; salvaged from raw "
                      f"({len(full)} chars): {stripped[:80]!r}", flush=True)
            except Exception:
                pass
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


def stream_response(messages, max_tokens=256, temperature=0.7):
    """Streaming generator: yield (text_chunk, prompt_toks, gen_toks) per token.

    Fast path: mlx_lm.stream_generate (no vision overhead, no numpy sync)
    Fallback: mlx_vlm.stream_generate (multimodal, slower)

    When speculative decoding is enabled, uses the draft model to propose
    multiple tokens that the target model verifies in parallel.
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
            prompt = prepare_prompt_lm(messages)
            extra = _kv_kwargs()
            extra["sampler"] = make_sampler(temp=temperature)

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

        _prepare_cache_for_request(messages)
        prompt = prepare_prompt_lm(messages)
        extra = _kv_kwargs()
        extra["sampler"] = make_sampler(temp=temperature)

        prefill_done = False
        for resp in lm_stream_generate(
            model, processor, prompt=prompt,
            max_tokens=max_tokens,
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

        full_text = []
        prompt_toks = 0
        gen_toks = 0
        first_token_time = None
        reset_thinking_filter()
        reset_stream_filter()

        debug_stream = os.environ.get("HU_DEBUG_STREAM", "").strip() in ("1", "true", "yes")
        debug_log = []
        with model_lock:
            for text, pt, gt in stream_response(messages, max_tokens, temperature):
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

    def _handle_non_stream(self, req, resp_id, t0):
        messages = req.get("messages", [])
        max_tokens = req.get("max_tokens", 256)
        temperature = req.get("temperature", 0.7)
        # Gemma 4 may use ~200-400 thinking tokens before producing visible output.
        # Budget extra only if the system prompt doesn't explicitly suppress it.
        # GEMMA_DISABLE_THINKING=1 skips the +512 budget entirely (~20-40% faster non-stream calls).
        if os.environ.get("GEMMA_DISABLE_THINKING", "").strip().lower() in ("1", "true", "yes"):
            internal_max = max_tokens
        else:
            internal_max = max_tokens + 512

        with model_lock:
            text, prompt_toks, gen_toks = generate_response(messages, internal_max, temperature)

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

        if req.get("stream", False):
            self._handle_stream(req, resp_id, t0)
        else:
            self._handle_non_stream(req, resp_id, t0)


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
    global server_realtime_mode

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

    load_model(args.model, adapter_path=args.adapter_path)

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
