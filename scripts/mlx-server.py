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
    - Prompt cache state tracking (future: cross-turn KV reuse)
    - Apple Silicon hardware detection (M5 TensorOps, Neural Accelerators, Metal version)
    - Real-time voice mode: optimized for low TTFT with aggressive KV compression
    - PLE-safe model validation for Gemma 4 (warns about broken quantizations)

The server exposes:
    POST /v1/chat/completions  — OpenAI-compatible chat endpoint (supports stream:true)
    GET  /v1/models            — list available models
    GET  /health               — health check (includes hardware info + tok/s stats)
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
        return defaults
    except Exception:
        return {}

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

prompt_cache_state = None

STOP_STRINGS = ("<end_of_turn>", "<eos>")

adapter_path_global = None
hw_info = {}
perf_stats = {"total_tokens": 0, "total_time": 0.0, "requests": 0}

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
    import mlx.core as mx
    from pathlib import Path

    model, tokenizer = load_fn(model_name)
    adapter_file = Path(adapter_path) / "adapters.safetensors"
    if adapter_file.exists():
        adapters = list(mx.load(str(adapter_file)).items())
        model.load_weights(adapters, strict=False)
        print(f"  Applied {len(adapters)} LoRA weight tensors from {adapter_file}", flush=True)
    return model, tokenizer


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


def prepare_prompt_lm(messages):
    """Format messages using mlx_lm's native chat template (fast text path)."""
    if hasattr(processor, "apply_chat_template"):
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
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


def _kv_kwargs():
    """Build KV cache kwargs for generate_step (works for both mlx_lm and mlx_vlm)."""
    extra = {}
    if turbo_cache is not None:
        extra["prompt_cache"] = turbo_cache
    elif kv_bits is not None:
        extra["kv_bits"] = int(kv_bits)
    return extra


def generate_response(messages, max_tokens=256, temperature=0.7):
    """Non-streaming: generate the full response at once."""
    has_imgs = _has_images(messages)

    if use_lm_path and not has_imgs:
        from mlx_lm import stream_generate as lm_stream_generate
        from mlx_lm.sample_utils import make_sampler
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
                prefill_done = True
            prompt_toks_out = getattr(resp, "prompt_tokens", prompt_toks_out)
            gen_toks_out = getattr(resp, "generation_tokens", gen_toks_out)
            t = resp.text or ""
            cleaned, hit_stop = strip_stop_tokens(t)
            text_parts.append(cleaned)
            if hit_stop:
                break
        return "".join(text_parts).strip(), prompt_toks_out, gen_toks_out

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
    return text.strip(), prompt_toks, gen_toks


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

    # Fast text path via mlx_lm
    if use_lm_path and not has_imgs:
        from mlx_lm import stream_generate as lm_stream_generate
        from mlx_lm.sample_utils import make_sampler

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
            if prompt_cache_state is not None and prompt_cache_state.token_ids is not None:
                health["cached_tokens"] = len(prompt_cache_state.token_ids)
            if adapter_path_global:
                health["adapter"] = adapter_path_global
            if speculative_enabled:
                health["speculative_decoding"] = True
                health["draft_tokens"] = speculative_draft_tokens
            if hw_info:
                health["hardware"] = hw_info
            if perf_stats["requests"] > 0:
                avg_tps = perf_stats["total_tokens"] / perf_stats["total_time"] if perf_stats["total_time"] > 0 else 0
                health["avg_tok_per_sec"] = round(avg_tps, 1)
                health["total_requests"] = perf_stats["requests"]
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

        with model_lock:
            for text, pt, gt in stream_response(messages, max_tokens, temperature):
                if first_token_time is None:
                    first_token_time = time.time()
                prompt_toks = pt
                gen_toks = gt
                full_text.append(text)

                chunk = {
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None,
                    }],
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
        if prompt_cache_state is not None and prompt_cache_state.token_ids is not None:
            reused = f" [cache:{len(prompt_cache_state.token_ids)} toks]"

        perf_stats["total_tokens"] += gen_toks
        perf_stats["total_time"] += elapsed
        perf_stats["requests"] += 1

        print(f"  -> {gen_toks} tokens in {elapsed:.1f}s ({tps:.1f} tok/s, TTFT {ttft:.2f}s){cache_tag}{spec_tag}{reused} | {preview}...", flush=True)

    def _handle_non_stream(self, req, resp_id, t0):
        messages = req.get("messages", [])
        max_tokens = req.get("max_tokens", 256)
        temperature = req.get("temperature", 0.7)

        with model_lock:
            text, prompt_toks, gen_toks = generate_response(messages, max_tokens, temperature)

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

    def do_POST(self):
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


def main():
    global kv_bits, kv_quant_scheme, prompt_cache_state, hw_info, speculative_draft_tokens

    hc = _load_human_config()

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
        "--speculative-tokens", type=int, default=4,
        help="Number of draft tokens to propose per step (default: 4).",
    )
    parser.add_argument(
        "--realtime", action="store_true",
        default=hc.get("realtime", False),
        help="Real-time voice mode: auto-enable TurboQuant 4-bit KV, aggressive caching, "
             "and optimized generation for lowest TTFT.",
    )
    args = parser.parse_args()

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
            print("Prompt cache: state tracking enabled (cross-turn reuse not yet wired)", flush=True)
        except ImportError:
            print("Prompt cache: not available (mlx_vlm.generate.PromptCacheState missing)", flush=True)

    load_model(args.model, adapter_path=args.adapter_path)

    if kv_bits is not None:
        _init_turbo_cache()

    if args.speculative_draft:
        speculative_draft_tokens = args.speculative_tokens
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
    cache_info = ", prompt-cache=tracking" if prompt_cache_state else ""
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
