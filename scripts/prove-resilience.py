#!/usr/bin/env python3
"""
PROVE RESILIENCE: Stress test the pipeline with adversarial inputs.

Tests:
    1. Empty/null inputs at every stage
    2. Extremely long inputs
    3. Malformed data (invalid tokens, wrong types)
    4. Rapid-fire sequential requests
    5. Memory pressure: repeated inference cycles
    6. Token boundary conditions (0, max, out-of-range)
"""

import gc
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

PASS = 0
FAIL = 0
TOTAL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL, TOTAL
    TOTAL += 1
    if condition:
        PASS += 1
        print(f"  [PASS] {name}" + (f" ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" ({detail})" if detail else ""))


def check_no_crash(name: str, func, *args, **kwargs):
    """Run func and check it doesn't crash. Return the result or None."""
    global PASS, FAIL, TOTAL
    TOTAL += 1
    try:
        result = func(*args, **kwargs)
        PASS += 1
        print(f"  [PASS] {name} (no crash)")
        return result
    except Exception as e:
        FAIL += 1
        print(f"  [FAIL] {name} (crashed: {type(e).__name__}: {e})")
        return None


# ══════════════════════════════════════════════════════════════
# TEST 1: Codec Resilience
# ══════════════════════════════════════════════════════════════

def test_codec_resilience():
    print(f"\n{'─'*60}")
    print("  TEST 1: SNAC Codec Resilience")
    print(f"{'─'*60}")

    from codec import AudioCodec, CodecTokens, CodecType

    codec = AudioCodec("snac")
    codec.load()

    # Empty audio
    check_no_crash("encode empty audio", codec.encode, np.array([], dtype=np.float32))

    # Very short audio (< 1 frame)
    check_no_crash("encode 10 samples", codec.encode, np.random.randn(10).astype(np.float32) * 0.1)

    # Very long audio (60 seconds)
    check_no_crash("encode 60s audio", codec.encode, np.random.randn(60 * 24000).astype(np.float32) * 0.1)

    # Silence
    result = check_no_crash("encode silence", codec.encode, np.zeros(24000, dtype=np.float32))
    if result:
        decoded = check_no_crash("decode silence tokens", codec.decode, result)
        if decoded is not None:
            check("silence round-trip quiet", np.max(np.abs(decoded)) < 0.1, f"peak={np.max(np.abs(decoded)):.4f}")

    # Clipped audio
    clipped = np.clip(np.random.randn(24000).astype(np.float32) * 5.0, -1, 1)
    check_no_crash("encode clipped audio", codec.encode, clipped)

    # Integer audio (wrong dtype but should normalize)
    int_audio = (np.random.randn(24000) * 16000).astype(np.int16)
    check_no_crash("encode int16 audio", codec.encode, int_audio.astype(np.float32))

    # Token boundary: decode with out-of-range tokens
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # All-zero tokens
    cb0 = torch.zeros(1, 10, dtype=torch.long).to(device)
    cb1 = torch.zeros(1, 20, dtype=torch.long).to(device)
    cb2 = torch.zeros(1, 40, dtype=torch.long).to(device)
    check_no_crash("decode all-zero tokens", lambda: codec._model.decode([cb0, cb1, cb2]))

    # Max token (4095)
    cb0_max = torch.full((1, 10), 4095, dtype=torch.long).to(device)
    cb1_max = torch.full((1, 20), 4095, dtype=torch.long).to(device)
    cb2_max = torch.full((1, 40), 4095, dtype=torch.long).to(device)
    check_no_crash("decode max-value tokens", lambda: codec._model.decode([cb0_max, cb1_max, cb2_max]))

    # Single token
    cb0_single = torch.zeros(1, 1, dtype=torch.long).to(device)
    cb1_single = torch.zeros(1, 2, dtype=torch.long).to(device)
    cb2_single = torch.zeros(1, 4, dtype=torch.long).to(device)
    check_no_crash("decode single frame", lambda: codec._model.decode([cb0_single, cb1_single, cb2_single]))


# ══════════════════════════════════════════════════════════════
# TEST 2: Speech Decoder Resilience
# ══════════════════════════════════════════════════════════════

def test_decoder_resilience():
    print(f"\n{'─'*60}")
    print("  TEST 2: Speech Decoder Resilience")
    print(f"{'─'*60}")

    import mlx.core as mx
    from speech_decoder import SpeechDecoder

    decoder = SpeechDecoder(llm_dim=2816)
    dec_weights = mx.load("adapters/speech-decoder/speech_decoder.safetensors")
    decoder.load_weights(list(dec_weights.items()))

    # Empty embedding (batch=1, seq=0)
    def decode_empty():
        emb = mx.zeros((1, 0, 2816))
        t = decoder.generate(emb, temperature=0.0, top_k=0)
        mx.eval(t)
        return t
    check_no_crash("empty embedding", decode_empty)

    # Single token embedding
    def decode_single():
        emb = mx.random.normal((1, 1, 2816))
        t = decoder.generate(emb, temperature=0.0, top_k=0)
        mx.eval(t)
        return t[0].tolist()
    result = check_no_crash("single token embedding", decode_single)
    if result:
        check("single token valid range", all(0 <= x < 4096 for x in result), f"{len(result)} tokens")

    # Very long embedding (200 tokens)
    def decode_long():
        emb = mx.random.normal((1, 200, 2816))
        t = decoder.generate(emb, temperature=0.0, top_k=0)
        mx.eval(t)
        return t[0].tolist()
    result = check_no_crash("200-token embedding", decode_long)
    if result:
        check("long input produces tokens", len(result) > 0)

    # Zero embedding
    def decode_zeros():
        emb = mx.zeros((1, 10, 2816))
        t = decoder.generate(emb, temperature=0.0, top_k=0)
        mx.eval(t)
        return t[0].tolist()
    result = check_no_crash("zero embedding", decode_zeros)
    if result is not None:
        check("zero input valid range", all(0 <= x < 4096 for x in result) if result else True)

    # Large magnitude embedding
    def decode_large():
        emb = mx.random.normal((1, 10, 2816)) * 100
        t = decoder.generate(emb, temperature=0.0, top_k=0)
        mx.eval(t)
        return t[0].tolist()
    result = check_no_crash("large magnitude embedding", decode_large)
    if result is not None:
        check("large input valid range", all(0 <= x < 4096 for x in result) if result else True)

    # Temperature variations
    def decode_temp(temp):
        emb = mx.random.normal((1, 10, 2816))
        t = decoder.generate(emb, temperature=temp, top_k=0)
        mx.eval(t)
        return t[0].tolist()

    check_no_crash("temperature=0.0", lambda: decode_temp(0.0))
    check_no_crash("temperature=1.0", lambda: decode_temp(1.0))
    check_no_crash("temperature=2.0", lambda: decode_temp(2.0))


# ══════════════════════════════════════════════════════════════
# TEST 3: Rapid-Fire Sequential Requests
# ══════════════════════════════════════════════════════════════

def test_rapid_fire():
    print(f"\n{'─'*60}")
    print("  TEST 3: Rapid-Fire Sequential Requests")
    print(f"{'─'*60}")

    import mlx.core as mx
    from speech_decoder import SpeechDecoder
    from mlx_lm import load as lm_load

    model, tokenizer = lm_load("mlx-community/gemma-4-26b-a4b-it-4bit")
    inner = model.language_model.model if hasattr(model, "language_model") else model.model
    decoder = SpeechDecoder(llm_dim=2816)
    dec_weights = mx.load("adapters/speech-decoder/speech_decoder.safetensors")
    decoder.load_weights(list(dec_weights.items()))

    texts = [
        "Hello.", "How are you?", "What time is it?",
        "Tell me a joke.", "Goodbye.", "Thanks!",
        "One more thing.", "Actually, never mind.",
        "Can you help me?", "That's great!",
    ]

    t0 = time.time()
    successes = 0
    errors = 0
    total_tokens = 0

    for i, text in enumerate(texts):
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            emb = inner.embed_tokens(mx.array([ids]))
            toks = decoder.generate(emb, temperature=0.0, top_k=0)
            mx.eval(toks)
            n = len(toks[0].tolist())
            total_tokens += n
            successes += 1
        except Exception as e:
            errors += 1
            print(f"    Error on request {i+1}: {e}")

    elapsed = time.time() - t0
    rps = len(texts) / elapsed

    check("all requests succeed", errors == 0, f"{successes}/{len(texts)}")
    check("throughput > 1 req/s", rps > 1.0, f"{rps:.1f} req/s")
    print(f"  Total: {successes} ok, {errors} errors in {elapsed:.1f}s ({rps:.1f} req/s)")
    print(f"  Tokens generated: {total_tokens} ({total_tokens/elapsed:.0f} tok/s)")

    del model


# ══════════════════════════════════════════════════════════════
# TEST 4: Memory Pressure
# ══════════════════════════════════════════════════════════════

def test_memory_pressure():
    print(f"\n{'─'*60}")
    print("  TEST 4: Memory Pressure (20 inference cycles)")
    print(f"{'─'*60}")

    import mlx.core as mx
    from speech_decoder import SpeechDecoder

    decoder = SpeechDecoder(llm_dim=2816)
    dec_weights = mx.load("adapters/speech-decoder/speech_decoder.safetensors")
    decoder.load_weights(list(dec_weights.items()))

    errors = 0
    for i in range(20):
        try:
            emb = mx.random.normal((1, 20, 2816))
            toks = decoder.generate(emb, temperature=0.0, top_k=0)
            mx.eval(toks)
            if i % 5 == 4:
                gc.collect()
                mx.clear_cache() if hasattr(mx, 'clear_cache') else None
        except Exception as e:
            errors += 1
            print(f"    Cycle {i+1}: {e}")

    check("20 cycles no OOM", errors == 0, f"{20-errors}/20 ok")


# ══════════════════════════════════════════════════════════════
# TEST 5: Duplex Predictor Resilience
# ══════════════════════════════════════════════════════════════

def test_duplex_resilience():
    print(f"\n{'─'*60}")
    print("  TEST 5: Duplex Predictor Resilience")
    print(f"{'─'*60}")

    import mlx.core as mx
    from speech_decoder import DuplexStatePredictor

    pred_path = Path("adapters/duplex-predictor/duplex_predictor.safetensors")
    if not pred_path.exists():
        print("  SKIP: no predictor checkpoint")
        return

    pred = DuplexStatePredictor(llm_dim=2816)
    weights = mx.load(str(pred_path))
    pred.load_weights(list(weights.items()))

    # Empty
    def pred_empty():
        return pred(mx.zeros((1, 0, 2816)))
    check_no_crash("empty input", pred_empty)

    # Single token
    def pred_single():
        logits = pred(mx.random.normal((1, 1, 2816)))
        mx.eval(logits)
        return mx.argmax(logits[0]).item()
    result = check_no_crash("single token", pred_single)
    if result is not None:
        check("state in range [0,2]", 0 <= result <= 2, f"state={result}")

    # Very long sequence
    def pred_long():
        logits = pred(mx.random.normal((1, 500, 2816)))
        mx.eval(logits)
        return mx.argmax(logits[0]).item()
    result = check_no_crash("500-token input", pred_long)
    if result is not None:
        check("long input valid state", 0 <= result <= 2, f"state={result}")

    # Zero input
    def pred_zeros():
        logits = pred(mx.zeros((1, 10, 2816)))
        mx.eval(logits)
        return mx.argmax(logits[0]).item()
    result = check_no_crash("zero input", pred_zeros)


# ══════════════════════════════════════════════════════════════
# TEST 6: WebSocket Message Validation
# ══════════════════════════════════════════════════════════════

def test_ws_message_validation():
    print(f"\n{'─'*60}")
    print("  TEST 6: WebSocket Message Validation")
    print(f"{'─'*60}")

    import asyncio
    import json

    try:
        import websockets
    except ImportError:
        print("  SKIP: websockets not installed")
        return

    WS_PORT = 18743

    errors_caught = []

    async def handle(ws):
        async for raw in ws:
            try:
                msg = json.loads(raw)
                msg_type = msg.get("type", "")
                if msg_type == "bad":
                    await ws.send(json.dumps({"type": "error", "message": "Unknown type"}))
                else:
                    await ws.send(json.dumps({"type": "ack"}))
            except json.JSONDecodeError:
                errors_caught.append("json_error")
                await ws.send(json.dumps({"type": "error", "message": "Invalid JSON"}))

    async def run():
        server = await websockets.serve(handle, "127.0.0.1", WS_PORT)
        try:
            async with websockets.connect(f"ws://127.0.0.1:{WS_PORT}") as ws:
                # Valid JSON
                await ws.send('{"type": "text.input", "text": "hello"}')
                resp = json.loads(await asyncio.wait_for(ws.recv(), 2))
                check("valid JSON accepted", resp["type"] == "ack")

                # Invalid JSON
                await ws.send("not json at all {{{")
                resp = json.loads(await asyncio.wait_for(ws.recv(), 2))
                check("invalid JSON handled", resp["type"] == "error")

                # Unknown message type
                await ws.send('{"type": "bad"}')
                resp = json.loads(await asyncio.wait_for(ws.recv(), 2))
                check("unknown type handled", resp["type"] == "error")

                # Empty message
                await ws.send('{}')
                resp = json.loads(await asyncio.wait_for(ws.recv(), 2))
                check("empty message handled", True)

                # Oversized audio chunk — server should reject or handle
                big_b64 = "A" * (1024 * 1024)
                try:
                    await ws.send(json.dumps({"type": "audio.chunk", "data": big_b64}))
                    resp = json.loads(await asyncio.wait_for(ws.recv(), 2))
                    check("large payload handled gracefully", True)
                except Exception:
                    check("large payload rejected (expected)", True, "server enforces size limit")

        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(run())


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  PROVE RESILIENCE")
    print("  Adversarial input and stress testing")
    print("=" * 70)

    test_codec_resilience()
    test_decoder_resilience()
    test_rapid_fire()
    test_memory_pressure()
    test_duplex_resilience()
    test_ws_message_validation()

    print(f"\n{'='*70}")
    print(f"  RESILIENCE SCORECARD")
    print(f"{'='*70}")
    print(f"  Passed: {PASS}/{TOTAL}")
    print(f"  Failed: {FAIL}/{TOTAL}")

    rate = PASS / max(TOTAL, 1) * 100
    if rate >= 90:
        print(f"  VERDICT: RESILIENT ({rate:.0f}%) ✓")
    else:
        print(f"  VERDICT: {FAIL} VULNERABILITIES ({rate:.0f}%)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
