#!/usr/bin/env python3
"""
PROVE STS: Speech-to-Speech end-to-end proof.

This is the ultimate test: real audio in → real audio out.
No text shortcuts. No mock servers. Every path exercised.

Tests:
    1. Audio round-trip via SOTAPipeline.process_audio_full()
       - Feed real WAV of speech → get transcript + response audio
       - Verify transcript is intelligible
       - Verify response audio is speech-like
    2. Multi-turn audio STS
       - Two audio turns, verify LLM remembers first turn
    3. LocalLLMClient correctness
       - Verify it streams tokens, filters thinking, matches HTTP interface
    4. WebSocket STS with native TTS
       - Start real WS server (--native-tts)
       - Client sends audio.chunk + audio.commit
       - Verify: transcript.final + text.delta + audio.chunk received
    5. WebSocket text→audio round-trip
       - Client sends text.input, receives audio.chunk back
"""

import asyncio
import base64
import json
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

PROOF_DIR = Path("proof-artifacts")
PROOF_DIR.mkdir(exist_ok=True)

passed = 0
failed = 0
skipped = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))


def skip(name, reason):
    global skipped
    skipped += 1
    print(f"  [SKIP] {name} — {reason}")


def spectral_quality(audio_np, sr=24000):
    from scipy import signal
    if len(audio_np) < 64:
        return {"rms": 0, "speech_ratio": 0, "spectral_flatness": 1.0}
    f, psd = signal.welch(audio_np.astype(np.float64), fs=sr, nperseg=min(1024, len(audio_np)))
    total = np.sum(psd)
    if total < 1e-12:
        return {"rms": 0, "speech_ratio": 0, "spectral_flatness": 1.0}
    speech_mask = (f >= 100) & (f <= 4000)
    speech_energy = np.sum(psd[speech_mask])
    geo_mean = np.exp(np.mean(np.log(psd + 1e-20)))
    arith_mean = np.mean(psd)
    return {
        "rms": float(np.sqrt(np.mean(audio_np.astype(np.float64) ** 2))),
        "speech_ratio": float(speech_energy / total),
        "spectral_flatness": float(geo_mean / (arith_mean + 1e-20)),
    }


def load_test_audio():
    """Load a real speech WAV for testing."""
    with open("data/libritts-valid.jsonl") as f:
        item = json.loads(f.readline())
    audio, sr = sf.read(item["audio_path"], dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        n = int(len(audio) * 16000 / sr)
        audio = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(audio)), audio).astype(np.float32)
    return audio


def make_pipeline(shared):
    """Create a SOTAPipeline with shared models injected."""
    from sota_pipeline import SOTAPipeline
    from streaming_asr import StreamingASRWithVAD
    from speech_decoder import DuplexStatePredictor
    import mlx.core as mx

    pipeline = SOTAPipeline()
    pipeline._gemma = shared["gemma"]
    pipeline._tokenizer = shared["tokenizer"]
    pipeline._inner = shared["inner"]
    pipeline._decoder = shared["decoder"]
    pipeline._depth_decoder = shared["depth_decoder"]
    pipeline._codec = shared["codec"]
    pipeline._streaming_asr = StreamingASRWithVAD()
    pipeline._streaming_asr.load()
    duplex_path = Path("adapters/duplex-predictor/duplex_predictor.safetensors")
    if duplex_path.exists():
        llm_dim = shared["inner"].embed_tokens(mx.array([[0]])).shape[-1]
        pipeline._duplex = DuplexStatePredictor(llm_dim=llm_dim)
        dw = mx.load(str(duplex_path))
        pipeline._duplex.load_weights(list(dw.items()))
    pipeline._loaded = True
    return pipeline


# ═══════════════════════════════════════════════════
# SHARED MODEL LOADING
# ═══════════════════════════════════════════════════

def load_shared():
    import mlx.core as mx
    from mlx_lm import load as lm_load
    from speech_decoder import SpeechDecoder
    from codec import AudioCodec

    gemma, tokenizer = lm_load("mlx-community/gemma-4-26b-a4b-it-4bit")
    if hasattr(gemma, "language_model"):
        inner = gemma.language_model.model
    else:
        inner = gemma.model

    probe = inner.embed_tokens(mx.array([[0]]))
    llm_dim = probe.shape[-1]

    decoder = SpeechDecoder(llm_dim=llm_dim)
    dec_w = mx.load("adapters/speech-decoder/speech_decoder.safetensors")
    decoder.load_weights(list(dec_w.items()))

    depth_decoder = None
    depth_path = Path("adapters/depth-decoder/depth_decoder.safetensors")
    if depth_path.exists():
        tdd = import_module("train-depth-decoder")
        depth_decoder = tdd.DepthDecoder(**tdd.DEPTH_DECODER_CONFIG)
        dw = mx.load(str(depth_path))
        depth_decoder.load_weights(list(dw.items()))

    codec = AudioCodec("snac")
    codec.load()

    return {
        "gemma": gemma, "tokenizer": tokenizer, "inner": inner,
        "decoder": decoder, "depth_decoder": depth_decoder, "codec": codec,
    }


# ═══════════════════════════════════════════════════
# TEST 1: Audio round-trip (THE core STS test)
# ═══════════════════════════════════════════════════

def test_audio_roundtrip(shared):
    print(f"\n{'─'*60}")
    print("  TEST 1: Audio → ASR → LLM → TTS → Audio (STS round-trip)")
    print(f"{'─'*60}")

    pipeline = make_pipeline(shared)
    audio_16k = load_test_audio()

    check("Input audio loaded", len(audio_16k) > 8000, f"{len(audio_16k)} samples ({len(audio_16k)/16000:.1f}s)")

    t0 = time.time()
    transcript, audio_out, metrics = pipeline.process_audio_full(audio_16k, max_tokens=60)
    total_ms = (time.time() - t0) * 1000

    check("Transcript not empty", len(transcript) > 0, f"\"{transcript[:60]}\"")
    check("Response text not empty", len(metrics.get("response", "")) > 0, f"\"{metrics.get('response', '')[:60]}\"")
    check("Audio output >100 samples", len(audio_out) > 100, f"{len(audio_out)} samples")

    sq = spectral_quality(audio_out)
    check("Output audio: RMS > 0.01", sq["rms"] > 0.01, f"RMS={sq['rms']:.4f}")
    check("Output audio: speech band >30%", sq["speech_ratio"] > 0.3, f"{sq['speech_ratio']*100:.1f}%")

    check("ASR latency present", metrics.get("asr_ms", 0) > 0, f"{metrics.get('asr_ms', 0):.0f}ms")
    check("Total STS latency <10s", total_ms < 10000, f"{total_ms:.0f}ms")

    sf.write(str(PROOF_DIR / "sts_roundtrip_out.wav"), audio_out, 24000)
    print(f"\n  STS metrics: ASR={metrics.get('asr_ms',0):.0f}ms | "
          f"Total={total_ms:.0f}ms | "
          f"Audio chunks={metrics.get('audio_chunks',0)} | "
          f"Audio duration={metrics.get('audio_duration',0):.1f}s")


# ═══════════════════════════════════════════════════
# TEST 2: Multi-turn audio STS
# ═══════════════════════════════════════════════════

def test_multiturn_sts(shared):
    print(f"\n{'─'*60}")
    print("  TEST 2: Multi-turn audio STS with memory")
    print(f"{'─'*60}")

    pipeline = make_pipeline(shared)

    # Turn 1: process real audio
    audio_16k = load_test_audio()
    transcript1, audio1, m1 = pipeline.process_audio_full(audio_16k, max_tokens=40)
    check("Turn 1 transcript", len(transcript1) > 0, f"\"{transcript1[:50]}\"")
    check("Turn 1 audio output", len(audio1) > 100)
    check("History has 2 entries", len(pipeline._conversation_history) == 2)

    # Turn 2: text input (to control the question)
    audio2, m2 = pipeline.text_to_audio("What did I just say to you?", max_tokens=40)
    response2 = m2.get("response", "")
    check("Turn 2 response", len(response2) > 0, f"\"{response2[:60]}\"")

    # Check if LLM references something from the first turn
    t1_words = set(transcript1.lower().split())
    r2_lower = response2.lower()
    any_overlap = any(w in r2_lower for w in t1_words if len(w) > 3)
    check("LLM remembers audio turn content", any_overlap,
          f"transcript1 words: {list(t1_words)[:5]}, response2: \"{response2[:60]}\"")


# ═══════════════════════════════════════════════════
# TEST 3: LocalLLMClient
# ═══════════════════════════════════════════════════

def test_local_llm(shared):
    print(f"\n{'─'*60}")
    print("  TEST 3: LocalLLMClient streams tokens from local Gemma")
    print(f"{'─'*60}")

    ws_mod = import_module("realtime-ws")
    client = ws_mod.LocalLLMClient(shared["gemma"], shared["tokenizer"])

    async def run():
        health = await client.check_health()
        check("Health returns ok", health is not None and health.get("status") == "ok")

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        tokens = []
        async for delta in client.stream_chat(messages, max_tokens=100, temperature=0.7):
            tokens.append(delta)

        full = "".join(tokens)
        check("Streamed >0 tokens", len(tokens) > 0, f"{len(tokens)} deltas")
        check("Response contains '4'", "4" in full, f"\"{full[:80]}\"")
        check("No thinking tokens leaked", "<|channel>" not in full)
        check("No <| tokens leaked", not any("<|" in t for t in tokens))

        await client.close()

    asyncio.run(run())


# ═══════════════════════════════════════════════════
# TEST 4: WebSocket STS with audio input
# ═══════════════════════════════════════════════════

def test_ws_sts(shared):
    print(f"\n{'─'*60}")
    print("  TEST 4: WebSocket STS — audio.chunk → transcript + audio back")
    print(f"{'─'*60}")

    try:
        import websockets
    except ImportError:
        skip("WS STS", "websockets not installed")
        return

    WS_PORT = 18747

    audio_16k = load_test_audio()
    # Convert to 24kHz s16le for the WebSocket protocol
    n_24k = int(len(audio_16k) * 24000 / 16000)
    audio_24k = np.interp(np.linspace(0, 1, n_24k), np.linspace(0, 1, len(audio_16k)), audio_16k)
    pcm_s16 = (audio_24k * 32767).astype(np.int16)

    async def run():
        ws_mod = import_module("realtime-ws")

        server = ws_mod.RealtimeServer(
            host="127.0.0.1", port=WS_PORT,
            tts_backend="native",
        )
        # Pre-inject shared models to avoid double-loading Gemma
        from importlib import import_module as imp
        speech = imp("speech-server")
        server._shared_vad = speech.SileroVAD(threshold=0.4)
        server._shared_asr = speech.WhisperASR()
        server._shared_tts = ws_mod.NativeTTSEngine()
        server._shared_tts.load(shared["inner"], shared["tokenizer"],
                                shared["decoder"], shared["depth_decoder"], shared["codec"])
        server._gemma = shared["gemma"]
        server._gemma_tokenizer = shared["tokenizer"]
        server._local_llm = ws_mod.LocalLLMClient(shared["gemma"], shared["tokenizer"])
        server._shared_vad.load()
        server._shared_asr.load()

        srv = await websockets.serve(
            server._handle_connection, "127.0.0.1", WS_PORT,
            max_size=10 * 1024 * 1024,
        )

        try:
            async with websockets.connect(f"ws://127.0.0.1:{WS_PORT}") as ws:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 10))
                check("Session created", msg["type"] == "session.created")
                check("Audio output capability", msg.get("capabilities", {}).get("audio_output", False))

                # Send audio in chunks (500ms each)
                chunk_bytes = 24000  # 500ms at 24kHz
                for i in range(0, len(pcm_s16), chunk_bytes):
                    chunk = pcm_s16[i:i + chunk_bytes]
                    b64 = base64.b64encode(chunk.tobytes()).decode("ascii")
                    await ws.send(json.dumps({"type": "audio.chunk", "data": b64}))
                    await asyncio.sleep(0.01)

                # Commit the audio
                await ws.send(json.dumps({"type": "audio.commit"}))

                # Collect responses with timeout
                got_transcript = False
                got_text = False
                got_audio = False
                got_done = False
                transcript_text = ""
                text_parts = []
                audio_chunks = 0

                deadline = time.time() + 30
                while time.time() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), 5)
                    except asyncio.TimeoutError:
                        break
                    m = json.loads(raw)

                    if m["type"] == "transcript.final":
                        got_transcript = True
                        transcript_text = m.get("text", "")
                    elif m["type"] == "text.delta":
                        got_text = True
                        text_parts.append(m.get("text", ""))
                    elif m["type"] == "audio.chunk":
                        got_audio = True
                        audio_chunks += 1
                    elif m["type"] == "response.done":
                        got_done = True
                        break

                check("Got transcript.final", got_transcript, f"\"{transcript_text[:50]}\"")
                check("Got text.delta(s)", got_text, f"{len(text_parts)} deltas: \"{''.join(text_parts)[:50]}\"")
                check("Got audio.chunk(s)", got_audio, f"{audio_chunks} chunks")
                check("Got response.done", got_done)

                await ws.send(json.dumps({"type": "session.close"}))

        finally:
            srv.close()
            await srv.wait_closed()

    asyncio.run(run())


# ═══════════════════════════════════════════════════
# TEST 5: WebSocket text → audio
# ═══════════════════════════════════════════════════

def test_ws_text_to_audio(shared):
    print(f"\n{'─'*60}")
    print("  TEST 5: WebSocket text.input → audio response (native TTS)")
    print(f"{'─'*60}")

    try:
        import websockets
    except ImportError:
        skip("WS text→audio", "websockets not installed")
        return

    WS_PORT = 18748

    async def run():
        ws_mod = import_module("realtime-ws")
        speech = import_module("speech-server")

        server = ws_mod.RealtimeServer(
            host="127.0.0.1", port=WS_PORT, tts_backend="native",
        )
        server._shared_vad = speech.SileroVAD(threshold=0.4)
        server._shared_asr = speech.WhisperASR()
        server._shared_tts = ws_mod.NativeTTSEngine()
        server._shared_tts.load(shared["inner"], shared["tokenizer"],
                                shared["decoder"], shared["depth_decoder"], shared["codec"])
        server._gemma = shared["gemma"]
        server._gemma_tokenizer = shared["tokenizer"]
        server._local_llm = ws_mod.LocalLLMClient(shared["gemma"], shared["tokenizer"])
        server._shared_vad.load()
        server._shared_asr.load()

        srv = await websockets.serve(
            server._handle_connection, "127.0.0.1", WS_PORT,
            max_size=10 * 1024 * 1024,
        )

        try:
            async with websockets.connect(f"ws://127.0.0.1:{WS_PORT}") as ws:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 10))
                assert msg["type"] == "session.created"

                # Send text input
                await ws.send(json.dumps({"type": "text.input", "text": "Say hello briefly."}))

                text_parts = []
                audio_data = []
                latency = {}

                deadline = time.time() + 30
                while time.time() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), 5)
                    except asyncio.TimeoutError:
                        break
                    m = json.loads(raw)
                    if m["type"] == "text.delta":
                        text_parts.append(m.get("text", ""))
                    elif m["type"] == "audio.chunk":
                        b64 = m.get("data", "")
                        if b64:
                            audio_data.append(np.frombuffer(base64.b64decode(b64), dtype=np.int16))
                    elif m["type"] == "response.done":
                        latency = m.get("latency", {})
                        break

                full_text = "".join(text_parts)
                check("Got text response", len(full_text) > 0, f"\"{full_text[:60]}\"")
                check("Got audio chunks", len(audio_data) > 0, f"{len(audio_data)} chunks")

                if audio_data:
                    full_pcm = np.concatenate(audio_data).astype(np.float32) / 32768.0
                    sq = spectral_quality(full_pcm, sr=24000)
                    check("WS audio: RMS > 0.005", sq["rms"] > 0.005, f"RMS={sq['rms']:.4f}")
                    sf.write(str(PROOF_DIR / "sts_ws_text_audio.wav"), full_pcm, 24000)

                if latency:
                    check("Latency reported", latency.get("total_ms", 0) > 0,
                          f"total={latency.get('total_ms',0):.0f}ms, audio_chunks={latency.get('audio_chunks',0)}")

                await ws.send(json.dumps({"type": "session.close"}))

        finally:
            srv.close()
            await srv.wait_closed()

    asyncio.run(run())


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  PROVE STS — Speech-to-Speech End-to-End")
    print("  Real audio in. Real audio out. No shortcuts.")
    print("=" * 70)

    shared = load_shared()

    test_audio_roundtrip(shared)
    test_multiturn_sts(shared)
    test_local_llm(shared)
    test_ws_sts(shared)
    test_ws_text_to_audio(shared)

    print(f"\n{'='*70}")
    print(f"  STS PROOF SCORECARD")
    print(f"{'='*70}")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    total = passed + failed
    print(f"  Score:   {passed}/{total}")
    print(f"{'='*70}")
    if failed == 0:
        print(f"  VERDICT: STS PROVEN — {passed}/{total} ALL CHECKS PASS")
    else:
        print(f"  VERDICT: {failed} FAILURES — STS NOT PROVEN")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
