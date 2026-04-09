#!/usr/bin/env python3
"""
WebSocket bidirectional realtime API for gemma-realtime.

Two TTS modes:
    Default:      Kokoro TTS (external, high quality, requires kokoro package)
    --native-tts: SNAC speech decoder + depth decoder (multi-codebook, on-device)

Features:
    - Whisper ASR (mlx-whisper on Apple Silicon)
    - Sentence-level streaming LLM → TTS
    - Full-duplex: client can send {"type": "interrupt"} to stop generation
    - VAD: Silero (if onnxruntime installed) or energy-based fallback

Protocol:
    ws://localhost:8742/v1/realtime

    Client -> Server:
        {"type": "audio.chunk", "data": "<base64 PCM 24kHz s16le>"}
        {"type": "audio.commit"}
        {"type": "text.input", "text": "..."}
        {"type": "config", "voice": "af_bella", "vad_threshold": 0.4, ...}
        {"type": "session.close"}
        {"type": "interrupt"}

    Server -> Client:
        {"type": "session.created", "session_id": "..."}
        {"type": "transcript.final", "text": "..."}
        {"type": "response.start"}
        {"type": "text.delta", "text": "..."}
        {"type": "text.done", "text": "..."}
        {"type": "audio.chunk", "data": "<base64 PCM 24kHz s16le>", "seq": N}
        {"type": "audio.done"}
        {"type": "response.done", "latency": {...}}
        {"type": "state.change", "state": "INTERRUPT"}
        {"type": "error", "message": "..."}

Usage:
    python3 scripts/realtime-ws.py                          # Kokoro TTS
    python3 scripts/realtime-ws.py --native-tts             # SNAC decoder
    python3 scripts/realtime-ws.py --port 8742 --llm-url http://localhost:8741
"""

import argparse
import asyncio
import base64
import json
import re
import time
import uuid

import numpy as np

SAMPLE_RATE = 24000
WHISPER_RATE = 16000
BYTES_PER_SAMPLE = 2
VAD_THRESHOLD = 0.4
SILENCE_TIMEOUT_S = 0.8
SENTENCE_BOUNDARY = re.compile(r"[.!?]\s*$|[.!?][\"']\s*$")
MIN_FLUSH_CHARS = 12
MAX_BUFFER_CHARS = 120


class RealtimeSession:
    """Manages state for a single WebSocket realtime session."""

    def __init__(self, session_id, llm_url, whisper_model, voice, vad_threshold):
        self.session_id = session_id
        self.llm_url = llm_url
        self.whisper_model = whisper_model
        self.voice = voice
        self.vad_threshold = vad_threshold

        self.messages = []
        self.audio_buffer = []
        self.is_recording = False
        self._interrupted = False
        self._asr = None
        self._tts = None
        self._vad = None
        self._llm = None
        self._sentence_buffer = ""

    async def initialize(self):
        """Lazy-load ASR, TTS, VAD, LLM client."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))

        from importlib import import_module
        speech = import_module("speech-server")

        self._vad = speech.SileroVAD(threshold=self.vad_threshold)
        self._asr = speech.WhisperASR(model_name=self.whisper_model)
        self._tts = speech.TTSEngine(voice=self.voice)
        self._llm = speech.LLMClient(base_url=self.llm_url)

        self._vad.load()
        self._asr.load()
        self._tts.load()

        self.messages.append({
            "role": "system",
            "content": (
                "You are a helpful voice assistant. Keep responses concise and conversational. "
                "Respond naturally as if speaking aloud."
            ),
        })

    async def close(self):
        if self._llm:
            await self._llm.close()


class NativeTTSEngine:
    """SNAC-based TTS using the speech decoder + depth decoder pipeline.

    Replaces Kokoro when --native-tts is set.
    """

    def __init__(self):
        self._decoder = None
        self._depth_decoder = None
        self._codec = None
        self._inner = None
        self._tokenizer = None
        self.available = False

    def load(self, inner_model, tokenizer, decoder, depth_decoder, codec):
        self._inner = inner_model
        self._tokenizer = tokenizer
        self._decoder = decoder
        self._depth_decoder = depth_decoder
        self._codec = codec
        self.available = True

    def synthesize(self, text: str) -> np.ndarray:
        import mlx.core as mx
        import torch

        ids = self._tokenizer.encode(text[:120], add_special_tokens=False)
        if not ids:
            return np.zeros(2400, dtype=np.float32)

        emb = self._inner.embed_tokens(mx.array([ids]))
        tokens_mx = self._decoder.generate(emb, temperature=0.0, top_k=0)
        mx.eval(tokens_mx)
        cb0_tokens = tokens_mx[0].tolist()

        if not cb0_tokens:
            return np.zeros(2400, dtype=np.float32)

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        cb0_t = torch.tensor(cb0_tokens, dtype=torch.long).unsqueeze(0).to(device)

        if self._depth_decoder is not None:
            cb0_mx = mx.array([cb0_tokens], dtype=mx.int32)
            cb1_mx, cb2_mx = self._depth_decoder.generate(cb0_mx)
            mx.eval(cb1_mx, cb2_mx)
            cb1_t = torch.tensor(cb1_mx[0].tolist(), dtype=torch.long).unsqueeze(0).to(device)
            cb2_t = torch.tensor(cb2_mx[0].tolist(), dtype=torch.long).unsqueeze(0).to(device)
        else:
            cb1_t = torch.zeros(1, len(cb0_tokens) * 2, dtype=torch.long).to(device)
            cb2_t = torch.zeros(1, len(cb0_tokens) * 4, dtype=torch.long).to(device)

        with torch.no_grad():
            audio = self._codec._model.decode([cb0_t, cb1_t, cb2_t])
        return audio.detach().cpu().numpy().squeeze()


class RealtimeServer:
    """WebSocket server implementing the realtime bidirectional protocol."""

    def __init__(self, host="0.0.0.0", port=8742, llm_url="http://localhost:8741",
                 whisper_model="mlx-community/whisper-small-mlx", voice="af_bella",
                 native_tts=False):
        self.host = host
        self.port = port
        self.llm_url = llm_url
        self.whisper_model = whisper_model
        self.voice = voice
        self.native_tts = native_tts
        self._sessions = {}
        self._shared_vad = None
        self._shared_asr = None
        self._shared_tts = None

    async def _ensure_shared_models(self):
        """Load heavyweight models once, share across all sessions."""
        if self._shared_vad is not None:
            return

        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from importlib import import_module
        speech = import_module("speech-server")

        self._shared_vad = speech.SileroVAD(threshold=VAD_THRESHOLD)
        self._shared_asr = speech.WhisperASR(model_name=self.whisper_model)

        if self.native_tts:
            import mlx.core as mx
            from mlx_lm import load as lm_load
            from speech_decoder import SpeechDecoder
            from codec import AudioCodec

            print("  Loading native SNAC TTS pipeline...", flush=True)
            gemma, tokenizer = lm_load("mlx-community/gemma-4-26b-a4b-it-4bit")
            if hasattr(gemma, "language_model"):
                inner = gemma.language_model.model
            else:
                inner = gemma.model

            probe = inner.embed_tokens(mx.array([[0]]))
            llm_dim = probe.shape[-1]

            decoder = SpeechDecoder(llm_dim=llm_dim)
            dec_weights = mx.load("adapters/speech-decoder/speech_decoder.safetensors")
            decoder.load_weights(list(dec_weights.items()))

            depth_decoder = None
            depth_path = Path("adapters/depth-decoder/depth_decoder.safetensors")
            if depth_path.exists():
                tdd = import_module("train-depth-decoder")
                depth_decoder = tdd.DepthDecoder()
                dw = mx.load(str(depth_path))
                depth_decoder.load_weights(list(dw.items()))
                print("    Depth decoder loaded (3-codebook)", flush=True)

            codec = AudioCodec("snac")
            codec.load()

            self._shared_tts = NativeTTSEngine()
            self._shared_tts.load(inner, tokenizer, decoder, depth_decoder, codec)
            self._gemma = gemma
            print("    Native SNAC TTS ready", flush=True)
        else:
            self._shared_tts = speech.TTSEngine(voice=self.voice)
            self._shared_tts.load()

        print("  Loading shared models (first connection)...", flush=True)
        self._shared_vad.load()
        self._shared_asr.load()

    async def _handle_connection(self, websocket):
        session_id = str(uuid.uuid4())[:12]
        session = RealtimeSession(
            session_id=session_id,
            llm_url=self.llm_url,
            whisper_model=self.whisper_model,
            voice=self.voice,
            vad_threshold=VAD_THRESHOLD,
        )
        self._sessions[session_id] = session

        try:
            await self._ensure_shared_models()
            session._vad = self._shared_vad
            session._asr = self._shared_asr
            session._tts = self._shared_tts

            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent))
            from importlib import import_module
            speech = import_module("speech-server")
            session._llm = speech.LLMClient(base_url=self.llm_url)

            session.messages.append({
                "role": "system",
                "content": (
                    "You are a helpful voice assistant. Keep responses concise and conversational. "
                    "Respond naturally as if speaking aloud."
                ),
            })

            await websocket.send(json.dumps({
                "type": "session.created",
                "session_id": session_id,
                "capabilities": {
                    "audio_input": True,
                    "audio_output": session._tts.available,
                    "text_input": True,
                    "text_output": True,
                    "vad": True,
                },
            }))

            print(f"  [{session_id}] Session started", flush=True)

            async for message in websocket:
                try:
                    msg = json.loads(message)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON",
                    }))
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "audio.chunk":
                    await self._handle_audio_chunk(session, websocket, msg)
                elif msg_type == "audio.commit":
                    await self._handle_audio_commit(session, websocket)
                elif msg_type == "text.input":
                    await self._handle_text_input(session, websocket, msg)
                elif msg_type == "config":
                    self._handle_config(session, msg)
                elif msg_type == "interrupt":
                    session._interrupted = True
                    await websocket.send(json.dumps({
                        "type": "state.change", "state": "INTERRUPT",
                    }))
                elif msg_type == "session.close":
                    break
                else:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                    }))

        except Exception as e:
            error_name = type(e).__name__
            if error_name not in ("ConnectionClosedOK", "ConnectionClosedError", "ConnectionClosed"):
                print(f"  [{session_id}] Error: {e}", flush=True)
        finally:
            await session.close()
            self._sessions.pop(session_id, None)
            print(f"  [{session_id}] Session ended", flush=True)

    async def _handle_audio_chunk(self, session, websocket, msg):
        """Process an incoming audio chunk."""
        b64_data = msg.get("data", "")
        try:
            raw = base64.b64decode(b64_data)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            await websocket.send(json.dumps({
                "type": "error", "message": "Invalid audio data",
            }))
            return

        if len(samples) > 0 and SAMPLE_RATE != WHISPER_RATE:
            n_out = int(len(samples) * WHISPER_RATE / SAMPLE_RATE)
            x_old = np.linspace(0, 1, len(samples))
            x_new = np.linspace(0, 1, n_out)
            samples_16k = np.interp(x_new, x_old, samples).astype(np.float32)
        else:
            samples_16k = samples

        is_speech = session._vad.is_speech(samples_16k)

        if is_speech:
            session.audio_buffer.append(samples_16k)
            session.is_recording = True
        elif session.is_recording:
            session.audio_buffer.append(samples_16k)

    async def _handle_audio_commit(self, session, websocket):
        """Process committed audio: run ASR -> LLM -> TTS."""
        if not session.audio_buffer:
            return

        audio = np.concatenate(session.audio_buffer)
        session.audio_buffer = []
        session.is_recording = False
        session._vad.reset()

        if len(audio) < WHISPER_RATE * 0.3:
            return

        t_start = time.time()

        transcript = await asyncio.get_event_loop().run_in_executor(
            None, session._asr.transcribe, audio
        )
        t_asr = time.time()

        if not transcript.strip():
            await websocket.send(json.dumps({
                "type": "transcript.final", "text": "",
            }))
            return

        await websocket.send(json.dumps({
            "type": "transcript.final",
            "text": transcript,
        }))

        await self._generate_response(session, websocket, transcript, t_start, t_asr)

    async def _handle_text_input(self, session, websocket, msg):
        """Handle direct text input (no ASR needed)."""
        text = msg.get("text", "").strip()
        if not text:
            return

        t_start = time.time()
        await self._generate_response(session, websocket, text, t_start, t_start)

    async def _generate_response(self, session, websocket, user_text, t_start, t_asr):
        """Run LLM streaming + TTS and send results over WebSocket."""
        session.messages.append({"role": "user", "content": user_text})

        await websocket.send(json.dumps({"type": "response.start"}))

        full_response = []
        sentence_buffer = ""
        audio_seq = 0
        first_token_time = None
        first_audio_time = None

        session._interrupted = False
        async for delta in session._llm.stream_chat(
            session.messages, max_tokens=256, temperature=0.7
        ):
            if session._interrupted:
                break

            if delta.startswith("<|channel>") or delta.startswith("<|"):
                continue

            now = time.time()
            if first_token_time is None:
                first_token_time = now

            full_response.append(delta)
            await websocket.send(json.dumps({
                "type": "text.delta", "text": delta,
            }))

            sentence_buffer += delta
            sentences = []
            while True:
                match = SENTENCE_BOUNDARY.search(sentence_buffer)
                if match and match.end() >= MIN_FLUSH_CHARS:
                    sentence = sentence_buffer[:match.end()].strip()
                    sentence_buffer = sentence_buffer[match.end():]
                    if sentence:
                        sentences.append(sentence)
                    continue
                if len(sentence_buffer) >= MAX_BUFFER_CHARS:
                    comma = sentence_buffer.rfind(",", 0, MAX_BUFFER_CHARS)
                    break_at = comma + 1 if comma > MIN_FLUSH_CHARS else MAX_BUFFER_CHARS
                    chunk = sentence_buffer[:break_at].strip()
                    sentence_buffer = sentence_buffer[break_at:]
                    if chunk:
                        sentences.append(chunk)
                    continue
                break

            for sent in sentences:
                audio = await asyncio.get_event_loop().run_in_executor(
                    None, session._tts.synthesize, sent
                )
                if audio is not None:
                    if first_audio_time is None:
                        first_audio_time = time.time()
                    peak = np.abs(audio).max()
                    safe = audio / max(peak, 1.0)
                    pcm_s16 = (safe * 32767).astype(np.int16)
                    b64 = base64.b64encode(pcm_s16.tobytes()).decode("ascii")
                    await websocket.send(json.dumps({
                        "type": "audio.chunk",
                        "data": b64,
                        "seq": audio_seq,
                        "format": "pcm_24k_s16le",
                    }))
                    audio_seq += 1

        remainder = sentence_buffer.strip()
        if remainder and session._tts.available:
            audio = await asyncio.get_event_loop().run_in_executor(
                None, session._tts.synthesize, remainder
            )
            if audio is not None:
                if first_audio_time is None:
                    first_audio_time = time.time()
                peak = np.abs(audio).max()
                safe = audio / max(peak, 1.0)
                pcm_s16 = (safe * 32767).astype(np.int16)
                b64 = base64.b64encode(pcm_s16.tobytes()).decode("ascii")
                await websocket.send(json.dumps({
                    "type": "audio.chunk",
                    "data": b64,
                    "seq": audio_seq,
                    "format": "pcm_24k_s16le",
                }))
                audio_seq += 1

        t_end = time.time()
        response_text = "".join(full_response)
        session.messages.append({"role": "assistant", "content": response_text})

        await websocket.send(json.dumps({
            "type": "text.done", "text": response_text,
        }))
        await websocket.send(json.dumps({"type": "audio.done"}))

        latency = {
            "asr_ms": round((t_asr - t_start) * 1000, 1),
            "llm_ttft_ms": round((first_token_time - t_asr) * 1000, 1) if first_token_time else None,
            "first_audio_ms": round((first_audio_time - t_start) * 1000, 1) if first_audio_time else None,
            "total_ms": round((t_end - t_start) * 1000, 1),
            "audio_chunks": audio_seq,
        }
        await websocket.send(json.dumps({
            "type": "response.done", "latency": latency,
        }))

        print(
            f"  [{session.session_id}] \"{user_text[:40]}\" -> {len(response_text)} chars, "
            f"{audio_seq} audio chunks, {latency['total_ms']:.0f}ms total",
            flush=True,
        )

    def _handle_config(self, session, msg):
        """Update session configuration."""
        if "voice" in msg:
            session.voice = msg["voice"]
            if session._tts:
                session._tts.voice = msg["voice"]
        if "vad_threshold" in msg:
            session.vad_threshold = msg["vad_threshold"]
            if session._vad:
                session._vad.threshold = msg["vad_threshold"]
        if "system_prompt" in msg:
            system_msgs = [m for m in session.messages if m["role"] == "system"]
            if system_msgs:
                system_msgs[0]["content"] = msg["system_prompt"]
            else:
                session.messages.insert(0, {"role": "system", "content": msg["system_prompt"]})

    async def start(self):
        try:
            import websockets
        except ImportError:
            print("ERROR: websockets not installed. Run: pip install websockets", flush=True)
            return

        print(f"\n{'='*60}", flush=True)
        print(f"  Gemma Realtime WebSocket Server", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  Endpoint: ws://{self.host}:{self.port}/v1/realtime", flush=True)
        print(f"  LLM:      {self.llm_url}", flush=True)
        tts_mode = "SNAC (native)" if self.native_tts else f"Kokoro ({self.voice})"
        print(f"  TTS:      {tts_mode}", flush=True)
        print(f"  ASR:      {self.whisper_model}", flush=True)
        print(f"{'='*60}\n", flush=True)

        async with websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            max_size=10 * 1024 * 1024,
            ping_interval=30,
            ping_timeout=10,
        ):
            print(f"Listening on ws://{self.host}:{self.port}/v1/realtime", flush=True)
            await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(
        description="Gemma Realtime WebSocket Server — bidirectional audio + text streaming",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8742)
    parser.add_argument("--llm-url", default="http://localhost:8741")
    parser.add_argument("--whisper-model", default="mlx-community/whisper-small-mlx")
    parser.add_argument("--voice", default="af_bella")
    parser.add_argument(
        "--native-tts", action="store_true",
        help="Use SNAC speech decoder instead of Kokoro TTS (multi-codebook audio)",
    )
    args = parser.parse_args()

    server = RealtimeServer(
        host=args.host,
        port=args.port,
        llm_url=args.llm_url,
        whisper_model=args.whisper_model,
        voice=args.voice,
        native_tts=args.native_tts,
    )
    asyncio.run(server.start())


if __name__ == "__main__":
    main()
