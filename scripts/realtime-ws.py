#!/usr/bin/env python3
"""
WebSocket bidirectional realtime API for gemma-realtime.

Five TTS backends (--tts flag):
    kokoro-onnx (default): Kokoro v1.0 via ONNX Runtime (82M, 4.5 MOS, 0.44x RTF)
    kokoro:                Kokoro via PyTorch (requires Python <3.13)
    voxtral:               Voxtral 4B via mlx-audio (CC BY-NC, 20 voices, 9 langs)
    native:                SNAC speech decoder + depth decoder (research prototype)
                           SELF-CONTAINED: loads Gemma locally for both LLM + TTS,
                           no separate mlx-server.py needed.
    fish:                  Fish Audio DAC codec + Fast AR depth decoder (true STS path)
                           SELF-CONTAINED: loads Gemma + Fish codec locally.
                           8 FSQ groups, 44.1kHz, MOSS-Speech layer splitting.
                           pip install fish-speech (or descript-audio-codec)

Features:
    - Whisper ASR (mlx-whisper on Apple Silicon)
    - Sentence-level streaming LLM -> TTS
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
    python3 scripts/realtime-ws.py                          # Kokoro TTS (default)
    python3 scripts/realtime-ws.py --tts voxtral            # Voxtral 4B MLX
    python3 scripts/realtime-ws.py --tts native             # SNAC decoder
    python3 scripts/realtime-ws.py --port 8742 --llm-url http://localhost:8741

    Tuning (env, read at import): GEMMA_MIN_FLUSH_CHARS, GEMMA_MAX_BUFFER_CHARS — guides/08-inference-sota-roadmap.md
"""

import argparse
import asyncio
import base64
import json
import os
import platform
import re
import sys
import time
import uuid
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

SAMPLE_RATE = 24000
WHISPER_RATE = 16000
BYTES_PER_SAMPLE = 2
VAD_THRESHOLD = 0.4
SILENCE_TIMEOUT_S = 0.8
SENTENCE_BOUNDARY = re.compile(r"[.!?]\s*$|[.!?][\"']\s*$")
MIN_FLUSH_CHARS = int(os.environ.get("GEMMA_MIN_FLUSH_CHARS", "12"))
MAX_BUFFER_CHARS = int(os.environ.get("GEMMA_MAX_BUFFER_CHARS", "120"))


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


class F5TTSEngine:
    """F5-TTS via MLX -- zero-shot voice cloning with flow-matching DiT.

    Provide a 5-10s mono 24kHz WAV reference to clone any voice.
    Slower than Kokoro (~7x RTF) but produces cloned speech.
    """

    def __init__(self, ref_audio_path=None, ref_audio_text=None, steps=8):
        self._ref_audio_path = ref_audio_path
        self._ref_audio_text = ref_audio_text
        self._steps = steps
        self._generate = None
        self.available = False

    def load(self):
        from f5_tts_mlx.generate import generate
        self._generate = generate
        self.available = True
        ref_info = f", ref={self._ref_audio_path}" if self._ref_audio_path else " (default voice)"
        print(f"  TTS: F5-TTS MLX loaded (voice cloning, {self._steps} steps{ref_info})", flush=True)

    def synthesize(self, text: str) -> np.ndarray | None:
        if not self._generate:
            return None
        import soundfile as sf
        import tempfile, os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            self._generate(
                generation_text=text,
                ref_audio_path=self._ref_audio_path,
                ref_audio_text=self._ref_audio_text,
                steps=self._steps,
                output_path=tmp_path,
            )
            data, sr = sf.read(tmp_path)
            if sr != 24000:
                x_old = np.linspace(0, 1, len(data))
                x_new = np.linspace(0, 1, int(len(data) * 24000 / sr))
                data = np.interp(x_new, x_old, data).astype(np.float32)
            return data.astype(np.float32)
        except Exception as e:
            print(f"  F5-TTS error: {e}", flush=True)
            return None
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class KokoroOnnxTTSEngine:
    """Kokoro TTS via ONNX Runtime -- 82M params, 4.5 MOS, 0.44x RTF.

    Uses kokoro-onnx (no spacy/torch dependency chain).
    Produces 24kHz mono float32 audio.
    """

    VOICES = [
        "af_bella", "af_nicole", "af_sarah", "af_sky",
        "am_adam", "am_michael",
        "bf_emma", "bf_isabella",
        "bm_george", "bm_lewis",
    ]

    def __init__(self, voice="af_bella"):
        self.voice = voice
        self._kokoro = None
        self.available = False

    def load(self):
        from kokoro_onnx import Kokoro
        from pathlib import Path

        base = Path(__file__).parent.parent / "models" / "kokoro"
        model_path = base / "kokoro-v1.0.onnx"
        voices_path = base / "voices-v1.0.bin"

        if not model_path.exists():
            print("  TTS: Kokoro ONNX model not found — download with:", flush=True)
            print(f"    mkdir -p {base}", flush=True)
            print(f"    curl -sL https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx -o {model_path}", flush=True)
            print(f"    curl -sL https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin -o {voices_path}", flush=True)
            return

        self._kokoro = Kokoro(str(model_path), str(voices_path))
        self.available = True
        print(f"  TTS: Kokoro ONNX loaded (voice={self.voice}, 82M, 4.5 MOS)", flush=True)

    def synthesize(self, text: str) -> np.ndarray | None:
        if not self._kokoro:
            return None
        samples, _sr = self._kokoro.create(text, voice=self.voice, speed=1.0)
        return samples


class NativeTTSEngine:
    """SNAC-based TTS using the speech decoder + depth decoder pipeline.

    Research prototype -- uses embed_tokens, not hidden states.
    For production quality, use KokoroOnnxTTSEngine instead.
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


class FishTTSEngine:
    """Fish Audio codec-based TTS with Fast AR depth decoding.

    Uses the Fish STS pipeline from fish_sts.py. When loaded in "tts" mode
    (text → audio), it still goes through Gemma's embeddings + Fish codec,
    giving higher quality than SNAC native TTS.

    For true STS (audio → audio, no text), use FishSTSPipeline directly.
    """

    def __init__(self):
        self._pipeline = None
        self._inner = None
        self._tokenizer = None
        self.available = False

    def load(self, inner_model, tokenizer):
        """Load Fish codec + Fast AR for TTS mode with trained weights."""
        from fish_sts import FishSpeechToSpeech, FishSTSConfig, FishSTSPipeline, PRESET_CONFIGS
        from codec import AudioCodec

        self._inner = inner_model
        self._tokenizer = tokenizer

        # Resolve trained weights (Phase C > B > A)
        w_path = FishSTSPipeline._resolve_weights(None)

        self._codec = AudioCodec("fish")
        try:
            self._codec.load()
            self._use_snac_fallback = False
        except Exception as e:
            print(f"  Fish TTS: codec load failed ({e}), falling back to SNAC", flush=True)
            self._codec = AudioCodec("snac")
            self._codec.load()
            self._use_snac_fallback = True

        if w_path is not None:
            import mlx.core as mx
            config = FishSTSPipeline.config_from_weights(str(w_path))
            self._sts_model = FishSpeechToSpeech(config)
            w = mx.load(str(w_path))
            self._sts_model.load_weights(list(w.items()), strict=False)
            print(f"  Fish TTS: loaded trained weights from {w_path} "
                  f"(llm_dim={config.llm_dim}, cb={config.fish_codebook_size})", flush=True)
        else:
            import mlx.core as mx
            probe = inner_model.embed_tokens(mx.array([[0]]))
            llm_dim = probe.shape[-1]
            config = FishSTSConfig(llm_dim=llm_dim)
            self._sts_model = FishSpeechToSpeech(config)
            print("  Fish TTS: WARNING — no trained weights, random init", flush=True)

        self.available = True
        print("  Fish TTS engine loaded", flush=True)

    def synthesize(self, text: str) -> np.ndarray:
        """Text → audio via Gemma embeddings + Fish codec."""
        import mlx.core as mx
        from codec import CodecTokens, CodecType

        ids = self._tokenizer.encode(text[:200], add_special_tokens=False)
        if not ids:
            return np.zeros(4410, dtype=np.float32)

        emb = self._inner.embed_tokens(mx.array([ids]))

        cb0_out, speech_hidden = self._sts_model.generate_cb0(
            emb, temperature=0.5, top_k=30
        )
        mx.eval(cb0_out, speech_hidden)

        if cb0_out.size == 0:
            return np.zeros(4410, dtype=np.float32)

        all_codes = self._sts_model.decode_depth(cb0_out, speech_hidden)
        mx.eval(all_codes)

        codec_type = CodecType.SNAC if self._use_snac_fallback else CodecType.FISH_DAC
        n_cb = 3 if self._use_snac_fallback else self._sts_model.config.fish_n_codebooks
        fr = 12.0 if self._use_snac_fallback else self._sts_model.config.fish_frame_rate
        out_tokens = CodecTokens(
            codes=np.array(all_codes[0].tolist(), dtype=np.int64),
            n_codebooks=n_cb,
            frame_rate=fr,
            codec_type=codec_type,
        )
        return self._codec.decode(out_tokens)


async def _send_tts(tts, text, websocket, audio_seq, first_audio_time):
    """Synthesize text and send audio chunks over WebSocket.

    MLX-based engines (NativeTTS, FishTTS, Voxtral, F5-TTS) must run on the
    main event-loop thread because GPU streams are thread-local.
    ONNX-based engines (KokoroOnnx) can safely use a thread-pool executor.
    """
    from importlib import import_module
    speech = import_module("speech-server")

    uses_mlx = isinstance(tts, (NativeTTSEngine, FishTTSEngine, F5TTSEngine,
                                speech.VoxtralTTSEngine))

    if uses_mlx:
        audio = tts.synthesize(text)
        await asyncio.sleep(0)
        if audio is not None:
            if first_audio_time is None:
                first_audio_time = time.time()
            await RealtimeServer._send_audio_chunk(websocket, audio, audio_seq)
            audio_seq += 1
    else:
        audio = await asyncio.get_event_loop().run_in_executor(
            None, tts.synthesize, text
        )
        if audio is not None:
            if first_audio_time is None:
                first_audio_time = time.time()
            await RealtimeServer._send_audio_chunk(websocket, audio, audio_seq)
            audio_seq += 1
    return audio_seq, first_audio_time


class LocalLLMClient:
    """Local Gemma LLM — runs generation in-process, no HTTP server needed.

    Drop-in replacement for speech.LLMClient when using --native-tts.
    Implements the same async stream_chat() interface.
    """

    def __init__(self, gemma_model, tokenizer):
        self._model = gemma_model
        self._tokenizer = tokenizer

    async def check_health(self):
        return {"status": "ok", "backend": "local"}

    async def stream_chat(self, messages, max_tokens=256, temperature=0.7):
        """Yield text deltas from local Gemma generation.

        Filters thinking tokens just like the HTTP LLMClient.
        MLX must run on the main thread so we yield to the event loop
        between tokens with asyncio.sleep(0).
        """
        import asyncio
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        # Gemma 4 format:
        #   <|channel>thought\n...thinking...<channel|>response<end_of_turn>
        # Note: opening = <|channel>, closing = <channel|> (reversed)
        in_thinking = False

        for resp in stream_generate(
            self._model, self._tokenizer, prompt=prompt,
            max_tokens=max_tokens, sampler=make_sampler(temp=temperature),
        ):
            content = resp.text or ""

            if "<end_of_turn>" in content:
                clean = content.split("<end_of_turn>")[0]
                if clean and not in_thinking:
                    yield clean
                return

            if "<|channel>" in content:
                in_thinking = True
                continue
            if "<channel|>" in content:
                in_thinking = False
                continue
            if content.startswith("<|") or content.startswith("<channel"):
                continue
            if "thought" == content.strip() and in_thinking:
                continue

            if in_thinking:
                continue

            if content:
                yield content
                await asyncio.sleep(0)

    async def close(self):
        pass


class RealtimeServer:
    """WebSocket server implementing the realtime bidirectional protocol."""

    def __init__(self, host="0.0.0.0", port=8742, llm_url="http://localhost:8741",
                 whisper_model="mlx-community/whisper-small-mlx", voice=None,
                 tts_backend="kokoro", tts_precision="6bit", denoise_steps=4,
                 draft_heads=None,
                 ref_audio_path=None, ref_audio_text=None, f5_steps=8):
        self.host = host
        self.port = port
        self.llm_url = llm_url
        self.whisper_model = whisper_model
        self.tts_backend = tts_backend
        self.tts_precision = tts_precision
        self.denoise_steps = denoise_steps
        self.draft_heads = draft_heads
        self.voice = voice
        self.ref_audio_path = ref_audio_path
        self.ref_audio_text = ref_audio_text
        self.f5_steps = f5_steps
        self._sessions = {}
        self._shared_vad = None
        self._shared_asr = None
        self._shared_tts = None
        self._iosurface_health_logged = False

    def _maybe_log_iosurface_health(self):
        if self._iosurface_health_logged or platform.system() != "Darwin":
            return
        self._iosurface_health_logged = True
        try:
            from native_hw import health_payload_for_http

            h = health_payload_for_http(None)
            if h:
                print(f"  libgemma_hw / IOSurface: {h}", flush=True)
        except Exception as e:
            print(f"  libgemma_hw health probe: {e}", flush=True)

    async def _ensure_shared_models(self):
        """Load heavyweight models once at startup, share across all sessions."""
        if self._shared_vad is not None:
            return
        print("  Loading models...", flush=True)

        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from importlib import import_module
        speech = import_module("speech-server")

        self._shared_vad = speech.SileroVAD(threshold=VAD_THRESHOLD)
        self._shared_asr = speech.WhisperASR(model_name=self.whisper_model)

        if self.tts_backend == "native":
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
                depth_decoder = tdd.DepthDecoder(**tdd.DEPTH_DECODER_CONFIG)
                dw = mx.load(str(depth_path))
                depth_decoder.load_weights(list(dw.items()))
                print("    Depth decoder loaded (3-codebook)", flush=True)

            codec = AudioCodec("snac")
            codec.load()

            self._shared_tts = NativeTTSEngine()
            self._shared_tts.load(inner, tokenizer, decoder, depth_decoder, codec)
            self._gemma = gemma
            self._gemma_tokenizer = tokenizer
            self._local_llm = LocalLLMClient(gemma, tokenizer)
            print("    Native SNAC TTS ready (self-contained, no HTTP LLM needed)", flush=True)
        elif self.tts_backend == "fish":
            import mlx.core as mx
            from mlx_lm import load as lm_load

            print("  Loading Fish DAC true STS pipeline...", flush=True)
            gemma, tokenizer = lm_load("mlx-community/gemma-4-26b-a4b-it-4bit")
            if hasattr(gemma, "language_model"):
                inner = gemma.language_model.model
            else:
                inner = gemma.model

            self._shared_tts = FishTTSEngine()
            self._shared_tts.load(inner, tokenizer)
            self._gemma = gemma
            self._gemma_tokenizer = tokenizer
            self._local_llm = LocalLLMClient(gemma, tokenizer)
            print("    Fish TTS ready (8-group FSQ, 44.1kHz, self-contained)", flush=True)
        elif self.tts_backend == "kokoro-onnx":
            voice = self.voice or "af_bella"
            self._shared_tts = KokoroOnnxTTSEngine(voice=voice)
            self._shared_tts.load()

            from mlx_lm import load as lm_load
            print("  Loading Gemma for local LLM inference...", flush=True)
            gemma, tokenizer = lm_load("mlx-community/gemma-4-26b-a4b-it-4bit")
            self._local_llm = LocalLLMClient(gemma, tokenizer)
            print("    Gemma loaded (self-contained, Kokoro ONNX + local LLM)", flush=True)
        elif self.tts_backend == "f5":
            self._shared_tts = F5TTSEngine(
                ref_audio_path=self.ref_audio_path,
                ref_audio_text=self.ref_audio_text,
                steps=self.f5_steps,
            )
            self._shared_tts.load()

            from mlx_lm import load as lm_load
            print("  Loading Gemma for local LLM inference...", flush=True)
            gemma, tokenizer = lm_load("mlx-community/gemma-4-26b-a4b-it-4bit")
            self._local_llm = LocalLLMClient(gemma, tokenizer)
            print("    Gemma loaded (self-contained, F5-TTS + local LLM)", flush=True)
        elif self.tts_backend == "voxtral":
            voice = self.voice or "cheerful_male"
            self._shared_tts = speech.VoxtralTTSEngine(
                voice=voice, precision=self.tts_precision,
                denoise_steps=self.denoise_steps,
                draft_heads=self.draft_heads,
            )
            self._shared_tts.load()
            from mlx_lm import load as lm_load
            print("  LLM: Loading Gemma for self-contained Voxtral mode...", flush=True)
            gemma, tokenizer = lm_load("mlx-community/gemma-4-26b-a4b-it-4bit")
            self._local_llm = LocalLLMClient(gemma, tokenizer)
            print("    Gemma loaded (self-contained, Voxtral + local LLM)", flush=True)
        else:
            voice = self.voice or "af_bella"
            self._shared_tts = speech.TTSEngine(voice=voice)
            self._shared_tts.load()

        self._shared_vad.load()
        self._shared_asr.load()
        self._maybe_log_iosurface_health()

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

            if hasattr(self, '_local_llm'):
                session._llm = self._local_llm
            else:
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

        transcript = session._asr.transcribe(audio)
        await asyncio.sleep(0)
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

    @staticmethod
    async def _send_audio_chunk(websocket, audio, audio_seq):
        """Normalize and send a single audio array as a WebSocket chunk."""
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
            session.messages, max_tokens=512, temperature=0.7
        ):
            if session._interrupted:
                break

            if delta.startswith("<|channel>") or delta.startswith("<|"):
                continue
            if delta.startswith("<channel|>") or delta.startswith("<channel"):
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
                audio_seq, first_audio_time = await _send_tts(
                    session._tts, sent, websocket, audio_seq, first_audio_time
                )

        remainder = sentence_buffer.strip()
        if remainder and session._tts.available:
            audio_seq, first_audio_time = await _send_tts(
                session._tts, remainder, websocket, audio_seq, first_audio_time
            )

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
        voice = self.voice or {"kokoro-onnx": "af_bella", "f5": "clone" if self.ref_audio_path else "default", "kokoro": "af_bella", "voxtral": "cheerful_male", "native": "SNAC", "fish": "Fish DAC 10cb"}.get(self.tts_backend, "?")
        precision_str = f", {self.tts_precision}" if self.tts_backend == "voxtral" else ""
        tts_mode = f"{self.tts_backend} ({voice}{precision_str})"
        print(f"  TTS:      {tts_mode}", flush=True)
        print(f"  ASR:      {self.whisper_model}", flush=True)
        print(f"{'='*60}\n", flush=True)

        await self._ensure_shared_models()

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
    parser.add_argument("--voice", default=None,
                        help="TTS voice (auto-selected per backend if omitted)")
    parser.add_argument(
        "--tts", choices=["kokoro-onnx", "f5", "kokoro", "voxtral", "native", "fish"],
        default="kokoro-onnx",
        help="TTS backend: kokoro-onnx (default, 82M, 4.5 MOS, fast), "
             "f5 (voice cloning, flow-matching DiT), "
             "kokoro (needs Python <3.13), voxtral (4B MLX), native (SNAC decoder), "
             "fish (Fish DAC 8-group FSQ, true STS path)",
    )
    parser.add_argument(
        "--ref-audio", default=None,
        help="Path to reference WAV (mono 24kHz, 5-10s) for F5-TTS voice cloning",
    )
    parser.add_argument(
        "--ref-text", default=None,
        help="Transcript of the reference audio for F5-TTS voice cloning",
    )
    parser.add_argument(
        "--f5-steps", type=int, default=8,
        help="F5-TTS denoising steps (default: 8). Lower = faster but lower quality.",
    )
    parser.add_argument(
        "--tts-precision", choices=["4bit", "6bit", "bf16"], default="6bit",
        help="Voxtral model precision (default: 6bit, best speed/quality for real-time)",
    )
    parser.add_argument(
        "--denoise-steps", type=int, default=4, choices=[2, 4, 6, 8],
        help="Voxtral denoising steps (default: 4, original: 8). Lower = faster.",
    )
    parser.add_argument(
        "--draft-heads", default=None,
        help="Draft heads for Voxtral speculative decoding; auto-detected if omitted (see draft_heads_resolve)",
    )
    parser.add_argument(
        "--no-draft-heads",
        action="store_true",
        help="Disable Voxtral draft heads even if default weights exist",
    )
    parser.add_argument(
        "--native-tts", action="store_true",
        help="Shorthand for --tts native",
    )
    args = parser.parse_args()
    if args.native_tts:
        args.tts = "native"

    from draft_heads_resolve import resolve_draft_heads_path

    if args.tts == "voxtral":
        draft_heads = None if args.no_draft_heads else resolve_draft_heads_path(args.draft_heads)
    else:
        draft_heads = None

    server = RealtimeServer(
        host=args.host,
        port=args.port,
        llm_url=args.llm_url,
        whisper_model=args.whisper_model,
        voice=args.voice,
        tts_backend=args.tts,
        tts_precision=args.tts_precision,
        denoise_steps=args.denoise_steps,
        draft_heads=draft_heads,
        ref_audio_path=args.ref_audio,
        ref_audio_text=args.ref_text,
        f5_steps=args.f5_steps,
    )
    asyncio.run(server.start())


if __name__ == "__main__":
    main()
