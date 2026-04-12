#!/usr/bin/env python3
"""
Real-time bidirectional speech server for gemma-realtime.

Cascaded pipeline: Microphone -> VAD -> Whisper ASR -> Gemma E4B -> TTS -> Speaker
Connects to the existing MLX server (mlx-server.py) via OpenAI-compatible HTTP API,
adds streaming ASR input and streaming TTS output with sentence-boundary buffering.

Architecture:
    Audio In -> Silero VAD -> Whisper (streaming) -> text
    text -> POST /v1/chat/completions (stream:true) -> text deltas
    text deltas -> sentence buffer -> Kokoro TTS -> Audio Out

Usage:
    python3 scripts/speech-server.py
    python3 scripts/speech-server.py --llm-url http://localhost:8741
    python3 scripts/speech-server.py --tts kokoro --voice af_bella
    python3 scripts/speech-server.py --no-mic --text-only

    Tuning (env, read at import): GEMMA_MIN_FLUSH_CHARS, GEMMA_MAX_BUFFER_CHARS — see guides/08-inference-sota-roadmap.md

Requirements:
    pip install mlx-whisper kokoro sounddevice numpy websockets silero-vad
"""

import argparse
import asyncio
import json
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

SAMPLE_RATE = 24000
WHISPER_RATE = 16000
CHUNK_DURATION_MS = 30
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
VAD_THRESHOLD = 0.4
SILENCE_TIMEOUT_MS = 800
SENTENCE_BOUNDARY = re.compile(r"[.!?]\s*$|[.!?][\"']\s*$")
MIN_FLUSH_CHARS = int(os.environ.get("GEMMA_MIN_FLUSH_CHARS", "12"))
MAX_BUFFER_CHARS = int(os.environ.get("GEMMA_MAX_BUFFER_CHARS", "120"))


class SileroVAD:
    """Voice Activity Detection using Silero VAD (ONNX, <1ms per frame)."""

    def __init__(self, threshold=VAD_THRESHOLD):
        self.threshold = threshold
        self._model = None
        self._ready = False

    def load(self):
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,
            )
            self._model = model
            self._get_speech_timestamps = utils[0]
            self._ready = True
            print("  VAD: Silero VAD loaded (ONNX)", flush=True)
        except Exception as e:
            print(f"  VAD: Silero load failed ({e}), using energy-based fallback", flush=True)
            self._ready = False

    def is_speech(self, audio_chunk_16k: np.ndarray) -> bool:
        if not self._ready:
            rms = np.sqrt(np.mean(audio_chunk_16k.astype(np.float32) ** 2))
            return rms > 0.01

        import torch
        audio_f32 = audio_chunk_16k.astype(np.float32)
        if np.abs(audio_f32).max() > 1.0:
            audio_f32 = audio_f32 / 32768.0

        # Silero expects 512-sample chunks at 16kHz
        chunk_size = 512
        max_conf = 0.0
        for start in range(0, len(audio_f32) - chunk_size + 1, chunk_size):
            chunk = torch.from_numpy(audio_f32[start:start + chunk_size])
            conf = self._model(chunk, 16000).item()
            max_conf = max(max_conf, conf)
            if conf > self.threshold:
                return True
        return max_conf > self.threshold

    def reset(self):
        if self._ready and hasattr(self._model, "reset_states"):
            self._model.reset_states()


class WhisperASR:
    """Streaming ASR using mlx-whisper or whisper on Apple Silicon."""

    def __init__(self, model_name="mlx-community/whisper-small-mlx"):
        self.model_name = model_name
        self._transcribe = None

    def load(self):
        try:
            import mlx_whisper
            self._transcribe = mlx_whisper.transcribe
            print(f"  ASR: mlx-whisper loaded ({self.model_name})", flush=True)
        except ImportError:
            try:
                import whisper
                self._model = whisper.load_model("small")
                self._transcribe = self._model.transcribe
                print("  ASR: OpenAI Whisper loaded (small)", flush=True)
            except ImportError:
                print("  ASR: No whisper backend available!", flush=True)
                print("    Install: pip install mlx-whisper", flush=True)
                raise

    def transcribe(self, audio_np: np.ndarray) -> str:
        if self._transcribe is None:
            return ""

        audio = audio_np.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0

        try:
            import mlx_whisper
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.model_name,
                language="en",
                fp16=True,
            )
        except Exception:
            result = self._transcribe(audio, language="en", fp16=True)

        text = result.get("text", "").strip()
        return text


class SentenceBuffer:
    """Buffer text deltas and flush at sentence boundaries for TTS."""

    def __init__(self, min_chars=MIN_FLUSH_CHARS, max_chars=MAX_BUFFER_CHARS):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buffer = ""

    def add(self, text: str) -> list[str]:
        self._buffer += text
        sentences = []

        while True:
            match = SENTENCE_BOUNDARY.search(self._buffer)
            if match and match.end() >= self.min_chars:
                sentence = self._buffer[: match.end()].strip()
                self._buffer = self._buffer[match.end() :]
                if sentence:
                    sentences.append(sentence)
                continue

            if len(self._buffer) >= self.max_chars:
                comma = self._buffer.rfind(",", 0, self.max_chars)
                break_at = comma + 1 if comma > self.min_chars else self.max_chars
                chunk = self._buffer[:break_at].strip()
                self._buffer = self._buffer[break_at:]
                if chunk:
                    sentences.append(chunk)
                continue

            break

        return sentences

    def flush(self) -> str | None:
        remainder = self._buffer.strip()
        self._buffer = ""
        return remainder if remainder else None

    def clear(self):
        self._buffer = ""


class TTSEngine:
    """Text-to-speech using Kokoro (82M params, Apache 2.0)."""

    def __init__(self, voice="af_bella", speed=1.0):
        self.voice = voice
        self.speed = speed
        self._pipeline = None
        self._available = False

    def load(self):
        try:
            from kokoro import KPipeline
            self._pipeline = KPipeline(lang_code="a")
            self._available = True
            print(f"  TTS: Kokoro loaded (voice={self.voice})", flush=True)
        except ImportError:
            print("  TTS: Kokoro not available — audio output disabled", flush=True)
            print("    Install: pip install kokoro soundfile", flush=True)
            self._available = False

    def synthesize(self, text: str) -> np.ndarray | None:
        if not self._available or not text.strip():
            return None

        try:
            samples = []
            for result in self._pipeline(
                text, voice=self.voice, speed=self.speed
            ):
                if result.audio is not None:
                    samples.append(result.audio.numpy() if hasattr(result.audio, "numpy") else np.array(result.audio))
            if not samples:
                return None
            audio = np.concatenate(samples)
            return audio
        except Exception as e:
            print(f"  TTS error: {e}", flush=True)
            return None

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def available(self) -> bool:
        return self._available


VOXTRAL_MODELS = {
    "4bit": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",       # 2.5GB, ~0.22x RTF
    "6bit": "mlx-community/Voxtral-4B-TTS-2603-mlx-6bit",       # 3.5GB, ~0.38x RTF
    "bf16": "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16",       # 8GB,   ~1.06x RTF
}


class VoxtralTTSEngine:
    """Text-to-speech using Voxtral 4B via mlx-audio (CC BY-NC 4.0).

    20 preset voices, 9 languages, 24kHz output.
    Defaults to 6-bit quantized model (~3.5GB, 0.38x RTF) -- best quality/speed
    balance for real-time pipelines. Use bf16 for maximum quality with streaming.
    """

    VOICES = [
        "casual_male", "casual_female", "cheerful_male", "cheerful_female",
        "serious_male", "serious_female", "narrator_male", "narrator_female",
        "fr_male", "fr_female", "de_male", "de_female",
        "es_male", "es_female", "it_male", "it_female",
        "pt_male", "pt_female", "ar_male", "hi_male", "hi_female",
    ]

    def __init__(self, voice="cheerful_male", model_path=None, precision="6bit",
                 denoise_steps=4, draft_heads=None):
        self.voice = voice
        self.model_path = model_path or VOXTRAL_MODELS.get(precision, VOXTRAL_MODELS["6bit"])
        self.precision = precision
        self.denoise_steps = denoise_steps
        self.draft_heads_path = draft_heads
        self._model = None
        self._draft_heads = None
        self._available = False

    def load(self):
        try:
            from mlx_audio.tts.utils import load as mlx_audio_load
            print(f"  TTS: Loading Voxtral {self.precision} from {self.model_path}...", flush=True)
            self._model = mlx_audio_load(self.model_path)
            if self.denoise_steps != 8:
                self._model.acoustic_transformer.args.n_denoising_steps = self.denoise_steps
            if self.draft_heads_path:
                from voxtral_speculative import DraftHeadSet
                self._draft_heads = DraftHeadSet.load(self.draft_heads_path)
                print(f"  TTS: Draft heads loaded from {self.draft_heads_path}", flush=True)
            self._available = True
            mode = "speculative" if self._draft_heads else "standard"
            print(f"  TTS: Voxtral loaded (voice={self.voice}, precision={self.precision}, "
                  f"denoise_steps={self.denoise_steps}, mode={mode})", flush=True)
        except Exception as e:
            print(f"  TTS: Voxtral not available — {e}", flush=True)
            print("    Install: pip install mlx-audio", flush=True)
            self._available = False

    @staticmethod
    def _estimate_max_frames(text, headroom=2.0):
        """Duration-proportional frame cap to prevent runaway generation."""
        n_words = len(text.split())
        est_seconds = n_words / 2.5  # ~150 WPM = 2.5 WPS
        max_frames = int(est_seconds * 12.5 * headroom)
        return max(50, min(max_frames, 500))

    def synthesize(self, text: str, max_tokens: int | None = None) -> np.ndarray | None:
        """Blocking synthesis -- uses speculative decoding when draft heads are loaded."""
        if not self._available or not text.strip():
            return None
        if max_tokens is None:
            max_tokens = self._estimate_max_frames(text)
        try:
            if self._draft_heads:
                from voxtral_speculative import speculative_generate
                audio, stats = speculative_generate(
                    self._model, text, self.voice, self._draft_heads,
                    max_tokens=max_tokens, verbose=True,
                )
                return audio if len(audio) > 0 else None
            chunks = []
            for result in self._model.generate(
                text=text, voice=self.voice, max_tokens=max_tokens,
            ):
                if result.audio is not None:
                    chunks.append(np.array(result.audio))
            if not chunks:
                return None
            return np.concatenate(chunks)
        except Exception as e:
            print(f"  TTS error: {e}", flush=True)
            return None

    def synthesize_stream(self, text: str, max_tokens: int | None = None):
        """Streaming synthesis -- yields audio chunks as they're generated.

        Each chunk is ~2s of audio, allowing playback to start before the full
        sentence is synthesized. Critical for bf16 where RTF >= 1.0.
        """
        if not self._available or not text.strip():
            return
        if max_tokens is None:
            max_tokens = self._estimate_max_frames(text)
        try:
            for result in self._model.generate(
                text=text, voice=self.voice,
                stream=True, streaming_interval=1.5,
                max_tokens=max_tokens,
            ):
                if result.audio is not None:
                    yield np.array(result.audio)
        except Exception as e:
            print(f"  TTS stream error: {e}", flush=True)

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def available(self) -> bool:
        return self._available


class LLMClient:
    """Async HTTP client for the MLX server's OpenAI-compatible streaming API."""

    def __init__(self, base_url="http://localhost:8741"):
        self.base_url = base_url.rstrip("/")
        self._session = None

    async def _ensure_session(self):
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession()

    async def check_health(self) -> dict | None:
        import aiohttp
        await self._ensure_session()
        try:
            async with self._session.get(f"{self.base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception:
            pass
        return None

    async def stream_chat(self, messages: list[dict], max_tokens=256, temperature=0.7):
        """Yield text deltas from streaming chat completion.

        Filters Gemma 4 thinking tokens: skips content between <|channel>thought
        and the next <|channel> marker (typically <|channel>response or end).
        """
        await self._ensure_session()
        import aiohttp

        payload = {
            "model": "local",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        in_thinking = False

        try:
            async with self._session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                async for line in resp.content:
                    line = line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if not content:
                            continue
                        if "<|channel>thought" in content:
                            in_thinking = True
                            continue
                        if "<|channel>" in content:
                            in_thinking = False
                            continue
                        if content.startswith("<|"):
                            continue
                        if in_thinking:
                            continue
                        yield content
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"  LLM stream error: {e}", flush=True)

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None


class AudioIO:
    """Microphone input and speaker output via sounddevice."""

    def __init__(self, input_rate=WHISPER_RATE, output_rate=SAMPLE_RATE):
        self.input_rate = input_rate
        self.output_rate = output_rate
        self._input_queue = queue.Queue()
        self._output_queue = queue.Queue()
        self._input_stream = None
        self._output_stream = None
        self._playing = threading.Event()

    def start_input(self):
        import sounddevice as sd
        self._input_stream = sd.InputStream(
            samplerate=self.input_rate,
            channels=1,
            dtype="float32",
            blocksize=int(self.input_rate * CHUNK_DURATION_MS / 1000),
            callback=self._input_callback,
        )
        self._input_stream.start()
        print(f"  Audio: Mic open ({self.input_rate}Hz)", flush=True)

    def _input_callback(self, indata, frames, time_info, status):
        self._input_queue.put(indata[:, 0].copy())

    def get_audio_chunk(self, timeout=0.1) -> np.ndarray | None:
        try:
            return self._input_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def play_audio(self, audio: np.ndarray):
        """Play audio array through speakers (blocking in caller's thread)."""
        import sounddevice as sd
        self._playing.set()
        try:
            sd.play(audio, self.output_rate, blocking=True)
        finally:
            self._playing.clear()

    def play_audio_async(self, audio: np.ndarray):
        """Play audio in background thread."""
        t = threading.Thread(target=self.play_audio, args=(audio,), daemon=True)
        t.start()
        return t

    @property
    def is_playing(self) -> bool:
        return self._playing.is_set()

    def stop(self):
        import sounddevice as sd
        if self._input_stream:
            self._input_stream.stop()
            self._input_stream.close()
        sd.stop()


class ConversationState:
    """Manages multi-turn conversation history."""

    def __init__(self, system_prompt=None, max_turns=10):
        self.max_turns = max_turns
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str):
        self.messages.append({"role": "assistant", "content": text})
        self._trim()

    def _trim(self):
        system = [m for m in self.messages if m["role"] == "system"]
        non_system = [m for m in self.messages if m["role"] != "system"]
        if len(non_system) > self.max_turns * 2:
            non_system = non_system[-(self.max_turns * 2):]
        self.messages = system + non_system

    def get_messages(self) -> list[dict]:
        return list(self.messages)


async def run_speech_pipeline(args):
    """Main speech pipeline loop."""

    print(f"\n{'='*60}", flush=True)
    print(f"  Gemma Realtime Speech Server", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  LLM backend: {args.llm_url}", flush=True)
    print(f"  Mode: {'text-only' if args.text_only else 'full audio'}", flush=True)
    print(f"{'='*60}\n", flush=True)

    vad = SileroVAD(threshold=args.vad_threshold)
    asr = WhisperASR(model_name=args.whisper_model)

    if args.tts == "voxtral":
        voice = args.voice or "cheerful_male"
        tts = VoxtralTTSEngine(voice=voice, precision=args.tts_precision,
                               denoise_steps=args.denoise_steps,
                               draft_heads=args.draft_heads)
    else:
        voice = args.voice or "af_bella"
        tts = TTSEngine(voice=voice, speed=args.speed)

    llm = LLMClient(base_url=args.llm_url)
    sentence_buf = SentenceBuffer()

    print("Loading components...", flush=True)
    if not args.text_only:
        vad.load()
        asr.load()
        tts.load()

    health = await llm.check_health()
    if health:
        engine = health.get("engine", "unknown")
        model = health.get("model", "unknown")
        tq = " + TurboQuant+" if health.get("turboquant_plus") else ""
        spec = " + speculative" if health.get("speculative_decoding") else ""
        print(f"  LLM: Connected — {model} via {engine}{tq}{spec}", flush=True)
    else:
        print(f"  LLM: WARNING — Cannot reach {args.llm_url}/health", flush=True)
        print(f"       Start the MLX server: python3 scripts/mlx-server.py --realtime", flush=True)

    conversation = ConversationState(
        system_prompt=args.system_prompt or (
            "You are a helpful voice assistant. Keep responses concise and conversational. "
            "Respond naturally as if speaking aloud — use short sentences, avoid markdown, "
            "lists, or code blocks."
        ),
        max_turns=args.max_turns,
    )

    audio_io = None
    if not args.text_only:
        try:
            audio_io = AudioIO()
            audio_io.start_input()
        except Exception as e:
            print(f"  Audio I/O failed: {e}", flush=True)
            print("  Falling back to text-only mode", flush=True)
            args.text_only = True

    print(f"\n{'='*60}", flush=True)
    if args.text_only:
        print("  Ready! Type your messages below.", flush=True)
    else:
        print("  Ready! Speak into your microphone.", flush=True)
        print("  (Press Ctrl+C to quit)", flush=True)
    print(f"{'='*60}\n", flush=True)

    try:
        if args.text_only:
            await _text_loop(llm, tts, sentence_buf, conversation, args)
        else:
            await _audio_loop(vad, asr, llm, tts, sentence_buf, conversation, audio_io, args)
    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)
    finally:
        if audio_io:
            audio_io.stop()
        await llm.close()


def _play_tts_blocking(tts, text):
    """Play TTS audio. Uses streaming for bf16 Voxtral (RTF > 1.0) to overlap
    generation with playback; uses blocking synthesis for everything else."""
    import sounddevice as sd
    use_stream = (
        hasattr(tts, 'synthesize_stream')
        and getattr(tts, 'precision', None) == 'bf16'
    )
    if use_stream:
        for chunk in tts.synthesize_stream(text):
            sd.play(chunk, tts.sample_rate, blocking=True)
    else:
        audio = tts.synthesize(text)
        if audio is not None:
            sd.play(audio, tts.sample_rate, blocking=True)


async def _text_loop(llm, tts, sentence_buf, conversation, args):
    """Text-only interactive loop (no microphone)."""
    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input("\nYou: ")
            )
        except EOFError:
            break

        if not user_input.strip():
            continue
        if user_input.strip().lower() in ("quit", "exit", "q"):
            break

        conversation.add_user(user_input)
        print("Assistant: ", end="", flush=True)

        full_response = []
        t0 = time.time()
        first_token = None

        async for delta in llm.stream_chat(
            conversation.get_messages(),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ):
            if first_token is None:
                first_token = time.time()
            print(delta, end="", flush=True)
            full_response.append(delta)

            if tts.available:
                sentences = sentence_buf.add(delta)
                for s in sentences:
                    await asyncio.get_event_loop().run_in_executor(
                        None, _play_tts_blocking, tts, s
                    )

        remainder = sentence_buf.flush()
        if remainder and tts.available:
            await asyncio.get_event_loop().run_in_executor(
                None, _play_tts_blocking, tts, remainder
            )

        elapsed = time.time() - t0
        ttft = (first_token - t0) if first_token else elapsed
        response_text = "".join(full_response)
        conversation.add_assistant(response_text)
        print(f"\n  [{elapsed:.1f}s, TTFT {ttft:.2f}s]", flush=True)


async def _audio_loop(vad, asr, llm, tts, sentence_buf, conversation, audio_io, args):
    """Full audio loop: mic -> VAD -> ASR -> LLM -> TTS -> speaker."""
    audio_buffer = []
    is_speaking = False
    silence_start = None

    print("Listening...", flush=True)

    while True:
        chunk = await asyncio.get_event_loop().run_in_executor(
            None, audio_io.get_audio_chunk, 0.05
        )

        if chunk is None:
            await asyncio.sleep(0.01)
            continue

        if audio_io.is_playing:
            continue

        speech_detected = vad.is_speech(chunk)

        if speech_detected:
            if not is_speaking:
                is_speaking = True
                audio_buffer = []
                print("\n  [listening...]", end="", flush=True)
            audio_buffer.append(chunk)
            silence_start = None
        elif is_speaking:
            audio_buffer.append(chunk)
            if silence_start is None:
                silence_start = time.time()
            elif (time.time() - silence_start) * 1000 > SILENCE_TIMEOUT_MS:
                is_speaking = False
                silence_start = None
                vad.reset()

                if len(audio_buffer) < 5:
                    audio_buffer = []
                    continue

                full_audio = np.concatenate(audio_buffer)
                audio_buffer = []

                print(" transcribing...", end="", flush=True)
                t_asr = time.time()
                transcript = await asyncio.get_event_loop().run_in_executor(
                    None, asr.transcribe, full_audio
                )
                asr_time = time.time() - t_asr

                if not transcript.strip():
                    print(" (empty)", flush=True)
                    continue

                print(f" done ({asr_time:.1f}s)", flush=True)
                print(f"  You: {transcript}", flush=True)

                conversation.add_user(transcript)
                print("  Assistant: ", end="", flush=True)

                full_response = []
                t0 = time.time()
                first_token = None
                tts_threads = []

                def _synth_and_play(text):
                    use_stream = (
                        hasattr(tts, 'synthesize_stream')
                        and getattr(tts, 'precision', None) == 'bf16'
                    )
                    if use_stream:
                        for chunk in tts.synthesize_stream(text):
                            audio_io.play_audio(chunk)
                    else:
                        audio = tts.synthesize(text)
                        if audio is not None:
                            audio_io.play_audio(audio)

                async for delta in llm.stream_chat(
                    conversation.get_messages(),
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                ):
                    if first_token is None:
                        first_token = time.time()
                    print(delta, end="", flush=True)
                    full_response.append(delta)

                    sentences = sentence_buf.add(delta)
                    for s in sentences:
                        await asyncio.get_event_loop().run_in_executor(
                            None, _synth_and_play, s
                        )

                remainder = sentence_buf.flush()
                if remainder:
                    await asyncio.get_event_loop().run_in_executor(
                        None, _synth_and_play, remainder
                    )

                elapsed = time.time() - t0
                ttft = (first_token - t0) if first_token else elapsed
                response_text = "".join(full_response)
                conversation.add_assistant(response_text)
                print(f"\n  [{elapsed:.1f}s total, TTFT {ttft:.2f}s, ASR {asr_time:.1f}s]", flush=True)
                print("\nListening...", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Gemma Realtime Speech Server — cascaded ASR + LLM + TTS pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full speech pipeline (requires microphone)
  %(prog)s

  # Text-only mode (type instead of speak)
  %(prog)s --text-only

  # Custom LLM backend
  %(prog)s --llm-url http://localhost:11434

  # Custom voice and speed
  %(prog)s --voice af_sarah --speed 1.1
""",
    )
    parser.add_argument(
        "--llm-url", default="http://localhost:8741",
        help="URL of the MLX server (default: http://localhost:8741)",
    )
    parser.add_argument(
        "--whisper-model", default="mlx-community/whisper-small-mlx",
        help="Whisper model for ASR (default: mlx-community/whisper-small-mlx)",
    )
    parser.add_argument(
        "--tts", choices=["kokoro", "voxtral"], default="kokoro",
        help="TTS backend: kokoro (82M, Apache 2.0) or voxtral (4B, CC BY-NC) (default: kokoro)",
    )
    parser.add_argument(
        "--tts-precision", choices=["4bit", "6bit", "bf16"], default="6bit",
        help="Voxtral model precision (default: 6bit, best speed/quality for real-time)",
    )
    parser.add_argument(
        "--denoise-steps", type=int, default=4, choices=[2, 4, 6, 8],
        help="Voxtral acoustic flow-matching denoising steps (default: 4, original: 8). "
             "Lower = faster but may reduce quality. 4 gives ~1.9x speedup.",
    )
    parser.add_argument(
        "--draft-heads", default=None,
        help="Draft heads .safetensors for speculative decoding. If omitted with --tts voxtral, "
        "uses VOXTRAL_DRAFT_HEADS or auto-detects heads-libri.safetensors / heads.safetensors.",
    )
    parser.add_argument(
        "--no-draft-heads",
        action="store_true",
        help="Disable draft heads for Voxtral even if default weights exist",
    )
    parser.add_argument(
        "--voice", default=None,
        help="TTS voice (kokoro: af_bella, voxtral: cheerful_male). Auto-selected if omitted.",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="TTS speaking speed (default: 1.0)",
    )
    parser.add_argument(
        "--vad-threshold", type=float, default=VAD_THRESHOLD,
        help=f"VAD speech detection threshold (default: {VAD_THRESHOLD})",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="Maximum tokens per LLM response (default: 256)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="LLM temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=10,
        help="Maximum conversation turns to keep in context (default: 10)",
    )
    parser.add_argument(
        "--system-prompt", default=None,
        help="Custom system prompt (default: voice assistant prompt)",
    )
    parser.add_argument(
        "--text-only", action="store_true",
        help="Text-only mode — type instead of speak (no mic/speaker needed)",
    )
    parser.add_argument(
        "--no-mic", action="store_true",
        help="Alias for --text-only",
    )
    args = parser.parse_args()
    if args.no_mic:
        args.text_only = True

    from draft_heads_resolve import resolve_draft_heads_path

    if args.tts == "voxtral":
        args.draft_heads = None if args.no_draft_heads else resolve_draft_heads_path(args.draft_heads)
    else:
        args.draft_heads = None

    asyncio.run(run_speech_pipeline(args))


if __name__ == "__main__":
    main()
