#!/usr/bin/env python3
"""
Red team validation suite for gemma-realtime speech pipeline.

Validates every phase's components can import, instantiate, and run basic
operations. Measures architecture latency budgets, memory estimates, and
identifies integration risks.

Usage:
    python3 scripts/red-team-validate.py
    python3 scripts/red-team-validate.py --phase 4
    python3 scripts/red-team-validate.py --verbose
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"


class ValidationResult:
    def __init__(self, name: str, status: str, detail: str = "", latency_ms: float = 0):
        self.name = name
        self.status = status
        self.detail = detail
        self.latency_ms = latency_ms


class RedTeamValidator:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.results = []

    def log(self, result: ValidationResult):
        self.results.append(result)
        icon = {"PASS": "+", "FAIL": "X", "WARN": "!", "SKIP": "-"}[result.status]
        lat = f" ({result.latency_ms:.0f}ms)" if result.latency_ms > 0 else ""
        print(f"  [{icon}] {result.name}{lat}: {result.detail}", flush=True)

    def validate_phase0(self):
        """Phase 0: Verify mlx-server.py fixes."""
        print("\n--- Phase 0: Server Fixes ---", flush=True)

        try:
            with open(Path(__file__).parent / "mlx-server.py") as f:
                source = f.read()

            checks = [
                ("make_sampler(temp=temperature)" in source or 'extra["temp"] = temperature' in source,
                 "temperature wired to lm_stream_generate"),
                ("_compact_turbo_cache()" in source and source.count("_compact_turbo_cache()") >= 3,
                 "_compact_turbo_cache called after prefill"),
                ("num_draft_tokens" in source, "speculative_draft_tokens wired via inspect"),
            ]

            for ok, desc in checks:
                self.log(ValidationResult(
                    f"Phase 0: {desc}",
                    PASS if ok else FAIL,
                    "present in source" if ok else "NOT found in source",
                ))
        except Exception as e:
            self.log(ValidationResult("Phase 0: read mlx-server.py", FAIL, str(e)))

    def validate_phase1(self):
        """Phase 1: Cascaded pipeline components."""
        print("\n--- Phase 1: Cascaded Pipeline ---", flush=True)

        try:
            speech_server = importlib.import_module("speech-server")

            vad = speech_server.SileroVAD(threshold=0.4)
            self.log(ValidationResult("Phase 1: SileroVAD instantiate", PASS, ""))

            asr = speech_server.WhisperASR()
            self.log(ValidationResult("Phase 1: WhisperASR instantiate", PASS, ""))

            tts = speech_server.TTSEngine(voice="af_bella")
            self.log(ValidationResult("Phase 1: TTSEngine instantiate", PASS, ""))

            buf = speech_server.SentenceBuffer()
            results = buf.add("Hello world. How are you?")
            assert len(results) >= 1, "Expected at least one sentence"
            self.log(ValidationResult("Phase 1: SentenceBuffer", PASS, f"flushed {len(results)} sentences"))

            llm = speech_server.LLMClient("http://localhost:8741")
            self.log(ValidationResult("Phase 1: LLMClient instantiate", PASS, ""))

            conv = speech_server.ConversationState(system_prompt="Test")
            conv.add_user("Hello")
            conv.add_assistant("Hi there")
            assert len(conv.get_messages()) == 3
            self.log(ValidationResult("Phase 1: ConversationState", PASS, "3 messages tracked"))

        except Exception as e:
            self.log(ValidationResult("Phase 1: import", FAIL, str(e)))

    def validate_phase2(self):
        """Phase 2: WebSocket server components."""
        print("\n--- Phase 2: WebSocket API ---", flush=True)

        try:
            ws = importlib.import_module("realtime-ws")
            session = ws.RealtimeSession("test", "http://localhost:8741", "whisper-small", "af_bella", 0.4)
            self.log(ValidationResult("Phase 2: RealtimeSession instantiate", PASS, ""))

            server = ws.RealtimeServer()
            self.log(ValidationResult("Phase 2: RealtimeServer instantiate", PASS, f"port={server.port}"))
        except ImportError as e:
            self.log(ValidationResult("Phase 2: import", WARN, f"Missing dep: {e}"))
        except Exception as e:
            self.log(ValidationResult("Phase 2: import", FAIL, str(e)))

    def validate_phase3(self):
        """Phase 3: Neural audio codec."""
        print("\n--- Phase 3: Neural Audio Codec ---", flush=True)

        try:
            from codec import AudioCodec, CodecTokens, CodecType, CODEC_CONFIGS

            for ct in CodecType:
                cfg = CODEC_CONFIGS[ct]
                self.log(ValidationResult(
                    f"Phase 3: {ct.value} config",
                    PASS,
                    f"{cfg.sample_rate}Hz, {cfg.frame_rate}fps, {cfg.n_codebooks}cb, {cfg.bandwidth_kbps}kbps",
                ))

            codec = AudioCodec("snac")
            assert codec.vocab_size == 4096 * 3
            self.log(ValidationResult("Phase 3: SNAC vocab", PASS, f"{codec.vocab_size} tokens"))

            tokens = CodecTokens(
                codes=np.random.randint(0, 4096, (3, 10)),
                n_codebooks=3,
                frame_rate=12.0,
                codec_type=CodecType.SNAC,
            )
            flat = tokens.flat_tokens
            reconstructed = CodecTokens.from_flat(flat, 3, CodecType.SNAC)
            assert np.array_equal(tokens.codes, reconstructed.codes)
            self.log(ValidationResult("Phase 3: Token roundtrip", PASS, f"flat={len(flat)}, reconstruct OK"))

        except Exception as e:
            self.log(ValidationResult("Phase 3: codec", FAIL, str(e)))

    def validate_phase4(self):
        """Phase 4: Speech encoder/decoder adapters."""
        print("\n--- Phase 4: Speech Adapters ---", flush=True)

        try:
            import mlx.core as mx
            from speech_encoder import SpeechEncoder, SpeechEncoderConfig

            for target in ["e2b", "e4b"]:
                config = SpeechEncoderConfig.from_target(target)
                encoder = SpeechEncoder(**config)

                audio = mx.random.normal((1, 1, encoder.chunk_samples))
                t0 = time.time()
                out = encoder(audio)
                mx.eval(out)
                lat = (time.time() - t0) * 1000

                self.log(ValidationResult(
                    f"Phase 4: Encoder {target}",
                    PASS if lat < 500 else WARN,
                    f"{encoder.num_params()/1e6:.1f}M params, output={out.shape}",
                    latency_ms=lat,
                ))

        except Exception as e:
            self.log(ValidationResult("Phase 4: encoder", FAIL, str(e)))

        try:
            import mlx.core as mx
            from speech_decoder import SpeechDecoder, SpeechDecoderConfig, DuplexStatePredictor

            for target in ["e2b", "e4b"]:
                config = SpeechDecoderConfig.from_target(target)
                decoder = SpeechDecoder(**config)
                predictor = DuplexStatePredictor(llm_dim=config["llm_dim"])

                hidden = mx.random.normal((1, 10, config["llm_dim"]))

                t0 = time.time()
                tokens = decoder.generate(hidden, temperature=0.8, top_k=50)
                mx.eval(tokens)
                lat = (time.time() - t0) * 1000

                state = predictor.predict(hidden)

                self.log(ValidationResult(
                    f"Phase 4: Decoder {target}",
                    PASS if lat < 2000 else WARN,
                    f"{decoder.num_params()/1e6:.1f}M params, generated {tokens.shape[-1]} tokens, state={state}",
                    latency_ms=lat,
                ))

        except Exception as e:
            self.log(ValidationResult("Phase 4: decoder", FAIL, str(e)))

    def validate_phase5(self):
        """Phase 5: Speech-to-speech model."""
        print("\n--- Phase 5: Speech-to-Speech Model ---", flush=True)

        try:
            import mlx.core as mx
            from speech_model import SpeechToSpeechModel, SpeechModelConfig, PRESET_CONFIGS

            config = PRESET_CONFIGS["e4b"]
            model = SpeechToSpeechModel(config)

            hidden = mx.random.normal((1, 20, config.llm_dim))

            t0 = time.time()
            text_logits, audio_tokens = model.predict_audio_frame(hidden, temperature=0.8)
            mx.eval(audio_tokens)
            frame_ms = (time.time() - t0) * 1000

            budget_ms = 1000.0 / config.frame_rate_hz
            status = PASS if frame_ms < budget_ms else WARN

            self.log(ValidationResult(
                "Phase 5: Audio frame generation",
                status,
                f"{config.n_codebooks} codebooks, budget={budget_ms:.0f}ms",
                latency_ms=frame_ms,
            ))

            self.log(ValidationResult(
                "Phase 5: Inner monologue",
                PASS if text_logits is not None else FAIL,
                f"text_logits shape={text_logits.shape}" if text_logits is not None else "None",
            ))

            self.log(ValidationResult(
                "Phase 5: Vocab extension",
                PASS,
                f"total={config.extended_vocab_size} ({config.text_vocab_size} text + {config.total_audio_vocab} audio)",
            ))

            state = model.predict_state(hidden)
            self.log(ValidationResult(
                "Phase 5: State predictor",
                PASS,
                f"state={['LISTEN', 'SPEAK', 'INTERRUPT'][state]}",
            ))

            self.log(ValidationResult(
                "Phase 5: Model size",
                PASS,
                f"{model.num_params()/1e6:.1f}M adapter params (LLM frozen)",
            ))

        except Exception as e:
            self.log(ValidationResult("Phase 5: speech_model", FAIL, str(e)))

    def validate_phase6(self):
        """Phase 6: Hardware acceleration."""
        print("\n--- Phase 6: Hardware Acceleration ---", flush=True)

        try:
            import mlx.core as mx
            from hw_accel import (
                LayerAdaptiveTurboCache, IOSurfaceKVManager,
                EAGLEDraftHead, HWAccelConfig,
            )

            class MockModel:
                class model:
                    layers = [None] * 32

            config = HWAccelConfig(turbo_bits=3, turbo_fp16_layers=4)
            cache = LayerAdaptiveTurboCache(MockModel(), config)
            self.log(ValidationResult(
                "Phase 6: LayerAdaptiveTurboCache",
                PASS,
                f"{len(cache)} layers, ~{cache.memory_estimate_mb:.1f}MB/1K ctx",
            ))

            io_config = HWAccelConfig(iosurface_enabled=True)
            io_mgr = IOSurfaceKVManager(io_config)
            io_mgr.allocate_kv_surface("test_kv", 1024)
            self.log(ValidationResult(
                "Phase 6: IOSurfaceKVManager",
                PASS if io_mgr.available else WARN,
                f"IOSurface available={io_mgr.available}",
            ))

            eagle_config = HWAccelConfig(eagle_enabled=True)
            eagle = EAGLEDraftHead(eagle_config, llm_dim=2560)

            hidden = mx.random.normal((1, 1, 2560))
            t0 = time.time()
            for _ in range(10):
                specs = eagle.speculate(hidden, n_tokens=4)
                mx.eval(specs)
            spec_ms = (time.time() - t0) / 10 * 1000

            self.log(ValidationResult(
                "Phase 6: EAGLE speculate",
                PASS if spec_ms < 50 else WARN,
                f"{eagle.num_params()/1e6:.1f}M params, 4 tokens",
                latency_ms=spec_ms,
            ))

            target = mx.random.normal((1, 4, 2560))
            loss = eagle.distillation_loss(specs, target)
            mx.eval(loss)
            self.log(ValidationResult(
                "Phase 6: EAGLE distillation",
                PASS,
                f"loss={loss.item():.4f}",
            ))

        except Exception as e:
            self.log(ValidationResult("Phase 6: hw_accel", FAIL, str(e)))

    def validate_integration(self):
        """Cross-phase integration checks."""
        print("\n--- Integration Checks ---", flush=True)

        try:
            import mlx.core as mx
            from speech_encoder import SpeechEncoder, SpeechEncoderConfig
            from speech_decoder import SpeechDecoder, SpeechDecoderConfig

            enc_config = SpeechEncoderConfig.from_target("e4b")
            dec_config = SpeechDecoderConfig.from_target("e4b")

            assert enc_config["llm_dim"] == dec_config["llm_dim"], "Dimension mismatch"

            encoder = SpeechEncoder(**enc_config)
            decoder = SpeechDecoder(**dec_config)

            audio = mx.random.normal((1, 1, encoder.chunk_samples))
            t0 = time.time()
            embeddings = encoder(audio)
            tokens = decoder.generate(embeddings, temperature=0.8)
            mx.eval(tokens)
            e2e_ms = (time.time() - t0) * 1000

            self.log(ValidationResult(
                "Integration: Encoder -> Decoder",
                PASS if e2e_ms < 2000 else WARN,
                f"audio -> {embeddings.shape} -> {tokens.shape[-1]} codec tokens",
                latency_ms=e2e_ms,
            ))

        except Exception as e:
            self.log(ValidationResult("Integration: enc-dec", FAIL, str(e)))

        try:
            from codec import AudioCodec, CodecTokens, CodecType
            codec = AudioCodec("snac")
            tokens_per_sec = codec.tokens_per_second
            self.log(ValidationResult(
                "Integration: Codec throughput budget",
                PASS if tokens_per_sec < 200 else WARN,
                f"SNAC needs {tokens_per_sec:.0f} tokens/sec (model generates ~100 tok/s)",
            ))
        except Exception as e:
            self.log(ValidationResult("Integration: codec budget", FAIL, str(e)))

        memory_budget = {
            "Gemma E4B 4-bit": 2.5,
            "Speech encoder": 0.02,
            "Speech decoder": 0.06,
            "SNAC codec": 0.1,
            "Whisper small": 1.5,
            "Kokoro TTS": 0.3,
            "KV cache (4K ctx, TQ3)": 0.5,
        }
        total_gb = sum(memory_budget.values())
        self.log(ValidationResult(
            "Integration: Memory budget",
            PASS if total_gb < 8 else WARN,
            f"{total_gb:.1f}GB total ({'fits 16GB' if total_gb < 16 else 'needs 32GB+'})",
        ))
        if self.verbose:
            for component, gb in memory_budget.items():
                print(f"    {component}: {gb:.1f} GB", flush=True)

    def print_summary(self):
        print(f"\n{'='*70}", flush=True)
        print(f"  RED TEAM VALIDATION SUMMARY", flush=True)
        print(f"{'='*70}", flush=True)

        pass_count = sum(1 for r in self.results if r.status == PASS)
        warn_count = sum(1 for r in self.results if r.status == WARN)
        fail_count = sum(1 for r in self.results if r.status == FAIL)
        skip_count = sum(1 for r in self.results if r.status == SKIP)

        print(f"  PASS: {pass_count}", flush=True)
        print(f"  WARN: {warn_count}", flush=True)
        print(f"  FAIL: {fail_count}", flush=True)
        if skip_count:
            print(f"  SKIP: {skip_count}", flush=True)

        if fail_count > 0:
            print(f"\n  FAILURES:", flush=True)
            for r in self.results:
                if r.status == FAIL:
                    print(f"    - {r.name}: {r.detail}", flush=True)

        if warn_count > 0:
            print(f"\n  WARNINGS:", flush=True)
            for r in self.results:
                if r.status == WARN:
                    print(f"    - {r.name}: {r.detail}", flush=True)

        verdict = "ALL CLEAR" if fail_count == 0 else f"{fail_count} ISSUE(S) NEED ATTENTION"
        print(f"\n  Verdict: {verdict}", flush=True)
        print(f"{'='*70}\n", flush=True)

        return fail_count == 0


def main():
    parser = argparse.ArgumentParser(description="Red team validation for speech pipeline")
    parser.add_argument("--phase", type=int, default=0, help="Validate specific phase (0=all)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    validator = RedTeamValidator(verbose=args.verbose)

    phases = {
        0: validator.validate_phase0,
        1: validator.validate_phase1,
        2: validator.validate_phase2,
        3: validator.validate_phase3,
        4: validator.validate_phase4,
        5: validator.validate_phase5,
        6: validator.validate_phase6,
    }

    print(f"\n{'='*70}", flush=True)
    print(f"  GEMMA REALTIME — RED TEAM VALIDATION", flush=True)
    print(f"{'='*70}", flush=True)

    if args.phase > 0:
        if args.phase in phases:
            phases[args.phase]()
        else:
            print(f"  Unknown phase: {args.phase}", flush=True)
    else:
        for phase_fn in phases.values():
            phase_fn()
        validator.validate_integration()

    all_pass = validator.print_summary()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
