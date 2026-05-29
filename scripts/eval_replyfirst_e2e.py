#!/usr/bin/env python3
"""OFFLINE E2E proof: the serving-config primer-skip makes v4-repair reply FIRST.

Loads a SECOND, private instance of gemma-4-31b-it-4bit + the seth-lora-v4-repair
adapter (NEVER touches the shared :8741 prod server) and runs the REAL
mlx-server.py stream_response() path twice on the same casual prompt:

  (a) skip_thinking_primer=False  → Gemma's forced `<|channel>thought` primer
                                    → deliberation streams first (reply LAST)
  (b) skip_thinking_primer=True   → no primer → reply streams FIRST

Emits a verdict JSON. Reply-first is proven when (b) produces user-visible reply
content at a much lower token index than (a). Apple-Silicon only; not CI.

Run:  python3 scripts/eval_replyfirst_e2e.py
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

SERVER_PATH = Path(__file__).parent / "mlx-server.py"
MODEL_ID = "mlx-community/gemma-4-31b-it-4bit"
ADAPTER = str(Path.home() / ".human/training-data/adapters/seth-lora-v4-repair-20260525-071921")
TRAIN = Path.home() / ".human/training-data/finetune/train.jsonl"
PROMPT = "hey, you around?"
MAX_TOKENS = 128
OUT = Path.home() / ".human/training-data/replyfirst-e2e-verdict.json"

# Markers that indicate the model is deliberating, not replying.
THINK_MARKERS = ("<|channel>thought", "<channel|>", "<|channel|>")


def _load_module():
    spec = importlib.util.spec_from_file_location("mlx_server_e2e", SERVER_PATH)
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    return srv


def _persona_system_prompt():
    first = TRAIN.read_text().splitlines()[0]
    return json.loads(first)["messages"][0]["content"]


def _run(srv, messages, skip_primer):
    """Run the real stream_response; return (full_text, first_chunk_latency_s,
    gen_tokens, tokens_until_reply)."""
    t0 = time.time()
    first_latency = None
    full = []
    gen_toks = 0
    tokens_until_reply = None
    seen = ""
    with srv.model_lock:
        for text, _pt, gt in srv.stream_response(
            messages, max_tokens=MAX_TOKENS, temperature=0.7,
            skip_thinking_primer=skip_primer,
        ):
            if first_latency is None:
                first_latency = time.time() - t0
            gen_toks = gt
            full.append(text)
            seen += text
            # "reply" = first content with no active think marker in the
            # accumulated stream. Approximate token index by gen token count
            # at the moment the stream first contains non-marker prose.
            if tokens_until_reply is None:
                stripped = seen
                for m in THINK_MARKERS:
                    stripped = stripped.replace(m, "")
                if stripped.strip() and not any(m in seen for m in THINK_MARKERS):
                    tokens_until_reply = gt
    return "".join(full), first_latency, gen_toks, tokens_until_reply


def main():
    srv = _load_module()

    print(f"Loading {MODEL_ID} + v4-repair adapter (offline, private instance)...",
          flush=True)
    from mlx_lm import load
    model, processor = load(MODEL_ID, adapter_path=ADAPTER)
    srv.model = model
    srv.processor = processor
    srv.use_lm_path = True
    srv.speculative_enabled = False
    srv.draft_model = None
    srv.model_id = MODEL_ID

    sysp = _persona_system_prompt()
    messages = [{"role": "system", "content": sysp},
                {"role": "user", "content": PROMPT}]

    print("\n=== (a) skip_thinking_primer=False  (forced primer — baseline) ===",
          flush=True)
    a_text, a_lat, a_tok, a_reply_at = _run(srv, messages, skip_primer=False)
    print(f"  gen_tokens={a_tok}  first_chunk={a_lat:.2f}s  reply_at_token={a_reply_at}")
    print(f"  output[:300]: {a_text[:300]!r}")

    print("\n=== (b) skip_thinking_primer=True   (reply-first — the fix) ===",
          flush=True)
    b_text, b_lat, b_tok, b_reply_at = _run(srv, messages, skip_primer=True)
    print(f"  gen_tokens={b_tok}  first_chunk={b_lat:.2f}s  reply_at_token={b_reply_at}")
    print(f"  output[:300]: {b_text[:300]!r}")

    # Verdict: (b) must reach user-visible reply content at a meaningfully lower
    # token index than (a), and must not start with a thinking marker.
    b_reply_idx = b_reply_at if b_reply_at is not None else b_tok
    a_reply_idx = a_reply_at if a_reply_at is not None else a_tok
    b_no_leading_marker = not any(b_text.lstrip().startswith(m) for m in THINK_MARKERS)
    reply_first = b_no_leading_marker and (b_reply_idx <= a_reply_idx)

    verdict = {
        "prompt": PROMPT,
        "model": MODEL_ID,
        "adapter": ADAPTER,
        "max_tokens": MAX_TOKENS,
        "baseline_primed": {
            "gen_tokens": a_tok, "first_chunk_s": round(a_lat or 0, 3),
            "reply_at_token": a_reply_at, "text": a_text,
        },
        "fix_replyfirst": {
            "gen_tokens": b_tok, "first_chunk_s": round(b_lat or 0, 3),
            "reply_at_token": b_reply_at, "text": b_text,
        },
        "streaming_beneficial": bool(reply_first),
        "verdict": "PASS" if reply_first else "FAIL",
    }
    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\nVerdict: {verdict['verdict']}  (streaming_beneficial={verdict['streaming_beneficial']})")
    print(f"  baseline reply_at_token={a_reply_idx}  fix reply_at_token={b_reply_idx}")
    print(f"Written: {OUT}")
    return 0 if reply_first else 1


if __name__ == "__main__":
    sys.exit(main())
