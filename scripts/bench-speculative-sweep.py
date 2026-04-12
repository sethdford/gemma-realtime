#!/usr/bin/env python3
"""
Measure MLX server streaming tok/s and TTFT; pair with GEMMA_SPECULATIVE_TOKENS sweeps.

The server reads draft token count at startup — restart mlx-server.py between N values.
See guides/08-inference-sota-roadmap.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

SYSTEM_PROMPT = (
    "You are a conversational voice assistant. Keep responses concise and natural, "
    "as if speaking aloud. 1-3 sentences max."
)


def _get_json(url: str, path: str, timeout: float = 5.0) -> dict | None:
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, json.JSONDecodeError, ValueError, OSError):
        return None


def _bench_stream(url: str, model: str | None, max_tokens: int) -> dict:
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "tell me something interesting in two short sentences."},
        ],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    if model:
        payload["model"] = model

    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    first_token_time = None
    total_tokens = 0

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                line = line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            total_tokens += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return {"error": str(e)}

    t_end = time.perf_counter()
    ttft_ms = (first_token_time - t0) * 1000 if first_token_time else None
    gen_time = (t_end - first_token_time) if first_token_time else 0.0
    tps = total_tokens / gen_time if gen_time > 0 else 0.0

    return {
        "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
        "tokens": total_tokens,
        "tps": round(tps, 2),
        "total_ms": round((t_end - t0) * 1000, 1),
    }


def _discover_model(url: str) -> str | None:
    data = _get_json(url, "/v1/models")
    if not data:
        return None
    models = data.get("data") or []
    if models:
        return models[0].get("id")
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="MLX speculative / streaming snapshot benchmark")
    p.add_argument("--url", default="http://127.0.0.1:8741", help="MLX server base URL")
    p.add_argument("--rounds", type=int, default=6, help="Streaming requests (after warmup)")
    p.add_argument("--warmup", type=int, default=1, help="Warmup requests (discarded)")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--json", action="store_true", help="Single JSON object on stdout")
    args = p.parse_args()
    base = args.url.rstrip("/")

    health = _get_json(base, "/health")
    if not health:
        print(f"error: no JSON /health at {base}", file=sys.stderr)
        return 1

    tuning = health.get("inference_tuning") or {}
    model = _discover_model(base)

    for _ in range(max(0, args.warmup)):
        _bench_stream(base, model, args.max_tokens)

    rows = []
    for _ in range(args.rounds):
        rows.append(_bench_stream(base, model, args.max_tokens))

    errs = [r for r in rows if r.get("error")]
    if errs:
        print(errs[0].get("error"), file=sys.stderr)
        return 1

    ttfts = [r["ttft_ms"] for r in rows if r.get("ttft_ms") is not None]
    tps_list = [r["tps"] for r in rows if r.get("tps") is not None]

    out = {
        "url": base,
        "model": model,
        "inference_tuning": tuning,
        "rounds": args.rounds,
        "max_tokens": args.max_tokens,
        "ttft_ms_median": round(statistics.median(ttfts), 1) if ttfts else None,
        "tps_median": round(statistics.median(tps_list), 2) if tps_list else None,
        "samples": rows,
    }

    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    print(f"url={base} model={model!r}")
    print(f"inference_tuning={json.dumps(tuning, sort_keys=True)}")
    if ttfts:
        print(f"ttft_ms median={out['ttft_ms_median']} (n={len(ttfts)})")
    if tps_list:
        print(f"tps median={out['tps_median']} (n={len(tps_list)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
