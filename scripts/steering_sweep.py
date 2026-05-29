#!/usr/bin/env python3
"""Phase 2 — activation-steering α-sweep validation (the SOTA evidence).

For each trait, sweep the steering coefficient α through the live server's
`steering` field and measure:
  (a) MONOTONICITY — does a direct numeric proxy for the trait move
      monotonically with α? (formality↓contractions, verbosity↑length,
      warmth↑warm-markers, humor↑laugh-markers)
  (b) CAPABILITY — do base instruction/reasoning probes still produce correct
      answers at each α? (the lora-scale-default-or-die lesson: over-steering
      destroys instruction-following).
Emits a safe α range per trait: the largest |α| that still shifts the proxy
WITHOUT breaking capability.

Run against a steering-enabled server (default :8742, NOT prod :8741).

Usage: scripts/steering_sweep.py [--url http://127.0.0.1:8742/v1] [--trait formality]
"""
import argparse
import json
import sys
import urllib.request

SYSTEM = ("You are Seth. Reply in his exact texting voice — natural, concise, real "
          "opinions and warmth, lowercase-leaning, no AI-assistant tells.")

# Neutral prompts that leave room for the trait to express.
PROMPTS = ["how's the project going", "what should we do this weekend",
           "did you hear about the news", "tell me about your day",
           "what do you think we should order"]

# α=1 already shifts noticeably and α≥2 over-steers to degenerate output (the
# lora-scale lesson), so map the boundary finely in [-2, 2].
ALPHAS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

# Capability probes: (prompt, substring that MUST appear in a correct answer).
CAP_PROBES = [
    ("What is 17 times 23? Reply with just the number.", "391"),
    ("Translate 'good morning' to Spanish. Reply with just the translation.", "buenos"),
    ("Complete: the capital of France is ____. One word.", "paris"),
]


def gen(url, system, user, steering, max_tokens=120):
    body = {"model": "gemma-4-31b-it-4bit",
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens, "temperature": 0.0}
    if steering:
        body["steering"] = steering
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    return (d["choices"][0]["message"]["content"] or "").strip()


def _words(t):
    return [w for w in "".join(c.lower() if (c.isalnum() or c.isspace()) else " "
                                for c in t).split()]


def proxy(trait, text):
    """Direct numeric proxy for the trait level in `text` (higher = more)."""
    w = _words(text)
    n = max(1, len(w))
    if trait == "verbosity":
        return float(len(text))  # longer = more verbose
    if trait == "formality":
        # MORE formal = FEWER contractions/casual markers. Return formality level
        # = 1 - casual_density, so it should rise with +α.
        casual = sum(1 for x in w if x in ("lol", "haha", "yeah", "u", "ur", "lemme",
                                           "gonna", "wanna", "nah", "ya", "hbu", "rn"))
        contractions = text.count("'")
        return 1.0 - (casual + contractions) / n
    if trait == "warmth":
        warm = sum(1 for x in w if x in ("love", "happy", "hug", "care", "sweet",
                                         "miss", "appreciate", "glad", "wonderful", "dear"))
        return warm / n + text.count("!") / n
    if trait == "humor":
        funny = sum(1 for x in w if x in ("lol", "lmao", "lmao", "haha", "hah",
                                          "joke", "funny", "wheezing", "plot"))
        return funny / n
    return 0.0


def capability_ok(url, alpha, trait):
    """Run capability probes at this α; return (passed, total)."""
    passed = 0
    for q, must in CAP_PROBES:
        try:
            ans = gen(url, "You are a precise assistant.", q, {trait: alpha} if alpha else None,
                      max_tokens=60).lower()
        except Exception:  # noqa: BLE001
            ans = ""
        if must in ans:
            passed += 1
    return passed, len(CAP_PROBES)


def is_monotonic(pairs):
    """pairs: [(alpha, proxy)] sorted by alpha. Return 'up'/'down'/'none'."""
    ys = [p for _, p in sorted(pairs)]
    up = all(b >= a - 1e-9 for a, b in zip(ys, ys[1:]))
    down = all(b <= a + 1e-9 for a, b in zip(ys, ys[1:]))
    return "up" if up and not down else "down" if down and not up else "none"


def sweep_trait(url, trait):
    print(f"=== {trait} ===", flush=True)
    rows = []
    for a in ALPHAS:
        vals = []
        for p in PROMPTS:
            try:
                txt = gen(url, SYSTEM, p, {trait: a} if a else None)
            except Exception as ex:  # noqa: BLE001
                txt = ""
                print(f"  [warn] α={a} {p[:20]!r}: {ex}", flush=True)
            vals.append(proxy(trait, txt))
        mean_proxy = sum(vals) / len(vals)
        cap_pass, cap_tot = capability_ok(url, a, trait)
        rows.append({"alpha": a, "proxy": round(mean_proxy, 4),
                     "capability": f"{cap_pass}/{cap_tot}", "cap_ok": cap_pass == cap_tot})
        print(f"  α={a:+.1f}  proxy={mean_proxy:.4f}  capability={cap_pass}/{cap_tot}", flush=True)
    mono = is_monotonic([(r["alpha"], r["proxy"]) for r in rows])
    safe = [r["alpha"] for r in rows if r["cap_ok"]]
    safe_range = [min(safe), max(safe)] if safe else []
    print(f"  -> monotonic: {mono} | capability-safe α range: {safe_range}\n", flush=True)
    return {"trait": trait, "monotonic": mono, "safe_alpha_range": safe_range, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description="Activation-steering α-sweep (Phase 2)")
    ap.add_argument("--url", default="http://127.0.0.1:8742/v1")
    ap.add_argument("--trait", action="append",
                    choices=["formality", "verbosity", "warmth", "humor"])
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()
    traits = args.trait or ["formality", "verbosity", "warmth", "humor"]
    results = [sweep_trait(args.url, t) for t in traits]
    print(json.dumps(results, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as fh:
            json.dump(results, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
