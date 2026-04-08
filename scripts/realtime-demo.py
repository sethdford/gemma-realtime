#!/usr/bin/env python3
"""
Real-Time Conversation Demo — Prove it works!

Streams a multi-turn conversation against a local LLM backend,
measuring TTFT, tokens/sec, and real-time factor for each response.
Shows tokens appearing in real-time just like Gemini Live.

Usage:
    python3 scripts/realtime-demo.py                    # auto-detect backend
    python3 scripts/realtime-demo.py --backend ollama   # force Ollama
    python3 scripts/realtime-demo.py --backend mlx      # force MLX
    python3 scripts/realtime-demo.py --interactive      # interactive chat mode
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error

BACKENDS = {
    "ollama": {"url": "http://localhost:11434/v1/chat/completions", "name": "Ollama"},
    "mlx":    {"url": "http://localhost:8742/v1/chat/completions",  "name": "MLX Server"},
    "llama":  {"url": "http://localhost:8741/v1/chat/completions",  "name": "llama.cpp"},
}

CONVERSATION_PROMPTS = [
    "Hey! What's the best way to spend a rainy Sunday afternoon?",
    "That sounds great. Do you think reading fiction or nonfiction is better for relaxation?",
    "Interesting take. What's a book you'd recommend to someone who's never really been into reading?",
    "Nice. Switching topics — if you could have dinner with anyone in history, who would it be and why?",
    "Ha, good choice. One more — what's your hot take that most people would disagree with?",
]

VOICE_PROMPTS = [
    "Hey, how's it going?",
    "What should I make for dinner tonight? Something quick.",
    "Good idea. What about dessert?",
    "Perfect. Hey, tell me something interesting I probably don't know.",
    "Whoa, that's wild. Alright, thanks! Talk later.",
]


def check_backend(url):
    """Check if a backend is responding."""
    try:
        base = url.rsplit("/v1/", 1)[0]
        for path in ["/v1/models", "/api/tags", "/health", "/"]:
            try:
                req = urllib.request.Request(base + path, method="GET")
                req.add_header("Accept", "application/json")
                resp = urllib.request.urlopen(req, timeout=2)
                if resp.status == 200:
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False


def discover_model(url):
    """Try to discover the model name from the backend."""
    try:
        base = url.rsplit("/v1/", 1)[0]
        req = urllib.request.Request(base + "/v1/models", method="GET")
        req.add_header("Accept", "application/json")
        resp = urllib.request.urlopen(req, timeout=2)
        data = json.loads(resp.read())
        if "data" in data and data["data"]:
            return data["data"][0].get("id", "default")
    except Exception:
        pass
    return None


def stream_chat(url, model, messages, max_tokens=256):
    """Send a chat completion request and stream the response."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")

    t_start = time.perf_counter()
    first_token_time = None
    tokens = []
    full_text = ""

    try:
        resp = urllib.request.urlopen(req, timeout=60)
        remainder = ""

        while True:
            raw = resp.read(4096)
            if not raw:
                break

            text = remainder + raw.decode("utf-8", errors="replace")
            lines = text.split("\n")
            remainder = lines[-1]

            for line in lines[:-1]:
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    event = json.loads(data_str)
                    choices = event.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            now = time.perf_counter()
                            if first_token_time is None:
                                first_token_time = now
                            tokens.append(now)
                            full_text += content
                            sys.stdout.write(content)
                            sys.stdout.flush()
                except json.JSONDecodeError:
                    continue

    except urllib.error.URLError as e:
        return None, f"Connection error: {e}"

    t_end = time.perf_counter()
    sys.stdout.write("\n")
    sys.stdout.flush()

    if not tokens:
        return None, "No tokens received"

    ttft = (first_token_time - t_start) * 1000
    total_time = t_end - t_start
    n_tokens = len(tokens)
    gen_time = t_end - first_token_time if first_token_time else total_time
    tps = n_tokens / gen_time if gen_time > 0 else 0
    ms_per_tok = (gen_time / n_tokens * 1000) if n_tokens > 0 else 0

    # Real-time factor: < 1.0 means faster than speech (150 wpm ≈ 3.3 tok/s)
    speech_rate_tps = 3.3
    rtf = speech_rate_tps / tps if tps > 0 else float("inf")

    stats = {
        "ttft_ms": ttft,
        "tokens": n_tokens,
        "tps": tps,
        "ms_per_tok": ms_per_tok,
        "total_ms": total_time * 1000,
        "rtf": rtf,
        "text": full_text,
    }
    return stats, None


def print_stats(stats, turn_num):
    """Print performance metrics for a turn."""
    rt = "REAL-TIME" if stats["rtf"] < 0.2 else "MARGINAL" if stats["rtf"] < 1.0 else "TOO SLOW"
    color = "\033[32m" if rt == "REAL-TIME" else "\033[33m" if rt == "MARGINAL" else "\033[31m"
    reset = "\033[0m"

    print(f"\n  {color}╭─ Turn {turn_num} ────────────────────────────────────╮{reset}")
    print(f"  {color}│{reset} TTFT: {stats['ttft_ms']:>7.0f} ms    Tokens: {stats['tokens']:>4d}         {color}│{reset}")
    print(f"  {color}│{reset} Speed: {stats['tps']:>6.1f} tok/s  Per-tok: {stats['ms_per_tok']:>5.1f} ms   {color}│{reset}")
    print(f"  {color}│{reset} RTF:  {stats['rtf']:>7.4f}       Status: {rt:<11s}  {color}│{reset}")
    print(f"  {color}╰──────────────────────────────────────────────╯{reset}\n")


def run_conversation(url, model, prompts, system_prompt=None):
    """Run a full multi-turn conversation with metrics."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    all_stats = []

    for i, prompt in enumerate(prompts):
        turn = i + 1
        print(f"\033[1;36m  You ({turn}/{len(prompts)}):\033[0m {prompt}\n")

        messages.append({"role": "user", "content": prompt})

        print(f"\033[1;33m  AI:\033[0m ", end="", flush=True)
        stats, error = stream_chat(url, model, messages, max_tokens=200)

        if error:
            print(f"\n\033[31m  Error: {error}\033[0m\n")
            break

        messages.append({"role": "assistant", "content": stats["text"]})
        print_stats(stats, turn)
        all_stats.append(stats)

    return all_stats


def run_interactive(url, model):
    """Interactive chat mode — type messages, see real-time responses."""
    messages = [{"role": "system", "content": "You are a helpful, witty, and concise assistant. Keep responses short and conversational."}]

    print("\n\033[1mInteractive mode — type messages, Ctrl+C to quit\033[0m\n")

    while True:
        try:
            user_input = input("\033[1;36m  You:\033[0m ")
            if not user_input.strip():
                continue
        except (KeyboardInterrupt, EOFError):
            print("\n\n\033[2m  Goodbye!\033[0m\n")
            break

        messages.append({"role": "user", "content": user_input})
        print(f"\033[1;33m  AI:\033[0m ", end="", flush=True)
        stats, error = stream_chat(url, model, messages, max_tokens=300)

        if error:
            print(f"\n\033[31m  Error: {error}\033[0m")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": stats["text"]})
        rt = "REAL-TIME" if stats["rtf"] < 0.2 else "ok"
        color = "\033[32m" if rt == "REAL-TIME" else "\033[33m"
        print(f"  \033[2m[{stats['tps']:.0f} tok/s, TTFT {stats['ttft_ms']:.0f}ms, {color}{rt}\033[0m\033[2m]\033[0m\n")


def main():
    parser = argparse.ArgumentParser(description="Real-time conversation demo")
    parser.add_argument("--backend", choices=["ollama", "mlx", "llama"], help="Force a specific backend")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive chat mode")
    parser.add_argument("--voice", action="store_true", help="Use short voice-style prompts")
    parser.add_argument("--url", help="Custom API URL")
    args = parser.parse_args()

    print()
    print("\033[1m╔══════════════════════════════════════════════════════════════╗\033[0m")
    print("\033[1m║  gemma-realtime — Live Conversation Demo                    ║\033[0m")
    print("\033[1m║  Streaming tokens in real-time on Apple Silicon             ║\033[0m")
    print("\033[1m╚══════════════════════════════════════════════════════════════╝\033[0m")
    print()

    # Auto-detect or use specified backend
    if args.url:
        url = args.url
        backend_name = "Custom"
    elif args.backend:
        url = BACKENDS[args.backend]["url"]
        backend_name = BACKENDS[args.backend]["name"]
    else:
        url = None
        backend_name = None
        for key, cfg in BACKENDS.items():
            print(f"  Checking {cfg['name']}... ", end="", flush=True)
            if check_backend(cfg["url"]):
                url = cfg["url"]
                backend_name = cfg["name"]
                print(f"\033[32mONLINE\033[0m")
                break
            else:
                print(f"\033[2moffline\033[0m")

    if not url:
        print("\n\033[31m  No backend found! Start one of:\033[0m")
        print("    ollama serve")
        print("    python3 scripts/mlx-server.py --model mlx-community/gemma-4-e4b-it-4bit")
        print("    bash scripts/llamacpp-serve.sh")
        sys.exit(1)

    # Discover model
    model = args.model
    if not model:
        model = discover_model(url)
        if not model:
            if "ollama" in url or "11434" in url:
                model = "gemma3:4b"
            else:
                model = "default"

    print(f"\n  Backend: \033[1m{backend_name}\033[0m")
    print(f"  Model:   \033[1m{model}\033[0m")
    print(f"  URL:     {url}")
    print()

    if args.interactive:
        run_interactive(url, model)
        return

    # Run the conversation demo
    prompts = VOICE_PROMPTS if args.voice else CONVERSATION_PROMPTS
    style = "voice" if args.voice else "conversation"

    print(f"  Running {len(prompts)}-turn {style} demo...\n")
    print("  ═══════════════════════════════════════════════════════════\n")

    system = "You are a friendly, witty assistant. Keep responses concise — 1-3 sentences max for voice mode." if args.voice else "You are a helpful, engaging assistant. Keep responses conversational and interesting, 2-4 sentences."

    all_stats = run_conversation(url, model, prompts, system_prompt=system)

    if not all_stats:
        print("\033[31m  No successful turns.\033[0m")
        sys.exit(1)

    # Aggregate results
    print("  ═══════════════════════════════════════════════════════════")
    print()

    avg_ttft = sum(s["ttft_ms"] for s in all_stats) / len(all_stats)
    avg_tps = sum(s["tps"] for s in all_stats) / len(all_stats)
    avg_rtf = sum(s["rtf"] for s in all_stats) / len(all_stats)
    total_tokens = sum(s["tokens"] for s in all_stats)
    p50_tps = sorted(s["tps"] for s in all_stats)[len(all_stats) // 2]

    is_realtime = avg_rtf < 0.2
    verdict_color = "\033[32m" if is_realtime else "\033[33m"
    verdict = "REAL-TIME" if is_realtime else "NOT REAL-TIME"

    print(f"  \033[1m╔══════════════════════════════════════════════════════╗\033[0m")
    print(f"  \033[1m║  CONVERSATION RESULTS                                ║\033[0m")
    print(f"  \033[1m╠══════════════════════════════════════════════════════╣\033[0m")
    print(f"  \033[1m║\033[0m  Turns:         {len(all_stats):>5d}                                \033[1m║\033[0m")
    print(f"  \033[1m║\033[0m  Total tokens:  {total_tokens:>5d}                                \033[1m║\033[0m")
    print(f"  \033[1m║\033[0m  Avg TTFT:      {avg_ttft:>5.0f} ms                              \033[1m║\033[0m")
    print(f"  \033[1m║\033[0m  Avg TPS:       {avg_tps:>5.1f} tok/s                           \033[1m║\033[0m")
    print(f"  \033[1m║\033[0m  P50 TPS:       {p50_tps:>5.1f} tok/s                           \033[1m║\033[0m")
    print(f"  \033[1m║\033[0m  Avg RTF:       {avg_rtf:>5.4f}                               \033[1m║\033[0m")
    print(f"  \033[1m║\033[0m                                                    \033[1m║\033[0m")
    print(f"  \033[1m║\033[0m  Verdict:  {verdict_color}\033[1m★ {verdict} ★\033[0m                          \033[1m║\033[0m")
    print(f"  \033[1m║\033[0m                                                    \033[1m║\033[0m")
    if is_realtime:
        margin = 0.2 / avg_rtf
        print(f"  \033[1m║\033[0m  {avg_tps:.0f} tok/s is {margin:.0f}x faster than human speech.        \033[1m║\033[0m")
        print(f"  \033[1m║\033[0m  This is Gemini Live territory — running locally.  \033[1m║\033[0m")
    print(f"  \033[1m╚══════════════════════════════════════════════════════╝\033[0m")
    print()


if __name__ == "__main__":
    main()
