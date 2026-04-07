#!/usr/bin/env python3
"""
Extract Facebook Messenger conversation pairs for LoRA fine-tuning.

Reads Facebook's data export (Download Your Information → JSON format)
and produces training pairs in the same format as extract-imessage.py.

Facebook export structure:
    your_facebook_activity/
    └── messages/
        └── inbox/
            ├── person1_abc123/
            │   └── message_1.json
            ├── person2_def456/
            │   └── message_1.json
            └── ...

Usage:
    python3 scripts/extract-facebook.py --export ~/Downloads/facebook-export
    python3 scripts/extract-facebook.py --export ~/Downloads/facebook-export --name "Your Name"
    python3 scripts/extract-facebook.py --export ~/Downloads/facebook-export --voice

Prerequisites:
    1. Go to Facebook → Settings → Your Information → Download Your Information
    2. Select: Format=JSON, Media Quality=Low, Date Range=All Time
    3. Request "Messages" only (fastest download)
    4. Download and unzip the archive
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

MAX_GAP_SECONDS = 3600
MIN_REPLY_LENGTH = 2

SKIP_PATTERNS = [
    "sent a photo",
    "sent a video",
    "sent a link",
    "sent an attachment",
    "liked a message",
    "reacted to your message",
    "changed the group photo",
    "set the emoji to",
    "named the group",
    "You sent a photo",
    "sent a GIF",
    "started a video chat",
    "missed your call",
    "audio_files",
]


def should_skip(text: str) -> bool:
    if not text or len(text.strip()) < MIN_REPLY_LENGTH:
        return True
    lower = text.lower()
    return any(p.lower() in lower for p in SKIP_PATTERNS)


def decode_facebook_text(text: str) -> str:
    """Facebook exports text in latin-1 encoded UTF-8. Decode properly."""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def find_inbox_dir(export_path: Path) -> Path | None:
    """Locate the messages/inbox directory in a Facebook export."""
    candidates = [
        export_path / "your_facebook_activity" / "messages" / "inbox",
        export_path / "messages" / "inbox",
        export_path / "inbox",
        export_path,
    ]
    for c in candidates:
        if c.is_dir() and any(c.iterdir()):
            jsons = list(c.rglob("message_*.json"))
            if jsons:
                return c
    return None


def extract_messages(inbox_dir: Path, your_name: str) -> list[dict]:
    """Extract all messages from the Facebook inbox directory."""
    messages = []
    conv_dirs = sorted(inbox_dir.iterdir())

    for conv_dir in conv_dirs:
        if not conv_dir.is_dir():
            continue

        chat_id = conv_dir.name
        msg_files = sorted(conv_dir.glob("message_*.json"))

        for msg_file in msg_files:
            try:
                with open(msg_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            for msg in data.get("messages", []):
                sender = decode_facebook_text(msg.get("sender_name", ""))
                content = msg.get("content", "")
                timestamp_ms = msg.get("timestamp_ms", 0)

                if not content or should_skip(content):
                    continue

                content = decode_facebook_text(content)
                is_from_me = sender.lower() == your_name.lower()
                unix_ts = timestamp_ms / 1000.0

                messages.append({
                    "is_from_me": is_from_me,
                    "text": content.strip(),
                    "timestamp": unix_ts,
                    "contact": sender,
                    "chat_id": chat_id,
                    "datetime": datetime.fromtimestamp(unix_ts).isoformat() if unix_ts > 0 else "",
                })

    messages.sort(key=lambda m: m["timestamp"])
    return messages


def group_by_chat(messages: list[dict]) -> dict[str, list[dict]]:
    chats = {}
    for msg in messages:
        chats.setdefault(msg["chat_id"], []).append(msg)
    for cid in chats:
        chats[cid].sort(key=lambda m: m["timestamp"])
    return chats


def build_conversation_windows(chat_messages: list[dict]) -> list[list[dict]]:
    windows = []
    current = []
    for msg in chat_messages:
        if current and msg["timestamp"] - current[-1]["timestamp"] > MAX_GAP_SECONDS:
            if current:
                windows.append(current)
            current = []
        current.append(msg)
    if current:
        windows.append(current)
    return windows


def extract_training_pairs(windows: list[list[dict]]) -> list[dict]:
    pairs = []
    for window in windows:
        for i, msg in enumerate(window):
            if msg["is_from_me"] and len(msg["text"]) >= MIN_REPLY_LENGTH:
                context_start = max(0, i - 5)
                context = window[context_start:i]
                if not context:
                    continue
                messages = []
                for ctx in context:
                    role = "assistant" if ctx["is_from_me"] else "user"
                    messages.append({"role": role, "content": ctx["text"]})
                messages.append({"role": "assistant", "content": msg["text"]})
                pairs.append({
                    "messages": messages,
                    "metadata": {
                        "chat_id": msg["chat_id"],
                        "timestamp": msg["datetime"],
                        "reply_length": len(msg["text"]),
                        "source": "facebook",
                    }
                })
    return pairs


def extract_voice_training_pairs(windows: list[list[dict]]) -> list[dict]:
    """Voice-optimized: shorter context, shorter replies."""
    MAX_VOICE_REPLY_LENGTH = 280
    MAX_VOICE_CONTEXT = 3
    pairs = []
    for window in windows:
        for i, msg in enumerate(window):
            if not msg["is_from_me"] or len(msg["text"]) < MIN_REPLY_LENGTH:
                continue
            if len(msg["text"]) > MAX_VOICE_REPLY_LENGTH:
                continue
            if msg["text"].startswith("http") or "\n\n" in msg["text"]:
                continue

            context_start = max(0, i - MAX_VOICE_CONTEXT)
            context = window[context_start:i]
            if not context:
                continue

            messages = []
            for ctx in context:
                role = "assistant" if ctx["is_from_me"] else "user"
                messages.append({"role": role, "content": ctx["text"]})
            messages.append({"role": "assistant", "content": msg["text"]})

            pairs.append({
                "messages": messages,
                "metadata": {
                    "chat_id": msg["chat_id"],
                    "timestamp": msg["datetime"],
                    "reply_length": len(msg["text"]),
                    "source": "facebook",
                    "format": "voice",
                }
            })
    return pairs


def extract_ground_truth(windows: list[list[dict]]) -> list[dict]:
    gt = []
    for window in windows:
        for i in range(len(window) - 1):
            incoming = window[i]
            reply = window[i + 1]
            if not incoming["is_from_me"] and reply["is_from_me"]:
                if len(reply["text"]) >= MIN_REPLY_LENGTH:
                    delay_s = reply["timestamp"] - incoming["timestamp"]
                    gt.append({
                        "incoming": incoming["text"],
                        "reply": reply["text"],
                        "delay_seconds": round(delay_s, 1),
                        "chat_id": incoming["chat_id"],
                        "timestamp": reply["datetime"],
                        "source": "facebook",
                    })
    return gt


def main():
    parser = argparse.ArgumentParser(
        description="Extract Facebook Messenger conversations for fine-tuning",
    )
    parser.add_argument("--export", required=True,
                        help="Path to unzipped Facebook data export")
    parser.add_argument("--name", default=None,
                        help="Your name as it appears in Facebook (auto-detected if omitted)")
    parser.add_argument("--output", default="data/facebook",
                        help="Output directory (default: data/facebook)")
    parser.add_argument("--voice", action="store_true",
                        help="Also generate voice-optimized training pairs")
    args = parser.parse_args()

    export_path = Path(args.export).expanduser()
    if not export_path.exists():
        print(f"ERROR: Export path not found: {export_path}")
        sys.exit(1)

    inbox_dir = find_inbox_dir(export_path)
    if not inbox_dir:
        print(f"ERROR: Could not find messages/inbox in {export_path}")
        print(f"Expected: {export_path}/your_facebook_activity/messages/inbox/")
        print(f"\nMake sure you downloaded 'Messages' in JSON format from Facebook.")
        sys.exit(1)

    # Auto-detect name by finding most common sender
    your_name = args.name
    if your_name is None:
        print("  Auto-detecting your name from messages...")
        sample_msgs = []
        for msg_file in list(inbox_dir.rglob("message_1.json"))[:20]:
            try:
                with open(msg_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for msg in data.get("messages", []):
                    sample_msgs.append(decode_facebook_text(msg.get("sender_name", "")))
            except Exception:
                pass

        from collections import Counter
        name_counts = Counter(sample_msgs)
        if name_counts:
            your_name = name_counts.most_common(1)[0][0]
            print(f"  Detected your name: {your_name}")
        else:
            print("ERROR: Could not auto-detect your name. Use --name 'Your Name'")
            sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting Facebook messages...")
    print(f"  Export:  {export_path}")
    print(f"  Inbox:   {inbox_dir}")
    print(f"  Name:    {your_name}")
    print(f"  Output:  {output_dir}")

    messages = extract_messages(inbox_dir, your_name)
    print(f"  Total messages: {len(messages)}")

    chats = group_by_chat(messages)
    print(f"  Conversations: {len(chats)}")

    all_windows = []
    for chat_msgs in chats.values():
        all_windows.extend(build_conversation_windows(chat_msgs))
    print(f"  Conversation windows: {len(all_windows)}")

    # Training pairs
    training = extract_training_pairs(all_windows)
    train_path = output_dir / "training_pairs.jsonl"
    with open(train_path, "w") as f:
        for pair in training:
            f.write(json.dumps(pair) + "\n")
    print(f"\n  Training pairs: {len(training)} -> {train_path}")

    # Voice training pairs
    if args.voice:
        voice = extract_voice_training_pairs(all_windows)
        voice_path = output_dir / "voice_training_pairs.jsonl"
        with open(voice_path, "w") as f:
            for pair in voice:
                f.write(json.dumps(pair) + "\n")
        print(f"  Voice training pairs: {len(voice)} -> {voice_path}")

    # Ground truth
    gt = extract_ground_truth(all_windows)
    gt_path = output_dir / "ground_truth.jsonl"
    with open(gt_path, "w") as f:
        for item in gt:
            f.write(json.dumps(item) + "\n")
    print(f"  Ground truth pairs: {len(gt)} -> {gt_path}")

    # Stats
    my_msgs = [m for m in messages if m["is_from_me"]]
    my_lengths = [len(m["text"]) for m in my_msgs]
    print(f"\n--- Stats ---")
    print(f"  Your messages: {len(my_msgs)}")
    print(f"  Other messages: {len(messages) - len(my_msgs)}")
    if my_lengths:
        print(f"  Avg reply length: {sum(my_lengths)/len(my_lengths):.0f} chars")
        print(f"  Median reply length: {sorted(my_lengths)[len(my_lengths)//2]} chars")

    if training:
        print(f"\n--- Sample training pair ---")
        print(json.dumps(training[0], indent=2))


if __name__ == "__main__":
    main()
