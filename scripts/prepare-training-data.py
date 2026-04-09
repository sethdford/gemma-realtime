#!/usr/bin/env python3
"""
Combine extracted conversation data from multiple sources into training-ready
JSONL files for Gemma 4 LoRA fine-tuning.

Merges data from:
  - iMessage (extract_imessage_pairs.py)
  - Facebook Messenger (extract-facebook.py)
  - Custom JSONL files

Produces:
  - train.jsonl — 90% of shuffled conversations for SFT training
  - valid.jsonl — 10% for validation
  - train_voice.jsonl — voice-optimized subset (short replies, tight context)
  - valid_voice.jsonl — voice validation

Usage:
    python3 scripts/prepare-training-data.py
    python3 scripts/prepare-training-data.py --sources data/imessage data/facebook
    python3 scripts/prepare-training-data.py --output ~/.human/training-data/finetune
    python3 scripts/prepare-training-data.py --voice  # include voice-optimized split
    python3 scripts/prepare-training-data.py --system-prompt "You are Seth. ..."
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path


DEFAULT_SYSTEM_PROMPT = (
    "You are a personal AI assistant fine-tuned to match your owner's communication "
    "style. Respond naturally, matching their tone, vocabulary, and typical message "
    "length. Be conversational and authentic."
)


def load_pairs(source_dir: Path, file_pattern: str = "training_pairs.jsonl") -> list[dict]:
    """Load training pairs from a source directory."""
    pairs_file = source_dir / file_pattern
    if not pairs_file.exists():
        return []

    pairs = []
    with open(pairs_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pairs


def add_system_prompt(pairs: list[dict], system_prompt: str) -> list[dict]:
    """Prepend a system prompt to each conversation if not already present."""
    result = []
    for pair in pairs:
        messages = pair.get("messages", [])
        if not messages:
            continue

        if messages[0].get("role") != "system":
            messages = [{"role": "system", "content": system_prompt}] + messages

        result.append({
            "messages": messages,
            "metadata": pair.get("metadata", {}),
        })
    return result


def deduplicate(pairs: list[dict]) -> list[dict]:
    """Remove exact duplicate conversations (by last assistant message)."""
    seen = set()
    unique = []
    for pair in pairs:
        msgs = pair.get("messages", [])
        assistant_msgs = [m["content"] for m in msgs if m["role"] == "assistant"]
        key = "|".join(assistant_msgs[-2:]) if len(assistant_msgs) >= 2 else "|".join(assistant_msgs)
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    return unique


def split_train_valid(pairs: list[dict], valid_ratio: float = 0.1) -> tuple[list[dict], list[dict]]:
    """Shuffle and split into train/valid sets."""
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - valid_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def write_jsonl(pairs: list[dict], path: Path):
    """Write pairs to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Combine conversation data sources into training-ready JSONL",
    )
    parser.add_argument("--sources", nargs="+", default=["data/imessage", "data/facebook"],
                        help="Source data directories (default: data/imessage data/facebook)")
    parser.add_argument("--custom", nargs="*", default=[],
                        help="Additional JSONL files to include")
    parser.add_argument("--output", default="data/finetune",
                        help="Output directory (default: data/finetune)")
    parser.add_argument("--voice", action="store_true",
                        help="Also prepare voice-optimized training data")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT,
                        help="System prompt to prepend to conversations")
    parser.add_argument("--no-system-prompt", action="store_true",
                        help="Don't add a system prompt")
    parser.add_argument("--valid-ratio", type=float, default=0.1,
                        help="Validation split ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output)

    print(f"\n{'='*60}")
    print(f"  Prepare Training Data")
    print(f"{'='*60}")

    # Collect from all sources
    all_pairs = []
    voice_pairs = []

    for source in args.sources:
        source_dir = Path(source)
        if not source_dir.exists():
            print(f"  SKIP: {source_dir} (not found)")
            continue

        pairs = load_pairs(source_dir, "training_pairs.jsonl")
        source_name = source_dir.name
        print(f"  {source_name}: {len(pairs)} training pairs")
        all_pairs.extend(pairs)

        if args.voice:
            vp = load_pairs(source_dir, "voice_training_pairs.jsonl")
            print(f"  {source_name}: {len(vp)} voice pairs")
            voice_pairs.extend(vp)

    # Load custom JSONL files
    for custom_file in args.custom:
        custom_path = Path(custom_file)
        if not custom_path.exists():
            print(f"  SKIP: {custom_path} (not found)")
            continue
        pairs = []
        with open(custom_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        pairs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        print(f"  custom ({custom_path.name}): {len(pairs)} pairs")
        all_pairs.extend(pairs)

    if not all_pairs:
        print(f"\n  ERROR: No training data found!")
        print(f"  Run one of these first:")
        print(f"    python3 scripts/extract_imessage_pairs.py")
        print(f"    python3 scripts/extract-facebook.py --export ~/Downloads/facebook-export")
        sys.exit(1)

    # Deduplicate
    before = len(all_pairs)
    all_pairs = deduplicate(all_pairs)
    dupes = before - len(all_pairs)
    if dupes:
        print(f"\n  Removed {dupes} duplicate conversations")

    # Add system prompt
    if not args.no_system_prompt:
        all_pairs = add_system_prompt(all_pairs, args.system_prompt)
        if voice_pairs:
            voice_pairs = add_system_prompt(voice_pairs, args.system_prompt)

    # Split and write
    train, valid = split_train_valid(all_pairs, args.valid_ratio)
    write_jsonl(train, output_dir / "train.jsonl")
    write_jsonl(valid, output_dir / "valid.jsonl")

    print(f"\n  Output: {output_dir}")
    print(f"  Training examples: {len(train)}")
    print(f"  Validation examples: {len(valid)}")

    if args.voice and voice_pairs:
        voice_pairs = deduplicate(voice_pairs)
        v_train, v_valid = split_train_valid(voice_pairs, args.valid_ratio)
        write_jsonl(v_train, output_dir / "train_voice.jsonl")
        write_jsonl(v_valid, output_dir / "valid_voice.jsonl")
        print(f"  Voice training: {len(v_train)}")
        print(f"  Voice validation: {len(v_valid)}")

    # Summary stats
    all_lengths = []
    for pair in all_pairs:
        for msg in pair.get("messages", []):
            if msg["role"] == "assistant":
                all_lengths.append(len(msg["content"]))

    if all_lengths:
        print(f"\n--- Response Stats ---")
        print(f"  Total responses: {len(all_lengths)}")
        print(f"  Avg length: {sum(all_lengths)/len(all_lengths):.0f} chars")
        print(f"  Median length: {sorted(all_lengths)[len(all_lengths)//2]} chars")

    print(f"\n  Next step:")
    print(f"    python3 scripts/finetune-gemma.py --target e4b --data {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
