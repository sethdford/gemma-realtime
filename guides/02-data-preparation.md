# Data Preparation

How to extract, clean, and prepare your personal conversation data for fine-tuning.

## Overview

The fine-tuning pipeline expects JSONL files with multi-turn conversations:

```json
{
  "messages": [
    {"role": "system", "content": "You are a personal assistant..."},
    {"role": "user", "content": "hey what's up"},
    {"role": "assistant", "content": "not much, just working on some code"},
    {"role": "user", "content": "nice what kind?"},
    {"role": "assistant", "content": "rust stuff, trying to optimize this parser"}
  ]
}
```

Each `assistant` message is YOUR response — the model learns to generate text that matches your style.

## Data Sources

### iMessage (macOS)

The easiest source if you're a Mac user. Your entire message history is in a SQLite database.

```bash
python3 scripts/extract_imessage_pairs.py
```

**What it does:**
- Reads `~/Library/Messages/chat.db` (you may need to grant Full Disk Access in System Preferences)
- Recovers messages from `attributedBody` blobs (modern macOS stores text there)
- Filters out verification codes, system messages, and short/empty messages
- Groups messages into conversation windows (1-hour gap boundary)
- Produces standard training pairs (6-message context) and voice-optimized pairs (3-message context, short replies)

**Output:**
```
data/imessage/
├── training_pairs.jsonl       # Full training pairs
├── voice_training_pairs.jsonl # Voice-optimized (short context, concise replies)
├── ground_truth.jsonl         # (incoming, your_reply) pairs for evaluation
└── timing_data.jsonl          # Response timing for pacing analysis
```

**Troubleshooting:**

If you get a permission error:
1. System Preferences → Privacy & Security → Full Disk Access
2. Add Terminal (or your terminal app)
3. Restart the terminal and try again

### Facebook Messenger

Download your data from Facebook, then extract it.

**Step 1: Download your data**

1. Go to [facebook.com/dyi](https://www.facebook.com/dyi) (Download Your Information)
2. Click "Request a download"
3. Settings:
   - **Format**: JSON (not HTML)
   - **Media Quality**: Low (we only need text)
   - **Date Range**: All Time
   - **Information**: Select only "Messages"
4. Click "Create File" and wait for the email notification
5. Download and unzip the archive

**Step 2: Extract conversations**

```bash
python3 scripts/extract-facebook.py \
  --export ~/Downloads/facebook-YourName \
  --voice
```

The script auto-detects your name from message frequency. If it gets it wrong:

```bash
python3 scripts/extract-facebook.py \
  --export ~/Downloads/facebook-YourName \
  --name "Your Full Name"
```

**What it does:**
- Finds the `messages/inbox/` directory in your export
- Decodes Facebook's latin-1-encoded UTF-8 text (yes, it's weird)
- Filters out "sent a photo", reactions, call notifications, etc.
- Same conversation windowing and pair extraction as iMessage

**Output:**
```
data/facebook/
├── training_pairs.jsonl
├── voice_training_pairs.jsonl  (if --voice)
└── ground_truth.jsonl
```

### Custom Data

You can add any JSONL file that follows the messages format. Each line should have:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Pass custom files when preparing training data:

```bash
python3 scripts/prepare-training-data.py \
  --custom my-slack-export.jsonl my-discord-export.jsonl
```

### WhatsApp

WhatsApp exports as a `.txt` file. You can convert it to JSONL with a simple script:

```python
import json, re, sys

pairs = []
messages = []
your_name = "Your Name"  # change this

for line in open(sys.argv[1]):
    match = re.match(r'\[(\d+/\d+/\d+, \d+:\d+:\d+)\] (.+?): (.+)', line.strip())
    if match:
        timestamp, sender, text = match.groups()
        if '<Media omitted>' in text:
            continue
        role = "assistant" if sender == your_name else "user"
        messages.append({"role": role, "content": text})

        if role == "assistant" and len(messages) >= 2:
            pairs.append({"messages": messages[-6:]})

for p in pairs:
    print(json.dumps(p))
```

```bash
python3 whatsapp_convert.py "WhatsApp Chat.txt" > data/whatsapp/training_pairs.jsonl
```

## Combining Sources

After extracting from all your sources:

```bash
python3 scripts/prepare-training-data.py \
  --sources data/imessage data/facebook \
  --custom data/whatsapp/training_pairs.jsonl \
  --voice \
  --output data/finetune
```

This:
1. Loads all training pairs from each source
2. Deduplicates (same conversations across platforms)
3. Adds a system prompt
4. Shuffles and splits 90/10 into train/valid
5. Also prepares voice-optimized splits

**Output:**
```
data/finetune/
├── train.jsonl        # 90% — for SFT training
├── valid.jsonl        # 10% — for validation loss
├── train_voice.jsonl  # Voice-optimized training
└── valid_voice.jsonl  # Voice-optimized validation
```

## Data Quality Tips

**More data isn't always better.** What matters:

1. **Recency** — Recent conversations reflect your current style better than messages from 5 years ago. Facebook lets you filter by date range.

2. **Diversity** — Include conversations with different people and topics. A model trained only on work Slack will sound robotic in casual chat.

3. **Reply length distribution** — For voice models, prefer shorter replies (< 280 chars). The voice extraction scripts already filter for this.

4. **Avoid toxic/embarrassing data** — The model WILL learn patterns from your worst messages. Consider filtering conversations you wouldn't want reproduced.

5. **Minimum viable dataset** — 500 training pairs is a reasonable minimum. 2,000-5,000 is a sweet spot. Beyond 10,000 you get diminishing returns for LoRA.

## Privacy Notes

All data extraction and training happens locally on your Mac. Nothing is sent to any cloud service.

- The extracted JSONL files contain your actual messages — treat them as sensitive
- The LoRA adapter (the fine-tuning output) also encodes your style — keep it private
- The scripts never upload anything; they only read local databases/files and write local JSONL

If you're sharing this repo or your adapter, make sure `data/` is in your `.gitignore`.
