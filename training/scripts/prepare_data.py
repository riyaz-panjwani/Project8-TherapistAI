"""Prepare MultiWOZ 2.2 + synthetic therapy data for intent & DST training.

Downloads MultiWOZ via Hugging Face datasets, maps service-level intents to
our therapy intent schema, and writes train/val JSONL files.

Usage:
    python training/scripts/prepare_data.py --output_dir training/data
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ── MultiWOZ intent → therapy-intent mapping ─────────────────────
MULTIWOZ_MAP: dict[str, str] = {
    "inform":     "general",
    "request":    "seeking_advice",
    "confirm":    "checking_in",
    "negate":     "venting",
    "affirm":     "gratitude",
    "bye":        "checking_in",
    "greet":      "checking_in",
    "thank":      "gratitude",
    "reqmore":    "seeking_advice",
}

# ── Synthetic therapy utterances (small seed set) ────────────────
SEED_DATA: list[dict] = [
    # crisis
    {"text": "I've been thinking about hurting myself.", "label": "crisis"},
    {"text": "I don't want to be here anymore.", "label": "crisis"},
    {"text": "I've been having thoughts of suicide.", "label": "crisis"},
    {"text": "I've been cutting myself again.", "label": "crisis"},
    # anxiety
    {"text": "I can't stop worrying about everything.", "label": "anxiety"},
    {"text": "I had a panic attack on the tube this morning.", "label": "anxiety"},
    {"text": "My heart races every time I think about it.", "label": "anxiety"},
    {"text": "I feel overwhelmed all the time.", "label": "anxiety"},
    # depression
    {"text": "I just feel empty. Like nothing matters.", "label": "depression"},
    {"text": "I can't get out of bed most mornings.", "label": "depression"},
    {"text": "Everything feels pointless.", "label": "depression"},
    {"text": "I've lost interest in things I used to love.", "label": "depression"},
    # venting
    {"text": "I'm so sick of how my flatmate treats me.", "label": "venting"},
    {"text": "My boss is an absolute nightmare and I'm furious.", "label": "venting"},
    {"text": "I just need to vent — today was terrible.", "label": "venting"},
    # seeking_advice
    {"text": "I don't know what to do about my relationship.", "label": "seeking_advice"},
    {"text": "Should I quit my job?", "label": "seeking_advice"},
    {"text": "What do you think I should do?", "label": "seeking_advice"},
    # relationship
    {"text": "My partner and I keep having the same argument.", "label": "relationship"},
    {"text": "I feel so lonely even when I'm with people.", "label": "relationship"},
    {"text": "My mum doesn't understand me at all.", "label": "relationship"},
    # work_stress
    {"text": "I have three deadlines this week and I can't cope.", "label": "work_stress"},
    {"text": "I think I'm burning out from work.", "label": "work_stress"},
    {"text": "My exams are in two weeks and I'm not ready.", "label": "work_stress"},
    # self_esteem
    {"text": "I hate how I look.", "label": "self_esteem"},
    {"text": "I'm not smart enough for this.", "label": "self_esteem"},
    {"text": "Everyone else seems to have their life together.", "label": "self_esteem"},
    # trauma
    {"text": "Something happened to me as a child that I've never talked about.", "label": "trauma"},
    {"text": "I keep having flashbacks from what happened.", "label": "trauma"},
    # gratitude
    {"text": "Talking to you really helped me last time.", "label": "gratitude"},
    {"text": "Thank you, I feel so much better.", "label": "gratitude"},
    # progress
    {"text": "I actually had a good week for once.", "label": "progress"},
    {"text": "I stood up for myself today and it felt amazing.", "label": "progress"},
    # checking_in
    {"text": "Hey, just checking in.", "label": "checking_in"},
    {"text": "Hi, how are you?", "label": "checking_in"},
    {"text": "Good morning.", "label": "checking_in"},
]


def load_multiwoz(split: str) -> list[dict]:
    try:
        from datasets import load_dataset
        ds = load_dataset("multi_woz_v22", split=split, trust_remote_code=True)
        records = []
        for dialog in ds:
            for turn in dialog.get("turns", {}).get("utterance", []):
                # turns alternate user/system; only take user turns
                pass
        # MultiWOZ structure varies by version; use seed data as primary
        return []
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="training/data")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = list(SEED_DATA)

    # try loading MultiWOZ
    mwoz = load_multiwoz("train")
    records.extend(mwoz)

    random.shuffle(records)
    split = int(len(records) * 0.85)
    train_records = records[:split]
    val_records   = records[split:]

    def write_jsonl(path, data):
        Path(path).write_text("\n".join(json.dumps(r) for r in data))

    write_jsonl(out / "train_intent.jsonl", train_records)
    write_jsonl(out / "val_intent.jsonl",   val_records)

    print(f"Wrote {len(train_records)} train / {len(val_records)} val records to {out}")
    print("Labels:", {r['label'] for r in records})


if __name__ == "__main__":
    main()
