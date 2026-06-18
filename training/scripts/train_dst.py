"""Fine-tune ConvBERT for Dialogue State Tracking (slot extraction).

Treats DST as a token-classification / NER task on MultiWOZ slots.
Each token is tagged with B/I/O for slots: TOPIC, ISSUE, PERSON, MOOD.

Usage:
    python training/scripts/train_dst.py \
        --output_dir training/checkpoints/dst \
        --epochs 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from datasets import Dataset
from sklearn.metrics import f1_score
import json
import numpy as np

DST_LABELS = ["O", "B-TOPIC", "I-TOPIC", "B-ISSUE", "I-ISSUE", "B-PERSON", "I-PERSON", "B-MOOD", "I-MOOD"]
LABEL2ID   = {l: i for i, l in enumerate(DST_LABELS)}
ID2LABEL   = {i: l for l in DST_LABELS for i, l in enumerate(DST_LABELS)}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def tokenize_and_align(examples, tokenizer, max_len=128):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding=False,
        max_length=max_len,
    )
    labels_out = []
    for i, label_seq in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned  = []
        prev_wid = None
        for wid in word_ids:
            if wid is None:
                aligned.append(-100)
            elif wid != prev_wid:
                aligned.append(label_seq[wid])
            else:
                tag = label_seq[wid]
                # convert B- to I- for continuation tokens
                aligned.append(tag + 1 if tag % 2 == 1 else tag)
            prev_wid = wid
        labels_out.append(aligned)
    tokenized["labels"] = labels_out
    return tokenized


def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, -1)
    flat_p = [p for pred_row, label_row in zip(preds, labels) for p, l in zip(pred_row, label_row) if l != -100]
    flat_l = [l for pred_row, label_row in zip(preds, labels) for p, l in zip(pred_row, label_row) if l != -100]
    return {"f1": f1_score(flat_l, flat_p, average="macro", zero_division=0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file",  default="training/data/train_dst.jsonl")
    parser.add_argument("--val_file",    default="training/data/val_dst.jsonl")
    parser.add_argument("--model_name",  default="YituTech/conv-bert-base")
    parser.add_argument("--output_dir",  default="training/checkpoints/dst")
    parser.add_argument("--epochs",      type=int,   default=4)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=3e-5)
    args = parser.parse_args()

    train_path = Path(args.train_file)
    val_path   = Path(args.val_file)
    if not train_path.exists() or not val_path.exists():
        print("DST training data not found. Provide train_dst.jsonl / val_dst.jsonl with {tokens, ner_tags} rows.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(DST_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_raw = load_jsonl(train_path)
    val_raw   = load_jsonl(val_path)

    def to_hf(records):
        return Dataset.from_dict({
            "tokens":   [r["tokens"]   for r in records],
            "ner_tags": [r["ner_tags"] for r in records],
        })

    train_ds = to_hf(train_raw).map(
        lambda ex: tokenize_and_align(ex, tokenizer), batched=True
    )
    val_ds   = to_hf(val_raw).map(
        lambda ex: tokenize_and_align(ex, tokenizer), batched=True
    )

    training_args = TrainingArguments(
        output_dir              = args.output_dir,
        num_train_epochs        = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        learning_rate           = args.lr,
        eval_strategy           = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        metric_for_best_model   = "f1",
        fp16                    = torch.cuda.is_available(),
        report_to               = "none",
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        tokenizer       = tokenizer,
        data_collator   = DataCollatorForTokenClassification(tokenizer),
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"Training ConvBERT DST on {len(train_ds)} examples…")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
