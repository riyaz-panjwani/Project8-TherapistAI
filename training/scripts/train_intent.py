"""Fine-tune RoBERTa on MultiWOZ + therapy intent labels.

Usage:
    python training/scripts/train_intent.py \
        --output_dir training/checkpoints/intent \
        --epochs 5 \
        --batch_size 16

Dataset: MultiWOZ 2.2 (intent labels) augmented with manually labelled
therapy utterances. Run prepare_data.py first to produce train.jsonl /
val.jsonl with {"text": ..., "label": ...} rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

INTENT_LABELS = [
    "venting", "seeking_advice", "anxiety", "depression", "crisis",
    "gratitude", "relationship", "work_stress", "self_esteem",
    "checking_in", "trauma", "progress", "general",
]
LABEL2ID = {l: i for i, l in enumerate(INTENT_LABELS)}
ID2LABEL = {i: l for i, l in enumerate(INTENT_LABELS)}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def make_dataset(records: list[dict], tokenizer, max_len: int = 128) -> Dataset:
    texts  = [r["text"]  for r in records]
    labels = [LABEL2ID[r["label"]] for r in records]
    enc = tokenizer(texts, truncation=True, padding=False, max_length=max_len)
    enc["labels"] = labels
    return Dataset.from_dict(enc)


def compute_metrics(pred):
    logits, labels = pred
    preds = logits.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file",  default="training/data/train_intent.jsonl")
    parser.add_argument("--val_file",    default="training/data/val_intent.jsonl")
    parser.add_argument("--model_name",  default="roberta-base")
    parser.add_argument("--output_dir",  default="training/checkpoints/intent")
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--max_len",     type=int,   default=128)
    args = parser.parse_args()

    train_path = Path(args.train_file)
    val_path   = Path(args.val_file)
    if not train_path.exists() or not val_path.exists():
        print("Run training/scripts/prepare_data.py first to create train/val splits.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(INTENT_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = make_dataset(load_jsonl(train_path), tokenizer, args.max_len)
    val_ds   = make_dataset(load_jsonl(val_path),   tokenizer, args.max_len)

    training_args = TrainingArguments(
        output_dir              = args.output_dir,
        num_train_epochs        = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        learning_rate           = args.lr,
        warmup_ratio            = 0.1,
        weight_decay            = 0.01,
        eval_strategy           = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        metric_for_best_model   = "f1_macro",
        logging_steps           = 20,
        fp16                    = torch.cuda.is_available(),
        report_to               = "none",
    )

    trainer = Trainer(
        model          = model,
        args           = training_args,
        train_dataset  = train_ds,
        eval_dataset   = val_ds,
        tokenizer      = tokenizer,
        data_collator  = DataCollatorWithPadding(tokenizer),
        compute_metrics= compute_metrics,
        callbacks      = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"Training RoBERTa intent classifier on {len(train_ds)} examples…")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
