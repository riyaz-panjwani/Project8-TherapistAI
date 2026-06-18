"""Train TherapistTransformer — multi-task intent + DST.

Usage:
    # 1. Download GloVe (once)
    python training/scripts/download_glove.py

    # 2. Prepare data
    python training/scripts/prepare_multitask_data.py

    # 3. Train
    python training/scripts/train_multitask.py

Checkpoints saved to: training/checkpoints/therapist_transformer/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# make backend importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))
from models.custom_transformer import (
    TherapistTransformer, Tokenizer,
    INTENT2ID, DST2ID, MAX_LEN,
)

CHECKPOINT_DIR = Path("training/checkpoints/therapist_transformer")
GLOVE_PATH     = Path("training/data/glove.6B.100d.txt")


# ── dataset ────────────────────────────────────────────────────────────────

class TherapyDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer: Tokenizer):
        self.records   = records
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec    = self.records[idx]
        ids    = self.tokenizer.encode(rec["text"])
        mask   = self.tokenizer.attention_mask(ids)
        intent = INTENT2ID[rec["intent"]]

        # DST labels aligned to encoded ids
        # ids = [CLS, tok1, tok2, ..., SEP, PAD, PAD, ...]
        # We skip CLS (pos 0) and SEP, pad with -100 (ignored by cross-entropy)
        tokens   = rec["tokens"]
        dst_strs = rec["dst_tags"]
        raw_dst  = [DST2ID[t] for t in dst_strs]

        dst_labels = [-100]                                   # [CLS] ignored
        for i in range(MAX_LEN - 2):                          # -2 for CLS+SEP
            if i < len(raw_dst):
                dst_labels.append(raw_dst[i])
            elif ids[i + 1] != self.tokenizer.word2id["[SEP]"] and mask[i + 1] == 1:
                dst_labels.append(DST2ID["O"])
            else:
                dst_labels.append(-100)
        dst_labels.append(-100)                               # [SEP] ignored

        # pad to MAX_LEN
        dst_labels = dst_labels[:MAX_LEN]
        dst_labels += [-100] * (MAX_LEN - len(dst_labels))

        return {
            "input_ids":      torch.tensor(ids,        dtype=torch.long),
            "attention_mask": torch.tensor(mask,       dtype=torch.long),
            "intent_label":   torch.tensor(intent,     dtype=torch.long),
            "dst_labels":     torch.tensor(dst_labels, dtype=torch.long),
        }


# ── training loop ───────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def evaluate(model, loader, intent_criterion, dst_criterion, device):
    model.eval()
    total_loss = intent_correct = total_intent = 0
    with torch.no_grad():
        for batch in loader:
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            int_y  = batch["intent_label"].to(device)
            dst_y  = batch["dst_labels"].to(device)

            int_log, dst_log = model(ids, mask)

            loss = intent_criterion(int_log, int_y) + 0.5 * dst_criterion(
                dst_log.view(-1, dst_log.size(-1)), dst_y.view(-1)
            )
            total_loss += loss.item()

            preds = int_log.argmax(dim=-1)
            intent_correct += (preds == int_y).sum().item()
            total_intent   += int_y.size(0)

    return total_loss / len(loader), intent_correct / total_intent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file",  default="training/data/multitask_train.jsonl")
    parser.add_argument("--val_file",    default="training/data/multitask_val.jsonl")
    parser.add_argument("--glove",       default=str(GLOVE_PATH))
    parser.add_argument("--output_dir",  default=str(CHECKPOINT_DIR))
    parser.add_argument("--epochs",      type=int,   default=60)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--patience",    type=int,   default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_recs = load_jsonl(Path(args.train_file))
    val_recs   = load_jsonl(Path(args.val_file))

    # build vocabulary from all training texts
    tokenizer = Tokenizer()
    tokenizer.build_vocab([r["text"] for r in train_recs + val_recs])
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    train_ds = TherapyDataset(train_recs, tokenizer)
    val_ds   = TherapyDataset(val_recs,   tokenizer)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size)

    model = TherapistTransformer(vocab_size=tokenizer.vocab_size).to(device)

    # transfer learning — init embeddings from GloVe
    if Path(args.glove).exists():
        model.load_glove_embeddings(args.glove, tokenizer)
    else:
        print(f"GloVe not found at {args.glove}. Run download_glove.py first.")
        print("Training with random embeddings (results will be weaker).")

    intent_criterion = nn.CrossEntropyLoss()
    dst_criterion    = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer        = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler        = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_count = 0

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>9}  {'Intent Acc':>10}")
    print("─" * 44)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_dl:
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            int_y  = batch["intent_label"].to(device)
            dst_y  = batch["dst_labels"].to(device)

            optimizer.zero_grad()
            int_log, dst_log = model(ids, mask)

            # multi-task loss: intent + 0.5 * DST
            intent_loss = intent_criterion(int_log, int_y)
            dst_loss    = dst_criterion(
                dst_log.view(-1, dst_log.size(-1)), dst_y.view(-1)
            )
            loss = intent_loss + 0.5 * dst_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        val_loss, val_acc = evaluate(model, val_dl, intent_criterion, dst_criterion, device)
        train_loss /= len(train_dl)

        print(f"{epoch:>5}  {train_loss:>10.4f}  {val_loss:>9.4f}  {val_acc:>9.1%}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            tokenizer.save(out_dir / "tokenizer.json")
            # save config for loading later
            config = {
                "vocab_size": tokenizer.vocab_size,
                "d_embed": 100, "d_model": 128, "n_heads": 4,
                "d_ff": 256, "n_layers": 3, "dropout": 0.3, "max_len": MAX_LEN,
            }
            (out_dir / "config.json").write_text(json.dumps(config, indent=2))
            print(f"       ✓ saved (val_loss={val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {out_dir}/best_model.pt")


if __name__ == "__main__":
    main()
