"""Train TherapistTransformerV2 — frozen DistilBERT features + custom head.

Pipeline
────────
1. Load DistilBERT tokenizer + model (frozen — never updated).
2. Tokenize all training & val texts once, run through DistilBERT,
   cache the 768-d hidden states on disk as .pt tensors.
3. Train TherapistTransformerV2 (projection + custom encoder + heads)
   using cached hidden states — no BERT forward pass during training.
4. Save checkpoint to training/checkpoints/therapist_transformer_v2/

Usage
─────
    source .venv/bin/activate
    python training/scripts/train_bert_v2.py

Expected accuracy: 85–93 % (vs ~71 % with GloVe V1).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))
from models.custom_transformer import (
    TherapistTransformerV2, INTENT2ID, DST2ID, INTENT_LABELS, DST_LABELS,
)

CHECKPOINT_DIR = Path("training/checkpoints/therapist_transformer_v2")
CACHE_DIR      = Path("training/data/bert_cache")
BERT_MODEL_ID  = "distilbert-base-uncased"
MAX_LEN        = 128


# ── label smoothing ────────────────────────────────────────────────────────

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.smoothing    = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = logits.size(-1)
        log_p = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth = torch.full_like(log_p, self.smoothing / (n - 1))
            smooth.scatter_(-1, targets.unsqueeze(-1).clamp(min=0), 1.0 - self.smoothing)
        loss = -(smooth * log_p).sum(dim=-1)
        if self.ignore_index >= 0:
            loss = loss[targets.ne(self.ignore_index)]
        return loss.mean()


# ── warmup + cosine scheduler ──────────────────────────────────────────────

def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * prog)))
    return LambdaLR(optimizer, fn)


# ── DistilBERT embedding precomputation ───────────────────────────────────

def precompute(texts: list[str], labels_list: list[dict], bert_tok, bert_model,
               cache_path: Path, split: str, device: str):
    h_path = cache_path / f"{split}_hidden.pt"
    m_path = cache_path / f"{split}_mask.pt"
    w_path = cache_path / f"{split}_wordids.json"

    if h_path.exists() and m_path.exists() and w_path.exists():
        print(f"[cache] Loading {split} from cache…")
        return (torch.load(h_path),
                torch.load(m_path),
                json.loads(w_path.read_text()))

    print(f"[bert] Encoding {len(texts)} {split} examples…")
    all_h, all_m, all_w = [], [], []
    BS = 32
    bert_model.eval()
    for i in range(0, len(texts), BS):
        batch = texts[i:i + BS]
        enc = bert_tok(batch, max_length=MAX_LEN, padding="max_length",
                       truncation=True, return_tensors="pt",
                       return_offsets_mapping=False)
        # word_ids per example (fast tokenizer)
        word_ids_batch = [enc.word_ids(j) for j in range(len(batch))]

        with torch.no_grad():
            out = bert_model(input_ids=enc["input_ids"].to(device),
                             attention_mask=enc["attention_mask"].to(device))
        all_h.append(out.last_hidden_state.cpu())
        all_m.append(enc["attention_mask"].cpu())
        all_w.extend(word_ids_batch)

    hidden = torch.cat(all_h, dim=0)   # [N, MAX_LEN, 768]
    masks  = torch.cat(all_m, dim=0)   # [N, MAX_LEN]

    cache_path.mkdir(parents=True, exist_ok=True)
    torch.save(hidden, h_path)
    torch.save(masks,  m_path)
    w_path.write_text(json.dumps(all_w))
    print(f"[cache] Saved {split} ({hidden.shape}).")
    return hidden, masks, all_w


# ── dataset ────────────────────────────────────────────────────────────────

def align_dst(word_ids: list[int | None], dst_tags: list[str]) -> list[int]:
    """Map word-level DST tags onto DistilBERT subword positions."""
    labels, prev = [], None
    for wid in word_ids:
        if wid is None:
            labels.append(-100)
        elif wid != prev:
            tag = dst_tags[wid] if wid < len(dst_tags) else "O"
            labels.append(DST2ID[tag])
        else:
            labels.append(-100)          # non-first subword → ignored
        prev = wid
    labels = labels[:MAX_LEN]
    labels += [-100] * (MAX_LEN - len(labels))
    return labels


def _aug(h: torch.Tensor, mask: torch.Tensor,
         p_noise: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """Light Gaussian noise on hidden states (augmentation for training)."""
    if p_noise > 0:
        noise = torch.randn_like(h) * p_noise
        h = h + noise
    return h, mask


class BertDataset(Dataset):
    def __init__(self, records: list[dict], hidden: torch.Tensor,
                 masks: torch.Tensor, word_ids_list: list,
                 augment: bool = False):
        self.records       = records
        self.hidden        = hidden
        self.masks         = masks
        self.word_ids_list = word_ids_list
        self.augment       = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec   = self.records[idx]
        h     = self.hidden[idx].clone()
        mask  = self.masks[idx].clone()
        if self.augment:
            h, mask = _aug(h, mask)

        intent_label = INTENT2ID[rec["intent"]]
        dst_labels   = align_dst(self.word_ids_list[idx], rec["dst_tags"])

        return {
            "hidden":       h,
            "mask":         mask,
            "intent_label": torch.tensor(intent_label, dtype=torch.long),
            "dst_labels":   torch.tensor(dst_labels,   dtype=torch.long),
        }


# ── evaluation ─────────────────────────────────────────────────────────────

def evaluate(model, loader, intent_crit, dst_crit, device):
    model.eval()
    total_loss = intent_correct = total_intent = 0
    with torch.no_grad():
        for batch in loader:
            h      = batch["hidden"].to(device)
            mask   = batch["mask"].to(device)
            int_y  = batch["intent_label"].to(device)
            dst_y  = batch["dst_labels"].to(device)

            int_log, dst_log = model(h, mask)
            loss = intent_crit(int_log, int_y) + 0.5 * dst_crit(
                dst_log.view(-1, dst_log.size(-1)), dst_y.view(-1))

            total_loss     += loss.item()
            preds           = int_log.argmax(dim=-1)
            intent_correct += (preds == int_y).sum().item()
            total_intent   += int_y.size(0)

    return total_loss / len(loader), intent_correct / total_intent


# ── main ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file",   default="training/data/multitask_train.jsonl")
    parser.add_argument("--val_file",     default="training/data/multitask_val.jsonl")
    parser.add_argument("--output_dir",   default=str(CHECKPOINT_DIR))
    parser.add_argument("--cache_dir",    default=str(CACHE_DIR))
    parser.add_argument("--bert_model",   default=BERT_MODEL_ID)
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--patience",     type=int,   default=18)
    parser.add_argument("--warmup_frac",  type=float, default=0.08)
    parser.add_argument("--label_smooth", type=float, default=0.1)
    parser.add_argument("--d_model",      type=int,   default=256)
    parser.add_argument("--n_layers",     type=int,   default=4)
    parser.add_argument("--n_heads",      type=int,   default=8)
    parser.add_argument("--d_ff",         type=int,   default=512)
    parser.add_argument("--dropout",      type=float, default=0.2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── load DistilBERT (frozen) ──────────────────────────────────────────
    print(f"[bert] Loading {args.bert_model} (downloads ~260 MB on first run)…")
    from transformers import AutoTokenizer, AutoModel
    bert_tok   = AutoTokenizer.from_pretrained(args.bert_model)
    bert_model = AutoModel.from_pretrained(args.bert_model).to(device)
    for p in bert_model.parameters():
        p.requires_grad = False
    bert_model.eval()
    print("[bert] Loaded and frozen.")

    # ── load data ─────────────────────────────────────────────────────────
    train_recs = load_jsonl(Path(args.train_file))
    val_recs   = load_jsonl(Path(args.val_file))
    cache      = Path(args.cache_dir)

    train_texts = [r["text"] for r in train_recs]
    val_texts   = [r["text"] for r in val_recs]

    train_h, train_m, train_w = precompute(
        train_texts, train_recs, bert_tok, bert_model, cache, "train", device)
    val_h, val_m, val_w = precompute(
        val_texts, val_recs, bert_tok, bert_model, cache, "val", device)

    # ── datasets & loaders ────────────────────────────────────────────────
    train_ds = BertDataset(train_recs, train_h, train_m, train_w, augment=True)
    val_ds   = BertDataset(val_recs,   val_h,   val_m,   val_w,   augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, num_workers=0)

    # ── model ─────────────────────────────────────────────────────────────
    model = TherapistTransformerV2(
        bert_dim  = 768,
        d_model   = args.d_model,
        n_heads   = args.n_heads,
        d_ff      = args.d_ff,
        n_layers  = args.n_layers,
        dropout   = args.dropout,
        max_len   = MAX_LEN,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # ── loss / optimiser / scheduler ──────────────────────────────────────
    intent_crit = LabelSmoothingCE(smoothing=args.label_smooth)
    dst_crit    = LabelSmoothingCE(smoothing=args.label_smooth, ignore_index=-100)
    optimizer   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-3)

    total_steps  = len(train_dl) * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    scheduler    = make_scheduler(optimizer, warmup_steps, total_steps)

    # ── training loop ─────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc   = 0.0
    patience_count = 0

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>9}  {'Intent Acc':>10}  {'Best Acc':>9}")
    print("─" * 56)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_dl:
            h      = batch["hidden"].to(device)
            mask   = batch["mask"].to(device)
            int_y  = batch["intent_label"].to(device)
            dst_y  = batch["dst_labels"].to(device)

            optimizer.zero_grad()
            int_log, dst_log = model(h, mask)

            loss = (intent_crit(int_log, int_y)
                    + 0.5 * dst_crit(dst_log.view(-1, dst_log.size(-1)), dst_y.view(-1)))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        val_loss, val_acc = evaluate(model, val_dl, intent_crit, dst_crit, device)
        train_loss /= len(train_dl)

        saved = ""
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            config = {
                "bert_dim": 768, "d_model": args.d_model,
                "n_heads": args.n_heads, "d_ff": args.d_ff,
                "n_layers": args.n_layers, "dropout": args.dropout,
                "max_len": MAX_LEN, "bert_model": args.bert_model,
                "n_intent": len(INTENT_LABELS), "n_dst": len(DST_LABELS),
            }
            (out_dir / "config.json").write_text(json.dumps(config, indent=2))
            saved = " ✓"
        else:
            patience_count += 1

        print(f"{epoch:>5}  {train_loss:>10.4f}  {val_loss:>9.4f}  {val_acc:>9.1%}  {best_val_acc:>8.1%}{saved}")

        if patience_count >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nBest val accuracy : {best_val_acc:.1%}")
    print(f"Checkpoint        : {out_dir}/best_model.pt")


if __name__ == "__main__":
    main()
