"""
TherapistTransformer — custom multi-task Transformer built from scratch.

Every component (attention, positional encoding, feed-forward, layer norm)
is hand-coded in PyTorch. No transformers library used for the architecture.

Transfer learning: the embedding layer is initialised from GloVe 100d
vectors, giving the model strong lexical representations without training
on billions of tokens.

Architecture
────────────
Input tokens
    │
    ▼
Embedding  (GloVe 100d, fine-tuned)
    │
    ▼
Linear projection  100 → d_model=256
    │
    ▼
Sinusoidal Positional Encoding
    │
    ▼
TransformerEncoder  (n_layers=4, Pre-LN)
  ├─ MultiHeadSelfAttention  (n_heads=8, d_k=32)
  ├─ Add & LayerNorm
  ├─ FeedForward  (d_ff=512, GELU)
  └─ Add & LayerNorm
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
[CLS] + mean-pool concat [B,2·d_model]  token hidden states
    │                                      │
    ▼                                      ▼
Intent Head                            DST Head
(MLP: 512→256→13, GELU)               (linear → 9 BIO tags)
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

# ── label sets (shared with training scripts) ──────────────────────────────

INTENT_LABELS = [
    "venting", "seeking_advice", "anxiety", "depression", "crisis",
    "gratitude", "relationship", "work_stress", "self_esteem",
    "checking_in", "trauma", "progress", "general",
]

DST_LABELS = [
    "O",
    "B-TOPIC", "I-TOPIC",
    "B-ISSUE", "I-ISSUE",
    "B-PERSON", "I-PERSON",
    "B-MOOD",  "I-MOOD",
]

INTENT2ID = {l: i for i, l in enumerate(INTENT_LABELS)}
ID2INTENT = {i: l for i, l in enumerate(INTENT_LABELS)}
DST2ID    = {l: i for i, l in enumerate(DST_LABELS)}
ID2DST    = {i: l for i, l in enumerate(DST_LABELS)}

# ── hyper-parameters ───────────────────────────────────────────────────────

D_EMBED  = 100    # GloVe vector size
D_MODEL  = 256    # internal Transformer dimension
N_HEADS  = 8      # attention heads  (D_MODEL / N_HEADS = 32)
D_FF     = 512    # feed-forward hidden size
N_LAYERS = 4      # Transformer encoder depth
DROPOUT  = 0.2    # dropout (lowered — larger dataset + model)
MAX_LEN  = 128    # max token sequence length

# ════════════════════════════════════════════════════════════════════════════
# 1. Tokenizer
# ════════════════════════════════════════════════════════════════════════════

class Tokenizer:
    """Word-level tokenizer with a fixed vocabulary built from training data."""

    PAD = "[PAD]"   # id 0
    UNK = "[UNK]"   # id 1
    CLS = "[CLS]"   # id 2 — prepended to every sequence; used for classification
    SEP = "[SEP]"   # id 3

    SPECIAL = [PAD, UNK, CLS, SEP]

    def __init__(self, max_len: int = MAX_LEN):
        self.max_len = max_len
        self.word2id: dict[str, int] = {t: i for i, t in enumerate(self.SPECIAL)}
        self.id2word: dict[int, str] = {i: t for i, t in enumerate(self.SPECIAL)}

    # ── vocabulary ────────────────────────────────────────────────────────

    def build_vocab(self, texts: list[str], max_vocab: int = 12_000):
        from collections import Counter
        counts: Counter = Counter()
        for t in texts:
            counts.update(self._split(t))
        for word, _ in counts.most_common(max_vocab - len(self.SPECIAL)):
            if word not in self.word2id:
                idx = len(self.word2id)
                self.word2id[word] = idx
                self.id2word[idx]  = word

    @property
    def vocab_size(self) -> int:
        return len(self.word2id)

    # ── encode ────────────────────────────────────────────────────────────

    def _split(self, text: str) -> list[str]:
        """Lowercase + split on whitespace/punctuation."""
        text = text.lower()
        text = re.sub(r"([^\w\s'])", r" \1 ", text)
        return text.split()

    def tokenize(self, text: str) -> list[str]:
        return self._split(text)

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        tokens = self._split(text)
        ids = [self.word2id.get(t, self.word2id[self.UNK]) for t in tokens]
        if add_special:
            ids = [self.word2id[self.CLS]] + ids + [self.word2id[self.SEP]]
        # truncate + pad
        ids = ids[:self.max_len]
        ids += [self.word2id[self.PAD]] * (self.max_len - len(ids))
        return ids

    def attention_mask(self, ids: list[int]) -> list[int]:
        return [1 if i != self.word2id[self.PAD] else 0 for i in ids]

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps({"word2id": self.word2id, "max_len": self.max_len}))

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        data = json.loads(Path(path).read_text())
        tok = cls(max_len=data["max_len"])
        tok.word2id = data["word2id"]
        tok.id2word = {int(i): w for w, i in tok.word2id.items()}
        return tok


# ════════════════════════════════════════════════════════════════════════════
# 2. Transformer components (all hand-coded)
# ════════════════════════════════════════════════════════════════════════════

if _TORCH_OK:

    class PositionalEncoding(nn.Module):
        """
        Sinusoidal positional encoding from 'Attention Is All You Need'.

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Adding these fixed vectors to token embeddings gives the model
        a sense of token order without learned position parameters.
        """

        def __init__(self, d_model: int, max_len: int = MAX_LEN, dropout: float = DROPOUT):
            super().__init__()
            self.dropout = nn.Dropout(dropout)

            # pre-compute the encoding matrix  [max_len, d_model]
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)           # [L, 1]
            div = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float)
                * -(math.log(10_000.0) / d_model)
            )                                                                          # [d_model/2]
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))                               # [1, L, d_model]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, T, d_model]
            return self.dropout(x + self.pe[:, :x.size(1)])


    class MultiHeadSelfAttention(nn.Module):
        """
        Multi-head self-attention.

        Projects input into Q, K, V spaces for each head, computes scaled
        dot-product attention, then projects the concatenated heads back.

        Scaled dot-product:  Attention(Q,K,V) = softmax(QK^T / √d_k) V
        """

        def __init__(self, d_model: int, n_heads: int, dropout: float = DROPOUT):
            super().__init__()
            assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
            self.n_heads = n_heads
            self.d_k = d_model // n_heads

            self.W_q = nn.Linear(d_model, d_model, bias=False)
            self.W_k = nn.Linear(d_model, d_model, bias=False)
            self.W_v = nn.Linear(d_model, d_model, bias=False)
            self.W_o = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            B, T, D = x.shape
            H, d_k = self.n_heads, self.d_k

            # project & split into heads  →  [B, H, T, d_k]
            def split(w):
                return w(x).view(B, T, H, d_k).transpose(1, 2)

            Q, K, V = split(self.W_q), split(self.W_k), split(self.W_v)

            # scaled dot-product attention
            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)   # [B, H, T, T]
            if mask is not None:
                # mask: [B, T] → [B, 1, 1, T]
                scores = scores.masked_fill(mask[:, None, None, :] == 0, -1e9)
            attn = self.dropout(torch.softmax(scores, dim=-1))

            # aggregate values and merge heads
            out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
            return self.W_o(out)


    class FeedForward(nn.Module):
        """
        Position-wise feed-forward: two linear layers with GELU activation.

        FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
        """

        def __init__(self, d_model: int, d_ff: int, dropout: float = DROPOUT):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    class TransformerEncoderLayer(nn.Module):
        """
        One Transformer encoder layer: self-attention + feed-forward,
        each wrapped with a residual connection and layer normalisation.

        Pre-LN variant (norm before sub-layer) — more stable than original
        Post-LN for deep models.
        """

        def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = DROPOUT):
            super().__init__()
            self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
            self.ff   = FeedForward(d_model, d_ff, dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            # pre-norm + residual
            x = x + self.attn(self.norm1(x), mask)
            x = x + self.ff(self.norm2(x))
            return x


    class TransformerEncoder(nn.Module):
        """Stack of TransformerEncoderLayer modules."""

        def __init__(
            self,
            n_layers: int,
            d_model:  int,
            n_heads:  int,
            d_ff:     int,
            dropout:  float = DROPOUT,
        ):
            super().__init__()
            self.layers = nn.ModuleList([
                TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            for layer in self.layers:
                x = layer(x, mask)
            return self.norm(x)


    # ════════════════════════════════════════════════════════════════════════
    # 3. Full multi-task model
    # ════════════════════════════════════════════════════════════════════════

    class TherapistTransformer(nn.Module):
        """
        Multi-task Transformer: shared encoder, two output heads.

        Intent head  — classifies the whole utterance via the [CLS] token.
        DST head     — tags each token with a BIO slot label (NER-style).

        Transfer learning: call load_glove_embeddings() to initialise the
        embedding layer from pre-trained GloVe 100d vectors. The embeddings
        are then fine-tuned jointly with the rest of the model.

        Intent representation: [CLS] hidden state concatenated with the
        mean-pooled non-pad hidden states → richer whole-utterance signal.
        """

        def __init__(
            self,
            vocab_size:    int,
            n_intent:      int  = len(INTENT_LABELS),
            n_dst:         int  = len(DST_LABELS),
            d_embed:       int  = D_EMBED,
            d_model:       int  = D_MODEL,
            n_heads:       int  = N_HEADS,
            d_ff:          int  = D_FF,
            n_layers:      int  = N_LAYERS,
            dropout:       float= DROPOUT,
            max_len:       int  = MAX_LEN,
            pad_id:        int  = 0,
        ):
            super().__init__()
            self.pad_id = pad_id

            # embedding + projection
            self.embedding  = nn.Embedding(vocab_size, d_embed, padding_idx=pad_id)
            self.proj       = nn.Linear(d_embed, d_model)
            self.pos_enc    = PositionalEncoding(d_model, max_len, dropout)

            # shared encoder
            self.encoder = TransformerEncoder(n_layers, d_model, n_heads, d_ff, dropout)

            # intent head: MLP over [CLS ∥ mean-pool] → richer utterance repr
            self.intent_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_intent),
            )
            self.dst_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model, n_dst),
            )

        def forward(
            self,
            input_ids: torch.Tensor,          # [B, T]
            attention_mask: Optional[torch.Tensor] = None,  # [B, T]
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Returns:
                intent_logits  [B, n_intent]
                dst_logits     [B, T, n_dst]
            """
            if attention_mask is None:
                attention_mask = (input_ids != self.pad_id).long()

            # embed → project → positional encoding
            x = self.proj(self.embedding(input_ids))    # [B, T, d_model]
            x = self.pos_enc(x)

            # shared Transformer encoder
            x = self.encoder(x, attention_mask)         # [B, T, d_model]

            # [CLS] token + mean-pool over non-pad tokens → richer intent repr
            cls_hidden  = x[:, 0, :]                    # [B, d_model]
            mask_f      = attention_mask.unsqueeze(-1).float()
            mean_hidden = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            intent_logits = self.intent_head(
                torch.cat([cls_hidden, mean_hidden], dim=-1)   # [B, 2·d_model]
            )

            # all token positions → DST tags
            dst_logits = self.dst_head(x)               # [B, T, n_dst]

            return intent_logits, dst_logits

        # ── transfer learning ────────────────────────────────────────────

        def load_glove_embeddings(
            self,
            glove_path: str | Path,
            tokenizer:  "Tokenizer",
        ) -> int:
            """
            Initialise embedding weights from GloVe vectors.

            Words not found in GloVe keep their random initialisation.
            Returns number of words successfully loaded.
            """
            glove_path = Path(glove_path)
            if not glove_path.exists():
                print(f"[embed] GloVe file not found at {glove_path}. Skipping.")
                return 0

            print(f"[embed] Loading GloVe vectors from {glove_path}…")
            glove: dict[str, np.ndarray] = {}
            with open(glove_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    word  = parts[0]
                    vec   = np.array(parts[1:], dtype=np.float32)
                    glove[word] = vec

            weight = self.embedding.weight.data
            loaded = 0
            for word, idx in tokenizer.word2id.items():
                if word in glove and idx < weight.shape[0]:
                    weight[idx] = torch.tensor(glove[word])
                    loaded += 1

            print(f"[embed] Loaded {loaded}/{tokenizer.vocab_size} vectors.")
            return loaded

        # ── convenience inference methods ────────────────────────────────

        def predict_intent(
            self,
            input_ids:      torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> tuple[str, float]:
            self.eval()
            with torch.no_grad():
                logits, _ = self(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=-1)[0]
                idx   = int(probs.argmax())
                return ID2INTENT[idx], float(probs[idx])

        def predict_dst(
            self,
            input_ids:      torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> list[tuple[str, str]]:
            """Returns list of (token_id, BIO_label) for non-PAD tokens."""
            self.eval()
            with torch.no_grad():
                _, logits = self(input_ids, attention_mask)
                preds = logits[0].argmax(dim=-1).tolist()
                mask  = (input_ids[0] != self.pad_id).tolist()
                return [(str(i), ID2DST[p]) for i, (p, m) in enumerate(zip(preds, mask)) if m]
