"""Intent classifier — uses custom TherapistTransformer if trained,
falls back to keyword heuristics so the app always runs.

Priority order:
  1. training/checkpoints/therapist_transformer_v2/  ← DistilBERT-backed V2 model (~85-93%)
  2. training/checkpoints/therapist_transformer/     ← GloVe V1 model (~71%)
  3. Keyword heuristics (no model needed)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

INTENT_LABELS = [
    "venting", "seeking_advice", "anxiety", "depression", "crisis",
    "gratitude", "relationship", "work_stress", "self_esteem",
    "checking_in", "trauma", "progress", "general",
]

CUSTOM_CHECKPOINT    = Path(__file__).parent / "../../training/checkpoints/therapist_transformer"
CUSTOM_CHECKPOINT_V2 = Path(__file__).parent / "../../training/checkpoints/therapist_transformer_v2"


@dataclass
class IntentResult:
    label: str
    score: float
    all_scores: dict[str, float]


class IntentClassifier:
    """Loads best available model; V2 (DistilBERT) > V1 (GloVe) > heuristics."""

    def __init__(self, device: str = "cpu"):
        self.device      = device
        self._model      = None
        self._tokenizer  = None   # V1 custom Tokenizer
        # V2-specific
        self._bert_tok   = None
        self._bert_model = None
        self._is_v2      = False
        self._load_model()

    def _load_model(self):
        if not _TORCH_AVAILABLE:
            return
        if self._try_load_v2():
            return
        self._try_load_v1()

    def _try_load_v2(self) -> bool:
        ckpt = CUSTOM_CHECKPOINT_V2.resolve()
        if not (ckpt / "best_model.pt").exists():
            return False
        try:
            import json
            from transformers import AutoTokenizer, AutoModel
            from models.custom_transformer import TherapistTransformerV2

            config = json.loads((ckpt / "config.json").read_text())
            bert_id = config.get("bert_model", "distilbert-base-uncased")

            bert_tok   = AutoTokenizer.from_pretrained(bert_id)
            bert_model = AutoModel.from_pretrained(bert_id).to(self.device)
            for p in bert_model.parameters():
                p.requires_grad = False
            bert_model.eval()

            model = TherapistTransformerV2(
                bert_dim = config.get("bert_dim", 768),
                d_model  = config.get("d_model", 256),
                n_heads  = config.get("n_heads", 8),
                d_ff     = config.get("d_ff", 512),
                n_layers = config.get("n_layers", 4),
                dropout  = config.get("dropout", 0.2),
                max_len  = config.get("max_len", 128),
            ).to(self.device)
            model.load_state_dict(
                torch.load(ckpt / "best_model.pt", map_location=self.device)
            )
            model.eval()

            self._model      = model
            self._bert_tok   = bert_tok
            self._bert_model = bert_model
            self._is_v2      = True
            print("[intent] Loaded TherapistTransformerV2 (DistilBERT features).")
            return True
        except Exception as e:
            print(f"[intent] Could not load V2 model: {e}.")
            return False

    def _try_load_v1(self):
        ckpt = CUSTOM_CHECKPOINT.resolve()
        if not (ckpt / "best_model.pt").exists():
            return
        try:
            import json
            from models.custom_transformer import TherapistTransformer, Tokenizer
            config = json.loads((ckpt / "config.json").read_text())
            model  = TherapistTransformer(**config)
            model.load_state_dict(
                torch.load(ckpt / "best_model.pt", map_location=self.device)
            )
            model.eval()
            self._model     = model
            self._tokenizer = Tokenizer.load(ckpt / "tokenizer.json")
            self._is_v2     = False
            print("[intent] Loaded custom TherapistTransformer V1 (GloVe).")
        except Exception as e:
            print(f"[intent] Could not load custom model: {e}. Using heuristics.")

    def classify(self, text: str) -> IntentResult:
        if self._model is not None:
            result = self._classify_custom(text)
            # When model confidence is low, blend with heuristic
            if result.score < 0.55:
                heuristic = self._classify_heuristic(text)
                if heuristic.score > 0.55 and heuristic.label != result.label:
                    return heuristic
            return result
        return self._classify_heuristic(text)

    _CRISIS_RE = re.compile(
        r"\b(suicid|kill myself|end my life|don.t want to live|self.?harm|"
        r"cut myself|hurt myself|want to die|end it all|no reason to live)\b",
        re.IGNORECASE,
    )

    # Strong domain-specific overrides: two or more hits → use heuristic label
    _STRONG_PATTERNS: dict[str, re.Pattern] = {
        "work_stress": re.compile(
            r"\b(boss|manager|deadline|workload|burnout|fired|redundant|"
            r"my job|at work|office|exam|dissertation|uni|university|coursework)\b",
            re.IGNORECASE,
        ),
        "relationship": re.compile(
            r"\b(my (partner|boyfriend|girlfriend|husband|wife|mum|mom|dad|"
            r"brother|sister|friend|family))\b",
            re.IGNORECASE,
        ),
        "anxiety": re.compile(
            r"\b(anxious|anxiety|panic|panic attack|worried|worrying|dread|"
            r"overwhelm|racing thoughts|can.t breathe)\b",
            re.IGNORECASE,
        ),
        "depression": re.compile(
            r"\b(depressed|depression|hopeless|numb|empty|worthless|can.t get up|"
            r"no point|nothing matters)\b",
            re.IGNORECASE,
        ),
        "gratitude": re.compile(
            r"\b(thank you|thanks|grateful|appreciate|helped me|feel better|"
            r"so helpful|really helped)\b",
            re.IGNORECASE,
        ),
    }

    def _classify_custom(self, text: str) -> IntentResult:
        # Safety override: crisis keywords always win regardless of model output
        if self._CRISIS_RE.search(text):
            scores = {l: 0.01 for l in INTENT_LABELS}
            scores["crisis"] = 0.99
            return IntentResult(label="crisis", score=0.99, all_scores=scores)

        if self._is_v2:
            enc = self._bert_tok(
                text, max_length=128, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            with torch.no_grad():
                out = self._bert_model(
                    input_ids=enc["input_ids"].to(self.device),
                    attention_mask=enc["attention_mask"].to(self.device),
                )
            label, score = self._model.predict_intent(
                out.last_hidden_state, enc["attention_mask"].to(self.device)
            )
        else:
            ids  = self._tokenizer.encode(text)
            mask = self._tokenizer.attention_mask(ids)
            t_ids  = torch.tensor([ids],  dtype=torch.long)
            t_mask = torch.tensor([mask], dtype=torch.long)
            label, score = self._model.predict_intent(t_ids, t_mask)

        # Strong pattern override: if 2+ strong hits for a class the model missed, trust them
        for override_label, pattern in self._STRONG_PATTERNS.items():
            hits = len(pattern.findall(text))
            if hits >= 2 and label != override_label:
                scores_out = {l: 0.05 for l in INTENT_LABELS}
                scores_out[override_label] = 0.85
                return IntentResult(label=override_label, score=0.85, all_scores=scores_out)
            if hits == 1 and score < 0.50 and label != override_label:
                scores_out = {l: 0.05 for l in INTENT_LABELS}
                scores_out[override_label] = 0.75
                return IntentResult(label=override_label, score=0.75, all_scores=scores_out)

        all_scores = {l: (score if l == label else 0.0) for l in INTENT_LABELS}
        return IntentResult(label=label, score=score, all_scores=all_scores)

    def _classify_heuristic(self, text: str) -> IntentResult:
        t = text.lower()
        scores: dict[str, float] = {label: 0.05 for label in INTENT_LABELS}

        patterns = {
            "crisis":        [r"\b(suicid|kill myself|end my life|don't want to live|self.?harm|cut myself|hurt myself)\b"],
            "anxiety":       [r"\b(anxious|anxiety|panic|overwhelm|worried|can't breathe|racing thoughts|dread)\b"],
            "depression":    [r"\b(depressed|hopeless|empty|numb|worthless|can't get up|no point|exhausted|sad)\b"],
            "venting":       [r"\b(so frustrated|i hate|fed up|can't take|sick of|pissed|furious|so angry)\b"],
            "seeking_advice":[r"\b(what should i|how do i|any advice|help me|what can i|what do you think)\b"],
            "gratitude":     [r"\b(thank you|thanks|grateful|appreciate|helped me|feeling better)\b"],
            "relationship":  [r"\b(boyfriend|girlfriend|partner|friend|family|mother|father|siblings|alone|lonely|relationship)\b"],
            "work_stress":   [r"\b(work|job|boss|deadline|career|exam|study|uni|college|school|fired|burnout)\b"],
            "self_esteem":   [r"\b(confident|ugly|stupid|not good enough|hate myself|loser|worthless)\b"],
            "checking_in":   [r"^(hi|hello|hey|good morning|good evening|how are you|just checking in)[\s!?\.]*$"],
            "trauma":        [r"\b(trauma|ptsd|abuse|assault|childhood|flashback|nightmare|never forget)\b"],
            "progress":      [r"\b(doing better|feeling better|made progress|proud of myself|achieved|improving)\b"],
        }

        for label, pats in patterns.items():
            for pat in pats:
                if re.search(pat, t):
                    scores[label] = min(scores[label] + 0.6, 0.95)

        if scores["crisis"] > 0.5:
            scores = {k: (0.95 if k == "crisis" else 0.01) for k in scores}

        total = sum(scores.values())
        scores = {k: v / total for k, v in scores.items()}
        best_label = max(scores, key=lambda k: scores[k])
        return IntentResult(label=best_label, score=scores[best_label], all_scores=scores)


_classifier: Optional[IntentClassifier] = None


def get_classifier() -> IntentClassifier:
    global _classifier
    if _classifier is None:
        device = "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        _classifier = IntentClassifier(device=device)
    return _classifier
