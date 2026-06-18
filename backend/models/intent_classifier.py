"""Intent classifier — uses custom TherapistTransformer if trained,
falls back to keyword heuristics so the app always runs.

Priority order:
  1. training/checkpoints/therapist_transformer/  ← our custom model
  2. Keyword heuristics (no model needed)
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

CUSTOM_CHECKPOINT = Path(__file__).parent / "../../training/checkpoints/therapist_transformer"


@dataclass
class IntentResult:
    label: str
    score: float
    all_scores: dict[str, float]


class IntentClassifier:
    """Loads custom TherapistTransformer if available, falls back to heuristics."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model     = None
        self._tokenizer = None
        self._load_model()

    def _load_model(self):
        if not _TORCH_AVAILABLE:
            return
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
            print("[intent] Loaded custom TherapistTransformer.")
        except Exception as e:
            print(f"[intent] Could not load custom model: {e}. Using heuristics.")

    def classify(self, text: str) -> IntentResult:
        if self._model is not None:
            return self._classify_custom(text)
        return self._classify_heuristic(text)

    _CRISIS_RE = re.compile(
        r"\b(suicid|kill myself|end my life|don.t want to live|self.?harm|"
        r"cut myself|hurt myself|want to die|end it all|no reason to live)\b",
        re.IGNORECASE,
    )

    def _classify_custom(self, text: str) -> IntentResult:
        # Safety override: crisis keywords always win regardless of model output
        if self._CRISIS_RE.search(text):
            scores = {l: 0.01 for l in INTENT_LABELS}
            scores["crisis"] = 0.99
            return IntentResult(label="crisis", score=0.99, all_scores=scores)

        ids  = self._tokenizer.encode(text)
        mask = self._tokenizer.attention_mask(ids)
        t_ids  = torch.tensor([ids],  dtype=torch.long)
        t_mask = torch.tensor([mask], dtype=torch.long)
        label, score = self._model.predict_intent(t_ids, t_mask)
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
