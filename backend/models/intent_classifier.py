"""RoBERTa-based intent classifier for therapy conversation turns.

Labels align with common therapeutic communication categories plus
crisis detection. In production, fine-tune on MultiWOZ + a therapy
dataset (see training/scripts/train_intent.py). This module ships with
a zero-shot fallback using keyword heuristics so the app runs before
training is done.
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

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

INTENT_LABELS = [
    "venting",          # expressing emotions without asking for advice
    "seeking_advice",   # asking for guidance / what should I do
    "anxiety",          # worry, fear, panic, overwhelm
    "depression",       # sadness, hopelessness, emptiness
    "crisis",           # self-harm, suicidal ideation — top priority
    "gratitude",        # thanking, positive feedback to therapist
    "relationship",     # conflict with others, loneliness
    "work_stress",      # job, career, academic pressure
    "self_esteem",      # confidence, identity, body image
    "checking_in",      # casual opener, "how are you" type messages
    "trauma",           # past experiences, PTSD
    "progress",         # reporting improvement, wins
    "general",          # catch-all
]

FINE_TUNED_MODEL_DIR = Path(__file__).parent / "../../training/checkpoints/intent"


@dataclass
class IntentResult:
    label: str
    score: float
    all_scores: dict[str, float]


class IntentClassifier:
    """Loads fine-tuned RoBERTa if available, falls back to heuristics."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._pipeline = None
        self._load_model()

    def _load_model(self):
        if not _TRANSFORMERS_AVAILABLE:
            return
        model_dir = FINE_TUNED_MODEL_DIR.resolve()
        if model_dir.exists() and (model_dir / "config.json").exists():
            self._pipeline = pipeline(
                "text-classification",
                model=str(model_dir),
                tokenizer=str(model_dir),
                device=0 if (self.device == "cuda") else -1,
                top_k=None,
            )

    def classify(self, text: str) -> IntentResult:
        if self._pipeline is not None:
            return self._classify_model(text)
        return self._classify_heuristic(text)

    def _classify_model(self, text: str) -> IntentResult:
        results = self._pipeline(text[:512])[0]
        scores = {r["label"]: r["score"] for r in results}
        best = max(results, key=lambda r: r["score"])
        return IntentResult(label=best["label"], score=best["score"], all_scores=scores)

    def _classify_heuristic(self, text: str) -> IntentResult:
        t = text.lower()
        scores: dict[str, float] = {label: 0.05 for label in INTENT_LABELS}

        patterns = {
            "crisis": [
                r"\b(suicid|kill myself|end my life|don't want to live|self.?harm|cut myself|hurt myself)\b"
            ],
            "anxiety": [
                r"\b(anxious|anxiety|panic|overwhelm|worried|can't breathe|racing thoughts|dread)\b"
            ],
            "depression": [
                r"\b(depressed|hopeless|empty|numb|worthless|can't get up|no point|exhausted|sad)\b"
            ],
            "venting": [
                r"\b(so frustrated|i hate|fed up|can't take|sick of|pissed|furious|so angry)\b"
            ],
            "seeking_advice": [
                r"\b(what should i|how do i|any advice|help me|what can i|what do you think)\b"
            ],
            "gratitude": [
                r"\b(thank you|thanks|grateful|appreciate|helped me|feeling better)\b"
            ],
            "relationship": [
                r"\b(boyfriend|girlfriend|partner|friend|family|mother|father|siblings|alone|lonely|relationship)\b"
            ],
            "work_stress": [
                r"\b(work|job|boss|deadline|career|exam|study|uni|college|school|fired|burnout)\b"
            ],
            "self_esteem": [
                r"\b(confident|ugly|stupid|not good enough|hate myself|loser|worthless)\b"
            ],
            "checking_in": [
                r"^(hi|hello|hey|good morning|good evening|how are you|just checking in)[\s!?\.]*$"
            ],
            "trauma": [
                r"\b(trauma|ptsd|abuse|assault|childhood|flashback|nightmare|never forget)\b"
            ],
            "progress": [
                r"\b(doing better|feeling better|made progress|proud of myself|achieved|improving)\b"
            ],
        }

        for label, pats in patterns.items():
            for pat in pats:
                if re.search(pat, t):
                    scores[label] = min(scores[label] + 0.6, 0.95)

        # crisis always wins if triggered
        if scores["crisis"] > 0.5:
            scores = {k: (0.95 if k == "crisis" else 0.01) for k in scores}

        total = sum(scores.values())
        scores = {k: v / total for k, v in scores.items()}
        best_label = max(scores, key=lambda k: scores[k])

        # normalise to sensible top score
        all_scores = scores.copy()
        if best_label == "general" and all(v < 0.3 for v in scores.values()):
            scores["general"] = 0.7

        return IntentResult(label=best_label, score=scores[best_label], all_scores=all_scores)


# module-level singleton
_classifier: Optional[IntentClassifier] = None


def get_classifier() -> IntentClassifier:
    global _classifier
    if _classifier is None:
        device = "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        _classifier = IntentClassifier(device=device)
    return _classifier
