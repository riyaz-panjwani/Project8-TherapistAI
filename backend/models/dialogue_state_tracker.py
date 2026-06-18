"""Dialogue State Tracker using ConvBERT for slot extraction.

Tracks a rolling DST state across the entire (single) session so the
therapist always knows the user's current emotional context. Ships with
a rule-based extractor as fallback before the model is trained.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from transformers import pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

FINE_TUNED_DST_DIR = Path(__file__).parent / "../../training/checkpoints/dst"


@dataclass
class DialogueState:
    """Slots tracked across the full conversation."""
    current_mood: str = "unknown"       # positive / neutral / negative / crisis
    mood_score: float = 0.5             # 0 = very negative, 1 = very positive
    active_topics: list[str] = field(default_factory=list)
    mentioned_people: list[str] = field(default_factory=list)
    disclosed_issues: list[str] = field(default_factory=list)
    needs_followup: bool = False
    turn_count: int = 0
    last_intent: str = "general"

    def to_dict(self) -> dict:
        return asdict(self)


class DialogueStateTracker:
    """Updates DialogueState from each user turn."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._ner_pipeline = None
        self._load_model()

    def _load_model(self):
        if not _TRANSFORMERS_AVAILABLE:
            return
        dst_dir = FINE_TUNED_DST_DIR.resolve()
        if dst_dir.exists() and (dst_dir / "config.json").exists():
            self._ner_pipeline = pipeline(
                "token-classification",
                model=str(dst_dir),
                tokenizer=str(dst_dir),
                device=0 if (self.device == "cuda") else -1,
                aggregation_strategy="simple",
            )

    def update(self, state: DialogueState, text: str, intent: str) -> DialogueState:
        """Return updated state given a new user utterance."""
        state.turn_count += 1
        state.last_intent = intent

        if self._ner_pipeline is not None:
            state = self._update_model(state, text, intent)
        else:
            state = self._update_heuristic(state, text, intent)

        return state

    def _update_heuristic(self, state: DialogueState, text: str, intent: str) -> DialogueState:
        t = text.lower()

        # ── mood scoring ────────────────────────────────────────────────
        negative_words = ["sad", "anxious", "depressed", "hopeless", "angry", "scared",
                          "overwhelmed", "tired", "worthless", "empty", "alone", "hopeless"]
        positive_words = ["happy", "better", "good", "grateful", "calm", "hopeful",
                          "excited", "proud", "okay", "fine", "improving"]
        crisis_words   = ["suicide", "kill myself", "self-harm", "hurt myself", "end it all"]

        neg_hits = sum(1 for w in negative_words if w in t)
        pos_hits = sum(1 for w in positive_words if w in t)
        crisis_hit = any(w in t for w in crisis_words)

        if crisis_hit:
            state.current_mood = "crisis"
            state.mood_score = 0.0
        else:
            delta = (pos_hits - neg_hits) * 0.1
            state.mood_score = max(0.0, min(1.0, state.mood_score + delta))
            if state.mood_score >= 0.65:
                state.current_mood = "positive"
            elif state.mood_score >= 0.4:
                state.current_mood = "neutral"
            else:
                state.current_mood = "negative"

        # ── topic extraction ─────────────────────────────────────────────
        topic_patterns = {
            "relationship": r"\b(partner|boyfriend|girlfriend|husband|wife|friend|family|mother|father|brother|sister)\b",
            "work":         r"\b(work|job|boss|career|office|colleagues|fired|deadline|salary)\b",
            "academic":     r"\b(exam|uni|university|college|school|study|grades|thesis|dissertation)\b",
            "health":       r"\b(health|doctor|medication|sleep|eating|exercise|sick|ill|pain)\b",
            "self-image":   r"\b(ugly|fat|stupid|worthless|hate myself|not good enough|confidence)\b",
            "trauma":       r"\b(trauma|abuse|assault|ptsd|flashback|childhood|nightmare)\b",
        }
        for topic, pattern in topic_patterns.items():
            if re.search(pattern, t) and topic not in state.active_topics:
                state.active_topics.append(topic)
                if topic not in state.disclosed_issues:
                    state.disclosed_issues.append(topic)

        # ── people mentions ───────────────────────────────────────────────
        people_re = r"\b(my (mum|mom|dad|father|mother|brother|sister|friend|partner|boyfriend|girlfriend|husband|wife|therapist|teacher|boss|colleague))\b"
        for match in re.finditer(people_re, t):
            person = match.group(1)
            if person not in state.mentioned_people:
                state.mentioned_people.append(person)

        # flag follow-up if crisis or seeking advice
        state.needs_followup = intent in ("crisis", "seeking_advice", "anxiety", "depression")

        return state

    def _update_model(self, state: DialogueState, text: str, intent: str) -> DialogueState:
        entities = self._ner_pipeline(text[:512])
        for ent in entities:
            label = ent.get("entity_group", "")
            word  = ent.get("word", "").strip()
            if label == "PERSON" and word not in state.mentioned_people:
                state.mentioned_people.append(word)
            elif label in ("TOPIC", "ISSUE") and word not in state.active_topics:
                state.active_topics.append(word)
        # still run heuristic for mood
        return self._update_heuristic(state, text, intent)


_tracker: Optional[DialogueStateTracker] = None


def get_tracker() -> DialogueStateTracker:
    global _tracker
    if _tracker is None:
        device = "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        _tracker = DialogueStateTracker(device=device)
    return _tracker
