"""Unit tests for heuristic intent classifier (no model weights needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from models.intent_classifier import IntentClassifier

clf = IntentClassifier()   # heuristic mode (no model dir)


def test_crisis_detected():
    r = clf.classify("I want to kill myself tonight")
    assert r.label == "crisis"
    assert r.score > 0.8


def test_anxiety_detected():
    r = clf.classify("I feel so anxious and overwhelmed all the time")
    assert r.label in ("anxiety", "depression")


def test_seeking_advice():
    r = clf.classify("What should I do about my relationship?")
    assert r.label == "seeking_advice"


def test_gratitude():
    r = clf.classify("Thank you so much, I feel so much better now")
    assert r.label == "gratitude"


def test_checking_in():
    r = clf.classify("hi")
    assert r.label == "checking_in"


def test_work_stress():
    r = clf.classify("I have three deadlines this week and I'm burning out at work")
    assert r.label == "work_stress"


def test_scores_sum_to_one():
    r = clf.classify("I feel really sad and lonely today")
    total = sum(r.all_scores.values())
    assert abs(total - 1.0) < 0.01


def test_crisis_wins_over_all():
    r = clf.classify("I'm so sad and I want to hurt myself")
    assert r.label == "crisis"
