"""Unit tests for heuristic dialogue state tracker."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from models.dialogue_state_tracker import DialogueStateTracker, DialogueState

tracker = DialogueStateTracker()


def fresh():
    return DialogueState()


def test_mood_goes_negative_on_sad_text():
    state = fresh()
    state = tracker.update(state, "I feel so sad and hopeless", "depression")
    assert state.current_mood in ("negative", "crisis")
    assert state.mood_score < 0.5


def test_crisis_mood_on_crisis_text():
    state = fresh()
    state = tracker.update(state, "I want to kill myself", "crisis")
    assert state.current_mood == "crisis"
    assert state.mood_score == 0.0


def test_topic_relationship_extracted():
    state = fresh()
    state = tracker.update(state, "My boyfriend and I keep fighting", "relationship")
    assert "relationship" in state.active_topics


def test_topic_work_extracted():
    state = fresh()
    state = tracker.update(state, "I'm so stressed about my job and deadlines", "work_stress")
    assert "work" in state.active_topics


def test_turn_counter_increments():
    state = fresh()
    state = tracker.update(state, "hello", "checking_in")
    state = tracker.update(state, "I feel okay", "general")
    assert state.turn_count == 2


def test_people_extracted():
    state = fresh()
    state = tracker.update(state, "My mum doesn't understand me", "relationship")
    assert any("mum" in p for p in state.mentioned_people)


def test_positive_mood_on_good_text():
    state = fresh()
    state.mood_score = 0.5
    state = tracker.update(state, "I feel really happy and grateful today", "gratitude")
    assert state.mood_score > 0.5
