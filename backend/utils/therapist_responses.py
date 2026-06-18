"""Response generation — local generative model + template fallback.

Pipeline:
  1. Try local BlenderBot (facebook/blenderbot-400M-distill) — empathetic,
     reads actual conversation content, no API key needed.
  2. Fall back to intent-keyed templates if model not loaded yet.
"""
from __future__ import annotations

import random
from models.dialogue_state_tracker import DialogueState


def build_response(
    intent: str,
    state: DialogueState,
    profile_summary: str,
    user_name: str,
    message_count: int,
    history_snippet: list[dict],
) -> str:
    # try local generative model first
    try:
        from models.response_generator import generate
        reply = generate(
            history=history_snippet,
            intent=intent,
            mood=state.current_mood,
            topics=state.active_topics,
            people=state.mentioned_people,
        )
        if reply:
            if intent == "crisis":
                reply += _CRISIS_RESOURCES
            return reply
    except Exception:
        pass

    # fallback to templates
    return _build_template_response(intent, state, user_name, message_count)


# ── template fallback ─────────────────────────────────────────────

_OPENINGS: dict[str, list[str]] = {
    "crisis":        ["I can hear how much pain you're in right now, and I'm really glad you told me.",
                      "That sounds incredibly difficult. Thank you for trusting me with this."],
    "anxiety":       ["That sounds exhausting to carry.",
                      "Anxiety has a way of making everything feel urgent at once."],
    "depression":    ["What you're describing sounds really heavy.",
                      "That kind of emptiness can be so draining."],
    "venting":       ["I'm here — let it all out.",
                      "That frustration makes total sense."],
    "seeking_advice":["Let's think through this together.",
                      "I want to make sure I understand the situation fully first."],
    "relationship":  ["That dynamic sounds complicated.",
                      "Navigating people we care about is never simple."],
    "work_stress":   ["Work pressure has a way of seeping into every corner of life.",
                      "That sounds like a lot of weight to carry."],
    "self_esteem":   ["I notice you're being quite hard on yourself there.",
                      "It takes a lot to say those things out loud."],
    "trauma":        ["Thank you for sharing something so personal.",
                      "It's okay to take this at whatever pace feels right."],
    "gratitude":     ["I'm really glad that's been helpful.",
                      "That's a real shift — and it's yours."],
    "progress":      ["That's genuinely worth celebrating.",
                      "Notice what you just said — that's growth."],
    "checking_in":   ["Good to have you here.", "It's good to hear from you."],
    "general":       ["I'm listening.", "Tell me more."],
}

_FOLLOWUPS: dict[str, list[str]] = {
    "crisis":        ["Are you safe right now?", "What does this moment feel like?"],
    "anxiety":       ["When did you first start feeling this way?", "Is there a specific thought that keeps coming back?"],
    "depression":    ["How long have you been carrying this?", "What does a typical day look like for you right now?"],
    "venting":       ["What part of this feels most unfair?", "How long has this been building?"],
    "seeking_advice":["What have you already tried?", "What does your gut say?"],
    "relationship":  ["How do you feel when you're around this person?", "What would feel like a fair outcome?"],
    "work_stress":   ["What's the part that's hardest to let go of?", "What would 'good enough' look like here?"],
    "self_esteem":   ["Where do you think that belief came from?", "What would you need to feel differently?"],
    "trauma":        ["Is this something you've been able to talk about before?", "How do you usually cope when it comes up?"],
    "gratitude":     ["What do you think made the difference?", "How does it feel compared to where you were?"],
    "progress":      ["What helped you get there?", "What do you want to build on next?"],
    "checking_in":   ["How have things been since we last spoke?", "What's been on your mind?"],
    "general":       ["What's been sitting with you most heavily?", "How are you really doing?"],
}

_CRISIS_RESOURCES = (
    "\n\n---\n**If you're in immediate danger, please reach out:**\n"
    "- **UK:** Samaritans 116 123 (free, 24/7)\n"
    "- **US:** 988 Suicide & Crisis Lifeline (call or text 988)\n"
    "- **International:** findahelpline.com\n---"
)


def _build_template_response(intent: str, state: DialogueState, user_name: str, message_count: int) -> str:
    opening  = random.choice(_OPENINGS.get(intent, _OPENINGS["general"]))
    followup = random.choice(_FOLLOWUPS.get(intent, _FOLLOWUPS["general"]))
    name_prefix = f"{user_name}, " if user_name and message_count <= 3 else ""
    reply = f"{name_prefix}{opening}\n\n{followup}"
    if intent == "crisis":
        reply += _CRISIS_RESOURCES
    return reply


def first_greeting(user_name: str = "") -> str:
    name = f", {user_name}" if user_name else ""
    return (
        f"Hello{name}. I'm glad you're here.\n\n"
        "This is a space just for you — no judgements, no rush. "
        "Everything you share stays between us, and I'll remember it all "
        "so you never have to repeat yourself.\n\n"
        "What's on your mind?"
    )


def returning_greeting(user_name: str, message_count: int, last_theme: str = "") -> str:
    name = f", {user_name}" if user_name else ""
    theme_line = f" Last time we were talking about {last_theme}." if last_theme else ""
    return (
        f"Welcome back{name}.{theme_line}\n\n"
        f"We've had {message_count} exchanges together, and I remember all of it. "
        "How are you doing today?"
    )
