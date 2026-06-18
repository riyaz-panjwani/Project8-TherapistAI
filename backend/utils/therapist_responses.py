"""Response generation — local generative model + rich template fallback.

Pipeline:
  1. Try local BlenderBot (facebook/blenderbot-400M-distill) — reads actual
     conversation content, no API key needed.
  2. Fall back to intent + DST-aware templates if model not loaded yet.
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
    # extract last user message for context
    last_user_msg = ""
    for m in reversed(history_snippet):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "")
            break

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

    return _build_template_response(intent, state, user_name, message_count, last_user_msg)


# ── Crisis resources ──────────────────────────────────────────────────────────

_CRISIS_RESOURCES = (
    "\n\n---\n**If you're in immediate danger, please reach out:**\n"
    "- **UK:** Samaritans 116 123 (free, 24/7)\n"
    "- **US:** 988 Suicide & Crisis Lifeline (call or text 988)\n"
    "- **Ireland:** Samaritans 116 123\n"
    "- **International:** findahelpline.com\n---"
)


# ── Templates ─────────────────────────────────────────────────────────────────

_OPENINGS: dict[str, list[str]] = {
    "crisis": [
        "I can hear how much pain you're in right now, and I'm really glad you told me.",
        "That sounds incredibly difficult. Thank you for trusting me with this.",
        "What you're sharing takes enormous courage. I'm here with you.",
        "I want you to know — you matter, and this moment matters. I'm listening.",
        "Thank you for telling me. You don't have to carry this alone.",
        "I'm really glad you came here instead of keeping this inside.",
    ],
    "anxiety": [
        "That sounds exhausting to carry — anxiety has a way of making everything feel urgent at once.",
        "That constant hum of worry takes a real toll. I hear you.",
        "Living with that level of tension in your body is genuinely draining.",
        "Anxiety can make even ordinary moments feel dangerous. That's really hard.",
        "It sounds like your nervous system has been in overdrive. That wears you out.",
        "The way you're describing it — that tight, relentless dread — I hear you.",
        "Racing thoughts at that intensity are physically exhausting, not just mentally.",
    ],
    "depression": [
        "What you're describing sounds really heavy.",
        "That kind of emptiness can be so draining — you're carrying a lot.",
        "Depression has a way of making everything feel distant and grey. I hear you.",
        "Just getting through the day when you feel like this is genuinely hard work.",
        "That numbness you're describing — it's one of the loneliest feelings.",
        "It takes something to even put that into words. I'm glad you did.",
        "Even reaching out when you feel this low is hard. I'm here.",
    ],
    "venting": [
        "I'm here — let it all out.",
        "That frustration makes total sense. Tell me what happened.",
        "You needed somewhere to put this, and I'm glad you came here.",
        "It sounds like today really tested you. I'm listening.",
        "Sometimes you just need to say it out loud without anyone fixing it.",
        "That sounds genuinely infuriating. I'm not going anywhere.",
        "Vent away. I'm here for all of it.",
    ],
    "seeking_advice": [
        "Let's think through this together.",
        "I want to make sure I understand the situation fully before we explore options.",
        "It sounds like you're genuinely wrestling with this. Let's look at it carefully.",
        "There's usually more than one path through something like this. Let's map it out.",
        "I'm glad you're asking — sometimes just talking it through helps clarify things.",
        "You don't have to figure this out alone. Let's take it step by step.",
        "Before I share any thoughts, I want to understand what matters most to you here.",
    ],
    "relationship": [
        "That dynamic sounds really complicated.",
        "Navigating people we care about — or used to care about — is never simple.",
        "Feeling unseen or disconnected from people close to you is genuinely painful.",
        "Relationships can hold so much of our sense of worth. That makes this feel a lot.",
        "It sounds like you're carrying the weight of something that should be shared.",
        "The loneliness of feeling invisible to someone you love is real.",
        "That kind of hurt from someone close cuts deep.",
    ],
    "work_stress": [
        "Work pressure has a way of seeping into every corner of life.",
        "That sounds like a lot of weight to carry — especially when there's no end in sight.",
        "When work becomes overwhelming it's hard to remember that you are more than your job.",
        "That level of pressure over time does real damage. I hear you.",
        "It sounds like you're running on empty and still being asked for more.",
        "Burnout sneaks up like this — you keep going until you can't.",
        "The dread before a difficult week at work can be exhausting even before it starts.",
    ],
    "self_esteem": [
        "I notice you're being quite hard on yourself there.",
        "It takes a lot to say those things out loud — and I want to sit with them with you.",
        "The way you're speaking about yourself — would you speak that way to someone you loved?",
        "That inner critic can be relentless. You don't have to believe everything it says.",
        "There's often a long history behind those beliefs about ourselves. Let's look at it.",
        "That kind of shame around who you are is painful to carry.",
        "The comparison you're making — it's worth questioning who set that standard.",
    ],
    "trauma": [
        "Thank you for sharing something so personal.",
        "It's okay to take this at whatever pace feels right for you.",
        "You don't owe anyone the whole story at once — including me.",
        "What you're carrying sounds like something you've held alone for a long time.",
        "It makes sense that this still affects you — trauma doesn't follow a schedule.",
        "The body holds these things even when the mind tries to move on.",
        "I'm honoured you felt safe enough to bring this here.",
    ],
    "gratitude": [
        "I'm really glad that's been helpful.",
        "That's a real shift — and it's yours, not mine.",
        "It means a lot to hear that. How are you feeling right now?",
        "Hearing that genuinely matters. I'm glad you came back.",
        "There's something powerful about being able to name that something helped.",
        "That warmth you're describing — hold onto it.",
        "I'm glad this space has been useful. That means something.",
    ],
    "progress": [
        "That's genuinely worth celebrating.",
        "Notice what you just said — that's real growth.",
        "Small steps like that are how lasting change happens.",
        "That took something. Don't minimise it.",
        "I love hearing this. What did it feel like in the moment?",
        "You did that. It wasn't luck — it was you choosing differently.",
        "That's the kind of thing that looks small from the outside and is huge from the inside.",
    ],
    "checking_in": [
        "Good to have you here.",
        "It's good to hear from you. How are things?",
        "Glad you came back. How have you been?",
        "Nice to see you. What's been on your mind?",
        "I've been here. How are you doing today?",
        "Welcome back. What's going on with you?",
    ],
    "general": [
        "I'm listening. Take your time.",
        "Tell me more.",
        "I'm here. Whatever's on your mind.",
        "Something brought you here today — what is it?",
        "You don't have to have the words perfectly. Just start.",
        "I've got time. What's sitting with you?",
    ],
}

_FOLLOWUPS: dict[str, list[str]] = {
    "crisis": [
        "Are you safe right now?",
        "What does this moment feel like for you?",
        "Have you told anyone else about these thoughts?",
        "How long have you been feeling this way?",
        "Is there anything that's been making it worse recently?",
        "What would you need right now to feel even slightly safer?",
    ],
    "anxiety": [
        "When did this level of worry start?",
        "Is there a specific thought that keeps coming back?",
        "What does your body do when the anxiety hits?",
        "What would it feel like to not have this running in the background?",
        "Has anything made it better, even briefly?",
        "What's the worst case scenario your mind keeps going to?",
        "Are there moments when the anxiety eases at all?",
    ],
    "depression": [
        "How long have you been carrying this?",
        "What does a typical day look like for you right now?",
        "When did you last feel even slightly okay?",
        "Is there anything — even something small — that brings a flicker of relief?",
        "Have you been able to tell anyone else how bad it's been?",
        "What does the hardest part of the day feel like?",
        "Are you looking after yourself at all — eating, sleeping?",
    ],
    "venting": [
        "What part of this feels most unfair?",
        "How long has this been building?",
        "What would you have needed instead?",
        "What do you wish you could say to them?",
        "Is this a pattern with this person, or was today unusual?",
        "How are you feeling now that you've said it out loud?",
        "What needs to change for this to stop happening?",
    ],
    "seeking_advice": [
        "What have you already tried?",
        "What does your gut say, underneath the noise?",
        "What outcome would feel right to you?",
        "What are you most afraid will happen if you do nothing?",
        "What's stopping you from the option you're already leaning toward?",
        "Who else is affected by this decision?",
        "What would you tell a friend in the same situation?",
    ],
    "relationship": [
        "How do you feel when you're around this person?",
        "What would feel like a fair outcome here?",
        "Has this always been the dynamic, or did something shift?",
        "What do you need from this person that you're not getting?",
        "If you knew things wouldn't change, what would you do?",
        "What does this relationship cost you?",
        "What would it feel like to be truly seen by them?",
    ],
    "work_stress": [
        "What's the part that's hardest to let go of at the end of the day?",
        "What would 'good enough' actually look like here?",
        "Is there anyone at work you feel you can be honest with?",
        "What's been keeping you going despite all this?",
        "When did it start feeling unmanageable?",
        "What would you tell someone you cared about in your position?",
        "What's one thing that would make the week even slightly more bearable?",
    ],
    "self_esteem": [
        "Where do you think that belief about yourself came from?",
        "What would you need to feel differently?",
        "Has there been a time in your life when you felt okay about yourself?",
        "What does that inner voice sound like — whose voice is it, really?",
        "What do people who know you well say about you?",
        "If the critical voice was wrong, what would be true instead?",
        "What would it mean for your life if you stopped believing that about yourself?",
    ],
    "trauma": [
        "Is this something you've been able to talk about before?",
        "How do you usually cope when it comes up?",
        "What has helped you get through moments when it felt overwhelming?",
        "Do you feel safe right now, talking about this?",
        "What part of it still feels hardest to sit with?",
        "Has anything shifted in how you see what happened, over time?",
        "Is there a part of you that has found a way to keep going despite this?",
    ],
    "gratitude": [
        "What do you think made the difference?",
        "How does it feel compared to where you were?",
        "What's changed in how you see things?",
        "What do you want to carry forward from this?",
        "Is there anything you want to work on next?",
        "What does it feel like to notice your own progress?",
    ],
    "progress": [
        "What helped you get there?",
        "What do you want to build on next?",
        "How did it feel in the moment when you realised you'd done it?",
        "What made this time different from before?",
        "What would you tell yourself a month ago?",
        "What's the next small step you can imagine taking?",
        "How do you want to hold onto this feeling?",
    ],
    "checking_in": [
        "How have things been since we last spoke?",
        "What's been on your mind?",
        "Anything in particular that made you come back today?",
        "How are you really doing?",
        "What's the first thing that comes to mind when I ask how you are?",
    ],
    "general": [
        "What's been sitting with you most heavily?",
        "How are you really doing?",
        "What made today different from other days?",
        "If you could put a word on what you're feeling, what would it be?",
        "What brought you here today?",
        "What would be most useful to talk about right now?",
    ],
}


def _extract_key_phrase(text: str, intent: str) -> str:
    """Pull a short phrase from the user message to anchor the response."""
    import re
    t = text.strip()
    # For work_stress pull the job/study detail
    if intent == "work_stress":
        m = re.search(r"(my boss|my manager|my job|my work|my exams?|my dissertation|my deadline|my workload)", t, re.I)
        if m:
            return m.group(1).lower()
    # For relationship pull the person mentioned
    if intent == "relationship":
        m = re.search(r"my (partner|boyfriend|girlfriend|husband|wife|mum|mom|dad|father|mother|brother|sister|friend|family)", t, re.I)
        if m:
            return m.group(0).lower()
    # For venting, pull the core frustration subject
    if intent == "venting":
        m = re.search(r"(my (flatmate|roommate|boss|colleague|friend|family|partner)|the (bus|train|commute|meeting|day))", t, re.I)
        if m:
            return m.group(1).lower()
    return ""


def _build_template_response(
    intent: str,
    state: DialogueState,
    user_name: str,
    message_count: int,
    last_user_msg: str = "",
) -> str:
    opening  = random.choice(_OPENINGS.get(intent, _OPENINGS["general"]))
    followup = random.choice(_FOLLOWUPS.get(intent, _FOLLOWUPS["general"]))

    name_prefix = f"{user_name}, " if user_name and message_count <= 6 else ""

    # Build context line from DST state + message content
    context_line = ""
    key_phrase = _extract_key_phrase(last_user_msg, intent) if last_user_msg else ""

    if key_phrase:
        context_line = f"\n\nWhat you said about {key_phrase} — that stood out to me."
    elif state.mentioned_people:
        person = state.mentioned_people[-1]
        context_line = f"\n\nIt sounds like {person} is connected to this."
    elif state.active_topics and intent in ("work_stress", "anxiety", "depression"):
        topic = state.active_topics[-1]
        context_line = f"\n\nThe {topic} piece seems to be central to what you're carrying."

    reply = f"{name_prefix}{opening}{context_line}\n\n{followup}"

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
    theme_line = f" Last time we touched on {last_theme}." if last_theme else ""

    greetings = [
        f"Welcome back{name}.{theme_line}\n\nI remember everything — you never have to start from scratch. How are you doing today?",
        f"Good to see you back{name}.{theme_line}\n\nWe've talked {message_count} times now. How have things been?",
        f"Hey{name}, I'm glad you came back.{theme_line}\n\nWhat's been going on since we last spoke?",
        f"Welcome back{name}. This is your space — always will be.{theme_line}\n\nHow are you feeling today?",
    ]
    return random.choice(greetings)
