"""Local generative response model — no API key needed.

Uses facebook/blenderbot-400M-distill, an empathetic conversational model
that runs on CPU. Falls back to rule-based templates if transformers is not
installed or the model fails to load.

Intent + DST context is injected as a persona prefix so the model stays
in a therapeutic register.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

_pipeline = None
_load_attempted = False

MODEL_ID = "facebook/blenderbot-400M-distill"
CACHE_DIR = Path(__file__).parent / "../../training/checkpoints/blenderbot"


def _load():
    global _pipeline, _load_attempted
    if _load_attempted:
        return _pipeline
    _load_attempted = True
    try:
        from transformers import pipeline
        print(f"[generator] Loading {MODEL_ID} (first run downloads ~800 MB)…")
        _pipeline = pipeline(
            "text2text-generation",
            model=MODEL_ID,
            cache_dir=str(CACHE_DIR),
        )
        print("[generator] Model ready.")
    except Exception as e:
        print(f"[generator] Could not load model: {e}. Using templates.")
        _pipeline = None
    return _pipeline


def generate(
    history: list[dict],
    intent: str,
    mood: str,
    topics: list[str],
    people: list[str],
    max_new_tokens: int = 120,
) -> Optional[str]:
    """Return a generated reply or None (caller falls back to templates)."""
    pipe = _load()
    if pipe is None:
        return None

    # BlenderBot takes a flat string of prior turns separated by </s>
    # We inject a soft persona line at the start
    persona = (
        "I am an empathetic therapist. "
        f"The person seems to be feeling {mood}. "
    )
    if topics:
        persona += f"Topics: {', '.join(topics)}. "

    turns = []
    for msg in history[-8:]:           # last 8 turns is plenty
        if msg["role"] == "user":
            turns.append(msg["content"])
        else:
            turns.append(msg["content"])

    # BlenderBot's expected format: utterances joined by </s>
    context = "  ".join(turns)
    input_text = persona + context

    try:
        result = pipe(
            input_text,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.85,
            top_p=0.92,
        )
        reply = result[0]["generated_text"].strip()
        # Remove any accidental echo of the input
        if reply.lower().startswith(persona.lower()[:30]):
            reply = reply[len(persona):].strip()
        return reply if len(reply) > 10 else None
    except Exception:
        return None
