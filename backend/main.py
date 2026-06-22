"""FastAPI + WebSocket backend — single persistent session per user."""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from memory.database import (
    init_db, get_db, get_or_create_profile, save_message,
    load_history, count_messages, update_profile,
)
from models.intent_classifier import get_classifier
from models.dialogue_state_tracker import get_tracker, DialogueState
from utils.therapist_responses import (
    build_response, first_greeting, returning_greeting,
)


def _build_summary(profile: "UserProfile", state: "DialogueState") -> str:
    """Build a short narrative summary from profile + current DST state."""
    import json as _json
    parts: list[str] = []
    themes = _json.loads(profile.recurring_themes)
    if themes:
        parts.append(f"recurring themes: {', '.join(themes[-5:])}")
    topics = _json.loads(profile.disclosed_topics)
    if topics:
        parts.append(f"discussed topics: {', '.join(topics[-5:])}")
    if state.mentioned_people:
        parts.append(f"mentioned people: {', '.join(state.mentioned_people[-4:])}")
    mood_hist = _json.loads(profile.mood_history)
    if mood_hist:
        recent_score = mood_hist[-1]["score"]
        mood_word = "low" if recent_score < -0.3 else ("positive" if recent_score > 0.3 else "neutral")
        parts.append(f"recent mood: {mood_word}")
    return "; ".join(parts) if parts else ""

# ── per-connection in-memory state (lost on restart, not persisted) ──────────
# Persistent state lives in SQLite via the DB helpers above.
_active_states: dict[str, DialogueState] = {}
_greeted_this_session: set[str] = set()  # prevents repeat greeting on reconnect

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # warm up models on startup
    get_classifier()
    get_tracker()
    yield


app = FastAPI(title="TherapistAI", lifespan=lifespan)

app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR / "static")),
    name="static",
)


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = FRONTEND_DIR / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    await websocket.accept()

    classifier = get_classifier()
    tracker    = get_tracker()

    # restore or init state
    if user_id not in _active_states:
        _active_states[user_id] = DialogueState()

    state   = _active_states[user_id]
    profile = await get_or_create_profile(db, user_id)
    count   = await count_messages(db, user_id)
    history = await load_history(db, user_id, limit=40)

    # send history replay so the UI repopulates on reconnect
    history_payload = [
        {
            "role":      m.role,
            "content":   m.content,
            "intent":    m.intent,
            "timestamp": m.timestamp.isoformat(),
        }
        for m in history
    ]
    await websocket.send_json({"type": "history", "messages": history_payload})

    # only greet on a genuinely fresh session (no messages at all yet)
    is_fresh = len(history) == 0
    if is_fresh:
        greeting = first_greeting(profile.display_name)
        await websocket.send_json({"type": "message", "role": "therapist", "content": greeting})
        await save_message(db, user_id, "therapist", greeting)
        _greeted_this_session.add(user_id)
    elif count > 0 and user_id not in _greeted_this_session:
        # returning user — greet once per server session, don't persist to DB
        import json as _json
        themes = _json.loads(profile.recurring_themes)
        last_theme = themes[-1] if themes else ""
        greeting = returning_greeting(profile.display_name, count, last_theme)
        await websocket.send_json({"type": "message", "role": "therapist", "content": greeting})
        _greeted_this_session.add(user_id)

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            # ── name registration ────────────────────────────────────────────
            if data.get("type") == "set_name":
                name = data.get("name", "").strip()[:64]
                await update_profile(db, user_id, display_name=name)
                profile = await get_or_create_profile(db, user_id)
                await websocket.send_json({"type": "ack_name", "name": name})
                continue

            user_text = data.get("content", "").strip()
            if not user_text:
                continue

            # ── intent detection ─────────────────────────────────────────────
            intent_result = classifier.classify(user_text)

            # ── dialogue state update ────────────────────────────────────────
            state = tracker.update(state, user_text, intent_result.label)
            _active_states[user_id] = state

            # ── persist user message ─────────────────────────────────────────
            await save_message(
                db, user_id, "user", user_text,
                intent=intent_result.label,
                intent_score=intent_result.score,
                dialogue_state=state.to_dict(),
            )

            # ── echo intent metadata to UI (for the debug panel) ────────────
            await websocket.send_json({
                "type":   "intent",
                "label":  intent_result.label,
                "score":  round(intent_result.score, 3),
                "state":  state.to_dict(),
            })

            # ── update profile ───────────────────────────────────────────────
            for topic in state.active_topics[-3:]:
                await update_profile(db, user_id, new_topic=topic)
            if state.active_topics:
                await update_profile(db, user_id, new_theme=state.active_topics[-1])
            await update_profile(db, user_id, mood_score=state.mood_score)

            # rebuild narrative summary every 5 user messages
            if count % 5 == 0:
                summary_text = _build_summary(profile, state)
                if summary_text:
                    await update_profile(db, user_id, summary=summary_text)

            profile = await get_or_create_profile(db, user_id)
            count   = await count_messages(db, user_id)
            recent  = await load_history(db, user_id, limit=10)
            history_snippet = [
                {"role": m.role, "content": m.content} for m in recent
            ]

            # ── generate therapist reply ─────────────────────────────────────
            reply = build_response(
                intent       = intent_result.label,
                state        = state,
                profile_summary = profile.summary,
                user_name    = profile.display_name,
                message_count = count,
                history_snippet = history_snippet,
            )

            await save_message(db, user_id, "therapist", reply,
                               intent=intent_result.label,
                               dialogue_state=state.to_dict())

            await websocket.send_json({
                "type":    "message",
                "role":    "therapist",
                "content": reply,
            })

    except WebSocketDisconnect:
        pass
    finally:
        # persist final state
        _active_states[user_id] = state
