"""SQLite persistent memory — one eternal session per user, no resets."""
import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import Column, Integer, String, Text, Float, DateTime, select, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

DB_PATH = Path(__file__).parent / "therapist.db"
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(64), nullable=False, index=True)
    role = Column(String(16), nullable=False)       # "user" | "therapist"
    content = Column(Text, nullable=False)
    intent = Column(String(64))                     # detected intent label
    intent_score = Column(Float)
    dialogue_state = Column(Text)                   # JSON blob
    timestamp = Column(DateTime, default=datetime.utcnow)


class UserProfile(Base):
    """Persistent longitudinal state — what the therapist remembers about you."""
    __tablename__ = "user_profiles"

    user_id = Column(String(64), primary_key=True)
    display_name = Column(String(128), default="")
    recurring_themes = Column(Text, default="[]")   # JSON list[str]
    disclosed_topics = Column(Text, default="[]")   # JSON list[str]
    mood_history = Column(Text, default="[]")       # JSON list[{ts, score}]
    last_seen = Column(DateTime, default=datetime.utcnow)
    session_count = Column(Integer, default=1)
    summary = Column(Text, default="")             # running narrative summary


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with SessionLocal() as session:
        yield session


# ── helper queries ──────────────────────────────────────────────────────────

async def get_or_create_profile(db: AsyncSession, user_id: str) -> UserProfile:
    result = await db.execute(select(UserProfile).where(UserProfile.user_id == user_id))
    profile = result.scalar_one_or_none()
    if not profile:
        profile = UserProfile(user_id=user_id)
        db.add(profile)
        await db.commit()
        await db.refresh(profile)
    return profile


async def save_message(
    db: AsyncSession,
    user_id: str,
    role: str,
    content: str,
    intent: str | None = None,
    intent_score: float | None = None,
    dialogue_state: dict | None = None,
) -> Message:
    msg = Message(
        user_id=user_id,
        role=role,
        content=content,
        intent=intent,
        intent_score=intent_score,
        dialogue_state=json.dumps(dialogue_state) if dialogue_state else None,
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg


async def load_history(db: AsyncSession, user_id: str, limit: int = 40) -> list[Message]:
    result = await db.execute(
        select(Message)
        .where(Message.user_id == user_id)
        .order_by(Message.timestamp.asc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def count_messages(db: AsyncSession, user_id: str) -> int:
    result = await db.execute(
        select(func.count()).where(Message.user_id == user_id, Message.role == "user")
    )
    return result.scalar_one()


async def update_profile(
    db: AsyncSession,
    user_id: str,
    *,
    new_theme: str | None = None,
    new_topic: str | None = None,
    mood_score: float | None = None,
    summary: str | None = None,
    display_name: str | None = None,
):
    profile = await get_or_create_profile(db, user_id)
    if new_theme:
        themes = json.loads(profile.recurring_themes)
        if new_theme not in themes:
            themes.append(new_theme)
            profile.recurring_themes = json.dumps(themes[-20:])
    if new_topic:
        topics = json.loads(profile.disclosed_topics)
        if new_topic not in topics:
            topics.append(new_topic)
            profile.disclosed_topics = json.dumps(topics[-50:])
    if mood_score is not None:
        history = json.loads(profile.mood_history)
        history.append({"ts": datetime.utcnow().isoformat(), "score": mood_score})
        profile.mood_history = json.dumps(history[-200:])
    if summary:
        profile.summary = summary
    if display_name:
        profile.display_name = display_name
    profile.last_seen = datetime.utcnow()
    await db.commit()
    return profile
