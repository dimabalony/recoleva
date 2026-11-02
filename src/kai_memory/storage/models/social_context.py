"""SocialContext and SocialExperience models — social awareness layer."""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, utc_now


class SocialContext(Base):
    """Social context for a chat/topic — norms, participants, type."""

    __tablename__ = "social_contexts"

    id: Mapped[_uuid.UUID] = mapped_column(
        default=_uuid.uuid4, server_default=text("gen_random_uuid()"), primary_key=True
    )
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    topic_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    context_type: Mapped[str] = mapped_column(String(20), nullable=False)
    context_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    participants: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    social_norms: Mapped[str | None] = mapped_column(Text, nullable=True)

    last_active_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, server_default=text("NOW()"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=text("NOW()"),
        onupdate=utc_now,
    )

    __table_args__ = (
        UniqueConstraint("chat_id", "topic_id"),
        Index("idx_social_ctx_private", "chat_id", unique=True, postgresql_where=text("topic_id IS NULL")),
    )


class SocialExperience(Base):
    """A social experience that taught the agent a norm."""

    __tablename__ = "social_experiences"

    id: Mapped[_uuid.UUID] = mapped_column(
        default=_uuid.uuid4, server_default=text("gen_random_uuid()"), primary_key=True
    )
    social_context_id: Mapped[_uuid.UUID] = mapped_column(ForeignKey("social_contexts.id"), nullable=False)
    experience: Mapped[str] = mapped_column(Text, nullable=False)
    learned_norm: Mapped[str | None] = mapped_column(Text, nullable=True)
    signal_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    salience: Mapped[float] = mapped_column(Float, default=0.5)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, server_default=text("NOW()"))

    __table_args__ = (Index("idx_socexp_context", "social_context_id", "salience"),)
