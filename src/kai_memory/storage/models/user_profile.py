"""UserProfile model - personal context for AI personalization."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, utc_now


class UserProfile(Base):
    """User profile for personalized AI context.

    Stores identity, goals, principles, current context, constraints,
    and relationships. Used by AI to make context-aware recommendations.
    """

    __tablename__ = "user_profile"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Multi-user support (default_user for single-user setup)
    user_id: Mapped[str] = mapped_column(String, nullable=False, default="default_user")

    # Classification
    category: Mapped[str] = mapped_column(String, nullable=False)
    key: Mapped[str] = mapped_column(String, nullable=False)

    # Content (flexible JSON structure)
    value: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    # Priority 1-10 (10 = highest importance)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=5)

    # Optional notes/context
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, server_default=text("NOW()"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=text("NOW()"),
        onupdate=utc_now,
    )

    __table_args__ = (
        Index("idx_user_profile_category", "user_id", "category"),
        Index("idx_user_profile_priority", "user_id", "priority"),
    )
