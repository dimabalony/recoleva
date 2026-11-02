"""ConfusionSet model — group of entities the system has confused or might confuse."""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime

from sqlalchemy import DateTime, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, utc_now


class ConfusionSet(Base):
    """Group of entities flagged for contrastive disambiguation.

    Populated by Sub-design C; M1 ships only the table + model so producers
    can target it without a follow-up migration.
    """

    __tablename__ = "confusion_sets"

    id: Mapped[_uuid.UUID] = mapped_column(
        default=_uuid.uuid4,
        server_default=text("gen_random_uuid()"),
        primary_key=True,
    )
    name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    entity_type: Mapped[str] = mapped_column(String(30), nullable=False)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    confusion_source: Mapped[str | None] = mapped_column(String(30), nullable=True)
    last_affirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, server_default=text("NOW()"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=text("NOW()"),
        onupdate=utc_now,
    )
