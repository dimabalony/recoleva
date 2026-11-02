"""MemoryEntityLink model — connects memory traces to entities."""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, utc_now


class MemoryEntityLink(Base):
    """Link between a memory trace and a memory entity."""

    __tablename__ = "memory_entity_links"

    id: Mapped[_uuid.UUID] = mapped_column(
        default=_uuid.uuid4, server_default=text("gen_random_uuid()"), primary_key=True
    )
    memory_id: Mapped[_uuid.UUID] = mapped_column(ForeignKey("memory_traces.id"), nullable=False)
    entity_id: Mapped[_uuid.UUID] = mapped_column(ForeignKey("memory_entities.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, server_default=text("NOW()"))

    __table_args__ = (
        UniqueConstraint("memory_id", "entity_id", "role"),
        Index("idx_mel_memory", "memory_id"),
        Index("idx_mel_entity", "entity_id"),
    )
