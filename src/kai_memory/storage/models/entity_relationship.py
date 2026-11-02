"""EntityRelationship model — temporal, typed relationships between entities."""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, utc_now


class EntityRelationship(Base):
    """Temporal relationship between two memory entities."""

    __tablename__ = "entity_relationships"

    id: Mapped[_uuid.UUID] = mapped_column(
        default=_uuid.uuid4, server_default=text("gen_random_uuid()"), primary_key=True
    )
    from_entity_id: Mapped[_uuid.UUID] = mapped_column(ForeignKey("memory_entities.id"), nullable=False)
    to_entity_id: Mapped[_uuid.UUID] = mapped_column(ForeignKey("memory_entities.id"), nullable=False)
    relationship_type: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    source_memory_id: Mapped[_uuid.UUID | None] = mapped_column(ForeignKey("memory_traces.id"), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, server_default=text("NOW()"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=text("NOW()"),
        onupdate=utc_now,
    )

    __table_args__ = (
        Index(
            "idx_er_from",
            "from_entity_id",
            postgresql_where=text("valid_until IS NULL"),
        ),
        Index(
            "idx_er_to",
            "to_entity_id",
            postgresql_where=text("valid_until IS NULL"),
        ),
        Index(
            "idx_er_from_type",
            "from_entity_id",
            "relationship_type",
            postgresql_where=text("valid_until IS NULL"),
        ),
        Index(
            "idx_er_to_type",
            "to_entity_id",
            "relationship_type",
            postgresql_where=text("valid_until IS NULL"),
        ),
    )
