"""MemoryTrace model — automatic memory (facts, episodes, impressions)."""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, utc_now


class MemoryTrace(Base):
    """Automatic memory — facts, episodes, impressions encoded from experience."""

    __tablename__ = "memory_traces"

    id: Mapped[_uuid.UUID] = mapped_column(
        default=_uuid.uuid4, server_default=text("gen_random_uuid()"), primary_key=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    memory_type: Mapped[str] = mapped_column(String(30), nullable=False)
    salience: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    initial_salience: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    significance: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=3)

    source_type: Mapped[str] = mapped_column(String(20), nullable=False)
    source_chat_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    source_topic_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    emotion_valence: Mapped[float | None] = mapped_column(Float, nullable=True)
    emotion_arousal: Mapped[float | None] = mapped_column(Float, nullable=True)

    access_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    access_timestamps: Mapped[list] = mapped_column(JSONB, default=list, server_default=text("'[]'::jsonb"))
    last_accessed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    invalidated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    invalidated_by: Mapped[_uuid.UUID | None] = mapped_column(ForeignKey("memory_traces.id"), nullable=True)
    invalidation_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    consolidated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)

    # v2 additive columns (migration 061). Both Python and server defaults
    # are required because tests construct MemoryTrace(...) directly; server
    # defaults don't fire when SQLAlchemy builds a row client-side.
    representations: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    encoding_signals: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )

    # Sub-design C INC 1 (migration 087) — typed_metadata + provenance.
    # typed_metadata is the cold, producer-read extraction blob (one versioned
    # JSONB column, reserved keys); the four provenance scalars are hot-path /
    # group-by columns (R1). Both Python and server defaults are set so tests
    # that construct MemoryTrace(...) client-side still get the empty default.
    # source_user_id lands NULL here and is populated by E (INC 8), its only
    # consumer. See backend-docs/ai-workforce/plans/design/2026-06-04-subdesign-C-architecture-and-plan.md §1.1.
    typed_metadata: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    source_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    source_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    invalidation_source: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, server_default=text("NOW()"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=text("NOW()"),
        onupdate=utc_now,
    )

    __table_args__ = (
        Index(
            "ix_memory_traces_durability_marker",
            text("(typed_metadata->>'durability')"),
            postgresql_where=text("typed_metadata->>'durability' IN ('durable', 'superseded')"),
        ),
        Index("idx_mt_salience", "salience", postgresql_where=text("invalidated_at IS NULL")),
        # DB-621 — referencing side of the ``invalidated_by`` self-FK. Without it
        # each trace the decay prune deletes made the referential check scan the
        # whole table. Mirrors migration 106.
        Index(
            "idx_mt_invalidated_by",
            "invalidated_by",
            postgresql_where=text("invalidated_by IS NOT NULL"),
        ),
        Index("idx_mt_type", "memory_type", postgresql_where=text("invalidated_at IS NULL")),
        Index("idx_mt_created", "created_at"),
        Index("idx_mt_source_chat", "source_chat_id", "source_topic_id"),
        Index(
            "idx_mt_consolidated",
            "consolidated_at",
            postgresql_where=text("invalidated_at IS NULL"),
        ),
        # Mirrors migration 039 DDL. Declared on the model (not only in the
        # migration) so `create_all`-based integration DBs carry the real HNSW
        # index — tests/integration/test_pgvector_index_plan.py EXPLAINs
        # against it to guard index eligibility of the search ORDER BY.
        Index(
            "idx_mt_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
