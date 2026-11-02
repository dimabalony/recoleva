"""EntityMergeCandidate model."""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, text
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, utc_now


class EntityMergeCandidate(Base):
    """Pair of entities flagged as potential duplicates."""

    __tablename__ = "entity_merge_candidates"

    id: Mapped[_uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        default=_uuid.uuid4,
        server_default=text("gen_random_uuid()"),
        primary_key=True,
    )
    entity_a_id: Mapped[_uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_entities.id"), nullable=False
    )
    entity_b_id: Mapped[_uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_entities.id"), nullable=False
    )
    source_relationship_id: Mapped[_uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("entity_relationships.id"), nullable=True
    )
    # Provenance, NOT ownership (DB-579). A candidate outlives the run that proposed it:
    # a PROPOSED row waits indefinitely for a human review click, and once MERGED or
    # REJECTED the row is the permanent record that stops ``_candidate_exists`` from
    # re-proposing a pair the human already judged. The nightly ``_cleanup_old_runs``
    # DELETE therefore has to null the pointer rather than be blocked by it (the
    # original NO ACTION, which failed the step every run) or take the candidate with
    # it (CASCADE, which would resurrect rejected pairs and silently empty the review
    # queue). Same shape as ``memory_artifacts.produced_by_run_id``; contrast
    # ``consolidation_run_rollbacks.run_id``, which IS owned by its run and cascades.
    source_consolidation_run_id: Mapped[_uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("consolidation_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    evidence_trace_ids: Mapped[list[_uuid.UUID] | None] = mapped_column(ARRAY(UUID(as_uuid=True)), nullable=True)

    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5, server_default="0.5")
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", server_default="pending")
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=text("NOW()"),
        nullable=False,
    )
