"""ConsolidationRun model — tracks nightly consolidation passes."""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, Integer, SmallInteger, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, utc_now


class ConsolidationRun(Base):
    """Record of a consolidation pass (Level 2 or Level 3)."""

    __tablename__ = "consolidation_runs"

    id: Mapped[_uuid.UUID] = mapped_column(
        default=_uuid.uuid4, server_default=text("gen_random_uuid()"), primary_key=True
    )
    level: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, server_default=text("NOW()"))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    traces_processed: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    run_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"))
    # M2 Wave 1 — ArtifactPipeline producers append per-artifact ChangesMadeEntry
    # rows here so preview_run / rollback_run (Wave 3) have the full audit trail.
    # Wrapped in MutableList so in-place ``run.changes_made.append(...)``
    # writes are tracked by SQLAlchemy's ORM unit-of-work.
    changes_made: Mapped[list[dict[str, Any]]] = mapped_column(
        MutableList.as_mutable(JSONB),
        default=list,
        server_default=text("'[]'::jsonb"),
        nullable=False,
    )

    __table_args__ = (Index("idx_consolrun_level", "level", "started_at"),)
