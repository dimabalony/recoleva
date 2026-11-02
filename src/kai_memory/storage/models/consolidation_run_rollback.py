"""ConsolidationRunRollback model — audit trail for rollback_run / rollback_chain.

M2 Wave 1 ships the table; Wave 3 ships the operations that write rows.
"""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, utc_now


class ConsolidationRunRollback(Base):
    """One row per ``rollback_run(run_id, ...)`` invocation.

    ``inverse_actions`` carries the recorded inverse of the run's
    ``changes_made`` list, sufficient for an un-rollback (rare).
    """

    __tablename__ = "consolidation_run_rollbacks"

    id: Mapped[_uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        default=_uuid.uuid4,
        server_default=text("gen_random_uuid()"),
        primary_key=True,
    )
    run_id: Mapped[_uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("consolidation_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    rolled_back_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, server_default=text("now()"), nullable=False
    )
    rolled_back_by: Mapped[str] = mapped_column(String(100), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    inverse_actions: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, server_default=text("'[]'::jsonb"), nullable=False
    )

    __table_args__ = (
        Index("idx_run_rollbacks_run", "run_id"),
        Index("idx_run_rollbacks_when", "rolled_back_at"),
    )
