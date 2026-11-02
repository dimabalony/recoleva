"""TraceInhibition model — competitive inhibition between traces."""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, utc_now


class TraceInhibition(Base):
    """Inhibition link from `winner_trace_id` over `loser_trace_id`.

    Produced by Sub-design C feedback handling. M1 ships the schema only.
    """

    __tablename__ = "trace_inhibitions"

    id: Mapped[_uuid.UUID] = mapped_column(
        default=_uuid.uuid4,
        server_default=text("gen_random_uuid()"),
        primary_key=True,
    )
    winner_trace_id: Mapped[_uuid.UUID] = mapped_column(
        ForeignKey("memory_traces.id", ondelete="CASCADE"), nullable=False
    )
    loser_trace_id: Mapped[_uuid.UUID] = mapped_column(
        ForeignKey("memory_traces.id", ondelete="CASCADE"), nullable=False
    )
    strength: Mapped[float] = mapped_column(Float, nullable=False, default=0.1, server_default="0.1")
    suppressed_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, server_default=text("NOW()"))

    __table_args__ = (
        UniqueConstraint("winner_trace_id", "loser_trace_id", name="uq_inhibitions_pair"),
        CheckConstraint(
            "winner_trace_id <> loser_trace_id",
            name="ck_inhibitions_no_self_loop",
        ),
        CheckConstraint(
            "strength > 0.0 AND strength <= 0.5",
            name="ck_inhibitions_strength_bound",
        ),
        Index("idx_inhibitions_loser", "loser_trace_id", "winner_trace_id"),
    )
