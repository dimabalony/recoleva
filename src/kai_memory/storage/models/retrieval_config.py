"""RetrievalConfig model — adaptive retrieval weights (single-row)."""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, utc_now


class RetrievalConfig(Base):
    """Single-row config for adaptive retrieval weight tuning."""

    __tablename__ = "retrieval_config"

    id: Mapped[_uuid.UUID] = mapped_column(
        default=_uuid.uuid4, server_default=text("gen_random_uuid()"), primary_key=True
    )
    weight_semantic: Mapped[float] = mapped_column(Float, nullable=False, default=0.35)
    weight_graph: Mapped[float] = mapped_column(Float, nullable=False, default=0.25)
    weight_recency: Mapped[float] = mapped_column(Float, nullable=False, default=0.15)
    weight_access: Mapped[float] = mapped_column(Float, nullable=False, default=0.15)
    weight_arousal: Mapped[float] = mapped_column(Float, nullable=False, default=0.10)
    # A2: 6th L1-priming component (per-trace decayed salience). DEFAULT 0.0 ⇒ inert
    # (salience * 0.0 == 0), so priming is unchanged until the operator tunes it up.
    weight_salience: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default=text("0.0"))
    # #273: 7th component. Intentions (Path 4) previously forged semantic=1.0 +
    # graph=1.0 on themselves (0.60 of unearned score) and crowded gold out of
    # the top-8. They now score on this component instead. STATIC: no code path
    # writes this column at runtime — the nightly `_adjust_retrieval_weights`
    # loop that drove the #241 graph runaway is deleted, and this weight is
    # versioned via migration only (097), never self-tuned.
    weight_intention: Mapped[float] = mapped_column(Float, nullable=False, default=0.15, server_default=text("0.15"))
    adjustment_history: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, server_default=text("'[]'::jsonb")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=text("NOW()"),
        onupdate=utc_now,
    )
