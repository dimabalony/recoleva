"""CognitiveConfig model — tunable magic numbers for cognitive memory."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column, validates

from .base import Base, utc_now


class CognitiveConfig(Base):
    """Single source of truth for numeric tuning knobs."""

    __tablename__ = "cognitive_config"

    key: Mapped[str] = mapped_column(String(50), primary_key=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    category: Mapped[str] = mapped_column(String(30), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=text("NOW()"),
        onupdate=utc_now,
    )

    @validates("value")
    def _coerce_value(self, _key: str, value: object) -> float:
        """Coerce ``value`` to ``float`` on every assignment — the durable type
        boundary for ALL callers (operator flip scripts, SSM sync, ORM writes).

        A string like ``'1.0'`` written into this ``Float`` column otherwise binds
        a Python ``str`` to a real parameter (``asyncpg.DataError: must be real
        number, not str``) and — worse — leaves every session that loads the row
        perpetually dirty, so any later flush (even a read-only script's autoflush)
        crashes. Coercing here makes ``obj.value = '1.0'`` store ``1.0`` cleanly.
        ``None``/non-numeric raise a clear error rather than corrupting the row.
        """
        if value is None:
            raise ValueError("cognitive_config.value must not be None")
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"cognitive_config.value must be numeric, got {value!r}") from exc
