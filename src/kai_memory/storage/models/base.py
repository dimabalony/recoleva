"""Base class and common imports for models."""

from __future__ import annotations

import uuid
from datetime import datetime, UTC
from typing import Any

from sqlalchemy import DateTime, Index, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative registry containing only memory-owned tables."""

    pass


def utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime.

    Persistence defaults intentionally retain real UTC time, independent of the
    host semantic clock. This preserves the existing ORM write contract.
    """
    return datetime.now(UTC)


# Re-export for convenience
__all__ = [
    "JSONB",
    "UUID",
    "Any",
    "Base",
    "DateTime",
    "Index",
    "Mapped",
    "String",
    "Text",
    "datetime",
    "mapped_column",
    "text",
    "utc_now",
    "uuid",
]
