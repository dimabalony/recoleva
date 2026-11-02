# app/models/agent_memory.py
"""Agent memory model — persistent knowledge the AI reads and writes."""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, text

from .base import Base, DateTime, JSONB, Mapped, String, Text, mapped_column, utc_now


class AgentMemory(Base):
    """Persistent memory entries the AI can search, read, and write.

    Always-loaded categories (injected into system prompt every turn):
    - 'instruction': Operating instructions (like AGENTS.md in OpenClaw)
    - 'persona': Persona/tone (like SOUL.md)
    - 'identity': Agent identity (like IDENTITY.md)
    - 'tool_notes': Local setup notes (like TOOLS.md)

    On-demand categories (retrieved via memory_search/memory_get):
    - 'longterm': Curated wisdom, lessons, decisions (like MEMORY.md)
    - 'daily': Daily observations, key: date e.g. '2026-03-03' (like memory/YYYY-MM-DD.md)
    - 'heartbeat': Heartbeat state (last_checks, observations, etc.)
    - Any other category the agent creates (project, work, health, etc.)
    """

    __tablename__ = "agent_memory"

    id: Mapped[_uuid.UUID] = mapped_column(
        default=_uuid.uuid4, server_default=text("gen_random_uuid()"), primary_key=True
    )
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    key: Mapped[str] = mapped_column(String(200), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[int] = mapped_column(default=5)
    source: Mapped[str | None] = mapped_column(String(100), nullable=True)
    memory_metadata: Mapped[dict] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"))
    # Embedding vector for semantic similarity search
    # Uses text-embedding-3-small = 1536 dimensions
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, server_default=text("NOW()"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, server_default=text("NOW()"), onupdate=utc_now
    )

    __table_args__ = (
        Index("idx_agent_memory_category", "category"),
        Index("idx_agent_memory_key", "category", "key", unique=True),
        Index("idx_agent_memory_priority", "priority"),
    )
