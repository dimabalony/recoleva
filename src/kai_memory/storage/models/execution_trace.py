"""ExecutionTrace model — Store A of procedural memory (design §3).

The raw, append-only execution log: one row per completed chat turn that used
tools, recording the *kind* of ask + the tool sequence as-happened + raw
(unjudged) success signals. **COLD by construction** — its own table, so it is
invisible to everything that reads ``memory_traces`` (L1 priming, L2 extraction,
L3 consolidation). Queried only by the offline recurrence/reliability eval and
(later) the L3 utility pass. NEVER edited after insert.

Privacy: ``task_signature`` is the secret-scrubbed, normalized user ask (readable
so the eval can SEE clusters — acceptable in this single-tenant deployment; a
multi-tenant deployment would drop to embedding-only, see the recorder). ``steps``
records tool NAMES + argument-key SHAPES, never argument values or outputs.
"""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, DateTime, Index, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, utc_now


class ExecutionTrace(Base):
    """One recorded tool-using turn — the raw evidence procedural memory needs."""

    __tablename__ = "execution_traces"

    id: Mapped[_uuid.UUID] = mapped_column(
        default=_uuid.uuid4, server_default=text("gen_random_uuid()"), primary_key=True
    )
    # The *kind* of ask — scrubbed + normalized trigger text. Clustering into
    # "same ask" happens at eval time by embedding cosine (θ pinned pre-eval).
    task_signature: Mapped[str] = mapped_column(Text, nullable=False)
    # Filled in BATCH by the eval / reflection pass — never embedded per-turn on
    # the hot path (cost). NULL until that batch runs.
    task_embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)
    # The tool sequence as-happened: [{"tool": name, "arg_keys": [...]}] — names +
    # argument-key shapes only, NEVER values/outputs; side-effecting-tool payloads
    # excluded upstream by the recorder.
    steps: Mapped[list] = mapped_column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))
    # success | failure | unknown. 'unknown' at write; the deferred reflection pass
    # upgrades it from the multi-turn signals (no redo, positive follow-up).
    outcome: Mapped[str] = mapped_column(String(20), nullable=False, default="unknown")
    how_measured: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Raw, UNJUDGED signals captured at write (tool_error_count, tool_call_count,
    # stop_reason, tokens, had_text_response). No judgement applied here.
    success_signals: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    cost_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Where it happened (for eval audit / dedup), not a hard FK — conversations
    # are not a modelled table here.
    conversation_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Multi-person scoping: a procedure recorded for person A must never be served
    # in person B's context. NULL for headless/self venues.
    source_user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, server_default=text("NOW()"), nullable=False
    )

    # Mirror migration 093 so the integration fixture (Base.metadata.create_all) and
    # Alembic autogenerate stay in lockstep — no phantom schema drift.
    __table_args__ = (
        Index("idx_exec_traces_created", "created_at"),
        # Mirror migration 111 for canonical execution replay/observation lookups.
        Index(
            "idx_exec_traces_assistant_message",
            text("(success_signals->>'assistant_message_id')"),
            postgresql_where=text("success_signals->>'assistant_message_id' IS NOT NULL"),
        ),
        Index(
            "idx_exec_traces_unembedded",
            "created_at",
            postgresql_where=text("task_embedding IS NULL"),
        ),
        # Was the serving lookup for the Phase-A exact-signature advisory, which
        # DB-487 deleted (serve-all has no signature match). Retained for now — the
        # index is unread but harmless; dropping it is a separate DDL change.
        Index("idx_exec_traces_serving", "task_signature", "source_user_id"),
    )
