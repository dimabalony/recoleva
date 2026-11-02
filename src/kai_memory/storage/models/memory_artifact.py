"""MemoryArtifact model — typed consolidation artifacts (EntitySchema, ContrastiveMap, …)."""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, utc_now


class MemoryArtifact(Base):
    """One row per produced artifact.

    Per-type active uniqueness is enforced by a partial unique index on
    (artifact_type, subject_entity_id, COALESCE(dimension, '')) — see
    migration 061.
    """

    __tablename__ = "memory_artifacts"

    id: Mapped[_uuid.UUID] = mapped_column(
        default=_uuid.uuid4,
        server_default=text("gen_random_uuid()"),
        primary_key=True,
    )
    artifact_type: Mapped[str] = mapped_column(String(50), nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    producer_name: Mapped[str] = mapped_column(String(100), nullable=False)
    producer_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    subject_entity_id: Mapped[_uuid.UUID | None] = mapped_column(
        ForeignKey("memory_entities.id", ondelete="CASCADE"), nullable=True
    )
    # Populated only for IdentityRecord; NULL otherwise.
    dimension: Mapped[str | None] = mapped_column(String(100), nullable=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    produced_by_run_id: Mapped[_uuid.UUID | None] = mapped_column(
        ForeignKey("consolidation_runs.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active", server_default="active")
    # supersedes_id: regular FK (target always exists at INSERT time).
    # superseded_by_id: DEFERRABLE INITIALLY DEFERRED — design §3.2 requires
    # ``UPDATE prev SET superseded_by_id=NEW_ID`` BEFORE the new row is
    # INSERTed, so the FK target doesn't yet exist at UPDATE time. Postgres
    # re-checks the deferred FK at savepoint commit, by which point the new
    # row is in. The integration-test fixture builds the schema via
    # ``Base.metadata.create_all`` (not Alembic), so the deferrable flag
    # has to live on the model too — migration 064 keeps the live DB in
    # lockstep.
    supersedes_id: Mapped[_uuid.UUID | None] = mapped_column(
        ForeignKey("memory_artifacts.id", ondelete="SET NULL"), nullable=True
    )
    superseded_by_id: Mapped[_uuid.UUID | None] = mapped_column(
        ForeignKey(
            "memory_artifacts.id",
            ondelete="SET NULL",
            deferrable=True,
            initially="DEFERRED",
        ),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, server_default=text("NOW()"))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Bi-temporal columns (M2.5 #1, migration 066). World-time validity,
    # distinct from system-time supersession. Producers populate these when
    # they have evidence about when a fact was true in the world; consumers
    # (D/E/G) read them. NULL t_valid_until = "still valid".
    t_valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    t_valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        # review-r1 fix 9: mirrors migration 062's CHECK constraint so the
        # integration test DB built by Base.metadata.create_all enforces the
        # same status-string allowlist as production. Extended in migration
        # 067 (D Wave 1) to allow 'permanent' for the Anchor row.
        CheckConstraint(
            "status IN ('active', 'superseded', 'expired', 'permanent')",
            name="ck_memory_artifacts_status",
        ),
        Index(
            "idx_artifacts_active_type",
            "artifact_type",
            "subject_entity_id",
            postgresql_where=text("status = 'active'"),
        ),
        Index(
            "idx_artifacts_expires",
            "expires_at",
            postgresql_where=text("status <> 'expired' AND expires_at IS NOT NULL"),
        ),
        Index("idx_artifacts_content_hash", "content_hash"),
        # DB-621 — referencing side of the ``consolidation_runs`` FK (SET NULL).
        # Without it every run the nightly ``_cleanup_old_runs`` deletes made the
        # referential check scan all artifacts. Mirrors migration 106.
        Index(
            "idx_artifacts_produced_by_run",
            "produced_by_run_id",
            postgresql_where=text("produced_by_run_id IS NOT NULL"),
        ),
        # review-r1 fix 6: partial UNIQUE index — at most one active artifact
        # per (artifact_type, subject_entity_id, COALESCE(dimension, '')).
        # Mirrored from migration 061's raw `CREATE UNIQUE INDEX` so that
        # `Base.metadata.create_all` (used by the integration test fixture)
        # builds the same constraint the live DB enforces.
        Index(
            "idx_artifacts_one_active",
            "artifact_type",
            "subject_entity_id",
            text("COALESCE(dimension, '')"),
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
    )
