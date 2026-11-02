"""MemoryEntity model — people, places, organizations in the knowledge graph."""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import DDL, BigInteger, DateTime, ForeignKey, Index, Integer, String, Text, event, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, utc_now


class MemoryEntity(Base):
    """Entity in the cognitive memory knowledge graph."""

    __tablename__ = "memory_entities"

    id: Mapped[_uuid.UUID] = mapped_column(
        default=_uuid.uuid4, server_default=text("gen_random_uuid()"), primary_key=True
    )
    entity_type: Mapped[str] = mapped_column(String(30), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    # Cross-script canonicalization key (Sub-design C deferred-cluster). Equal
    # keys ⇒ same clean transliteration, so Cyrillic↔Latin variants of a name
    # resolve to one entity. Populated forward via `translit_key(name)` on write;
    # NULL on pre-088 rows until a backfill (the NULL makes the resolver clause a
    # no-op for those rows, never a wrong match). Indexed for the resolver lookup.
    canonical_key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    aliases: Mapped[list] = mapped_column(JSONB, default=list, server_default=text("'[]'::jsonb"))
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    data: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"))
    namespace: Mapped[str | None] = mapped_column(String(100), nullable=True)

    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, server_default=text("NOW()")
    )
    last_mentioned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    mention_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Snapshot of mention_count written only by L3 consolidation; used for stable
    # ORDER BY in the cached system prompt (see _build_cognitive_context).
    ranking_score: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    telegram_user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    merged_into_id: Mapped[_uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_entities.id"), nullable=True
    )
    merged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    merge_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, server_default=text("NOW()"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=text("NOW()"),
        onupdate=utc_now,
    )

    __table_args__ = (
        Index(
            "idx_mentity_type_name_null_tg",
            "entity_type",
            "name",
            unique=True,
            postgresql_where=text("telegram_user_id IS NULL"),
        ),
        Index(
            "idx_mentity_tg_user",
            "telegram_user_id",
            unique=True,
            postgresql_where=text("telegram_user_id IS NOT NULL"),
        ),
        Index("idx_mentity_type", "entity_type"),
        # DB-621 — the per-message lookups. ``find_entities_by_names`` filters on
        # ``name`` alone (the unique index above leads on entity_type, so the
        # planner never used it) and then on ``aliases @> [x]``; the lowercase
        # fallback compares ``lower(name)`` and the lowercased alias array.
        # Declared here as well as in migration 106 so ``create_all`` test
        # databases carry the real plans (tests/integration/test_db621_index_plans.py).
        Index(
            "idx_mentity_name_active",
            "name",
            postgresql_where=text("merged_into_id IS NULL"),
        ),
        Index(
            "idx_mentity_lower_name_active",
            text("lower(name)"),
            postgresql_where=text("merged_into_id IS NULL"),
        ),
        Index(
            "idx_mentity_aliases_gin",
            "aliases",
            postgresql_using="gin",
            postgresql_ops={"aliases": "jsonb_path_ops"},
            postgresql_where=text("merged_into_id IS NULL"),
        ),
        Index(
            "idx_mentity_aliases_lower_gin",
            text("aliases_lower(aliases)"),
            postgresql_using="gin",
            postgresql_where=text("merged_into_id IS NULL"),
        ),
        # Resolver cross-script lookup filters on (entity_type, canonical_key).
        Index(
            "idx_mentity_type_canonical_key",
            "entity_type",
            "canonical_key",
            postgresql_where=text("canonical_key IS NOT NULL"),
        ),
        Index(
            "idx_memory_entities_ranking_score",
            ranking_score.desc(),
            "id",
            postgresql_where=text("summary IS NOT NULL"),
        ),
    )


#: IMMUTABLE helper behind ``idx_mentity_aliases_lower_gin``: the alias array with
#: every element lowercased, so ``aliases_lower(aliases) ?| ARRAY[tokens]`` is a
#: GIN probe. A non-array value (nothing writes one, but the expression is
#: evaluated on every row write) yields an empty array instead of failing the
#: write. Plain SQL — no superuser needed on RDS. Mirrors migration 106; keep
#: the two texts identical. Registered ``before_create`` so ``create_all`` builds
#: the function before the expression index that references it.
ALIASES_LOWER_DDL = (
    "CREATE OR REPLACE FUNCTION aliases_lower(jsonb) RETURNS jsonb "
    "LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS $$ "
    "SELECT CASE WHEN jsonb_typeof($1) = 'array' "
    "THEN (SELECT COALESCE(jsonb_agg(lower(v)), '[]'::jsonb) FROM jsonb_array_elements_text($1) AS a(v)) "
    "ELSE '[]'::jsonb END $$"
)

event.listen(MemoryEntity.__table__, "before_create", DDL(ALIASES_LOWER_DDL))
