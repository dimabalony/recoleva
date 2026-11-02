"""One-time data-cleanup helpers for cross-script entity canonicalization.

Companion to migration 088 / ``kai_memory.canonicalization``. The
shipped resolver only PREVENTS new cross-script forks; these helpers repair the
EXISTING corrupted data (Sub-design C deferred cluster). PROD WRITES — run only
behind explicit human sign-off, on a per-id list, inside a transaction.

Three operations, each precision-first:

- ``backfill_canonical_key`` — populate ``canonical_key = translit_key(name)`` on
  existing rows the migration left NULL. Empty keys (garbage/punct-only names)
  stay NULL so they can never mass-match.
- ``merge_into_survivor`` — fold a human-confirmed set of fork rows into a
  CALLER-DESIGNATED survivor, regardless of mention_count (so e.g. the agent-Kai
  row survives even though a person-Kai variant has more mentions). Mirrors the
  resolver's telegram_user_id guard: never merges across two distinct non-NULL
  ``telegram_user_id`` values. Relationship + entity-link re-pointing reuses the
  battle-tested ``consolidation._dedup_and_transfer_relationships``.
- ``close_relationship`` — soft-close a wrong edge via ``valid_until``; never a
  hard delete (history is preserved).
"""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import load_only

from kai_memory.host_calls import utc_now
from kai_memory.identity import identity_merge_survivor
from kai_memory.storage.models.entity_relationship import EntityRelationship
from kai_memory.storage.models.memory_entity import MemoryEntity
from kai_memory.consolidation.consolidation import _dedup_and_transfer_relationships
from kai_memory.canonicalization import translit_key


async def backfill_canonical_key(session: AsyncSession, *, batch_size: int = 500) -> int:
    """Set ``canonical_key = translit_key(name)`` on active, unkeyed rows.

    Only loads id/name/canonical_key (never the embedding vector) to stay light
    on the 512 MB container. Rows whose ``translit_key`` is empty are left NULL.
    Returns the number of rows actually keyed.
    """
    stmt = (
        select(MemoryEntity)
        .where(
            MemoryEntity.merged_into_id.is_(None),
            MemoryEntity.canonical_key.is_(None),
        )
        .options(load_only(MemoryEntity.id, MemoryEntity.name, MemoryEntity.canonical_key))
    )
    result = await session.execute(stmt)
    rows = result.scalars().all()
    count = 0
    for entity in rows:
        key = translit_key(entity.name or "")
        if not key:
            continue
        entity.canonical_key = key
        count += 1
        if count % batch_size == 0:
            await session.flush()
    await session.flush()
    return count


async def merge_into_survivor(
    session: AsyncSession,
    survivor_id: _uuid.UUID,
    absorbed_ids: list[_uuid.UUID],
    *,
    reason: str,
) -> dict:
    """Fold ``absorbed_ids`` into ``survivor_id`` (survivor is forced, not picked).

    Skips (does NOT merge) any absorbed row whose non-NULL ``telegram_user_id``
    differs from the survivor's non-NULL ``telegram_user_id`` — mirroring the
    resolver guard so two distinct real users can never be collapsed.
    """
    with session.no_autoflush:
        survivor = await session.get(MemoryEntity, survivor_id)
        if survivor is None:
            raise ValueError(f"survivor {survivor_id} not found")

        absorbed_rows = {
            entity_id: await session.get(MemoryEntity, entity_id)
            for entity_id in absorbed_ids
            if entity_id != survivor_id
        }
        # Validate the complete batch before any earlier fork can be mutated.
        for absorbed in absorbed_rows.values():
            if absorbed is not None:
                protected = identity_merge_survivor(survivor, absorbed)
                if protected is not None and protected is not survivor:
                    raise ValueError("cannot absorb bound agent identity")

    merged: list[_uuid.UUID] = []
    skipped_tg_conflict: list[_uuid.UUID] = []
    skipped_missing: list[_uuid.UUID] = []
    now = utc_now()

    for absorbed_id in absorbed_ids:
        if absorbed_id == survivor_id:
            continue
        absorbed = absorbed_rows[absorbed_id]
        if absorbed is None:
            skipped_missing.append(absorbed_id)
            continue
        # Explicit non-NULL (not truthiness): a telegram_user_id of 0 is still a
        # distinct id and must block a conflicting merge.
        if (
            survivor.telegram_user_id is not None
            and absorbed.telegram_user_id is not None
            and survivor.telegram_user_id != absorbed.telegram_user_id
        ):
            skipped_tg_conflict.append(absorbed_id)
            continue

        aliases = list(survivor.aliases or [])
        if absorbed.name != survivor.name and absorbed.name not in aliases:
            aliases.append(absorbed.name)
        for alias in absorbed.aliases or []:
            if alias != survivor.name and alias not in aliases:
                aliases.append(alias)
        survivor.aliases = aliases

        absorbed.merged_into_id = survivor.id
        absorbed.merged_at = now
        absorbed.merge_reason = reason
        await session.flush()

        await _dedup_and_transfer_relationships(session, survivor.id, absorbed.id)
        merged.append(absorbed_id)

    return {
        "survivor": survivor_id,
        "merged": merged,
        "skipped_tg_conflict": skipped_tg_conflict,
        "skipped_missing": skipped_missing,
    }


async def close_relationship(
    session: AsyncSession,
    rel_id: _uuid.UUID,
    *,
    now: datetime | None = None,
) -> bool:
    """Soft-close a relationship by stamping ``valid_until``. Never deletes.

    Returns True if it closed an active edge, False if the edge is missing or
    already closed.
    """
    rel = await session.get(EntityRelationship, rel_id)
    if rel is None:
        return False
    if rel.valid_until is not None:
        return False
    rel.valid_until = now or utc_now()
    await session.flush()
    return True
