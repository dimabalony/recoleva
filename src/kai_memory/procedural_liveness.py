"""DB-487 ride-along (b) — the nightly rule-liveness assertion.

The check that would have caught DB-436 on 2026-07-13. Capture was minting rules
the whole time, every serving query required ``status='active'``, and no rule ever
reached ``active`` — so ``procedural.served`` had ZERO occurrences in all of
production history and nobody noticed for 26 days. Nothing in the system asserted
"capture is on, therefore something should eventually be servable".

This is that assertion, and it is deliberately STATELESS: no counter to persist, no
row to keep in sync. "Zero active rules for ≥3 nights" is expressed as *the store
has held rules for at least ``STALE_DAYS`` days and none of them is active* — the
same condition, derived from ``created_at`` instead of from a streak the process
would lose on restart.

Runs as a nightly consolidation step; no-op when capture is disabled.
"""

from __future__ import annotations

import logging
from datetime import UTC

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.host_calls import utc_now
from kai_memory.storage.models.memory_trace import MemoryTrace
from kai_memory.store import cognitive_config
from kai_memory.procedural.procedural_scope import PROC_KINDS, PROC_STATUS_ACTIVE

logger = logging.getLogger("app.services.memory.procedural_liveness")

STALE_DAYS = 3  # rules older than this with none active ⇒ the pipeline is stuck


def is_stale(total: int, active: int, oldest_age_days: float) -> bool:
    """The pure predicate behind the warning — unit-tested without a DB.

    Rules exist, none is active, and the oldest has been sitting there for at least
    ``STALE_DAYS``. An empty store is NOT stale: capture may simply have had nothing
    to capture, which is a different (and normal) state.
    """
    return total > 0 and active == 0 and oldest_age_days >= STALE_DAYS


async def assert_rule_liveness(session: AsyncSession) -> dict[str, int]:
    """Warn when capture is enabled but the rule store has gone inert.

    Returns a small stats dict for the consolidation summary. Never raises for a
    data reason — this is an observability step, not a gate.
    """
    if not cognitive_config.get_bool("procedural_rule_extraction_enabled"):
        return {"skipped": 1}

    status = MemoryTrace.typed_metadata["procedural"]["status"].astext
    stmt = select(
        func.count(),
        func.count().filter(status == PROC_STATUS_ACTIVE),
        func.min(MemoryTrace.created_at),
    ).where(
        MemoryTrace.memory_type == "procedural",
        MemoryTrace.invalidated_at.is_(None),
        MemoryTrace.typed_metadata["procedural"]["kind"].astext.in_(PROC_KINDS),
    )
    total_raw, active_raw, oldest = (await session.execute(stmt)).one()
    total = int(total_raw or 0)
    active = int(active_raw or 0)

    age_days = 0.0
    if oldest is not None:
        if oldest.tzinfo is None:
            oldest = oldest.replace(tzinfo=UTC)
        age_days = (utc_now() - oldest).total_seconds() / 86400.0

    stale = is_stale(total, active, age_days)
    if stale:
        # The DB-436 signature exactly: rules exist, capture is on, nothing can serve.
        logger.warning(
            "procedural.liveness.no_active_rules total=%d active=0 oldest_age_days=%.1f — "
            "capture is enabled but NOTHING can serve; check the activation path",
            total,
            age_days,
        )
    logger.info(
        "procedural.liveness total=%d active=%d oldest_age_days=%.1f stale_days=%d",
        total,
        active,
        age_days,
        STALE_DAYS,
    )
    return {"total": total, "active": active, "stale": int(stale)}
