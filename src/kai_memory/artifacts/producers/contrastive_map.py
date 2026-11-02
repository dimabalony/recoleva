"""ContrastiveMapProducer — design §3.3.2.

Wave 2 ships this as a **graceful no-op**. The
``confusion_sets`` table exists (M1 migration 061) but stays empty
until M3 wave 2 ships ``ConfusionSetProducer`` to populate it. So we
register here, run nightly, and return an empty subject list — the
pipeline records "ContrastiveMap: 0 produced" telemetry without any
errors.

When M3 lands, replace ``discover_subjects`` with the real query
sketched below; the rest of the producer (payload generation,
supersession) stays identical.
"""

from __future__ import annotations

import logging
import uuid as _uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.artifacts.producer import BaseProducer, ProducerSubject
from kai_memory.artifacts.types import ArtifactType, ProducedArtifact

logger = logging.getLogger("app.services.memory.artifacts.producers.contrastive_map")


class ContrastiveMapProducer(BaseProducer):
    """No-op until M3 ConfusionSetProducer populates ``confusion_sets``."""

    name = "ContrastiveMapProducer"
    version = 1
    artifact_type: ArtifactType = "ContrastiveMap"

    async def discover_subjects(self, session: AsyncSession, run_id: _uuid.UUID) -> list[ProducerSubject]:
        # Probe the table — if any confusion_set rows exist, log a warning
        # so the M3 implementer notices that the no-op is suppressing real
        # work. Until then, return [] cleanly.
        result = await session.execute(text("SELECT 1 FROM confusion_sets LIMIT 1"))
        if result.first() is not None:
            logger.warning(
                "ContrastiveMapProducer: confusion_sets has rows but M3 producer "
                "logic is not yet wired — Wave 2 still no-ops. Implement the real "
                "discovery query before M3 ConfusionSetProducer ships."
            )
        # When M3 lands, replace with a query like:
        #   SELECT id FROM confusion_sets cs
        #   LEFT JOIN memory_artifacts ma
        #     ON ma.subject_entity_id = cs.id
        #    AND ma.artifact_type = 'ContrastiveMap'
        #    AND ma.status = 'active'
        #   WHERE ma.id IS NULL
        #      OR cs.last_affirmed_at > ma.created_at;
        return []

    async def produce(
        self,
        session: AsyncSession,
        subject: ProducerSubject,
        run_id: _uuid.UUID,
    ) -> ProducedArtifact | None:
        # Unreachable in Wave 2 — discover_subjects always returns [].
        # Raise rather than silently no-op so a future bug that bypasses
        # the discovery gate fails loudly.
        raise NotImplementedError(
            "ContrastiveMapProducer.produce() is not implemented until M3. "
            "Wave 2 discover_subjects returns [] so this should be unreachable."
        )


__all__ = ["ContrastiveMapProducer"]
