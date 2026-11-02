"""ArtifactProducer Protocol + BaseProducer ABC (M2 Wave 1).

Producers are pure functions of ``(session_state, run_id)``. Each runs inside
a savepoint owned by ``ArtifactPipeline._run_one``. Producer failures are
contained: their savepoint rolls back, the pipeline keeps going.

The default ``supersede()`` implements the exact 5-step transaction sequence
specified in design §3.2 — implementer-quoted comment lives there. The four
edge cases the design enumerates are exercised end-to-end by the Wave 1
test suite:

1. ``prev is None``                          → INSERT, action='create'
2. ``prev is not None`` and hashes match     → no DB write, action='noop'
3. ``prev is not None`` and hashes differ    → UPDATE prev + INSERT new,
                                                action='supersede'
4. UPDATE returns zero rows                  → another producer beat us to it,
                                                action='conflict' (no raise)
"""

from __future__ import annotations

import abc
import hashlib
import json
import logging
import uuid as _uuid
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import insert, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.storage.models.memory_artifact import MemoryArtifact
from kai_memory.artifacts.types import (
    Artifact,
    ArtifactType,
    ChangesMadeAction,
    ChangesMadeEntry,
    ProducedArtifact,
)
from kai_memory.types import EntityId

logger = logging.getLogger("app.services.memory.artifacts.producer")

# Default Sonnet model for Sonnet-backed producers. Centralized here (the module
# every producer already imports) so the L3 producer fleet moves together on a
# model bump instead of drifting across per-producer copies.
SONNET_MODEL = "sonnet-5.5"


@dataclass(frozen=True)
class ProducerSubject:
    """Opaque subject handle returned by ``discover_subjects``.

    Wave 1 keeps this generic — Wave 2 producers can carry whatever
    discovery shape they need (entity_id, candidate cue phrase, trajectory
    cluster, …). The pipeline never inspects ``key`` beyond logging.
    """

    key: str
    payload: dict[str, Any]


@runtime_checkable
class ArtifactProducer(Protocol):
    """Producer Protocol — implemented by every nightly artifact producer.

    Concrete producers should subclass ``BaseProducer`` to inherit the
    default ``supersede()`` (5-step transaction). Wave 2 will register the
    five concrete producers (EntitySchema, ContrastiveMap, CueMapping,
    Trajectory, IdentityRecord) on this Protocol.
    """

    name: str
    version: int
    artifact_type: ArtifactType

    async def discover_subjects(self, session: AsyncSession, run_id: _uuid.UUID) -> list[ProducerSubject]: ...

    async def produce(
        self,
        session: AsyncSession,
        subject: ProducerSubject,
        run_id: _uuid.UUID,
    ) -> ProducedArtifact | None: ...

    async def supersede(
        self,
        session: AsyncSession,
        prev: Artifact | None,
        new: ProducedArtifact,
        run_id: _uuid.UUID,
    ) -> ChangesMadeEntry: ...


def _canonical_json(payload: dict[str, Any]) -> str:
    """Deterministic JSON for content-hashing.

    ``sort_keys=True`` + ``separators=(',', ':')`` matches what Postgres'
    ``jsonb`` would produce with stable key ordering. Producers that pass
    Pydantic ``.model_dump(mode='json')`` payloads get JSON-safe values
    automatically.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def compute_content_hash(payload: dict[str, Any]) -> str:
    """Public so producers / tests can pre-compute the same hash."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class BaseProducer(abc.ABC):
    """ABC implementing the §3.2 default ``supersede()`` transaction.

    Concrete producers MUST set ``name``, ``version``, ``artifact_type`` as
    class attributes and implement ``discover_subjects`` + ``produce``. They
    MAY override ``supersede`` if their artifact type needs a non-default
    transition (e.g. Trajectory's append-mode mutation). The default below
    is the spec-quoted 5-step sequence and should suit four of the five
    Wave 2 producers.
    """

    name: str = ""
    version: int = 0
    artifact_type: ArtifactType  # Subclasses must set; declared here for typing.

    @abc.abstractmethod
    async def discover_subjects(self, session: AsyncSession, run_id: _uuid.UUID) -> list[ProducerSubject]: ...

    @abc.abstractmethod
    async def produce(
        self,
        session: AsyncSession,
        subject: ProducerSubject,
        run_id: _uuid.UUID,
    ) -> ProducedArtifact | None: ...

    async def supersede(
        self,
        session: AsyncSession,
        prev: Artifact | None,
        new: ProducedArtifact,
        run_id: _uuid.UUID,
    ) -> ChangesMadeEntry:
        """Default content-hash-dedup supersession (design §3.2, two-step variant).

        Runs inside the producer's savepoint (caller's responsibility).

        Sequence:
          1. compute ``new.content_hash`` if not pre-set
          2. prev present and hashes match → ``noop``, no DB write
          3. prev present → UPDATE prev SET ``status='superseded'``,
             ``superseded_by_id=NEW_ID`` WHERE id=prev.id AND status='active'
             RETURNING id. Atomic concurrent-supersession guard. If zero rows
             come back, another producer already superseded prev — skip step
             4 entirely (no INSERT, no orphan) and return ``conflict``
             pointing at prev.id.
          4. INSERT new row with ``status='active'`` and ``supersedes_id=prev.id``.
             Partial unique index allows this because step 3 already moved
             prev out of the active set.
          5. emit ``ChangesMadeEntry`` with ``create`` / ``supersede``.

        **Schema invariant** (migration 064): the
        ``memory_artifacts_superseded_by_id_fkey`` FK is
        ``DEFERRABLE INITIALLY DEFERRED``. Without that, step 3 would fail
        because NEW_ID does not yet exist when the UPDATE runs. Postgres
        re-checks the deferred FK at savepoint commit, by which point
        step 4 has inserted the row. The supersedes_id FK stays
        non-deferrable — it always points at an already-existing prev row.

        **Why UPDATE-first beats INSERT-first:** the partial unique index
        ``idx_artifacts_one_active WHERE status = 'active'`` is non-deferrable
        (Postgres does not support deferrable partial indexes). With
        INSERT-first ordering, both rows would be momentarily 'active'
        between INSERT and the supersede UPDATE, violating the index.
        With UPDATE-first ordering, prev leaves the active set immediately
        and the new INSERT enters cleanly.
        """
        producer_name = new.producer_name or self.name
        producer_version = new.producer_version if new.producer_version is not None else self.version

        # Step 0 — Sub-design D Wave 1 defense-in-depth: refuse permanent prev.
        # The DB trigger ``prevent_permanent_artifact_modification`` (migration
        # 068) is the ground truth, but raising here gives a clearer error and
        # avoids a savepoint roundtrip on a write that's guaranteed to fail.
        # Permanent artifacts (today: the Anchor row) are mutable only via the
        # admin path, never from a producer.
        if prev is not None and prev.status == "permanent":
            raise PermissionError(
                f"BaseProducer.supersede refused: prev artifact {prev.id} "
                f"(artifact_type={prev.artifact_type}) is permanent. "
                "Permanent artifacts are mutable only via the admin path "
                "(SET LOCAL app.allow_permanent_edit = 'true')."
            )

        # Step 1 — compute content hash if not pre-set.
        content_hash = new.content_hash or compute_content_hash(new.payload)

        # Step 2 — content-hash dedup.
        if prev is not None and prev.content_hash == content_hash:
            logger.debug(
                "BaseProducer.supersede noop: prev %s hash matches new (%s)",
                prev.id,
                content_hash[:12],
            )
            return ChangesMadeEntry(
                action="noop",
                artifact_id=prev.id,
                prev_artifact_id=prev.id,
                artifact_type=self.artifact_type,
                subject_entity_id=new.subject_entity_id,
                human_summary=new.human_summary,
                producer_name=producer_name,
                producer_version=producer_version,
            )

        new_id = _uuid.uuid4()

        # Step 3 — UPDATE prev to mark superseded AND point at new_id in one
        # statement. Concurrent-supersession guard via ``status='active'``
        # in WHERE + RETURNING. Relies on the DEFERRABLE FK so superseded_by_id
        # may reference a row that hasn't been inserted yet.
        if prev is not None:
            guard_stmt = (
                update(MemoryArtifact)
                .where(
                    MemoryArtifact.id == prev.id,
                    MemoryArtifact.status == "active",
                )
                .values(status="superseded", superseded_by_id=new_id)
                .returning(MemoryArtifact.id)
            )
            result = await session.execute(guard_stmt)
            updated_ids = result.scalars().all()
            if not updated_ids:
                # Another producer already superseded prev. The winning
                # producer has already inserted its successor, so we skip
                # step 4 entirely — no orphan, no double-active row.
                logger.warning(
                    "BaseProducer.supersede conflict: prev=%s already non-active",
                    prev.id,
                )
                return ChangesMadeEntry(
                    action="conflict",
                    artifact_id=None,
                    prev_artifact_id=prev.id,
                    artifact_type=self.artifact_type,
                    subject_entity_id=new.subject_entity_id,
                    human_summary=new.human_summary,
                    producer_name=producer_name,
                    producer_version=producer_version,
                )

        # Step 4 — INSERT new active row. Safe against the partial unique
        # index because step 3 already moved prev (if any) out of 'active'.
        insert_stmt = insert(MemoryArtifact).values(
            id=new_id,
            artifact_type=self.artifact_type,
            schema_version=new.schema_version,
            producer_name=producer_name,
            producer_version=producer_version,
            subject_entity_id=new.subject_entity_id,
            dimension=new.dimension,
            payload=new.payload,
            content_hash=content_hash,
            produced_by_run_id=run_id,
            status="active",
            supersedes_id=prev.id if prev is not None else None,
            expires_at=new.expires_at,
        )
        await session.execute(insert_stmt)

        # Step 5 — emit ChangesMadeEntry.
        action: ChangesMadeAction = "supersede" if prev is not None else "create"
        return ChangesMadeEntry(
            action=action,
            artifact_id=new_id,
            prev_artifact_id=prev.id if prev is not None else None,
            artifact_type=self.artifact_type,
            subject_entity_id=new.subject_entity_id,
            human_summary=new.human_summary,
            producer_name=producer_name,
            producer_version=producer_version,
        )

    # ── helpers shared by Wave 2 producers ────────────────

    async def _fetch_active(
        self,
        session: AsyncSession,
        subject_entity_id: EntityId | None,
        dimension: str | None = None,
    ) -> Artifact | None:
        """Return the current active artifact for this (type, subject, dim).

        Implements the read half of the supersede transaction. Wave 2
        producers can call this from ``produce()`` to materialise the
        ``prev`` argument they pass into ``supersede()``. Uses a parameterised
        query that mirrors ``idx_artifacts_one_active``.
        """
        stmt = text(
            """
            SELECT id, artifact_type, schema_version, producer_name,
                   producer_version, subject_entity_id, payload,
                   content_hash, status, supersedes_id, created_at, expires_at
              FROM memory_artifacts
             WHERE artifact_type = :artifact_type
               AND status = 'active'
               AND (subject_entity_id = :subject_entity_id
                    OR (subject_entity_id IS NULL AND :subject_entity_id IS NULL))
               AND COALESCE(dimension, '') = COALESCE(:dimension, '')
             LIMIT 1
            """
        )
        row = (
            (
                await session.execute(
                    stmt,
                    {
                        "artifact_type": self.artifact_type,
                        "subject_entity_id": subject_entity_id,
                        "dimension": dimension,
                    },
                )
            )
            .mappings()
            .first()
        )
        if row is None:
            return None
        return Artifact(
            id=row["id"],
            artifact_type=row["artifact_type"],
            schema_version=row["schema_version"],
            producer_name=row["producer_name"],
            producer_version=row["producer_version"],
            subject_entity_id=row["subject_entity_id"],
            payload=row["payload"],
            content_hash=row["content_hash"],
            status=row["status"],
            supersedes_id=row["supersedes_id"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
        )


__all__ = [
    "ArtifactProducer",
    "BaseProducer",
    "ProducerSubject",
    "compute_content_hash",
]
