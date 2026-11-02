"""Typed contracts for the artifact pipeline (M2 Wave 1).

Layered on top of the M1 pieces that already exist:
- ``Artifact`` (frozen dataclass) and ``ArtifactType`` Literal — defined in
  ``kai_memory.types`` and re-exported here so producers don't reach
  into the parent package.
- Pydantic payload models (``EntitySchemaPayload``, ``ContrastiveMapPayload``,
  …) — defined in ``kai_memory.artifact_schemas`` and re-exported here.

Wave-1-new types in this module:
- ``ArtifactStatus`` Literal mirroring the DB CHECK constraint
- ``ChangesMadeAction`` Literal of the five action verbs the design enumerates
- ``ChangesMadeEntry`` — the dict written into ``consolidation_runs.changes_made``
  by ``ArtifactPipeline`` (Wave 3 ``preview_run`` / ``rollback_run`` consume it)
- ``ProducedArtifact`` — the value an ``ArtifactProducer.produce()`` returns
  before the pipeline persists it. Carries ``human_summary`` for §3.4.1 preview.

Wave 2 producers extend the payload models if their algorithm needs additional
fields (per §14 deferred refinements). The pydantic models stay
``extra="forbid"`` so any extension lands in a reviewed schema bump rather than
silently widening payloads on disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# Re-export so ``from kai_memory.artifacts import Artifact`` works.
from kai_memory.artifact_schemas import (
    AnchorPayload,
    ChallengeEvent,
    ContrastiveMapMember,
    ContrastiveMapPayload,
    CueMappingPayload,
    EntitySchemaPayload,
    IdentityRecordPayload,
    PatternPayload,
    PersonalityTraitPayload,
    ReinforcementEvent,
    RelationshipRef,
    StancePayload,
    TrainingDialogPayload,
    TrajectoryEvent,
    TrajectoryPayload,
)
from kai_memory.types import Artifact, ArtifactId, ArtifactType, EntityId

# ── DB-aligned status enum ────────────────────────────────
# Mirrors the ``ck_memory_artifacts_status`` CHECK constraint
# (migration 062, extended in 067 for D Wave 1's ``permanent``). Keep
# this in lockstep with that constraint — if a future migration adds
# a status value, update both this Literal and the CHECK in alembic.
ArtifactStatus = Literal["active", "superseded", "expired", "permanent"]


# ── ChangesMadeEntry — preview / rollback audit row ───────
# - ``create``     fresh artifact, no prior
# - ``supersede``  prior → new (both rows updated)
# - ``expire``     status=active → status=expired (e.g. CueMapping archive)
# - ``noop``       content_hash matched prior → no DB write
# - ``conflict``   another producer in this savepoint already superseded prev
ChangesMadeAction = Literal["create", "supersede", "expire", "noop", "conflict"]


@dataclass(frozen=True)
class ChangesMadeEntry:
    """One audit row appended to ``consolidation_runs.changes_made``.

    Wave 3 ``rollback_run`` reads this back and computes the inverse action.
    Stored as JSONB; serialize via ``to_dict()``.
    """

    action: ChangesMadeAction
    artifact_id: ArtifactId | None
    prev_artifact_id: ArtifactId | None
    artifact_type: ArtifactType
    subject_entity_id: EntityId | None
    human_summary: str
    producer_name: str
    producer_version: int

    def to_dict(self) -> dict[str, Any]:
        """Render to JSON-safe dict for JSONB persistence."""
        return {
            "action": self.action,
            "artifact_id": str(self.artifact_id) if self.artifact_id else None,
            "prev_artifact_id": (str(self.prev_artifact_id) if self.prev_artifact_id else None),
            "artifact_type": self.artifact_type,
            "subject_entity_id": (str(self.subject_entity_id) if self.subject_entity_id else None),
            "human_summary": self.human_summary,
            "producer_name": self.producer_name,
            "producer_version": self.producer_version,
        }


# ── ProducedArtifact — producer.produce() return type ─────
# The pipeline persists this via ``BaseProducer.supersede()``. Producers must
# set ``content_hash`` if they want hash-based dedup (the default behaviour is
# to compute it from the canonical-JSON of ``payload`` if left as ``None``).
@dataclass(frozen=True)
class ProducedArtifact:
    """The in-memory artifact a producer hands back to the pipeline.

    ``payload`` carries the producer's structured fields and is persisted
    verbatim to ``memory_artifacts.payload`` (JSONB). Producers SHOULD pass a
    Pydantic payload model from ``artifact_schemas.py`` and call
    ``model.model_dump(mode='json')`` at the boundary so timestamps and UUIDs
    serialize correctly.

    ``human_summary`` is the short natural-language description Wave 3's
    ``format_run_preview`` uses. Keep it under 200 chars.
    """

    artifact_type: ArtifactType
    schema_version: int
    subject_entity_id: EntityId | None
    dimension: str | None
    payload: dict[str, Any]
    human_summary: str
    content_hash: str | None = None
    expires_at: Any = None  # datetime | None — ``Any`` to avoid forcing tz import here

    # Optional producer-supplied fields. The pipeline can override these via
    # the producer's ``name`` / ``version`` attributes if not provided.
    producer_name: str | None = None
    producer_version: int | None = None
    # Free-form metadata — does NOT land in memory_artifacts (no column for it)
    # but is preserved in the per-producer ChangesMadeEntry.human_summary if
    # the producer wants to render it. Kept to keep the dataclass open for
    # producer-internal bookkeeping without needing a payload-shape change.
    notes: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "AnchorPayload",
    "Artifact",
    "ArtifactId",
    "ArtifactStatus",
    "ArtifactType",
    "ChallengeEvent",
    "ChangesMadeAction",
    "ChangesMadeEntry",
    "ContrastiveMapMember",
    "ContrastiveMapPayload",
    "CueMappingPayload",
    "EntityId",
    "EntitySchemaPayload",
    "IdentityRecordPayload",
    "PatternPayload",
    "PersonalityTraitPayload",
    "ProducedArtifact",
    "ReinforcementEvent",
    "RelationshipRef",
    "StancePayload",
    "TrainingDialogPayload",
    "TrajectoryEvent",
    "TrajectoryPayload",
]
