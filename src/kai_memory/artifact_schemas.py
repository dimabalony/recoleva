"""Pydantic v2 payload models for consolidation artifacts.

Each ArtifactType has a concrete model with `schema_version: Literal[1] = 1`
locked. ContrastiveMapPayload and TrajectoryPayload contain nested submodels;
both call `model_rebuild()` at module load.

Wave 2 (M2) extends the M1 stubs to match the Wave 2 producer specs in
`backend-docs/ai-workforce/plans/design/2026-05-02-memory-v2b-artifact-pipeline-design.md` §3.3.
No payloads exist on disk yet (M1 shipped only the table; first M2 nightly
populates it), so changing the v1 shape is safe.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class _ArtifactPayload(BaseModel):
    """Common config for all payload models — strict, immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RelationshipRef(BaseModel):
    """One outgoing edge from an entity, captured in EntitySchemaPayload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    predicate: str
    target_entity_id: uuid.UUID
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class EntitySchemaPayload(_ArtifactPayload):
    """Compiled canonical view of an entity (design §3.3.1)."""

    schema_version: Literal[1] = 1
    canonical_name: str
    entity_type: str
    aliases: list[str] = Field(default_factory=list)
    canonical_fields: dict = Field(default_factory=dict)
    # INC 3 §3.5 — decoupled structured store: the render-facing ``canonical_fields``
    # stays a flat dict[str,str]; the durable-attribute aggregator's structured slot
    # values (value / observed_at / source_trace_id / durability) live here. Cutover
    # target — the renderer migrates to read this only after shadow validates, so it
    # defaults empty and is additive/back-compatible with existing artifacts.
    canonical_slot_values: dict = Field(default_factory=dict)
    relationships: list[RelationshipRef] = Field(default_factory=list)
    last_evidence_trace_id: uuid.UUID | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class ContrastiveMapMember(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_id: uuid.UUID
    distinguishing_features: dict = Field(default_factory=dict)


class ContrastiveMapPayload(_ArtifactPayload):
    """Authoritative contrastive representation for a confusion set."""

    schema_version: Literal[1] = 1
    confusion_set_id: uuid.UUID
    confusion_source: str | None = None
    members: list[ContrastiveMapMember]
    rationale: str | None = None


class CueMappingPayload(_ArtifactPayload):
    """Pattern → EntitySchema hydration mapping (design §3.3.3).

    ``target_schema_ids`` is a list because one cue can hydrate several
    schemas at once (e.g. "наша собака" → {Луна + Тучка}). Wave 2's
    initial implementation produces single-target mappings; the list shape
    keeps room for M3 multi-target adjudication without a schema bump.
    """

    schema_version: Literal[1] = 1
    cue_pattern: str
    # An empty target list would persist a no-op cue that the prompt builder
    # has nothing to hydrate from. Reject at validation time.
    target_schema_ids: list[uuid.UUID] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    fired_count_30d: int = Field(ge=0, default=0)


class TrajectoryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: uuid.UUID
    # AwareDatetime rejects naive datetime values at validation time so the
    # ruff DTZ enforcement applied to runtime code can't be silently bypassed
    # by a producer that constructs payloads from `datetime.now()`.
    timestamp: AwareDatetime
    summary: str


class TrajectoryPayload(_ArtifactPayload):
    """Time-ordered narrative across related traces (design §3.3.4).

    Status invariant (enforced in ``_check_status_closed_at_consistency``):
    ``status='closed'`` ⇔ ``closed_at`` is set. An open trajectory has
    ``closed_at IS NULL``; a closed trajectory must record when it closed.
    Without this validator a producer could ship a "closed but never closed"
    artifact that confuses Wave 3's preview/rollback rendering.
    """

    schema_version: Literal[1] = 1
    subject_entity_ids: list[uuid.UUID]
    events: list[TrajectoryEvent]
    status: Literal["open", "closed"] = "open"
    opened_at: AwareDatetime
    closed_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _check_status_closed_at_consistency(self) -> "TrajectoryPayload":
        if self.status == "closed" and self.closed_at is None:
            raise ValueError("TrajectoryPayload status='closed' requires closed_at to be set")
        if self.status == "open" and self.closed_at is not None:
            raise ValueError("TrajectoryPayload status='open' must have closed_at=None")
        return self


class IdentityRecordPayload(_ArtifactPayload):
    """Structured drift record (design §3.3.5).

    Uniqueness key is ``(subject_entity_id, dimension)``. Multiple records
    can coexist for the same subject if they describe different dimensions.
    The artifacts table enforces per-(type, subject, dimension) uniqueness
    via the partial unique index from migration 061. The producer MUST set
    ``subject_entity_id`` to a non-NULL value (Postgres treats NULLs as
    distinct in unique indexes — see §4.1 invariant note).
    """

    schema_version: Literal[1] = 1
    subject_entity_id: uuid.UUID
    dimension: str
    direction: Literal["increasing", "decreasing", "stable"]
    magnitude: float = Field(ge=0.0, le=1.0)
    driving_observations: list[uuid.UUID]
    # See TrajectoryEvent.timestamp note — AwareDatetime forbids naive values
    # so producers can't bypass DTZ enforcement at the schema layer.
    epoch_start: AwareDatetime
    epoch_end: AwareDatetime | None = None


class TrainingDialogPayload(_ArtifactPayload):
    """Type spec preserved for Layer 3; producer dropped from v2 scope."""

    schema_version: Literal[1] = 1
    messages: list[dict] = Field(default_factory=list)


# ── Sub-design D — persona pipeline payloads (Wave 1+2) ────────


class AnchorPayload(_ArtifactPayload):
    """Frozen foundational identity (design §3.2).

    Stored as a single ``status='permanent'`` row. Mutable only via the
    admin path (``SET LOCAL app.allow_permanent_edit = 'true'``); the DB
    trigger ``prevent_permanent_artifact_modification`` (migration 068)
    refuses any other modification, and ``BaseProducer.supersede`` raises
    ``PermissionError`` on a permanent ``prev`` for fast app-layer fail.

    ``content`` is rendered verbatim by ``PersonaComposer.compose()`` as
    the foundational text block — advisor-with-care framing per
    arXiv:2603.00024.
    """

    schema_version: Literal[1] = 1
    # Min 100 chars matches migration 069's self-validation: anything
    # shorter is almost certainly a placeholder that slipped review.
    # Max 8000: D Wave 1 freezes the current prod persona/soul content
    # verbatim (~5.8k chars) without trimming. Post-Wave-3 admin-path
    # edits are expected to bring the Anchor down toward the ~1-2k char
    # range design §3.2 envisioned.
    content: str = Field(min_length=100, max_length=8000)
    framing_locked_at: AwareDatetime
    schema_revision: int = Field(ge=1, default=1)


class PersonalityTraitPayload(_ArtifactPayload):
    """Slow-evolving personality dimension (design §3.3).

    Supersedable per-dimension (Wave 2). Decay model:

        strength = magnitude
                 * exp(-lambda * days_since_last_reinforced)
                 * (1 + rehearsal_count * 0.2)

    Per-layer lambda lives in ``cognitive_config`` (migration 067).
    ``guidance_text`` is the Sonnet-generated prose the composer renders.
    """

    schema_version: Literal[1] = 1
    dimension: str = Field(min_length=1, max_length=64)
    direction: Literal["increasing", "decreasing", "stable"]
    magnitude: float = Field(ge=0.0, le=1.0)
    guidance_text: str = Field(min_length=1, max_length=500)
    driving_observations: list[uuid.UUID] = Field(default_factory=list)
    epoch_start: AwareDatetime
    last_reinforced_at: AwareDatetime
    rehearsal_count: int = Field(ge=0, default=0)


class ReinforcementEvent(BaseModel):
    """Evidence that supports a Stance (Wave 2 StanceProducer)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["agreed", "outcome_validated", "held_under_pushback"]
    trace_id: uuid.UUID | None = None
    occurred_at: AwareDatetime
    note: str = Field(default="", max_length=300)


class ChallengeEvent(BaseModel):
    """Evidence that contradicts a Stance (Wave 2 StanceProducer)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["disagreed", "outcome_invalidated"]
    trace_id: uuid.UUID | None = None
    occurred_at: AwareDatetime
    note: str = Field(default="", max_length=300)


class StancePayload(_ArtifactPayload):
    """Kai's own opinion (design §3.4).

    All Stances use ``subject_entity_id = kai_self_entity_id`` (the row
    in ``memory_entities`` matching ``name='Kai' AND entity_type='agent'``)
    so the partial unique index
    ``idx_artifacts_one_active`` enforces per-(subject, dimension)
    uniqueness despite Postgres NULL-distinct semantics.

    Reinforcement signals append without superseding; supersession
    happens when accumulated evidence shifts the position itself.
    """

    schema_version: Literal[1] = 1
    topic: str = Field(min_length=1, max_length=64)
    position: str = Field(min_length=1, max_length=300)
    reasoning: str = Field(min_length=1, max_length=600)
    reinforcement_signals: list[ReinforcementEvent] = Field(default_factory=list)
    challenge_signals: list[ChallengeEvent] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    last_validated_at: AwareDatetime


class PatternPayload(_ArtifactPayload):
    """Recent observation, fast-decaying (design §3.5).

    No per-dimension uniqueness — multiple active Patterns are allowed
    (sliding window, default 7 days). Promotes to ``PersonalityTrait``
    when ``reinforcement_count >= persona_pattern_promotion_threshold``
    (default 3) — promotion runs in-pipeline same-run for clean
    ``changes_made`` attribution.
    """

    schema_version: Literal[1] = 1
    dimension: str = Field(min_length=1, max_length=64)
    observation: str = Field(min_length=1, max_length=250)
    driving_traces: list[uuid.UUID] = Field(default_factory=list)
    observed_at: AwareDatetime
    # Always at least 1 — the first observation creates the row.
    reinforcement_count: int = Field(ge=1, default=1)


# Resolve forward refs for nested submodels.
ContrastiveMapPayload.model_rebuild()
TrajectoryPayload.model_rebuild()
EntitySchemaPayload.model_rebuild()
StancePayload.model_rebuild()
