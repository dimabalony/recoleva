"""Typed contracts for the Memory v2 interface.

MemoryClient is the supported host lifecycle. MemoryInterface remains the
compatibility backend protocol; these shared values describe stored memory.

This module is type-only — no production wiring lives here. The interface
gains a concrete implementation in v2a Task 8 via the adapter wrapping
CognitiveMemoryService.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

TraceId = uuid.UUID
EntityId = uuid.UUID
ArtifactId = uuid.UUID
ConsolidationRunId = uuid.UUID

# All 11 existing values; "relationship" kept for compatibility — defined but
# not emitted by L2/L3 today.
MemoryType = Literal[
    "fact",
    "episode",
    "impression",
    "preference",
    "relationship",
    "plan",
    "intention",
    "status_change",
    "decision",
    "social_norm",
    "self_reflection",
]

# Expanded from 5 to 9 (Task 4 will wire the new four into extraction).
EntityType = Literal[
    "person",
    "animal",
    "place",
    "organization",
    "topic",
    "project",
    "object",
    "event",
    "group",
    # D Wave 1 — kai-self entity row uses 'agent' so it's distinguishable
    # from human persons + can be matched by predicate alone in producer
    # lookups (no need for a separate name-equality check across millions
    # of rows).
    "agent",
]

ArtifactType = Literal[
    "EntitySchema",
    "ContrastiveMap",
    "CueMapping",
    "Trajectory",
    "IdentityRecord",
    "TrainingDialog",
    # Sub-design D - persona pipeline (Wave 1+2).
    "Anchor",  # frozen foundational identity (status='permanent')
    "PersonalityTrait",  # slow-evolving personality dimension
    "Stance",  # Kai's own opinions
    "Pattern",  # recent observation, fast-decaying
]

LinkRole = Literal["subject", "about", "mentioned", "participant"]

Modality = Literal["text", "image", "audio", "transcript", "sensor"]


@dataclass(frozen=True)
class Representation:
    modality: Modality
    payload_ref: str
    embedding_ref: str | None
    provenance: str


@dataclass(frozen=True)
class EncodingSignals:
    """Captured at encoding time. Drives OEC mode + Haiku scoring.

    Stored in memory_traces.encoding_signals JSONB.
    """

    identity: float = 0.0
    correction: float = 0.0
    introduction: float = 0.0
    haiku_novelty: float | None = None
    haiku_arousal: float | None = None
    oec_mode: Literal["defer", "eager", "deep"] | None = None


@dataclass(frozen=True)
class Cue:
    kind: Literal["semantic", "entity_name", "descriptor", "entity_id"]
    value: str | EntityId
    confidence: float = 1.0


@dataclass(frozen=True)
class RetrievalContext:
    chat_id: int | None
    topic_id: int | None
    venue: str = "telegram"
    current_namespace: str | None = None
    current_timestamp: datetime | None = None
    conversation_entities: list[EntityId] = field(default_factory=list)
    conversation_mood: str | None = None
    limit: int = 8


@dataclass(frozen=True)
class TracePrime:
    """One trace surfaced by priming, with scoring breakdown."""

    trace_id: TraceId
    content: str
    created_at: datetime
    memory_type: MemoryType
    score: float
    score_breakdown: dict[str, float]
    competition: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Artifact:
    """Typed consolidation artifact. Payload shape depends on artifact_type.

    payload is a dict matching the corresponding Pydantic model in
    artifact_schemas.py — consumers are expected to validate via the matching
    model before trusting fields.
    """

    id: ArtifactId
    artifact_type: ArtifactType
    schema_version: int
    producer_name: str
    producer_version: int
    subject_entity_id: EntityId | None
    payload: dict
    content_hash: str
    status: Literal["active", "superseded", "expired", "permanent"]
    supersedes_id: ArtifactId | None
    created_at: datetime
    expires_at: datetime | None


@dataclass(frozen=True)
class RetrievalResult:
    primed_traces: list[TracePrime]
    relevant_artifacts: list[Artifact]
    total_candidates: int
    retrieval_latency_ms: int


@dataclass(frozen=True)
class Experience:
    """An encoded experience. One or more Representations.

    `id` is optional on encode() (assigned by storage layer if None). If
    provided (e.g., when re-encoding after feedback), the storage layer uses it
    as the stable identity.
    """

    chat_id: int | None
    topic_id: int | None
    venue: str
    timestamp: datetime
    memory_type: MemoryType
    content: str
    entity_links: list[tuple[EntityId, LinkRole]]
    significance: int
    id: TraceId | None = None
    representations: list[Representation] = field(default_factory=list)
    signals: EncodingSignals = field(default_factory=EncodingSignals)
    emotion_valence: float | None = None
    emotion_arousal: float | None = None
    source_trace_ids: list[TraceId] = field(default_factory=list)


@dataclass(frozen=True)
class EncodingResult:
    trace_ids: list[TraceId]
    entities_touched: list[EntityId]
    contradictions_detected: int
    mode: Literal["defer", "eager", "deep"]


@dataclass(frozen=True)
class ArtifactQuery:
    """For export_artifacts() — used by future LoRA/adaptation consumers."""

    artifact_type: ArtifactType | None = None
    schema_version_min: int | None = None
    produced_after: datetime | None = None
    subject_entity_id: EntityId | None = None
    # Confidence ordering is per-artifact-type. For types without a
    # `confidence` payload field (IdentityRecord, EntitySchema) this filter is
    # a no-op; consumers checking confidence should inspect the payload.
    min_confidence: float | None = None
    status: Literal["active", "superseded", "expired", "permanent", "any"] = "active"


@dataclass(frozen=True)
class ConsolidationTrigger:
    level: Literal[2, 3]
    venue: str
    chat_id: int | None
    topic_id: int | None
    reason: Literal["silence_timer", "eager", "deep_eager", "nightly"]
    since_message_id: int | None = None


@dataclass(frozen=True)
class ConsolidationReport:
    run_id: ConsolidationRunId
    level: Literal[2, 3]
    traces_created: int
    traces_invalidated: int
    artifacts_produced: dict[ArtifactType, int]
    duration_ms: int
    failures: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FeedbackTarget:
    kind: Literal["trace", "artifact"]
    id: TraceId | ArtifactId


@dataclass(frozen=True)
class FeedbackSignal:
    kind: Literal["affirmation", "weak_contradiction", "strong_correction", "factual_refinement"]
    evidence_text: str
    source_message_id: int | None
    source_trace_id: TraceId | None


@dataclass(frozen=True)
class FeedbackResult:
    action: Literal["noop", "demote", "invalidate", "rewrite"]
    inhibition_id: uuid.UUID | None
    new_trace_id: TraceId | None
    invalidated_artifact_ids: list[ArtifactId]


@runtime_checkable
class MemoryInterface(Protocol):
    """Substrate-agnostic contract for memory access.

    When both Experience.signals and the signals parameter are provided, the
    explicit signals parameter wins (used by encoders that compute signals
    separately from the experience construction).
    """

    async def retrieve(self, cues: list[Cue], context: RetrievalContext) -> RetrievalResult: ...

    async def encode(
        self,
        experience: Experience,
        signals: EncodingSignals | None = None,
    ) -> EncodingResult: ...

    async def consolidate(self, trigger: ConsolidationTrigger) -> ConsolidationReport: ...

    async def feedback(self, target: FeedbackTarget, signal: FeedbackSignal) -> FeedbackResult: ...

    async def export_artifacts(self, query: ArtifactQuery) -> AsyncIterator[Artifact]: ...
