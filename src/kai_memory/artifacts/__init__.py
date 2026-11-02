"""Artifact pipeline package — M2 Wave 1 foundation.

Public surface for producers (Wave 2) and consumers (Wave 3):
- ``ArtifactProducer`` Protocol + ``BaseProducer`` ABC (`producer.py`)
- ``ProducedArtifact`` / ``ChangesMadeEntry`` and the existing payload models
  (`types.py`)

The orchestrator lives one level up at
``kai_memory.artifact_pipeline``.
"""

from __future__ import annotations

from kai_memory.artifacts.producer import (
    ArtifactProducer,
    BaseProducer,
    ProducerSubject,
)
from kai_memory.artifacts.types import (
    Artifact,
    ArtifactStatus,
    ArtifactType,
    ChangesMadeAction,
    ChangesMadeEntry,
    ContrastiveMapPayload,
    CueMappingPayload,
    EntitySchemaPayload,
    IdentityRecordPayload,
    ProducedArtifact,
    RelationshipRef,
    TrajectoryEvent,
    TrajectoryPayload,
)

__all__ = [
    "Artifact",
    "ArtifactProducer",
    "ArtifactStatus",
    "ArtifactType",
    "BaseProducer",
    "ChangesMadeAction",
    "ChangesMadeEntry",
    "ContrastiveMapPayload",
    "CueMappingPayload",
    "EntitySchemaPayload",
    "IdentityRecordPayload",
    "ProducedArtifact",
    "ProducerSubject",
    "RelationshipRef",
    "TrajectoryEvent",
    "TrajectoryPayload",
]
