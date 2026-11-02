"""Pydantic artifact-payload validation per v2-B (M2) §3.3 producer specs.

Each artifact type has a Pydantic v2 model with `schema_version: Literal[1] = 1`.
ContrastiveMapPayload and TrajectoryPayload use forward refs and call
model_rebuild() at module load.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError


def test_imports_from_module() -> None:
    from kai_memory.artifact_schemas import (
        ContrastiveMapMember,
        ContrastiveMapPayload,
        CueMappingPayload,
        EntitySchemaPayload,
        IdentityRecordPayload,
        RelationshipRef,
        TrainingDialogPayload,
        TrajectoryEvent,
        TrajectoryPayload,
    )

    expected = {
        ContrastiveMapMember,
        ContrastiveMapPayload,
        CueMappingPayload,
        EntitySchemaPayload,
        IdentityRecordPayload,
        RelationshipRef,
        TrainingDialogPayload,
        TrajectoryEvent,
        TrajectoryPayload,
    }
    assert all(m is not None for m in expected)


def test_identity_record_payload_minimal() -> None:
    from kai_memory.artifact_schemas import IdentityRecordPayload

    payload = IdentityRecordPayload(
        subject_entity_id=uuid.uuid4(),
        dimension="communication_style",
        direction="increasing",
        magnitude=0.4,
        driving_observations=[uuid.uuid4(), uuid.uuid4()],
        epoch_start=datetime.now(UTC),
        epoch_end=None,
    )
    assert payload.schema_version == 1
    assert payload.dimension == "communication_style"
    assert payload.magnitude == 0.4


def test_identity_record_schema_version_locked() -> None:
    from kai_memory.artifact_schemas import IdentityRecordPayload

    with pytest.raises(ValidationError):
        IdentityRecordPayload(
            schema_version=2,  # type: ignore[arg-type]
            subject_entity_id=uuid.uuid4(),
            dimension="x",
            direction="stable",
            magnitude=0.0,
            driving_observations=[],
            epoch_start=datetime.now(UTC),
            epoch_end=None,
        )


def test_identity_record_direction_enum() -> None:
    from kai_memory.artifact_schemas import IdentityRecordPayload

    with pytest.raises(ValidationError):
        IdentityRecordPayload(
            subject_entity_id=uuid.uuid4(),
            dimension="x",
            direction="sideways",  # type: ignore[arg-type]
            magnitude=0.5,
            driving_observations=[],
            epoch_start=datetime.now(UTC),
            epoch_end=None,
        )


def test_identity_record_magnitude_bounded() -> None:
    """Magnitude is a float in [0.0, 1.0]; values outside the range fail."""
    from kai_memory.artifact_schemas import IdentityRecordPayload

    for bad in (-0.1, 1.5):
        with pytest.raises(ValidationError):
            IdentityRecordPayload(
                subject_entity_id=uuid.uuid4(),
                dimension="x",
                direction="stable",
                magnitude=bad,
                driving_observations=[],
                epoch_start=datetime.now(UTC),
                epoch_end=None,
            )


def test_entity_schema_payload_minimal() -> None:
    from kai_memory.artifact_schemas import EntitySchemaPayload

    payload = EntitySchemaPayload(
        canonical_name="Луна",
        entity_type="animal",
        canonical_fields={"color": "white"},
        relationships=[],
        aliases=["Luna"],
    )
    assert payload.schema_version == 1
    assert payload.canonical_name == "Луна"
    assert payload.confidence == 1.0  # default
    assert payload.last_evidence_trace_id is None


def test_entity_schema_payload_with_relationships() -> None:
    from kai_memory.artifact_schemas import EntitySchemaPayload, RelationshipRef

    target = uuid.uuid4()
    payload = EntitySchemaPayload(
        canonical_name="Луна",
        entity_type="animal",
        relationships=[
            RelationshipRef(predicate="owned_by", target_entity_id=target, confidence=0.9),
        ],
        last_evidence_trace_id=uuid.uuid4(),
        confidence=0.95,
    )
    assert payload.relationships[0].predicate == "owned_by"
    assert payload.relationships[0].target_entity_id == target


def test_contrastive_map_payload_with_forward_ref() -> None:
    from kai_memory.artifact_schemas import (
        ContrastiveMapMember,
        ContrastiveMapPayload,
    )

    member = ContrastiveMapMember(
        entity_id=uuid.uuid4(),
        distinguishing_features={"color": "white"},
    )
    payload = ContrastiveMapPayload(
        confusion_set_id=uuid.uuid4(),
        members=[member],
    )
    assert payload.schema_version == 1
    assert len(payload.members) == 1
    assert payload.members[0].distinguishing_features == {"color": "white"}


def test_cue_mapping_payload() -> None:
    from kai_memory.artifact_schemas import CueMappingPayload

    target = uuid.uuid4()
    payload = CueMappingPayload(
        cue_pattern="наша собака",
        target_schema_ids=[target],
        confidence=0.85,
        fired_count_30d=12,
    )
    assert payload.schema_version == 1
    assert payload.confidence == 0.85
    assert payload.target_schema_ids == [target]
    assert payload.fired_count_30d == 12


def test_cue_mapping_confidence_bounded() -> None:
    from kai_memory.artifact_schemas import CueMappingPayload

    with pytest.raises(ValidationError):
        CueMappingPayload(
            cue_pattern="x",
            target_schema_ids=[uuid.uuid4()],
            confidence=1.5,
        )


def test_cue_mapping_confidence_lower_bounded() -> None:
    """CueMappingPayload.confidence rejects negative values (Field ge=0.0)."""
    from kai_memory.artifact_schemas import CueMappingPayload

    with pytest.raises(ValidationError):
        CueMappingPayload(
            cue_pattern="x",
            target_schema_ids=[uuid.uuid4()],
            confidence=-0.1,
        )


def test_cue_mapping_rejects_empty_target_list() -> None:
    """target_schema_ids must have at least one element."""
    from kai_memory.artifact_schemas import CueMappingPayload

    with pytest.raises(ValidationError):
        CueMappingPayload(
            cue_pattern="x",
            target_schema_ids=[],
            confidence=0.9,
        )


def test_trajectory_payload_closed_requires_closed_at() -> None:
    """status='closed' without closed_at fails validation."""
    from kai_memory.artifact_schemas import TrajectoryEvent, TrajectoryPayload

    now = datetime.now(UTC)
    ev = TrajectoryEvent(trace_id=uuid.uuid4(), timestamp=now, summary="x")
    with pytest.raises(ValidationError):
        TrajectoryPayload(
            subject_entity_ids=[uuid.uuid4()],
            events=[ev],
            status="closed",  # missing closed_at
            opened_at=now,
            closed_at=None,
        )


def test_trajectory_payload_open_must_have_null_closed_at() -> None:
    """status='open' with closed_at set is rejected."""
    from kai_memory.artifact_schemas import TrajectoryEvent, TrajectoryPayload

    now = datetime.now(UTC)
    ev = TrajectoryEvent(trace_id=uuid.uuid4(), timestamp=now, summary="x")
    with pytest.raises(ValidationError):
        TrajectoryPayload(
            subject_entity_ids=[uuid.uuid4()],
            events=[ev],
            status="open",
            opened_at=now,
            closed_at=now,  # must be None when open
        )


def test_trajectory_payload_construction() -> None:
    from kai_memory.artifact_schemas import TrajectoryEvent, TrajectoryPayload

    now = datetime.now(UTC)
    ev = TrajectoryEvent(
        trace_id=uuid.uuid4(),
        timestamp=now,
        summary="Phishing attempt acknowledged",
    )
    payload = TrajectoryPayload(
        subject_entity_ids=[uuid.uuid4()],
        events=[ev],
        status="open",
        opened_at=now,
    )
    assert payload.schema_version == 1
    assert payload.status == "open"
    assert payload.closed_at is None


def test_training_dialog_payload_minimal() -> None:
    """TrainingDialog dropped from v2 producer scope but type spec preserved."""
    from kai_memory.artifact_schemas import TrainingDialogPayload

    payload = TrainingDialogPayload(messages=[])
    assert payload.schema_version == 1


# ── AwareDatetime guards (review-r1 fix 7 + 10) ────────────


def test_identity_record_rejects_naive_datetime():
    """v2a §7.4: AwareDatetime forbids tzinfo=None on epoch_start."""
    from datetime import datetime as _dt

    from kai_memory.artifact_schemas import IdentityRecordPayload

    with pytest.raises(ValidationError):
        IdentityRecordPayload(
            subject_entity_id=uuid.uuid4(),
            dimension="x",
            direction="stable",
            magnitude=0.0,
            driving_observations=[],
            epoch_start=_dt(2026, 5, 1, 12, 0),  # naive — must reject  # noqa: DTZ001
            epoch_end=None,
        )


def test_trajectory_event_rejects_naive_datetime():
    from datetime import datetime as _dt

    from kai_memory.artifact_schemas import TrajectoryEvent

    with pytest.raises(ValidationError):
        TrajectoryEvent(
            trace_id=uuid.uuid4(),
            timestamp=_dt(2026, 5, 1, 12, 0),  # naive — must reject  # noqa: DTZ001
            summary="x",
        )


def test_identity_record_docstring_references_migration_061():
    """review-r1 fix 10: docstring used to say "migration 052" before the
    branch was rebased onto migration 061."""
    from kai_memory.artifact_schemas import IdentityRecordPayload

    assert IdentityRecordPayload.__doc__ is not None
    assert "061" in IdentityRecordPayload.__doc__
    assert "052" not in IdentityRecordPayload.__doc__
