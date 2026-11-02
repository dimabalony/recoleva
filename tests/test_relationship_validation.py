from __future__ import annotations

import logging
import uuid

import pytest

from kai_memory.storage.models.memory_entity import MemoryEntity
from kai_memory.consolidation import consolidation


def _entity(entity_id: uuid.UUID, name: str, entity_type: str) -> MemoryEntity:
    return MemoryEntity(id=entity_id, name=name, entity_type=entity_type)


def _validate(from_entity: MemoryEntity, to_entity: MemoryEntity, predicate: str):
    validator = getattr(consolidation, "_validate_relationship_classification", None)
    assert validator is not None, "relationship validator must exist"
    return validator(from_entity, to_entity, predicate, 0.84)


def test_relationship_validator_accepts_allowed_direction() -> None:
    bali = _entity(uuid.UUID("00000000-0000-0000-0000-000000000001"), "Bali", "place")
    indonesia = _entity(uuid.UUID("00000000-0000-0000-0000-000000000002"), "Indonesia", "place")

    result = _validate(bali, indonesia, "located_in")

    assert result.ok is True
    assert result.corrected_from_id == bali.id
    assert result.corrected_to_id == indonesia.id
    assert result.predicate == "located_in"


def test_relationship_validator_corrects_inverted_direction() -> None:
    bali = _entity(uuid.UUID("00000000-0000-0000-0000-000000000001"), "Bali", "place")
    indonesia = _entity(uuid.UUID("00000000-0000-0000-0000-000000000002"), "Indonesia", "place")

    result = _validate(indonesia, bali, "located_in")

    assert result.ok is True
    assert result.corrected_from_id == bali.id
    assert result.corrected_to_id == indonesia.id
    assert "swapped" in result.reason


def test_relationship_validator_rejects_unknown_predicate(caplog) -> None:
    olga = _entity(uuid.uuid4(), "Olga", "person")
    clouds = _entity(uuid.uuid4(), "облачками", "topic")

    with caplog.at_level(logging.WARNING):
        result = _validate(clouds, olga, "reminds_about")

    assert result.ok is False
    assert "unknown predicate" in result.reason
    assert "reminds_about" in caplog.text


def test_relationship_validator_rejects_type_mismatch_after_swap() -> None:
    bali = _entity(uuid.uuid4(), "Bali", "place")
    indonesia = _entity(uuid.uuid4(), "Indonesia", "place")

    result = _validate(bali, indonesia, "works_at")

    assert result.ok is False
    assert "not allowed" in result.reason


def test_relationship_validator_orders_symmetric_predicate_by_uuid() -> None:
    higher = _entity(uuid.UUID("00000000-0000-0000-0000-0000000000ff"), "Olga", "person")
    lower = _entity(uuid.UUID("00000000-0000-0000-0000-000000000001"), "Misha", "person")

    result = _validate(higher, lower, "friend_of")

    assert result.ok is True
    assert result.corrected_from_id == lower.id
    assert result.corrected_to_id == higher.id
    assert "canonical" in result.reason


@pytest.mark.parametrize(
    ("predicate", "from_type", "to_type"),
    [
        ("user_of", "person", "organization"),
        ("user_of", "agent", "tool"),
        ("user_of", "organization", "project"),
        ("resident_of", "person", "place"),
        ("customer_of", "person", "organization"),
        ("alternative_to", "object", "object"),
        ("employee_of", "person", "organization"),
        ("patient_of", "person", "organization"),
        ("competitor_of", "organization", "organization"),
        ("practitioner_of", "person", "topic"),
        ("creator_of", "person", "project"),
        ("prospective_buyer_of", "person", "object"),
        ("assistant_of", "agent", "person"),
        ("business_registered_in", "organization", "place"),
        ("business_registered_in", "person", "place"),
        ("cardholder_of", "person", "object"),
        ("companion_of", "animal", "pet"),
        ("complement_of", "topic", "topic"),
        ("leader_of", "person", "project"),
        ("patient_with", "person", "topic"),
        ("practitioner_at", "person", "organization"),
        ("prospective_customer_of", "person", "organization"),
        ("renter_of", "person", "object"),
        ("traveling_to", "person", "place"),
        ("used_in", "object", "place"),
        ("visits_regularly", "person", "place"),
        ("hometown_of", "place", "person"),
        ("member_of", "person", "project"),
        ("member_of", "agent", "project"),
        ("owner_of", "person", "group"),
    ],
)
def test_relationship_validator_accepts_approved_predicate_rules(
    predicate: str,
    from_type: str,
    to_type: str,
) -> None:
    from_entity = _entity(uuid.UUID("00000000-0000-0000-0000-000000000011"), "from", from_type)
    to_entity = _entity(uuid.UUID("00000000-0000-0000-0000-000000000022"), "to", to_type)

    result = _validate(from_entity, to_entity, predicate)

    assert result.ok is True
    assert result.corrected_from_id == from_entity.id
    assert result.corrected_to_id == to_entity.id


@pytest.mark.parametrize(
    ("predicate", "from_type", "to_type"),
    [
        ("customer_of", "organization", "person"),
        ("employee_of", "organization", "person"),
        ("hometown_of", "person", "place"),
        ("owner_of", "group", "person"),
        ("practitioner_at", "organization", "person"),
        ("prospective_customer_of", "organization", "person"),
        ("renter_of", "object", "person"),
        ("resident_of", "place", "person"),
    ],
)
def test_relationship_validator_swaps_approved_directional_inversions(
    predicate: str,
    from_type: str,
    to_type: str,
) -> None:
    from_entity = _entity(uuid.UUID("00000000-0000-0000-0000-000000000011"), "from", from_type)
    to_entity = _entity(uuid.UUID("00000000-0000-0000-0000-000000000022"), "to", to_type)

    result = _validate(from_entity, to_entity, predicate)

    assert result.ok is True
    assert result.corrected_from_id == to_entity.id
    assert result.corrected_to_id == from_entity.id
    assert "swapped" in result.reason


@pytest.mark.parametrize("predicate", ["leads", "managed_by"])
def test_relationship_validator_keeps_unapproved_synonyms_rejected(predicate: str) -> None:
    person = _entity(uuid.uuid4(), "person", "person")
    project = _entity(uuid.uuid4(), "project", "project")

    result = _validate(person, project, predicate)

    assert result.ok is False
    assert "unknown predicate" in result.reason


def test_relationship_validator_rejects_employee_of_project_target() -> None:
    person = _entity(uuid.uuid4(), "person", "person")
    project = _entity(uuid.uuid4(), "project", "project")

    result = _validate(person, project, "employee_of")

    assert result.ok is False
    assert "not allowed" in result.reason
