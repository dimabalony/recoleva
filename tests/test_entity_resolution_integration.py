from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from kai_memory.storage.models.entity_merge_candidate import EntityMergeCandidate
from kai_memory.storage.models.entity_relationship import EntityRelationship
from kai_memory.storage.models.memory_entity import MemoryEntity
from kai_memory.consolidation.consolidation import _create_relationship_from_classification


class _RelationshipSession:
    def __init__(self, relationships: list[EntityRelationship] | None = None) -> None:
        self.relationships = relationships or []
        self.added: list[object] = []

    def add(self, obj: object) -> None:
        self.added.append(obj)
        if isinstance(obj, EntityRelationship):
            self.relationships.append(obj)

    async def flush(self) -> None:
        return None

    async def execute(self, _stmt) -> object:
        now = datetime.now(UTC)
        for rel in self.relationships:
            if rel.valid_until is None:
                rel.valid_until = now
        return object()


def _entity(name: str, tg_id: int) -> MemoryEntity:
    return MemoryEntity(id=uuid.uuid4(), name=name, entity_type="person", telegram_user_id=tg_id)


def _active_partner_rels_for(entity_id: uuid.UUID, rels: list[EntityRelationship]) -> list[EntityRelationship]:
    return [
        rel
        for rel in rels
        if rel.valid_until is None
        and rel.relationship_type == "partner_of"
        and (rel.from_entity_id == entity_id or rel.to_entity_id == entity_id)
    ]


@pytest.mark.asyncio
async def test_pavel_scenario_replaces_partner_conflict():
    pavel = _entity("Pavel", 100)
    natasha = _entity("Natasha", 200)
    katya = _entity("Katya", 300)
    old_rel = EntityRelationship(
        id=uuid.uuid4(),
        from_entity_id=pavel.id,
        to_entity_id=natasha.id,
        relationship_type="partner_of",
        confidence=0.88,
    )
    session = _RelationshipSession([old_rel])

    pair = {
        "a_id": pavel.id,
        "b_id": katya.id,
        "entity_a": pavel,
        "entity_b": katya,
        "name_a": "Pavel",
        "name_b": "Katya",
        "shared_contents": ["Pavel and Katya are partners"],
        "earliest_date": datetime.now(UTC),
        "existing_rels_a": [{"type": "partner_of", "other_name": "Natasha", "confidence": 0.88, "id": str(old_rel.id)}],
        "existing_rels_b": [],
    }
    classification = {
        "relationship_type": "partner_of",
        "confidence": 0.99,
        "conflict_action": "replace",
        "conflicts_with": [str(old_rel.id)],
    }

    await _create_relationship_from_classification(session, pair, classification)  # type: ignore[arg-type]

    assert old_rel.valid_until is not None
    active_rels = _active_partner_rels_for(pavel.id, session.relationships)
    assert len(active_rels) == 1
    pair = {active_rels[0].from_entity_id, active_rels[0].to_entity_id}
    assert pair == {pavel.id, katya.id}
    assert active_rels[0].confidence == 0.99


@pytest.mark.asyncio
async def test_same_person_conflict_creates_merge_candidate_without_relationship():
    pavel = _entity("Pavel", 100)
    pavel_alias = _entity("Pavel B", 101)
    session = _RelationshipSession()
    pair = {
        "a_id": pavel.id,
        "b_id": pavel_alias.id,
        "entity_a": pavel,
        "entity_b": pavel_alias,
        "name_a": "Pavel",
        "name_b": "Pavel B",
        "shared_contents": ["Pavel B is the same Pavel"],
        "earliest_date": datetime.now(UTC),
        "source_consolidation_run_id": uuid.uuid4(),
    }

    await _create_relationship_from_classification(
        session,  # type: ignore[arg-type]
        pair,
        {"relationship_type": "same_person", "confidence": 0.93, "conflict_action": "same_person"},
    )

    assert not any(isinstance(obj, EntityRelationship) for obj in session.added)
    candidates = [obj for obj in session.added if isinstance(obj, EntityMergeCandidate)]
    assert len(candidates) == 1
    candidate_pair = {candidates[0].entity_a_id, candidates[0].entity_b_id}
    assert candidate_pair == {pavel.id, pavel_alias.id}
    assert candidates[0].status == "pending"
