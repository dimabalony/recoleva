from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from kai_memory.storage.models.memory_entity import MemoryEntity
from kai_memory.store.cognitive_memory import CognitiveMemoryService

pytestmark = pytest.mark.asyncio


def _entity(
    name: str,
    entity_type: str = "person",
    merged_into_id: uuid.UUID | None = None,
    telegram_user_id: int | None = None,
) -> MemoryEntity:
    return MemoryEntity(
        id=uuid.uuid4(),
        name=name,
        entity_type=entity_type,
        aliases=[],
        merged_into_id=merged_into_id,
        telegram_user_id=telegram_user_id,
    )


def _rows_result(rows: list[MemoryEntity]) -> MagicMock:
    result = MagicMock()
    result.scalars.return_value.all.return_value = rows
    return result


def _session_for_entities(entities: list[MemoryEntity]) -> AsyncMock:
    session = AsyncMock()

    async def execute(stmt):
        compiled = stmt.compile(dialect=postgresql.dialect())
        sql = str(compiled)
        params = compiled.params

        names = params.get("name_1", [])
        rows = [e for e in entities if e.name in names]

        if "memory_entities.entity_type =" in sql:
            rows = [e for e in rows if e.entity_type == params["entity_type_1"]]
        if "memory_entities.merged_into_id IS NULL" in sql:
            rows = [e for e in rows if e.merged_into_id is None]
        if "memory_entities.telegram_user_id IS NULL" in sql:
            rows = [e for e in rows if e.telegram_user_id is None]

        return _rows_result(rows)

    session.execute = AsyncMock(side_effect=execute)
    return session


async def test_find_entities_by_names_filters_by_entity_type():
    person = _entity("Python", entity_type="person")
    topic = _entity("Python", entity_type="topic")
    session = _session_for_entities([person, topic])

    svc = CognitiveMemoryService(session)
    results = await svc.find_entities_by_names(["Python"], entity_type="person")

    assert results == [person]


async def test_find_entities_by_names_excludes_merged_entities():
    active = _entity("Elena")
    merged = _entity("Natasha", merged_into_id=active.id)
    session = _session_for_entities([active, merged])

    svc = CognitiveMemoryService(session)
    results = await svc.find_entities_by_names(["Elena", "Natasha"])

    assert results == [active]


async def test_find_entities_by_names_without_filters_returns_both_entity_types():
    person = _entity("Python", entity_type="person")
    topic = _entity("Python", entity_type="topic")
    session = _session_for_entities([person, topic])

    svc = CognitiveMemoryService(session)
    results = await svc.find_entities_by_names(["Python"])

    assert results == [person, topic]


async def test_only_ungrounded_excludes_tg_grounded():
    grounded = _entity("Pavel", telegram_user_id=100)
    ungrounded = _entity("Pavel")
    session = _session_for_entities([grounded, ungrounded])

    svc = CognitiveMemoryService(session)
    results = await svc.find_entities_by_names(["Pavel"], only_ungrounded=True)

    assert results == [ungrounded]
