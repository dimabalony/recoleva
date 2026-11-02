from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from kai_memory.storage.models.memory_entity import MemoryEntity
from kai_memory.store.cognitive_memory import CognitiveMemoryService

pytestmark = pytest.mark.asyncio


def _rows_result(rows: list[MemoryEntity]) -> MagicMock:
    result = MagicMock()
    result.scalars.return_value.all.return_value = rows
    return result


async def test_traverse_entity_graph_excludes_merged_entities():
    active_id = uuid.uuid4()
    merged_id = uuid.uuid4()

    session = AsyncMock()

    async def execute(stmt, *_args, **_kwargs):
        sql = str(stmt)
        if sql.count("merged_into_id IS NULL") >= 2:
            rows = [SimpleNamespace(entity_id=active_id, energy=0.8)]
        else:
            rows = [
                SimpleNamespace(entity_id=merged_id, energy=0.9),
                SimpleNamespace(entity_id=active_id, energy=0.8),
            ]
        result = MagicMock()
        result.__iter__ = MagicMock(return_value=iter(rows))
        return result

    session.execute = AsyncMock(side_effect=execute)
    svc = CognitiveMemoryService(session)

    results = await svc.traverse_entity_graph([uuid.uuid4()])

    result_ids = {entity_id for entity_id, _energy in results}
    assert active_id in result_ids
    assert merged_id not in result_ids


async def test_find_entities_by_names_excludes_merged_entity_from_search_results():
    merged = MemoryEntity(
        id=uuid.uuid4(),
        name="Natasha_old",
        entity_type="person",
        aliases=[],
        merged_into_id=uuid.uuid4(),
    )
    session = AsyncMock()

    async def execute(stmt):
        compiled = stmt.compile(dialect=postgresql.dialect())
        sql = str(compiled)
        rows = [] if "memory_entities.merged_into_id IS NULL" in sql else [merged]
        return _rows_result(rows)

    session.execute = AsyncMock(side_effect=execute)
    svc = CognitiveMemoryService(session)

    results = await svc.find_entities_by_names(["Natasha_old"])

    assert results == []
