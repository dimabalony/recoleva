"""DB-621 — ``find_entities_by_lowercase_tokens`` is two index-eligible statements.

The old single statement OR-ed ``lower(name) IN (...)`` with an EXISTS over the
unnested alias array, which no index can serve. These unit tests pin the
statement shapes (so a "simplification" back into one OR fails here), the merge
semantics between the two halves, and the pre-106 fallback the alias half takes
when ``aliases_lower(jsonb)`` is not in the database; the plans themselves are
guarded on a real Postgres in ``tests/integration/test_db621_index_plans.py``.
"""

from __future__ import annotations

import logging
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from kai_memory.storage.models.memory_entity import MemoryEntity
from kai_memory.store.cognitive_memory import CognitiveMemoryService, reset_aliases_lower_probe

pytestmark = pytest.mark.asyncio

_SERVICE_LOGGER = "kai_memory.store.cognitive_memory"
#: Marks the compiled probe statement apart from the two entity statements.
_PROBE_MARKER = "to_regprocedure"


@pytest.fixture(autouse=True)
def _unprobed():
    """The probe result is cached for the life of the process — clear it around
    every test so the cases are independent and order-insensitive."""
    reset_aliases_lower_probe()
    yield
    reset_aliases_lower_probe()


def _entity(name: str, aliases: list[str] | None = None) -> MemoryEntity:
    return MemoryEntity(id=uuid.uuid4(), name=name, entity_type="person", aliases=aliases or [])


def _rows(rows: list[MemoryEntity]) -> MagicMock:
    result = MagicMock()
    result.scalars.return_value.all.return_value = rows
    return result


def _probe_answer(exists: bool) -> MagicMock:
    result = MagicMock()
    result.scalar.return_value = exists
    return result


def _session(by_name: list[MemoryEntity], by_alias: list[MemoryEntity], *, helper_exists: bool = True) -> AsyncMock:
    """Answers the ``aliases_lower`` probe with ``helper_exists``, the name
    statement first and the alias statement second, and keeps the compiled SQL
    of each so the tests can look at the shapes.

    ``statements`` is everything executed in order; ``probes`` and ``queries``
    are its probe and entity-statement halves. ``helper_exists`` is writable on
    the session so a test can make migration 106 land mid-process.
    """
    session = AsyncMock()
    session.statements = []
    session.probes = []
    session.queries = []
    session.helper_exists = helper_exists

    async def execute(stmt):
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        session.statements.append(sql)
        if _PROBE_MARKER in sql:
            session.probes.append(sql)
            return _probe_answer(session.helper_exists)
        session.queries.append(sql)
        return _rows(by_name if len(session.queries) == 1 else by_alias)

    session.execute = execute
    return session


async def test_two_statements_each_shaped_for_its_index():
    session = _session([], [])
    svc = CognitiveMemoryService(session)

    await svc.find_entities_by_lowercase_tokens(["Kai", "Annie"], limit=10)

    name_sql, alias_sql = session.queries
    name_where = name_sql[name_sql.index("WHERE") :]
    alias_where = alias_sql[alias_sql.index("WHERE") :]
    assert "lower(memory_entities.name) IN" in name_where
    assert "EXISTS" not in name_where and "aliases" not in name_where
    assert "aliases_lower(memory_entities.aliases) ?|" in alias_where
    assert " OR " not in alias_where and "EXISTS" not in alias_where
    for where in (name_where, alias_where):
        assert "memory_entities.merged_into_id IS NULL" in where
        assert "LIMIT" in where


async def test_short_tokens_are_dropped_and_nothing_runs_when_none_remain():
    session = _session([], [])
    svc = CognitiveMemoryService(session)

    assert await svc.find_entities_by_lowercase_tokens(["a", "of", ""], limit=10) == []
    assert session.statements == []


async def test_alias_half_is_skipped_when_names_already_fill_the_limit():
    kai, anna = _entity("Kai"), _entity("Anna")
    session = _session([kai, anna], [_entity("Never asked")])
    svc = CognitiveMemoryService(session)

    found = await svc.find_entities_by_lowercase_tokens(["kai", "anna", "hq"], limit=2)

    assert found == [kai, anna]
    # One statement total: no alias statement, and no probe for a helper the
    # lookup never needed.
    assert len(session.queries) == 1
    assert session.probes == []
    assert len(session.statements) == 1


async def test_halves_are_merged_without_duplicates_and_capped():
    kai = _entity("Kai")
    anna = _entity("Anna", ["Annie"])
    doha = _entity("Doha", ["HQ"])
    # Anna matches by name AND alias — she must appear once, and all three fit.
    session = _session([kai, anna], [anna, doha])
    found = await CognitiveMemoryService(session).find_entities_by_lowercase_tokens(
        ["kai", "anna", "annie", "headq"], limit=3
    )
    assert found == [kai, anna, doha]

    # The cap applies across both halves: one name hit plus one alias hit.
    session = _session([kai], [anna, doha])
    found = await CognitiveMemoryService(session).find_entities_by_lowercase_tokens(["kai", "annie", "headq"], limit=2)
    assert found == [kai, anna]


async def test_pre_106_schema_falls_back_to_the_unindexed_alias_statement(caplog):
    """CI's priming benchmark (and any deploy window before ``alembic upgrade``)
    runs this code against a database without migration 106. The alias half must
    answer anyway, from the old EXISTS, rather than raise UndefinedFunction."""
    session = _session([], [], helper_exists=False)
    svc = CognitiveMemoryService(session)

    with caplog.at_level(logging.WARNING, logger=_SERVICE_LOGGER):
        await svc.find_entities_by_lowercase_tokens(["Kai", "Annie"], limit=10)

        name_sql, alias_sql = session.queries
        assert "lower(memory_entities.name) IN" in name_sql[name_sql.index("WHERE") :]
        alias_where = alias_sql[alias_sql.index("WHERE") :]
        assert "jsonb_array_elements_text(aliases)" in alias_where
        assert "lower(a.v) = ANY(" in alias_where
        assert "aliases_lower" not in alias_where
        assert "memory_entities.merged_into_id IS NULL" in alias_where
        assert "LIMIT" in alias_where
        assert " OR " not in alias_where  # still two statements, not the old single OR

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "migration 106 not applied" in warnings[0]

        # A second lookup re-probes — a miss must not stick, or a process that
        # started inside the deploy window would never see 106 land — but it
        # does not warn again.
        caplog.clear()
        await svc.find_entities_by_lowercase_tokens(["Kai", "Annie"], limit=10)
        assert len(session.probes) == 2
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


async def test_the_helper_appearing_mid_process_switches_to_the_indexed_statement():
    """Migration 106 landing under a running process: the next lookup re-probes,
    finds the helper, uses the GIN statement, and stops probing from then on."""
    session = _session([], [], helper_exists=False)
    svc = CognitiveMemoryService(session)

    await svc.find_entities_by_lowercase_tokens(["Kai", "Annie"], limit=10)
    assert "jsonb_array_elements_text(aliases)" in session.queries[1]

    session.helper_exists = True
    await svc.find_entities_by_lowercase_tokens(["Kai", "Annie"], limit=10)
    assert "aliases_lower(memory_entities.aliases) ?|" in session.queries[3]

    await svc.find_entities_by_lowercase_tokens(["Kai", "Annie"], limit=10)
    assert "aliases_lower(memory_entities.aliases) ?|" in session.queries[5]
    assert len(session.probes) == 2  # the positive answer is cached; no third probe


async def test_a_positive_probe_runs_once_per_process():
    """Two lookups against a migrated database: four entity statements, one probe."""
    session = _session([], [])
    svc = CognitiveMemoryService(session)

    await svc.find_entities_by_lowercase_tokens(["Kai", "Annie"], limit=10)
    await svc.find_entities_by_lowercase_tokens(["Doha", "Anna"], limit=10)

    assert len(session.probes) == 1
    assert len(session.queries) == 4
    assert all("aliases_lower(memory_entities.aliases) ?|" in sql for sql in session.queries[1::2])
