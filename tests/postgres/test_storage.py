"""Real PostgreSQL proofs for disposable storage ownership and atomicity."""

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import Column, Integer, MetaData, Table, event, text
from sqlalchemy.ext.asyncio import create_async_engine

from kai_memory.storage import UnsafeStorageError, bootstrap_disposable, compose_metadata, metadata

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest_asyncio.fixture(loop_scope="session")
async def fresh_engine(sdk_empty_db_url):
    """Each case borrows an engine over its own empty SDK test database."""
    engine = create_async_engine(sdk_empty_db_url)
    try:
        yield engine
    finally:
        await engine.dispose()


async def _inventory(engine):
    """Snapshot names and definitions, including functions the hook could overwrite."""
    async with engine.connect() as conn:
        return (
            await conn.execute(
                text("""
            SELECT 'relation', n.nspname, c.relname, c.relkind::text
            FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE n.nspname NOT LIKE 'pg_%' AND n.nspname <> 'information_schema'
            UNION ALL
            SELECT 'routine', n.nspname, p.proname, pg_get_functiondef(p.oid)
            FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
            WHERE n.nspname NOT LIKE 'pg_%' AND n.nspname <> 'information_schema'
            UNION ALL
            SELECT 'type', n.nspname, t.typname, t.typtype::text
            FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace
            WHERE n.nspname NOT LIKE 'pg_%' AND n.nspname <> 'information_schema'
            UNION ALL
            SELECT 'schema', n.nspname, '', '' FROM pg_namespace n
            WHERE n.nspname NOT LIKE 'pg_%' AND n.nspname <> 'information_schema'
            ORDER BY 1, 2, 3, 4
        """)
            )
        ).all()


@pytest.mark.parametrize(
    "existing",
    [
        "CREATE TABLE occupied (id integer)",
        "CREATE VIEW occupied AS SELECT 1 AS id",
        "CREATE SEQUENCE occupied",
        "CREATE FUNCTION aliases_lower(jsonb) RETURNS jsonb LANGUAGE sql AS 'SELECT $1'",
        "CREATE PROCEDURE occupied() LANGUAGE sql AS 'SELECT 1'",
        "CREATE TYPE occupied AS ENUM ('reserved')",
        "CREATE SCHEMA occupied",
    ],
)
async def test_rejects_existing_objects_before_any_mutation(fresh_engine, existing):
    async with fresh_engine.begin() as conn:
        await conn.execute(text(existing))
    before = await _inventory(fresh_engine)
    with pytest.raises(UnsafeStorageError):
        await bootstrap_disposable(fresh_engine)
    assert await _inventory(fresh_engine) == before


async def _assert_aliases_and_private_upsert(engine):
    async with engine.begin() as conn:
        assert (await conn.execute(text('SELECT aliases_lower(\'["ALICE", "Bob"]\'::jsonb)'))).scalar_one() == [
            "alice",
            "bob",
        ]
        assert (await conn.execute(text("SELECT aliases_lower('{}'::jsonb)"))).scalar_one() == []
        assert (await conn.execute(text("SELECT aliases_lower(NULL::jsonb)"))).scalar_one() is None
        for label in ("before", "after"):
            await conn.execute(
                text("""
                INSERT INTO social_contexts (chat_id, context_type, context_name)
                VALUES (17, 'private', :label)
                ON CONFLICT (chat_id) WHERE topic_id IS NULL
                DO UPDATE SET context_name=EXCLUDED.context_name
            """),
                {"label": label},
            )
        assert (
            await conn.execute(text("SELECT context_name FROM social_contexts WHERE chat_id=17"))
        ).scalars().all() == ["after"]
        assert (await conn.execute(text("SELECT to_regclass('ix_memory_traces_durability_marker')"))).scalar_one()


async def test_bootstrap_accepts_extension_objects_and_borrows_engine(fresh_engine, monkeypatch):
    async with fresh_engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION vector"))
    dispose = AsyncMock()
    monkeypatch.setattr(type(fresh_engine), "dispose", dispose)
    await bootstrap_disposable(fresh_engine)
    dispose.assert_not_awaited()
    async with fresh_engine.connect() as conn:
        tables = (await conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public'"))).scalars().all()
    assert set(tables) == set(metadata.tables)
    await _assert_aliases_and_private_upsert(fresh_engine)
    # Let fixture teardown dispose the engine normally.
    monkeypatch.undo()


async def test_composed_create_all_preserves_function_hook(fresh_engine):
    composed = MetaData()
    compose_metadata(composed)
    Table("host_only", composed, Column("id", Integer, primary_key=True))
    async with fresh_engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION vector"))
        await conn.run_sync(composed.create_all)
    assert "host_only" not in metadata.tables
    await _assert_aliases_and_private_upsert(fresh_engine)


async def test_failure_rolls_back_entire_bootstrap_and_keeps_engine(fresh_engine, monkeypatch):
    composed = MetaData()
    compose_metadata(composed)
    before = await _inventory(fresh_engine)

    def fail_after_tables(*args, **kwargs):
        raise RuntimeError("forced after table creation")

    event.listen(composed, "after_create", fail_after_tables)
    dispose = AsyncMock()
    monkeypatch.setattr(type(fresh_engine), "dispose", dispose)
    with pytest.raises(RuntimeError, match="forced after table creation"):
        await bootstrap_disposable(fresh_engine, schema=composed)
    dispose.assert_not_awaited()
    assert await _inventory(fresh_engine) == before
    async with fresh_engine.connect() as conn:
        assert (await conn.execute(text("SELECT count(*) FROM pg_extension WHERE extname='vector'"))).scalar_one() == 0
    monkeypatch.undo()
