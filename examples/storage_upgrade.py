"""Disposable-only M1->M2 upgrade, injected failure, retry and rollback rehearsal.

This is an SDK-internal proof harness shipped with its exact source revision,
not a supported host migration API; the private shape helper is version-coupled.
No production migration API. The caller supplies a *fresh* disposable PostgreSQL
URL. bootstrap_disposable refuses any preexisting user object before any write.
The actual host migration is Alembic 111 (concurrent index creation); this
transactional rehearsal proves schema/data compatibility, not production locking
or concurrent-index cancellation recovery. No model inference is performed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import hashlib
import os
import uuid
from pathlib import Path

from sqlalchemy import MetaData, inspect, text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from kai_memory.storage.disposable import check_url_is_local

from kai_memory.storage import bootstrap_disposable, compose_metadata
from kai_memory.storage.compatibility import (
    CURRENT_SCHEMA_REVISION,
    M1_SCHEMA_REVISION,
    REPLAY_INDEX,
    _shape,
    inspect_storage,
    require_compatible_storage,
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _json_shape(value):
    if isinstance(value, dict):
        return {k: _json_shape(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json_shape(v) for v in value]
    if isinstance(value, set):
        return sorted([_json_shape(v) for v in value], key=str)
    return value


def historical_schema(dialect):
    """Derive M1 only after checking independently captured historical metadata.

    Fixture captured by executing all storage modules at DB-864 c08ca8fb, not by
    subtracting an index from today's metadata and calling the result historical.
    """
    baseline = json.loads(Path(__file__).with_name("storage_m1_schema.json").read_text())
    schema = compose_metadata(MetaData())
    replay = next(i for i in schema.tables["execution_traces"].indexes if i.name == REPLAY_INDEX)
    schema.tables["execution_traces"].indexes.remove(replay)
    actual = {n: _json_shape(_shape(t, dialect)) for n, t in schema.tables.items()}
    if actual != baseline["tables"]:
        raise AssertionError("Current metadata minus replay index differs from captured historical M1 schema")
    return schema, baseline["source_commit"]


async def rehearse(url: str, *, disposable: bool = False) -> dict:
    if disposable is not True:
        raise ValueError("Explicit disposable=True acknowledgment is required")
    check_url_is_local(url, production_url=os.environ.get("DATABASE_URL"))
    engine = create_async_engine(url)
    try:
        old, baseline = historical_schema(engine.dialect)
        # This is the only entry point to mutation. It refuses nonempty storage.
        await bootstrap_disposable(engine, schema=old)
        from kai_memory.storage import metadata
        from kai_memory.storage.models.memory_entity import MemoryEntity
        from kai_memory.storage.models.memory_trace import MemoryTrace
        from kai_memory.storage.models.memory_artifact import MemoryArtifact
        from kai_memory.storage.models.user_profile import UserProfile
        from kai_memory.storage.models.cognitive_config import CognitiveConfig
        from kai_memory.storage.models.execution_trace import ExecutionTrace

        async with async_sessionmaker(engine)() as session:
            entity = MemoryEntity(entity_type="agent", name="Synthetic upgrade agent")
            session.add(entity)
            await session.flush()
            session.add_all(
                [
                    MemoryTrace(
                        content="Synthetic upgrade fact",
                        memory_type="fact",
                        source_type="chat",
                        embedding=[0.01] * 1536,
                    ),
                    MemoryArtifact(
                        artifact_type="IdentityRecord",
                        producer_name="synthetic-upgrade",
                        subject_entity_id=entity.id,
                        payload={"fixture": True},
                        content_hash="a" * 64,
                    ),
                    UserProfile(
                        id=uuid.uuid4(),
                        user_id="synthetic-upgrade",
                        category="identity",
                        key="name",
                        value={"name": "Synthetic"},
                    ),
                    CognitiveConfig(key="synthetic_upgrade", value=0.5, category="fixture"),
                    ExecutionTrace(
                        task_signature="synthetic upgrade sentinel",
                        outcome="unknown",
                        cost_tokens=0,
                        success_signals={"assistant_message_id": 880},
                    ),
                ]
            )
            await session.commit()

        async def fingerprint():
            result = {}
            async with engine.connect() as conn:
                for name in sorted(metadata.tables):
                    # Table identifiers come from SDK metadata, never external input.
                    count, content = (
                        await conn.execute(
                            text(
                                f"SELECT count(*), coalesce(string_agg(t::text, '|' ORDER BY t::text), '') "
                                f'FROM public."{name}" AS t'
                            )
                        )
                    ).one()
                    result[name] = (count, hashlib.sha256(content.encode()).hexdigest())
            return result

        original_data = await fingerprint()
        before = await inspect_storage(engine)
        _require(before.revision == M1_SCHEMA_REVISION and not before.compatible, before)
        index = next(i for i in metadata.tables["execution_traces"].indexes if i.name == REPLAY_INDEX)

        async def mutate(*, rollback=False, fail=False):
            async with engine.begin() as conn:
                await conn.execute(text("SELECT pg_advisory_xact_lock(862741)"))
                names = await conn.run_sync(lambda c: set(inspect(c).get_table_names(schema="public")))
                _require(names == set(metadata.tables), "Rehearsal refuses host-owned tables")
                report = await inspect_storage(conn)
                _require(report.revision == (CURRENT_SCHEMA_REVISION if rollback else M1_SCHEMA_REVISION), report)
                await conn.execute(text("SET LOCAL search_path TO public"))
                await conn.run_sync(lambda c: index.drop(c) if rollback else index.create(c))
                if fail:
                    raise RuntimeError("injected disposable upgrade failure")

        try:
            await mutate(fail=True)
        except RuntimeError as exc:
            _require(str(exc) == "injected disposable upgrade failure", "Storage rehearsal invariant failed")
        else:
            raise AssertionError("Failure injection did not run")
        failed = await inspect_storage(engine)
        _require(failed == before, failed)
        _require(await fingerprint() == original_data, "Failed upgrade changed rows")
        await mutate()
        upgraded = await require_compatible_storage(engine)
        _require(await fingerprint() == original_data, "Upgrade changed rows")
        await mutate(rollback=True)
        rolled_back = await inspect_storage(engine)
        _require(rolled_back == before, rolled_back)
        _require(await fingerprint() == original_data, "Rollback changed rows")
        await mutate()
        await require_compatible_storage(engine)
        _require(await fingerprint() == original_data, "Retry changed rows")
        async with engine.connect() as conn:
            rows = (await conn.execute(text("SELECT task_signature, success_signals FROM execution_traces"))).all()
        _require(rows == [("synthetic upgrade sentinel", {"assistant_message_id": 880})], rows)
        return {
            "historical_source_commit": baseline,
            "old_revision": before.revision,
            "new_revision": upgraded.revision,
            "failed_upgrade_rolled_back": failed == before,
            "downgrade_restored_old_schema": rolled_back == before,
            "retry_compatible": True,
            "sentinel_retained": True,
            "all_table_checksums_retained": True,
            "seeded_tables": sorted(name for name, (count, _) in original_data.items() if count),
            "production_migration_qualified": False,
        }
    finally:
        await engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--disposable", action="store_true", required=True)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = asyncio.run(rehearse(args.database_url, disposable=args.disposable))
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
