"""Artifact export pages storage without borrowing sessions across consumer yields."""

from contextlib import suppress
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from kai_memory.ownership import current_owner
from kai_memory.sdk import Chat, HostAdapters, MemoryClient
from kai_memory.types import ArtifactQuery


@pytest.fixture
async def export_host():
    state = SimpleNamespace(rows=[], queries=[], open_sessions=0, page_lengths=[], revoke_on_close=False)
    policy = SimpleNamespace(read=True)
    policy.may_read = lambda _: policy.read
    client = None

    class Session:
        async def execute(self, stmt):
            assert current_owner() is client
            params = stmt.compile().params
            assert "LIMIT" in str(stmt) and "OFFSET" not in str(stmt)
            limit = next(value for key, value in params.items() if key.startswith("param") and type(value) is int)
            assert limit == 200
            cursor_dates = [
                value for key, value in params.items() if key.startswith("param") and isinstance(value, datetime)
            ]
            cursor_ids = [value for key, value in params.items() if key.startswith("param") and isinstance(value, UUID)]
            rows = sorted(state.rows, key=lambda row: (row.created_at, row.id))
            if cursor_ids:
                assert "(memory_artifacts.created_at, memory_artifacts.id) >" in str(stmt)
                cursor = cursor_dates[-1], cursor_ids[-1]
                rows = [row for row in rows if (row.created_at, row.id) > cursor]
            page = rows[:limit]
            state.queries.append(stmt)
            state.page_lengths.append(len(page))
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: page))

        async def commit(self):
            pass

        async def rollback(self):
            pass

        async def close(self):
            state.open_sessions -= 1
            if state.revoke_on_close:
                policy.read = False

    def sessions():
        state.open_sessions += 1
        return Session()

    client = MemoryClient(
        HostAdapters(
            inference=SimpleNamespace(complete=AsyncMock()),
            embedder=SimpleNamespace(embed=AsyncMock()),
            history=SimpleNamespace(),
            session_factory=sessions,
            policy=policy,
            overrides={
                "scrub_secrets": lambda text: text,
                "validate_skill_content": lambda _: [],
                "redact_urls_for_trace": lambda _: None,
            },
        )
    )
    yield client, state, policy
    await client.close()


def row(number, confidence=1.0):
    return SimpleNamespace(
        id=UUID(int=number),
        created_at=datetime(2026, 10, 24, tzinfo=UTC),
        artifact_type="EntitySchema",
        schema_version=1,
        producer_name="synthetic",
        producer_version=1,
        subject_entity_id=None,
        payload={"confidence": confidence, "synthetic": number},
        content_hash=str(number),
        status="active",
        supersedes_id=None,
        expires_at=None,
    )


async def test_export_more_than_one_page_uses_timestamp_uuid_keyset(export_host):
    client, state, _ = export_host
    # Every timestamp ties; UUID order must break ties without skips or duplicates.
    state.rows = [row(number) for number in reversed(range(1, 406))]
    results = []
    async for artifact in client.export_artifacts(Chat(chat_id=1)):
        assert state.open_sessions == 0
        results.append(artifact)
    assert [artifact.id.int for artifact in results] == list(range(1, 406))
    assert state.page_lengths == [200, 200, 5]


async def test_export_full_filtered_page_advances_raw_cursor(export_host):
    client, state, _ = export_host
    state.rows = [row(number, confidence=0.1 if number <= 200 else 0.9) for number in range(1, 204)]
    results = [artifact async for artifact in client.export_artifacts(Chat(), ArtifactQuery(min_confidence=0.8))]
    assert [artifact.id.int for artifact in results] == [201, 202, 203]
    assert state.page_lengths == [200, 3]
    assert state.open_sessions == 0


async def test_early_consumer_exit_releases_session_and_owner_operation(export_host):
    client, state, _ = export_host
    state.rows = [row(number) for number in range(1, 406)]
    stream = client.export_artifacts(Chat())
    assert (await anext(stream)).id.int == 1
    assert state.open_sessions == 0 and len(state.queries) == 1
    assert await client.close(timeout=0.01)
    assert client.state == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        await anext(stream)
    await stream.aclose()
    assert len(state.queries) == 1


async def test_revocation_stops_cached_page_delivery(export_host):
    client, state, policy = export_host
    state.rows = [row(1), row(2)]
    stream = client.export_artifacts(Chat())
    await anext(stream)
    policy.read = False
    with pytest.raises(PermissionError):
        await anext(stream)
    assert state.open_sessions == 0 and len(state.queries) == 1


async def test_revocation_stops_next_query_after_full_filtered_page(export_host):
    client, state, _ = export_host
    state.rows = [row(number, 0.1) for number in range(1, 402)]
    state.revoke_on_close = True
    stream = client.export_artifacts(Chat(), ArtifactQuery(min_confidence=0.8))
    with pytest.raises(PermissionError):
        await anext(stream)
    assert len(state.queries) == 1 and state.open_sessions == 0
    with suppress(Exception):
        await stream.aclose()


async def test_export_retains_all_query_constraints_on_every_page(export_host):
    client, state, _ = export_host
    state.rows = [row(number) for number in range(1, 202)]
    subject = UUID(int=700)
    since = datetime(2026, 1, 1, tzinfo=UTC)
    query = ArtifactQuery(
        artifact_type="EntitySchema",
        schema_version_min=1,
        produced_after=since,
        subject_entity_id=subject,
        status="permanent",
    )
    assert len([artifact async for artifact in client.export_artifacts(Chat(), query)]) == 201
    assert len(state.queries) == 2
    for stmt in state.queries:
        compiled = stmt.compile()
        sql = str(compiled)
        assert "memory_artifacts.artifact_type =" in sql
        assert "memory_artifacts.schema_version >=" in sql
        assert "memory_artifacts.created_at >" in sql
        assert "memory_artifacts.subject_entity_id =" in sql
        assert "memory_artifacts.status =" in sql
        assert {"EntitySchema", "permanent", subject, since}.issubset(
            value for value in compiled.params.values() if not isinstance(value, list)
        )
