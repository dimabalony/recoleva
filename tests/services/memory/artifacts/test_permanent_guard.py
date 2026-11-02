"""BaseProducer.supersede() refuses to mutate a permanent prev artifact.

The DB trigger ``prevent_permanent_artifact_modification`` (migration
068) is the ground truth. This test guards the application-layer
fail-fast that gives a clearer error than the trigger's RAISE EXCEPTION,
*and* prevents the SQL from being issued at all (no wasted savepoint
roundtrip on a doomed write).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from kai_memory.artifacts.producer import BaseProducer, ProducerSubject
from kai_memory.artifacts.types import (
    Artifact,
    ChangesMadeEntry,
    ProducedArtifact,
)


class _DummyProducer(BaseProducer):
    """Minimal subclass so we can exercise the inherited ``supersede``."""

    name = "dummy"
    version = 1
    artifact_type = "Anchor"

    async def discover_subjects(self, session, run_id):  # pragma: no cover
        return []

    async def produce(
        self, session, subject: ProducerSubject, run_id: uuid.UUID
    ) -> ProducedArtifact | None:  # pragma: no cover
        return None


def _permanent_artifact() -> Artifact:
    return Artifact(
        id=uuid.uuid4(),
        artifact_type="Anchor",
        schema_version=1,
        producer_name="bootstrap",
        producer_version=0,
        subject_entity_id=None,
        payload={"content": "x" * 200},
        content_hash="abc",
        status="permanent",
        supersedes_id=None,
        created_at=datetime.now(UTC),
        expires_at=None,
    )


def _attempted_replacement() -> ProducedArtifact:
    return ProducedArtifact(
        artifact_type="Anchor",
        schema_version=1,
        subject_entity_id=None,
        dimension=None,
        payload={"content": "y" * 200},
        human_summary="attempt to rewrite a permanent artifact",
    )


@pytest.mark.asyncio
async def test_supersede_refuses_permanent_prev() -> None:
    producer = _DummyProducer()
    with pytest.raises(PermissionError, match="permanent"):
        # session=None is fine — the guard fires before any SQL is touched.
        await producer.supersede(
            session=None,  # type: ignore[arg-type]
            prev=_permanent_artifact(),
            new=_attempted_replacement(),
            run_id=uuid.uuid4(),
        )


@pytest.mark.asyncio
async def test_supersede_error_names_the_artifact_id() -> None:
    """Error message includes prev.id + artifact_type so logs are actionable."""
    producer = _DummyProducer()
    prev = _permanent_artifact()
    with pytest.raises(PermissionError) as excinfo:
        await producer.supersede(
            session=None,  # type: ignore[arg-type]
            prev=prev,
            new=_attempted_replacement(),
            run_id=uuid.uuid4(),
        )
    msg = str(excinfo.value)
    assert str(prev.id) in msg
    assert "Anchor" in msg


@pytest.mark.asyncio
async def test_supersede_with_active_prev_does_not_raise(monkeypatch) -> None:
    """Sanity: the guard ONLY fires for status='permanent', not status='active'.

    Stubs the SQL execute to record calls so we can verify the normal
    supersede path was reached (UPDATE prev + INSERT new).
    """
    producer = _DummyProducer()
    prev = Artifact(
        id=uuid.uuid4(),
        artifact_type="Anchor",
        schema_version=1,
        producer_name="dummy",
        producer_version=1,
        subject_entity_id=None,
        payload={"content": "x"},
        content_hash="abc",
        status="active",
        supersedes_id=None,
        created_at=datetime.now(UTC),
        expires_at=None,
    )

    class _StubResult:
        def scalars(self):
            class _S:
                def all(self_inner):
                    return [prev.id]

            return _S()

    class _StubSession:
        def __init__(self):
            self.calls: list[str] = []

        async def execute(self, stmt):
            stmt_str = str(stmt)
            self.calls.append(stmt_str)
            return _StubResult()

    session = _StubSession()
    entry = await producer.supersede(
        session=session,  # type: ignore[arg-type]
        prev=prev,
        new=_attempted_replacement(),
        run_id=uuid.uuid4(),
    )
    assert isinstance(entry, ChangesMadeEntry)
    assert entry.action == "supersede"
    # UPDATE prev + INSERT new = 2 SQL calls.
    assert len(session.calls) == 2
