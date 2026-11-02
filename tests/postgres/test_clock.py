"""Host clock semantics against disposable PostgreSQL and synthetic rows."""

import asyncio
import math
import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from kai_memory.storage.models.cognitive_config import CognitiveConfig

from kai_memory.storage.models.consolidation_run import ConsolidationRun
from kai_memory.storage.models.entity_relationship import EntityRelationship
from kai_memory.storage.models.memory_artifact import MemoryArtifact
from kai_memory.storage.models.memory_entity import MemoryEntity
from kai_memory.storage.models.memory_entity_link import MemoryEntityLink
from kai_memory.storage.models.memory_trace import MemoryTrace
from kai_memory.artifacts.producer import ProducerSubject
from kai_memory.artifacts.producers.entity_schema import EntitySchemaProducer
from kai_memory.artifacts.producers.pattern_producer import PatternProducer
from kai_memory.artifacts.producers.trait_producer import TraitProducer
from kai_memory.artifacts.producers.trajectory import TrajectoryProducer
from kai_memory import Chat
from kai_memory.consolidation import consolidation
from kai_memory.persona_composer import PersonaComposer
from kai_memory.sdk import HostAdapters, MemoryClient
from kai_memory.store import cognitive_config

pytestmark = pytest.mark.asyncio(loop_scope="session")
NOW = datetime(2040, 6, 2, 0, 1, tzinfo=UTC)
MIDNIGHT = NOW.replace(hour=0, minute=0)


CONFIG = {
    "base_decay": 0.1,
    "silent_salience_threshold": 0.01,
    "silent_age_days": 30,
    "durable_supersede_grace_days": 30,
    "run_retention_days": 90,
    "entity_schema_trace_window_days": 180,
    "entity_schema_trace_window_limit": 20,
    "persona_pattern_window_days": 7,
    "persona_max_active_patterns": 8,
    "entityschema_deterministic_enabled": 0,
    "entityschema_live": 0,
    "persona_trait_decay_lambda": 0.1,
    "persona_trait_strength_threshold": 0.5,
    "persona_max_active_traits": 1,
    "persona_stance_decay_lambda": 0.1,
    "persona_max_active_stances": 1,
    "trajectory_max_per_run": 10,
    "trajectory_event_min_count": 2,
}


@pytest_asyncio.fixture(loop_scope="session")
async def db_session(sdk_db_url):
    """Use a fresh database: decay/prune scan the whole store, not only our IDs."""
    engine = create_async_engine(sdk_db_url)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            session.add_all(
                [CognitiveConfig(key=key, value=value, category="clock-test") for key, value in CONFIG.items()]
            )
            await session.flush()
            yield session
            await session.rollback()
    finally:
        await engine.dispose()


async def _owned(operation, session):
    """Start the coroutine in SDK scope without mutating process-global ports."""
    async with MemoryClient(
        HostAdapters(
            inference=object(),
            embedder=object(),
            history=object(),
            session_factory=lambda: None,
            clock=lambda: NOW,
            overrides={
                "scrub_secrets": lambda value: value,
                "validate_skill_content": lambda value: [],
                "redact_urls_for_trace": lambda value: value,
            },
        )
    ) as client:

        async def configured():
            await cognitive_config.load_config(session)
            assert cognitive_config.is_loaded()
            assert all(cognitive_config.get(key) == value for key, value in CONFIG.items())
            return await operation()

        return await client.run_owned(lambda: asyncio.create_task(configured()))


def _trace(content, **overrides):
    return MemoryTrace(
        **{
            "content": content,
            "memory_type": "fact",
            "source_type": "test",
            "created_at": NOW - timedelta(days=10),
            "salience": 0.8,
            "initial_salience": 0.8,
            "significance": 3,
            **overrides,
        }
    )


@pytest.mark.parametrize("accessed", [False, True])
async def test_decay_daily_gate_uses_host_utc_midnight(db_session, accessed):
    await db_session.execute(text("SET LOCAL TIME ZONE 'America/Los_Angeles'"))
    common = {"access_count": int(accessed)}
    if accessed:
        common["last_accessed_at"] = NOW - timedelta(days=10)
    due = _trace("due", consolidated_at=MIDNIGHT - timedelta(microseconds=1), **common)
    done = _trace("done", consolidated_at=MIDNIGHT, **common)
    db_session.add_all([due, done])
    await db_session.flush()

    async def decay():
        return await consolidation._apply_decay(db_session)

    assert await _owned(decay, db_session) == 1
    await db_session.refresh(due)
    await db_session.refresh(done)
    expected = 0.8 * math.exp(-CONFIG["base_decay"] * 3 / 5 * 10)
    assert due.salience == pytest.approx(expected)
    assert due.consolidated_at == NOW
    assert done.salience == 0.8
    assert done.consolidated_at == MIDNIGHT
    assert await _owned(decay, db_session) == 0


async def test_prune_age_access_and_supersede_grace_use_host_clock(db_session):
    age = CONFIG["silent_age_days"]
    grace = CONFIG["durable_supersede_grace_days"]
    old = NOW - timedelta(days=age + 1)
    db_session.add_all(
        [
            _trace("silent", created_at=old, salience=0.0),
            _trace("young", created_at=NOW - timedelta(days=age - 1), salience=0.0),
            _trace("retrieved", created_at=old, salience=0.0, last_accessed_at=NOW),
            _trace(
                "grace",
                created_at=old,
                salience=0.0,
                typed_metadata={
                    "durability": "superseded",
                    "superseded_at": NOW.isoformat(),
                },
            ),
            _trace(
                "grace expired",
                created_at=old,
                salience=0.0,
                typed_metadata={
                    "durability": "superseded",
                    "superseded_at": (NOW - timedelta(days=grace + 1)).isoformat(),
                },
            ),
        ]
    )
    await db_session.flush()
    assert await _owned(lambda: consolidation._prune_silent_traces(db_session), db_session) == 2
    remaining = set((await db_session.execute(select(MemoryTrace.content))).scalars())
    assert remaining == {"young", "retrieved", "grace"}


async def test_run_retention_uses_host_clock(db_session):
    days = CONFIG["run_retention_days"]
    old = NOW - timedelta(days=days + 1)
    expired = ConsolidationRun(level=3, status="completed", started_at=old, completed_at=old)
    boundary = ConsolidationRun(level=3, status="failed", started_at=old, completed_at=NOW - timedelta(days=days))
    running = ConsolidationRun(level=3, status="running", started_at=old)
    db_session.add_all([expired, boundary, running])
    await db_session.flush()
    assert await _owned(lambda: consolidation._cleanup_old_runs(db_session), db_session) == 1
    remaining = set((await db_session.execute(select(ConsolidationRun.id))).scalars())
    assert remaining == {boundary.id, running.id}


async def test_entity_trace_window_uses_host_clock(db_session):
    days = int(CONFIG["entity_schema_trace_window_days"])
    entity = MemoryEntity(name="Synthetic clock subject", entity_type="person")
    recent = _trace("recent", created_at=NOW - timedelta(days=days) + timedelta(seconds=1))
    boundary = _trace("boundary", created_at=NOW - timedelta(days=days))
    db_session.add_all([entity, recent, boundary])
    await db_session.flush()
    db_session.add_all(
        [MemoryEntityLink(memory_id=row.id, entity_id=entity.id, role="subject") for row in [recent, boundary]]
    )
    await db_session.flush()
    producer = EntitySchemaProducer()
    assert await _owned(lambda: producer._recent_trace_contents(db_session, entity.id), db_session) == ["recent"]


async def test_entity_relationship_expiry_uses_host_clock(db_session):
    entity = MemoryEntity(name="Synthetic subject", entity_type="person", data={"role": "tester"})
    targets = [MemoryEntity(name=f"Synthetic target {i}", entity_type="person") for i in range(3)]
    db_session.add_all([entity, *targets])
    await db_session.flush()
    db_session.add_all(
        [
            EntityRelationship(
                from_entity_id=entity.id,
                to_entity_id=target.id,
                relationship_type="knows",
                valid_until=until,
                confidence=1.0,
            )
            for target, until in zip(targets, [NOW, NOW + timedelta(seconds=1), None], strict=True)
        ]
    )
    await db_session.flush()
    subject = ProducerSubject(key=str(entity.id), payload={"entity_id": entity.id})
    result = await _owned(lambda: EntitySchemaProducer().produce(db_session, subject, uuid.uuid4()), db_session)
    assert result is not None
    assert {str(rel["target_entity_id"]) for rel in result.payload["relationships"]} == {
        str(targets[1].id),
        str(targets[2].id),
    }


async def test_persona_pattern_window_uses_host_clock(db_session):
    days = float(CONFIG["persona_pattern_window_days"])
    for name, observed_at in [
        ("recent", NOW - timedelta(days=days) + timedelta(seconds=1)),
        ("boundary", NOW - timedelta(days=days)),
    ]:
        db_session.add(
            MemoryArtifact(
                artifact_type="Pattern",
                producer_name="synthetic-clock-test",
                content_hash=name,
                created_at=observed_at,
                payload={"dimension": name, "observation": name, "observed_at": observed_at.isoformat()},
            )
        )
    await db_session.flush()
    rows = await _owned(lambda: PersonaComposer(db_session)._fetch_recent_patterns(), db_session)
    assert [row.observation for row in rows] == ["recent"]


def _artifact(kind, name, payload, **overrides):
    return MemoryArtifact(
        artifact_type=kind,
        producer_name="synthetic-clock-test",
        content_hash=name,
        created_at=NOW,
        payload=payload,
        **overrides,
    )


async def test_pattern_expiry_strict_boundary_uses_host_clock(db_session):
    cutoff = NOW - timedelta(days=CONFIG["persona_pattern_window_days"])
    rows = [
        _artifact("Pattern", name, {"observed_at": observed.isoformat()})
        for name, observed in [
            ("stale", cutoff - timedelta(microseconds=1)),
            ("boundary", cutoff),
            ("recent", cutoff + timedelta(microseconds=1)),
        ]
    ]
    db_session.add_all(rows)
    await db_session.flush()
    assert await _owned(lambda: PatternProducer()._expire_stale(db_session), db_session) == 1
    statuses = dict((await db_session.execute(select(MemoryArtifact.content_hash, MemoryArtifact.status))).all())
    assert statuses == {"stale": "expired", "boundary": "active", "recent": "active"}
    assert await _owned(lambda: PatternProducer()._expire_stale(db_session), db_session) == 0


async def test_trait_demotion_threshold_and_future_clamp_use_host_clock(db_session):
    # Exact binary fractions at age zero make the equality boundary unambiguous.
    # The aged row starts above threshold but decays below it only on host time.
    cases = [
        ("aged", 0.75, NOW - timedelta(days=10)),
        ("below", 0.25, NOW),
        ("boundary", 0.5, NOW),
        ("above", 0.75, NOW),
        ("future below", 0.25, NOW + timedelta(days=10)),
    ]
    db_session.add_all(
        [
            _artifact(
                "PersonalityTrait",
                name,
                {
                    "magnitude": magnitude,
                    "last_reinforced_at": reinforced.isoformat(),
                    "rehearsal_count": 0,
                },
            )
            for name, magnitude, reinforced in cases
        ]
    )
    await db_session.flush()
    assert await _owned(lambda: TraitProducer()._demote_below_threshold(db_session), db_session) == 3
    statuses = dict((await db_session.execute(select(MemoryArtifact.content_hash, MemoryArtifact.status))).all())
    assert statuses == {
        "aged": "superseded",
        "below": "superseded",
        "boundary": "active",
        "above": "active",
        "future below": "superseded",
    }
    assert await _owned(lambda: TraitProducer()._demote_below_threshold(db_session), db_session) == 0


async def test_trajectory_discovery_strict_lookback_uses_host_clock(db_session):
    cutoff = NOW - timedelta(days=30)
    entities = [MemoryEntity(name=name, entity_type="person") for name in ["eligible", "boundary", "old"]]
    db_session.add_all(entities)
    await db_session.flush()
    for entity, edge in zip(
        entities, [cutoff + timedelta(microseconds=1), cutoff, cutoff - timedelta(microseconds=1)], strict=True
    ):
        traces = [_trace(entity.name + " edge", created_at=edge), _trace(entity.name + " recent", created_at=NOW)]
        db_session.add_all(traces)
        await db_session.flush()
        db_session.add_all([MemoryEntityLink(memory_id=row.id, entity_id=entity.id, role="subject") for row in traces])
    await db_session.flush()
    subjects = await _owned(lambda: TrajectoryProducer().discover_subjects(db_session, uuid.uuid4()), db_session)
    assert [subject.key for subject in subjects] == [str(entities[0].id)]
    assert subjects[0].payload["first_trace_at"] == cutoff + timedelta(microseconds=1)
    assert subjects[0].payload["last_trace_at"] == NOW


@pytest.mark.parametrize("kind", ["PersonalityTrait", "Stance"])
async def test_persona_decay_ranking_and_future_clamp_use_host_clock(db_session, kind):
    trait = kind == "PersonalityTrait"
    value_key = "magnitude" if trait else "confidence"
    time_key = "last_reinforced_at" if trait else "last_validated_at"
    output_key = "guidance_text" if trait else "position"
    # Database wall time sees all rows in the future and picks the 0.9 row.
    # Host time decays that row below the fresh 0.6 winner. Without the
    # future-age clamp the 0.5 future row wrongly outranks that winner.
    db_session.add_all(
        [
            _artifact(
                kind,
                name,
                {
                    value_key: value,
                    time_key: timestamp.isoformat(),
                    output_key: name,
                    "rehearsal_count": 0,
                },
            )
            for name, value, timestamp in [
                ("old strongest", 0.9, NOW - timedelta(days=10)),
                ("fresh winner", 0.6, NOW),
                ("future weaker", 0.5, NOW + timedelta(days=10)),
            ]
        ]
    )
    await db_session.flush()
    composer = PersonaComposer(db_session)
    operation = composer._fetch_active_traits if trait else lambda: composer._fetch_relevant_stances(Chat())
    rows = await _owned(operation, db_session)
    assert [getattr(row, output_key) for row in rows] == ["fresh winner"]
