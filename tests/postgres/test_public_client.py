"""M5 named clients over disposable storage; only inference/embeddings are recorded."""

import asyncio
from datetime import UTC, datetime, timedelta
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from kai_memory import AgentIdentity, Chat, HostAdapters, MemoryClient
from memory_test_host import FixedEmbedder
from kai_memory.procedural_api import RuleCandidate
from kai_memory.storage.models.cognitive_config import CognitiveConfig
from kai_memory.storage.models.memory_entity import MemoryEntity
from kai_memory.storage.models.memory_trace import MemoryTrace

pytestmark = pytest.mark.asyncio(loop_scope="session")
NOW = datetime(2026, 10, 31, tzinfo=UTC)


class History:
    def __init__(self):
        self.messages = []

    async def record(self, message):
        self.messages.append(message)

    async def read_page(self, chat, after, limit):
        rows = sorted(
            [m for m in self.messages if (m.chat_id, m.topic_id, m.venue) == (chat.chat_id, chat.topic_id, chat.venue)],
            key=lambda m: (m.created_at, m.id),
        )
        if after is not None:
            rows = [
                m
                for m in rows
                if m.created_at > after.created_at
                or (m.created_at == after.created_at and after.id is not None and m.id > after.id)
            ]
        return rows[:limit]

    async def senders_seen_before(self, senders, before):
        return {m.sender_user_id for m in self.messages if m.sender_user_id in senders and m.created_at < before}


class Inference:
    def __init__(self):
        self.subject = "Mira"
        self.value = "Cedar City"
        self.calls = 0

    async def complete(self, prompt, *, model, max_tokens, thinking_mode="adaptive"):
        if "You are a memory extraction system" not in prompt:
            return None
        self.calls += 1
        return json.dumps(
            {
                "memories": [
                    {
                        "content": f"{self.subject} lives in {self.value}",
                        "memory_type": "fact",
                        "significance": 4,
                        "entities": [{"name": self.subject, "type": "person", "matched_participant_key": None}],
                        "facts": [
                            {
                                "subject": self.subject,
                                "field": "city",
                                "value": self.value,
                                "polarity": "is",
                                "slot": "current_location",
                                "durability": "durable",
                            }
                        ],
                        "confidence": 0.9,
                        "is_entity_eligible": True,
                    }
                ],
                "social_experiences": [],
            }
        )


@pytest.fixture
def host():
    return SimpleNamespace(
        name="atlas",
        policy=SimpleNamespace(
            is_admin=lambda actor: actor == 10,
            respect=lambda actor: {10: 100, 20: 75}.get(actor, 0),
            may_read=lambda chat: True,
            may_write=lambda chat: True,
        ),
    )


def adapters(host, factory, inference):
    return HostAdapters(
        inference=inference,
        embedder=FixedEmbedder(),
        history=History(),
        session_factory=factory,
        identity=AgentIdentity(host.name, host.name.title()),
        policy=host.policy,
        clock=lambda: NOW + timedelta(days=1),
        overrides={
            "scrub_secrets": lambda v: v,
            "validate_skill_content": lambda v: [],
            "redact_urls_for_trace": lambda v: v,
        },
    )


async def test_named_client_capture_sharing_and_denied_scope(host, sdk_db_url, loopback_only):
    engine = create_async_engine(sdk_db_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as session:
            session.add_all(
                [
                    CognitiveConfig(key="procedural_rule_extraction_enabled", value=1, category="test"),
                    CognitiveConfig(key="procedural_semantic_serving_mode", value=2, category="test"),
                ]
            )
            await session.commit()
        async with MemoryClient(adapters(host, factory, Inference())) as client:
            chat = Chat(chat_id=101, topic_id=7, venue="synthetic", is_group=True, sender_user_id=20)
            teacher = await client.record_message(chat, "user", "Remember: use short paragraphs", sender_id=20, at=NOW)
            candidate = RuleCandidate(
                teacher.text,
                "communication",
                "Use short paragraphs",
                "When answering",
                directive_strength="instruction",
            )
            captured = await client.procedural.capture(chat, [candidate], [teacher.id])
            assert captured.changed == 1 and captured.decisions == ("minted",)
            shared = await client.procedural.serve(replace(chat, sender_user_id=99))
            assert len(shared.served_rule_ids) == 1 and "Use short paragraphs" in shared.block
            denied = await client.procedural.serve(replace(chat, topic_id=8, sender_user_id=99))
            assert denied.served_rule_ids == () and not denied.block
            async with factory() as session:
                row = await session.get(MemoryTrace, shared.served_rule_ids[0])
                proc = row.typed_metadata["procedural"]
                assert proc["provenance"] == {"taught_by": 20, "respect": 75}
                assert proc["source_message_ids"] == [str(teacher.id)]
                assert proc["source_chat_ids"] == ["101:7"]
    finally:
        await engine.dispose()


async def test_public_extract_mixed_script_has_one_persisted_subject(host, sdk_db_url, loopback_only):
    engine = create_async_engine(sdk_db_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    inference = Inference()
    try:
        async with MemoryClient(adapters(host, factory, inference)) as client:
            chat = Chat(chat_id=101, venue="synthetic", sender_user_id=10, sender_name="Teacher")
            for batch, (subject, city) in enumerate((("Mira", "Cedar City"), ("Мира", "Fern City"))):
                inference.subject, inference.value = subject, city
                for offset in range(6):
                    await client.record_message(
                        chat,
                        "user" if offset % 2 == 0 else "assistant",
                        f"{subject} lives in {city}.",
                        sender_id=10 if offset % 2 == 0 else None,
                        sender_name="Teacher" if offset % 2 == 0 else None,
                        at=NOW + timedelta(minutes=batch * 10 + offset),
                    )
                result = await client.extract(chat)
                assert result["completed"] and result["traces_created"] == 1, result
            assert inference.calls == 2
            async with factory() as session:
                entities = (
                    (await session.execute(select(MemoryEntity).where(MemoryEntity.entity_type == "person")))
                    .scalars()
                    .all()
                )
                subjects = [e for e in entities if e.name in {"Mira", "Мира"} or "Мира" in (e.aliases or [])]
                assert len(subjects) == 1
                assert subjects[0].telegram_user_id is None  # The teacher is not the described person.
                traces = (
                    (await session.execute(select(MemoryTrace).where(MemoryTrace.memory_type == "fact")))
                    .scalars()
                    .all()
                )
                assert len(traces) == 2
                facts = [fact for trace in traces for fact in trace.typed_metadata["facts"]]
                assert len(facts) == 2 and {fact["subject_id"] for fact in facts} == {str(subjects[0].id)}
                assert {fact["subject"] for fact in facts} == {"Mira", "Мира"}
    finally:
        await engine.dispose()


async def test_public_persona_composes_persisted_layers_without_inference(host, sdk_db_url, loopback_only):
    from kai_memory.storage.models.memory_artifact import MemoryArtifact

    engine = create_async_engine(sdk_db_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    inference = Inference()
    try:
        async with factory() as session:
            payloads = [
                ("Anchor", "permanent", {"content": f"{host.name.title()} synthetic anchor"}),
                (
                    "PersonalityTrait",
                    "active",
                    {"guidance_text": "Explain uncertainty", "magnitude": 0.9, "last_reinforced_at": NOW.isoformat()},
                ),
                (
                    "Stance",
                    "active",
                    {"position": "Prefer evidence", "confidence": 0.9, "last_validated_at": NOW.isoformat()},
                ),
                (
                    "Pattern",
                    "active",
                    {"observation": "Uses examples", "dimension": "style", "observed_at": NOW.isoformat()},
                ),
            ]
            session.add_all(
                [
                    MemoryArtifact(
                        artifact_type=kind,
                        status=status,
                        producer_name="synthetic",
                        payload=payload,
                        content_hash=str(index) * 64,
                    )
                    for index, (kind, status, payload) in enumerate(payloads)
                ]
            )
            await session.commit()
        async with MemoryClient(adapters(host, factory, inference)) as client:
            chat = Chat(chat_id=101, venue="synthetic", sender_user_id=10)
            result = await client.compose_persona(chat)
            assert result == await client.compose_persona(chat)
            assert (
                result.index("synthetic anchor")
                < result.index("Explain uncertainty")
                < result.index("Prefer evidence")
                < result.index("Uses examples")
            )
            assert inference.calls == 0
    finally:
        await engine.dispose()


@pytest.mark.parametrize(
    "windowed,failure",
    [
        (True, "timeout"),
        (True, "semantic"),
        (True, "entity"),
        (True, "graph"),
        (True, "intention"),
        (True, "access"),
        (False, "graph"),
        (False, "access"),
    ],
)
async def test_priming_completed_rows_survive_real_recovery(
    host, sdk_db_url, loopback_only, monkeypatch, windowed, failure
):
    from kai_memory.retrieval import memory_priming as mp
    from kai_memory.retrieval.iterative_retrieval import SegmentCue, WindowCues

    engine = create_async_engine(sdk_db_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as seed:
            row = MemoryTrace(
                content="Mira lives in Cedar City",
                memory_type="fact",
                source_type="chat",
                created_at=NOW,
                salience=0.8,
                emotion_arousal=0.2,
                typed_metadata={"durability": "durable"},
                source_path="recorded-test",
            )
            seed.add(row)
            await seed.commit()
            trace_id = row.id

        async with MemoryClient(adapters=adapters(host, factory, Inference())) as client, factory() as session:
            svc = mp.MemoryPrimingService(session, weights_override={"semantic": 1.0})
            managed = []
            cancelled = []
            rollback = AsyncMock(wraps=session.rollback)
            monkeypatch.setattr(session, "rollback", rollback)

            async def search(*args, **kwargs):
                if managed and failure in {"timeout", "semantic"}:
                    if failure == "semantic":
                        raise RuntimeError("scripted later search failure")
                    try:
                        await asyncio.Event().wait()
                    finally:
                        cancelled.append(True)
                found = (await session.scalars(select(MemoryTrace).where(MemoryTrace.id == trace_id))).one()
                managed.append(found)
                # Only the subsequent wait is short; the real initial query has its normal budget.
                monkeypatch.setattr(mp, "_SEMANTIC_TIMEOUT_S", 0.01)
                return [(found, 0.9)]

            monkeypatch.setattr(svc.memory_svc, "search_traces_by_vector", search)
            monkeypatch.setattr(svc.memory_svc, "search_traces_raw", search)
            monkeypatch.setattr(
                svc.memory_svc,
                "find_entities_by_names",
                AsyncMock(
                    return_value=[SimpleNamespace(id=trace_id)],
                    side_effect=RuntimeError("scripted entity failure") if failure == "entity" else None,
                ),
            )
            monkeypatch.setattr(svc.memory_svc, "get_entity_linked_traces", AsyncMock(return_value=[]))
            monkeypatch.setattr(svc.memory_svc, "get_entity_linked_traces_with_entity", AsyncMock(return_value=[]))
            monkeypatch.setattr(
                svc.memory_svc,
                "traverse_entity_graph",
                AsyncMock(
                    return_value=[],
                    side_effect=RuntimeError("scripted graph failure") if failure == "graph" else None,
                ),
            )
            monkeypatch.setattr(
                svc.memory_svc,
                "get_intention_traces",
                AsyncMock(
                    return_value=[],
                    side_effect=RuntimeError("scripted intention failure") if failure == "intention" else None,
                ),
            )
            if failure == "access":
                monkeypatch.setattr(
                    svc.memory_svc, "record_access_batch", AsyncMock(side_effect=RuntimeError("access"))
                )
            cues = WindowCues(segments=[SegmentCue("Mira", 1.0, [1.0])], is_windowed=True)
            if failure in {"timeout", "semantic"}:
                cues.segments.append(SegmentCue("x", 0.5, [0.5]))
            # Capitalized name long enough to exercise entity/graph/intention paths.
            cues.segments[0] = SegmentCue("Mira lives", 1.0, [1.0])
            formatted, ids, top = await client.run_owned(
                lambda: asyncio.create_task(
                    svc.prime_with_traces(
                        "Mira lives",
                        window_cues=cues if windowed else None,
                    )
                )
            )
            rollback.assert_awaited_once()
            assert bool(cancelled) == (failure == "timeout")
            assert inspect(managed[0]).persistent  # recovery must not expunge the caller's row
            assert inspect(managed[0]).expired_attributes
            assert ids == [trace_id]
            assert "Mira lives in Cedar City" in formatted
            trace, score, components = top[0]
            assert trace is not managed[0] and inspect(trace).transient
            assert trace.id == trace_id and trace.content == "Mira lives in Cedar City"
            assert trace.created_at == NOW and trace.memory_type == "fact"
            assert trace.typed_metadata == {"durability": "durable"}
            assert trace.source_path == "recorded-test"
            assert score == pytest.approx(0.9) and components["semantic"] == pytest.approx(0.9)
            assert components["salience"] == pytest.approx(0.8)
            await session.commit()

        async with factory() as verify:
            stored = await verify.get(MemoryTrace, trace_id)
            assert stored.access_count == (0 if failure == "access" else 1)
            assert stored.content == "Mira lives in Cedar City"
    finally:
        await engine.dispose()
