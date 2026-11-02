"""Disposable PostgreSQL proof for the public procedural lifecycle (DB-879)."""

from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from kai_memory import ports
from kai_memory.procedural_api import ExecutionEvidence, ProceduralMemory, RuleCandidate, ToolEvidence
from kai_memory.sdk import Chat, Message
from kai_memory.storage.models.execution_trace import ExecutionTrace
from kai_memory.storage.models.memory_trace import MemoryTrace
from kai_memory.store import cognitive_config, embeddings

pytestmark = pytest.mark.asyncio(loop_scope="session")
NOW = datetime(2026, 10, 24, tzinfo=UTC)


@pytest_asyncio.fixture(loop_scope="session")
async def db_session(sdk_db_url):
    """Whole-store rollback assertions require a fresh store without committed suite seeds."""
    engine = create_async_engine(sdk_db_url)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            async with session.begin():
                yield session
                await session.rollback()
    finally:
        await engine.dispose()


@pytest.fixture
def procedural_host(db_session, monkeypatch):
    import kai_memory.procedural_api as api

    state = SimpleNamespace(messages=[])

    @asynccontextmanager
    async def operation():
        yield

    @asynccontextmanager
    async def sessions():
        # The integration fixture owns the surrounding rollback transaction.
        async with db_session.begin_nested():
            yield db_session

    monkeypatch.setattr(api, "session_scope", sessions)
    policy = SimpleNamespace(
        is_admin=lambda actor: actor == 10,
        respect=lambda actor: 100 if actor == 10 else 0,
        may_read=lambda chat: chat.chat_id == 1,
        may_write=lambda chat: chat.chat_id == 1,
    )
    monkeypatch.setitem(ports.__dict__, "get_host_policy", lambda: policy)
    monkeypatch.setattr(cognitive_config, "get_bool", lambda name: True)
    monkeypatch.setattr(cognitive_config, "get", lambda name: "2")
    previous = ports.bind(
        clock=lambda: NOW,
        scrub_secrets=lambda value: value,
        validate_skill_content=lambda value: [],
        redact_urls_for_trace=lambda value: value,
        strip_msg_meta_envelope=lambda value: value,
        is_admin_user=lambda actor, ids: actor == 10,
        senders_seen_before=AsyncMock(return_value=set()),
        read_conversation_page=AsyncMock(side_effect=lambda *args: state.messages),
    )
    state.api = ProceduralMemory(operation, AsyncMock())
    state.chat = Chat(chat_id=1, venue="example", sender_user_id=10)
    state.teacher = Message(1, None, "example", "user", "Keep answers concise", NOW, sender_user_id=10)
    state.messages.append(state.teacher)
    state.rule = RuleCandidate(
        "Keep answers concise",
        "communication",
        "Use concise answers",
        "When answering",
        directive_strength="instruction",
        scope_hint="this_person",
    )
    monkeypatch.setattr(embeddings, "generate_embedding", AsyncMock(return_value=[1.0] + [0.0] * 1535))
    try:
        yield state
    finally:
        ports.bind(**previous)


async def test_public_capture_serving_execution_observation_and_replay(procedural_host, db_session):
    host = procedural_host
    result = await host.api.capture(host.chat, [host.rule], [host.teacher.id])
    assert result.changed == 1 and result.decisions == ("minted",)
    result = await host.api.capture(host.chat, [host.rule], [host.teacher.id], history_after=None)
    assert result.changed == 0 and result.decisions == ("duplicate",)
    serving = await host.api.serve(host.chat)
    assert serving.block and len(serving.served_rule_ids) == 1
    rid = serving.served_rule_ids[0]
    assistant = Message(
        1,
        None,
        "example",
        "assistant",
        "Done",
        NOW + timedelta(seconds=1),
        message_metadata={"served_rule_ids": [str(rid)]},
    )
    host.messages.append(assistant)
    evidence = ExecutionEvidence(assistant.id, tools=(ToolEvidence("search", ("query",)),), served_rule_ids=(rid,))
    receipt = await host.api.record_execution(host.chat, "Find the answer", evidence)
    assert receipt.status == "recorded"
    duplicate = await host.api.record_execution(host.chat, "Find the answer", evidence)
    assert duplicate.status == "duplicate" and duplicate.execution_trace_id == receipt.execution_trace_id
    trace = await db_session.get(ExecutionTrace, receipt.execution_trace_id)
    assert trace.outcome == "unknown" and "rule_compliance" not in trace.success_signals
    host.messages.append(
        Message(1, None, "example", "user", "That worked", NOW + timedelta(seconds=2), sender_user_id=10)
    )
    observed = await host.api.observe(
        host.chat, assistant.id, outcome="success", evidence_quote="That worked", compliance={rid: "followed"}
    )
    assert observed.status == "recorded" and observed.outcome_updated and observed.compliance_updated == 1
    again = await host.api.observe(
        host.chat, assistant.id, outcome="success", evidence_quote="That worked", compliance={rid: "followed"}
    )
    assert again.status == "duplicate"
    rule = await db_session.get(MemoryTrace, rid)
    assert rule.typed_metadata["procedural"]["utility"]["s"] == 0  # nightly ledger owns scoring


async def test_public_capture_embedding_rollback_then_retry_is_durable(procedural_host, db_session, monkeypatch):
    host = procedural_host
    second = replace(host.rule, trigger="When searching", rule="Use search results")
    failure = True

    async def embed(value):
        if "searching" in value and failure:
            raise RuntimeError("scripted transient embedding failure")
        return ([0.0, 1.0] if "searching" in value else [1.0, 0.0]) + [0.0] * 1534

    monkeypatch.setattr(embeddings, "generate_embedding", embed)
    result = await host.api.capture(host.chat, [host.rule, second], [host.teacher.id])
    assert result.status == "failed" and result.changed == 0
    assert not (await db_session.execute(select(MemoryTrace))).scalars().all()
    failure = False
    result = await host.api.capture(host.chat, [host.rule, second], [host.teacher.id])
    assert result.changed == 2
    result = await host.api.capture(host.chat, [host.rule, second], [host.teacher.id])
    assert result.changed == 0 and result.decisions == ("duplicate", "duplicate")
    assert len((await db_session.execute(select(MemoryTrace))).scalars().all()) == 2


async def test_same_batch_supersession_and_person_scope_are_preserved(procedural_host, db_session):
    host = procedural_host
    first = replace(host.rule, source_quote="First instruction")
    negative = replace(host.rule, rule="Do not use concise answers", source_quote="Second instruction")
    last = replace(host.rule, source_quote="Third instruction")
    host.messages = [
        replace(host.teacher, id=uuid4(), text=quote, created_at=NOW + timedelta(seconds=index))
        for index, quote in enumerate(("First instruction", "Second instruction", "Third instruction"))
    ]
    source_ids = [message.id for message in host.messages]
    result = await host.api.capture(host.chat, [first, negative, last], source_ids)
    assert result.decisions == ("minted", "superseded", "superseded")
    rows = (await db_session.execute(select(MemoryTrace))).scalars().all()
    survivors = [row for row in rows if row.invalidated_at is None]
    assert len(rows) == 3 and len(survivors) == 1
    assert survivors[0].typed_metadata["procedural"]["guidance"] == host.rule.rule
    replay = await host.api.capture(host.chat, [first, negative, last], source_ids)
    assert replay.changed == 0 and replay.decisions == ("duplicate", "duplicate", "duplicate")
    assert len((await db_session.execute(select(MemoryTrace))).scalars().all()) == 3
    # Another trusted person's otherwise identical private rule cannot fold into it.
    policy = ports.get_host_policy()
    policy.respect = lambda actor: 100 if actor in (10, 20) else 0
    teacher = replace(host.teacher, id=uuid4(), sender_user_id=20)
    host.messages.append(teacher)
    result = await host.api.capture(replace(host.chat, sender_user_id=20), [host.rule], [teacher.id])
    assert result.decisions == ("minted",)
    served = await host.api.serve(replace(host.chat, sender_user_id=20))
    assert len(served.served_rule_ids) == 1 and served.served_rule_ids[0] != survivors[0].id


async def test_tool_free_compliance_remains_message_anchored(procedural_host, db_session):
    host = procedural_host
    await host.api.capture(host.chat, [host.rule], [host.teacher.id])
    serving = await host.api.serve(host.chat)
    rid = serving.served_rule_ids[0]
    assistant = Message(
        1,
        None,
        "example",
        "assistant",
        "Done",
        NOW + timedelta(seconds=1),
        message_metadata={"served_rule_ids": [str(rid)]},
    )
    host.messages.append(assistant)
    receipt = await host.api.record_execution(host.chat, "ask", ExecutionEvidence(assistant.id, served_rule_ids=(rid,)))
    assert receipt.status == "tool_free" and receipt.execution_trace_id is None
    assert not (await db_session.execute(select(ExecutionTrace))).scalars().all()
    observed = await host.api.observe(host.chat, assistant.id, compliance={rid: "violated"})
    assert observed.compliance_updated == 1
    observed = await host.api.observe(host.chat, assistant.id, compliance={rid: "violated"})
    assert observed.status == "duplicate"
    rule = await db_session.get(MemoryTrace, rid)
    proc = rule.typed_metadata["procedural"]
    assert len(proc["compliance_events"]) == 1 and proc["utility"]["f"] == 0


@pytest.mark.parametrize("retirement", ["supersede", "invalidate"])
async def test_retired_teaching_cannot_resurrect_by_flipping_group_safe(procedural_host, db_session, retirement):
    from kai_memory.store.cognitive_memory import CognitiveMemoryService

    host = procedural_host
    original_rule = replace(host.rule, scope_hint="global", group_safe=False)
    created = await host.api.capture(host.chat, [original_rule], [host.teacher.id])
    assert created.decisions == ("minted",)
    original = (await db_session.execute(select(MemoryTrace))).scalar_one()
    if retirement == "supersede":
        correction = replace(
            host.teacher, id=uuid4(), text="Stop using concise answers", created_at=NOW + timedelta(seconds=1)
        )
        host.messages.append(correction)
        replacement = replace(original_rule, source_quote=correction.text, rule="Do not use concise answers")
        changed = await host.api.capture(host.chat, [replacement], [correction.id])
        assert changed.decisions == ("superseded",)
    else:
        await CognitiveMemoryService(db_session).invalidate_trace(original.id)
    before = (await db_session.execute(select(MemoryTrace))).scalars().all()
    survivors = {row.id for row in before if row.invalidated_at is None}
    embeddings_before = embeddings.generate_embedding.await_count
    replay = await host.api.capture(host.chat, [replace(original_rule, group_safe=True)], [host.teacher.id])
    assert replay.status == "completed" and replay.changed == 0 and replay.decisions == ("duplicate",)
    assert embeddings.generate_embedding.await_count == embeddings_before
    after = (await db_session.execute(select(MemoryTrace))).scalars().all()
    assert len(after) == len(before) and original.invalidated_at is not None
    assert {row.id for row in after if row.invalidated_at is None} == survivors


@pytest.mark.parametrize(
    "stored,candidate_scope,matched",
    [
        (
            {"person": None, "venue": "any", "group_safe": False},
            {"person": None, "venue": "any", "group_safe": True},
            True,
        ),
        ({}, {"person": None, "venue": "any"}, True),
        ({"person": 10, "venue": "wake"}, {"person": 10, "venue": "wake"}, True),
        ({"person": 10, "venue": None}, {"person": 10, "venue": None}, True),
        ({"person": "10", "venue": "wake"}, {"person": 10, "venue": "wake"}, False),
        ({"person": 20, "venue": "wake"}, {"person": 10, "venue": "wake"}, False),
        ({"person": 10, "venue": "wake"}, {"person": 10, "venue": "any"}, False),
        ({"person": None, "venue": None}, {"person": None, "venue": "any"}, False),
        ({"person": None, "venue": "any"}, {"person": None, "venue": None}, False),
    ],
)
async def test_replay_scope_identity_preserves_postgres_json_types(db_session, stored, candidate_scope, matched):
    from kai_memory.procedural.rule_capture import _replay_scope_identity

    existing = MemoryTrace(
        content="Synthetic scope identity",
        memory_type="procedural",
        source_type="conversation",
        typed_metadata={"procedural": {"scope": stored}},
    )
    db_session.add(existing)
    await db_session.flush()
    found = (
        await db_session.execute(
            select(MemoryTrace.id).where(
                MemoryTrace.id == existing.id,
                *_replay_scope_identity(MemoryTrace.typed_metadata["procedural"]["scope"], candidate_scope),
            )
        )
    ).scalar_one_or_none()
    assert (found is not None) is matched


async def test_batched_replay_covers_old_tombstones_beyond_one_chunk(db_session, monkeypatch):
    from kai_memory.procedural.rule_capture import _load_replayed_keys, _replay_key, _REPLAY_BATCH_SIZE

    scope = {"person": None, "venue": "any", "group_safe": False}
    keys = [
        _replay_key(str(uuid4()), f"Synthetic guidance {index}", "When answering", scope)
        for index in range(_REPLAY_BATCH_SIZE + 1)
    ]
    db_session.add_all(
        [
            MemoryTrace(
                content="Synthetic old teaching",
                memory_type="procedural",
                source_type="conversation",
                invalidated_at=NOW,
                typed_metadata={
                    "procedural": {
                        "source_message_ids": [key[0]],
                        "guidance": key[1],
                        "trigger_pattern": key[2],
                        "scope": scope,
                    }
                },
            )
            for key in keys
        ]
    )
    await db_session.flush()
    execute = AsyncMock(wraps=db_session.execute)
    monkeypatch.setattr(db_session, "execute", execute)
    assert await _load_replayed_keys(db_session, keys) == set(keys)
    assert execute.await_count == 2
    assert all(len(call.args[0].selected_columns[0].clauses) <= _REPLAY_BATCH_SIZE for call in execute.await_args_list)


async def test_public_minted_superseded_teaching_remains_tombstoned_within_batch(procedural_host, db_session):
    host = procedural_host
    original = replace(host.rule, source_quote="Positive teaching", scope_hint="global")
    opposite = replace(original, source_quote="Negative teaching", rule="Do not use concise answers")
    first = replace(host.teacher, text=original.source_quote)
    second = replace(host.teacher, id=uuid4(), text=opposite.source_quote, created_at=NOW + timedelta(seconds=1))
    host.messages = [first, second]
    result = await host.api.capture(host.chat, [original, opposite, original], [first.id, second.id])
    assert result.decisions == ("minted", "superseded", "duplicate")
    rows = (await db_session.execute(select(MemoryTrace))).scalars().all()
    assert len(rows) == 2
    survivors = [row for row in rows if row.invalidated_at is None]
    assert len(survivors) == 1 and survivors[0].typed_metadata["procedural"]["guidance"] == opposite.rule


async def test_replay_batch_empty_store_returns_no_hits(db_session):
    from kai_memory.procedural.rule_capture import _load_replayed_keys, _replay_key

    keys = [_replay_key(str(uuid4()), "Synthetic guidance", "When answering", {"person": None, "venue": "any"})]
    assert await _load_replayed_keys(db_session, keys) == set()


@pytest.mark.parametrize("public", [True, False])
async def test_malformed_category_does_not_rollback_valid_stored_rule(procedural_host, db_session, caplog, public):
    from dataclasses import asdict
    from kai_memory.procedural import rule_capture

    host = procedural_host
    private_text = "recognizable-private-malformed-category"
    inputs = [host.rule, replace(host.rule, category={private_text: private_text})]
    with caplog.at_level("INFO", logger=rule_capture.logger.name):
        if public:
            result = await host.api.capture(host.chat, inputs, [host.teacher.id])
            assert result.status == "completed"
            changed, decisions = result.changed, result.decisions
        else:
            decisions = []
            changed = await rule_capture.capture_rules(
                db_session,
                host.chat,
                [asdict(value) for value in inputs],
                host.messages,
                None,
                respect_of=ports.get_host_policy().respect,
                decisions=decisions,
            )
    assert changed == 1 and tuple(decisions) == ("minted", "rejected:offschema")
    rows = (await db_session.execute(select(MemoryTrace))).scalars().all()
    assert len(rows) == 1 and rows[0].typed_metadata["procedural"]["guidance"] == host.rule.rule
    assert private_text not in caplog.text


@pytest.mark.parametrize("venue", ["event", "heartbeat", "wake"])
async def test_public_headless_record_observe_and_scope(procedural_host, db_session, monkeypatch, venue):
    host = procedural_host
    monkeypatch.setitem(
        ports.__dict__,
        "get_host_policy",
        lambda: SimpleNamespace(
            may_read=lambda chat: True, may_write=lambda chat: True, is_admin=lambda actor: actor == 10
        ),
    )
    chat = Chat(chat_id=None, venue=venue, sender_user_id=10)
    teacher = replace(host.teacher, chat_id=None, venue=venue)
    assistant = Message(None, None, venue, "assistant", "Done", NOW + timedelta(seconds=1))
    host.messages[:] = [teacher, assistant]
    receipt = await host.api.record_execution(
        chat, "Find the answer", ExecutionEvidence(assistant.id, tools=(ToolEvidence("search", ("query",)),))
    )
    assert receipt.status == "recorded"
    trace = await db_session.get(ExecutionTrace, receipt.execution_trace_id)
    assert trace.conversation_ref is None and trace.success_signals["venue"] == venue
    host.messages.append(
        Message(None, None, venue, "user", "That worked", NOW + timedelta(seconds=2), sender_user_id=10)
    )
    original_signals = dict(trace.success_signals)
    for field, wrong, original in [
        ("source_user_id", 99, 10),
        ("conversation_ref", "event:1:None", None),
        ("success_signals", {**original_signals, "venue": "other"}, original_signals),
    ]:
        setattr(trace, field, wrong)
        await db_session.flush()
        rejected = await host.api.observe(chat, assistant.id, outcome="success", evidence_quote="That worked")
        assert rejected.status == "rejected" and rejected.reason == "trace_scope"
        assert trace.outcome == "unknown"
        setattr(trace, field, original)
        await db_session.flush()
    observed = await host.api.observe(chat, assistant.id, outcome="success", evidence_quote="That worked")
    assert observed.status == "recorded" and observed.outcome_updated
    duplicate = await host.api.observe(chat, assistant.id, outcome="success", evidence_quote="That worked")
    assert duplicate.status == "duplicate"


@pytest.mark.parametrize("legacy", [False, True])
async def test_reinforced_wording_survives_retirement_and_restart(procedural_host, db_session, legacy):
    host = procedural_host
    minted = await host.api.capture(host.chat, [host.rule], [host.teacher.id])
    assert minted.decisions == ("minted",)
    original = (await db_session.execute(select(MemoryTrace))).scalar_one()
    original_id = original.id
    teacher = replace(host.teacher, id=uuid4(), text="Please use brief replies", created_at=NOW + timedelta(seconds=1))
    host.messages.append(teacher)
    teaching = replace(host.rule, source_quote=teacher.text, rule="Use brief replies")
    variants = [replace(teaching, rule=f"Use brief replies variant {index}") for index in range(55)]
    teaching = variants[0]
    if legacy:
        from dataclasses import asdict
        from kai_memory.procedural.rule_capture import capture_rules

        decisions = []
        changed = await capture_rules(
            db_session,
            host.chat,
            [asdict(value) for value in variants],
            [teacher],
            None,
            respect_of=ports.get_host_policy().respect,
            decisions=decisions,
        )
        reinforced = SimpleNamespace(changed=changed, decisions=tuple(decisions))
    else:
        reinforced = await host.api.capture(host.chat, variants, [teacher.id])
    assert reinforced.changed == 1 and reinforced.decisions == ("reinforced",) + ("duplicate",) * 54
    original = await db_session.get(MemoryTrace, original_id)
    assert original.typed_metadata["procedural"]["guidance"] == host.rule.rule
    consumed = original.typed_metadata["procedural"]["consumed_teachings"]
    assert len(consumed) == 55 and consumed[0]["guidance"] == teaching.rule
    original.invalidated_at = NOW + timedelta(seconds=2)
    await db_session.flush()
    db_session.expunge_all()
    from kai_memory.procedural.rule_capture import _load_replayed_keys, _replay_key

    exact_key = _replay_key(str(teacher.id), teaching.rule, teaching.trigger, {"person": 10, "venue": "any"})
    string_actor = _replay_key(str(teacher.id), teaching.rule, teaching.trigger, {"person": "10", "venue": "any"})
    null_venue = _replay_key(str(teacher.id), teaching.rule, teaching.trigger, {"person": 10, "venue": None})
    assert await _load_replayed_keys(db_session, [exact_key, string_actor, null_venue]) == {exact_key}

    # A new facade and reloaded ORM rows cannot rely on the previous capture's in-memory keys.
    @asynccontextmanager
    async def restarted_operation():
        yield

    host.api = ProceduralMemory(restarted_operation, AsyncMock())
    replay = await host.api.capture(host.chat, [teaching], [teacher.id])
    assert replay.changed == 0 and replay.decisions == ("duplicate",)
    # Neither a distinct scope nor genuinely different wording is an exact replay.
    changed_scope = replace(teaching, scope_hint="global")
    scoped = await host.api.capture(host.chat, [changed_scope], [teacher.id])
    assert scoped.changed == 1 and scoped.decisions == ("minted",)
    changed_wording = replace(teaching, rule="Use compact answers")
    reworded = await host.api.capture(host.chat, [changed_wording], [teacher.id])
    assert reworded.changed == 1 and reworded.decisions == ("minted",)


async def test_reinforced_identity_rolls_back_and_retry_consumes_it(procedural_host, db_session, monkeypatch):
    host = procedural_host
    assert (await host.api.capture(host.chat, [host.rule], [host.teacher.id])).decisions == ("minted",)
    original = (await db_session.execute(select(MemoryTrace))).scalar_one()
    original_id = original.id
    teacher = replace(host.teacher, id=uuid4(), text="Please use brief replies", created_at=NOW + timedelta(seconds=1))
    host.messages.append(teacher)
    teaching = replace(host.rule, source_quote=teacher.text, rule="Use brief replies")
    failure = replace(teaching, trigger="Fail embedding")

    async def embed(value):
        if "fail embedding" in value.lower():
            raise RuntimeError("synthetic embedding failure")
        return [1.0] + [0.0] * 1535

    monkeypatch.setattr(embeddings, "generate_embedding", embed)
    rejected = await host.api.capture(host.chat, [teaching, failure], [teacher.id])
    assert rejected.status == "failed"
    original = await db_session.get(MemoryTrace, original_id)
    proc = original.typed_metadata["procedural"]
    assert str(teacher.id) not in proc["source_message_ids"] and not proc.get("consumed_teachings")
    retry = await host.api.capture(host.chat, [teaching], [teacher.id])
    assert retry.decisions == ("reinforced",)
    original = await db_session.get(MemoryTrace, original_id)
    assert original.typed_metadata["procedural"]["consumed_teachings"][0]["guidance"] == teaching.rule


@pytest.mark.parametrize("operation", ["record_execution", "capture", "observe"])
async def test_saturated_public_history_rejects_without_mutating_borrowed_transaction(
    procedural_host, db_session, monkeypatch, operation
):
    from kai_memory.storage.models.memory_entity import MemoryEntity

    host = procedural_host
    assistant = Message(1, None, "example", "assistant", "Done", NOW + timedelta(seconds=1))
    host.messages[:] = [host.teacher, assistant] + [replace(host.teacher, id=uuid4()) for _ in range(998)]
    owner = MemoryEntity(id=uuid4(), entity_type="person", name="Owner sentinel")
    db_session.add(owner)
    await db_session.flush()
    owner_id = owner.id
    outer = db_session.get_transaction()
    assert outer is not None
    before_rules = set((await db_session.execute(select(MemoryTrace.id))).scalars().all())

    before_traces = set((await db_session.execute(select(ExecutionTrace.id))).scalars().all())
    with monkeypatch.context() as patch:
        commit, rollback, close = AsyncMock(), AsyncMock(), AsyncMock()
        patch.setattr(db_session, "commit", commit)
        patch.setattr(db_session, "rollback", rollback)
        patch.setattr(db_session, "close", close)
        calls = {
            "record_execution": lambda: host.api.record_execution(
                host.chat, "ask", ExecutionEvidence(assistant.id, tools=(ToolEvidence("search"),))
            ),
            "capture": lambda: host.api.capture(host.chat, [host.rule], [host.teacher.id]),
            "observe": lambda: host.api.observe(host.chat, assistant.id),
        }
        result = await calls[operation]()
        assert result.status == "rejected" and result.reason == "history_saturated"
        assert db_session.get_transaction() is outer and outer.is_active
        assert (
            await db_session.execute(select(MemoryEntity.id).where(MemoryEntity.id == owner_id))
        ).scalar_one() == owner_id
        assert set((await db_session.execute(select(MemoryTrace.id))).scalars().all()) == before_rules
        assert set((await db_session.execute(select(ExecutionTrace.id))).scalars().all()) == before_traces
        commit.assert_not_awaited()
        rollback.assert_not_awaited()
        close.assert_not_awaited()
