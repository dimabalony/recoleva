"""Public procedural contracts, using canonical synthetic history and no inference."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, UTC
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from kai_memory import ports

# Import before writer mocks so lazy signature-helper imports cannot retain them.
import kai_memory.procedural.procedural_recorder  # noqa: F401
from kai_memory.procedural import rule_capture
from kai_memory.procedural_api import (
    ExecutionEvidence,
    ProceduralMemory,
    RuleCandidate,
    ToolEvidence,
    resolve_evidence_speaker,
)
from kai_memory.sdk import Chat, Message
from kai_memory.store import cognitive_config

NOW = datetime(2026, 10, 24, tzinfo=UTC)


def message(role="user", actor=10, text="Use concise answers", **kwargs):
    return Message(
        chat_id=1,
        topic_id=None,
        venue="example",
        role=role,
        text=text,
        created_at=NOW + timedelta(seconds=kwargs.pop("offset", 0)),
        sender_user_id=actor,
        **kwargs,
    )


class Policy:
    def is_admin(self, actor):
        return actor == 10

    def respect(self, actor):
        return 100 if actor == 10 else 75 if actor == 20 else 0

    def may_read(self, chat):
        return chat.chat_id == 1

    def may_write(self, chat):
        return self.may_read(chat)


@pytest.fixture
def runtime(monkeypatch):
    import kai_memory.procedural_api as api

    state = SimpleNamespace(messages=[], rows=[], results=[])

    @asynccontextmanager
    async def operation():
        yield

    @asynccontextmanager
    async def nested():
        old_rows = list(state.rows)
        try:
            yield
        except BaseException:
            state.rows[:] = old_rows
            raise

    class ScalarRows(list):
        def all(self):
            return self

    def result(statement):
        from sqlalchemy.dialects.postgresql import array

        if "capture_key" in str(statement):
            return SimpleNamespace(scalar_one=lambda: getattr(state, "capture_available", True))
        value = state.results.pop(0) if state.results else None
        columns = getattr(statement, "selected_columns", ())
        if value is None and len(columns) and isinstance(columns[0], array):
            value = [False] * len(columns[0].clauses)
        return SimpleNamespace(
            scalar_one=lambda: value, scalar_one_or_none=lambda: value, scalars=lambda: ScalarRows(value or [])
        )

    session = SimpleNamespace(
        begin_nested=nested, execute=AsyncMock(side_effect=lambda statement, *args, **kwargs: result(statement))
    )

    @asynccontextmanager
    async def sessions():
        yield session

    monkeypatch.setattr(api, "session_scope", sessions)
    monkeypatch.setitem(ports.__dict__, "get_host_policy", lambda: Policy())
    monkeypatch.setattr(cognitive_config, "get_bool", lambda name: True)
    previous = ports.bind(
        read_conversation_page=AsyncMock(side_effect=lambda *args: state.messages),
        scrub_secrets=lambda value: value,
        validate_skill_content=lambda value: [],
        redact_urls_for_trace=lambda value: value,
        strip_msg_meta_envelope=lambda value: value,
        clock=lambda: NOW,
        is_admin_user=lambda actor, ids: actor == 10,
        senders_seen_before=AsyncMock(return_value=set()),
    )
    state.facade = ProceduralMemory(operation, AsyncMock())
    state.session = session
    try:
        yield state
    finally:
        ports.bind(**previous)


def candidate(**changes):
    return RuleCandidate("Use concise answers", "communication", "Keep answers concise", "When answering", **changes)


def test_wrong_speaker_and_ambiguous_quote_cannot_teach(runtime):
    raw = {"source_quote": "Use concise answers", "category": "communication", "rule": "Be brief", "instructor_id": 10}
    gate = rule_capture.evaluate_candidate(raw, [message(actor=99)], None, Policy().respect)
    assert not gate.accepted and gate.reason == "untrusted"
    gate = rule_capture.evaluate_candidate(raw, [message(), message(actor=99)], None, Policy().respect)
    assert not gate.accepted and gate.reason == "multi_match_untrusted"
    gate = rule_capture.evaluate_candidate(raw, [message(actor=20)], None, Policy().respect)
    assert gate.accepted and gate.instructor_id == 20 and gate.respect == 75


def test_offschema_rejection_never_logs_candidate_values_or_keys(runtime, caplog):
    secret = "recognizable-private-synthetic-text"
    raw = {"source_quote": "Use concise answers", "category": secret, "rule": secret, "trigger": secret, secret: secret}
    gate = rule_capture.evaluate_candidate(raw, [message()], None, Policy().respect)
    assert gate.reason == "offschema"
    assert secret not in caplog.text and "Use concise answers" not in caplog.text
    assert "category_not_allowed" in caplog.text


def row(person=None, venue="any", invalidated=False):
    return SimpleNamespace(
        id=uuid4(),
        invalidated_at=NOW if invalidated else None,
        typed_metadata={"procedural": {"scope": {"person": person, "venue": venue}}},
    )


def test_dedup_ignores_invalidated_and_other_person_or_venue(runtime):
    excluded = [row(invalidated=True), row(person=20), row(venue="wake")]
    assert rule_capture._dedup_decision(
        [1.0, 0.0], "+", True, [(item, [1.0, 0.0], "+") for item in excluded], {"person": 10, "venue": "any"}
    ) == ("new", None)
    matching = row(person=10)
    action, found = rule_capture._dedup_decision(
        [1.0, 0.0], "+", True, [(matching, [1.0, 0.0], "+")], {"person": 10, "venue": "any"}
    )
    assert action == "increment" and found is matching


async def test_capture_reads_history_and_returns_rejections(runtime, monkeypatch):
    import kai_memory.store.cognitive_memory as storage

    runtime.messages = [message(actor=99)]
    monkeypatch.setattr(rule_capture, "_load_existing_rules", AsyncMock(side_effect=lambda session: []))
    writer = AsyncMock()
    monkeypatch.setattr(storage, "CognitiveMemoryService", lambda session: writer)
    result = await runtime.facade.capture(
        Chat(chat_id=1, venue="example", sender_user_id=99), [candidate()], [runtime.messages[0].id]
    )
    assert result.status == "completed" and result.changed == 0 and result.decisions == ("rejected:untrusted",)
    writer.store_procedural_rule.assert_not_awaited()
    result = await runtime.facade.capture(Chat(chat_id=2, venue="example"), [candidate()], [runtime.messages[0].id])
    assert result.reason == "scope_denied"
    result = await runtime.facade.capture(Chat(chat_id=1, venue="example"), [candidate()], [uuid4()])
    assert result.reason == "source_anchor"


async def test_capture_failure_rolls_back_reports_failure_and_can_retry(runtime, monkeypatch):
    import kai_memory.store.cognitive_memory as storage
    import kai_memory.store.embeddings as embeddings

    runtime.messages = [message()]
    monkeypatch.setattr(rule_capture, "_load_existing_rules", AsyncMock(side_effect=lambda session: []))
    calls = 0

    async def embed(value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("scripted embedding failure")
        return [1.0, 0.0] if calls % 2 else [0.0, 1.0]

    async def store(content, **kwargs):
        item = SimpleNamespace(id=uuid4(), invalidated_at=None, typed_metadata=kwargs["typed_metadata"])
        runtime.rows.append(item)
        return item

    monkeypatch.setattr(embeddings, "generate_embedding", embed)
    monkeypatch.setattr(storage, "CognitiveMemoryService", lambda session: SimpleNamespace(store_procedural_rule=store))
    chat = Chat(chat_id=1, venue="example", sender_user_id=10)
    from dataclasses import replace

    second = replace(candidate(), trigger="When searching", rule="Use search results")
    result = await runtime.facade.capture(chat, [candidate(), second], [runtime.messages[0].id])
    assert result.status == "failed" and result.changed == 0 and not runtime.rows
    result = await runtime.facade.capture(chat, [candidate()], [runtime.messages[0].id])
    assert result.status == "completed" and result.changed == 1 and len(runtime.rows) == 1


async def test_same_batch_supersession_matches_only_survivors(runtime, monkeypatch):
    import kai_memory.store.cognitive_memory as storage
    import kai_memory.store.embeddings as embeddings

    runtime.messages = [message()]
    old = row()
    old.typed_metadata["procedural"]["guidance"] = "Use concise answers"
    monkeypatch.setattr(rule_capture, "_load_existing_rules", AsyncMock(return_value=[(old, [1.0, 0.0], "+")]))
    monkeypatch.setattr(embeddings, "generate_embedding", AsyncMock(return_value=[1.0, 0.0]))
    invalidated = []

    async def invalidate(value, **kwargs):
        invalidated.append(value)

    async def store(content, **kwargs):
        item = SimpleNamespace(id=uuid4(), invalidated_at=None, typed_metadata=kwargs["typed_metadata"])
        runtime.rows.append(item)
        return item

    monkeypatch.setattr(
        storage,
        "CognitiveMemoryService",
        lambda session: SimpleNamespace(store_procedural_rule=store, invalidate_trace=invalidate),
    )
    # Opposite instruction replaces old, then the latest positive instruction
    # must replace that survivor rather than reinforcing the already dead old row.
    inputs = [candidate(directive_strength="instruction"), candidate(directive_strength="instruction")]
    from dataclasses import replace

    inputs[0] = replace(inputs[0], rule="Do not use concise answers")
    result = await runtime.facade.capture(
        Chat(chat_id=1, venue="example", sender_user_id=10), inputs, [runtime.messages[0].id]
    )
    assert result.status == "completed" and result.decisions == ("superseded", "superseded")
    assert invalidated == [old.id, runtime.rows[0].id]


async def test_tool_free_does_not_invent_trace_and_wrong_actor_rejected(runtime, monkeypatch):
    from kai_memory.procedural import procedural_recorder

    recorder = AsyncMock()
    monkeypatch.setattr(procedural_recorder, "record_execution_trace", recorder)
    rid = uuid4()
    assistant = message(role="assistant", actor=None, offset=1, message_metadata={"served_rule_ids": [str(rid)]})
    runtime.messages = [message(), assistant]
    evidence = ExecutionEvidence(assistant.id, served_rule_ids=(rid,))
    result = await runtime.facade.record_execution(Chat(chat_id=1, venue="example", sender_user_id=10), "ask", evidence)
    assert result.status == "tool_free" and result.execution_trace_id is None
    runtime.session.execute.assert_not_awaited()
    result = await runtime.facade.record_execution(Chat(chat_id=1, venue="example", sender_user_id=99), "ask", evidence)
    assert result.status == "rejected" and result.reason == "assistant_anchor"
    recorder.assert_not_awaited()


async def test_recorded_execution_is_idempotent_and_does_not_imply_following(runtime, monkeypatch):
    from kai_memory.procedural import procedural_recorder

    assistant = message(role="assistant", actor=None, offset=1)
    runtime.messages = [message(), assistant]
    trace_id = uuid4()

    async def record(session, chat, metadata, task_text, **kwargs):
        assert "rule_compliance" not in metadata
        metadata["execution_trace_id"] = str(trace_id)

    recorder = AsyncMock(side_effect=record)
    monkeypatch.setattr(procedural_recorder, "record_execution_trace", recorder)
    evidence = ExecutionEvidence(assistant.id, tools=(ToolEvidence("search", ("query",)),))
    chat = Chat(chat_id=1, venue="example", sender_user_id=10)
    result = await runtime.facade.record_execution(chat, "ask", evidence)
    assert result.status == "recorded" and result.execution_trace_id == trace_id
    runtime.results = [None, SimpleNamespace(id=trace_id)]
    result = await runtime.facade.record_execution(chat, "ask", evidence)
    assert result.status == "duplicate" and result.execution_trace_id == trace_id
    assert recorder.await_count == 1


def test_evidence_wrong_speaker_forwards_reply_quotes_and_ambiguity(runtime):
    assert resolve_evidence_speaker("great", [message(actor=99, text="great")]) == 99
    assert resolve_evidence_speaker("great", [message(text="great"), message(actor=99, text="great")]) is None
    assert resolve_evidence_speaker("great", [message(text="great", message_metadata={"forwarded": True})]) is None
    assert resolve_evidence_speaker("great", [message(text="[Replying to X: great]")]) is None


async def test_wrong_speaker_cannot_set_outcome(runtime):
    assistant = message(role="assistant", actor=None, offset=1)
    runtime.messages = [message(), assistant, message(actor=99, text="great", offset=2)]
    trace = SimpleNamespace(source_user_id=10, conversation_ref="example:1:None", outcome="unknown")
    runtime.results = [trace]
    result = await runtime.facade.observe(
        Chat(chat_id=1, venue="example", sender_user_id=10), assistant.id, outcome="success", evidence_quote="great"
    )
    assert result.status == "rejected" and result.reason == "evidence_speaker" and trace.outcome == "unknown"


async def test_serving_only_returns_exposure_and_denied_scope_is_empty(runtime, monkeypatch):
    from kai_memory.procedural import procedural_serving

    rid = uuid4()
    serve = AsyncMock(return_value=("standing guidance", [rid]))
    monkeypatch.setattr(procedural_serving, "get_learned_behaviors", serve)
    result = await runtime.facade.serve(Chat(chat_id=1, venue="example"))
    assert result.block == "standing guidance" and result.served_rule_ids == (rid,)
    denied = await runtime.facade.serve(Chat(chat_id=2, venue="example"))
    assert denied.block is None and denied.status == "scope_denied" and serve.await_count == 1


async def test_history_cursor_is_forwarded_and_missing_trigger_fails_closed(runtime):
    assistant = message(role="assistant", actor=None, offset=1)
    runtime.messages = [assistant]
    cursor = object()
    result = await runtime.facade.record_execution(
        Chat(chat_id=1, venue="example"), "ask", ExecutionEvidence(assistant.id), history_after=cursor
    )
    assert result.status == "rejected" and result.reason == "assistant_anchor"
    assert ports.read_conversation_page.await_args.args[2] is cursor


async def test_plain_turn_compliance_is_idempotent_and_scope_checked(runtime):
    rid = uuid4()
    assistant = message(role="assistant", actor=None, offset=1, message_metadata={"served_rule_ids": [str(rid)]})
    runtime.messages = [message(), assistant]
    rule = row(person=10)
    rule.id = rid
    rule.typed_metadata["procedural"]["kind"] = "rule"
    # trace lookup, eligibility read, message-anchored writer read.
    runtime.results = [None, [rule], [rule]]
    chat = Chat(chat_id=1, venue="example", sender_user_id=10)
    result = await runtime.facade.observe(chat, assistant.id, compliance={rid: "followed"})
    assert result.status == "recorded" and result.compliance_updated == 1
    assert "utility" not in rule.typed_metadata["procedural"]
    runtime.results = [None, [rule], [rule]]
    result = await runtime.facade.observe(chat, assistant.id, compliance={rid: "violated"})
    assert result.status == "duplicate" and result.compliance_updated == 0
    assert rule.typed_metadata["procedural"]["compliance_events"][0]["verdict"] == "followed"
    rule.typed_metadata["procedural"]["scope"]["person"] = 99
    runtime.results = [None, [rule]]
    result = await runtime.facade.observe(chat, assistant.id, compliance={rid: "violated"})
    assert result.status == "rejected" and result.reason == "rule_scope"


async def test_outcome_requires_followup_evidence_not_prior_praise(runtime):
    assistant = message(role="assistant", actor=None, offset=1)
    runtime.messages = [message(text="great"), assistant]
    trace = SimpleNamespace(source_user_id=10, conversation_ref="example:1:None", outcome="unknown")
    runtime.results = [trace]
    result = await runtime.facade.observe(
        Chat(chat_id=1, venue="example", sender_user_id=10), assistant.id, outcome="success", evidence_quote="great"
    )
    assert result.reason == "evidence_speaker" and trace.outcome == "unknown"


async def test_serving_failure_is_observable(runtime, monkeypatch):
    from kai_memory.procedural import procedural_serving

    monkeypatch.setattr(procedural_serving, "get_learned_behaviors", AsyncMock(side_effect=RuntimeError("store down")))
    result = await runtime.facade.serve(Chat(chat_id=1, venue="example"))
    assert result.status == "failed" and result.block is None


def test_multi_teacher_source_message_belongs_to_instructor(runtime):
    admin = message()
    roster = message(actor=20)
    raw = {"source_quote": "Use concise answers", "category": "communication", "rule": "Be brief"}
    gate = rule_capture.evaluate_candidate(raw, [admin, roster], None, Policy().respect)
    assert gate.accepted and gate.instructor_id == 20 and gate.respect == 75
    assert gate.matched_message_id == str(roster.id)


async def test_public_replay_of_retired_teaching_is_duplicate_before_embedding(runtime, monkeypatch):
    import kai_memory.store.embeddings as embeddings

    runtime.messages = [message()]
    runtime.results = [[True]]  # prior source/guidance/scope match, including invalidated rows
    monkeypatch.setattr(rule_capture, "_load_existing_rules", AsyncMock(side_effect=lambda session: []))
    embed = AsyncMock(side_effect=AssertionError("duplicate must not embed"))
    monkeypatch.setattr(embeddings, "generate_embedding", embed)
    result = await runtime.facade.capture(
        Chat(chat_id=1, venue="example", sender_user_id=10), [candidate()], [runtime.messages[0].id]
    )
    assert result.status == "completed" and result.changed == 0 and result.decisions == ("duplicate",)
    embed.assert_not_awaited()


def test_canonical_trigger_skips_tool_results_and_preserves_explicit_origin(runtime):
    from kai_memory.procedural_api import _assistant_anchor

    original = message(actor=10)
    steerer = message(actor=99, offset=1)
    tool = message(actor=None, offset=2, message_metadata={"source_kind": "tool_result"})
    assistant = message(
        role="assistant", actor=None, offset=3, message_metadata={"procedural_origin_message_id": str(original.id)}
    )
    assert _assistant_anchor(
        [original, steerer, tool, assistant], assistant.id, Chat(chat_id=1, venue="example", sender_user_id=10)
    ) == (assistant, 10)
    assistant.message_metadata = {}
    assert _assistant_anchor(
        [original, tool, assistant], assistant.id, Chat(chat_id=1, venue="example", sender_user_id=10)
    ) == (assistant, 10)


def test_shared_group_canonical_attribution_stays_null(runtime):
    from kai_memory.procedural_api import _assistant_anchor

    original = message(actor=10)
    assistant = message(
        role="assistant",
        actor=None,
        offset=1,
        message_metadata={"native_procedural_attribution": "shared_group_context"},
    )
    assert _assistant_anchor(
        [original, assistant], assistant.id, Chat(chat_id=1, venue="example", is_group=True, sender_user_id=None)
    ) == (assistant, None)
    assert (
        _assistant_anchor(
            [original, assistant], assistant.id, Chat(chat_id=1, venue="example", is_group=True, sender_user_id=10)
        )
        is None
    )


@pytest.mark.parametrize("exposure", ["prior", "future", "absent"])
async def test_public_reteaching_uses_prior_canonical_exposure_once(runtime, monkeypatch, exposure):
    import kai_memory.store.embeddings as embeddings

    existing = row()
    proc = existing.typed_metadata["procedural"]
    proc.update(
        {
            "kind": "rule",
            "status": "active",
            "guidance": "Keep answers concise",
            "source_message_ids": [str(uuid4())],
            "evidence_count": 1,
            "utility": {"s": 0, "f": 0, "event_keys": []},
        }
    )
    reteacher = message(offset=2)
    # This earlier stranger's identical quote is exposure history, never a teaching anchor.
    stranger = message(actor=99, offset=0)
    assistant = message(
        role="assistant",
        actor=None,
        offset=1 if exposure == "prior" else 3,
        message_metadata={"served_rule_ids": [str(existing.id)]},
    )
    runtime.messages = [stranger, reteacher] if exposure == "absent" else [stranger, assistant, reteacher]
    monkeypatch.setattr(
        rule_capture, "_load_existing_rules", AsyncMock(side_effect=lambda session: [(existing, [1.0, 0.0], "pos")])
    )
    monkeypatch.setattr(embeddings, "generate_embedding", AsyncMock(return_value=[1.0, 0.0]))
    chat = Chat(chat_id=1, venue="example", sender_user_id=10)
    result = await runtime.facade.capture(chat, [candidate()], [reteacher.id])
    assert result.status == "completed" and result.decisions == ("reinforced",) and result.changed == 1
    proc = existing.typed_metadata["procedural"]
    assert proc["evidence_count"] == 2 and proc["recurrence_count"] == 1
    assert proc.get("recurrence_events", []) == ([str(reteacher.id)] if exposure == "prior" else [])
    assert proc["utility"] == {"s": 0, "f": 0, "event_keys": []}  # scoring remains nightly
    result = await runtime.facade.capture(chat, [candidate()], [reteacher.id])
    assert result.decisions == ("duplicate",) and result.changed == 0
    proc = existing.typed_metadata["procedural"]
    assert proc["evidence_count"] == 2 and proc["recurrence_count"] == 1
    assert proc.get("recurrence_events", []) == ([str(reteacher.id)] if exposure == "prior" else [])


@pytest.mark.parametrize("verdicts", [("misapplied",), ("followed", "misapplied"), ("misapplied", "violated")])
async def test_tool_free_unrecordable_verdict_rejects_entire_batch_before_writes(runtime, monkeypatch, verdicts):
    from kai_memory.procedural import rule_capture

    ids = [uuid4() for _ in verdicts]
    assistant = message(
        role="assistant", actor=None, offset=1, message_metadata={"served_rule_ids": list(map(str, ids))}
    )
    runtime.messages = [message(), assistant]
    runtime.results = [None]  # no execution trace: tool-free message anchoring applies
    writer = AsyncMock(side_effect=AssertionError("rejected batch must not write"))
    monkeypatch.setattr(rule_capture, "record_message_anchored_compliance", writer)
    result = await runtime.facade.observe(
        Chat(chat_id=1, venue="example", sender_user_id=10),
        assistant.id,
        compliance=dict(zip(ids, verdicts, strict=True)),
    )
    assert result.status == "rejected" and result.reason == "verdict_not_recordable"
    assert not result.outcome_updated and result.compliance_updated == 0
    assert runtime.session.execute.await_count == 1  # no eligibility reads or pending-event writes
    writer.assert_not_awaited()


async def test_execution_trace_keeps_misapplied_observation_and_exact_replay_is_duplicate(runtime):
    rid = uuid4()
    assistant = message(role="assistant", actor=None, offset=1, message_metadata={"served_rule_ids": [str(rid)]})
    runtime.messages = [message(), assistant]
    rule = row(person=10)
    rule.id = rid
    rule.typed_metadata["procedural"]["kind"] = "rule"
    trace = SimpleNamespace(source_user_id=10, conversation_ref="example:1:None", success_signals={})
    chat = Chat(chat_id=1, venue="example", sender_user_id=10)
    runtime.results = [trace, [rule]]
    result = await runtime.facade.observe(chat, assistant.id, compliance={rid: "misapplied"})
    assert result.status == "recorded" and result.compliance_updated == 1
    assert trace.success_signals["rule_compliance"] == {str(rid): "misapplied"}
    runtime.results = [trace, [rule]]
    result = await runtime.facade.observe(chat, assistant.id, compliance={rid: "misapplied"})
    assert result.status == "duplicate" and result.compliance_updated == 0
    assert trace.success_signals["rule_compliance"] == {str(rid): "misapplied"}


@pytest.mark.parametrize("verdict", ["followed", "violated"])
async def test_tool_free_exact_existing_observation_remains_duplicate(runtime, verdict):
    rid = uuid4()
    assistant = message(role="assistant", actor=None, offset=1, message_metadata={"served_rule_ids": [str(rid)]})
    runtime.messages = [message(), assistant]
    rule = row(person=10)
    rule.id = rid
    event = {"message_id": str(assistant.id), "verdict": verdict, "at": NOW.isoformat()}
    rule.typed_metadata["procedural"].update({"kind": "rule", "compliance_events": [event]})
    runtime.results = [None, [rule], [rule]]
    result = await runtime.facade.observe(
        Chat(chat_id=1, venue="example", sender_user_id=10), assistant.id, compliance={rid: verdict}
    )
    assert result.status == "duplicate" and result.compliance_updated == 0
    assert rule.typed_metadata["procedural"]["compliance_events"] == [event]


@pytest.mark.parametrize("operation", ["record_execution", "capture", "serve", "observe"])
async def test_public_failure_diagnostic_contains_only_event_and_exception_type(
    runtime, monkeypatch, caplog, operation
):
    import kai_memory.procedural_api as api

    private_text = "recognizable-private-conversation-and-credential-value"

    @asynccontextmanager
    async def failing_sessions():
        raise RuntimeError(private_text)
        yield  # pragma: no cover -- async context manager must fail on entry

    monkeypatch.setattr(api, "session_scope", failing_sessions)
    chat = Chat(chat_id=1, venue="example", sender_user_id=10)
    assistant_id = uuid4()
    calls = {
        "record_execution": lambda: runtime.facade.record_execution(
            chat, private_text, ExecutionEvidence(assistant_id)
        ),
        "capture": lambda: runtime.facade.capture(chat, [candidate()], [uuid4()]),
        "serve": lambda: runtime.facade.serve(chat),
        "observe": lambda: runtime.facade.observe(chat, assistant_id),
    }
    result = await calls[operation]()
    assert result.status == "failed"
    records = [record for record in caplog.records if record.name == "kai_memory.procedural_api"]
    assert len(records) == 1
    assert records[0].getMessage() == f"procedural.public.{operation}.failed type=RuntimeError"
    assert records[0].exc_info is None and records[0].stack_info is None
    assert private_text not in caplog.text and "Traceback" not in caplog.text


@pytest.mark.parametrize("person,venue", [(None, "any"), (10, "wake"), (10, None)])
def test_replay_scope_uses_typed_jsonb_identity_and_ignores_group_safe(person, venue):
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import JSONB, dialect
    from sqlalchemy.sql.elements import BindParameter
    from sqlalchemy.sql.visitors import iterate
    from kai_memory.storage.models.memory_trace import MemoryTrace

    column = MemoryTrace.typed_metadata["procedural"]["scope"]
    scope = {"person": person, "venue": venue, "group_safe": False}
    conditions = rule_capture._replay_scope_identity(column, scope)
    flipped = rule_capture._replay_scope_identity(column, {**scope, "group_safe": True})
    assert select(1).where(*conditions).compare(select(1).where(*flipped))
    values = [
        item.value
        for condition in conditions
        for item in iterate(condition)
        if isinstance(item, BindParameter) and isinstance(item.type, JSONB)
    ]
    import json

    assert sorted(map(json.dumps, values)) == sorted(map(json.dumps, [None, person, "any", venue]))
    processor = JSONB().bind_processor(dialect())
    assert processor(None) == "null" and processor(10) == "10" and processor("10") == '"10"'
    compiled = str(select(1).where(*conditions).compile(dialect=dialect()))
    assert "->>" not in compiled  # text coercion must not merge integer and string actors
    assert " IS NULL" not in compiled  # explicit JSON null is a typed JSONB value


async def test_public_replay_batches_more_than_64_distinct_keys_without_truncation(runtime, monkeypatch):
    from dataclasses import replace
    from kai_memory.store import embeddings

    runtime.messages = [message()]
    count = rule_capture._REPLAY_BATCH_SIZE + 1
    inputs = [replace(candidate(), rule=f"Use response style {index}") for index in range(count)]
    runtime.results = [[True] * rule_capture._REPLAY_BATCH_SIZE, [True]]
    monkeypatch.setattr(rule_capture, "_load_existing_rules", AsyncMock(return_value=[]))
    embed = AsyncMock(side_effect=AssertionError("old tombstones must prevent embeddings"))
    monkeypatch.setattr(embeddings, "generate_embedding", embed)
    result = await runtime.facade.capture(
        Chat(chat_id=1, venue="example", sender_user_id=10), inputs, [runtime.messages[0].id]
    )
    assert result.status == "completed" and result.changed == 0 and result.decisions == ("duplicate",) * count
    queries = [
        call.args[0] for call in runtime.session.execute.await_args_list if "capture_key" not in str(call.args[0])
    ]
    assert len(queries) == 2
    assert [len(query.selected_columns[0].clauses) for query in queries] == [64, 1]
    assert all("invalidated_at" not in str(query) and "LIMIT" not in str(query) for query in queries)
    embed.assert_not_awaited()


async def test_public_same_batch_tombstones_preserve_minted_superseded_replay(runtime, monkeypatch):
    from dataclasses import replace
    import kai_memory.store.cognitive_memory as storage
    from kai_memory.store import embeddings

    first = message(text="Positive teaching")
    second = message(text="Negative teaching", offset=1)
    runtime.messages = [first, second]
    positive = replace(candidate(directive_strength="instruction"), source_quote=first.text)
    negative = replace(positive, source_quote=second.text, rule="Do not keep answers concise")
    monkeypatch.setattr(rule_capture, "_load_existing_rules", AsyncMock(return_value=[]))
    monkeypatch.setattr(embeddings, "generate_embedding", AsyncMock(return_value=[1.0, 0.0]))

    async def store(content, **kwargs):
        item = SimpleNamespace(id=uuid4(), invalidated_at=None, typed_metadata=kwargs["typed_metadata"])
        runtime.rows.append(item)
        return item

    async def invalidate(identity, **kwargs):
        next(item for item in runtime.rows if item.id == identity).invalidated_at = NOW

    monkeypatch.setattr(
        storage,
        "CognitiveMemoryService",
        lambda session: SimpleNamespace(store_procedural_rule=store, invalidate_trace=invalidate),
    )
    result = await runtime.facade.capture(
        Chat(chat_id=1, venue="example", sender_user_id=10), [positive, negative, positive], [first.id, second.id]
    )
    assert result.decisions == ("minted", "superseded", "duplicate") and len(runtime.rows) == 2
    assert runtime.rows[0].invalidated_at is not None and runtime.rows[1].invalidated_at is None
    assert runtime.session.execute.await_count == 2  # transaction guard plus replay batch


async def test_reinforcement_persists_candidate_identity_through_same_batch_retirement(runtime, monkeypatch):
    from dataclasses import replace
    import kai_memory.store.cognitive_memory as storage
    from kai_memory.store import embeddings

    first = message(text="Positive teaching")
    second = message(text="Negative teaching", offset=1)
    runtime.messages = [first, second]
    original = row()
    original.typed_metadata["procedural"].update(
        {
            "kind": "rule",
            "status": "active",
            "guidance": "Keep answers concise",
            "trigger_pattern": "When answering",
            "source_message_ids": [str(uuid4())],
        }
    )
    runtime.rows = [original]
    positive = replace(candidate(directive_strength="instruction"), source_quote=first.text, rule="Use brief responses")
    negative = replace(positive, source_quote=second.text, rule="Do not use brief responses")
    monkeypatch.setattr(rule_capture, "_load_existing_rules", AsyncMock(return_value=[(original, [1.0, 0.0], "pos")]))
    monkeypatch.setattr(embeddings, "generate_embedding", AsyncMock(return_value=[1.0, 0.0]))

    async def store(content, **kwargs):
        item = SimpleNamespace(id=uuid4(), invalidated_at=None, typed_metadata=kwargs["typed_metadata"])
        runtime.rows.append(item)
        return item

    async def invalidate(identity, **kwargs):
        next(item for item in runtime.rows if item.id == identity).invalidated_at = NOW

    monkeypatch.setattr(
        storage,
        "CognitiveMemoryService",
        lambda session: SimpleNamespace(store_procedural_rule=store, invalidate_trace=invalidate),
    )
    result = await runtime.facade.capture(
        Chat(chat_id=1, venue="example", sender_user_id=10), [positive, negative, positive], [first.id, second.id]
    )
    assert result.decisions == ("reinforced", "superseded", "duplicate")
    assert runtime.rows[-1].typed_metadata["procedural"]["guidance"] == negative.rule
    assert original.typed_metadata["procedural"]["guidance"] == "Keep answers concise"
    assert original.typed_metadata["procedural"]["consumed_teachings"] == [
        {
            "source_message_id": str(first.id),
            "guidance": positive.rule,
            "trigger_pattern": positive.trigger,
            "scope": {"person": None, "venue": "any"},
        }
    ]


async def test_rejected_candidates_do_not_run_replay_queries(runtime, monkeypatch):
    runtime.messages = [message(actor=99)]
    monkeypatch.setattr(rule_capture, "_load_existing_rules", AsyncMock(return_value=[]))
    result = await runtime.facade.capture(
        Chat(chat_id=1, venue="example", sender_user_id=99), [candidate(), candidate()], [runtime.messages[0].id]
    )
    assert result.decisions == ("rejected:untrusted", "rejected:untrusted")
    runtime.session.execute.assert_awaited_once()
    assert "capture_key" in str(runtime.session.execute.await_args.args[0])


async def test_batched_capture_keeps_rejection_and_mutation_diagnostic_order(runtime, monkeypatch, caplog):
    from dataclasses import replace
    import kai_memory.store.cognitive_memory as storage
    from kai_memory.store import embeddings

    runtime.messages = [message()]
    monkeypatch.setattr(rule_capture, "_load_existing_rules", AsyncMock(return_value=[]))
    monkeypatch.setattr(embeddings, "generate_embedding", AsyncMock(return_value=[1.0, 0.0]))

    async def store(content, **kwargs):
        return SimpleNamespace(id=uuid4(), invalidated_at=None, typed_metadata=kwargs["typed_metadata"])

    monkeypatch.setattr(storage, "CognitiveMemoryService", lambda session: SimpleNamespace(store_procedural_rule=store))
    with caplog.at_level("INFO", logger=rule_capture.logger.name):
        result = await runtime.facade.capture(
            Chat(chat_id=1, venue="example", sender_user_id=10),
            [candidate(), replace(candidate(), category="unknown")],
            [runtime.messages[0].id],
        )
    assert result.decisions == ("minted", "rejected:offschema")
    events = [record.getMessage().split()[0] for record in caplog.records if record.name == rule_capture.logger.name]
    assert events == ["procedural.rule.minted", "procedural.rule.rejected_offschema", "procedural.rule.rejected"]


@pytest.mark.parametrize("public", [True, False])
@pytest.mark.parametrize("shape", ["list", "dict"])
async def test_malformed_category_rejection_preserves_valid_neighbors_and_log_privacy(
    runtime, monkeypatch, caplog, public, shape
):
    from dataclasses import asdict, replace
    import kai_memory.store.cognitive_memory as storage
    from kai_memory.store import embeddings

    private_text = "recognizable-private-category-content"
    malformed = [private_text] if shape == "list" else {private_text: private_text}
    inputs = [candidate(), replace(candidate(), category=malformed)]
    runtime.messages = [message()]
    monkeypatch.setattr(rule_capture, "_load_existing_rules", AsyncMock(return_value=[]))
    monkeypatch.setattr(embeddings, "generate_embedding", AsyncMock(return_value=[1.0, 0.0]))

    async def store(content, **kwargs):
        stored = SimpleNamespace(id=uuid4(), invalidated_at=None, typed_metadata=kwargs["typed_metadata"])
        runtime.rows.append(stored)
        return stored

    monkeypatch.setattr(storage, "CognitiveMemoryService", lambda session: SimpleNamespace(store_procedural_rule=store))
    chat = Chat(chat_id=1, venue="example", sender_user_id=10)
    with caplog.at_level("INFO", logger=rule_capture.logger.name):
        if public:
            result = await runtime.facade.capture(chat, inputs, [runtime.messages[0].id])
            assert result.status == "completed"
            changed, decisions = result.changed, result.decisions
        else:
            decisions = []
            changed = await rule_capture.capture_rules(
                runtime.session,
                chat,
                [asdict(value) for value in inputs],
                runtime.messages,
                None,
                respect_of=Policy().respect,
                decisions=decisions,
            )
    assert changed == 1 and tuple(decisions) == ("minted", "rejected:offschema")
    assert len(runtime.rows) == 1 and runtime.rows[0].typed_metadata["procedural"]["category"] == "communication"
    assert private_text not in caplog.text
    assert "reason=offschema category=invalid" in caplog.text


@pytest.mark.parametrize("operation", ["record_execution", "observe"])
async def test_execution_anchor_lookup_keeps_indexable_key_literal(runtime, operation):
    from sqlalchemy.dialects.postgresql.asyncpg import dialect

    assistant = message(role="assistant", actor=None, offset=1)
    runtime.messages = [message(), assistant]
    trace = SimpleNamespace(id=uuid4(), source_user_id=10, conversation_ref="example:1:None")
    chat = Chat(chat_id=1, venue="example", sender_user_id=10)
    if operation == "record_execution":
        runtime.results = [None, trace]  # advisory lock, then existing execution
        result = await runtime.facade.record_execution(
            chat, "ask", ExecutionEvidence(assistant.id, tools=(ToolEvidence("search"),))
        )
    else:
        runtime.results = [trace]
        result = await runtime.facade.observe(chat, assistant.id)
    assert result.status == "duplicate"
    statement = runtime.session.execute.await_args.args[0]
    compiled = statement.compile(dialect=dialect())
    assert list(compiled.params.values()) == [str(assistant.id)]
    assert "->> 'assistant_message_id') IS NOT NULL" in str(compiled)
    assert "->> 'assistant_message_id') = $1::VARCHAR" in str(compiled)
    assert str(compiled).endswith("FOR UPDATE") is (operation == "observe")


@pytest.mark.parametrize("mismatch", [None, "actor", "ref", "venue"])
async def test_headless_observation_retains_actor_and_venue_scope(runtime, monkeypatch, mismatch):
    from dataclasses import replace

    monkeypatch.setitem(
        ports.__dict__,
        "get_host_policy",
        lambda: SimpleNamespace(
            may_read=lambda chat: True, may_write=lambda chat: True, is_admin=lambda actor: actor == 10
        ),
    )
    chat = Chat(chat_id=None, venue="event", sender_user_id=10)
    assistant = replace(message(role="assistant", actor=None, offset=1), chat_id=None, venue="event")
    runtime.messages = [
        replace(message(), chat_id=None, venue="event"),
        assistant,
        replace(message(text="great", offset=2), chat_id=None, venue="event"),
    ]
    trace = SimpleNamespace(
        source_user_id=99 if mismatch == "actor" else 10,
        conversation_ref="event:1:None" if mismatch == "ref" else None,
        success_signals={"venue": "wake" if mismatch == "venue" else "event"},
        outcome="unknown",
    )
    runtime.results = [trace]
    result = await runtime.facade.observe(chat, assistant.id, outcome="success", evidence_quote="great")
    if mismatch:
        assert result.status == "rejected" and result.reason == "trace_scope"
        assert trace.outcome == "unknown"
    else:
        assert result.status == "recorded" and result.outcome_updated


@pytest.mark.parametrize("enabled, inputs", [(False, [{"rule": "unused"}]), (True, [])])
async def test_disabled_or_empty_legacy_capture_does_not_acquire_guard(runtime, monkeypatch, enabled, inputs):
    monkeypatch.setattr(cognitive_config, "get_bool", lambda name: enabled)
    assert await rule_capture.capture_rules(runtime.session, Chat(chat_id=1, venue="example"), inputs, [], None) == 0
    runtime.session.execute.assert_not_awaited()


@pytest.mark.parametrize("telemetry_failure", [False, True])
async def test_busy_public_capture_is_explicit_and_does_not_read_or_mutate_rules(
    runtime, monkeypatch, telemetry_failure
):
    runtime.messages = [message()]
    runtime.capture_available = False
    load = AsyncMock(side_effect=AssertionError("busy capture must not load rules"))
    monkeypatch.setattr(rule_capture, "_load_existing_rules", load)
    counter = Mock(side_effect=RuntimeError("synthetic telemetry failure") if telemetry_failure else None)
    with ports.scoped({"telemetry": SimpleNamespace(counter=counter)}, inherit=True):
        result = await runtime.facade.capture(
            Chat(chat_id=1, venue="example", sender_user_id=10), [candidate()], [runtime.messages[0].id]
        )
    counter.assert_called_once_with("memory.procedural.capture_busy", 1)
    assert result.status == "failed" and result.reason == "capture_busy"
    assert result.changed == 0 and result.decisions == () and runtime.rows == []
    load.assert_not_awaited()


@pytest.mark.parametrize("operation", ["record_execution", "capture", "observe"])
@pytest.mark.parametrize("window", [999, 1000, 1001, "wrong_scope"])
async def test_history_saturation_has_specific_rejection_before_writes(runtime, monkeypatch, operation, window):
    from dataclasses import replace

    teacher = message()
    assistant = message(role="assistant", actor=None, offset=2000)
    runtime.messages = [teacher, assistant] + [
        message(offset=index + 1) for index in range((window if isinstance(window, int) else 1000) - 2)
    ]
    if window == "wrong_scope":
        runtime.messages[0] = replace(teacher, venue="another-host")
    capture = AsyncMock(return_value=0)
    monkeypatch.setattr(rule_capture, "capture_rules", capture)
    runtime.session.commit = AsyncMock()
    runtime.session.rollback = AsyncMock()
    runtime.session.close = AsyncMock()
    chat = Chat(chat_id=1, venue="example", sender_user_id=10)
    calls = {
        "record_execution": lambda: runtime.facade.record_execution(chat, "ask", ExecutionEvidence(assistant.id)),
        "capture": lambda: runtime.facade.capture(chat, [candidate()], [teacher.id]),
        "observe": lambda: runtime.facade.observe(chat, assistant.id),
    }
    result = await calls[operation]()
    if window == 999:
        assert (
            result.status
            == {"record_execution": "tool_free", "capture": "completed", "observe": "duplicate"}[operation]
        )
    else:
        assert result.status == ("failed" if window == "wrong_scope" else "rejected")
        assert (result.reason == "history_saturated") is (window != "wrong_scope")
        runtime.session.execute.assert_not_awaited()
        capture.assert_not_awaited()
        assert runtime.rows == []
    runtime.session.commit.assert_not_awaited()
    runtime.session.rollback.assert_not_awaited()
    runtime.session.close.assert_not_awaited()
