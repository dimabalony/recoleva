"""Deterministic turn receipts without inference, a database, or a chat engine."""

import asyncio
import uuid
from contextlib import asynccontextmanager
from dataclasses import FrozenInstanceError
from datetime import datetime, UTC
from unittest.mock import AsyncMock

import pytest

from kai_memory import turn
from kai_memory.recognition import PrimedTurn, StagedRecognition
from kai_memory.retrieval.iterative_retrieval import LIVE, RepriseState


@asynccontextmanager
async def _operation():
    yield


@pytest.fixture
def host(monkeypatch):
    staged = []
    writes = []
    trace_id = uuid.UUID(int=1)

    @asynccontextmanager
    async def session():
        yield object()

    async def recognize(_session, text, ctx, cues, *, observation, phase, input_id, defer_access):
        assert defer_access
        decision = observation.begin(phase, input_id=input_id)
        decision.complete("retrieved", retrieved=[trace_id], selected=[trace_id])
        value = StagedRecognition(PrimedTurn("synthetic recall", [trace_id], [], None), ctx, access_deferred=True)
        staged.append(value)
        return value

    async def reprise(cues, text, ctx, ids, *, observation):
        value = RepriseState(ctx, [[1.0, 0.0]], ids, LIVE, observation=observation)

        async def retrieve(summary):
            return [(str(uuid.UUID(int=2)), "synthetic mid-turn recall", 0.9)]

        monkeypatch.setattr(value, "_reprime", retrieve)
        return value

    monkeypatch.setattr(turn, "recognize", recognize)
    monkeypatch.setattr(turn, "build_window_cues", AsyncMock(return_value=None))
    monkeypatch.setattr(turn, "build_reprise_state", reprise)
    monkeypatch.setattr(StagedRecognition, "_apply_learning_writes", lambda self: writes.append(self))
    with turn.ports.scoped({"session_scope": session, "clock": lambda: datetime(2026, 1, 1, tzinfo=UTC)}):
        yield turn.TurnMemory(operation=_operation, chat_ctx=None), staged, writes


async def test_input_acknowledgement_is_separate_from_retrieval_and_commit(host):
    memory, staged, writes = host
    prepared = await memory.prepare_input("i1", "synthetic input")
    assert prepared.priming_context == "synthetic recall"
    receipt = memory.snapshot()
    decision = receipt.decisions[0]
    assert decision.retrieved_ids == decision.selected_ids == (str(uuid.UUID(int=1)),)
    assert decision.delivery == "prepared" and not decision.committed
    assert not writes
    assert memory.accept_input("i1")
    assert not memory.accept_input("i1")
    assert memory.snapshot().decisions[0].delivery == "accepted"
    assert memory.snapshot().decisions[0].committed and writes == staged
    assert decision.delivery == "prepared"  # Prior snapshots never change.
    with pytest.raises(FrozenInstanceError):
        decision.delivery = "accepted"
    await memory.close()


@pytest.mark.parametrize("uncertain,status", [(False, "rejected"), (True, "unknown")])
async def test_rejected_and_duplicate_steering_do_not_write(host, uncertain, status):
    memory, staged, writes = host
    await memory.prepare_input("i1", "initial")
    memory.accept_input("i1")
    await memory.prepare_input("s1", "steer", kind="steering")
    assert memory.reject_input("s1", uncertain=uncertain)
    assert not memory.accept_input("s1")
    duplicate = await memory.prepare_input("s1", "steer", kind="steering")
    assert duplicate.outcome == "duplicate" and duplicate.priming_context is None
    assert len(staged) == 2 and len(writes) == 1
    decisions = memory.snapshot().decisions
    assert decisions[1].delivery == status and not decisions[1].committed
    assert decisions[2].outcome == "duplicate" and not decisions[2].committed
    await memory.close()


async def test_accepted_steering_commits_once(host):
    memory, staged, writes = host
    await memory.prepare_input("i1", "initial")
    await memory.prepare_input("s1", "steer", kind="steering")
    assert memory.accept_input("s1") and not memory.accept_input("s1")
    duplicate = await memory.prepare_input("s1", "replacement", kind="steering")
    assert duplicate.outcome == "duplicate" and len(writes) == 1
    assert staged[0].outcome is None and staged[1].outcome == "committed"
    await memory.close()
    assert staged[0].outcome == "discarded"


async def test_summary_and_tool_result_are_read_only_and_ack_is_not_model_use(host):
    memory, _, writes = host
    await memory.prepare_input("i1", "initial")
    memory.consider_summary("synthetic reasoning", call_id="tool1")
    prepared = await memory.prepare_tool_result("tool1")
    assert prepared.lines and "recalled while thinking" in prepared.lines[0]
    assert memory.snapshot().decisions[-1].delivery == "prepared"
    memory.acknowledge_tool_result("tool1", written=True)
    decision = memory.snapshot().decisions[-1]
    assert decision.delivery == "written_unacknowledged" and not decision.committed
    assert decision.trigger == "tool_result"
    memory.acknowledge_tool_result("tool1", accepted=True)
    assert memory.snapshot().decisions[-1].delivery == "accepted"
    assert not writes
    assert (await memory.prepare_tool_result("tool1")).lines == ()
    await memory.close()
    assert not writes


async def test_confirmed_tool_context_acceptance_is_read_only(host):
    memory, _, writes = host
    await memory.prepare_input("i1", "initial")
    memory.consider_summary("synthetic reasoning", call_id="tool1")
    await memory.prepare_tool_result("tool1")
    memory.acknowledge_tool_result("tool1", accepted=True)
    assert memory.snapshot().decisions[-1].delivery == "accepted"
    assert not memory.snapshot().decisions[-1].committed and not writes
    await memory.close()


async def test_tool_free_turn_reports_no_injection_point(host):
    memory, _, writes = host
    await memory.prepare_input("i1", "initial")
    await memory.close(has_summary=True)
    assert memory.snapshot().decisions[-1].outcome == "no_injection_point"
    assert memory.snapshot().status == "closed" and not writes


async def test_explicit_final_summary_does_not_revise(host):
    memory, _, writes = host
    await memory.prepare_input("i1", "initial")
    memory.consider_summary("synthetic reasoning", injection_point=False)
    assert memory.snapshot().decisions[-1].outcome == "no_injection_point"
    assert not writes
    await memory.close()


async def test_close_discards_unsettled_inputs_and_fences_operations(host):
    memory, staged, writes = host
    await memory.prepare_input("i1", "initial")
    await memory.close()
    assert staged[0].outcome == "discarded" and not writes
    assert memory.snapshot().decisions[0].delivery == "discarded"
    with pytest.raises(RuntimeError, match="closed"):
        memory.accept_input("i1")
    with pytest.raises(RuntimeError, match="closed"):
        await memory.prepare_input("i2", "late")


async def test_close_joins_cancelled_retrieval_cleanup(host, monkeypatch):
    memory, _, _ = host
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    await memory.prepare_input("i1", "initial")

    async def retrieve(summary):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()
            await release.wait()

    monkeypatch.setattr(memory._reprise, "_reprime", retrieve)
    memory.consider_summary("synthetic reasoning", call_id="tool1")
    await entered.wait()
    closing = asyncio.create_task(memory.close())
    await cancelled.wait()
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert memory.snapshot().status == "closing"
    release.set()
    await memory.close()
    assert memory.snapshot().status == "closed"


async def test_tool_result_without_reasoning_summary_is_empty(host):
    memory, _, writes = host
    await memory.prepare_input("i1", "initial")
    memory.consider_summary(None, call_id="tool1")
    assert (await memory.prepare_tool_result("tool1")).lines == ()
    assert any(decision.outcome == "no_summary" for decision in memory.snapshot().decisions)
    assert not writes
    await memory.close()


async def test_duplicate_and_busy_tool_cues_do_not_replace_pending_recall(host):
    memory, _, _ = host
    await memory.prepare_input("i1", "initial")
    memory.consider_summary("one", call_id="tool1")
    memory.consider_summary("duplicate", call_id="tool1")
    assert memory.snapshot().decisions[-1].outcome == "duplicate"
    memory.consider_summary("two", call_id="tool2")
    assert memory.snapshot().decisions[-1].outcome == "busy"
    assert (await memory.prepare_tool_result("tool1")).lines
    with pytest.raises(ValueError, match="pending summary"):
        await memory.prepare_tool_result("tool2")
    await memory.close()


async def test_cleanup_uses_owning_hosts_bindings(host, monkeypatch):
    memory, _, _ = host
    bindings = {"clock": lambda: datetime(2026, 1, 1, tzinfo=UTC)}

    class Owner:
        def run_owned(self, fn):
            with turn.ports.scoped(bindings):
                return fn()

    await memory.prepare_input("i1", "initial")
    memory._owner = Owner()
    memory._owner.accepts_work = lambda: True
    entered = asyncio.Event()

    async def pending_retrieval(_summary):
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(memory._reprise, "_reprime", pending_retrieval)
    memory.consider_summary("one", call_id="tool1")
    await entered.wait()

    def wrong_clock():
        raise AssertionError("Another hosts binding leaked into cleanup")

    with turn.ports.scoped({"clock": wrong_clock}):
        await memory.close()
    assert memory.snapshot().status == "closed"


@pytest.mark.parametrize("input_id", ["", "not a token", "x" * 129])
async def test_invalid_receipt_identity_is_rejected_before_retrieval(host, input_id):
    memory, staged, _ = host
    with pytest.raises(ValueError, match="receipt token"):
        await memory.prepare_input(input_id, "synthetic input")
    assert not staged
    await memory.close()


async def test_existing_host_observation_keeps_schema_and_terminal_status(host):
    observation = turn.MemoryObservation(model="synthetic", scope="scope", input_ids=["i1"])
    memory = turn.TurnMemory(operation=_operation, chat_ctx=None, observation=observation)
    await memory.prepare_input("i1", "initial")
    memory.bind_diagnostics(thread_id="thread1", turn_id="turn1")
    assert memory.diagnostic_snapshot()["operation_id"] == observation.operation_id
    assert memory.diagnostic_snapshot()["native_thread_id"] == "thread1"
    await memory.prepare_input("s1", "steering", kind="steering")
    assert memory.diagnostic_snapshot()["decisions"][-1]["phase"] == "steered"
    counters = memory.counters
    counters["fired"] = 99
    assert memory.counters["fired"] != 99
    await memory.close(status="cancelled")
    assert observation.snapshot()["terminal_status"] == "cancelled"


async def test_owner_discard_before_turn_close_updates_receipts(host):
    memory, staged, _ = host
    await memory.prepare_input("i1", "initial")
    staged[0].discard()  # Client close may discard staged handles before its turns.
    await memory.close()
    decision = memory.snapshot().decisions[0]
    assert decision.delivery == "discarded" and not decision.committed


@pytest.mark.parametrize("uncertain,expected", [(False, "not_sent"), (True, "unknown")])
async def test_abandoned_prepared_recall_ignores_stale_acknowledgements(host, uncertain, expected):
    memory, _, writes = host
    await memory.prepare_input("i1", "initial")
    memory.consider_summary("synthetic reasoning", call_id="tool1")
    assert (await memory.prepare_tool_result("tool1")).lines
    assert not memory.abandon_tool_call("unrelated")
    assert memory.abandon_tool_call("tool1", uncertain=uncertain)
    assert not memory.abandon_tool_call("tool1")
    memory.acknowledge_tool_result("tool1", written=True, accepted=True)
    decisions = [decision for decision in memory.snapshot().decisions if decision.call_id == "tool1"]
    assert decisions and all(decision.delivery == expected for decision in decisions)
    assert not (await memory.prepare_tool_result("tool1")).lines
    assert not writes
    await memory.close()


@pytest.mark.parametrize("accepted", [False, True])
async def test_abandonment_does_not_erase_confirmed_delivery_evidence(host, accepted):
    memory, _, _ = host
    await memory.prepare_input("i1", "initial")
    memory.consider_summary("synthetic reasoning", call_id="tool1")
    await memory.prepare_tool_result("tool1")
    memory.acknowledge_tool_result("tool1", written=True, accepted=accepted)
    memory.abandon_tool_call("tool1")
    memory.acknowledge_tool_result("tool1", accepted=True)
    assert memory.snapshot().decisions[-1].delivery == ("accepted" if accepted else "written_unacknowledged")
    await memory.close()
