"""Actual SDK ownership, authority and deadlines around synthetic retrieval."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from kai_memory import recognition, turn
from kai_memory.ownership import current_owner
from kai_memory.sdk import Chat, HostAdapters, MemoryClient, Message
from kai_memory.recognition import PrimedTurn
from kai_memory.store import cognitive_config


class Policy:
    read = True
    write = True

    def may_read(self, chat):
        return self.read

    def may_write(self, chat):
        return self.write

    def is_admin(self, actor):
        return True

    def respect(self, actor):
        return 100


@pytest.fixture
async def runtime(monkeypatch):
    policy = Policy()
    trace_id = uuid4()
    tracked = MagicMock()
    writes = AsyncMock()
    reprise = SimpleNamespace(cancel=MagicMock(), close=AsyncMock(), counters={})

    @asynccontextmanager
    async def session():
        yield object()

    monkeypatch.setattr(cognitive_config, "load_config", AsyncMock())

    async def prime(_session, _text, **kwargs):
        decision = kwargs["observation"].begin(kwargs["phase"], input_id=kwargs["input_id"])
        decision.complete("retrieved", retrieved=[trace_id], selected=[trace_id])
        return PrimedTurn("synthetic", [trace_id], [], None)

    monkeypatch.setattr(recognition, "_prime_memories", prime)
    monkeypatch.setattr(recognition, "_track_priming_reconsolidation", tracked)
    monkeypatch.setattr(recognition, "_record_deferred_access", writes)
    monkeypatch.setattr(turn, "build_window_cues", AsyncMock(return_value=None))
    monkeypatch.setattr(turn, "build_reprise_state", AsyncMock(return_value=reprise))
    client = MemoryClient(
        HostAdapters(
            inference=SimpleNamespace(complete=AsyncMock()),
            embedder=SimpleNamespace(embed=AsyncMock()),
            history=SimpleNamespace(),
            session_factory=lambda: None,
            policy=policy,
            overrides={
                "session_scope": session,
                "scrub_secrets": lambda text: text,
                "validate_skill_content": lambda text: [],
                "redact_urls_for_trace": lambda text: None,
            },
        )
    )
    yield SimpleNamespace(
        client=client, policy=policy, writes=writes, tracked=tracked, reprise=reprise, trace_id=trace_id
    )
    await client.close()


async def test_client_close_deadline_retains_resistant_turn_cleanup(runtime):
    entered, cancelled, release, finished = (asyncio.Event() for _ in range(4))
    owners = []

    async def resistant_close(**_):
        owners.append(current_owner())
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        finally:
            finished.set()

    runtime.reprise.close.side_effect = resistant_close
    controller = await runtime.client.begin_turn(Chat(chat_id=1))
    await controller.prepare_input("initial", "synthetic")
    try:
        result = await asyncio.wait_for(runtime.client.close(timeout=0.01), timeout=1)
        await entered.wait()
        assert result is False
        assert runtime.client.state == "closing"
        assert controller.snapshot().status == "closing"
        assert not finished.is_set()
        assert owners == [runtime.client]
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), timeout=1)
        await controller.close()
    for _ in range(20):
        if runtime.client.state == "closed":
            break
        await asyncio.sleep(0)
    assert runtime.client.state == "closed"
    assert controller.snapshot().status == "closed"


@pytest.mark.parametrize("revocation", ["read", "write"])
async def test_policy_revoked_after_prepare_prevents_learning(runtime, revocation):
    controller = await runtime.client.begin_turn(Chat(chat_id=1))
    await controller.prepare_input("initial", "synthetic")
    setattr(runtime.policy, revocation, False)
    assert controller.accept_input("initial") is False
    assert await runtime.client.drain()
    runtime.tracked.assert_not_called()
    runtime.writes.assert_not_awaited()
    assert not controller.snapshot().decisions[0].committed


async def test_read_only_host_can_prepare_but_never_commit_access(runtime):
    runtime.policy.write = False
    controller = await runtime.client.begin_turn(Chat(chat_id=1))
    prepared = await controller.prepare_input("initial", "synthetic")
    assert prepared.priming_context == "synthetic"
    assert controller.accept_input("initial") is False
    assert await runtime.client.drain()
    runtime.writes.assert_not_awaited()
    runtime.tracked.assert_not_called()


async def test_cached_procedural_facade_serializes_capture_per_client(runtime, monkeypatch):
    from kai_memory import procedural_api
    from kai_memory.procedural import rule_capture
    from kai_memory.procedural_api import RuleCandidate

    assert runtime.client.procedural is runtime.client.procedural
    message = Message(1, None, "example", "user", "Be concise", datetime.now(UTC), sender_user_id=7)
    monkeypatch.setattr(procedural_api, "_history", AsyncMock(return_value=[message]))
    monkeypatch.setattr(cognitive_config, "get_bool", lambda _: True)
    entered, release = asyncio.Event(), asyncio.Event()
    count = 0
    active = 0
    peak = 0

    async def capture(*_, **__):
        nonlocal count, active, peak
        assert current_owner() is runtime.client
        count += 1
        active += 1
        peak = max(peak, active)
        entered.set()
        await release.wait()
        active -= 1
        return 1

    monkeypatch.setattr(rule_capture, "capture_rules", capture)
    candidate = RuleCandidate("Be concise", "communication", "Be brief", "Always")
    first = asyncio.create_task(runtime.client.procedural.capture(Chat(chat_id=1), [candidate], [message.id]))
    await entered.wait()
    second = asyncio.create_task(runtime.client.procedural.capture(Chat(chat_id=1), [candidate], [message.id]))
    try:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert count == 1
    finally:
        release.set()
        receipts = await asyncio.gather(first, second)
    assert peak == 1 and count == 2
    assert all(receipt.status == "completed" for receipt in receipts)


async def test_failed_reprise_close_drains_owned_work_before_failed_receipt_and_pruning(runtime, monkeypatch):
    from kai_memory.retrieval.iterative_retrieval import LIVE, RepriseState

    retrieving, retrieval_cancelled, release_retrieval = (asyncio.Event() for _ in range(3))
    writing, release_access, access_finished, close_failed = (asyncio.Event() for _ in range(4))

    async def retrieve(_summary):
        retrieving.set()
        try:
            await asyncio.Future()
        finally:
            retrieval_cancelled.set()
            await release_retrieval.wait()

    async def write(_trace_ids):
        writing.set()
        await release_access.wait()
        access_finished.set()

    failure = RuntimeError("synthetic reprise cleanup failed")

    async def fail_close(**_kwargs):
        close_failed.set()
        raise failure

    async def reprise(_cues, _text, ctx, ids, *, observation):
        state = RepriseState(ctx, [[1.0, 0.0]], ids, LIVE, observation=observation)
        monkeypatch.setattr(state, "_reprime", retrieve)
        monkeypatch.setattr(state, "close", fail_close)
        return state

    runtime.writes.side_effect = write
    monkeypatch.setattr(turn, "build_reprise_state", reprise)
    controller = await runtime.client.begin_turn(Chat(chat_id=1))
    await controller.prepare_input("initial", "synthetic")
    assert controller.accept_input("initial")
    controller.consider_summary("synthetic reasoning", call_id="tool1")
    await asyncio.wait_for(asyncio.gather(retrieving.wait(), writing.wait()), timeout=1)
    closing = asyncio.create_task(controller.close())
    try:
        await asyncio.wait_for(asyncio.gather(close_failed.wait(), retrieval_cancelled.wait()), timeout=1)
        assert controller.snapshot().status == "closing"
        assert controller.diagnostic_snapshot()["terminal_status"] == "pending"
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        release_retrieval.set()
        await asyncio.sleep(0)
        assert not access_finished.is_set()
        assert controller.snapshot().status == "closing"
        assert not await runtime.client.drain(timeout=0)
        release_access.set()
        with pytest.raises(RuntimeError, match="synthetic reprise cleanup failed") as first:
            await asyncio.wait_for(controller.close(), timeout=1)
        assert first.value is failure
        assert access_finished.is_set()
        assert controller.snapshot().status == "closed"
        assert controller.diagnostic_snapshot()["terminal_status"] == "failed"
        with pytest.raises(RuntimeError, match="synthetic reprise cleanup failed") as repeated:
            await controller.close()
        assert repeated.value is failure
        replacement = await runtime.client.begin_turn(Chat(chat_id=1))
        assert controller not in runtime.client._turns
        await replacement.close()
    finally:
        release_retrieval.set()
        release_access.set()
        await asyncio.gather(closing, return_exceptions=True)
        try:
            await controller.close()
        except RuntimeError:
            pass


async def test_abandonment_releases_pending_call_but_retains_actual_old_cleanup(runtime, monkeypatch):
    from kai_memory.retrieval.iterative_retrieval import LIVE, RepriseState

    started, cleaning, release = (asyncio.Event() for _ in range(3))
    retrievals = []

    async def retrieve(summary):
        retrievals.append(summary)
        if summary == "first":
            started.set()
            try:
                await asyncio.Future()
            finally:
                cleaning.set()
                await release.wait()
        return [(str(uuid4()), "synthetic memory", 0.9)]

    async def reprise(_cues, _text, ctx, ids, *, observation):
        state = RepriseState(ctx, [[1.0, 0.0]], ids, LIVE, observation=observation)
        monkeypatch.setattr(state, "_reprime", retrieve)
        return state

    monkeypatch.setattr(turn, "build_reprise_state", reprise)
    controller = await runtime.client.begin_turn(Chat(chat_id=1))
    await controller.prepare_input("initial", "synthetic")
    controller.consider_summary("first", call_id="old")
    await started.wait()
    try:
        assert not controller.abandon_tool_call("unrelated")
        assert not cleaning.is_set()
        assert controller.abandon_tool_call("old")
        assert not controller.abandon_tool_call("old")
        await cleaning.wait()
        assert not await runtime.client.drain(timeout=0)
        controller.consider_summary("while cleaning", call_id="busy")
        assert not (await controller.prepare_tool_result("busy")).lines
        assert retrievals == ["first"]
        assert any(decision.outcome == "busy" for decision in controller.snapshot().decisions)
        release.set()
        assert await runtime.client.drain(timeout=1)
        controller.consider_summary("later", call_id="new")
        assert (await controller.prepare_tool_result("new")).lines
        assert retrievals == ["first", "later"]
        old = [decision for decision in controller.snapshot().decisions if decision.call_id == "old"]
        assert old and all(decision.delivery == "not_sent" and not decision.committed for decision in old)
    finally:
        release.set()
        await controller.close()
