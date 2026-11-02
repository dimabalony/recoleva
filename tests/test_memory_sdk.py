"""MemoryClient rules that need no database (DB-854).

The database-backed ownership rules (two stores, commit after close, work in flight
during close, a drain timeout) run in ``tests/integration/host_contract``.
"""

from __future__ import annotations

from memory_host_policy import synthetic_policy

import asyncio
from datetime import UTC, datetime

import pytest

from kai_memory import Chat, HostAdapters, MemoryClient, MemorySettings, Message, NoopTelemetry, ports
from kai_memory.host_calls import utc_now
from kai_memory.ownership import current_owner

FIXED = datetime(2030, 1, 2, 3, 4, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(ports, "_BOUND", {})


class _History:
    def __init__(self) -> None:
        self.messages: list[Message] = []

    async def record(self, message: Message) -> None:
        self.messages.append(message)

    async def read_page(self, chat, after, limit):
        return self.messages[:limit]

    async def senders_seen_before(self, senders, before):
        return set()

    async def delete(self, chat_id, topic_id):
        self.messages.clear()


class _Inference:
    async def complete(self, prompt, *, model, max_tokens, thinking_mode="adaptive"):
        return None


class _Embedder:
    async def embed(self, texts, *, api_key=None):
        return [[1.0] for _ in texts]


def _adapters(**kw) -> HostAdapters:
    base = {
        "inference": _Inference(),
        "embedder": _Embedder(),
        "history": _History(),
        "session_factory": lambda: None,
        "clock": lambda: FIXED,
    }
    kw["overrides"] = synthetic_policy(**kw.get("overrides", {}))
    return HostAdapters(**{**base, **kw})


def test_a_client_binds_every_port_only_inside_its_own_work():
    """The process default stays the host's; the client's adapters apply only in its scope."""
    default = object()
    ports.bind(embedder=default)
    adapters = _adapters()
    client = MemoryClient(adapters)
    assert ports.embedder is default  # creating a client binds nothing process-wide

    def inside():
        return ports.embedder, ports.get_settings(), ports.telemetry, ports.ChatContext, current_owner()

    embedder, settings, telemetry, chat_context, owner = client.run_owned(inside)
    assert embedder is adapters.embedder
    assert settings is adapters.settings and isinstance(settings, MemorySettings)
    assert isinstance(adapters.telemetry, NoopTelemetry) and telemetry.span("test", "fake") is None
    assert chat_context is Chat and owner is client
    assert ports.embedder is default and current_owner() is None
    asyncio.run(client.close())


def test_a_client_supplies_the_whole_contract_so_nothing_falls_through_to_the_process_default():
    client = MemoryClient(_adapters())
    assert set(client._bindings) == set(ports._HOST_SYMBOLS)
    asyncio.run(client.close())


def test_clients_are_independent_and_tasks_keep_their_owners_adapters():
    """Two open clients; a task each starts inherits its owner's clock, even after the other runs."""

    async def main():
        first = MemoryClient(_adapters(clock=lambda: datetime(2031, 1, 1, tzinfo=UTC)))
        second = MemoryClient(_adapters(clock=lambda: datetime(2032, 1, 1, tzinfo=UTC)))
        release = asyncio.Event()

        async def later():
            await release.wait()
            return utc_now().year, current_owner()

        task_first = first.run_owned(lambda: asyncio.create_task(later()))
        task_second = second.run_owned(lambda: asyncio.create_task(later()))
        release.set()
        assert await task_first == (2031, first)
        assert await task_second == (2032, second)
        await first.close()
        await second.close()

    asyncio.run(main())


def test_the_process_default_host_still_binds_after_import_while_a_client_is_open():
    client = MemoryClient(_adapters())
    ports.bind(clock=lambda: datetime(2040, 1, 1, tzinfo=UTC))
    assert utc_now().year == 2040  # Kai's path: no owner, the late process binding applies
    assert client.run_owned(utc_now) == FIXED  # the client's own clock inside its work
    asyncio.run(client.close())


def test_an_unknown_override_is_refused():
    with pytest.raises(KeyError, match="not memory host ports"):
        MemoryClient(_adapters(overrides={"not_a_port": object()}))


def test_overrides_replace_sdk_defaults():
    def scrub(text: str) -> str:
        return "[scrubbed]"

    client = MemoryClient(_adapters(overrides={"scrub_secrets": scrub}))
    assert client.run_owned(lambda: ports.scrub_secrets) is scrub
    asyncio.run(client.close())


def test_record_message_stamps_the_clock_and_goes_to_history_until_closed():
    history = _History()
    client = MemoryClient(_adapters(history=history))
    message = asyncio.run(client.record_message(Chat(chat_id=1), "user", "hello", sender_id=9))
    assert history.messages == [message]
    assert message.created_at == FIXED and message.sender_user_id == 9
    assert asyncio.run(client.close()) is True and client.state == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(client.record_message(Chat(chat_id=1), "user", "late"))


def test_close_cancels_its_own_tasks_on_a_drain_timeout_and_a_second_close_waits():
    async def main():
        client = MemoryClient(_adapters())
        never = asyncio.Event()
        task = client.run_owned(lambda: asyncio.create_task(never.wait()))
        client.adopt(task)
        first, second = await asyncio.gather(client.close(timeout=0.05), client.close())
        assert (first, second) == (False, False)
        assert task.cancelled() and client.state == "closed"

    asyncio.run(main())


def test_sdk_inference_adapter_receives_memorys_model_name():
    seen: list[tuple[str | None, int, str]] = []

    class Recording:
        async def complete(self, prompt, *, model, max_tokens, thinking_mode="adaptive"):
            seen.append((model, max_tokens, thinking_mode))
            return "ok"

    client = MemoryClient(_adapters(inference=Recording()))

    async def call():
        provider = ports.get_provider_for_model(ports.get_settings(), "haiku-5.5")
        disabled = await ports.call_llm_text(provider, "prompt", max_tokens=99, thinking_mode="disabled")
        adaptive = await ports.call_llm_text(provider, "prompt", max_tokens=99)
        return disabled, adaptive

    async def main():  # a task created in the client's work runs with its adapters
        return await client.run_owned(lambda: asyncio.create_task(call()))

    assert asyncio.run(main()) == ("ok", "ok")
    assert seen == [("haiku-5.5", 99, "disabled"), ("haiku-5.5", 99, "adaptive")]
    asyncio.run(client.close())


def test_a_cancelled_close_keeps_closing_and_every_close_reports_the_one_real_result():
    async def main():
        client = MemoryClient(_adapters())
        release = asyncio.Event()
        task = client.run_owned(lambda: asyncio.create_task(release.wait()))
        client.adopt(task)
        first = asyncio.create_task(client.close(timeout=5))
        await asyncio.sleep(0.01)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert client.state == "closing" and not client.accepts_work()  # still fenced, still owned
        assert not task.done()
        release.set()
        assert await asyncio.gather(client.close(), client.close()) == [True, True]
        assert client.state == "closed" and task.done() and not task.cancelled()
        assert await client.close() is True

    asyncio.run(main())


def test_close_fences_before_it_first_awaits():
    async def main():
        client = MemoryClient(_adapters())
        # Scheduled in this order, the call runs right after close() first yields.
        closing = asyncio.create_task(client.close())
        late = asyncio.create_task(client.record_message(Chat(chat_id=1), "user", "late"))
        with pytest.raises(RuntimeError, match="this MemoryClient is closing"):
            await late
        assert await closing is True

    asyncio.run(main())


def test_close_discards_recognitions_still_unsettled():
    from kai_memory.recognition import PrimedTurn, StagedRecognition

    async def main():
        client = MemoryClient(_adapters())
        primed = PrimedTurn(None, [], [], None)
        settled, unsettled = (client.run_owned(lambda: StagedRecognition(primed, None, owner=client)) for _ in range(2))
        client.track_staged(settled)
        client.track_staged(unsettled)
        assert settled.commit() is True
        await client.close()
        assert (settled.outcome, unsettled.outcome) == ("committed", "discarded")
        assert unsettled.commit() is False

    asyncio.run(main())


def test_clients_never_share_an_embedding_or_primed_baseline_entry():
    """The same text or chat under two clients' embedders keeps two entries, and Kai's none."""
    from kai_memory.retrieval import iterative_retrieval

    class Constant:
        def __init__(self, value: float) -> None:
            self.value = value

        async def embed(self, texts, *, api_key=None):
            return [[self.value] * 1536 for _ in texts]

    async def main():
        first, second = MemoryClient(_adapters(embedder=Constant(1.0))), MemoryClient(_adapters(embedder=Constant(2.0)))

        async def embed_and_churn(ids):
            return await iterative_retrieval._embed_cached("the same cue"), iterative_retrieval._churn(
                Chat(chat_id=7), ids
            )

        a = await first.run_owned(lambda: asyncio.create_task(embed_and_churn(["t1"])))
        b = await second.run_owned(lambda: asyncio.create_task(embed_and_churn(["t2"])))
        again = await first.run_owned(lambda: asyncio.create_task(embed_and_churn(["t3"])))
        assert (a, b) == (([1.0] * 1536, None), ([2.0] * 1536, None))  # no shared vector, no shared baseline
        assert again == ([1.0] * 1536, 1.0)  # first's own baseline: t1 dropped
        assert len(first.caches["cue_embeddings"]) == len(second.caches["cue_embeddings"]) == 1
        assert len(iterative_retrieval._EMBED_CACHE) == 0 and iterative_retrieval._PREV_PRIMED == {}
        await first.close()
        await second.close()

    iterative_retrieval.reset_state_for_testing()
    asyncio.run(main())


def test_each_client_reads_and_caches_its_own_stores_retrieval_weights():
    """Same chat, two stores with different weights: each client scores with, and caches, its own."""
    from types import SimpleNamespace

    from kai_memory.retrieval import memory_priming

    def store(semantic: float):
        config = SimpleNamespace(
            weight_semantic=semantic,
            weight_graph=0.1,
            weight_recency=0.1,
            weight_access=0.1,
            weight_arousal=0.1,
            weight_salience=0.0,
            weight_intention=0.1,
        )

        class Session:  # the database boundary: the one retrieval_config row
            reads = 0

            async def execute(self, statement):
                Session.reads += 1
                return SimpleNamespace(scalar_one_or_none=lambda: config)

        return Session

    async def main():
        session_a, session_b = store(0.1), store(0.9)
        first, second = MemoryClient(_adapters()), MemoryClient(_adapters())

        async def weights(session_cls):
            return (await memory_priming.MemoryPrimingService(session_cls())._get_cached_weights())["semantic"]

        used = [
            await first.run_owned(lambda: asyncio.create_task(weights(session_a))),
            await second.run_owned(lambda: asyncio.create_task(weights(session_b))),
            await first.run_owned(lambda: asyncio.create_task(weights(session_a))),
        ]
        assert used == [0.1, 0.9, 0.1]
        assert (session_a.reads, session_b.reads) == (1, 1)  # A's second read came from A's cache
        assert memory_priming._WEIGHTS_CACHE is None  # Kai's process cache untouched
        await first.close()
        await second.close()

    memory_priming.invalidate_weights_cache()
    asyncio.run(main())


def test_reconsolidation_candidates_stay_with_the_client_that_committed_them():
    import uuid

    from kai_memory.consolidation import consolidation_trigger

    chat, trace = Chat(chat_id=7), uuid.uuid4()
    first, second = MemoryClient(_adapters()), MemoryClient(_adapters())
    first.run_owned(lambda: consolidation_trigger.track_retrieved_traces(chat, [trace]))
    assert second.run_owned(lambda: consolidation_trigger.pop_priority_candidates(chat)) is None
    assert consolidation_trigger.pop_priority_candidates(chat) is None  # nor Kai's registry
    assert first.run_owned(lambda: consolidation_trigger.pop_priority_candidates(chat)) == [trace]
    asyncio.run(first.close())
    asyncio.run(second.close())


def test_each_client_keeps_its_own_cognitive_config_while_another_loads_different_rows():
    """Codex's repro: A loads flag=1, B loads flag=0 mid-run, A must still read 1 (and Kai's cache neither)."""
    from types import SimpleNamespace

    from kai_memory.store import cognitive_config

    def session(value: float):
        rows = [SimpleNamespace(key="procedural_rule_extraction_enabled", value=value)]

        class Session:  # the database boundary: this store's cognitive_config rows
            async def execute(self, statement):
                return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: rows))

        return Session()

    async def main():
        first, second = MemoryClient(_adapters()), MemoryClient(_adapters())
        b_loaded = asyncio.Event()

        async def run_a():
            await cognitive_config.load_config(session(1.0))
            await b_loaded.wait()  # A is mid-run while B loads
            return cognitive_config.get_bool("procedural_rule_extraction_enabled"), cognitive_config.is_loaded()

        async def run_b():
            await cognitive_config.load_config(session(0.0))
            b_loaded.set()
            return cognitive_config.get_bool("procedural_rule_extraction_enabled"), cognitive_config.is_loaded()

        a = first.run_owned(lambda: asyncio.create_task(run_a()))
        b = second.run_owned(lambda: asyncio.create_task(run_b()))
        assert await asyncio.gather(a, b) == [(True, True), (False, True)]
        assert cognitive_config._CACHE == cognitive_config._DEFAULTS and not cognitive_config.is_loaded()
        await first.close()
        await second.close()

    cognitive_config.reset_cache()
    asyncio.run(main())


def test_each_client_probes_its_own_store_for_the_aliases_helper():
    """A positive probe on one store must not send another client's store down the indexed path."""
    from types import SimpleNamespace

    from kai_memory.store import cognitive_memory

    def store(has_helper: bool):
        class Session:  # the database boundary: to_regprocedure on this store
            probes = 0

            async def execute(self, statement):
                Session.probes += 1
                return SimpleNamespace(scalar=lambda: has_helper)

        return Session

    async def main():
        with_helper, without = store(True), store(False)
        first, second = MemoryClient(_adapters()), MemoryClient(_adapters())

        async def probe(session_cls):
            return await cognitive_memory.CognitiveMemoryService(session_cls())._aliases_lower_exists()

        seen = [
            await first.run_owned(lambda: asyncio.create_task(probe(with_helper))),
            await second.run_owned(lambda: asyncio.create_task(probe(without))),
            await first.run_owned(lambda: asyncio.create_task(probe(with_helper))),
        ]
        assert seen == [True, False, True]
        assert (with_helper.probes, without.probes) == (1, 1)  # first's positive answer is first's own
        assert not cognitive_memory._ALIASES_LOWER_STATE.available  # nor Kai's
        await first.close()
        await second.close()

    cognitive_memory.reset_aliases_lower_probe()
    asyncio.run(main())


def test_each_client_gates_its_own_consolidation_llm_calls():
    from kai_memory.consolidation import consolidation

    first, second = MemoryClient(_adapters()), MemoryClient(_adapters())
    gate_first = first.run_owned(consolidation._get_sonnet_semaphore)
    assert first.run_owned(consolidation._get_sonnet_semaphore) is gate_first
    assert second.run_owned(consolidation._get_sonnet_semaphore) is not gate_first
    assert consolidation._get_sonnet_semaphore() is not gate_first  # Kai's process gate
    asyncio.run(first.close())
    asyncio.run(second.close())


def _parent_that_adopts(child_work):
    """An owned task that starts an owned child and finishes first (a retry, a sub-write)."""
    from kai_memory.ownership import adopt_if_owned

    async def parent():
        child = asyncio.create_task(child_work())
        adopt_if_owned(child)
        await asyncio.sleep(0)
        return child

    return parent


def test_drain_waits_for_a_child_adopted_while_it_waits_under_one_deadline():
    async def main():
        client = MemoryClient(_adapters())
        release = asyncio.Event()
        parent = client.run_owned(lambda: asyncio.create_task(_parent_that_adopts(release.wait)()))
        client.adopt(parent)
        assert await client.drain(0.2) is False  # the parent finished; its child has not
        child = parent.result()
        assert not child.done()
        asyncio.get_running_loop().call_later(0.05, release.set)
        assert await client.drain(2) is True and child.done()
        assert await client.close() is True

    asyncio.run(main())


def test_close_cancels_every_owned_task_descendants_included_and_owns_nothing_running():
    async def main():
        client = MemoryClient(_adapters())
        never = asyncio.Event()
        parent = client.run_owned(lambda: asyncio.create_task(_parent_that_adopts(never.wait)()))
        client.adopt(parent)
        await asyncio.sleep(0.01)
        child = parent.result()
        assert await client.close(timeout=0.1) is False
        assert child.cancelled() and client.state == "closed" and client._running() == set()

    asyncio.run(main())


def test_a_task_adopted_after_close_starts_is_cancelled_and_awaited():
    async def main():
        client = MemoryClient(_adapters())
        release, never = asyncio.Event(), asyncio.Event()

        async def late_parent():
            await release.wait()  # still running when close() starts
            return await _parent_that_adopts(never.wait)()

        parent = client.run_owned(lambda: asyncio.create_task(late_parent()))
        client.adopt(parent)
        closing = asyncio.create_task(client.close(timeout=5))
        await asyncio.sleep(0)
        release.set()
        assert await closing is True  # nothing needed the timeout's cancel
        child = parent.result()
        assert child.cancelled() and client._running() == set()

    asyncio.run(main())


def test_an_owned_timer_does_not_reschedule_once_its_client_is_closing():
    from kai_memory.consolidation import consolidation_trigger

    async def main():
        client = MemoryClient(_adapters())
        chat = Chat(chat_id=11)
        client.run_owned(lambda: consolidation_trigger.schedule_consolidation(chat))
        (timer,) = client.caches["pending_timers"].values()
        assert consolidation_trigger._pending_timers == {}  # never Kai's registry
        closing = asyncio.create_task(client.close(timeout=0.1))
        await asyncio.sleep(0)
        client.run_owned(lambda: consolidation_trigger.schedule_consolidation(chat))  # a retry while closing
        assert client.caches["pending_timers"] == {(chat.venue, chat.chat_id, chat.topic_id): timer}
        assert await closing is False  # the sleeping timer was cancelled at the timeout
        assert timer.done() and client._running() == set()  # (the timer returns quietly on cancel)

    asyncio.run(main())


def test_a_naive_host_clock_fails_at_the_clock_port():
    client = MemoryClient(_adapters(clock=lambda: datetime(2030, 1, 2, 3, 4)))  # noqa: DTZ001 — the host's mistake
    with pytest.raises(ValueError, match="naive datetime"):
        client.run_owned(utc_now)
    asyncio.run(client.close())


def test_record_message_refuses_naive_times_from_the_caller_or_the_clock():
    client = MemoryClient(_adapters())
    with pytest.raises(ValueError, match="timezone-aware"):
        asyncio.run(client.record_message(Chat(chat_id=1), "user", "hi", at=datetime(2030, 1, 1)))  # noqa: DTZ001
    asyncio.run(client.close())
    naive = MemoryClient(_adapters(clock=lambda: datetime(2030, 1, 1)))  # noqa: DTZ001 — the host's mistake
    with pytest.raises(ValueError, match="naive datetime"):
        asyncio.run(naive.record_message(Chat(chat_id=1), "user", "hi"))
    asyncio.run(naive.close())


def test_close_is_bounded_by_its_timeout_even_with_an_operation_stuck_in_flight():
    class StuckHistory(_History):
        def __init__(self) -> None:
            super().__init__()
            self.release = asyncio.Event()

        async def record(self, message):
            await self.release.wait()
            await super().record(message)

    async def main():
        history = StuckHistory()
        client = MemoryClient(_adapters(history=history))
        stuck = asyncio.create_task(client.record_message(Chat(chat_id=1), "user", "slow"))
        await asyncio.sleep(0)
        assert await asyncio.wait_for(client.close(timeout=0.1), 2) is False
        assert client.state == "closing" and not stuck.done()
        history.release.set()
        await stuck  # only real completion releases the closing owner
        assert client.state == "closed"

    asyncio.run(main())


def test_the_sdk_conversation_reader_reads_after_since_with_a_limit_like_kais():
    from datetime import timedelta

    from kai_memory.sdk import _HistoryConversations

    start = datetime(2030, 1, 1, tzinfo=UTC)

    class Log(_History):
        async def read_page(self, chat, after, limit):
            rows = [
                m
                for m in self.messages
                if (m.chat_id, m.topic_id, m.venue) == (chat.chat_id, chat.topic_id, chat.venue)
                and (after is None or m.created_at > after.created_at)
            ]
            return rows[:limit]

    log = Log()
    log.messages = [
        Message(
            chat_id=1,
            topic_id=None,
            venue="telegram",
            role="user",
            text=str(i),
            created_at=start + timedelta(minutes=i),
        )
        for i in range(300)
    ]
    # Interleave other scopes so a dropped or misforwarded identity changes results.
    from dataclasses import replace

    log.messages = [
        row
        for message in log.messages
        for row in (
            replace(message, chat_id=2, text="other chat"),
            replace(message, topic_id=7, text="other topic"),
            replace(message, venue="web", text="other venue"),
            message,
        )
    ]
    reader = _HistoryConversations(log)(None)
    recent = asyncio.run(reader.get_messages_since(1, None, start + timedelta(minutes=250), venue="telegram"))
    assert [m.text for m in recent] == [str(i) for i in range(251, 300)]
    assert len(asyncio.run(reader.get_messages_since(1, None, start - timedelta(minutes=1)))) == 200
    for chat_id, topic_id, venue, expected in (
        (2, None, "telegram", "other chat"),
        (1, 7, "telegram", "other topic"),
        (1, None, "web", "other venue"),
    ):
        messages = asyncio.run(
            reader.get_messages_since(chat_id, topic_id, start + timedelta(minutes=250), venue=venue)
        )
        assert [message.text for message in messages] == [expected] * 49


def test_close_returns_at_deadline_and_retains_cancellation_cleanup_until_it_finishes():
    from kai_memory.recognition import PrimedTurn, StagedRecognition

    async def main():
        client = MemoryClient(_adapters())
        other = MemoryClient(_adapters())
        started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def work():
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleaning.set()
                await release.wait()  # a provider/session still owns a real resource

        task = client.run_owned(lambda: asyncio.create_task(work()))
        client.adopt(task)
        await started.wait()
        closing = asyncio.create_task(client.close(timeout=0.02))
        try:
            # The cleanup gate is only released in finally; old unbounded gather cannot pass.
            assert await asyncio.wait_for(asyncio.shield(closing), 1) is False
            await asyncio.wait_for(cleaning.wait(), 1)
            assert client.state == "closing" and not task.done()
            assert await client.close(timeout=0) is False
            assert not task.done()  # a repeat close must not cancel cleanup a second time
            with pytest.raises(RuntimeError, match="closing"):
                await client.record_message(Chat(chat_id=1), "user", "late")
            late = StagedRecognition(PrimedTurn(None, [], [], None), None, owner=client)
            client.track_staged(late)
            assert late.outcome == "discarded" and late.commit() is False
            await other.record_message(Chat(chat_id=2), "user", "unaffected")
            assert await other.close() is True
        finally:
            release.set()
            await task
            await closing
            await other.close()
        assert client.state == "closed"
        assert await client.close() is False  # one stable result, even after eventual cleanup

    asyncio.run(main())


def test_overdue_cleanup_keeps_owner_alive_without_external_references_and_consumes_failure():
    import gc
    import weakref

    async def main():
        loop = asyncio.get_running_loop()
        errors = []
        old_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: errors.append(context))
        refs = []
        started = asyncio.Event()

        async def work():
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup = loop.create_future()
                refs.append(weakref.ref(cleanup))
                await cleanup  # no external strong reference to this waiter
                raise RuntimeError("discarded cleanup failure") from None

        client = MemoryClient(_adapters())
        task = client.run_owned(lambda: asyncio.create_task(work()))
        client.adopt(task)
        await started.wait()
        owner_ref, task_ref = weakref.ref(client), weakref.ref(task)
        closing = asyncio.create_task(client.close(timeout=0.02))
        try:
            assert await asyncio.wait_for(asyncio.shield(closing), 1) is False
            del task, client, closing
            gc.collect()
            assert owner_ref() is not None and owner_ref().state == "closing"
            assert task_ref() is not None and refs[0]() is not None
            refs[0]().set_result(None)
            # drain observes completion without retrieving the task's failure itself.
            assert await owner_ref().drain(1) is True
            await asyncio.sleep(0)
            gc.collect()
            assert task_ref() is None and owner_ref() is None
            assert errors == []
        finally:
            if refs and refs[0]() is not None and not refs[0]().done():
                refs[0]().set_result(None)
            if task_ref() is not None:
                await asyncio.gather(task_ref(), return_exceptions=True)
            loop.set_exception_handler(old_handler)

    asyncio.run(main())


async def test_sdk_explicit_call_helper_override_remains_authoritative():
    seen = []

    async def complete(provider, prompt, max_tokens=2048, span=None, **kwargs):
        seen.append(kwargs)
        return "host override"

    client = MemoryClient(_adapters(overrides={"call_llm_text": complete}))

    async def call():
        return await ports.call_llm_text(None, "prompt", thinking_mode="disabled")

    assert await client.run_owned(lambda: asyncio.create_task(call())) == "host override"
    assert seen == [{"thinking_mode": "disabled"}]
    await client.close()


@pytest.mark.parametrize("raises", [False, True])
def test_stage_a_shadow_is_retained_and_settled_by_client(monkeypatch, caplog, raises):
    """DB-849: the supported SDK owner retains the actual shadow task through settlement."""
    import gc
    import weakref
    from unittest.mock import AsyncMock
    from kai_memory.retrieval import iterative_retrieval as ir
    from kai_memory.retrieval.memory_priming import MemoryPrimingService

    async def main():
        client = MemoryClient(_adapters(session_factory=lambda: AsyncMock()))
        entered, release = asyncio.Event(), asyncio.Event()
        refs = []

        async def prime(*args, **kwargs):
            refs.append(weakref.ref(asyncio.current_task()))
            entered.set()
            await release.wait()
            if raises:
                raise RuntimeError("synthetic shadow failure")
            return "", [], []

        monkeypatch.setattr(MemoryPrimingService, "prime_with_traces", prime)
        cues = ir.WindowCues(segments=[ir.SegmentCue("synthetic", 1.0, [1.0])], is_windowed=True)
        try:
            await client.run_owned(
                lambda: asyncio.create_task(ir._run_stage_a_shadow(cues, "synthetic", Chat(), [], ir.SHADOW))
            )
            await asyncio.wait_for(entered.wait(), 1)
            gc.collect()
            assert refs[0]() is not None and refs[0]() in client._tasks
            assert not await client.drain(timeout=0)
            release.set()
            assert await client.drain(timeout=1)
            assert not client._tasks
            assert await client.close()
        finally:
            release.set()
            await client.close()

    with caplog.at_level("DEBUG"):
        asyncio.run(main())
    assert any("counterfactual failed (ignored)" in record.message for record in caplog.records) is raises
