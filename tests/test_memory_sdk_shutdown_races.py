"""Shutdown deadlines retain only the client whose cleanup is still running."""

from memory_host_policy import synthetic_policy

import asyncio

import pytest

from kai_memory import sdk
from kai_memory.sdk import HostAdapters, MemoryClient


def _client():
    return MemoryClient(
        HostAdapters(
            overrides=synthetic_policy(),
            inference=object(),
            embedder=object(),
            history=object(),
            session_factory=lambda: None,
        )
    )


def _start_cleanup_worker(client):
    entered, cleaning, release, settled = (asyncio.Event() for _ in range(4))

    async def work():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()

    task = client.run_owned(lambda: asyncio.create_task(work()))
    client.adopt(task)
    # The SDK's settlement callback precedes this signal.
    task.add_done_callback(lambda _task: settled.set())
    return task, entered, cleaning, release, settled


async def test_cancelled_close_caller_preserves_deadline_cleanup_and_stable_result():
    client = _client()
    task, entered, cleaning, release, settled = _start_cleanup_worker(client)
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 1)
        closing = asyncio.create_task(client.close(timeout=0.02))
        await asyncio.sleep(0)  # close has fenced the client and installed its shielded closer
        assert client.state == "closing"
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert await asyncio.wait_for(client.close(timeout=0), 1) is False
        await asyncio.wait_for(cleaning.wait(), 1)
        assert client.state == "closing" and client in sdk._CLOSING_CLIENTS
        assert task.cancelling() == 1
        assert await client.close(timeout=0) is False
        assert task.cancelling() == 1 and not task.done()
        release.set()
        await asyncio.wait_for(settled.wait(), 1)
        assert client.state == "closed" and client not in sdk._CLOSING_CLIENTS
        assert await client.close() is False
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 1)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await client.close()


async def test_settling_one_retained_owner_does_not_release_or_recancel_the_other():
    first, second = _client(), _client()
    first_work = _start_cleanup_worker(first)
    second_work = _start_cleanup_worker(second)
    task_a, entered_a, cleaning_a, release_a, settled_a = first_work
    task_b, entered_b, cleaning_b, release_b, settled_b = second_work
    try:
        await asyncio.wait_for(asyncio.gather(entered_a.wait(), entered_b.wait()), 1)
        assert await asyncio.gather(first.close(timeout=0.02), second.close(timeout=0.02)) == [False, False]
        await asyncio.wait_for(asyncio.gather(cleaning_a.wait(), cleaning_b.wait()), 1)
        assert {first, second} <= sdk._CLOSING_CLIENTS
        release_a.set()
        await asyncio.wait_for(settled_a.wait(), 1)
        assert first.state == "closed" and first not in sdk._CLOSING_CLIENTS
        assert second.state == "closing" and second in sdk._CLOSING_CLIENTS
        assert not task_b.done() and task_b.cancelling() == 1
        assert await first.close() is False
        assert not task_b.done() and task_b.cancelling() == 1
        release_b.set()
        await asyncio.wait_for(settled_b.wait(), 1)
        assert second.state == "closed" and second not in sdk._CLOSING_CLIENTS
    finally:
        release_a.set()
        release_b.set()
        await asyncio.wait_for(asyncio.gather(task_a, task_b, return_exceptions=True), 1)
        await asyncio.gather(first.close(), second.close())
