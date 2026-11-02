"""Foreground operations remain alive through actual host resource cleanup."""

from memory_host_policy import synthetic_policy

import asyncio
import gc
import weakref

from kai_memory.sdk import Chat, HostAdapters, MemoryClient


class _History:
    def __init__(self):
        self.entered = asyncio.Event()
        self.waiter = None
        self.cleaned = False

    async def record(self, message):
        waiter = asyncio.get_running_loop().create_future()
        self.waiter = weakref.ref(waiter)
        self.entered.set()
        try:
            await waiter
        finally:
            self.cleaned = True


def _client(history):
    return MemoryClient(
        HostAdapters(
            overrides=synthetic_policy(),
            inference=object(),
            embedder=object(),
            history=history,
            session_factory=lambda: None,
        )
    )


async def test_close_retains_an_unreferenced_foreground_task_until_cleanup():
    history = _History()
    client = _client(history)
    task = asyncio.create_task(client.record_message(Chat(), "user", "synthetic message"))
    task_ref = weakref.ref(task)
    try:
        await asyncio.wait_for(history.entered.wait(), 1)
        assert await client.close(timeout=0) is False
        del task
        gc.collect()
        await asyncio.sleep(0)
        assert task_ref() is not None
        assert client.state == "closing" and not history.cleaned
        waiter = history.waiter()
        assert waiter is not None
        waiter.set_result(None)
        await asyncio.wait_for(task_ref(), 1)
        assert history.cleaned and client.state == "closed"
    finally:
        task = task_ref()
        if task is not None:
            waiter = history.waiter() if history.waiter is not None else None
            if waiter is not None and not waiter.done():
                waiter.set_result(None)
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 1)
        await client.close(timeout=0)


async def test_nested_operation_keeps_foreground_task_until_outer_operation_exits():
    history = _History()
    client = _client(history)

    async def nested():
        async with client._operation():
            async with client._operation():
                pass
            await history.record(None)

    task = asyncio.create_task(nested())
    task_ref = weakref.ref(task)
    try:
        await asyncio.wait_for(history.entered.wait(), 1)
        assert await client.close(timeout=0) is False
        del task
        gc.collect()
        await asyncio.sleep(0)
        assert task_ref() is not None
        assert client.state == "closing" and not history.cleaned
        waiter = history.waiter()
        assert waiter is not None
        waiter.set_result(None)
        await asyncio.wait_for(task_ref(), 1)
        assert history.cleaned and client.state == "closed"
    finally:
        task = task_ref()
        if task is not None:
            waiter = history.waiter() if history.waiter is not None else None
            if waiter is not None and not waiter.done():
                waiter.set_result(None)
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 1)
        await client.close(timeout=0)
