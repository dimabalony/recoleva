"""Which memory client owns the work running in this context (DB-854).

A ``MemoryClient`` runs each operation with itself as the owner (a ContextVar that
asyncio tasks inherit, set together with its ``ports.scoped`` bindings). Memory work
that outlives the call finds its owner here:

* a ``StagedRecognition`` records the owner it was recognised under, and its
  ``commit()`` runs inside that owner's bindings, or is refused once the owner
  stopped accepting work;
* a background task memory creates is ``adopt``-ed by the owner, which drains it
  on ``close()`` and requests cancellation on a drain timeout; unfinished cleanup
  keeps the client alive and ``closing`` until the work actually settles;
* state memory keeps between calls (the retrieval-weights cache, the cue-embedding
  cache, the previous-primed baseline, the reconsolidation candidates) lives on the
  owner (``owned_state``), so clients over different stores or adapters never share
  an entry.

Kai calls memory with no owner: everything then works as it did before.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol, TypeVar, cast

T = TypeVar("T")


class Owner(Protocol):
    """What memory needs from the client that owns a piece of work."""

    # Memory's between-call state for this owner, by name (see ``owned_state``).
    caches: dict[str, Any]

    def accepts_work(self) -> bool:
        """False once the owner is closing: new work for it (a commit, a timer) is refused."""
        ...

    def run_owned(self, fn: Callable[[], T]) -> T:
        """Run ``fn`` with this owner's bindings and ownership in effect."""
        ...

    def adopt(self, task: asyncio.Task[Any]) -> None:
        """Track a background task until it finishes; the owner drains or cancels it.

        An owner that is closing cancels the task at once.
        """
        ...

    def track_staged(self, staged: Any) -> None:
        """Hold a staged result the owner rejects (discards) if it is unsettled at close."""
        ...


_OWNER: ContextVar[Owner | None] = ContextVar("memory_owner", default=None)


def current_owner() -> Owner | None:
    """The owner of the work running in this context, or None (Kai's own calls)."""
    return _OWNER.get()


@contextmanager
def owned_by(owner: Owner) -> Iterator[None]:
    """Make ``owner`` the owner of the work in this context and the tasks created in it."""
    token = _OWNER.set(owner)
    try:
        yield
    finally:
        _OWNER.reset(token)


def adopt_if_owned(task: asyncio.Task[Any]) -> bool:
    """Hand ``task`` to the current owner, if any. True when an owner took it."""
    owner = _OWNER.get()
    if owner is None:
        return False
    owner.adopt(task)
    return True


def owned_state[S](name: str, factory: Callable[[], S], process: S) -> S:
    """The current owner's ``name`` state (made with ``factory`` on first use), or ``process`` (Kai's)."""
    owner = _OWNER.get()
    if owner is None:
        return process
    try:
        return cast(S, owner.caches[name])
    except KeyError:
        value = owner.caches[name] = factory()
        return value
