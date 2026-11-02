"""Recognition: the priming hook every chat engine runs before a model turn (DB-839).

`_prime_memories` runs Level 1 priming plus procedural serving and returns a
`PrimedTurn`; `_track_priming_reconsolidation` flags the surfaced traces for L2.
Both moved here from `chat_engine`, which re-exports them so its tool loops and
every `chat_engine.X` patch target keep working.

`recognize` wraps the hook in a `StagedRecognition`, so a host that may still
reject the turn (a native steer the runtime never acknowledges) can hold back
the learning writes: `commit()` applies them once, `discard()` drops them. A
deferred access write runs as a task (also on `StagedRecognition.access_task`).
Under a `MemoryClient` the task belongs to that client, which drains it on close;
otherwise it sits in this module's registry and `drain_pending_access` lets a host
await it.

Imports of other memory modules stay lazy or `TYPE_CHECKING`-only:
`app.services.chat/__init__` imports `chat_engine`, which imports this module.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from kai_memory import ports
from kai_memory.ownership import Owner, current_owner

if TYPE_CHECKING:
    from kai_memory.retrieval.memory_observation import MemoryObservation
    from kai_memory.ports import ChatContext

logger = logging.getLogger("app.services.memory.recognition")


class PrimedTurn(NamedTuple):
    """What the shared priming hook hands each tool loop (Phase C §3.2 / DB-487 §3)."""

    priming_context: str | None  # L1 associations — UNCACHED suffix
    trace_ids: list[uuid.UUID]  # priming-surfaced trace ids (reconsolidation tracking)
    served_rule_ids: list[str]  # procedural rule ids served this turn
    learned_behaviors: str | None  # standing-rules block — CACHED prefix


async def _prime_memories(
    session: Any,
    last_user_msg: str | None,
    chat_ctx: ChatContext | None = None,
    window_cues: Any = None,
    *,
    observation: MemoryObservation | None = None,
    phase: str = "initial",
    input_id: str | None = None,
    record_access: bool = True,
) -> PrimedTurn:
    """Run Level 1 priming + procedural serving. Returns a `PrimedTurn`.

    `chat_ctx` rides along on the structured memory.retrieve.complete log
    so per-chat / per-topic / per-venue analysis is possible.

    `served_rule_ids` is stamped into `metadata["served_rule_ids"]` by every tool loop
    before the recorder hook, so L2 Part 5 can judge rule compliance from persisted
    metadata. Empty in shadow/off mode.

    `learned_behaviors` is the standing-rules block. It is returned SEPARATELY from
    `priming_context` because the two go to opposite sides of the prompt-cache
    boundary: L1 associations are per-turn retrieval (uncached suffix), standing rules
    are stable context (cached prefix — DB-487 §3). Concatenating them, as this hook
    did before DB-487, would drag the rules block below the boundary and re-bill it
    every turn.

    `window_cues` (L1.5 Stage A) is fed to prime ONLY when `priming_context_cues` is
    LIVE and the cues are a real multi-message window; otherwise priming uses today's
    single-message path (byte-identical). The flag gate lives here — prime itself is
    flag-agnostic (the shadow counterfactual passes cues explicitly).

    A turn with no user text skips L1 priming (there is no cue to prime on) but STILL
    serves rules. Standing rules are not retrieval — serve-all needs no query — so
    gating them behind `last_user_msg` would silently withhold them from exactly the
    text-less turns (heartbeat/wake) that Stage 3 exists to serve. That shape of
    structural block is what DB-436 was.

    `record_access=False` skips L1's access-count write; `recognize(defer_access=True)`
    uses it to move that write to `StagedRecognition.commit()`.
    """
    priming_context: str | None = None
    trace_ids: list = []
    served_rule_ids: list[str] = []
    learned_behaviors: str | None = None

    # Two INDEPENDENT fail-open scopes. A shared one would let an L1 retrieval hiccup
    # take the standing rules down with it — and rules are stable context, not this
    # turn's retrieval, so there is no reason for one to depend on the other.
    memory = observation.begin(phase, input_id=input_id) if observation is not None else None
    if last_user_msg:
        try:
            from kai_memory.retrieval.iterative_retrieval import LIVE, context_cues_state
            from kai_memory.retrieval.memory_priming import MemoryPrimingService

            live_cues = (
                window_cues
                if (window_cues is not None and window_cues.is_windowed and context_cues_state() == LIVE)
                else None
            )
            svc = (
                MemoryPrimingService(session, observation=memory)
                if memory is not None
                else MemoryPrimingService(session)
            )
            priming_context, trace_ids = await svc.prime(
                last_user_msg, chat_ctx=chat_ctx, record_access=record_access, window_cues=live_cues
            )
        except Exception:
            if memory is not None:
                memory.complete("failed")
            logger.warning("Memory priming failed", exc_info=True)
            priming_context, trace_ids = None, []

    if memory is not None and memory.outcome == "pending":
        memory.complete(
            "retrieved" if trace_ids else "empty" if last_user_msg else "no_input",
            retrieved=trace_ids,
            selected=trace_ids if priming_context else [],
        )

    # Procedural serving (DB-487) — serve-all over the deterministic scope filter, NEVER
    # L1 scoring (that re-triggers the #241 runaway). Flag-gated: off/shadow return
    # (None, []). get_learned_behaviors is itself fail-open; this guards the import.
    rules = observation.begin(phase, "rules", input_id=input_id) if observation is not None else None
    if chat_ctx is not None:
        try:
            from kai_memory.procedural.procedural_serving import get_learned_behaviors

            learned_behaviors, ids = (
                await get_learned_behaviors(session, chat_ctx, observation=rules)
                if rules is not None
                else await get_learned_behaviors(session, chat_ctx)
            )
            served_rule_ids = [str(i) for i in ids]
        except Exception:
            if rules is not None:
                rules.complete("failed")
            logger.warning("Procedural serving failed", exc_info=True)
            learned_behaviors, served_rule_ids = None, []

    if rules is not None and rules.outcome == "pending":
        rules.complete(
            "retrieved" if served_rule_ids else "empty",
            retrieved=served_rule_ids,
            selected=served_rule_ids if learned_behaviors else [],
        )
    return PrimedTurn(priming_context, trace_ids, served_rule_ids, learned_behaviors)


def _track_priming_reconsolidation(chat_ctx: ChatContext, trace_ids: list) -> None:
    """Flag priming-surfaced traces for L2 to treat as contradiction candidates."""
    if chat_ctx.chat_id is None or not trace_ids:
        return
    try:
        from kai_memory.consolidation.consolidation_trigger import track_retrieved_traces

        track_retrieved_traces(chat_ctx, trace_ids)
    except Exception:
        logger.debug("track_retrieved_traces failed", exc_info=True)


# Deferred access writes run after the turn's priming session has closed. The registry
# keeps each task from being garbage-collected mid-flight and lets a host drain them;
# a task leaves it when it finishes, whatever the outcome. A task created for a
# memory client is that client's instead (``ownership``), never this registry's.
_access_tasks: set[asyncio.Task[None]] = set()


async def _record_deferred_access(trace_ids: list[uuid.UUID]) -> None:
    """Write L1 access for a committed recognition in its own session. Fail-open."""
    try:
        from kai_memory.store.cognitive_memory import CognitiveMemoryService

        async with ports.session_scope() as session:
            await CognitiveMemoryService(session).record_access_batch(trace_ids)
    except Exception:
        logger.warning("Deferred access recording failed", exc_info=True)


def _schedule_deferred_access(trace_ids: list[uuid.UUID]) -> asyncio.Task[None]:
    asyncio.get_running_loop()  # raise before the coroutine exists, so none is left un-awaited
    task = asyncio.create_task(_record_deferred_access(trace_ids), name="recognition-access")
    owner = current_owner()
    if owner is not None:
        owner.adopt(task)
    else:
        _access_tasks.add(task)
        task.add_done_callback(_access_tasks.discard)
    return task


async def drain_pending_access(timeout: float) -> bool:  # noqa: ASYNC109 — reports leftovers, never cancels
    """Wait up to `timeout` seconds for the deferred access writes in flight.

    Returns True when none remain. The writes are fail-open, so this never raises
    for a failed one; a write still running at the timeout is left to finish.
    """
    pending = set(_access_tasks)
    if not pending:
        return True
    _done, still_running = await asyncio.wait(pending, timeout=timeout)
    return not still_running


@dataclass(slots=True)
class StagedRecognition:
    """One turn's recognition, with its learning writes held until the host settles it.

    The learning writes are reconsolidation tracking and, when `access_deferred`, the
    L1 access-count write. `commit()` applies them; `discard()` drops them. Whichever
    runs first settles the recognition (see `outcome`) and returns True; every later
    `commit()` or `discard()` is a no-op that returns False. Neither raises.

    `access_task` is the deferred access write `commit()` started, if any. It sits in
    the module registry until it finishes, so `drain_pending_access` covers it.

    `owner` is the memory client the recognition ran under (None for Kai's calls).
    `commit()` applies the writes inside that client's bindings, so they reach its
    store and its task set. Closing the client discards every recognition still
    unsettled; a commit that races the close (owner closing, recognition not yet
    discarded) is refused: it returns False, logs, and writes nothing.
    """

    primed: PrimedTurn
    chat_ctx: ChatContext | None
    access_deferred: bool = False
    owner: Owner | None = None
    commit_guard: Callable[[], bool] | None = None
    access_task: asyncio.Task[None] | None = field(default=None, init=False)
    _outcome: Literal["committed", "discarded"] | None = field(default=None, init=False)

    @property
    def outcome(self) -> Literal["committed", "discarded"] | None:
        return self._outcome

    def commit(self) -> bool:
        """Apply the learning writes. True only for the call that settles it."""
        if self._outcome is not None:
            return False
        owner = self.owner
        if owner is not None and not owner.accepts_work():
            logger.warning("Recognition commit refused: its memory client is closing or closed; nothing written")
            return False
        if self.commit_guard is not None:
            try:
                permitted = self.commit_guard() if owner is None else owner.run_owned(self.commit_guard)
            except Exception:
                permitted = False
            if not permitted:
                self.discard()
                return False
        self._outcome = "committed"
        if owner is None:
            self._apply_learning_writes()
        else:
            owner.run_owned(self._apply_learning_writes)
        return True

    def _apply_learning_writes(self) -> None:
        if self.chat_ctx is not None:
            try:
                _track_priming_reconsolidation(self.chat_ctx, self.primed.trace_ids)
            except Exception:
                logger.warning("Recognition reconsolidation tracking failed", exc_info=True)
        if self.access_deferred and self.primed.trace_ids:
            try:
                self.access_task = _schedule_deferred_access(self.primed.trace_ids)
            except Exception:
                logger.warning("Deferred access scheduling failed", exc_info=True)

    def discard(self) -> bool:
        """Drop the learning writes. True only for the call that settles it."""
        if self._outcome is not None:
            return False
        self._outcome = "discarded"
        return True


async def recognize(
    session: Any,
    last_user_msg: str | None,
    chat_ctx: ChatContext | None = None,
    window_cues: Any = None,
    *,
    observation: MemoryObservation | None = None,
    phase: str = "initial",
    input_id: str | None = None,
    defer_access: bool = False,
) -> StagedRecognition:
    """Prime a turn and stage its learning writes for `commit()` or `discard()`.

    `defer_access=True` holds L1's access-count write until `commit()`, so a turn the
    host rejects never counts as an access. Without it the write happens during
    priming, as it does for the tool loops.
    """
    primed = await _prime_memories(
        session,
        last_user_msg,
        chat_ctx=chat_ctx,
        window_cues=window_cues,
        observation=observation,
        phase=phase,
        input_id=input_id,
        record_access=not defer_access,
    )
    owner = current_owner()
    staged = StagedRecognition(primed, chat_ctx, access_deferred=defer_access, owner=owner)
    if owner is not None:
        owner.track_staged(staged)
    return staged
