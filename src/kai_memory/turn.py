"""Client-owned staged recognition and read-only mid-turn recall (DB-878)."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Literal

from kai_memory import ports
from kai_memory.ownership import Owner
from kai_memory.recognition import StagedRecognition, recognize
from kai_memory.retrieval.iterative_retrieval import RepriseState, build_reprise_state, build_window_cues
from kai_memory.retrieval.memory_observation import MemoryObservation


_RECEIPT_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")


@dataclass(frozen=True, slots=True)
class DecisionReceipt:
    phase: str
    kind: str
    trigger: str
    input_ids: tuple[str, ...]
    call_id: str | None
    outcome: str
    delivery: str
    retrieved_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    committed: bool


@dataclass(frozen=True, slots=True)
class TurnReceipt:
    status: str
    decisions: tuple[DecisionReceipt, ...]


@dataclass(frozen=True, slots=True)
class PreparedContext:
    input_id: str
    outcome: str
    priming_context: str | None
    learned_behaviors: str | None
    trace_ids: tuple[str, ...]
    served_rule_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreparedRecall:
    call_id: str
    lines: tuple[str, ...]


@dataclass(slots=True)
class _Input:
    kind: str
    staged: StagedRecognition
    prepared: PreparedContext
    observation_input_id: str | None
    delivery: str = "prepared"


class TurnMemory:
    """One turn; acceptance means host acknowledgement, never proven model use.

    The client supplies its operation context and owner. Retrieval therefore uses
    the same bindings and task ownership as other client operations. A host must
    call ``accept_input`` only after confirmed delivery. Mid-turn delivery never
    records accesses or reconsolidation. ``close`` discards unsettled inputs and
    joins Stage C work; it never closes host resources.
    """

    def __init__(
        self,
        *,
        operation: Callable[[], AbstractAsyncContextManager[None]],
        chat_ctx: Any,
        owner: Owner | None = None,
        model: str = "unspecified",
        scope: str = "turn",
        observation: MemoryObservation | None = None,
        read_guard: Callable[[], bool] | None = None,
    ) -> None:
        self._operation = operation
        self._ctx = chat_ctx
        self._owner = owner
        self._read_guard = read_guard
        self._host_observation = observation is not None
        self._observation = (
            observation if observation is not None else MemoryObservation(model=model, scope=scope, input_ids=[])
        )
        self._inputs: dict[str, _Input] = {}
        self._reprise: RepriseState | None = None
        self._lock = asyncio.Lock()
        self._closed = False
        self._settled = False
        self._close_task: asyncio.Task[None] | None = None
        self._calls: set[str] = set()
        self._abandoned_calls: set[str] = set()
        self._pending_call: str | None = None
        self._recalls: dict[str, PreparedRecall] = {}
        self._has_summary = False
        self._terminal_status = "closed"

    def _check_open(self) -> None:
        if self._closed or (self._owner is not None and not self._owner.accepts_work()):
            raise RuntimeError("Memory turn is closing or closed")

    def _check_disclosure(self) -> None:
        self._check_open()
        if self._read_guard is not None and not self._read_guard():
            raise PermissionError("memory host policy denied reading this scope")

    async def prepare_input(
        self,
        input_id: str,
        text: str | None,
        *,
        messages: list[dict[str, Any]] | None = None,
        kind: Literal["initial", "steering"] = "initial",
    ) -> PreparedContext:
        """Retrieve once per canonical input ID; duplicates return no context."""
        if not isinstance(input_id, str) or not _RECEIPT_ID.fullmatch(input_id):
            raise ValueError("input_id must be a 1-128 character receipt token (letters, digits, _, ., :, -)")
        if kind not in {"initial", "steering"}:
            raise ValueError("kind must be initial or steering")
        async with self._operation(), self._lock:
            self._check_open()
            if input_id in self._inputs:
                decision = self._observation.begin("steered" if kind == "steering" else kind, input_id=input_id)
                decision.complete("duplicate")
                return PreparedContext(input_id, "duplicate", None, None, (), ())
            if kind == "initial" and any(item.kind == "initial" for item in self._inputs.values()):
                raise ValueError("A turn has only one initial input")
            phase = "steered" if kind == "steering" else kind
            observation_input_id = None if kind == "initial" and self._host_observation else input_id
            window = messages if messages is not None else [{"role": "user", "content": [{"text": text or ""}]}]
            cues = await build_window_cues(window, self._ctx)
            async with ports.session_scope() as session:
                staged = await recognize(
                    session,
                    text,
                    self._ctx,
                    cues,
                    observation=self._observation,
                    phase=phase,
                    input_id=observation_input_id,
                    defer_access=True,
                )
            try:
                self._check_disclosure()
                primed = staged.primed
                prepared = PreparedContext(
                    input_id,
                    "prepared",
                    primed.priming_context,
                    primed.learned_behaviors,
                    tuple(str(value) for value in primed.trace_ids),
                    tuple(primed.served_rule_ids),
                )
                self._inputs[input_id] = _Input(phase, staged, prepared, observation_input_id)
                self._observation.prepared(phase, input_id=observation_input_id)
                if kind == "initial":
                    self._reprise = await build_reprise_state(
                        cues,
                        text,
                        self._ctx,
                        primed.trace_ids,
                        observation=self._observation,
                    )
                self._check_disclosure()
                return prepared
            except BaseException:
                staged.discard()
                raise

    def accept_input(self, input_id: str, *, turn_id: str | None = None) -> bool:
        """Commit learning only for the first confirmed acknowledgement."""
        self._check_open()
        item = self._inputs.get(input_id)
        if item is None or not item.staged.commit():
            return False
        item.delivery = "accepted"
        self._observation.delivery(item.kind, "accepted", input_id=item.observation_input_id, turn_id=turn_id)
        return True

    def reject_input(self, input_id: str, *, uncertain: bool = False, sent: bool = True) -> bool:
        """Discard uncertain or rejected delivery without counting it as use."""
        item = self._inputs.get(input_id)
        if item is None or not item.staged.discard():
            return False
        item.delivery = "unknown" if uncertain else "rejected" if sent else "not_sent"
        self._observation.delivery(item.kind, item.delivery, input_id=item.observation_input_id)
        return True

    def consider_summary(
        self,
        summary: str | None,
        *,
        call_id: str | None = None,
        injection_point: bool = True,
    ) -> None:
        """Start Stage C from a supplied summary; never revise a final answer."""
        self._check_disclosure()
        if self._reprise is None:
            raise RuntimeError("Prepare an initial input before considering a summary")
        self._has_summary = self._has_summary or bool(summary and summary.strip())
        if call_id is not None:
            if not isinstance(call_id, str) or not _RECEIPT_ID.fullmatch(call_id):
                raise ValueError("call_id must be a 1-128 character receipt token (letters, digits, _, ., :, -)")
            if call_id in self._calls:
                decision = self._observation.begin("reprise", trigger="reasoning_summary")
                decision.call_id = call_id
                decision.complete("duplicate")
                return
            if self._pending_call is not None:
                decision = self._observation.begin("reprise", trigger="reasoning_summary")
                decision.call_id = call_id
                decision.complete("busy")
                return
            self._calls.add(call_id)
            self._pending_call = call_id

        def consider() -> None:
            assert self._reprise is not None
            self._reprise.consider(
                [summary] if summary else [], "tool_use" if injection_point else None, call_id=call_id
            )

        if self._owner is None:
            consider()
        else:
            self._owner.run_owned(consider)

    async def prepare_tool_result(self, call_id: str) -> PreparedRecall:
        """Collect transient recall for the tool result, without a learning write."""
        async with self._operation(), self._lock:
            self._check_open()
            if call_id in self._abandoned_calls:
                return PreparedRecall(call_id, ())
            if call_id in self._recalls:
                decision = self._observation.begin("tool_result", trigger="tool_result")
                decision.call_id = call_id
                decision.complete("duplicate")
                return PreparedRecall(call_id, ())
            if self._reprise is None or self._pending_call != call_id:
                raise ValueError("No pending summary for this tool call")
            lines = await self._reprise.collect()
            if call_id in self._abandoned_calls:
                self._observation.abandon_call(call_id)
                return PreparedRecall(call_id, ())
            self._check_disclosure()
            source = next(
                (
                    value
                    for value in reversed(self._observation.snapshot()["decisions"])
                    if value["phase"] == "reprise" and value["call_id"] == call_id and value["outcome"] != "duplicate"
                ),
                None,
            )
            decision = self._observation.begin("tool_result", trigger="tool_result")
            decision.call_id = call_id
            decision.complete(
                "prepared" if lines else "empty",
                retrieved=source["retrieved_ids"] if source else (),
                selected=source["selected_ids"] if source and lines else (),
            )
            if lines:
                decision.delivery = "prepared"
            result = PreparedRecall(call_id, tuple(lines))
            self._recalls[call_id] = result
            self._pending_call = None
            return result

    def abandon_tool_call(self, call_id: str, *, uncertain: bool = False) -> bool:
        """Release one aborted injection opportunity; stale acknowledgements are ignored."""
        if self._settled or call_id not in self._calls or call_id in self._abandoned_calls:
            return False
        self._abandoned_calls.add(call_id)
        if self._pending_call == call_id:
            self._pending_call = None
            if self._reprise is not None:
                if self._owner is None:
                    self._reprise.cancel()
                else:
                    self._owner.run_owned(self._reprise.cancel)
        self._observation.abandon_call(call_id, uncertain=uncertain)
        return True

    def acknowledge_tool_result(self, call_id: str, *, accepted: bool = False, written: bool = False) -> None:
        """Separate confirmed acceptance from a write with unknown acceptance."""
        # Transport may settle prepared receipts while cleanup is still draining.
        if call_id in self._abandoned_calls:
            return
        if self._settled:
            return  # A retained transport may replay an acknowledgement after detach.
        if call_id not in self._recalls:
            raise ValueError("Tool result was not prepared")
        self._observation.call_delivery(
            call_id, "accepted" if accepted else "written_unacknowledged" if written else "unknown"
        )

    @property
    def counters(self) -> dict[str, int]:
        """Detached existing Stage C counters for host diagnostics."""
        return dict(self._reprise.counters) if self._reprise is not None else {}

    def bind_diagnostics(
        self,
        *,
        thread_id: str | None = None,
        turn_id: str | None = None,
        trace_id: str | None = None,
        observed_model: str | None = None,
    ) -> None:
        self._observation.bind(
            thread_id=thread_id,
            turn_id=turn_id,
            trace_id=trace_id,
            observed_model=observed_model,
        )

    def diagnostic_snapshot(self, status: str = "pending") -> dict[str, Any]:
        """Preserve the existing diagnostic schema for host compatibility."""
        return self._observation.snapshot(status)

    def snapshot(self) -> TurnReceipt:
        """Return detached immutable evidence; delivery does not imply model use."""
        decisions = []
        for decision in self._observation.snapshot()["decisions"]:
            committed = any(
                value.staged.outcome == "committed"
                and (value.observation_input_id is None or key in decision["canonical_input_ids"])
                and value.kind == decision["phase"]
                and decision["outcome"] != "duplicate"
                for key, value in self._inputs.items()
            )
            delivery = decision["delivery"]
            if decision["outcome"] != "duplicate":
                for key, item in self._inputs.items():
                    if (
                        item.observation_input_id is None or key in decision["canonical_input_ids"]
                    ) and item.kind == decision["phase"]:
                        delivery = item.delivery
            decisions.append(
                DecisionReceipt(
                    decision["phase"],
                    decision["kind"],
                    decision["trigger"],
                    tuple(decision["canonical_input_ids"]),
                    decision["call_id"],
                    decision["outcome"],
                    delivery,
                    tuple(decision["retrieved_ids"]),
                    tuple(decision["selected_ids"]),
                    committed,
                )
            )
        return TurnReceipt(
            "closed" if self._settled else "closing" if self._closed else "open",
            tuple(decisions),
        )

    def begin_close(self) -> None:
        """Fence immediately; asynchronous close still owns the actual cleanup."""
        self._closed = True
        for item in self._inputs.values():
            if item.staged.discard() or (item.staged.outcome == "discarded" and item.delivery == "prepared"):
                item.delivery = "discarded"
        if self._reprise is not None:
            if self._owner is None:
                self._reprise.cancel()
            else:
                self._owner.run_owned(self._reprise.cancel)

    async def close(self, *, has_summary: bool = False, status: str = "closed") -> None:
        """Fence new work and retain cleanup even if the caller is cancelled."""
        if self._close_task is None:
            self._terminal_status = status
        self.begin_close()
        self._has_summary = self._has_summary or has_summary
        for item in self._inputs.values():
            if item.staged.discard() or (item.staged.outcome == "discarded" and item.delivery == "prepared"):
                item.delivery = "discarded"
        if self._close_task is None:

            def start_cleanup() -> asyncio.Task[None]:
                task = asyncio.create_task(self._finish_close(), name="memory-turn-close")
                adopt_cleanup = getattr(self._owner, "adopt_cleanup", None)
                if adopt_cleanup is not None:
                    adopt_cleanup(task)
                return task

            self._close_task = start_cleanup() if self._owner is None else self._owner.run_owned(start_cleanup)
        await asyncio.shield(self._close_task)

    async def _finish_close(self) -> None:
        async with self._lock:
            for item in self._inputs.values():
                if item.staged.discard() or (item.staged.outcome == "discarded" and item.delivery == "prepared"):
                    item.delivery = "discarded"
            failure: BaseException | None = None
            try:
                if self._reprise is not None:
                    try:
                        await self._reprise.close(has_summary=self._has_summary)
                    except BaseException as exc:
                        failure = exc
                        await self._reprise.drain()
            finally:
                access_tasks = [
                    item.staged.access_task for item in self._inputs.values() if item.staged.access_task is not None
                ]
                if access_tasks:
                    await asyncio.shield(asyncio.gather(*access_tasks, return_exceptions=True))
            self._observation.finish("failed" if failure is not None else self._terminal_status)
            self._settled = True
            if failure is not None:
                raise failure
