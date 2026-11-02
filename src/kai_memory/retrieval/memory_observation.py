"""Bounded, content-free receipts for memory crossing the native turn boundary."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import json
import logging
import re
import time
from typing import Any, cast
from uuid import UUID, uuid4

# Pinned to the pre-DB-842 module path: CloudWatch queries, feature-watch criteria and
# log filters match the logger name, which must not follow the module's move.
_LEGACY_LOGGER_NAME = "app.services.chat.memory_observation"
logger = logging.getLogger(_LEGACY_LOGGER_NAME)
MAX_DECISIONS = 64
MAX_IDS = 32
MAX_SNAPSHOT_BYTES = 64 * 1024
_TOKEN = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")


def _token(value: object) -> str | None:
    value = str(value) if isinstance(value, UUID) else value if isinstance(value, str) else ""
    return value if _TOKEN.fullmatch(value) else None


def _ids(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    for value in values:
        token = _token(value)
        if token and token not in result:
            result.append(token)
        if len(result) == MAX_IDS:
            break
    return result


def _emit(event: str, payload: dict[str, Any]) -> None:
    # Telemetry failure must not change a model/tool/cleanup outcome.
    try:
        logger.info("%s %s", event, json.dumps({"event_type": event, **payload}, allow_nan=False))
    except Exception:
        pass


@dataclass
class MemoryDecision:
    owner: MemoryObservation
    decision_id: str
    phase: str
    kind: str
    trigger: str
    canonical_input_ids: list[str]
    mode: str = "live"
    outcome: str = "pending"
    delivery: str = "not_prepared"
    retrieved_ids: list[str] = field(default_factory=list)
    selected_ids: list[str] = field(default_factory=list)
    retrieved_count: int = 0
    selected_count: int = 0
    issues: dict[str, int] = field(default_factory=dict)
    native_turn_id: str | None = None
    call_id: str | None = None
    started: float = field(default_factory=lambda: time.monotonic())
    duration_ms: int | None = None
    retained: bool = True

    def issue(self, reason: str) -> None:
        if reason in {"failed", "timeout", "embedding_failed", "capped"}:
            self.issues[reason] = min(1_000_000, self.issues.get(reason, 0) + 1)

    def complete(
        self,
        outcome: str,
        *,
        retrieved: Sequence[object] = (),
        selected: Sequence[object] = (),
        retrieval_pending: bool = False,
    ) -> None:
        if self.outcome != "pending":
            return
        if not retrieval_pending:
            self.retrieval_finished()
        self.retrieved_count = len(retrieved)
        self.selected_count = len(selected)
        self.retrieved_ids = _ids(retrieved)
        self.selected_ids = _ids(selected)
        if outcome in {"retrieved", "empty"} and self.issues.keys() & {"failed", "timeout", "embedding_failed"}:
            outcome = "partial" if self.retrieved_ids else "timeout" if "timeout" in self.issues else "failed"
        self.outcome = outcome
        self.emit()

    def retrieval_finished(self, *, retrieved: Sequence[object] | None = None) -> None:
        if self.owner.finished:
            return
        if self.duration_ms is None:
            self.duration_ms = max(0, int((time.monotonic() - self.started) * 1000))
        if retrieved is not None:
            self.retrieved_count = len(retrieved)
            self.retrieved_ids = _ids(retrieved)
        if self.outcome != "pending":
            # A timeout/cancel receipt precedes native worker settlement. Its
            # late result is evidence of retrieval, never permission to inject.
            self.emit()

    def payload(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "phase": self.phase,
            "kind": self.kind,
            "trigger": self.trigger,
            "canonical_input_ids": self.canonical_input_ids,
            "mode": self.mode,
            "outcome": self.outcome,
            "delivery": self.delivery,
            "retrieved_ids": self.retrieved_ids,
            "selected_ids": self.selected_ids,
            "retrieved_count": self.retrieved_count,
            "selected_count": self.selected_count,
            "ids_truncated": self.retrieved_count > len(self.retrieved_ids)
            or self.selected_count > len(self.selected_ids),
            "injected_ids": self.selected_ids if self.delivery == "accepted" else [],
            "written_ids": self.selected_ids if self.delivery == "written_unacknowledged" else [],
            "issues": dict(self.issues),
            "duration_ms": self.duration_ms,
            "native_turn_id": self.native_turn_id,
            "call_id": _token(self.call_id),
        }

    def emit(self) -> None:
        if self.retained and not self.owner.finished:
            _emit("native.memory.decision", {**self.owner.identity(), **self.payload()})


class MemoryObservation:
    def __init__(self, *, model: str, scope: str, input_ids: list[str]) -> None:
        self.operation_id = str(uuid4())
        self.model = _token(model)
        self.observed_model: str | None = None
        self.scope = _token(scope)
        self.input_ids = _ids(input_ids)
        self.input_count = len(input_ids)
        self.trace_id: str | None = None
        self.native_thread_id: str | None = None
        self.native_turn_id: str | None = None
        self.decisions: list[MemoryDecision] = []
        self.omitted_decisions = 0
        self.finished = False
        self.terminal_status = "pending"
        self._final_json: str | None = None

    def identity(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "engine": "native_codex",
            "model": self.model,
            "model_source": "configured",
            "observed_model": self.observed_model,
            "scope": self.scope,
            "trace_id": self.trace_id,
            "native_thread_id": self.native_thread_id,
            "native_turn_id": self.native_turn_id,
        }

    def bind(
        self,
        *,
        thread_id: str | None = None,
        turn_id: str | None = None,
        trace_id: str | None = None,
        observed_model: str | None = None,
    ) -> None:
        if observed_model is not None:
            self.observed_model = _token(observed_model)
        if thread_id is not None:
            self.native_thread_id = _token(thread_id)
        if turn_id is not None:
            self.native_turn_id = _token(turn_id)
        if trace_id is not None:
            self.trace_id = _token(trace_id)

    def begin(
        self, phase: str, kind: str = "memory", *, input_id: str | None = None, trigger: str = "user_message"
    ) -> MemoryDecision:
        retained = len(self.decisions) < MAX_DECISIONS and not self.finished
        decision = MemoryDecision(
            self,
            str(uuid4()),
            phase,
            kind,
            trigger,
            _ids([input_id]) if input_id is not None else list(self.input_ids),
            native_turn_id=self.native_turn_id,
            retained=retained,
        )
        if retained:
            self.decisions.append(decision)
        else:
            self.omitted_decisions = min(1_000_000, self.omitted_decisions + 1)
        return decision

    def prepared(self, phase: str, *, input_id: str | None = None) -> None:
        for decision in self.decisions:
            if decision.phase == phase and decision.delivery == "not_prepared" and decision.selected_ids:
                if input_id is None or input_id in decision.canonical_input_ids:
                    decision.delivery = "prepared"

    def delivery(self, phase: str, status: str, *, input_id: str | None = None, turn_id: str | None = None) -> None:
        if turn_id is not None:
            self.bind(turn_id=turn_id)
        if status == "accepted" and input_id is not None and _token(input_id) and input_id not in self.input_ids:
            self.input_count += 1
            self.input_ids = [*self.input_ids, input_id][-MAX_IDS:]
        for decision in self.decisions:
            if decision.phase != phase:
                continue
            if input_id is not None and input_id not in decision.canonical_input_ids:
                continue
            if status == "accepted":
                decision.native_turn_id = _token(turn_id)
            if decision.delivery == "prepared":
                decision.delivery = status
                decision.emit()

    def call_delivery(self, call_id: str, status: str) -> None:
        """Settle prepared tool context without inferring model use."""
        if status not in {"accepted", "written_unacknowledged", "unknown"}:
            raise ValueError("Invalid tool context delivery status")
        for decision in self.decisions:
            if decision.call_id == call_id and (
                decision.delivery == "prepared"
                or (status == "accepted" and decision.delivery in {"written_unacknowledged", "unknown"})
            ):
                decision.delivery = status
                decision.emit()

    def abandon_call(self, call_id: str, *, uncertain: bool = False) -> None:
        """Discard context for an aborted response without undoing confirmed delivery."""
        for decision in self.decisions:
            if decision.call_id != call_id:
                continue
            if decision.outcome == "pending":
                # Retained IDs are bounded samples; preserve their full counts.
                decision.outcome = "discarded"
                decision.emit()
            if decision.delivery in {"not_prepared", "prepared"}:
                decision.delivery = "unknown" if uncertain else "not_sent"
                decision.emit()

    def tool_response(self, call_id: str, written: bool) -> None:
        for decision in self.decisions:
            if decision.call_id == call_id and decision.delivery == "prepared":
                decision.delivery = "written_unacknowledged" if written else "unknown"
                decision.emit()

    def snapshot(self, status: str = "pending") -> dict[str, Any]:
        if self._final_json is not None:
            return cast(dict[str, Any], json.loads(self._final_json))
        result: dict[str, Any] = {
            **self.identity(),
            "terminal_status": status,
            "canonical_input_ids": list(self.input_ids),
            "canonical_input_count": self.input_count,
            "input_ids_truncated": self.input_count > len(self.input_ids),
            "decisions": [],
            "omitted_decisions": self.omitted_decisions,
            "omitted_snapshot_decisions": len(self.decisions),
        }
        # Reserve room for counters/status and JSON list punctuation. Individual
        # decision events remain bounded independently by MAX_IDS and token size.
        used = len(json.dumps(result).encode()) + 128
        for decision in self.decisions:
            payload = decision.payload()
            size = len(json.dumps(payload).encode()) + 2
            if used + size > MAX_SNAPSHOT_BYTES:
                break
            result["decisions"].append(payload)
            used += size
        result["omitted_snapshot_decisions"] -= len(result["decisions"])
        # A snapshot never exposes the live lists to metadata consumers.
        return cast(dict[str, Any], json.loads(json.dumps(result)))

    def finish(self, status: str) -> dict[str, Any]:
        if not self.finished:
            for decision in self.decisions:
                if decision.outcome == "pending":
                    decision.complete("cancelled" if status == "cancelled" else "failed")
                if decision.delivery == "prepared":
                    decision.delivery = "unknown"
            self.finished = True
            self.terminal_status = status
            self._final_json = json.dumps(self.snapshot(status))
            _emit("native.memory.turn", json.loads(self._final_json))
        return self.snapshot()
