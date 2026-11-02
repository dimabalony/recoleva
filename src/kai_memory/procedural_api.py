"""Portable procedural lifecycle; authority and canonical history belong to the host.

Serving records exposure, never compliance. Observations park evidence for the
existing nightly ledger; no public operation writes utility or changes D027.
"""

from __future__ import annotations

import asyncio
import logging

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from sqlalchemy import Select, literal_column, select

from kai_memory import ports
from kai_memory.host_calls import session_scope, utc_now
from kai_memory.procedural.procedural_scope import LEDGER_VERDICTS, PERSISTED_VERDICTS, MessageAnchor

if TYPE_CHECKING:
    from kai_memory.sdk import Chat, Message
    from kai_memory.storage.models.execution_trace import ExecutionTrace

Operation = Callable[[], AbstractAsyncContextManager[None]]
LoadConfig = Callable[[], Awaitable[None]]
_HISTORY_LIMIT = 1000
logger = logging.getLogger(__name__)


def _assistant_execution_lookup(message_id: UUID) -> Select[tuple[ExecutionTrace]]:
    from kai_memory.storage.models.execution_trace import ExecutionTrace

    # A fixed key keeps the expression index eligible under generic prepared plans.
    anchor = ExecutionTrace.success_signals[literal_column("'assistant_message_id'")].astext
    return select(ExecutionTrace).where(anchor.is_not(None), anchor == str(message_id))


@dataclass(frozen=True)
class RuleCandidate:
    source_quote: str
    category: str
    rule: str
    trigger: str
    directive_strength: str = "correction"
    scope_hint: str = "global"
    group_safe: bool = False
    expected_tools: list[str] | None = None


@dataclass(frozen=True)
class ToolEvidence:
    name: str
    argument_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionEvidence:
    assistant_message_id: UUID
    tools: tuple[ToolEvidence, ...] = ()
    served_rule_ids: tuple[UUID, ...] = ()
    tool_error_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None
    pending_approval: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecordingReceipt:
    status: Literal["recorded", "tool_free", "duplicate", "disabled", "rejected", "failed"]
    execution_trace_id: UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None


@dataclass(frozen=True)
class CaptureReceipt:
    """Capture result; failed/capture_busy is contention callers may retry later."""

    status: Literal["completed", "disabled", "rejected", "failed"]
    changed: int = 0
    decisions: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class ServingReceipt:
    block: str | None
    served_rule_ids: tuple[UUID, ...] = ()
    status: str = "completed"


@dataclass(frozen=True)
class OutcomeReceipt:
    status: Literal["recorded", "duplicate", "rejected", "disabled", "failed"]
    outcome_updated: bool = False
    compliance_updated: int = 0
    reason: str | None = None


def resolve_evidence_speaker(evidence: str, messages: Sequence[Any]) -> int | None:
    """Quote attribution fails closed when several distinct speakers match."""
    from kai_memory.procedural.rule_capture import _anchorable_text, _norm

    needle = _norm(evidence)
    if not needle:
        return None
    speakers = {
        getattr(message, "sender_user_id", None)
        for message in messages
        if (text := _anchorable_text(message)) is not None and needle in text
    }
    return next(iter(speakers)) if len(speakers) == 1 else None


def _same_chat(message: Any, chat: Chat) -> bool:
    return (message.chat_id, message.topic_id, message.venue) == (chat.chat_id, chat.topic_id, chat.venue)


class HistoryWindowSaturated(ValueError):
    """The host must provide a smaller window retaining the required anchors."""


async def _history(session: Any, chat: Chat, after: Any = None) -> list[Message]:
    messages = list(await ports.read_conversation_page(session, chat, after, _HISTORY_LIMIT))
    if any(not _same_chat(message, chat) for message in messages):
        raise ValueError("history adapter returned another conversational scope")
    if len(messages) >= _HISTORY_LIMIT:
        raise HistoryWindowSaturated("procedural history window saturated; use a bounded host history view")
    return sorted(messages, key=lambda message: message.created_at)


def is_user_trigger(message: Any) -> bool:
    metadata = getattr(message, "message_metadata", None) or {}
    return (
        message.role == "user"
        and not getattr(message, "tool_results", None)
        and metadata.get("source_kind") != "tool_result"
    )


def _assistant_anchor(messages: Sequence[Message], message_id: UUID, chat: Chat) -> tuple[Message, int | None] | None:
    trigger: Message | None = None
    for index, message in enumerate(messages):
        if is_user_trigger(message):
            trigger = message
        if message.id != message_id:
            continue
        if message.role != "assistant":
            return None
        metadata = message.message_metadata or {}
        origin = metadata.get("procedural_origin_message_id")
        if origin:
            trigger = next(
                (
                    item
                    for item in messages[:index]
                    if str(item.id) == str(origin) and is_user_trigger(item) and item.created_at <= message.created_at
                ),
                None,
            )
        if trigger is None:
            return None
        actor = trigger.sender_user_id
        if chat.is_group and metadata.get("native_procedural_attribution") == "shared_group_context":
            actor = None
        if actor != chat.sender_user_id:
            return None
        return message, actor
    return None


class ProceduralMemory:
    """Facade owned by one MemoryClient's operation and adapter scope.

    ``history_after`` uses the host History adapter's existing cursor contract.
    The bounded window must include each source message and the assistant's
    preceding actual human trigger; missing or saturated windows fail closed.
    Saturation returns rejected/history_saturated; provide a smaller anchored window.
    """

    def __init__(self, operation: Operation, load_config: LoadConfig) -> None:
        self._operation = operation
        self._load_config = load_config
        self._capture_lock = asyncio.Lock()

    async def record_execution(
        self, chat: Chat, task_text: str, evidence: ExecutionEvidence, *, history_after: Any = None
    ) -> RecordingReceipt:
        from kai_memory.procedural.procedural_recorder import record_execution_trace
        from kai_memory.store import cognitive_config

        async with self._operation():
            policy = ports.get_host_policy()
            if not policy.may_read(chat) or not policy.may_write(chat):
                return RecordingReceipt("rejected", reason="scope_denied")
            await self._load_config()
            if not cognitive_config.get_bool("procedural_recorder_enabled"):
                return RecordingReceipt("disabled")
            try:
                async with session_scope() as session:
                    messages = await _history(session, chat, history_after)
                    anchor = _assistant_anchor(messages, evidence.assistant_message_id, chat)
                    if anchor is None:
                        return RecordingReceipt("rejected", reason="assistant_anchor")
                    assistant, _ = anchor
                    # Host history, not a model's claimed served IDs, is authoritative.
                    served = tuple(str(value) for value in assistant.message_metadata.get("served_rule_ids", ()))
                    if set(map(str, evidence.served_rule_ids)) != set(served):
                        return RecordingReceipt("rejected", reason="served_rules_mismatch")
                    metadata: dict[str, Any] = {
                        "assistant_message_id": str(assistant.id),
                        "served_rule_ids": list(served),
                        "tool_calls": [
                            {"name": tool.name, "input": dict.fromkeys(tool.argument_keys)} for tool in evidence.tools
                        ],
                        "tool_error_count": evidence.tool_error_count,
                        "input_tokens": evidence.input_tokens,
                        "output_tokens": evidence.output_tokens,
                        "stop_reason": evidence.stop_reason,
                        "pending_approval": list(evidence.pending_approval),
                    }
                    if not evidence.tools and not evidence.pending_approval:
                        return RecordingReceipt("tool_free", metadata={"served_rule_ids": list(served)})
                    # A transaction advisory lock serializes replay for this persisted message.
                    from sqlalchemy import text

                    await session.execute(
                        text("SELECT pg_advisory_xact_lock(:key)"), {"key": evidence.assistant_message_id.int % (2**63)}
                    )
                    existing = (await session.execute(_assistant_execution_lookup(assistant.id))).scalar_one_or_none()
                    if existing is not None:
                        return RecordingReceipt(
                            "duplicate",
                            existing.id,
                            {"execution_trace_id": str(existing.id), "served_rule_ids": list(served)},
                        )
                    await record_execution_trace(session, chat, metadata, task_text, raise_on_error=True)
                    trace_id = metadata.get("execution_trace_id")
                    if trace_id is None:
                        return RecordingReceipt("rejected", reason="empty_execution")
                    return RecordingReceipt(
                        "recorded", UUID(trace_id), {"execution_trace_id": trace_id, "served_rule_ids": list(served)}
                    )
            except HistoryWindowSaturated:
                return RecordingReceipt("rejected", reason="history_saturated")
            except Exception as exc:
                logger.warning("procedural.public.record_execution.failed type=%s", type(exc).__name__)
                return RecordingReceipt("failed", reason="recording_failed")

    async def capture(
        self,
        chat: Chat,
        candidates: Sequence[RuleCandidate],
        source_message_ids: Sequence[UUID],
        *,
        history_after: Any = None,
    ) -> CaptureReceipt:
        from kai_memory.procedural.rule_capture import ProceduralCaptureBusy, capture_rules
        from kai_memory.store import cognitive_config

        async with self._operation():
            policy = ports.get_host_policy()
            if not policy.may_read(chat) or not policy.may_write(chat):
                return CaptureReceipt("rejected", reason="scope_denied")
            await self._load_config()
            if not cognitive_config.get_bool("procedural_rule_extraction_enabled"):
                return CaptureReceipt("disabled")
            try:
                async with self._capture_lock, session_scope() as session:
                    messages = await _history(session, chat, history_after)
                    wanted = set(source_message_ids)
                    window = [message for message in messages if message.id in wanted]
                    if len(window) != len(wanted) or not wanted:
                        return CaptureReceipt("rejected", reason="source_anchor")
                    decisions: list[str] = []
                    changed = await capture_rules(
                        session,
                        chat,
                        [asdict(candidate) for candidate in candidates],
                        window,
                        None,
                        respect_of=policy.respect,
                        raise_on_error=True,
                        decisions=decisions,
                        replay_guard=True,
                        served_history=messages,
                    )
                    return CaptureReceipt("completed", changed, tuple(decisions))
            except ProceduralCaptureBusy:
                return CaptureReceipt("failed", reason="capture_busy")
            except HistoryWindowSaturated:
                return CaptureReceipt("rejected", reason="history_saturated")
            except Exception as exc:
                logger.warning("procedural.public.capture.failed type=%s", type(exc).__name__)
                return CaptureReceipt("failed", reason="capture_rolled_back")

    async def serve(self, chat: Chat) -> ServingReceipt:
        from kai_memory.procedural.procedural_serving import get_learned_behaviors

        async with self._operation():
            if not ports.get_host_policy().may_read(chat):
                return ServingReceipt(None, status="scope_denied")
            await self._load_config()
            try:
                async with session_scope() as session:
                    block, ids = await get_learned_behaviors(session, chat, raise_on_error=True)
                    return ServingReceipt(block, tuple(ids))
            except Exception as exc:
                logger.warning("procedural.public.serve.failed type=%s", type(exc).__name__)
                return ServingReceipt(None, status="failed")

    async def observe(
        self,
        chat: Chat,
        assistant_message_id: UUID,
        *,
        outcome: str | None = None,
        evidence_quote: str = "",
        compliance: Mapping[UUID, str] | None = None,
        history_after: Any = None,
    ) -> OutcomeReceipt:
        from kai_memory.procedural.rule_capture import record_message_anchored_compliance
        from kai_memory.procedural.procedural_serving import _in_scope
        from kai_memory.storage.models.execution_trace import ExecutionTrace
        from kai_memory.storage.models.memory_trace import MemoryTrace
        from kai_memory.store import cognitive_config

        async with self._operation():
            policy = ports.get_host_policy()
            if not policy.may_read(chat) or not policy.may_write(chat):
                return OutcomeReceipt("rejected", reason="scope_denied")
            await self._load_config()
            if not cognitive_config.get_bool("procedural_outcome_enabled"):
                return OutcomeReceipt("disabled")
            if outcome not in (None, "success", "failure") or any(
                value not in PERSISTED_VERDICTS for value in (compliance or {}).values()
            ):
                return OutcomeReceipt("rejected", reason="invalid_verdict")
            try:
                async with session_scope() as session:
                    messages = await _history(session, chat, history_after)
                    anchored = _assistant_anchor(messages, assistant_message_id, chat)
                    if anchored is None:
                        return OutcomeReceipt("rejected", reason="assistant_anchor")
                    assistant, actor = anchored
                    served = set(map(str, assistant.message_metadata.get("served_rule_ids", ())))
                    verdicts = {str(key): value for key, value in (compliance or {}).items()}
                    if not set(verdicts) <= served:
                        return OutcomeReceipt("rejected", reason="rule_not_served")
                    trace = (
                        await session.execute(_assistant_execution_lookup(assistant.id).with_for_update())
                    ).scalar_one_or_none()
                    if trace is None and (trace_id := assistant.message_metadata.get("execution_trace_id")):
                        trace = (
                            await session.execute(
                                select(ExecutionTrace).where(ExecutionTrace.id == UUID(str(trace_id))).with_for_update()
                            )
                        ).scalar_one_or_none()
                    expected_ref = f"{chat.venue}:{chat.chat_id}:{chat.topic_id}" if chat.chat_id is not None else None
                    if trace is not None and (
                        trace.source_user_id != actor
                        or trace.conversation_ref != expected_ref
                        # Headless venues share a null reference; retain their recorded venue boundary.
                        or (expected_ref is None and trace.success_signals.get("venue") != chat.venue)
                    ):
                        return OutcomeReceipt("rejected", reason="trace_scope")
                    if trace is None and any(verdict not in LEDGER_VERDICTS for verdict in verdicts.values()):
                        return OutcomeReceipt("rejected", reason="verdict_not_recordable")
                    rows: Sequence[MemoryTrace] = []
                    if verdicts:
                        rows = (
                            (
                                await session.execute(
                                    select(MemoryTrace)
                                    .where(
                                        MemoryTrace.id.in_([UUID(key) for key in verdicts]),
                                        MemoryTrace.invalidated_at.is_(None),
                                        MemoryTrace.memory_type == "procedural",
                                    )
                                    .order_by(MemoryTrace.id)
                                    .with_for_update()
                                )
                            )
                            .scalars()
                            .all()
                        )
                        if len(rows) != len(verdicts):
                            return OutcomeReceipt("rejected", reason="missing_rule")
                        from kai_memory.procedural.procedural_scope import PROC_KINDS

                        for row in rows:
                            proc = (row.typed_metadata or {}).get("procedural") or {}
                            person = (proc.get("scope") or {}).get("person")
                            if proc.get("kind") not in PROC_KINDS or not _in_scope(proc, chat):
                                return OutcomeReceipt("rejected", reason="rule_scope")
                            if not (policy.is_admin(actor) or (person is not None and person == actor)):
                                return OutcomeReceipt("rejected", reason="untrusted_actor")
                    updated = False
                    if outcome is not None:
                        following = messages[messages.index(assistant) + 1 :]
                        speaker = resolve_evidence_speaker(evidence_quote, following)
                        if speaker is None or not (policy.is_admin(speaker) or speaker == actor):
                            return OutcomeReceipt("rejected", reason="evidence_speaker")
                        if trace is None:
                            return OutcomeReceipt("rejected", reason="tool_free_outcome")
                        if trace.outcome == "unknown":
                            trace.outcome = outcome
                            trace.how_measured = "l2_turn_reflection"
                            from kai_memory.procedural.rule_capture import _norm

                            trace.success_signals = {
                                **(trace.success_signals or {}),
                                "outcome_evidence": _norm(evidence_quote)[:300],
                                "outcome_at": utc_now().isoformat(),
                            }
                            updated = True
                    count = 0
                    if verdicts:
                        for row in rows:
                            verdict = verdicts[str(row.id)]
                            if trace is not None:
                                signals = dict(trace.success_signals or {})
                                existing = dict(signals.get("rule_compliance") or {})
                                if str(row.id) in existing:
                                    continue
                                existing[str(row.id)] = verdict
                                trace.success_signals = {**signals, "rule_compliance": existing}
                                count += 1
                            elif verdict in LEDGER_VERDICTS:
                                count += await record_message_anchored_compliance(
                                    session,
                                    {
                                        str(row.id): (
                                            verdict,
                                            MessageAnchor(str(assistant.id), actor, assistant.created_at),
                                        )
                                    },
                                    None,
                                    actor_is_trusted=lambda value, person: (
                                        policy.is_admin(value) or (person is not None and person == value)
                                    ),
                                )
                    return OutcomeReceipt("recorded" if updated or count else "duplicate", updated, count)
            except HistoryWindowSaturated:
                return OutcomeReceipt("rejected", reason="history_saturated")
            except Exception as exc:
                logger.warning("procedural.public.observe.failed type=%s", type(exc).__name__)
                return OutcomeReceipt("failed", reason="observation_rolled_back")
