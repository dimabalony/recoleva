"""Procedural-memory recorder — Phase A dual-write (design §3).

Invoked at the post-turn point of ALL THREE chat_engine tool loops (mirror of
``schedule_consolidation``; enforced by ``test_triple_loop_sync``). For a
completed tool-using turn it writes BOTH stores:

- **Store A** (``execution_traces``): the raw, cold execution log.
- **Store B** (``memory_traces`` ``memory_type='procedural'``): a low-salience
  CANDIDATE procedure, held out of every L1/L2/L3 read until serving is enabled.

Gated by ``procedural_recorder_enabled`` (default on). Fail-open: recording must
never break a turn. Everything text is ``scrub_secrets``-ed; ``steps`` carry tool
names + argument-key SHAPES only, never values/outputs.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.storage.models.execution_trace import ExecutionTrace
from kai_memory.store import cognitive_config
from kai_memory.host_calls import redact_urls_for_trace, scrub_secrets, strip_msg_meta_envelope
from kai_memory.store.cognitive_memory import CognitiveMemoryService

if TYPE_CHECKING:
    from kai_memory.ports import ChatContext

# Pinned to the pre-DB-842 module path: CloudWatch queries, feature-watch criteria and
# log filters match the logger name, which must not follow the module's move.
_LEGACY_LOGGER_NAME = "app.services.chat.procedural_recorder"
logger = logging.getLogger(_LEGACY_LOGGER_NAME)

_CANDIDATE_SALIENCE = 0.1  # Store B ships low-salience; the L3 utility pass owns promotion.
_WHITESPACE = re.compile(r"\s+")
# idx_exec_traces_serving btree-indexes the FULL task_signature, and Postgres
# btree index rows are BYTE-limited (~2704 bytes for btree v4; prod hit an 8191
# ceiling). A long ask (8816 bytes seen in prod) overflows the index on flush
# (ProgramLimitExceededError) → the recorder's savepoint rolls back → the record
# is silently dropped. Cap the normalized signature well under the limit, by
# ENCODED BYTES (Cyrillic is 2 bytes/char in UTF-8; char count alone is unsafe).
_SIGNATURE_MAX_BYTES = 2000


def _truncate_to_byte_cap(text: str, max_bytes: int) -> str:
    """Truncate ``text`` to at most ``max_bytes`` UTF-8 bytes without splitting a
    multibyte character (``errors="ignore"`` drops any dangling partial sequence),
    then rstrip so a cut landing on a space stays idempotent."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore").rstrip()


def normalize_signature(task_text: str) -> str:
    """Stable, scrubbed signature of the *kind* of ask: strip the ``[msg_meta: {...}]``
    envelope the chat pipeline prepends, then scrub secrets, collapse whitespace, casefold.

    Clustering into "same ask" is done at eval time by embedding cosine over this
    string, so the envelope MUST be removed first: its unique ``message_id``/``time``
    inject per-row noise and its constant ``sender`` tag inflates cross-ask similarity,
    which would corrupt the recurrence signal. Public because the serving path shares it."""
    stripped = strip_msg_meta_envelope(task_text or "")
    privacy_safe = redact_urls_for_trace(scrub_secrets(stripped)) or ""
    normalized = _WHITESPACE.sub(" ", privacy_safe).strip().casefold()
    # Truncate LAST so both the record and serve paths (which share this function)
    # cap identically and exact-match serving stays consistent. See _SIGNATURE_MAX_BYTES.
    return _truncate_to_byte_cap(normalized, _SIGNATURE_MAX_BYTES)


def _steps_from_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool NAMES + argument-key SHAPES, never values. A malformed entry degrades
    to just its name rather than raising."""
    steps: list[dict[str, Any]] = []
    for tc in tool_calls:
        name = tc.get("name")
        if not name:
            continue
        raw = tc.get("input")
        arg_keys = sorted((redact_urls_for_trace(str(key)) or "") for key in raw) if isinstance(raw, dict) else []
        safe_name = redact_urls_for_trace(scrub_secrets(str(name))) or ""
        steps.append({"tool": safe_name, "arg_keys": arg_keys})
    return steps


async def record_execution_trace(
    session: AsyncSession,
    chat_ctx: ChatContext,
    metadata: dict[str, Any],
    task_text: str | None,
    *,
    raise_on_error: bool = False,
) -> None:
    """Dual-write Store A + Store B for a completed turn. No-op when disabled, on a
    tool-less turn, or with no task text. Fail-open — swallows every error."""
    try:
        if not cognitive_config.get_bool("procedural_recorder_enabled"):
            return
        # System turns (heartbeat/wake) ARE recorded — they carry sender_user_id=NULL,
        # which keeps them distinguishable for a future heartbeat self-improvement loop.
        # Serving and the recurrence eval exclude them; the cold log stays complete.
        tool_calls = metadata.get("tool_calls") or []
        # Phase C §5.2.1 — headless STALL record: a headless turn that returned at an
        # approval gate never ran its tools, so it has no completed tool_calls. The
        # caller passes metadata["pending_approval"] = [tool names]; those pending
        # tools count as steps so the nightly headless scorer (§5.2.2) can mark the
        # turn a failure (pending_approval present) instead of leaving it 'unknown'.
        pending_approval = metadata.get("pending_approval") or []
        if pending_approval and not tool_calls:
            tool_calls = [{"name": name} for name in pending_approval]
        signature = normalize_signature(task_text or "")
        if not tool_calls or not signature:
            return  # a procedure is a tool sequence for a real ask; nothing else qualifies

        steps = _steps_from_tool_calls(tool_calls)
        if not steps:
            return  # tool_calls present but every entry was nameless → no real procedure
        success_signals = {
            "tool_call_count": len(tool_calls),
            "tool_error_count": int(metadata.get("tool_error_count", 0) or 0),
            "stop_reason": metadata.get("stop_reason"),
            "output_tokens": int(metadata.get("output_tokens", 0) or 0),
            # Phase C §1.2 — venue distinguishes heartbeat vs wake (both NULL sender);
            # served_rule_ids joins every serving to its L2 compliance row (§4.1);
            # pending_approval marks a headless stall (§5.2.2 failure signal).
            "venue": chat_ctx.venue,
        }
        if metadata.get("assistant_message_id"):
            success_signals["assistant_message_id"] = str(metadata["assistant_message_id"])
        if metadata.get("served_rule_ids"):
            success_signals["served_rule_ids"] = metadata["served_rule_ids"]
        if pending_approval:
            success_signals["pending_approval"] = pending_approval
        cost_tokens = int(metadata.get("input_tokens", 0) or 0) + int(metadata.get("output_tokens", 0) or 0)
        conversation_ref = None
        if chat_ctx.chat_id is not None:
            conversation_ref = f"{chat_ctx.venue}:{chat_ctx.chat_id}:{chat_ctx.topic_id}"

        # SAVEPOINT: isolate both writes so a recorder failure rolls back cleanly
        # and NEVER leaves the shared session's transaction in an aborted state
        # (which would blow up the next write in the same turn).
        async with session.begin_nested():
            # Store A — the raw cold log. task_embedding stays NULL (batch-embedded
            # later by the eval, never on the hot path).
            exec_trace = ExecutionTrace(
                task_signature=signature,
                steps=steps,
                outcome="unknown",  # multi-turn signals (redo, follow-up) upgrade this later
                how_measured="phase_a_write_time_signals",
                success_signals=success_signals,
                cost_tokens=cost_tokens,
                conversation_ref=conversation_ref,
                source_user_id=chat_ctx.sender_user_id,
            )
            session.add(exec_trace)
            await session.flush()  # need exec_trace.id for Store B provenance

            # Store B — low-salience CANDIDATE procedure in memory_traces, held out
            # of every L1/L2/L3 read (procedural exclusion). Status / provenance /
            # validity live in typed_metadata (no schema change).
            tool_names = [s["tool"] for s in steps]
            content = scrub_secrets(f"Procedure for: {signature} — tools: {', '.join(tool_names)}")
            await CognitiveMemoryService(session).store_procedural_candidate(
                content,
                salience=_CANDIDATE_SALIENCE,
                source_chat_id=chat_ctx.chat_id,
                source_topic_id=chat_ctx.topic_id,
                source_user_id=chat_ctx.sender_user_id,
                typed_metadata={
                    "procedural": {
                        "status": "candidate",
                        "execution_trace_id": str(exec_trace.id),  # provenance FK → Store A
                        "tool_names": tool_names,
                        "valid_from": None,  # validity interval; invalidate-don't-delete via invalidated_at
                        "valid_until": None,
                    }
                },
            )
        # Phase C §1.2 — stamp the trace id back onto the turn metadata AFTER the
        # savepoint commits, so the handler persists it into the assistant row's
        # message_metadata. L2 Part 3 joins turn outcomes to the trace through it.
        metadata["execution_trace_id"] = str(exec_trace.id)

        # Observability + feature-watch marker (a few tool-using turns/day; INFO).
        logger.info("procedural.recorded exec_id=%s steps=%d cost_tokens=%d", exec_trace.id, len(steps), cost_tokens)
    except Exception as exc:
        # Instrumentation must NEVER break a turn.
        logger.warning("procedural recorder failed type=%s", type(exc).__name__)
        if raise_on_error:
            raise
