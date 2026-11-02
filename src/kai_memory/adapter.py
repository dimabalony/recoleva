"""MemoryInterface adapter wrapping CognitiveMemoryService.

v2a §3.5. Concrete implementation of the v2 Protocol — every new consumer
should target this adapter rather than `CognitiveMemoryService` directly.
Existing direct callers are grandfathered in `tests/test_memory_interface_discipline.py::ALLOWED_DIRS`.

For M1, the adapter delegates:
- `retrieve` → `MemoryPrimingService.prime` (shaped into RetrievalResult).
- `encode` → `CognitiveMemoryService.store_trace` + `link_trace_to_entity`.
- `consolidate(level=3, reason="nightly")` → `consolidation.consolidate()`.
  Other levels/reasons are accepted but not yet wired (eager L2 / deep
  encoding ride with Sub-design C).
- `feedback("strong_correction")` → `invalidate_trace`. Other signal kinds
  are recorded as `noop` until Sub-design C lights up the inhibition path.
- `export_artifacts` → empty `AsyncIterator[Artifact]` (no producers in M1;
  the M2 producers will populate it).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.retrieval.memory_priming import MemoryPrimingService
from kai_memory.store.cognitive_memory import CognitiveMemoryService
from kai_memory.types import (
    Artifact,
    ArtifactQuery,
    ConsolidationReport,
    ConsolidationTrigger,
    Cue,
    EncodingResult,
    EncodingSignals,
    Experience,
    FeedbackResult,
    FeedbackSignal,
    FeedbackTarget,
    RetrievalContext,
    RetrievalResult,
    TracePrime,
)


class CognitiveMemoryAdapter:
    """Wraps `CognitiveMemoryService` to implement `MemoryInterface`."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.memory_svc = CognitiveMemoryService(session)
        self.priming_svc = MemoryPrimingService(session)

    async def retrieve(self, cues: list[Cue], context: RetrievalContext) -> RetrievalResult:
        """Surface relevant traces for the cues in `context`.

        M1 routes through `MemoryPrimingService.prime_with_traces`, which
        runs the four cognitive-memory paths and returns the ranked trace
        rows directly. review-r1 fix 4: previously the adapter called
        `prime()` for IDs then re-fetched the same rows via
        `get_traces_batch` — N+1 on every chat message.
        """
        message = " ".join(_stringify_cue(c) for c in cues)
        t0 = time.time()
        _formatted, _trace_ids, top = await self.priming_svc.prime_with_traces(
            message,
            limit=context.limit,
            chat_ctx=_proxy_chat_ctx(context),
        )
        latency_ms = int((time.time() - t0) * 1000)

        # review-r1 fix 5: populate TracePrime.score with the real scoring
        # output (was hardcoded 0.0 in M1 before the review).
        # review-r2 fix 3: populate score_breakdown from the same component
        # dict the structured priming log emits — Protocol contract promises
        # `dict[str, float]`; consumers used to receive `{}` for everything.
        primed: list[TracePrime] = [
            TracePrime(
                trace_id=trace.id,
                content=trace.content,
                created_at=trace.created_at,
                memory_type=trace.memory_type,  # type: ignore[arg-type]
                score=float(score),
                score_breakdown={k: float(v) for k, v in components.items()},
            )
            for trace, score, components in top
        ]

        return RetrievalResult(
            primed_traces=primed,
            relevant_artifacts=[],  # M1 has no artifact producers
            total_candidates=len(primed),
            retrieval_latency_ms=latency_ms,
        )

    async def encode(
        self,
        experience: Experience,
        signals: EncodingSignals | None = None,
    ) -> EncodingResult:
        """Persist an experience as a trace + entity links.

        Per the Protocol contract: when both `experience.signals` and the
        explicit `signals` parameter carry data, the explicit parameter
        wins. M1 stores signals in `memory_traces.encoding_signals` JSONB
        via the new column added in migration 061; the Wave 2 store_trace
        call doesn't yet accept signals, so M1 records traces without
        signals and a follow-up wires signals.payload through.
        """
        del signals  # Reserved for Sub-design C (OEC mode + Haiku scoring).

        trace = await self.memory_svc.store_trace(
            content=experience.content,
            memory_type=experience.memory_type,
            significance=experience.significance,
            source_type="conversation",  # M1: encode is conversation-driven
            source_chat_id=experience.chat_id,
            source_topic_id=experience.topic_id,
            emotion_valence=experience.emotion_valence,
            emotion_arousal=experience.emotion_arousal,
        )
        entities_touched: list[uuid.UUID] = []
        for entity_id, role in experience.entity_links:
            await self.memory_svc.link_trace_to_entity(trace.id, entity_id, role)
            entities_touched.append(entity_id)

        return EncodingResult(
            trace_ids=[trace.id],
            entities_touched=entities_touched,
            contradictions_detected=0,  # M1 doesn't run contradiction at encode
            mode="defer",  # Sub-design C will populate eager / deep modes.
        )

    async def consolidate(self, trigger: ConsolidationTrigger) -> ConsolidationReport:
        """Delegate to the existing nightly consolidation entrypoint.

        M1 only wires `level=3, reason="nightly"`. Other combinations return
        an empty report so callers can call the method without crashing.
        """
        run_id = uuid.uuid4()
        t0 = time.time()
        if trigger.level == 3 and trigger.reason == "nightly":
            from kai_memory.consolidation import consolidation as consolidation_module

            result = await consolidation_module.consolidate()
            failures = list(result.get("failures", [])) if isinstance(result, dict) else []
            traces_created = int(result.get("new_traces", 0)) if isinstance(result, dict) else 0
            traces_invalidated = int(result.get("contradictions", 0)) if isinstance(result, dict) else 0
        else:
            failures = []
            traces_created = 0
            traces_invalidated = 0

        return ConsolidationReport(
            run_id=run_id,
            level=trigger.level,
            traces_created=traces_created,
            traces_invalidated=traces_invalidated,
            artifacts_produced={},  # No artifact producers in M1.
            duration_ms=int((time.time() - t0) * 1000),
            failures=failures,
        )

    async def feedback(self, target: FeedbackTarget, signal: FeedbackSignal) -> FeedbackResult:
        """Apply a feedback signal.

        M1 wires only `strong_correction` against a trace target — the
        invalidation path. Sub-design C lights up the inhibition + rewrite
        actions; until then the other actions return as `noop`.
        """
        if target.kind == "trace" and signal.kind == "strong_correction" and isinstance(target.id, uuid.UUID):
            await self.memory_svc.invalidate_trace(
                target.id,
                invalidated_by=signal.source_trace_id,
                confidence=0.9,
            )
            return FeedbackResult(
                action="invalidate",
                inhibition_id=None,
                new_trace_id=None,
                invalidated_artifact_ids=[],
            )

        return FeedbackResult(
            action="noop",
            inhibition_id=None,
            new_trace_id=None,
            invalidated_artifact_ids=[],
        )

    async def export_artifacts(self, query: ArtifactQuery) -> AsyncIterator[Artifact]:
        """Yield artifacts matching `query`.

        M1 has no artifact producers, so the iterator is always empty;
        the empty-list loop keeps this function an async generator (so
        callers can `async for` directly) without an unreachable `yield`.
        Sub-design B's producers will populate the body.
        """
        del query
        empty: tuple[Artifact, ...] = ()
        for artifact in empty:
            yield artifact


def _stringify_cue(cue: Cue) -> str:
    """Render a Cue value into a string fragment for priming."""
    return str(cue.value)


def _proxy_chat_ctx(context: RetrievalContext):
    """Return an object exposing chat_id / topic_id / venue for priming."""
    return _RetrievalContextChatProxy(
        chat_id=context.chat_id,
        topic_id=context.topic_id,
        venue=context.venue,
    )


class _RetrievalContextChatProxy:
    """Duck-typed adapter exposing the three attrs prime() reads."""

    __slots__ = ("chat_id", "topic_id", "venue")

    def __init__(
        self,
        chat_id: int | None,
        topic_id: int | None,
        venue: str,
    ) -> None:
        self.chat_id = chat_id
        self.topic_id = topic_id
        self.venue = venue


__all__ = ["CognitiveMemoryAdapter"]
