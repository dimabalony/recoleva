"""StanceProducer — Sub-design D Wave 2 §3.4.

Kai's own positions / opinions / takes. Sourced from agent
self-reflection traces (which include thinking-block self-corrections
ingested at L2) and contradiction signals. All Stances key on
``subject_entity_id = kai_self_entity_id`` so the partial unique index
``idx_artifacts_one_active`` enforces per-(subject, topic) uniqueness
without bumping into Postgres' NULL-distinct semantics — the same
reason IdentityRecord uses a non-NULL subject (M2 §4.1 invariant).

Pipeline shape:

1. ``discover_subjects`` — single Sonnet call over recent
   self_reflection traces returning candidate (topic, position,
   reasoning, confidence) tuples + zero-or-more
   reinforcement/challenge events to merge into existing stances.
2. ``produce`` — emits ``StancePayload`` with
   ``reinforcement_signals`` / ``challenge_signals`` initialized from
   the candidate's discovered events.
3. ``supersede`` (overridden) — if an active Stance for the topic
   already exists, append the discovered events into its
   ``reinforcement_signals`` / ``challenge_signals`` arrays IN PLACE
   without superseding (no new row, no version bump). Confidence is
   updated in place too. The BaseProducer 5-step supersede only fires
   on position change OR confidence drift past ``_REFINEMENT_DELTA``;
   challenge-majority does NOT currently trigger supersession or
   decrement confidence (see "Deferred" below).

This append-in-place departure from the M2 default supersede is a
locked design call (§3.4 + §3.6: "ReinforcementEvent appended (no
supersession); when sufficient new evidence accumulates, StanceProducer
may supersede the artifact with a refined position").

**Deferred** — §3.4 says confidence "drops with invalidations" but no
code path decrements on ``challenge_signals`` accumulation. Today a
stance can hit 10 challenges against 2 reinforcements and stay at
confidence=0.9. ``_expire_stale_stances`` gate 2 (staleness) catches
this indirectly within ``persona_stance_retention_days`` as Sonnet
stops reinforcing the topic — but the direct "confidence drops on
challenge majority" path is not implemented. Filed as a follow-up.
"""

from __future__ import annotations

from kai_memory.host_calls import get_model_roles

from kai_memory.identity import resolve_agent_identity

import json
import logging
import re
import uuid as _uuid
from datetime import timedelta
from typing import Any

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.host_calls import (
    call_llm_text,
    get_provider_for_model,
    get_settings,
    start_generation_span,
    utc_now,
)
from kai_memory.storage.models.memory_artifact import MemoryArtifact
from kai_memory.storage.models.memory_trace import MemoryTrace
from kai_memory.store import cognitive_config
from kai_memory.artifact_schemas import (
    ChallengeEvent,
    ReinforcementEvent,
    StancePayload,
)
from kai_memory.artifacts.producer import (
    BaseProducer,
    ProducerSubject,
    compute_content_hash,
)
from kai_memory.artifacts.types import (
    Artifact,
    ArtifactType,
    ChangesMadeEntry,
    ProducedArtifact,
)

logger = logging.getLogger("app.services.memory.artifacts.producers.stance_producer")


_STANCE_LOOKBACK_DAYS = 14

# Confidence delta that triggers position-refinement supersession.
# When the new candidate's confidence differs from the existing row's by
# at least this much, we issue a real BaseProducer.supersede instead of
# appending in place. Keeps stance prose stable across small noise but
# allows the position to evolve when the evidence shifts decisively.
_REFINEMENT_DELTA = 0.25

_STANCE_PROMPT_TEMPLATE = """\
You are analysing the agent's recent self-reflection traces from the
last {window} day(s) to identify the agent's OWN POSITIONS / OPINIONS
("stances") on topics.

Traces (chronological):
{traces}

For each stance the agent demonstrated, return a JSON object:
{{
  "topic": "<snake_case, <= 64 chars; e.g. hype_skepticism>",
  "position": "<<= 250 chars; the agent's view stated in plain prose>",
  "reasoning": "<<= 500 chars; why the agent holds this view>",
  "confidence": 0.0-1.0,
  "events": [
    {{
      "kind": "agreed" | "outcome_validated" | "held_under_pushback"
            | "disagreed" | "outcome_invalidated",
      "trace_id": "<one of the trace ids above, or null>",
      "note": "<<= 200 chars optional context>"
    }}
  ]
}}

Rules:
- "topic" must be snake_case.
- "position" addresses the agent in second person ("You believe ...")
  OR states the position factually. No instructions to the user.
- "events" can be empty.
- "kind" determines whether the event is reinforcement (agreed,
  outcome_validated, held_under_pushback) or challenge (disagreed,
  outcome_invalidated).
- Only include stances with confidence >= 0.4 (per
  persona_stance_confidence_floor).
- If no stances surfaced, return [].
- Output JSON ARRAY only, no commentary, no code fences.
"""


class StanceProducer(BaseProducer):
    """Per-topic Stance artifacts for the agent self-entity."""

    name = "StanceProducer"
    version = 1
    artifact_type: ArtifactType = "Stance"

    async def discover_subjects(self, session: AsyncSession, run_id: _uuid.UUID) -> list[ProducerSubject]:
        agent_self_id = await self._get_kai_self_entity_id(session)
        if agent_self_id is None:
            logger.debug("StanceProducer skipped: no kai-self entity in memory_entities")
            return []

        # Step 0 — expire stale stances before capacity is computed. Mirrors
        # TraitProducer._demote_below_threshold (called at top of its
        # discover_subjects). Two-gate OR: confidence below floor (Sonnet
        # wouldn't accept this topic today) OR last_validated_at older than
        # the retention window (Sonnet stopped refreshing this stance). Pre-fix,
        # the 6-stance cap blocked all new stances forever once filled — see
        # `backend-docs/ai-workforce/plans/design/2026-05-06-memory-v2d-persona-pipeline-design.md`
        # §3.6 "Stance challenged + invalidated → marks as expired".
        await self._expire_stale_stances(session, agent_self_id)

        traces = await self._fetch_recent_traces(session)
        if len(traces) < 2:
            logger.debug(
                "StanceProducer skipped: only %d self_reflection traces (need 2+)",
                len(traces),
            )
            return []

        candidates = await self._call_sonnet_stance_extraction(traces)
        if not candidates:
            return []

        # Cap. Same-topic candidates append/supersede an existing row
        # (no capacity consumed). Fresh topics consume capacity.
        cap = int(cognitive_config.get("persona_max_active_stances"))
        active_topics = await self._fetch_active_stance_topics(session, agent_self_id)
        capacity_remaining = max(cap - len(active_topics), 0)

        valid_trace_ids = {t.id for t in traces}

        accepted: list[ProducerSubject] = []
        dropped_topics: list[str] = []
        for c in candidates:
            topic = c["topic"]
            is_existing = topic in active_topics
            if not is_existing and capacity_remaining <= 0:
                # No silent drops — accumulate and log once at end of loop.
                # When the cap bites we want a CloudWatch trail so the
                # next "0 produced" night is diagnosable from logs alone.
                dropped_topics.append(topic)
                continue
            if not is_existing:
                capacity_remaining -= 1
            accepted.append(
                ProducerSubject(
                    key=f"stance::{agent_self_id}::{topic}",
                    payload={
                        "agent_self_id": agent_self_id,
                        "topic": topic,
                        "position": c["position"],
                        "reasoning": c["reasoning"],
                        "confidence": c["confidence"],
                        "events": c["events"],
                        "valid_trace_ids": valid_trace_ids,
                    },
                )
            )
        if dropped_topics:
            # First 5 topics are enough to spot a pattern; longer lists are
            # redundant since they all dropped for the same reason.
            logger.info(
                "StanceProducer: %d candidate(s) dropped at cap=%d (active=%d, topics=%s)",
                len(dropped_topics),
                cap,
                len(active_topics),
                dropped_topics[:5],
            )
        return accepted

    async def produce(
        self,
        session: AsyncSession,
        subject: ProducerSubject,
        run_id: _uuid.UUID,
    ) -> ProducedArtifact | None:
        agent_self_id: _uuid.UUID = subject.payload["agent_self_id"]
        topic: str = subject.payload["topic"]
        position: str = subject.payload["position"]
        reasoning: str = subject.payload["reasoning"]
        confidence: float = subject.payload["confidence"]
        events_raw: list[dict[str, Any]] = subject.payload.get("events", [])
        valid_trace_ids: set[_uuid.UUID] = subject.payload["valid_trace_ids"]

        reinforcement_signals, challenge_signals = _split_events(events_raw, valid_trace_ids)
        now = utc_now()
        payload_model = StancePayload(
            topic=topic,
            position=position,
            reasoning=reasoning,
            reinforcement_signals=reinforcement_signals,
            challenge_signals=challenge_signals,
            confidence=confidence,
            last_validated_at=now,
        )

        # Stable content_hash on the position + reasoning + confidence
        # subset. Excludes signals + last_validated_at because both are
        # mutated in-place during reinforcement append.
        content_hash = compute_content_hash(
            {
                "subject_entity_id": str(agent_self_id),
                "topic": topic,
                "position": position,
                "reasoning": reasoning,
                "confidence": round(confidence, 4),
            }
        )

        return ProducedArtifact(
            artifact_type="Stance",
            schema_version=1,
            subject_entity_id=agent_self_id,
            dimension=topic,  # M2 partial unique index keys on dimension; reuse for topic
            payload=payload_model.model_dump(mode="json"),
            content_hash=content_hash,
            human_summary=(
                f"stance[{topic}] conf={confidence:.2f} +{len(reinforcement_signals)}/-{len(challenge_signals)}"
            ),
        )

    async def supersede(
        self,
        session: AsyncSession,
        prev: Artifact | None,
        new: ProducedArtifact,
        run_id: _uuid.UUID,
    ) -> ChangesMadeEntry:
        """Stance-specific supersede: append-in-place by default; supersede only on big drift.

        - prev present + small confidence delta + position unchanged →
          append the new payload's reinforcement/challenge events into
          the existing row's signal arrays. No new row, no supersession.
        - prev present + significant drift (confidence delta >=
          ``_REFINEMENT_DELTA`` OR new position differs from existing) →
          fall through to the BaseProducer default for the supersede
          transaction.
        - prev None → BaseProducer 'create' branch.

        Auto-expiry of low-confidence stances happens via
        ``_expire_low_confidence`` called from ``discover_subjects``-time
        cleanup; that step is intentionally lightweight and SQL-only so
        it doesn't take a Sonnet trip.
        """
        producer_name = new.producer_name or self.name
        producer_version = new.producer_version if new.producer_version is not None else self.version

        if prev is None:
            return await super().supersede(session, prev, new, run_id)

        # Permanent guard handled by base.
        if prev.status == "permanent":
            return await super().supersede(session, prev, new, run_id)

        prev_payload = prev.payload or {}
        prev_position = str(prev_payload.get("position", ""))
        prev_confidence = _coerce_float(prev_payload.get("confidence"))
        new_position = str(new.payload.get("position", ""))
        new_confidence = _coerce_float(new.payload.get("confidence"))

        position_changed = prev_position.strip() != new_position.strip()
        confidence_drift = abs(new_confidence - prev_confidence)

        if position_changed or confidence_drift >= _REFINEMENT_DELTA:
            # Real supersession — refined position or large confidence
            # shift. Defer to BaseProducer's 5-step transaction.
            return await super().supersede(session, prev, new, run_id)

        # Append-in-place. Merge new events into existing arrays without
        # creating a new row. This is the "Stance reinforced → append
        # without supersession" path from design §3.6.
        new_reinforcements: list[dict[str, Any]] = list(new.payload.get("reinforcement_signals", []))
        new_challenges: list[dict[str, Any]] = list(new.payload.get("challenge_signals", []))

        if not new_reinforcements and not new_challenges:
            # Nothing new to record — short-circuit to noop. This is the
            # exact same shape BaseProducer.supersede uses for hash-match
            # noops; pipeline sees a single audit row with action='noop'.
            #
            # Touch ``last_validated_at`` on the row so the staleness gate
            # in ``_expire_stale_stances`` resets — "I still hold this view,
            # no new evidence" is itself a validation signal. Without this
            # touch, stable opinions that Sonnet keeps surfacing (but finds
            # no new events for) would drift to "stale" within
            # ``persona_stance_retention_days`` and be expired.
            touched_payload = dict(prev_payload)
            touched_payload["last_validated_at"] = utc_now().isoformat()
            await session.execute(
                update(MemoryArtifact).where(MemoryArtifact.id == prev.id).values(payload=touched_payload)
            )
            return ChangesMadeEntry(
                action="noop",
                artifact_id=prev.id,
                prev_artifact_id=prev.id,
                artifact_type=self.artifact_type,
                subject_entity_id=new.subject_entity_id,
                human_summary=f"stance[{prev_payload.get('topic')}] no new evidence",
                producer_name=producer_name,
                producer_version=producer_version,
            )

        merged_reinforcements = list(prev_payload.get("reinforcement_signals", [])) + new_reinforcements
        merged_challenges = list(prev_payload.get("challenge_signals", [])) + new_challenges
        # Keep arrays bounded — older signals fall off when capacity hit.
        # Cap at 50 each: enough history to compute drift, small enough
        # the JSONB row stays under reasonable size.
        merged_reinforcements = merged_reinforcements[-50:]
        merged_challenges = merged_challenges[-50:]

        updated_payload = dict(prev_payload)
        updated_payload["reinforcement_signals"] = merged_reinforcements
        updated_payload["challenge_signals"] = merged_challenges
        updated_payload["last_validated_at"] = utc_now().isoformat()
        # Confidence: weighted average; weight new sample at 0.3 to keep
        # the position stable under noise.
        updated_payload["confidence"] = round(prev_confidence * 0.7 + new_confidence * 0.3, 4)

        await session.execute(
            update(MemoryArtifact).where(MemoryArtifact.id == prev.id).values(payload=updated_payload)
        )

        return ChangesMadeEntry(
            action="supersede",  # in-place mutation — recorded as supersede so rollback can revert payload
            artifact_id=prev.id,
            prev_artifact_id=prev.id,
            artifact_type=self.artifact_type,
            subject_entity_id=new.subject_entity_id,
            human_summary=(
                f"stance[{prev_payload.get('topic')}] appended "
                f"+{len(new_reinforcements)} reinforcement / "
                f"+{len(new_challenges)} challenge events"
            ),
            producer_name=producer_name,
            producer_version=producer_version,
        )

    # ── helpers ───────────────────────────────────────────

    async def _get_kai_self_entity_id(self, session: AsyncSession) -> _uuid.UUID | None:
        entity = await resolve_agent_identity(session)
        return entity.id if entity else None

    async def _fetch_active_stance_topics(
        self,
        session: AsyncSession,
        agent_self_id: _uuid.UUID,
    ) -> set[str]:
        result = await session.execute(
            text(
                """
                SELECT dimension FROM memory_artifacts
                 WHERE artifact_type = 'Stance'
                   AND status = 'active'
                   AND subject_entity_id = :subject
                """
            ),
            {"subject": agent_self_id},
        )
        return {row[0] for row in result.all() if row[0] is not None}

    async def _expire_stale_stances(
        self,
        session: AsyncSession,
        agent_self_id: _uuid.UUID,
    ) -> int:
        """Expire active Stances that drift below confidence floor or stop refreshing.

        Pure SQL UPDATE, runs at top of ``discover_subjects`` before capacity is
        computed. Matches ``TraitProducer._demote_below_threshold`` shape: silent
        UPDATE + INFO log, no separate audit emit. The audit trail is the row's
        own ``status`` history — query ``memory_artifacts`` for
        ``status='superseded' AND superseded_by_artifact_id IS NULL`` to see
        system-driven expiries.

        Two-gate OR semantics:

        1. ``confidence < persona_stance_confidence_floor`` (default 0.4) — if
           Sonnet's discovery wouldn't accept this topic today (same floor as
           ``_parse_stance_response`` filtering), the stance shouldn't be active.
        2. ``last_validated_at < now() - persona_stance_retention_days`` (default
           7) — if Sonnet hasn't refreshed this stance in the retention window,
           it's gone cold. Matches ``persona_pattern_window_days`` precedent
           (artifact lifespan axis), not ``_STANCE_LOOKBACK_DAYS`` (the
           extraction input axis — wrong dimension).
        """
        floor = float(cognitive_config.get("persona_stance_confidence_floor"))
        retention_days = float(cognitive_config.get("persona_stance_retention_days"))
        cutoff = utc_now() - timedelta(days=retention_days)

        result = await session.execute(
            text(
                """
                UPDATE memory_artifacts
                   SET status = 'superseded'
                 WHERE artifact_type = 'Stance'
                   AND status = 'active'
                   AND subject_entity_id = :subject
                   AND (
                           COALESCE((payload->>'confidence')::float, 0.0) < :floor
                        OR COALESCE(
                               (payload->>'last_validated_at')::timestamptz,
                               '1970-01-01T00:00:00+00:00'::timestamptz
                           ) < :cutoff
                   )
                RETURNING dimension,
                          (payload->>'confidence')::float AS conf,
                          payload->>'last_validated_at' AS last_validated_at
                """
            ),
            {"subject": agent_self_id, "floor": floor, "cutoff": cutoff},
        )
        rows = result.all()
        if rows:
            # Log per-stance detail so a future CloudWatch grep can see which
            # gate fired for each — confidence-gate hits show conf<floor;
            # staleness-gate hits show recent conf with stale timestamp.
            logger.info(
                "StanceProducer: expired %d stale stance(s) (floor=%.2f, retention_days=%.0f, cutoff=%s); details=%s",
                len(rows),
                floor,
                retention_days,
                cutoff.isoformat(),
                [{"dim": r.dimension, "conf": r.conf, "last_validated_at": r.last_validated_at} for r in rows],
            )
        return len(rows)

    async def _fetch_recent_traces(self, session: AsyncSession) -> list[MemoryTrace]:
        cutoff = utc_now() - timedelta(days=_STANCE_LOOKBACK_DAYS)
        result = await session.execute(
            select(MemoryTrace)
            .where(
                MemoryTrace.memory_type == "self_reflection",
                MemoryTrace.invalidated_at.is_(None),
                MemoryTrace.created_at > cutoff,
            )
            .order_by(MemoryTrace.created_at)
            .limit(50)
        )
        return list(result.scalars().all())

    async def _call_sonnet_stance_extraction(
        self,
        traces: list[MemoryTrace],
    ) -> list[dict[str, Any]]:
        trace_lines = "\n".join(f"- [{t.id}] {t.created_at.isoformat()} :: {_truncate(t.content, 280)}" for t in traces)
        prompt = _STANCE_PROMPT_TEMPLATE.format(
            window=_STANCE_LOOKBACK_DAYS,
            traces=trace_lines,
        )
        try:
            provider = get_provider_for_model(get_settings(), get_model_roles().artifact)
            span = start_generation_span("consolidation/stance-extract", get_model_roles().artifact)
            # max_tokens=8192: stance JSON is one array entry per dimension
            # with stance value + driving_trace_ids. Same truncation tail as
            # IdentityRecord — bumped together since both used 2048 and
            # showed 0-output on prod.
            # See pattern_producer.py for the thinking_mode="disabled"
            # rationale — same bug class, same evidence base.
            raw = await call_llm_text(
                provider,
                prompt,
                max_tokens=8192,
                span=span,
                thinking_mode="disabled",
                task_name="memory.stance.extract",
            )
        except Exception:
            logger.warning(
                "StanceProducer: Sonnet call failed; skipping",
                exc_info=True,
            )
            return []
        return _parse_stance_response(raw)


# ── parsing helpers ───────────────────────────────────────


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_stance_response(raw: str | None) -> list[dict[str, Any]]:
    if not raw:
        return []
    text_ = raw.strip()
    if text_.startswith("```"):
        text_ = re.sub(r"^```[a-zA-Z]*\n?", "", text_)
        text_ = re.sub(r"\n?```$", "", text_).strip()
    match = _JSON_ARRAY_RE.search(text_)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("StanceProducer: malformed Sonnet JSON: %s", text_[:200])
        return []
    if not isinstance(parsed, list):
        return []
    floor = float(cognitive_config.get("persona_stance_confidence_floor"))
    cleaned: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        topic = item.get("topic")
        position = item.get("position")
        reasoning = item.get("reasoning")
        confidence = _coerce_float(item.get("confidence"))
        if not isinstance(topic, str) or not topic.strip():
            continue
        if not isinstance(position, str) or not position.strip():
            continue
        if not isinstance(reasoning, str) or not reasoning.strip():
            continue
        if confidence < floor:
            continue
        events = item.get("events") or []
        if not isinstance(events, list):
            events = []
        cleaned.append(
            {
                "topic": topic.strip()[:64],
                "position": position.strip()[:300],
                "reasoning": reasoning.strip()[:600],
                "confidence": min(1.0, max(0.0, confidence)),
                "events": [e for e in events if isinstance(e, dict)],
            }
        )
    return cleaned


_REINFORCEMENT_KINDS = {"agreed", "outcome_validated", "held_under_pushback"}
_CHALLENGE_KINDS = {"disagreed", "outcome_invalidated"}


def _split_events(
    events_raw: list[dict[str, Any]],
    valid_trace_ids: set[_uuid.UUID],
) -> tuple[list[ReinforcementEvent], list[ChallengeEvent]]:
    """Split Sonnet's events list into typed reinforcement/challenge arrays.

    Sonnet hallucinates trace_ids occasionally — drop any that aren't in
    ``valid_trace_ids`` instead of polluting the payload (matches
    IdentityRecord's defence pattern).
    """
    reinforcements: list[ReinforcementEvent] = []
    challenges: list[ChallengeEvent] = []
    now = utc_now()
    for ev in events_raw:
        kind = ev.get("kind")
        if kind not in _REINFORCEMENT_KINDS and kind not in _CHALLENGE_KINDS:
            continue
        trace_id_raw = ev.get("trace_id")
        trace_uuid: _uuid.UUID | None = None
        if isinstance(trace_id_raw, str):
            try:
                parsed = _uuid.UUID(trace_id_raw)
            except (TypeError, ValueError):
                parsed = None
            if parsed is not None and parsed in valid_trace_ids:
                trace_uuid = parsed
        note = ev.get("note") or ""
        if not isinstance(note, str):
            note = ""
        note = note[:300]
        if kind in _REINFORCEMENT_KINDS:
            try:
                reinforcements.append(
                    ReinforcementEvent(
                        kind=kind,  # type: ignore[arg-type]
                        trace_id=trace_uuid,
                        occurred_at=now,
                        note=note,
                    )
                )
            except Exception:
                continue
        else:
            try:
                challenges.append(
                    ChallengeEvent(
                        kind=kind,  # type: ignore[arg-type]
                        trace_id=trace_uuid,
                        occurred_at=now,
                        note=note,
                    )
                )
            except Exception:
                continue
    return reinforcements, challenges


def _coerce_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


__all__ = [
    "StanceProducer",
    "_parse_stance_response",
]
