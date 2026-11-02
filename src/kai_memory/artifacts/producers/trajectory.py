"""TrajectoryProducer — design §3.3.4.

Builds time-ordered narratives from trace clusters around a single
entity. Wave 2 ships single-entity-only; multi-entity narratives are
M3+. Append-mode for existing-open trajectories: a new event extends
the existing artifact's payload via supersession (each pipeline run
emits a fresh row that supersedes the prior open row). Closed
trajectories never re-supersede; new events on a closed subject create
a new trajectory.

Cap: ``cognitive_config.trajectory_max_per_run`` (default 20). One
Sonnet call per subject summarizes the events together — cheaper than
per-event summarization and keeps the narrative voice consistent.
"""

from __future__ import annotations

from kai_memory.host_calls import get_model_roles

import logging
import uuid as _uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.storage.models.memory_trace import MemoryTrace
from kai_memory.store import cognitive_config
from kai_memory.host_calls import (
    call_llm_text,
    get_provider_for_model,
    get_settings,
    parse_json_array_response,
    start_generation_span,
    utc_now,
)
from kai_memory.artifact_schemas import TrajectoryEvent, TrajectoryPayload
from kai_memory.artifacts.producer import BaseProducer, ProducerSubject
from kai_memory.artifacts.types import ArtifactType, ProducedArtifact

logger = logging.getLogger("app.services.memory.artifacts.producers.trajectory")


# Lookback for the recent_clusters CTE — design §3.3.4 hardcodes 30 days.
# Trajectories are about narrative arcs; events older than 30 days that
# never coalesced into a trajectory are unlikely to retroactively fold
# into a new one.
_LOOKBACK_DAYS = 30

# §3.3.4 close criterion: 14 days without new events ⇒ status='closed'.
_CLOSE_DAYS = 14

# Sonnet prompt template — single call per subject summarises all events.
# Keep instructions minimal to control token cost; Wave 3 may add an
# instruction to surface relationship arcs once the cutover lands.
_SUMMARY_PROMPT_TEMPLATE = """\
You are summarising a sequence of memory traces about ONE subject (entity_id={entity_id}).
Return a JSON array; one object per trace, in the order given.
Each object: {{"trace_id": "...", "summary": "<one sentence, factual, <= 25 words>"}}.

Traces (chronological):
{traces}

Output JSON only, no commentary."""


class TrajectoryProducer(BaseProducer):
    """Per-subject trajectory artifact producer."""

    name = "TrajectoryProducer"
    version = 1
    artifact_type: ArtifactType = "Trajectory"

    async def discover_subjects(self, session: AsyncSession, run_id: _uuid.UUID) -> list[ProducerSubject]:
        cap = int(cognitive_config.get("trajectory_max_per_run"))
        min_count = int(cognitive_config.get("trajectory_event_min_count"))
        result = await session.execute(
            text(
                """
                WITH recent_clusters AS (
                    SELECT mel.entity_id,
                           COUNT(*) AS trace_count,
                           MIN(mt.created_at) AS first_trace_at,
                           MAX(mt.created_at) AS last_trace_at
                      FROM memory_entity_links mel
                      JOIN memory_traces mt ON mt.id = mel.memory_id
                     WHERE mt.invalidated_at IS NULL
                       AND mt.created_at > CAST(:now AS timestamptz) - (:lookback_days * INTERVAL '1 day')
                     GROUP BY mel.entity_id
                    HAVING COUNT(*) >= :min_count
                ),
                existing_trajectories AS (
                    SELECT subject_entity_id,
                           MAX(created_at) AS latest_at,
                           BOOL_OR((payload->>'status')::text = 'open') AS has_open
                      FROM memory_artifacts
                     WHERE artifact_type = 'Trajectory' AND status = 'active'
                       AND jsonb_array_length(payload->'subject_entity_ids') = 1
                     GROUP BY subject_entity_id
                )
                SELECT rc.entity_id,
                       rc.first_trace_at,
                       rc.last_trace_at,
                       COALESCE(ext.has_open, FALSE) AS has_open,
                       ext.latest_at AS existing_latest_at
                  FROM recent_clusters rc
                  LEFT JOIN existing_trajectories ext ON ext.subject_entity_id = rc.entity_id
                 WHERE
                       ext.subject_entity_id IS NULL
                    OR (NOT ext.has_open AND rc.last_trace_at > ext.latest_at)
                    OR (ext.has_open AND rc.last_trace_at > ext.latest_at)
                 ORDER BY rc.last_trace_at DESC
                 LIMIT :cap
                """
            ),
            {"cap": cap, "min_count": min_count, "lookback_days": _LOOKBACK_DAYS, "now": utc_now()},
        )
        rows = result.mappings().all()
        if not rows:
            return []
        logger.debug("TrajectoryProducer discovered %d subjects (cap=%d)", len(rows), cap)
        return [
            ProducerSubject(
                key=str(row["entity_id"]),
                payload={
                    "entity_id": row["entity_id"],
                    "first_trace_at": row["first_trace_at"],
                    "last_trace_at": row["last_trace_at"],
                    "has_open": bool(row["has_open"]),
                    "existing_latest_at": row["existing_latest_at"],
                },
            )
            for row in rows
        ]

    async def produce(
        self,
        session: AsyncSession,
        subject: ProducerSubject,
        run_id: _uuid.UUID,
    ) -> ProducedArtifact | None:
        entity_id = subject.payload["entity_id"]
        has_open = subject.payload["has_open"]
        existing_latest_at = subject.payload.get("existing_latest_at")

        # Clamp the fetch window:
        # - has_open=True   → only events newer than the existing trajectory
        # - has_open=False  → start of the discovery window (last 30 days)
        # Without the False-branch clamp we'd reload the entity's entire
        # history even though discover_subjects only flagged it on the
        # 30-day window.
        discovery_window_start = utc_now() - timedelta(days=_LOOKBACK_DAYS)
        since = existing_latest_at if has_open else discovery_window_start
        traces = await self._fetch_traces(session, entity_id, since=since)
        if not traces:
            # has_open=True but the open trajectory already covers all traces
            # in the lookback window — nothing new to append.
            return None

        prev = await self._fetch_active(session, entity_id, dimension=None)
        prior_events = self._prior_events_from_artifact(prev) if has_open and prev else []

        # One Sonnet call summarises every trace in this run's batch.
        new_events = await self._summarise_events(entity_id, traces)
        all_events = prior_events + new_events

        # Close the trajectory if the most recent event is older than the
        # close threshold — even though we just added events, if those
        # events are themselves stale (rare), close.
        last_ts = max(ev.timestamp for ev in all_events)
        now = utc_now()
        is_closed = (now - last_ts) > timedelta(days=_CLOSE_DAYS)
        status: Literal["open", "closed"] = "closed" if is_closed else "open"

        first_ts = self._payload_opened_at(prev) if has_open and prev else min(ev.timestamp for ev in all_events)

        payload_model = TrajectoryPayload(
            subject_entity_ids=[entity_id],
            events=all_events,
            status=status,
            opened_at=first_ts,
            closed_at=last_ts if is_closed else None,
        )

        return ProducedArtifact(
            artifact_type="Trajectory",
            schema_version=1,
            subject_entity_id=entity_id,
            dimension=None,
            payload=payload_model.model_dump(mode="json"),
            human_summary=_compose_summary(entity_id, len(all_events), status),
        )

    # ── helpers ───────────────────────────────────────────

    async def _fetch_traces(
        self,
        session: AsyncSession,
        entity_id: _uuid.UUID,
        since: datetime | None,
    ) -> list[MemoryTrace]:
        # Raw SELECT against memory_entity_links to avoid the cross-package
        # MemoryEntityLink ORM coupling. Hydrating MemoryTrace objects keeps
        # the rest of produce() typed.
        sql = """
            SELECT mt.id
              FROM memory_entity_links mel
              JOIN memory_traces mt ON mt.id = mel.memory_id
             WHERE mel.entity_id = :entity_id
               AND mt.invalidated_at IS NULL
        """
        if since is not None:
            sql += " AND mt.created_at > :since"
        sql += " ORDER BY mt.created_at ASC"

        params: dict[str, Any] = {"entity_id": entity_id}
        if since is not None:
            params["since"] = since

        ids = [row[0] for row in (await session.execute(text(sql), params)).all()]
        if not ids:
            return []
        result = await session.execute(
            select(MemoryTrace).where(MemoryTrace.id.in_(ids)).order_by(MemoryTrace.created_at)
        )
        return list(result.scalars().all())

    def _prior_events_from_artifact(self, artifact: Any) -> list[TrajectoryEvent]:
        """Decode existing payload's events array into TrajectoryEvent objects."""
        if artifact is None:
            return []
        payload = artifact.payload or {}
        events_raw = payload.get("events", [])
        out: list[TrajectoryEvent] = []
        for ev in events_raw:
            try:
                out.append(TrajectoryEvent.model_validate(ev))
            except Exception:
                logger.warning("TrajectoryProducer: dropping malformed prior event %s", ev)
        return out

    def _payload_opened_at(self, artifact: Any) -> datetime:
        payload = artifact.payload or {}
        opened_at = payload.get("opened_at")
        if isinstance(opened_at, str):
            return datetime.fromisoformat(opened_at)
        if isinstance(opened_at, datetime):
            if opened_at.tzinfo is None:
                return opened_at.replace(tzinfo=UTC)
            return opened_at
        return utc_now()

    async def _summarise_events(
        self,
        entity_id: _uuid.UUID,
        traces: list[MemoryTrace],
    ) -> list[TrajectoryEvent]:
        """Single Sonnet call summarises every trace in ``traces``.

        On Sonnet failure, falls back to a deterministic content-prefix
        summary so the artifact still ships — design §3.3.4 prefers shipping
        an artifact with a less-polished summary over skipping the producer.
        """
        traces_text = "\n".join(f"- [{t.id}] {t.created_at.isoformat()} :: {_truncate(t.content, 280)}" for t in traces)
        prompt = _SUMMARY_PROMPT_TEMPLATE.format(entity_id=entity_id, traces=traces_text)
        summary_by_id = await self._call_sonnet_summaries(prompt)
        events: list[TrajectoryEvent] = []
        for trace in traces:
            summary = summary_by_id.get(str(trace.id)) or _fallback_summary(trace)
            ts = trace.created_at if trace.created_at.tzinfo is not None else trace.created_at.replace(tzinfo=UTC)
            events.append(
                TrajectoryEvent(
                    trace_id=trace.id,
                    timestamp=ts,
                    summary=summary,
                )
            )
        return events

    async def _call_sonnet_summaries(self, prompt: str) -> dict[str, str]:
        """Run the Sonnet summarisation prompt and return ``{trace_id: summary}``."""
        try:
            provider = get_provider_for_model(get_settings(), get_model_roles().artifact)
            span = start_generation_span("consolidation/trajectory-summary", get_model_roles().artifact)
            # max_tokens=8192: trajectory summaries ship one JSON entry per
            # trace in the batch. With trace_count up to the
            # _LOOKBACK_DAYS-bounded batch size, 4096 occasionally truncated
            # the JSON tail — producing low-quality / fallback-only events
            # for the latter half of the array. Same fix shape as
            # PatternProducer's 2048→8192 bump (PR #122). Sonnet's pricing
            # only charges for emitted tokens so the ceiling is a safety
            # bound, not a cost lever.
            raw = await call_llm_text(
                provider,
                prompt,
                max_tokens=8192,
                span=span,
                thinking_mode="disabled",
                task_name="memory.trajectory.summary",
            )
        except Exception:
            logger.warning(
                "TrajectoryProducer: Sonnet call failed; falling back to deterministic summaries",
                exc_info=True,
            )
            return {}

        if not raw:
            return {}
        items = parse_json_array_response(raw) or []
        out: dict[str, str] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            tid = item.get("trace_id")
            summary = item.get("summary")
            if isinstance(tid, str) and isinstance(summary, str) and summary.strip():
                out[tid] = summary.strip()
        return out


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def _fallback_summary(trace: MemoryTrace) -> str:
    """Deterministic fallback when Sonnet doesn't return a summary for this trace."""
    head = _truncate(trace.content.replace("\n", " ").strip(), 120)
    return f"[{trace.memory_type}] {head}"


def _compose_summary(entity_id: _uuid.UUID, n_events: int, status: str) -> str:
    return f"Trajectory<{entity_id}> {n_events} event{'s' if n_events != 1 else ''} ({status})"
