"""IdentityRecordProducer — design §3.3.5.

Detects directional drift in the agent's behaviour from
``self_reflection`` traces and ships one structured record per
``(subject, dimension)``. Wave 2 ships **agent-self only**;
user-targeted IdentityRecords are M3+.

Discovery: agent's canonical self-entity + ≥3 self_reflection traces
inside the drift window (default 7 days). One Sonnet call returns the
list of dimensions with directional drift. Per-dimension supersession
respects the ``idx_artifacts_one_active`` partial unique index.

§4.1 invariant: ``subject_entity_id`` MUST be non-NULL on every
produced artifact (Postgres treats NULLs as distinct in unique
indexes — without a non-NULL subject the partial unique index would
allow two active records for the same dimension).
"""

from __future__ import annotations

from kai_memory.host_calls import get_model_roles

from kai_memory.identity import resolve_agent_identity

import json
import logging
import re
import uuid as _uuid
from datetime import datetime, timedelta, UTC
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.host_calls import (
    call_llm_text,
    get_provider_for_model,
    get_settings,
    start_generation_span,
    utc_now,
)
from kai_memory.storage.models.memory_trace import MemoryTrace
from kai_memory.store import cognitive_config
from kai_memory.artifact_schemas import IdentityRecordPayload
from kai_memory.artifacts._dimension_normalizer import (
    canonical_list_for_prompt,
    normalize as normalize_dimension,
)
from kai_memory.artifacts.producer import BaseProducer, ProducerSubject
from kai_memory.artifacts.types import ArtifactType, ProducedArtifact

logger = logging.getLogger("app.services.memory.artifacts.producers.identity_record")


_AGENT_SELF_ENTITY_TYPE: str = "agent"

# Shared enum for Sonnet's drift direction. Frozen at module load so
# repeated membership checks in _filter_eligible don't reallocate each
# call (per AGENTS.md: "Define shared constants using frozenset in the
# module that owns the enum for repeated enum checks").
_ALLOWED_DIRECTIONS: frozenset[str] = frozenset({"increasing", "decreasing", "stable"})

# §3.3.5 prompt — Sonnet returns a JSON array of drifted dimensions.
# Magnitude is a float in [0.0, 1.0]; direction is one of
# {increasing, decreasing, stable}. Stable + magnitude=0 is permitted
# but ignored at the apply step (no record created for nil drift).
_DRIFT_PROMPT_TEMPLATE = """\
You are analysing the agent's self-reflection traces from the last {window} day(s).

Traces (chronological):
{trace_summaries}

Identify dimensions where the agent shows DIRECTIONAL drift.

PREFERRED DIMENSION VOCABULARY (use these names when they fit; coin a new
snake_case name only when none of these capture the drift):
{canonical_dimensions}

For each dimension with drift, return a JSON object:
{{ "dimension": "...", "direction": "increasing" | "decreasing" | "stable",
   "magnitude": 0.0-1.0, "driving_trace_ids": ["..."] }}

Rules:
- Prefer the vocabulary above; emit a NEW snake_case name only if no listed
  name fits the drift you're describing.
- Only return dimensions with magnitude >= {min_magnitude:.2f}.
- driving_trace_ids must reference trace ids from the list above.
- If no clear drift, return [].
- Output JSON ARRAY only, no commentary.
"""


class IdentityRecordProducer(BaseProducer):
    """Per-dimension drift records for the agent's self-entity."""

    name = "IdentityRecordProducer"
    version = 1
    artifact_type: ArtifactType = "IdentityRecord"

    async def discover_subjects(self, session: AsyncSession, run_id: _uuid.UUID) -> list[ProducerSubject]:
        """Pull self-reflection traces, run drift analysis ONCE, return one
        subject per eligible dimension.

        Discovering per-dimension lets the pipeline's per-subject ``produce``
        loop materialise multiple ``IdentityRecord`` artifacts in a single
        pass — design §3.3.5 explicitly allows multiple records when each
        targets a distinct ``(subject_entity_id, dimension)`` pair, and the
        partial unique index ``idx_artifacts_one_active`` enforces uniqueness
        within that key, not within the producer.
        """
        agent_self_id = await self._get_agent_self_entity_id(session)
        if agent_self_id is None:
            logger.debug(
                "IdentityRecordProducer skipped: no configured agent-self entity found",
            )
            return []

        window_days = float(cognitive_config.get("identity_record_drift_window_days"))
        cutoff = utc_now() - timedelta(days=window_days)
        result = await session.execute(
            select(MemoryTrace)
            .where(
                MemoryTrace.memory_type == "self_reflection",
                MemoryTrace.invalidated_at.is_(None),
                MemoryTrace.created_at > cutoff,
            )
            .order_by(MemoryTrace.created_at)
        )
        traces = list(result.scalars().all())
        if len(traces) < 3:
            logger.debug(
                "IdentityRecordProducer skipped: only %d self_reflection traces in last %.1f days (need 3+)",
                len(traces),
                window_days,
            )
            return []

        dimensions = await self._call_sonnet_drift_analysis_for_traces(traces, window_days)
        eligible = self._filter_eligible(dimensions)
        if not eligible:
            return []

        valid_trace_ids = {t.id for t in traces}
        # Trace-derived epoch_start: the earliest trace's timestamp. Used
        # by produce() so re-runs with the same trace pool produce the
        # same content_hash (stability is what makes BaseProducer.supersede()
        # short-circuit to action='noop' instead of churning supersessions
        # every L3 pass). The window cutoff would shift between runs; this
        # anchor doesn't.
        earliest_ts = _ensure_aware(min(t.created_at for t in traces))
        return [
            ProducerSubject(
                key=f"agent-self:{agent_self_id}::{drift['dimension']}",
                payload={
                    "agent_self_id": agent_self_id,
                    "drift": drift,
                    "valid_trace_ids": valid_trace_ids,
                    "fallback_trace_ids": [t.id for t in traces],
                    "epoch_start_seed": earliest_ts,
                },
            )
            for drift in eligible
        ]

    async def produce(
        self,
        session: AsyncSession,
        subject: ProducerSubject,
        run_id: _uuid.UUID,
    ) -> ProducedArtifact | None:
        drift: dict[str, Any] = subject.payload["drift"]
        agent_self_id: _uuid.UUID = subject.payload["agent_self_id"]
        valid_trace_ids: set[_uuid.UUID] = subject.payload["valid_trace_ids"]
        fallback_trace_ids: list[_uuid.UUID] = subject.payload["fallback_trace_ids"]

        # Validate driving_trace_ids against the in-window trace pool so
        # Sonnet hallucinations don't pollute the payload.
        driving: list[_uuid.UUID] = []
        for tid in drift.get("driving_trace_ids", []):
            if not isinstance(tid, str):
                continue
            try:
                parsed = _uuid.UUID(tid)
            except (TypeError, ValueError):
                continue
            if parsed in valid_trace_ids:
                driving.append(parsed)
        if not driving:
            # Fallback to the in-window traces themselves; better than
            # an empty list when Sonnet fails to attribute.
            driving = list(fallback_trace_ids)

        # Stable epoch_start: the earliest trace timestamp from the
        # in-window pool, computed once at discovery and passed through.
        # This anchors the artifact to evidence rather than to clock time
        # — re-running L3 against the same trace pool produces an identical
        # content_hash, so BaseProducer.supersede() short-circuits to
        # action='noop' on the next nightly run.
        epoch_start: datetime = subject.payload["epoch_start_seed"]

        payload_model = IdentityRecordPayload(
            subject_entity_id=agent_self_id,
            dimension=drift["dimension"],
            direction=drift["direction"],
            magnitude=_coerce_magnitude(drift.get("magnitude")),
            driving_observations=driving,
            epoch_start=epoch_start,
            epoch_end=None,
        )

        return ProducedArtifact(
            artifact_type="IdentityRecord",
            schema_version=1,
            subject_entity_id=agent_self_id,
            dimension=drift["dimension"],
            payload=payload_model.model_dump(mode="json"),
            human_summary=(
                f"agent {drift['direction']} on {drift['dimension']} "
                f"(magnitude {_coerce_magnitude(drift.get('magnitude')):.2f})"
            ),
        )

    # ── helpers ───────────────────────────────────────────

    async def _get_agent_self_entity_id(self, session: AsyncSession) -> _uuid.UUID | None:
        entity = await resolve_agent_identity(session)
        return entity.id if entity else None

    async def _call_sonnet_drift_analysis_for_traces(
        self,
        traces: list[MemoryTrace],
        window_days: float,
    ) -> list[dict[str, Any]]:
        """Single Sonnet call across all in-window traces.

        Called once per ``discover_subjects`` invocation; the returned list
        seeds per-dimension subjects.
        """
        min_magnitude = float(cognitive_config.get("identity_record_min_magnitude"))
        trace_summaries = "\n".join(
            f"- [{t.id}] {t.created_at.isoformat()} :: {_truncate(t.content, 320)}" for t in traces
        )
        prompt = _DRIFT_PROMPT_TEMPLATE.format(
            window=int(window_days),
            trace_summaries=trace_summaries,
            min_magnitude=min_magnitude,
            canonical_dimensions=canonical_list_for_prompt(),
        )
        try:
            provider = get_provider_for_model(get_settings(), get_model_roles().artifact)
            span = start_generation_span("consolidation/identity-drift", get_model_roles().artifact)
            # max_tokens=8192: drift JSON is one array entry per dimension
            # with magnitude + driving_trace_ids. 2048 truncated mid-array on
            # prod (CloudWatch: chronic ~9/24h truncations 2026-05-09/10),
            # parser saw incomplete JSON and returned [], producer committed
            # 0 artifacts. Re-diagnosed the "drift vs snapshot" 0-output as
            # the same bug class as #122's PatternProducer truncation.
            # See pattern_producer.py for the thinking_mode="disabled"
            # rationale — same bug class, same evidence base.
            raw = await call_llm_text(
                provider,
                prompt,
                max_tokens=8192,
                span=span,
                thinking_mode="disabled",
                task_name="memory.identity.drift",
            )
        except Exception:
            logger.warning(
                "IdentityRecordProducer: Sonnet call failed; skipping",
                exc_info=True,
            )
            return []
        return _parse_drift_response(raw)

    def _filter_eligible(self, dimensions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply min_magnitude + structural filters to Sonnet's drift list.

        Returns one entry per dimension that should yield an artifact (the
        partial unique index already deduplicates by ``(subject, dimension)``,
        so we pass everything that passes the threshold). Coerces magnitude
        defensively — Sonnet occasionally returns it as a string.
        """
        min_magnitude = float(cognitive_config.get("identity_record_min_magnitude"))
        # Dict keyed by normalized dimension so two raw synonyms (e.g.
        # "context_isolation_discipline" and "context_bleed_control" both
        # normalizing to "context_discipline") collapse into ONE candidate.
        # Without this, BaseProducer.supersede() would still keep the DB
        # honest via the partial unique index, but we'd waste an extra
        # produce() call per collision. Keep the highest-magnitude variant
        # so the strongest evidence wins; ties keep the first occurrence
        # (Sonnet's own ordering, which usually places strongest first).
        by_canonical: dict[str, dict[str, Any]] = {}
        for d in dimensions:
            magnitude = _coerce_magnitude(d.get("magnitude"))
            if magnitude < min_magnitude:
                continue
            dimension = d.get("dimension")
            direction = d.get("direction")
            if not dimension or not isinstance(dimension, str):
                continue
            if direction not in _ALLOWED_DIRECTIONS:
                continue
            # Soft-whitelist normalization: collapse known synonyms; unknown
            # names pass through so Sonnet can still coin new dimensions as
            # the system grows (see _dimension_normalizer for rationale).
            normalized = normalize_dimension(dimension)
            if not normalized:
                continue
            # Carry the coerced magnitude + normalized dimension so produce()
            # doesn't re-parse and the partial unique index sees canonical
            # names only.
            entry = {**d, "magnitude": magnitude, "dimension": normalized}
            existing = by_canonical.get(normalized)
            if existing is None or magnitude > float(existing["magnitude"]):
                by_canonical[normalized] = entry
        eligible: list[dict[str, Any]] = list(by_canonical.values())
        # Sort by magnitude descending so multi-dimension runs surface the
        # strongest drift first in changes_made (purely cosmetic — the
        # pipeline aggregates all of them).
        eligible.sort(key=lambda d: float(d["magnitude"]), reverse=True)
        return eligible


# ── parsing helpers ───────────────────────────────────────


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_drift_response(raw: str | None) -> list[dict[str, Any]]:
    """Tolerant parse: strip code-fences, locate the outermost JSON array."""
    if not raw:
        return []
    text_ = raw.strip()
    # Strip optional ```json ... ``` fences.
    if text_.startswith("```"):
        text_ = re.sub(r"^```[a-zA-Z]*\n?", "", text_)
        text_ = re.sub(r"\n?```$", "", text_).strip()
    match = _JSON_ARRAY_RE.search(text_)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("IdentityRecordProducer: malformed Sonnet JSON: %s", text_[:200])
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def _coerce_magnitude(value: Any) -> float:
    """Float-coerce Sonnet's magnitude reply, clamping to [0.0, 1.0].

    Sonnet occasionally returns ``"0.4"`` or ``null``; the partial
    unique index doesn't care, but ``IdentityRecordPayload`` will reject
    a non-float at validation time and the > comparison in the eligibility
    filter would raise ``TypeError`` against a non-numeric value. Coerce
    with a 0.0 default.
    """
    if value is None:
        return 0.0
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not _isfinite(coerced):
        return 0.0
    return max(0.0, min(1.0, coerced))


def _isfinite(x: float) -> bool:
    import math

    return math.isfinite(x)


def _ensure_aware(ts: datetime) -> datetime:
    """Coerce a naive datetime to UTC. Aware datetimes pass through."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts


__all__ = [
    "IdentityRecordProducer",
    "_coerce_magnitude",  # exposed for unit tests
    "_parse_drift_response",  # exposed for unit tests
]
