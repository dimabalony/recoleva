"""PersonaComposer — deterministic persona prompt-block render (Sub-design D).

Wave 1: ``compose_anchor_only()`` is what prompt-builder calls. Renders the
Anchor row + empty section headers (kept for token-boundary stability).

Wave 2: producers populate Trait / Stance / Pattern artifacts in shadow.
``compose()`` is the Wave-3-ready render path that reads them all; tests
exercise it during Wave 2 even though no caller uses it yet.

Wave 3: prompt-builder switches the call site from ``compose_anchor_only``
to ``compose`` and the Anchor-only method is deleted. Single-line
production change, easy git-revert rollback.

Determinism guarantees (design §3.7):
- All ORDER BY clauses include ``id ASC`` tie-breaks so two consecutive
  ``compose()`` calls with no DB change produce bit-identical output.
- Empty fetches still render their section header — keeps token boundaries
  stable across compose calls.
- Read-only over ``memory_artifacts``; READ COMMITTED isolation is fine
  because the partial unique index + supersede ordering ensure a coherent
  set of active rows at any snapshot.
"""

from __future__ import annotations

import logging
import uuid as _uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.host_calls import utc_now

if TYPE_CHECKING:
    from kai_memory.ports import ChatContext


logger = logging.getLogger("app.services.memory.persona_composer")


@dataclass(frozen=True)
class _AnchorRow:
    content: str


@dataclass(frozen=True)
class _TraitRow:
    guidance_text: str


@dataclass(frozen=True)
class _StanceRow:
    position: str


@dataclass(frozen=True)
class _PatternRow:
    observation: str


@dataclass(frozen=True)
class _TrajectoryEvent:
    """A single event inside a Trajectory artifact, already weight-resolved.

    ``weight`` is the access-weighted score used by the midterm top-N
    tier. Derived from ``memory_traces.access_count`` (defaults to 0
    when the underlying trace is missing or was pruned). Higher means
    "more anchored in the agent's working memory".
    """

    timestamp: datetime
    summary: str
    weight: float = 0.0


@dataclass(frozen=True)
class _TrajectoryRow:
    """One active trajectory pre-rendered into three event tiers.

    - ``events_recent``: last 90 days, verbatim (no further filtering).
    - ``events_midterm``: 90-365 days, capped to top-N by weight, ordered
      chronologically inside the tier.
    - ``older_count`` + ``older_first`` / ``older_last``: >365 days, only
      a count and time-span summary (single line at render time).
    """

    subject_name: str
    events_recent: list[_TrajectoryEvent] = field(default_factory=list)
    events_midterm: list[_TrajectoryEvent] = field(default_factory=list)
    older_count: int = 0
    older_first: datetime | None = None
    older_last: datetime | None = None


class PersonaComposer:
    """Composes the persona prompt block from typed artifacts.

    No LLM call at compose time. Pure read + render.
    """

    def __init__(self, session: AsyncSession):
        self._session = session

    async def compose_anchor_only(
        self,
        chat_ctx: ChatContext,
        budget_chars: int = 4000,
    ) -> str:
        """Wave 1 / Wave 2 render path.

        Returns Anchor content + empty Personality / Views / Recently-observed
        section headers. Wave 3 swaps the prompt-builder call site to
        ``compose()`` for the full render.

        ``chat_ctx`` is kept in the signature for API parity with ``compose()``
        so the Wave 3 cutover is a single-line swap at the call site.
        """
        del chat_ctx
        anchor = await self._fetch_anchor()
        result = self._render(anchor, [], [], [], [])
        self._budget_warn(result, budget_chars)
        return result

    async def compose(
        self,
        chat_ctx: ChatContext,
        budget_chars: int = 4000,
    ) -> str:
        """Wave 3 render path. Anchor + active Traits + Stances + Patterns + Trajectories.

        Composition order (cache-stable prefix to variable suffix):
          1. Anchor (always full)
          2. Active Traits (top N by strength * recency)
          3. Active Stances (relevance-filtered, top N)
          4. Active Patterns (last 7 days, by recency)
          5. Active Trajectories (per-subject, three-tier event filter:
             recent verbatim / midterm top-N by access weight / older
             collapsed)

        Total bounded by per-layer count caps (cognitive_config), not by
        ``budget_chars``. ``budget_chars`` is a sanity-check that emits a
        warning if exceeded — means a per-layer cap was misconfigured.
        """
        anchor = await self._fetch_anchor()
        traits = await self._fetch_active_traits()
        stances = await self._fetch_relevant_stances(chat_ctx)
        patterns = await self._fetch_recent_patterns()
        trajectories = await self._fetch_recent_trajectories()
        result = self._render(anchor, traits, stances, patterns, trajectories)
        self._budget_warn(result, budget_chars)
        return result

    # ── DB reads ──────────────────────────────────────────────

    async def _fetch_anchor(self) -> _AnchorRow | None:
        """Return the single permanent Anchor row, or None if not seeded yet.

        Determinism: orders by created_at ASC, id ASC. There SHOULD only ever
        be one Anchor row (DB trigger plus admin-path discipline), but
        ordering is explicit for repeatability if a future migration adds
        a second Anchor by accident.
        """
        row = (
            (
                await self._session.execute(
                    text(
                        """
                        SELECT payload->>'content' AS content
                          FROM memory_artifacts
                         WHERE artifact_type = 'Anchor' AND status = 'permanent'
                         ORDER BY created_at ASC, id ASC
                         LIMIT 1
                        """
                    )
                )
            )
            .mappings()
            .first()
        )
        if row is None or row["content"] is None:
            return None
        return _AnchorRow(content=str(row["content"]))

    async def _fetch_active_traits(self) -> list[_TraitRow]:
        """Active PersonalityTrait rows ordered by computed strength.

        Strength formula (design §3.3):

            strength = magnitude
                     * exp(-lambda * days_since_last_reinforced)
                     * (1 + rehearsal_count * 0.2)

        Computed SQL-side using the cognitive_config's
        ``persona_trait_decay_lambda`` so ordering is stable per row
        regardless of session timing. Tie-breaks on
        ``last_reinforced_at``, ``created_at``, ``id`` (design §3.7
        determinism guarantee).
        """
        from kai_memory.store import cognitive_config

        limit = int(cognitive_config.get("persona_max_active_traits"))
        lambda_ = float(cognitive_config.get("persona_trait_decay_lambda"))
        rows = (
            (
                await self._session.execute(
                    text(
                        """
                        SELECT payload->>'guidance_text' AS guidance_text
                          FROM memory_artifacts
                         WHERE artifact_type = 'PersonalityTrait'
                           AND status = 'active'
                         ORDER BY (
                                (payload->>'magnitude')::float
                                * exp(
                                    -CAST(:lambda_ AS double precision)
                                    * GREATEST(
                                        EXTRACT(EPOCH FROM (
                                            CAST(:now AS timestamptz)
                                            - (payload->>'last_reinforced_at')::timestamptz
                                        )) / 86400.0,
                                        0.0
                                    )
                                )
                                * (1 + COALESCE((payload->>'rehearsal_count')::int, 0) * 0.2)
                            ) DESC,
                            (payload->>'last_reinforced_at')::timestamptz DESC,
                            created_at DESC,
                            id ASC
                         LIMIT :limit
                        """
                    ),
                    {"lambda_": lambda_, "limit": limit, "now": utc_now()},
                )
            )
            .mappings()
            .all()
        )
        return [_TraitRow(guidance_text=str(r["guidance_text"])) for r in rows if r["guidance_text"]]

    async def _fetch_relevant_stances(
        self,
        chat_ctx: ChatContext,
    ) -> list[_StanceRow]:
        """Active Stance rows ordered by confidence * decay * recency.

        Wave 2 ranks by ``confidence * decay_factor`` per design §3.7;
        the decay factor uses ``persona_stance_decay_lambda`` against
        ``last_validated_at``. Wave 3 will add ``chat_ctx``-relevance
        filtering (entity overlap with the current conversation, design
        §14 deferred refinement); Wave 2 keeps it simple TopK so the
        composer ships with a stable ordering before relevance scoring
        gets tuned.
        """
        del chat_ctx  # Wave 3 will use this for entity-overlap relevance.
        from kai_memory.store import cognitive_config

        limit = int(cognitive_config.get("persona_max_active_stances"))
        lambda_ = float(cognitive_config.get("persona_stance_decay_lambda"))
        rows = (
            (
                await self._session.execute(
                    text(
                        """
                        SELECT payload->>'position' AS position
                          FROM memory_artifacts
                         WHERE artifact_type = 'Stance'
                           AND status = 'active'
                         ORDER BY (
                                (payload->>'confidence')::float
                                * exp(
                                    -CAST(:lambda_ AS double precision)
                                    * GREATEST(
                                        EXTRACT(EPOCH FROM (
                                            CAST(:now AS timestamptz)
                                            - (payload->>'last_validated_at')::timestamptz
                                        )) / 86400.0,
                                        0.0
                                    )
                                )
                            ) DESC,
                            (payload->>'last_validated_at')::timestamptz DESC,
                            created_at DESC,
                            id ASC
                         LIMIT :limit
                        """
                    ),
                    {"lambda_": lambda_, "limit": limit, "now": utc_now()},
                )
            )
            .mappings()
            .all()
        )
        return [_StanceRow(position=str(r["position"])) for r in rows if r["position"]]

    async def _fetch_recent_patterns(self) -> list[_PatternRow]:
        """Active Pattern rows within the configured window.

        Filters by ``persona_pattern_window_days`` (default 7); orders by
        most-recently-observed first; capped at
        ``persona_max_active_patterns``.

        Dimension de-duplication: ``DISTINCT ON (dimension)`` keeps only the
        freshest active row per dimension. Pattern's supersede path
        reinforces in place by content_hash, so two active rows with the
        SAME dimension already imply distinct observations — but the
        composer surfaces only one per dimension to keep the persona prompt
        crisp. Persona payload normalization (see
        ``_dimension_normalizer.normalize``) ensures the dimension column
        is canonicalized at write time, so DISTINCT collapses synonyms
        too. Inner query orders by ``dimension, observed_at DESC,
        created_at DESC, id ASC`` to satisfy Postgres' rule that DISTINCT
        ON columns must lead ORDER BY; outer query re-orders for the
        composer.

        IdentityRecord intentionally does NOT use DISTINCT ON: the
        ``idx_artifacts_one_active`` partial unique index guarantees
        at most one active row per ``(subject_entity_id, dimension)``
        pair already.
        """
        from kai_memory.store import cognitive_config

        limit = int(cognitive_config.get("persona_max_active_patterns"))
        window_days = float(cognitive_config.get("persona_pattern_window_days"))
        rows = (
            (
                await self._session.execute(
                    text(
                        """
                        SELECT observation
                          FROM (
                            SELECT DISTINCT ON (payload->>'dimension')
                                   payload->>'observation' AS observation,
                                   (payload->>'observed_at')::timestamptz AS observed_at,
                                   created_at,
                                   id
                              FROM memory_artifacts
                             WHERE artifact_type = 'Pattern'
                               AND status = 'active'
                               AND (payload->>'observed_at')::timestamptz
                                   > CAST(:now AS timestamptz) - (:window_days * INTERVAL '1 day')
                             ORDER BY payload->>'dimension',
                                      (payload->>'observed_at')::timestamptz DESC,
                                      created_at DESC,
                                      id ASC
                          ) AS deduped
                         ORDER BY observed_at DESC,
                                  created_at DESC,
                                  id ASC
                         LIMIT :limit
                        """
                    ),
                    {"window_days": window_days, "limit": limit, "now": utc_now()},
                )
            )
            .mappings()
            .all()
        )
        return [_PatternRow(observation=str(r["observation"])) for r in rows if r["observation"]]

    async def _fetch_recent_trajectories(self) -> list[_TrajectoryRow]:
        """Active Trajectory artifacts pre-filtered into three event tiers.

        Wave 3 cutover blocker — on prod the largest Trajectory had
        1046 events / 145 KB (2026-05-18). Rendering verbatim blows the
        prompt budget on the first call. The three-tier filter caps each
        trajectory to ``persona_trajectory_max_events`` and folds
        long-tail history into a single collapsed line.

        Tier rules (config-driven):

        1. Last ``persona_trajectory_recent_window_days`` (default 90d):
           verbatim, no further filtering.
        2. ``recent_window..older_window`` (default 90d-365d): top-N by
           access-weighted score, ordered chronologically.
        3. Older than ``persona_trajectory_older_window_days`` (default
           365d): collapsed into a single "N events between X and Y" line.

        Weight = ``memory_traces.access_count`` for the underlying trace
        (defaults to 0 when the trace was pruned or never linked). This
        matches the agent's working-memory signal — events whose source
        trace gets touched repeatedly are anchor events, even if old.

        Trajectories themselves are capped at
        ``persona_max_active_trajectories`` per render, ordered by most
        recent ``opened_at`` first (the trajectory's narrative arc start)
        so the persona block reflects what Kai is currently "tracking".
        """
        from kai_memory.store import cognitive_config

        traj_limit = int(cognitive_config.get("persona_max_active_trajectories"))
        max_events = int(cognitive_config.get("persona_trajectory_max_events"))
        recent_days = float(cognitive_config.get("persona_trajectory_recent_window_days"))
        older_days = float(cognitive_config.get("persona_trajectory_older_window_days"))

        artifacts = (
            (
                await self._session.execute(
                    text(
                        """
                        SELECT a.id AS artifact_id,
                               COALESCE(e.name, '(unknown subject)') AS subject_name,
                               a.payload->'events' AS events
                          FROM memory_artifacts a
                          LEFT JOIN memory_entities e ON e.id = a.subject_entity_id
                         WHERE a.artifact_type = 'Trajectory'
                           AND a.status = 'active'
                         ORDER BY (a.payload->>'opened_at')::timestamptz DESC,
                                  a.created_at DESC,
                                  a.id ASC
                         LIMIT :traj_limit
                        """
                    ),
                    {"traj_limit": traj_limit},
                )
            )
            .mappings()
            .all()
        )
        if not artifacts:
            return []

        # Collect every trace_id we'll need an access_count for, then run
        # ONE batch lookup. Avoids N+1 even on the worst-case 1046-event
        # trajectory (one round-trip per render regardless of size).
        all_trace_ids: set[_uuid.UUID] = set()
        for art in artifacts:
            events_raw = art["events"] or []
            for ev in events_raw:
                # Legacy or corrupt payloads can carry scalars / nulls in the
                # events array; calling ``.get()`` on those throws
                # AttributeError and aborts compose(). Skip silently and let
                # the per-event parser apply the same guard downstream.
                if not isinstance(ev, dict):
                    continue
                tid = ev.get("trace_id")
                if isinstance(tid, str):
                    try:
                        all_trace_ids.add(_uuid.UUID(tid))
                    except ValueError:
                        continue
        access_by_trace: dict[_uuid.UUID, int] = {}
        if all_trace_ids:
            access_rows = (
                await self._session.execute(
                    text(
                        """
                        SELECT id, COALESCE(access_count, 0) AS access_count
                          FROM memory_traces
                         WHERE id = ANY(:ids)
                        """
                    ),
                    {"ids": list(all_trace_ids)},
                )
            ).all()
            access_by_trace = {row.id: int(row.access_count) for row in access_rows}

        now = utc_now()
        recent_cutoff = now - timedelta(days=recent_days)
        older_cutoff = now - timedelta(days=older_days)
        rows: list[_TrajectoryRow] = []
        for art in artifacts:
            events_raw = art["events"] or []
            recent: list[_TrajectoryEvent] = []
            midterm: list[_TrajectoryEvent] = []
            older_first: datetime | None = None
            older_last: datetime | None = None
            older_count = 0
            for ev in events_raw:
                parsed = self._parse_trajectory_event(ev, access_by_trace)
                if parsed is None:
                    continue
                if parsed.timestamp >= recent_cutoff:
                    recent.append(parsed)
                elif parsed.timestamp >= older_cutoff:
                    midterm.append(parsed)
                else:
                    older_count += 1
                    older_first = parsed.timestamp if older_first is None else min(older_first, parsed.timestamp)
                    older_last = parsed.timestamp if older_last is None else max(older_last, parsed.timestamp)

            # Recent tier is verbatim — but we still cap so a freak 500-event
            # last-90-days trajectory can't blow the whole budget alone. Keep
            # the most recent N when the cap bites; that preserves the "what
            # just happened" signal which is the recent tier's whole point.
            recent.sort(key=lambda e: e.timestamp, reverse=True)
            recent = recent[:max_events]
            recent.sort(key=lambda e: e.timestamp)

            slots_left = max(0, max_events - len(recent))
            if slots_left and midterm:
                # Weight-rank then re-chronologically-order so the rendered
                # block reads as a coherent timeline, not as a top-N list.
                midterm = sorted(midterm, key=lambda e: e.weight, reverse=True)[:slots_left]
                midterm.sort(key=lambda e: e.timestamp)
            else:
                midterm = []

            rows.append(
                _TrajectoryRow(
                    subject_name=str(art["subject_name"]),
                    events_recent=recent,
                    events_midterm=midterm,
                    older_count=older_count,
                    older_first=older_first,
                    older_last=older_last,
                )
            )
        return rows

    @staticmethod
    def _parse_trajectory_event(
        ev: Any,
        access_by_trace: dict[_uuid.UUID, int],
    ) -> _TrajectoryEvent | None:
        """Best-effort coerce a raw JSONB event into a typed row.

        Returns None if the event is malformed (not a dict, or missing
        timestamp / summary). Defensive against producer drift — an
        existing 1046-event trajectory may have legacy entries with
        slightly different shapes, including bare scalars or nulls.
        """
        if not isinstance(ev, dict):
            return None
        ts_raw = ev.get("timestamp")
        summary = ev.get("summary")
        if not isinstance(ts_raw, str) or not isinstance(summary, str):
            return None
        try:
            timestamp = datetime.fromisoformat(ts_raw)
        except ValueError:
            return None
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        weight = 0.0
        tid_raw = ev.get("trace_id")
        if isinstance(tid_raw, str):
            try:
                weight = float(access_by_trace.get(_uuid.UUID(tid_raw), 0))
            except ValueError:
                weight = 0.0
        return _TrajectoryEvent(timestamp=timestamp, summary=summary, weight=weight)

    # ── render ────────────────────────────────────────────────

    @staticmethod
    def _render(
        anchor: _AnchorRow | None,
        traits: list[_TraitRow],
        stances: list[_StanceRow],
        patterns: list[_PatternRow],
        trajectories: list[_TrajectoryRow],
    ) -> str:
        parts: list[str] = []
        if anchor is not None:
            parts.append(anchor.content)

        # Always emit section headers — empty fetches render as headers
        # without items, NOT as missing sections (design §3.7 determinism
        # guarantee). Keeps token boundaries stable across compose calls,
        # which matters for prompt-cache prefix stability.
        parts.append("\n## Personality")
        for t in traits:
            parts.append(f"- {t.guidance_text}")

        parts.append("\n## Views")
        for s in stances:
            parts.append(f"- {s.position}")

        parts.append("\n## Recently observed")
        for p in patterns:
            parts.append(f"- {p.observation}")

        parts.append("\n## Trajectories")
        for traj in trajectories:
            parts.append(f"\n### {traj.subject_name}")
            if traj.older_count and traj.older_first and traj.older_last:
                span = f"{traj.older_first.date().isoformat()} to {traj.older_last.date().isoformat()}"
                parts.append(
                    f"- _Earlier ({span}): {traj.older_count} event{'s' if traj.older_count != 1 else ''} folded._"
                )
            for ev in traj.events_midterm:
                parts.append(f"- ({ev.timestamp.date().isoformat()}) {ev.summary}")
            for ev in traj.events_recent:
                parts.append(f"- ({ev.timestamp.date().isoformat()}) {ev.summary}")

        return "\n".join(parts)

    @staticmethod
    def _budget_warn(rendered: str, budget_chars: int) -> None:
        if len(rendered) > budget_chars:
            logger.warning(
                "PersonaComposer exceeded budget_chars: %d > %d "
                "(per-layer cap may be misconfigured — check cognitive_config)",
                len(rendered),
                budget_chars,
            )


__all__ = ["PersonaComposer"]
