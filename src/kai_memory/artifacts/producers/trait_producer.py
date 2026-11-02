"""TraitProducer — Sub-design D Wave 2 §3.3.

Slow-evolving personality dimensions for the agent (Kai). Produces
``PersonalityTrait`` artifacts keyed on
``(kai_self_entity_id, dimension)``. The partial unique index
``idx_artifacts_one_active`` enforces single-active-per-(subject, dim).

Pipeline shape:

1. Decay first — recompute ``strength`` for every active Trait and
   demote those below ``persona_trait_strength_threshold`` to
   ``status='superseded'`` BEFORE we discover new subjects. Decay
   removes capacity, discovery fills it; doing them in this order
   keeps the cap check below honest.
2. ``discover_subjects`` — pick dimensions with accumulated active
   ``Pattern`` evidence + recent self_reflection traces. Single Sonnet
   call returns a list of (dimension, direction, magnitude,
   guidance_text). Per-run cap = ``persona_max_active_traits`` (default
   12, configurable).
3. ``produce`` — emits a ``PersonalityTraitPayload`` with a content
   hash on the stable subset (subject + dimension + direction +
   magnitude + guidance_text); rehearsal_count reflects accumulated
   Pattern reinforcement_count over the same dimension at production
   time.
4. ``supersede`` — inherits the BaseProducer 5-step transaction. Same
   subject+dimension means an existing Trait is superseded with the new
   payload (drift on direction/magnitude/text triggers a fresh row).

Decay model (design §3.3, locked):

    strength = magnitude
             * exp(-lambda * days_since_last_reinforced)
             * (1 + rehearsal_count * 0.2)

where ``lambda = persona_trait_decay_lambda`` (default 0.0055 ≈ 1/180d).
"""

from __future__ import annotations

from kai_memory.host_calls import get_model_roles

from kai_memory.identity import resolve_agent_identity

import json
import logging
import math
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
from kai_memory.artifact_schemas import PersonalityTraitPayload
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

logger = logging.getLogger("app.services.memory.artifacts.producers.trait_producer")


# Lookback for trait drift discovery — broader than the Pattern window.
_TRAIT_LOOKBACK_DAYS = 30

_TRAIT_PROMPT_TEMPLATE = """\
You are analysing the agent's accumulated patterns and self-reflection
traces from the last {window} day(s) to infer SLOW-EVOLVING personality
traits.

Active patterns (dimension :: observation :: reinforcement_count):
{patterns}

Recent self-reflection traces (chronological):
{traces}

For each dimension where the evidence supports a *durable* trait (not a
one-off observation), return a JSON object:

{{ "dimension": "...",
   "direction": "increasing" | "decreasing" | "stable",
   "magnitude": 0.0-1.0,
   "guidance_text": "<<= 400 chars; how this trait should shape the agent's behaviour>" }}

Rules:
- Only return dimensions with magnitude >= 0.3.
- "guidance_text" is the prose the agent's prompt will render. Address
  the agent in second person ("You ..."). No instructions to the user.
- "dimension" must be snake_case, <= 64 chars.
- If no dimension has durable evidence, return [].
- Output JSON ARRAY only, no commentary, no code fences.
"""


class TraitProducer(BaseProducer):
    """Per-dimension PersonalityTrait artifacts for the agent self-entity."""

    name = "TraitProducer"
    version = 1
    artifact_type: ArtifactType = "PersonalityTrait"

    async def discover_subjects(self, session: AsyncSession, run_id: _uuid.UUID) -> list[ProducerSubject]:
        # Step 1 — decay/demote first so the cap check below sees freed
        # capacity. The decay step is idempotent (same input → same
        # output); running it every L3 pass is the design's intent.
        await self._demote_below_threshold(session)

        agent_self_id = await self._get_kai_self_entity_id(session)
        if agent_self_id is None:
            logger.debug("TraitProducer skipped: no kai-self entity in memory_entities")
            return []

        # Step 1b — staleness gate. Mirrors StanceProducer._expire_stale_stances
        # (added in PR #137). Strength-decay alone takes ~290 days for a
        # magnitude=1.0 trait to fall below threshold at lambda=0.0055; the
        # retention gate cuts the trait at the artifact-lifespan window
        # instead so capacity frees up when Sonnet stops surfacing the
        # dimension. last_reinforced_at is touched on the noop supersede
        # path (see ``supersede`` override below) so stable traits that
        # Sonnet keeps surfacing don't expire from this gate.
        await self._expire_stale_traits(session, agent_self_id)

        # Step 2 — gather signal: active Patterns + recent self_reflection.
        patterns = await self._fetch_active_patterns(session)
        traces = await self._fetch_recent_traces(session)
        if not patterns and len(traces) < 3:
            logger.debug(
                "TraitProducer skipped: %d patterns + %d traces (need patterns OR >=3 traces)",
                len(patterns),
                len(traces),
            )
            return []

        candidates = await self._call_sonnet_trait_inference(patterns, traces)
        if not candidates:
            return []

        # Step 3 — cap. Same-dimension candidates supersede an existing
        # active row without consuming new capacity (UPDATE-then-INSERT
        # leaves the active count unchanged). Fresh dimensions consume
        # capacity. ``capacity_remaining`` measures how many fresh
        # dimensions we can still admit before exceeding
        # ``persona_max_active_traits``.
        cap = int(cognitive_config.get("persona_max_active_traits"))
        active_dims = await self._fetch_active_trait_dimensions(session, agent_self_id)
        capacity_remaining = max(cap - len(active_dims), 0)

        # Index pattern → reinforcement_count for rehearsal_count seeding.
        pattern_count_by_dim: dict[str, int] = {}
        for p in patterns:
            dim = str(p.get("dimension"))
            pattern_count_by_dim[dim] = max(
                pattern_count_by_dim.get(dim, 0),
                int(p.get("reinforcement_count", 1)),
            )

        accepted: list[ProducerSubject] = []
        dropped_dims: list[str] = []
        for c in candidates:
            dim = c["dimension"]
            is_supersede = dim in active_dims
            if not is_supersede and capacity_remaining <= 0:
                # Cap reached AND this would be a fresh dimension. Accumulate
                # the dimension name and log once at end of loop — gives
                # CloudWatch a per-night trail of which fresh dimensions
                # Sonnet wanted to surface but couldn't fit. Mirrors
                # StanceProducer's capacity-drop pattern (PR #137).
                dropped_dims.append(dim)
                continue
            if not is_supersede:
                capacity_remaining -= 1
            accepted.append(
                ProducerSubject(
                    key=f"trait::{agent_self_id}::{dim}",
                    payload={
                        "agent_self_id": agent_self_id,
                        "dimension": dim,
                        "direction": c["direction"],
                        "magnitude": c["magnitude"],
                        "guidance_text": c["guidance_text"],
                        "rehearsal_count_seed": pattern_count_by_dim.get(dim, 0),
                    },
                )
            )

        if dropped_dims:
            # First 5 dimensions are enough to spot a pattern; longer lists
            # are redundant since they all dropped for the same reason.
            logger.info(
                "TraitProducer: %d candidate(s) dropped at cap=%d (active=%d, dimensions=%s)",
                len(dropped_dims),
                cap,
                len(active_dims),
                dropped_dims[:5],
            )
        return accepted

    async def produce(
        self,
        session: AsyncSession,
        subject: ProducerSubject,
        run_id: _uuid.UUID,
    ) -> ProducedArtifact | None:
        agent_self_id: _uuid.UUID = subject.payload["agent_self_id"]
        dimension: str = subject.payload["dimension"]
        direction: str = subject.payload["direction"]
        magnitude: float = subject.payload["magnitude"]
        guidance_text: str = subject.payload["guidance_text"]
        rehearsal_count: int = subject.payload.get("rehearsal_count_seed", 0)

        now = utc_now()
        payload_model = PersonalityTraitPayload(
            dimension=dimension,
            direction=direction,  # type: ignore[arg-type]
            magnitude=magnitude,
            guidance_text=guidance_text,
            driving_observations=[],
            epoch_start=now,
            last_reinforced_at=now,
            rehearsal_count=rehearsal_count,
        )

        # Stable content_hash: subject + dimension + direction + magnitude +
        # guidance_text. Excludes timestamps so re-runs short-circuit to
        # noop when the trait shape didn't drift.
        content_hash = compute_content_hash(
            {
                "subject_entity_id": str(agent_self_id),
                "dimension": dimension,
                "direction": direction,
                "magnitude": round(magnitude, 4),
                "guidance_text": guidance_text,
            }
        )

        return ProducedArtifact(
            artifact_type="PersonalityTrait",
            schema_version=1,
            subject_entity_id=agent_self_id,
            dimension=dimension,
            payload=payload_model.model_dump(mode="json"),
            content_hash=content_hash,
            human_summary=f"trait[{dimension}] {direction} mag={magnitude:.2f}",
        )

    async def supersede(
        self,
        session: AsyncSession,
        prev: Artifact | None,
        new: ProducedArtifact,
        run_id: _uuid.UUID,
    ) -> ChangesMadeEntry:
        """Trait-specific supersede: touch ``last_reinforced_at`` on noop path.

        BaseProducer's hash-match noop returns ``action='noop'`` without
        writing to the row. For Traits that means the retention staleness
        gate in ``_expire_stale_traits`` will eventually expire a perfectly
        stable trait that Sonnet keeps surfacing in the same shape.

        Mirror Stance's PR-#137 fix: on hash match, UPDATE the existing
        row's ``last_reinforced_at`` in JSONB before returning the noop
        ChangesMadeEntry. This is a payload-only mutation (no schema
        change) — ``last_reinforced_at`` is already a field on
        ``PersonalityTraitPayload``.

        Drift cases (hash differs) fall through to ``super().supersede``
        for the standard 5-step transaction, which writes a fresh row with
        ``last_reinforced_at=now`` on its payload.
        """
        producer_name = new.producer_name or self.name
        producer_version = new.producer_version if new.producer_version is not None else self.version

        if prev is None or prev.status == "permanent":
            return await super().supersede(session, prev, new, run_id)

        new_hash = new.content_hash or compute_content_hash(new.payload)
        if prev.content_hash != new_hash:
            return await super().supersede(session, prev, new, run_id)

        # Hash match — noop path, but touch last_reinforced_at to reset the
        # staleness clock. Without this, retention=7 expires every stable
        # trait after 7 days even when Sonnet keeps surfacing it.
        touched_payload = dict(prev.payload or {})
        touched_payload["last_reinforced_at"] = utc_now().isoformat()
        await session.execute(
            update(MemoryArtifact).where(MemoryArtifact.id == prev.id).values(payload=touched_payload)
        )
        logger.debug(
            "TraitProducer.supersede noop: touched last_reinforced_at on %s (hash=%s)",
            prev.id,
            new_hash[:12],
        )
        return ChangesMadeEntry(
            action="noop",
            artifact_id=prev.id,
            prev_artifact_id=prev.id,
            artifact_type=self.artifact_type,
            subject_entity_id=new.subject_entity_id,
            human_summary=new.human_summary,
            producer_name=producer_name,
            producer_version=producer_version,
        )

    # ── decay / demotion ──────────────────────────────────

    async def _expire_stale_traits(
        self,
        session: AsyncSession,
        agent_self_id: _uuid.UUID,
    ) -> int:
        """Expire active Traits that haven't been re-reinforced within retention.

        Pure SQL UPDATE, runs at top of ``discover_subjects`` AFTER
        ``_demote_below_threshold`` (strength gate) so both eviction paths
        clear capacity before the cap check. Mirrors
        ``StanceProducer._expire_stale_stances`` (PR #137) shape: silent
        UPDATE + INFO log, no separate audit emit. The audit trail is the
        row's own ``status`` history.

        Single gate: ``last_reinforced_at < now() - persona_trait_retention_days``
        (default 7d). The strength gate (``_demote_below_threshold``) is
        the OTHER eviction path; this method only handles the staleness
        axis. Touched in lockstep by the ``supersede`` noop override so
        stable traits Sonnet keeps surfacing don't expire.
        """
        retention_days = float(cognitive_config.get("persona_trait_retention_days"))
        cutoff = utc_now() - timedelta(days=retention_days)

        result = await session.execute(
            text(
                """
                UPDATE memory_artifacts
                   SET status = 'superseded'
                 WHERE artifact_type = 'PersonalityTrait'
                   AND status = 'active'
                   AND subject_entity_id = :subject
                   AND COALESCE(
                           (payload->>'last_reinforced_at')::timestamptz,
                           '1970-01-01T00:00:00+00:00'::timestamptz
                       ) < :cutoff
                RETURNING dimension,
                          (payload->>'magnitude')::float AS magnitude,
                          payload->>'last_reinforced_at' AS last_reinforced_at
                """
            ),
            {"subject": agent_self_id, "cutoff": cutoff},
        )
        rows = result.all()
        if rows:
            logger.info(
                "TraitProducer: expired %d stale trait(s) (retention_days=%.0f, cutoff=%s); details=%s",
                len(rows),
                retention_days,
                cutoff.isoformat(),
                [{"dim": r.dimension, "mag": r.magnitude, "last_reinforced_at": r.last_reinforced_at} for r in rows],
            )
        return len(rows)

    async def _demote_below_threshold(self, session: AsyncSession) -> int:
        """Recompute strength on each active Trait; demote below threshold.

        SQL-side computation keeps us off the network. ``last_reinforced_at``
        is a JSONB string in ISO-8601; we cast to timestamptz and EXTRACT
        epoch days against now(). Idempotent — re-running on the same row
        produces the same strength and same outcome.
        """
        threshold = float(cognitive_config.get("persona_trait_strength_threshold"))
        lambda_ = float(cognitive_config.get("persona_trait_decay_lambda"))

        result = await session.execute(
            text(
                """
                UPDATE memory_artifacts
                   SET status = 'superseded'
                 WHERE artifact_type = 'PersonalityTrait'
                   AND status = 'active'
                   AND (
                       (payload->>'magnitude')::float
                       * exp(
                           -CAST(:lambda_ AS double precision)
                           * GREATEST(
                               EXTRACT(EPOCH FROM (
                                   CAST(:now AS timestamptz) - (payload->>'last_reinforced_at')::timestamptz
                               )) / 86400.0,
                               0.0
                           )
                       )
                       * (1 + COALESCE((payload->>'rehearsal_count')::int, 0) * 0.2)
                       < CAST(:threshold AS double precision)
                   )
                """
            ),
            {"lambda_": lambda_, "threshold": threshold, "now": utc_now()},
        )
        rowcount = result.rowcount or 0  # type: ignore[attr-defined]
        if rowcount > 0:
            logger.info("TraitProducer demoted %d traits below strength threshold", rowcount)
        return rowcount

    # ── helpers ───────────────────────────────────────────

    async def _get_kai_self_entity_id(self, session: AsyncSession) -> _uuid.UUID | None:
        entity = await resolve_agent_identity(session)
        return entity.id if entity else None

    async def _fetch_active_trait_dimensions(
        self,
        session: AsyncSession,
        agent_self_id: _uuid.UUID,
    ) -> set[str]:
        result = await session.execute(
            text(
                """
                SELECT dimension FROM memory_artifacts
                 WHERE artifact_type = 'PersonalityTrait'
                   AND status = 'active'
                   AND subject_entity_id = :subject
                """
            ),
            {"subject": agent_self_id},
        )
        return {row[0] for row in result.all() if row[0] is not None}

    async def _fetch_active_patterns(self, session: AsyncSession) -> list[dict[str, Any]]:
        result = await session.execute(
            text(
                """
                SELECT payload->>'dimension' AS dimension,
                       payload->>'observation' AS observation,
                       COALESCE((payload->>'reinforcement_count')::int, 1) AS reinforcement_count
                  FROM memory_artifacts
                 WHERE artifact_type = 'Pattern'
                   AND status = 'active'
                 ORDER BY (payload->>'observed_at')::timestamptz DESC, id ASC
                 LIMIT 50
                """
            )
        )
        rows = result.mappings().all()
        return [dict(r) for r in rows]

    async def _fetch_recent_traces(self, session: AsyncSession) -> list[MemoryTrace]:
        cutoff = utc_now() - timedelta(days=_TRAIT_LOOKBACK_DAYS)
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

    async def _call_sonnet_trait_inference(
        self,
        patterns: list[dict[str, Any]],
        traces: list[MemoryTrace],
    ) -> list[dict[str, Any]]:
        pattern_lines = (
            "\n".join(
                f"- {p['dimension']} :: {_truncate(str(p['observation']), 200)} :: x{p['reinforcement_count']}"
                for p in patterns
            )
            or "(none)"
        )
        trace_lines = (
            "\n".join(f"- [{t.id}] {t.created_at.isoformat()} :: {_truncate(t.content, 240)}" for t in traces)
            or "(none)"
        )
        prompt = _TRAIT_PROMPT_TEMPLATE.format(
            window=_TRAIT_LOOKBACK_DAYS,
            patterns=pattern_lines,
            traces=trace_lines,
        )
        try:
            provider = get_provider_for_model(get_settings(), get_model_roles().artifact)
            span = start_generation_span("consolidation/trait-inference", get_model_roles().artifact)
            # max_tokens=8192: trait inference can return many dimensions
            # with magnitude + guidance_text. Same truncation tail as
            # IdentityRecord/Stance — bumped together since all three used
            # 2048 and showed 0-output on prod.
            # See pattern_producer.py for the thinking_mode="disabled"
            # rationale — same bug class, same evidence base.
            raw = await call_llm_text(
                provider,
                prompt,
                max_tokens=8192,
                span=span,
                thinking_mode="disabled",
                task_name="memory.trait.infer",
            )
        except Exception:
            logger.warning(
                "TraitProducer: Sonnet call failed; skipping",
                exc_info=True,
            )
            return []
        return _parse_trait_response(raw)

    # Public for tests — make decay strength easy to reproduce.
    @staticmethod
    def compute_strength(
        magnitude: float,
        days_since_last_reinforced: float,
        rehearsal_count: int,
        decay_lambda: float | None = None,
    ) -> float:
        lam = float(cognitive_config.get("persona_trait_decay_lambda")) if decay_lambda is None else float(decay_lambda)
        days = max(days_since_last_reinforced, 0.0)
        return magnitude * math.exp(-lam * days) * (1 + max(rehearsal_count, 0) * 0.2)


# ── parsing helpers ───────────────────────────────────────

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_trait_response(raw: str | None) -> list[dict[str, Any]]:
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
        logger.warning("TraitProducer: malformed Sonnet JSON: %s", text_[:200])
        return []
    if not isinstance(parsed, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        dimension = item.get("dimension")
        direction = item.get("direction")
        magnitude = item.get("magnitude")
        guidance_text = item.get("guidance_text")
        if not isinstance(dimension, str) or not dimension.strip():
            continue
        if direction not in ("increasing", "decreasing", "stable"):
            continue
        if not isinstance(guidance_text, str) or not guidance_text.strip():
            continue
        try:
            mag_f = float(magnitude)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if not math.isfinite(mag_f):
            continue
        mag_f = max(0.0, min(1.0, mag_f))
        if mag_f < 0.3:
            continue
        cleaned.append(
            {
                "dimension": dimension.strip()[:64],
                "direction": direction,
                "magnitude": mag_f,
                "guidance_text": guidance_text.strip()[:500],
            }
        )
    return cleaned


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


__all__ = [
    "TraitProducer",
    "_parse_trait_response",
]
