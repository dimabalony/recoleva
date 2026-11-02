"""PatternProducer — Sub-design D Wave 2 §3.5.

Recent observations of agent behaviour across recent traces. Patterns
are the *fast-decaying* layer of the persona pipeline — sliding window
of ``persona_pattern_window_days`` (default 7), capped at
``persona_max_active_patterns`` (default 8).

Pipeline shape (mirrors IdentityRecordProducer):

1. ``discover_subjects`` — scan ``self_reflection`` traces over the
   pattern window, run ONE Sonnet call to extract observation candidates
   tagged with a ``dimension``. Returns one ``ProducerSubject`` per
   candidate.
2. ``produce`` — emits a ``PatternPayload`` with ``reinforcement_count``
   defaulting to 1.
3. ``supersede`` (overridden) — content_hash on ``(dimension, observation)``
   matches an existing active Pattern → increment ``reinforcement_count``
   in place (stays content_hash-stable; we hash on the dimension+observation
   pair so increments don't change the hash). When ``reinforcement_count``
   crosses ``persona_pattern_promotion_threshold``, the producer's
   pipeline-level helper schedules a ``PersonalityTrait`` for the
   dimension via ``_promote_to_trait`` recorded as a separate
   ``ChangesMadeEntry``.
4. Auto-expiry of patterns older than the window happens at the start of
   ``discover_subjects`` so each L3 run sees a clean active set.

Wave 2 keeps the Sonnet prompt single-shot + JSON-array (Identity-style).
No few-shot, no chain-of-thought.
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

from sqlalchemy import insert, select, text, update
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
from kai_memory.artifact_schemas import PatternPayload, PersonalityTraitPayload
from kai_memory.artifacts._dimension_normalizer import (
    canonical_list_for_prompt,
    normalize as normalize_dimension,
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

logger = logging.getLogger("app.services.memory.artifacts.producers.pattern_producer")


# Single-shot extraction prompt. Sonnet returns a JSON array; each item is
# {dimension, observation, driving_trace_ids}. Producer caps post-parse
# (per-run cap = ``persona_max_active_patterns`` from cognitive_config).
_PATTERN_PROMPT_TEMPLATE = """\
You are analysing the agent's recent self-reflection traces.

Traces (chronological, last {window} day(s)):
{trace_summaries}

Identify SHORT recurring observations of the agent's behaviour, each tagged
with a single behavioural dimension.

PREFERRED DIMENSION VOCABULARY (use these names when they fit; coin a new
snake_case name only when none of these capture the behaviour):
{canonical_dimensions}

For each pattern, return a JSON object:
{{ "dimension": "...", "observation": "<<= 200 chars>", "driving_trace_ids": ["..."] }}

Rules:
- "observation" must be a single, factual, present-tense sentence. <= 200 chars.
- "dimension" must be a snake_case label, <= 64 chars.
- Prefer the vocabulary above; emit a NEW snake_case name only if no listed
  name fits the behaviour you're describing.
- driving_trace_ids must reference trace ids from the list above.
- If nothing recurs, return [].
- Output JSON ARRAY only, no commentary, no code fences.
"""


class PatternProducer(BaseProducer):
    """Produces ``Pattern`` artifacts; promotes to ``PersonalityTrait`` in-pipeline.

    Subject discovery is a single Sonnet call over the recent
    self_reflection trace pool. Each returned ``(dimension, observation)``
    becomes one ``ProducerSubject``. ``produce()`` returns a fresh
    ``ProducedArtifact`` with ``reinforcement_count=1``; ``supersede()``
    is overridden so a content_hash hit increments the existing row's
    ``reinforcement_count`` rather than INSERTing a duplicate.
    """

    name = "PatternProducer"
    version = 1
    artifact_type: ArtifactType = "Pattern"

    async def discover_subjects(self, session: AsyncSession, run_id: _uuid.UUID) -> list[ProducerSubject]:
        # Step 1 — auto-expire patterns past the window. Single UPDATE keeps
        # the active set bounded before discovery runs; without this the cap
        # check below would refuse new patterns even when stale ones occupy
        # all the slots.
        await self._expire_stale(session)

        window_days = float(cognitive_config.get("persona_pattern_window_days"))
        cap = int(cognitive_config.get("persona_max_active_patterns"))
        cutoff = utc_now() - timedelta(days=window_days)

        # Step 2 — pull recent self_reflection traces (mirror Identity
        # discovery; thinking-block events surface in self_reflection
        # traces via L2 extraction).
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
        if len(traces) < 2:
            logger.debug(
                "PatternProducer skipped: only %d self_reflection traces in last %.1f days",
                len(traces),
                window_days,
            )
            return []

        candidates = await self._call_sonnet_pattern_extraction(traces, window_days)
        if not candidates:
            return []

        valid_trace_ids = {t.id for t in traces}
        # Cap candidates per design §3.5 + §6 Wave 2 (single-call producer).
        # The cap is on candidates, not on the resulting active rows — the
        # supersede path handles dedup against existing actives separately.
        candidates = candidates[:cap]

        return [
            ProducerSubject(
                key=f"pattern::{c['dimension']}::{c['observation'][:40]}",
                payload={
                    "dimension": c["dimension"],
                    "observation": c["observation"],
                    "driving_trace_ids": c.get("driving_trace_ids", []),
                    "valid_trace_ids": valid_trace_ids,
                },
            )
            for c in candidates
        ]

    async def produce(
        self,
        session: AsyncSession,
        subject: ProducerSubject,
        run_id: _uuid.UUID,
    ) -> ProducedArtifact | None:
        dimension: str = subject.payload["dimension"]
        observation: str = subject.payload["observation"]
        valid_trace_ids: set[_uuid.UUID] = subject.payload["valid_trace_ids"]

        driving: list[_uuid.UUID] = []
        for tid in subject.payload.get("driving_trace_ids", []):
            if not isinstance(tid, str):
                continue
            try:
                parsed = _uuid.UUID(tid)
            except (TypeError, ValueError):
                continue
            if parsed in valid_trace_ids:
                driving.append(parsed)

        observed_at = utc_now()
        payload_model = PatternPayload(
            dimension=dimension,
            observation=observation,
            driving_traces=driving,
            observed_at=observed_at,
            reinforcement_count=1,
        )

        # Pre-compute the content_hash on the (dimension, observation)
        # pair only — reinforcement_count and observed_at MUST NOT be
        # part of the hash, otherwise every new run would create a new
        # row instead of reinforcing.
        stable_hash_input = {"dimension": dimension, "observation": observation}
        content_hash = compute_content_hash(stable_hash_input)

        return ProducedArtifact(
            artifact_type="Pattern",
            schema_version=1,
            subject_entity_id=None,  # Patterns aren't subject-keyed; uniqueness is via content_hash
            dimension=dimension,
            payload=payload_model.model_dump(mode="json"),
            content_hash=content_hash,
            human_summary=f"pattern[{dimension}] {_truncate(observation, 80)}",
        )

    async def supersede(
        self,
        session: AsyncSession,
        prev: Artifact | None,
        new: ProducedArtifact,
        run_id: _uuid.UUID,
    ) -> ChangesMadeEntry:
        """Pattern-specific supersede: reinforce existing or insert new.

        Patterns aren't subject-keyed, so the BaseProducer's
        ``_fetch_active`` helper (which queries on subject+dimension) is
        not the right resolver. We look up by content_hash instead. If a
        match is found in the active set, increment its
        ``reinforcement_count`` in place (no new row, no supersession).
        Otherwise INSERT a fresh row with reinforcement_count=1.

        Promotion to PersonalityTrait happens in
        ``_maybe_promote_to_trait`` — emits a separate
        ``ChangesMadeEntry`` recorded by the pipeline via the
        ``post_actions`` hook. Wave 2 records the promotion as a side
        effect of this producer's run.
        """
        del prev  # We resolve our own prev via content_hash.

        existing = await self._fetch_existing_by_hash(session, new.content_hash)
        producer_name = new.producer_name or self.name
        producer_version = new.producer_version if new.producer_version is not None else self.version

        if existing is not None:
            # Reinforce in place. Increment reinforcement_count + bump
            # observed_at so the decay clock reflects the latest evidence.
            new_count = int(existing.payload.get("reinforcement_count", 1)) + 1
            updated_payload = dict(existing.payload)
            updated_payload["reinforcement_count"] = new_count
            updated_payload["observed_at"] = utc_now().isoformat()
            await session.execute(
                update(MemoryArtifact).where(MemoryArtifact.id == existing.id).values(payload=updated_payload)
            )

            # Promotion check — synchronous within the producer's
            # savepoint so the audit trail attributes both rows to the
            # same run (design §3.6 + §6 Wave 2).
            promotion_count = await self._maybe_promote_to_trait(
                session,
                pattern_id=existing.id,
                dimension=str(existing.payload["dimension"]),
                observation=str(existing.payload["observation"]),
                reinforcement_count=new_count,
                run_id=run_id,
            )
            if promotion_count > 0:
                # Surface the promotion in the human_summary so the
                # ChangesMadeEntry preview makes the cross-row impact
                # visible in changes_made.
                summary = (
                    f"reinforced pattern[{existing.payload['dimension']}] "
                    f"-> count={new_count} (+1 PersonalityTrait promotion)"
                )
            else:
                summary = f"reinforced pattern[{existing.payload['dimension']}] -> count={new_count}"
            return ChangesMadeEntry(
                action="supersede",  # in-place mutation of an active row
                artifact_id=existing.id,
                prev_artifact_id=existing.id,
                artifact_type=self.artifact_type,
                subject_entity_id=None,
                human_summary=summary,
                producer_name=producer_name,
                producer_version=producer_version,
            )

        # No matching active pattern → fresh INSERT. Mirrors the
        # BaseProducer.supersede 'create' branch but skips the UPDATE-prev
        # step (no prev for content_hash misses).
        new_id = _uuid.uuid4()
        await session.execute(
            insert(MemoryArtifact).values(
                id=new_id,
                artifact_type=self.artifact_type,
                schema_version=new.schema_version,
                producer_name=producer_name,
                producer_version=producer_version,
                subject_entity_id=None,
                dimension=new.dimension,
                payload=new.payload,
                content_hash=new.content_hash,
                produced_by_run_id=run_id,
                status="active",
                supersedes_id=None,
                expires_at=new.expires_at,
            )
        )

        return ChangesMadeEntry(
            action="create",
            artifact_id=new_id,
            prev_artifact_id=None,
            artifact_type=self.artifact_type,
            subject_entity_id=None,
            human_summary=new.human_summary,
            producer_name=producer_name,
            producer_version=producer_version,
        )

    async def _fetch_active(
        self,
        session: AsyncSession,
        subject_entity_id: Any,
        dimension: str | None = None,
    ) -> Artifact | None:
        """Override: PatternProducer dedup keys on content_hash, not (subject, dim).

        The pipeline's ``_iterate_subjects`` calls this to materialise
        ``prev`` for ``supersede``. Returning ``None`` here forces our
        overridden ``supersede`` to do its own content_hash lookup —
        keeping the dedup logic in one place. The base class's
        (subject_entity_id, dimension) lookup would be wrong for
        Patterns: subject is always NULL and multiple patterns can
        share a dimension if they describe different observations.
        """
        del session, subject_entity_id, dimension
        return None

    # ── helpers ───────────────────────────────────────────

    async def _expire_stale(self, session: AsyncSession) -> int:
        """Mark patterns past the window as expired. Returns row count."""
        window_days = float(cognitive_config.get("persona_pattern_window_days"))
        result = await session.execute(
            text(
                """
                UPDATE memory_artifacts
                   SET status = 'expired'
                 WHERE artifact_type = 'Pattern'
                   AND status = 'active'
                   AND (payload->>'observed_at')::timestamptz
                       < CAST(:now AS timestamptz) - (:window_days * INTERVAL '1 day')
                """
            ),
            {"window_days": window_days, "now": utc_now()},
        )
        rowcount = result.rowcount or 0  # type: ignore[attr-defined]
        if rowcount > 0:
            logger.debug("PatternProducer expired %d stale patterns", rowcount)
        return rowcount

    async def _fetch_existing_by_hash(
        self,
        session: AsyncSession,
        content_hash: str | None,
    ) -> Artifact | None:
        if not content_hash:
            return None
        row = (
            (
                await session.execute(
                    text(
                        """
                        SELECT id, artifact_type, schema_version, producer_name,
                               producer_version, subject_entity_id, payload,
                               content_hash, status, supersedes_id, created_at, expires_at
                          FROM memory_artifacts
                         WHERE artifact_type = 'Pattern'
                           AND status = 'active'
                           AND content_hash = :content_hash
                         LIMIT 1
                        """
                    ),
                    {"content_hash": content_hash},
                )
            )
            .mappings()
            .first()
        )
        if row is None:
            return None
        return Artifact(
            id=row["id"],
            artifact_type=row["artifact_type"],
            schema_version=row["schema_version"],
            producer_name=row["producer_name"],
            producer_version=row["producer_version"],
            subject_entity_id=row["subject_entity_id"],
            payload=row["payload"],
            content_hash=row["content_hash"],
            status=row["status"],
            supersedes_id=row["supersedes_id"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
        )

    async def _maybe_promote_to_trait(
        self,
        session: AsyncSession,
        *,
        pattern_id: _uuid.UUID,
        dimension: str,
        observation: str,
        reinforcement_count: int,
        run_id: _uuid.UUID,
    ) -> int:
        """Promote pattern → PersonalityTrait when reinforcement crosses threshold.

        Returns the number of Trait artifacts created (0 or 1). Skips
        cleanly if a Trait for this dimension already exists active —
        that's TraitProducer's job to supersede when drift accumulates.
        """
        threshold = int(cognitive_config.get("persona_pattern_promotion_threshold"))
        if reinforcement_count < threshold:
            return 0

        agent_self_id = await self._get_kai_self_entity_id(session)
        if agent_self_id is None:
            logger.debug(
                "PatternProducer: skipping promotion (no kai-self entity); pattern=%s",
                pattern_id,
            )
            return 0

        # Skip if an active PersonalityTrait for this dimension exists —
        # TraitProducer is the canonical writer for trait drift; we don't
        # want PatternProducer racing it on supersession.
        existing = (
            await session.execute(
                text(
                    """
                    SELECT 1 FROM memory_artifacts
                     WHERE artifact_type = 'PersonalityTrait'
                       AND status = 'active'
                       AND subject_entity_id = :subject
                       AND dimension = :dimension
                     LIMIT 1
                    """
                ),
                {"subject": agent_self_id, "dimension": dimension},
            )
        ).first()
        if existing is not None:
            return 0

        # Bootstrap a minimal Trait. TraitProducer will refine + supersede
        # when accumulated evidence justifies a different magnitude /
        # direction; this row exists so the composer has SOMETHING to
        # surface immediately after promotion fires.
        now = utc_now()
        trait_payload = PersonalityTraitPayload(
            dimension=dimension,
            direction="stable",
            magnitude=0.4,  # mid-band; TraitProducer will recompute
            guidance_text=_truncate(observation, 500),
            driving_observations=[],
            epoch_start=now,
            last_reinforced_at=now,
            rehearsal_count=reinforcement_count,
        )
        trait_dump = trait_payload.model_dump(mode="json")
        # Trait content_hash uses the stable subset so re-running the same
        # promotion is idempotent and matches what TraitProducer would
        # later produce for content_hash dedup.
        trait_hash = compute_content_hash(
            {
                "subject_entity_id": str(agent_self_id),
                "dimension": dimension,
                "direction": "stable",
                "magnitude": 0.4,
                "guidance_text": trait_payload.guidance_text,
            }
        )
        new_trait_id = _uuid.uuid4()
        await session.execute(
            insert(MemoryArtifact).values(
                id=new_trait_id,
                artifact_type="PersonalityTrait",
                schema_version=1,
                producer_name=self.name,
                producer_version=self.version,
                subject_entity_id=agent_self_id,
                dimension=dimension,
                payload=trait_dump,
                content_hash=trait_hash,
                produced_by_run_id=run_id,
                status="active",
                supersedes_id=None,
                expires_at=None,
            )
        )
        logger.info(
            "PatternProducer promoted pattern=%s -> PersonalityTrait=%s (dimension=%s, count=%d)",
            pattern_id,
            new_trait_id,
            dimension,
            reinforcement_count,
        )
        return 1

    async def _get_kai_self_entity_id(self, session: AsyncSession) -> _uuid.UUID | None:
        entity = await resolve_agent_identity(session)
        return entity.id if entity else None

    async def _call_sonnet_pattern_extraction(
        self,
        traces: list[MemoryTrace],
        window_days: float,
    ) -> list[dict[str, Any]]:
        trace_summaries = "\n".join(
            f"- [{t.id}] {t.created_at.isoformat()} :: {_truncate(t.content, 320)}" for t in traces
        )
        prompt = _PATTERN_PROMPT_TEMPLATE.format(
            window=int(window_days),
            trace_summaries=trace_summaries,
            canonical_dimensions=canonical_list_for_prompt(),
        )
        try:
            provider = get_provider_for_model(get_settings(), get_model_roles().artifact)
            span = start_generation_span("consolidation/pattern-extract", get_model_roles().artifact)
            # max_tokens=8192: PatternProducer returns a JSON array of
            # (dimension, observation, driving_trace_ids) tuples — one per
            # candidate. The 2048 ceiling truncated EVERY production run and
            # silently committed 0 artifacts (parser saw incomplete JSON,
            # returned []). 8192 gives 4x headroom; Sonnet's per-call cost
            # at this ceiling stays well under budget for nightly L3.
            # thinking_mode="disabled": adaptive thinking consumes the
            # max_tokens budget non-deterministically — verified 2026-05-13
            # via scripts/verify_disable_thinking.py (33% truncation rate
            # on this very call). For structured JSON extraction the
            # internal reasoning happens regardless; the visible thinking
            # blocks add zero quality (NO-thinking actually produced MORE
            # patterns: 15 vs 12 on average).
            raw = await call_llm_text(
                provider,
                prompt,
                max_tokens=8192,
                span=span,
                thinking_mode="disabled",
                task_name="memory.pattern.extract",
            )
        except Exception:
            logger.warning(
                "PatternProducer: Sonnet call failed; skipping",
                exc_info=True,
            )
            return []
        return _parse_pattern_response(raw)


# ── parsing helpers ───────────────────────────────────────


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_pattern_response(raw: str | None) -> list[dict[str, Any]]:
    """Tolerant parse: strip code fences, locate outermost JSON array.

    Identical shape to ``_parse_drift_response`` in identity_record.py
    but tuned for the Pattern schema (dimension, observation,
    driving_trace_ids).
    """
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
        logger.warning("PatternProducer: malformed Sonnet JSON: %s", text_[:200])
        return []
    if not isinstance(parsed, list):
        return []
    # Dict keyed by (canonical_dimension, observation) so two raw entries
    # whose dimensions normalize to the same canonical AND share an
    # observation merge into one — driving_trace_ids unioned. Without this
    # dedup, identical artifacts collide on supersede()'s content_hash
    # lookup and the second entry becomes a wasted reinforce. Preserve
    # first-occurrence order via dict insertion order.
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        dimension = item.get("dimension")
        observation = item.get("observation")
        if not isinstance(dimension, str) or not isinstance(observation, str):
            continue
        if not dimension.strip() or not observation.strip():
            continue
        # Schema cap mirrors PatternPayload (250 chars). Trim defensively
        # so the validator doesn't reject otherwise-valid candidates.
        # Dimension passes through the soft-whitelist normalizer — known
        # synonyms collapse to a canonical name, unknown names pass
        # through untouched (Sonnet keeps the freedom to coin new ones).
        normalized = normalize_dimension(dimension.strip()[:64])
        if not normalized:
            continue
        observation_trimmed = observation.strip()[:250]
        # Defensive shape coercion: Sonnet occasionally emits
        # driving_trace_ids as a single string instead of a list.
        # Iterating a string yields per-character "trace ids" that
        # all pass isinstance(tid, str), polluting the union. Normalize
        # any non-list value to an empty list before filtering.
        raw_driving = item.get("driving_trace_ids", [])
        if not isinstance(raw_driving, list):
            raw_driving = []
        trace_ids = [tid for tid in raw_driving if isinstance(tid, str)]
        key = (normalized, observation_trimmed)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = {
                "dimension": normalized,
                "observation": observation_trimmed,
                "driving_trace_ids": list(trace_ids),
            }
        else:
            # Union driving_trace_ids while preserving order + uniqueness.
            seen = set(existing["driving_trace_ids"])
            for tid in trace_ids:
                if tid not in seen:
                    existing["driving_trace_ids"].append(tid)
                    seen.add(tid)
    return list(by_key.values())


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


__all__ = [
    "PatternProducer",
    "_parse_pattern_response",
]
