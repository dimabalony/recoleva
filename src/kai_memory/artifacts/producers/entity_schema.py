"""EntitySchemaProducer — design §3.3.1.

Replaces `memory_entities.summary` (variable-length prose) with a
structured ``EntitySchemaPayload`` that the prompt builder can format
deterministically into a cache-friendly block. Wave 2 only ships the
producer; the prompt builder switch happens in Wave 3.

Discovery: §3.3.1 SQL with ``active_schemas`` + ``recent_trace_activity``
CTEs. Trigger conditions:
  (1) entity has at least one active trace AND no active EntitySchema
      ("mentioned but lacks")
  (2) entity has trace activity newer than its current active EntitySchema
      ("new evidence")

Cap: ``cognitive_config.entity_schema_max_per_run`` (default 50).
"""

from __future__ import annotations

from kai_memory.host_calls import get_model_roles

import hashlib
import json
import logging
import re
import uuid as _uuid
from functools import cache
from sqlalchemy import or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.host_calls import (
    call_llm_text,
    get_provider_for_model,
    get_settings,
    start_generation_span,
    utc_now,
)
from kai_memory.identity import AGENT_IDENTITY_DATA_KEY
from kai_memory.storage.models.entity_relationship import EntityRelationship
from kai_memory.storage.models.memory_entity import MemoryEntity
from kai_memory.store import cognitive_config
from kai_memory.artifact_schemas import EntitySchemaPayload, RelationshipRef
from kai_memory.slot_profile import PREDICATE_SLOT_MAP, SlotProfile, aggregate_slot_profile
from kai_memory.slot_summarizer import curate_profile
from kai_memory.artifacts.producer import (
    BaseProducer,
    ProducerSubject,
    compute_content_hash,
)
from kai_memory.artifacts.types import ArtifactType, ProducedArtifact

logger = logging.getLogger("app.services.memory.artifacts.producers.entity_schema")

_TRACE_TRUNCATE_CHARS = 300
_TRACE_EXTRACTION_TASK = "memory.entity_schema.extract_from_traces"
_TRACE_EXTRACTION_PROMPT_TEMPLATE = """\
You are extracting stable canonical_fields for an EntitySchema.

Entity: {entity_name}
Entity type: {entity_type}

Return ONLY a JSON object. NO markdown fences. NO commentary. NO arrays.
Use string keys and string values only. Maximum 10 keys.

Extract STABLE CANONICAL facts likely to still be true in 6 months.
Do NOT extract recent events, one-off incidents, temporary plans, task status,
protocol steps, product comparisons, or transient troubleshooting details.
Skip facts that are unclear or only implied. If nothing qualifies, return {{}}.

Use short standardized snake_case keys.
Suggested keys:
- person: role, relationship, location, hometown, languages, register,
  communication, status, preference, expertise
- animal/pet: species, breed, color, owner, location, companion, feed,
  medication, nickname, weight
- organization: kind, location, role, owner, contact, status
- place: kind, city, country, address, owner, purpose
- tool: category, owner, purpose, platform, status

Observations:
{trace_text}
"""


@cache
def _symmetric_relationship_types() -> frozenset[str]:
    # Lazy import avoids the consolidation -> producers -> entity_schema cycle.
    from kai_memory.consolidation.consolidation import get_symmetric_relationship_predicates

    return get_symmetric_relationship_predicates()


# SQL kept verbatim from design §3.3.1 — see that section for the rationale
# behind the join shape. ``recent_trace_activity`` joins
# ``memory_entity_links`` to ``memory_traces`` on the documented FK
# (``mel.memory_id`` → ``memory_traces.id``).
_DISCOVERY_SQL = text(
    """
    WITH active_schemas AS (
        SELECT subject_entity_id, MAX(created_at) AS latest_schema_at
        FROM memory_artifacts
        WHERE artifact_type = 'EntitySchema' AND status = 'active'
        GROUP BY subject_entity_id
    ),
    recent_trace_activity AS (
        SELECT mel.entity_id, MAX(mt.created_at) AS latest_trace_at,
               (array_agg(mt.id ORDER BY mt.created_at DESC))[1] AS latest_trace_id
        FROM memory_entity_links mel
        JOIN memory_traces mt ON mt.id = mel.memory_id
        WHERE mt.invalidated_at IS NULL
        GROUP BY mel.entity_id
    )
    SELECT me.id AS entity_id,
           rta.latest_trace_at,
           rta.latest_trace_id
      FROM memory_entities me
      JOIN recent_trace_activity rta ON rta.entity_id = me.id
      LEFT JOIN active_schemas as_ ON as_.subject_entity_id = me.id
     WHERE me.merged_into_id IS NULL
       AND (
           as_.subject_entity_id IS NULL
           OR rta.latest_trace_at > as_.latest_schema_at
       )
     ORDER BY rta.latest_trace_at DESC
     LIMIT :cap
    """
)

_TRACE_WINDOW_SQL = text(
    """
    SELECT mt.content
      FROM memory_entity_links mel
      JOIN memory_traces mt ON mt.id = mel.memory_id
     WHERE mel.entity_id = :entity_id
       AND mt.invalidated_at IS NULL
       AND mt.created_at > (CAST(:now AS timestamptz) - (:window_days * INTERVAL '1 day'))
     ORDER BY mt.created_at DESC
     LIMIT :limit
    """
)

# INC 3 §3.4 — row contract for the slot aggregator: per fact-bearing trace
# touching this entity, its facts + created_at + the memory-physics signals
# (Phase A §13). ``salience`` carries significance-gated Ebbinghaus decay +
# access reinforcement (run nightly by ``consolidation._apply_decay``) and is the
# aggregator's rank key; ``significance`` is carried for observability / future
# per-fact decay. Slot facts are pre-attributed (``subject_id`` set by the
# write-path/backfill), so no single-link join is needed.
_SLOT_FACTS_SQL = text(
    """
    SELECT mt.id::text AS trace_id, mt.created_at,
           mt.salience, mt.significance,
           mt.typed_metadata->'facts' AS facts
      FROM memory_entity_links mel
      JOIN memory_traces mt ON mt.id = mel.memory_id
     WHERE mel.entity_id = :entity_id
       AND mt.invalidated_at IS NULL
       AND jsonb_typeof(mt.typed_metadata->'facts') = 'array'
       AND jsonb_array_length(mt.typed_metadata->'facts') > 0
    """
)

# Below this many populated slots, the deterministic profile is too thin to trust —
# the flip (later) falls back to Sonnet. In shadow we only log it.
_MIN_ATTRIBUTED_FIELDS = 3

# Hard size cap on the profile card (design §3.4 — ranked by importance, never
# alphabetical, then capped).
_MAX_PROFILE_SLOTS = 12


class EntitySchemaProducer(BaseProducer):
    """Compose canonical EntitySchema artifacts from existing entity state."""

    name = "EntitySchemaProducer"
    version = 1
    artifact_type: ArtifactType = "EntitySchema"

    async def discover_subjects(self, session: AsyncSession, run_id: _uuid.UUID) -> list[ProducerSubject]:
        cap = int(cognitive_config.get("entity_schema_max_per_run"))
        result = await session.execute(_DISCOVERY_SQL, {"cap": cap})
        rows = result.mappings().all()
        if not rows:
            return []
        logger.debug("EntitySchemaProducer discovered %d subjects (cap=%d)", len(rows), cap)
        return [
            ProducerSubject(
                key=str(row["entity_id"]),
                payload={
                    "entity_id": row["entity_id"],
                    "latest_trace_id": row["latest_trace_id"],
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
        latest_trace_id = subject.payload.get("latest_trace_id")

        entity = await session.get(MemoryEntity, entity_id)
        if entity is None or entity.merged_into_id is not None:
            return None

        # INC 3 cutover flags. The NEW curated card ships LIVE for ALL entity types when
        # `entityschema_live` is on. (The earlier person-only `entityschema_live_for_person`
        # flag was retired once the all-types flag became the live cutover — it was fully
        # subsumed.) `entityschema_deterministic_enabled` still runs the shadow path
        # (NEW-vs-Sonnet agreement diff) for live monitoring.
        shadow_enabled = cognitive_config.get_bool("entityschema_deterministic_enabled")
        use_new_live = cognitive_config.get_bool("entityschema_live")

        # Sonnet's card is the live fallback (and the shadow-diff baseline). Compute it
        # up-front UNLESS we're on the pure-live path (a live flag on with shadow off):
        # there the NEW card is tried first and Sonnet is only used if it's thin/empty, so
        # a live rollout doesn't pay for an (often cold-cache) Sonnet extraction when a
        # rich new card ships. When shadow is on we compute Sonnet anyway — the agreement
        # diff needs it. ``canonical_fields`` is the working live card from here on.
        defer_sonnet = use_new_live and not shadow_enabled
        canonical_fields = {} if defer_sonnet else await self._canonical_fields(session, entity)

        # Pull active outgoing relationships plus inbound canonical rows for
        # symmetric predicates. Directional inbound edges are excluded because
        # ``RelationshipRef`` represents relationships from this entity to a
        # target entity.
        rel_rows = (
            (
                await session.execute(
                    select(EntityRelationship).where(
                        or_(
                            EntityRelationship.from_entity_id == entity.id,
                            (
                                (EntityRelationship.to_entity_id == entity.id)
                                & (EntityRelationship.relationship_type.in_(_symmetric_relationship_types()))
                            ),
                        ),
                        (EntityRelationship.valid_until.is_(None)) | (EntityRelationship.valid_until > utc_now()),
                    )
                )
            )
            .scalars()
            .all()
        )
        relationships = [
            RelationshipRef(
                predicate=rel.relationship_type,
                target_entity_id=rel.to_entity_id if rel.from_entity_id == entity.id else rel.from_entity_id,
                confidence=rel.confidence,
            )
            for rel in rel_rows
        ]

        # INC 3 cutover decision (flags + the deferred-Sonnet choice were resolved above;
        # the new card needs `relationships`, so it's computed here, after the rel query).
        # The new card is shipped LIVE only when `use_new_live` AND it's non-None; a
        # thin/empty/failed card (None) keeps Sonnet — the self-protecting "NEW where it's
        # good, Sonnet where it's thin" fallback. The new card is still computed whenever
        # the shadow flag is on (to log the agreement diff vs Sonnet for ALL types).
        if shadow_enabled or use_new_live:
            new_card = await self._deterministic_card(
                session, entity, canonical_fields, relationships, log_shadow=shadow_enabled
            )
            if use_new_live and new_card:
                canonical_fields = new_card
            elif defer_sonnet:
                # NEW not shipped and Sonnet was deferred → load it now as the live card.
                canonical_fields = await self._canonical_fields(session, entity)

        # last_evidence_trace_id comes from the discovery row (most recent
        # trace touching this entity). Fall back to a fresh query if the
        # discovery payload didn't carry one — defensive against future
        # discovery rewrites that drop the column.
        if latest_trace_id is None:
            latest_trace_id = await self._latest_trace_id(session, entity.id)

        # Default confidence: 1.0 unless the entity has thin evidence (only
        # one or two traces). Producer-internal heuristic — Wave 3 may
        # widen this once we have priming data.
        confidence = await self._confidence_from_evidence(session, entity.id)

        payload_model = EntitySchemaPayload(
            canonical_name=entity.name,
            entity_type=entity.entity_type,
            aliases=list(entity.aliases) if entity.aliases else [],
            canonical_fields=canonical_fields,
            relationships=relationships,
            last_evidence_trace_id=latest_trace_id,
            confidence=confidence,
        )
        payload_dict = payload_model.model_dump(mode="json")

        # Pre-compute content_hash from the *content* subset of the payload,
        # excluding ``last_evidence_trace_id``. That field is provenance —
        # the most-recent trace pointer changes every time a new trace
        # touches the entity, even when nothing canonical (name, aliases,
        # canonical_fields, relationships) actually changed. Without this,
        # ``BaseProducer.supersede()`` never short-circuits to
        # action='noop' on byte-identical canonical content, churning
        # daily supersessions on hot entities (observed: 7 versions in
        # 6 days, payload byte-identical except for the trace pointer).
        # ``confidence`` stabilises at 1.0 once an entity has >=3 traces
        # (per ``_confidence_from_evidence``), so it doesn't churn either.
        stable_payload = {k: v for k, v in payload_dict.items() if k != "last_evidence_trace_id"}
        content_hash = compute_content_hash(stable_payload)

        return ProducedArtifact(
            artifact_type="EntitySchema",
            schema_version=1,
            subject_entity_id=entity.id,
            dimension=None,
            payload=payload_dict,
            content_hash=content_hash,
            human_summary=_compose_human_summary(entity.name, entity.entity_type, len(relationships)),
        )

    # ── helpers ────────────────────────────────────────────

    async def _canonical_fields(self, session: AsyncSession, entity: MemoryEntity) -> dict[str, str]:
        """Return cached canonical fields or extract them from recent linked traces."""
        # §8 deep-merge-corruption guard: entity.data is also the find_or_create
        # deep-merge target, so strip provenance meta (_source/_v) and any
        # non-string junk before treating it as the canonical_fields cache. Only
        # short-circuit when real string fields survive.
        if isinstance(entity.data, dict) and entity.data:
            cached = {k: v for k, v in entity.data.items() if not k.startswith("_") and isinstance(v, str)}
            if cached:
                return cached

        trace_contents = await self._recent_trace_contents(session, entity.id)
        if not trace_contents:
            return {}

        trace_text = "\n".join(f"- {_truncate(content, _TRACE_TRUNCATE_CHARS)}" for content in trace_contents)
        prompt = _TRACE_EXTRACTION_PROMPT_TEMPLATE.format(
            entity_name=entity.name,
            entity_type=entity.entity_type,
            trace_text=trace_text,
        )
        try:
            provider = get_provider_for_model(get_settings(), get_model_roles().artifact)
            span = start_generation_span("consolidation/entity-schema-extract", get_model_roles().artifact)
            raw = await call_llm_text(
                provider,
                prompt,
                max_tokens=1024,
                span=span,
                thinking_mode="disabled",
                task_name=_TRACE_EXTRACTION_TASK,
            )
        except Exception:
            logger.warning(
                "EntitySchemaProducer canonical_fields extraction failed for entity_id=%s",
                entity.id,
                exc_info=True,
            )
            return {}

        fields = _parse_canonical_fields_response(raw, entity.id)
        if fields:
            # §8: source/version-tag the cache entry so the cutover can clear
            # only Sonnet-built caches and the read short-circuit stays
            # source-aware. The tag keys are _-prefixed → stripped on read.
            binding = (
                {AGENT_IDENTITY_DATA_KEY: entity.data[AGENT_IDENTITY_DATA_KEY]}
                if isinstance(entity.data, dict) and AGENT_IDENTITY_DATA_KEY in entity.data
                else {}
            )
            entity.data = {**fields, **binding, "_source": "sonnet", "_v": 1}
        else:
            raw_text = raw or ""
            logger.warning(
                "EntitySchemaProducer extraction returned empty for entity %s (%s); "
                "no canonical_fields populated. sonnet_response_len=%d sonnet_response_sha256=%s",
                entity.id,
                entity.name,
                len(raw_text),
                hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            )
        return fields

    async def _relationship_pairs(
        self, session: AsyncSession, relationships: list[RelationshipRef]
    ) -> list[tuple[str, str]]:
        """(predicate, target_entity_name) pairs for slot-mappable relationships."""
        mapped = [
            (r.predicate, r.target_entity_id)
            for r in relationships
            if (r.predicate or "").strip().lower() in PREDICATE_SLOT_MAP
        ]
        if not mapped:
            return []
        ids = list({tid for _, tid in mapped})
        rows = await session.execute(select(MemoryEntity.id, MemoryEntity.name).where(MemoryEntity.id.in_(ids)))
        names = {row.id: row.name for row in rows}
        return [(pred, names.get(tid) or "") for pred, tid in mapped]

    async def _slot_profile(
        self, session: AsyncSession, entity: MemoryEntity, relationships: list[RelationshipRef]
    ) -> tuple[SlotProfile, bool]:
        """INC 3 §3.4 slot-keyed aggregator (cache-bypassed). SHADOW-only.

        Aggregates the entity's slot-tagged facts (``slot != off_schema``, attributed
        via ``subject_id``) into the clean profile card — SINGLE slots recency-win,
        MULTI accumulate distinct, relationships fold into slots, ranked by importance.

        Returns (profile, would_fallback). ``would_fallback`` is True when fewer than
        ``_MIN_ATTRIBUTED_FIELDS`` slots are populated — the flip (later) would then
        defer to Sonnet rather than ship a thin profile.
        """
        result = await session.execute(_SLOT_FACTS_SQL, {"entity_id": entity.id})
        rows = [
            {
                "trace_id": str(r.trace_id),
                "created_at": r.created_at,
                "facts": r.facts or [],
                "salience": r.salience,  # Phase A §13 — parent-trace rank signal
                "significance": r.significance,  # carried for observability / future per-fact decay
            }
            for r in result
        ]
        rel_pairs = await self._relationship_pairs(session, relationships)
        profile = aggregate_slot_profile(
            rows, str(entity.id), entity.entity_type, rel_pairs=rel_pairs, max_slots=_MAX_PROFILE_SLOTS
        )
        return profile, len(profile.canonical_fields) < _MIN_ATTRIBUTED_FIELDS

    async def _deterministic_card(
        self,
        session: AsyncSession,
        entity: MemoryEntity,
        sonnet_fields: dict[str, str],
        relationships: list[RelationshipRef],
        *,
        log_shadow: bool,
    ) -> dict[str, str] | None:
        """Compute the NEW curated card (aggregate_slot_profile → curate_profile), and —
        when ``log_shadow`` — log an agreement diff vs Sonnet (§7 diagnostic).

        Returns the curated ``canonical_fields`` for the live cutover, or ``None`` when
        the new path is too thin (``would_fallback``), empty, or errors — so the caller
        ships Sonnet and NEVER an empty card. Never raises into the producer (both the
        shadow diagnostic and the live cutover must be failsafe). One Haiku call per
        entity (curate_profile is already input-capped)."""
        try:
            profile, would_fallback = await self._slot_profile(session, entity, relationships)
            # §12 P2-C — capture the off-schema-gate counts from the deterministic
            # profile BEFORE curating (curate_profile rebuilds the profile without them).
            off_schema_durable_count = profile.off_schema_durable_count
            off_schema_retained_count = profile.off_schema_retained_count
            # Whole-entity curation: one cheap Haiku call turns ALL the structured facts
            # into a concise but descriptive card (grouped, minutiae abstracted). The
            # SUBSTRATE (canonical_slot_values) is preserved; any failure keeps the dump.
            profile = await curate_profile(profile, entity_name=entity.name)
            det_fields = profile.canonical_fields
            if log_shadow:
                s_keys, d_keys = set(sonnet_fields), set(det_fields)
                logger.info(
                    "entityschema.shadow_diff %s",
                    json.dumps(
                        {
                            "event_type": "entityschema.shadow_diff",
                            "entity_id": str(entity.id),
                            "sonnet_field_count": len(s_keys),
                            "deterministic_field_count": len(d_keys),
                            "only_sonnet": sorted(s_keys - d_keys),
                            "only_deterministic": sorted(d_keys - s_keys),
                            "value_divergences": sorted(
                                k for k in s_keys & d_keys if sonnet_fields[k] != det_fields[k]
                            ),
                            "slot_value_count": sum(len(v) for v in profile.canonical_slot_values.values()),
                            "off_schema_durable_count": off_schema_durable_count,
                            "off_schema_retained_count": off_schema_retained_count,
                            "would_fallback_to_sonnet": would_fallback,
                        }
                    ),
                )
            # Live-cutover guard: never ship a thin/empty card — defer to Sonnet.
            if would_fallback or not det_fields:
                return None
            return det_fields
        except Exception:
            logger.warning("entityschema deterministic card failed for entity_id=%s", entity.id, exc_info=True)
            return None

    async def _recent_trace_contents(self, session: AsyncSession, entity_id: _uuid.UUID) -> list[str]:
        limit = max(0, int(cognitive_config.get("entity_schema_trace_window_limit") or 20))
        window_days = max(0, int(cognitive_config.get("entity_schema_trace_window_days") or 180))
        if limit == 0 or window_days == 0:
            return []

        result = await session.execute(
            _TRACE_WINDOW_SQL,
            {
                "entity_id": entity_id,
                "limit": limit,
                "window_days": window_days,
                "now": utc_now(),
            },
        )
        contents = [str(content) for content in result.scalars().all() if content]
        return contents[:limit]

    async def _latest_trace_id(self, session: AsyncSession, entity_id: _uuid.UUID) -> _uuid.UUID | None:
        row = (
            await session.execute(
                text(
                    """
                    SELECT mt.id
                      FROM memory_entity_links mel
                      JOIN memory_traces mt ON mt.id = mel.memory_id
                     WHERE mel.entity_id = :entity_id
                       AND mt.invalidated_at IS NULL
                     ORDER BY mt.created_at DESC
                     LIMIT 1
                    """
                ),
                {"entity_id": entity_id},
            )
        ).first()
        return row[0] if row else None

    async def _confidence_from_evidence(self, session: AsyncSession, entity_id: _uuid.UUID) -> float:
        """0.5 with one trace, 0.75 with two, 1.0 with three or more.

        Wave 2 heuristic — Wave 3 may replace once priming data lands.
        Caps at 3 to avoid scanning thin entities' full history.
        """
        row = (
            await session.execute(
                text(
                    """
                    SELECT COUNT(*) AS n
                      FROM (
                          SELECT 1 FROM memory_entity_links mel
                          JOIN memory_traces mt ON mt.id = mel.memory_id
                         WHERE mel.entity_id = :entity_id
                           AND mt.invalidated_at IS NULL
                         LIMIT 3
                      ) AS capped
                    """
                ),
                {"entity_id": entity_id},
            )
        ).first()
        trace_count = int(row[0]) if row else 0
        if trace_count >= 3:
            return 1.0
        if trace_count == 2:
            return 0.75
        return 0.5


def _parse_canonical_fields_response(raw: str | None, entity_id: _uuid.UUID) -> dict[str, str]:
    if raw is None:
        logger.warning(
            "EntitySchemaProducer canonical_fields extraction returned no output for entity_id=%s",
            entity_id,
        )
        return {}

    try:
        parsed = json.loads(_strip_markdown_fences(raw))
    except json.JSONDecodeError:
        logger.warning(
            "EntitySchemaProducer canonical_fields extraction returned malformed JSON for entity_id=%s",
            entity_id,
            exc_info=True,
        )
        return {}

    if not isinstance(parsed, dict):
        logger.warning(
            "EntitySchemaProducer canonical_fields extraction returned non-dict JSON for entity_id=%s",
            entity_id,
        )
        return {}

    if not all(isinstance(key, str) and isinstance(value, str) for key, value in parsed.items()):
        logger.warning(
            "EntitySchemaProducer canonical_fields extraction returned non-string fields for entity_id=%s",
            entity_id,
        )
        return {}

    return {key: value for key, value in parsed.items() if not key.startswith("_")}


def _strip_markdown_fences(raw: str) -> str:
    text_ = raw.strip()
    if text_.startswith("```"):
        text_ = re.sub(r"^```[a-zA-Z]*\n?", "", text_)
        text_ = re.sub(r"\n?```$", "", text_).strip()
    return text_


def _truncate(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    return content[: limit - 3].rstrip() + "..."


def _compose_human_summary(name: str, entity_type: str, rel_count: int) -> str:
    """Short preview line for §3.4.1 run preview (Wave 3)."""
    return f"{name} ({entity_type}, {rel_count} active relationship{'s' if rel_count != 1 else ''})"
