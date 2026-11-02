"""Cognitive memory service — store, search, entity operations, invalidation."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid as _uuid
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from sqlalchemy import ARRAY, String, bindparam, func, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.host_calls import scrub_secrets, utc_now, validate_skill_content
from kai_memory.identity import AGENT_IDENTITY_DATA_KEY
from kai_memory.ownership import owned_state
from kai_memory.storage.models.memory_entity import MemoryEntity
from kai_memory.storage.models.memory_entity_link import MemoryEntityLink
from kai_memory.storage.models.memory_trace import MemoryTrace
from kai_memory.storage.models.retrieval_config import RetrievalConfig
from kai_memory.store import cognitive_config
from kai_memory.store.cognitive_memory_constants import (
    SERVABLE_MEMORY_TYPES,
    DEFAULT_RETRIEVAL_WEIGHTS,
    EntityType,
    LinkRole,
    MemoryType,
    SourceType,
)
from kai_memory.store.embeddings import generate_embedding, is_zero_vector
from kai_memory.canonicalization import translit_key
from kai_memory.romanization import candidate_aliases

# Pinned to the pre-DB-842 module path: CloudWatch queries, feature-watch criteria and
# log filters match the logger name, which must not follow the module's move.
_LEGACY_LOGGER_NAME = "app.services.cognitive_memory"
logger = logging.getLogger(_LEGACY_LOGGER_NAME)

MERGE_CHAIN_MAX_DEPTH = 10

# ── HNSW search tuning ─────────────────────────────────────
# `idx_mt_embedding` is HNSW (migration 039, `vector_cosine_ops`, default build
# params m=16 / ef_construction=64). An HNSW index scan yields at most
# `hnsw.ef_search` candidates (pgvector default 40) BEFORE the WHERE filters
# run; our largest search LIMIT is ~40 (priming asks limit*2) and the
# post-filters (invalidated_at, servable-type allowlist, merged-entity
# anti-join) discard some candidates, so the default leaves no recall
# headroom. 100 buys >2x headroom over the worst-case LIMIT at sub-millisecond
# extra cost on a ~10k-vector index — versus the seq-scan brute force this
# replaces, which burned the entire 4s L1 semantic-search budget.
HNSW_EF_SEARCH = 100

# ── Pre-106 alias-lookup fallback ──────────────────────────
# ``aliases_lower(jsonb)`` and its GIN index ship in migration 106, so code can
# meet a database that has not run it yet: the CI priming benchmark runs a PR's
# code against the shared database over an SSM tunnel without migrating it, and
# any deploy window where the app starts before ``alembic upgrade`` finishes
# looks the same. Probing keeps the hot path exception-free — without it every
# alias lookup would raise UndefinedFunction and silently lose the alias half of
# priming. Delete this, ``_ALIASES_LOWER_PROBE`` and the fallback branch in
# ``find_entities_by_lowercase_tokens`` once 106 is live everywhere.
_ALIASES_LOWER_PROBE = text("SELECT to_regprocedure('aliases_lower(jsonb)') IS NOT NULL")


class _AliasesLowerProbe:
    """What one database has shown about the helper.

    ``available`` is True once the helper has been SEEN. Only a positive probe is
    cached: the helper can appear under a running process (that is exactly what the
    deploy window is), and a cached miss would pin that process to the unindexed
    fallback until restart. So a miss re-probes on the next alias lookup — one
    extra ``to_regprocedure`` read, trivial beside the sequential scan the
    fallback itself costs, and it stops the moment 106 lands. ``warned``: the
    fallback warns once, not once per lookup.
    """

    def __init__(self) -> None:
        self.available = False
        self.warned = False


#: Kai's database. A memory client probes its own store (``ownership``, DB-854).
_ALIASES_LOWER_STATE = _AliasesLowerProbe()


def reset_aliases_lower_probe() -> None:
    """Forget the cached ``aliases_lower(jsonb)`` probe state. Tests only."""
    _ALIASES_LOWER_STATE.available = False
    _ALIASES_LOWER_STATE.warned = False


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Deep-merge `overlay` into a copy of `base` (v2a §5.4).

    Nested dict values are merged recursively. Lists are replaced wholesale
    (not concatenated) to avoid unbounded growth. Overlay keys win on
    conflicts; base is never mutated.
    """
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _scrub_typed_metadata(blob: Any) -> Any:
    """Recursively scrub secrets from every string *value* in a typed_metadata
    blob before persistence (Sub-design C §1.1, CR 3356167732).

    ``store_trace`` already scrubs ``content``; the cold extraction blob carries
    Haiku-extracted user data (and occasional credential-term values) and must
    get the same treatment. Dict keys are reserved schema slots — left intact;
    only values are recursed/scrubbed. Non-strings pass through unchanged.
    """
    if isinstance(blob, str):
        return scrub_secrets(blob)
    if isinstance(blob, dict):
        return {k: _scrub_typed_metadata(v) for k, v in blob.items()}
    if isinstance(blob, list):
        return [_scrub_typed_metadata(item) for item in blob]
    return blob


# Durable identity facts get a salience FLOOR (not significance=5, which would make a
# wrong fact immortal) so decay can't prune them past the 0.01 line. The DECAY-pass
# floor (consolidation.py) is the load-bearing mechanism; this initial floor helps
# low-significance durable facts start above the decay floor. See the root-cause doc.
DURABLE_INITIAL_SALIENCE = 0.35


def _has_durable_fact(typed_metadata: dict | None) -> bool:
    """True if this trace carries at least one durable identity fact — either the
    trace-level ``durability`` marker (stamped at store time) or any per-fact
    ``durability == "durable"`` tag that L2 extraction emits into ``facts``."""
    if not typed_metadata:
        return False
    if typed_metadata.get("durability") == "durable":
        return True
    facts = typed_metadata.get("facts")
    return isinstance(facts, list) and any(isinstance(f, dict) and f.get("durability") == "durable" for f in facts)


class CognitiveMemoryService:
    """Core service for the cognitive memory system.

    Provides store, search, entity operations, access tracking,
    and trace invalidation (contradiction handling).
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ── Store ──────────────────────────────────────────────

    async def store_trace(
        self,
        content: str,
        memory_type: MemoryType,
        significance: int = 3,
        source_type: SourceType = "conversation",
        source_chat_id: int | None = None,
        source_topic_id: int | None = None,
        emotion_valence: float | None = None,
        emotion_arousal: float | None = None,
        embedding: list[float] | None = None,
        typed_metadata: dict | None = None,
        source_path: str | None = None,
    ) -> MemoryTrace:
        """Store a new memory trace with initial salience derived from significance.

        When ``embedding`` is provided, it is used directly and the OpenAI call
        is skipped (useful when the caller batch-embeds beforehand).

        Sub-design C INC 1: ``typed_metadata`` (the cold extraction blob) and
        ``source_path`` (trace origin) are written only when
        ``typed_metadata_write_enabled`` is on; off → the columns keep their
        defaults ({} / NULL), byte-identical to pre-087. ``source_user_id`` is
        deliberately NOT a parameter here — its population is deferred to E.
        """
        content = scrub_secrets(content)
        violations = validate_skill_content(content)
        if violations:
            logger.warning(
                "Memory content violations — downgrading significance to 1: %s",
                violations,
            )
            significance = 1

        # Durable identity facts (citizenship, employer, DOB, …) must survive the
        # Ebbinghaus decay that otherwise prunes a significance-3 trace past the 0.01
        # floor at ~68 days (write-path fact-loss root cause — see
        # backend-docs/ai-workforce/research/2026-07-19-write-path-fact-loss-rootcause.md). Give them an
        # initial-salience FLOOR. We deliberately do NOT bump `significance` — that would
        # make a WRONG fact immortal and pollute the significance scale. Retention is by
        # salience floor; invalidation (invalidated_at) stays the escape hatch.
        #
        # Codex #6: gate the initial floor behind the write flag TOO. With the flag OFF no
        # trace-level marker persists, so the nightly decay floor never fires — bumping
        # only the initial salience would leave a flag-off half-state (a sig-1 durable
        # starting at 0.35 vs the pre-PR 0.2). Floor + marker now switch together, so
        # flag-off is byte-identical to pre-PR behavior.
        durable = _has_durable_fact(typed_metadata)
        write_enabled = cognitive_config.get_bool("typed_metadata_write_enabled")
        initial_sal = significance / 5.0
        if durable and write_enabled:
            initial_sal = max(initial_sal, DURABLE_INITIAL_SALIENCE)
        trace = MemoryTrace(
            content=content,
            memory_type=memory_type,
            significance=significance,
            salience=initial_sal,
            initial_salience=initial_sal,
            source_type=source_type,
            source_chat_id=source_chat_id,
            source_topic_id=source_topic_id,
            emotion_valence=emotion_valence,
            emotion_arousal=emotion_arousal,
        )
        if write_enabled:
            if typed_metadata is not None:
                # Scrub secrets from the blob's string values, mirroring the
                # content scrub above — the extraction payload can carry
                # credential-term values that must not persist verbatim.
                blob = _scrub_typed_metadata(typed_metadata)
                if durable:
                    # Stamp a trace-level marker (derived from the per-fact durability
                    # tags) so the nightly decay/prune SQL can cheaply read
                    # `typed_metadata->>'durability'` — the persisted signal is what
                    # makes the DECAY-pass floor (the load-bearing mechanism) work.
                    blob["durability"] = "durable"
                trace.typed_metadata = blob
            if source_path is not None:
                trace.source_path = source_path
        self.session.add(trace)
        await self.session.flush()

        # Last-line guard: never persist a zero-norm embedding. A zero vector
        # NaN-sorts to the top of every pgvector search and poisons retrieval
        # corpus-wide. NULL is filtered out cleanly and can be backfilled.
        if embedding is not None and is_zero_vector(embedding):
            logger.warning("Refusing zero-norm embedding for trace %s — storing NULL for backfill", trace.id)
            embedding = None

        if embedding is not None:
            trace.embedding = embedding
            await self.session.flush()
        else:
            try:
                computed = await generate_embedding(content)
                trace.embedding = computed
                await self.session.flush()
            except Exception:
                # Embedding API down (EmbeddingError) or any other failure —
                # keep the trace with a NULL embedding (content is preserved;
                # the nightly re-embed backfills it). Never store a zero vector.
                logger.warning("Embedding unavailable for trace %s — stored with NULL embedding", trace.id)

        return trace

    async def store_procedural_candidate(
        self,
        content: str,
        *,
        salience: float,
        source_chat_id: int | None,
        source_topic_id: int | None,
        source_user_id: int | None,
        typed_metadata: dict,
    ) -> MemoryTrace:
        """Store a Store-B procedural CANDIDATE (procedural-memory Phase A).

        A dedicated writer (vs ``store_trace``) so the procedural recorder can set
        the exact fields Phase A needs — explicit low ``salience``, ``source_user_id``
        for multi-person scoping, and ``typed_metadata`` UNCONDITIONALLY (status /
        provenance / validity) — while keeping ``MemoryTrace`` construction INSIDE
        the memory-service boundary (test_memory_interface_discipline). No embedding:
        procedural candidates are held out of every L1/L2/L3 read.
        """
        trace = MemoryTrace(
            content=content,
            memory_type="procedural",
            salience=salience,
            initial_salience=salience,
            source_type="conversation",
            source_chat_id=source_chat_id,
            source_topic_id=source_topic_id,
            source_user_id=source_user_id,
            typed_metadata=typed_metadata,
        )
        self.session.add(trace)
        await self.session.flush()
        return trace

    async def store_procedural_rule(
        self,
        content: str,
        *,
        embedding: list[float] | None,
        source_user_id: int | None,
        typed_metadata: dict,
        source_chat_id: int | None = None,
        source_topic_id: int | None = None,
    ) -> MemoryTrace:
        """Store a Phase-C procedural RULE/HABIT (memory_type='procedural', kind
        rule|habit — design §1.1).

        Mirrors ``store_procedural_candidate`` but sets an ``embedding`` (rules are
        served by a dedicated cosine path, §3.1 — candidates stay NULL-embedded)
        and NORMAL significance (no ``significance=5`` override — prune's type
        exclusion already keeps procedural rows, and unread salience decay is
        harmless; App. A C-10). Keeps ``MemoryTrace`` construction inside the
        memory-service boundary (test_memory_interface_discipline).

        ``content`` is scrubbed at rest like ``store_trace``: a rule is embedded
        and served back into the system prompt (dedicated cosine path, §3.1), so a
        secret captured verbatim in a "teach" instruction must never resurface.
        """
        content = scrub_secrets(content)
        trace = MemoryTrace(
            content=content,
            memory_type="procedural",
            source_type="conversation",
            source_chat_id=source_chat_id,
            source_topic_id=source_topic_id,
            source_user_id=source_user_id,
            embedding=embedding,
            typed_metadata=typed_metadata,
        )
        self.session.add(trace)
        await self.session.flush()
        return trace

    # ── Search ─────────────────────────────────────────────

    @staticmethod
    def _no_merged_entity_links_filter():
        return ~(
            select(MemoryEntityLink.id)
            .join(MemoryEntity, MemoryEntity.id == MemoryEntityLink.entity_id)
            .where(
                MemoryEntityLink.memory_id == MemoryTrace.id,
                MemoryEntity.merged_into_id.isnot(None),
            )
            .correlate(MemoryTrace)
            .exists()
        )

    @staticmethod
    def _semantic_search_stmt(
        query_embedding: list[float],
        limit: int,
        memory_types: list[str] | None = None,
    ):
        """Shared semantic-search SELECT over memory_traces.

        ORDER BY must stay the BARE ``embedding <=> :vec`` operator (ascending
        distance): pgvector can only serve an index scan when the ORDER BY is
        exactly the indexed distance expression. The old wrapped form
        (``1 - distance AS similarity … ORDER BY similarity DESC``) made
        ``idx_mt_embedding`` ineligible, so every search seq-scanned ~10.9k
        1536-dim vectors (prod 08-26: lifetime idx_scan=0, 43% of L1 semantic
        searches blew the 4s timeout and failed open to empty priming).

        ``ORDER BY distance ASC`` ≡ ``ORDER BY (1 - distance) DESC`` — result
        order is mathematically identical; ``similarity`` stays in the SELECT
        list for downstream scoring. Guarded by
        tests/test_pgvector_search_ordering.py (compiled SQL) and
        tests/integration/test_pgvector_index_plan.py (EXPLAIN on real PG).
        """
        distance = MemoryTrace.embedding.cosine_distance(query_embedding)
        similarity_expr = (1 - distance).label("similarity")
        stmt = (
            select(MemoryTrace, similarity_expr)
            .where(
                MemoryTrace.invalidated_at.is_(None),
                MemoryTrace.embedding.isnot(None),
                MemoryTrace.memory_type.in_(SERVABLE_MEMORY_TYPES),  # allowlist (see constant)
                CognitiveMemoryService._no_merged_entity_links_filter(),
            )
            .order_by(distance)
            .limit(limit)
        )
        if memory_types:
            stmt = stmt.where(MemoryTrace.memory_type.in_(memory_types))
        return stmt

    async def _pin_semantic_search_plan(self) -> None:
        """Pin the semantic-search plan for the statement that follows.

        ``set_config(…, is_local := true)`` ≡ ``SET LOCAL`` — all three GUCs die
        at commit/rollback; one statement = one round trip.

        - ``hnsw.ef_search = 100``: candidate-budget headroom (see
          ``HNSW_EF_SEARCH``).
        - ``enable_seqscan/enable_sort = off``: index eligibility is NOT
          sufficient on prod — the db.t4g.micro cost model PREFERS the 5.26s
          seq scan (and, with only seqscan penalized, a 5.33s Bitmap+Sort via
          ``idx_mt_type``) over the ~ms ``idx_mt_embedding`` scan (EXPLAIN
          ANALYZE on prod, 2026-08-27: fully pinned → Index Scan, 640ms cold,
          ms warm). These are cost PENALTIES, not prohibitions, so a plan that
          has no alternative still executes. Safe here because the pinned
          window contains exactly one statement — the top-k search whose ORDER
          BY is precisely the expression the index serves — so forbidding
          seq+sort cannot produce a wrong plan, only the right one.

        CAVEAT: ``_restore_planner_defaults`` reopens seq scans right after
        the search; if any statement is ever added BETWEEN pin and restore
        (or the search's ORDER BY stops being index-eligible), these toggles
        must be revisited — they would distort that statement's plan too.
        """
        await self.session.execute(
            text(
                "SELECT set_config('hnsw.ef_search', :ef, true), "
                "set_config('enable_seqscan', 'off', true), "
                "set_config('enable_sort', 'off', true)"
            ).bindparams(ef=str(HNSW_EF_SEARCH))
        )

    async def _restore_planner_defaults(self) -> None:
        """Reopen seq scans/sorts for the REST of the transaction (priming
        Paths 2-4 and prompt loaders reuse this session's transaction — the
        pin must not distort their plans). Runs on the success path only: every
        error/timeout path rolls the session back, which discards the local
        GUCs anyway. ``hnsw.ef_search`` deliberately stays at 100 — it only
        affects HNSW scans."""
        await self.session.execute(
            text("SELECT set_config('enable_seqscan', 'on', true), set_config('enable_sort', 'on', true)")
        )

    async def search_traces(
        self,
        query: str,
        limit: int = 10,
        memory_types: list[str] | None = None,
    ) -> list[tuple[MemoryTrace, float]]:
        """Hybrid search: vector similarity + salience + arousal re-ranking."""
        try:
            query_embedding = await generate_embedding(query)
        except Exception:
            logger.warning("Embedding failed for query, falling back to empty")
            return []

        stmt = self._semantic_search_stmt(query_embedding, limit * 2, memory_types)

        await self._pin_semantic_search_plan()
        result = await self.session.execute(stmt)
        await self._restore_planner_defaults()
        rows = result.all()

        scored = []
        for trace, similarity in rows:
            arousal_boost = (trace.emotion_arousal or 0) * 0.1
            score = similarity * 0.6 + trace.salience * 0.3 + arousal_boost
            scored.append((trace, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    async def search_traces_raw(
        self,
        query: str,
        limit: int = 20,
        memory_types: list[str] | None = None,
    ) -> list[tuple[MemoryTrace, float]]:
        """Vector search returning raw cosine similarity (no re-ranking).

        Used by MemoryPrimingService for 5-component scoring where each
        component is scored independently.
        """
        try:
            query_embedding = await generate_embedding(query)
        except Exception:
            logger.warning("Embedding failed for query, falling back to empty")
            return []
        # Split-stage timing: the embed round trip is timed inside
        # `generate_embedding` (`embeddings.embed_call`); everything below is
        # the pgvector stage. The 4s L1 guard wraps BOTH, so a bare "semantic
        # search timed out" cannot say which stage burned the budget — and the
        # two have opposite fixes (client bounds vs index/query work).
        db_started = time.monotonic()

        stmt = self._semantic_search_stmt(query_embedding, limit, memory_types)

        await self._pin_semantic_search_plan()
        result = await self.session.execute(stmt)
        await self._restore_planner_defaults()
        rows = list(result.tuples().all())
        logger.info(
            "embeddings.db_query %s",
            json.dumps(
                {
                    "event_type": "embeddings.db_query",
                    "stage": "pgvector_search",
                    "caller": "search_traces_raw",
                    "duration_ms": round((time.monotonic() - db_started) * 1000, 1),
                    "rows": len(rows),
                    "limit": limit,
                }
            ),
        )
        return rows

    async def search_traces_by_vector(
        self,
        embedding: list[float],
        limit: int = 20,
        memory_types: list[str] | None = None,
    ) -> list[tuple[MemoryTrace, float]]:
        """Vector search using a pre-computed embedding (no generate_embedding call).

        Returns (trace, cosine_similarity) pairs sorted by similarity DESC.
        """
        stmt = self._semantic_search_stmt(embedding, limit, memory_types)

        await self._pin_semantic_search_plan()
        result = await self.session.execute(stmt)
        await self._restore_planner_defaults()
        return list(result.tuples().all())

    # ── Access tracking ────────────────────────────────────

    async def record_access(self, trace_id: _uuid.UUID) -> None:
        """Atomic increment of access_count + append timestamp to JSONB."""
        now = utc_now()
        now_iso = now.isoformat()
        stmt = (
            update(MemoryTrace)
            .where(MemoryTrace.id == trace_id)
            .values(
                access_count=MemoryTrace.access_count + 1,
                last_accessed_at=now,
                access_timestamps=func.coalesce(MemoryTrace.access_timestamps, text("'[]'::jsonb"))
                + func.cast(
                    func.jsonb_build_array(now_iso),
                    MemoryTrace.access_timestamps.type,
                ),
            )
        )
        await self.session.execute(stmt)

    async def record_access_batch(self, trace_ids: list[_uuid.UUID]) -> None:
        """Batch version — single UPDATE for N trace IDs."""
        if not trace_ids:
            return
        now = utc_now()
        now_iso = now.isoformat()
        stmt = (
            update(MemoryTrace)
            .where(MemoryTrace.id.in_(trace_ids))
            .values(
                access_count=MemoryTrace.access_count + 1,
                last_accessed_at=now,
                access_timestamps=func.coalesce(MemoryTrace.access_timestamps, text("'[]'::jsonb"))
                + func.cast(
                    func.jsonb_build_array(now_iso),
                    MemoryTrace.access_timestamps.type,
                ),
            )
        )
        await self.session.execute(stmt)

    # ── Entity operations ──────────────────────────────────

    async def find_or_create_entity(
        self,
        name: str,
        entity_type: EntityType = "person",
        namespace: str | None = None,
        telegram_user_id: int | None = None,
        data: dict | None = None,
        source_text: str | None = None,
    ) -> MemoryEntity:
        """Find by Telegram user ID, name/alias, or create.

        v2a §5.4: when `data` is provided and an existing row is found, the
        incoming dict is deep-merged into `entity.data`.

        v2a §5.5: `source_text` enables bidirectional alias expansion BEFORE
        the lookup query. Non-ASCII names contribute their unidecode form;
        ASCII names whose normalized romanization matches a Cyrillic token in
        `source_text` recover that token as a candidate. Both directions
        round-trip into `entity.aliases` so future lookups by either form hit.
        """
        # Stable identity bindings belong to the host, never generic entity data.
        if data and AGENT_IDENTITY_DATA_KEY in data:
            data = {key: value for key, value in data.items() if key != AGENT_IDENTITY_DATA_KEY}
        now = utc_now()

        # Pre-compute alias candidates for bidirectional lookup.
        alias_candidates = candidate_aliases(name, source_text)
        lookup_names = [name, *alias_candidates]

        # Cross-script canonical key (Sub-design C): folds Cyrillic↔Latin
        # transliteration variance so e.g. a lookup of "Misha" reaches an existing
        # "Миша" entity (canonical_key "misha") even when the Cyrillic form is not
        # in this message's source_text. Empty for garbage/punctuation-only names
        # — in that case the clause is omitted so it never mass-matches NULLs.
        canonical_key = translit_key(name)

        if telegram_user_id is not None:
            result = await self.session.execute(
                select(MemoryEntity).where(
                    MemoryEntity.telegram_user_id == telegram_user_id,
                    MemoryEntity.merged_into_id.is_(None),
                )
            )
            entity = result.scalar_one_or_none()
            if entity:
                self._merge_aliases(entity, name, alias_candidates)
                if data:
                    entity.data = _deep_merge(entity.data or {}, data)
                await self._bump_mention(entity.id, now)
                return entity

        match_clauses: list[Any] = [
            MemoryEntity.name.in_(lookup_names),
            *[MemoryEntity.aliases.op("@>")(func.jsonb_build_array(n)) for n in lookup_names],
        ]
        if canonical_key:
            match_clauses.append(MemoryEntity.canonical_key == canonical_key)
        filters = [
            MemoryEntity.entity_type == entity_type,
            MemoryEntity.merged_into_id.is_(None),
            or_(*match_clauses),
        ]
        if telegram_user_id is None:
            filters.append(MemoryEntity.telegram_user_id.is_(None))

        stmt = (
            select(MemoryEntity)
            .where(*filters)
            .order_by(
                (MemoryEntity.name == name).desc(),
                MemoryEntity.mention_count.desc(),
            )
            .limit(1)
        )
        result = await self.session.execute(stmt)
        entity = result.scalar_one_or_none()

        if entity:
            if telegram_user_id is not None:
                if entity.telegram_user_id is None:
                    entity.telegram_user_id = telegram_user_id
                elif entity.telegram_user_id != telegram_user_id:
                    return await self._create_entity(
                        name,
                        entity_type,
                        namespace,
                        telegram_user_id,
                        now,
                        data=data,
                        aliases=alias_candidates,
                    )
            # Cross-script recovery signal (Sub-design C, feature-watch): the
            # entity was found but the looked-up name/aliases did not surface it
            # — so the canonical_key clause is what matched. This is the fork the
            # resolver just avoided. INFO so it reaches CloudWatch. Logs a stable
            # entity id + type + a non-reversible key fingerprint, never the raw
            # personal names, to keep PII out of centralized logs.
            if canonical_key and not self._matched_by_surface(entity, lookup_names):
                key_fp = hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()[:12]
                logger.info(
                    "cognitive.entity.cross_script_merge entity_id=%s type=%r key_fp=%s",
                    entity.id,
                    entity_type,
                    key_fp,
                )
            self._merge_aliases(entity, name, alias_candidates)
            if data:
                entity.data = _deep_merge(entity.data or {}, data)
            await self._bump_mention(entity.id, now)
            return entity

        # Tier-0 shadow probe (task #8) — LOG-ONLY, behavior-neutral. This is a
        # text-mention (tg_id None) MISS about to create a fork; measure what a
        # future Tier-0 cross-partition resolver WOULD find here, WITHOUT changing
        # the returned entity. Fires only on this path; fail-open (never breaks
        # resolution). See backend-docs/ai-workforce/plans/design/2026-07-03-entity-resolution-v2-design.md §3.
        if telegram_user_id is None and cognitive_config.get_bool("tier0_shadow_probe_enabled"):
            await self._tier0_shadow_probe(name, entity_type, namespace, lookup_names, canonical_key)

        return await self._create_entity(
            name,
            entity_type,
            namespace,
            telegram_user_id,
            now,
            data=data,
            aliases=alias_candidates,
        )

    async def _tier0_shadow_probe(
        self,
        name: str,
        entity_type: EntityType,
        namespace: str | None,
        lookup_names: list[str],
        canonical_key: str,
    ) -> None:
        """LOG-ONLY Tier-0 measurement (task #8). MUST NOT mutate state or change
        the caller's return value — it only reads and logs.

        Runs the resolver's SAME surface match (name / alias / canonical_key,
        type, active) but WITHOUT the ``telegram_user_id IS NULL`` partition, so
        it sees tg-bound rows too. Because the caller's NULL-tg lookup already
        missed, every row this finds is necessarily tg-bound — exactly the Tier-0
        target: a text mention that forks today instead of reaching the canonical
        tg-bound row. Emits one structured ``tier0_shadow`` line per miss with a
        candidate, answering H1 (how often a cross-partition match exists),
        H2 (is it unique), H3 (is the matched surface high-specificity), and
        H4 (an id-keyed audit sample). Fail-open: any error is swallowed so
        instrumentation can never break resolution.

        PII: logs a non-reversible name/key fingerprint + ``candidate_entity_id``
        (not raw names) — matching the ``cross_script_merge`` convention above and
        the multi-tenant portability direction. The H4 misbind audit anchors on
        ``candidate_entity_id`` (join to the source trace / edges in the DB, which
        is the real ground truth, not the surface string).
        """
        try:
            match_clauses: list[Any] = [
                MemoryEntity.name.in_(lookup_names),
                *[MemoryEntity.aliases.op("@>")(func.jsonb_build_array(n)) for n in lookup_names],
            ]
            if canonical_key:
                match_clauses.append(MemoryEntity.canonical_key == canonical_key)
            stmt = (
                select(MemoryEntity)
                .where(
                    MemoryEntity.entity_type == entity_type,
                    MemoryEntity.merged_into_id.is_(None),
                    or_(*match_clauses),
                )
                .order_by(MemoryEntity.mention_count.desc())
                # Bounded: the probe only needs to tell 0 / 1 / 2+ apart (no
                # hypothesis needs the exact count — 2+ ⇒ Tier-0 refuses either
                # way), so cap the scan at 2. order_by keeps the single kept row
                # on a would_bind the highest-mention one.
                .limit(2)
            )
            # ``no_autoflush``: a read must NOT flush pending session state — that
            # would change flush timing inside the transaction and make the
            # "behavior-neutral" guarantee non-airtight (and a stray autoflush
            # error would be swallowed by the except below).
            with self.session.no_autoflush:
                candidates = (await self.session.execute(stmt)).scalars().all()
            count = len(candidates)  # capped at 2 by limit(2): a count of 2 means ">=2"
            if count < 1:
                return  # no cross-partition match — nothing to measure (keep log noise low)

            would_bind = count == 1  # H2: exactly one (limit(2) returns 2 if >=2 exist)
            ambiguous = count >= 2  # H2: 2+ candidates → Tier-0 refuses regardless of the true count
            specificity: str | None = None
            matched_via: str | None = None
            candidate_id: str | None = None
            candidate_mentions: int | None = None
            tier0_would_bind_under_gate = False
            if would_bind:
                candidate = candidates[0]
                matched_via, matched_surface = self._tier0_matched_surface(candidate, lookup_names, canonical_key)
                # H3: high-specificity gate binds only a multi-token (full-name-grade)
                # surface, never a bare single given-name token.
                specificity = "multi_token" if len((matched_surface or "").split()) >= 2 else "single_token"
                tier0_would_bind_under_gate = specificity == "multi_token"
                candidate_id = str(candidate.id)  # H4: audit anchor (join to trace/edges in DB)
                candidate_mentions = candidate.mention_count

            payload = {
                "event": "resolver.tier0_shadow",
                "entity_type": entity_type,
                "namespace": namespace,
                "name_fp": hashlib.sha256(name.encode("utf-8")).hexdigest()[:12],
                "key_fp": (hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()[:12] if canonical_key else None),
                "candidate_count": count,  # H1 (bounded by limit(2); 2 == ">=2")
                "would_bind": would_bind,  # H2
                "ambiguous": ambiguous,  # H2: candidate_count >= 2 (the cap ceiling)
                "match_specificity": specificity,  # H3
                "matched_via": matched_via,
                "tier0_would_bind_under_gate": tier0_would_bind_under_gate,
                "candidate_entity_id": candidate_id,  # H4 (id-keyed; no raw names)
                "candidate_mention_count": candidate_mentions,  # H4
            }
            # Inline-serialize the payload as the message (basicConfig drops
            # ``extra=`` keys); "tier0_shadow" is the greppable marker.
            logger.info("tier0_shadow %s", json.dumps(payload))
        except Exception:
            logger.warning("tier0_shadow probe failed (ignored)", exc_info=True)

    @staticmethod
    def _tier0_matched_surface(
        candidate: MemoryEntity, lookup_names: list[str], canonical_key: str
    ) -> tuple[str | None, str | None]:
        """Which surface matched the shadow candidate + that surface string.

        Mirrors the resolver's OR-clause precedence (name, then alias, then
        canonical_key) so the specificity classification measures the surface the
        gate would actually bind on.
        """
        lookup_set = set(lookup_names)
        if candidate.name in lookup_set:
            return "name", candidate.name
        alias_hit = next((a for a in (candidate.aliases or []) if a in lookup_set), None)
        if alias_hit is not None:
            return "alias", alias_hit
        if canonical_key and candidate.canonical_key == canonical_key:
            return "canonical_key", candidate.canonical_key
        return None, None

    @staticmethod
    def _matched_by_surface(entity: MemoryEntity, lookup_names: list[str]) -> bool:
        """Whether `entity` was reachable by name/alias (vs only canonical_key).

        Used to tell a cross-script (canonical_key-only) recovery apart from an
        ordinary name/alias hit, so the observability log fires only for the
        former. The OR-filter's only other branch is canonical_key, so
        ``not matched_by_surface`` ⇒ matched by canonical_key.
        """
        if entity.name in lookup_names:
            return True
        return bool(set(entity.aliases or []) & set(lookup_names))

    @staticmethod
    def _merge_aliases(entity: MemoryEntity, name: str, candidates: list[str]) -> None:
        """Append `candidates` (and the lookup `name` if mismatched) to entity.aliases.

        Existing aliases are preserved; duplicates and the entity's own name
        are skipped.
        """
        existing = list(entity.aliases or [])
        existing_set = set(existing)
        appended = list(existing)
        for candidate in [name, *candidates]:
            if candidate == entity.name or candidate in existing_set:
                continue
            existing_set.add(candidate)
            appended.append(candidate)
        if appended != existing:
            entity.aliases = appended

    async def _create_entity(
        self,
        name: str,
        entity_type: str,
        namespace: str | None,
        telegram_user_id: int | None,
        now: datetime,
        *,
        data: dict | None = None,
        aliases: list[str] | None = None,
    ) -> MemoryEntity:
        values: dict = {
            "name": name,
            "entity_type": entity_type,
            "namespace": namespace,
            "mention_count": 1,
            "last_mentioned_at": now,
        }
        # Cross-script canonical key (Sub-design C); omitted for empty-key
        # (garbage) names so the column stays NULL rather than "".
        key = translit_key(name)
        if key:
            values["canonical_key"] = key
        if telegram_user_id is not None:
            values["telegram_user_id"] = telegram_user_id
        if data:
            values["data"] = data
        if aliases:
            values["aliases"] = [a for a in aliases if a != name]

        if telegram_user_id is None:
            insert_stmt = (
                pg_insert(MemoryEntity)
                .values(**values)
                .on_conflict_do_update(
                    index_elements=["entity_type", "name"],
                    index_where=MemoryEntity.telegram_user_id.is_(None),
                    set_={
                        "mention_count": MemoryEntity.mention_count + 1,
                        "last_mentioned_at": now,
                    },
                )
                .returning(MemoryEntity)
            )
        else:
            insert_stmt = (
                pg_insert(MemoryEntity)
                .values(**values)
                .on_conflict_do_update(
                    index_elements=["telegram_user_id"],
                    index_where=MemoryEntity.telegram_user_id.isnot(None),
                    set_={
                        "mention_count": MemoryEntity.mention_count + 1,
                        "last_mentioned_at": now,
                    },
                )
                .returning(MemoryEntity)
            )

        result = await self.session.execute(insert_stmt)
        return result.scalar_one()

    async def _bump_mention(self, entity_id: _uuid.UUID, now: datetime) -> None:
        """Atomic increment of mention_count and update last_mentioned_at."""
        stmt = (
            update(MemoryEntity)
            .where(MemoryEntity.id == entity_id)
            .values(
                mention_count=MemoryEntity.mention_count + 1,
                last_mentioned_at=now,
            )
        )
        await self.session.execute(stmt)

    async def link_trace_to_entity(
        self,
        trace_id: _uuid.UUID,
        entity_id: _uuid.UUID,
        role: LinkRole = "about",
    ) -> MemoryEntityLink:
        """Link a memory trace to an entity. Silently idempotent on duplicates."""
        entity = await self.session.get(MemoryEntity, entity_id)
        visited = {entity_id}
        for _ in range(MERGE_CHAIN_MAX_DEPTH):
            next_id = getattr(entity, "merged_into_id", None) if entity is not None else None
            if not isinstance(next_id, _uuid.UUID) or next_id in visited:
                break
            survivor = await self.session.get(MemoryEntity, next_id)
            if survivor is None:
                break
            entity_id = survivor.id
            entity = survivor
            visited.add(entity_id)

        stmt = (
            pg_insert(MemoryEntityLink)
            .values(memory_id=trace_id, entity_id=entity_id, role=role)
            .on_conflict_do_nothing(
                index_elements=["memory_id", "entity_id", "role"],
            )
            .returning(MemoryEntityLink)
        )
        result = await self.session.execute(stmt)
        existing = result.scalar_one_or_none()
        if existing:
            return existing

        # ON CONFLICT DO NOTHING returned no row — fetch the existing link
        fetch = select(MemoryEntityLink).where(
            MemoryEntityLink.memory_id == trace_id,
            MemoryEntityLink.entity_id == entity_id,
            MemoryEntityLink.role == role,
        )
        result = await self.session.execute(fetch)
        return result.scalar_one()

    # ── Invalidation (contradiction handling) ──────────────

    async def invalidate_trace(
        self,
        trace_id: _uuid.UUID,
        invalidated_by: _uuid.UUID | None = None,
        confidence: float = 1.0,
        invalidation_source: str | None = None,
    ) -> None:
        """Mark a trace as invalidated — preserves history, excluded from search.

        Sub-design C INC 1: ``invalidation_source`` records the correction-WRITER
        ('l2_inline' | 'l3_batch') that wrote this invalidation edge, so E (INC 8)
        can separate the two correction streams via ``GROUP BY``. Stamped only
        when ``typed_metadata_write_enabled`` is on; off → the edge is written
        exactly as today (source left NULL).
        """
        now = utc_now()
        values: dict = {
            "invalidated_at": now,
            "invalidated_by": invalidated_by,
            "invalidation_confidence": confidence,
        }
        if invalidation_source is not None and cognitive_config.get_bool("typed_metadata_write_enabled"):
            values["invalidation_source"] = invalidation_source
        stmt = (
            update(MemoryTrace).where(MemoryTrace.id == trace_id, MemoryTrace.invalidated_at.is_(None)).values(**values)
        )
        await self.session.execute(stmt)

    # ── Graph traversal ────────────────────────────────────

    async def traverse_entity_graph(
        self,
        start_entity_ids: list[_uuid.UUID],
        max_depth: int = 3,
        min_confidence: float = 0.3,
    ) -> list[tuple[_uuid.UUID, float]]:
        """Spreading activation via recursive CTE with cycle detection.

        Returns list of (entity_id, activation_energy) pairs reachable from
        start_entity_ids within max_depth hops. Energy decays by
        confidence * 0.6 per hop for diminishing returns with distance.
        """
        if not start_entity_ids:
            return []

        cte_sql = text("""
            WITH RECURSIVE graph AS (
                SELECT
                    id AS entity_id,
                    1.0::float AS energy,
                    0 AS depth,
                    ARRAY[id] AS visited
                FROM memory_entities
                WHERE id = ANY(:start_ids)
                  AND merged_into_id IS NULL

                UNION ALL

                SELECT
                    CASE
                        WHEN er.from_entity_id = g.entity_id THEN er.to_entity_id
                        ELSE er.from_entity_id
                    END AS entity_id,
                    g.energy * er.confidence * (
                        0.4 + 0.2 * LEAST(1.0, LN(1 + COALESCE(me_target.mention_count, 0)) / LN(10))
                    ) AS energy,
                    g.depth + 1 AS depth,
                    g.visited || CASE
                        WHEN er.from_entity_id = g.entity_id THEN er.to_entity_id
                        ELSE er.from_entity_id
                    END AS visited
                FROM graph g
                JOIN entity_relationships er ON (
                    er.from_entity_id = g.entity_id OR er.to_entity_id = g.entity_id
                )
                JOIN memory_entities me_target ON me_target.id = CASE
                    WHEN er.from_entity_id = g.entity_id THEN er.to_entity_id
                    ELSE er.from_entity_id
                END
                    AND me_target.merged_into_id IS NULL
                WHERE g.depth < :max_depth
                  AND er.valid_until IS NULL
                  AND er.confidence >= :min_confidence
                  AND NOT (
                    CASE
                        WHEN er.from_entity_id = g.entity_id THEN er.to_entity_id
                        ELSE er.from_entity_id
                    END = ANY(g.visited)
                  )
            )
            SELECT entity_id, MAX(energy) AS energy
            FROM graph
            WHERE depth > 0
            GROUP BY entity_id
            ORDER BY energy DESC
            LIMIT :result_limit
        """)

        from kai_memory.store import cognitive_config

        result = await self.session.execute(
            cte_sql,
            {
                "start_ids": start_entity_ids,
                "max_depth": max_depth,
                "min_confidence": min_confidence,
                "result_limit": cognitive_config.get_int("graph_traversal_limit"),
            },
        )
        return [(row.entity_id, row.energy) for row in result]

    async def get_entity_linked_traces(
        self,
        entity_ids: list[_uuid.UUID],
        limit: int = 10,
    ) -> list[MemoryTrace]:
        """Get non-invalidated traces linked to any of the given entity IDs."""
        if not entity_ids:
            return []
        stmt = (
            select(MemoryTrace)
            .join(MemoryEntityLink, MemoryEntityLink.memory_id == MemoryTrace.id)
            .where(
                MemoryEntityLink.entity_id.in_(entity_ids),
                MemoryTrace.invalidated_at.is_(None),
                MemoryTrace.memory_type.in_(SERVABLE_MEMORY_TYPES),  # allowlist — procedural stays out of priming
            )
            .distinct()
            .order_by(MemoryTrace.salience.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_entity_linked_traces_with_entity(
        self,
        entity_ids: list[_uuid.UUID],
        limit: int = 10,
    ) -> list[tuple[MemoryTrace, _uuid.UUID]]:
        """Get (trace, linked_entity_id) pairs for energy-aware graph scoring."""
        if not entity_ids:
            return []
        stmt = (
            select(MemoryTrace, MemoryEntityLink.entity_id)
            .join(MemoryEntityLink, MemoryEntityLink.memory_id == MemoryTrace.id)
            .where(
                MemoryEntityLink.entity_id.in_(entity_ids),
                MemoryTrace.invalidated_at.is_(None),
                MemoryTrace.memory_type.in_(SERVABLE_MEMORY_TYPES),  # allowlist — procedural stays out of priming
            )
            .order_by(MemoryTrace.salience.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.tuples().all())

    async def get_intention_traces(
        self,
        entity_ids: list[_uuid.UUID],
        limit: int = 5,
    ) -> list[MemoryTrace]:
        """Get intention-type traces linked to mentioned entities."""
        if not entity_ids:
            return []
        stmt = (
            select(MemoryTrace)
            .join(MemoryEntityLink, MemoryEntityLink.memory_id == MemoryTrace.id)
            .where(
                MemoryEntityLink.entity_id.in_(entity_ids),
                MemoryTrace.memory_type == "intention",
                MemoryTrace.invalidated_at.is_(None),
            )
            .order_by(MemoryTrace.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    # ── Retrieval config ──────────────────────────────────

    async def get_retrieval_weights(self) -> dict[str, float]:
        """Read adaptive retrieval weights from the single-row config table."""
        result = await self.session.execute(select(RetrievalConfig).limit(1))
        config = result.scalar_one_or_none()
        if config:
            return {
                "semantic": config.weight_semantic,
                "graph": config.weight_graph,
                "recency": config.weight_recency,
                "access": config.weight_access,
                "arousal": config.weight_arousal,
                "salience": config.weight_salience,  # A2 — default 0.0 (inert)
                "intention": config.weight_intention,  # #273 — static, never auto-tuned
            }
        # Fallback defaults matching the seeded row
        return dict(DEFAULT_RETRIEVAL_WEIGHTS)

    async def find_entities_by_names(
        self,
        names: list[str],
        entity_type: str | None = None,
        only_ungrounded: bool = False,
    ) -> list[MemoryEntity]:
        """Find entities matching any of the given names (exact or alias)."""
        if not names:
            return []
        # Exact name match
        stmt = select(MemoryEntity).where(
            MemoryEntity.name.in_(names),
            MemoryEntity.merged_into_id.is_(None),
        )
        if entity_type is not None:
            stmt = stmt.where(MemoryEntity.entity_type == entity_type)
        if only_ungrounded:
            stmt = stmt.where(MemoryEntity.telegram_user_id.is_(None))
        result = await self.session.execute(stmt)
        found = list(result.scalars().all())
        found_ids = {e.id for e in found}
        found_names = {e.name for e in found}

        # Batch alias search for unmatched names — single query with OR
        remaining = [n for n in names if n not in found_names]
        if remaining:
            alias_conditions = [MemoryEntity.aliases.op("@>")(func.jsonb_build_array(name)) for name in remaining]
            alias_stmt = select(MemoryEntity).where(
                or_(*alias_conditions),
                MemoryEntity.merged_into_id.is_(None),
            )
            if entity_type is not None:
                alias_stmt = alias_stmt.where(MemoryEntity.entity_type == entity_type)
            if only_ungrounded:
                alias_stmt = alias_stmt.where(MemoryEntity.telegram_user_id.is_(None))
            alias_result = await self.session.execute(alias_stmt)
            for entity in alias_result.scalars().all():
                if entity.id not in found_ids:
                    found.append(entity)
                    found_ids.add(entity.id)

        return found

    async def _aliases_lower_exists(self) -> bool:
        """Whether migration 106's ``aliases_lower(jsonb)`` helper is in the database.

        A ``to_regprocedure`` read, cached per database once it answers True and
        repeated while it answers False — the helper can appear under a running
        process, so a miss must not stick. The warning is one-shot regardless.
        See ``_ALIASES_LOWER_PROBE`` above for why the pre-106 window exists.
        """
        probe = owned_state("aliases_lower_probe", _AliasesLowerProbe, _ALIASES_LOWER_STATE)
        if probe.available:
            return True
        exists = bool((await self.session.execute(_ALIASES_LOWER_PROBE)).scalar())
        if exists:
            probe.available = True
        elif not probe.warned:
            probe.warned = True
            logger.warning(
                "aliases_lower(jsonb) is absent — migration 106 not applied; "
                "alias lookup falls back to the unindexed form"
            )
        return exists

    async def find_entities_by_lowercase_tokens(
        self,
        tokens: list[str],
        limit: int = 10,
    ) -> list[MemoryEntity]:
        """Case-insensitive lookup against entity names AND aliases.

        v2a §9 Task 6b fallback for priming when the capitalized-name regex
        returns no candidates. Tokens shorter than 3 chars are dropped (the
        caller already filters, but defensive).

        The alias half has two shapes: the GIN-eligible one, and the pre-106
        unindexed EXISTS for a database that has not run migration 106 yet
        (see ``_ALIASES_LOWER_PROBE``). Results are identical either way.
        """
        lowered = sorted({t.lower() for t in tokens if len(t) >= 3})
        if not lowered:
            return []
        # Two statements, each served by its own index (DB-621). The old single
        # statement OR-ed ``lower(name) IN (...)`` with an EXISTS over the
        # unnested alias array, and an OR with a subquery is something no index
        # can serve: it full-scanned memory_entities on every priming fallback.
        # Name half: expression index ``idx_mentity_lower_name_active``.
        by_name = (
            select(MemoryEntity)
            .where(
                MemoryEntity.merged_into_id.is_(None),
                func.lower(MemoryEntity.name).in_(lowered),
            )
            .limit(limit)
        )
        found = list((await self.session.execute(by_name)).scalars().all())
        if len(found) >= limit:
            return found[:limit]
        # Alias half: ``aliases_lower(aliases) ?| tokens`` against the GIN over the
        # IMMUTABLE lowercasing helper (idx_mentity_aliases_lower_gin). asyncpg
        # can't infer a text-array type from a bare Python list, so the bind
        # carries an explicit ARRAY(String) type — without it the first call
        # crashes with "could not determine data type of parameter".
        # The probe runs here, not before the name half: a lookup the names
        # already answered must cost exactly one statement. Once it answers
        # True it is never repeated.
        tokens_param = bindparam("lowered_tokens", value=lowered, type_=ARRAY(String()))
        if await self._aliases_lower_exists():
            alias_match: Any = func.aliases_lower(MemoryEntity.aliases).op("?|")(tokens_param)
        else:
            # Pre-106 schema: the old unindexed EXISTS over the unnested alias
            # array. Still its own statement — the name half stays separate in
            # both branches, because re-OR-ing them would also make the name
            # half unservable by its index.
            alias_match = text(
                "EXISTS (SELECT 1 FROM jsonb_array_elements_text(aliases) AS a(v) "
                "WHERE lower(a.v) = ANY(:lowered_tokens))"
            ).bindparams(tokens_param)
        by_alias = (
            select(MemoryEntity)
            .where(
                MemoryEntity.merged_into_id.is_(None),
                alias_match,
            )
            .limit(limit)
        )
        seen = {entity.id for entity in found}
        for entity in (await self.session.execute(by_alias)).scalars().all():
            if entity.id in seen:
                continue
            found.append(entity)
            seen.add(entity.id)
            if len(found) >= limit:
                break
        return found

    # ── Read helpers ───────────────────────────────────────

    async def get_trace(self, trace_id: _uuid.UUID) -> MemoryTrace | None:
        """Get a single trace by ID."""
        result = await self.session.execute(select(MemoryTrace).where(MemoryTrace.id == trace_id))
        return result.scalar_one_or_none()

    async def get_person_entity_ids_for_trace(self, trace_id: _uuid.UUID) -> list[_uuid.UUID]:
        """Person entities linked to a trace — its SUBJECT candidates (DB-591).

        Person-typed only, and deliberately across every link role: a trace's subject
        may be linked as 'subject', 'about' or 'participant' depending on which writer
        stored it, so filtering by role would make older traces read as subject-less.
        Topics/organizations are excluded because two facts about different people
        routinely share one ('work', 'health') — an overlap there is not a shared
        subject, which is the only question this answers.

        Ids are resolved through ``merged_into_id`` to the survivor. ``link_trace_to_entity``
        only follows the chain at WRITE time, so a link stored before a dedup merge still
        points at the absorbed row — while the caller compares against subject ids that came
        from ``find_or_create_entity``, which always returns a survivor. Left unresolved, a
        name fork folded after the fact would make one person look like two and silently
        block their own legitimate corrections.
        """
        stmt = (
            select(MemoryEntityLink.entity_id)
            .join(MemoryEntity, MemoryEntity.id == MemoryEntityLink.entity_id)
            .where(MemoryEntityLink.memory_id == trace_id, MemoryEntity.entity_type == "person")
        )
        result = await self.session.execute(stmt)
        return await self.resolve_entity_survivors(result.scalars().all())

    async def resolve_entity_survivors(self, entity_ids: Iterable[_uuid.UUID]) -> list[_uuid.UUID]:
        """Map each entity id to its merge-chain survivor, de-duplicated, order-preserving.

        The one place identity comparison resolves merges (DB-591). Resolution walks the
        chain a LEVEL AT A TIME across all inputs, so the whole call costs at most
        ``MERGE_CHAIN_MAX_DEPTH`` queries no matter how many ids it is handed — a
        per-id ``get_entity`` walk would be O(ids x depth) sequential reads on the L2
        write path, which the repo's path rules forbid. Bounding the INPUT instead
        would be worse: a dropped subject reads as a mismatch, which is exactly the
        false skip this gate exists to avoid.

        Failure degrades to the ORIGINAL id, never to nothing — a missing row, a broken
        chain, a cycle, or a chain deeper than the cap all map an id to itself, so the
        comparison reads as "unresolved" rather than silently losing a subject. Same
        semantics as ``get_entity`` returning None, which callers already treat that way.
        """
        originals = list(dict.fromkeys(entity_ids))
        if not originals:
            return []
        cursor: dict[_uuid.UUID, _uuid.UUID] = {oid: oid for oid in originals}
        seen_per_id: dict[_uuid.UUID, set[_uuid.UUID]] = {oid: {oid} for oid in originals}
        resolved: dict[_uuid.UUID, _uuid.UUID] = {}
        for _ in range(MERGE_CHAIN_MAX_DEPTH):
            pending = {oid: cid for oid, cid in cursor.items() if oid not in resolved}
            if not pending:
                break
            rows = (
                await self.session.execute(
                    select(MemoryEntity.id, MemoryEntity.merged_into_id).where(
                        MemoryEntity.id.in_(set(pending.values()))
                    )
                )
            ).all()
            edges = {row_id: merged_into for row_id, merged_into in rows}
            for oid, cid in pending.items():
                if cid not in edges:
                    logger.warning("Merge chain broken at %s while resolving %s", cid, oid)
                    resolved[oid] = oid
                    continue
                next_id = edges[cid]
                if not isinstance(next_id, _uuid.UUID):
                    resolved[oid] = cid  # terminal survivor
                elif next_id in seen_per_id[oid]:
                    logger.warning("Merge cycle detected while resolving entity %s", oid)
                    resolved[oid] = oid
                else:
                    seen_per_id[oid].add(next_id)
                    cursor[oid] = next_id
        for oid in originals:
            if oid not in resolved:
                logger.warning("Merge chain exceeds max depth %d starting from %s", MERGE_CHAIN_MAX_DEPTH, oid)
                resolved[oid] = oid
        survivors: list[_uuid.UUID] = []
        emitted: set[_uuid.UUID] = set()
        for oid in originals:
            survivor = resolved[oid]
            if survivor not in emitted:
                emitted.add(survivor)
                survivors.append(survivor)
        return survivors

    async def get_traces_batch(self, trace_ids: list[_uuid.UUID]) -> list[MemoryTrace]:
        """Fetch multiple traces in a single query."""
        if not trace_ids:
            return []
        result = await self.session.execute(select(MemoryTrace).where(MemoryTrace.id.in_(trace_ids)))
        return list(result.scalars().all())

    async def get_entity(self, entity_id: _uuid.UUID) -> MemoryEntity | None:
        """Get a single entity by ID, following merged_into_id to the survivor."""
        entity = await self.session.get(MemoryEntity, entity_id)
        if entity is None:
            return None
        visited = {entity_id}
        for _ in range(MERGE_CHAIN_MAX_DEPTH):
            next_id = getattr(entity, "merged_into_id", None)
            if not isinstance(next_id, _uuid.UUID):
                return entity
            if next_id in visited:
                logger.warning("Merge cycle detected while fetching entity %s", entity_id)
                return None
            visited.add(next_id)
            survivor = await self.session.get(MemoryEntity, next_id)
            if survivor is None:
                logger.warning("Merge chain broken at %s -> %s", entity.id, next_id)
                return None
            entity = survivor
        logger.warning(
            "Merge chain exceeds max depth %d starting from %s",
            MERGE_CHAIN_MAX_DEPTH,
            entity_id,
        )
        return None

    async def get_entity_traces(self, entity_id: _uuid.UUID, limit: int = 20) -> list[MemoryTrace]:
        """Get memory traces linked to an entity, ordered by salience."""
        stmt = (
            select(MemoryTrace)
            .join(MemoryEntityLink, MemoryEntityLink.memory_id == MemoryTrace.id)
            .where(
                MemoryEntityLink.entity_id == entity_id,
                MemoryTrace.invalidated_at.is_(None),
            )
            .order_by(MemoryTrace.salience.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
