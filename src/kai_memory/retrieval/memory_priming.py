"""Level 1 memory priming — per-message retrieval for prompt injection."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncSession

import uuid

if TYPE_CHECKING:
    from kai_memory.ports import ChatContext
    from kai_memory.retrieval.iterative_retrieval import WindowCues
    from kai_memory.retrieval.memory_observation import MemoryDecision

from kai_memory.host_calls import recover_session, utc_now
from kai_memory.ownership import current_owner
from kai_memory.storage.models.memory_trace import MemoryTrace
from kai_memory.store import cognitive_config
from kai_memory.store.cognitive_memory import CognitiveMemoryService
from kai_memory.retrieval.memory_mechanisms import MechanismConfig, apply_mechanisms
from kai_memory.store.cognitive_memory_constants import DEFAULT_RETRIEVAL_WEIGHTS
from kai_memory.romanization import CYRILLIC_STOPWORDS

# Pinned to the pre-DB-842 module path: CloudWatch queries, feature-watch criteria and
# log filters match the logger name, which must not follow the module's move.
_LEGACY_LOGGER_NAME = "app.services.chat.memory_priming"
logger = logging.getLogger(_LEGACY_LOGGER_NAME)

# Cache retrieval weights — data changes only during nightly consolidation
_WEIGHTS_CACHE: tuple[float, dict[str, float]] | None = None
_WEIGHTS_CACHE_TTL = 300  # 5 min — explicit invalidation handles real changes
_WEIGHTS_CACHE_NAME = "retrieval_weights"  # a memory client's own copy (``ownership``)

# Regex for capitalized words (Latin and Cyrillic) — extracts likely entity names
_ENTITY_NAME_RE = re.compile(r"(?:[A-Z\u0410-\u042F][a-z\u0430-\u044F]+(?:\s+[A-Z\u0410-\u042F][a-z\u0430-\u044F]+)*)")

# Common sentence starters to filter out of entity extraction
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "Where",
        "What",
        "When",
        "How",
        "The",
        "This",
        "That",
        "There",
        "Here",
        "Which",
        "Who",
        "Why",
        "Can",
        "Could",
        "Would",
        "Should",
        "Will",
        "Have",
        "Has",
        "Does",
        "Did",
        "But",
        "And",
        "Also",
        "Just",
        "Maybe",
        "Please",
        "Sure",
        "Yes",
        "Thanks",
        "Hello",
        "Hey",
        # Russian
        "Как",
        "Что",
        "Где",
        "Когда",
        "Зачем",
        "Почему",
        "Кто",
        "Это",
        "Вот",
        "Тут",
        "Там",
        "Ещё",
        "Уже",
        "Тоже",
        "Пожалуйста",
        "Привет",
    }
)

# Recency half-life in days (~0.5 after 7 days)
_RECENCY_HALF_LIFE_DAYS = 7.0
# Max expected access count for log normalization
_MAX_EXPECTED_ACCESS = 50

# Soft cap on the chars sent to the embedding model. Hard byte truncation
# happens inside ``generate_embedding`` (see ``embeddings._MAX_INPUT_BYTES``);
# this is the priming-layer ceiling that produces sensible search behavior
# before truncation kicks in. Semantic queries longer than ~2k chars rarely
# improve recall and waste embed tokens.
_MAX_QUERY_CHARS = 2000

# Marker emitted by ``format_event_message`` that separates the YAML-ish
# event header (title/summary/participants/etc.) from the raw event content
# (e.g. a full meeting transcript). For event messages we embed the header
# only — the content can be a 45kB+ Fireflies transcript that blows the
# OpenAI 8192-token ceiling (caught in CloudWatch 2026-05-14 via the
# diagnostic added in PR #136).
_EVENT_CONTENT_SEPARATOR = "\n--- content ---\n"


def _derive_embedding_query(last_message: str) -> str:
    """Pick the right slice of ``last_message`` to feed to the embedder.

    Three rules in priority order:

    1. **Event messages** — start with ``[EVENT]`` and contain the
       ``--- content ---`` separator. Embed the header section only
       (title, fireflies_summary, participants, keywords). The header is
       <2kB; the content (transcript) can be 45kB+ and blows the model's
       token ceiling. The header is also a stronger semantic anchor for
       retrieval — title + summary + participants discriminate better
       than verbatim transcript text.
    2. **Otherwise** — hard-truncate to ``_MAX_QUERY_CHARS``. Semantic
       search rarely benefits from longer queries and we want cheap
       embed calls.

    The hard byte truncation in ``embeddings.py`` still applies on top of
    whatever this function returns, so anything pathological still gets
    bounded at the API boundary.
    """
    if last_message.startswith("[EVENT]") and _EVENT_CONTENT_SEPARATOR in last_message:
        header, _, _ = last_message.partition(_EVENT_CONTENT_SEPARATOR)
        # Header is typically <2kB but bound anyway for safety.
        return header[:_MAX_QUERY_CHARS]
    return last_message[:_MAX_QUERY_CHARS]


# Semantic search timeout — embedding call to OpenAI + pgvector query.
_SEMANTIC_TIMEOUT_S = 4.0
# L1.5 windowed Path 1 (LIVE hot path): each per-segment pgvector call is bounded by
# _SEMANTIC_TIMEOUT_S, but a deep window has up to ~12 segments. Cap the number of
# searches to the highest-weight segments (older/low-weight ones contribute negligibly
# under weighted-max) and bound their TOTAL wall-clock so worst-case latency stays sane
# on the 2 GB ARM instance.
_WINDOWED_MAX_SEARCH_SEGMENTS = 8
_WINDOWED_TOTAL_BUDGET_S = 4.0
# Counters for observability (semantic success/timeout/error rates)
_semantic_ok: int = 0
_semantic_timeout: int = 0
_semantic_error: int = 0
_windowed_budget_skipped: int = 0


def _snapshot_trace(trace: MemoryTrace) -> MemoryTrace:
    """Keep loaded values across recovery without detaching the session's row.

    Ranked traces are read-only; access recording writes by ID. A separate
    transient instance preserves the model's column surface for downstream
    readers without retaining ORM identity or rollback expiry. Container values
    are shared shallowly because retrieval and its re-rankers do not mutate them.
    """
    return MemoryTrace(**{column.key: getattr(trace, column.key, None) for column in inspect(MemoryTrace).column_attrs})


class MemoryPrimingService:
    """Level 1: Per-message memory retrieval for prompt injection.

    4 search paths: semantic, entity-linked, graph traversal, intentions.
    5-component scoring from retrieval_config table.
    """

    def __init__(
        self,
        session: AsyncSession,
        weights_override: dict[str, float] | None = None,
        *,
        observation: MemoryDecision | None = None,
    ) -> None:
        """`weights_override` is for benchmark/experiment callers ONLY.

        When provided, scoring uses exactly these weights for THIS instance —
        the module-level DB-weights cache is neither read nor written, so
        concurrent live-chat priming is untouched. The live request path never
        passes it and reads the cached `retrieval_config` weights exactly as
        before. Never monkeypatch class attributes to inject weights in-app —
        that leaks the override into every concurrent chat (the reason this
        parameter exists; see the 2026-07-02 benchmark-gated adaptation design).
        """
        self.session = session
        self.observation = observation
        self.memory_svc = CognitiveMemoryService(session)
        self._weights_override = dict(weights_override) if weights_override is not None else None

    async def _recover_session(self) -> None:
        """Clean the session after a DB-side failure inside a priming path.

        Delegates to the shared ``app.db.recover_session`` — the same rollback-then-
        invalidate behaviour the procedural capture and serving paths need, kept in one
        place so a fix to it reaches all three.
        """
        await recover_session(self.session)

    async def prime(
        self,
        last_message: str,
        limit: int = 8,
        chat_ctx: ChatContext | None = None,
        record_access: bool = True,
        window_cues: WindowCues | None = None,
    ) -> tuple[str, list[uuid.UUID]]:
        """Retrieve relevant memories for the current message.

        Returns (formatted_string, retrieved_trace_ids). The trace IDs are the
        memories surfaced to the agent — callers hand them to
        track_retrieved_traces() so Level 2 extraction can treat them as
        priority contradiction candidates.

        `chat_ctx` is optional but recommended: when provided, its chat_id /
        topic_id / venue ride along on the structured `memory.retrieve.complete`
        log so per-chat / per-topic / per-venue analyses become possible.

        `record_access=False` skips the access-count write so a read-only caller
        (e.g. the offline gold-eval harness) doesn't pollute the access signal it
        is measuring. Real request-path callers keep the default (True).

        `window_cues` (L1.5 Stage A LIVE) replaces the single last-message cue with a
        segmented, recency-weighted window: Path 1 does per-segment searches merged
        weighted-max, Paths 2/3 take the per-segment entity union scaled by best segment
        weight. `None` (default) = today's exact single-message behavior (byte-identical).
        """
        formatted, top_ids, _top = await self.prime_with_traces(
            last_message, limit=limit, chat_ctx=chat_ctx, record_access=record_access, window_cues=window_cues
        )
        return formatted, top_ids

    async def prime_with_traces(
        self,
        last_message: str,
        limit: int = 8,
        chat_ctx: ChatContext | None = None,
        record_access: bool = True,
        window_cues: WindowCues | None = None,
        emit_log: bool = True,
    ) -> tuple[str, list[uuid.UUID], list[tuple[MemoryTrace, float, dict[str, float]]]]:
        """Variant of `prime()` that returns the ranked (trace, score, components) triples.

        review-r1 fix 4: prevents the adapter's `retrieve()` from re-fetching
        the same MemoryTrace rows it just discarded.

        review-r2 fix 3: also returns the per-component scoring dict so the
        adapter can populate `TracePrime.score_breakdown` from the same
        values the structured priming log emits — instead of leaving an
        empty dict on the Protocol surface.
        """
        if not last_message or not last_message.strip():
            return "", [], []

        # L1.5 Stage A: cue_mode rides on the retrieve.complete log so live-analysis
        # joins don't need to stitch the memory.window.shadow event.
        cue_mode = "windowed" if (window_cues is not None and window_cues.is_windowed) else "last_msg"

        t0 = time.time()
        # Weights first: a failed weights read recovers (rolls back) the session, which
        # would expire candidate rows loaded before it and break scoring them.
        weights = self._weights_override if self._weights_override is not None else await self._get_cached_weights()
        candidates, paths = await self._collect_candidates(last_message, limit, window_cues=window_cues)
        if not candidates:
            # emit_log=False for internal read-only primes (L1.5 shadow counterfactual +
            # Stage C re-primes) — otherwise every one emits a duplicate retrieve.complete
            # that double-counts the live turn's retrieval.
            if emit_log:
                self._emit_retrieve_complete_log(
                    chat_ctx,
                    candidates_total=0,
                    returned=0,
                    primed=[],
                    duration_ms=int((time.time() - t0) * 1000),
                    cue_mode=cue_mode,
                )
            return "", [], []

        top = self._score_candidates(candidates, weights, limit)
        duration_ms = int((time.time() - t0) * 1000)

        if emit_log:
            primed_payload = self._build_primed_payload(top, paths)
            self._emit_retrieve_complete_log(
                chat_ctx,
                candidates_total=len(candidates),
                returned=len(top),
                primed=primed_payload,
                duration_ms=duration_ms,
                cue_mode=cue_mode,
            )

        top_ids = [t.id for t, _, _ in top]

        # Candidates are read snapshots: access-write recovery may expire the
        # session's rows without breaking either rendering or the returned top.
        formatted = _format_priming_context([(trace, score) for trace, score, _ in top])

        if record_access:
            try:
                await self.memory_svc.record_access_batch(top_ids)
            except Exception:
                logger.debug("Batch access recording failed", exc_info=True)
                await self._recover_session()

        return formatted, top_ids, top

    @staticmethod
    def _build_primed_payload(
        top: list[tuple],
        paths: dict[str, set[str]],
    ) -> list[dict]:
        """Render the per-trace `primed` entries for the structured log.

        review-r1 fix 8: every leaf float passes through `_finite()` so
        NaN/Infinity (possible when cosine distance hits a zero-magnitude
        embedding) never lands in the JSON log. Strict downstream sinks
        (CloudWatch, Langfuse, OpenSearch) reject or silently drop
        non-standard JSON tokens; emitting 0.0 instead is the safe fallback.
        """
        out: list[dict] = []
        for rank, (trace, score, components) in enumerate(top, start=1):
            tid = str(trace.id)
            sb = {
                # M1 active components
                "semantic": _finite(components.get("semantic", 0.0)),
                "graph": _finite(components.get("graph", 0.0)),
                "recency": _finite(components.get("recency", 0.0)),
                "access": _finite(components.get("access", 0.0)),
                "arousal": _finite(components.get("arousal", 0.0)),
                # A2 — salience is now a scored component (weight defaults 0.0 → inert).
                "salience": _finite(components.get("salience", 0.0)),
                # #273 — intentions score here instead of forging semantic/graph 1.0.
                "intention": _finite(components.get("intention", 0.0)),
                # Raw observed salience kept alongside the scored value for analysis.
                "salience_observed": _finite(getattr(trace, "salience", 0.0) or 0.0),
                # M3 placeholders (mechanism not yet active)
                "memory_type_bias": 0.0,
                "anaphoric_recency_multiplier": 1.0,
                "diversity_penalty": 0.0,
                "inhibition_penalty": 0.0,
            }
            out.append(
                {
                    "trace_id": tid,
                    "memory_type": getattr(trace, "memory_type", None),
                    "rank": rank,
                    "score": _finite(score),
                    "score_breakdown": sb,
                    "path_attribution": sorted(paths.get(tid, set())),
                }
            )
        return out

    @staticmethod
    def _emit_retrieve_complete_log(
        chat_ctx: Any,
        *,
        candidates_total: int,
        returned: int,
        primed: list[dict],
        duration_ms: int,
        cue_mode: str = "last_msg",
    ) -> None:
        """Emit memory.retrieve.complete per backend-docs/ai-workforce/observability/memory-v2-telemetry.md.

        Hotfix (2026-05-01): the project's logging.basicConfig uses Python's
        default formatter (`%(levelname)s:%(name)s:%(message)s`), which
        silently drops `extra={}` keys — the 13 fields specified in the
        telemetry contract were never reaching CloudWatch. Inline-serialize
        the payload as the message body so consumers see the full JSON
        without changing the project-wide logging config.
        """
        chat_id = getattr(chat_ctx, "chat_id", None) if chat_ctx is not None else None
        topic_id = getattr(chat_ctx, "topic_id", None) if chat_ctx is not None else None
        venue = getattr(chat_ctx, "venue", None) if chat_ctx is not None else None
        payload = {
            "event_type": "memory.retrieve.complete",
            "timestamp": utc_now().isoformat(),
            "chat_id": chat_id,
            "topic_id": topic_id,
            "venue": venue,
            "intent": None,  # M1 has no query-type routing
            # L1.5 Stage A: "last_msg" (today's single cue) or "windowed" (segmented
            # window cues fed Path 1 + Paths 2/3). Lets live analysis distinguish the
            # cue construction without joining the memory.window.shadow event.
            "cue_mode": cue_mode,
            "candidates_total": candidates_total,
            "returned": returned,
            "duration_ms": duration_ms,
            "score_floor_applied": False,  # M1 has no quality gate
            "primed": primed,
        }
        logger.info("memory.retrieve.complete %s", json.dumps(payload))

    async def _collect_candidates(
        self, last_message: str, limit: int, window_cues: WindowCues | None = None
    ) -> tuple[dict[str, tuple[MemoryTrace, dict[str, float]]], dict[str, set[str]]]:
        """Run the 4 search paths sequentially and merge candidates.

        Returns ({trace_id: (trace, components)}, {trace_id: paths_set}). The
        path-attribution map records every retrieval path that surfaced each
        trace (semantic / entity_link / graph / intention) for the structured
        memory.retrieve.complete log.

        AsyncSession is NOT safe for concurrent operations, so we call each
        path sequentially instead of using asyncio.gather(). Semantic search
        is wrapped in asyncio.wait_for() with ``_SEMANTIC_TIMEOUT_S``. Each
        path that touches the DB recovers the session on failure (see
        ``_recover_session``) so a poisoned transaction can't cascade.

        ``window_cues`` (L1.5 Stage A LIVE) routes to the windowed collector; ``None``
        keeps this exact single-message path (byte-identical to today).
        """
        if window_cues is not None:
            return await self._collect_candidates_windowed(window_cues, limit)
        # Skip entity extraction on very short messages — regex is wasted work
        entity_names = extract_entity_names(last_message) if len(last_message) >= 5 else []
        entities = []
        if entity_names:
            try:
                entities = await self.memory_svc.find_entities_by_names(entity_names)
            except Exception:
                if self.observation is not None:
                    self.observation.issue("failed")
                logger.warning("Entity lookup failed", exc_info=True)
                await self._recover_session()
        elif len(last_message) >= 5:
            # v2a §9 Task 6b: lowercase fallback. ~59% of priming events
            # have no capitalized words; without this fallback, lowercase
            # mentions of known entities (e.g. "луна") miss Path 2/3/4.
            tokens = _extract_lowercase_tokens(last_message)
            if tokens:
                try:
                    entities = await self.memory_svc.find_entities_by_lowercase_tokens(tokens)
                except Exception:
                    if self.observation is not None:
                        self.observation.issue("failed")
                    logger.warning("Lowercase entity fallback failed", exc_info=True)
                    await self._recover_session()
        entity_ids = [e.id for e in entities]

        all_traces: dict[str, tuple[MemoryTrace, dict[str, float]]] = {}
        paths: dict[str, set[str]] = {}

        # Embedding query: header-only for events, char-capped for everything
        # else. ``last_message`` is preserved for the other 3 paths (entity
        # extraction, graph traversal, intention search) which work fine on
        # the full content. Only the semantic path benefits from a focused
        # query — and only the semantic path makes the embed call that can
        # blow the OpenAI ceiling.
        embedding_query = _derive_embedding_query(last_message)

        # Path 1: Semantic search (with timeout)
        global _semantic_ok, _semantic_timeout, _semantic_error
        try:
            semantic_res = await asyncio.wait_for(
                self.memory_svc.search_traces_raw(embedding_query, limit=limit * 2),
                timeout=_SEMANTIC_TIMEOUT_S,
            )
            for trace, similarity in semantic_res:
                tid = str(trace.id)
                if tid not in all_traces:
                    all_traces[tid] = (_snapshot_trace(trace), {})
                all_traces[tid][1]["semantic"] = similarity
                paths.setdefault(tid, set()).add("semantic")
            _semantic_ok += 1
        except TimeoutError:
            if self.observation is not None:
                self.observation.issue("timeout")
            _semantic_timeout += 1
            total = _semantic_ok + _semantic_timeout + _semantic_error
            pct = _semantic_timeout / total * 100 if total else 0
            logger.warning(
                "Semantic search timed out (%.1fs) — timeout_rate=%.1f%% (%d/%d)",
                _SEMANTIC_TIMEOUT_S,
                pct,
                _semantic_timeout,
                total,
            )
            # asyncio.wait_for cancels the inner coroutine when the timeout
            # fires, but asyncpg's cancel-in-flight leaves the SQLAlchemy
            # session with an unrolled-back transaction. Without recovery
            # here, Paths 2-4 below (entity-linked, graph, intentions) and
            # downstream prompt_builder loaders all reuse this session and
            # fail with PendingRollbackError (8s2b cascade) — one
            # user-visible chat outage per occurrence.
            await self._recover_session()
        except Exception:
            if self.observation is not None:
                self.observation.issue("failed")
            _semantic_error += 1
            logger.warning("Semantic search failed in priming", exc_info=True)
            # Same poisoned-session concern as the TimeoutError branch.
            await self._recover_session()

        # Path 2: Direct entity links score a full 1.0 on the graph component
        if entity_ids:
            try:
                linked_res = await self.memory_svc.get_entity_linked_traces(entity_ids, limit=limit)
                for trace in linked_res:
                    tid = str(trace.id)
                    if tid not in all_traces:
                        all_traces[tid] = (_snapshot_trace(trace), {})
                    all_traces[tid][1]["graph"] = 1.0
                    paths.setdefault(tid, set()).add("entity_link")
            except Exception:
                if self.observation is not None:
                    self.observation.issue("failed")
                logger.warning("Entity-linked search failed in priming", exc_info=True)
                await self._recover_session()

        # Path 3: Graph traversal
        if entity_ids:
            try:
                graph_res = await self.memory_svc.traverse_entity_graph(entity_ids, max_depth=3)
                if graph_res:
                    reached_ids = [eid for eid, _ in graph_res]
                    energy_map = dict(graph_res)
                    try:
                        trace_entity_pairs = await self.memory_svc.get_entity_linked_traces_with_entity(
                            reached_ids, limit=limit
                        )
                        for trace, linked_entity_id in trace_entity_pairs:
                            tid = str(trace.id)
                            if tid not in all_traces:
                                all_traces[tid] = (_snapshot_trace(trace), {})
                            trace_energy = energy_map.get(linked_entity_id, 0)
                            existing = all_traces[tid][1].get("graph", 0)
                            all_traces[tid][1]["graph"] = max(existing, trace_energy)
                            paths.setdefault(tid, set()).add("graph")
                    except Exception:
                        if self.observation is not None:
                            self.observation.issue("failed")
                        logger.warning("Graph trace fetch failed in priming", exc_info=True)
                        await self._recover_session()
            except Exception:
                if self.observation is not None:
                    self.observation.issue("failed")
                logger.warning("Graph traversal failed in priming", exc_info=True)
                await self._recover_session()

        # Path 4: Intentions — prospective memories that should surface strongly.
        #
        # #273: this path used to write `semantic = 1.0` and `graph = 1.0` onto
        # every intention trace. Both were forgeries: an intention is not
        # semantically perfect and not graph-perfect, it merely ARRIVED via a
        # different path. Those two fake 1.0s bought 0.60 of unearned composite
        # score and crowded genuinely-answering traces out of the top-8 (48/70
        # → 56/70 golds@8 on the frozen pool when stripped).
        #
        # Intentions now score on their OWN component, weighted by
        # `intention` (static — see `_INTENTION_WEIGHT`). `semantic`/`graph`
        # are left untouched: whatever Paths 1-3 legitimately earned for this
        # trace stands, and if those paths never saw it, the components stay
        # absent (⇒ 0.0) rather than being invented here. The invariant this
        # restores: a component score means "scored X on signal S", never
        # "arrived via path P".
        if entity_ids:
            try:
                intention_res = await self.memory_svc.get_intention_traces(entity_ids, limit=3)
                for trace in intention_res:
                    tid = str(trace.id)
                    if tid not in all_traces:
                        all_traces[tid] = (_snapshot_trace(trace), {})
                    all_traces[tid][1]["intention"] = 1.0
                    paths.setdefault(tid, set()).add("intention")
            except Exception:
                if self.observation is not None:
                    self.observation.issue("failed")
                logger.warning("Intention search failed in priming", exc_info=True)
                await self._recover_session()

        return all_traces, paths

    async def _collect_candidates_windowed(
        self, window_cues: WindowCues, limit: int
    ) -> tuple[dict[str, tuple[MemoryTrace, dict[str, float]]], dict[str, set[str]]]:
        """L1.5 Stage A LIVE collection over a segmented, recency-weighted window (§A1).

        - **Path 1** (semantic): one pgvector search per segment on its pre-computed
          embedding; each candidate's ``semantic`` = weighted-MAX over segments of
          ``segment_weight * cosine_sim`` (weighted-max, not sum — a faint distant cue
          can't accumulate to dominate; the #241-signature we watch). The newest segment
          at weight 1.0 reproduces today's semantic for newest-cued traces; older segments
          only ADD faintly-weighted candidates.
        - **Paths 2/3** (entity/graph): the per-segment ENTITY UNION (same
          ``extract_entity_names`` + ``_extract_lowercase_tokens`` as the single-message
          path), each entity carrying its BEST segment weight. Path 2 (direct link) scores
          ``graph`` = that entity weight; Path 3 (traversal) scales reached-trace energy by
          the max seed weight (v1 approximation — traversal pools its seeds, losing
          per-source attribution; exact per-source scaling is a follow-up).
        - **Path 4** (intentions): union seed, ``intention`` = 1.0 (prospective memories
          are not dampened by a faint old-entity weight — the union INCLUSION is the win).

        Scoring is unchanged (``_score_candidates``); only the semantic/graph component
        values differ. Read-side only; the same ``_recover_session`` discipline applies.
        """
        all_traces: dict[str, tuple[MemoryTrace, dict[str, float]]] = {}
        paths: dict[str, set[str]] = {}

        # ── Path 1: per-segment semantic, merged weighted-max ────────────────
        # Bound the search count + total wall-clock (CR): highest-weight segments first,
        # cap the number, and stop once the shared budget is spent. Low-weight tail
        # segments barely move a weighted-max score anyway.
        global _semantic_ok, _semantic_timeout, _semantic_error
        search_segments = sorted(window_cues.segments, key=lambda s: s.weight, reverse=True)[
            :_WINDOWED_MAX_SEARCH_SEGMENTS
        ]
        global _windowed_budget_skipped
        deadline = time.monotonic() + _WINDOWED_TOTAL_BUDGET_S
        for index, seg in enumerate(search_segments):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                skipped = len(search_segments) - index
                _windowed_budget_skipped += skipped
                if self.observation is not None:
                    self.observation.issue("timeout")
                logger.warning(
                    "Windowed Path 1 total budget (%.1fs) exhausted; skipped_segments=%d",
                    _WINDOWED_TOTAL_BUDGET_S,
                    skipped,
                )
                break
            timeout = min(_SEMANTIC_TIMEOUT_S, remaining)
            try:
                seg_rows = await asyncio.wait_for(
                    self.memory_svc.search_traces_by_vector(seg.embedding, limit=limit * 2),
                    timeout=timeout,
                )
                _semantic_ok += 1
            except TimeoutError:
                if self.observation is not None:
                    self.observation.issue("timeout")
                _semantic_timeout += 1
                logger.warning("Windowed semantic search timed out (%.1fs)", timeout)
                await self._recover_session()
                continue
            except Exception:
                if self.observation is not None:
                    self.observation.issue("failed")
                _semantic_error += 1
                logger.warning("Windowed semantic search failed", exc_info=True)
                await self._recover_session()
                continue
            for trace, similarity in seg_rows:
                tid = str(trace.id)
                if tid not in all_traces:
                    all_traces[tid] = (_snapshot_trace(trace), {})
                score = seg.weight * float(similarity)
                if score > all_traces[tid][1].get("semantic", 0.0):
                    all_traces[tid][1]["semantic"] = score
                paths.setdefault(tid, set()).add("semantic")

        # ── Entity union across segments, tracking each entity's best weight ─
        entity_weight: dict[Any, float] = {}
        for seg in window_cues.segments:
            if len(seg.text) < 5:
                continue
            names = extract_entity_names(seg.text)
            seg_entities: list = []
            if names:
                try:
                    seg_entities = await self.memory_svc.find_entities_by_names(names)
                except Exception:
                    if self.observation is not None:
                        self.observation.issue("failed")
                    logger.warning("Windowed entity lookup failed", exc_info=True)
                    await self._recover_session()
            else:
                tokens = _extract_lowercase_tokens(seg.text)
                if tokens:
                    try:
                        seg_entities = await self.memory_svc.find_entities_by_lowercase_tokens(tokens)
                    except Exception:
                        if self.observation is not None:
                            self.observation.issue("failed")
                        logger.warning("Windowed lowercase entity fallback failed", exc_info=True)
                        await self._recover_session()
            for e in seg_entities:
                if e.id not in entity_weight or seg.weight > entity_weight[e.id]:
                    entity_weight[e.id] = seg.weight
        entity_ids = list(entity_weight.keys())
        max_seed_weight = max(entity_weight.values(), default=0.0)

        # ── Path 2: direct entity links, graph = entity's best segment weight ─
        if entity_ids:
            try:
                linked_pairs = await self.memory_svc.get_entity_linked_traces_with_entity(entity_ids, limit=limit)
                for trace, linked_entity_id in linked_pairs:
                    tid = str(trace.id)
                    if tid not in all_traces:
                        all_traces[tid] = (_snapshot_trace(trace), {})
                    w = entity_weight.get(linked_entity_id, 0.0)
                    all_traces[tid][1]["graph"] = max(all_traces[tid][1].get("graph", 0.0), w)
                    paths.setdefault(tid, set()).add("entity_link")
            except Exception:
                if self.observation is not None:
                    self.observation.issue("failed")
                logger.warning("Windowed entity-linked search failed", exc_info=True)
                await self._recover_session()

        # ── Path 3: graph traversal, energy scaled by max seed weight ────────
        if entity_ids:
            try:
                graph_res = await self.memory_svc.traverse_entity_graph(entity_ids, max_depth=3)
                if graph_res:
                    reached_ids = [eid for eid, _ in graph_res]
                    energy_map = dict(graph_res)
                    try:
                        trace_entity_pairs = await self.memory_svc.get_entity_linked_traces_with_entity(
                            reached_ids, limit=limit
                        )
                        for trace, linked_entity_id in trace_entity_pairs:
                            tid = str(trace.id)
                            if tid not in all_traces:
                                all_traces[tid] = (_snapshot_trace(trace), {})
                            trace_energy = energy_map.get(linked_entity_id, 0) * max_seed_weight
                            existing = all_traces[tid][1].get("graph", 0)
                            all_traces[tid][1]["graph"] = max(existing, trace_energy)
                            paths.setdefault(tid, set()).add("graph")
                    except Exception:
                        if self.observation is not None:
                            self.observation.issue("failed")
                        logger.warning("Windowed graph trace fetch failed", exc_info=True)
                        await self._recover_session()
            except Exception:
                if self.observation is not None:
                    self.observation.issue("failed")
                logger.warning("Windowed graph traversal failed", exc_info=True)
                await self._recover_session()

        # ── Path 4: intentions (union seed, undampened) ──────────────────────
        if entity_ids:
            try:
                intention_res = await self.memory_svc.get_intention_traces(entity_ids, limit=3)
                for trace in intention_res:
                    tid = str(trace.id)
                    if tid not in all_traces:
                        all_traces[tid] = (_snapshot_trace(trace), {})
                    all_traces[tid][1]["intention"] = 1.0
                    paths.setdefault(tid, set()).add("intention")
            except Exception:
                if self.observation is not None:
                    self.observation.issue("failed")
                logger.warning("Windowed intention search failed", exc_info=True)
                await self._recover_session()

        return all_traces, paths

    def _score_candidates(
        self,
        candidates: dict[str, tuple[MemoryTrace, dict[str, float]]],
        weights: dict[str, float],
        limit: int,
        mechanism_config: MechanismConfig | None = None,
    ) -> list[tuple[MemoryTrace, float, dict[str, float]]]:
        """Apply 5-component weighted scoring, then the M3 re-ranker slot, and
        return top-N.

        Returns (trace, score, full_components_dict). The components dict
        includes the per-component values populated for the structured
        memory.retrieve.complete log payload.

        ``mechanism_config`` gates the A/B/C/D re-rankers (Sub-design C). It
        defaults to an inert config and INC 2 registers no mechanisms, so the
        slot is the identity function and output is byte-identical to the bare
        5-component scorer. The harness/self-tests pass a non-inert config.
        """
        now = utc_now()
        scored: list[tuple[MemoryTrace, float, dict[str, float]]] = []
        for _tid, (trace, components) in candidates.items():
            recency = _compute_recency_score(trace.created_at, now)
            access = _compute_access_score(trace.access_count)
            arousal = trace.emotion_arousal or 0

            full_components = dict(components)
            full_components["recency"] = recency
            full_components["access"] = access
            full_components["arousal"] = arousal
            full_components["salience"] = _compute_salience_score(trace)  # A2 — weight defaults 0.0
            score = weighted_component_score(full_components, weights)
            scored.append((trace, score, full_components))

        scored.sort(key=lambda x: x[1], reverse=True)
        # M3 re-ranker slot (Sub-design C). Identity when inert / no mechanisms
        # registered (INC 2), so this is byte-identical to scored[:limit].
        ranked = apply_mechanisms(scored, mechanism_config or MechanismConfig())
        return ranked[:limit]

    async def _get_cached_weights(self) -> dict[str, float]:
        """Return retrieval weights, cached for _WEIGHTS_CACHE_TTL seconds.

        Under a memory client the cache is that client's (its store's weights);
        otherwise it is the process-wide ``_WEIGHTS_CACHE``.
        """
        global _WEIGHTS_CACHE
        owner = current_owner()
        entry = _WEIGHTS_CACHE if owner is None else owner.caches.get(_WEIGHTS_CACHE_NAME)
        now_ts = time.time()
        if entry is not None:
            cached_at, cached = entry
            if now_ts - cached_at < _WEIGHTS_CACHE_TTL:
                return cached
        try:
            weights = await self.memory_svc.get_retrieval_weights()
        except Exception:
            logger.warning("Failed to load retrieval weights, using defaults", exc_info=True)
            # A deferred-access prime (DB-839) runs no access write afterwards whose own
            # recovery would clean this session, and later readers share it. Recover here,
            # and don't cache the fallback, so the next prime retries the DB.
            await self._recover_session()
            return dict(DEFAULT_RETRIEVAL_WEIGHTS)
        if owner is None:
            _WEIGHTS_CACHE = (now_ts, weights)
        else:
            owner.caches[_WEIGHTS_CACHE_NAME] = (now_ts, weights)
        return weights


def invalidate_weights_cache() -> None:
    """Clear the retrieval weights cache (the current memory client's, or the process one)."""
    global _WEIGHTS_CACHE
    owner = current_owner()
    if owner is None:
        _WEIGHTS_CACHE = None
    else:
        owner.caches.pop(_WEIGHTS_CACHE_NAME, None)


# Cyrillic chars in the character class are intentional; ambiguity warning suppressed.
_LOWERCASE_TOKEN_RE = re.compile(r"[A-Za-zА-яёЁ]{3,}")


def _finite(value: float | int | None) -> float:
    """Coerce non-finite floats (NaN / +Inf / -Inf / None) to 0.0.

    review-r1 fix 8: keeps the structured `memory.retrieve.complete` log
    JSON-strict — Python's default `json.dumps(allow_nan=True)` writes
    `NaN` / `Infinity` tokens that strict sinks (CloudWatch, Langfuse,
    OpenSearch) reject or silently truncate.
    """
    if value is None:
        return 0.0
    f = float(value)
    return f if math.isfinite(f) else 0.0


def _extract_lowercase_tokens(message: str) -> list[str]:
    """v2a §9 Task 6b: tokenize the message into ≥3-char alphabetic words.

    Lowercased, deduplicated, order-preserving. Stop-words dropped.
    """
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in _LOWERCASE_TOKEN_RE.findall(message):
        token = raw.lower()
        if token in seen:
            continue
        # Filter both English-style stopwords (case-folded against _STOP_WORDS,
        # which is title-cased) AND Cyrillic function words from romanization's
        # canonical set (lowercase). Without the latter, "или / она / они /
        # его / тот / наш / ..." pass through and trigger spurious DB hits on
        # every Russian message — review-r1 finding.
        if raw[:1].upper() + raw[1:].lower() in _STOP_WORDS or raw.title() in _STOP_WORDS:
            continue
        if token in CYRILLIC_STOPWORDS:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def extract_entity_names(text: str) -> list[str]:
    """Extract likely entity names from text using capitalization heuristics."""
    matches = _ENTITY_NAME_RE.findall(text)
    # Deduplicate, preserve order, filter stop words
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        # Skip common sentence starters (check first word for multi-word matches)
        first_word = m.split()[0] if " " in m else m
        if first_word in _STOP_WORDS:
            # For multi-word like "The Garden", skip. For single word like "Where", skip.
            if " " not in m:
                continue
            # Multi-word: strip the stop word prefix (e.g., "The Maria" is unlikely but handle)
            remainder = m[len(first_word) :].strip()
            if remainder and remainder not in seen:
                seen.add(remainder)
                result.append(remainder)
            continue
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


def _compute_recency_score(created_at: datetime, now: datetime) -> float:
    """Exponential decay: 1.0 today, ~0.5 after 7 days, ~0.1 after 30 days."""
    if created_at.tzinfo is None:
        return 0.5  # fallback for naive datetimes
    age_days = max(0, (now - created_at).total_seconds() / 86400)
    decay_rate = math.log(2) / _RECENCY_HALF_LIFE_DAYS
    return math.exp(-decay_rate * age_days)


def _compute_access_score(access_count: int) -> float:
    """Logarithmic normalization: log(1 + count) / log(1 + max_expected)."""
    if access_count <= 0:
        return 0
    return math.log(1 + access_count) / math.log(1 + _MAX_EXPECTED_ACCESS)


def _compute_salience_score(trace: MemoryTrace) -> float:
    """A2 — the trace's CURRENT (decayed) salience as a 0–1 priming component.

    Salience already bakes in significance-gated Ebbinghaus decay + access
    reinforcement (run nightly by ``consolidation._apply_decay``). Clamped to [0, 1]
    and NaN/None-coerced so a stray value can't distort the weighted sum. Only
    matters when ``weight_salience`` > 0 (it defaults 0.0 → inert)."""
    s = _finite(getattr(trace, "salience", 0.0) or 0.0)
    return max(0.0, min(1.0, s))


# The base scorer components, in fixed order. Shared by the live scorer
# (_score_candidates) and the offline replay harness so the two cannot diverge
# (Sub-design C INC 2 — paired A/B replay over the frozen candidate pool).
# A2 added ``salience`` as the 6th component; its weight DEFAULTS 0.0, so the
# ``salience * 0.0`` term is exactly 0 and the score stays bit-identical to the
# pre-salience 5-component scorer until the weight is tuned up.
SCORE_COMPONENTS = ("semantic", "graph", "recency", "access", "arousal", "salience", "intention")


def weighted_component_score(components: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted sum over the scorer components. Identical math for live scoring and
    frozen-pool replay. ``weights.get(k, 0.0)`` tolerates a weights dict that predates a
    component (e.g. a cached 5-key dict across a deploy) — a missing weight contributes 0.
    Adding ``salience`` at weight 0.0 is a no-op (``salience * 0.0 == 0``)."""
    return sum(float(components.get(k, 0.0)) * float(weights.get(k, 0.0)) for k in SCORE_COMPONENTS)


def _format_trace_entry(trace, now: datetime) -> str:
    """Format a single trace as a priming-block line, with relative-time label.

    v2a §6.2. Recent traces (younger than `temporal_relative_label_min_days`)
    are labeled only by memory_type. Older traces gain a `remembered Xd ago`
    suffix so the agent can disambiguate "current" from "stale" recall.

    Naive timestamps are treated as UTC (legacy rows). Future timestamps
    (clock skew) get no label — we don't lie about the past.
    """
    created = trace.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    age_days = (now - created).total_seconds() / 86400

    mtype = trace.memory_type
    base_prefix = f"[{mtype} — act on this]" if mtype == "intention" else f"[{mtype}]"

    if age_days < 0:
        return f"- {base_prefix} {trace.content}"

    min_days = cognitive_config.get("temporal_relative_label_min_days")
    if age_days < min_days:
        return f"- {base_prefix} {trace.content}"

    if age_days < 2:
        label = "remembered yesterday"
    elif age_days < 7:
        label = f"remembered {int(age_days)}d ago"
    elif age_days < 30:
        label = f"remembered {int(age_days / 7)}w ago"
    elif age_days < 365:
        label = f"remembered {int(age_days / 30)}mo ago"
    else:
        label = f"remembered {int(age_days / 365)}y ago"

    if mtype == "intention":
        prefix = f"[{mtype} — act on this, {label}]"
    else:
        prefix = f"[{mtype}, {label}]"
    return f"- {prefix} {trace.content}"


def _format_priming_context(results: list[tuple], now: datetime | None = None) -> str:
    """Format priming results for prompt injection.

    Content is already scrubbed at storage time in store_trace(); no double-scrub.
    `now` defaults to utc_now() so callers don't have to thread the clock.
    """
    if not results:
        return ""
    when = now if now is not None else utc_now()
    lines = ["## Active Associations"]
    for trace, _score in results:
        lines.append(_format_trace_entry(trace, when))
    return "\n".join(lines)
