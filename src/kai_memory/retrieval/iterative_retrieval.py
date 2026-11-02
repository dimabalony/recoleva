"""L1.5 iterative retrieval — window cues (Stage A) + thinking-cued re-priming (Stage C/C2).

Design of record: backend-docs/ai-workforce/plans/design/2026-07-21-l15-iterative-retrieval-design.md
(v1.2.x). Linear DB-327 (Stage A) + DB-329 (Stage C+C2).

Everything here ships behind two 3-state flags (``0=off / 1=shadow / 2=live``),
both DEFAULT 1=shadow, so at merge there is **no live behavior change**: Stage A
computes a windowed candidate diff and logs it (injects nothing); Stage C runs
the full trigger + retrieve + dedup and logs would-have-injected (injects nothing);
C2 logs would-have-revised (never calls the model). The flip to live is the
operator's deliberate, reversible step.

Containment (design §Containment, the #241 + echo-chamber boundary) — all five
invariants live here or are enforced by callers:

1. No writes to ``retrieval_config`` / salience / significance / decay; every knob
   below is a versioned module constant, never a ``cognitive_config`` row.
2. (Stage B / buffer NOT built in this increment — deferred on churn data.)
3. Echo-chamber: injected lines carry the ``[recalled while thinking]`` provenance
   marker; re-prime retrieval is read-only (``record_access=False``); no C→L2 edge.
4. Bedrock asymmetry fail-open: Bedrock extracts no thinking, so ``consider()`` sees
   an empty thinking list and no-ops — today's behavior.
5. Injected content never enters the memory store: the caller appends the block
   INSIDE ``toolResult.content`` (invisible to ``ConversationMessage.text`` → invisible
   to L2), and L2 additionally skips any line carrying the marker (memory_extraction).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from kai_memory.host_calls import strip_echoed_metadata, strip_msg_meta_envelope, utc_now
from kai_memory import ports
from kai_memory.ownership import adopt_if_owned, owned_state
from kai_memory.store import cognitive_config
from kai_memory.retrieval.memory_observation import MemoryDecision, MemoryObservation

if TYPE_CHECKING:
    from kai_memory.ports import ChatContext

# Pinned to the pre-DB-842 module path: CloudWatch queries, feature-watch criteria and
# log filters match the logger name, which must not follow the module's move.
_LEGACY_LOGGER_NAME = "app.services.chat.iterative_retrieval"
logger = logging.getLogger(_LEGACY_LOGGER_NAME)

# ---------------------------------------------------------------------------
# Flags (3-state) — read from cognitive_config, seeded by migration 099.
# ---------------------------------------------------------------------------

FLAG_CONTEXT_CUES = "priming_context_cues"  # Stage A
FLAG_THINKING_CUES = "priming_thinking_cues"  # Stage C + C2

OFF, SHADOW, LIVE = 0, 1, 2
# Shared enum-membership sets (repo guideline: a frozenset in the module that owns the enum).
_VALID_STATES = frozenset({OFF, SHADOW, LIVE})
ACTIVE_STATES = frozenset({SHADOW, LIVE})  # flag is doing something (shadow logs / live injects)


def _flag_state(key: str) -> int:
    """Read a 3-state flag as an int (``0=off / 1=shadow / 2=live``).

    ``cognitive_config.value`` is a FLOAT column; round to the nearest state so a
    stored ``1.0`` reads as SHADOW. Anything outside {0,1,2} clamps to OFF as the
    safe default (never silently "live").
    """
    raw = round(cognitive_config.get(key))
    return raw if raw in _VALID_STATES else OFF


def context_cues_state() -> int:
    return _flag_state(FLAG_CONTEXT_CUES)


def thinking_cues_state() -> int:
    return _flag_state(FLAG_THINKING_CUES)


# ---------------------------------------------------------------------------
# Versioned constants (NOT cognitive_config rows — containment invariant #1).
# Conservative defaults; sweepable against the Tier-2 follow-up fixture set.
# ---------------------------------------------------------------------------

RECALL_MARKER = "[recalled while thinking]"

# Stage A — segment weighting (design §A1). Newest message anchors at 1.0; each
# older message decays geometrically to a floor; ~30 messages deep lands at floor.
_SEG_NEWEST_WEIGHT = 1.0
_SEG_DECAY = 0.6
_SEG_WEIGHT_FLOOR = 0.05
# A message longer than this is split into position-weighted segments (later parts
# brighter). This also fixes the pre-existing header-only transcript degradation:
# a long event/transcript message's CONTENT becomes a cue instead of the header only.
_LONG_MSG_CHARS = 4000
_MAX_LONG_SEGMENTS = 4
# Chars fed to the embedder per segment (priming-layer ceiling; hard byte truncation
# still applies inside generate_embedding).
_MAX_SEGMENT_CHARS = 2000
# Cap how deep into the window we build individual segments; older than this collapse
# into one floor-weighted chunk.
_RECENT_WINDOW = 8

# Stage C — novelty gate + relevance floors + caps (design §Stage C).
_NOVELTY_COSINE_MAX = 0.85  # thinking must be < this cosine vs the cue set to re-prime
_MID_TOOL_RELEVANCE_FLOOR = 0.45  # semantic-sim floor for a mid-tool injection
_REVISE_RELEVANCE_FLOOR = 0.60  # HIGHER floor for C2 (an extra model call must earn itself)
_MAX_REPRIME_ROUNDS = 3  # injection rounds per turn
_MAX_LINES_PER_INJECTION = 3  # Search-o1 condense lesson — terse beats broad
_MAX_LINES_PER_TURN = 9
_REPRIME_BUDGET_S = 2.0  # hard await budget; skip + log on timeout
_MAX_REVISES_PER_TURN = 1
_CUE_SNIPPET_CHARS = 120  # hard-truncated thinking-cue snippet on the reprime shadow log

# Marginal-cost cap: the per-turn cue embed cache (design §A2 — each segment embeds
# ONCE). Content-hash keyed, bounded LRU, shared across turns for the same content.
_EMBED_CACHE_MAX = 1024
_EMBED_CACHE: OrderedDict[str, list[float]] = OrderedDict()

# Turn-over-turn primed set, per (venue, chat_id, topic_id) — feeds ``primed_set_churn``
# (the Stage B GO/NO-GO gate; MUST ship). Bounded by pruning alongside the caller.
_PREV_PRIMED: dict[tuple[Any, Any, Any], set[str]] = {}
_PREV_PRIMED_MAX_KEYS = 512


def reset_state_for_testing() -> None:
    """Test-only — clear module caches so tests don't leak state into each other."""
    _EMBED_CACHE.clear()
    _PREV_PRIMED.clear()


# ---------------------------------------------------------------------------
# Embedding cache (A2)
# ---------------------------------------------------------------------------


async def _embed_cached(text: str) -> list[float] | None:
    """Embed ``text`` once, memoised by content hash (design §A2).

    Returns ``None`` on embed failure (fail-open: a missing cue just doesn't
    contribute — never raises into the tool loop). A repeated string within or
    across turns costs zero marginal embeds.
    """
    key = hashlib.sha1(text.encode("utf-8")).hexdigest()  # content cache key, not security
    cache = owned_state("cue_embeddings", OrderedDict, _EMBED_CACHE)  # a memory client's own embedder
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached
    try:
        from kai_memory.store.embeddings import generate_embedding

        emb = await generate_embedding(text)
    except Exception:
        logger.debug("cue embed failed (ignored)", exc_info=True)
        return None
    cache[key] = emb
    cache.move_to_end(key)
    while len(cache) > _EMBED_CACHE_MAX:
        cache.popitem(last=False)
    return emb


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors. 0.0 on degenerate input."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(dot / ((na**0.5) * (nb**0.5)))


# ---------------------------------------------------------------------------
# Stage A — window cue construction
# ---------------------------------------------------------------------------

_EVENT_CONTENT_SEPARATOR = "\n--- content ---\n"


def extract_window_user_texts(messages: list[dict[str, Any]] | None) -> list[str]:
    """Return the USER-message texts in the window, oldest→newest.

    Kai's assistant replies are permanently excluded (design §A1: they paraphrase
    already-surfaced memories; including them closes a self-echo loop). Tool-result
    user messages (content is a list of ``toolResult`` blocks, no plain text) yield
    no text and are skipped.
    """
    if not messages:
        return []
    out: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(b["text"] for b in content if isinstance(b, dict) and "text" in b)
        else:
            text = ""
        # A sidecar marks canonical content, which must be trusted as authored
        # even when the user is literally discussing an envelope. Only
        # pre-cutover messages without a sidecar need legacy stripping.
        if isinstance(msg.get(ports.MODEL_METADATA_KEY), dict):
            text = text.strip()
        else:
            text = strip_msg_meta_envelope(text).strip()
        if text:
            out.append(text)
    return out


def _segment_message(text: str, base_weight: float) -> list[tuple[str, float]]:
    """Split one message into (segment_text, weight) pairs.

    Short messages → a single segment at ``base_weight``. Long messages (and
    event/transcript messages) → position-weighted segments, later parts brighter
    (fixing the pre-existing header-only transcript degradation). Segment weights
    scale ``base_weight`` so an old long message still floors correctly.
    """
    # Event messages: embed BOTH header and content (today: header only). Split on
    # the content separator first so the header keeps a full segment.
    pieces: list[str]
    if text.startswith("[EVENT]") and _EVENT_CONTENT_SEPARATOR in text:
        header, _, body = text.partition(_EVENT_CONTENT_SEPARATOR)
        pieces = [header, *_chunk(body, _MAX_SEGMENT_CHARS)]
    elif len(text) > _LONG_MSG_CHARS:
        pieces = _chunk(text, _MAX_SEGMENT_CHARS)
    else:
        return [(text[:_MAX_SEGMENT_CHARS], base_weight)]

    pieces = [p for p in (s.strip() for s in pieces) if p][:_MAX_LONG_SEGMENTS]
    if not pieces:
        return []
    if len(pieces) == 1:
        return [(pieces[0][:_MAX_SEGMENT_CHARS], base_weight)]
    # Position ramp: earliest piece dimmest, last piece full base_weight.
    out: list[tuple[str, float]] = []
    n = len(pieces)
    for i, piece in enumerate(pieces):
        # ramp from _SEG_DECAY..1.0 across pieces
        ramp = _SEG_DECAY + (1.0 - _SEG_DECAY) * (i / (n - 1))
        out.append((piece[:_MAX_SEGMENT_CHARS], base_weight * ramp))
    return out


def _chunk(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def build_segment_cues(window_texts: list[str]) -> list[tuple[str, float]]:
    """Build the segmented, recency-weighted cue set (design §A1).

    Returns ``[(segment_text, weight), ...]``. The newest message anchors at 1.0
    (split into position segments when long); recent messages decay geometrically;
    messages older than ``_RECENT_WINDOW`` collapse into one floor-weighted chunk.

    INVARIANT: a single-message window (``len == 1``) returns exactly one segment
    at weight 1.0 — the byte-identical-to-today cue.
    """
    if not window_texts:
        return []
    segments: list[tuple[str, float]] = []
    n = len(window_texts)
    # newest = index n-1, weight 1.0 (possibly multi-segment)
    segments.extend(_segment_message(window_texts[-1], _SEG_NEWEST_WEIGHT))
    # recent individually decayed
    for dist in range(1, min(_RECENT_WINDOW, n)):
        idx = n - 1 - dist
        weight = max(_SEG_WEIGHT_FLOOR, _SEG_DECAY**dist)
        segments.append((window_texts[idx][:_MAX_SEGMENT_CHARS], weight))
    # older collapsed into a single floor chunk
    if n > _RECENT_WINDOW:
        older = window_texts[: n - _RECENT_WINDOW]
        joined = "\n".join(older)[-_MAX_SEGMENT_CHARS:]
        if joined.strip():
            segments.append((joined, _SEG_WEIGHT_FLOOR))
    return segments


@dataclass
class SegmentCue:
    """One window segment: its text, recency weight, and (pre-computed) embedding."""

    text: str
    weight: float
    embedding: list[float]


@dataclass
class WindowCues:
    """The turn's window cue set (design §A1), built ONCE before priming.

    ``segments`` are ordered newest-first (segments[0] anchors at weight 1.0). Each
    carries its embedding (A2 cache) so Path 1 does per-segment searches and Stage C
    reads them for the novelty gate without re-embedding.

    ``is_windowed`` is True only for a REAL multi-message window (≥2 user messages
    with Stage A shadow/live). A single-message turn yields ``is_windowed=False`` with
    one weight-1.0 segment built from ``_derive_embedding_query`` (header-only for
    events) — identical to today's single cue, so the LIVE priming path can safely
    fall back to today's behavior (the empty-window byte-identical invariant).
    """

    segments: list[SegmentCue] = field(default_factory=list)
    is_windowed: bool = False

    @property
    def embeddings(self) -> list[list[float]]:
        return [s.embedding for s in self.segments]


async def build_window_cues(
    messages: list[dict[str, Any]] | None,
    chat_ctx: ChatContext | None = None,
) -> WindowCues | None:
    """Build the per-turn :class:`WindowCues` (design §A1/A2), once, BEFORE priming.

    Returns ``None`` when both flags are OFF or there is no user text — the caller then
    uses today's exact single-message priming path (byte-identical). Otherwise returns
    a WindowCues whose segments are already embedded (A2 content-hash cache):

    - Stage A shadow/live AND ≥2 user messages ⇒ full segmented window (``is_windowed=True``).
    - Otherwise (single message, or Stage A off but Stage C on) ⇒ ONE weight-1.0 cue
      built from ``_derive_embedding_query(newest)`` (``is_windowed=False``) — the same
      embedding today's semantic path uses, so LIVE priming stays byte-identical and
      Stage C novelty still has a cue.

    Fail-open: any error returns ``None`` (today's behavior). Never raises.
    """
    try:
        a_state = context_cues_state()
        c_state = thinking_cues_state()
        if a_state == OFF and c_state == OFF:
            return None
        window_texts = extract_window_user_texts(messages)
        if not window_texts:
            return None

        if a_state in ACTIVE_STATES and len(window_texts) > 1:
            raw = build_segment_cues(window_texts)
            is_windowed = True
        else:
            # Single cue = exactly today's semantic query (header-only for events).
            from kai_memory.retrieval.memory_priming import _derive_embedding_query

            raw = [(_derive_embedding_query(window_texts[-1]), _SEG_NEWEST_WEIGHT)]
            is_windowed = False

        segments: list[SegmentCue] = []
        for text, weight in raw:
            emb = await _embed_cached(text)
            if emb is not None:
                segments.append(SegmentCue(text=text, weight=weight, embedding=emb))
        if not segments:
            return None
        # A single surviving segment can't be a real "window" — demote so LIVE priming
        # falls back to today's path (never a 1-segment windowed collection).
        if len(segments) < 2:
            is_windowed = False
        return WindowCues(segments=segments, is_windowed=is_windowed)
    except Exception:
        logger.warning("build_window_cues failed; falling back to last-message priming", exc_info=True)
        return None


def _churn(chat_ctx: ChatContext | None, primed_ids: list[uuid.UUID]) -> float | None:
    """``primed_set_churn`` — fraction of the PREVIOUS turn's primed traces that
    dropped this turn (design §Evals; the Stage B GO/NO-GO gate — MUST ship).

    Returns ``None`` on the first turn of a conversation (no baseline). Updates the
    per-key baseline as a side effect. Bounded to ``_PREV_PRIMED_MAX_KEYS``.
    """
    if chat_ctx is None:
        return None
    key = (chat_ctx.venue, chat_ctx.chat_id, chat_ctx.topic_id)
    now_set = {str(i) for i in primed_ids}
    previous = owned_state("previous_primed", dict, _PREV_PRIMED)
    prev = previous.get(key)
    previous[key] = now_set
    if len(previous) > _PREV_PRIMED_MAX_KEYS:
        # drop an arbitrary oldest-ish key (dict preserves insertion order)
        previous.pop(next(iter(previous)), None)
    if not prev:
        return None
    dropped = len(prev - now_set)
    return dropped / len(prev)


# ---------------------------------------------------------------------------
# The per-turn reprise state — Stage C + C2, created once after priming.
# ---------------------------------------------------------------------------


class RepriseState:
    """Per-turn iterative-retrieval state (Stage C + C2).

    Created once per turn (after L1 priming) by :func:`build_reprise_state`. Holds:
    the turn's cue embeddings (for the novelty gate), the already-primed trace ids
    (dedup base), round/line caps, and the ``reprime.*`` counters. Mutates nothing
    shared — the in-flight re-prime task operates on its OWN session and returns
    trace UUIDs + formatted lines only.

    When both flags are OFF this is a no-op shell: ``consider``/``collect``/``cancel``
    /``maybe_revise`` all short-circuit, preserving byte-identical behavior.
    """

    def __init__(
        self,
        chat_ctx: ChatContext | None,
        cue_embeddings: list[list[float]],
        primed_trace_ids: list[uuid.UUID],
        thinking_state: int,
        *,
        observation: MemoryObservation | None = None,
        inactive_reason: str = "disabled",
    ) -> None:
        self.chat_ctx = chat_ctx
        self._observation = observation
        self._inactive_reason = inactive_reason
        self._decision: MemoryDecision | None = None
        self._retrieved_ids: list[str] = []
        self._retrieval_outcome = "pending"
        self._retrieval_tasks: set[asyncio.Task] = set()
        self._considered = False
        self._cue_embeddings = cue_embeddings
        self._seen_ids: set[str] = {str(i) for i in primed_trace_ids}
        self._state = thinking_state  # OFF / SHADOW / LIVE
        self._task: asyncio.Task | None = None
        self._pending_cue = ""  # thinking text of the in-flight re-prime (for the shadow log)
        self._injections = 0
        self._lines_used = 0
        self._revises = 0
        self.counters: dict[str, int] = {
            "fired": 0,
            "skipped_gate": 0,
            "skipped_budget": 0,
            "cancelled": 0,
            "noop": 0,
            "suppressed_beyond_cap": 0,
            "would_revise": 0,
            "revised": 0,
        }

    @property
    def active(self) -> bool:
        return self._state in ACTIVE_STATES

    # -- Stage C: kick / collect / cancel -------------------------------------

    def consider(
        self, thinking_blocks: list[str] | None, stop_reason: str | None, *, call_id: str | None = None
    ) -> None:
        """Consider a re-prime after a thinking block (design §Stage C trigger).

        Kicks an asyncio task overlapped with tool execution when:
        - the flag is shadow/live,
        - there IS thinking (Bedrock extracts none → no-op, fail-open),
        - ``stop_reason == "tool_use"`` (a final response never receives a mid-tool
          injection — C2 handles finals separately),
        - the round cap is not yet hit.

        The novelty gate (cosine < 0.85 vs the cue set) runs INSIDE the task so this
        stays cheap and synchronous. Only ONE task is in flight at a time (each
        iteration awaits :meth:`collect` before the next ``consider``).
        """
        self._considered = True
        decision = None
        if self._observation is not None:
            decision = self._observation.begin("reprise", trigger="reasoning_summary")
            decision.mode = {OFF: "off", SHADOW: "shadow", LIVE: "live"}.get(self._state, "off")
            decision.call_id = call_id
        if not self.active or stop_reason != "tool_use":
            if decision is not None:
                decision.complete(self._inactive_reason if not self.active else "no_injection_point")
            return
        text = _join_thinking(thinking_blocks)
        if not text:
            if decision is not None:
                decision.complete("no_summary")
            return
        if self._injections >= _MAX_REPRIME_ROUNDS or self._lines_used >= _MAX_LINES_PER_TURN:
            self.counters["suppressed_beyond_cap"] += 1
            if decision is not None:
                decision.complete("capped")
            self._emit_reprime_log("suppressed_beyond_cap", 0, 0, cue_snippet=text)
            return
        if (self._task is not None and not self._task.done()) or (
            self._observation is not None and any(not owned.done() for owned in self._retrieval_tasks)
        ):
            # A native timed-out waiter may return before its worker finishes
            # cleanup. Preserve its decision/result fields until true settlement.
            if decision is not None:
                decision.complete("busy")
            return
        self._pending_cue = text
        self._decision = decision
        self._retrieved_ids = []
        self._retrieval_outcome = "pending"
        self._task = asyncio.create_task(self._observed_reprime(text, decision))
        adopt_if_owned(self._task)
        self._retrieval_tasks.add(self._task)
        self._task.add_done_callback(self._retrieval_tasks.discard)
        self._task.add_done_callback(_swallow_task_result)

    async def _observed_reprime(
        self, text: str, decision: MemoryDecision | None
    ) -> list[tuple[str, str, float]] | None:
        try:
            found = await self._reprime(text)
            self._retrieval_outcome = "novelty_gated" if found is None else "retrieved" if found else "empty"
            if decision is not None and decision.issues.keys() & {"failed", "timeout", "embedding_failed"}:
                self._retrieval_outcome = (
                    "partial" if self._retrieved_ids else ("timeout" if "timeout" in decision.issues else "failed")
                )
            return found
        except asyncio.CancelledError:
            self._retrieval_outcome = "cancelled"
            raise
        except Exception:
            self._retrieval_outcome = "failed"
            raise
        finally:
            if decision is not None:
                decision.retrieval_finished(retrieved=self._retrieved_ids)

    async def _reprime(self, thinking_text: str) -> list[tuple[str, str, float]] | None:
        """Novelty-gate, then read-only retrieve on the thinking cue.

        Returns ``[(trace_id, one_line, relevance), ...]`` (deduped vs the seen set and
        floored), ``[]`` when it retrieved but nothing new survived, or ``None`` when the
        NOVELTY GATE filtered it (thinking not novel vs the cue set). collect() maps those
        three to the ``fired`` / ``noop`` / ``skipped_gate`` funnel outcomes — keeping them
        distinct so the shadow logs show WHY a consideration didn't fire, not just that it
        didn't. Runs on its OWN DB session; NEVER records access (design §Stage C).
        """
        emb = await _embed_cached(thinking_text)
        if emb is None:
            if self._decision is not None:
                self._decision.issue("embedding_failed")
            return None  # can't judge novelty without an embedding → treat as gated
        # Novelty gate — thinking must be genuinely new vs the cue set.
        novelty = max((_cosine(emb, c) for c in self._cue_embeddings), default=0.0)
        if novelty >= _NOVELTY_COSINE_MAX:
            return None  # collect() logs skipped_gate
        try:
            from kai_memory.ports import session_scope
            from kai_memory.retrieval.memory_priming import MemoryPrimingService

            async with session_scope() as s:
                svc = (
                    MemoryPrimingService(s, observation=self._decision)
                    if self._decision is not None
                    else MemoryPrimingService(s)
                )
                # Read-only: record_access=False (design §Stage C — mandatory).
                _fmt, _ids, top = await svc.prime_with_traces(
                    thinking_text, chat_ctx=self.chat_ctx, record_access=False, emit_log=False
                )
                self._retrieved_ids = [str(trace.id) for trace, _, _ in top]
                out: list[tuple[str, str, float]] = []
                for trace, _score, components in top:
                    tid = str(trace.id)
                    if tid in self._seen_ids:
                        continue
                    relevance = float(components.get("semantic", 0.0) or 0.0)
                    if relevance < _MID_TOOL_RELEVANCE_FLOOR:
                        continue
                    out.append((tid, _one_line(trace), relevance))
                return out
        except Exception:
            if self._decision is not None:
                self._decision.issue("failed")
            logger.debug("re-prime retrieval failed (ignored)", exc_info=True)
            return []

    async def collect(self) -> list[str]:
        """Await the in-flight re-prime task (hard 2s budget), return injection lines.

        Applies dedup, the per-injection (≤3) and per-turn (≤9) line caps, records the
        surfaced ids into the seen set, and emits ``memory.reprime.complete``. In SHADOW
        mode it logs would-have-injected and returns ``[]`` (injects nothing). Returns
        the marker + formatted lines only in LIVE mode.
        """
        task = self._task
        self._task = None
        if task is None:
            return []
        decision = self._decision
        t0 = time.time()
        try:
            # Native cancellation has one owner even when close races the tool
            # handler's wait. wait_for must not inject a second cancellation into
            # a worker already unwinding database cleanup.
            waiter = asyncio.shield(task) if self._observation is not None else task
            found = await asyncio.wait_for(waiter, timeout=_REPRIME_BUDGET_S)
        except TimeoutError:
            if self._observation is not None:
                self._request_cancel(task)
            else:
                task.cancel()
            self.counters["skipped_budget"] += 1
            if decision is not None:
                decision.complete("timeout", retrieved=self._retrieved_ids, retrieval_pending=not task.done())
            self._emit_reprime_log("skipped_budget", 0, int((time.time() - t0) * 1000), cue_snippet=self._pending_cue)
            return []
        except asyncio.CancelledError:
            if self._observation is not None:
                self._request_cancel(task)
            self.counters["cancelled"] += 1
            if decision is not None:
                decision.complete("cancelled", retrieved=self._retrieved_ids, retrieval_pending=not task.done())
            self._emit_reprime_log("cancelled", 0, int((time.time() - t0) * 1000))
            return []
        except Exception:
            if decision is not None:
                decision.complete("failed")
            logger.debug("re-prime task raised (ignored)", exc_info=True)
            return []

        # Funnel: None = novelty gate filtered it; [] = retrieved but nothing new after
        # dedup/floor. Distinct outcomes so the shadow logs carry the denominators.
        if found is None:
            if decision is not None:
                decision.complete("failed" if decision.issues else "novelty_gated")
            self.counters["skipped_gate"] += 1
            self._emit_reprime_log("skipped_gate", 0, int((time.time() - t0) * 1000), cue_snippet=self._pending_cue)
            return []
        if not found:
            if decision is not None:
                decision.complete("empty", retrieved=self._retrieved_ids)
            self.counters["noop"] += 1
            self._emit_reprime_log("noop", 0, int((time.time() - t0) * 1000), cue_snippet=self._pending_cue)
            return []

        # Cap enforcement.
        remaining_turn = _MAX_LINES_PER_TURN - self._lines_used
        if self._injections >= _MAX_REPRIME_ROUNDS or remaining_turn <= 0:
            if decision is not None:
                decision.complete("capped", retrieved=self._retrieved_ids)
            self.counters["suppressed_beyond_cap"] += 1
            self._emit_reprime_log("suppressed_beyond_cap", 0, int((time.time() - t0) * 1000))
            return []
        take = min(_MAX_LINES_PER_INJECTION, remaining_turn, len(found))
        chosen = found[:take]
        if take < len(found):
            if decision is not None:
                decision.issue("capped")
            self.counters["suppressed_beyond_cap"] += 1

        for tid, _line, _rel in chosen:
            self._seen_ids.add(tid)
        self._injections += 1
        self._lines_used += len(chosen)
        self.counters["fired"] += 1
        duration_ms = int((time.time() - t0) * 1000)
        self._emit_reprime_log(
            "fired",
            len(chosen),
            duration_ms,
            trace_ids=[tid for tid, _l, _r in chosen],
            cue_snippet=self._pending_cue,
        )

        if decision is not None:
            decision.complete(
                "retrieved",
                retrieved=self._retrieved_ids,
                selected=[tid for tid, _, _ in chosen] if self._state == LIVE else [],
            )
            if self._state == LIVE and decision.selected_ids:
                decision.delivery = "prepared"
        if self._state != LIVE:
            # Shadow — log would-have-injected, inject nothing.
            return []
        return [RECALL_MARKER, *(line for _tid, line, _rel in chosen)]

    def cancel(self) -> None:
        """Cancel the in-flight re-prime task (design §Stage C — cancel at every
        early exit: approval pauses + CancelledByNewMessage handlers)."""
        task = self._task
        self._task = None
        if self._observation is not None:
            for owned in self._retrieval_tasks:
                if not owned.cancelling():
                    owned.cancel()
        if task is not None and not task.done():
            if self._observation is None or not task.cancelling():
                task.cancel()
            self.counters["cancelled"] += 1
            self._emit_reprime_log("cancelled", 0, 0, cue_snippet=self._pending_cue)

    @staticmethod
    def _request_cancel(task: asyncio.Task) -> None:
        if not task.cancelling():
            task.cancel()
        # collect is the latency boundary. The retained worker belongs to close,
        # which joins its actual cleanup before the turn can publish a final receipt.

    async def drain(self) -> None:
        """Join actual retrieval cleanup independently of diagnostic finalization."""
        tasks = list(self._retrieval_tasks)
        for task in tasks:
            if not task.cancelling():
                task.cancel()
        await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))

    async def close(self, *, has_summary: bool = False) -> None:
        """Join native diagnostic work before its final receipt; legacy cancel stays synchronous."""
        try:
            self.cancel()
        finally:
            await self.drain()
        if self._decision is not None and self._decision.outcome == "pending":
            outcome = self._retrieval_outcome
            if outcome in {"retrieved", "partial"}:
                outcome = "discarded"
            elif outcome == "pending":
                outcome = "cancelled"  # Cancelled before the worker first ran.
            self._decision.complete(outcome, retrieved=self._retrieved_ids)
        if not self._considered and self._observation is not None:
            decision = self._observation.begin("reprise", trigger="reasoning_summary")
            decision.mode = {OFF: "off", SHADOW: "shadow", LIVE: "live"}.get(self._state, "off")
            decision.complete(
                self._inactive_reason if not self.active else "no_injection_point" if has_summary else "no_summary"
            )
            self._considered = True

    # -- C2: recall-and-revise on final responses -----------------------------

    async def maybe_revise(
        self,
        *,
        provider: Any,
        system_prompt: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
        thinking_blocks: list[str] | None,
        draft: str,
        metadata: dict[str, Any],
        can_revise_live: bool = True,
        ownership_lost: Callable[[], Awaitable[bool]] | None = None,
    ) -> str:
        """C2 — recall-and-revise on a final response (design §C2).

        Considers a re-prime on the final-turn thinking; on a strong-novel hit (a
        HIGHER relevance floor than mid-tool injection), takes ONE bounded revise
        step: same request + an assistant message carrying the draft + a user-role
        message containing ONLY the marker + recall lines (in-memory, NEVER
        persisted). Returns sanitized R2 when it contains visible text; otherwise
        retains the draft. When R2 is used, only R2 is sent or persisted. Cap 1
        revise/turn; the revise response can never trigger another revise.

        ``can_revise_live=False`` forces shadow behavior even when the flag is LIVE —
        used by the streaming loop, where R1 was already streamed to the user
        token-by-token and cannot be un-sent (the "R1 never sent" invariant). Live
        revision is confined to the non-streaming loops.

        SHADOW (or ``can_revise_live=False``): logs would-have-revised, returns the
        draft unchanged (no model call). Returns the draft unchanged whenever nothing
        qualifies.
        """
        if not self.active or self._revises >= _MAX_REVISES_PER_TURN:
            return draft
        text = _join_thinking(thinking_blocks)
        if not text:
            return draft
        recall = await self._reprime_for_revise(text)
        if not recall:
            return draft

        self._revises += 1
        chosen = recall[:_MAX_LINES_PER_INJECTION]
        revise_ids = [tid for tid, _l, _r in chosen]
        lines = [RECALL_MARKER, *(line for _tid, line, _rel in chosen)]
        if self._state != LIVE or not can_revise_live:
            self.counters["would_revise"] += 1
            self._emit_reprime_log("would_revise", len(lines) - 1, 0, trace_ids=revise_ids, cue_snippet=text)
            return draft

        # LIVE — one revise generation. Build an in-memory request; R1 + recall block
        # never leave this call, R2 is what the caller sends/persists.
        try:
            if ownership_lost is not None and await ownership_lost():
                logger.warning("process_completion event=loop_abort result=ownership_lost phase=c2")
                return draft
            revise_messages = [
                *messages,
                {"role": "assistant", "content": [{"text": draft}]},
                {"role": "user", "content": [{"text": "\n".join(lines)}]},
            ]
            r2 = await provider.converse(
                system_prompt,
                revise_messages,
                tools,
                max_tokens=max_tokens,
                temperature=1.0,
            )
            from kai_memory.ports import record_generation_provenance

            revised_text = strip_echoed_metadata(r2.text)
            if revised_text != r2.text:
                logger.warning("echoed metadata stripped path=c2_generation")
            used_for_response = bool(revised_text.strip())
            record_generation_provenance(metadata, getattr(r2, "provenance", None), used_for_response=used_for_response)
            if r2.thinking_blocks:
                metadata.setdefault("thinking_blocks", []).extend(r2.thinking_blocks)
            metadata["input_tokens"] = metadata.get("input_tokens", 0) + r2.usage.get("input", 0)
            metadata["output_tokens"] = metadata.get("output_tokens", 0) + r2.usage.get("output", 0)
            metadata["c2_revised"] = used_for_response
            if not used_for_response:
                return draft
            self.counters["revised"] += 1
            self._emit_reprime_log("revised", len(lines) - 1, 0, trace_ids=revise_ids, cue_snippet=text)
            return revised_text
        except ports.CancelledByNewMessage:
            # DB-624 — a turn the user already replaced must not fall back to
            # publishing its draft. The loop's cancel handler owns this exit.
            raise
        except Exception:
            logger.warning("C2 revise call failed; returning original draft", exc_info=True)
            return draft

    async def _reprime_for_revise(self, thinking_text: str) -> list[tuple[str, str, float]]:
        """Like :meth:`_reprime` but with the HIGHER C2 relevance floor."""
        emb = await _embed_cached(thinking_text)
        if emb is None:
            return []
        novelty = max((_cosine(emb, c) for c in self._cue_embeddings), default=0.0)
        if novelty >= _NOVELTY_COSINE_MAX:
            self.counters["skipped_gate"] += 1
            return []
        try:
            from kai_memory.ports import session_scope
            from kai_memory.retrieval.memory_priming import MemoryPrimingService

            async with session_scope() as s:
                svc = MemoryPrimingService(s)
                _fmt, _ids, top = await svc.prime_with_traces(
                    thinking_text, chat_ctx=self.chat_ctx, record_access=False, emit_log=False
                )
                out: list[tuple[str, str, float]] = []
                for trace, _score, components in top:
                    tid = str(trace.id)
                    if tid in self._seen_ids:
                        continue
                    relevance = float(components.get("semantic", 0.0) or 0.0)
                    if relevance < _REVISE_RELEVANCE_FLOOR:
                        continue
                    out.append((tid, _one_line(trace), relevance))
                return out
        except Exception:
            logger.debug("C2 re-prime retrieval failed (ignored)", exc_info=True)
            return []

    def _emit_reprime_log(
        self,
        outcome: str,
        lines: int,
        duration_ms: int,
        *,
        trace_ids: list[str] | None = None,
        cue_snippet: str | None = None,
    ) -> None:
        payload = {
            "event_type": "memory.reprime.complete",
            "timestamp": utc_now().isoformat(),
            "chat_id": getattr(self.chat_ctx, "chat_id", None),
            "topic_id": getattr(self.chat_ctx, "topic_id", None),
            "venue": getattr(self.chat_ctx, "venue", None),
            "mode": {OFF: "off", SHADOW: "shadow", LIVE: "live"}.get(self._state, "off"),
            "outcome": outcome,
            "lines": lines,
            "duration_ms": duration_ms,
            "injections_this_turn": self._injections,
            "lines_this_turn": self._lines_used,
        }
        # Observability (user request): on fired / would_revise / revised, carry WHICH
        # memories the thought would surface (join ids→content offline) + a hard-truncated
        # snippet of the cueing thought. It's Kai's own thinking (already persisted in
        # message_metadata), so no new sensitivity class enters the logs.
        if trace_ids is not None:
            payload["would_inject_trace_ids"] = [str(t) for t in trace_ids]
        if cue_snippet is not None and self._observation is None:
            payload["cue_snippet"] = cue_snippet[:_CUE_SNIPPET_CHARS]
        logger.info("memory.reprime.complete %s", json.dumps(payload))


def _swallow_task_result(task: asyncio.Task) -> None:
    """Done-callback so a cancelled/failed re-prime task never leaks an
    'exception was never retrieved' warning (design §Stage C F3)."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.debug("re-prime task ended with exception (ignored): %s", exc)


def _join_thinking(thinking_blocks: list[str] | None) -> str:
    if not thinking_blocks:
        return ""
    return "\n".join(t for t in thinking_blocks if isinstance(t, str) and t.strip()).strip()


def _one_line(trace: Any) -> str:
    """Render a trace as a single terse recalled line, carrying the provenance marker."""
    content = (getattr(trace, "content", "") or "").strip().replace("\n", " ")
    if len(content) > 220:
        content = content[:217] + "..."
    mtype = getattr(trace, "memory_type", None) or "memory"
    return f"- {RECALL_MARKER} [{mtype}] {content}"


# ---------------------------------------------------------------------------
# Injection helper — the BLOCKING safety invariant (design §Stage C F1/F7).
# ---------------------------------------------------------------------------


def inject_into_tool_results(tool_results: list[dict[str, Any]], lines: list[str]) -> bool:
    """Append the recall block INSIDE the first ``toolResult.content`` array.

    This is the loop-hint placement (chat_engine.py:726-731 / :1173-1178 / :1695-1700)
    — NEVER a sibling text block in the message-level ``tool_results`` list. Rationale
    (design §Containment #5, the red-team BLOCKER): a sibling top-level text block is
    read by ``ConversationMessage.text`` → rendered by L2 as user-spoken conversation →
    re-extracted by Haiku as user-attributed facts (the write-path echo chamber). Inner
    placement is invisible to ``.text``/L2. Returns True if a block was injected.
    """
    if not lines:
        return False
    for tr in tool_results:
        content = tr.get("toolResult", {}).get("content")
        if isinstance(content, list):
            content.append({"text": "\n".join(lines)})
            return True
    return False


# ---------------------------------------------------------------------------
# Entry point — build the per-turn reprise state + run Stage A shadow.
# ---------------------------------------------------------------------------


async def build_reprise_state(
    window_cues: WindowCues | None,
    last_user_msg: str | None,
    chat_ctx: ChatContext | None,
    primed_trace_ids: list[uuid.UUID],
    *,
    observation: MemoryObservation | None = None,
) -> RepriseState:
    """Create the per-turn :class:`RepriseState` and run the Stage A shadow counterfactual.

    Consumes the pre-built ``window_cues`` (from :func:`build_window_cues`, made once
    BEFORE priming so LIVE can feed the actual retrieval). Mirrored across all 3 tool
    loops (+ resume), called once right after L1 priming. Never takes the main request
    session — the shadow counterfactual and Stage C re-primes each open their OWN
    session, so instrumentation can never poison the session the tool loop reuses.
    Fail-open: any error returns an inert (OFF-state) RepriseState.

    Byte-identical invariant: with both flags OFF, ``window_cues`` is None and this
    returns an inert state with no embeds/logs.
    """
    try:
        c_state = thinking_cues_state()
        a_state = context_cues_state()
        if a_state == OFF and c_state == OFF:
            return RepriseState(chat_ctx, [], primed_trace_ids, OFF, observation=observation)

        cue_embeddings = window_cues.embeddings if window_cues is not None else []

        # Stage A shadow/live — full windowed-prime counterfactual + telemetry.
        if a_state in ACTIVE_STATES:
            await _run_stage_a_shadow(window_cues, last_user_msg, chat_ctx, primed_trace_ids, a_state)

        # Stage C needs a cue set to judge novelty against. With NO cue embeddings (empty
        # user text, or every cue embed failed) the novelty gate would default to 0.0 and
        # fire on EVERY thinking block — disable Stage C instead (it can't measure novelty).
        effective_c = c_state if cue_embeddings else OFF
        return RepriseState(
            chat_ctx,
            cue_embeddings,
            primed_trace_ids,
            effective_c,
            observation=observation,
            inactive_reason="no_cue_embeddings" if c_state != OFF and not cue_embeddings else "disabled",
        )
    except Exception:
        logger.warning("build_reprise_state failed; iterative retrieval inert this turn", exc_info=True)
        return RepriseState(chat_ctx, [], primed_trace_ids, OFF, observation=observation, inactive_reason="failed")


async def _run_stage_a_shadow(
    window_cues: WindowCues | None,
    last_user_msg: str | None,
    chat_ctx: ChatContext | None,
    primed_trace_ids: list[uuid.UUID],
    a_state: int,
) -> None:
    """Emit ``memory.window.shadow`` (``cue_mode`` / ``window_decisive_rate`` /
    ``primed_set_churn``; design §Evals). Injects nothing (Stage A is read-side).

    The SHADOW counterfactual — a FULL 4-path windowed prime (read-only, own session)
    diffed against the live top-8 — is dispatched as a BACKGROUND task, NOT awaited: it
    ~doubles priming work and this runs pre-LLM on every shadow turn, so keeping it on the
    request path would add that latency to the 2 GB ARM hot path. Backgrounding overlaps it
    with the LLM call instead (it's read-only + isolated on its own session; nothing on the
    turn depends on its result — it only logs). churn is computed synchronously first so the
    per-key baseline advances regardless of task timing.

    Own-session (session-poisoning lesson, memory_priming.py:178-185): a search failure
    (asyncpg cancel → unrolled-back tx) must NEVER poison the tool loop's session."""
    churn = _churn(chat_ctx, primed_trace_ids)
    is_windowed = window_cues is not None and window_cues.is_windowed
    cue_mode = "windowed" if is_windowed else "last_msg"

    # In LIVE, primed_trace_ids ALREADY are the windowed top-8 — the counterfactual would
    # be identical, so only SHADOW+windowed runs it (and off the hot path).
    if a_state == SHADOW and is_windowed and last_user_msg and window_cues is not None:
        task = asyncio.create_task(
            _shadow_counterfactual_and_log(
                window_cues, last_user_msg, chat_ctx, list(primed_trace_ids), churn, cue_mode, a_state
            )
        )
        task.add_done_callback(_swallow_task_result)
        adopt_if_owned(task)
        return

    # No counterfactual (live, or single-cue) — emit synchronously (cheap, no I/O).
    _emit_window_shadow_log(chat_ctx, a_state, cue_mode, window_cues, None, churn)


async def _shadow_counterfactual_and_log(
    window_cues: WindowCues,
    last_user_msg: str,
    chat_ctx: ChatContext | None,
    primed_trace_ids: list[uuid.UUID],
    churn: float | None,
    cue_mode: str,
    a_state: int,
) -> None:
    """Background body: run the full windowed prime read-only on its own session, diff its
    top-8 vs the live top-8, then emit ``memory.window.shadow``. Fail-open."""
    windowed_decisive = None
    try:
        from kai_memory.ports import session_scope
        from kai_memory.retrieval.memory_priming import MemoryPrimingService

        async with session_scope() as s:
            svc = MemoryPrimingService(s)
            _fmt, windowed_ids, _top = await svc.prime_with_traces(
                last_user_msg, chat_ctx=chat_ctx, record_access=False, window_cues=window_cues, emit_log=False
            )
            live_set = {str(i) for i in primed_trace_ids}
            windowed_set = {str(i) for i in windowed_ids}
            windowed_decisive = 1 if windowed_set != live_set else 0
    except Exception:
        logger.debug("stage-a windowed-prime counterfactual failed (ignored)", exc_info=True)
    _emit_window_shadow_log(chat_ctx, a_state, cue_mode, window_cues, windowed_decisive, churn)


def _emit_window_shadow_log(
    chat_ctx: ChatContext | None,
    a_state: int,
    cue_mode: str,
    window_cues: WindowCues | None,
    windowed_decisive: int | None,
    churn: float | None,
) -> None:
    payload = {
        "event_type": "memory.window.shadow",
        "timestamp": utc_now().isoformat(),
        "chat_id": getattr(chat_ctx, "chat_id", None),
        "topic_id": getattr(chat_ctx, "topic_id", None),
        "venue": getattr(chat_ctx, "venue", None),
        "mode": {SHADOW: "shadow", LIVE: "live"}.get(a_state, "shadow"),
        "cue_mode": cue_mode,
        "segments": len(window_cues.segments) if window_cues is not None else 0,
        "window_decisive_rate": windowed_decisive,  # per-turn 0/1; nightly aggregates
        "primed_set_churn": churn,
    }
    logger.info("memory.window.shadow %s", json.dumps(payload))
