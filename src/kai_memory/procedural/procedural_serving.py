"""Procedural-memory serving (design §3, redesigned for serve-all — DB-487).

``get_learned_behaviors`` renders the **standing rules block**: every rule that
passes the deterministic scope filter is served, every turn, in a byte-stable
category-grouped block that rides the CACHED system-prompt prefix.

**Why cosine retrieval is gone.** DB-436's 2026-08-06 prod diagnosis measured the
old cosine path on 200 real turns across 3 embedding bases: 0/200 turns served at any
floor ≥0.40, and 5 of 6 rules could not retrieve their own source quote. The
2026-08-08 research survey (``backend-docs/ai-workforce/research/
2026-08-08-rule-serving-and-prompt-learning-research.md``) explains why —
similarity retrieval is the DECLARATIVE mechanism; standing behavioral rules are
the PRODUCTION-system analog, and productions are not retrieved, they all match
every cycle. The deterministic scope filter (``_in_scope``) is our production-LHS.

The shape, per that survey's Q4 recommendations:

- **serve-all in scope**, capped at ``SERVE_MAX_RULES`` (25) — the cap is a
  curation-forcing device, not a capacity guard (per-instruction accuracy is
  85-93% at n=10 and stable to ~150, IFScale);
- the cap is enforced by **utility-ledger eviction** — utility decides WHICH
  rules survive the cap, never the order they are rendered in; equal utility
  (the common case — an unscored rule sits at the ceiling) evicts the OLDEST
  rule, so a freshly taught one never loses its slot to a stale one (DB-581);
- rendering is **category-grouped and deterministic within category**, so the
  block is byte-identical turn to turn and the prompt cache keeps hitting. It is
  NEVER utility-sorted per turn (cache churn for no behavioral gain), and it
  carries no per-turn counters for the same reason;
- the header states conflict resolution explicitly ("the more specific rule
  wins") — LLMs have no arbitration primitive.

Modes (``procedural_semantic_serving_mode``): 0=off, 1=shadow (compute + log
``procedural.serve.shadow``, inject nothing), 2=live. Fail-open throughout —
serving never breaks a turn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import logging
import uuid

from sqlalchemy import BigInteger, cast, exists, func, literal, or_, select
from sqlalchemy.orm import defer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.storage.models.memory_trace import MemoryTrace
from kai_memory.store import cognitive_config
from kai_memory.host_calls import scrub_secrets, validate_skill_content
from kai_memory.retrieval.memory_observation import MemoryDecision
from kai_memory.procedural.procedural_scope import (
    CATEGORY_ORDER,
    CATEGORY_TITLES,
    PROC_KINDS,
    PROC_STATUS_ACTIVE,
    conv_key,
    normalize_conv_keys,
)

if TYPE_CHECKING:
    from kai_memory.ports import ChatContext

# Pinned to the pre-DB-842 module path: CloudWatch queries, feature-watch criteria and
# log filters match the logger name, which must not follow the module's move.
_LEGACY_LOGGER_NAME = "app.services.chat.procedural_serving"
logger = logging.getLogger(_LEGACY_LOGGER_NAME)

# ── Versioned module constants (design §1.4 — NOT config rows) ───────────────
# Cap 25 (raised from DB-487's 10). Always a curation-forcing device, never a capacity
# guard: IFScale measures per-instruction compliance at 85-93% for n=10 and stable to
# ~150, so 10 was never the limit of what a model can follow — it was pressure to curate
# before we had live data. Research Q4(a) set the review point at ~25 and prod reached it
# (10/10 active rules, empty utility ledger), which is what made DB-581's tiebreak bite.
# 25 removes the eviction pressure while the ledger fills. Beyond 25 the answer is nightly
# rule consolidation/merging (research Q4(c)), NOT a bigger number — the block rides the
# cached prefix of every turn, so each rule is permanent prompt weight.
SERVE_MAX_RULES = 25
# The kind/status vocabularies are shared repo-wide from procedural_scope, so the two
# queries here can never diverge from each other or from the write side on a typo
# (AGENTS.md: define shared constants for repeated enum checks).
# Headless venues — Stage-3 serving (design §9); gated out of Stage 2 conversational serving.
_HEADLESS_VENUES = frozenset({"heartbeat", "wake"})

_UNCATEGORIZED = "Other"
# Hard cap on a single rule's rendered guidance. The text is model-authored with no
# length contract, and since DB-487 it sits in the CACHED prefix of EVERY turn — so one
# pathological rule would inflate the prompt permanently, not just once. Bounding at
# render (not at capture) keeps the stored rule intact and the block a pure, byte-stable
# function of stored content. 300 chars is the length the ledger's cost term already
# assumes ("a rule's guidance is <=300 chars", procedural_reinforcement.estimate_tokens).
_MAX_GUIDANCE_CHARS = 300
# DB-629 — the same budget for the trigger, but its OWN constant. The two are equal today
# and are not the same rule: guidance is capped because it rides the cached chat prefix
# every turn, the trigger because it reaches the L2 judge prompt. Sharing one name would
# mean raising either bound silently moves the other. Tests size their fixtures from this
# constant, never from a literal, so raising it cannot quietly empty its own boundary test
# (the #327 failure mode, three times over).
_MAX_TRIGGER_CHARS = 300
_TRUNCATION_SUFFIX = "…"


def _proc(row: MemoryTrace) -> dict:
    return (row.typed_metadata or {}).get("procedural", {}) or {}


def _u_norm(proc: dict) -> float:
    """clamp(U, 0, 1) — an ABSOLUTE clamp against fixed constants, never relative
    to peers (design §3.1.5). A rule with no ledger yet (U is None) is un-demoted,
    so it survives eviction at the utility ceiling (utility is demotion-only — §4.2)."""
    u = (proc.get("utility") or {}).get("U")
    if u is None:
        return 1.0
    return max(0.0, min(1.0, float(u)))


def _u_raw(proc: dict) -> float | None:
    """The stored ledger value, un-clamped and un-defaulted — ``None`` for a rule the
    ledger has never scored. Logged alongside ``_u_norm`` so an eviction line shows
    both what the ledger holds and what eviction actually compared."""
    u = (proc.get("utility") or {}).get("U")
    return None if u is None else float(u)


def _eviction_rank(row: MemoryTrace) -> tuple[float, float, str]:
    """Cap-eviction order (DB-581): lowest utility evicted first, oldest breaks the tie.

    Ties are the normal case here, not a corner: ``_u_norm`` returns the ceiling for
    every rule the ledger has not scored yet, so a store of recently taught rules is
    one large tie and the tiebreak IS the eviction policy. It used to be ``str(id)``,
    i.e. UUID order — arbitrary, and able to drop a rule taught seconds ago in favour
    of a stale one that had never earned its slot. Recency breaks the tie the way a
    person would: among rules the ledger cannot separate, the one you just taught
    outranks the one you taught months ago.

    The list is sorted ASCENDING and the first ``SERVE_MAX_RULES`` survive, so every
    component reads "smaller is safer": ``-U`` (higher utility first), then negated
    creation time (newer first), then the id as the final deterministic tiebreak so
    the surviving SET never depends on input order. A row with no ``created_at``
    (an ORM object that was never persisted) ranks as oldest.

    This is the EVICTION order only. Render order stays ``_render_key``, which reads
    neither utility nor time — that separation is what keeps the block byte-stable in
    the cached prompt prefix."""
    created = getattr(row, "created_at", None)
    recency = created.timestamp() if created is not None else float("-inf")
    return (-_u_norm(_proc(row)), -recency, str(row.id))


def _is_origin_conversation(proc: dict, chat_ctx: ChatContext) -> bool:
    """DB-436 mechanism 1 — is THIS conversation one that taught the rule?

    ``source_chat_ids`` is stamped by ``rule_capture`` with the conv keys that
    contributed evidence. Everyone present in that conversation witnessed the
    correction, so re-surfacing the rule there discloses nothing new — which is
    what makes it safe in a group even when the rule is not globally group-safe.
    Uses the SHARED key/normalization helpers so the read side can never drift
    from the write side (see ``procedural_scope``).
    """
    key = conv_key(chat_ctx.chat_id, chat_ctx.topic_id)
    if key is None:
        return False
    return key in normalize_conv_keys(proc.get("source_chat_ids") or [])


def _in_scope(proc: dict, chat_ctx: ChatContext) -> bool:
    """The leakage firewall (design §3.1.4) and — since DB-487 — the ONLY selector.

    Applied in the serving path so a person-A rule never surfaces in person-B's
    context, and NULL-sender (heartbeat/wake) turns see only global rules. E9
    asserts zero scope violations. This filter is deliberately UNCHANGED by the
    serve-all redesign: it is the production-system LHS, the thing that decides
    whether a standing rule may fire in this context at all.

    DB-436 widened the GROUP gate only. It used to be "group ⇒ must be group_safe",
    which — with ``group_safe`` hardcoded ``False`` at mint — made serving dead on
    arrival in Kai's ~all-group traffic. A rule may serve in a group when ANY of
    three provenance-aware conditions holds:

    1. it is globally ``group_safe`` (impersonal; the pre-existing condition), or
    2. this conversation TAUGHT it (``_is_origin_conversation``) — the audience
       already witnessed the correction, or
    3. it is person-scoped AND this sender IS that person — the rule is about the
       very person now speaking, so it discloses nothing they do not already know.

    The person firewall is evaluated FIRST and is unchanged: a person-A rule is
    rejected before the group gate is ever consulted, so none of the three widenings
    can leak one person's rule to another.
    """
    scope = proc.get("scope") or {}
    # venue
    if scope.get("venue", "any") not in ("any", chat_ctx.venue):
        return False
    # person scoping — the firewall, checked BEFORE the group gate so no group-gate
    # branch can widen a person-scoped rule beyond its person.
    person = scope.get("person")
    if chat_ctx.sender_user_id is None:
        if person is not None:
            return False  # NULL-sender serves ONLY global rules
    elif person is not None and person != chat_ctx.sender_user_id:
        return False
    # group gate (default not group-safe), widened by provenance — see docstring
    if chat_ctx.is_group and not (
        # `is True`, not truthy: the SQL prefilter tests `group_safe::text = 'true'`, so a
        # stored `"false"` (JSON string) or `1` would be TRUTHY here but FALSE there —
        # making SQL stricter than Python, the one direction the superset contract forbids,
        # and failing silently. `_judge_group_safe` writes a strict bool, so this is the
        # same value by construction; anything else is malformed and stays closed. [review-r1]
        scope.get("group_safe") is True
        or _is_origin_conversation(proc, chat_ctx)
        or (person is not None and person == chat_ctx.sender_user_id)
    ):
        return False
    return True


# The active procedural store is tens-to-hundreds of rows by design (§3.1). Bound
# the serve-all scan anyway so this path can never materialize an unbounded table
# into Python if that assumption ever breaks (the memory-retrieval budget rule).
_ACTIVE_RULES_HARD_CAP = 500


async def _has_active_rules(session: AsyncSession, chat_ctx: ChatContext) -> bool:
    """Cheap, **scope-aware** EXISTS guard (design §8 E8): skip the row load entirely
    when no rule could ever serve THIS context. A global guard would fire for every
    conversational user the moment any person-/heartbeat-scoped rule exists, even
    though ``_in_scope`` would exclude it.

    **Contract: this SQL is a SUPERSET of ``_in_scope``, never a subset.** It may admit
    a turn whose rules the Python filter then rejects (a wasted query — cheap), but it
    must never reject a turn that has a servable rule (a silent behavior removal —
    exactly the DB-436 bug, where the SQL's `group_safe = true` requirement made the
    guard return false for every group turn and serving died regardless of ``_in_scope``).
    Exactness lives in ``_in_scope``; this query only prunes.
    ``tests/integration/test_procedural_phase_c.py::_assert_layers_agree`` pins it.

    Note there is deliberately NO ``embedding IS NOT NULL`` condition: serve-all never
    embeds, so a NULL-embedded active rule IS servable, and requiring an embedding here
    would make SQL stricter than Python — the forbidden direction."""
    scope = MemoryTrace.typed_metadata["procedural"]["scope"]
    conds = [
        MemoryTrace.memory_type == "procedural",
        MemoryTrace.invalidated_at.is_(None),
        MemoryTrace.typed_metadata["procedural"]["kind"].astext.in_(PROC_KINDS),
        MemoryTrace.typed_metadata["procedural"]["status"].astext == PROC_STATUS_ACTIVE,
        # venue: missing → 'any' (parity with scope.get("venue", "any"))
        func.coalesce(scope["venue"].astext, "any").in_(("any", chat_ctx.venue)),
    ]
    if chat_ctx.is_group:
        # DB-436: mirror the WIDENED group gate — group_safe OR origin-conversation OR
        # sender-match. This is the layer the DB-436 trap lives in: extending only the
        # Python `_in_scope` would leave this EXISTS returning false for group turns, so
        # the rule load is skipped and serving stays dead however correct the filter is.
        group_ok = [
            # group safety: missing → 'false' (default not group-safe)
            func.coalesce(scope["group_safe"].astext, "false") == "true"
        ]
        if (key := conv_key(chat_ctx.chat_id, chat_ctx.topic_id)) is not None:
            src = cast(MemoryTrace.typed_metadata["procedural"]["source_chat_ids"], JSONB)
            origin = [src.has_key(key)]
            if chat_ctx.topic_id is None:
                # Legacy rows stored a bare int chat id, which normalize_conv_keys maps
                # to the topic-less key. `?` only matches string elements, so match the
                # numeric form too rather than under-selecting here.
                origin.append(src.contains(func.to_jsonb(literal(chat_ctx.chat_id, BigInteger))))
            group_ok.append(or_(*origin))
        if chat_ctx.sender_user_id is not None:
            group_ok.append(scope["person"].astext == str(chat_ctx.sender_user_id))
        conds.append(or_(*group_ok))
    if chat_ctx.sender_user_id is None:
        conds.append(scope["person"].astext.is_(None))  # NULL-sender → only global rules
    else:
        conds.append(or_(scope["person"].astext.is_(None), scope["person"].astext == str(chat_ctx.sender_user_id)))
    return bool((await session.execute(select(exists().where(*conds)))).scalar())


async def _active_rules(session: AsyncSession) -> list[MemoryTrace]:
    """Active, non-invalidated procedural rule/habit rows (tens-to-hundreds), hard-
    capped so the serve-all path always has a budget. Ordered by id so the row set
    handed to the deterministic renderer is itself stable."""
    stmt = (
        select(MemoryTrace)
        # Serve-all never embeds, and nothing downstream (_in_scope, _u_norm,
        # _render_key, _is_quarantined, _render) touches the vector. Deferring it keeps
        # up to 500 x 1536 float4 — several MB — off the wire on EVERY conversational
        # turn, which matters on a 2 GB box.
        .options(defer(MemoryTrace.embedding))
        .where(
            MemoryTrace.memory_type == "procedural",
            MemoryTrace.invalidated_at.is_(None),
            MemoryTrace.typed_metadata["procedural"]["kind"].astext.in_(PROC_KINDS),
            MemoryTrace.typed_metadata["procedural"]["status"].astext == PROC_STATUS_ACTIVE,
        )
        .order_by(MemoryTrace.id)
        .limit(_ACTIVE_RULES_HARD_CAP)
    )
    rows = list((await session.execute(stmt)).scalars().all())
    if len(rows) == _ACTIVE_RULES_HARD_CAP:
        # Never truncate silently. The store is tens-to-hundreds by design, so hitting
        # this bound means that assumption broke — and rules beyond the cut are dropped
        # BEFORE scope filtering and utility eviction, i.e. an arbitrary slice, not the
        # best ten. A silent version of exactly this is what DB-436 was.
        logger.warning(
            "procedural.serve.scan_cap_hit n=%d — active rule scan saturated; rules beyond the "
            "cap are excluded before scope filtering",
            _ACTIVE_RULES_HARD_CAP,
        )
    return rows


def _category(proc: dict) -> str:
    category = str(proc.get("category") or "")
    return category if category in CATEGORY_TITLES else _UNCATEGORIZED


def _render_key(row: MemoryTrace) -> tuple[int, str, str]:
    """Deterministic render position: category group, then guidance text, then id.

    Pure function of stored content — NOT of utility, recency, or the current turn.
    That is what makes the block byte-stable across turns, which is what keeps the
    cached prompt prefix hitting."""
    proc = _proc(row)
    category = _category(proc)
    order = CATEGORY_ORDER.index(category) if category in CATEGORY_ORDER else len(CATEGORY_ORDER)
    return (order, str(proc.get("guidance") or ""), str(row.id))


def _is_quarantined(proc: dict) -> bool:
    """The render-time security re-screen (design §3.3): a rule whose guidance trips the
    central validators must never reach the prompt, however it got into the store."""
    guidance = str(proc.get("guidance") or "")
    return bool(validate_skill_content(guidance)) or scrub_secrets(guidance) != guidance


def select_servable_rules(rows: list[MemoryTrace], chat_ctx: ChatContext) -> list[MemoryTrace]:
    """Serve-all selection (design DB-487 §2): every in-scope rule, capped at
    ``SERVE_MAX_RULES`` by **utility eviction**, returned in deterministic render order.

    Utility appears exactly once, as the eviction signal deciding WHICH rules survive
    an over-cap store — never as the render order (that would churn the prompt cache
    every night the ledger moves; research Q4(b)). Utility ties — the common case, since
    every unscored rule sits at the ceiling — evict the OLDEST first and fall back to id
    for determinism (``_eviction_rank``, DB-581). Every evicted rule is logged. Shared by
    the live path and the tests, so a cap/ordering regression cannot hide behind a
    test-only code path.

    Quarantined rules are dropped HERE rather than only at render, so their ids never
    reach ``served_rule_ids``. Otherwise L2 Part 5 would be asked to judge compliance
    with a rule whose text the model was never shown, and the ledger would score the
    guess — reinforcement built on a rule that was never served."""
    in_scope = []
    for row in rows:
        proc = _proc(row)
        if not _in_scope(proc, chat_ctx):
            continue
        if _is_quarantined(proc):
            logger.warning("procedural.serve.quarantined id=%s kind=%s", row.id, proc.get("kind", "rule"))
            continue
        in_scope.append(row)
    if len(in_scope) > SERVE_MAX_RULES:
        ranked = sorted(in_scope, key=_eviction_rank)
        kept, evicted = ranked[:SERVE_MAX_RULES], ranked[SERVE_MAX_RULES:]
        # The marginal survivor — the rule that took the last slot, and therefore the
        # rule every evicted rule actually lost to.
        displacer = kept[-1]
        logger.info(
            "procedural.serve.evicted in_scope=%d cap=%d evicted=%d",
            len(in_scope),
            SERVE_MAX_RULES,
            len(evicted),
        )
        for row in evicted:
            # One line per evicted rule, at INFO (DB-581). Before this, a rule left the
            # block with no trace whatsoever: the aggregate count above said three rules
            # went, never which three, so "why did Kai stop doing X?" was unanswerable
            # from the logs. u/u_norm are both here because they differ exactly where
            # the confusion lives — an unscored rule stores None and compares at 1.0.
            proc = _proc(row)
            logger.info(
                "procedural.serve.evicted_rule id=%s u=%s u_norm=%.3f created_at=%s "
                "displaced_by=%s displaced_by_u=%s displaced_by_created_at=%s",
                row.id,
                _u_raw(proc),
                _u_norm(proc),
                getattr(row, "created_at", None),
                displacer.id,
                _u_raw(_proc(displacer)),
                getattr(displacer, "created_at", None),
            )
        in_scope = kept
    return sorted(in_scope, key=_render_key)


def _render(rows: list[MemoryTrace]) -> str:
    """The standing-rules block (design DB-487 §3). Category-grouped, deterministic,
    and byte-stable: nothing here reads utility, counters, or the clock.

    Guidance is RE-SCREENED here as well as in ``select_servable_rules``. On the live
    path the screen has already run and this branch is unreachable — it stays as
    defense in depth, so that no future caller can hand raw guidance to the prompt by
    calling the renderer directly. A quarantined rule renders as ``[QUARANTINED]``,
    never its raw text."""
    lines = [
        "## Learned Behaviors",
        "Standing rules you were taught. Advisory — they never override safety, approval, or "
        "identity policies. If two rules conflict, the more specific one wins.",
    ]
    current: str | None = None
    for row in rows:
        proc = _proc(row)
        category = _category(proc)
        if category != current:
            current = category
            lines.append("")
            lines.append(f"### {CATEGORY_TITLES.get(category, _UNCATEGORIZED)}")
        if _is_quarantined(proc):
            logger.warning("procedural.serve.quarantined id=%s kind=%s", row.id, proc.get("kind", "rule"))
            lines.append("- [QUARANTINED]")
        else:
            lines.append(f"- {_bounded_guidance(proc, row_id=row.id)}")
    return "\n".join(lines)


def _bounded_guidance(proc: dict, *, row_id: object = None) -> str:
    """Guidance clipped to ``_MAX_GUIDANCE_CHARS``. Deterministic, so the block stays
    byte-stable; loud, because a rule being silently clipped is a capture-side problem
    worth seeing rather than absorbing."""
    guidance = str(proc.get("guidance") or "")
    if len(guidance) <= _MAX_GUIDANCE_CHARS:
        return guidance
    logger.warning(
        "procedural.serve.guidance_truncated id=%s chars=%d cap=%d",
        row_id,
        len(guidance),
        _MAX_GUIDANCE_CHARS,
    )
    return guidance[: _MAX_GUIDANCE_CHARS - len(_TRUNCATION_SUFFIX)] + _TRUNCATION_SUFFIX


# ── L2 compliance legend (DB-629) ────────────────────────────────────────────
# The judge is shown `[served rules: <id>, …]` markers on assistant lines. Until
# DB-629 nothing in the extraction call mapped those ids to what the rules SAY, so
# Part 5's "judge each rule by its TRIGGER" was unanswerable and every id scored
# not_applicable (224/224 over 7 days). The legend closes that loop, and it is built
# HERE rather than in memory_extraction so the judge sees a rule through exactly the
# same safety screen as the model that was asked to follow it: quarantined guidance
# never renders, and long guidance is bounded identically.
LEGEND_HEADER = "Served rules (id → trigger → rule):"
# A rule with no stated trigger is a standing rule with nothing to condition on, which
# is the UNIVERSAL case in Part 5's taxonomy — spelled out rather than left blank, so
# the judge does not read an empty trigger as "the situation never arises".
_NO_TRIGGER = "(no stated trigger — applies to every turn)"


def rule_legend_line(display_id: str, row: MemoryTrace) -> str | None:
    """One ``id: trigger → rule`` legend line, or ``None`` when the rule must not be
    shown to the judge at all.

    Returning ``None`` for a quarantined rule is deliberate and is the same rule the
    serving path enforces in ``select_servable_rules``: a rule whose text the model was
    never allowed to see must not be judged, because the ledger would then be scoring a
    guess. The caller drops the id from the marker too, so the judge is never handed an
    id it has no text for."""
    proc = _proc(row)
    if _is_quarantined(proc):
        logger.warning("procedural.legend.quarantined id=%s kind=%s", row.id, proc.get("kind", "rule"))
        return None
    guidance = _bounded_guidance(proc, row_id=row.id)
    if not guidance.strip():
        return None
    trigger = _bounded_trigger(proc, row_id=row.id)
    if trigger is None:
        return None
    return f"{display_id}: {trigger} → {guidance}"


def _bounded_trigger(proc: dict, *, row_id: object = None) -> str | None:
    """The trigger, security-screened and length-bounded, or ``None`` to drop the rule.

    DB-629 put ``trigger_pattern`` into a prompt for the first time — until the legend it
    was stored-but-never-rendered, so ``_is_quarantined`` screening only ``guidance`` was
    sufficient. It no longer is: capture screens the trigger once at
    ``evaluate_candidate``, and this is the same render-time re-screen guidance already
    gets, so no future caller can hand raw stored text to a model. The length bound is the
    other half — guidance is capped and an unbounded trigger would walk straight around
    that budget."""
    trigger = str(proc.get("trigger_pattern") or "").strip()
    if not trigger:
        return _NO_TRIGGER
    if validate_skill_content(trigger) or scrub_secrets(trigger) != trigger:
        logger.warning("procedural.legend.trigger_quarantined id=%s", row_id)
        return None
    if len(trigger) > _MAX_TRIGGER_CHARS:
        logger.warning(
            "procedural.legend.trigger_truncated id=%s chars=%d cap=%d",
            row_id,
            len(trigger),
            _MAX_TRIGGER_CHARS,
        )
        return trigger[: _MAX_TRIGGER_CHARS - len(_TRUNCATION_SUFFIX)] + _TRUNCATION_SUFFIX
    return trigger


def render_rule_legend(entries: list[tuple[str, MemoryTrace]]) -> tuple[str, dict[str, str]]:
    """Render the served-rules legend block plus the ``display id -> rule id`` map.

    Sorted by display id so the block is a pure function of its inputs — byte-stable for
    a given window, which is what keeps it from churning anything downstream."""
    lines: list[str] = []
    alias: dict[str, str] = {}
    ordered = sorted(entries, key=lambda e: e[0])
    if len(ordered) > SERVE_MAX_RULES:
        # The serving path caps ONE turn at SERVE_MAX_RULES, but this is the union across
        # every turn in the extraction window, so rule churn mid-window could push it past
        # the cap and put an unbudgeted block in the L2 prompt. Sorted-by-id keeps the
        # truncation deterministic (the block stays a pure function of its inputs); loud,
        # because reaching this at all means the store is churning faster than a window.
        logger.warning("procedural.legend.capped entries=%d cap=%d", len(ordered), SERVE_MAX_RULES)
        ordered = ordered[:SERVE_MAX_RULES]
    for display_id, row in ordered:
        line = rule_legend_line(display_id, row)
        if line is None:
            continue
        lines.append(f"- {line}")
        alias[display_id] = str(row.id)
    if not lines:
        return "", {}
    return "\n".join([LEGEND_HEADER, *lines]), alias


async def load_rules_by_id(session: AsyncSession, rule_ids: list[uuid.UUID]) -> list[MemoryTrace]:
    """Load live procedural rule rows by id for the legend. Invalidated (retired) rules
    are excluded: their text is no longer a standing rule, so judging against it would
    feed the ledger evidence for a rule that no longer exists."""
    if not rule_ids:
        return []
    stmt = (
        select(MemoryTrace)
        # The legend reads guidance and trigger only — same reason _active_rules defers it.
        .options(defer(MemoryTrace.embedding))
        .where(
            MemoryTrace.id.in_(rule_ids),
            MemoryTrace.memory_type == "procedural",
            MemoryTrace.invalidated_at.is_(None),
            MemoryTrace.typed_metadata["procedural"]["kind"].astext.in_(PROC_KINDS),
        )
        # The IN list is caller-bounded, but an explicit row cap is the standing rule for
        # this table (the nightly scan bounds it the same way) — one long window must not
        # be able to load an arbitrary number of rule rows on the 2 GB instance. Ordered,
        # so a direct caller passing more ids gets a STABLE subset rather than whatever
        # Postgres returns first — the legend is documented as a pure function of inputs.
        .order_by(MemoryTrace.id)
        .limit(SERVE_MAX_RULES)
    )
    return list((await session.execute(stmt)).scalars().all())


async def get_learned_behaviors(
    session: AsyncSession,
    chat_ctx: ChatContext,
    *,
    observation: MemoryDecision | None = None,
    raise_on_error: bool = False,
) -> tuple[str | None, list[uuid.UUID]]:
    """Serve the standing-rules block. Returns ``(block, served_rule_ids)``.

    Off/shadow return ``(None, [])`` (shadow logs the candidate ids first). Live
    returns the rendered block — which the caller places in the CACHED prompt prefix
    — plus every rendered rule id for the compliance ledger. Fail-open: any error
    yields ``(None, [])``, so serving never breaks a turn.
    """
    try:
        mode = int(cognitive_config.get("procedural_semantic_serving_mode"))
        if observation is not None:
            observation.mode = {0: "off", 1: "shadow"}.get(mode, "live")
        if mode == 0:
            if observation is not None:
                observation.complete("disabled")
            return None, []

        # Stage 2 = "conversation venues only". BOTH headless venues (heartbeat AND
        # wake) are Stage-3 serving (design §9), gated behind the rung-2/3 flag.
        if chat_ctx.venue in _HEADLESS_VENUES and not cognitive_config.get_bool("procedural_habit_promotion_enabled"):
            if observation is not None:
                observation.complete("venue_gated")
            return None, []

        # Both reads run inside ONE SAVEPOINT. Serving is fail-open, but "open" must not
        # mean "destructive": the caller's session may carry uncommitted work, so a bare
        # rollback on failure could discard it. A savepoint confines a failed statement to
        # this advisory read — the only thing that should be lost when serving fails.
        async with session.begin_nested():
            # E8 short-circuit — skip the row load when no in-scope rule can exist.
            if not await _has_active_rules(session, chat_ctx):
                return None, []
            rows = select_servable_rules(await _active_rules(session), chat_ctx)
        if not rows:
            return None, []
        ids = [r.id for r in rows]

        if mode == 1:  # shadow — compute everything, inject nothing
            if observation is not None:
                observation.complete("retrieved", retrieved=ids)
            logger.info(
                "procedural.serve.shadow venue=%s n=%d ids=%s",
                chat_ctx.venue,
                len(ids),
                [str(i) for i in ids],
            )
            return None, []

        block = _render(rows)
        if observation is not None:
            observation.complete("retrieved", retrieved=ids, selected=ids)
        logger.info("procedural.served ids=%s venue=%s n=%d", [str(i) for i in ids], chat_ctx.venue, len(ids))
        return block, ids
    except Exception as exc:
        if observation is not None:
            observation.complete("failed")
        logger.warning("procedural serving failed type=%s", type(exc).__name__)
        if raise_on_error:
            raise
        return None, []
