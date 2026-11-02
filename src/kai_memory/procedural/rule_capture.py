"""Phase C §2.2 — the deterministic trust gate (the injection-by-habit firewall).

A Part-4 behavior-rule candidate (produced by the L2 Haiku pass, §2.1) is minted
into the ``memory_type='procedural'`` rule store ONLY if it clears every gate here.
The load-bearing control is **quote-anchored provenance**: the candidate's
``source_quote`` must substring-match a genuine ``role='user'`` message in the
extraction window, and that DB row's ``sender_user_id`` — never Haiku's own
attribution — is the instructor. This kills:

- **tool-result injection** structurally — text inside a fetched webpage/file is
  never a ``role='user'`` message;
- **reply-quote smuggling** — the handler prepends ``[Replying to X: "..."]`` to
  the replier's row; the quoted span (first ``reply_quote_len`` chars, §1.3) is
  excluded from matching, and legacy rows without that marker that start with
  ``[Replying to `` fail closed;
- **forward smuggling** — rows stamped ``forwarded=true`` never anchor;
- **multi-match** — if the quote matches several user rows, ALL their senders must
  pass the trust check.

Then: category allowlist (form/process only, never stance), central security
screen (``scrub_secrets`` + ``validate_skill_content``, reject-not-downgrade),
scrub-at-rest of the stored quote, and dedup/conflict (§4.3). Invoked from
``extract_from_conversation`` beside ``_store_social_experiences``,
savepoint-isolated, fail-open, behind ``procedural_rule_extraction_enabled``.

**First-teach activation (DB-487).** A candidate that clears the whole chain above
mints ``status='active'`` immediately. The three old promotion ladders are gone:
the memory-verb requirement, the "2nd occurrence from a different conversation"
gate, and the communication-probe hold. DB-436's prod diagnosis is why — in 26 days
NO rule ever reached ``active``, so nothing could ever serve: corrections (all six
real rules) never carried a memory verb, the promotion ladder needed a ≥0.90-cosine
re-extraction match that never happened, and the comm probe was a ``return PENDING``
stub holding 3 of 6 rules forever. The trust chain — admin-only, quote-anchored,
security-screened, category-allowlisted — is the safety property; the ladders only
added latency to a gate that was already closed. The behavioral safety net is the
serving ON/OFF eval plus the biweekly ELEPHANT run, not an un-wired probe.

**Respect — teaching is OPEN at >= 75 (DB-487, superseded scope 2026-08-29).** The
gate is no longer admin-only. Every matched instructor resolves to a respect score
from an operator-confirmed roster (``PROCEDURAL_RESPECT_ROSTER``, env/SSM — editable
without a deploy); admins resolve to 100 from the admin ids. Anyone not on the roster
scores 0 and is rejected with a logged reason.

The rationale is observing explicitly trusted teachers before further hardening.
Tracking is a first-class part of the feature,
not decoration: every non-admin mint logs teacher/respect/rule/scope, every rejection
logs its reason and score, a never-before-seen sender attempting to teach gets its own
marker for a roster decision, and each rule carries ``provenance.taught_by``. That
record is what DB-574's hardening decision will be made from.

What did NOT open: the quote-anchored chain, the category allowlist, both security
screens, and one-strike probation for roster-taught rules. Gain scales by ``respect/100``,
so a 75-taught rule must out-perform an admin-taught one to survive cap eviction, and it
alone can be demoted: since DB-648 an admin-taught rule is never suspended for
non-compliance — its utility still ranks it, but a violation escalates instead of demoting.
"""

from __future__ import annotations

import logging
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from kai_memory.host_calls import (
    get_settings,
    is_admin_user,
    scrub_secrets,
    strip_msg_meta_envelope,
    utc_now,
    validate_skill_content,
)
from kai_memory.store import cognitive_config
from kai_memory.procedural.procedural_scope import (
    CATEGORY_ORDER,
    MessageAnchor,
    PROC_KINDS,
    PROC_STATUS_ACTIVE,
    REACTIVATABLE_STATUSES,
    conv_key as _conv_key,
    normalize_conv_keys,
    trusted_actor,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from kai_memory.sdk import Message as ConversationMessage
    from kai_memory.ports import ChatContext

# Pinned to the pre-DB-842 module path: CloudWatch queries, feature-watch criteria and
# log filters match the logger name, which must not follow the module's move.
_LEGACY_LOGGER_NAME = "app.services.chat.rule_capture"
logger = logging.getLogger(_LEGACY_LOGGER_NAME)

# ── Versioned module constants (design §1.4/§2.2 — NOT config rows) ──────────
# Derived from the shared ordered vocabulary so the write-side allowlist and the
# read-side render grouping can never diverge (see procedural_scope).
CATEGORY_ALLOWLIST = frozenset(CATEGORY_ORDER)
PROBATION_DAYS = 7
_RECURRENCE_EVENTS_TAIL = 50  # bound pending recurrence msg ids (mirrors the ledger's event_keys tail)
# DB-629 — same bound for pending message-anchored compliance events. The nightly pass
# ages these out on its own lookback window, so the tail only guards the pathological
# case where reinforcement is disabled and nothing is draining the list.
_COMPLIANCE_EVENTS_TAIL = 50
# How many teaching conversations a rule remembers. Pre-DB-487 this was an ACTIVATION
# counter (2 distinct chats promoted a candidate) and the list was allowed to fill and
# freeze. Under serve-all it is a SERVING key — origin-conversation serving (DB-436
# mechanism 1) matches on it — so a frozen list means a rule re-taught in a third
# conversation never serves in the conversation that just taught it. It is now a
# most-recent-wins ring: still bounded for the 2 GB box, but the newest teaching
# conversation always gets in.
_DISTINCT_CONVS_TRACKED = 5
_REPLY_QUOTE_MARKER = "[Replying to "
_WHITESPACE = re.compile(r"\s+")

# ── Respect (DB-487, superseded scope 2026-08-29) ────────────────────────────
# Teaching is OPEN at respect >= 75. Observation before hardening: a 75-taught rule
# gets whatever scope the extractor infers, deliberately unrestricted, and the live
# data feeds DB-574's hardening decisions.
#
# What still bounds the blast radius, unchanged by the opening:
#   - the quote-anchored trust chain (only a genuine role='user' message can anchor;
#     tool results, forwards and reply-quotes can never mint);
#   - the category allowlist — form/process only, never stance;
#   - the central security screen at capture AND the render-time quarantine;
#   - probation: ONE violated verdict suspends a new ROSTER-taught rule. An admin-taught
#     rule is exempt from status demotion since DB-648 — a violation there is a serving
#     bug and escalates (procedural.rule.ignored) instead of forgetting the instruction
#     (procedural_reinforcement.is_protected_from_demotion);
#   - gain scales by respect/100 for BOTH, so a 75-taught rule needs a better track record
#     than an admin-taught one to survive cap eviction, and — being unprotected — to clear
#     the demotion floor at all (procedural_reinforcement.respect_multiplier).
RESPECT_ADMIN = 100  # the configured admin — resolved from TELEGRAM_ADMIN_USER_IDS
RESPECT_ROSTER_MEMBER = 75  # an operator-confirmed teacher from PROCEDURAL_RESPECT_ROSTER
RESPECT_UNKNOWN = 0  # anyone not on the roster — cannot teach
MINT_RESPECT_MIN = RESPECT_ROSTER_MEMBER

# Back-compat alias: this constant was named for the pre-supersession "anyone with
# conversation history" rule, which the confirmed roster replaced.
RESPECT_KNOWN_PERSON = RESPECT_ROSTER_MEMBER


@dataclass
class GateResult:
    """Outcome of the deterministic trust gate for one candidate."""

    accepted: bool
    # ok | no_anchor | untrusted | multi_match_untrusted | offschema | security
    reason: str
    instructor_id: int | None = None
    status: str | None = None  # active (first-teach) — DB-487 removed the candidate ladders
    scrubbed_quote: str | None = None
    # id of the first trusted user row the quote anchored to — computed ONCE here so
    # capture_rules never re-walks the window (single injection-defense surface).
    matched_message_id: str | None = None
    # The instructor's respect score and the threshold it was judged against, so an
    # accept AND a rejection can both explain themselves (DB-487 acceptance).
    respect: int = RESPECT_UNKNOWN
    respect_threshold: int = MINT_RESPECT_MIN
    # The sender the quote anchored to, set even when the gate REJECTS — `instructor_id`
    # is only meaningful on an accept, but the tracking requirement is precisely about
    # who tried and was turned away.
    attempted_by: int | None = None


def _norm(text: str) -> str:
    """Whitespace-normalized, envelope-stripped, casefolded — the matching space
    (design §2.2.1). Envelope strip uses the shared legacy-input JSON parser."""
    return _WHITESPACE.sub(" ", strip_msg_meta_envelope(text or "")).strip().casefold()


def _anchorable_text(msg: ConversationMessage) -> str | None:
    """The searchable text of a window row, or None if the row may not anchor a
    quote (design §2.2.1). Only ``role='user'`` rows anchor; forwarded rows and
    legacy ``[Replying to`` rows fail closed; the reply-quote span is excluded."""
    if getattr(msg, "role", None) != "user":
        return None
    meta = getattr(msg, "message_metadata", None) or {}
    if meta.get("forwarded"):
        return None  # forward refusal — not the sender's own words
    raw = msg.text or ""
    rql = meta.get("reply_quote_len")
    if isinstance(rql, int) and rql > 0:
        raw = raw[rql:]  # exclude the verbatim reply-quote prefix span
    elif raw.lstrip().startswith(_REPLY_QUOTE_MARKER):
        return None  # legacy row without the marker — fail closed
    return _norm(raw)


def parse_respect_roster(roster_str: str | None) -> dict[int, int]:
    """Parse ``PROCEDURAL_RESPECT_ROSTER`` — ``"<user_id>:<respect>"`` pairs, comma-
    separated — into ``{user_id: respect}``.

    Deliberately strict and fail-CLOSED: a malformed entry is dropped with a warning
    rather than guessed at. This string decides who may write standing instructions
    into Kai's system prompt, so a typo must cost someone their teaching rights, never
    silently grant them. Admins are NOT listed here (they resolve to 100 from the
    admin ids), and a roster entry may not exceed ``RESPECT_ADMIN``.
    """
    roster: dict[int, int] = {}
    for raw in (roster_str or "").split(","):
        entry = raw.strip()
        if not entry:
            continue
        uid_str, _, score_str = entry.partition(":")
        uid_str, score_str = uid_str.strip(), score_str.strip()
        # Digits only: a leading "-" marks a Telegram CHAT id, not a user id, so accepting
        # one would silently seat an un-matchable entry in a list meant to be auditable.
        if not uid_str.isdigit() or not score_str.isdigit():
            logger.warning("procedural.respect.roster_entry_malformed entry=%r — ignored", entry)
            continue
        score = int(score_str)
        if not 0 <= score <= RESPECT_ADMIN:
            logger.warning("procedural.respect.roster_entry_out_of_range entry=%r — ignored", entry)
            continue
        roster[int(uid_str)] = score
    return roster


def roster_respect_of(admin_ids_str: str | None, roster_str: str | None = None) -> Callable[[int | None], int]:
    """Build the respect resolver: admin → 100, roster member → its configured score,
    everyone else → 0.

    The roster is CONFIG, not code (``PROCEDURAL_RESPECT_ROSTER``, env/SSM — same path
    as the admin ids), so an operator can add or revoke a teacher without a deploy. Admins are
    resolved from the admin ids rather than duplicated into the roster, so the two lists
    cannot drift apart and grant someone teaching rights nobody meant to.
    """
    roster = parse_respect_roster(roster_str)

    def _respect(sender_user_id: int | None) -> int:
        if is_admin_user(sender_user_id, admin_ids_str):
            return RESPECT_ADMIN
        if sender_user_id is None:
            return RESPECT_UNKNOWN
        return roster.get(sender_user_id, RESPECT_UNKNOWN)

    return _respect


def admin_respect_of(admin_ids_str: str | None) -> Callable[[int | None], int]:
    """Admin-only resolver — the fail-closed default for ``evaluate_candidate`` so the
    pure gate stays DB-free and never widens teaching just because a caller forgot to
    pass a resolver."""
    return roster_respect_of(admin_ids_str, None)


def _security_violation(text: str) -> bool:
    """Central validators (design §2.2.4) — a secret or dangerous directive in the
    RULE text rejects (never downgrades). Mirrors the serving-render re-screen."""
    return bool(validate_skill_content(text)) or scrub_secrets(text) != text


def _log_gate_rejection(reason: str, category: str) -> None:
    if reason == "offschema":
        logger.warning("procedural.rule.rejected_offschema reason=category_not_allowed")
    elif reason == "security":
        logger.warning("procedural.rule.blocked_security category=%s", category)


def evaluate_candidate(
    candidate: dict[str, Any],
    window_messages: list[ConversationMessage],
    admin_ids_str: str | None,
    respect_of: Callable[[int | None], int] | None = None,
    *,
    diagnostics: bool = True,
) -> GateResult:
    """The pure deterministic trust gate (design §2.2). No DB, no LLM. Returns a
    ``GateResult``; the orchestrator handles dedup + persistence for accepted ones.

    ``respect_of`` maps a matched sender id to a respect score; it defaults to the
    admin-only resolver, which is exactly the pre-DB-487 trust test."""
    respect_of = respect_of or admin_respect_of(admin_ids_str)
    source_quote = str(candidate.get("source_quote") or "")
    quote_norm = _norm(source_quote)
    if not quote_norm:
        return GateResult(accepted=False, reason="no_anchor")

    # 1. Quote-anchored provenance — the load-bearing control. Attribution comes
    #    from the matched DB row's sender, never from Haiku. Collect (sender, id)
    #    per match so capture_rules can reuse the id without re-walking the window.
    matched: list[tuple[int | None, str | None]] = []
    for msg in window_messages:
        text = _anchorable_text(msg)
        if text is not None and quote_norm in text:
            mid = getattr(msg, "id", None)
            matched.append((getattr(msg, "sender_user_id", None), str(mid) if mid is not None else None))
    if not matched:
        return GateResult(accepted=False, reason="no_anchor")
    matched_senders = [s for s, _ in matched]

    # 2. Trust, expressed as respect (DB-487) — multi-match fail-closed: the WEAKEST
    #    matched sender decides, so an admin quote that also appears in an off-roster
    #    member's message can never mint. MINT_RESPECT_MIN is RESPECT_ROSTER_MEMBER (75),
    #    so the door is open to confirmed roster members, not admins alone.
    respects = [respect_of(s) for s in matched_senders]
    respect = min(respects)
    # The weakest matched sender both decides the verdict and is the one to name: on a
    # multi-match rejection, the interesting party is whoever dragged the score down.
    weakest = min(zip(respects, range(len(matched_senders)), strict=False))[1]
    attempted_by = matched_senders[weakest]
    if respect < MINT_RESPECT_MIN:
        reason = "multi_match_untrusted" if len(matched_senders) > 1 else "untrusted"
        return GateResult(accepted=False, reason=reason, respect=respect, attempted_by=attempted_by)
    # The WEAKEST matched sender is the instructor of record, so `provenance.taught_by`
    # and `provenance.respect` always describe the same person. Taking the first non-None
    # sender instead would, on a quote matching two different trusted people, attribute
    # the rule to one and score it from the other.
    instructor_id = (
        attempted_by if attempted_by is not None else next((s for s in matched_senders if s is not None), None)
    )
    matched_message_id = next((mid for sender, mid in matched if sender == instructor_id and mid is not None), None)

    # 3. Category allowlist. Neither category nor model-supplied fields are safe log data.
    category = str(candidate.get("category") or "")
    if category not in CATEGORY_ALLOWLIST:
        if diagnostics:
            _log_gate_rejection("offschema", category)
        return GateResult(accepted=False, reason="offschema", respect=respect, attempted_by=attempted_by)

    # 4. Security screen on the RULE text (reject-not-downgrade).
    rule_text = str(candidate.get("rule") or "")
    trigger_text = str(candidate.get("trigger") or "")
    if _security_violation(rule_text) or _security_violation(trigger_text):
        if diagnostics:
            _log_gate_rejection("security", category)
        return GateResult(accepted=False, reason="security", respect=respect, attempted_by=attempted_by)

    # 5. First-teach activation (DB-487): clearing the chain above IS the bar.
    #    Nothing mints as a candidate any more — see the module docstring for why
    #    the three old promotion ladders were removed rather than tuned.
    # 6. Scrub-at-rest — matched raw above, store scrubbed (§2.2.6).
    return GateResult(
        accepted=True,
        reason="ok",
        instructor_id=instructor_id,
        status=PROC_STATUS_ACTIVE,
        scrubbed_quote=scrub_secrets(source_quote),
        matched_message_id=matched_message_id,
        respect=respect,
        attempted_by=attempted_by,
    )


# ── Dedup / conflict (design §4.3) — deterministic, no nightly adjudicator ────

_DEDUP_SAME = 0.90  # ≥ same-polarity → evidence increment, not a new row
_DEDUP_NEW = 0.75  # [0.75, 0.90) same-polarity → a genuinely different rule
_NEGATION = re.compile(
    r"\b(never|not|no|don'?t|do not|avoid|stop|нет|не|нельзя|больше не)\b",
    re.IGNORECASE,
)


def _polarity(text: str) -> str:
    """Coarse affirmative/negative polarity. Embeddings cluster "use X" and "never
    use X" together (high cosine), so polarity disambiguates a reinforcing repeat
    from a contradicting override (design §4.3)."""
    return "neg" if _NEGATION.search(text or "") else "pos"


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


def _dedup_decision(
    new_emb: list[float],
    new_polarity: str,
    is_instruction: bool,
    existing: list[tuple[Any, list[float], str]],
    scope: dict[str, Any] | None = None,
) -> tuple[str, Any]:
    """Decide how a fresh candidate relates to the existing rule store (design §4.3).

    Returns ``(action, matched_row|None)`` where action is:
    - ``increment`` — same-polarity cosine ≥0.90: reinforce an existing rule, no new row;
    - ``supersede`` — opposite-polarity cosine ≥0.75 from a trusted instruction: latest wins;
    - ``new`` — a genuinely different rule (or the store is empty).
    """
    best_same: tuple[float, Any] = (0.0, None)
    best_opp: tuple[float, Any] = (0.0, None)
    for row, emb, pol in existing:
        if getattr(row, "invalidated_at", None) is not None:
            continue
        if scope is not None:
            row_scope = ((row.typed_metadata or {}).get("procedural") or {}).get("scope") or {}
            if row_scope.get("person") != scope.get("person") or row_scope.get("venue", "any") != scope.get(
                "venue", "any"
            ):
                continue
        cos = _cosine(new_emb, emb)
        if pol == new_polarity:
            if cos > best_same[0]:
                best_same = (cos, row)
        elif cos > best_opp[0]:
            best_opp = (cos, row)
    if best_same[0] >= _DEDUP_SAME:
        return "increment", best_same[1]
    if is_instruction and best_opp[0] >= _DEDUP_NEW:
        return "supersede", best_opp[1]
    return "new", None


def _judge_group_safe(candidate: dict[str, Any], person: int | None) -> bool:
    """DB-436 mechanism 3 — mint-time group-safety judgment.

    Was hardcoded ``False`` (a placeholder), which combined with `_in_scope`'s group
    gate to make rule serving DEAD in practice: Kai's traffic is ~all group venues, so
    every minted rule was unservable from birth (0 serves in 7d over 1,680 assistant
    messages).

    A rule is group-safe only when it is IMPERSONAL: the extractor judged it carries no
    personal information AND it is not person-scoped. The ``person is None`` conjunct is
    the firewall — it is enforced in code, not trusted from the model, so a hallucinated
    ``group_safe: true`` on a ``this_person`` rule can never widen that rule's audience.
    Missing/garbage key → ``False``, i.e. the old behaviour.
    """
    if person is not None:
        return False
    return candidate.get("group_safe") is True


def _build_rule_metadata(
    candidate: dict[str, Any],
    gate: GateResult,
    *,
    now: Any,
    source_message_id: str | None,
    source_chat_id: int | None = None,
    source_topic_id: int | None = None,
) -> dict[str, Any]:
    """Assemble the ``typed_metadata.procedural`` blob for a minted rule (§1.1)."""
    strength = str(candidate.get("directive_strength") or "correction")
    scope = _candidate_scope_identity(candidate, gate.instructor_id)
    venue, person = scope["venue"], scope["person"]
    category = str(candidate.get("category") or "")
    # DB-487 — the communication-probe hold is GONE. It was a `return PENDING` stub
    # (procedural_comm_probe._probe_rule) that held 3 of the 6 real rules forever, and
    # the ELEPHANT harness it was waiting on never got wired to it. The behavioral
    # safety net for stance-shaped comm rules is now the serving ON/OFF eval plus the
    # biweekly ELEPHANT run — a check that actually executes.
    status = gate.status
    is_active = status == PROC_STATUS_ACTIVE
    probation_until = (now + timedelta(days=PROBATION_DAYS)).isoformat() if is_active else None
    return {
        "kind": "rule",
        "status": status,
        "source": "admin_instruction" if strength == "instruction" else "admin_correction",
        "category": category,
        "trigger_pattern": str(candidate.get("trigger") or ""),
        "guidance": str(candidate.get("rule") or ""),
        "expected_tools": candidate.get("expected_tools") or None,
        "scope": {"person": person, "venue": venue, "group_safe": _judge_group_safe(candidate, person)},
        "instructed_by": gate.instructor_id,
        # DB-487 — who taught this rule and how much that teaching is trusted. Read by
        # the reinforcement gain hook (G scales by respect/100) and by the DB-574
        # hardening review, which needs to see how each teacher's rules actually behave.
        "provenance": {"taught_by": gate.instructor_id, "respect": gate.respect},
        "source_message_ids": [source_message_id] if source_message_id else [],
        # Conversations that contributed evidence (§2.3), keyed on (chat_id, topic_id)
        # so different supergroup topics count separately. Since DB-487 this is pure
        # PROVENANCE — it feeds origin-conversation serving (DB-436 mechanism 1), not
        # an activation ladder. NULL chat (headless/unknown) contributes nothing.
        "source_chat_ids": ([k] if (k := _conv_key(source_chat_id, source_topic_id)) is not None else []),
        "source_quote": gate.scrubbed_quote,
        "execution_trace_ids": [],
        # Same key set as the reactivation reset in _apply_evidence — should_suspend
        # reads fail_streak, so a mint-time row missing it would present a different
        # ledger shape than a recovered one.
        "utility": {
            "s": 0,
            "f": 0,
            "n_served": 0,
            "streak": 0,
            "fail_streak": 0,
            "U": None,
            "last_outcome_at": None,
            "event_keys": [],
        },
        "evidence_count": 1,
        "recurrence_count": 0,
        "probation_until": probation_until,
        "trait_dimension": None,
        "valid_from": now.isoformat(),
        "valid_until": None,
        "supersedes": None,
        "promoted_at": now.isoformat() if is_active else None,
        "last_served_at": None,
        "last_followed_at": None,
    }


async def find_first_time_senders(
    session: AsyncSession,
    window_messages: list[ConversationMessage],
    admin_ids_str: str | None,
) -> set[int]:
    """Which non-admin senders in this window have NEVER written to Kai before it.

    Respect no longer comes from conversation history — the confirmed roster decides
    that — but "a sender Kai has never seen before is attempting to teach" is exactly
    the event surfaced for an operator roster decision, so the history probe
    earns its keep as a TRACKING signal instead of a scoring one.

    "Before" is strict: the window's own messages are already persisted, so "has any
    message at all" would report nobody as new. Fail-closed to the empty set — a DB
    hiccup must degrade the log line, never the gate (the roster alone decides minting).
    """
    try:
        from kai_memory import ports

        senders = {
            s
            for msg in window_messages
            if (s := getattr(msg, "sender_user_id", None)) is not None and not is_admin_user(s, admin_ids_str)
        }
        if not senders:
            return set()
        created = [c for msg in window_messages if (c := getattr(msg, "created_at", None)) is not None]
        if not created:
            return set()
        window_start = min(created)
        # The conversation-history port answers it (DB-843); in production a GROUP BY
        # backed by idx_conversation_sender_created (migration 104).
        async with session.begin_nested():
            seen_before = await ports.senders_seen_before(session, senders, window_start)
        return {s for s in senders if s not in seen_before}
    except Exception:
        logger.warning("procedural first-time-sender probe failed (tracking only)", exc_info=True)
        return set()


async def _load_existing_rules(session: AsyncSession) -> list[tuple[Any, list[float], str]]:
    from sqlalchemy import select

    from kai_memory.storage.models.memory_trace import MemoryTrace

    stmt = select(MemoryTrace).where(
        MemoryTrace.memory_type == "procedural",
        MemoryTrace.invalidated_at.is_(None),
        MemoryTrace.embedding.isnot(None),
        MemoryTrace.typed_metadata["procedural"]["kind"].astext.in_(PROC_KINDS),
    )
    rows = list((await session.execute(stmt)).scalars().all())
    out: list[tuple[Any, list[float], str]] = []
    for row in rows:
        proc = (row.typed_metadata or {}).get("procedural", {}) or {}
        emb = list(row.embedding) if row.embedding is not None else []
        out.append((row, emb, _polarity(str(proc.get("guidance") or ""))))
    return out


def _replay_scope_identity(scope_column: Any, scope: dict[str, Any]) -> tuple[Any, Any]:
    """Match dedup's person/venue identity without model-supplied disclosure hints."""
    from sqlalchemy import func, literal
    from sqlalchemy.dialects.postgresql import JSONB

    # Raw JSONB preserves integer vs string actors and explicit JSON null venues.
    # Missing person means null; only a missing venue defaults to "any".
    return (
        func.coalesce(scope_column["person"], literal(None, type_=JSONB)) == literal(scope.get("person"), type_=JSONB),
        func.coalesce(scope_column["venue"], literal("any", type_=JSONB))
        == literal(scope.get("venue", "any"), type_=JSONB),
    )


_REPLAY_BATCH_SIZE = 64
_ReplayKey = tuple[str, str, str, str, str]


def _replay_key(source_id: str, guidance: str, trigger: str, scope: dict[str, Any]) -> _ReplayKey:
    return (source_id, guidance, trigger, json.dumps(scope.get("person")), json.dumps(scope.get("venue", "any")))


def _consumed_teaching(key: _ReplayKey) -> dict[str, Any]:
    source, guidance, trigger, person, venue = key
    return {
        "source_message_id": source,
        "guidance": guidance,
        "trigger_pattern": trigger,
        "scope": {"person": json.loads(person), "venue": json.loads(venue)},
    }


def _persist_consumed_teaching(row: Any, key: _ReplayKey) -> None:
    """Retain exact transformed teachings through retirement, without tail eviction.

    Like source_message_ids, this provenance grows with retained teaching history;
    recurrence-event bounds must never discard durable replay protection. Legacy
    alternate wording never stored here cannot be reconstructed retroactively.
    """
    proc = dict((row.typed_metadata or {}).get("procedural") or {})
    consumed = list(proc.get("consumed_teachings") or [])
    teaching = _consumed_teaching(key)
    if teaching not in consumed:
        consumed.append(teaching)
        proc["consumed_teachings"] = consumed
        row.typed_metadata = {**(row.typed_metadata or {}), "procedural": proc}


def _candidate_scope_identity(candidate: dict[str, Any], instructor_id: int | None) -> dict[str, Any]:
    hint = str(candidate.get("scope_hint") or "global")
    return {
        "person": instructor_id if hint == "this_person" else None,
        "venue": "heartbeat" if hint == "heartbeat" else "wake" if hint == "wake" else "any",
    }


async def _load_replayed_keys(session: AsyncSession, keys: list[_ReplayKey]) -> set[_ReplayKey]:
    """One table scan per fixed-size key batch; never materialize stored tombstones.

    Predicate work remains O(stored rows * requested keys); this adds no index or
    history limit. Each result is a boolean array bounded by the batch size.
    """
    from sqlalchemy import and_, func, or_, select
    from sqlalchemy.dialects.postgresql import array
    from kai_memory.storage.models.memory_trace import MemoryTrace

    replayed: set[_ReplayKey] = set()
    proc = MemoryTrace.typed_metadata["procedural"]
    for start in range(0, len(keys), _REPLAY_BATCH_SIZE):
        batch = keys[start : start + _REPLAY_BATCH_SIZE]
        predicates = [
            or_(
                and_(
                    proc["source_message_ids"].contains([source_id]),
                    proc["guidance"].astext == guidance,
                    proc["trigger_pattern"].astext == trigger,
                    *_replay_scope_identity(proc["scope"], {"person": json.loads(person), "venue": json.loads(venue)}),
                ),
                proc["consumed_teachings"].contains(
                    [_consumed_teaching((source_id, guidance, trigger, person, venue))]
                ),
            )
            for source_id, guidance, trigger, person, venue in batch
        ]
        # A single aggregate over the relation avoids correlated per-key scans.
        statement = select(array([func.coalesce(func.bool_or(predicate), False) for predicate in predicates])).where(
            MemoryTrace.memory_type == "procedural"
        )
        flags = (await session.execute(statement)).scalar_one()
        if (
            not isinstance(flags, (list, tuple))
            or len(flags) != len(batch)
            or any(type(flag) is not bool for flag in flags)
        ):
            raise RuntimeError("invalid procedural replay batch result")
        replayed.update(key for key, flag in zip(batch, flags, strict=True) if flag)
    return replayed


def _remember_stored_replay(replayed: set[_ReplayKey], row: Any, source_id: str | None) -> None:
    proc = (row.typed_metadata or {}).get("procedural") or {}
    sources = proc.get("source_message_ids")
    guidance, trigger = proc.get("guidance"), proc.get("trigger_pattern")
    if (
        source_id
        and isinstance(sources, list)
        and source_id in sources
        and isinstance(guidance, str)
        and isinstance(trigger, str)
    ):
        replayed.add(_replay_key(source_id, guidance, trigger, proc.get("scope") or {}))


class ProceduralCaptureBusy(RuntimeError):
    """Another transaction owns capture; retry without consuming teaching history."""


async def lock_procedural_capture(session: AsyncSession) -> None:
    """Serialize capture until the outer transaction settles, including legacy L2.

    Extraction takes this before its first writes to preserve lock order, holding
    it through inference/pages. Capture-enabled store work trades throughput for
    consistent dedup; acquisition never queues database connections behind it.
    This does not expand the one-client-per-store contract.
    """
    from sqlalchemy import text

    acquired = (
        await session.execute(text("SELECT pg_try_advisory_xact_lock(:capture_key)"), {"capture_key": 0x4B414950524F43})
    ).scalar_one()
    if acquired is not True:
        from kai_memory import ports

        try:
            ports.telemetry.counter("memory.procedural.capture_busy", 1)
        except Exception as exc:
            logger.debug("procedural.capture_busy.metric_failed type=%s", type(exc).__name__)
        raise ProceduralCaptureBusy


async def capture_rules(
    session: AsyncSession,
    chat_ctx: ChatContext,
    candidates: list[dict[str, Any]],
    window_messages: list[ConversationMessage],
    admin_ids_str: str | None,
    *,
    respect_of: Callable[[int | None], int] | None = None,
    raise_on_error: bool = False,
    decisions: list[str] | None = None,
    replay_guard: bool = False,
    served_history: list[ConversationMessage] | None = None,
) -> int:
    """Gate + dedup + mint Part-4 behavior-rule candidates (design §2.2/§4.3).

    Savepoint-isolated, fail-open, behind ``procedural_rule_extraction_enabled``
    (Stage 1 ships 0.0 → no-op). Returns the number of rows minted/reinforced.
    ``window_messages`` alone anchors teaching trust; optional canonical
    ``served_history`` supplies prior exposure without widening that trust window.
    """
    if not cognitive_config.get_bool("procedural_rule_extraction_enabled"):
        return 0
    if not candidates:
        return 0
    minted = 0
    try:
        from kai_memory.procedural.procedural_recorder import normalize_signature
        from kai_memory.store.cognitive_memory import CognitiveMemoryService
        from kai_memory.store.embeddings import generate_embedding

        # Respect comes from CONFIG, not the DB — no query, nothing to fail.
        respect_of = respect_of or roster_respect_of(admin_ids_str, get_settings().procedural_respect_roster)
        # The first-time-sender probe is a read, resolved BEFORE the savepoint opens: a
        # failed statement inside a savepoint aborts it, and this probe swallows its own
        # errors, so a hiccup here would silently poison the writes that follow.
        first_time_senders = await find_first_time_senders(session, window_messages, admin_ids_str)
        async with session.begin_nested():
            await lock_procedural_capture(session)
            existing = await _load_existing_rules(session)
            service = CognitiveMemoryService(session)
            gates = (
                [
                    evaluate_candidate(candidate, window_messages, admin_ids_str, respect_of, diagnostics=False)
                    for candidate in candidates
                ]
                if replay_guard
                else []
            )
            requested = (
                list(
                    dict.fromkeys(
                        _replay_key(
                            gate.matched_message_id,
                            str(candidate.get("rule") or ""),
                            str(candidate.get("trigger") or ""),
                            _candidate_scope_identity(candidate, gate.instructor_id),
                        )
                        for candidate, gate in zip(candidates, gates, strict=True)
                        if gate.accepted and gate.matched_message_id
                    )
                )
                if replay_guard
                else []
            )
            replayed = await _load_replayed_keys(session, requested)
            for index, candidate in enumerate(candidates):
                gate = (
                    gates[index]
                    if replay_guard
                    else evaluate_candidate(candidate, window_messages, admin_ids_str, respect_of)
                )
                if not gate.accepted:
                    category = candidate.get("category")
                    safe_category = (
                        category if isinstance(category, str) and category in CATEGORY_ALLOWLIST else "invalid"
                    )
                    if replay_guard:
                        _log_gate_rejection(gate.reason, safe_category)
                    if decisions is not None:
                        decisions.append("rejected:" + gate.reason)
                    # DB-487 ride-along (a): every rejection is now visible. The bare
                    # `continue` here is why DB-436's 26 days of "capture is fine, nothing
                    # ever activates" had no log trail to read. Category is model-supplied
                    # and the quote is not logged, so this line carries no user content.
                    logger.info(
                        "procedural.rule.rejected reason=%s category=%s respect=%d threshold=%d",
                        gate.reason,
                        safe_category,
                        gate.respect,
                        gate.respect_threshold,
                    )
                    # A sender the assistant has NEVER seen before trying to teach is a
                    # roster decision waiting to happen, not just another rejection. Give
                    # it its own marker so it can be alerted on without grepping the noise.
                    if gate.respect < MINT_RESPECT_MIN and gate.attempted_by in first_time_senders:
                        logger.warning(
                            "procedural.rule.rejected_new_sender teacher=%s respect=%d — a sender with no "
                            "prior history attempted to teach; surface for a roster decision",
                            gate.attempted_by,
                            gate.respect,
                        )
                    continue
                trigger = str(candidate.get("trigger") or "")
                guidance = str(candidate.get("rule") or "")
                # First matched user row is the source message (idempotency key, §4.4);
                # computed ONCE by the gate, no second window walk.
                src_msg_id = gate.matched_message_id
                now = utc_now()
                meta = _build_rule_metadata(
                    candidate,
                    gate,
                    now=now,
                    source_message_id=src_msg_id,
                    source_chat_id=chat_ctx.chat_id,
                    source_topic_id=chat_ctx.topic_id,
                )
                if (
                    replay_guard
                    and src_msg_id
                    and _replay_key(src_msg_id, guidance, trigger, meta["scope"]) in replayed
                ):
                    if decisions is not None:
                        decisions.append("duplicate")
                    continue
                emb = await generate_embedding(normalize_signature(trigger) or guidance)
                if not emb:
                    if raise_on_error:
                        raise RuntimeError("procedural embedding unavailable")
                    continue
                polarity = _polarity(guidance)
                is_instruction = str(candidate.get("directive_strength")) == "instruction"
                action, matched = _dedup_decision(emb, polarity, is_instruction, existing, meta["scope"])

                if action == "increment":
                    # served=True ⇒ the rule WAS served this window and still got
                    # corrected (content failure); served=False ⇒ retrieval failure
                    # (§4.2 tripwire split).
                    served = _rule_was_served(
                        str(matched.id),
                        served_history if served_history is not None else window_messages,
                        before_message_id=src_msg_id if served_history is not None else None,
                    )
                    if _apply_evidence(
                        matched,
                        src_msg_id,
                        served=served,
                        source_chat_id=chat_ctx.chat_id,
                        source_topic_id=chat_ctx.topic_id,
                    ):
                        minted += 1
                        if decisions is not None:
                            decisions.append("reinforced")
                    elif decisions is not None:
                        decisions.append("duplicate")
                    if src_msg_id:
                        consumed_key = _replay_key(src_msg_id, guidance, trigger, meta["scope"])
                        _persist_consumed_teaching(matched, consumed_key)
                        if replay_guard:
                            replayed.add(consumed_key)
                    if replay_guard:
                        _remember_stored_replay(replayed, matched, src_msg_id)
                    continue
                if action == "supersede" and matched is not None:
                    # Latest trusted instruction wins deterministically at capture:
                    # the loser is invalidated (invalidate-don't-delete), the winner
                    # records `supersedes` below.
                    await service.invalidate_trace(matched.id, invalidation_source="l2_inline")
                    existing = [entry for entry in existing if entry[0].id != matched.id]

                if action == "supersede" and matched is not None:
                    meta["supersedes"] = str(matched.id)
                content = f"When {meta['trigger_pattern']} → {meta['guidance']}"
                row = await service.store_procedural_rule(
                    content,
                    embedding=emb,
                    source_user_id=gate.instructor_id,
                    typed_metadata={"procedural": meta},
                    source_chat_id=chat_ctx.chat_id,
                    source_topic_id=chat_ctx.topic_id,
                )
                if decisions is not None:
                    decisions.append("superseded" if action == "supersede" else "minted")
                if replay_guard:
                    _remember_stored_replay(replayed, row, src_msg_id)
                existing.append((row, emb, polarity))  # so intra-window dups fold
                minted += 1
                logger.info(
                    "procedural.rule.minted id=%s status=%s category=%s action=%s taught_by=%s respect=%d",
                    row.id,
                    meta["status"],
                    meta["category"],
                    action,
                    gate.instructor_id,
                    gate.respect,
                )
                if gate.respect < RESPECT_ADMIN:
                    # Teaching is open to the roster in this build precisely so its use can
                    # be observed before further hardening. A non-admin mint gets its own log line
                    # carrying the full shape of what was granted — teacher, score, rule,
                    # and the scope it will serve in.
                    logger.info(
                        "procedural.rule.minted_by_non_admin id=%s taught_by=%s respect=%d category=%s scope=%s",
                        row.id,
                        gate.instructor_id,
                        gate.respect,
                        meta["category"],
                        meta["scope"],
                    )
    except ProceduralCaptureBusy:
        raise
    except Exception as exc:
        # The savepoint rolled back every write in this call, so nothing persisted —
        # reset the count so the return value never over-reports (§ docstring contract).
        logger.warning("procedural rule capture failed type=%s", type(exc).__name__)
        minted = 0
        if decisions is not None:
            decisions.clear()
        if raise_on_error:
            raise
    return minted


def _rule_was_served(
    rule_id: str, window_messages: list[ConversationMessage], *, before_message_id: str | None = None
) -> bool:
    """True if this rule id was served on any assistant turn in the window (§4.2 —
    distinguishes a content failure from a retrieval failure for the tripwire)."""
    if before_message_id is not None:
        boundary = next(
            (
                index
                for index, message in enumerate(window_messages)
                if str(getattr(message, "id", "")) == before_message_id
            ),
            None,
        )
        if boundary is None:
            return False
        window_messages = window_messages[:boundary]
    for msg in window_messages:
        if getattr(msg, "role", None) != "assistant":
            continue
        served = (getattr(msg, "message_metadata", None) or {}).get("served_rule_ids") or []
        if rule_id in {str(s) for s in served}:
            return True
    return False


def _apply_evidence(
    matched: Any,
    src_msg_id: str | None,
    *,
    served: bool = False,
    source_chat_id: int | None = None,
    source_topic_id: int | None = None,
) -> bool:
    """Reinforce an existing rule with a fresh trusted occurrence (design §4.3/§4.4).

    Evidence is idempotent: a ``source_message_id`` already in ``source_message_ids``
    never double-increments. A hit on an already-active rule is a recurrence event
    whose ``served`` state (§4.2) separates a content failure (served) from a
    retrieval failure (not served). Since DB-487 there is no candidate→active ladder
    here — rules mint active — but a SUSPENDED rule that gets taught again returns
    straight to active on the same first-teach principle, with a clean ledger.
    Returns True when the row was mutated.
    """
    proc = dict((matched.typed_metadata or {}).get("procedural", {}) or {})
    ids = list(proc.get("source_message_ids") or [])
    if src_msg_id and src_msg_id in ids:
        return False  # idempotent — already counted
    if src_msg_id:
        ids.append(src_msg_id)
    proc["source_message_ids"] = ids
    proc["evidence_count"] = int(proc.get("evidence_count", 0) or 0) + 1
    # Track the conversations that taught this rule (§2.3), keyed on (chat_id, topic_id)
    # so different supergroup topics count separately. NULL chat contributes nothing.
    # Bounded at _DISTINCT_CONVS_TRACKED (2 GB EC2) as a most-recent-wins ring — see the
    # constant. Normalize any legacy bare-int entry to the "{chat_id}:" conv-key form
    # BEFORE dedup, so a stored [100] plus fresh chat-100 evidence resolves to the SAME
    # key (["100:"]) rather than being counted as a second conversation.
    conv_ids = normalize_conv_keys(proc.get("source_chat_ids") or [])
    conv_key = _conv_key(source_chat_id, source_topic_id)
    if conv_key is not None:
        # Move-to-end, then evict the OLDEST if over the bound. Never refuse the new key
        # (the pre-DB-487 behaviour): origin-conversation serving reads this list, so a
        # frozen list means a rule re-taught here can never serve here. Re-teaching in a
        # conversation already present also refreshes its recency, so the place you keep
        # correcting the rule is the last one to fall off the ring.
        if conv_key in conv_ids:
            conv_ids.remove(conv_key)
        conv_ids.append(conv_key)
        conv_ids = conv_ids[-_DISTINCT_CONVS_TRACKED:]
    proc["source_chat_ids"] = conv_ids
    if proc.get("status") == PROC_STATUS_ACTIVE:
        proc["recurrence_count"] = int(proc.get("recurrence_count", 0) or 0) + 1
        if served:
            # Content failure: the rule WAS surfaced and still got corrected. Record
            # the correcting message id for the nightly ledger's recurrence f+=2 (§4.2).
            # Utility is written ONLY nightly (#241 rule); capture leaves the pending
            # event. Tail-bounded so the pending list can't grow unbounded.
            if src_msg_id:
                rec = list(proc.get("recurrence_events") or [])
                rec.append(src_msg_id)
                proc["recurrence_events"] = rec[-_RECURRENCE_EVENTS_TAIL:]
            logger.error("procedural.recurrence.tripwire served=true id=%s", matched.id)
        else:
            # Not-served recurrence: the rule was out of scope for that turn (or serving
            # was off), so the correction is not the rule's fault — NO utility penalty
            # (design refinement, team-lead ruling). Under serve-all this should become
            # rare; when it persists it points at the scope filter, not the rule.
            # (Log token kept verbatim — prod queries key off it.)
            logger.warning("procedural.recurrence.retrieval_miss id=%s", matched.id)
    elif proc.get("status") in REACTIVATABLE_STATUSES:
        # Fresh trusted teaching reactivates (DB-487 first-teach). Pre-DB-487 this
        # dropped a suspended rule to `candidate` and let the 2-distinct-conversation
        # ladder re-promote it — but that ladder is gone, so recovering to `candidate`
        # would now be a permanent grave. `candidate` is here only for legacy rows the
        # activation backfill did not cover (e.g. one minted mid-deploy).
        #
        # Reset the ACT-R ledger to a CLEAN slate — otherwise the stale utility floor
        # (s/f/n_served/U from before suspension) would re-suspend the rule on the very
        # next nightly run with zero new failures (dead-recovery bug). The trap: a bare
        # reset re-exposes the rule to the SAME pre-suspension traces still inside the
        # 2-day lookback window, which would re-gather and re-contaminate the fresh
        # ledger. So we also stamp a per-rule `scored_after` watermark = recovery time;
        # the nightly ledger scores ONLY traces created after it. Probation restarts too,
        # so a reactivated roster-taught rule is one failure away from suspension again —
        # that is the brake that makes immediate reactivation safe. (An admin-taught rule
        # is exempt from status demotion since DB-648, so it is only ever here if the
        # ledger suspended it before that guard shipped.) The 30-day retire clock resets
        # with suspended_at.
        now = utc_now()
        previous_status = str(proc.get("status"))
        proc["status"] = PROC_STATUS_ACTIVE
        proc.pop("suspended_at", None)
        proc["promoted_at"] = now.isoformat()
        proc["probation_until"] = (now + timedelta(days=PROBATION_DAYS)).isoformat()
        proc["scored_after"] = now.isoformat()
        # ...and drop any PENDING recurrence. `scored_after` filters execution TRACES
        # only; gather_events also emits recurrence events straight from this list, so a
        # pre-suspension recurrence left here would re-apply f+=2 to the fresh ledger and
        # — with probation restarted — suspend the rule again on the very next nightly
        # pass. That is the dead-recovery bug wearing a different hat.
        proc.pop("recurrence_events", None)
        # DB-629: `compliance_events` has the identical property — a pre-suspension
        # plain-turn `violated` verdict left here would re-apply to the reset ledger and,
        # with probation restarted, suspend the rule again on the very next nightly pass,
        # on zero new evidence.
        proc.pop("compliance_events", None)
        proc["utility"] = {
            "s": 0,
            "f": 0,
            "n_served": 0,
            "streak": 0,
            "fail_streak": 0,
            "U": None,
            "last_outcome_at": None,
            "event_keys": [],
        }
        logger.info(
            "procedural.rule.reinforced id=%s recovered=%s->active (ledger reset + watermark + probation)",
            matched.id,
            previous_status,
        )
    # Reassign so SQLAlchemy detects the JSONB change.
    meta = dict(matched.typed_metadata or {})
    meta["procedural"] = proc
    matched.typed_metadata = meta
    return True


async def record_message_anchored_compliance(
    session: AsyncSession,
    msg_updates: dict[str, tuple[str, MessageAnchor]],
    admin_ids: str | None,
    *,
    actor_is_trusted: Callable[[int | None, int | None], bool] | None = None,
) -> int:
    """Record L2 Part-5 verdicts from PLAIN (non-tool) turns as pending ledger events on
    the rule rows (DB-629). Returns the number of rules written. Runs inside the caller's
    savepoint; the caller is fail-open.

    This lives beside ``_apply_evidence``'s ``recurrence_events`` write because it is the
    same move: capture-time code parks a keyed, trust-scoped event on the rule, and the
    nightly ledger is still the only thing that touches ``utility`` (#241 containment).

    Why it exists: a tool-using turn carries an ``execution_trace_id`` and its verdict
    lands on the trace, but a plain reply — where the anti-meta-commentary and tone rules
    actually live — has ``served_rule_ids`` and nothing else. Every correct ``followed``
    verdict on one used to be computed, logged, and dropped for want of somewhere to put
    it, which is half of why the utility ledger sat empty for 61 days.

    Trust-scoped exactly like the trace path (§4.2): a verdict counts only when the turn's
    triggering human is an admin or the rule's own ``scope.person``, so a stranger in a
    group can neither manufacture reinforcement nor demote someone else's rule."""
    import uuid

    from sqlalchemy import select
    from sqlalchemy.orm import defer

    from kai_memory.storage.models.memory_trace import MemoryTrace

    rule_uuids: list[uuid.UUID] = []
    for rid in msg_updates:
        try:
            rule_uuids.append(uuid.UUID(rid))
        except ValueError:
            continue
    if not rule_uuids:
        return 0

    written = 0
    rows = (
        await session.execute(
            select(MemoryTrace)
            # Nothing below reads the vector; keep Vector(1536) off the wire (2 GB box).
            .options(defer(MemoryTrace.embedding))
            # A rule retired between serving and extraction must not receive an event:
            # the nightly pass skips invalidated rows, so the entry would never be
            # gathered AND never aged out — a permanent row in typed_metadata recording
            # a verdict about a rule that no longer stands.
            .where(MemoryTrace.id.in_(rule_uuids), MemoryTrace.invalidated_at.is_(None))
            # This is a read-modify-write of the whole typed_metadata document, so two
            # extraction passes finishing at once (different conversations, same standing
            # rule — the normal case under serve-all) would otherwise read the same
            # pending list and overwrite each other's event. Ordering by id makes the
            # lock acquisition deterministic, so two such writers queue instead of
            # deadlocking. NOTE: the nightly ledger and `_apply_evidence` are the other
            # full-document writers of this key and do NOT take this lock yet — the
            # remaining race is extraction-vs-nightly, tracked as follow-up work rather
            # than widened here (CodeRabbit review round 2).
            .order_by(MemoryTrace.id)
            .with_for_update()
        )
    ).scalars()
    for row in rows:
        entry = msg_updates.get(str(row.id))
        if entry is None:
            # Defensive: a KeyError here would abort the caller's savepoint and take the
            # trace-anchored verdicts down with it, which is a far worse outcome than
            # skipping one rule whose id spelling did not round-trip.
            logger.warning("procedural.compliance.unmatched_rule rule=%s — skipped", row.id)
            continue
        verdict, anchor = entry
        meta = dict(row.typed_metadata or {})
        proc = dict(meta.get("procedural") or {})
        scope_person = (proc.get("scope") or {}).get("person")
        is_trusted = actor_is_trusted or (lambda actor, person: trusted_actor(actor, person, admin_ids))
        if not is_trusted(anchor.source_user_id, scope_person):
            logger.info("procedural.compliance.untrusted rule=%s verdict=%s — not recorded", row.id, verdict)
            continue
        events = [e for e in (proc.get("compliance_events") or []) if isinstance(e, dict)]
        if any(str(e.get("message_id")) == anchor.message_id for e in events):
            continue  # idempotent: this anchor already carries an event
        events.append(
            {
                "message_id": anchor.message_id,
                "verdict": verdict,
                # The nightly pass ages entries out on this timestamp, which is what makes
                # the idempotency key safe to prune: a key is released only once no
                # extraction can still re-produce its event.
                "at": (anchor.at or utc_now()).isoformat(),
            }
        )
        proc["compliance_events"] = events[-_COMPLIANCE_EVENTS_TAIL:]
        meta["procedural"] = proc
        row.typed_metadata = meta
        written += 1
        logger.info(
            "procedural.compliance.message_anchored rule=%s msg=%s verdict=%s pending=%d",
            row.id,
            anchor.message_id,
            verdict,
            len(proc["compliance_events"]),
        )
    return written
