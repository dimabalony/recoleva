"""Phase C §4.2/§4.4 — the nightly reinforcement/demotion ledger.

The **only** writer of ``typed_metadata.procedural.utility``. Runs once per nightly
consolidation pass, gated by ``procedural_reinforcement_enabled``. It reads
yesterday's served ``execution_traces`` and the rule store, turns **rule-level
compliance events** (never overall turn outcome) into an ACT-R utility, and demotes
rules whose utility floors out.

**Who taught the rule decides what a failure means (DB-648).** A rule a full-trust
person taught (``is_protected_from_demotion``) still loses utility, and that utility
still ranks it for cap eviction, but it is never suspended or retired for non-compliance.
That the system keeps failing to obey it is a serving bug, so when it crosses a trigger
that would have suspended it, the pass logs ``procedural.rule.ignored`` at ERROR, once
per episode.

**#241 containment, mechanically** (the weights-runaway cautionary tale):

- nightly-only, bounded; **fixed versioned module constants** (``G``/``C``/θ) shipped
  with the code — nothing self-adjusts, nothing tunes itself from its own output;
- **zero contact** with ``retrieval_config``, ``SCORE_COMPONENTS``, L1 weights, or
  trace salience — it reads ``execution_traces`` + writes one JSONB sub-key on
  ``memory_traces`` procedural rows and nothing else;
- idempotent: every event carries a deterministic key (§4.4); duplicate/overlapping
  nightly runs re-present the same events and skip already-applied keys, so the
  ledger is stable under repeated runs.

Credit assignment (Codex blocker C-2): a compliance/outcome event counts only from
an **admin or the rule's ``scope.person``** (untrusted praise can't manufacture
reinforcement); the deterministic tool check is inherently trusted (it is code, not
a user utterance). Turn outcomes feed trace labels — never this ledger.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.host_calls import get_settings, utc_now
from kai_memory.storage.models.execution_trace import ExecutionTrace
from kai_memory.storage.models.memory_trace import MemoryTrace
from kai_memory.store import cognitive_config
from kai_memory.procedural.procedural_scope import (
    LEDGER_VERDICTS,
    PROC_KINDS,
    PROC_STATUS_ACTIVE,
    PROC_STATUS_SUSPENDED,
    VERDICT_FOLLOWED,
    VERDICT_VIOLATED,
    trusted_actor,
)

logger = logging.getLogger("app.services.memory.procedural_reinforcement")

# ── Versioned module constants (design §1.4/§4.2 — NOT config rows, #241 rule) ─
THETA_DEMOTE = 0.4  # utility floor: below this (after min evidence) an established rule suspends
G_BY_SOURCE: dict[str, float] = {"admin_instruction": 1.0, "admin_correction": 0.9, "l3_promotion": 0.7}
_G_DEFAULT = 0.9  # unknown provenance → correction-grade gain
C_PER_KTOKEN = 0.3  # cost term: tokens(rule)/1000 x 0.3
MIN_SERVED_FOR_UTILITY_DEMOTION = 5  # no utility demotion before n_served ≥ 5 (§4.2)
CONSECUTIVE_FAILURES_TO_SUSPEND = 2  # established rules: 2 consecutive failures
SUSPEND_RETIRE_DAYS = 30  # suspended → retired (invalidate-don't-delete) after 30d
LOOKBACK_DAYS = 2  # served-trace window; idempotency keys make the overlap safe
MAX_RULES = 1000  # supported eligible rule/habit store; larger stores refuse the whole pass
MAX_TRACES = 5000  # bound the nightly served-trace scan

# A row with no ``source`` is read as a correction — for gain AND for the DB-648 guard,
# so the two can never disagree about what kind of rule it is. rule_capture always
# stamps a source, so this only covers malformed rows.
_SOURCE_DEFAULT = "admin_correction"
# DB-648 — the sources a PERSON taught, through rule_capture (the only rule writer, which
# mints exactly these two). For these, a ``violated`` verdict means the system failed to
# honour the instruction, not that the instruction is unwanted. Taught at full trust
# (``is_protected_from_demotion``) they still lose utility — eviction ranking under
# SERVE_MAX_RULES reads it — but are never auto-suspended or retired for it.
# ``l3_promotion`` (reserved: priced in G_BY_SOURCE, minted by no code today) is
# deliberately absent: for a rule nobody asked for, "nobody acts on this" genuinely does
# mean noise, so it keeps the demotion path unchanged.
USER_TAUGHT_SOURCES: frozenset[str] = frozenset({"admin_instruction", "admin_correction"})
# The respect score of a full-trust teacher — rule_capture.RESPECT_ADMIN, restated here
# because this module must not import rule_capture (it pulls in the Telegram layer); a
# unit test pins the two together.
FULL_TRUST_RESPECT = 100
# The demotion triggers, named so a guarded crossing can say which one it would have been.
TRIGGER_PROBATION = "probation"  # on probation and a failure applied this run
TRIGGER_FAIL_STREAK = "fail_streak"  # CONSECUTIVE_FAILURES_TO_SUSPEND consecutive failures
TRIGGER_UTILITY = "utility"  # U < THETA_DEMOTE after MIN_SERVED_FOR_UTILITY_DEMOTION servings
# A protected rule found already suspended — only reachable for a row suspended before the
# guard existed. It is held back from retirement and paged once so it can be re-taught.
# Named apart from the three above so it greps apart from a would-have-suspended crossing.
TRIGGER_LEGACY_SUSPENDED = "legacy_suspended"
# typed_metadata.procedural key that dedupes the ``procedural.rule.ignored`` page: set when
# a protected rule first crosses a trigger, cleared once it is followed again.
IGNORED_MARKER_KEY = "ignored"

# The producer/consumer contract lives in procedural_scope so a rename cannot silently
# desynchronise the judge prompt from this ledger (see PERSISTED_VERDICTS there).
_FOLLOWED = VERDICT_FOLLOWED
_VIOLATED = VERDICT_VIOLATED
_RECURRENCE = "recurrence"  # a trusted correction targeting an active rule → f+=2
_NUDGE_REJECTED = "nudge_rejected"  # a rejected wake carrying this habit's id → f+=2 (Stage 3)
# DB-629 — event-key kind for a verdict anchored to a PLAIN assistant turn. Distinct from
# ``compliance:`` because the two are pruned against different windows (trace ids vs the
# rule's own pending list); sharing one prefix would let the trace prune evict a message
# key whose event is still live, and the next pass would re-apply it.
_COMPLIANCE_MSG = "compliance_msg"


@dataclass(frozen=True)
class Event:
    """One rule-level ledger event with its idempotency key (design §4.4)."""

    verdict: str  # followed | violated | recurrence | nudge_rejected
    key: str  # "{event_type}:{source_id}" — dedup across duplicate nightly runs


@dataclass
class LedgerDelta:
    changed: bool
    failed: bool  # any failure event applied this run (drives probation demotion)


# ── Pure utility + event application (unit-tested; no DB, no LLM) ─────────────


def estimate_tokens(text: str) -> int:
    """Cheap, stable token estimate (≈4 chars/token) for the cost term. A rule's
    guidance is ≤300 chars, so C is tiny — exact tokenization isn't worth a dep."""
    return max(1, len(text or "") // 4)


def respect_multiplier(respect: int | None) -> float:
    """Gain multiplier for the instructor's respect score: ``respect / 100`` (DB-487).

    This is the counterweight to opening teaching at respect >= 75. An admin-taught rule
    (100) keeps full gain, so nothing about existing rules changes; a roster-taught rule
    (75) earns at 0.75x, which means it needs a *better* observed track record than an
    admin rule to survive cap eviction. Only the roster-taught rule can also be demoted by
    it: since DB-648 a rule whose multiplier is 1.0 is exempt from status demotion
    (``is_protected_from_demotion``), so for it utility is ranking only, while a 75-taught
    rule must clear the demotion floor too. Trust is graded, not binary — the ledger, not
    the gate, is where a weaker teacher's rules are held to a higher bar.

    A missing score (pre-DB-487 rows, or a rule minted before provenance existed) maps to
    1.0 rather than 0.0: absence of evidence is not a demotion, and treating it as one
    would retroactively suspend every legacy rule on the next nightly pass. Values are
    clamped so a malformed roster entry can never inflate gain above admin level.
    """
    if respect is None:
        return 1.0
    try:
        scaled = float(respect) / 100.0
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, scaled))


def compute_utility(s: int, f: int, source: str, token_count: int, *, respect: int | None = None) -> float:
    """``U = P·G - C`` (design §4.2). ``P`` Laplace-smoothed, ``G`` by provenance scaled
    by the teacher's respect (``respect/100`` — DB-487), ``C`` a small token-cost
    penalty. Demotion-only — there is no θ_promote."""
    p = (s + 1) / (s + f + 2)
    g = G_BY_SOURCE.get(source, _G_DEFAULT) * respect_multiplier(respect)
    c = (token_count / 1000.0) * C_PER_KTOKEN
    return p * g - c


def apply_events(
    util: dict[str, Any],
    *,
    source: str,
    token_count: int,
    events: list[Event],
    now_iso: str,
    respect: int | None = None,
) -> LedgerDelta:
    """Apply new events to a utility ledger IN PLACE, idempotently (design §4.2/§4.4).

    Skips any event whose key is already in ``util.event_keys``; recomputes ``U``
    only when something changed. followed → s+1/streak+1; violated → f+1/streak=0;
    recurrence & nudge_rejected → f+2/streak=0. n_served counts only *servings*
    (followed/violated), never corrections. Returns whether anything changed and
    whether a failure applied (for probation demotion)."""
    keys = list(util.get("event_keys") or [])
    key_set = set(keys)
    s = int(util.get("s", 0) or 0)
    f = int(util.get("f", 0) or 0)
    streak = int(util.get("streak", 0) or 0)
    fail_streak = int(util.get("fail_streak", 0) or 0)
    n_served = int(util.get("n_served", 0) or 0)

    changed = False
    failed = False
    for ev in events:
        if ev.key in key_set:
            continue
        key_set.add(ev.key)
        keys.append(ev.key)
        changed = True
        if ev.verdict == _FOLLOWED:
            s += 1
            streak += 1
            fail_streak = 0
            n_served += 1
        elif ev.verdict == _VIOLATED:
            f += 1
            streak = 0
            fail_streak += 1
            n_served += 1
            failed = True
        elif ev.verdict in (_RECURRENCE, _NUDGE_REJECTED):
            f += 2  # a trusted correction / rejected nudge stings 2x a single violation
            streak = 0
            fail_streak += 1
            failed = True

    if not changed:
        return LedgerDelta(changed=False, failed=False)

    util["s"] = s
    util["f"] = f
    util["streak"] = streak
    util["fail_streak"] = fail_streak
    util["n_served"] = n_served
    util["U"] = compute_utility(s, f, source, token_count, respect=respect)
    util["last_outcome_at"] = now_iso
    # Keep the FULL applied-key set here — a fixed tail would evict keys whose trace
    # is still inside the lookback window and re-apply them next run (inflating s/f).
    # run_reinforcement prunes this to the current window's keys, so it stays bounded
    # by the window size, never by an arbitrary constant (design §4.4).
    util["event_keys"] = keys
    return LedgerDelta(changed=True, failed=failed)


def prune_event_keys(
    util: dict[str, Any],
    *,
    window_trace_ids: set[str],
    pending_recurrence: set[str],
    pending_compliance: set[str] | None = None,
) -> bool:
    """Drop applied-event keys that can never be re-gathered — compliance/nudge keys
    whose trace has aged out of the lookback window, message-anchored compliance keys
    whose entry has left ``compliance_events``, and recurrence keys already pruned from
    ``recurrence_events``. Keeps ``event_keys`` bounded by the window (not a fixed tail)
    WITHOUT ever evicting a key that could re-occur (design §4.4).
    Returns True if the list changed."""
    pending_compliance = pending_compliance or set()
    before = util.get("event_keys") or []
    kept: list[str] = []
    for k in before:
        kind, _, sid = k.partition(":")
        if kind in ("compliance", "nudge"):
            if sid in window_trace_ids:
                kept.append(k)
        elif kind == _COMPLIANCE_MSG:
            # DB-629 message-anchored compliance. Retained while its entry is still in
            # ``compliance_events`` — which the runner ages out on the SAME window as a
            # trace. Dropping the key any earlier would let the next pass re-apply the
            # same verdict and inflate s/f; that is exactly the double-count the event
            # key exists to prevent.
            if sid in pending_compliance:
                kept.append(k)
        elif kind == "recurrence":
            if sid in pending_recurrence:
                kept.append(k)
        else:
            kept.append(k)  # unknown key kind — retain defensively
    if kept != before:
        util["event_keys"] = kept
        return True
    return False


def _on_probation(proc: dict[str, Any], now: datetime) -> bool:
    raw = proc.get("probation_until")
    if not raw:
        return False
    try:
        until = datetime.fromisoformat(str(raw))
    except ValueError:
        return False
    if until.tzinfo is None:
        until = until.replace(tzinfo=now.tzinfo)
    return now < until


def rule_source(proc: dict[str, Any]) -> str:
    """The rule's provenance source, with the same missing-value default the gain uses."""
    return str(proc.get("source") or _SOURCE_DEFAULT)


def is_protected_from_demotion(proc: dict[str, Any]) -> bool:
    """True for a rule a full-trust person taught: never auto-suspended or retired for
    non-compliance (DB-648).

    Non-compliance on a rule someone asked for is a bug report about the serving path,
    not evidence the rule is unwanted — suspending it stops it being served, compliance
    gets worse, and the person has to teach it again (the DB-646 duplicates).

    Two conditions, both already in the data:

    - ``source`` is user-taught (``USER_TAUGHT_SOURCES``);
    - the recorded respect is exactly full trust (``FULL_TRUST_RESPECT``, the admin), or
      absent. Protecting the absent case is an inference from history, not from the row:
      teaching was admin-only until DB-487 started stamping ``provenance.respect``, so
      every row without it was admin-taught. A roster-taught rule (75) keeps today's
      demotion — DB-487 made the demotion floor the counterweight to opening teaching at
      75, and this guard does not quietly remove it (whether roster rules deserve the
      same protection is an open product question; answering yes is this one condition).

    Deliberately NOT ``respect_multiplier(...) == 1.0``: that clamps 150 and maps garbage
    to 1.0 so a malformed score cannot demote. Here the safe direction is reversed — a
    malformed or out-of-range score must not WIDEN protection, so it falls back to
    today's demotion (``typed_metadata`` is unconstrained JSONB).
    """
    if rule_source(proc) not in USER_TAUGHT_SOURCES:
        return False
    respect = (proc.get("provenance") or {}).get("respect")
    if respect is None:
        return True
    if isinstance(respect, bool):
        return False  # bool is an int subclass; True must not read as a score
    try:
        return float(respect) == FULL_TRUST_RESPECT
    except (TypeError, ValueError):
        return False


def demotion_trigger(proc: dict[str, Any], util: dict[str, Any], now: datetime, *, failed_this_run: bool) -> str | None:
    """Which active → suspended trigger this ledger crosses, IGNORING who taught the rule
    (design §4.2). Probation: a single failure this run. Established: 2 consecutive
    failures OR ``U < θ`` after ``n_served ≥ 5`` (no utility demotion before minimum
    evidence). ``None`` when nothing fires or the rule is not active."""
    if proc.get("status") != PROC_STATUS_ACTIVE:
        return None
    if _on_probation(proc, now) and failed_this_run:
        return TRIGGER_PROBATION
    if int(util.get("fail_streak", 0) or 0) >= CONSECUTIVE_FAILURES_TO_SUSPEND:
        return TRIGGER_FAIL_STREAK
    u = util.get("U")
    n_served = int(util.get("n_served", 0) or 0)
    if u is not None and n_served >= MIN_SERVED_FOR_UTILITY_DEMOTION and float(u) < THETA_DEMOTE:
        return TRIGGER_UTILITY
    return None


def should_suspend(proc: dict[str, Any], util: dict[str, Any], now: datetime, *, failed_this_run: bool) -> bool:
    """Active → suspended (design §4.2): a demotion trigger fires AND the rule is not
    protected. A protected rule escalates instead (``update_ignored_marker``)."""
    if is_protected_from_demotion(proc):
        return False
    return demotion_trigger(proc, util, now, failed_this_run=failed_this_run) is not None


def ignored_trigger(proc: dict[str, Any], util: dict[str, Any], now: datetime, *, failed_this_run: bool) -> str | None:
    """For a PROTECTED rule: the demotion that would have happened, if any — a trigger on
    an active rule, or ``legacy_suspended`` for a row demoted before the guard existed."""
    if proc.get("status") == PROC_STATUS_SUSPENDED:
        return TRIGGER_LEGACY_SUSPENDED
    return demotion_trigger(proc, util, now, failed_this_run=failed_this_run)


def update_ignored_marker(
    proc: dict[str, Any], util: dict[str, Any], trigger: str | None, now_iso: str
) -> tuple[bool, bool]:
    """Dedupe the ``procedural.rule.ignored`` page to ONCE per episode (DB-648).

    An episode starts when a protected rule first crosses a trigger: the marker is
    stamped and the caller pages. While it is set, further crossings are silent — the
    rule stays over the fail-streak or utility line night after night, and a nightly
    page for the same fact is noise. It ends only when the rule is back in compliance:
    no trigger crossing AND the latest ledger event was a ``followed`` (``fail_streak``
    is 0). A quiet night or a duplicate nightly run therefore cannot re-arm it.

    Mutates ``proc`` in place. Returns ``(page, changed)``."""
    marker = proc.get(IGNORED_MARKER_KEY)
    if trigger is not None:
        if marker:
            return False, False
        proc[IGNORED_MARKER_KEY] = {"at": now_iso, "trigger": trigger}
        return True, True
    if marker and int(util.get("fail_streak", 0) or 0) == 0:
        proc.pop(IGNORED_MARKER_KEY, None)
        return False, True
    return False, False


def _suspended_too_long(proc: dict[str, Any], now: datetime) -> bool:
    if proc.get("status") != PROC_STATUS_SUSPENDED:
        return False
    if is_protected_from_demotion(proc):
        # DB-648: retirement is the second half of the same demotion. A protected rule
        # can only be suspended if the ledger demoted it before this guard shipped; it
        # waits for re-teaching (rule_capture reactivates it) and is never invalidated.
        return False
    raw = proc.get("suspended_at")
    if not raw:
        return False
    try:
        since = datetime.fromisoformat(str(raw))
    except ValueError:
        return False
    if since.tzinfo is None:
        since = since.replace(tzinfo=now.tzinfo)
    return (now - since) >= timedelta(days=SUSPEND_RETIRE_DAYS)


# ── Event gathering from the day's served traces (trust-scoped) ──────────────


# The credit boundary lives in procedural_scope alongside the verdict vocabulary, so the
# nightly ledger and the capture-time compliance writer cannot drift apart on who is
# allowed to move a rule's utility. Re-exported under the old private name so the many
# call sites below read unchanged.
_trusted_actor = trusted_actor


def _resolve_compliance(
    trace: ExecutionTrace,
    rule_id: str,
    expected_tools: list[str],
    scope_person: int | None,
    admin_ids: str | None,
) -> str | None:
    """A single verdict for (rule, trace): the L2 Part-5 judgment if present (trust-
    scoped to admin/scope.person), else the deterministic tool check (inherently
    trusted — code, not an utterance) which only ever yields ``followed`` (§4.1)."""
    signals = trace.success_signals or {}
    verdict = (signals.get("rule_compliance") or {}).get(rule_id)
    if verdict in (_FOLLOWED, _VIOLATED) and _trusted_actor(trace.source_user_id, scope_person, admin_ids):
        return str(verdict)
    # An untrusted model verdict is simply IGNORED — but must NOT suppress the
    # inherently-trusted deterministic tool check below (that evidence is code-derived,
    # not a user utterance), so we fall through rather than returning None here.
    if expected_tools:
        tools = {s.get("tool") for s in (trace.steps or []) if isinstance(s, dict)}
        if set(expected_tools) <= tools:
            return _FOLLOWED
    return None


def after_watermark(traces: list[ExecutionTrace], scored_after_iso: str | None) -> list[ExecutionTrace]:
    """Drop traces at/before a rule's ``scored_after`` recovery watermark (design §4.2)
    so pre-recovery evidence can never re-contribute to the reset ledger. A rule with
    no watermark scores its full window unchanged."""
    if not scored_after_iso:
        return traces
    try:
        cutoff = datetime.fromisoformat(str(scored_after_iso))
    except ValueError:
        return traces
    # Normalize BOTH sides to UTC-aware: a legacy naive watermark vs an aware
    # created_at (or vice-versa) would otherwise raise TypeError on comparison and
    # abort the whole nightly pass. Production timestamps are UTC.
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=UTC)
    out: list[ExecutionTrace] = []
    for t in traces:
        created = getattr(t, "created_at", None)
        if created is None:
            continue
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        if created > cutoff:
            out.append(t)
    return out


def _entry_after_watermark(entry: dict[str, Any], scored_after_iso: str | None) -> bool:
    """True when a pending compliance entry post-dates a rule's ``scored_after`` recovery
    watermark — the ``after_watermark`` rule, for evidence that is not a trace. An
    undatable entry is refused rather than admitted: unlike a trace, it carries its own
    timestamp, so a missing one means it cannot be placed relative to a recovery."""
    if not scored_after_iso:
        return True
    try:
        cutoff = datetime.fromisoformat(str(scored_after_iso))
        at = datetime.fromisoformat(str(entry.get("at")))
    except (TypeError, ValueError):
        return False
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=UTC)
    if at.tzinfo is None:
        at = at.replace(tzinfo=UTC)
    return at > cutoff


def gather_events(
    row: MemoryTrace,
    proc: dict[str, Any],
    traces: list[ExecutionTrace],
    admin_ids: str | None,
) -> list[Event]:
    """Collect this rule's new ledger events from the day's served traces + its own
    capture-recorded recurrences + (Stage-3 dormant) rejected-nudge signals."""
    rule_id = str(row.id)
    scope_person = (proc.get("scope") or {}).get("person")
    expected_tools = [t for t in (proc.get("expected_tools") or []) if t]
    events: list[Event] = []

    for trace in traces:
        verdict = _resolve_compliance(trace, rule_id, expected_tools, scope_person, admin_ids)
        if verdict:
            events.append(Event(verdict=verdict, key=f"compliance:{trace.id}"))
        # Rejected-nudge (§4.2). Populated by the Stage-3 wake-care loop
        # (wake_events.origin_rule_ids → trace signal); dormant until then. Trust-
        # scoped like compliance — a stranger's rejected nudge can't demote a rule.
        signals = trace.success_signals or {}
        rejected_here = signals.get("nudge_outcome") == "rejected" and rule_id in {
            str(x) for x in (signals.get("origin_rule_ids") or [])
        }
        if rejected_here and _trusted_actor(trace.source_user_id, scope_person, admin_ids):
            events.append(Event(verdict=_NUDGE_REJECTED, key=f"nudge:{trace.id}"))

    # Compliance on a PLAIN turn (DB-629): a served reply with no execution trace, so L2
    # recorded the verdict on the rule itself. Trust was already scoped at write time —
    # the assistant row has no sender, so the credit boundary has to resolve through the
    # triggering human while the window is still in hand.
    scored_after = proc.get("scored_after")
    for entry in proc.get("compliance_events") or []:
        if not isinstance(entry, dict):
            continue
        verdict = str(entry.get("verdict") or "")
        msg_id = str(entry.get("message_id") or "")
        if not (msg_id and verdict in LEDGER_VERDICTS):
            continue
        # The recovery watermark applies here too. `after_watermark` filters TRACES; a
        # pending plain-turn entry reaches the ledger straight off the rule row, so
        # without this a pre-suspension verdict would score against the reset ledger of a
        # just-reactivated rule and re-suspend it during probation. Capture pops the list
        # on reactivation (rule_capture._apply_evidence) — this is the second lock.
        if _entry_after_watermark(entry, scored_after):
            events.append(Event(verdict=verdict, key=f"{_COMPLIANCE_MSG}:{msg_id}"))

    # Recurrence: a trusted correction targeting this (active) rule, recorded at
    # capture time as a message id (§4.3). Applied here (utility is nightly-only).
    for msg_id in proc.get("recurrence_events") or []:
        events.append(Event(verdict=_RECURRENCE, key=f"recurrence:{msg_id}"))
    return events


def prune_compliance_events(proc: dict[str, Any], now: datetime) -> bool:
    """Age message-anchored compliance entries out of the rule's pending list on the SAME
    window the served-trace scan uses (DB-629). Returns True if the list changed.

    Age, not applied-ness, is the eviction rule. Dropping an entry the moment its event
    applied would prune its idempotency key too, and an L2 re-run over the same window
    would then re-append and re-count it. Holding entries for the full lookback window
    means a key is only ever released once no extraction can still produce it."""
    before = proc.get("compliance_events") or []
    if not before:
        return False
    cutoff = now - timedelta(days=LOOKBACK_DAYS)
    kept: list[dict[str, Any]] = []
    for entry in before:
        if not isinstance(entry, dict):
            continue  # malformed row — drop it rather than carry it forever
        try:
            at = datetime.fromisoformat(str(entry.get("at")))
        except (TypeError, ValueError):
            continue  # undatable entry can never be aged safely → drop
        if at.tzinfo is None:
            at = at.replace(tzinfo=UTC)
        if at > cutoff:
            kept.append(entry)
    if kept != before:
        proc["compliance_events"] = kept
        return True
    return False


# ── Nightly runner ───────────────────────────────────────────────────────────


async def _load_rules(session: AsyncSession) -> list[MemoryTrace]:
    """Load at most MAX_RULES + 1 eligible rows, with one saturation sentinel.

    The supported store has at most MAX_RULES non-invalidated procedural rule/habit
    rows (all statuses). The caller must refuse oversized stores before any scoring
    or retirement; ordering alone cannot guarantee coverage beyond the limit.

    Deliberately NOT ``with_for_update()``, though this is a read-modify-write of the same
    ``typed_metadata`` document that ``rule_capture.record_message_anchored_compliance``
    locks. This runs inside a SAVEPOINT of the long consolidation transaction
    (``consolidation._run_step``), and a savepoint commit does not release row locks — they
    would be held on every rule row until the WHOLE nightly run commits, minutes later.
    Every L2 extraction in that window would then block on a user-facing path. The residual
    race is one nightly run overwriting a compliance event committed during its own
    transaction: a single lost observation, rare, and self-healing on the next verdict.
    Stalling chat extraction nightly to avoid it is the worse trade. Closing it properly
    means an atomic JSONB merge for this key rather than a wider lock (follow-up).
    """
    stmt = (
        select(MemoryTrace)
        .where(
            MemoryTrace.memory_type == "procedural",
            MemoryTrace.invalidated_at.is_(None),
            MemoryTrace.typed_metadata["procedural"]["kind"].astext.in_(PROC_KINDS),
        )
        .order_by(MemoryTrace.created_at.asc(), MemoryTrace.id.asc())
        .limit(MAX_RULES + 1)
    )
    return list((await session.execute(stmt)).scalars().all())


async def _load_served_traces(session: AsyncSession, cutoff: datetime) -> list[ExecutionTrace]:
    # CHRONOLOGICAL order is load-bearing, not cosmetic: apply_events mutates
    # fail_streak sequentially (followed→0, violated→+1), so the order traces are
    # applied decides whether a rule hits 2 consecutive failures → whether it
    # suspends. Without a deterministic ORDER BY the same events could flip the
    # demotion outcome run-to-run. created_at ASC replays real chronology; id is a
    # stable tie-breaker for same-timestamp rows.
    stmt = (
        select(ExecutionTrace)
        .where(
            ExecutionTrace.created_at >= cutoff,
            ExecutionTrace.success_signals.has_key("served_rule_ids"),  # JSONB ? operator
        )
        .order_by(ExecutionTrace.created_at.asc(), ExecutionTrace.id.asc())
        .limit(MAX_TRACES)
    )
    rows = list((await session.execute(stmt)).scalars().all())
    if len(rows) == MAX_TRACES:
        # The ascending window is bounded for safety, but at this volume the newest
        # traces would silently age past the cutoff unprocessed. Prod is ~200/2d, so
        # this WARN would fire long (~25x) before that's a real risk — surface it early.
        logger.warning("procedural.reinforce.trace_cap_hit n=%d — served-trace window saturated", MAX_TRACES)
    return rows


async def run_reinforcement(session: AsyncSession) -> dict[str, int]:
    """Nightly reinforcement/demotion pass (design §4.2). No-op when the flag is off.
    Idempotent under duplicate runs. Never touches retrieval config or salience.

    Above MAX_RULES eligible rows, skip the entire pass without ledger/retirement
    writes. ``rules_seen`` is the bounded sentinel count, not the store total;
    ``rules`` is zero because none were processed. Consolidation exposes this
    refusal under ``rule_scoring`` alongside a content-free warning.
    """
    if not cognitive_config.get_bool("procedural_reinforcement_enabled"):
        return {"skipped": 1}
    now = utc_now()
    now_iso = now.isoformat()
    admin_ids = getattr(get_settings(), "telegram_admin_user_ids", "")
    rows = await _load_rules(session)
    if len(rows) > MAX_RULES:
        logger.warning(
            "procedural.reinforce.rule_cap_exceeded limit=%d rules_seen=%d — reinforcement skipped; "
            "eligible rule store exceeds supported limit",
            MAX_RULES,
            len(rows),
        )
        return {
            "skipped": 1,
            "rule_scan_saturated": 1,
            "rule_limit": MAX_RULES,
            "rules_seen": len(rows),
            "rules": 0,
        }
    if not rows:
        # Heartbeat: emit the summary even at 0 rules so the feature-watch has a
        # reliable nightly signal that the enabled ledger ran (design §9 observability).
        logger.info("procedural.reinforce rules=0 reinforced=0 suspended=0 retired=0 ignored=0")
        return {"rules": 0}
    traces = await _load_served_traces(session, now - timedelta(days=LOOKBACK_DAYS))
    served_map: dict[str, list[ExecutionTrace]] = {}
    for trace in traces:
        for rid in (trace.success_signals or {}).get("served_rule_ids") or []:
            served_map.setdefault(str(rid), []).append(trace)

    stats = {"rules": len(rows), "reinforced": 0, "suspended": 0, "retired": 0, "ignored": 0}
    for row in rows:
        proc = dict((row.typed_metadata or {}).get("procedural", {}) or {})
        util = dict(proc.get("utility") or {})
        source = rule_source(proc)
        token_count = estimate_tokens(str(proc.get("guidance") or row.content or ""))
        # DB-487: the instructor's respect score scales the gain term by respect/100, so
        # a roster-taught rule (75) earns at 0.75x and must build a better record than an
        # admin-taught one to clear THETA_DEMOTE. Missing on pre-DB-487 rows, which
        # respect_multiplier maps to 1.0 rather than demoting them retroactively.
        respect = (proc.get("provenance") or {}).get("respect")

        # Watermark: after a suspended→candidate recovery, score ONLY traces created
        # after the recovery time, so the pre-recovery evidence that suspended it can't
        # re-contaminate the reset ledger while still inside the lookback window (§4.2).
        window_traces = after_watermark(served_map.get(str(row.id), []), proc.get("scored_after"))
        # Age expired message-anchored events out BEFORE gathering, so the plain-turn path
        # scores the same window the trace path does. Pruning afterwards would let a
        # verdict older than the lookback window apply exactly once on its way out — which
        # a trace-anchored verdict can never do, because the trace query already excluded
        # it (CodeRabbit review round 2).
        row_changed = prune_compliance_events(proc, now)
        events = gather_events(row, proc, window_traces, admin_ids)
        delta = apply_events(
            util, source=source, token_count=token_count, events=events, now_iso=now_iso, respect=respect
        )
        row_changed = row_changed or delta.changed
        if delta.changed:
            proc["utility"] = util
            stats["reinforced"] += 1
            # Applied recurrences are now in event_keys — drop them from the pending
            # list so they can never re-apply (idempotency, §4.4).
            applied = set(util.get("event_keys") or [])
            pending = [m for m in (proc.get("recurrence_events") or []) if f"recurrence:{m}" not in applied]
            if pending != (proc.get("recurrence_events") or []):
                proc["recurrence_events"] = pending
        # Prune the applied-key ledger to the current window so it stays bounded
        # without ever evicting a still-reachable key (the tail-eviction re-apply bug).
        window_ids = {str(t.id) for t in window_traces}
        pending_rec = {str(m) for m in (proc.get("recurrence_events") or [])}
        pending_comp = {str(e.get("message_id")) for e in (proc.get("compliance_events") or []) if isinstance(e, dict)}
        if prune_event_keys(
            util,
            window_trace_ids=window_ids,
            pending_recurrence=pending_rec,
            pending_compliance=pending_comp,
        ):
            proc["utility"] = util
            row_changed = True

        if should_suspend(proc, util, now, failed_this_run=delta.failed):
            proc["status"] = "suspended"
            proc["suspended_at"] = now_iso
            row_changed = True
            stats["suspended"] += 1
            logger.warning(
                # source/respect appended (DB-648) so a query can prove no full-trust rule
                # was ever suspended; the original fields keep their positions.
                "procedural.rule.suspended id=%s U=%s fail_streak=%s n_served=%s source=%s respect=%s",
                row.id,
                util.get("U"),
                util.get("fail_streak"),
                util.get("n_served"),
                source,
                (proc.get("provenance") or {}).get("respect"),
            )
        elif is_protected_from_demotion(proc):
            # DB-648: the ledger above still moved (U keeps ranking this rule for cap
            # eviction); only the status transition is withheld. Escalate instead — the
            # log is the pager (DB-467), and this is the "rule being ignored" alarm DB-646
            # needed a human to spot by eye.
            trigger = ignored_trigger(proc, util, now, failed_this_run=delta.failed)
            page, marker_changed = update_ignored_marker(proc, util, trigger, now_iso)
            row_changed = row_changed or marker_changed
            if page:
                stats["ignored"] += 1
                logger.error(
                    "procedural.rule.ignored id=%s source=%s respect=%s trigger=%s fail_streak=%s n_served=%s U=%s "
                    "— a user-taught rule was held back from demotion (DB-648): the serving path is not "
                    "getting it obeyed",
                    row.id,
                    source,
                    (proc.get("provenance") or {}).get("respect"),
                    trigger,
                    util.get("fail_streak"),
                    util.get("n_served"),
                    util.get("U"),
                )

        retired = False
        if _suspended_too_long(proc, now):
            from kai_memory.store.cognitive_memory import CognitiveMemoryService

            await CognitiveMemoryService(session).invalidate_trace(row.id, invalidation_source="procedural_retire")
            retired = True
            stats["retired"] += 1
            logger.info("procedural.rule.retired id=%s (suspended > %dd)", row.id, SUSPEND_RETIRE_DAYS)

        if row_changed and not retired:
            meta = dict(row.typed_metadata or {})
            meta["procedural"] = proc
            row.typed_metadata = meta

    logger.info(
        "procedural.reinforce rules=%d reinforced=%d suspended=%d retired=%d ignored=%d",
        stats["rules"],
        stats["reinforced"],
        stats["suspended"],
        stats["retired"],
        stats["ignored"],
    )
    return stats
