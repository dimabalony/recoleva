"""Phase C §4.2/§4.4 — reinforcement/demotion ledger (pure logic).

DB-free unit coverage of the #241-safe ledger: the rule-level compliance-event
matrix, idempotency under duplicate runs, utility ordering, demotion reachability,
and trust-scoped event gathering. The nightly wiring is integration-tested.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from kai_memory import procedural_reinforcement as pr
from kai_memory.procedural_reinforcement import (
    THETA_DEMOTE,
    Event,
    after_watermark,
    apply_events,
    compute_utility,
    gather_events,
    prune_event_keys,
    respect_multiplier,
    should_suspend,
)

_NOW = datetime(2026, 7, 20, tzinfo=UTC)
_NOW_ISO = _NOW.isoformat()
ADMIN_IDS = "111"
ADMIN = 111
STRANGER = 999


def _util(**over) -> dict:
    base = {"s": 0, "f": 0, "n_served": 0, "streak": 0, "fail_streak": 0, "U": None, "event_keys": []}
    base.update(over)
    return base


def _ev(verdict: str, key: str) -> Event:
    return Event(verdict=verdict, key=key)


# ── Utility (§4.2) ───────────────────────────────────────────────────────────


def test_compute_utility_provenance_ordering():
    # Same evidence, stronger provenance → higher utility (G: instruction>correction>l3).
    s, f, tok = 3, 1, 12
    u_instr = compute_utility(s, f, "admin_instruction", tok)
    u_corr = compute_utility(s, f, "admin_correction", tok)
    u_l3 = compute_utility(s, f, "l3_promotion", tok)
    assert u_instr > u_corr > u_l3


def test_compute_utility_cost_penalty_reduces_u():
    assert compute_utility(3, 1, "admin_instruction", 20000) < compute_utility(3, 1, "admin_instruction", 1)


# ── Event matrix (§4.2) ──────────────────────────────────────────────────────


def test_followed_increments_s_and_streak():
    u = _util()
    d = apply_events(
        u, source="admin_instruction", token_count=10, events=[_ev("followed", "compliance:t1")], now_iso=_NOW_ISO
    )
    assert d.changed and not d.failed
    assert u["s"] == 1 and u["streak"] == 1 and u["fail_streak"] == 0 and u["n_served"] == 1
    assert u["U"] is not None


def test_violated_increments_f_and_resets_streak():
    u = _util(s=2, streak=2)
    d = apply_events(
        u, source="admin_instruction", token_count=10, events=[_ev("violated", "compliance:t2")], now_iso=_NOW_ISO
    )
    assert d.changed and d.failed
    assert u["f"] == 1 and u["streak"] == 0 and u["fail_streak"] == 1 and u["n_served"] == 1


def test_recurrence_stings_double():
    u = _util()
    d = apply_events(
        u, source="admin_correction", token_count=10, events=[_ev("recurrence", "recurrence:m1")], now_iso=_NOW_ISO
    )
    assert d.failed
    assert u["f"] == 2 and u["fail_streak"] == 1
    assert u["n_served"] == 0  # a correction is not a serving


def test_nudge_rejected_stings_double():
    u = _util()
    apply_events(u, source="l3_promotion", token_count=10, events=[_ev("nudge_rejected", "nudge:w1")], now_iso=_NOW_ISO)
    assert u["f"] == 2 and u["n_served"] == 0


# ── Idempotency (§4.4) — stability under duplicate/overlapping runs ───────────


def test_apply_is_idempotent_under_duplicate_runs():
    events = [_ev("followed", "compliance:t1"), _ev("violated", "compliance:t2")]
    u1 = _util()
    apply_events(u1, source="admin_instruction", token_count=10, events=events, now_iso=_NOW_ISO)
    snapshot = {k: u1[k] for k in ("s", "f", "n_served", "streak", "fail_streak", "U")}

    # A duplicate nightly run re-presents the SAME events → no double count.
    d2 = apply_events(u1, source="admin_instruction", token_count=10, events=events, now_iso=_NOW_ISO)
    assert d2.changed is False
    assert {k: u1[k] for k in snapshot} == snapshot


def test_new_event_after_dupes_still_applies_once():
    u = _util()
    apply_events(
        u, source="admin_instruction", token_count=10, events=[_ev("followed", "compliance:t1")], now_iso=_NOW_ISO
    )
    # t1 repeats (skipped), t2 is new (applied once).
    apply_events(
        u,
        source="admin_instruction",
        token_count=10,
        events=[_ev("followed", "compliance:t1"), _ev("followed", "compliance:t2")],
        now_iso=_NOW_ISO,
    )
    assert u["s"] == 2


def test_apply_retains_all_keys_no_fixed_tail():
    # No fixed tail cap — a fixed tail would evict keys whose trace is still in the
    # lookback window and let them re-apply. run_reinforcement prunes by window instead.
    u = _util()
    events = [_ev("followed", f"compliance:t{i}") for i in range(70)]
    apply_events(u, source="admin_instruction", token_count=10, events=events, now_iso=_NOW_ISO)
    assert len(u["event_keys"]) == 70  # ALL keys retained
    assert u["s"] == 70


def test_prune_event_keys_drops_out_of_window_and_stays_idempotent():
    # 70 compliance events applied; only 30 traces remain in the window on the next run.
    u = _util()
    events = [_ev("followed", f"compliance:t{i}") for i in range(70)]
    apply_events(u, source="admin_instruction", token_count=10, events=events, now_iso=_NOW_ISO)
    in_window = {f"t{i}" for i in range(40, 70)}  # t40..t69 still in the 2-day window
    changed = pr.prune_event_keys(u, window_trace_ids=in_window, pending_recurrence=set())
    assert changed is True
    assert len(u["event_keys"]) == 30  # aged-out keys dropped; in-window kept

    # Re-presenting the in-window events must NOT re-apply (their keys survived).
    rerun = [_ev("followed", f"compliance:t{i}") for i in range(40, 70)]
    d = apply_events(u, source="admin_instruction", token_count=10, events=rerun, now_iso=_NOW_ISO)
    assert d.changed is False
    assert u["s"] == 70  # NO inflation


def test_prune_event_keys_covers_all_three_key_kinds():
    # compliance + nudge keys retained iff their trace is in-window; recurrence keys
    # retained iff still pending; unknown kinds retained defensively.
    util = {
        "event_keys": [
            "compliance:t1",  # in-window → kept
            "compliance:t9",  # aged out → dropped
            "nudge:t1",  # in-window → kept
            "nudge:t9",  # aged out → dropped
            "recurrence:m1",  # still pending → kept
            "recurrence:m2",  # applied+pruned → dropped
            "weird:x",  # unknown kind → retained defensively
        ]
    }
    changed = pr.prune_event_keys(util, window_trace_ids={"t1"}, pending_recurrence={"m1"})
    assert changed is True
    assert set(util["event_keys"]) == {"compliance:t1", "nudge:t1", "recurrence:m1", "weird:x"}


def test_prune_event_keys_noop_when_nothing_drops():
    util = {"event_keys": ["compliance:t1", "recurrence:m1"]}
    changed = pr.prune_event_keys(util, window_trace_ids={"t1"}, pending_recurrence={"m1"})
    assert changed is False  # all keys still relevant → no write


# ── Demotion reachability (§4.2) — a failing rule actually suspends ──────────
# DB-648: demotion now applies only to rules no full-trust person taught, so these pin the
# path on the LIVE unprotected class — a roster-taught rule (respect 75), which is what
# rule_capture really mints below full trust. Using an admin-taught rule here would make
# every "does not suspend" assertion pass vacuously; using `l3_promotion` would test a
# source no code can mint (it appears only in the matrix below, as the code branches on it).

_ROSTER = {"source": "admin_correction", "provenance": {"taught_by": 7, "respect": 75}}
_UNPROTECTED = [pytest.param(_ROSTER, id="roster_taught_75")]
_L3 = {"source": "l3_promotion"}
# Protected provenances: the admin, and a legacy row with no recorded respect.
_ADMIN = {"source": "admin_instruction", "provenance": {"taught_by": ADMIN, "respect": 100}}
_LEGACY = {"source": "admin_correction"}
_PROTECTED = [pytest.param(_ADMIN, id="admin_instruction_100"), pytest.param(_LEGACY, id="legacy_no_respect")]


def _probation_proc(**over) -> dict:
    return {"status": "active", "probation_until": (_NOW + timedelta(days=3)).isoformat(), **over}


def _fail_streak_ledger() -> dict:
    n = pr.CONSECUTIVE_FAILURES_TO_SUSPEND
    return _util(f=n, fail_streak=n, n_served=n)


def _utility_floor_ledger() -> dict:
    # fail_streak held under the fast path so ONLY the U < θ branch can fire.
    n = pr.MIN_SERVED_FOR_UTILITY_DEMOTION
    return _util(s=1, f=n, n_served=n + 1, fail_streak=1, U=THETA_DEMOTE / 2)


@pytest.mark.parametrize("prov", _UNPROTECTED)
def test_probation_single_failure_suspends(prov):
    proc = _probation_proc(**prov)
    u = _util(f=1, fail_streak=1)
    assert should_suspend(proc, u, _NOW, failed_this_run=True) is True


def test_probation_no_failure_does_not_suspend():
    proc = _probation_proc(**_ROSTER)
    u = _util(s=1, streak=1)
    assert should_suspend(proc, u, _NOW, failed_this_run=False) is False


@pytest.mark.parametrize("prov", _UNPROTECTED)
def test_established_two_consecutive_failures_suspend(prov):
    proc = {"status": "active", **prov}  # no probation
    assert should_suspend(proc, _fail_streak_ledger(), _NOW, failed_this_run=True) is True


@pytest.mark.parametrize("prov", _UNPROTECTED)
def test_established_utility_floor_after_min_served(prov):
    proc = {"status": "active", **prov}
    assert should_suspend(proc, _utility_floor_ledger(), _NOW, failed_this_run=False) is True


def test_no_utility_demotion_before_min_served():
    proc = {"status": "active", **_ROSTER}
    u = _util(s=0, f=2, n_served=pr.MIN_SERVED_FOR_UTILITY_DEMOTION - 1, fail_streak=1, U=0.15)
    assert should_suspend(proc, u, _NOW, failed_this_run=False) is False


def test_reachability_five_violations_suspend():
    """End-to-end: five served-and-violated events drive an established rule under θ
    with enough evidence — demotion IS reachable (the lifecycle isn't a dead end)."""
    proc = {"status": "active", **_ROSTER, "guidance": "do the thing"}
    u = _util()
    events = [_ev("violated", f"compliance:t{i}") for i in range(5)]
    d = apply_events(u, source="admin_correction", token_count=10, events=events, now_iso=_NOW_ISO, respect=75)
    assert should_suspend(proc, u, _NOW, failed_this_run=d.failed) is True


def test_reachability_utility_floor_specifically():
    """The U<θ demotion branch, isolated from the fail_streak branch: interleave
    follows and violations so fail_streak never reaches 2, yet U drops below θ after
    n_served ≥ 5 (CodeRabbit: exercise the utility-floor path, not fail_streak)."""
    proc = {"status": "active", **_ROSTER, "guidance": "do the thing"}
    u = _util()
    verdicts = ["violated", "followed", "violated", "followed", "violated", "followed", "violated"]
    events = [_ev(v, f"compliance:t{i}") for i, v in enumerate(verdicts)]
    apply_events(u, source="admin_correction", token_count=100, events=events, now_iso=_NOW_ISO, respect=75)
    assert u["fail_streak"] < pr.CONSECUTIVE_FAILURES_TO_SUSPEND  # NOT the fail-streak branch
    assert u["n_served"] >= pr.MIN_SERVED_FOR_UTILITY_DEMOTION
    assert u["U"] < pr.THETA_DEMOTE
    assert should_suspend(proc, u, _NOW, failed_this_run=False) is True  # utility-floor path


# ── DB-648: a rule a person taught is never demoted for the system's non-compliance ──


def test_protection_keys_on_the_writers_real_vocabulary():
    """The guard must key on what rule_capture actually mints, not on names invented here.
    Build real metadata through the writer for both directive strengths, admin-taught, and
    check every source it can emit is protected and priced by G_BY_SOURCE."""
    from kai_memory.procedural.rule_capture import RESPECT_ADMIN, GateResult, _build_rule_metadata

    gate = GateResult(accepted=True, reason="ok", instructor_id=ADMIN, status="active", respect=RESPECT_ADMIN)
    for strength in ("instruction", "correction"):
        proc = _build_rule_metadata(
            {"directive_strength": strength, "rule": "r", "trigger": "t", "category": "format"},
            gate,
            now=_NOW,
            source_message_id="m1",
        )
        assert proc["source"] in pr.USER_TAUGHT_SOURCES
        assert proc["source"] in pr.G_BY_SOURCE
        assert pr.is_protected_from_demotion(proc) is True
    assert "l3_promotion" in pr.G_BY_SOURCE and "l3_promotion" not in pr.USER_TAUGHT_SOURCES
    assert pr.FULL_TRUST_RESPECT == RESPECT_ADMIN


@pytest.mark.parametrize(
    ("proc", "protected"),
    [
        (_ADMIN, True),
        (_LEGACY, True),  # pre-DB-487 row: no respect recorded, every such rule was admin-taught
        ({}, True),  # no source → read as admin_correction, exactly as the gain reads it
        (_L3, False),
        (_ROSTER, False),  # DB-487's demotion counterweight for respect-75 teaching stays
        ({"source": "something_new"}, False),  # unknown provenance never widens protection
        # A malformed or out-of-range persisted score must not WIDEN protection
        # (respect_multiplier maps both to 1.0 — right for gain, wrong for this guard).
        ({"source": "admin_instruction", "provenance": {"respect": "n/a"}}, False),
        ({"source": "admin_instruction", "provenance": {"respect": 150}}, False),
        ({"source": "admin_instruction", "provenance": {"respect": True}}, False),
        ({"source": "admin_instruction", "provenance": {"respect": 99}}, False),
    ],
)
def test_is_protected_from_demotion_matrix(proc, protected):
    assert pr.is_protected_from_demotion(proc) is protected


@pytest.mark.parametrize("prov", _PROTECTED)
@pytest.mark.parametrize(
    ("proc_state", "ledger", "failed", "trigger"),
    [
        pytest.param(_probation_proc(), _util(f=1, fail_streak=1, n_served=1), True, "probation", id="probation"),
        pytest.param({"status": "active"}, _fail_streak_ledger(), True, "fail_streak", id="fail_streak"),
        pytest.param({"status": "active"}, _utility_floor_ledger(), False, "utility", id="utility"),
    ],
)
def test_protected_rule_never_suspends_on_any_trigger(prov, proc_state, ledger, failed, trigger):
    """Every trigger genuinely FIRES on this ledger (so the assertion is not vacuous), and
    the rule still does not suspend — the first slip in a fresh rule's probation week
    included."""
    proc = {**proc_state, **prov}
    assert pr.demotion_trigger(proc, ledger, _NOW, failed_this_run=failed) == trigger
    assert should_suspend(proc, ledger, _NOW, failed_this_run=failed) is False


def test_protected_rule_still_loses_utility_for_ranking():
    """Only the STATUS transition is guarded. Cap eviction (DB-581) ranks by U, so a
    protected rule the system keeps failing must still sink below a rule it obeys."""
    obeyed, ignored = _util(), _util()
    apply_events(
        obeyed, source="admin_instruction", token_count=10, events=[_ev("followed", "compliance:a")], now_iso=_NOW_ISO
    )
    d = apply_events(
        ignored,
        source="admin_instruction",
        token_count=10,
        events=[_ev("violated", f"compliance:t{i}") for i in range(pr.CONSECUTIVE_FAILURES_TO_SUSPEND)],
        now_iso=_NOW_ISO,
    )
    assert d.changed and d.failed
    assert ignored["f"] == pr.CONSECUTIVE_FAILURES_TO_SUSPEND
    assert ignored["fail_streak"] == pr.CONSECUTIVE_FAILURES_TO_SUSPEND
    assert ignored["n_served"] == pr.CONSECUTIVE_FAILURES_TO_SUSPEND
    assert ignored["U"] < obeyed["U"]


def test_ignored_marker_pages_once_per_episode():
    """One ERROR per crossing, not one per nightly: stamp on the first crossing, stay silent
    while it persists, survive a quiet night, and re-arm only after a `followed`."""
    proc: dict = {"status": "active", **_ADMIN}
    failing = _util(fail_streak=pr.CONSECUTIVE_FAILURES_TO_SUSPEND)

    assert pr.update_ignored_marker(proc, failing, "fail_streak", _NOW_ISO) == (True, True)
    assert proc[pr.IGNORED_MARKER_KEY] == {"at": _NOW_ISO, "trigger": "fail_streak"}
    # Still crossing on the next nightly → silent, nothing to write.
    assert pr.update_ignored_marker(proc, failing, "utility", _NOW_ISO) == (False, False)
    # A quiet night (no trigger, but the last event was still a failure) keeps the marker.
    assert pr.update_ignored_marker(proc, _util(fail_streak=1), None, _NOW_ISO) == (False, False)
    assert pr.IGNORED_MARKER_KEY in proc
    # A `followed` reset the streak and nothing crosses → the episode ends.
    assert pr.update_ignored_marker(proc, _util(s=1, fail_streak=0), None, _NOW_ISO) == (False, True)
    assert pr.IGNORED_MARKER_KEY not in proc
    # ...so the next crossing is a new episode and pages again.
    assert pr.update_ignored_marker(proc, failing, "probation", _NOW_ISO) == (True, True)


def test_ignored_marker_is_not_cleared_while_a_trigger_still_crosses():
    """A `followed` alone does not end the episode while U is still under the floor —
    otherwise every night Kai obeys the rule once would page 'this rule is ignored'."""
    proc: dict = {"status": "active", **_ADMIN, pr.IGNORED_MARKER_KEY: {"at": _NOW_ISO, "trigger": "utility"}}
    assert pr.update_ignored_marker(proc, _util(s=1, fail_streak=0), "utility", _NOW_ISO) == (False, False)
    assert pr.IGNORED_MARKER_KEY in proc


@pytest.mark.parametrize("prov", _PROTECTED)
def test_protected_rule_already_suspended_is_never_retired(prov):
    """A protected rule can only be suspended if the ledger demoted it before this guard
    shipped. Retirement is the second half of that same demotion, so it must not run."""
    long_ago = (_NOW - timedelta(days=pr.SUSPEND_RETIRE_DAYS + 1)).isoformat()
    proc = {"status": "suspended", "suspended_at": long_ago, **prov}
    assert pr._suspended_too_long(proc, _NOW) is False
    assert pr.ignored_trigger(proc, _util(), _NOW, failed_this_run=False) == pr.TRIGGER_LEGACY_SUSPENDED


@pytest.mark.parametrize("prov", _UNPROTECTED)
def test_unprotected_rule_still_retires_after_the_suspension_window(prov):
    long_ago = (_NOW - timedelta(days=pr.SUSPEND_RETIRE_DAYS + 1)).isoformat()
    proc = {"status": "suspended", "suspended_at": long_ago, **prov}
    assert pr._suspended_too_long(proc, _NOW) is True


def test_untrusted_verdict_falls_through_to_deterministic_tool_check():
    """Security fix — an untrusted L2 verdict must be IGNORED, not suppress the
    inherently-trusted deterministic tool check (code-derived, not a user utterance)."""
    row = _row()
    proc = {"scope": {"person": None}, "expected_tools": ["wake"]}
    # Stranger's L2 verdict says 'violated', but the tool WAS used → deterministic followed.
    traces = [_trace("t1", source_user_id=STRANGER, compliance={"rule-1": "violated"}, steps=[{"tool": "wake"}])]
    events = gather_events(row, proc, traces, ADMIN_IDS)
    assert [e.verdict for e in events] == ["followed"]  # untrusted verdict ignored; tool-check wins


def test_after_watermark_naive_cutoff_vs_aware_trace_no_typeerror():
    """§4.2 stability — a naive legacy watermark vs an aware created_at must not raise
    TypeError (which would abort the whole nightly pass)."""
    aware = SimpleNamespace(id="t", created_at=datetime(2026, 7, 20, 12, tzinfo=UTC))
    naive_cutoff = "2026-07-20T06:00:00"  # no offset
    assert after_watermark([aware], naive_cutoff) == [aware]  # aware(12:00) > naive→UTC(06:00), no crash


# ── Event gathering: trust-scoping + deterministic tool check (§4.1/§4.2) ────


def _trace(tid: str, *, source_user_id=None, compliance=None, steps=None, extra_signals=None):
    signals: dict = {"served_rule_ids": ["rule-1"]}
    if compliance is not None:
        signals["rule_compliance"] = compliance
    if extra_signals:
        signals.update(extra_signals)
    return SimpleNamespace(
        id=tid,
        source_user_id=source_user_id,
        success_signals=signals,
        steps=steps or [],
    )


def _row(rule_id="rule-1"):
    return SimpleNamespace(id=rule_id)


def test_gather_l2_verdict_from_admin_counts():
    row = _row()
    proc = {"scope": {"person": None}}
    traces = [_trace("t1", source_user_id=ADMIN, compliance={"rule-1": "followed"})]
    events = gather_events(row, proc, traces, ADMIN_IDS)
    assert [e.verdict for e in events] == ["followed"]
    assert events[0].key == "compliance:t1"


def test_gather_l2_verdict_from_untrusted_ignored():
    row = _row()
    proc = {"scope": {"person": None}}  # global rule → only admin evidence counts
    traces = [_trace("t1", source_user_id=STRANGER, compliance={"rule-1": "violated"})]
    assert gather_events(row, proc, traces, ADMIN_IDS) == []


def test_gather_l2_verdict_from_scope_person_counts():
    row = _row()
    proc = {"scope": {"person": 555}}  # person-scoped → that person's evidence counts
    traces = [_trace("t1", source_user_id=555, compliance={"rule-1": "violated"})]
    events = gather_events(row, proc, traces, ADMIN_IDS)
    assert [e.verdict for e in events] == ["violated"]


def test_gather_deterministic_tool_check_followed():
    row = _row()
    # A tool-shaped rule; a headless trace (NULL sender) whose steps ⊇ expected_tools.
    proc = {"scope": {"person": None}, "expected_tools": ["wake"]}
    traces = [_trace("t1", source_user_id=None, steps=[{"tool": "wake"}, {"tool": "read"}])]
    events = gather_events(row, proc, traces, ADMIN_IDS)
    assert [e.verdict for e in events] == ["followed"]  # deterministic check is always trusted


def test_gather_deterministic_tool_absent_is_ignored():
    row = _row()
    proc = {"scope": {"person": None}, "expected_tools": ["wake"]}
    traces = [_trace("t1", source_user_id=None, steps=[{"tool": "read"}])]  # wake absent → not a violation
    assert gather_events(row, proc, traces, ADMIN_IDS) == []


def test_after_watermark_filters_pre_recovery_traces():
    cutoff = datetime(2026, 7, 20, tzinfo=UTC)
    older = SimpleNamespace(id="t_old", created_at=cutoff - timedelta(hours=1))
    newer = SimpleNamespace(id="t_new", created_at=cutoff + timedelta(hours=1))
    assert after_watermark([older, newer], cutoff.isoformat()) == [newer]  # pre-recovery dropped
    assert after_watermark([older, newer], None) == [older, newer]  # no watermark → all


def test_gather_recurrence_events():
    row = _row()
    proc = {"scope": {"person": None}, "recurrence_events": ["m1", "m2"]}
    events = gather_events(row, proc, [], ADMIN_IDS)
    assert {e.key for e in events} == {"recurrence:m1", "recurrence:m2"}
    assert all(e.verdict == "recurrence" for e in events)


def test_gather_nudge_rejected_from_trusted_actor():
    row = _row()
    proc = {"scope": {"person": None}}  # global rule → admin actor counts
    traces = [
        _trace("t1", source_user_id=ADMIN, extra_signals={"nudge_outcome": "rejected", "origin_rule_ids": ["rule-1"]})
    ]
    events = gather_events(row, proc, traces, ADMIN_IDS)
    assert any(e.verdict == "nudge_rejected" and e.key == "nudge:t1" for e in events)


def test_gather_nudge_rejected_from_untrusted_ignored():
    """Security (§4.2): a stranger's rejected nudge can't demote a global/admin rule."""
    row = _row()
    proc = {"scope": {"person": None}}
    traces = [
        _trace(
            "t1", source_user_id=STRANGER, extra_signals={"nudge_outcome": "rejected", "origin_rule_ids": ["rule-1"]}
        )
    ]
    assert gather_events(row, proc, traces, ADMIN_IDS) == []


# ── DB-487: respect scales the gain term (the counterweight to opening teaching) ──


def test_admin_respect_leaves_gain_unchanged():
    """Opening teaching must not silently re-price EXISTING rules. Admin respect (100)
    is exactly 1.0, so every admin-taught rule's utility is byte-identical to before."""
    assert respect_multiplier(100) == 1.0
    baseline = compute_utility(3, 1, "admin_instruction", 40)
    assert compute_utility(3, 1, "admin_instruction", 40, respect=100) == baseline


def test_missing_respect_is_not_a_demotion():
    """Legacy rows carry no provenance. Treating absent respect as 0 would retroactively
    suspend every pre-DB-487 rule on the next nightly pass."""
    assert respect_multiplier(None) == 1.0
    assert compute_utility(3, 1, "admin_instruction", 40, respect=None) == compute_utility(
        3, 1, "admin_instruction", 40
    )


def test_roster_taught_rule_earns_at_three_quarters():
    assert respect_multiplier(75) == 0.75
    admin_u = compute_utility(4, 1, "admin_correction", 40, respect=100)
    roster_u = compute_utility(4, 1, "admin_correction", 40, respect=75)
    assert roster_u < admin_u


def test_roster_taught_rule_needs_a_better_record_to_clear_the_floor():
    """The behavioural point, not just arithmetic: at an identical track record a
    75-taught rule can sit BELOW the demotion floor while the admin's sits above it, so
    a weaker teacher's rule must actually earn its place."""
    record = dict(s=2, f=2, source="admin_correction", token_count=40)
    admin_u = compute_utility(**record, respect=100)
    roster_u = compute_utility(**record, respect=75)
    assert admin_u >= THETA_DEMOTE > roster_u

    # ...and a stronger record lifts the same 75-taught rule back over the floor.
    assert compute_utility(s=9, f=0, source="admin_correction", token_count=40, respect=75) >= THETA_DEMOTE


def test_respect_multiplier_is_clamped_against_a_malformed_score():
    """A bad roster entry must never inflate gain above admin level."""
    assert respect_multiplier(9999) == 1.0
    assert respect_multiplier(-50) == 0.0
    assert respect_multiplier("nonsense") == 1.0


def test_apply_events_forwards_respect_into_the_recomputed_utility():
    """The wiring, not just the formula: apply_events is the only writer of util["U"],
    so a dropped respect kwarg would silently restore full gain for every teacher."""
    events = [Event(verdict="followed", key="compliance:t1")]

    def _fresh() -> dict:
        # Built independently: a shallow copy would share the event_keys LIST between the
        # two ledgers, so the second apply_events would see the first's applied key.
        return {"s": 0, "f": 0, "n_served": 0, "streak": 0, "fail_streak": 0, "U": None, "event_keys": []}

    admin_util, roster_util = _fresh(), _fresh()

    apply_events(admin_util, source="admin_correction", token_count=40, events=events, now_iso="x", respect=100)
    apply_events(roster_util, source="admin_correction", token_count=40, events=events, now_iso="x", respect=75)

    assert roster_util["U"] < admin_util["U"]
    assert admin_util["U"] == compute_utility(1, 0, "admin_correction", 40, respect=100)
    assert roster_util["U"] == compute_utility(1, 0, "admin_correction", 40, respect=75)


# ── DB-629: the two new pure helpers (CodeRabbit PR round 1) ────────────────


def _entry(at_iso, verdict="followed", msg_id="m1"):
    return {"message_id": msg_id, "verdict": verdict, "at": at_iso}


def test_entry_after_watermark_admits_only_post_recovery_evidence():
    from kai_memory.procedural_reinforcement import _entry_after_watermark

    mark = "2026-09-10T12:00:00+00:00"
    assert _entry_after_watermark(_entry("2026-09-10T12:00:01+00:00"), mark) is True
    assert _entry_after_watermark(_entry("2026-09-10T11:59:59+00:00"), mark) is False
    # Exactly AT the watermark is refused: the watermark is stamped at recovery, so an
    # entry sharing its instant is pre-recovery evidence, not new evidence.
    assert _entry_after_watermark(_entry(mark), mark) is False
    # No watermark at all → a rule that never recovered scores its full window.
    assert _entry_after_watermark(_entry("2026-01-01T00:00:00+00:00"), None) is True


def test_entry_after_watermark_refuses_an_undatable_entry():
    """Unlike a trace, a pending entry carries its own timestamp — a missing or malformed
    one cannot be placed relative to a recovery, so it is refused rather than admitted.
    Admitting it would let pre-suspension evidence score against a reset ledger."""
    from kai_memory.procedural_reinforcement import _entry_after_watermark

    mark = "2026-09-10T12:00:00+00:00"
    assert _entry_after_watermark({"message_id": "m1", "verdict": "followed"}, mark) is False
    assert _entry_after_watermark(_entry("not-a-timestamp"), mark) is False
    assert _entry_after_watermark(_entry(None), mark) is False
    # A naive stored timestamp is read as UTC, not rejected (prod timestamps are UTC).
    assert _entry_after_watermark(_entry("2026-09-10T12:00:01"), mark) is True


def test_prune_compliance_events_drops_aged_and_malformed_entries():
    from datetime import datetime, timedelta

    from kai_memory.procedural_reinforcement import LOOKBACK_DAYS, prune_compliance_events

    now = datetime.fromisoformat("2026-09-12T12:00:00+00:00")
    fresh = _entry((now - timedelta(hours=1)).isoformat(), msg_id="fresh")
    aged = _entry((now - timedelta(days=LOOKBACK_DAYS + 1)).isoformat(), msg_id="aged")
    proc = {"compliance_events": [fresh, aged, "not-a-dict", {"message_id": "x", "verdict": "followed"}]}

    assert prune_compliance_events(proc, now) is True
    assert proc["compliance_events"] == [fresh]  # aged, non-dict and undatable all dropped
    assert prune_compliance_events(proc, now) is False  # idempotent — nothing left to drop


def test_prune_compliance_events_is_a_noop_on_an_empty_list():
    from datetime import datetime

    from kai_memory.procedural_reinforcement import prune_compliance_events

    now = datetime.fromisoformat("2026-09-12T12:00:00+00:00")
    assert prune_compliance_events({}, now) is False
    assert prune_compliance_events({"compliance_events": []}, now) is False


def test_prune_event_keys_retains_a_still_pending_compliance_msg_key():
    """The key must survive exactly as long as its entry can still be re-gathered.
    Released early, the next nightly pass re-applies the same verdict and inflates s/f."""
    util = {"event_keys": ["compliance_msg:m1", "compliance_msg:gone", "compliance:t1"]}
    changed = prune_event_keys(
        util,
        window_trace_ids={"t1"},
        pending_recurrence=set(),
        pending_compliance={"m1"},
    )
    assert changed is True
    assert util["event_keys"] == ["compliance_msg:m1", "compliance:t1"]


def test_prune_event_keys_without_pending_compliance_drops_message_keys():
    """Default (no pending list supplied) means nothing is pending, so a message key has
    nothing to hold it — the callers that pre-date DB-629 keep their exact behaviour."""
    util = {"event_keys": ["compliance_msg:m1", "recurrence:r1"]}
    assert prune_event_keys(util, window_trace_ids=set(), pending_recurrence={"r1"}) is True
    assert util["event_keys"] == ["recurrence:r1"]
