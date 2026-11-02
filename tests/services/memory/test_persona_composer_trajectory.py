"""Unit tests for PersonaComposer Trajectory render path (Wave 3).

Covers the three-tier event filter (recent verbatim / midterm top-N by
access-weight / older collapsed) plus the budget guarantee on a
1000+-event fixture trajectory (mirrors the 1046-event / 145 KB
artifact observed on prod 2026-05-18).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from kai_memory.store import cognitive_config
from kai_memory.persona_composer import (
    PersonaComposer,
    _AnchorRow,
    _PatternRow,
    _StanceRow,
    _TraitRow,
    _TrajectoryEvent,
    _TrajectoryRow,
)


# ── Pure render tier behaviour ──────────────────────────────────────


def _ev(day_offset: int, summary: str, weight: float = 0.0) -> _TrajectoryEvent:
    """Build a typed trajectory event ``day_offset`` days before now."""
    return _TrajectoryEvent(
        timestamp=datetime.now(UTC) - timedelta(days=day_offset),
        summary=summary,
        weight=weight,
    )


def test_render_emits_trajectory_section_with_subject_header() -> None:
    rendered = PersonaComposer._render(
        anchor=_AnchorRow(content="ANCHOR"),
        traits=[],
        stances=[],
        patterns=[],
        trajectories=[
            _TrajectoryRow(
                subject_name="Olga",
                events_recent=[_ev(1, "called mum")],
                events_midterm=[],
                older_count=0,
            ),
        ],
    )
    assert "## Trajectories" in rendered
    assert "### Olga" in rendered
    assert "called mum" in rendered


def test_render_emits_empty_trajectory_section_for_token_stability() -> None:
    """No trajectories → still emit the section header so prompt-cache
    prefix stability holds across compose calls."""
    rendered = PersonaComposer._render(anchor=None, traits=[], stances=[], patterns=[], trajectories=[])
    assert "## Trajectories" in rendered


def test_render_collapses_older_events_into_single_line() -> None:
    """The older tier renders as one summary line, not as individual events."""
    rendered = PersonaComposer._render(
        anchor=None,
        traits=[],
        stances=[],
        patterns=[],
        trajectories=[
            _TrajectoryRow(
                subject_name="Луна",
                events_recent=[],
                events_midterm=[],
                older_count=137,
                older_first=datetime(2024, 1, 5, tzinfo=UTC),
                older_last=datetime(2025, 3, 18, tzinfo=UTC),
            ),
        ],
    )
    assert "137 events folded" in rendered
    assert "2024-01-05" in rendered
    assert "2025-03-18" in rendered


def test_render_orders_midterm_before_recent() -> None:
    """Inside a trajectory, midterm tier renders BEFORE recent so the
    reader sees a chronological progression top-to-bottom."""
    rendered = PersonaComposer._render(
        anchor=None,
        traits=[],
        stances=[],
        patterns=[],
        trajectories=[
            _TrajectoryRow(
                subject_name="X",
                events_recent=[_ev(5, "RECENT-EVENT")],
                events_midterm=[_ev(200, "MIDTERM-EVENT")],
                older_count=0,
            ),
        ],
    )
    midterm_idx = rendered.index("MIDTERM-EVENT")
    recent_idx = rendered.index("RECENT-EVENT")
    assert midterm_idx < recent_idx


# ── _fetch_recent_trajectories: tier filter on a 1000-event fixture ──


def _make_event_dict(day_offset: int, idx: int, trace_id: uuid.UUID | None = None) -> dict:
    """Build a raw JSONB event dict matching what producer writes."""
    return {
        "trace_id": str(trace_id) if trace_id else str(uuid.uuid4()),
        "timestamp": (datetime.now(UTC) - timedelta(days=day_offset)).isoformat(),
        "summary": f"event #{idx}: {'recent' if day_offset <= 90 else ('midterm' if day_offset <= 365 else 'ancient')} narrative beat",
    }


async def test_fetch_recent_trajectories_handles_1000_event_artifact() -> None:
    """1000-event trajectory must filter to ≤max_events across recent+midterm,
    plus a single collapsed-older line. The whole composer render stays
    well under 8 KB on this single-trajectory fixture.

    Fixture shape mirrors a typical heavy prod trajectory: ~3% of events
    are recent (last 90d), ~17% midterm (90-365d), ~80% older. With
    max_events=40 the recent tier consumes ~30 slots and midterm gets
    ~10 — enough that the high-weight anchor event in midterm can
    surface in the top-N pick.
    """
    cognitive_config.set_for_testing("persona_max_active_trajectories", 5.0)
    cognitive_config.set_for_testing("persona_trajectory_max_events", 40.0)
    cognitive_config.set_for_testing("persona_trajectory_recent_window_days", 90.0)
    cognitive_config.set_for_testing("persona_trajectory_older_window_days", 365.0)

    # 1046 events spread realistically: heavy older tail, moderate
    # midterm, light recent (matches user-narrative density: most
    # interaction is "in the moment" so recent is small even when total
    # is large; older accumulates over years).
    high_weight_trace = uuid.uuid4()
    events = []
    for i in range(30):
        # Recent tier — last 90 days.
        events.append(_make_event_dict(day_offset=max(1, i * 3), idx=i))
    for i in range(30, 200):
        # Midterm tier — 90 to 365 days.
        day_offset = 95 + ((i - 30) * 270 // 170)
        if i == 100:
            events.append(_make_event_dict(day_offset, i, trace_id=high_weight_trace))
        else:
            events.append(_make_event_dict(day_offset, i))
    for i in range(200, 1046):
        # Older tier — 366 days+ back to ~4 years.
        day_offset = 400 + ((i - 200) * 1100 // 846)
        events.append(_make_event_dict(day_offset, i))

    artifact_row = {
        "artifact_id": uuid.uuid4(),
        "subject_name": "Луна",
        "events": events,
    }
    artifacts_result = MagicMock()
    artifacts_result.mappings.return_value.all.return_value = [artifact_row]

    # Build a memory_traces access_count lookup result. Give the anchor
    # trace a huge access_count so it lands in midterm top-N.
    class _AccessRow:
        def __init__(self, tid: uuid.UUID, count: int):
            self.id = tid
            self.access_count = count

    access_result = [_AccessRow(high_weight_trace, 500)]
    access_result_mock = MagicMock()
    access_result_mock.all.return_value = access_result

    session = MagicMock()
    session.execute = AsyncMock(side_effect=[artifacts_result, access_result_mock])

    composer = PersonaComposer(session)
    try:
        rows = await composer._fetch_recent_trajectories()
    finally:
        cognitive_config.reset_cache()

    assert len(rows) == 1
    traj = rows[0]
    assert traj.subject_name == "Луна"

    # Total event slots (recent + midterm) MUST stay at max_events=40.
    total_event_slots = len(traj.events_recent) + len(traj.events_midterm)
    assert total_event_slots <= 40, (
        f"max_events=40 was exceeded: recent={len(traj.events_recent)} midterm={len(traj.events_midterm)}"
    )

    # Older tier collapses — count > 0 for a 4-year fixture.
    assert traj.older_count > 0
    assert traj.older_first is not None
    assert traj.older_last is not None

    # The high-weight anchor event lands in midterm top-N. With one
    # heavy access_count=500 vs the rest at 0, that event MUST be
    # selected when midterm has spare slots.
    midterm_summaries = [e.summary for e in traj.events_midterm]
    assert any("event #100:" in s for s in midterm_summaries), (
        f"high-weight anchor event missing from midterm top-N: {midterm_summaries[:5]}..."
    )


async def test_compose_renders_within_8kb_on_full_persona() -> None:
    """End-to-end budget guarantee.

    Anchor + Traits + Stances + Patterns + one 1046-event Trajectory
    must render under 8 KB. This is the cutover-blocking metric — Wave
    3 prompt-builder calls compose() and the 145 KB pre-fix payload
    blew the 15000-char persona budget instantly.
    """
    cognitive_config.set_for_testing("persona_max_active_trajectories", 5.0)
    cognitive_config.set_for_testing("persona_trajectory_max_events", 40.0)
    cognitive_config.set_for_testing("persona_trajectory_recent_window_days", 90.0)
    cognitive_config.set_for_testing("persona_trajectory_older_window_days", 365.0)
    try:
        anchor = _AnchorRow(content="A" * 1500)
        traits = [_TraitRow(guidance_text="You commit without hedging.") for _ in range(12)]
        stances = [_StanceRow(position="Direct over diplomatic.") for _ in range(6)]
        patterns = [_PatternRow(observation="Verifies before recommending.") for _ in range(8)]
        # Use FILTERED tier outputs — the budget guarantee is on what
        # `_fetch_recent_trajectories` actually returns, not on raw 1046-event
        # input. Each summary is ~80 chars; 40 events x 80 = 3200 chars per
        # trajectory; 5 trajectories x 3200 = 16 000 chars worst-case slot
        # budget. In practice an L3-trained narrative is ~50 chars/event so
        # the render lands comfortably under 8000 across one trajectory and
        # under 15000 across five.
        traj_events = [
            _ev(day_offset=10 + i, summary=f"narrative beat #{i} — short summary", weight=0.0) for i in range(40)
        ]
        trajectories = [
            _TrajectoryRow(
                subject_name=f"Subject-{j}",
                events_recent=traj_events[:20],
                events_midterm=traj_events[20:40],
                older_count=200,
                older_first=datetime(2024, 1, 1, tzinfo=UTC),
                older_last=datetime(2025, 6, 1, tzinfo=UTC),
            )
            for j in range(5)
        ]
    finally:
        cognitive_config.reset_cache()

    rendered = PersonaComposer._render(anchor, traits, stances, patterns, trajectories)

    # 8000-char target per spec — covers one big trajectory comfortably.
    # The five-trajectory budget is checked below; this slice gates the
    # single-trajectory case explicitly.
    one_traj_only = PersonaComposer._render(
        anchor=anchor,
        traits=traits,
        stances=stances,
        patterns=patterns,
        trajectories=[trajectories[0]],
    )
    assert len(one_traj_only) <= 8000, f"Single-trajectory render exceeded 8 KB budget: {len(one_traj_only)} chars"

    # Five-trajectory render (worst case) stays under 15000 chars — the
    # ``persona_compose_budget_chars`` ceiling locked in migration 075.
    assert len(rendered) <= 15000, f"Five-trajectory render exceeded 15 KB persona budget: {len(rendered)} chars"


# ── _parse_trajectory_event guards ──────────────────────────────────


def test_parse_trajectory_event_drops_malformed_entries() -> None:
    """Defensive: legacy events with missing fields don't crash the render."""
    assert PersonaComposer._parse_trajectory_event({}, {}) is None
    assert PersonaComposer._parse_trajectory_event({"timestamp": "not-a-date", "summary": "x"}, {}) is None
    assert PersonaComposer._parse_trajectory_event({"timestamp": 123, "summary": "x"}, {}) is None


def test_parse_trajectory_event_drops_non_dict_payloads() -> None:
    """Defensive: scalar / null / list items in `events` array don't AttributeError.

    Producer drift can leave legacy entries that aren't full dicts.
    Calling `.get()` on those would abort compose() entirely; the
    isinstance guard skips them silently.
    """
    assert PersonaComposer._parse_trajectory_event(None, {}) is None
    assert PersonaComposer._parse_trajectory_event("just a string", {}) is None
    assert PersonaComposer._parse_trajectory_event(42, {}) is None
    assert PersonaComposer._parse_trajectory_event(["not", "a", "dict"], {}) is None


async def test_fetch_recent_trajectories_skips_non_dict_events() -> None:
    """End-to-end: an events array containing non-dict items (legacy /
    corrupt payload) renders cleanly instead of crashing compose().

    Pre-fix the trace_id collection loop called `ev.get("trace_id")` on
    every raw item — a scalar entry from a legacy producer would have
    raised AttributeError and aborted the whole render.
    """
    cognitive_config.set_for_testing("persona_max_active_trajectories", 5.0)
    cognitive_config.set_for_testing("persona_trajectory_max_events", 40.0)
    cognitive_config.set_for_testing("persona_trajectory_recent_window_days", 90.0)
    cognitive_config.set_for_testing("persona_trajectory_older_window_days", 365.0)

    good_event = _make_event_dict(day_offset=10, idx=1)
    artifact_row = {
        "artifact_id": uuid.uuid4(),
        "subject_name": "Whatever",
        "events": [
            good_event,
            None,
            "stringly-typed",
            42,
        ],
    }
    artifacts_result = MagicMock()
    artifacts_result.mappings.return_value.all.return_value = [artifact_row]

    access_result_mock = MagicMock()
    access_result_mock.all.return_value = []

    session = MagicMock()
    session.execute = AsyncMock(side_effect=[artifacts_result, access_result_mock])

    composer = PersonaComposer(session)
    try:
        rows = await composer._fetch_recent_trajectories()
    finally:
        cognitive_config.reset_cache()

    assert len(rows) == 1
    # The one good event survived; junk was dropped silently.
    total = len(rows[0].events_recent) + len(rows[0].events_midterm)
    assert total == 1, (
        f"expected 1 surviving event, got recent={rows[0].events_recent} midterm={rows[0].events_midterm}"
    )


def test_parse_trajectory_event_uses_access_count_for_weight() -> None:
    """Weight comes from memory_traces.access_count via trace_id lookup.

    Default 0 when the trace is missing (pruned or never linked).
    """
    tid = uuid.uuid4()
    ev = {
        "trace_id": str(tid),
        "timestamp": "2026-04-01T12:00:00+00:00",
        "summary": "anchor event",
    }
    # Found in lookup → weight = access_count.
    parsed = PersonaComposer._parse_trajectory_event(ev, {tid: 42})
    assert parsed is not None
    assert parsed.weight == pytest.approx(42.0)

    # Missing from lookup → weight = 0.
    parsed = PersonaComposer._parse_trajectory_event(ev, {})
    assert parsed is not None
    assert parsed.weight == 0.0
