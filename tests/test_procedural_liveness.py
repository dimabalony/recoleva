"""DB-487 ride-along (b) — the pure predicate behind the nightly liveness warning.

The DB-object side is covered in tests/integration/test_procedural_phase_c.py; this
pins the decision itself, which is the part that has to be exactly right: a check that
fires on day one is noise nobody reads, and a check that never fires is what let
DB-436 sit unnoticed for 26 days.
"""

from __future__ import annotations

import pytest

from kai_memory.procedural_liveness import STALE_DAYS, is_stale


def test_rules_with_none_active_past_the_window_is_stale():
    """The DB-436 signature exactly: 6 rules, 0 active, sitting there for weeks."""
    assert is_stale(total=6, active=0, oldest_age_days=26.0) is True


def test_freshly_captured_rules_are_not_yet_stale():
    """A rule minted an hour ago and not yet active is normal, not an alarm."""
    assert is_stale(total=1, active=0, oldest_age_days=0.2) is False


def test_an_empty_store_is_not_stale():
    """Capture enabled with nothing captured is a different (and normal) state — the
    conversations simply contained no teaching. Warning here would train the operator
    to ignore the line, which is how the real signal gets lost."""
    assert is_stale(total=0, active=0, oldest_age_days=0.0) is False


def test_any_active_rule_clears_the_alarm():
    assert is_stale(total=9, active=1, oldest_age_days=99.0) is False


@pytest.mark.parametrize(
    ("age", "expected"),
    [(STALE_DAYS - 0.1, False), (STALE_DAYS, True), (STALE_DAYS + 10, True)],
)
def test_boundary_is_inclusive_at_stale_days(age, expected):
    assert is_stale(total=3, active=0, oldest_age_days=age) is expected
