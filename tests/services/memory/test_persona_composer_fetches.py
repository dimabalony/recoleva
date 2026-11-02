"""Unit tests for PersonaComposer Wave 2 fetch methods.

These tests mock the AsyncSession's ``execute`` to verify the SQL
parameter wiring + render-layer behaviour without needing the
testcontainer. The full DB-side ORDER BY ranking + trigger interactions
remain integration-test territory (currently skipped — see
``tests/integration/test_persona_composer.py`` for the CI cascade
issue).

Locked here: per-fetch SQL parameters from cognitive_config and the
deterministic render shape across all three sections + Anchor.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

from kai_memory.store import cognitive_config
from memory_test_support import ChatContext
from kai_memory.persona_composer import (
    PersonaComposer,
    _AnchorRow,
    _PatternRow,
    _StanceRow,
    _TraitRow,
)


def _make_chat_ctx() -> ChatContext:
    return ChatContext(
        chat_id=12345,
        topic_id=None,
        is_group=False,
        venue="telegram",
        sender_user_id=99,
        sender_name="user",
    )


def _stub_session_with_rows(per_query_rows: list[list[dict]]) -> MagicMock:
    """Build a session whose ``execute`` returns a different ``rows`` set
    for each call, in order."""
    session = MagicMock()
    results = []
    for rows in per_query_rows:
        result = MagicMock()
        result.mappings.return_value.first.return_value = rows[0] if rows else None
        result.mappings.return_value.all.return_value = rows
        results.append(result)
    session.execute = AsyncMock(side_effect=results)
    return session


# ── render shape ───────────────────────────────────────────


def test_render_emits_all_section_headers_when_lists_are_empty() -> None:
    """Empty fetches still emit all section headers (Personality / Views /
    Recently observed / Trajectories).

    Design §3.7 determinism: keeps token boundaries stable across
    consecutive compose() calls. The Trajectories header lands here in
    Wave 3 (see PR-C) so the cache-prefix shape doesn't shift again at
    cutover.
    """
    rendered = PersonaComposer._render(_AnchorRow(content="ANCHOR"), [], [], [], [])
    assert "ANCHOR" in rendered
    assert "## Personality" in rendered
    assert "## Views" in rendered
    assert "## Recently observed" in rendered
    assert "## Trajectories" in rendered


def test_render_renders_each_layer_as_bullet_lines() -> None:
    rendered = PersonaComposer._render(
        _AnchorRow(content="A"),
        traits=[_TraitRow(guidance_text="T1"), _TraitRow(guidance_text="T2")],
        stances=[_StanceRow(position="S1")],
        patterns=[_PatternRow(observation="P1")],
        trajectories=[],
    )
    assert "- T1" in rendered
    assert "- T2" in rendered
    assert "- S1" in rendered
    assert "- P1" in rendered


def test_render_with_no_anchor_omits_anchor_text_but_keeps_headers() -> None:
    rendered = PersonaComposer._render(None, [], [], [], [])
    assert "## Personality" in rendered
    assert "## Views" in rendered
    assert "## Recently observed" in rendered
    assert "## Trajectories" in rendered


# ── compose() wires fetch methods ──────────────────────────


async def test_compose_calls_all_three_fetches_and_renders() -> None:
    """compose() invokes Anchor + Trait + Stance + Pattern fetches and
    composes them into a single rendered block."""
    session = MagicMock()
    composer = PersonaComposer(session)

    # Replace each fetch with a small async stub returning a known shape.
    async def _fetch_anchor():
        return _AnchorRow(content="ANCHOR-CONTENT")

    async def _fetch_traits():
        return [_TraitRow(guidance_text="TRAIT-A")]

    async def _fetch_stances(chat_ctx):
        del chat_ctx
        return [_StanceRow(position="STANCE-A")]

    async def _fetch_patterns():
        return [_PatternRow(observation="PATTERN-A")]

    async def _fetch_trajectories():
        return []

    composer._fetch_anchor = _fetch_anchor  # type: ignore[method-assign]
    composer._fetch_active_traits = _fetch_traits  # type: ignore[method-assign]
    composer._fetch_relevant_stances = _fetch_stances  # type: ignore[method-assign]
    composer._fetch_recent_patterns = _fetch_patterns  # type: ignore[method-assign]
    composer._fetch_recent_trajectories = _fetch_trajectories  # type: ignore[method-assign]

    out = await composer.compose(_make_chat_ctx())
    assert "ANCHOR-CONTENT" in out
    # Canonical order: Anchor → Personality → Views → Recently observed → Trajectories.
    anchor_idx = out.index("ANCHOR-CONTENT")
    trait_idx = out.index("TRAIT-A")
    stance_idx = out.index("STANCE-A")
    pattern_idx = out.index("PATTERN-A")
    traj_idx = out.index("## Trajectories")
    assert anchor_idx < trait_idx < stance_idx < pattern_idx < traj_idx


async def test_compose_anchor_only_unchanged_by_wave2() -> None:
    """compose_anchor_only renders ONLY the Anchor + empty section headers.

    This is what prompt-builder calls in Wave 1 + Wave 2. Wave 3 swaps
    the call site; this test prevents that swap from happening
    accidentally during Wave 2 review.
    """
    session = MagicMock()
    composer = PersonaComposer(session)

    async def _fetch_anchor():
        return _AnchorRow(content="ANCHOR")

    # If compose_anchor_only ever starts calling Trait/Stance/Pattern
    # fetches, this stub would never be hit and the test would fail.
    async def _should_not_be_called(*args, **kwargs):
        raise AssertionError("compose_anchor_only must NOT fetch Trait/Stance/Pattern in Wave 2")

    composer._fetch_anchor = _fetch_anchor  # type: ignore[method-assign]
    composer._fetch_active_traits = _should_not_be_called  # type: ignore[method-assign]
    composer._fetch_relevant_stances = _should_not_be_called  # type: ignore[method-assign]
    composer._fetch_recent_patterns = _should_not_be_called  # type: ignore[method-assign]

    out = await composer.compose_anchor_only(_make_chat_ctx())
    assert "ANCHOR" in out
    assert "## Personality" in out
    assert "- " not in out  # No bullet lines emitted.


# ── fetch method SQL parameters ────────────────────────────


async def test_fetch_active_traits_uses_cognitive_config_lambda_and_limit() -> None:
    """Inspect the kwargs passed into session.execute to lock parameter wiring."""
    session = _stub_session_with_rows([[]])
    composer = PersonaComposer(session)

    cognitive_config.set_for_testing("persona_max_active_traits", 7.0)
    cognitive_config.set_for_testing("persona_trait_decay_lambda", 0.01)
    try:
        await composer._fetch_active_traits()
    finally:
        cognitive_config.reset_cache()

    args, kwargs = session.execute.call_args
    # Second positional arg is the params dict.
    params = args[1] if len(args) > 1 else kwargs.get("params", {})
    assert params.pop("now").tzinfo is not None
    assert params == {"lambda_": 0.01, "limit": 7}


async def test_fetch_recent_patterns_uses_cognitive_config_window_and_limit() -> None:
    session = _stub_session_with_rows([[]])
    composer = PersonaComposer(session)

    cognitive_config.set_for_testing("persona_max_active_patterns", 5.0)
    cognitive_config.set_for_testing("persona_pattern_window_days", 14.0)
    try:
        await composer._fetch_recent_patterns()
    finally:
        cognitive_config.reset_cache()

    args, _ = session.execute.call_args
    params = args[1]
    assert params.pop("now").tzinfo is not None
    assert params == {"window_days": 14.0, "limit": 5}


async def test_fetch_relevant_stances_uses_cognitive_config_lambda_and_limit() -> None:
    session = _stub_session_with_rows([[]])
    composer = PersonaComposer(session)

    cognitive_config.set_for_testing("persona_max_active_stances", 3.0)
    cognitive_config.set_for_testing("persona_stance_decay_lambda", 0.05)
    try:
        await composer._fetch_relevant_stances(_make_chat_ctx())
    finally:
        cognitive_config.reset_cache()

    args, _ = session.execute.call_args
    params = args[1]
    assert params.pop("now").tzinfo is not None
    assert params == {"lambda_": 0.05, "limit": 3}


async def test_fetch_active_traits_drops_rows_with_null_guidance_text() -> None:
    """Defensive: NULL guidance_text rows are filtered (shouldn't happen,
    but the schema allows the JSONB key to be absent)."""
    session = _stub_session_with_rows([[{"guidance_text": "T1"}, {"guidance_text": None}]])
    composer = PersonaComposer(session)
    rows = await composer._fetch_active_traits()
    assert len(rows) == 1
    assert rows[0].guidance_text == "T1"


# ── budget warning (smoke) ─────────────────────────────────


def test_budget_warn_fires_when_over_limit(caplog) -> None:
    PersonaComposer._budget_warn("x" * 5000, budget_chars=4000)
    assert any("exceeded budget_chars" in rec.message for rec in caplog.records)


def test_budget_warn_silent_under_limit(caplog) -> None:
    PersonaComposer._budget_warn("short", budget_chars=4000)
    assert not any("exceeded budget_chars" in rec.message for rec in caplog.records)


# ── P0 budget bump (2026-05-11) ────────────────────────────


def test_persona_compose_budget_default_is_18000() -> None:
    """Default value must be 18000 chars (Wave 3 cutover sizing).

    Migration history:

    * 067 set the key at 4000 (Anchor-only era).
    * 072 bumped 4000 → 6000 after prompt_builder started reading the
      config (10 breaches in 5h with the 4000 ceiling).
    * 075 bumps 6000 → 15000 for Wave 3: ``compose()`` adds Traits +
      Stances + Patterns under the Anchor. 2026-05-16 prod smoke
      measured ``compose_ok len=12404`` against ``ai_workforce``.
    * 083 bumps 15000 → 18000 after the pre-cutover smoke measured
      17864 chars against prod.

    Locking the default here so a regression can't silently revert and
    re-emit the "exceeded budget_chars" warning on every prompt build.
    """
    from kai_memory.store.cognitive_config import _DEFAULTS

    assert _DEFAULTS["persona_compose_budget_chars"] == 18000.0


# Touch unused so ChatContext stays a soft dep for typing only.
_ = uuid
