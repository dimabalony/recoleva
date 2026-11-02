"""Structural guards on the relationship-classifier prompt (PR-F).

The 2026-05-18 → 2026-05-20 prod data showed 17+ bad rels emitted by Sonnet:
direction inversions (Indonesia part_of Corfu), slash-bundled predicates
(employee_of / candidate_at), made-up predicates (transitioning_to_member_of).
PR-B (#153) added a validator that defensively rejects those at insert time.
PR-F (this PR) attacks the root cause — embed the canonical allowlist + direction
examples in the prompt so Sonnet emits canonical predicates in the first place.

This module locks the prompt-render invariants so a future change can't
silently drop the allowlist or the direction guidance.
"""

from __future__ import annotations

import asyncio
import re
from unittest.mock import patch

import pytest

from kai_memory.consolidation.consolidation import (
    _RELATIONSHIP_PREDICATE_ALLOWLIST,
    _DIRECTION_EXAMPLES,
    _classify_pair_relationship,
    _render_allowlist_for_prompt,
)


# ── allowlist render ─────────────────────────────────────────────────


def test_render_allowlist_includes_every_predicate() -> None:
    """Every predicate in the allowlist must appear in the prompt block.

    If a future PR adds a predicate to ``_RELATIONSHIP_PREDICATE_ALLOWLIST``,
    it surfaces in the prompt automatically — this test fails if the
    renderer drifts out of sync.
    """
    rendered = _render_allowlist_for_prompt()
    for predicate in _RELATIONSHIP_PREDICATE_ALLOWLIST:
        assert predicate in rendered, f"predicate {predicate!r} missing from rendered allowlist"


def test_render_allowlist_marks_symmetric_predicates() -> None:
    """Symmetric predicates must carry a `(symmetric)` annotation.

    Sonnet uses the marker to know which predicates allow either ordering
    of (from, to). Asymmetric predicates must NOT carry it.
    """
    rendered = _render_allowlist_for_prompt()
    # partner_of is symmetric.
    assert re.search(r"^- partner_of .*\(symmetric\)$", rendered, re.MULTILINE)
    # employee_of is asymmetric.
    assert re.search(r"^- employee_of —.*organization", rendered, re.MULTILINE)
    assert not re.search(r"^- employee_of —.*\(symmetric\)", rendered, re.MULTILINE)


def test_render_allowlist_carries_direction_label_from_rule() -> None:
    """Each line's direction text comes from the rule's ``direction_label``.

    Verifies the renderer pulls from the source-of-truth dataclass rather
    than hardcoding text.
    """
    rendered = _render_allowlist_for_prompt()
    for predicate, rule in _RELATIONSHIP_PREDICATE_ALLOWLIST.items():
        # Match line ``- {predicate} — {direction_label}`` (followed by optional symmetric marker).
        pattern = rf"^- {re.escape(predicate)} — {re.escape(rule.canonical_direction)}"
        assert re.search(pattern, rendered, re.MULTILINE), (
            f"line for {predicate!r} missing canonical_direction={rule.canonical_direction!r}"
        )


# ── direction examples ──────────────────────────────────────────────


def test_direction_examples_cover_known_inversion_traps() -> None:
    """The example block has to address the specific inversion patterns we
    saw in prod (May 18-20). Each pattern below was bad-direction emitted
    at least once on prod.
    """
    examples_text = "\n".join(_DIRECTION_EXAMPLES)
    assert "user_of" in examples_text, "Kai user_of trap missing — saw 'Kai user_of Elena'"
    assert "hometown_of" in examples_text, "Minsk/Katya direction trap missing"
    assert "part_of" in examples_text, "Indonesia/Corfu hierarchy trap missing"
    assert "owner_of" in examples_text, "облачками/Olga owner_of trap missing"
    assert "employee_of" in examples_text or "customer_of" in examples_text, (
        "org-direction trap missing (Examplecorp or Shopee)"
    )


def test_direction_examples_use_NOT_phrasing() -> None:
    """Each example shows the wrong direction explicitly with 'NOT'.

    The 'NOT X' construct teaches Sonnet to recognize inverted patterns
    instead of just memorizing the correct one. Without 'NOT', Sonnet
    pattern-matches positively but stays vulnerable to subtle inversions
    (e.g. 'Corfu (place) located_in Indonesia (place)' vs 'Indonesia part_of Corfu').
    """
    has_not_pattern = sum(1 for ex in _DIRECTION_EXAMPLES if "NOT " in ex)
    # Most examples should carry the NOT — at least 4 of 6 to keep flexibility.
    assert has_not_pattern >= 4, f"Only {has_not_pattern} of {len(_DIRECTION_EXAMPLES)} examples carry 'NOT' phrasing"


# ── _classify_pair_relationship prompt assembly ─────────────────────


def _make_pair() -> dict:
    return {
        "name_a": "Misha",
        "name_b": "Examplecorp",
        "shared_contents": ["Misha joined Examplecorp in 2021 and left in 2023"],
        "existing_rels_a": [],
        "existing_rels_b": [],
    }


def test_classifier_prompt_embeds_allowlist_block() -> None:
    """The prompt passed to Sonnet contains the rendered allowlist block."""
    captured: dict[str, str] = {}

    async def fake_call(prompt: str, _span: str, *, thinking_mode: str = "adaptive") -> str:
        captured["prompt"] = prompt
        return '{"relationship_type": "no_relationship"}'

    with patch("kai_memory.consolidation.consolidation._call_sonnet_text", side_effect=fake_call):
        result = asyncio.run(_classify_pair_relationship(_make_pair()))

    assert result is None  # no_relationship → returns None
    prompt = captured["prompt"]
    # Every allowlisted predicate must be in the prompt.
    for predicate in _RELATIONSHIP_PREDICATE_ALLOWLIST:
        assert predicate in prompt, f"predicate {predicate!r} missing from classifier prompt"


def test_classifier_prompt_includes_reject_instruction() -> None:
    """The prompt explicitly tells Sonnet to return ``no_relationship``
    when no allowlisted predicate fits, rather than inventing one.
    """
    captured: dict[str, str] = {}

    async def fake_call(prompt: str, _span: str, *, thinking_mode: str = "adaptive") -> str:
        captured["prompt"] = prompt
        return "null"

    with patch("kai_memory.consolidation.consolidation._call_sonnet_text", side_effect=fake_call):
        asyncio.run(_classify_pair_relationship(_make_pair()))

    prompt = captured["prompt"]
    assert "no_relationship" in prompt, "reject instruction missing — Sonnet won't know to self-reject"
    assert "rejected at validation time" in prompt, (
        "validator-rejection warning missing — Sonnet needs to know invalid predicates fail downstream"
    )


def test_classifier_prompt_includes_direction_examples() -> None:
    """The 6 direction examples land in the prompt verbatim."""
    captured: dict[str, str] = {}

    async def fake_call(prompt: str, _span: str, *, thinking_mode: str = "adaptive") -> str:
        captured["prompt"] = prompt
        return "null"

    with patch("kai_memory.consolidation.consolidation._call_sonnet_text", side_effect=fake_call):
        asyncio.run(_classify_pair_relationship(_make_pair()))

    prompt = captured["prompt"]
    for example in _DIRECTION_EXAMPLES:
        assert example in prompt, f"direction example missing from prompt: {example!r}"


def test_classifier_returns_none_on_no_relationship_token() -> None:
    """Sonnet returning ``{"relationship_type": "no_relationship"}`` maps
    to ``None`` so the downstream pipeline treats it as "no clear rel"
    rather than emitting a Validation-rejection log line.
    """

    async def fake_call(_prompt: str, _span: str, *, thinking_mode: str = "adaptive") -> str:
        return '{"relationship_type": "no_relationship", "confidence": 0.0}'

    with patch("kai_memory.consolidation.consolidation._call_sonnet_text", side_effect=fake_call):
        result = asyncio.run(_classify_pair_relationship(_make_pair()))

    assert result is None


def test_classifier_accepts_canonical_predicate_unchanged() -> None:
    """Sanity check: a normal canonical-predicate response still parses to a dict."""

    async def fake_call(_prompt: str, _span: str, *, thinking_mode: str = "adaptive") -> str:
        return (
            '{"relationship_type": "employee_of", "description": "Misha at Examplecorp 2021-2023", '
            '"confidence": 0.9, "conflict_action": "none", "conflicts_with": []}'
        )

    with patch("kai_memory.consolidation.consolidation._call_sonnet_text", side_effect=fake_call):
        result = asyncio.run(_classify_pair_relationship(_make_pair()))

    assert result is not None
    assert result["relationship_type"] == "employee_of"
    assert result["confidence"] == pytest.approx(0.9)
