"""Unit tests for Tier-3 entity resolution pure logic (design §3 blocking + 3b).

DB-touching producer behavior (candidate creation, PROPOSED-vs-PENDING, dedup,
guards, end-to-end merge) lives in tests/integration/test_entity_resolution_tier3.py.
"""

from __future__ import annotations

import uuid

from kai_memory.entity_resolution_tier3 import (
    BlockEntity,
    blocked_pairs,
    classify_deterministic,
    levenshtein_at_most_1,
)


def _e(canonical_key: str, entity_type: str = "person", tg: int | None = None) -> BlockEntity:
    return BlockEntity(uuid.uuid4(), entity_type, canonical_key, tg, 5)


# ── levenshtein_at_most_1 ────────────────────────────────────────────────────


def test_levenshtein_equal():
    assert levenshtein_at_most_1("danil", "danil") is True


def test_levenshtein_one_substitution():
    assert levenshtein_at_most_1("misha", "mysha") is True


def test_levenshtein_one_insertion():
    assert levenshtein_at_most_1("danil", "daniil") is True  # insert one 'i'


def test_levenshtein_one_deletion():
    assert levenshtein_at_most_1("daniil", "danil") is True


def test_levenshtein_two_edits_false():
    assert levenshtein_at_most_1("danil", "danya") is False  # 2 substitutions


def test_levenshtein_length_gap_gt_1_false():
    assert levenshtein_at_most_1("misha", "dimitri") is False


# ── classify_deterministic (3b) ──────────────────────────────────────────────


def test_subset_single_token_into_full_name():
    assert classify_deterministic("misha", "misha petrov") == "subset"
    assert classify_deterministic("misha petrov", "misha") == "subset"  # order-independent


def test_subset_identical_single_token():
    assert classify_deterministic("pavel", "pavel") == "subset"


def test_near_exact_multi_token_spelling_wobble():
    assert classify_deterministic("danil zosim", "daniil zosim") == "near_exact"


def test_near_exact_single_token_spelling_wobble():
    assert classify_deterministic("danil", "daniil") == "near_exact"


def test_near_exact_digit_guard_rejects_digit_difference():
    # A ≤1-edit wobble on a DIGIT is a different ticket/model/version, never a
    # spelling variant — must NOT classify as near_exact (auto-merge risk).
    assert classify_deterministic("db-216", "db-219") is None
    assert classify_deterministic("db-103", "db-100") is None
    assert classify_deterministic("v1", "v2") is None
    # multi-token: the digit-differing first token would otherwise be a near_exact
    assert classify_deterministic("db-216 alpha", "db-219 alpha") is None


def test_near_exact_digit_guard_allows_letter_wobble_with_equal_digits():
    # The guard only blocks DIGIT differences; a genuine letter wobble whose digits
    # match is still a near_exact.
    assert classify_deterministic("agent-2", "agemt-2") == "near_exact"
    assert classify_deterministic("danil zosim", "daniil zosim") == "near_exact"  # no digits at all


def test_diminutive_formal_is_not_deterministic():
    # misha↔mikhail is a name-equivalence problem — 3c territory, never 3b.
    assert classify_deterministic("misha", "mikhail") is None


def test_shared_given_name_different_surname_is_not_deterministic():
    # Two full names sharing only the given token: blocked (later) but NOT auto-merge.
    assert classify_deterministic("misha petrov", "misha smirnov") is None


def test_multi_token_subset_two_tokens_is_not_deterministic():
    # subset rule requires the shorter key be EXACTLY one token.
    assert classify_deterministic("danil zosim", "danil zosim moscow") is None


def test_empty_key_never_classifies():
    assert classify_deterministic("", "misha") is None
    assert classify_deterministic("misha", "") is None


# ── blocked_pairs (3a) ───────────────────────────────────────────────────────


def test_blocking_token_overlap():
    ents = [_e("misha"), _e("misha petrov")]
    assert blocked_pairs(ents) == [(0, 1)]


def test_blocking_first_token_prefix():
    # danil / daniil share no full token but share the 4-char prefix "dani".
    ents = [_e("danil"), _e("daniil")]
    assert blocked_pairs(ents) == [(0, 1)]


def test_blocking_excludes_cross_entity_type():
    ents = [_e("misha", "person"), _e("misha", "organization")]
    assert blocked_pairs(ents) == []  # same surface, different type → never paired


def test_blocking_disjoint_names_not_paired():
    ents = [_e("misha petrov"), _e("leo tolstoy")]
    assert blocked_pairs(ents) == []


def test_blocking_prefix_needs_min_4_chars():
    # 3-char first tokens never prefix-block (too short to be discriminating).
    ents = [_e("ann"), _e("ana")]
    assert blocked_pairs(ents) == []


def test_blocking_is_deduped_and_ordered():
    # Three entities all sharing "danil" → 3 unique unordered pairs, sorted.
    ents = [_e("danil"), _e("danil zosim"), _e("daniil zosim")]
    assert blocked_pairs(ents) == [(0, 1), (0, 2), (1, 2)]


def test_blocking_skips_empty_canonical_key():
    ents = [_e(""), _e("misha")]
    assert blocked_pairs(ents) == []


def test_blocking_caps_oversized_bucket():
    # A common token shared by many entities would expand k² — the bucket is
    # skipped past max_bucket (here 3) instead of producing every pair.
    ents = [_e("misha petrov"), _e("misha onoshko"), _e("misha ivanov"), _e("misha bystrov")]
    # All four share token "misha" (bucket size 4). With max_bucket=3 the token
    # bucket is skipped; the ≥4-char first-token prefix "misha" bucket is the same
    # four members and is likewise skipped → no pairs.
    assert blocked_pairs(ents, max_bucket=3) == []
    # Under the default cap the same four fully pair up (6 pairs).
    assert len(blocked_pairs(ents)) == 6
