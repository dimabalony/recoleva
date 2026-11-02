"""Type definitions and validation sets for cognitive memory domain."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

MemoryType = Literal[
    "fact",
    "episode",
    "impression",
    "preference",
    "relationship",
    "plan",
    "intention",
    "status_change",
    "decision",
    "social_norm",
    "self_reflection",
    "procedural",
]
SourceType = Literal["conversation", "meeting", "consolidation", "self_reflection"]
EntityType = Literal[
    "person",
    "animal",
    "place",
    "organization",
    "topic",
    "project",
    "object",
    "event",
    "group",
    "agent",
]
LinkRole = Literal["subject", "about", "mentioned", "participant"]
RunStatus = Literal["running", "completed", "failed", "deferred"]


class ConflictAction(StrEnum):
    NONE = "none"
    REPLACE = "replace"
    SAME_PERSON = "same_person"
    SEQUENTIAL = "sequential"


class MergeCandidateStatus(StrEnum):
    PENDING = "pending"
    MERGED = "merged"
    REJECTED = "rejected"
    # Tier 3 (entity-resolution v2): a candidate HELD for human review — the
    # nightly merge sweep (`_merge_entities`) only executes PENDING, so PROPOSED
    # rows never auto-merge. The `status` column is a plain VARCHAR(20) with no
    # CHECK/enum constraint (migration 051), so this value needs no migration.
    PROPOSED = "proposed"


MEMORY_TYPES: frozenset[str] = frozenset(
    {
        "fact",
        "episode",
        "impression",
        "preference",
        "relationship",
        "plan",
        "intention",
        "status_change",
        "decision",
        "social_norm",
        "self_reflection",
        "procedural",
    }
)
# Types eligible for L1 priming retrieval. An ALLOWLIST (not a procedural
# denylist) so a future new memory_type can't silently leak into priming — it
# must be added here deliberately. ``procedural`` (Phase A) is intentionally out:
# procedural candidates are served via their own advisory path, never as an L1
# scoring weight (#241-runaway guard).
SERVABLE_MEMORY_TYPES: frozenset[str] = MEMORY_TYPES - {"procedural"}
VALID_SOURCE_TYPES: frozenset[str] = frozenset({"conversation", "meeting", "consolidation", "self_reflection"})
VALID_ENTITY_TYPES: frozenset[str] = frozenset(
    {
        "person",
        "animal",
        "place",
        "organization",
        "topic",
        "project",
        "object",
        "event",
        "group",
        # Sub-design C INC 1: Kai-the-assistant gets a real type instead of
        # collapsing to 'topic'/'person'. The resolver keys on
        # (name.lower(), entity_type), so agent-Kai never merges into person-Kai.
        "agent",
    }
)
VALID_LINK_ROLES: frozenset[str] = frozenset({"subject", "about", "mentioned", "participant"})
VALID_RUN_STATUSES: frozenset[str] = frozenset({"running", "completed", "failed", "deferred"})

# #273 — the intention component's static weight.
#
# Sizing rationale (the value the sweep arbitrates around): under the forgery an
# intention received `1.0*w_semantic + 1.0*w_graph` = 0.35 + 0.25 = 0.60 of
# unearned composite score, on top of whatever recency/access/arousal it truly
# earned. That is why they crowded gold out. A weight here is NOT a like-for-like
# replacement for that 0.60 — the whole point is that intentions surface on their
# own merit, at a level that keeps them servable when genuinely relevant without
# outranking traces that actually answer the question.
#
# Sizing against the SCALE of the honest components, which is the trap here:
# semantic is a *weighted* signal, so the most any trace can earn from a perfect
# semantic match is 0.35 — not 1.0. A naive "0.30, just under the 0.60 forgery"
# therefore still outranks a trace matching at semantic 0.85 (0.85*0.35 =
# 0.2975 < 0.30), i.e. it reproduces the crowding-out bug at a lower number.
# The intention component is a BONUS on top of the recency/access/arousal an
# intention already earns, so it must be sized against a realistic composite,
# not against the forgery it replaces.
#
# 0.15 is the pre-sweep candidate. At w=0.15 (with a fresh intention also
# earning ~0.15 recency ⇒ ~0.30 total):
#   * a genuinely answering trace (semantic 0.85 ⇒ 0.2975, plus its own recency)
#     ranks AHEAD — gold stops being crowded out;
#   * a weak-but-sticky trace (the ~59% band at semantic ~0.34 ⇒ 0.119) ranks
#     BEHIND — a relevant intention still surfaces, which is the point of
#     Path 4. Both directions are pinned by tests.
# CALIBRATION STATUS: provisional pending the paired sweep — the issue requires
# the weight be measured, not guessed. Do not treat this as tuned until the PR
# body carries the paired numbers.
_INTENTION_WEIGHT: float = 0.15

DEFAULT_RETRIEVAL_WEIGHTS: dict[str, float] = {
    "semantic": 0.35,
    "graph": 0.25,
    "recency": 0.15,
    "access": 0.15,
    "arousal": 0.10,
    # A2: 6th component. DEFAULT 0.0 ⇒ inert — ``salience * 0.0 == 0`` keeps the
    # weighted sum (and ranking) bit-identical to the pre-salience 5-component scorer
    # until the operator tunes it up and measures recall@8.
    "salience": 0.0,
    # #273: 7th component. Intentions (prospective memories, Path 4) used to
    # FORGE ``semantic = 1.0`` and ``graph = 1.0`` on themselves — scores they
    # never earned — which crowded genuinely relevant traces out of the top-8
    # (measured: 48/70 → 56/70 golds@8 on the frozen pool when stripped).
    # Arriving via path P is not evidence of scoring X on signal P, so
    # intentions now carry their OWN component and semantic/graph keep whatever
    # each trace actually earned. The weight is what makes intentions surface;
    # it is a STATIC versioned constant (see migration 097) — deliberately not
    # adaptive. The #241 runaway (`_adjust_retrieval_weights` drove graph
    # 0.25 → 0.945 over ~267 nightly passes, costing recall@8 0.75 → 0.60) is
    # why: nothing in this dict self-adjusts, ever.
    "intention": _INTENTION_WEIGHT,
}
