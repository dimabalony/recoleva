"""Tier 3 entity resolution (entity-resolution v2 design §3) — nightly
blocked-candidate merge for the partial-name / mixed-script / spelling fork
class the write-path tiers can't catch (`Misha`⊂`Misha Petrov`, `Павел`/`Павелл
Петров`, `Pavel`×2). Runs in L3 consolidation, NOT the hot write-path.

Pipeline:
- **3a Blocking** (`blocked_pairs`): a wide, cheap recall net over active entities
  of ONE ``entity_type`` — pairs sharing a ``canonical_key`` token OR a ≥4-char
  first-token prefix. In-memory inverted index (O(n + pairs), not O(n²)); the
  full same-type set is swept each run (no created_at window — a fork of any age
  that keeps growing must stay catchable).
- **3b Deterministic classify** (`classify_deterministic`): a blocked pair matches
  a hard rule — a single-token subset of a longer key, or a ≤1-edit first-token
  spelling wobble with all other tokens equal.
- **3b UNIQUENESS GUARD** (in `run_tier3_resolution`): a matched pair may
  auto-merge ONLY if it is *mutually unique* — each entity's sole 3b partner is
  the other. A bare given name (`misha`) that subset-matches ≥2 distinct full names
  (`misha petrov`/`misha ivanov`/…) is AMBIGUOUS; auto-merging into one
  arbitrarily is a cross-human misbind, so those pairs are NOT persisted (only
  counted per run) — a future 3c pass re-derives them. (The Tier-0 "unique hit"
  lesson applied to Tier 3.)
- **Auto-merge is `subset`-only.** Only a mutually-unique `subset` pair may execute
  — that shape (a bare given name uniquely subsetting one full name) is near-
  certain same-person. `near_exact` (≤1 edit) is **review-only by design**: it can
  conflate mutually-unique but genuinely-distinct common names (`anna`/`inna`), so
  it is ALWAYS written PROPOSED and never auto-merges, regardless of the flag.
- **3d Review surface**: mutually-unique deterministic pairs become
  ``EntityMergeCandidate`` rows (ambiguous pairs are counted, not persisted).
  With ``tier3_auto_merge_enabled`` OFF (default) they are **PROPOSED** — held for
  human review, never executed (the nightly ``_merge_entities`` sweep only runs
  PENDING). With it ON, subset pairs are **PENDING** — executed same night by that
  sweep, which re-checks the ``telegram_user_id`` guard in ``_execute_merge``.

3c (Haiku adjudication of the ambiguous, blocked-but-not-deterministic middle) is
a separate fast-follow — this module leaves those pairs untouched.

Safety invariants (design §5): never across ``entity_type`` (blocking is per
type); never two distinct non-NULL ``telegram_user_id`` (refused here AND by the
executor); precision-first (only the deterministic rule auto-qualifies); soft-
merge only (the executor sets ``merged_into_id``).
"""

from __future__ import annotations

import json
import logging
import uuid as _uuid
from collections import defaultdict
from typing import NamedTuple

from sqlalchemy import String, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.storage.models.entity_merge_candidate import EntityMergeCandidate
from kai_memory.storage.models.memory_entity import MemoryEntity
from kai_memory.store import cognitive_config
from kai_memory.store.cognitive_memory_constants import MergeCandidateStatus

logger = logging.getLogger("app.services.memory.entity_resolution_tier3")

_PREFIX_LEN = 4  # first-token prefix block length (3a)
# Cap on members in one blocking bucket. A bucket expands to k² pairs, so a very
# common token/prefix (many entities sharing a stopword-ish given name) could
# blow up the nightly sweep. Beyond this the bucket is SKIPPED (logged, not
# silent): such a bucket is a common name where every candidate would be
# uniqueness-guard-AMBIGUOUS anyway, so no auto-merge is lost.
_MAX_BUCKET_MEMBERS = 200
_CONFIDENCE = {"subset": 0.95, "near_exact": 0.9}  # stored on the candidate row


class BlockEntity(NamedTuple):
    """The minimal projection Tier 3 blocking + classification needs."""

    id: _uuid.UUID
    entity_type: str
    canonical_key: str
    telegram_user_id: int | None
    mention_count: int


def _tokens(canonical_key: str) -> list[str]:
    return [t for t in (canonical_key or "").split() if t]


def _digits(token: str) -> str:
    """The digit characters of ``token``, in order (``db-216`` → ``216``)."""
    return "".join(c for c in token if c.isdigit())


def levenshtein_at_most_1(a: str, b: str) -> bool:
    """True iff ``a`` and ``b`` are within one edit (sub/ins/del). Bounded and
    early-exiting — never computes a full matrix."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:  # single substitution allowed
        return sum(ca != cb for ca, cb in zip(a, b, strict=True)) == 1
    # lengths differ by exactly 1 → the shorter must embed in the longer with one
    # insertion. Walk both with a single allowed skip on the longer.
    short, lng = (a, b) if la < lb else (b, a)
    i = j = 0
    skipped = False
    while i < len(short) and j < len(lng):
        if short[i] == lng[j]:
            i += 1
            j += 1
        elif skipped:
            return False
        else:
            skipped = True
            j += 1  # skip one char in the longer string
    return True


def classify_deterministic(key_a: str, key_b: str) -> str | None:
    """The 3b hard rule. Returns ``"subset"`` | ``"near_exact"`` | ``None``.

    - **subset**: the shorter key is exactly ONE token and that token is a member
      of the longer key's token set (``misha`` ⊂ ``misha petrov``; also identical
      single-token keys like ``pavel``/``pavel``).
    - **near_exact**: equal token count, every token except the first is identical,
      and the first tokens are within one edit (``pavel petrov``/``pavell petrov``;
      also single-token ``danil``/``daniil``).
    """
    ta, tb = _tokens(key_a), _tokens(key_b)
    if not ta or not tb:
        return None
    short, lng = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if len(short) == 1 and short[0] in set(lng):
        return "subset"
    if len(ta) == len(tb) and ta[1:] == tb[1:] and levenshtein_at_most_1(ta[0], tb[0]):
        # Digit-guard: a ≤1-edit wobble on a DIGIT is a different ticket / model /
        # version (DB-216 vs DB-219, v1 vs v2), never a spelling variant. Only
        # letter-level wobbles (danil/daniil) are real near-exacts — catastrophic
        # to merge otherwise if auto-merge is ever enabled.
        if _digits(ta[0]) != _digits(tb[0]):
            return None
        return "near_exact"
    return None


def blocked_pairs(entities: list[BlockEntity], *, max_bucket: int = _MAX_BUCKET_MEMBERS) -> list[tuple[int, int]]:
    """3a blocking: indices of entity pairs worth classifying.

    A pair is blocked iff (same ``entity_type`` AND) their ``canonical_key``s share
    a token OR their first tokens share a ``_PREFIX_LEN``-char prefix. Uses two
    inverted indexes (token → members, first-prefix → members) so only entities
    that actually collide on a key are paired — no O(n²) all-pairs scan.

    A bucket bigger than ``max_bucket`` is SKIPPED (logged) to bound the k²
    pair expansion of a pathologically common token/prefix; nothing is lost that
    the uniqueness guard wouldn't have held as AMBIGUOUS anyway.
    """
    by_token: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_prefix: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, ent in enumerate(entities):
        toks = _tokens(ent.canonical_key)
        if not toks:
            continue
        for tok in set(toks):
            by_token[(ent.entity_type, tok)].append(idx)
        first = toks[0]
        if len(first) >= _PREFIX_LEN:
            by_prefix[(ent.entity_type, first[:_PREFIX_LEN])].append(idx)

    pairs: set[tuple[int, int]] = set()
    for buckets in (by_token, by_prefix):
        for key, members in buckets.items():
            if len(members) < 2:
                continue
            if len(members) > max_bucket:
                logger.warning(
                    "tier3.blocking.bucket_skipped key=%r size=%d (>%d) — not expanding k^2 pairs",
                    key,
                    len(members),
                    max_bucket,
                )
                continue
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    lo, hi = members[a], members[b]
                    pairs.add((lo, hi) if lo < hi else (hi, lo))
    return sorted(pairs)


def _distinct_telegram_users(a: BlockEntity, b: BlockEntity) -> bool:
    """True iff both carry a non-NULL telegram_user_id and they differ — two real
    chat participants, which must never merge (design §5)."""
    return (
        a.telegram_user_id is not None and b.telegram_user_id is not None and a.telegram_user_id != b.telegram_user_id
    )


async def _candidate_exists(session: AsyncSession, a_id: _uuid.UUID, b_id: _uuid.UUID) -> bool:
    """Any existing candidate for this unordered pair, ANY status — so we don't
    re-propose a duplicate and, critically, don't re-surface a human-REJECTED pair."""
    lo_new, hi_new = sorted((a_id, b_id), key=str)
    lo_col = func.least(EntityMergeCandidate.entity_a_id, EntityMergeCandidate.entity_b_id)
    hi_col = func.greatest(EntityMergeCandidate.entity_a_id, EntityMergeCandidate.entity_b_id)
    result = await session.execute(select(EntityMergeCandidate.id).where(lo_col == lo_new, hi_col == hi_new).limit(1))
    return result.scalar_one_or_none() is not None


async def _load_active_entities(session: AsyncSession) -> list[BlockEntity]:
    """Active, keyed entities only (never the merged tombstones or NULL-key junk)."""
    result = await session.execute(
        select(
            MemoryEntity.id,
            MemoryEntity.entity_type,
            MemoryEntity.canonical_key,
            MemoryEntity.telegram_user_id,
            MemoryEntity.mention_count,
        ).where(
            MemoryEntity.merged_into_id.is_(None),
            MemoryEntity.canonical_key.isnot(None),
            cast(MemoryEntity.canonical_key, String) != "",
        )
    )
    return [
        BlockEntity(
            id=row.id,
            entity_type=row.entity_type,
            canonical_key=row.canonical_key,
            telegram_user_id=row.telegram_user_id,
            mention_count=row.mention_count,
        )
        for row in result
    ]


async def run_tier3_resolution(session: AsyncSession) -> dict[str, int]:
    """L3 producer: block active entities, classify the deterministic pairs under a
    UNIQUENESS GUARD, and write merge candidates.

    A deterministic pair auto-qualifies for merge ONLY when it is *mutually
    unique* — each entity's sole 3b-qualifying same-type partner is the other. An
    entity that 3b-qualifies against ≥2 others (a bare given name like ``misha``
    matching four distinct full-name Mishas) is AMBIGUOUS: auto-merging it into one
    arbitrarily is a cross-human misbind. Ambiguous pairs are NOT persisted (they
    flooded the review queue and are recomputed nightly) — only counted in
    ``stats["ambiguous"]`` for the run summary; a future 3c pass re-derives them.

    Candidate status: PENDING (executed by the ``_merge_entities`` sweep that runs
    right after) only when the pair is unique AND ``tier3_auto_merge_enabled`` is
    on; otherwise PROPOSED. This producer never merges directly.
    """
    stats = {
        "blocked_pairs": 0,
        "deterministic": 0,
        "proposed": 0,
        "pending": 0,
        "ambiguous": 0,
        "near_exact_review": 0,
        "skipped_existing": 0,
        "skipped_distinct_tg": 0,
    }
    if not cognitive_config.get_bool("tier3_producer_enabled"):
        return stats

    auto_merge = cognitive_config.get_bool("tier3_auto_merge_enabled")
    entities = await _load_active_entities(session)
    pairs = blocked_pairs(entities)
    stats["blocked_pairs"] = len(pairs)

    # Pass 1 — collect deterministic, tg-safe pairs and count each entity's degree
    # in the 3b-qualifying graph (for the uniqueness guard).
    qualifying: list[tuple[int, int, str]] = []
    degree: dict[int, int] = defaultdict(int)
    for i, j in pairs:
        a, b = entities[i], entities[j]
        rule = classify_deterministic(a.canonical_key, b.canonical_key)
        if rule is None:
            continue  # blocked-but-not-deterministic → 3c (fast-follow), untouched here
        # Safety: distinct real users never merge (blocking already keeps types apart).
        if _distinct_telegram_users(a, b):
            stats["skipped_distinct_tg"] += 1
            continue
        qualifying.append((i, j, rule))
        degree[i] += 1
        degree[j] += 1
    stats["deterministic"] = len(qualifying)

    # Pass 2 — write a candidate per qualifying pair; the uniqueness guard decides
    # whether it may execute or is held for review.
    for i, j, rule in qualifying:
        a, b = entities[i], entities[j]
        unique = degree[i] == 1 and degree[j] == 1
        # Ambiguous (a bare given name subsetting ≥2 full names) is NOT persisted:
        # ~1k such pairs/run made the human-review queue unauditable, and they are
        # recomputed every night anyway (a future 3c Haiku pass can re-derive them
        # from the same blocking). Count for the run summary; never a row.
        if not unique:
            stats["ambiguous"] += 1
            continue
        if await _candidate_exists(session, a.id, b.id):
            stats["skipped_existing"] += 1
            continue
        # Auto-merge is reserved for the PROVABLY same-person tier: a
        # mutually-unique SUBSET (a bare given name that uniquely subsets exactly
        # one full name). near_exact (≤1 edit) is REVIEW-ONLY by design — it can
        # conflate genuinely distinct common names (anna/inna) that ARE mutually
        # unique, so the uniqueness guard can't protect it; it always waits for a
        # human (one review click for a rare spelling variant).
        if rule == "near_exact":
            status, reason, bucket = MergeCandidateStatus.PROPOSED, f"tier3:{rule}", "near_exact_review"
        elif auto_merge:  # subset + unique + auto-merge on
            status, reason, bucket = MergeCandidateStatus.PENDING, f"tier3:{rule}", "pending"
        else:  # subset + unique + auto-merge off (auto-eligible once the flag flips)
            status, reason, bucket = MergeCandidateStatus.PROPOSED, f"tier3:{rule}", "proposed"
        session.add(
            EntityMergeCandidate(
                entity_a_id=a.id,
                entity_b_id=b.id,
                confidence=_CONFIDENCE[rule],
                reason=reason,
                status=status,
            )
        )
        await session.flush()
        stats[bucket] += 1

    # Structured summary for observability + feature-watch (inline-serialized:
    # basicConfig drops ``extra=`` keys). "tier3.resolution.complete" is the marker.
    logger.info("tier3.resolution.complete %s", json.dumps({"auto_merge": auto_merge, **stats}))
    return stats


async def select_proposed_candidates(session: AsyncSession, *, limit: int = 200) -> list[dict]:
    """§3d review surface: the held PROPOSED queue with both entity names, for a
    human to audit before ``tier3_auto_merge_enabled`` is flipped on."""
    a_ent = MemoryEntity.__table__.alias("a_ent")
    b_ent = MemoryEntity.__table__.alias("b_ent")
    result = await session.execute(
        select(
            EntityMergeCandidate.id,
            EntityMergeCandidate.reason,
            EntityMergeCandidate.confidence,
            EntityMergeCandidate.created_at,
            a_ent.c.name.label("a_name"),
            b_ent.c.name.label("b_name"),
        )
        .join(a_ent, a_ent.c.id == EntityMergeCandidate.entity_a_id)
        .join(b_ent, b_ent.c.id == EntityMergeCandidate.entity_b_id)
        .where(EntityMergeCandidate.status == MergeCandidateStatus.PROPOSED)
        .order_by(EntityMergeCandidate.created_at.desc())
        .limit(limit)
    )
    return [
        {
            "id": str(row.id),
            "reason": row.reason,
            "confidence": row.confidence,
            "a_name": row.a_name,
            "b_name": row.b_name,
            "created_at": str(row.created_at),
        }
        for row in result
    ]
