"""In-process cache for cognitive_config values.

Kai's calls (no memory client) share the process cache ``_CACHE``. Under a
``MemoryClient`` the cache and its loaded flag are that client's own
(held on the client, ``ownership``): a client reads its own store's rows, and another
client loading different rows mid-run cannot change them (DB-854).
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.storage.models.cognitive_config import CognitiveConfig
from kai_memory.ownership import current_owner

# Pinned to the pre-DB-842 module path: CloudWatch queries, feature-watch criteria and
# log filters match the logger name, which must not follow the module's move.
_LEGACY_LOGGER_NAME = "app.services.cognitive_config"
logger = logging.getLogger(_LEGACY_LOGGER_NAME)

_DEFAULTS: dict[str, float] = {
    # decay
    "base_decay": 0.1,
    "spaced_bonus_cap": 0.2,
    "spaced_bonus_scale": 7.0,
    # pattern
    "cooccurrence_threshold": 3,
    "pair_limit": 10,
    # identity
    "similarity_threshold": 0.75,
    "sustained_reflection_min": 3,
    "sustained_days_min": 3,
    "evolution_window_days": 14,
    # pruning
    "silent_salience_threshold": 0.01,
    "silent_age_days": 30,
    "run_retention_days": 90,
    "timestamp_cap": 20,
    # Part C supersession grace (DB-315 / Codex #3): a durable trace demoted to
    # 'superseded' is spared from pruning for this window, so a WRONG newer fact that
    # superseded a correct older one can still be caught by contradiction-detection
    # before the old value is lost. Full re-promotion-on-invalidation is a follow-up.
    "durable_supersede_grace_days": 30,
    # retrieval
    "max_depth": 3,
    "min_confidence": 0.3,
    "relationship_min_confidence": 0.5,
    "graph_traversal_limit": 20,
    "temporal_relative_label_min_days": 1.0,
    # limits
    "decay_batch_size": 500,
    "entity_summary_limit": 10,
    "contradiction_batch": 30,
    # Memory v2-B Wave 1 (migration 064) — artifact lifecycle policy.
    "artifact_superseded_retention_days": 30.0,
    "artifact_stable_state_days": 14.0,
    "trajectory_event_min_count": 3.0,
    "identity_record_drift_window_days": 7.0,
    "identity_record_stable_days": 14.0,
    "cue_mapping_archive_days": 30.0,
    # Memory v2-B Wave 2 (migration 065) — producer caps and thresholds.
    "entity_schema_max_per_run": 50.0,
    "entity_schema_trace_window_limit": 20.0,
    "entity_schema_trace_window_days": 180.0,
    "trajectory_max_per_run": 20.0,
    "cue_min_cooccurrence": 5.0,
    "cue_adjudication_batch_max": 10.0,
    "cue_high_confidence_threshold": 0.8,
    "identity_record_min_magnitude": 0.3,
    # Sub-design D — persona pipeline (migrations 067 D Wave 1 + Wave 2).
    # Defaults mirror the migration's INSERT so unit tests that exercise
    # producers without ``load_config()`` (e.g. mocked DB sessions) still
    # see plausible numbers. Production reads the row values back into the
    # cache via ``load_config()``.
    "persona_max_active_traits": 12.0,
    "persona_max_active_patterns": 8.0,
    "persona_max_active_stances": 6.0,
    "persona_pattern_window_days": 7.0,
    "persona_pattern_promotion_threshold": 3.0,
    # Wave 3 adds Traits + Stances + Patterns under the Anchor — observed
    # ``compose()`` renders at ~12.4k chars on prod (smoke 2026-05-16). The
    # 6000 ceiling was set for the Anchor-only era and fires the "exceeded
    # budget_chars" warning on every render. 18000 covers Wave 3 prod smoke
    # size with headroom; revisit at M3 if per-layer caps tighten.
    "persona_compose_budget_chars": 18000.0,
    "persona_trait_decay_lambda": 0.0055,
    "persona_pattern_decay_lambda": 0.143,
    "persona_stance_decay_lambda": 0.0167,
    "persona_trait_strength_threshold": 0.2,
    "persona_trait_supersede_min_evidence": 5.0,
    "persona_stance_confidence_floor": 0.4,
    # How long an active Stance survives without Sonnet refreshing it.
    # 7 days mirrors persona_pattern_window_days (the artifact-lifespan axis).
    # When last_validated_at drifts past this, _expire_stale_stances flips
    # the row to status='superseded' so capacity opens for new stances.
    "persona_stance_retention_days": 7.0,
    # How long an active Trait survives without Sonnet re-reinforcing it.
    # Mirrors persona_stance_retention_days — same 7-day artifact-lifespan
    # axis. When last_reinforced_at drifts past this, _expire_stale_traits
    # flips the row to status='superseded' so capacity opens for new
    # traits. Strength-decay (persona_trait_strength_threshold +
    # persona_trait_decay_lambda) is the OTHER eviction path; the
    # retention gate kicks in faster when Sonnet stops surfacing the
    # dimension regardless of decayed strength.
    "persona_trait_retention_days": 7.0,
    # Sub-design D Wave 3 — Trajectory render-time event-filter.
    # Wave 2 producers built 1046-event / 145 KB trajectories on prod;
    # the composer caps per-trajectory event slots so rendering stays
    # inside the persona-block budget without losing chronology.
    #
    # ``persona_max_active_trajectories``: max trajectory artifacts
    #   surfaced per compose() call. Ordered by most-recent opened_at.
    # ``persona_trajectory_max_events``: per-trajectory total event
    #   slots across the recent + midterm tiers (older tier is always
    #   one collapsed line).
    # ``persona_trajectory_recent_window_days``: events within this
    #   window render verbatim (subject to the max_events cap).
    # ``persona_trajectory_older_window_days``: events beyond this
    #   window collapse into a single "N events between X and Y" line.
    "persona_max_active_trajectories": 5.0,
    "persona_trajectory_max_events": 40.0,
    "persona_trajectory_recent_window_days": 90.0,
    "persona_trajectory_older_window_days": 365.0,
    # Sub-design C INC 1 (migration 087) — master write-flag for the typed_metadata +
    # provenance substrate. **Code default flipped 0.0 → 1.0 (write-path fact-loss fix,
    # 2026-07-19).** This is the ACTIVATION LEVER for durable-fact retention: only when
    # the typed_metadata column is populated can the nightly decay/prune read the
    # persisted `durability` marker and floor durable facts above the prune line.
    # DECOUPLED from entityschema_live — this flag only POPULATES the column (+ scrubs
    # the blob); the EntitySchema card path stays independently gated. Prod may still
    # override via an SSM-set cognitive_config row (the user's activation call, not ours).
    "typed_metadata_write_enabled": 1.0,
    # INC 3 Step-1: when ON, the EntitySchema producer runs the deterministic
    # canonical_fields aggregator in SHADOW (computes + logs an agreement diff vs
    # Sonnet, but does NOT flip — Sonnet stays authoritative). Ships OFF (0.0);
    # seeded by migration 089 (mirrors this default). The flip is a later step.
    "entityschema_deterministic_enabled": 0.0,
    # INC 3 cutover (Sub-design C): when ON, the EntitySchema producer ships the NEW
    # curated card (aggregate_slot_profile → curate_profile) as the LIVE canonical_fields
    # for ALL entity types — still only when it's non-thin (a thin/empty/failed new card
    # falls back to Sonnet automatically, so it's "NEW where it's good, Sonnet where it's
    # thin"). Ships OFF (0.0); the prod flip is the user's deliberate, reversible step.
    # Not migration-seeded — `get()` falls back to this default until a row exists.
    # (The earlier person-only `entityschema_live_for_person` flag was retired once this
    # all-types flag became the live cutover — it was fully subsumed.)
    "entityschema_live": 0.0,
    # Phase 0 benchmark-gated weight-experiment loop (2026-07-02 design): the
    # weekly job is REPORT-ONLY (never writes retrieval_config), so it defaults
    # ON; the flag exists for instant off without a deploy. Not migration-seeded
    # — `get()` falls back to this default until a row exists.
    "weight_experiment_enabled": 1.0,
    # Tier-0 shadow probe (entity-resolution v2, task #8): a LOG-ONLY,
    # behavior-neutral instrument in ``find_or_create_entity``. On a text-mention
    # (tg_id None) miss, it measures what a Tier-0 cross-partition lookup WOULD
    # find (H1-H4) without changing the returned entity. Defaults ON (1.0) — it's
    # read-only + fail-open. Not migration-seeded (mirrors weight_experiment_enabled)
    # — ``get()`` falls back to this default until a row exists; toggle off without
    # a deploy via an INSERT/UPDATE on cognitive_config + ``load_config()``.
    "tier0_shadow_probe_enabled": 1.0,
    # Tier-3 nightly entity resolution (entity-resolution v2 §3). Two flags,
    # not migration-seeded (mirror weight_experiment_enabled — ``get()`` falls
    # back to these until a row exists; toggle without a deploy via cognitive_config).
    #   tier3_producer_enabled — generate blocked merge candidates each L3 run.
    #     Default ON (1.0): SAFE — it only WRITES candidate rows (PROPOSED, or
    #     PENDING when auto-merge is on) + a report; it never merges by itself.
    #   tier3_auto_merge_enabled — the risky promote step. Default OFF (0.0):
    #     3b-deterministic candidates are written PROPOSED (held) instead of
    #     PENDING, so nothing auto-merges. The operator audits the PROPOSED queue,
    #     then flips this ON to let 3b-deterministic pairs execute (via the
    #     existing _merge_entities PENDING sweep, which re-checks the tg_id guard).
    "tier3_producer_enabled": 1.0,
    "tier3_auto_merge_enabled": 0.0,
    # Procedural memory Phase A (procedural-and-proactive-memory-design §3).
    #   procedural_recorder_enabled — dual-write the execution log (Store A,
    #     cold) + a low-salience procedural CANDIDATE trace (Store B). Default ON
    #     (1.0): cheap + behavior-neutral (Store A is its own table; Store B's
    #     memory_type='procedural' is excluded from every L1/L2/L3 read).
    #     Not migration-seeded (mirrors weight_experiment_enabled).
    #   procedural_serving_enabled — REMOVED in DB-487 with the Phase-A
    #     exact-signature advisory it gated (zero call sites). Any existing ROW is
    #     deliberately left in place: a downgrade to 103 restores the reader but does
    #     not re-run 094's seed, so deleting it would swap an operator value for a
    #     baked-in default on rollback.
    "procedural_recorder_enabled": 1.0,
    # Procedural memory Phase C (2026-07-11-procedural-phase-c-design §1.4) — keys
    # seeded by migration 094 (stage kill-switches + serving mode). Everything else in
    # Phase C is a versioned module constant, not a config row. All ship 0.0/off →
    # Stage 1 is behavior-neutral; flips are the operator's job.
    #   procedural_serve_min_cosine — REMOVED in DB-487: serve-all has no cosine gate.
    #     Migration 104 deliberately LEAVES the row in place: a downgrade to 103 restores
    #     the reader but does not re-run 094's seed, so deleting it would swap an operator
    #     value for a baked-in default on rollback.
    "procedural_rule_extraction_enabled": 0.0,  # L2 Parts 3-5 + capture gate active
    "procedural_semantic_serving_mode": 0.0,  # 0=off, 1=shadow (log-only), 2=live
    "procedural_outcome_enabled": 0.0,  # outcome-inference passes
    "procedural_reinforcement_enabled": 0.0,  # nightly ledger pass
    "procedural_habit_promotion_enabled": 0.0,  # rung-2 L3 step
    "procedural_trait_feed_enabled": 0.0,  # rung-3 evidence stream + compose pickup
    # L1.5 iterative retrieval (2026-07-21-l15-iterative-retrieval-design; DB-327/DB-329).
    # Two 3-state flags (0=off / 1=shadow / 2=live), seeded by migration 099. Both ship
    # 1.0=SHADOW → no live behavior change at merge: Stage A computes a windowed candidate
    # diff + logs it (injects nothing); Stage C runs the full trigger/retrieve/dedup and
    # logs would-have-injected (injects nothing); C2 logs would-have-revised (never calls
    # the model). Every other L1.5 knob is a versioned module constant (iterative_retrieval.py),
    # never a config row — the #241 containment boundary. The live flip is the operator's job.
    "priming_context_cues": 1.0,  # Stage A — continuous window cues
    "priming_thinking_cues": 1.0,  # Stage C + C2 — thinking-cued re-priming / recall-and-revise
}

_CACHE: dict[str, float] = dict(_DEFAULTS)
_LOADED = False


class _OwnedConfig:
    """A memory client's copy of the cache and its loaded flag."""

    def __init__(self) -> None:
        self.values: dict[str, float] = dict(_DEFAULTS)
        self.loaded = False


def _owned() -> _OwnedConfig | None:
    owner = current_owner()
    if owner is None:
        return None
    config = owner.caches.get("cognitive_config")
    if config is None:
        config = owner.caches["cognitive_config"] = _OwnedConfig()
    return config  # type: ignore[no-any-return]


def _values() -> dict[str, float]:
    owned = _owned()
    return _CACHE if owned is None else owned.values


async def load_config(session: AsyncSession) -> None:
    """Prime the cache from the cognitive_config table."""
    global _LOADED
    owned = _owned()
    values = _CACHE if owned is None else owned.values
    try:
        result = await session.execute(select(CognitiveConfig))
        for row in result.scalars().all():
            values[row.key] = row.value
        if owned is None:
            _LOADED = True
        else:
            owned.loaded = True
    except Exception:
        logger.warning("Failed to load cognitive_config, using defaults", exc_info=True)


def get(key: str) -> float:
    """Return a config value, falling back to baked-in defaults."""
    return _values().get(key, _DEFAULTS.get(key, 0.0))


def get_int(key: str) -> int:
    return int(get(key))


def get_bool(key: str) -> bool:
    """Interpret a 0/1-style config value as a boolean enable-flag.

    Sub-design C ships several enable flags (typed_metadata_write_enabled,
    later query_type_routing_enabled, …) as 0.0/1.0 floats in this cache.
    """
    return get(key) > 0.5


def is_loaded() -> bool:
    owned = _owned()
    return _LOADED if owned is None else owned.loaded


def reset_cache() -> None:
    """Test-only — restore defaults."""
    global _LOADED
    _CACHE.clear()
    _CACHE.update(_DEFAULTS)
    _LOADED = False


def set_for_testing(key: str, value: float) -> None:
    """Test-only setter — overrides one cache entry without touching the table.

    Tests that need to flip a tunable knob mid-run should call this rather
    than reaching into ``_CACHE`` directly. ``reset_cache()`` undoes any
    overrides applied via this helper.
    """
    _CACHE[key] = value
