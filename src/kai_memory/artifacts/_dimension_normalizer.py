"""Shared dimension vocabulary for Pattern + IdentityRecord producers.

Both producers emit dimension labels from Sonnet (snake_case strings like
"context_discipline", "honesty_under_pressure"). Without a canonicalization
step, synonyms accumulate: prod 2026-05-15 had 79 active dimensions, ~half
of which were duplicates of the same concept under different names (e.g. 8
variants of "context discipline", 5 variants of "honesty under pressure").
This file holds the canonical vocabulary + alias map + normalize() helper
that both producers call to snap synonyms to a single label.

**Critical design constraint:** the whitelist is SOFT, not a hard cap. The
prompt instructs Sonnet to "prefer these canonical names but emit a new
snake_case name if none fits". When ``normalize()`` sees a dimension that
isn't in either set, it passes through unchanged — Sonnet retains the
power to coin new dimensions as the system grows. Only known synonyms
collapse.

Census source: ``backend-docs/ai-workforce/research/2026-05-15-dimension-census.md`` (Pattern +
IdentityRecord active rows on prod 2026-05-15). The map encodes the
deliberate grouping decisions from that census.
"""

from __future__ import annotations

# Seed canonical vocabulary — chosen from the most-active variant in each
# census cluster. New canonical names can be added by raising a row's
# observed frequency above the variant currently in CANONICAL_DIMENSIONS;
# no code change required to ADD a dimension Sonnet starts emitting (it
# passes through unchanged), only when we decide to alias new variants.
CANONICAL_DIMENSIONS: frozenset[str] = frozenset(
    {
        # Persona pipeline core (synonym-prone)
        "context_discipline",
        "honesty_under_pressure",
        "proactive_synthesis",
        "register_calibration",
        "effort_past_capability_wall",
        "compression_under_constraint",
        "overconfident_point_estimation",
        "assertion_before_verification",
        "grammatical_gender_accuracy",
        "urgency_calibration",
        "values_model_accuracy",
        "epistemic_calibration",
        # Singletons — already canonical, no aliases known yet but listed
        # here so the prompt can suggest them (vocabulary hint).
        "cross_domain_synthesis",
        "emotional_attunement",
        "technical_depth",
        "uncertainty_flagging",
        "hyper_specificity",
        "scope_drift",
        "self_correction_on_error",
        "self_awareness",
        "sycophancy_resistance",
        "continuity_tracking_gap",
        "persistence_gap",
        "operational_hygiene",
        "deceptive_behavior",
    }
)


# Alias map — synonyms → canonical name. Sourced from the 2026-05-15
# dimension census. Adding entries here is safe; the normalize() helper
# falls back to passthrough for unknown dimensions, so a missing alias
# costs at most one nightly run before another synonym surfaces.
ALIAS_MAP: dict[str, str] = {
    # Context discipline cluster
    "context_boundary_enforcement": "context_discipline",
    "context_isolation": "context_discipline",
    "context_isolation_discipline": "context_discipline",
    "context_separation_discipline": "context_discipline",
    "context_bleed_control": "context_discipline",
    "context_bleed_failure": "context_discipline",
    "context_bleeding": "context_discipline",
    "context_bleed_prevention": "context_discipline",
    # Honesty cluster
    "honesty_under_discomfort": "honesty_under_pressure",
    "honesty_under_social_pressure": "honesty_under_pressure",
    "epistemic_honesty": "honesty_under_pressure",
    "direct_error_acknowledgment": "honesty_under_pressure",
    "error_acknowledgment": "honesty_under_pressure",
    "error_acknowledgment_directness": "honesty_under_pressure",
    # Proactive / structured synthesis cluster
    "proactive_synthesis_and_follow_through": "proactive_synthesis",
    "proactive_synthesis_and_urgency_flagging": "proactive_synthesis",
    "structured_synthesis": "proactive_synthesis",
    "structured_complexity_synthesis": "proactive_synthesis",
    "structured_comparative_reasoning": "proactive_synthesis",
    "structured_multi_criteria_synthesis": "proactive_synthesis",
    "structured_prioritization": "proactive_synthesis",
    # Register cluster
    "register_adaptation": "register_calibration",
    "register_differentiation": "register_calibration",
    "per_user_register_calibration": "register_calibration",
    "per_person_register_calibration": "register_calibration",
    # Capability-boundary cluster
    "capability_boundary_stopping": "effort_past_capability_wall",
    "effort_cutoff_at_capability_wall": "effort_past_capability_wall",
    "effort_extension_past_capability_wall": "effort_past_capability_wall",
    "effort_extension_past_capability_boundaries": "effort_past_capability_wall",
    "proactive_effort_extension": "effort_past_capability_wall",
    # Compression cluster
    "compression_failure": "compression_under_constraint",
    "compression_failure_under_pressure": "compression_under_constraint",
    "argument_compression_failure": "compression_under_constraint",
    "constraint_adaptive_specificity": "compression_under_constraint",
    "constraint_adaptive_pivoting": "compression_under_constraint",
    # Overconfident-estimation cluster
    "overconfident_point_estimates": "overconfident_point_estimation",
    "overconfident_numeric_estimates": "overconfident_point_estimation",
    # Verification / assertion cluster
    "assertion_without_verification": "assertion_before_verification",
    # Grammar / gender cluster
    "grammatical_gender_error": "grammatical_gender_accuracy",
    # Values-model cluster
    "values_model_accuracy_for_third_parties": "values_model_accuracy",
    "values_model_accuracy_for_non_interlocutors": "values_model_accuracy",
    "non_interlocutor_values_modeling": "values_model_accuracy",
    # Epistemic cluster
    "epistemic_humility": "epistemic_calibration",
    "technical_confidence_calibration": "epistemic_calibration",
    # NOTE: "uncertainty_flagging" intentionally left distinct from
    # epistemic_calibration — different surface behavior.
}


# Suffixes the normalizer strips before alias-map lookup. Sonnet often
# decorates dimensions with context tails ("_under_pressure",
# "_failure_mode", "_in_practice") that mean roughly the same thing as
# the bare form. Stripping these collapses the long tail of variants
# without growing ALIAS_MAP for every combination.
_TRAILING_SUFFIXES: tuple[str, ...] = (
    "_under_pressure",
    "_under_discomfort",
    "_failure_mode",
    "_failure",
    "_gap",
    "_in_practice",
    "_when_warranted",
    "_when_warranted_failure",
)


def _strip_suffixes(dim: str) -> str:
    """Strip the longest-matching known suffix once. Idempotent.

    Only the literal suffixes in ``_TRAILING_SUFFIXES`` are stripped — not
    a general regex — so the normalization stays conservative and
    inspectable. If multiple suffixes match, the longest wins (longest-prefix
    rule by string length so "_under_pressure" beats "_pressure").
    """
    for suffix in sorted(_TRAILING_SUFFIXES, key=len, reverse=True):
        if dim.endswith(suffix) and dim != suffix:
            return dim[: -len(suffix)]
    return dim


def normalize(dim: str) -> str:
    """Map a Sonnet-emitted dimension to its canonical name.

    Rules:

    1. Lowercase the input (Sonnet occasionally capitalizes).
    2. If already in ``CANONICAL_DIMENSIONS``, return unchanged.
    3. If in ``ALIAS_MAP``, return the mapped canonical.
    4. Strip a known trailing suffix once; re-check ALIAS_MAP and
       CANONICAL_DIMENSIONS.
    5. **Passthrough** the input (lowercased only) — Sonnet is allowed to
       coin new dimensions, and we'd rather see them than swallow them.

    Empty input returns empty string (caller decides whether to skip).
    """
    if not dim:
        return ""
    lowered = dim.strip().lower()
    if not lowered:
        return ""
    if lowered in CANONICAL_DIMENSIONS:
        return lowered
    if lowered in ALIAS_MAP:
        return ALIAS_MAP[lowered]
    stripped = _strip_suffixes(lowered)
    if stripped != lowered:
        if stripped in CANONICAL_DIMENSIONS:
            return stripped
        if stripped in ALIAS_MAP:
            return ALIAS_MAP[stripped]
    return lowered


def canonical_list_for_prompt() -> str:
    """Render canonical names as a sorted, comma-joined string for the
    Sonnet prompt's vocabulary-hint section.
    """
    return ", ".join(sorted(CANONICAL_DIMENSIONS))


__all__ = [
    "ALIAS_MAP",
    "CANONICAL_DIMENSIONS",
    "canonical_list_for_prompt",
    "normalize",
]
