"""What makes a retrieval-weight mix valid (memory's rule for its ``retrieval_config`` row).

Extracted from the weekly weight experiment when that harness moved to host-owned
``app.services.benchmarks`` (DB-843): the experiment proposes mixes and the operator's
``scripts/adopt_retrieval_weights.py`` writes one, and both must agree on this rule.
"""

from __future__ import annotations

import math

WEIGHT_KEYS: tuple[str, ...] = ("semantic", "graph", "recency", "access", "arousal", "salience")
WEIGHT_FLOOR = 0.02  # a nonzero component must carry at least this; 0.0 = component off
WEIGHT_CEILING = 0.60
_SUM_TOLERANCE = 0.01
_EPS = 1e-9


def validate_weights(weights: dict[str, float]) -> list[str]:
    """Validation shared by the proposer and the adoption script. [] = valid.

    Floor semantics: 0.0 is allowed (component off — e.g. salience today), but a
    nonzero weight below WEIGHT_FLOOR is rejected as meaningless noise-tuning.
    """
    errors: list[str] = []
    for key in WEIGHT_KEYS:
        if key not in weights:
            errors.append(f"missing key: {key}")
    for key in weights:
        if key not in WEIGHT_KEYS:
            errors.append(f"unknown key: {key}")
    if errors:
        return errors
    for key, value in weights.items():
        v = float(value)
        if not math.isfinite(v):
            errors.append(f"{key}={v} is not a finite number")
        elif v < 0.0 or v > WEIGHT_CEILING + _EPS:
            errors.append(f"{key}={v} outside [0.0, {WEIGHT_CEILING}] (ceiling)")
        elif 0.0 < v < WEIGHT_FLOOR - _EPS:
            errors.append(f"{key}={v} is nonzero below the {WEIGHT_FLOOR} floor (use 0.0 to disable)")
    total = sum(float(v) for v in weights.values())
    if abs(total - 1.0) > _SUM_TOLERANCE:
        errors.append(f"weights sum to {total:.4f}, expected 1.0 ±{_SUM_TOLERANCE}")
    return errors
