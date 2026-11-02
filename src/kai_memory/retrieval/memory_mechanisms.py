"""M3 re-ranker slot — the `MechanismConfig` contract (Sub-design C INC 2, A3).

The cognitive-memory scorer (`memory_priming._score_candidates`) splits into a
pure base-scoring stage and a re-ranker slot. Every M3 mechanism (A quality
gate, B diversity, C query-type routing, D source confidence) is a pure
``(ranked_candidates, MechanismConfig) -> ranked_candidates`` function plugged
into this one slot behind a single flag-dict.

INC 2 ships the **signature + slot only**, EMPTY: no mechanism is registered
(`_MECHANISMS == ()`) and the default config is inert, so `apply_mechanisms`
is the identity function and scorer output is byte-identical to pre-INC-2.
Formulas land in INC 4-7 (design §1.4 / §2). See
backend-docs/ai-workforce/plans/design/2026-06-04-subdesign-C-architecture-and-plan.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# One ranked candidate as produced by the base scorer: (trace, score, components).
# The trace is duck-typed (the live MemoryTrace or a replay stand-in), so it stays
# ``Any`` — this re-ranker contract needs no memory-model import (keeps the module
# outside the memory-service boundary per test_memory_interface_discipline).
RankedItem = tuple[Any, float, dict[str, Any]]


@dataclass(frozen=True)
class MechanismConfig:
    """Flag-dict gating the M3 re-rankers. Every field defaults to its inert
    value, so a default ``MechanismConfig()`` is a no-op (``is_inert()`` True).

    Fields fix the *contract* for the later increments; the formulas are
    deferred (R2). A mechanism only fires when its flag leaves the inert value:
      - ``query_type``                Mechanism C (INC 4): None => no routing.
      - ``quality_gate_floor``        Mechanism A (INC 5): 0.0 => no floor.
      - ``diversity_lambda``          Mechanism B (INC 7): 0.0 => no penalty.
      - ``source_confidence_weight``  Mechanism D (INC 6): 0.0 => no reweight.
    """

    query_type: str | None = None
    quality_gate_floor: float = 0.0
    diversity_lambda: float = 0.0
    source_confidence_weight: float = 0.0

    def is_inert(self) -> bool:
        """True when no mechanism flag is active => the slot is identity."""
        return (
            self.query_type is None
            and self.quality_gate_floor == 0.0
            and self.diversity_lambda == 0.0
            and self.source_confidence_weight == 0.0
        )


# A re-ranker: pure function over the ranked list. Must not mutate its input.
Mechanism = Callable[[list[RankedItem], MechanismConfig], list[RankedItem]]

# The active re-ranker registry. EMPTY in INC 2 — INC 4-7 append their
# mechanisms here (each behind its own MechanismConfig flag).
_MECHANISMS: tuple[Mechanism, ...] = ()


def apply_mechanisms(
    ranked: list[RankedItem],
    config: MechanismConfig,
    mechanisms: tuple[Mechanism, ...] | None = None,
) -> list[RankedItem]:
    """Fold the ranked candidates through the registered re-rankers.

    Pure and order-preserving by default: returns ``ranked`` unchanged when the
    registry is empty OR the config is inert (the INC 2 identity invariant).
    Each mechanism is applied left-to-right and must return a new list.

    ``mechanisms`` overrides the production registry (used by the measurement
    harness / self-tests to exercise the slot with a mechanism-sized re-ranker).
    """
    active = _MECHANISMS if mechanisms is None else mechanisms
    if not active or config.is_inert():
        return ranked
    result = ranked
    for mechanism in active:
        result = mechanism(result, config)
    return result
