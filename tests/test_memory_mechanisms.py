"""MechanismConfig contract + apply_mechanisms slot (Sub-design C INC 2, A3).

The re-ranker slot is a pure ``(ranked_candidates, MechanismConfig) -> ranked``
function behind one flag-dict. INC 2 ships it EMPTY: no mechanisms registered
and inert config => identity (byte-identical scorer output). Formulas for the
A/B/C/D re-rankers land in INC 4-7; this fixes the signature + slot only.
"""

from __future__ import annotations

import pytest

from kai_memory.retrieval.memory_mechanisms import (
    MechanismConfig,
    apply_mechanisms,
)


def _ranked(*labels):
    """Build a minimal ranked list: (trace_stub, score, components)."""
    return [(object(), float(len(labels) - i), {"label": lbl}) for i, lbl in enumerate(labels)]


def test_default_config_is_inert():
    assert MechanismConfig().is_inert() is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query_type": "episodic"},
        {"quality_gate_floor": 0.3},
        {"diversity_lambda": 0.2},
        {"source_confidence_weight": 0.5},
    ],
)
def test_any_set_flag_makes_config_non_inert(kwargs):
    assert MechanismConfig(**kwargs).is_inert() is False


def test_apply_mechanisms_identity_when_registry_empty():
    """INC 2 production path: no mechanisms registered => ranked returned unchanged."""
    ranked = _ranked("a", "b", "c")
    out = apply_mechanisms(ranked, MechanismConfig(), mechanisms=())
    assert out == ranked
    assert [c["label"] for _, _, c in out] == ["a", "b", "c"]


def test_apply_mechanisms_identity_when_config_inert_even_with_mechanisms():
    """A registered mechanism must NOT fire while its flag is inert."""

    def reverser(ranked, config):
        return list(reversed(ranked))

    ranked = _ranked("a", "b", "c")
    out = apply_mechanisms(ranked, MechanismConfig(), mechanisms=(reverser,))
    assert [c["label"] for _, _, c in out] == ["a", "b", "c"]


def test_apply_mechanisms_runs_registered_mechanism_when_active():
    """The slot genuinely re-ranks when config is non-inert (proves it isn't a no-op stub)."""

    def reverser(ranked, config):
        return list(reversed(ranked))

    ranked = _ranked("a", "b", "c")
    out = apply_mechanisms(ranked, MechanismConfig(query_type="episodic"), mechanisms=(reverser,))
    assert [c["label"] for _, _, c in out] == ["c", "b", "a"]


def test_apply_mechanisms_folds_multiple_mechanisms_in_order():
    def tag(name):
        def mech(ranked, config):
            return [(t, s, {**c, name: True}) for t, s, c in ranked]

        return mech

    ranked = _ranked("a")
    out = apply_mechanisms(ranked, MechanismConfig(query_type="x"), mechanisms=(tag("m1"), tag("m2")))
    assert out[0][2]["m1"] is True and out[0][2]["m2"] is True


def test_production_registry_is_empty_in_inc2():
    """INC 2 ships zero active re-rankers — the default registry is empty."""
    from kai_memory.retrieval.memory_mechanisms import _MECHANISMS

    assert _MECHANISMS == ()


def test_score_candidates_invokes_the_slot(monkeypatch):
    """_score_candidates actually routes through apply_mechanisms (wiring proof).

    Identity tests alone can't distinguish 'wired + inert' from 'not wired', so
    register a reverser + a non-inert config and confirm the scorer's output
    order flips — proving the slot is live inside _score_candidates.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from kai_memory.storage.models.base import utc_now
    from kai_memory.retrieval import memory_mechanisms
    from kai_memory.retrieval.memory_priming import MemoryPrimingService

    now = utc_now()

    def trace(tid):
        return SimpleNamespace(id=tid, created_at=now, access_count=0, emotion_arousal=0.0)

    candidates = {
        "a": (trace("a"), {"semantic": 0.9}),
        "b": (trace("b"), {"semantic": 0.1}),
    }
    weights = {"semantic": 1.0, "graph": 0.0, "recency": 0.0, "access": 0.0, "arousal": 0.0}
    svc = MemoryPrimingService(MagicMock())

    # Default (inert, empty registry) => identity: higher semantic ranks first.
    top = svc._score_candidates(candidates, weights, 2)
    assert [t.id for t, _, _ in top] == ["a", "b"]

    # Register a reverser + non-inert config => the slot fires inside the scorer.
    def reverser(ranked, config):
        return list(reversed(ranked))

    monkeypatch.setattr(memory_mechanisms, "_MECHANISMS", (reverser,))
    top2 = svc._score_candidates(candidates, weights, 2, mechanism_config=MechanismConfig(query_type="x"))
    assert [t.id for t, _, _ in top2] == ["b", "a"]
