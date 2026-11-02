"""Structural guards on ``consolidation._run_artifact_pipeline`` registrations.

The L3 pipeline registers producers in a specific order (priority + dependency
chain). Tests in this module read the source of ``_run_artifact_pipeline`` and
assert which producers are / are NOT registered, so accidental re-adds or
silent reorderings show up as test failures rather than as a quiet shift in
nightly behaviour.

ContrastiveMapProducer is intentionally absent: ``discover_subjects`` is a
hardcoded no-op until M3 ships ConfusionSetProducer. Re-registering it makes
producer-health dashboards (3x/night "0 produced") noisy without adding signal.
"""

from __future__ import annotations

import ast
import inspect

from kai_memory.consolidation import consolidation


def _registered_producers() -> list[str]:
    source = inspect.getsource(consolidation._run_artifact_pipeline)
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "register":
            continue
        if not node.args:
            continue
        arg = node.args[0]
        # We expect ``pipeline.register(SomeProducer())`` — pull out the class
        # name from the inner Call node.
        if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
            names.append(arg.func.id)
    return names


def test_contrastive_map_not_registered() -> None:
    """ContrastiveMap stays off the L3 pipeline until M3."""
    assert "ContrastiveMapProducer" not in _registered_producers()


def test_expected_producers_registered_in_priority_order() -> None:
    """Lock the registration order so accidental reshuffles surface.

    EntitySchema first (CueMapping reads its outputs). Pattern before Trait
    so TraitProducer sees same-run Patterns. Stance last (independent).
    """
    expected = [
        "EntitySchemaProducer",
        "TrajectoryProducer",
        "IdentityRecordProducer",
        "CueMappingProducer",
        "PatternProducer",
        "TraitProducer",
        "StanceProducer",
    ]
    assert _registered_producers() == expected
