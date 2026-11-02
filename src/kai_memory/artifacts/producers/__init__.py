"""Concrete artifact producers for M2 Wave 2 + Sub-design D Wave 2.

Each producer subclasses ``BaseProducer`` and implements
``discover_subjects`` + ``produce``. ``supersede`` is inherited from
``BaseProducer`` (the §3.2 5-step content-hash-dedup transaction) unless a
producer needs custom transition behaviour (PatternProducer overrides for
content_hash dedup; StanceProducer overrides for append-in-place semantics).

Order of construction (priority per design §6 Wave 2):
1. EntitySchemaProducer  — replaces entity.summary; no Sonnet
2. TrajectoryProducer    — narrative grouping; one Sonnet/subject
3. IdentityRecordProducer — agent-self drift detection; one Sonnet/run
4. CueMappingProducer    — phrase→schema mapping; up to 10 Sonnet/run
5. ContrastiveMapProducer — registered no-op until M3 confusion sets
6. PatternProducer       — D Wave 2; recent observations, fast decay
7. TraitProducer         — D Wave 2; reads patterns + traces, slow decay
8. StanceProducer        — D Wave 2; agent positions, append-in-place

Order matters: TraitProducer reads Patterns produced earlier in the same
run (Pattern → Trait promotion is also handled inline by
PatternProducer for the threshold-crossing case; TraitProducer sweeps
the longer-term drift signal).
"""

from __future__ import annotations

from kai_memory.artifacts.producers.contrastive_map import (
    ContrastiveMapProducer,
)
from kai_memory.artifacts.producers.cue_mapping import CueMappingProducer
from kai_memory.artifacts.producers.entity_schema import EntitySchemaProducer
from kai_memory.artifacts.producers.identity_record import (
    IdentityRecordProducer,
)
from kai_memory.artifacts.producers.pattern_producer import PatternProducer
from kai_memory.artifacts.producers.stance_producer import StanceProducer
from kai_memory.artifacts.producers.trait_producer import TraitProducer
from kai_memory.artifacts.producers.trajectory import TrajectoryProducer

__all__ = [
    "ContrastiveMapProducer",
    "CueMappingProducer",
    "EntitySchemaProducer",
    "IdentityRecordProducer",
    "PatternProducer",
    "StanceProducer",
    "TraitProducer",
    "TrajectoryProducer",
]
