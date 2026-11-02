"""INC 3 A3 — subject attribution.

A fact's raw ``subject`` (an entity name emitted by Haiku, or re-derived during
backfill) is resolved to a canonical entity *id* by matching it against the
trace's candidate entities using #201 cross-script canonicalization
(``same_canonical_entity``). Persisting the resolved id — not the raw string —
keeps ``Миша`` / ``Misha`` from smearing one person's facts across two entities.

Pure + dependency-light so it is unit-testable in isolation and shared verbatim
by the live L2 extraction path and the one-time backfill tool.
"""

from __future__ import annotations

import uuid as _uuid
from collections.abc import Iterable, Sequence
from typing import Any

from kai_memory.canonicalization import same_canonical_entity

# A candidate entity: (id, name, entity_type). entity_type is accepted for
# symmetry / future type-aware matching but resolution is name-canonical (a
# subject string carries no reliable type).
Candidate = tuple[_uuid.UUID, str, str]


def resolve_subject_id(subject: str | None, candidates: Sequence[Candidate]) -> _uuid.UUID | None:
    """Return the canonical entity id a fact's ``subject`` denotes, or None.

    Matches ``subject`` against each candidate's name via ``same_canonical_entity``
    (cross-script translit key). Returns the first canonical match; an empty /
    garbage subject, or no match, yields None (the fact stays unattributed — the
    aggregator's single-link/Sonnet fallback handles it). Precision-first: never
    mints an entity, never guesses past a key match.
    """
    if not subject or not subject.strip():
        return None
    # Collect DISTINCT matching entity ids (aliases of one entity collapse). A
    # single match resolves; zero or an ambiguous >1 (two different entities
    # sharing a canonical key) yields None — precision-first, order-independent.
    matched = {entity_id for entity_id, name, _etype in candidates if same_canonical_entity(subject, name)}
    if len(matched) == 1:
        return next(iter(matched))
    return None


def candidates_from_entities(entities: Iterable[Any]) -> list[Candidate]:
    """Expand resolved entity objects into match candidates (name + each alias).

    Aliases are included as separate candidates so a cross-script subject matches
    even when the canonical row's primary name is in the other script.
    """
    out: list[Candidate] = []
    for e in entities or []:
        eid = getattr(e, "id", None)
        name = getattr(e, "name", None)
        etype = getattr(e, "entity_type", None) or "person"
        if eid is None or not name:
            continue
        out.append((eid, name, etype))
        for alias in getattr(e, "aliases", None) or []:
            if alias:
                out.append((eid, alias, etype))
    return out


def attribute_fact_subjects(facts: Iterable[Any], entities: Iterable[Any]) -> None:
    """In place: stamp each fact's ``subject_id`` (canonical entity id) from its
    raw ``subject`` name, matched against ``entities``. Leaves the raw ``subject``
    intact; sets ``subject_id`` to None when absent/unresolved.
    """
    candidates = candidates_from_entities(entities)
    for fact in facts or []:
        if not isinstance(fact, dict) or not fact.get("subject"):
            continue
        sid = resolve_subject_id(fact.get("subject"), candidates)
        fact["subject_id"] = str(sid) if sid is not None else None
