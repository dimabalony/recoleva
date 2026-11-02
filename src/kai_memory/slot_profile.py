"""INC 3 — durable-attribute aggregator (pure).

Turns the slot-tagged facts (``slot`` + ``durability`` from the schema-constrained
write-path, #217, and the backfill, #218) into a clean profile card.

**The inversion (§5):** schema is a hint, not a gate. Typed slots route the facts
they understand; every *durable* off-schema fact is RETAINED — keyed by its own
``field`` — so fieldless-type entities (topic/place/org/product) get structured,
universal cards instead of empty ones. There is no recurrence gate: a single
mention survives.

**Phase A (§13) — ranking IS the memory physics.** Facts are ranked by their
parent trace's **salience**, which already bakes in significance-gated Ebbinghaus
decay + access reinforcement (run nightly by ``consolidation._apply_decay``):
durable, high-significance facts stay high-salience and persist; transient,
low-significance ones fade and drop at the cap. Junk is *forgotten by decay*, not
suppressed by a hand-maintained blocklist — the inc2 durability rubric keeps
agent-operational facts out of the durable set up front, and decay handles any
leakage. (A fact inherits its parent trace's salience; a value seen across several
traces takes the MAX — the most-salient mention.)

Per entity:
  * **Typed slots** — on-schema facts (attributed via ``subject_id``, valid for the
    type): SINGLE → newest-wins; MULTI → dedup, rank by (salience, recency), cap;
    a ``labeled`` MULTI slot renders "field: value" while bare MULTI stays plain.
  * **Off-schema durables** — retained field-keyed (one card field per ``field``),
    cross-field value-deduped.
  * value-level ``is_not`` tombstones a specific value (normalized).
  * fold active ``EntityRelationship`` rows into slots via ``PREDICATE_SLOT_MAP``.
  * rank ``importance → recency → salience`` (NEVER alphabetical), one total-card
    cap (``max_slots``); off-schema fields fall back to importance 0 (after typed).

DECOUPLED STORAGE (§3.5): returns BOTH the render-facing flat ``canonical_fields``
(``dict[str, str]``, derived) and the structured ``canonical_slot_values``
(per value: ``value / field / observed_at / source_trace_id / durability / slot``).
Pure
(no DB / LLM) → unit-testable in isolation.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from kai_memory.canonicalization import translit_key
from kai_memory.durable_schema import Cardinality, slots_for

# Static, audited predicate→slot fold (design §3.4). Targets durable_schema slot
# keys; a pair whose slot is not valid for the aggregated entity's type is skipped.
# (``owner_of`` removed — §12 P2-D: there is no ``owns`` slot, so it was a no-op.)
PREDICATE_SLOT_MAP: dict[str, str] = {
    "works_at": "employer",
    "located_in": "current_location",
}

# Cross-field value-dedup threshold (§12 P2-A): two retained off-schema fields whose
# VALUE token-sets overlap at least this much collapse (car vs vehicle_ownership).
_VALUE_DEDUP_OVERLAP = 0.8

# Value-level ``is_not`` tombstone match threshold: a negation erases a stored value
# when their token sets share at least one token AND overlap (Jaccard) >= this. Lets
# "I don't have a Li Auto car" {li,auto,car} erase a stored "Li Auto L6" {li,auto,l6}
# (2/4 = 0.5) — they diverge only at the suffix — while staying tight enough that
# "black coffee" vs "black tea" (1/3) and "Li Auto" vs "Li Ning" (1/3) do NOT match.
_TOMBSTONE_MIN_TOKEN_OVERLAP = 0.4

# Fallback importance for synthetic / off-schema field keys not in the typed schema —
# they rank AFTER the typed slots but are still retained (§5 universal coverage).
_OFF_SCHEMA_IMPORTANCE = 0

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class SlotProfile:
    """Aggregator output — flat render-facing fields + the decoupled structured store."""

    canonical_fields: dict[str, str]
    canonical_slot_values: dict[str, list[dict[str, Any]]]
    # §12 P2-C telemetry: off-schema durable candidates entering the gate, and the
    # number retained onto the card (so "gate too tight" is distinguishable from
    # "no input"). Default 0 so callers (e.g. the summarizer) can rebuild without them.
    off_schema_durable_count: int = field(default=0)
    off_schema_retained_count: int = field(default=0)


def _normalized_equal(a: str, b: str) -> bool:
    ka = translit_key(a or "")
    return bool(ka) and ka == translit_key(b or "")


def _tokens(s: str) -> frozenset[str]:
    """Significant tokens of a single string: split on whitespace / underscore / slash,
    #201-normalize each part (``translit_key``), drop empties.

    Slash is a separator too: ``translit_key`` strips ``/`` *without* a break, so
    "Minsk/Klimovichi" would otherwise collapse to one opaque token and defeat
    token-overlap matching; splitting on it first keeps {minsk, klimovichi}.
    """
    return frozenset(t for t in (translit_key(p) for p in re.split(r"[\s_/]+", str(s))) if t)


def _tombstone_matches(value: str, tombstone: str) -> bool:
    """Does a value-level ``is_not`` tombstone erase this stored ``value``?

    Exact #201-normalized equality is the fast path. Otherwise loosen to token-set
    overlap so a correction that diverges only at the suffix still lands — "I don't
    have a Li Auto car" must erase a stored "Li Auto L6". Requires BOTH a shared token
    AND Jaccard overlap >= ``_TOMBSTONE_MIN_TOKEN_OVERLAP``, so a single common word
    ("li" in "Li Auto" vs "Li Ning", "black" in "black coffee" vs "black tea") can't
    over-match and wrongly erase a distinct fact.
    """
    if _normalized_equal(value, tombstone):
        return True
    vt, tt = _tokens(value), _tokens(tombstone)
    shared = vt & tt
    if not shared:
        return False
    return len(shared) / len(vt | tt) >= _TOMBSTONE_MIN_TOKEN_OVERLAP


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


# Connector tokens dropped from the near-duplicate key so trivial wording variants
# collapse ("asymptomatic for 2 years" ≈ "asymptomatic past 2 years").
_DEDUP_CONNECTORS = frozenset({"for", "past", "the", "of", "a", "an", "in", "on", "at", "to", "and"})


def _dedup_key(field: Any, value: Any) -> tuple[str, ...]:
    """Near-duplicate key: the sorted set of significant (#201-normalized) tokens of
    field+value, with connector words dropped. Subsumes exact-normalized dedup and
    collapses light near-dupes; only an identical significant-token-set merges, so
    distinct values ('170 cm' vs '175 cm', 'high' vs 'low …') stay separate."""
    raw = f"{field or ''} {value or ''}"
    toks = [t for t in (translit_key(p) for p in re.split(r"[\s_]+", raw) if p.strip()) if t]
    sig = [t for t in toks if t not in _DEDUP_CONNECTORS]
    return tuple(sorted(set(sig or toks)))


def _is_self_contained(value: str) -> bool:
    """A value that reads fine without its field label: a multi-word, all-alphabetic
    noun phrase (e.g. 'autoimmune thyroiditis'). '170 cm', '75B', 'confirmed' are not."""
    toks = value.split()
    return len(toks) >= 2 and all(t.replace("-", "").isalpha() for t in toks)


def _label(field: Any, value: str) -> str:
    """Render a labeled MULTI-slot value with its field as context ('height: 170 cm'),
    skipping the prefix when redundant: empty field, value already labeled (has ':'),
    a self-contained noun phrase, or the field already inside the value."""
    field = (str(field) if field is not None else "").strip()
    value = value.strip()
    if not field or ":" in value or _is_self_contained(value):
        return value
    fk = translit_key(field)
    if fk and fk in translit_key(value):
        return value
    return f"{field}: {value}"


def _record(it: dict[str, Any], slot: str) -> dict[str, Any]:
    return {
        "value": str(it["value"]),
        "field": it.get("field"),
        "observed_at": _iso(it["_dt"]),
        "source_trace_id": it["source_trace_id"],
        "durability": it["durability"],
        "slot": slot,
    }


def _reduce_multi(surviving: list[dict[str, Any]], slot: str, cap: int) -> list[dict[str, Any]]:
    """Collapse near-duplicates, then rank by (salience DESC, recency DESC) and cap.

    ``surviving`` is newest-first, so a dedup group's first-seen item is its newest
    (the representative). Each group inherits the MAX salience across its mentions —
    the most-salient mention wins (Phase A §13). Salience already bakes in
    significance-gated decay, so a durable high-significance value outranks a faded
    transient one, which falls below the cap.
    """
    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    for it in surviving:
        key = _dedup_key(it.get("field"), it["value"])
        group = groups.get(key)
        if group is None:
            groups[key] = {"rep": it, "salience": it["salience"]}
        else:
            group["salience"] = max(group["salience"], it["salience"])
    ordered = sorted(groups.values(), key=lambda g: (g["salience"], g["rep"]["_dt"]), reverse=True)
    return [_record(g["rep"], slot) for g in ordered[:cap]]


def _reduce_one_slot(
    surviving: list[dict[str, Any]], slot: str, cardinality: Cardinality, cap: int
) -> list[dict[str, Any]]:
    """Reduce one slot's tombstone-survivors: SINGLE → newest-wins; MULTI → dedup +
    salience/recency rank + cap."""
    if cardinality == Cardinality.SINGLE:
        return [_record(surviving[0], slot)]
    return _reduce_multi(surviving, slot, cap)


def _survivors(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Newest-first, drop blank values, apply value-level ``is_not`` tombstoning.

    Tombstoning uses token-overlap matching (``_tombstone_matches``) so a correction
    that diverges only at the suffix ("I don't have a Li Auto car") still erases the
    stored value ("Li Auto L6"), not just an exact-normalized restatement.
    """
    items.sort(key=lambda x: x["_dt"], reverse=True)
    tombstones = [str(it["value"] or "") for it in items if it.get("polarity") == "is_not"]
    is_items = [it for it in items if it.get("polarity") != "is_not" and str(it["value"] or "").strip()]
    return [it for it in is_items if not any(_tombstone_matches(str(it["value"]), t) for t in tombstones)]


def _as_salience(value: Any) -> float:
    """Coerce a raw per-trace salience into a float, defaulting to the column default
    (0.5) when missing/non-numeric — so ranking never KeyErrors on a thin row."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.5


def _max_salience(items: list[dict[str, Any]]) -> float:
    """Highest parent-trace salience among a slot's items (Phase A §13 rank key)."""
    return max((it["salience"] for it in items), default=0.0)


def _field_max_salience(items: list[dict[str, Any]]) -> dict[str, float]:
    """Per off-schema ``field`` → max parent-trace salience across its mentions."""
    out: dict[str, float] = {}
    for it in items:
        fkey = str(it.get("field") or "").strip()
        if fkey:
            out[fkey] = max(out.get(fkey, 0.0), it["salience"])
    return out


def _max_recency(records: list[dict[str, Any]]) -> str:
    """Newest ``observed_at`` (ISO string) among records; '' when none — recency tiebreak."""
    return max((r["observed_at"] or "" for r in records), default="")


def _retain_off_schema(items: list[dict[str, Any]], cap: int) -> dict[str, list[dict[str, Any]]]:
    """§6.1 / §12 P2-B — field-keyed retention of off-schema durables (NO recurrence
    gate). Group by ``field``; reduce each field MULTI-style (single-valued fields
    collapse to one salience-winning value, multi-valued accumulate distinct). Returns
    ``{field_name: records}`` — a queryable, Sonnet-shaped substrate."""
    by_field: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        fkey = str(it.get("field") or "").strip()
        if fkey:
            by_field[fkey].append(it)
    out: dict[str, list[dict[str, Any]]] = {}
    for fkey, group in by_field.items():
        surviving = _survivors(group)
        if surviving:
            out[fkey] = _reduce_multi(surviving, fkey, cap)
    return out


def _value_token_set(records: list[dict[str, Any]]) -> frozenset[str]:
    toks: set[str] = set()
    for r in records:
        toks |= _tokens(str(r["value"]))
    return frozenset(toks)


def _dedup_off_schema_by_value(
    field_records: dict[str, list[dict[str, Any]]], order: list[str]
) -> dict[str, list[dict[str, Any]]]:
    """§12 P2-A — collapse off-schema fields whose VALUE token-sets overlap >= the
    threshold even when the field name differs (car vs vehicle_ownership → one).
    ``order`` is best-first; the higher-ranked field of a dup pair survives."""
    kept: dict[str, list[dict[str, Any]]] = {}
    kept_sets: list[frozenset[str]] = []
    for fkey in order:
        ts = _value_token_set(field_records[fkey])
        if ts and any(ts and ks and len(ts & ks) / len(ts | ks) >= _VALUE_DEDUP_OVERLAP for ks in kept_sets):
            continue
        kept[fkey] = field_records[fkey]
        kept_sets.append(ts)
    return kept


def aggregate_slot_profile(
    rows: list[dict[str, Any]],
    entity_id: str,
    entity_type: str | None,
    *,
    rel_pairs: Iterable[tuple[str, str]] = (),
    max_slots: int = 12,
    multi_cap: int = 8,
) -> SlotProfile:
    """Aggregate one entity's slot-tagged facts into a ranked profile card.

    Typed slots route by the schema; durable off-schema facts are RETAINED (§5/§6.1,
    no recurrence gate) keyed by their own ``field`` — giving universal coverage for
    fieldless-type entities. ``max_slots`` is the single total-card-field cap.
    """
    entity_id = str(entity_id)
    slot_meta = {s.key: s for s in slots_for(entity_type)}  # typed slots: cardinality + importance + labeled

    by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    off_schema_durable: list[dict[str, Any]] = []
    off_schema_candidates = 0  # §12 P2-C telemetry (durable off-schema entering retention)
    for row in rows:
        created = row.get("created_at") or _EPOCH
        trace_id = row.get("trace_id")
        # Parent-trace salience (Phase A §13) — significance-gated decay + reinforcement,
        # applied nightly by consolidation. Every fact on this trace inherits it as its
        # rank signal. ``significance`` is carried on the row contract for observability /
        # future per-fact decay; salience already encodes it, so ranking uses salience.
        salience = _as_salience(row.get("salience"))
        for fact in row.get("facts") or []:
            if not isinstance(fact, dict):
                continue
            if str(fact.get("subject_id") or "") != entity_id:
                continue  # attribution: only this entity's facts
            slot = fact.get("slot")
            # ``claimed_durability`` (hardening R-1) = Haiku's raw tri-state opinion; it
            # drives the off-schema quarantine below. It falls back to ``durability`` for
            # traces written/backfilled before the split (those carry Haiku's tag in
            # ``durability`` directly), so retention behavior is unchanged for old data.
            claimed = fact.get("claimed_durability") or fact.get("durability") or "transient"
            item = {
                "_dt": created,
                "value": fact.get("value"),
                "field": fact.get("field"),
                "polarity": fact.get("polarity"),
                "durability": fact.get("durability") or "transient",
                "source_trace_id": trace_id,
                "salience": salience,
            }
            if slot == "off_schema":
                # §6.1 quarantine: retain off-schema facts Haiku CLAIMED durable (the
                # slot-promotion signal), NOT the code-stamped tag (which is uniformly
                # transient for off_schema under D-1). The record reflects the claimed
                # status so the summarizer sees it as durable; decay still fades junk.
                if claimed == "durable":
                    off_schema_candidates += 1
                    item["durability"] = "durable"
                    off_schema_durable.append(item)
                continue
            if not slot or slot not in slot_meta:
                continue
            by_slot[slot].append(item)

    slot_records: dict[str, list[dict[str, Any]]] = {}
    slot_salience: dict[str, float] = {}  # Phase A §13 rank key — max parent-trace salience
    for slot, items in by_slot.items():
        surviving = _survivors(items)
        if not surviving:
            continue
        slot_records[slot] = _reduce_one_slot(surviving, slot, slot_meta[slot].cardinality, multi_cap)
        slot_salience[slot] = _max_salience(surviving)

    # §6.1 / §12 P2-B — field-keyed off-schema retention + cross-field value-dedup.
    # Order off-schema fields by max salience (recency tiebreak); the higher-salience
    # field of a value-dup pair survives (Phase A §13).
    off_field_records = _retain_off_schema(off_schema_durable, multi_cap)
    field_salience = _field_max_salience(off_schema_durable)
    off_order = sorted(
        off_field_records,
        key=lambda f: (field_salience.get(f, 0.0), _max_recency(off_field_records[f])),
        reverse=True,
    )
    off_field_records = _dedup_off_schema_by_value(off_field_records, off_order)
    off_schema_keys: set[str] = set()
    for fkey, recs in off_field_records.items():
        if fkey in slot_records:
            continue  # a populated typed slot owns the key
        slot_records[fkey] = recs
        slot_salience[fkey] = field_salience.get(fkey, 0.0)
        off_schema_keys.add(fkey)

    _fold_relationships(rel_pairs, slot_records, slot_meta)

    # Rank importance → recency → salience(tiebreak); stable, least-significant first.
    # NEVER alphabetical. ``slot_meta.get`` (§12 P1-A) gives synthetic/off-schema keys a
    # fallback importance so an unknown/None entity type can't KeyError. Salience (Phase A
    # §13) is the final tiebreak — durable facts persist, transient ones fade.
    def _importance(s: str) -> int:
        meta = slot_meta.get(s)
        return meta.importance if meta is not None else _OFF_SCHEMA_IMPORTANCE

    ranked = list(slot_records)
    ranked.sort(key=lambda s: slot_salience.get(s, 0.0), reverse=True)
    ranked.sort(key=lambda s: _max_recency(slot_records[s]), reverse=True)
    ranked.sort(key=_importance, reverse=True)
    ranked = ranked[:max_slots]  # single total-card-field cap (§12 P2-B)

    canonical_fields: dict[str, str] = {}
    for s in ranked:
        meta = slot_meta.get(s)
        if meta is not None and meta.cardinality == Cardinality.MULTI and meta.labeled:
            canonical_fields[s] = ", ".join(_label(r["field"], r["value"]) for r in slot_records[s])
        else:
            canonical_fields[s] = ", ".join(r["value"] for r in slot_records[s])
    canonical_slot_values = {s: slot_records[s] for s in ranked}
    retained = sum(len(slot_records[s]) for s in ranked if s in off_schema_keys)
    return SlotProfile(
        canonical_fields=canonical_fields,
        canonical_slot_values=canonical_slot_values,
        off_schema_durable_count=off_schema_candidates,
        off_schema_retained_count=retained,
    )


def _fold_relationships(
    rel_pairs: Iterable[tuple[str, str]],
    slot_records: dict[str, list[dict[str, Any]]],
    slot_meta: dict[str, Any],
) -> None:
    """Fold (predicate, object) pairs into slots in place — facts win."""
    for predicate, obj in rel_pairs:
        slot = PREDICATE_SLOT_MAP.get((predicate or "").strip().lower())
        obj = (obj or "").strip()
        if not slot or not obj or slot not in slot_meta:
            continue
        rec = {
            "value": obj,
            "field": None,  # relationship-derived (no source fact field)
            "observed_at": None,
            "source_trace_id": None,
            "durability": "durable",
            "slot": slot,
        }
        existing = slot_records.get(slot)
        if slot_meta[slot].cardinality == Cardinality.SINGLE:
            if not existing:
                slot_records[slot] = [rec]
        else:
            lst = existing or []
            if not any(_normalized_equal(obj, r["value"]) for r in lst):
                lst.append(rec)
            slot_records[slot] = lst
