"""Unit tests for the INC 3 durable-attribute aggregator (slot_profile) + the two
slot-description tweaks."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

# ── (A) the two minimal slot-description tweaks ─────────────────────────────


def _person_slot(key: str):
    from kai_memory.durable_schema import slots_for

    return next(s for s in slots_for("person") if s.key == key)


def test_occupation_description_mentions_field_of_expertise():
    desc = _person_slot("occupation").description
    assert "field of expertise" in desc
    assert "cosmetologist" in desc


def test_stable_preferences_excludes_product_attributes():
    desc = _person_slot("stable_preferences").description
    assert "The person's own" in desc
    assert "attributes of a product" in desc


def test_slot_durability_rubric_untouched_by_tweaks():
    # The tweaks must touch ONLY the two Slot.descriptions, not the shared rubric.
    from kai_memory.durable_schema import SLOT_DURABILITY_RUBRIC

    assert SLOT_DURABILITY_RUBRIC.startswith("· slot:")
    assert 'When uncertain → "transient".' in SLOT_DURABILITY_RUBRIC


# ── (B) the aggregator (slot_profile) ───────────────────────────────────────

_EID = str(uuid.uuid4())


def _dt(day: int) -> datetime:
    return datetime(2026, 6, day, tzinfo=UTC)


def _fact(slot, value, *, durability="durable", polarity="is", subject_id=_EID, field="f"):
    return {
        "slot": slot,
        "durability": durability,
        "polarity": polarity,
        "subject_id": subject_id,
        "field": field,
        "value": value,
    }


def _row(trace_id, day, facts, *, salience=0.5, significance=3):
    # salience defaults to the column default (0.5) so existing tests rank by the
    # recency tiebreak; salience-sensitive tests pass it explicitly.
    return {
        "trace_id": trace_id,
        "created_at": _dt(day),
        "facts": facts,
        "salience": salience,
        "significance": significance,
    }


def _agg(rows, **kw):
    from kai_memory.slot_profile import aggregate_slot_profile

    return aggregate_slot_profile(rows, _EID, "person", **kw)


def test_full_name_renders_display_only_single_slot():
    """A3/C-1: full_name is a SINGLE slot that renders on the card (display-only). The
    aggregator only produces card fields — it never feeds entity resolution/dedup — so a
    free-text full_name cannot re-introduce the #201 script forks."""
    rows = [
        _row("t1", 1, [_fact("full_name", "Михаил", field="full_name")]),
        _row("t2", 5, [_fact("full_name", "Михаил Петров", field="full_name")]),
    ]
    prof = _agg(rows)
    assert prof.canonical_fields["full_name"] == "Михаил Петров"  # newest-wins
    recs = prof.canonical_slot_values["full_name"]
    assert len(recs) == 1 and recs[0]["slot"] == "full_name"


def test_single_slot_newest_wins():
    rows = [
        _row("t1", 1, [_fact("current_location", "Minsk")]),
        _row("t2", 5, [_fact("current_location", "Corfu")]),
    ]
    prof = _agg(rows)
    assert prof.canonical_fields["current_location"] == "Corfu"
    recs = prof.canonical_slot_values["current_location"]
    assert len(recs) == 1
    assert recs[0]["value"] == "Corfu"
    assert recs[0]["slot"] == "current_location"
    assert recs[0]["source_trace_id"] == "t2"
    assert recs[0]["observed_at"] == _dt(5).isoformat()
    assert recs[0]["durability"] == "durable"


def test_multi_slot_accumulates_distinct_newest_first():
    rows = [
        _row("t1", 1, [_fact("languages", "Russian")]),
        _row("t2", 5, [_fact("languages", "English")]),
        _row("t3", 6, [_fact("languages", "English")]),  # exact dup of the day-5 value
    ]
    prof = _agg(rows)
    assert prof.canonical_fields["languages"] == "English, Russian"  # newest-first, deduped
    assert [r["value"] for r in prof.canonical_slot_values["languages"]] == ["English", "Russian"]


def test_off_schema_and_unattributed_facts_excluded():
    other = str(uuid.uuid4())
    rows = [
        _row("t1", 1, [_fact("off_schema", "noise", durability="operational")]),
        _row("t2", 2, [_fact("current_location", "Tokyo", subject_id=other)]),  # other entity
        _row("t3", 3, [_fact("citizenship", "RU")]),
    ]
    prof = _agg(rows)
    assert "current_location" not in prof.canonical_fields  # other entity's fact dropped
    assert set(prof.canonical_fields) == {"citizenship"}


def test_is_not_tombstones_value():
    rows = [
        _row("t1", 1, [_fact("citizenship", "RU")]),
        _row("t2", 2, [_fact("citizenship", "RU", polarity="is_not")]),
    ]
    prof = _agg(rows)
    assert "citizenship" not in prof.canonical_fields  # the only value was tombstoned


def test_relationship_fold_fills_absent_single_slot_but_facts_win():
    # works_at -> employer; fact-derived employer wins over the fold.
    rows = [_row("t1", 1, [_fact("employer", "Acme")])]
    prof = _agg(rows, rel_pairs=[("works_at", "Examplecorp")])
    assert prof.canonical_fields["employer"] == "Acme"  # fact wins

    # with no employer fact, the fold fills it.
    prof2 = _agg([_row("t1", 1, [_fact("citizenship", "RU")])], rel_pairs=[("works_at", "Examplecorp")])
    assert prof2.canonical_fields["employer"] == "Examplecorp"
    rec = prof2.canonical_slot_values["employer"][0]
    assert rec["source_trace_id"] is None and rec["observed_at"] is None  # relationship-derived


def test_relationship_fold_skips_slot_invalid_for_type():
    # owner_of -> owns, but 'owns' is not a person slot → dropped.
    prof = _agg([_row("t1", 1, [_fact("citizenship", "RU")])], rel_pairs=[("owner_of", "Nora")])
    assert "owns" not in prof.canonical_fields


def test_ranking_by_importance_not_alphabetical():
    # current_location (importance 9) must rank before citizenship (6) and occupation (7),
    # never alphabetical (which would put citizenship first).
    rows = [
        _row("t1", 1, [_fact("citizenship", "RU")]),
        _row("t1", 1, [_fact("occupation", "founder")]),
        _row("t1", 1, [_fact("current_location", "Corfu")]),
    ]
    prof = _agg(rows)
    assert list(prof.canonical_fields) == ["current_location", "occupation", "citizenship"]


def test_max_slots_cap_keeps_highest_importance():
    rows = [
        _row("t1", 1, [_fact("citizenship", "RU")]),  # imp 6
        _row("t1", 1, [_fact("current_location", "Corfu")]),  # imp 9
        _row("t1", 1, [_fact("occupation", "founder")]),  # imp 7
    ]
    prof = _agg(rows, max_slots=2)
    assert list(prof.canonical_fields) == ["current_location", "occupation"]


def test_record_shape_has_all_keys():
    prof = _agg([_row("t1", 1, [_fact("citizenship", "RU", durability="durable")])])
    rec = prof.canonical_slot_values["citizenship"][0]
    assert set(rec) == {"value", "field", "observed_at", "source_trace_id", "durability", "slot"}


# ── Task #10: multi-slot refinement (label + recurrence-rank + cap + dedup) ──


def test_labeled_multi_slot_renders_field_prefix():
    # physical_attributes is a labeled MULTI slot → "height: 170 cm".
    rows = [_row("t1", 1, [_fact("physical_attributes", "170 cm", field="height")])]
    prof = _agg(rows)
    assert prof.canonical_fields["physical_attributes"] == "height: 170 cm"
    # field survives into the structured store
    assert prof.canonical_slot_values["physical_attributes"][0]["field"] == "height"


def test_labeled_multi_skips_prefix_for_self_contained_value():
    # a multi-word, all-alphabetic value reads fine without the field label.
    rows = [_row("t1", 1, [_fact("chronic_health_conditions", "autoimmune thyroiditis", field="diagnosis")])]
    prof = _agg(rows)
    assert prof.canonical_fields["chronic_health_conditions"] == "autoimmune thyroiditis"


def test_labeled_multi_labels_fragment_value():
    rows = [_row("t1", 1, [_fact("chronic_health_conditions", "confirmed", field="gilberts_syndrome")])]
    prof = _agg(rows)
    assert prof.canonical_fields["chronic_health_conditions"] == "gilberts_syndrome: confirmed"


def test_bare_multi_slot_not_labeled():
    # languages is a bare (labeled=False) MULTI slot → no field prefix.
    rows = [_row("t1", 1, [_fact("languages", "English", field="language")])]
    prof = _agg(rows)
    assert prof.canonical_fields["languages"] == "English"


def test_higher_salience_fact_ranks_above_lower():
    # Phase A §13: a higher-salience value outranks a lower-salience one even when the
    # low-salience value is NEWER (recency alone would invert the order).
    rows = [
        _row("t1", 9, [_fact("languages", "English")], salience=0.2),  # newer, low salience
        _row("t2", 1, [_fact("languages", "Russian")], salience=0.9),  # older, high salience
    ]
    prof = _agg(rows)
    assert prof.canonical_fields["languages"] == "Russian, English"  # salience beats recency


def test_low_significance_transient_drops_at_cap_high_significance_persists():
    # Phase A §13: salience encodes significance-gated decay — the high-significance
    # (durable) value survives the cap, the low-significance (transient, fast-decayed)
    # value drops, even though it is newer.
    rows = [
        _row("t1", 9, [_fact("languages", "Klingon")], salience=0.1, significance=1),
        _row("t2", 1, [_fact("languages", "English")], salience=0.95, significance=5),
    ]
    prof = _agg(rows, multi_cap=1)
    assert [r["value"] for r in prof.canonical_slot_values["languages"]] == ["English"]


def test_max_salience_taken_across_duplicate_mentions():
    # Phase A §13: a value seen across traces inherits the MAX salience of its mentions.
    # The newest mention (the rendered representative) is the LOW-salience one, so if the
    # group used the rep's own salience it would rank below Spanish; MAX puts it first.
    rows = [
        _row("t1", 1, [_fact("languages", "English")], salience=0.95),  # older mention, HIGH salience
        _row("t2", 5, [_fact("languages", "English")], salience=0.1),  # newest mention (rep), LOW salience
        _row("t3", 3, [_fact("languages", "Spanish")], salience=0.5),  # competitor, mid salience
    ]
    prof = _agg(rows)
    assert prof.canonical_fields["languages"] == "English, Spanish"


def test_multi_slot_caps_at_n():
    rows = [_row(f"t{i}", i, [_fact("languages", f"Lang{i}")]) for i in range(1, 11)]  # 10 distinct
    prof = _agg(rows, multi_cap=3)
    assert len(prof.canonical_slot_values["languages"]) == 3


def test_near_duplicate_values_collapse():
    rows = [
        _row("t1", 1, [_fact("chronic_health_conditions", "asymptomatic for 2 years", field="status")]),
        _row("t2", 2, [_fact("chronic_health_conditions", "asymptomatic past 2 years", field="status")]),
    ]
    prof = _agg(rows)
    # connector-variant near-dupes collapse to a single value
    assert len(prof.canonical_slot_values["chronic_health_conditions"]) == 1


def test_payload_accepts_canonical_slot_values():
    # EntitySchemaPayload must carry the decoupled structured store (cutover target).
    from kai_memory.artifact_schemas import EntitySchemaPayload

    p = EntitySchemaPayload(
        canonical_name="Misha",
        entity_type="person",
        canonical_fields={"current_location": "Corfu"},
        canonical_slot_values={"current_location": [{"value": "Corfu", "slot": "current_location"}]},
    )
    assert p.canonical_slot_values["current_location"][0]["value"] == "Corfu"
    assert p.canonical_fields == {"current_location": "Corfu"}


# ── INC 3 inversion increment 1: off-schema field-keyed retention ──────────


def _off(value, *, durability="durable", field="x", polarity="is"):
    return _fact("off_schema", value, durability=durability, field=field, polarity=polarity)


def test_offschema_durable_retained_field_keyed():
    rows = [_row("t1", 1, [_off("SampleGym", field="gym")])]
    prof = _agg(rows)
    assert prof.canonical_fields["gym"] == "SampleGym"  # keyed by the fact's field, not a blob
    rec = prof.canonical_slot_values["gym"][0]
    assert rec["value"] == "SampleGym" and rec["slot"] == "gym" and rec["field"] == "gym"
    assert "other_durable_attributes" not in prof.canonical_fields  # single-blob retired


def test_singleton_now_survives_no_recurrence_gate():
    # the helmet/Li-Auto era recurrence gate is gone: one trace is enough.
    prof = _agg([_row("t1", 1, [_off("ankle sprain", field="injury")])])
    assert prof.canonical_fields["injury"] == "ankle sprain"


def test_offschema_transient_and_operational_durability_not_retained():
    rows = [
        _row("t1", 1, [_off("bloating", durability="transient", field="symptom")]),
        _row("t2", 2, [_off("noise", durability="operational", field="thing")]),
    ]
    assert _agg(rows).canonical_fields == {}  # only durable off-schema is retained


def test_offschema_retained_by_claimed_durability_under_d1():
    """R-1 regression: a write-path off-schema fact whose CODE-STAMPED durability is
    'transient' (D-1 stamps every off_schema transient) but whose Haiku
    ``claimed_durability`` is 'durable' is STILL retained in the §6.1 quarantine — the
    split keeps slot-promotion alive. The record reflects the claimed-durable status."""
    fact = {
        "slot": "off_schema",
        "durability": "transient",  # code-stamped (D-1)
        "claimed_durability": "durable",  # Haiku's raw opinion
        "polarity": "is",
        "subject_id": _EID,
        "field": "hobby",
        "value": "freediving",
    }
    prof = _agg([_row("t1", 1, [fact])])
    assert prof.canonical_fields.get("hobby") == "freediving"
    assert prof.off_schema_durable_count == 1
    assert prof.canonical_slot_values["hobby"][0]["durability"] == "durable"


def test_offschema_not_retained_when_claimed_not_durable_under_d1():
    """Inverse of R-1: code-stamped transient + claimed transient ⇒ junk stays out."""
    fact = {
        "slot": "off_schema",
        "durability": "transient",
        "claimed_durability": "transient",
        "polarity": "is",
        "subject_id": _EID,
        "field": "mood",
        "value": "tired",
    }
    assert _agg([_row("t1", 1, [fact])]).canonical_fields == {}


def test_operational_field_no_longer_hard_blocked():
    # Phase A §13: the hand-maintained operational blocklist is GONE. A durable-tagged
    # operational field is no longer suppressed at aggregation — the inc2 durability
    # rubric keeps such facts out of the durable set up front, and decay fades any
    # leakage. So a durable off-schema field flows straight through.
    rows = [_row("t1", 1, [_off("0", field="exit_code")])]
    prof = _agg(rows)
    assert prof.canonical_fields["exit_code"] == "0"  # not blocklisted anymore


def test_cross_field_value_dedup():
    # "car: Li Auto" and "vehicle_ownership: Li Auto" collapse to one.
    rows = [
        _row("t1", 1, [_off("Li Auto", field="car")]),
        _row("t2", 2, [_off("Li Auto", field="vehicle_ownership")]),
    ]
    prof = _agg(rows)
    assert len([k for k in prof.canonical_slot_values if k in ("car", "vehicle_ownership")]) == 1


def test_entity_type_none_does_not_crash():
    from kai_memory.slot_profile import aggregate_slot_profile

    prof = aggregate_slot_profile([_row("t1", 1, [_off("SampleGym", field="gym")])], _EID, None)
    assert prof.canonical_fields["gym"] == "SampleGym"  # retained, no KeyError on slots_for(None)==[]


def test_non_person_entity_gets_field_keyed_card():
    from kai_memory.slot_profile import aggregate_slot_profile

    rows = [_row("t1", 1, [_off("Beijing", field="capital")]), _row("t2", 2, [_off("CNY", field="currency")])]
    prof = aggregate_slot_profile(rows, _EID, "topic")  # fieldless type → structured, not empty
    assert prof.canonical_fields["capital"] == "Beijing"
    assert prof.canonical_fields["currency"] == "CNY"


def test_typed_slot_outranks_off_schema_field():
    rows = [_row("t1", 1, [_fact("current_location", "Corfu")]), _row("t1", 1, [_off("SampleGym", field="gym")])]
    order = list(_agg(rows).canonical_fields)
    assert order[0] == "current_location" and order[-1] == "gym"


def test_total_card_field_cap():
    rows = [_row(f"t{i}", i + 1, [_off(f"V{i}", field=f"f{i}")]) for i in range(20)]
    assert len(_agg(rows, max_slots=5).canonical_fields) == 5


def test_telemetry_counts_on_profile():
    # Two durable off-schema candidates with the same VALUE under different fields →
    # both counted, but cross-field value-dedup collapses them to one retained.
    rows = [
        _row("t1", 1, [_off("Li Auto", field="car")]),  # retained
        _row("t2", 2, [_off("Li Auto", field="vehicle_ownership")]),  # value-deduped away
    ]
    prof = _agg(rows)
    assert prof.off_schema_durable_count == 2  # both durable candidates entered retention
    assert prof.off_schema_retained_count == 1  # value-dedup collapsed to one


def test_owner_of_removed_from_predicate_map():
    from kai_memory.slot_profile import PREDICATE_SLOT_MAP

    assert "owner_of" not in PREDICATE_SLOT_MAP
    assert "owns" not in PREDICATE_SLOT_MAP.values()


def test_aggregate_slot_profile_is_pure():
    # Pure: no DB / LLM / logging side-effects → deterministic, and the input rows
    # (and their fact dicts) are NOT mutated.
    import copy

    from kai_memory.slot_profile import aggregate_slot_profile

    rows = [
        _row("t1", 1, [_fact("languages", "English"), _fact("languages", "Russian")], salience=0.7),
        _row("t2", 2, [_off("SampleGym", field="gym")], salience=0.4),
    ]
    before = copy.deepcopy(rows)
    a = aggregate_slot_profile(rows, _EID, "person")
    b = aggregate_slot_profile(rows, _EID, "person")
    assert a == b  # deterministic
    assert rows == before  # input not mutated


def test_missing_salience_defaults_and_does_not_crash():
    # A thin row with no salience key falls back to the column default (0.5) — the
    # aggregator must not KeyError on the rank path.
    from kai_memory.slot_profile import aggregate_slot_profile

    rows = [{"trace_id": "t1", "created_at": _dt(1), "facts": [_fact("citizenship", "RU")]}]
    prof = aggregate_slot_profile(rows, _EID, "person")
    assert prof.canonical_fields["citizenship"] == "RU"


# ── Token-overlap tombstone matching ("I don't have X" actually erases X) ────


def test_tombstone_token_overlap_erases_suffix_divergent_value():
    # "I don't have a Li Auto car" must erase the stored "Li Auto L6" — same field,
    # shares {li, auto}, diverges only at the suffix (Jaccard 2/4 = 0.5 >= 0.4).
    rows = [
        _row("t1", 1, [_off("Li Auto L6", field="vehicle_ownership")]),
        _row("t2", 2, [_off("Li Auto car", field="vehicle_ownership", polarity="is_not")]),
    ]
    prof = _agg(rows)
    assert "vehicle_ownership" not in prof.canonical_fields  # erased by token-overlap


def test_tombstone_token_overlap_handles_leading_negation_word():
    # "no Li Auto car" {no,li,auto,car} vs "Li Auto L6" {li,auto,l6} → 2/5 = 0.4 (== threshold).
    rows = [
        _row("t1", 1, [_off("Li Auto L6", field="vehicle_ownership")]),
        _row("t2", 2, [_off("no Li Auto car", field="vehicle_ownership", polarity="is_not")]),
    ]
    prof = _agg(rows)
    assert "vehicle_ownership" not in prof.canonical_fields


def test_tombstone_replacement_then_newer_value_wins():
    # is_not "Minsk" erases stored "Minsk/Klimovichi" (slash splits into
    # {minsk, klimovichi}; shares {minsk}, 1/2 = 0.5 >= 0.4); a newer "Perm" then
    # stands alone. Without erasure both would coexist ("Perm, Minsk/Klimovichi").
    rows = [
        _row("t1", 1, [_off("Minsk/Klimovichi", field="hometown")]),
        _row("t2", 2, [_off("Minsk", field="hometown", polarity="is_not")]),
        _row("t3", 3, [_off("Perm", field="hometown")]),
    ]
    prof = _agg(rows)
    assert prof.canonical_fields["hometown"] == "Perm"  # old value erased, replacement wins


def test_tombstone_does_not_overmatch_single_shared_token_coffee_tea():
    # "black coffee" {black,coffee} vs "black tea" {black,tea}: only {black} shared,
    # Jaccard 1/3 < 0.4 → the stored "black tea" must NOT be erased.
    rows = [
        _row("t1", 1, [_off("black tea", field="drink")]),
        _row("t2", 2, [_off("black coffee", field="drink", polarity="is_not")]),
    ]
    prof = _agg(rows)
    assert prof.canonical_fields["drink"] == "black tea"  # not over-matched


def test_tombstone_does_not_overmatch_brand_prefix():
    # is_not "Li Auto" {li,auto} vs stored "Li Ning" {li,ning}: only {li} shared,
    # 1/3 < 0.4 → an unrelated brand survives the negation.
    rows = [
        _row("t1", 1, [_off("Li Ning", field="apparel_brand")]),
        _row("t2", 2, [_off("Li Auto", field="apparel_brand", polarity="is_not")]),
    ]
    prof = _agg(rows)
    assert prof.canonical_fields["apparel_brand"] == "Li Ning"


def test_tombstone_matches_unit():
    from kai_memory.slot_profile import _tokens, _tombstone_matches

    # exact #201-normalized equality stays the fast path (regression)
    assert _tombstone_matches("Li Auto L6", "Li Auto L6")
    assert _tombstone_matches("RU", "ru")
    # token-overlap matches and guards
    assert _tombstone_matches("Li Auto L6", "Li Auto car")
    assert _tombstone_matches("Minsk/Klimovichi", "Minsk")
    assert not _tombstone_matches("black tea", "black coffee")
    assert not _tombstone_matches("Li Ning", "Li Auto")
    # empty input / no shared token never matches (and no ZeroDivisionError)
    assert not _tombstone_matches("", "Li Auto")
    assert not _tombstone_matches("Perm", "Minsk")
    # slash is a token separator so translit_key can't glue compounds into one token
    assert _tokens("Minsk/Klimovichi") == frozenset({"minsk", "klimovichi"})
