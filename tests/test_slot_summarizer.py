"""Unit tests for the INC 3 whole-entity profile curator (slot_summarizer)."""

from __future__ import annotations

import pytest

from kai_memory import slot_summarizer as ss
from kai_memory.slot_profile import SlotProfile

pytestmark = pytest.mark.asyncio


def _rec(value, *, field="f", observed_at=None, durability="durable", slot="chronic_health_conditions", tid="t1"):
    return {
        "value": value,
        "field": field,
        "observed_at": observed_at,
        "source_trace_id": tid,
        "durability": durability,
        "slot": slot,
    }


def _profile():
    # WHOLE entity: a MULTI typed slot + a SINGLE typed slot + off-schema fields
    # (one of them granular logbook minutiae).
    return SlotProfile(
        canonical_fields={
            "chronic_health_conditions": "injury: ankle sprain",
            "current_location": "Corfu",
            "gym": "SampleGym",
            "supplement_dose": "5000 IU",
        },
        canonical_slot_values={
            "chronic_health_conditions": [
                _rec("ankle sprain", field="injury", observed_at="2025-12-10T00:00:00+00:00", tid="t1"),
            ],
            "current_location": [
                _rec("Corfu", field="city", observed_at="2025-06-01T00:00:00+00:00", slot="current_location", tid="t3"),
            ],
            "gym": [_rec("SampleGym", field="gym", observed_at="2025-05-01T00:00:00+00:00", slot="gym", tid="t4")],
            "supplement_dose": [
                _rec(
                    "5000 IU",
                    field="supplement_dose",
                    observed_at="2025-04-01T00:00:00+00:00",
                    slot="supplement_dose",
                    tid="t5",
                ),
            ],
        },
    )


def _patch_llm(monkeypatch, fn):
    monkeypatch.setattr("kai_memory.ports.call_llm_text", fn)


async def test_curation_input_is_whole_entity_not_just_multi(monkeypatch):
    # The prompt must carry EVERY slot/field — SINGLE + off-schema too, not just MULTI.
    seen = {}

    async def fake_call(provider, prompt, *a, **k):
        seen["prompt"] = prompt
        return '{"summary": "x"}'

    _patch_llm(monkeypatch, fake_call)
    await ss.curate_profile(_profile(), entity_name="Misha", provider=object())
    p = seen["prompt"]
    assert "ankle sprain" in p  # MULTI typed slot
    assert "Corfu" in p  # SINGLE typed slot (excluded by the old summarizer)
    assert "SampleGym" in p  # off-schema field (excluded by the old summarizer)
    assert "5000 IU" in p  # granular off-schema field


async def test_curated_card_replaces_canonical_fields(monkeypatch):
    curated = {"health": "ankle sprain (healing)", "location": "Corfu", "fitness": "trains at SampleGym gym"}

    async def fake_call(*a, **k):
        import json

        return json.dumps(curated)

    _patch_llm(monkeypatch, fake_call)
    out = await ss.curate_profile(_profile(), entity_name="Misha", provider=object())
    assert out.canonical_fields == curated  # whole card replaced by the curated gist


async def test_canonical_slot_values_unchanged(monkeypatch):
    async def fake_call(*a, **k):
        return '{"health": "ankle sprain", "summary": "lives in Corfu"}'

    _patch_llm(monkeypatch, fake_call)
    prof = _profile()
    out = await ss.curate_profile(prof, entity_name="Misha", provider=object())
    # SUBSTRATE preserved verbatim — curation only rewrites the card text.
    assert out.canonical_slot_values == prof.canonical_slot_values
    assert out.canonical_slot_values is prof.canonical_slot_values


async def test_granular_field_dropped_from_card_but_kept_in_substrate(monkeypatch):
    # The curated card abstracts away the granular "supplement_dose" minutiae, but the
    # full value stays in the queryable substrate.
    async def fake_call(*a, **k):
        return '{"health": "recovering from a ankle sprain", "supplements": "takes vitamin D"}'

    _patch_llm(monkeypatch, fake_call)
    prof = _profile()
    out = await ss.curate_profile(prof, entity_name="Misha", provider=object())
    assert "supplement_dose" not in out.canonical_fields  # minutiae abstracted out of the card
    substrate_values = [r["value"] for recs in out.canonical_slot_values.values() for r in recs]
    assert "5000 IU" in substrate_values  # ...but still in the substrate


async def test_llm_failure_falls_back_to_deterministic(monkeypatch):
    async def fake_call(*a, **k):
        return None  # call_llm_text returns None on failure/truncation

    _patch_llm(monkeypatch, fake_call)
    prof = _profile()
    out = await ss.curate_profile(prof, entity_name="Misha", provider=object())
    assert out.canonical_fields == prof.canonical_fields  # deterministic card preserved


async def test_one_call_per_entity(monkeypatch):
    calls = []

    async def fake_call(provider, prompt, *a, **k):
        calls.append(prompt)
        return '{"summary": "curated"}'

    _patch_llm(monkeypatch, fake_call)
    out = await ss.curate_profile(_profile(), entity_name="Misha", provider=object())
    assert len(calls) == 1  # exactly ONE Haiku call for the whole entity
    assert out.canonical_fields == {"summary": "curated"}


async def test_no_facts_skips_llm(monkeypatch):
    called = False

    async def fake_call(*a, **k):
        nonlocal called
        called = True
        return "{}"

    _patch_llm(monkeypatch, fake_call)
    prof = SlotProfile(canonical_fields={}, canonical_slot_values={})
    out = await ss.curate_profile(prof, entity_name="Misha", provider=object())
    assert called is False  # no facts → no LLM call
    assert out is prof  # unchanged


async def test_empty_curation_keeps_deterministic_card(monkeypatch):
    # The model returns nothing usable → keep the deterministic card whole.
    async def fake_call(*a, **k):
        return "{}"

    _patch_llm(monkeypatch, fake_call)
    prof = _profile()
    out = await ss.curate_profile(prof, entity_name="Misha", provider=object())
    assert out.canonical_fields == prof.canonical_fields


async def test_input_length_cap_truncates_to_top_facts(monkeypatch):
    # A fact-heavy entity is capped to the top _MAX_CURATION_FACTS (substrate is already
    # priority-ranked) so the one Haiku call stays bounded.
    n = ss._MAX_CURATION_FACTS
    recs = [_rec(f"fact{i:03d}", field="languages", slot="languages", tid=f"t{i}") for i in range(n + 5)]
    prof = SlotProfile(
        canonical_fields={"languages": "many"},
        canonical_slot_values={"languages": recs},
    )
    seen = {}

    async def fake_call(provider, prompt, *a, **k):
        seen["prompt"] = prompt
        return '{"languages": "polyglot"}'

    _patch_llm(monkeypatch, fake_call)
    await ss.curate_profile(prof, entity_name="Misha", provider=object())
    p = seen["prompt"]
    assert "fact000" in p  # top-priority fact kept
    assert f"fact{n + 4:03d}" not in p  # beyond the cap → dropped from the prompt
    assert p.count("observed=") == n  # exactly the cap number of fact lines rendered


async def test_cap_round_robins_across_slots_no_starvation(monkeypatch):
    # A fact-heavy EARLY slot must not starve a LATER slot out of the whole-entity
    # prompt: round-robin selection takes each slot's top fact before any slot's deeper
    # facts, so the later slot's top-ranked fact survives the cap.
    n = ss._MAX_CURATION_FACTS
    heavy = [_rec(f"lang{i:03d}", field="languages", slot="languages", tid=f"t{i}") for i in range(n + 10)]
    prof = SlotProfile(
        canonical_fields={"languages": "many", "current_location": "Corfu"},
        canonical_slot_values={
            "languages": heavy,  # early, fact-heavy slot
            "current_location": [_rec("Corfu", field="city", slot="current_location", tid="loc")],  # later slot
        },
    )
    seen = {}

    async def fake_call(provider, prompt, *a, **k):
        seen["prompt"] = prompt
        return '{"summary": "x"}'

    _patch_llm(monkeypatch, fake_call)
    await ss.curate_profile(prof, entity_name="Misha", provider=object())
    p = seen["prompt"]
    assert "Corfu" in p  # later slot's top fact survives despite the fact-heavy early slot
    assert "lang000" in p  # early slot's top fact also kept
    assert p.count("observed=") == n  # still capped to the budget
