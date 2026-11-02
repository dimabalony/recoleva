"""Unit tests for the INC 3 durable-attribute (slot + durability) backfill tool."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai_memory import durable_backfill as bf

pytestmark = pytest.mark.asyncio


class _Ent:
    def __init__(self, name, etype="person"):
        self.id = uuid.uuid4()
        self.name = name
        self.entity_type = etype


def _fact(field, value, *, slot=None, durability=None, subject_id=None):
    f = {"field": field, "value": value, "polarity": "is"}
    if slot is not None:
        f["slot"] = slot
    if durability is not None:
        f["durability"] = durability
    if subject_id is not None:
        f["subject_id"] = subject_id
    return f


# ── idempotency ─────────────────────────────────────────────────────────────


async def test_facts_need_classification_idempotency():
    assert bf.facts_need_classification([_fact("a", "b")]) is True
    assert bf.facts_need_classification([_fact("a", "b", slot="off_schema")]) is False


# ── pure classification (classify_trace_facts) ──────────────────────────────


async def test_classify_sets_valid_person_slot_and_durability():
    misha = _Ent("Misha", "person")
    facts = [_fact("citizenship", "Russian", subject_id=str(misha.id))]
    classifier = lambda content, fs: [{"slot": "citizenship", "durability": "durable"}]  # noqa: E731
    method = bf.classify_trace_facts("Misha is Canadian", facts, [misha], slot_classifier=classifier)
    assert method == "classified"
    assert facts[0]["slot"] == "citizenship"
    assert facts[0]["durability"] == "durable"


async def test_classify_normalizes_slot_case():
    misha = _Ent("Misha", "person")
    facts = [_fact("citizenship", "Russian", subject_id=str(misha.id))]
    classifier = lambda content, fs: [{"slot": "Citizenship", "durability": "durable"}]  # noqa: E731
    bf.classify_trace_facts("x", facts, [misha], slot_classifier=classifier)
    assert facts[0]["slot"] == "citizenship"  # normalized by the shared validator


async def test_classify_degrades_real_slot_without_subject_id():
    misha = _Ent("Misha", "person")
    facts = [_fact("citizenship", "Russian")]  # no subject_id
    classifier = lambda content, fs: [{"slot": "citizenship", "durability": "durable"}]  # noqa: E731
    bf.classify_trace_facts("x", facts, [misha], slot_classifier=classifier)
    assert facts[0]["slot"] == "off_schema"


async def test_classify_degrades_slot_invalid_for_entity_type():
    examplecorp = _Ent("Examplecorp", "organization")
    facts = [_fact("citizenship", "Russian", subject_id=str(examplecorp.id))]  # person slot on an org
    classifier = lambda content, fs: [{"slot": "citizenship", "durability": "durable"}]  # noqa: E731
    bf.classify_trace_facts("x", facts, [examplecorp], slot_classifier=classifier)
    assert facts[0]["slot"] == "off_schema"


async def test_classify_defaults_missing_and_bad_durability_to_transient():
    facts = [_fact("exit_code", "0")]
    classifier = lambda content, fs: [{"slot": "off_schema", "durability": "nonsense"}]  # noqa: E731
    bf.classify_trace_facts("x", facts, [], slot_classifier=classifier)
    assert facts[0]["slot"] == "off_schema"
    assert facts[0]["durability"] == "transient"


async def test_classify_off_schema_is_transient_under_d1():
    """Hardening D-1: code-stamped durability is COMPUTED from the resolved slot, not
    trusted from the classifier — an off_schema fact is transient even when the
    classifier says "operational". R-1: the raw tri-state opinion is preserved as
    ``claimed_durability`` (advisory, feeds the off-schema quarantine)."""
    facts = [_fact("exit_code", "0")]
    classifier = lambda content, fs: [{"slot": "off_schema", "durability": "operational"}]  # noqa: E731
    bf.classify_trace_facts("x", facts, [], slot_classifier=classifier)
    assert facts[0]["durability"] == "transient"  # retention signal (binary)
    assert facts[0]["claimed_durability"] == "operational"  # raw tri-state preserved (R-1)


async def test_classify_durability_computed_not_trusted_under_d1():
    """D-1: a classifier claiming a durable slot is transient (or vice-versa) is
    overridden — the code-stamped tag follows slot membership. citizenship ⇒ durable;
    the classifier's raw "transient" survives only as claimed_durability."""
    misha = _Ent("Misha", "person")
    facts = [_fact("citizenship", "Russian", subject_id=str(misha.id))]
    classifier = lambda content, fs: [{"slot": "citizenship", "durability": "transient"}]  # noqa: E731  wrong tag
    bf.classify_trace_facts("x", facts, [misha], slot_classifier=classifier)
    assert facts[0]["slot"] == "citizenship"
    assert facts[0]["durability"] == "durable"  # computed from slot, not the classifier's "transient"
    assert facts[0]["claimed_durability"] == "transient"  # raw opinion preserved separately


async def test_classify_no_classifier_leaves_unresolved():
    facts = [_fact("citizenship", "Russian")]
    method = bf.classify_trace_facts("x", facts, [], slot_classifier=None)
    assert method == "unresolved"
    assert "slot" not in facts[0]  # untouched, left for a later pass


async def test_classify_length_mismatch_raises():
    facts = [_fact("a", "b"), _fact("c", "d")]
    classifier = lambda content, fs: [{"slot": "off_schema", "durability": "transient"}]  # noqa: E731  (1 for 2)
    with pytest.raises(ValueError, match="misaligned"):
        bf.classify_trace_facts("x", facts, [], slot_classifier=classifier)


async def test_classify_preserves_field_value_subject():
    misha = _Ent("Misha", "person")
    facts = [
        {"field": "citizenship", "value": "Russian", "polarity": "is", "subject": "Misha", "subject_id": str(misha.id)}
    ]
    classifier = lambda content, fs: [{"slot": "citizenship", "durability": "durable"}]  # noqa: E731
    bf.classify_trace_facts("x", facts, [misha], slot_classifier=classifier)
    assert facts[0]["field"] == "citizenship"
    assert facts[0]["value"] == "Russian"
    assert facts[0]["subject"] == "Misha"
    assert facts[0]["subject_id"] == str(misha.id)


# ── §14 Haiku classifier ────────────────────────────────────────────────────


async def test_classifier_prompt_embeds_catalog_and_durability_rubric():
    from kai_memory.durable_schema import render_schema_for_prompt

    assert "{schema_catalog}" in bf._CLASSIFIER_PROMPT
    assert "durability" in bf._CLASSIFIER_PROMPT
    assert "off_schema" in bf._CLASSIFIER_PROMPT
    # the catalog renderer is the single source of truth for slot keys
    assert "citizenship" in render_schema_for_prompt()


async def test_haiku_slot_classifier_parses_slot_and_durability(monkeypatch):
    misha = _Ent("Misha", "person")
    facts = [
        {"field": "citizenship", "value": "Russian", "subject_id": str(misha.id)},
        {"field": "exit_code", "value": "0", "subject_id": None},
    ]

    async def fake_call(*a, **k):
        return (
            '[{"field":"citizenship","value":"Russian","slot":"citizenship","durability":"durable"},'
            '{"field":"exit_code","value":"0","slot":"off_schema","durability":"operational"}]'
        )

    monkeypatch.setattr("kai_memory.ports.call_llm_text", fake_call)
    out = await bf.haiku_slot_classifier("Misha is Canadian", facts, [misha], provider=object())
    assert out == [
        {"slot": "citizenship", "durability": "durable"},
        {"slot": "off_schema", "durability": "operational"},
    ]


async def test_haiku_slot_classifier_guards_malformed_items(monkeypatch):
    facts = [{"field": "a", "value": "b"}]

    async def fake_call(*a, **k):
        return '["garbage"]'

    monkeypatch.setattr("kai_memory.ports.call_llm_text", fake_call)
    out = await bf.haiku_slot_classifier("x", facts, [], provider=object())
    assert out == [{"slot": None, "durability": None}]  # malformed → empty signal, validator defaults later


# ── runner ──────────────────────────────────────────────────────────────────


def _db_ok_session():
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(scalar=MagicMock(return_value="ai_workforce")))
    return session


async def test_run_backfill_dry_run_writes_nothing():
    misha = _Ent("Misha", "person")
    traces = [
        {
            "id": "t1",
            "content": "Misha is Canadian",
            "tm": {"schema_version": 2, "facts": [_fact("citizenship", "Russian", subject_id=str(misha.id))]},
            "entities": [misha],
        }
    ]
    session = _db_ok_session()

    async def classifier(content, facts, ents):
        return [{"slot": "citizenship", "durability": "durable"}]

    with patch.object(bf, "_fetch_fact_bearing_traces", AsyncMock(return_value=traces)):
        report = await bf.run_backfill(session, apply=False, llm_slot_classifier=classifier)

    assert report["mode"] == "dry-run"
    assert report["counts"]["classified"] == 1
    assert report["counts"]["updated"] == 0
    update_calls = [c for c in session.execute.await_args_list if "UPDATE" in str(c.args[0])]
    assert update_calls == []
    session.commit.assert_not_called()
    assert report["samples"] and report["samples"][0]["slot"] == "citizenship"


async def test_run_backfill_apply_updates_commits_and_bumps_schema_version():
    misha = _Ent("Misha", "person")
    traces = [
        {
            "id": "t1",
            "content": "x",
            "tm": {"schema_version": 2, "facts": [_fact("citizenship", "Russian", subject_id=str(misha.id))]},
            "entities": [misha],
        }
    ]
    session = _db_ok_session()

    async def classifier(content, facts, ents):
        return [{"slot": "citizenship", "durability": "durable"}]

    with patch.object(bf, "_fetch_fact_bearing_traces", AsyncMock(return_value=traces)):
        report = await bf.run_backfill(session, apply=True, llm_slot_classifier=classifier)

    assert report["counts"]["updated"] == 1
    assert traces[0]["tm"]["schema_version"] == 3
    update_calls = [c for c in session.execute.await_args_list if "UPDATE" in str(c.args[0])]
    assert len(update_calls) == 1
    session.commit.assert_awaited_once()


async def test_run_backfill_skips_already_classified_idempotent():
    traces = [
        {
            "id": "t1",
            "content": "x",
            "tm": {"schema_version": 3, "facts": [_fact("citizenship", "Russian", slot="citizenship", subject_id="x")]},
            "entities": [_Ent("Misha")],
        }
    ]
    session = _db_ok_session()

    async def classifier(content, facts, ents):
        raise AssertionError("classifier must not be called for an already-classified trace")

    with patch.object(bf, "_fetch_fact_bearing_traces", AsyncMock(return_value=traces)):
        report = await bf.run_backfill(session, apply=True, llm_slot_classifier=classifier)
    assert report["counts"]["skipped_done"] == 1
    assert report["counts"]["updated"] == 0
    session.commit.assert_awaited_once()


async def test_run_backfill_refuses_wrong_database():
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(scalar=MagicMock(return_value="staging_db")))
    with pytest.raises(SystemExit, match="staging_db"):
        await bf.run_backfill(session, apply=True)


async def test_run_backfill_uses_sync_classifier_injection():
    misha = _Ent("Misha", "person")
    traces = [
        {
            "id": "t1",
            "content": "x",
            "tm": {"facts": [_fact("citizenship", "Russian", subject_id=str(misha.id))]},
            "entities": [misha],
        }
    ]
    session = _db_ok_session()
    classifier = lambda content, facts: [{"slot": "citizenship", "durability": "durable"}]  # noqa: E731
    with patch.object(bf, "_fetch_fact_bearing_traces", AsyncMock(return_value=traces)):
        report = await bf.run_backfill(session, apply=False, slot_classifier=classifier)
    assert report["counts"]["classified"] == 1
    assert traces[0]["tm"]["facts"][0]["slot"] == "citizenship"


# ── shared rubric (single source of truth) ──────────────────────────────────


async def test_slot_durability_rubric_shared_with_forward_prompt():
    """The backfill classifier and the forward EXTRACTION_PROMPT must embed the
    SAME durability rubric constant — single source of truth, no drift."""
    from kai_memory.extraction.memory_extraction import EXTRACTION_PROMPT
    from kai_memory.durable_schema import SLOT_DURABILITY_RUBRIC, render_schema_for_prompt

    forward = EXTRACTION_PROMPT.format(
        agent_display_name="Agent",
        served_rules_legend="",
        conversation="",
        thinking_blocks="",
        existing_memories="",
        participant_context="",
        current_timestamp="",
        schema_catalog=render_schema_for_prompt(),
        slot_durability_rubric=SLOT_DURABILITY_RUBRIC,
        procedural_block="",
        procedural_response_note="",
    )
    backfill = bf._CLASSIFIER_PROMPT.format(
        slot_durability_rubric=SLOT_DURABILITY_RUBRIC,
        schema_catalog=render_schema_for_prompt(),
        content="",
        facts="",
    )
    assert SLOT_DURABILITY_RUBRIC in forward
    assert SLOT_DURABILITY_RUBRIC in backfill


async def test_grounding_rubric_clauses_present():
    """Increment 2: the shared rubric carries the possession-vs-discussion grounding
    clause + the operational/lifestyle sharpenings, rendered in BOTH prompts."""
    from kai_memory.extraction.memory_extraction import EXTRACTION_PROMPT
    from kai_memory.durable_schema import SLOT_DURABILITY_RUBRIC, render_schema_for_prompt

    rubric = SLOT_DURABILITY_RUBRIC
    # (a) possession-vs-discussion grounding (Li Auto / H1B fix)
    assert "grounding:" in rubric
    assert "discussed, researched, compared" in rubric
    # Grounding v2 (2026-07-03): the rule now ENCODES weak predicates instead of
    # dropping them; the fabrication-guard phrase widened to "ownership, skill,
    # or identity". Whitespace-normalized (the clause line-wraps).
    assert "never as an ownership, skill, or identity fact" in " ".join(rubric.split())
    # (b) operational sharpened to the agent's own capabilities/process (automation_capability / NEIL)
    assert "AGENT'S OWN" in rubric and "capabilities" in rubric and "automation features" in rubric
    # (c) durable lifestyle clarification (gym/sport/tools were mis-tagged transient)
    assert "lifestyle attributes" in rubric and "membership" in rubric

    forward = EXTRACTION_PROMPT.format(
        agent_display_name="Agent",
        served_rules_legend="",
        conversation="",
        thinking_blocks="",
        existing_memories="",
        participant_context="",
        current_timestamp="",
        schema_catalog=render_schema_for_prompt(),
        slot_durability_rubric=rubric,
        procedural_block="",
        procedural_response_note="",
    )
    backfill = bf._CLASSIFIER_PROMPT.format(
        slot_durability_rubric=rubric, schema_catalog=render_schema_for_prompt(), content="", facts=""
    )
    assert "discussed, researched, compared" in forward  # forward extraction prompt carries grounding
    assert "discussed, researched, compared" in backfill  # backfill classifier carries grounding


# ── gate-sample selection ───────────────────────────────────────────────────


def _sfact(subject, *, etype="person", slot="off_schema"):
    return {
        "trace_id": "t",
        "subject": subject,
        "entity_type": etype,
        "field": "f",
        "value": "v",
        "slot": slot,
        "durability": "transient",
    }


async def test_select_sample_caps_to_size():
    facts = [_sfact(f"E{i}") for i in range(10)]
    out = bf._select_sample(facts, sample_size=3, ensure_entities=[])
    assert len(out) == 3


async def test_select_sample_always_includes_ensured_even_beyond_cap():
    facts = [_sfact("Misha"), _sfact("Elena"), _sfact("Katya")] + [_sfact(f"E{i}") for i in range(10)]
    out = bf._select_sample(facts, sample_size=2, ensure_entities=["Misha", "Elena", "Katya"])
    subjects = {f["subject"] for f in out}
    # all three key entities present despite sample_size=2
    assert {"Misha", "Elena", "Katya"} <= subjects


async def test_select_sample_ensure_matches_cross_script():
    facts = [_sfact("Миша"), _sfact("Other")]
    out = bf._select_sample(facts, sample_size=1, ensure_entities=["Misha"])
    assert any(f["subject"] == "Миша" for f in out)  # Cyrillic subject matched a Latin ensure-name


async def test_run_backfill_sample_has_trace_id_and_entity_type():
    misha = _Ent("Misha", "person")
    traces = [
        {
            "id": "trace-42",
            "content": "Misha is Canadian",
            "tm": {"facts": [_fact("citizenship", "Russian", subject_id=str(misha.id))]},
            "entities": [misha],
        }
    ]
    session = _db_ok_session()
    classifier = lambda content, facts: [{"slot": "citizenship", "durability": "durable"}]  # noqa: E731
    with patch.object(bf, "_fetch_fact_bearing_traces", AsyncMock(return_value=traces)):
        report = await bf.run_backfill(session, apply=False, slot_classifier=classifier)
    s = report["samples"][0]
    assert s["trace_id"] == "trace-42"
    assert s["entity_type"] == "person"
    assert s["subject"] == "Misha"
    assert s["slot"] == "citizenship"


async def test_run_backfill_ensures_key_entities_in_sample():
    misha, other = _Ent("Misha", "person"), _Ent("Other", "person")
    traces = [
        {
            "id": "t-other",
            "content": "x",
            "tm": {"facts": [_fact("a", "b", subject_id=str(other.id))]},
            "entities": [other],
        },
        {
            "id": "t-misha",
            "content": "y",
            "tm": {"facts": [_fact("citizenship", "Russian", subject_id=str(misha.id))]},
            "entities": [misha],
        },
    ]
    session = _db_ok_session()

    async def classifier(content, facts, ents):
        return [{"slot": "off_schema", "durability": "transient"} for _ in facts]

    with patch.object(bf, "_fetch_fact_bearing_traces", AsyncMock(return_value=traces)):
        # sample_size=1 but Misha must still be guaranteed in
        report = await bf.run_backfill(
            session, apply=False, sample_size=1, ensure_entities=["Misha"], llm_slot_classifier=classifier
        )
    assert any(s["subject"] == "Misha" for s in report["samples"])


# ── --force reclassify ──────────────────────────────────────────────────────


async def test_run_backfill_force_reclassifies_already_slotted():
    """A trace whose facts already carry a slot is skipped by default, but --force
    re-classifies it (overwriting slot/durability)."""
    misha = _Ent("Misha", "person")

    def make_traces():
        return [
            {
                "id": "t1",
                "content": "x",
                "tm": {
                    "schema_version": 3,
                    "facts": [_fact("citizenship", "Russian", slot="off_schema", subject_id=str(misha.id))],
                },
                "entities": [misha],
            }
        ]

    classifier = lambda content, facts: [{"slot": "citizenship", "durability": "durable"}]  # noqa: E731

    # default: idempotent skip (already slotted)
    traces = make_traces()
    session = _db_ok_session()
    with patch.object(bf, "_fetch_fact_bearing_traces", AsyncMock(return_value=traces)):
        report = await bf.run_backfill(session, apply=False, slot_classifier=classifier)
    assert report["counts"]["skipped_done"] == 1
    assert report["counts"]["classified"] == 0

    # --force: re-classified, off_schema overwritten to citizenship
    traces = make_traces()
    session = _db_ok_session()
    with patch.object(bf, "_fetch_fact_bearing_traces", AsyncMock(return_value=traces)):
        report = await bf.run_backfill(session, apply=False, force=True, slot_classifier=classifier)
    assert report["counts"]["skipped_done"] == 0
    assert report["counts"]["classified"] == 1
    assert traces[0]["tm"]["facts"][0]["slot"] == "citizenship"


# ── DB-315 halt guard (Codex #4+#5) ─────────────────────────────────────────


async def test_amain_halts_without_rewire_ack(monkeypatch):
    """The backfill RUN entrypoint refuses to run until it is rewired for DB-315
    (first_party_subject_ids + trace-level marker) and DB315_BACKFILL_REWIRED=1 is set —
    so it can never silently half-apply the fix. Raises before touching the DB."""
    monkeypatch.delenv("DB315_BACKFILL_REWIRED", raising=False)
    with pytest.raises(SystemExit, match="HALTED"):
        await bf._amain(apply=False, sample_size=1, ensure_entities=[], force=False)
