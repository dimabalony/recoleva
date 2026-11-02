"""Unit tests for the INC 3 A3 subject-attribution backfill tool."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai_memory import subject_backfill as bf

pytestmark = pytest.mark.asyncio


class _Ent:
    def __init__(self, name, etype="person"):
        self.id = uuid.uuid4()
        self.name = name
        self.entity_type = etype


def _fact(field, value, subject_id=None):
    f = {"field": field, "value": value, "polarity": "is"}
    if subject_id is not None:
        f["subject_id"] = subject_id
    return f


# ── pure attribution ────────────────────────────────────────────────────────


async def test_single_link_attributes_all_facts_deterministically():
    misha = _Ent("Misha")
    facts = [_fact("role", "founder"), _fact("city", "Corfu")]
    method = bf.attribute_trace_facts("Misha stuff", facts, [misha])
    assert method == "single_link"
    assert all(f["subject_id"] == str(misha.id) for f in facts)
    assert facts[0]["subject"] == "Misha"  # name back-filled


async def test_multi_link_uses_injected_assigner_and_resolves_by_canonicalization():
    misha = _Ent("Misha")
    katya = _Ent("Katya")
    facts = [_fact("role", "founder"), _fact("city", "Corfu")]
    # assigner names subjects (cross-script for fact 0)
    assigner = lambda content, fs: ["Миша", "Katya"]  # noqa: E731
    method = bf.attribute_trace_facts("multi", facts, [misha, katya], subject_assigner=assigner)
    assert method == "llm"
    assert facts[0]["subject_id"] == str(misha.id)  # 'Миша' → canonical Misha
    assert facts[1]["subject_id"] == str(katya.id)


async def test_multi_link_without_assigner_stays_unresolved():
    facts = [_fact("role", "founder")]
    method = bf.attribute_trace_facts("multi", facts, [_Ent("Misha"), _Ent("Katya")])
    assert method == "unresolved"
    assert facts[0]["subject_id"] is None


async def test_facts_need_attribution_idempotency():
    assert bf.facts_need_attribution([_fact("a", "b")]) is True
    assert bf.facts_need_attribution([_fact("a", "b", subject_id="x")]) is False


# ── runner ──────────────────────────────────────────────────────────────────


def _db_ok_session():
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(scalar=MagicMock(return_value="ai_workforce")))
    return session


async def test_run_backfill_dry_run_writes_nothing():
    misha = _Ent("Misha")
    traces = [
        {
            "id": "t1",
            "content": "Misha is a founder",
            "tm": {"schema_version": 1, "facts": [_fact("role", "founder")]},
            "entities": [misha],
        }
    ]
    session = _db_ok_session()
    with patch.object(bf, "_fetch_fact_bearing_traces", AsyncMock(return_value=traces)):
        report = await bf.run_backfill(session, apply=False)

    assert report["mode"] == "dry-run"
    assert report["counts"]["single_link"] == 1
    assert report["counts"]["updated"] == 0
    # no UPDATE executed, no commit
    update_calls = [c for c in session.execute.await_args_list if "UPDATE" in str(c.args[0])]
    assert update_calls == []
    session.commit.assert_not_called()
    assert report["samples"] and report["samples"][0]["subject_id"] == str(misha.id)


async def test_run_backfill_apply_updates_and_commits():
    traces = [{"id": "t1", "content": "x", "tm": {"facts": [_fact("role", "founder")]}, "entities": [_Ent("Misha")]}]
    session = _db_ok_session()
    with patch.object(bf, "_fetch_fact_bearing_traces", AsyncMock(return_value=traces)):
        report = await bf.run_backfill(session, apply=True)
    assert report["counts"]["updated"] == 1
    update_calls = [c for c in session.execute.await_args_list if "UPDATE" in str(c.args[0])]
    assert len(update_calls) == 1
    session.commit.assert_awaited_once()


async def test_run_backfill_skips_already_attributed_idempotent():
    traces = [
        {
            "id": "t1",
            "content": "x",
            "tm": {"facts": [_fact("role", "founder", subject_id="already")]},
            "entities": [_Ent("Misha")],
        }
    ]
    session = _db_ok_session()
    with patch.object(bf, "_fetch_fact_bearing_traces", AsyncMock(return_value=traces)):
        report = await bf.run_backfill(session, apply=True)
    assert report["counts"]["skipped_done"] == 1
    assert report["counts"]["updated"] == 0
    session.commit.assert_awaited_once()  # commit still called (no-op)


async def test_run_backfill_refuses_wrong_database():
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(scalar=MagicMock(return_value="staging_db")))
    with pytest.raises(SystemExit, match="staging_db"):
        await bf.run_backfill(session, apply=True)


async def test_multi_link_assigner_length_mismatch_raises():
    """Assigner returning fewer names than facts must fail fast, not silently
    drop tail facts."""
    facts = [_fact("a", "b"), _fact("c", "d")]
    assigner = lambda content, fs: ["OnlyOne"]  # noqa: E731  (1 name, 2 facts)
    with pytest.raises(ValueError, match="misaligned"):
        bf.attribute_trace_facts("x", facts, [_Ent("Misha"), _Ent("Katya")], subject_assigner=assigner)


# ── §14 Haiku assigner + multi-link runner wiring ───────────────────────────


async def test_haiku_subject_assigner_parses_names(monkeypatch):
    facts = [{"field": "role", "value": "founder"}, {"field": "city", "value": "Corfu"}]

    async def fake_call(*a, **k):
        return '[{"field":"role","value":"founder","subject":"Misha"},{"field":"city","value":"Corfu","subject":null}]'

    monkeypatch.setattr("kai_memory.ports.call_llm_text", fake_call)
    out = await bf.haiku_subject_assigner("Misha founded it in Corfu", facts, [_Ent("Misha")], provider=object())
    assert out == ["Misha", None]


async def test_run_backfill_multi_link_uses_llm_assigner_and_resolves():
    e1, e2 = _Ent("Misha"), _Ent("Katya")
    traces = [{"id": "t1", "content": "x", "tm": {"facts": [_fact("role", "founder")]}, "entities": [e1, e2]}]
    session = _db_ok_session()

    async def assigner(content, facts, cands):
        return ["Миша"]  # cross-script → resolves to e1

    with patch.object(bf, "_fetch_fact_bearing_traces", AsyncMock(return_value=traces)):
        report = await bf.run_backfill(session, apply=False, llm_subject_assigner=assigner)

    assert report["counts"]["llm"] == 1
    assert traces[0]["tm"]["facts"][0]["subject_id"] == str(e1.id)


async def test_run_backfill_zero_link_does_not_use_llm_assigner():
    """A trace linking ZERO person/org entities must not call the LLM assigner
    (no candidates → can't resolve) and stays unresolved."""
    called = False

    async def assigner(content, facts, cands):
        nonlocal called
        called = True
        return [None]

    traces = [{"id": "t1", "content": "x", "tm": {"facts": [_fact("role", "founder")]}, "entities": []}]
    session = _db_ok_session()
    with patch.object(bf, "_fetch_fact_bearing_traces", AsyncMock(return_value=traces)):
        report = await bf.run_backfill(session, apply=False, llm_subject_assigner=assigner)

    assert called is False
    assert report["counts"]["unresolved"] == 1
