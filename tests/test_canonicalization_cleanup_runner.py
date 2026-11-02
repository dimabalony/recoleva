"""Unit tests for the canonicalization cleanup RUNNER.

The runner reads a plan JSON ({backfill, merges[], close_edges[]}), executes it
against a session, and — critically — in dry-run mode ROLLS BACK (never commits).
DB-touching helpers are mocked here; the real transaction/rollback behavior is
covered by the integration test.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from kai_memory import canonicalization_cleanup_runner as runner

pytestmark = pytest.mark.asyncio


# ── transaction safety gate ─────────────────────────────────────────────────


async def test_dry_run_rolls_back_and_never_commits(monkeypatch):
    monkeypatch.setattr(runner, "execute_plan", AsyncMock(return_value={"dry_run": True}))
    session = AsyncMock()

    report = await runner.run_plan(session, {"merges": []}, apply=False)

    session.rollback.assert_awaited_once()
    session.commit.assert_not_called()
    assert report["dry_run"] is True


async def test_apply_commits_and_never_rolls_back(monkeypatch):
    monkeypatch.setattr(runner, "execute_plan", AsyncMock(return_value={"dry_run": False}))
    session = AsyncMock()

    await runner.run_plan(session, {"merges": []}, apply=True)

    session.commit.assert_awaited_once()
    session.rollback.assert_not_called()


# ── plan orchestration ──────────────────────────────────────────────────────


async def test_execute_plan_runs_backfill_only_when_flagged(monkeypatch):
    backfill = AsyncMock(return_value=42)
    monkeypatch.setattr(runner, "backfill_canonical_key", backfill)
    monkeypatch.setattr(runner, "merge_into_survivor", AsyncMock())
    monkeypatch.setattr(runner, "close_relationship", AsyncMock())
    session = AsyncMock()

    on = await runner.execute_plan(session, {"backfill": True}, dry_run=True)
    assert on["backfill_count"] == 42
    backfill.assert_awaited_once()

    backfill.reset_mock()
    off = await runner.execute_plan(session, {"backfill": False}, dry_run=True)
    assert off["backfill_count"] is None
    backfill.assert_not_awaited()


async def test_execute_plan_merges_each_entry_and_reports_names_and_footprint(monkeypatch):
    sid = str(uuid.uuid4())
    aid = str(uuid.uuid4())
    survivor = MagicMock(name="survivor")
    survivor.name = "Kai"
    absorbed = MagicMock()
    absorbed.name = "Kai (person)"
    by_id = {uuid.UUID(sid): survivor, uuid.UUID(aid): absorbed}

    session = AsyncMock()
    session.get = AsyncMock(side_effect=lambda model, _id: by_id.get(_id))
    monkeypatch.setattr(runner, "_absorbed_footprint", AsyncMock(return_value={"active_rels": 4, "links": 134}))
    merge = AsyncMock(return_value={"merged": [uuid.UUID(aid)], "skipped_tg_conflict": [], "skipped_missing": []})
    monkeypatch.setattr(runner, "merge_into_survivor", merge)
    monkeypatch.setattr(runner, "backfill_canonical_key", AsyncMock())
    monkeypatch.setattr(runner, "close_relationship", AsyncMock())

    plan = {"merges": [{"survivor_id": sid, "absorbed_ids": [aid], "reason": "cleanup"}]}
    report = await runner.execute_plan(session, plan, dry_run=True)

    m = report["merges"][0]
    assert m["survivor_name"] == "Kai"
    assert m["absorbed"][0]["name"] == "Kai (person)"
    assert m["absorbed"][0]["active_rels"] == 4
    assert m["absorbed"][0]["links"] == 134
    merge.assert_awaited_once()
    # survivor passed as UUID, reason threaded through
    args, kwargs = merge.call_args
    assert args[1] == uuid.UUID(sid)
    assert kwargs["reason"] == "cleanup"


async def test_execute_plan_closes_each_edge(monkeypatch):
    eid = str(uuid.uuid4())
    rel = MagicMock()
    rel.relationship_type = "manager_of"
    session = AsyncMock()
    session.get = AsyncMock(return_value=rel)
    close = AsyncMock(return_value=True)
    monkeypatch.setattr(runner, "close_relationship", close)
    monkeypatch.setattr(runner, "merge_into_survivor", AsyncMock())
    monkeypatch.setattr(runner, "backfill_canonical_key", AsyncMock())

    report = await runner.execute_plan(session, {"close_edges": [eid]}, dry_run=True)

    e = report["edges"][0]
    assert e["id"] == eid
    assert e["type"] == "manager_of"
    assert e["closed"] is True
    close.assert_awaited_once_with(session, uuid.UUID(eid))


# ── CLI session wiring (import-binding regression) ──────────────────────────


async def test_amain_uses_live_session_factory_after_init(monkeypatch):
    """Regression: _amain must read kai_memory.ports.SessionLocal (set by init_engine at
    runtime), NOT a name bound to None at import time. `from kai_memory.ports import
    SessionLocal` captures None; after init_engine() the local stays None and
    `SessionLocal()` blows up. The fix reads the module attribute."""
    import kai_memory.ports as app_db

    fake_session = AsyncMock()
    monkeypatch.setattr(app_db, "SessionLocal", None)
    monkeypatch.setattr(app_db, "init_engine", lambda: setattr(app_db, "SessionLocal", lambda: fake_session))
    monkeypatch.setattr(
        runner, "run_plan", AsyncMock(return_value={"dry_run": True, "backfill_count": None, "merges": [], "edges": []})
    )
    monkeypatch.setattr(runner, "load_plan", lambda p: {"merges": []})
    monkeypatch.setattr(runner, "print_report", lambda r: None)

    await runner._amain("/tmp/whatever.json", apply=False)

    fake_session.close.assert_awaited_once()


async def test_amain_apply_refuses_wrong_database(monkeypatch):
    """--apply must abort if current_database() != 'ai_workforce' (safety gate)."""
    import kai_memory.ports as app_db

    fake_session = AsyncMock()
    fake_session.execute = AsyncMock(return_value=MagicMock(scalar_one=MagicMock(return_value="staging_db")))
    monkeypatch.setattr(app_db, "SessionLocal", lambda: fake_session)
    run = AsyncMock()
    monkeypatch.setattr(runner, "run_plan", run)
    monkeypatch.setattr(runner, "load_plan", lambda p: {"merges": []})

    with pytest.raises(SystemExit, match="staging_db"):
        await runner._amain("/tmp/whatever.json", apply=True)

    run.assert_not_awaited()  # never executed the plan
    fake_session.close.assert_awaited_once()  # session still cleaned up


# ── plan loading ────────────────────────────────────────────────────────────


async def test_load_plan_reads_json(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({"backfill": True, "merges": [], "close_edges": []}))
    plan = runner.load_plan(str(p))
    assert plan["backfill"] is True
    assert plan["merges"] == []
