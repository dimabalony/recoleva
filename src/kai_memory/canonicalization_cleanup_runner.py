"""Runner CLI for the one-time canonicalization data-cleanup.

Reads a PLAN JSON at runtime (no PII in git) and executes it against the DB:

    {
      "backfill": true,
      "merges": [
        {"survivor_id": "<uuid>", "absorbed_ids": ["<uuid>", ...], "reason": "..."}
      ],
      "close_edges": ["<relationship-uuid>", ...]
    }

Two modes:
  --dry-run (DEFAULT): execute the whole plan inside ONE transaction, print the
    exact per-cluster plan (survivor + absorbed names/ids, # relationships and
    # entity_links that would re-point/dedup, merge/skip outcomes, edge closes),
    then ROLL BACK. Zero commit. This output is the human checkpoint.
  --apply: same execution, but COMMIT at the end.

Usage (via SSM in the prod container):
    python -m app.services.memory.canonicalization_cleanup_runner \
        --plan /tmp/plan.json [--apply]

Assert the active DB before --apply (current_database()='ai_workforce').
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid as _uuid

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory import ports
from kai_memory.storage.models.entity_relationship import EntityRelationship
from kai_memory.storage.models.memory_entity import MemoryEntity
from kai_memory.storage.models.memory_entity_link import MemoryEntityLink
from kai_memory.canonicalization_cleanup import (
    backfill_canonical_key,
    close_relationship,
    merge_into_survivor,
)


def load_plan(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data: dict = json.load(f)
    return data


async def _absorbed_footprint(session: AsyncSession, absorbed_id: _uuid.UUID) -> dict:
    """Count active relationships + entity_links touching an absorbed entity."""
    active_rels = (
        await session.execute(
            select(func.count())
            .select_from(EntityRelationship)
            .where(
                EntityRelationship.valid_until.is_(None),
                (EntityRelationship.from_entity_id == absorbed_id) | (EntityRelationship.to_entity_id == absorbed_id),
            )
        )
    ).scalar() or 0
    links = (
        await session.execute(
            select(func.count()).select_from(MemoryEntityLink).where(MemoryEntityLink.entity_id == absorbed_id)
        )
    ).scalar() or 0
    return {"active_rels": int(active_rels), "links": int(links)}


async def execute_plan(session: AsyncSession, plan: dict, *, dry_run: bool) -> dict:
    """Run backfill + merges + edge-closes, returning a structured report.

    Does NOT commit or roll back — ``run_plan`` owns the transaction outcome.
    """
    report: dict = {"dry_run": dry_run, "backfill_count": None, "merges": [], "edges": []}

    if plan.get("backfill"):
        report["backfill_count"] = await backfill_canonical_key(session)

    for entry in plan.get("merges", []):
        survivor_id = _uuid.UUID(entry["survivor_id"])
        absorbed_ids = [_uuid.UUID(a) for a in entry["absorbed_ids"]]
        survivor = await session.get(MemoryEntity, survivor_id)

        absorbed_info = []
        for aid in absorbed_ids:
            absorbed = await session.get(MemoryEntity, aid)
            footprint = await _absorbed_footprint(session, aid)
            absorbed_info.append(
                {
                    "id": str(aid),
                    "name": absorbed.name if absorbed else None,
                    **footprint,
                }
            )

        result = await merge_into_survivor(session, survivor_id, absorbed_ids, reason=entry["reason"])
        report["merges"].append(
            {
                "survivor_id": str(survivor_id),
                "survivor_name": survivor.name if survivor else None,
                "absorbed": absorbed_info,
                "merged": [str(x) for x in result["merged"]],
                "skipped_tg_conflict": [str(x) for x in result["skipped_tg_conflict"]],
                "skipped_missing": [str(x) for x in result["skipped_missing"]],
            }
        )

    for edge_id in plan.get("close_edges", []):
        eid = _uuid.UUID(edge_id)
        rel = await session.get(EntityRelationship, eid)
        closed = await close_relationship(session, eid)
        report["edges"].append(
            {
                "id": edge_id,
                "type": rel.relationship_type if rel else None,
                "closed": closed,
            }
        )

    return report


async def run_plan(session: AsyncSession, plan: dict, *, apply: bool) -> dict:
    """Execute the plan and finalize the transaction: commit iff ``apply``."""
    report = await execute_plan(session, plan, dry_run=not apply)
    if apply:
        await session.commit()
    else:
        await session.rollback()
    return report


def print_report(report: dict) -> None:
    mode = "APPLY (committed)" if not report["dry_run"] else "DRY-RUN (rolled back)"
    print(f"\n=== canonicalization cleanup — {mode} ===")
    print(f"backfill_count: {report['backfill_count']}")
    for m in report["merges"]:
        print(f"\nMERGE → survivor {m['survivor_name']!r} ({m['survivor_id']})")
        for a in m["absorbed"]:
            print(f"   ← {a['name']!r} ({a['id']})  rels={a['active_rels']} links={a['links']}")
        print(f"   merged={m['merged']}")
        if m["skipped_tg_conflict"]:
            print(f"   ⚠️ SKIPPED (tg-conflict)={m['skipped_tg_conflict']}")
        if m["skipped_missing"]:
            print(f"   ⚠️ SKIPPED (missing)={m['skipped_missing']}")
    for e in report["edges"]:
        print(f"\nCLOSE EDGE {e['id']} [{e['type']}] closed={e['closed']}")
    print("\n=== end ===")


async def _amain(plan_path: str, apply: bool) -> None:
    # Read the live module attribute — `ports.SessionLocal` is None until
    # init_engine() sets it; binding it locally at import time would freeze None.
    if ports.SessionLocal is None:
        ports.init_engine()
    assert ports.SessionLocal is not None
    plan = load_plan(plan_path)
    session = ports.SessionLocal()
    try:
        if apply:
            # Safety gate: never COMMIT cleanup mutations against the wrong DB.
            db_name = (await session.execute(text("SELECT current_database()"))).scalar_one()
            if db_name != "ai_workforce":
                raise SystemExit(f"refusing --apply against database {db_name!r} (expected 'ai_workforce')")
        report = await run_plan(session, plan, apply=apply)
    finally:
        await session.close()
    print_report(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonicalization data-cleanup runner")
    parser.add_argument("--plan", required=True, help="Path to plan JSON")
    parser.add_argument("--apply", action="store_true", help="Commit (default: dry-run + rollback)")
    args = parser.parse_args()
    asyncio.run(_amain(args.plan, args.apply))


if __name__ == "__main__":
    main()
