"""INC 3 A3 — one-time subject-attribution BACKFILL tool (BUILD; run later).

Kai CLI: ``python -m app.services.memory.subject_backfill --help``.
Use the Kai namespace to install its host bindings before running this tool.

Re-attributes subjects on the existing fact-bearing traces (their facts predate
A3 and carry no subject). It does NOT change field/value/polarity — it only adds
a resolved ``subject_id`` (+ a raw ``subject`` name) per fact, then bumps the
trace's ``typed_metadata.schema_version`` to 2.

Attribution reuses the SAME live attributor (``resolve_subject_id`` + #201
canonicalization):
  - single-link traces (exactly one linked active person/org entity) → every
    fact attributes to it, deterministically, no LLM (the §3 ~6.8% subset);
  - multi-link traces → an injected per-fact ``subject_assigner`` (a small LLM
    call, finalized at execution time) names each fact's subject, which is then
    resolved to an id over the trace's candidates; absent an assigner they stay
    unresolved (a later pass).

SAFETY: dry-run by default (no writes); asserts ``current_database()='ai_workforce'``;
SELECT + UPDATE-``typed_metadata`` only; idempotent (skips already-attributed
traces). Execution is a prod write run under the user's hand AFTER the
sample-accuracy spot-check (the A3 mini-gate). This module is unit-tested, not
run against prod here.
"""

from __future__ import annotations

from kai_memory.host_calls import get_model_roles

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.subject_attribution import candidates_from_entities, resolve_subject_id

# A per-fact subject assigner: (trace_content, facts) -> list[str|None] aligned
# to facts. Injected (real impl = the §14 Haiku call); None ⇒ skip multi-link
# traces this pass.
SubjectAssigner = Callable[[str, list[dict[str, Any]]], list[str | None]]
# The §14 LLM assigner is async + also sees the candidate entities.
AsyncSubjectAssigner = Callable[[str, list[dict[str, Any]], Sequence[Any]], Awaitable[list[str | None]]]

_HAIKU_MODEL = "haiku-5.5"  # matches memory_extraction.HAIKU_MODEL (live A3 model)

_ASSIGNER_PROMPT = """Below is a memory and the entities it mentions. For each fact, name the single \
entity (from the listed candidates) the fact is about, or null if it's about none of them. \
Never guess — return null when unsure.

Memory: {content}

Candidate entities:
{candidates}

Facts (in order):
{facts}

Output ONLY a JSON array, in the SAME order as the facts, each item \
{{"field": "...", "value": "...", "subject": "<one candidate name, or null>"}}."""


async def haiku_subject_assigner(
    content: str,
    facts: list[dict[str, Any]],
    candidates: Sequence[Any],
    *,
    provider: Any | None = None,
) -> list[str | None]:
    """§14 — name each fact's subject via Haiku (build only; run under user's hand).

    Returns subject NAMES aligned to ``facts`` (None when the model abstains).
    The caller resolves names → canonical ids via ``resolve_subject_id`` (which
    keeps the precision-first >1-match-or-null semantics), so this only NAMES.
    """
    from kai_memory.ports import get_settings
    from kai_memory.ports import call_llm_text, parse_json_array_response, get_provider_for_model

    cand_lines = (
        "\n".join(f"- {getattr(c, 'name', None)} ({getattr(c, 'entity_type', None)})" for c in candidates) or "(none)"
    )
    fact_lines = "\n".join(f"{i}. {f.get('field')}={f.get('value')}" for i, f in enumerate(facts))
    prompt = _ASSIGNER_PROMPT.format(content=(content or "")[:2000], candidates=cand_lines, facts=fact_lines)

    if provider is None:
        provider = get_provider_for_model(get_settings(), get_model_roles().auxiliary)
    raw = await call_llm_text(
        provider, prompt, max_tokens=1400, thinking_mode="disabled", task_name="inc3_backfill_subject"
    )
    parsed = parse_json_array_response(raw or "")
    out: list[str | None] = []
    for i in range(len(facts)):
        item = parsed[i] if i < len(parsed) and isinstance(parsed[i], dict) else {}
        subj = item.get("subject")
        # Guard malformed LLM output: only forward a non-empty STRING (resolve_subject_id
        # calls .strip()); numbers/dicts/lists → None.
        out.append(subj if isinstance(subj, str) and subj.strip() else None)
    return out


def facts_need_attribution(facts: Sequence[Any]) -> bool:
    """True if any dict fact still lacks a resolved subject_id (idempotency)."""
    return any(isinstance(f, dict) and not f.get("subject_id") for f in facts)


def attribute_trace_facts(
    content: str,
    facts: list[dict[str, Any]],
    person_org_entities: Sequence[Any],
    *,
    subject_assigner: SubjectAssigner | None = None,
) -> str:
    """Attribute every fact of one trace in place; return the method used.

    ``single_link`` (one person/org → all facts to it), ``llm`` (assigner named
    subjects, resolved by canonicalization), or ``unresolved`` (left for later).
    """
    dict_facts = [f for f in facts if isinstance(f, dict)]
    if not dict_facts:
        return "empty"

    if len(person_org_entities) == 1:
        sid = str(person_org_entities[0].id)
        name = getattr(person_org_entities[0], "name", None)
        for f in dict_facts:
            f["subject_id"] = sid
            f.setdefault("subject", name)
        return "single_link"

    if subject_assigner is not None:
        candidates = candidates_from_entities(person_org_entities)
        names = subject_assigner(content, dict_facts)
        if len(names) != len(dict_facts):
            raise ValueError(
                f"subject_assigner returned {len(names)} names for {len(dict_facts)} facts — "
                "misaligned attribution would silently drop facts"
            )
        for f, subj in zip(dict_facts, names, strict=True):
            if subj:
                f["subject"] = subj
                rid = resolve_subject_id(subj, candidates)
                f["subject_id"] = str(rid) if rid is not None else None
            else:
                f.setdefault("subject_id", None)
        return "llm"

    for f in dict_facts:
        f.setdefault("subject_id", None)
    return "unresolved"


async def _fetch_fact_bearing_traces(session: AsyncSession) -> list[dict[str, Any]]:
    """Read fact-bearing traces + their linked ACTIVE person/org entities."""
    rows = (
        (
            await session.execute(
                text("""
        SELECT mt.id::text AS id, mt.content, mt.typed_metadata
        FROM memory_traces mt
        WHERE jsonb_typeof(mt.typed_metadata->'facts') = 'array'
          AND jsonb_array_length(mt.typed_metadata->'facts') > 0
        ORDER BY mt.created_at
    """)
            )
        )
        .mappings()
        .all()
    )
    out = []
    for r in rows:
        ents = (
            (
                await session.execute(
                    text("""
            SELECT e.id::text AS id, e.name, e.entity_type
            FROM memory_entity_links l
            JOIN memory_entities e ON e.id = l.entity_id AND e.merged_into_id IS NULL
            WHERE l.memory_id = :tid AND e.entity_type IN ('person','organization')
        """),
                    {"tid": r["id"]},
                )
            )
            .mappings()
            .all()
        )
        out.append(
            {
                "id": r["id"],
                "content": r["content"],
                "tm": dict(r["typed_metadata"] or {}),
                "entities": [_Ent(e["id"], e["name"], e["entity_type"]) for e in ents],
            }
        )
    return out


class _Ent:
    __slots__ = ("entity_type", "id", "name")

    def __init__(self, eid, name, etype):
        self.id = eid
        self.name = name
        self.entity_type = etype


async def run_backfill(
    session: AsyncSession,
    *,
    apply: bool = False,
    sample_size: int = 20,
    subject_assigner: SubjectAssigner | None = None,
    llm_subject_assigner: AsyncSubjectAssigner | None = None,
) -> dict[str, Any]:
    """Attribute subjects across all fact-bearing traces. dry-run unless ``apply``.

    Single-link traces attribute deterministically. Multi-link traces use
    ``llm_subject_assigner`` (the §14 Haiku call, awaited here) when provided,
    else stay unresolved. ``subject_assigner`` (sync) is the test injection point.
    Returns a report (counts by method + a sample for the accuracy spot-check).
    """
    db = (await session.execute(text("SELECT current_database()"))).scalar()
    if db != "ai_workforce":
        raise SystemExit(f"refusing to run against database {db!r} (expected 'ai_workforce')")

    traces = await _fetch_fact_bearing_traces(session)
    counts = {"single_link": 0, "llm": 0, "unresolved": 0, "empty": 0, "skipped_done": 0, "updated": 0}
    samples: list[dict[str, Any]] = []

    for t in traces:
        facts = t["tm"].get("facts") or []
        if not facts_need_attribution(facts):
            counts["skipped_done"] += 1
            continue
        # Multi-link + an LLM assigner: await the §14 names, then feed them to the
        # (sync) attributor via a trivial closure — keeps attribute_trace_facts pure.
        per_trace_assigner = subject_assigner
        if per_trace_assigner is None and llm_subject_assigner is not None and len(t["entities"]) > 1:
            dict_facts = [f for f in facts if isinstance(f, dict)]
            names = await llm_subject_assigner(t["content"], dict_facts, t["entities"])
            per_trace_assigner = lambda _c, _f, _names=names: _names  # noqa: E731
        method = attribute_trace_facts(t["content"], facts, t["entities"], subject_assigner=per_trace_assigner)
        counts[method] = counts.get(method, 0) + 1
        t["tm"]["facts"] = facts
        t["tm"]["schema_version"] = 2

        for f in facts:
            if isinstance(f, dict) and len(samples) < sample_size:
                samples.append(
                    {
                        "trace": (t["content"] or "")[:160],
                        "field": f.get("field"),
                        "value": f.get("value"),
                        "subject": f.get("subject"),
                        "subject_id": f.get("subject_id"),
                        "method": method,
                    }
                )

        if apply:
            await session.execute(
                text("UPDATE memory_traces SET typed_metadata = :tm WHERE id = :id"),
                {"tm": json.dumps(t["tm"]), "id": t["id"]},
            )
            counts["updated"] += 1

    if apply:
        await session.commit()
    return {"mode": "apply" if apply else "dry-run", "traces": len(traces), "counts": counts, "samples": samples}


def print_report(report: dict[str, Any]) -> None:
    print(f"\n=== A3 subject backfill — {report['mode']} ===")
    print(f"fact-bearing traces: {report['traces']}")
    print(f"counts: {report['counts']}")
    print(f"\n--- sample (spot-check: is the assigned subject right for the fact?) [{len(report['samples'])}] ---")
    for s in report["samples"]:
        print(f"  [{s['method']}] {s['field']}={s['value']!r} → subject={s['subject']!r} id={s['subject_id']}")
        print(f"     trace: {s['trace']!r}")


async def _amain(apply: bool, sample_size: int) -> None:
    from kai_memory import ports

    if ports.SessionLocal is None:
        ports.init_engine()
    assert ports.SessionLocal is not None
    session = ports.SessionLocal()
    try:
        # §14 Haiku assigner wired for multi-link traces; single-link attribute
        # deterministically. Dry-run first → review the sample-accuracy report
        # (the A3 mini-gate) → only then --apply, under the user's hand.
        report = await run_backfill(
            session, apply=apply, sample_size=sample_size, llm_subject_assigner=haiku_subject_assigner
        )
    finally:
        await session.close()
    print_report(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="INC 3 A3 subject-attribution backfill")
    parser.add_argument("--apply", action="store_true", help="Commit (default: dry-run, no writes)")
    parser.add_argument("--sample-size", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(_amain(args.apply, args.sample_size))


if __name__ == "__main__":
    main()
