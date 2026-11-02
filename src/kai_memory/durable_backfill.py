"""INC 3 — one-time durable-attribute (slot + durability) BACKFILL tool (BUILD; run later).

Kai CLI: ``python -m app.services.memory.durable_backfill --help``.
Use the Kai namespace to install its host bindings before running this tool.

Re-classifies the facts on existing fact-bearing traces into the durable-attribute
schema: each fact gains a ``slot`` (a ``durable_schema`` key for the subject's
entity type, or ``"off_schema"``) and a ``durability`` tag
(``durable | transient | operational``). It does NOT change
``field/value/polarity/subject/subject_id`` — slot+durability are ADDITIVE — and
then bumps the trace's ``typed_metadata.schema_version`` to 3.

This is the backfill twin of the forward write-path
(``app/services/chat/memory_extraction.py``): it reuses the SAME slot catalog
(``render_schema_for_prompt``), the SAME durability rubric, and — critically — the
SAME degrade-not-reject validator (``_validate_and_default_fact_slots``) so the
backfilled facts are identical in shape to freshly-extracted ones. Mirrors the A3
``subject_backfill.py`` pattern (single live source of truth, injected LLM call,
dry-run-first).

Classification needs the §14 Haiku call for EVERY fact-bearing trace (unlike A3,
there is no deterministic single-link shortcut — a slot is a per-fact judgment).
The async classifier is injected and finalized at execution time; the sync
``slot_classifier`` is the test-injection seam.

SAFETY: dry-run by default (no writes); refuses unless ``current_database()='ai_workforce'``;
SELECT + UPDATE-``typed_metadata`` only; idempotent (skips traces whose facts already
carry a ``slot``). Execution is a prod write run under the user's hand AFTER the
sample-accuracy spot-check (the slot/durability mini-gate, §4). Unit-tested, not run
against prod here.
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

# Reuse the EXACT write-path validator so backfilled facts match freshly-extracted
# ones byte-for-byte (slot lower-casing + degrade-not-reject). Single source of
# truth — re-implementing it here would let the two paths drift.
from kai_memory.extraction.memory_extraction import _validate_and_default_fact_slots
from kai_memory.canonicalization import same_canonical_entity
from kai_memory.durable_schema import SLOT_DURABILITY_RUBRIC, render_schema_for_prompt

# A per-trace slot classifier: (trace_content, facts) -> list of {"slot","durability"}
# aligned to facts. Injected (real impl = the §14 Haiku call); the sync form is the
# test seam.
SlotClassifier = Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]]
# The §14 LLM classifier is async + also sees the candidate entities (for type-aware
# slot routing).
AsyncSlotClassifier = Callable[[str, list[dict[str, Any]], Sequence[Any]], Awaitable[list[dict[str, Any]]]]

_HAIKU_MODEL = "haiku-5.5"  # matches memory_extraction.HAIKU_MODEL (live L2 model)

# Embeds the SAME ``SLOT_DURABILITY_RUBRIC`` and slot catalog as the forward
# EXTRACTION_PROMPT (single source of truth in durable_schema.py) so backfill and
# forward classify by identical rules. ``{slot_durability_rubric}`` sits immediately
# above the catalog so its "listed below" reads correctly.
_CLASSIFIER_PROMPT = """You are re-classifying already-extracted facts about entities into a durable-attribute \
schema. For EACH fact below set two fields:

  {slot_durability_rubric}

Slot-mapping is the PRIMARY signal (bounded by the schema keys). Durability is SECONDARY; default to \
"transient" when unsure.

Durable-attribute slot catalog — valid slot keys per entity type:
{schema_catalog}

Memory: {content}

Facts (in order), each with its subject entity and that entity's type:
{facts}

Output ONLY a JSON array, in the SAME order as the facts, each item \
{{"field": "...", "value": "...", "slot": "<a catalog key or off_schema>", \
"durability": "durable|transient|operational"}}."""


def _entity_type_by_id(entities: Sequence[Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for e in entities or []:
        eid = getattr(e, "id", None)
        etype = getattr(e, "entity_type", None)
        if eid is not None and etype:
            out[str(eid)] = etype
    return out


async def haiku_slot_classifier(
    content: str,
    facts: list[dict[str, Any]],
    entities: Sequence[Any],
    *,
    provider: Any | None = None,
) -> list[dict[str, Any]]:
    """§14 — classify each fact's slot + durability via Haiku (build only; run under
    the user's hand).

    Returns ``{"slot","durability"}`` dicts aligned to ``facts`` (raw model strings;
    the caller runs them through ``_validate_and_default_fact_slots`` which defaults
    and degrades). Malformed items collapse to ``{"slot": None, "durability": None}``
    so the validator supplies safe defaults.
    """
    from kai_memory.ports import get_settings
    from kai_memory.ports import call_llm_text, parse_json_array_response, get_provider_for_model

    type_by_id = _entity_type_by_id(entities)
    fact_lines = "\n".join(
        f"{i}. subject={f.get('subject') or '(unknown)'} "
        f"(type={type_by_id.get(str(f.get('subject_id')), 'unknown')}) "
        f"field={f.get('field')} value={f.get('value')}"
        for i, f in enumerate(facts)
    )
    prompt = _CLASSIFIER_PROMPT.format(
        slot_durability_rubric=SLOT_DURABILITY_RUBRIC,
        schema_catalog=render_schema_for_prompt(),
        content=(content or "")[:2000],
        facts=fact_lines,
    )

    if provider is None:
        provider = get_provider_for_model(get_settings(), get_model_roles().auxiliary)
    raw = await call_llm_text(
        provider, prompt, max_tokens=2800, thinking_mode="disabled", task_name="inc3_backfill_slot"
    )
    parsed = parse_json_array_response(raw or "")
    out: list[dict[str, Any]] = []
    for i in range(len(facts)):
        item = parsed[i] if i < len(parsed) and isinstance(parsed[i], dict) else {}
        out.append({"slot": item.get("slot"), "durability": item.get("durability")})
    return out


def facts_need_classification(facts: Sequence[Any]) -> bool:
    """True if any dict fact still lacks a ``slot`` key (idempotency gate)."""
    return any(isinstance(f, dict) and "slot" not in f for f in facts)


def classify_trace_facts(
    content: str,
    facts: list[dict[str, Any]],
    entities: Sequence[Any],
    *,
    slot_classifier: SlotClassifier | None = None,
) -> str:
    """Classify every fact of one trace in place; return the method used.

    ``classified`` (the classifier named slot/durability, then the shared validator
    defaulted/degraded each), ``unresolved`` (no classifier this pass → facts left
    untouched), or ``empty`` (no dict facts).
    """
    dict_facts = [f for f in facts if isinstance(f, dict)]
    if not dict_facts:
        return "empty"

    if slot_classifier is None:
        return "unresolved"

    results = slot_classifier(content, dict_facts)
    if len(results) != len(dict_facts):
        raise ValueError(
            f"slot_classifier returned {len(results)} results for {len(dict_facts)} facts — "
            "misaligned classification would mis-tag facts"
        )
    for f, res in zip(dict_facts, results, strict=True):
        res = res if isinstance(res, dict) else {}
        f["slot"] = res.get("slot")
        f["durability"] = res.get("durability")

    # Identical degrade-not-reject pass as the write-path: defaults missing slot →
    # off_schema, bad durability → transient, and downgrades a real slot that lacks a
    # resolved subject_id or is invalid for the subject's entity type.
    _validate_and_default_fact_slots(dict_facts, list(entities))
    return "classified"


class _Ent:
    __slots__ = ("entity_type", "id", "name")

    def __init__(self, eid, name, etype):
        self.id = eid
        self.name = name
        self.entity_type = etype


async def _fetch_fact_bearing_traces(session: AsyncSession) -> list[dict[str, Any]]:
    """Read fact-bearing traces + their linked ACTIVE entities (any type, so a
    subject_id of any entity type resolves for slot validation)."""
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
            WHERE l.memory_id = :tid
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


def _select_sample(
    facts: list[dict[str, Any]],
    sample_size: int,
    ensure_entities: Sequence[str],
) -> list[dict[str, Any]]:
    """Pick a rateable gate sample. Every fact whose subject canonically matches an
    ``ensure_entities`` name (cross-script via #201) is ALWAYS included — even past
    ``sample_size`` — so the key person profiles (e.g. Misha/Elena/Katya) are
    guaranteed present; the remainder fills up to ``sample_size`` in collection
    (recency) order.
    """
    ensure = [e for e in (ensure_entities or []) if e and e.strip()]

    def _is_ensured(f: dict[str, Any]) -> bool:
        subj = f.get("subject")
        if not isinstance(subj, str) or not subj:
            return False
        return any(same_canonical_entity(subj, name) for name in ensure)

    ensured = [f for f in facts if _is_ensured(f)]
    rest = [f for f in facts if not _is_ensured(f)]
    fill = max(0, sample_size - len(ensured))
    return ensured + rest[:fill]


async def run_backfill(
    session: AsyncSession,
    *,
    apply: bool = False,
    sample_size: int = 20,
    ensure_entities: Sequence[str] | None = None,
    force: bool = False,
    slot_classifier: SlotClassifier | None = None,
    llm_slot_classifier: AsyncSlotClassifier | None = None,
) -> dict[str, Any]:
    """Classify slot+durability across all fact-bearing traces. dry-run unless ``apply``.

    Every fact-bearing trace needing classification is sent to ``llm_slot_classifier``
    (the §14 Haiku call, awaited here) when provided; ``slot_classifier`` (sync) is
    the test injection point. Returns a report (counts by method + a rateable sample,
    grouped-printable, with the ``ensure_entities`` subjects guaranteed present).

    ``force`` re-classifies EVERY fact-bearing trace even when its facts already
    carry a ``slot`` (overrides the idempotent skip) — used to re-apply tags after a
    prompt/schema change. Default stays idempotent.
    """
    db = (await session.execute(text("SELECT current_database()"))).scalar()
    if db != "ai_workforce":
        raise SystemExit(f"refusing to run against database {db!r} (expected 'ai_workforce')")

    traces = await _fetch_fact_bearing_traces(session)
    counts = {"classified": 0, "unresolved": 0, "empty": 0, "skipped_done": 0, "updated": 0}
    classified_facts: list[dict[str, Any]] = []

    for t in traces:
        facts = t["tm"].get("facts") or []
        if not force and not facts_need_classification(facts):
            counts["skipped_done"] += 1
            continue

        # Await the §14 slot/durability call, then feed the results to the (sync)
        # classifier via a trivial closure — keeps classify_trace_facts pure.
        per_trace_classifier = slot_classifier
        if per_trace_classifier is None and llm_slot_classifier is not None:
            dict_facts = [f for f in facts if isinstance(f, dict)]
            results = await llm_slot_classifier(t["content"], dict_facts, t["entities"])
            per_trace_classifier = lambda _c, _f, _r=results: _r  # noqa: E731

        method = classify_trace_facts(t["content"], facts, t["entities"], slot_classifier=per_trace_classifier)
        counts[method] = counts.get(method, 0) + 1
        t["tm"]["facts"] = facts
        t["tm"]["schema_version"] = 3

        if method == "classified":
            type_by_id = {str(getattr(e, "id", "")): getattr(e, "entity_type", None) for e in t["entities"]}
            name_by_id = {str(getattr(e, "id", "")): getattr(e, "name", None) for e in t["entities"]}
            for f in facts:
                if isinstance(f, dict):
                    sid = str(f.get("subject_id"))
                    classified_facts.append(
                        {
                            "trace_id": t["id"],
                            # Prefer the canonical entity name (via subject_id) over the
                            # raw fact subject — better for display + ensure-matching.
                            "subject": name_by_id.get(sid) or f.get("subject"),
                            "entity_type": type_by_id.get(sid, "unknown"),
                            "field": f.get("field"),
                            "value": f.get("value"),
                            "slot": f.get("slot"),
                            "durability": f.get("durability"),
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
    samples = _select_sample(classified_facts, sample_size, ensure_entities or [])
    return {"mode": "apply" if apply else "dry-run", "traces": len(traces), "counts": counts, "samples": samples}


def print_report(report: dict[str, Any]) -> None:
    print(f"\n=== INC 3 durable-attribute backfill — {report['mode']} ===")
    print(f"fact-bearing traces: {report['traces']}")
    print(f"counts: {report['counts']}")

    # Group the sample by entity so the rater sees each profile's facts together.
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for s in report["samples"]:
        groups.setdefault((s["subject"], s["entity_type"]), []).append(s)

    print(f"\n--- sample by entity (spot-check: right slot + durability?) [{len(report['samples'])} facts] ---")
    for (subject, etype), items in groups.items():
        print(f"\n  {subject!r} ({etype}) — {len(items)} fact(s):")
        for s in items:
            print(
                f"    [{s['trace_id']}] {s['field']}={s['value']!r} → slot={s['slot']!r} durability={s['durability']!r}"
            )


async def _amain(apply: bool, sample_size: int, ensure_entities: Sequence[str], force: bool) -> None:
    import os

    # HALT GUARD (DB-315, Codex #4+#5). This build-only tool shares the write-path
    # validator ``_validate_and_default_fact_slots``, which since DB-315:
    #   (1) gates family/health durability on ``first_party_subject_ids`` (the P-1 gate) —
    #       the backfill calls it WITHOUT that set, so every admin family/health fact would
    #       be stamped transient; and
    #   (2) expects the trace-level ``typed_metadata->>'durability'='durable'`` marker to be
    #       derived at store time — the backfill writes typed_metadata directly and never
    #       sets it, so backfilled durables would be unprotected by the retention floor.
    # Running as-is would silently half-apply the fix (the R-1 lesson). Wire BOTH (mirror
    # ExtractionService._first_party_subject_ids for #1 and store_trace's marker stamping
    # for #2), then set DB315_BACKFILL_REWIRED=1 to run.
    if os.getenv("DB315_BACKFILL_REWIRED") != "1":
        raise SystemExit(
            "durable_backfill is HALTED (DB-315): it needs first_party_subject_ids (P-1 "
            "family/health gate) AND trace-level durability-marker derivation wired before "
            "running — see the guard comment. After wiring both, set DB315_BACKFILL_REWIRED=1."
        )

    from kai_memory import ports

    if ports.SessionLocal is None:
        ports.init_engine()
    assert ports.SessionLocal is not None
    session = ports.SessionLocal()
    try:
        # §14 Haiku classifier wired for every fact-bearing trace. Dry-run first →
        # review the sample-accuracy report (the slot/durability mini-gate) → only
        # then --apply, under the user's hand.
        report = await run_backfill(
            session,
            apply=apply,
            sample_size=sample_size,
            ensure_entities=ensure_entities,
            force=force,
            llm_slot_classifier=haiku_slot_classifier,
        )
    finally:
        await session.close()
    print_report(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="INC 3 durable-attribute (slot + durability) backfill")
    parser.add_argument("--apply", action="store_true", help="Commit (default: dry-run, no writes)")
    parser.add_argument(
        "--sample", type=int, default=20, dest="sample_size", help="Max facts in the rateable gate sample."
    )
    parser.add_argument(
        "--ensure-entities",
        default="",
        help="Comma-separated subject names always included in the sample (e.g. 'Misha,Elena,Katya').",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-classify every fact-bearing trace even if its facts already carry a slot "
        "(override the idempotent skip — e.g. to re-apply tags after a prompt change).",
    )
    args = parser.parse_args()
    ensure = [s.strip() for s in args.ensure_entities.split(",") if s.strip()]
    asyncio.run(_amain(args.apply, args.sample_size, ensure, args.force))


if __name__ == "__main__":
    main()
