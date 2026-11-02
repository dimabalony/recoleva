"""INC 3 — whole-entity profile curator (Letta core/archival two-tier).

A cheap Haiku step that runs ON TOP of the deterministic ``aggregate_slot_profile``
(which stays pure). It curates the entity's accumulated structured facts into a
concise but richly DESCRIPTIVE profile card.

Two-tier memory (MemGPT/Letta core-vs-archival + Generative-Agents reflection):
  * the **CARD** (``canonical_fields``) = curated, descriptive *gist* — the compact
    "core memory" that is always in the prompt;
  * the **SUBSTRATE** (``canonical_slot_values``) = full detail, the "archival
    memory" queried on demand. Curation only rewrites the card text; it NEVER
    deletes from the substrate, which is returned UNCHANGED.

Curation groups related facts (supplements together, training together, health
together), abstracts granular logbook minutiae (exact measurements, increments,
timing windows, step-by-step protocols) into higher-level statements, and keeps the
most salient — biased toward descriptive inclusion over a terse list.

One Haiku call per entity covers ALL its facts (cost); the input is capped to the
top-priority facts (``canonical_slot_values`` is already ranked importance → recency
→ salience) so a fact-heavy entity can't blow up the prompt.

Graceful: any failure (provider, LLM, parse) keeps the deterministic card — the card
is never broken. Shadow-only — wired into the EntitySchemaProducer shadow path; live
profiles still come from Sonnet.
"""

from __future__ import annotations

from kai_memory.host_calls import get_model_roles

import logging
from typing import Any

from kai_memory.host_calls import utc_now
from kai_memory.slot_profile import SlotProfile

logger = logging.getLogger("app.services.memory.slot_summarizer")

_HAIKU_MODEL = "haiku-5.5"  # matches memory_extraction.HAIKU_MODEL (live L2 model)

# Input-length cap: curate the top-priority facts only (``canonical_slot_values`` is
# already ranked importance → recency → salience, and records within a slot by
# salience), so a fact-heavy entity can't blow up the one Haiku call.
_MAX_CURATION_FACTS = 50

# ONE principled instruction (no rule-list — avoids prompt-bloat / overfitting).
_CURATION_PROMPT = """You are writing one entity's profile card from its accumulated structured facts. \
Today is {today}. The full detail is preserved elsewhere; this card is the compact, always-visible gist.

Entity: {entity_name}

Facts (each: slot/field, value, observed date, durability):
{fact_block}

Write a concise but richly descriptive profile card — the durable, defining attributes you'd mention \
introducing this entity. Group related facts (supplements together, training together, health together). \
Abstract granular logbook minutiae (exact measurements, increments, timing windows, step-by-step protocols) \
into higher-level statements — keep descriptive texture, cut only the minutiae. Keep the most salient; favor \
a few descriptive lines over a terse list. State only what the facts support: never upgrade weak predicates \
(interest, research, considering) into identity, skill, or ownership claims.

Return ONLY a JSON object mapping each grouped attribute (a short snake_case key) to its descriptive phrase \
(a string). No markdown, no commentary."""


def _collect_facts(profile: SlotProfile) -> list[tuple[str, dict[str, Any]]]:
    """The entity's ``(slot, record)`` pairs, capped to the top ``_MAX_CURATION_FACTS``.

    Selected ROUND-ROBIN by record rank across slots — each slot's top fact before any
    slot's second, in the aggregator's slot-priority order — so a fact-heavy early slot
    can't starve later slots out of the "whole-entity" prompt. Within a slot records are
    already salience-ranked, so depth N is that slot's N-th most salient fact.
    """
    slots = list(profile.canonical_slot_values.items())
    out: list[tuple[str, dict[str, Any]]] = []
    depth = 0
    added = True
    while added and len(out) < _MAX_CURATION_FACTS:
        added = False
        for slot, recs in slots:
            if depth < len(recs):
                out.append((slot, recs[depth]))
                added = True
                if len(out) >= _MAX_CURATION_FACTS:
                    return out
        depth += 1
    return out


def _render_facts(facts: list[tuple[str, dict[str, Any]]]) -> str:
    lines: list[str] = []
    for slot, r in facts:
        observed = r.get("observed_at")
        observed = observed[:10] if isinstance(observed, str) else "unknown"
        field = r.get("field")
        label = f"{slot}/{field}" if field and field != slot else slot
        lines.append(f'- {label}: "{r.get("value")}" (observed={observed}, {r.get("durability")})')
    return "\n".join(lines)


async def _curate_via_haiku(
    facts: list[tuple[str, dict[str, Any]]],
    entity_name: str | None,
    provider: Any | None,
) -> dict[str, str]:
    """One Haiku call → curated ``{attribute: phrase}`` card. Returns ``{}`` on any
    failure (caller keeps the deterministic card). Imports are lazy so the LLM call is
    monkeypatchable."""
    from kai_memory.ports import get_settings
    from kai_memory.ports import call_llm_text, parse_json_object_response, get_provider_for_model

    prompt = _CURATION_PROMPT.format(
        today=utc_now().date().isoformat(),
        entity_name=entity_name or "(unknown)",
        fact_block=_render_facts(facts),
    )
    if provider is None:
        provider = get_provider_for_model(get_settings(), get_model_roles().auxiliary)
    raw = await call_llm_text(
        provider, prompt, max_tokens=1400, thinking_mode="disabled", task_name="inc3_profile_curation"
    )
    parsed = parse_json_object_response(raw or "")
    return {k: v for k, v in parsed.items() if isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip()}


async def curate_profile(
    profile: SlotProfile,
    *,
    entity_name: str | None = None,
    provider: Any | None = None,
) -> SlotProfile:
    """Curate the WHOLE entity into a concise but richly descriptive profile card.

    The CARD (``canonical_fields``) becomes the curated gist; the SUBSTRATE
    (``canonical_slot_values``) is returned UNCHANGED — full detail stays queryable
    (Letta core-vs-archival). One Haiku call per entity. Any failure keeps the
    deterministic card (never raises). Curation is type-agnostic — it groups across
    all of the entity's slots regardless of entity type.
    """
    try:
        facts = _collect_facts(profile)
        if not facts:
            return profile
        card = await _curate_via_haiku(facts, entity_name, provider)
        if not card:
            return profile
        # Substrate is preserved verbatim — only the card text is curated.
        return SlotProfile(canonical_fields=card, canonical_slot_values=profile.canonical_slot_values)
    except Exception:
        logger.warning("profile curation failed; keeping deterministic card", exc_info=True)
        return profile
