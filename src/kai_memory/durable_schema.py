"""Durable-attribute schema for entity profiles (INC 3 redesign, design v2 §3.1).

The schema governs ONLY the profile-summary card (``canonical_fields``) — NOT
what we remember. Episodic ``memory_traces`` keep every fact with full context;
this schema is the curated, durable, *queryable* summary layer on top.

How it is used:
  * **Write path** (§3.2): schema-constrained Haiku extraction routes each fact
    into one of an entity type's slots, or marks it ``off_schema`` with a
    ``durability`` tag. Noise (operational telemetry, point-in-time episodic)
    has no slot, so it never reaches the card — clean by construction.
  * **Quarantine + promotion** (§3.6): off-schema ``durable`` facts are kept and
    queryable; a recurring one is promoted into this schema after review. So a
    missing slot loses nothing — it just isn't on the card until promotion. We
    therefore SEED the slots the data already proves durable (see the §5.0
    sizing report) rather than wait for the promotion cycle to rediscover them.

Cardinality:
  * ``SINGLE`` — one current value; the newest observation wins. History is kept
    in the structured store (for the timeline); only the current value renders.
  * ``MULTI`` — accumulate distinct values (languages, relationships, …).

``importance`` ranks slots on the card (1-10, higher first) under the size cap —
NEVER alphabetical (§3.4; the Generative-Agents / Letta lesson).

Person slots were curated with the user against the live §5.0 field-frequency
data (2026-06-20); org/place/animal are seeded from the same data. project/agent
gained generic slot groups (2026-07-02 non-person coverage) so recurring projects
and agents can clear the deterministic-card thin gate; topic/object/event/group
stay intentionally fieldless.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Cardinality(StrEnum):
    SINGLE = "single"  # newest-wins
    MULTI = "multi"  # accumulate distinct values


@dataclass(frozen=True)
class Slot:
    """One durable-attribute slot. ``description`` doubles as the extraction
    guidance surfaced to Haiku, so write it as an instruction, not a label."""

    key: str
    description: str
    cardinality: Cardinality
    importance: int  # 1-10, higher ranks earlier on the profile card
    # MULTI collection slots whose ``field`` carries essential context (height,
    # diagnosis, …). The aggregator renders these as "field: value"; bare-value
    # MULTI slots (languages, citizenship, allergies) and all SINGLE slots leave
    # this False. See slot_profile.aggregate_slot_profile.
    labeled: bool = False


# --- PERSON -----------------------------------------------------------------
# Curated with the user (2026-06-20) against live field frequencies. The bulk of
# person facts in prod are operational telemetry or point-in-time episodic - by
# design those map to NO slot and stay episodic.
_PERSON: list[Slot] = [
    Slot(
        "full_name",
        "The person's full or more-complete name — a surname / patronymic added to "
        "a known first name (e.g. 'Mikhail Petrov'). Keep the ORIGINAL script "
        "(never transliterate) and record only when it adds a name part beyond what "
        "is already known. Display-only: it labels the profile card and is NEVER "
        "used to match, merge, or dedup entities.",
        Cardinality.SINGLE,
        10,
    ),
    Slot(
        "current_location",
        "Where the person currently lives or is based (city and/or country). "
        "A move replaces it; the newest mention wins.",
        Cardinality.SINGLE,
        9,
    ),
    Slot(
        "key_relationships",
        "Durable relationships to other people or pets: spouse, partner, "
        "children, parents, siblings, close friends, pets. "
        "Format each as 'relation: name' (e.g. 'wife: Katya').",
        Cardinality.MULTI,
        9,
        labeled=True,
    ),
    Slot(
        "chronic_health_conditions",
        "Ongoing / chronic medical conditions or diagnoses (e.g. Gilbert's "
        "syndrome). NOT one-off symptoms, lab values, or visit findings — those "
        "are episodic.",
        Cardinality.MULTI,
        8,
        labeled=True,
    ),
    Slot(
        "allergies",
        "Allergies and intolerances (food, drug, environmental). Safety-relevant and durable.",
        Cardinality.MULTI,
        8,
    ),
    Slot(
        "date_of_birth",
        "Date of birth, or age with the date it was stated (so it can be aged forward).",
        Cardinality.SINGLE,
        7,
    ),
    Slot(
        "occupation",
        "Current job, profession, primary role, or field of expertise (e.g. "
        "'cosmetologist', 'software engineer'). A career change replaces it.",
        Cardinality.SINGLE,
        7,
    ),
    Slot(
        "employer",
        "Current employer / company. Earlier employers are kept as history; the newest is current.",
        Cardinality.SINGLE,
        6,
    ),
    Slot(
        "languages",
        "Languages the person speaks.",
        Cardinality.MULTI,
        6,
    ),
    Slot(
        "citizenship",
        "Nationality / passport country. May be more than one.",
        Cardinality.MULTI,
        6,
    ),
    Slot(
        "hometown",
        "Where the person is originally from (birthplace / city of origin).",
        Cardinality.SINGLE,
        5,
    ),
    Slot(
        "stable_preferences",
        "The person's own durable tastes, values, and standing preferences (style, "
        "food, brands, communication tone). NOT a one-off choice, and NOT attributes "
        "of a product or thing they're discussing.",
        Cardinality.MULTI,
        5,
        labeled=True,
    ),
    Slot(
        "interests",
        "Standing interests and active research topics — things the person "
        "actively explores, researches, or considers (a hobby domain, a purchase "
        "being weighed). Encode the WEAK predicate in the value ('researching X', "
        "'interested in Y') — never as a skill, ownership, or identity claim.",
        Cardinality.MULTI,
        5,
    ),
    Slot(
        "physical_attributes",
        "Slowly-changing physical attributes useful for shopping/fit: height, "
        "clothing sizes, body measurements. Format each as 'attribute: value' "
        "(e.g. 'height: 170cm').",
        Cardinality.MULTI,
        4,
        labeled=True,
    ),
]

# --- ORGANIZATION -----------------------------------------------------------
_ORGANIZATION: list[Slot] = [
    Slot(
        "relationship",
        "The organization's standing relationship to people in the agent's world "
        "— employer, client, vendor, landlord, bank, insurer, clinic. Name the "
        "person in the value when known. (Renamed from relationship_to_user by "
        "migration 092 — data-safe, values verbatim.)",
        Cardinality.SINGLE,
        7,
    ),
    Slot(
        "location",
        "Where the organization is based or primarily operates.",
        Cardinality.SINGLE,
        6,
    ),
    Slot(
        "industry",
        "Sector / what the organization does.",
        Cardinality.SINGLE,
        6,
    ),
    Slot(
        "services_or_products",
        "Notable services or products the organization provides to the user.",
        Cardinality.MULTI,
        4,
    ),
]

# --- PLACE ------------------------------------------------------------------
_PLACE: list[Slot] = [
    Slot(
        "region",
        "Country / region / larger area this place is in.",
        Cardinality.SINGLE,
        6,
    ),
    Slot(
        "significance",
        "The place's standing significance in the agent's social world — whose "
        "home base, workplace location, or frequent destination it is. Name the "
        "person in the value. A change replaces it.",
        Cardinality.SINGLE,
        6,
    ),
    Slot(
        "place_type",
        "Kind of place (city, neighborhood, venue, landmark, …).",
        Cardinality.SINGLE,
        5,
    ),
]


# --- PROJECT ------------------------------------------------------------------
# Non-person coverage (2026-07-02): generic slot groups so recurring projects
# accumulate typed durable facts and clear the deterministic-card thin gate.
_PROJECT: list[Slot] = [
    Slot(
        "purpose",
        "What the project is / is for — its goal or domain in one phrase. A pivot replaces it.",
        Cardinality.SINGLE,
        7,
    ),
    Slot(
        "status",
        "Current lifecycle status (active, shipped, paused, abandoned). The newest mention wins.",
        Cardinality.SINGLE,
        6,
    ),
    Slot(
        "roles",
        "Durable roles people hold in the project (owner, maintainer, "
        "contributor, client). Name the person in each value, formatted "
        "'person — role'.",
        Cardinality.MULTI,
        6,
    ),
]

# --- AGENT --------------------------------------------------------------------
_AGENT: list[Slot] = [
    Slot(
        "purpose",
        "What this agent is for — its standing purpose or domain.",
        Cardinality.SINGLE,
        6,
    ),
    Slot(
        "capabilities",
        "Standing capabilities the agent keeps — kinds of tasks it can repeatedly "
        "perform for the people it serves. NOT one-off run outcomes, telemetry, or "
        "internal tool/feature inventories (those are operational).",
        Cardinality.MULTI,
        5,
    ),
]

# --- ANIMAL (pets) ----------------------------------------------------------
_ANIMAL: list[Slot] = [
    Slot(
        "species",
        "Species of the animal (dog, cat, …).",
        Cardinality.SINGLE,
        7,
    ),
    Slot(
        "owner",
        "Who owns / cares for the animal. Format 'name'.",
        Cardinality.SINGLE,
        7,
    ),
    Slot(
        "breed",
        "Breed, if known.",
        Cardinality.SINGLE,
        5,
    ),
]


# Exhaustive over the EntityType values - fieldless types map to [] explicitly
# so the schema is obviously complete (topic/object/event/group carry no durable
# attributes; most entities of those types are correctly empty).
DURABLE_ATTRIBUTE_SCHEMA: dict[str, list[Slot]] = {
    "person": _PERSON,
    "organization": _ORGANIZATION,
    "place": _PLACE,
    "animal": _ANIMAL,
    "project": _PROJECT,
    "agent": _AGENT,
    "topic": [],
    "object": [],
    "event": [],
    "group": [],
}


def slots_for(entity_type: str | None) -> list[Slot]:
    """Durable slots for an entity type ([] if unknown or fieldless).

    Returns a fresh list (the ``Slot`` items are frozen) so callers can't mutate
    the module-global registry in place. Off-schema durable facts are retained
    field-keyed by ``slot_profile.aggregate_slot_profile`` (the inversion), so there
    is no synthesis-only catch-all slot here — the schema types only what it knows.
    """
    if not entity_type:
        return []
    return list(DURABLE_ATTRIBUTE_SCHEMA.get(entity_type, []))


def schema_keys(entity_type: str | None) -> frozenset[str]:
    """Valid slot keys for an entity type."""
    return frozenset(s.key for s in slots_for(entity_type))


# Per-fact slot + durability instruction block — the SINGLE SOURCE OF TRUTH shared
# by the forward write-path (``EXTRACTION_PROMPT`` in memory_extraction.py) and the
# one-time ``durable_backfill`` classifier, so the two paths stay aligned. Both embed
# it as a ``{slot_durability_rubric}`` ``.format`` value, immediately ABOVE the
# rendered slot catalog (so "listed below" reads correctly). The leading indent of
# the first line comes from the embedding template; the rest carries its own.
SLOT_DURABILITY_RUBRIC = """\
· slot: one of the schema-catalog keys listed below for the subject entity's
    type, or "off_schema" if no key fits. Always use "off_schema" for entity
    types with no catalog (topic, object, event, group), for
    operational telemetry (exit codes, run ids, file sizes, mutation counts,
    download statuses), and for any point-in-time episodic value that should
    NOT appear on a stable profile card.
  · durability: "durable" (stays true over time — citizenship, employer,
    hometown, chronic condition, key relationship, language spoken, and stable
    lifestyle attributes like a gym/club membership, a sport practiced,
    regularly-used tools, or signature items; for non-person subjects likewise
    an organization's industry or its relationship to people the agent knows, a
    project's purpose or a person's role in it, an agent's standing
    capabilities, and a place's significance to the people in the agent's
    life), "transient" (point-in-time or
    episodic — today's lab value, an account balance, a booking, a one-off
    choice, a single symptom), or "operational" (about the AGENT'S OWN system,
    tools, or run-time process — exit_code, run_id, file_size,
    applied_mutations, download_status, automation features, internal
    tool/feature names). A standing capability the agent keeps (what it can
    repeatedly do for the people it serves) is "durable"; run-level outcomes and
    internal tool/feature inventories stay "operational". When uncertain → "transient".
  · grounding: record only attributes the subject actually HAS or IS. Do NOT
    record things the subject merely discussed, researched, compared, asked
    about, or was shown, as if they were the subject's attributes. Record active
    research or interest AS the weak predicate it is — an interests fact whose
    value says 'researching X' or 'interested in Y' — never as an ownership,
    skill, or identity fact; a product or topic under discussion is otherwise
    "off_schema"/"transient"."""


def render_schema_for_prompt() -> str:
    """Per-entity-type slot catalog for embedding in the Haiku extraction prompt.

    Only includes entity types that have defined durable slots.  Fieldless types
    (topic, object, event, group) are omitted — facts about those entities
    correctly map to ``"off_schema"`` by definition.

    Each line is ``  - <key>: <description>``.  The full ``Slot.description`` is
    surfaced (it is written as extraction guidance for Haiku — see the ``Slot``
    docstring); only internal whitespace/newlines are collapsed so each slot
    occupies a single line.

    Example output (abbreviated):
        person:
          - current_location: Where the person currently lives or is based …
          - key_relationships: Durable relationships to other people or pets …
        organization:
          - relationship: The organization's standing relationship to people …
    """
    lines: list[str] = []
    for entity_type, slots in DURABLE_ATTRIBUTE_SCHEMA.items():
        if not slots:
            continue
        lines.append(f"{entity_type}:")
        for slot in slots:
            # Collapse the multi-line description to one line; keep it complete
            # so Haiku gets the full per-slot extraction guidance.
            desc = " ".join(slot.description.split())
            lines.append(f"  - {slot.key}: {desc}")
    return "\n".join(lines)


# All slot keys across all types — for validation / metrics.
ALL_SLOT_KEYS: frozenset[str] = frozenset(s.key for slots in DURABLE_ATTRIBUTE_SCHEMA.values() for s in slots)


# ── Durability classification (write-path fact-loss fix, design 2026-07-20) ─────
# Durability is a property of a fact's TYPE, computed IN CODE from the resolved
# slot (hardening D-1) — NOT from Haiku's free judgment. A fact whose final slot is
# one of these curated PERSON-identity slots is stamped ``durable`` (earning the
# salience-floor retention that survives Ebbinghaus decay); every other slot —
# including ``off_schema`` and the softer person slots below — stays ``transient``
# and decays normally. Haiku's job is slot resolution; the retention tag is derived
# so a hallucinated "durable" can never floor a junk fact.
#
# SCOPE: person-identity slots only. The write-path fact-loss the floor fixes is
# about people (surname, "Olga = mother", employer, past injury). Non-person
# durable facts (an org's industry, a project's purpose) stay transient this pass
# — a noted follow-up, NOT a regression: the floor ships flag-decoupled and was
# never active before, so nothing that used to be retained stops being retained.
#
# EXCLUDED person slots (stay transient, promotable later): stable_preferences,
# interests, physical_attributes — softer / accumulating slots whose flooding risk
# outweighs the retention need; the schema still types them for the card.

# Floor for ANY subject — identity attributes that are low-harm to retain about a
# third party (a friend's citizenship or employer is not privacy-sensitive).
DURABLE_ANY_SUBJECT_SLOTS: frozenset[str] = frozenset(
    {
        "full_name",
        "current_location",
        "date_of_birth",
        "occupation",
        "employer",
        "hometown",
        "languages",
        "citizenship",
    }
)

# Floor ONLY for the first-party / primary subject (hardening P-1). These are MULTI
# with no per-subject ceiling, and Kai is universal multi-person memory (NO owner),
# so a third party's family/health ("my friend's mom is sick") must NOT mint a
# durable-floored trace — it is captured but stays transient and decays. The
# first-party predicate is applied at the D-1 stamping site (memory_extraction).
FIRST_PARTY_DURABLE_SLOTS: frozenset[str] = frozenset(
    {
        "key_relationships",
        "chronic_health_conditions",
        "allergies",
    }
)

# Every slot eligible for a durable tag (the first-party subset is additionally
# gated by the P-1 predicate at stamping time).
DURABLE_SLOT_SET: frozenset[str] = DURABLE_ANY_SUBJECT_SLOTS | FIRST_PARTY_DURABLE_SLOTS

# SINGLE-cardinality durable slots — the ones Part C's nightly supersede-release
# acts on: SINGLE is newest-wins, so an older value is genuinely superseded and its
# trace can be released to decay. MULTI durable slots accumulate distinct values and
# are never superseded. Derived from the schema so it stays in sync with cardinality.
DURABLE_SINGLE_SLOTS: frozenset[str] = frozenset(
    s.key for s in _PERSON if s.key in DURABLE_SLOT_SET and s.cardinality == Cardinality.SINGLE
)
