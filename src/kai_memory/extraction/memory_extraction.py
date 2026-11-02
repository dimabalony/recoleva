"""Level 2 memory extraction — post-conversation reflection via Haiku."""

from __future__ import annotations

from kai_memory.host_calls import get_model_roles

import json
import logging
import math
import uuid as _uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.host_calls import (
    call_llm_text,
    get_provider_for_model,
    get_settings,
    parse_json_object_response,
    start_generation_span,
    utc_now,
)
from kai_memory import ports
from kai_memory.identity import AGENT_IDENTITY_DATA_KEY, ensure_agent_identity, get_agent_identity
from kai_memory.store import cognitive_config
from kai_memory.storage.models.social_context import SocialContext, SocialExperience
from kai_memory.retrieval.iterative_retrieval import RECALL_MARKER
from kai_memory.procedural.procedural_scope import (
    ALL_VERDICTS,
    LEDGER_VERDICTS,
    MessageAnchor,
    PERSISTED_VERDICTS,
    VERDICT_FOLLOWED,
    VERDICT_MISAPPLIED,
    VERDICT_NOT_APPLICABLE,
    VERDICT_VIOLATED,
)
from kai_memory.store.cognitive_memory import CognitiveMemoryService
from kai_memory.store.cognitive_memory_constants import MEMORY_TYPES, VALID_ENTITY_TYPES, EntityType
from kai_memory.store.embeddings import generate_embeddings_batch
from kai_memory.ephemeral_probe import is_ephemeral_probe_chat
from kai_memory.extraction_cursor import (
    MAX_PAGES_PER_RUN,
    PAGE_SIZE,
    ExtractionCursor,
    page_key,
    read_messages_page,
)
from kai_memory.durable_schema import (
    DURABLE_ANY_SUBJECT_SLOTS,
    FIRST_PARTY_DURABLE_SLOTS,
    SLOT_DURABILITY_RUBRIC,
    render_schema_for_prompt,
    schema_keys,
)
from kai_memory.subject_attribution import attribute_fact_subjects

if TYPE_CHECKING:
    from kai_memory.ports import ChatContext

# Pinned to the pre-DB-842 module path: CloudWatch queries, feature-watch criteria and
# log filters match the logger name, which must not follow the module's move.
_LEGACY_LOGGER_NAME = "app.services.chat.memory_extraction"
logger = logging.getLogger(_LEGACY_LOGGER_NAME)


def _strip_recalled_lines(text: str | None) -> str:
    """Drop any line carrying the ``[recalled while thinking]`` marker.

    L1.5 Stage C — defense-in-depth marker-skip (design §Containment #5). Any line
    carrying this marker is a re-primed "recalled while thinking" association Kai saw
    mid-turn, NOT user-spoken conversation. The primary containment is placement
    (injected INSIDE toolResult.content → invisible to ConversationMessage.text), so
    in practice these lines never reach msg.text; this strip is the belt for any future
    placement drift, so Kai's own associations can never be re-extracted as
    user-attributed facts (the write-path echo chamber).
    """
    if not text:
        return ""
    if RECALL_MARKER not in text:
        return text
    return "\n".join(line for line in text.splitlines() if RECALL_MARKER not in line)


HAIKU_MODEL = "haiku-5.5"
MAX_CONVERSATION_CHARS = 80_000  # ~27K Haiku 5.5 tokens; ~5 clean meetings or ~2 tool-heavy ones per L2 batch
MAX_EXISTING_MEMORIES = 20

EntityMapKey = tuple[str, str]


def _bounded_thinking_text(parts: list[tuple[str, str]]) -> str:
    """Join the newest thinking within budget, retaining each summary's label.

    Slice before concatenating so neither one large legacy block nor the full
    extraction window creates an unbounded joined prompt temporary.
    """
    separator = "\n---\n"
    remaining = MAX_CONVERSATION_CHARS
    newest: list[str] = []
    for label, content in reversed(parts):
        separator_size = len(separator) if newest else 0
        available = remaining - separator_size - len(label)
        if available <= 0:
            break
        retained = content[-available:]
        newest.append(label + retained)
        remaining -= separator_size + len(label) + len(retained)
        if len(content) >= available:
            break
    return separator.join(reversed(newest)) if newest else "(none)"


# v2a §5.8: lossy semantic normalizer for Haiku-produced memory_type drift.
# Keys are observed out-of-Literal values; values are canonical MemoryType
# entries. New entries must be approved + reflected in the prompt rubric.
_MEMORY_TYPE_NORMALIZER: dict[str, str] = {
    "estimate": "fact",
    "observation": "impression",
    "insight": "impression",
    "boundary": "social_norm",
    "rule": "social_norm",
    "opinion": "preference",
    "recommendation": "plan",
    "advice": "plan",
    "task": "intention",
    "todo": "intention",
}


EXTRACTION_PROMPT = """\
You are a memory extraction system. Analyze the conversation below and extract \
discrete memories AND social signals.

## Part 1: Memories
For each memory, provide:
- content: the memory in natural language (one sentence)
- memory_type: one of the 11 values listed below. Pick the closest match —
  do not invent new types.
    - fact: a verifiable claim ("Misha works at Examplecorp")
    - episode: a remembered event or scene ("we celebrated Maria's birthday")
    - impression: a subjective observation or noticing ("the meeting felt rushed")
    - preference: a like, dislike, or chosen option ("prefers tea to coffee")
    - relationship: a connection between people or entities (used rarely)
    - plan: a forward-directed intent or schedule ("flying to Corfu next week")
    - intention: a personal commitment or to-do ("I will call mom Sunday")
    - status_change: a transition in someone's state ("Misha quit his job")
    - decision: a chosen path among alternatives ("we picked Postgres over Mongo")
    - social_norm: a social rule or boundary in this chat
      ("don't bring up salary in the family group")
    - self_reflection: agent's introspection about its own behavior
- significance: 1-5 (1=trivial, 5=critical/permanent like allergies)
- emotion_valence: -1.0 to 1.0 (negative to positive), null if neutral
- emotion_arousal: 0.0 to 1.0 (calm to intense), null if calm
- entities: list of {{"name": "...", "type": "...", "matched_participant_key": null}}
  objects. type is one of:
    person, animal, place, organization, topic, project, object, event, group, agent
  - animal: named pets and animals (e.g. "Луна", "Тучка")
  - object: a specific named item ("the new espresso machine")
  - event: a named happening ("our wedding", "Misha's PR review")
  - group: a collective reference ("the twins", "щенки", "the team")
  - agent: an AI assistant or agent (e.g. "{agent_display_name}", this assistant itself) — use
    this for the assistant, NOT "person", so it never merges with a human of
    the same name
  - **Always keep entity names in their original script.** If the source
    mentions a name in Cyrillic, return it in Cyrillic — do not
    transliterate to Latin (no "Луна" → "Luna", no "Миша" → "Misha").
    Transliteration can create duplicate entities and bifurcate the graph.
- contradicts_memory_id: if this fact contradicts an existing memory, provide that
  memory's ID (UUID string)
- facts: list of {{"subject": "...", "field": "...", "value": "...", \
"polarity": "is" | "is_not", "slot": "...", "durability": "..."}}
  objects — canonical key/value facts. "subject" is the ENTITY NAME the fact is
  ABOUT — exactly as it appears in `entities`, same script ("Миша" not "Misha" if
  the source is Cyrillic). A fact about Misha's role: {{"subject": "Misha", "field":
  "role", "value": "founder", "polarity": "is", "slot": "occupation", "durability": "durable"}}.
  Attribute each fact to the ONE entity it describes; never guess. Use polarity
  "is_not" for negations ("does NOT work at X"). Omit or use [] when there are no
  clean key/value facts.
  A durable person attribute (full name/surname, employer, occupation, chronic
  condition, allergy, citizenship, hometown, date of birth, language) or an
  IMMEDIATE-FAMILY relationship MUST be emitted HERE as a facts[] entry — this is
  the ONLY channel durable retention reads. Put immediate family in the
  "key_relationships" slot with the value formatted "relation: Name". The subject
  MUST be the PERSON WHO HAS the relationship (for a first-person statement like "my
  mom is Olga", that is the SPEAKER), never the related person — e.g. Misha saying
  "my mom is Olga" → {{"subject": "Misha", "field": "mother", "value": "mother:
  Olga", "polarity": "is", "slot": "key_relationships", "durability": "durable"}}
  (NOT subject "Olga"/value "son: Misha"). Never leave such a fact only in `content`
  or only as a relationships[] edge.
  For each fact additionally set:
  {slot_durability_rubric}

  Durable-attribute slot catalog — valid slot keys per entity type:
{schema_catalog}
- relationships: list of {{"subject": "...", "predicate": "...", "object": "...",
  "confidence": 0.0-1.0}} objects — typed edges between entities ({{"subject":
  "Misha", "predicate": "works_at", "object": "Examplecorp", "confidence": 0.9}}). Omit
  or use [] when there is no clear relationship. For an IMMEDIATE-FAMILY tie
  (parent, sibling, spouse, partner, child) ALSO emit a facts[] key_relationships
  entry (above) — a relationships[] edge alone is invisible to durable retention.
- confidence: 0.0-1.0 — your confidence that this extraction is accurate
- is_entity_eligible: true / false — set FALSE when this memory is ABOUT a
  credential or secret token (access_token, api_key, id_token, sb-credentials, …)
  that must NOT become a tracked entity; true for everything else

## Part 2: Social Experiences (usually empty)
Detect SOCIAL SIGNALS — moments where someone's reaction provides information
about how to behave in this chat. Look for these 6 signal types:

1. EXPLICIT CORRECTIONS: "don't bring that up", "not here", direct feedback
2. BOUNDARY ASSERTIONS: "that's private", "I'd rather not discuss"
3. POSITIVE REINFORCEMENT: thanks, elaboration, mirroring, laughter, "exactly"
4. IMPLICIT WITHDRAWAL: topic pivots, short replies, going quiet after
5. TONE MISMATCHES: casual response to serious topic (or vice versa)
6. THIRD-PARTY REPAIR: someone else softening your statement

For each signal, emit a social_experience with:
- experience: brief factual description of what happened
- learned_norm: tentative social rule (optional, null if unsure)
- signal_type: one of correction, boundary, positive, withdrawal, tone_mismatch, repair

Most conversations have ZERO social experiences. That's fine.
Only extract when a clear signal occurred.
{procedural_block}
## Context
The current time is {current_timestamp}.

Each user / assistant line below is prefixed by an ISO-8601 timestamp in
[brackets]. Use those timestamps to determine WHEN each fact happened.
Embed temporal context in extracted `content` using absolute dates ("on
2026-04-15") when the source mentions them, or clear relative anchors
grounded in those timestamps ("three days ago" when appropriate). Never
strip timestamps from source material. If the source line is from
[2026-04-17T09:00Z] and current_timestamp is [2026-04-20T08:00Z], a fact
about "today" in that source line happened three days ago as of now —
reflect that.

The agent's existing memories (for novelty/contradiction assessment):
{existing_memories}

The agent's thinking blocks (internal observations not shared with user):
{thinking_blocks}

Known chat participants for entity resolution:
{participant_context}

Conversation:
{conversation}
{served_rules_legend}
## Rules
- Skip routine greetings, small talk, and information available from external sources
- Focus on: facts about people and other tracked entities (places, organizations,
  projects, agents), decisions, status changes, preferences, plans, emotional observations
- Episodes should capture the *feel* of significant conversations, not just facts
- If a fact contradicts an existing memory, flag it with contradicts_memory_id
- For person entities that clearly match a known chat participant, set
  matched_participant_key to that participant's key; otherwise use null
- Be selective — 3 good memories > 10 mediocre ones
- IDENTITY EXCEPTION to selectivity: ALWAYS extract a durable identity fact about a
  person even when it is said once, in passing, or amid small talk — full
  name/surname, immediate family (parent, sibling, spouse, partner, child),
  employer, occupation, chronic health conditions, allergies, citizenship,
  hometown, date of birth, languages spoken. These are never "mediocre"; emit each
  as a structured facts[] entry (see the facts spec above), not only as prose.

## Response Format
Respond with a JSON object containing two keys: "memories" and "social_experiences".
{procedural_response_note}
Example:
{{
  "memories": [
    {{"content": "Misha works at Examplecorp", "memory_type": "fact", "significance": 3, \
"emotion_valence": null, "emotion_arousal": null, \
"entities": [{{"name": "Misha", "type": "person"}}, {{"name": "Examplecorp", "type": "organization"}}], \
"contradicts_memory_id": null, \
"facts": [{{"subject": "Misha", "field": "role", "value": "founder", \
"polarity": "is", "slot": "occupation", "durability": "durable"}}], \
"relationships": [{{"subject": "Misha", "predicate": "works_at", "object": "Examplecorp", "confidence": 0.9}}], \
"confidence": 0.85, "is_entity_eligible": true}}
  ],
  "social_experiences": [
    {{"experience": "Olga asked to keep private conversations private", \
"learned_norm": "Privacy boundaries in family chat are valued", \
"signal_type": "boundary"}}
  ]
}}

If nothing worth extracting, respond with: {{"memories": [], "social_experiences": []}}
"""

# Phase C §2.1 — Parts 3-5, rendered into the SAME Haiku call ONLY when
# procedural_rule_extraction_enabled is on (Stage 1 ships 0.0 → these placeholders
# format to "", so the prompt and its token cost are unchanged; behavior-neutral).
# NOTE: injected via .format() as pre-built strings, so literal braces here are
# fine — they are NOT re-interpreted by the outer .format() call.
_PROCEDURAL_BLOCK = """
## Part 3: Turn Outcomes (ONLY for assistant lines tagged [turn:N])
Some assistant lines carry a [turn:N] tag — those were tool-using turns. For each
tagged turn, judge from the human's SUBSEQUENT messages whether it succeeded:
- "success": the human accepted / moved on approvingly
- "failure_redo": the human asked to redo the same thing
- "failure_corrected": the human corrected HOW it was done (implies a general rule)
- "unclear": ambiguous OR the human stayed silent (SILENCE IS UNCLEAR, NEVER success)
For "failure_corrected", set "correction_instruction" to the implied rule as ONE
imperative sentence. Always set "evidence" to a verbatim fragment copied from a
human conversation line that justifies the judgment.

## Part 4: Behavior Rules (usually ZERO)
Moments where a participant instructs or corrects the assistant's OWN behavior
(format, tools/steps, when or whether to act). NOT rules: opinions, facts about
people, one-off task parameters, or anything about the assistant's stance/agreement.
Two strengths: "instruction" (an imperative, durable directive) and "correction"
(a fix of a specific output that implies a general preference). "source_quote" MUST
be verbatim text copied from ONE conversation line. Most conversations have ZERO.
Set "group_safe" true ONLY when the rule is impersonal — it names no person and
reveals nothing private, so anyone in any group could read it without harm.

## Part 5: Rule Compliance (ONLY when a line shows [served rules: ...])
Served rules are STANDING rules — the assistant is shown all of them every turn, not
a selection matched to this conversation. The "Served rules" legend below the Context
section gives, for every id that appears in a [served rules: ...] marker, that rule's
TRIGGER and the rule itself. Use the id EXACTLY as the legend spells it when you answer.
Judge each rule by its TRIGGER:
- CONDITIONAL trigger — it names a task type or condition ("when drafting an email for
  external send", "when a clinician has told him something"). If that task or condition
  did not come up, the situation did not arise: "not_applicable" is the DEFAULT here and
  by far the most common verdict overall.
- UNIVERSAL trigger — it matches every turn ("any message composition",
  "conversation participation", "tool-use and message sending", or equivalent).
  Its situation arises on EVERY tagged turn, so judge the tagged turn against the
  rule and answer "followed" or "violated" — never "not_applicable". For a
  prohibition ("do not X") "followed" means X is absent from the tagged turn; for
  a positive directive ("always do Y") "followed" means the turn did Y. A silently
  honored prohibition IS compliance, not an inapplicable rule.
For each served rule id:
- "not_applicable": the rule's situation did not arise this turn. Use this whenever the
  rule's trigger clearly did not come up — do NOT reward-guess "followed" for a rule
  that simply never applied, and do NOT mark "violated" merely because the assistant
  did not do what an unrelated rule describes.
- "followed": the rule's situation DID arise and the assistant's tagged turn honored it.
- "violated": the rule's situation DID arise and the assistant's tagged turn did not.
- "misapplied": the rule's situation did NOT arise, yet the assistant applied it anyway.
  This is the wrong-rule-fired case: serving every standing rule every turn makes
  applying an irrelevant rule the realistic failure, not failing to find a relevant one.
"""

_PROCEDURAL_RESPONSE_NOTE = """When procedural markers appear above, ALSO include any of these keys \
(omit or use [] when empty):
- "turn_outcomes": [{"turn": N, "outcome": "success|failure_redo|failure_corrected|unclear", \
"correction_instruction": "<one imperative sentence or null>", "evidence": "<verbatim fragment>"}]
- "behavior_rules": [{"rule": "<the rule, imperative>", "trigger": "<task-TYPE this applies to>", \
"category": "format|workflow|tool_use|communication|scheduling", \
"directive_strength": "instruction|correction", "scope_hint": "global|this_person|heartbeat|wake", \
"group_safe": true|false, \
"source_quote": "<verbatim text from ONE conversation line>"}]
- "rule_compliance": [{"rule_id": "<served rule id>", "verdict": "followed|violated|not_applicable|misapplied"}]
"""


# DB-629 — characters of a rule UUID used as its DISPLAY id in the served-rule markers
# and the legend. 8 hex is 4.3e9 values against a store capped at SERVE_MAX_RULES (25)
# in-scope rules, so a prefix collision inside one window is a curiosity, not a plan —
# and ``_display_ids`` falls back to full ids the moment one happens.
_SHORT_ID_CHARS = 8

# Bumped to 3 for INC 3 schema-constrained extraction: facts now carry
# ``slot`` (a durable-attribute schema key or "off_schema") and ``durability``.
# Since hardening D-1 (2026-07-20) ``durability`` is COMPUTED in code from the
# resolved slot (``durable`` | ``transient``) rather than trusted from Haiku.
TYPED_METADATA_SCHEMA_VERSION = 3

_FALSEY_TOKENS = frozenset({"false", "0", "no", "n", "f", "off", ""})

# Valid values for Haiku's RAW durability opinion, preserved on each fact as
# ``claimed_durability`` (tri-state incl. "operational"). This is ADVISORY analytics
# for the §6.1 off-schema quarantine / slot-promotion signal — NOT a retention signal.
# The code-stamped ``durability`` (computed from the slot, D-1) is the binary
# durable/transient tag that retention/floor/marker read.
_CLAIMED_DURABILITY: frozenset[str] = frozenset({"durable", "transient", "operational"})


def _coerce_bool(value: object, *, default: bool) -> bool:
    """Best-effort bool from Haiku output, which may emit a JSON bool OR a
    string (``"false"``). Plain ``bool("false")`` is ``True``, which for the
    ``is_entity_eligible`` credential guard is the dangerous false-negative —
    so string falsey-tokens map to False explicitly. None/absent → ``default``.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in _FALSEY_TOKENS
    return bool(value)


def _coerce_confidence(value: object) -> float | None:
    """Coerce a Haiku-emitted confidence to ``float | None`` (Sub-design C §1.1, CR review).

    Haiku may emit a numeric string (``"0.85"``), an out-of-range number, a JSON
    bool, or junk text. The column/blob slot is ``float | null``, so: parse
    numerics (incl. numeric strings) and clamp to ``[0, 1]``; bools, NaN, and
    non-numeric junk become ``None``.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return max(0.0, min(1.0, f))


def _coerce_list(value: object) -> list[Any]:
    """Coerce a Haiku list-slot to a clean list (Sub-design C §1.1, CR 3355128362).

    Haiku may emit ``facts``/``events``/``entities``/``relationships`` as a bare
    dict, string, or scalar instead of a list. ``value or []`` only neutralises
    falsy values — a truthy non-list (``"x"``, ``{...}``) would persist a
    malformed blob that the deterministic aggregation downstream cannot read.
    So: non-lists become ``[]``; within a list, ``None`` entries are dropped.
    """
    if isinstance(value, list):
        return [item for item in value if item is not None]
    return []


def _coerce_entities(value: object) -> list[Any]:
    """Validate model entity shapes before name lookup, participant binding and storage."""
    entities = []
    for entry in _coerce_list(value):
        name = entry.get("name") if isinstance(entry, dict) else entry
        if not isinstance(name, str) or not name.strip():
            continue
        if isinstance(entry, dict):
            entry = dict(entry)
            if not isinstance(entry.get("type"), str):
                # Keep the existing contextual defaults: matched participants are
                # people; otherwise an absent/unknown type defaults to topic.
                entry["type"] = None
            key = entry.get("matched_participant_key")
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                entry["matched_participant_key"] = None
        entities.append(entry)
    return entities


def _durability_for_slot(slot: str, subject_id: Any, first_party_subject_ids: frozenset[str]) -> str:
    """Compute a fact's ``durability`` from its FINAL resolved slot (hardening D-1).

    Durability is a property of the fact's TYPE, not Haiku's free judgment: a fact
    that resolves to a curated durable-identity slot is ``durable``; everything else
    is ``transient``. The first-party subset (family/health, P-1) is durable ONLY
    when the subject is a first-party entity — a third party's family/health is
    captured but stays transient (Kai is universal, ownerless memory). Any
    Haiku-emitted ``durability`` is ignored: a hallucinated tag can never floor junk.
    """
    if slot in DURABLE_ANY_SUBJECT_SLOTS:
        return "durable"
    if slot in FIRST_PARTY_DURABLE_SLOTS and subject_id and str(subject_id) in first_party_subject_ids:
        return "durable"
    return "transient"


def _validate_and_default_fact_slots(
    facts: list[Any],
    resolved_entities: list[Any],
    *,
    first_party_subject_ids: frozenset[str] = frozenset(),
) -> None:
    """In place: default/validate ``slot`` and COMPUTE ``durability`` on each fact.

    Must be called **after** :func:`attribute_fact_subjects` so that
    ``fact["subject_id"]`` is already set (or absent/None).

    Slot default (structural coercion):
    - ``slot`` missing / not a non-empty string → ``"off_schema"``

    Semantic validators (each *downgrades* ``slot`` to ``"off_schema"``, never
    drops the fact):
    1. Non-``"off_schema"`` slot without a resolved ``subject_id`` → ``"off_schema"``
       (can't know the entity type, so can't validate the key).
    2. Non-``"off_schema"`` slot that is not a valid key for the subject's
       entity type → ``"off_schema"`` (Haiku hallucinated or cross-typed).

    Durability (hardening D-1): the code-stamped ``durability`` is computed from the
    FINAL slot via :func:`_durability_for_slot` — Haiku's raw tag is discarded so a
    hallucinated "durable" can't floor junk. ``first_party_subject_ids`` gates the P-1
    family/health subset (empty ⇒ those stay transient — the safe default). Haiku's raw
    opinion is preserved as ``claimed_durability`` (tri-state) for the §6.1 off-schema
    quarantine only (hardening R-1) — advisory analytics, never a retention signal.
    """
    if not facts:
        return

    # Build entity-id → entity_type map from the already-resolved entities.
    entity_type_by_id: dict[str, str] = {}
    for e in resolved_entities or []:
        eid = getattr(e, "id", None)
        etype = getattr(e, "entity_type", None)
        if eid is not None and etype:
            entity_type_by_id[str(eid)] = etype

    for fact in facts:
        if not isinstance(fact, dict):
            continue

        # ── structural slot default ──────────────────────────────────────────
        # Canonicalize to lower-case: every schema key is lower_snake_case, so a
        # Haiku-emitted "Occupation"/"CITIZENSHIP" must normalize before the
        # schema_keys check below, else a valid durable slot is needlessly lost.
        slot = fact.get("slot")
        fact["slot"] = slot.strip().lower() if isinstance(slot, str) and slot.strip() else "off_schema"

        # ── semantic validators (may downgrade slot to off_schema) ───────────
        if fact["slot"] != "off_schema":
            # Validator 1: a real slot requires a resolved subject_id — otherwise
            # the slot could smear across co-mentioned entities (design §3.2).
            subject_id = fact.get("subject_id")
            if not subject_id:
                fact["slot"] = "off_schema"
            else:
                # Validator 2: the slot must be a valid key for the subject's entity
                # type. ``schema_keys`` returns an empty set for an unknown/unresolvable
                # type, so a subject_id we can't map to an entity type also degrades —
                # precision-first (design §3.2: a missed durable is recoverable, a
                # leaked one is the failure we're fixing).
                entity_type = entity_type_by_id.get(str(subject_id))
                if fact["slot"] not in schema_keys(entity_type):
                    fact["slot"] = "off_schema"

        # ── R-1: preserve Haiku's RAW opinion as ``claimed_durability`` (tri-state) ──
        # BEFORE overwriting. This advisory tag is what the §6.1 off-schema quarantine
        # (slot_profile off-schema retention + off_schema_durable_count) reads to keep
        # durable-by-Haiku off-schema facts alive as a slot-promotion signal — NOT a
        # retention signal. Without it, D-1's uniform "transient" would starve the
        # quarantine (the inert-tag failure reborn one level up).
        raw = fact.get("durability")
        fact["claimed_durability"] = (
            raw.strip().lower() if isinstance(raw, str) and raw.strip().lower() in _CLAIMED_DURABILITY else "transient"
        )

        # ── D-1: durability derived from the FINAL slot, never trusted from Haiku ─
        # The ONLY signal retention/floor/marker read (binary durable/transient).
        fact["durability"] = _durability_for_slot(fact["slot"], fact.get("subject_id"), first_party_subject_ids)


def _subject_ids_from_facts(facts: Any) -> frozenset[str]:
    """The distinct non-empty ``subject_id`` values in a facts list (DB-591).

    Tolerant by construction: a non-list ``facts``, a non-dict entry, or a fact with
    no resolved subject each contribute nothing rather than raising. Shared by the
    new-item and existing-trace sides of the contradiction subject gate so the two
    can never drift apart.
    """
    return frozenset(
        str(f["subject_id"])
        for f in (facts if isinstance(facts, list) else [])
        if isinstance(f, dict) and f.get("subject_id")
    )


def _build_typed_metadata(item: dict[str, Any]) -> dict[str, Any]:
    """Assemble the typed_metadata JSONB blob (Sub-design C §1.1) from one
    parsed Haiku memory item.

    Every slot is a reserved key, graceful on absence (R3): a missing key
    yields its empty default, never an error — so an old/sparse Haiku response
    still produces a valid (mostly-empty) blob. INC 1 populates
    facts/relationships/entities/confidence/is_entity_eligible from the additive
    prompt; ``events`` stays reserved-empty until Trajectory wires it,
    ``quality.novelty`` is reserved for A (INC 5), and ``query_type_hint`` for
    OEC (M4). The one model-emitted ``confidence`` feeds both D's prior
    (top-level) and A's quality floor (quality.extraction_confidence).
    """
    confidence = _coerce_confidence(item.get("confidence"))
    return {
        "schema_version": TYPED_METADATA_SCHEMA_VERSION,
        "facts": _coerce_list(item.get("facts")),
        "events": _coerce_list(item.get("events")),
        "entities": _coerce_list(item.get("entities")),
        "relationships": _coerce_list(item.get("relationships")),
        "confidence": confidence,
        "quality": {
            "extraction_confidence": confidence,
            "novelty": None,
        },
        "is_entity_eligible": _coerce_bool(item.get("is_entity_eligible"), default=True),
        "query_type_hint": None,
    }


def _dedupe_turn_outcomes(
    pending: list[tuple[str, str, int | None, str]],
) -> tuple[list[tuple[str, str, int | None, str]], dict[str, set[str]]]:
    """Drop turn-outcome rows whose trace_id carries CONFLICTING outcomes (design
    §5.1 malformed-model-output rule), collapse same-outcome repeats to the first.

    Returns ``(kept, conflicted)`` where ``conflicted`` maps each dropped trace_id to
    its set of contradictory outcomes (for the caller to log). Pure — unit-testable."""
    outcomes_by_trace: dict[str, set[str]] = {}
    for trace_id, mapped, _speaker, _evidence in pending:
        outcomes_by_trace.setdefault(trace_id, set()).add(mapped)
    conflicted = {tid: outs for tid, outs in outcomes_by_trace.items() if len(outs) > 1}
    seen: set[str] = set()
    kept: list[tuple[str, str, int | None, str]] = []
    for row in pending:
        tid = row[0]
        if tid in conflicted or tid in seen:
            continue
        seen.add(tid)
        kept.append(row)
    return kept, conflicted


class ExtractionStoreFailure(RuntimeError):
    """An extracted item could not be written; the run must not advance its cursor.

    ``page_cursor`` is where the failing page started, so repeated failures are
    counted per page, not per run.
    """

    def __init__(self, message: str, page_cursor: ExtractionCursor | None) -> None:
        super().__init__(message)
        self.page_cursor = page_cursor


class ExtractionService:
    """Level 2: Extract memories from conversation pauses."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        llm_text_fn: Callable[..., Awaitable[str | None]] | None = None,
    ) -> None:
        self.session = session
        # The one LLM call of a batch (DB-840). None keeps ports' call_llm_text, read at
        # call time; the memory eval injects a recorded or budgeted extractor instead.
        self._llm_text_fn = llm_text_fn
        self.memory_svc = CognitiveMemoryService(session)
        self.conv_manager = ports.ConversationManager(session)
        self.store_errors: list[str] = []

    async def extract_from_conversation(
        self,
        chat_ctx: ChatContext,
        since: datetime,
        priority_contradiction_candidates: list[_uuid.UUID] | None = None,
    ) -> dict[str, int]:
        """Extract memories from recent conversation, in one read of up to 200 messages.

        The L2 trigger uses ``extract_after_cursor`` instead, which pages the whole
        backlog; this single-read form serves callers that own a bounded window
        (the capture probe).

        priority_contradiction_candidates: trace IDs retrieved during priming
        in this chat — Haiku pays special attention to these as possible
        contradiction targets.

        Returns dict with counts: {"traces_created": N, "entities_found": N, "contradictions": N}
        """
        # NOTE: Extraction runs for ALL chats including groups. This is intentional —
        # the cognitive memory design uses universal memory with social judgment
        # (via social_contexts and social_norms) rather than data isolation.
        # See backend-docs/ai-workforce/plans/design/2026-04-01-cognitive-memory-architecture-design.md Section 6.
        # Headless venues (event, heartbeat) deliberately use chat_id=None;
        # `get_messages_since` translates the comparison to `IS NULL` via
        # SQLAlchemy. Do NOT coerce to 0 here — that would query for a
        # non-existent telegram chat 0 and return zero rows.
        stats = self._new_stats()

        # 1. Load messages since last extraction
        messages = await self.conv_manager.get_messages_since(
            chat_ctx.chat_id, chat_ctx.topic_id, since, venue=chat_ctx.venue
        )
        await self._extract_batch(chat_ctx, messages, stats, priority_contradiction_candidates)
        self._log_extraction_complete(chat_ctx, stats)
        return stats

    async def extract_after_cursor(
        self,
        chat_ctx: ChatContext,
        cursor: ExtractionCursor | None,
        priority_contradiction_candidates: list[_uuid.UUID] | None = None,
        *,
        skip_page_at: str | None = None,
    ) -> tuple[dict[str, Any], ExtractionCursor | None]:
        """Extract from every message after ``cursor``, page by page (DB-794).

        Each keyset page of up to PAGE_SIZE messages is one extraction batch (one
        Haiku call); at most MAX_PAGES_PER_RUN pages per run. Returns the stats and
        the cursor after the last message read and processed, which the caller
        persists in the same commit as the extraction writes. Pages past the cap
        stay above the returned cursor (``backlog_remaining``) for the next run.

        A failed item write raises ExtractionStoreFailure, so the cursor stays put
        and the run retries the window. ``skip_page_at`` (a ``page_key``, set by the
        trigger after repeated store failures on one page) lets only the page that
        starts there be passed: its message ids go to ``skipped_after_retries`` and
        the run ends, so any later page's failure gets its own retries.
        """
        stats: dict[str, Any] = {
            **self._new_stats(),
            # DB-591 rule: initialized up front, so zero is measured, not missing.
            "messages_read": 0,
            "pages_read": 0,
            "cursor_advanced": False,
            "backlog_remaining": False,
            "skipped_after_retries": [],
        }
        has_more = False
        for _ in range(MAX_PAGES_PER_RUN):
            # One row past the page tells whether another page is waiting.
            rows = await read_messages_page(self.session, chat_ctx, cursor, PAGE_SIZE + 1)
            has_more = len(rows) > PAGE_SIZE
            page = rows[:PAGE_SIZE]
            if not page:
                break
            stats["pages_read"] += 1
            stats["messages_read"] += len(page)
            page_start = cursor
            failures_before = stats["store_failures"]
            await self._extract_batch(chat_ctx, page, stats, priority_contradiction_candidates)
            if stats["store_failures"] > failures_before:
                if skip_page_at is None or page_key(page_start) != skip_page_at:
                    raise ExtractionStoreFailure(
                        f"{stats['store_failures'] - failures_before} item write(s) failed: {self.store_errors[-1]}",
                        page_cursor=page_start,
                    )
                stats["skipped_after_retries"].extend(str(m.id) for m in page)
                logger.warning(
                    "L2 extraction skipped %d messages after repeated store failures: chat_id=%s, topic_id=%s, %s",
                    len(page),
                    chat_ctx.chat_id,
                    chat_ctx.topic_id,
                    self.store_errors[-1],
                )
                # End the run on the skip: earlier pages and the skip commit, and the
                # drain timer (backlog_remaining) carries on from past this page.
                cursor = ExtractionCursor.after(page[-1])
                stats["cursor_advanced"] = True
                break
            cursor = ExtractionCursor.after(page[-1])
            stats["cursor_advanced"] = True
            if not has_more:
                break
        stats["backlog_remaining"] = has_more
        self._log_extraction_complete(chat_ctx, stats)
        return stats, cursor

    @staticmethod
    def _new_stats() -> dict[str, int]:
        return {
            "traces_created": 0,
            "entities_found": 0,
            "contradictions": 0,
            # DB-591: both gates' counters are initialized, not lazily created — a key
            # that only appears when its branch fires makes "zero" and "not measured"
            # indistinguishable downstream, and these two exist to be measured.
            "contradictions_skipped_subject_mismatch": 0,
            "contradictions_subject_unknown": 0,
            "social_experiences": 0,
            "third_party_family_transient": 0,  # P-1 telemetry (refinement 2)
            "store_failures": 0,
            "malformed_items": 0,
        }

    @staticmethod
    def _normalize_items(items: list[Any], stats: dict[str, int]) -> list[dict[str, Any]]:
        normalized = []
        for item in items:
            content = item.get("content") if isinstance(item, dict) else None
            if not isinstance(content, str) or not content.strip():
                stats["malformed_items"] += 1
                continue
            raw_significance = item.get("significance", 3)
            try:
                significance = float(raw_significance) if not isinstance(raw_significance, bool) else 3.0
            except (TypeError, ValueError, OverflowError):
                significance = 3.0
            if not math.isfinite(significance):
                significance = 3.0
            raw_type = item.get("memory_type", "fact")
            normalized.append(
                {
                    **item,
                    "content": content.strip(),
                    "significance": int(max(1, min(5, significance))),
                    "entities": _coerce_entities(item.get("entities")),
                    "facts": _coerce_list(item.get("facts")),
                    "memory_type": raw_type if isinstance(raw_type, str) else "fact",
                }
            )
        return normalized

    async def _extract_batch(
        self,
        chat_ctx: ChatContext,
        messages: list,
        stats: dict[str, int],
        priority_contradiction_candidates: list[_uuid.UUID] | None,
    ) -> None:
        """Run one extraction batch (one Haiku call) over ``messages``, adding to ``stats``."""
        chat_id = chat_ctx.chat_id
        topic_id = chat_ctx.topic_id
        if not messages:
            logger.debug("No messages to extract from (chat_id=%s, topic_id=%s)", chat_id, topic_id)
            return

        # Phase C §2.1 — Parts 3-5 render into this SAME Haiku call only when
        # capture is enabled. Off (Stage 1 default) → no [turn:N] tags, empty
        # procedural prompt blocks: the call is byte-for-byte unchanged.
        procedural_enabled = cognitive_config.get_bool("procedural_rule_extraction_enabled")

        # DB-629 pre-pass: resolve the rules served in this window to their TEXT before
        # any marker is rendered, because both the marker id form (short vs full) and
        # which ids may appear at all are decided by what the legend could load. A rule
        # that cannot be rendered safely is dropped from the markers too — the judge is
        # never shown an id it has no text for (see procedural_serving.rule_legend_line).
        served_legend = ""
        served_alias: dict[str, str] = {}  # display id -> rule id
        served_display: dict[str, str] = {}  # rule id -> display id
        if procedural_enabled:
            served_legend, served_alias, served_display = await self._build_served_rules_legend(messages)

        # 2. Build conversation text + thinking blocks (v2a §6.1: per-message
        # ISO-8601 timestamps so Haiku can ground "today" / "yesterday" in
        # the source line's clock, not just the current_timestamp.)
        conversation_parts: list[str] = []
        thinking_parts: list[tuple[str, str]] = []
        # §2.1 turn spine: local N → execution_trace_id map (the LLM never sees UUIDs).
        turn_map: dict[int, str] = {}
        for msg in messages:
            role = msg.role
            # L1.5 Stage C: strip any [recalled while thinking] lines before they can
            # become L2 input (defense-in-depth; placement already keeps them out of
            # msg.text). See _strip_recalled_lines.
            text = _strip_recalled_lines(msg.text)
            if text:
                sender = f" ({msg.sender_name})" if msg.sender_name and role == "user" else ""
                created = getattr(msg, "created_at", None)
                ts_prefix = f"[{created.isoformat()}] " if created is not None else ""
                marker = ""
                if procedural_enabled and role == "assistant":
                    meta = msg.message_metadata or {}
                    trace_id = meta.get("execution_trace_id")
                    if trace_id:
                        n = len(turn_map) + 1
                        turn_map[n] = str(trace_id)
                        marker += f" [turn:{n}]"
                    # DB-629: render the DISPLAY id (short, collision-checked), and only
                    # for rules the legend below actually explains. An id with no legend
                    # entry is omitted rather than shown opaquely.
                    served = [
                        served_display[str(s)] for s in (meta.get("served_rule_ids") or []) if str(s) in served_display
                    ]
                    if served:
                        marker += f" [served rules: {', '.join(served)}]"
                conversation_parts.append(f"{ts_prefix}[{role}{sender}]{marker}: {text}")
            # Harvest thinking blocks from assistant message metadata
            if role == "assistant" and msg.message_metadata:
                for tb in msg.message_metadata.get("thinking_blocks", []):
                    if isinstance(tb, str) and tb.strip():
                        thinking_parts.append(("", tb))
                summaries = msg.message_metadata.get("reasoning_summaries", [])
                if isinstance(summaries, list):
                    for summary in summaries:
                        if isinstance(summary, str) and summary.strip():
                            # Native Codex supplies summaries, not raw thinking.
                            # Keep that provenance explicit and exclude recalled
                            # facts from being extracted again as observations.
                            cleaned = _strip_recalled_lines(summary)
                            if cleaned.strip():
                                thinking_parts.append(
                                    ("[Assistant reasoning summary, not a user statement]\n", cleaned)
                                )

        conversation_text = "\n".join(conversation_parts)
        if len(conversation_text) > MAX_CONVERSATION_CHARS:
            conversation_text = conversation_text[-MAX_CONVERSATION_CHARS:]

        # Skip if conversation is too short / only greetings
        if len(conversation_text.strip()) < 50:
            logger.debug("Conversation too short for extraction")
            return

        if procedural_enabled:
            from kai_memory.procedural.rule_capture import lock_procedural_capture

            # Acquire before memory/social/outcome writes, not only at later rule capture.
            await lock_procedural_capture(self.session)

        # 3. Load existing memories as context for novelty assessment
        existing_memories = await self._load_existing_context(
            conversation_text,
            priority_candidates=priority_contradiction_candidates,
        )

        # 4. Call Haiku for extraction (returns both memories and social experiences)
        thinking_text = _bounded_thinking_text(thinking_parts)
        # Headless venues (chat_id IS NULL) skip social context — _ensure_social_context
        # would violate the social_contexts.chat_id NOT NULL constraint.
        if chat_ctx.chat_id is not None:
            await self._ensure_social_context(chat_ctx)
        participant_context, participant_map = await self._load_participant_context(chat_ctx)
        extraction = await self._call_haiku(
            conversation_text,
            thinking_text,
            existing_memories,
            participant_context=participant_context,
            procedural_enabled=procedural_enabled,
            served_rules_legend=served_legend,
        )
        items = self._normalize_items(extraction.get("memories", []), stats)
        social_experiences = extraction.get("social_experiences", [])
        turn_outcomes = extraction.get("turn_outcomes", [])
        behavior_rules = extraction.get("behavior_rules", [])
        rule_compliance = extraction.get("rule_compliance", [])
        if not items and not social_experiences and not turn_outcomes and not behavior_rules and not rule_compliance:
            return

        # 5a. Batch-embed all item contents
        item_contents = [item.get("content", "").strip() for item in items]
        try:
            embeddings = await generate_embeddings_batch(
                [c for c in item_contents if c],
            )
        except Exception:
            logger.warning("Batch embedding failed, falling back to per-item", exc_info=True)
            embeddings = None

        # 5b. Batch-resolve all entity names
        all_entity_names: list[str] = []
        for item in items:
            for raw_entity in item.get("entities", []):
                if isinstance(raw_entity, dict):
                    name = (raw_entity.get("name") or "").strip()
                elif isinstance(raw_entity, str):
                    name = raw_entity.strip()
                else:
                    name = ""
                if name:
                    all_entity_names.append(name)
        try:
            resolved_entities = await self.memory_svc.find_entities_by_names(
                list(set(all_entity_names)),
                only_ungrounded=True,
            )
            entity_map = {(e.name.lower(), e.entity_type): e for e in resolved_entities}
        except Exception:
            logger.warning("Batch entity resolution failed", exc_info=True)
            entity_map = {}

        try:
            matched_entity_map = await self._resolve_matched_participant_entities(items, participant_map)
            entity_map.update(matched_entity_map)
        except Exception:
            logger.warning("Participant-aware entity resolution failed", exc_info=True)

        # 5c. Store extracted items using pre-computed embeddings and entity map
        emb_idx = 0
        for item in items:
            content = item.get("content", "").strip()
            emb = None
            if content and embeddings is not None and emb_idx < len(embeddings):
                emb = embeddings[emb_idx]
                emb_idx += 1
            elif not content:
                pass  # _store_item will skip empty content
            else:
                emb_idx += 1  # advance index even when embeddings is None
            try:
                # A savepoint per item keeps a DB error inside this item: the
                # transaction stays usable and the failure is counted, not hidden.
                async with self.session.begin_nested():
                    await self._store_item(item, chat_ctx, stats, embedding=emb, entity_map=entity_map)
            except Exception as exc:
                self._record_store_failure(stats, exc)
                logger.warning("Failed to store extracted item type=%s", type(exc).__name__)

        # 6. Persist experiences into the social context ensured before extraction.
        # Headless venues (chat_id IS NULL) skip this — _store_social_experiences
        # would just emit a "no social context found" warning and silently drop
        # the experiences anyway.
        if chat_ctx.chat_id is not None and social_experiences:
            try:
                async with self.session.begin_nested():
                    await self._store_social_experiences(chat_ctx, social_experiences, stats)
            except Exception as exc:
                self._record_store_failure(stats, exc)
                logger.warning("Failed to store social experiences type=%s", type(exc).__name__)

        # 7. Phase C — turn-outcome labeling (§5.1) + rule capture (§2.2).
        # Capture errors must escape: the extraction savepoint rolls back the
        # entire run and retains its cursor so no teaching is lost (DB-865).
        if procedural_enabled:
            if turn_outcomes and turn_map:
                await self._apply_turn_outcomes(turn_outcomes, turn_map, messages, chat_ctx)
            if behavior_rules:
                from kai_memory.procedural.rule_capture import capture_rules

                from kai_memory.host_calls import get_host_policy

                admin_ids = getattr(get_settings(), "telegram_admin_user_ids", "")
                await capture_rules(
                    self.session,
                    chat_ctx,
                    behavior_rules,
                    messages,
                    admin_ids,
                    respect_of=get_host_policy().respect,
                    raise_on_error=True,
                )
            if rule_compliance:
                await self._apply_rule_compliance(rule_compliance, messages, served_alias)

    def _record_store_failure(self, stats: dict[str, int], exc: Exception) -> None:
        """Count a failed write so the run fails (or, after retries, skips) visibly."""
        stats["store_failures"] += 1
        self.store_errors.append(type(exc).__name__)

    @staticmethod
    def _log_extraction_complete(chat_ctx: ChatContext, stats: dict[str, int]) -> None:
        logger.info(
            "Level 2 extraction complete: chat_id=%s, topic_id=%s, traces=%d, entities=%d, "
            "contradictions=%d, contradictions_skipped_subject_mismatch=%d, "
            "contradictions_subject_unknown=%d, social_experiences=%d, third_party_family_transient=%d",
            chat_ctx.chat_id,
            chat_ctx.topic_id,
            stats["traces_created"],
            stats["entities_found"],
            stats["contradictions"],
            # DB-591: the two gate counters ride the same line as `contradictions` —
            # a refusal is only readable next to the invalidations it sits beside, and
            # `subject_unknown` is the gate's coverage gap, which is the number to watch.
            stats["contradictions_skipped_subject_mismatch"],
            stats["contradictions_subject_unknown"],
            stats["social_experiences"],
            stats["third_party_family_transient"],
        )

    @staticmethod
    def _display_ids(rule_ids: list[str]) -> dict[str, str]:
        """``rule id -> display id`` for the served-rule markers (DB-629).

        Short ids (first 8 hex) by default: the markers repeat on every assistant line,
        so full UUIDs cost kilobytes per extraction call for a window that serves a dozen
        standing rules, and a long opaque UUID is also the form a model is most likely to
        garble when echoing it back. The moment two ids share a prefix the whole window
        falls back to full ids — an ambiguous marker would let one rule's verdict land on
        another rule's ledger, which is worse than the tokens."""
        shorts = {rid: rid[:_SHORT_ID_CHARS] for rid in rule_ids}
        if len(set(shorts.values())) != len(shorts):
            logger.warning("procedural.legend.short_id_collision n=%d — falling back to full ids", len(shorts))
            return {rid: rid for rid in rule_ids}
        return shorts

    async def _build_served_rules_legend(self, messages: list) -> tuple[str, dict[str, str], dict[str, str]]:
        """Resolve this window's served rule ids to their trigger/guidance text (DB-629).

        Returns ``(legend_block, display id -> rule id, rule id -> display id)``.

        This is the fix for the structural gap that made DB-590 inert: the judge was shown
        `[served rules: <uuid>]` and asked to "judge each rule by its TRIGGER", but nothing
        in the call ever said what any rule's trigger WAS, so ``not_applicable`` was the
        only answer available to it. Fail-open — a legend that cannot be built yields no
        markers at all, which degrades Part 5 to silence rather than to guesswork."""
        rule_ids: set[str] = set()
        for msg in messages:
            if getattr(msg, "role", None) != "assistant":
                continue
            meta = getattr(msg, "message_metadata", None) or {}
            for rid in meta.get("served_rule_ids") or []:
                rule_ids.add(str(rid))
        if not rule_ids:
            return "", {}, {}
        try:
            from kai_memory.procedural.procedural_serving import (
                SERVE_MAX_RULES,
                load_rules_by_id,
                render_rule_legend,
            )

            # Bound BEFORE the load, not only at render: a renderer-only cap still pays
            # for every row the query returned.
            ordered_ids = sorted(rule_ids)
            if len(ordered_ids) > SERVE_MAX_RULES:
                logger.warning("procedural.legend.capped served=%d cap=%d", len(ordered_ids), SERVE_MAX_RULES)
                ordered_ids = ordered_ids[:SERVE_MAX_RULES]
            uuids: list[_uuid.UUID] = []
            for rid in ordered_ids:
                try:
                    uuids.append(_uuid.UUID(rid))
                except ValueError:
                    logger.warning("procedural.legend.bad_rule_id id=%s — skipped", rid)
            async with self.session.begin_nested():
                rows = await load_rules_by_id(self.session, uuids)
            display = self._display_ids(sorted(str(r.id) for r in rows))
            legend, alias = render_rule_legend([(display[str(r.id)], r) for r in rows])
            # Only ids the legend actually explains may be marked up; `alias` is the
            # authority, not `display` (rule_legend_line drops quarantined/empty rules).
            reverse = {rid: d for d, rid in alias.items()}
            logger.info(
                "procedural.legend.built served=%d explained=%d short_ids=%s",
                len(rule_ids),
                len(alias),
                all(len(d) == _SHORT_ID_CHARS for d in alias) if alias else None,
            )
            return legend, alias, reverse
        except Exception:
            logger.warning("procedural served-rules legend build failed (ignored)", exc_info=True)
            return "", {}, {}

    async def _apply_turn_outcomes(
        self,
        turn_outcomes: list[dict[str, Any]],
        turn_map: dict[int, str],
        window_messages: list,
        chat_ctx: ChatContext,
    ) -> None:
        """Label execution traces from L2 Part-3 turn outcomes (design §5.1).

        Same-window-only, savepoint-isolated, fail-open, gated by
        ``procedural_outcome_enabled``. Updates ONLY traces still ``outcome='unknown'``
        and ONLY when the evidencing speaker (quote-anchor-resolved from the window)
        is an admin OR the trace's own ``source_user_id`` — untrusted praise can
        never manufacture a success. ``unclear`` (incl. silence) never writes.
        """
        if not cognitive_config.get_bool("procedural_outcome_enabled"):
            return
        try:
            from sqlalchemy import select

            from kai_memory.storage.models.execution_trace import ExecutionTrace
            from kai_memory.procedural.rule_capture import _norm
            from kai_memory.ports import is_admin_user

            outcome_field = {"success": "success", "failure_redo": "failure", "failure_corrected": "failure"}
            admin_ids = getattr(get_settings(), "telegram_admin_user_ids", "")

            # First pass: resolve each item to (trace_id, mapped, speaker, evidence),
            # so the trace rows load in ONE batched query rather than N round-trips.
            pending: list[tuple[str, str, int | None, str]] = []
            for item in turn_outcomes:
                if not isinstance(item, dict):
                    continue
                mapped = outcome_field.get(str(item.get("outcome") or ""))
                if mapped is None:
                    continue  # 'unclear'/unknown → leave the trace unknown
                try:
                    n = int(str(item.get("turn")))
                except (TypeError, ValueError):
                    continue
                trace_id = turn_map.get(n)
                if not trace_id:
                    continue
                evidence = str(item.get("evidence") or "")
                speaker = self._resolve_evidence_speaker(evidence, window_messages)
                if speaker is None:
                    continue  # evidence must anchor to a trusted speaker in THIS window
                pending.append((trace_id, mapped, speaker, evidence))

            if not pending:
                return

            # Reject conflicting outcomes for the same turn (§ malformed-model-output
            # rule): if the model emitted both 'success' and a failure for one
            # trace_id, neither is trustworthy — drop the trace rather than letting
            # list order pick a winner. Same-outcome repeats collapse to one.
            pending, conflicted = _dedupe_turn_outcomes(pending)
            for tid, outs in conflicted.items():
                logger.warning("procedural.outcome.conflict trace=%s outcomes=%s — skipped", tid, sorted(outs))
            if not pending:
                return
            trace_by_id = {
                str(t.id): t
                for t in (
                    await self.session.execute(
                        select(ExecutionTrace).where(ExecutionTrace.id.in_([_uuid.UUID(p[0]) for p in pending]))
                    )
                ).scalars()
            }

            async with self.session.begin_nested():
                for trace_id, mapped, speaker, evidence in pending:
                    trace = trace_by_id.get(trace_id)
                    if trace is None or trace.outcome != "unknown":
                        continue
                    if not (is_admin_user(speaker, admin_ids) or speaker == trace.source_user_id):
                        continue
                    trace.outcome = mapped
                    trace.how_measured = "l2_turn_reflection"
                    signals = dict(trace.success_signals or {})
                    signals["outcome_evidence"] = _norm(evidence)[:300]
                    signals["outcome_at"] = utc_now().isoformat()
                    trace.success_signals = signals
                    logger.info("procedural.outcome.updated trace=%s outcome=%s", trace_id, mapped)
        except Exception:
            logger.warning("procedural turn-outcome apply failed (ignored)", exc_info=True)

    @staticmethod
    def _served_anchors(window_messages: list) -> tuple[dict[str, str], dict[str, MessageAnchor]]:
        """Split this window's servings into trace anchors and message anchors (DB-629).

        Returns ``(rule_id -> trace_id, rule_id -> message anchor)``. Every served rule
        appears in EXACTLY ONE of the two maps — one verdict, one anchor, one event.

        The message map is why this exists. A tool-using turn carries an
        ``execution_trace_id`` and the ledger reads its trace; a PLAIN reply — where the
        anti-meta-commentary and tone rules actually live — carries ``served_rule_ids``
        and nothing else, so before this every correct ``followed`` verdict on one was
        computed, logged, and dropped on the floor.

        Only the LAST serving per rule is kept, and only one of it. The judge emits ONE
        verdict per rule for the WHOLE window (the response schema has no turn field), so
        anchoring every served turn would count a single observation eight or twelve times
        over — and a lone ``violated`` verdict would then blow ``fail_streak`` past the
        2-failure suspension threshold inside one window and silently retire a rule the
        user taught on purpose. That fan-out predates DB-629 on the trace side (two tool
        turns serving one rule produced two ledger events from one judgment); it is
        collapsed here so both anchor kinds obey the same rule.

        A trace anchor is preferred over a message anchor for the same rule, because the
        trace row is where the existing compliance telemetry lives and where the nightly
        deterministic tool check already looks. That check is unaffected either way: it
        reads ``expected_tools`` against each trace's own steps, not this verdict."""
        trace_anchors: dict[str, str] = {}
        msg_anchors: dict[str, MessageAnchor] = {}
        from kai_memory.procedural_api import is_user_trigger

        last_user_id: int | None = None
        for msg in window_messages:
            role = getattr(msg, "role", None)
            if role == "user":
                if is_user_trigger(msg):
                    last_user_id = getattr(msg, "sender_user_id", None)
                continue
            if role != "assistant":
                continue
            meta = getattr(msg, "message_metadata", None) or {}
            rule_ids = [str(rid) for rid in (meta.get("served_rule_ids") or [])]
            if not rule_ids:
                continue
            trace_id = meta.get("execution_trace_id")
            if trace_id:
                for rid in rule_ids:
                    trace_anchors[rid] = str(trace_id)  # last tool turn in the window wins
                    msg_anchors.pop(rid, None)  # a trace anchor supersedes an earlier plain one
                continue
            msg_id = getattr(msg, "id", None)
            if msg_id is None:
                continue
            actor_id = last_user_id
            if origin := meta.get("procedural_origin_message_id"):
                actor_id = next(
                    (
                        getattr(item, "sender_user_id", None)
                        for item in window_messages
                        if str(getattr(item, "id", "")) == str(origin) and is_user_trigger(item)
                    ),
                    None,
                )
            if (
                meta.get("procedural_is_group") is True
                and meta.get("native_procedural_attribution") == "shared_group_context"
            ):
                actor_id = None
            anchor = MessageAnchor(
                message_id=str(msg_id),
                # The turn's TRIGGERING human, mirroring what procedural_recorder stamps
                # on an execution trace as source_user_id — the assistant row itself has
                # no sender, so the credit boundary has to resolve through the human whose
                # message the reply answers.
                source_user_id=actor_id,
                at=getattr(msg, "created_at", None),
            )
            for rid in rule_ids:
                if rid not in trace_anchors:  # a trace anchor already claimed this rule
                    msg_anchors[rid] = anchor  # last plain serving in the window wins
        return trace_anchors, msg_anchors

    async def _apply_rule_compliance(
        self,
        rule_compliance: list[dict[str, Any]],
        window_messages: list,
        id_alias: dict[str, str] | None = None,
    ) -> None:
        """Persist L2 Part-5 rule-compliance verdicts so the nightly reinforcement ledger
        (§4.2) can read them. Same-window only, savepoint-isolated, fail-open.

        Two anchors (DB-629). A rule served on a tool-using turn attaches to that turn's
        execution trace, unchanged. A rule served ONLY on plain replies has no trace to
        attach to, so its verdict lands in the rule row's
        ``typed_metadata.procedural.compliance_events`` as a pending ledger event keyed by
        the anchor message id — the same shape ``rule_capture`` already uses for
        ``recurrence_events``. ``utility`` is still written ONLY by the nightly pass
        (#241 containment); this path writes a sibling key and never touches it.

        ``not_applicable`` verdicts write nothing (§4.2) — and since DB-487's serve-all
        they are the majority verdict by design, because every standing rule is shown
        every turn whether or not its situation arose. Silence for an inapplicable rule
        is the correct ledger outcome: it is neither a success nor a failure."""
        try:
            from kai_memory.storage.models.execution_trace import ExecutionTrace

            served_map, msg_anchors = self._served_anchors(window_messages)
            if not served_map and not msg_anchors:
                return

            # Collect verdicts per rule_id, then DROP any rule with conflicting
            # verdicts (both followed and violated) — malformed model output must
            # not let list order decide reinforcement (mirror _dedupe_turn_outcomes).
            verdicts_by_rule: dict[str, set[str]] = {}
            seen: dict[str, int] = dict.fromkeys(ALL_VERDICTS, 0) | {"unrecognized": 0}
            alias = id_alias or {}
            for item in rule_compliance:
                if not isinstance(item, dict):
                    continue
                # DB-629: the model answers with the legend's DISPLAY id (short form).
                # Resolve it back to the rule id, still accepting a full id for a model
                # that echoes the underlying UUID (and for the pre-legend call shape).
                raw_id = str(item.get("rule_id") or "").strip().lower()
                rid = alias.get(raw_id, raw_id)
                # Canonicalize a full id the model echoed in its own spelling (no dashes,
                # braces, mixed case) so it joins against the stored ids rather than
                # silently missing. A short/unknown id is left as-is and simply won't match.
                try:
                    rid = str(_uuid.UUID(rid))
                except ValueError:
                    pass
                # str() BEFORE the dict lookup: a malformed model payload can put a list
                # or dict here, and `unhashable in dict` raises TypeError, which the outer
                # handler would turn into "drop every verdict in this window".
                verdict = str(item.get("verdict"))
                # "unrecognized" is its own bucket: a model emitting a verdict outside the
                # schema is a different failure from one judging everything inapplicable,
                # and silently dropping it would make the two look identical.
                seen[verdict if verdict in seen else "unrecognized"] += 1
                if rid and verdict in PERSISTED_VERDICTS:
                    verdicts_by_rule.setdefault(rid, set()).add(verdict)
            # Serve-all makes not_applicable the majority verdict by design, so the
            # persisted count alone can no longer tell "Part 5 judged nothing" apart from
            # "Part 5 judged everything inapplicable". Log the incoming distribution —
            # this is the signal that says whether serve-all is feeding the ledger at all.
            logger.info(
                "procedural.compliance.verdicts served_rules=%d followed=%d violated=%d "
                "not_applicable=%d misapplied=%d unrecognized=%d",
                # Every served rule, both anchor kinds — the plain-turn half is the half
                # that used to be invisible here.
                len(served_map) + len(msg_anchors),
                seen[VERDICT_FOLLOWED],
                seen[VERDICT_VIOLATED],
                seen[VERDICT_NOT_APPLICABLE],
                seen[VERDICT_MISAPPLIED],
                seen["unrecognized"],
            )
            # Any two DISTINCT verdicts for one rule are contradictory — list order must
            # never decide what the ledger reads (mirror _dedupe_turn_outcomes).
            conflicted = {rid for rid, vs in verdicts_by_rule.items() if len(vs) > 1}
            for rid in conflicted:
                logger.warning(
                    "procedural.compliance.conflict rule=%s verdicts=%s — skipped",
                    rid,
                    "+".join(sorted(verdicts_by_rule[rid])),
                )

            # trace_id -> {rule_id: verdict}, non-conflicting persisted verdicts only.
            # rule_id -> (verdict, anchor) for rules served only on plain turns (DB-629).
            updates: dict[str, dict[str, str]] = {}
            msg_updates: dict[str, tuple[str, MessageAnchor]] = {}
            for rid, verdicts in verdicts_by_rule.items():
                if rid in conflicted:
                    continue
                verdict = next(iter(verdicts))
                trace_id = served_map.get(rid)
                if trace_id:
                    updates.setdefault(trace_id, {})[rid] = verdict
                    continue
                # The ledger only ever acts on followed/violated; `misapplied` is inert
                # telemetry that rides along on a trace row that already exists. There is
                # no reader for it here, so minting a pending event for it would be
                # storage for nobody — it stays in the counter line above.
                anchor = msg_anchors.get(rid)
                if anchor is not None and verdict in LEDGER_VERDICTS:
                    msg_updates[rid] = (verdict, anchor)
            if not updates and not msg_updates:
                return

            trace_ids: list[_uuid.UUID] = []
            for tid in updates:
                try:
                    trace_ids.append(_uuid.UUID(tid))
                except ValueError:
                    continue

            async with self.session.begin_nested():
                if trace_ids:
                    traces = (
                        await self.session.execute(select(ExecutionTrace).where(ExecutionTrace.id.in_(trace_ids)))
                    ).scalars()
                    for trace in traces:
                        # .get(), not [] — a stored id whose spelling does not round-trip
                        # would raise here and abort the savepoint, taking the pending
                        # message-anchored writes below down with it.
                        trace_verdicts = updates.get(str(trace.id))
                        if trace_verdicts is None:
                            logger.warning("procedural.compliance.unmatched_trace trace=%s — skipped", trace.id)
                            continue
                        sig = dict(trace.success_signals or {})
                        comp = dict(sig.get("rule_compliance") or {})
                        comp.update(trace_verdicts)
                        sig["rule_compliance"] = comp
                        trace.success_signals = sig
                        logger.info(
                            "procedural.compliance.persisted trace=%s verdicts=%d",
                            trace.id,
                            len(trace_verdicts),
                        )
                if msg_updates:
                    from kai_memory.procedural.rule_capture import record_message_anchored_compliance

                    await record_message_anchored_compliance(
                        self.session, msg_updates, getattr(get_settings(), "telegram_admin_user_ids", "")
                    )
        except Exception:
            logger.warning("procedural rule-compliance persist failed (ignored)", exc_info=True)

    @staticmethod
    def _resolve_evidence_speaker(evidence: str, window_messages: list) -> int | None:
        """Return the sender_user_id of the user row this evidence fragment quotes,
        or None if it anchors nowhere (design §5.1 — quote-anchored, same-window).

        Anchors through ``_anchorable_text`` so the SAME firewall as capture applies:
        forwarded rows and reply-quote spans can never supply outcome evidence."""
        from kai_memory.procedural_api import resolve_evidence_speaker

        return resolve_evidence_speaker(evidence, window_messages)

    async def _load_existing_context(
        self,
        conversation_text: str,
        priority_candidates: list[_uuid.UUID] | None = None,
    ) -> str:
        """Load relevant existing memories for novelty assessment."""
        try:
            results = await self.memory_svc.search_traces(conversation_text[:500], limit=MAX_EXISTING_MEMORIES)
        except Exception:
            return "(no existing memories available)"

        lines = []
        # Prepend priority candidates (recently retrieved — possible contradiction targets)
        if priority_candidates:
            lines.append("# Pay special attention to possible contradictions with these recently-retrieved memories:")
            try:
                priority_traces = await self.memory_svc.get_traces_batch(priority_candidates[:10])
                for trace in priority_traces:
                    if trace.invalidated_at is None:
                        lines.append(f"- [id={trace.id}] [{trace.memory_type}] {trace.content}")
            except Exception:
                logger.debug("Failed to load priority contradiction candidates", exc_info=True)
            lines.append("")

        if not results:
            if not lines:
                return "(no existing memories)"
            return "\n".join(lines)
        for trace, _score in results:
            lines.append(f"- [id={trace.id}] [{trace.memory_type}] {trace.content}")
        return "\n".join(lines)

    async def _load_participant_context(self, chat_ctx: ChatContext) -> tuple[str, dict[str, dict[str, Any]]]:
        """Load recent chat participants for L2 entity resolution."""
        participants: list[dict] = []

        if chat_ctx.chat_id is not None:
            query = select(SocialContext).where(SocialContext.chat_id == chat_ctx.chat_id)
            if chat_ctx.topic_id is None:
                query = query.where(SocialContext.topic_id.is_(None))
            else:
                query = query.where(SocialContext.topic_id == chat_ctx.topic_id)
            result = await self.session.execute(query)
            social_context = result.scalar_one_or_none()
            raw_participants = getattr(social_context, "participants", None)
            if isinstance(raw_participants, list):
                participants = list(raw_participants)

        if not participants and chat_ctx.extra_participants:
            participants = list(chat_ctx.extra_participants)

        if not participants:
            return "(no known chat participants)", {}

        if len(participants) > 20:
            cutoff = utc_now() - timedelta(days=14)
            participants = [p for p in participants if self._participant_seen_after(p, cutoff)]

        participant_map: dict[str, dict[str, Any]] = {}
        lines = []
        for participant in participants:
            if not isinstance(participant, dict):
                continue
            tg_id = participant.get("telegram_user_id")
            email = participant.get("email")
            name = (participant.get("name") or "").strip()
            if not name:
                continue
            key = str(tg_id) if tg_id is not None else email or name
            if not key:
                continue
            participant_map[key] = participant
            username = participant.get("username")
            id_part = f" (@{username})" if username else (f" ({email})" if email else "")
            lines.append(f"- {key}: {name}{id_part}")

        if not lines:
            return "(no known chat participants)", {}
        return "\n".join(lines), participant_map

    @staticmethod
    def _participant_seen_after(participant: dict[str, Any], cutoff: datetime) -> bool:
        last_seen = participant.get("last_seen_at")
        if not last_seen:
            # Include-on-missing: covers legacy rows that pre-date this field and
            # meeting extra_participants that ride in without timestamps. Stricter
            # "exclude unknowns" would silently hide those participants from the
            # extraction prompt for large chats (>20 members).
            return True
        try:
            parsed = datetime.fromisoformat(str(last_seen).replace("Z", "+00:00"))
        except ValueError:
            return False
        if parsed.tzinfo is None and cutoff.tzinfo is not None:
            parsed = parsed.replace(tzinfo=cutoff.tzinfo)
        return parsed >= cutoff

    async def _call_haiku(
        self,
        conversation_text: str,
        thinking_text: str,
        existing_memories: str,
        *,
        participant_context: str = "(no known chat participants)",
        procedural_enabled: bool = False,
        served_rules_legend: str = "",
    ) -> dict[str, Any]:
        """Call Haiku to extract memories and social experiences.

        Returns dict with keys 'memories', 'social_experiences', and (when
        ``procedural_enabled``) 'turn_outcomes', 'behavior_rules', 'rule_compliance'.
        """
        prompt = EXTRACTION_PROMPT.format(
            agent_display_name=get_agent_identity().display_name,
            conversation=conversation_text,
            thinking_blocks=thinking_text,
            existing_memories=existing_memories,
            participant_context=participant_context,
            current_timestamp=utc_now().isoformat(),
            schema_catalog=render_schema_for_prompt(),
            slot_durability_rubric=SLOT_DURABILITY_RUBRIC,
            procedural_block=_PROCEDURAL_BLOCK if procedural_enabled else "",
            procedural_response_note=_PROCEDURAL_RESPONSE_NOTE if procedural_enabled else "",
            # Leading newline only when there IS a legend, so the off path stays
            # byte-identical to pre-DB-629 (the placeholder collapses to nothing).
            served_rules_legend=f"\n{served_rules_legend}\n" if served_rules_legend else "",
        )

        provider = get_provider_for_model(get_settings(), get_model_roles().extraction)
        span = start_generation_span("memory/l2-extraction", get_model_roles().extraction)
        # 11000 retains the old 8192-token headroom with Haiku 5.5's ~30%
        # larger token counts. The old cap allowed ~15-20 memory entries with
        # full entity lists. The call_llm_text default (2048) truncates
        # mid-JSON on meetings with substantial extractable content —
        # observed 2026-04-15 11:44 iOS team meeting produced 11 complete
        # memory entries then cut off mid-string at token 2048, breaking
        # all JSON parser fallbacks.
        llm_text_fn = self._llm_text_fn or call_llm_text
        raw_text = await llm_text_fn(provider, prompt, max_tokens=11000, span=span, thinking_mode="disabled")
        if not raw_text:
            # Failed/truncated inference must roll back the run and retain its cursor.
            # A completed JSON response with empty lists is handled normally below.
            raise RuntimeError("memory extraction returned no completion")
        parsed = parse_json_object_response(raw_text)

        def _as_list(key: str) -> list:
            val = parsed.get(key)
            return val if isinstance(val, list) else []

        # Normalize: always return the base keys as lists; procedural keys ride
        # along only when they were requested (else they stay empty).
        return {
            "memories": _as_list("memories"),
            "social_experiences": _as_list("social_experiences"),
            "turn_outcomes": _as_list("turn_outcomes"),
            "behavior_rules": _as_list("behavior_rules"),
            "rule_compliance": _as_list("rule_compliance"),
        }

    async def _resolve_matched_participant_entities(
        self,
        items: list[dict[str, Any]],
        participant_map: dict[str, dict[str, Any]],
    ) -> dict[EntityMapKey, Any]:
        """Bind extracted person entities to validated Telegram participants."""
        if not participant_map:
            return {}

        entity_map: dict[EntityMapKey, Any] = {}
        for item in items:
            for raw_entity in item.get("entities", []):
                if not isinstance(raw_entity, dict):
                    continue
                key = raw_entity.get("matched_participant_key")
                participant = participant_map.get(str(key)) if key is not None else None
                if participant is None:
                    raw_entity["matched_participant_key"] = None
                    continue
                name = (raw_entity.get("name") or participant.get("name") or "").strip()
                if not name:
                    continue
                entity_type = raw_entity.get("type", "person") or "person"
                validated_type: EntityType = entity_type if entity_type in VALID_ENTITY_TYPES else "person"  # type: ignore[assignment]
                # review-r1 fix 2: matched-participant entities ARE people by
                # definition; mutate raw_entity['type'] so _store_item, which
                # defaults missing types to 'topic' (v2a §5.2), reads the same
                # value the resolver used as the entity_map key. Without this
                # the lookup misses on (name, 'topic') vs (name, 'person')
                # and the Telegram-user binding is silently lost.
                raw_entity["type"] = validated_type
                tg_id = participant.get("telegram_user_id")
                # v2a §5.5: thread the item's content as source_text so the
                # reverse-direction alias scan (ASCII name + Cyrillic source
                # token) can recover the missing form before the DB lookup.
                source_text = (item.get("content") or "").strip() or None
                entity = await self.memory_svc.find_or_create_entity(
                    name,
                    validated_type,
                    telegram_user_id=tg_id if isinstance(tg_id, int) else None,
                    source_text=source_text,
                )
                entity_map[(name.lower(), validated_type)] = entity
        return entity_map

    @staticmethod
    def _first_party_subject_ids(resolved_entities: list[Any]) -> frozenset[str]:
        """Entity ids that count as the FIRST-PARTY subject for the P-1 durability gate.

        P-1 (operator-confirmed 2026-07-20): first-party = a subject entity whose
        ``telegram_user_id`` is a configured admin — the EXISTING trust boundary
        (``is_admin_user``). Kai invents no owner concept and stays universal;
        multi-admin ⇒ all admins are first-party (bounded, few, already the trust gate).
        Durable family/health facts (key_relationships, chronic_health_conditions,
        allergies) are floored ONLY for these subjects; a third party's family/health is
        captured but stays transient. Config-read lives HERE (the call-site seam), so the
        pure validator gets only the derived id set.
        """
        from kai_memory.ports import is_admin_user  # local import: avoid circular import

        admin_str = getattr(get_settings(), "telegram_admin_user_ids", "")
        return frozenset(
            str(e.id)
            for e in resolved_entities or []
            if getattr(e, "id", None) is not None and is_admin_user(getattr(e, "telegram_user_id", None), admin_str)
        )

    async def _store_item(
        self,
        item: dict[str, Any],
        chat_ctx: ChatContext,
        stats: dict[str, int],
        embedding: list[float] | None = None,
        entity_map: dict[EntityMapKey, Any] | None = None,
    ) -> None:
        """Store one extracted memory item."""
        content = item.get("content", "").strip()
        if not content:
            return

        raw_type = item.get("memory_type", "fact")
        if raw_type in MEMORY_TYPES:
            memory_type = raw_type
        else:
            normalized = _MEMORY_TYPE_NORMALIZER.get(raw_type)
            # Hotfix (2026-05-01): basicConfig's default formatter drops
            # `extra={}` keys — inline-serialize the payload as the message
            # so CloudWatch sees the full JSON. See memory_priming.py for
            # the same pattern on memory.retrieve.complete.
            if normalized:
                logger.info(
                    "memory.encoding.type_normalized %s",
                    json.dumps(
                        {
                            "event_type": "memory.encoding.type_normalized",
                            "raw": raw_type,
                            "resolved": normalized,
                        }
                    ),
                )
                memory_type = normalized
            else:
                logger.warning(
                    "memory.encoding.type_coerced %s",
                    json.dumps(
                        {
                            "event_type": "memory.encoding.type_coerced",
                            "raw": raw_type,
                            "resolved": "fact",
                        }
                    ),
                )
                memory_type = "fact"
        significance = max(1, min(5, int(item.get("significance", 3))))

        # INC 3 A3: resolve entities BEFORE building typed_metadata so each fact
        # can be attributed to a canonical entity id (subject → id). v2a §5.2:
        # unknown / missing entity types default to 'topic', NOT 'person'.
        if entity_map is None:
            entity_map = {}
        resolved_entities: list[Any] = []
        for raw_entity in item.get("entities", []):
            raw_data: dict = {}
            if isinstance(raw_entity, dict):
                name = (raw_entity.get("name") or "").strip()
                entity_type = raw_entity.get("type")
                provided_data = raw_entity.get("data")
                if isinstance(provided_data, dict):
                    # Stable self identity is host-owned, never model-supplied data.
                    raw_data = {key: value for key, value in provided_data.items() if key != AGENT_IDENTITY_DATA_KEY}
                if not entity_type:
                    logger.debug("Entity has no type; defaulting to 'topic'")
                    entity_type = "topic"
            elif isinstance(raw_entity, str):
                name = raw_entity.strip()
                entity_type = "topic"
                logger.debug("Entity came as plain string; defaulting to 'topic'")
            else:
                continue
            if not name:
                continue
            if entity_type not in VALID_ENTITY_TYPES:
                logger.warning("Unknown entity type; coercing to 'topic'")
                entity_type = "topic"
            validated_type: EntityType = entity_type  # type: ignore[assignment]
            entity = entity_map.get((name.lower(), validated_type))
            if validated_type == "agent" and get_agent_identity().recognizes(name):
                entity = await ensure_agent_identity(self.session)
                entity_map[(name.lower(), validated_type)] = entity
            if entity is None:
                # v2a §5.5: thread the item's content as source_text so the
                # reverse-direction alias scan ("Misha" + Cyrillic "Миша" in
                # source → bind both forms) fires.
                source_text = (item.get("content") or "").strip() or None
                if raw_data:
                    entity = await self.memory_svc.find_or_create_entity(
                        name, validated_type, data=raw_data, source_text=source_text
                    )
                else:
                    entity = await self.memory_svc.find_or_create_entity(name, validated_type, source_text=source_text)
                # Memoize so a repeated entity across items skips the DB round-trip.
                entity_map[(name.lower(), validated_type)] = entity
            resolved_entities.append(entity)

        # A3: stamp subject_id (canonical entity the fact is about) onto each
        # fact, matched against the resolved entities; keeps the raw subject.
        attribute_fact_subjects(item.get("facts", []), resolved_entities)

        # INC 3 + D-1: validate each fact's slot (degrade bogus/unresolvable slots to
        # "off_schema" without dropping the fact) and COMPUTE durability from the final
        # slot. ``first_party_subject_ids`` gates the P-1 family/health subset: floored
        # only when the subject is a first-party (admin) entity — a third party's
        # family/health is captured but stays transient (Kai is universal, ownerless).
        first_party_subject_ids = self._first_party_subject_ids(resolved_entities)
        _validate_and_default_fact_slots(
            item.get("facts", []), resolved_entities, first_party_subject_ids=first_party_subject_ids
        )

        # P-1 telemetry (refinement 2): count captured-but-not-floored third-party
        # family/health facts. If the eval/lived-experience shows this loses real facts
        # (e.g. the admin's mother's OWN health), the data tells us whether to expand
        # first-party to degree-1 — we deliberately do NOT build degree-1 now.
        stats["third_party_family_transient"] = stats.get("third_party_family_transient", 0) + sum(
            1
            for f in item.get("facts", [])
            if isinstance(f, dict) and f.get("slot") in FIRST_PARTY_DURABLE_SLOTS and f.get("durability") != "durable"
        )

        trace = await self.memory_svc.store_trace(
            content=content,
            memory_type=memory_type,
            significance=significance,
            source_type="conversation",
            source_chat_id=chat_ctx.chat_id,
            source_topic_id=chat_ctx.topic_id,
            emotion_valence=item.get("emotion_valence"),
            emotion_arousal=item.get("emotion_arousal"),
            embedding=embedding,
            # Sub-design C INC 1: L2-inline traces carry the rich extraction
            # blob + their origin. store_trace ignores these when the write
            # flag is off, so this is harmless until typed_metadata_write_enabled.
            typed_metadata=_build_typed_metadata(item),
            source_path="l2_inline",
        )
        stats["traces_created"] += 1

        # Handle contradiction — two gates (DB-591) before the old trace is invalidated.
        contradicts_id = item.get("contradicts_memory_id")
        if contradicts_id:
            await self._maybe_invalidate_contradicted(
                contradicts_id,
                new_trace_id=trace.id,
                item=item,
                chat_ctx=chat_ctx,
                resolved_entities=resolved_entities,
                stats=stats,
            )

        # Link the already-resolved entities (resolution + subject attribution
        # happened above, before the trace's typed_metadata was built). 'about'
        # role; 'topic'-default typing is applied during resolution.
        for entity in resolved_entities:
            await self.memory_svc.link_trace_to_entity(trace.id, entity.id, "about")
            stats["entities_found"] += 1

    @staticmethod
    def _fact_subject_ids(facts: Any, resolved_entities: list[Any]) -> frozenset[str]:
        """Canonical entity ids the item's facts are ABOUT (its subjects), or empty.

        Primary source is ``facts[].subject_id`` — stamped just above by
        ``attribute_fact_subjects`` (A3), so it is the resolved canonical id, not a
        raw name. Fallback for a sparse/legacy item with no facts: the PERSON
        entities the item resolved. Topics/organizations are deliberately excluded
        from the fallback — two facts about different people routinely share a topic
        ("work", "health"), and an overlap there is not a shared subject.

        Empty means UNKNOWN, never "no subject": the caller must not read it as a
        mismatch (see ``_maybe_invalidate_contradicted``).
        """
        ids = _subject_ids_from_facts(facts)
        if ids:
            return frozenset(ids)
        return frozenset(
            str(e.id)
            for e in resolved_entities or []
            if getattr(e, "id", None) is not None and getattr(e, "entity_type", None) == "person"
        )

    async def _trace_subject_ids(self, trace: Any) -> frozenset[str]:
        """The subjects an EXISTING trace is about — same shape as ``_fact_subject_ids``.

        ``typed_metadata.facts[].subject_id`` when the trace carries the INC-1 blob;
        otherwise the person entities linked to it, so a pre-``typed_metadata`` trace
        (the majority in prod until the write flag is fully rolled out) still has a
        subject rather than reading as unknown.
        """
        tm = getattr(trace, "typed_metadata", None)
        facts = tm.get("facts") if isinstance(tm, dict) else None
        # Persisted subject_ids are RAW ids frozen at write time — the link source
        # resolves merges, and this one has to as well or the gate is only half
        # merge-aware. The gap widens as typed_metadata_write_enabled rolls out,
        # because newer traces are exactly the ones carrying subject_id.
        parsed: list[_uuid.UUID] = []
        for raw in _subject_ids_from_facts(facts):
            try:
                parsed.append(_uuid.UUID(raw))
            except (ValueError, AttributeError, TypeError):
                continue
        if parsed:
            return frozenset(str(eid) for eid in await self.memory_svc.resolve_entity_survivors(parsed))
        # No usable stamped subject (absent, or every value unparseable) — fall back
        # to the linked person entities rather than reporting "no subject".
        trace_id = getattr(trace, "id", None)
        if trace_id is None:
            return frozenset()
        return frozenset(str(eid) for eid in await self.memory_svc.get_person_entity_ids_for_trace(trace_id))

    async def _maybe_invalidate_contradicted(
        self,
        contradicts_id: Any,
        *,
        new_trace_id: _uuid.UUID,
        item: dict[str, Any],
        chat_ctx: ChatContext,
        resolved_entities: list[Any],
        stats: dict[str, int],
    ) -> None:
        """Invalidate the trace Haiku flagged as contradicted — behind two gates (DB-591).

        Haiku's ``contradicts_memory_id`` is an ADVISORY signal over a semantically
        retrieved candidate set, and invalidation is destructive: retrieval filters
        ``invalidated_at IS NULL``, so obeying a wrong flag silently deletes a memory.
        The 2026-08-14 incident is the proof — a synthetic capture-probe statement
        ("my legal name is …") shared the ``full_name`` slot with the admin's real
        legal-name trace, Haiku called it a contradiction, and Kai lost the fact.

        Gate A (hard): an EPHEMERAL probe write may never invalidate anything. The
        probe harness owns only the rows it creates; mutating pre-existing memory is
        outside that contract in every case, so this returns before the lookup.

        Gate B (subject): a contradiction is a claim about ONE subject. Invalidate
        only when the two subjects are not provably different. "Provably" is load-
        bearing and asymmetric on purpose: an unknown subject on either side keeps
        today's behavior (invalidate) rather than silently disabling the supersede
        path where subjects are unresolved, and is logged so the prevalence is
        measurable instead of assumed.
        """
        # Gate A — ephemeral probe writes are inert against real memory.
        if is_ephemeral_probe_chat(chat_ctx.chat_id):
            logger.warning(
                "memory.encoding.contradiction_skipped_ephemeral_probe %s",
                json.dumps(
                    {
                        "event_type": "memory.encoding.contradiction_skipped_ephemeral_probe",
                        "chat_id": chat_ctx.chat_id,
                        "topic_id": chat_ctx.topic_id,
                        "contradicts_memory_id": str(contradicts_id),
                        "new_trace_id": str(new_trace_id),
                    }
                ),
            )
            return

        try:
            old_id = _uuid.UUID(contradicts_id)
        except (ValueError, AttributeError, TypeError):
            logger.debug("Invalid contradicts_memory_id: %s", contradicts_id)
            return

        old_trace = await self.memory_svc.get_trace(old_id)
        if old_trace is None:
            logger.debug("Skipping contradiction — trace %s does not exist", old_id)
            return

        # Gate B — subject-gated. Resolving the OLD subject costs a query, so it is
        # only paid when the NEW subject is known enough to compare against.
        new_subjects = self._fact_subject_ids(item.get("facts"), resolved_entities)
        old_subjects = await self._trace_subject_ids(old_trace) if new_subjects else frozenset()
        if new_subjects and old_subjects:
            if new_subjects.isdisjoint(old_subjects):
                logger.warning(
                    "memory.encoding.contradiction_skipped_subject_mismatch %s",
                    json.dumps(
                        {
                            "event_type": "memory.encoding.contradiction_skipped_subject_mismatch",
                            "chat_id": chat_ctx.chat_id,
                            "contradicts_memory_id": str(old_id),
                            "new_trace_id": str(new_trace_id),
                            "new_subject_ids": sorted(new_subjects),
                            "old_subject_ids": sorted(old_subjects),
                        }
                    ),
                )
                stats["contradictions_skipped_subject_mismatch"] = (
                    stats.get("contradictions_skipped_subject_mismatch", 0) + 1
                )
                return
        else:
            # Not a mismatch — an UNVERIFIABLE one. Counted so the gate's real
            # coverage is a measured number, not a hope.
            logger.info(
                "memory.encoding.contradiction_subject_unknown %s",
                json.dumps(
                    {
                        "event_type": "memory.encoding.contradiction_subject_unknown",
                        "chat_id": chat_ctx.chat_id,
                        "contradicts_memory_id": str(old_id),
                        "new_trace_id": str(new_trace_id),
                        "side": "new" if not new_subjects else "old",
                    }
                ),
            )
            stats["contradictions_subject_unknown"] = stats.get("contradictions_subject_unknown", 0) + 1

        await self.memory_svc.invalidate_trace(
            old_id,
            invalidated_by=new_trace_id,
            confidence=0.8,
            invalidation_source="l2_inline",
        )
        stats["contradictions"] += 1

    _VALID_SIGNAL_TYPES = frozenset({"correction", "boundary", "positive", "withdrawal", "tone_mismatch", "repair"})

    async def _store_social_experiences(
        self,
        chat_ctx: ChatContext,
        experiences: list[dict[str, Any]],
        stats: dict[str, int],
    ) -> None:
        """Persist social experiences linked to this chat's social context."""
        # Resolve the social_context_id (created by _ensure_social_context just before)
        query = select(SocialContext.id).where(SocialContext.chat_id == chat_ctx.chat_id)
        if chat_ctx.topic_id is None:
            query = query.where(SocialContext.topic_id.is_(None))
        else:
            query = query.where(SocialContext.topic_id == chat_ctx.topic_id)
        result = await self.session.execute(query)
        context_id = result.scalar_one_or_none()
        if context_id is None:
            logger.warning("No social context found for experience storage")
            return

        for exp in experiences:
            experience_text = (exp.get("experience") or "").strip()
            if not experience_text:
                continue
            learned_norm = exp.get("learned_norm")
            if learned_norm is not None:
                learned_norm = str(learned_norm).strip() or None
            signal_type = exp.get("signal_type")
            if signal_type not in self._VALID_SIGNAL_TYPES:
                signal_type = None
            try:
                self.session.add(
                    SocialExperience(
                        social_context_id=context_id,
                        experience=experience_text,
                        learned_norm=learned_norm,
                        signal_type=signal_type,
                        salience=0.7,
                    )
                )
                stats["social_experiences"] += 1
            except Exception:
                logger.warning("Failed to store social experience")
        await self.session.flush()

    async def _ensure_social_context(
        self,
        chat_ctx: ChatContext,
    ) -> None:
        """Create social_contexts row on first extraction per (chat_id, topic_id).

        Headless venues (event, heartbeat) have chat_id=None by design and
        no human counterpart whose social context we'd be tracking. The
        social_contexts table also has a NOT NULL constraint on chat_id, so
        attempting an upsert would raise IntegrityError. Skip cleanly.
        """
        if chat_ctx.chat_id is None:
            return
        chat_id = chat_ctx.chat_id
        topic_id = chat_ctx.topic_id
        context_type = "group" if chat_ctx.is_group else "personal"
        now = utc_now()
        context_name = chat_ctx.chat_title or chat_ctx.topic_title
        query = select(SocialContext).where(SocialContext.chat_id == chat_id)
        if topic_id is None:
            query = query.where(SocialContext.topic_id.is_(None))
        else:
            query = query.where(SocialContext.topic_id == topic_id)
        result = await self.session.execute(query)
        social_context = result.scalar_one_or_none()
        last_active_at = getattr(social_context, "last_active_at", None)
        if social_context is not None and last_active_at is not None:
            if last_active_at.tzinfo is None and now.tzinfo is not None:
                last_active_at = last_active_at.replace(tzinfo=now.tzinfo)
            if (
                abs((now - last_active_at).total_seconds()) <= 60
                and (social_context.context_name or None) == (context_name or None)
                and social_context.context_type == context_type
            ):
                if chat_ctx.sender_user_id is not None:
                    await self._upsert_participant(chat_ctx, now)
                return

        values = {
            "chat_id": chat_id,
            "topic_id": topic_id,
            "context_type": context_type,
            "last_active_at": now,
        }
        if context_name:
            values["context_name"] = context_name
        update_values: dict[str, Any] = {"last_active_at": now}
        if context_name:
            update_values["context_name"] = context_name
        stmt = pg_insert(SocialContext).values(**values)
        # NULL != NULL in SQL, so the composite (chat_id, topic_id) index
        # can't match when topic_id IS NULL.  Target the partial unique index
        # via index_elements + index_where (ON CONFLICT ON CONSTRAINT needs a
        # named CONSTRAINT, but migration 042 created a unique INDEX).
        if topic_id is None:
            stmt = stmt.on_conflict_do_update(
                index_elements=["chat_id"],
                index_where=SocialContext.topic_id.is_(None),
                set_=update_values,
            )
        else:
            stmt = stmt.on_conflict_do_update(
                index_elements=["chat_id", "topic_id"],
                set_=update_values,
            )
        await self.session.execute(stmt)

        if chat_ctx.sender_user_id is not None:
            await self._upsert_participant(chat_ctx, now)

    async def _upsert_participant(self, chat_ctx: ChatContext, now: datetime) -> None:
        """Add or update the current sender in the social context participant list."""
        if chat_ctx.sender_user_id is None:
            return

        participant = {
            "telegram_user_id": chat_ctx.sender_user_id,
            "name": chat_ctx.sender_name,
            "username": chat_ctx.sender_username,
            "last_seen_at": now.isoformat(),
        }
        result = await self.session.execute(
            text("""
                UPDATE social_contexts
                SET participants = (
                    SELECT COALESCE(jsonb_agg(elem), '[]'::jsonb)
                    FROM (
                        SELECT elem
                        FROM jsonb_array_elements(COALESCE(participants, '[]'::jsonb)) AS elem
                        WHERE elem->>'telegram_user_id' IS DISTINCT FROM :tg_id_text
                        UNION ALL
                        SELECT CAST(:new_participant AS jsonb)
                    ) sub
                ),
                updated_at = CAST(:now AS timestamptz)
                WHERE chat_id = :chat_id
                  AND topic_id IS NOT DISTINCT FROM :topic_id
            """),
            {
                "now": now,
                "tg_id_text": str(chat_ctx.sender_user_id),
                "new_participant": json.dumps(participant),
                "chat_id": chat_ctx.chat_id,
                "topic_id": chat_ctx.topic_id,
            },
        )
        # asyncpg may report rowcount=-1 when the driver can't determine the
        # count; we only warn on confirmed 0 to avoid false alarms.
        rowcount = getattr(result, "rowcount", None)
        if rowcount == 0:
            logger.warning(
                "No social context found for participant registration (chat_id=%s, topic_id=%s)",
                chat_ctx.chat_id,
                chat_ctx.topic_id,
            )
