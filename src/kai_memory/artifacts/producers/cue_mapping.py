"""CueMappingProducer — design §3.3.3.

Maps short user phrases ("наша собака", "the meeting") to the
EntitySchema artifacts they should hydrate. The design's preferred
source is the ``memory.retrieve.complete`` priming-event log on
CloudWatch — Wave 2 ships a **DB fallback** (memory_traces +
memory_entity_links sampling) per the §3.3.3 deferral note, on the
grounds that CloudWatch query infrastructure is heavier than the
shadow-period iteration budget.

The fallback approximates dominance by counting how many distinct
entities link to each trace — single-entity-linked traces are treated
as the priming-event's "dominant entity" surrogate. N-grams come from
``memory_traces.content`` (agent-stored memory text), not the original
user query — Wave 3 may swap to CloudWatch once the consumer ships.

Caps + thresholds (cognitive_config keys):
- ``cue_min_cooccurrence`` (default 5): min (phrase, entity) co-occurrences
- ``cue_adjudication_batch_max`` (default 10): max Sonnet calls per run
- ``cue_high_confidence_threshold`` (default 0.8): Sonnet confidence floor

Supersession key: ``(artifact_type='CueMapping', subject_entity_id=NULL,
dimension=<cue_pattern>)``. Each cue is its own row per §3.3.3.
"""

from __future__ import annotations

from kai_memory.host_calls import get_model_roles

import collections
import json
import logging
import re
import uuid as _uuid
from datetime import timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.host_calls import (
    call_llm_text,
    get_provider_for_model,
    get_settings,
    start_generation_span,
    utc_now,
)
from kai_memory.store import cognitive_config
from kai_memory.artifact_schemas import CueMappingPayload
from kai_memory.artifacts.producer import BaseProducer, ProducerSubject
from kai_memory.artifacts.types import ArtifactType, ProducedArtifact

logger = logging.getLogger("app.services.memory.artifacts.producers.cue_mapping")


_LOOKBACK_DAYS = 30
_NGRAM_MIN = 1
_NGRAM_MAX = 3

# Stop-list — short, English+Russian function words that pollute n-gram
# tallies. Wave 2 keeps this minimal; n-gram dominance does most of the
# filtering, and per-candidate Sonnet adjudication catches the rest.
# Cyrillic stop-words intentionally use Cyrillic glyphs that look like
# Latin counterparts; the homoglyph suppression sits inline on the
# Russian-tokens line so the literal comparisons stay correct.
#
# 2026-05-17 expansion — pronouns + role/category nouns + auxiliary verbs
# added after CueMapping rejection logs from 2026-05-16 showed Sonnet
# rejecting these at 95%+ confidence (e.g. "his", "may", "assistant").
# The DEBUG→INFO log flip in PR #141 made the noise visible; this filter
# stops paying Sonnet tokens to re-discover what we already know.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "for",
        "at",
        "by",
        "with",
        "from",
        "this",
        "that",
        "these",
        "those",
        "it",
        "as",
        "be",
        "been",
        "was",
        "were",
        # English pronouns
        "he",
        "she",
        "they",
        "them",
        "him",
        "her",
        "his",
        "hers",
        "their",
        "theirs",
        "i",
        "me",
        "my",
        "mine",
        "we",
        "us",
        "our",
        "ours",
        "you",
        "your",
        "yours",
        # English aux/modal verbs Sonnet flagged as common-noise.
        "may",
        "might",
        "can",
        "could",
        "would",
        "should",
        "shall",
        "will",
        "must",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        # Role / category nouns — sender/receiver labels and bot
        # vocabulary. All confirmed-False in the 2026-05-16 rejection
        # batch with confidence ≥ 0.95 ("'assistant' is a generic role").
        "assistant",
        "user",
        "agent",
        "system",
        "bot",
        "model",
        "ai",
        "llm",
        # 2026-05-20 v2 additions — confirmed via 24h CueMapping rejection
        # diagnostic (3 L3 passes, 9 unique noise phrases each). All
        # rejected at confidence >= 0.90 with rationale "generic English
        # connector / generic noun, not a disambiguating cue".
        # Negation that #144 missed.
        "not",
        # Comparison / connector words.
        "than",
        "rather",
        # Generic activity nouns — pollute n-gram tallies on health /
        # workout traces without naming a specific entity.
        "training",
        "session",
        "recovery",
        # Russian
        "и",
        "в",
        "не",
        "на",
        "что",
        "он",
        "она",
        "оно",
        "они",
        "это",
        "как",
        "у",  # noqa: RUF001
        "к",
        "с",  # noqa: RUF001
        "по",
        "о",  # noqa: RUF001
        "за",
        "от",
        "из",
        # Russian pronouns. Cyrillic glyphs that homoglyph to Latin
        # letters carry the standard RUF001 suppression per the existing
        # convention earlier in this stop-list.
        "его",  # noqa: RUF001
        "её",
        "ее",  # noqa: RUF001
        "их",
        "ему",
        "ей",
        "им",
        "мне",
        "тебя",
        "нас",
        "вас",
        "я",
        "ты",
        "мы",
        "вы",
    }
)

_TOKEN_RE = re.compile(r"[\wЀ-ӿ]+", re.UNICODE)

# Year token — 4-digit string in 1900-2099. Tighter than "any 4-digit
# number" so codes like "4242" don't get blanket-dropped.
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")

# Any-purpose digit check — short numeric strings (≤4 chars all-digit)
# carry essentially zero entity-disambiguation signal.
_ALL_DIGITS_RE = re.compile(r"^\d+$")


def _is_noise_token(token: str) -> bool:
    """Discard tokens that carry no entity-disambiguation signal.

    Catches: existing len<2 / stopword filters PLUS years (1900-2099)
    and short all-digit strings. Applied inside ``_extract_ngrams``
    before the n-gram cross-product runs, so noise tokens never inflate
    any n-gram tally and never get joined into a multi-word phrase.

    Compact dates (``2026-05-16``) and multi-token date fragments
    (``2026 05``, ``05 13``) are handled implicitly by this filter
    because ``_TOKEN_RE`` (``[\\wЀ-ӿ]+``) splits on ``-`` / ``/`` /
    whitespace — each component lands here individually and gets caught
    by the year regex or the short-numeric rule.
    """
    if len(token) < 2:
        return True
    if token in _STOPWORDS:
        return True
    if _YEAR_RE.match(token):
        return True
    if _ALL_DIGITS_RE.match(token) and len(token) <= 4:
        return True
    return False


class CueMappingProducer(BaseProducer):
    """Sonnet-adjudicated phrase→schema mappings for prompt hydration."""

    name = "CueMappingProducer"
    version = 1
    artifact_type: ArtifactType = "CueMapping"

    async def discover_subjects(self, session: AsyncSession, run_id: _uuid.UUID) -> list[ProducerSubject]:
        min_cooccur = int(cognitive_config.get("cue_min_cooccurrence"))
        batch_max = int(cognitive_config.get("cue_adjudication_batch_max"))

        # Pull recent single-entity-dominated traces. "Dominance" surrogate
        # in the DB-fallback path = trace has exactly one
        # memory_entity_link with role='subject' (or fall back to 'about').
        rows = await self._fetch_dominated_traces(session)
        if not rows:
            return []

        # Tally trace-level (phrase, entity_id) co-occurrence across the
        # dominated set. ``set(...)`` collapses repeated n-grams within a
        # single trace so a verbose trace can't push a phrase over the
        # threshold by sheer term-frequency.
        phrase_entity_counts: dict[str, collections.Counter[_uuid.UUID]] = collections.defaultdict(collections.Counter)
        for row in rows:
            entity_id: _uuid.UUID = row["entity_id"]
            content: str = row["content"] or ""
            for ngram in set(_extract_ngrams(content, _NGRAM_MIN, _NGRAM_MAX)):
                phrase_entity_counts[ngram][entity_id] += 1

        # One winner per phrase: the active key for ``CueMapping`` is
        # ``dimension=<phrase>``, so emitting two subjects with the same
        # phrase but different entity_ids would let the second subject
        # supersede the first within the same run, leaving a final
        # mapping decided by iteration order. Pick the strongest entity
        # per phrase up front and gate it on ``cue_min_cooccurrence``.
        candidates: list[tuple[str, _uuid.UUID, int]] = []
        for phrase, counter in phrase_entity_counts.items():
            if not counter:
                continue
            entity_id, count = counter.most_common(1)[0]
            if count >= min_cooccur:
                candidates.append((phrase, entity_id, count))

        if not candidates:
            return []

        # Alias-collision filter — drop single-token short phrases that
        # exactly match the candidate's entity name/alias. These reach
        # Sonnet today as confirmed-True at ~0.72-0.75 but get rejected
        # by the 0.80 threshold for ambiguity ("'olga' is a common given
        # name shared by many people"). Filtering at discovery saves the
        # Sonnet call and frees a batch_max slot for a real cue. Done
        # before the cap so legitimate multi-token candidates don't get
        # crowded out by ambiguous-name noise.
        candidates = await self._drop_alias_collisions(session, candidates)
        if not candidates:
            return []

        # Cap at batch_max strongest candidates.
        candidates.sort(key=lambda c: c[2], reverse=True)
        capped = candidates[:batch_max]
        logger.debug(
            "CueMappingProducer discovered %d candidates (min_cooccur=%d, capped=%d)",
            len(candidates),
            min_cooccur,
            len(capped),
        )
        return [
            ProducerSubject(
                key=f"{phrase}::{entity_id}",
                payload={
                    "phrase": phrase,
                    "entity_id": entity_id,
                    "cooccurrence_count": count,
                },
            )
            for phrase, entity_id, count in capped
        ]

    async def produce(
        self,
        session: AsyncSession,
        subject: ProducerSubject,
        run_id: _uuid.UUID,
    ) -> ProducedArtifact | None:
        phrase: str = subject.payload["phrase"]
        entity_id: _uuid.UUID = subject.payload["entity_id"]
        count: int = subject.payload["cooccurrence_count"]

        # Resolve the entity's active EntitySchema (so target_schema_ids
        # points to a real schema, and so adjudication has context). If none
        # exists yet (EntitySchemaProducer may have skipped the entity this
        # run), defer the cue to next run rather than emitting a dangling
        # pointer.
        active_schema = await self._lookup_active_entity_schema(session, entity_id)
        if active_schema is None:
            logger.debug(
                "CueMappingProducer skip phrase=%r — no active EntitySchema for entity %s",
                phrase,
                entity_id,
            )
            return None
        target_schema_id, schema_payload = active_schema

        adjudication = await self._adjudicate_with_sonnet(phrase, schema_payload)
        if adjudication is None:
            return None

        threshold = float(cognitive_config.get("cue_high_confidence_threshold"))
        if not adjudication["confirmed"] or adjudication["confidence"] < threshold:
            # INFO (not DEBUG) so CloudWatch keeps the rejection trail. The
            # producer is currently emitting 0 artifacts despite 10
            # candidates/pass (verified 2026-05-14); surfacing per-candidate
            # rationale lets the next investigation read the pattern from
            # logs alone without re-running the producer locally.
            entity_name = schema_payload.get("canonical_name", "?")
            logger.info(
                "CueMappingProducer rejected phrase=%r entity=%r confidence=%.2f "
                "confirmed=%s threshold=%.2f rationale=%s",
                phrase,
                entity_name,
                adjudication["confidence"],
                adjudication["confirmed"],
                threshold,
                adjudication.get("rationale", "")[:200],
            )
            return None

        payload_model = CueMappingPayload(
            cue_pattern=phrase,
            target_schema_ids=[target_schema_id],
            confidence=adjudication["confidence"],
            fired_count_30d=count,
        )
        return ProducedArtifact(
            artifact_type="CueMapping",
            schema_version=1,
            # subject_entity_id stays NULL — cues aren't owned by an entity;
            # `dimension` carries the cue pattern so the partial unique index
            # gives one active row per phrase.
            subject_entity_id=None,
            dimension=phrase,
            payload=payload_model.model_dump(mode="json"),
            human_summary=f"cue {phrase!r} → schema {target_schema_id} (conf {adjudication['confidence']:.2f})",
        )

    # ── helpers ───────────────────────────────────────────

    async def _drop_alias_collisions(
        self,
        session: AsyncSession,
        candidates: list[tuple[str, _uuid.UUID, int]],
    ) -> list[tuple[str, _uuid.UUID, int]]:
        """Drop (phrase, entity, count) triples where phrase IS the entity's name.

        Single-token short phrases (<= 10 chars, alphabetic only) that match
        the candidate entity's ``name`` are rejected at discovery rather
        than paying for a Sonnet adjudication that always lands in the
        ambiguity bucket. Heuristic ladder applied per candidate:

        - Phrase must be a single token (no whitespace).
        - Phrase length <= 10 chars (after lowercasing) — raised from 6
          on 2026-05-20 after 7-char names like ``mariana`` were slipping
          through to Sonnet. Covers Aleksei (7), Svetlana (8), Vladimir
          (8) without grabbing multi-syllable nouns (>=11 chars).
        - Phrase letters must be purely alphabetic (no digits/punct) —
          n-gram noise like ``"05"`` is already gone but be defensive.
        - Entity ``name`` (case-insensitive, normalized to ``_TOKEN_RE``
          single token) must equal phrase.

        Batched entity lookup keeps this to one round-trip regardless of
        candidate count.
        """
        if not candidates:
            return []
        # Pre-filter: only candidates whose phrase shape could collide.
        # Multi-token / long / non-alpha phrases short-circuit without
        # needing a DB row.
        #
        # 2026-05-20 raised the length cap from 6 → 10 chars: at 6,
        # ``mariana`` (7) and ``andrei`` (6, edge) were slipping through
        # to Sonnet and getting rejected at confidence 0.72-0.75 every
        # L3 pass. 10 chars covers the common-given-name tail (Aleksei,
        # Svetlana, Vladimir) without grabbing legitimate multi-syllable
        # nouns. Tune up further if cap-collisions remain visible in
        # CueMapping rejection logs.
        check_indices: list[int] = []
        for idx, (phrase, _entity_id, _count) in enumerate(candidates):
            if " " in phrase or len(phrase) > 10:
                continue
            # ``str.isalpha()`` is Unicode-aware so accented Latin
            # ("José", "Łukasz") and Cyrillic both count as alphabetic.
            # A narrow regex would over-filter the long tail of names.
            if not phrase.isalpha():
                continue
            check_indices.append(idx)
        if not check_indices:
            return candidates

        check_entity_ids = {candidates[i][1] for i in check_indices}
        name_rows = await session.execute(
            text(
                """
                SELECT id, name
                  FROM memory_entities
                 WHERE id = ANY(:ids)
                """
            ),
            {"ids": list(check_entity_ids)},
        )
        names_by_id: dict[_uuid.UUID, str] = {row.id: (row.name or "") for row in name_rows}

        dropped: set[int] = set()
        for idx in check_indices:
            phrase, entity_id, _count = candidates[idx]
            entity_name = names_by_id.get(entity_id, "")
            if not entity_name:
                continue
            # Whole-word tokens of the entity name, lowercased. The
            # ``phrase in name_tokens`` check below gates whenever the
            # candidate phrase equals ANY token of the name, so both
            # "olga" and "petrov" get gated for "Olga Petrov".
            # Partial / substring matches ("ир") still pass through —
            # tokenization is the membership boundary, which is the
            # behaviour we want (avoids over-filtering short prefixes).
            name_tokens = _TOKEN_RE.findall(entity_name.lower())
            if not name_tokens:
                continue
            if phrase in name_tokens:
                dropped.add(idx)

        if dropped:
            logger.info(
                "CueMappingProducer alias-collision drop: %d candidate(s) — examples=%s",
                len(dropped),
                [f"{candidates[i][0]}→{names_by_id.get(candidates[i][1], '?')}" for i in list(dropped)[:5]],
            )
        return [c for i, c in enumerate(candidates) if i not in dropped]

    async def _fetch_dominated_traces(self, session: AsyncSession) -> list[dict[str, Any]]:
        cutoff = utc_now() - timedelta(days=_LOOKBACK_DAYS)
        # "Dominated" = trace has exactly one entity link (any role). In the
        # CloudWatch path this would be derived from the priming event's
        # primed_traces array; here we approximate with a per-trace COUNT.
        result = await session.execute(
            text(
                """
                WITH trace_link_counts AS (
                    SELECT mel.memory_id,
                           COUNT(DISTINCT mel.entity_id) AS distinct_entities,
                           (array_agg(mel.entity_id))[1] AS dominant_entity_id
                      FROM memory_entity_links mel
                     GROUP BY mel.memory_id
                )
                SELECT mt.id AS trace_id,
                       mt.content,
                       tlc.dominant_entity_id AS entity_id
                  FROM memory_traces mt
                  JOIN trace_link_counts tlc ON tlc.memory_id = mt.id
                 WHERE mt.invalidated_at IS NULL
                   AND mt.created_at > :cutoff
                   AND tlc.distinct_entities = 1
                """
            ),
            {"cutoff": cutoff},
        )
        return [dict(row) for row in result.mappings().all()]

    async def _lookup_active_entity_schema(
        self, session: AsyncSession, entity_id: _uuid.UUID
    ) -> tuple[_uuid.UUID, dict[str, Any]] | None:
        """Return ``(schema_id, payload_dict)`` for the entity's active
        EntitySchema, or ``None`` if none exists.

        We need the payload for adjudication context (canonical_name,
        canonical_fields), so the lookup pulls both columns in one round-trip.
        """
        row = (
            await session.execute(
                text(
                    """
                    SELECT id, payload FROM memory_artifacts
                     WHERE artifact_type = 'EntitySchema'
                       AND status = 'active'
                       AND subject_entity_id = :entity_id
                     LIMIT 1
                    """
                ),
                {"entity_id": entity_id},
            )
        ).first()
        if row is None:
            return None
        schema_id = row[0]
        payload = row[1] if isinstance(row[1], dict) else {}
        return schema_id, payload

    async def _adjudicate_with_sonnet(self, phrase: str, schema_payload: dict[str, Any]) -> dict[str, Any] | None:
        # Distil schema_payload to the few fields Sonnet needs for entity
        # fit. Passing the raw payload would burn tokens on irrelevant
        # plumbing (relationships, schema_version, etc.); the canonical
        # name + entity_type + canonical_fields are what disambiguate.
        canonical_name = schema_payload.get("canonical_name") or "(unknown)"
        entity_type = schema_payload.get("entity_type") or "(unknown)"
        canonical_fields = schema_payload.get("canonical_fields") or {}
        aliases = schema_payload.get("aliases") or []
        # Truncate fields to keep the prompt small — typical schema has 5-10
        # canonical_fields with short values.
        fields_summary = (
            ", ".join(f"{k}={_truncate(str(v), 60)}" for k, v in list(canonical_fields.items())[:8]) or "(none)"
        )
        aliases_summary = ", ".join(aliases[:5]) or "(none)"
        prompt = (
            f"You are deciding whether the phrase {phrase!r} is a reliable cue "
            f"for retrieving information about a specific entity.\n\n"
            f"Entity:\n"
            f"  canonical_name: {canonical_name}\n"
            f"  entity_type:    {entity_type}\n"
            f"  aliases:        {aliases_summary}\n"
            f"  canonical_fields: {fields_summary}\n\n"
            f"Return STRICT JSON: "
            f'{{"confirmed": true|false, "confidence": 0.0-1.0, "rationale": "..."}}.\n\n'
            f"Be conservative — false positives pollute the priming prompt.\n"
            f"A reliable cue should be specific (not just a stop-word phrase) "
            f"AND closely associated with THIS entity in conversational context. "
            f"Phrases that match the entity's canonical name/aliases or refer to "
            f"a clearly disambiguating canonical field are good cues; generic "
            f"category words ('the dog', 'meeting') are bad cues unless the "
            f"context establishes a unique referent."
        )
        try:
            provider = get_provider_for_model(get_settings(), get_model_roles().artifact)
            span = start_generation_span("consolidation/cue-adjudication", get_model_roles().artifact)
            # max_tokens=2048 + thinking_mode="disabled": adaptive thinking
            # consumed the entire 512 budget on prod (silent skip via the
            # `\{.*\}` regex losing its closing brace). 2048 is generous for
            # the small adjudication object; thinking adds zero quality on
            # JSON extraction (see scripts/verify_disable_thinking.py).
            raw = await call_llm_text(
                provider,
                prompt,
                max_tokens=2048,
                span=span,
                thinking_mode="disabled",
                task_name="memory.cue.adjudicate",
            )
        except Exception:
            logger.warning(
                "CueMappingProducer: Sonnet call failed; skipping",
                exc_info=True,
            )
            return None

        return _parse_adjudication_response(raw)


# ── parsing helpers ───────────────────────────────────────


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_adjudication_response(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    body = raw.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\n?", "", body)
        body = re.sub(r"\n?```$", "", body).strip()
    match = _JSON_OBJ_RE.search(body)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("CueMappingProducer: malformed adjudication JSON: %s", body[:200])
        return None
    if not isinstance(parsed, dict):
        return None
    confirmed = _parse_confirmed(parsed.get("confirmed"))
    if confirmed is None:
        # Treat ambiguous values as a parse failure rather than silently
        # defaulting to True/False — the design says "be conservative".
        logger.warning(
            "CueMappingProducer: ambiguous 'confirmed' value %r; rejecting cue",
            parsed.get("confirmed"),
        )
        return None
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "confirmed": confirmed,
        "confidence": max(0.0, min(1.0, confidence)),
        "rationale": str(parsed.get("rationale", "")),
    }


def _parse_confirmed(value: Any) -> bool | None:
    """Strict-ish boolean parsing for the Sonnet adjudication response.

    Accepts native bools and the canonical lowercase string forms
    (``"true"`` / ``"false"``). Any other shape — including
    ``"yes"``, ``"1"``, numeric values, ``None``, missing — returns
    ``None`` so the caller can reject the response rather than coerce
    a typo into ``True``.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised == "true":
            return True
        if normalised == "false":
            return False
    return None


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def _extract_ngrams(content: str, n_min: int, n_max: int) -> list[str]:
    """Return lowercase 1-3 word n-grams from ``content``, noise-filtered.

    Token-level pass only — ``_is_noise_token`` drops stopwords, single
    chars, 4-digit years (1900-2099), and short all-digit strings.
    ``_TOKEN_RE`` splits on ``-`` / ``/`` / whitespace, so date fragments
    like ``2026-05-16`` arrive as ``("2026", "05", "16")`` — each
    component gets caught by the year / short-numeric rule individually,
    and the n-gram cross-product is already clean.

    Without the token-level cut, a noise token like ``2026`` would pair
    with every legitimate neighbour and inflate joint counts for
    ``"2026 wine"``, ``"code 2026"``, etc.
    """
    tokens = [t.lower() for t in _TOKEN_RE.findall(content)]
    tokens = [t for t in tokens if not _is_noise_token(t)]
    out: list[str] = []
    for n in range(n_min, n_max + 1):
        for i in range(0, len(tokens) - n + 1):
            out.append(" ".join(tokens[i : i + n]))
    return out


__all__ = [
    "CueMappingProducer",
    "_extract_ngrams",
    "_is_noise_token",
    "_parse_adjudication_response",
]
