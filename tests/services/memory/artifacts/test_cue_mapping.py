"""Unit tests for CueMappingProducer pure helpers + skip paths.

DB-dependent paths (subject discovery SQL, end-to-end produce against real
EntitySchema rows) live in ``tests/integration/test_artifact_pipeline.py``.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from kai_memory.artifacts.producers.cue_mapping import (
    CueMappingProducer,
    _extract_ngrams,
    _is_noise_token,
    _parse_adjudication_response,
)
from kai_memory.artifacts.producer import ProducerSubject


# ── Pure helpers ──────────────────────────────────────────


def test_extract_ngrams_filters_stopwords_and_short_tokens() -> None:
    text = "The quick brown fox runs in the forest"
    ngrams = _extract_ngrams(text, 1, 3)
    # "the", "in" filtered as stopwords; single-char tokens dropped.
    assert "quick" in ngrams
    assert "the" not in ngrams
    assert "in" not in ngrams
    # 2-grams and 3-grams over surviving tokens.
    assert "quick brown" in ngrams
    assert "quick brown fox" in ngrams


def test_extract_ngrams_handles_cyrillic() -> None:
    text = "Луна это белая собака"
    ngrams = _extract_ngrams(text, 1, 2)
    # "это" is in the stop-list; the remaining tokens should survive.
    assert "луна" in ngrams
    assert "белая" in ngrams
    assert "собака" in ngrams
    assert "это" not in ngrams


# ── 2026-05-17 discovery noise filter — token level ──────────────────


def test_is_noise_token_strips_years() -> None:
    """4-digit years in the 1900-2099 band are pure noise as cues.

    Note: the year regex is an explicit fast-path; the broader
    "all-digits len ≤ 4" rule catches the same tokens anyway. The two
    rules together mean every 4-digit-or-shorter pure number is noise,
    which matches the user's spec exactly.
    """
    assert _is_noise_token("2026") is True
    assert _is_noise_token("1999") is True
    assert _is_noise_token("year") is False
    # Words containing a year still survive — only the bare year is noise.
    assert _is_noise_token("y2k26") is False


def test_is_noise_token_strips_short_numeric() -> None:
    """All-digit tokens of length ≤4 ('05', '13', '2025') are noise."""
    assert _is_noise_token("05") is True
    assert _is_noise_token("13") is True
    assert _is_noise_token("12345") is False  # 5+ digits could be a real id


def test_is_noise_token_strips_pronouns_and_role_nouns() -> None:
    """Stop-list expansion catches pronouns + role/category nouns."""
    for tok in ("his", "her", "they", "them", "my", "your", "we"):
        assert _is_noise_token(tok) is True, f"english pronoun {tok!r} should be noise"
    # Cyrillic-Latin homoglyph noqa matches the pattern in cue_mapping.py.
    for tok in ("его", "её", "ее", "их", "я", "ты", "мы"):  # noqa: RUF001
        assert _is_noise_token(tok) is True, f"russian pronoun {tok!r} should be noise"
    for tok in ("assistant", "user", "agent", "system", "bot", "model"):
        assert _is_noise_token(tok) is True, f"role noun {tok!r} should be noise"


def test_is_noise_token_strips_modal_aux() -> None:
    """English aux/modals that Sonnet's rejection log flagged at 0.95+."""
    for tok in ("may", "might", "can", "could", "would", "should", "must"):
        assert _is_noise_token(tok) is True, f"aux {tok!r} should be noise"


def test_is_noise_token_strips_v2_additions() -> None:
    """2026-05-20 v2 additions: ``not``, ``than``, ``rather``, and the
    generic activity nouns ``training`` / ``session`` / ``recovery``.

    Diagnostic evidence: 24h CueMapping rejection log showed each of
    these phrases rejected once per L3 pass (3x in 24h) at confidence
    ≥0.90. ``not`` in particular was a clear miss in the original #144
    stop-list — it's a textbook English negation that should never
    have been considered a cue.
    """
    for tok in ("not", "than", "rather"):
        assert _is_noise_token(tok) is True, f"connector/negation {tok!r} should be noise (v2 addition)"
    for tok in ("training", "session", "recovery"):
        assert _is_noise_token(tok) is True, f"generic activity noun {tok!r} should be noise (v2 addition)"


def test_is_noise_token_keeps_real_cues() -> None:
    """Negative control — legitimate cue tokens must survive."""
    for tok in ("луна", "собака", "wine", "tradernet", "viabtc", "fireflies"):
        assert _is_noise_token(tok) is False, f"real cue {tok!r} should survive"


# ── End-to-end via tokenization ──────────────────────────────────────


def test_extract_ngrams_drops_compact_dates_via_tokenization() -> None:
    """Compact-date single tokens like ``2026-05-16`` get split on the
    hyphen by ``_TOKEN_RE`` into ``("2026", "05", "16")``. Each component
    is then dropped by ``_is_noise_token`` (year regex + short-numeric).
    No multi-token n-gram filter needed.
    """
    ngrams = _extract_ngrams("meeting 2026-05-16 with assistant", 1, 3)
    # All date components and the role noun are gone.
    assert "2026" not in ngrams
    assert "05" not in ngrams
    assert "16" not in ngrams
    assert "assistant" not in ngrams
    assert "2026 05" not in ngrams  # never assembled
    assert "05 16" not in ngrams
    # Real cue survives.
    assert "meeting" in ngrams


def test_extract_ngrams_drops_multi_token_dates_via_tokenization() -> None:
    """Multi-token date fragments like ``'2026 05 13'`` get dropped at the
    token level — each numeric token is < 5 chars and matched by the
    year regex or short-numeric rule before any n-gram is assembled.
    """
    ngrams = _extract_ngrams("event on 2026 05 13 about wine club", 1, 3)
    assert "2026" not in ngrams
    assert "05 13" not in ngrams
    assert "2026 05" not in ngrams
    assert "wine" in ngrams
    assert "wine club" in ngrams
    assert "event" in ngrams


def test_extract_ngrams_drops_slash_separated_dates() -> None:
    """``16/05/2026`` tokenizes to ``("16", "05", "2026")`` — each falls
    to ``_is_noise_token``. Slash and hyphen are equivalent here."""
    ngrams = _extract_ngrams("dinner 16/05/2026 at house", 1, 2)
    assert "16" not in ngrams
    assert "2026" not in ngrams
    assert "16 05" not in ngrams
    assert "dinner" in ngrams
    assert "house" in ngrams


def test_extract_ngrams_keeps_real_phrases() -> None:
    """Negative control — real cue phrases survive the filter."""
    ngrams = _extract_ngrams("our wine club met sue this evening", 1, 3)
    assert "wine" in ngrams
    assert "wine club" in ngrams
    assert "evening" in ngrams


def test_parse_adjudication_response_strips_code_fence() -> None:
    raw = """```json
{"confirmed": true, "confidence": 0.92, "rationale": "specific cue"}
```"""
    parsed = _parse_adjudication_response(raw)
    assert parsed is not None
    assert parsed["confirmed"] is True
    assert parsed["confidence"] == pytest.approx(0.92)
    assert "specific" in parsed["rationale"]


def test_parse_adjudication_response_clamps_confidence() -> None:
    raw = '{"confirmed": false, "confidence": 1.7}'
    parsed = _parse_adjudication_response(raw)
    assert parsed is not None
    assert parsed["confidence"] == 1.0


def test_parse_adjudication_response_returns_none_on_garbage() -> None:
    assert _parse_adjudication_response("not json at all") is None
    assert _parse_adjudication_response(None) is None


def test_parse_adjudication_string_false_is_false_not_truthy() -> None:
    """A literal ``"false"`` string must not coerce to True via bool()."""
    parsed = _parse_adjudication_response('{"confirmed": "false", "confidence": 0.95}')
    assert parsed is not None
    assert parsed["confirmed"] is False  # was True under the old bool() coercion


def test_parse_adjudication_accepts_string_true_for_confirmed() -> None:
    parsed = _parse_adjudication_response('{"confirmed": "true", "confidence": 0.7}')
    assert parsed is not None
    assert parsed["confirmed"] is True


def test_parse_adjudication_rejects_ambiguous_confirmed_value() -> None:
    """Anything other than bool/canonical-string is rejected as a parse failure."""
    assert _parse_adjudication_response('{"confirmed": "yes", "confidence": 0.9}') is None
    assert _parse_adjudication_response('{"confirmed": 1, "confidence": 0.9}') is None
    assert _parse_adjudication_response('{"confidence": 0.9}') is None  # missing


# ── Skip paths (no DB needed) ─────────────────────────────


async def test_produce_skips_when_no_active_entity_schema() -> None:
    """If the entity has no active EntitySchema, the cue defers to next run."""
    producer = CueMappingProducer()
    session = MagicMock()
    # _lookup_active_entity_schema returns None when no row found
    session.execute = AsyncMock(return_value=_empty_first())

    subject = ProducerSubject(
        key="phrase::ent",
        payload={
            "phrase": "наша собака",
            "entity_id": uuid.uuid4(),
            "cooccurrence_count": 7,
        },
    )
    result = await producer.produce(session, subject, uuid.uuid4())
    assert result is None


class _FakeAdjudicatingProducer(CueMappingProducer):
    """Test-double that overrides adjudication via subclass instead of
    rebinding the private coroutine on the instance (per coding standard
    "no private-field mutation from outside the class")."""

    def __init__(self, response: dict | None) -> None:
        super().__init__()
        self._fake_response = response
        self.captured: list[tuple[str, dict]] = []

    async def _adjudicate_with_sonnet(self, phrase, schema_payload):
        self.captured.append((phrase, schema_payload))
        return self._fake_response


async def test_produce_skips_when_sonnet_rejects() -> None:
    target_id = uuid.uuid4()
    schema_payload = {
        "canonical_name": "TestEntity",
        "entity_type": "person",
        "canonical_fields": {"role": "frontend"},
        "aliases": ["Test"],
    }
    schema_lookup = MagicMock()
    schema_lookup.first = MagicMock(return_value=(target_id, schema_payload))
    session = MagicMock()
    session.execute = AsyncMock(return_value=schema_lookup)

    producer = _FakeAdjudicatingProducer(response={"confirmed": True, "confidence": 0.5, "rationale": "weak"})
    subject = ProducerSubject(
        key="phrase::ent",
        payload={"phrase": "phrase", "entity_id": uuid.uuid4(), "cooccurrence_count": 8},
    )
    result = await producer.produce(session, subject, uuid.uuid4())
    assert result is None
    # Adjudication received the schema payload, not just the entity_id.
    assert producer.captured == [("phrase", schema_payload)]


async def test_produce_rejection_logs_at_info_with_entity_and_rationale(caplog) -> None:
    """The rejection log was DEBUG; flipped to INFO 2026-05-15 so CloudWatch
    keeps the trail by default. Must include entity name + rationale so the
    next 0-emit investigation can read the pattern from logs alone.
    """
    import logging as _logging

    target_id = uuid.uuid4()
    schema_payload = {
        "canonical_name": "Maria",
        "entity_type": "person",
        "canonical_fields": {},
        "aliases": [],
    }
    schema_lookup = MagicMock()
    schema_lookup.first = MagicMock(return_value=(target_id, schema_payload))
    session = MagicMock()
    session.execute = AsyncMock(return_value=schema_lookup)

    producer = _FakeAdjudicatingProducer(
        response={"confirmed": False, "confidence": 0.42, "rationale": "phrase too generic"},
    )
    subject = ProducerSubject(
        key="phrase::ent",
        payload={"phrase": "the team", "entity_id": uuid.uuid4(), "cooccurrence_count": 6},
    )

    with caplog.at_level(_logging.INFO, logger="app.services.memory.artifacts.producers.cue_mapping"):
        result = await producer.produce(session, subject, uuid.uuid4())

    assert result is None
    rejection_records = [
        r for r in caplog.records if r.levelno == _logging.INFO and "CueMappingProducer rejected" in r.getMessage()
    ]
    assert len(rejection_records) == 1
    msg = rejection_records[0].getMessage()
    assert "Maria" in msg
    assert "phrase too generic" in msg
    assert "the team" in msg


async def test_produce_emits_artifact_when_sonnet_confirms() -> None:
    target_id = uuid.uuid4()
    schema_payload = {
        "canonical_name": "Луна",
        "entity_type": "animal",
        "canonical_fields": {"breed": "Pomeranian"},
        "aliases": ["Luna"],
    }
    schema_lookup = MagicMock()
    schema_lookup.first = MagicMock(return_value=(target_id, schema_payload))
    session = MagicMock()
    session.execute = AsyncMock(return_value=schema_lookup)

    producer = _FakeAdjudicatingProducer(response={"confirmed": True, "confidence": 0.9, "rationale": "strong"})
    subject = ProducerSubject(
        key="phrase::ent",
        payload={"phrase": "наша собака", "entity_id": uuid.uuid4(), "cooccurrence_count": 12},
    )
    result = await producer.produce(session, subject, uuid.uuid4())
    assert result is not None
    assert result.artifact_type == "CueMapping"
    assert result.subject_entity_id is None  # cues live under dimension scope
    assert result.dimension == "наша собака"
    assert result.payload["target_schema_ids"] == [str(target_id)]
    assert result.payload["confidence"] == pytest.approx(0.9)
    assert result.payload["fired_count_30d"] == 12
    # Schema payload threaded through to the adjudication call.
    assert producer.captured == [("наша собака", schema_payload)]


# ── Alias-collision filter ───────────────────────────────────────────


async def test_drop_alias_collisions_handles_7_to_10_char_names() -> None:
    """v2 raises the length cap from 6 to 10 chars so names like
    ``elena`` (7 chars), ``aleksei`` (7), and ``svetlana`` (8) are
    gated at discovery, not paid for at Sonnet adjudication.

    Diagnostic evidence: 24h CueMapping rejection log showed
    ``elena`` rejected 3x at confidence 0.72-0.75 with rationale
    "common given name". The original 6-char cap let it through.
    """
    producer = CueMappingProducer()
    elena_id = uuid.uuid4()
    aleksei_id = uuid.uuid4()

    class _Row:
        def __init__(self, eid, name):
            self.id = eid
            self.name = name

    rows = [_Row(elena_id, "Elena Petrov"), _Row(aleksei_id, "Aleksei Smirnov")]
    name_lookup = MagicMock()
    name_lookup.__iter__ = lambda self: iter(rows)
    session = MagicMock()
    session.execute = AsyncMock(return_value=name_lookup)

    candidates = [
        ("elena", elena_id, 8),  # 7 chars, was slipping through
        ("aleksei", aleksei_id, 6),  # 7 chars
    ]
    survivors = await producer._drop_alias_collisions(session, candidates)
    survivor_phrases = {c[0] for c in survivors}
    assert "elena" not in survivor_phrases
    assert "aleksei" not in survivor_phrases


async def test_drop_alias_collisions_handles_10_char_boundary() -> None:
    """Exactly 10 chars (the cap inclusive bound) still passes through
    the alias check. Just-over-cap (11) does not.

    Two boundary candidates exercised together:
    - ``svetlanaaa`` (10 chars, len == cap) → check runs, matches the
      entity's first-name token → DROPPED.
    - ``photographs`` (11 chars, len > cap) → check skipped early,
      returns as-is regardless of any name match → KEPT.
    """
    producer = CueMappingProducer()
    sv_id = uuid.uuid4()
    pg_id = uuid.uuid4()

    class _Row:
        def __init__(self, eid, name):
            self.id = eid
            self.name = name

    # Entity name contains the 10-char phrase as a single token, so the
    # `if phrase in name_tokens` membership check fires.
    rows = [_Row(sv_id, "Svetlanaaa Ivanova"), _Row(pg_id, "Photographs Studio")]
    name_lookup = MagicMock()
    name_lookup.__iter__ = lambda self: iter(rows)
    session = MagicMock()
    session.execute = AsyncMock(return_value=name_lookup)

    candidates = [
        ("svetlanaaa", sv_id, 8),  # 10 chars exactly — at the cap
        ("photographs", pg_id, 8),  # 11 chars — over cap, alias check skipped
    ]
    assert len("svetlanaaa") == 10
    assert len("photographs") == 11

    survivors = await producer._drop_alias_collisions(session, candidates)
    survivor_phrases = {c[0] for c in survivors}
    # 10-char is at-the-cap → check runs → matches → drops.
    assert "svetlanaaa" not in survivor_phrases
    # 11-char is over the cap → check never runs → candidate survives.
    assert "photographs" in survivor_phrases


async def test_drop_alias_collisions_still_caps_at_10_chars() -> None:
    """Names of 11+ chars do NOT pass through the alias check (cap holds).

    Without the upper cap, multi-syllable nouns like ``alphabetical``
    or ``photograph`` would hit the entity lookup unnecessarily.
    """
    producer = CueMappingProducer()
    session = MagicMock()
    session.execute = AsyncMock()

    # 11-char single token — skipped, DB not consulted.
    candidates = [("photographs", uuid.uuid4(), 5)]
    assert len("photographs") > 10  # sanity
    survivors = await producer._drop_alias_collisions(session, candidates)
    assert survivors == candidates
    session.execute.assert_not_awaited()


async def test_drop_alias_collisions_kills_single_name_phrases(caplog) -> None:
    """Single-token phrase matching the entity's name → dropped pre-Sonnet.

    Yesterday's rejection log showed Sonnet rejecting "olga"→Olga at
    confidence 0.72 (ambiguous given name). Filtering at discovery saves
    the Sonnet call and frees a batch_max slot for a real cue.
    """
    import logging as _logging

    producer = CueMappingProducer()
    olga_id = uuid.uuid4()
    luna_id = uuid.uuid4()

    # Mock the entity-name lookup query.
    class _Row:
        def __init__(self, eid, name):
            self.id = eid
            self.name = name

    rows = [_Row(olga_id, "Olga Petrov"), _Row(luna_id, "Луна")]
    name_lookup = MagicMock()
    name_lookup.__iter__ = lambda self: iter(rows)

    session = MagicMock()
    session.execute = AsyncMock(return_value=name_lookup)

    candidates = [
        ("olga", olga_id, 8),  # single-token alias collision → drop
        ("луна", luna_id, 7),  # single-token alias collision → drop
        ("white pomeranian", luna_id, 9),  # multi-token cue → keep
        ("ир", olga_id, 5),  # partial match, NOT in entity name tokens → keep
    ]

    with caplog.at_level(_logging.INFO, logger="app.services.memory.artifacts.producers.cue_mapping"):
        survivors = await producer._drop_alias_collisions(session, candidates)

    survivor_phrases = {c[0] for c in survivors}
    assert "olga" not in survivor_phrases
    assert "луна" not in survivor_phrases
    assert "white pomeranian" in survivor_phrases
    assert "ир" in survivor_phrases

    drop_logs = [r for r in caplog.records if "alias-collision drop" in r.getMessage()]
    assert len(drop_logs) == 1
    assert "2 candidate(s)" in drop_logs[0].getMessage()


async def test_drop_alias_collisions_skips_lookup_when_no_candidates_qualify() -> None:
    """Multi-token / long / non-alpha candidates short-circuit without DB."""
    producer = CueMappingProducer()
    session = MagicMock()
    session.execute = AsyncMock()

    candidates = [
        ("our dog", uuid.uuid4(), 8),  # multi-token
        ("longerthansixchars", uuid.uuid4(), 9),  # > 6 chars
        ("05", uuid.uuid4(), 6),  # non-alpha (caught by other filters)
    ]
    survivors = await producer._drop_alias_collisions(session, candidates)
    assert survivors == candidates
    # Did NOT hit the database for the alias check.
    session.execute.assert_not_awaited()


async def test_drop_alias_collisions_handles_accented_latin_and_cyrillic() -> None:
    """``str.isalpha()`` is Unicode-aware, so accented Latin (José,
    Łukasz) and Cyrillic (Луна) all qualify for the alias-collision
    check — a narrow ASCII+Cyrillic regex would over-filter the long
    tail of real names.
    """
    producer = CueMappingProducer()
    jose_id = uuid.uuid4()
    lukasz_id = uuid.uuid4()
    luna_id = uuid.uuid4()

    class _Row:
        def __init__(self, eid, name):
            self.id = eid
            self.name = name

    rows = [
        _Row(jose_id, "José García"),
        _Row(lukasz_id, "Łukasz K"),
        _Row(luna_id, "Луна"),
    ]
    name_lookup = MagicMock()
    name_lookup.__iter__ = lambda self: iter(rows)
    session = MagicMock()
    session.execute = AsyncMock(return_value=name_lookup)

    candidates = [
        ("josé", jose_id, 9),  # accented Latin, matches name token → drop
        ("łukasz", lukasz_id, 8),  # accented Latin, matches → drop
        ("луна", luna_id, 7),  # Cyrillic, matches → drop
        ("garcía", jose_id, 6),  # second name token, matches → drop
    ]
    survivors = await producer._drop_alias_collisions(session, candidates)
    assert survivors == []
    # All four candidates were considered (each is single-token, alpha, ≤6 chars).
    session.execute.assert_awaited_once()


async def test_drop_alias_collisions_keeps_when_entity_name_missing() -> None:
    """Defensive — if memory_entities row is gone, keep the candidate."""
    producer = CueMappingProducer()
    olga_id = uuid.uuid4()

    name_lookup = MagicMock()
    name_lookup.__iter__ = lambda self: iter([])  # no rows back
    session = MagicMock()
    session.execute = AsyncMock(return_value=name_lookup)

    candidates = [("olga", olga_id, 8)]
    survivors = await producer._drop_alias_collisions(session, candidates)
    assert survivors == candidates


async def test_drop_alias_collisions_empty_candidates_returns_empty() -> None:
    producer = CueMappingProducer()
    session = MagicMock()
    session.execute = AsyncMock()
    survivors = await producer._drop_alias_collisions(session, [])
    assert survivors == []
    session.execute.assert_not_awaited()


# ── Helpers ───────────────────────────────────────────────


def _empty_first() -> MagicMock:
    """A SQLAlchemy result whose .first() returns None."""
    result = MagicMock()
    result.first = MagicMock(return_value=None)
    return result
