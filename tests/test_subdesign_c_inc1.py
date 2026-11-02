"""Sub-design C INC 1 — typed_metadata + provenance foundation.

Lowest-risk substrate increment: 5 additive columns on ``memory_traces``
(one versioned JSONB extraction blob + four provenance scalars), an additive
Haiku extraction prompt, the ``agent`` EntityType, and write-side plumbing
gated behind ``typed_metadata_write_enabled``. Nothing reads the columns yet.

Schema-layer tests (this block) assert the model column contract — runnable
without a database. Behavioural round-trips that need Postgres live in
``tests/integration/`` and run under testcontainers in CI.

Design: backend-docs/ai-workforce/plans/design/2026-06-04-subdesign-C-architecture-and-plan.md (§1.1).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import BigInteger
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB

from kai_memory.storage.models.memory_trace import MemoryTrace
from kai_memory.store import cognitive_config
from memory_test_support import ChatContext
from kai_memory.extraction.memory_extraction import ExtractionService
from kai_memory.store.cognitive_memory import CognitiveMemoryService
from kai_memory.store.cognitive_memory_constants import VALID_ENTITY_TYPES


@pytest.fixture(autouse=True)
def _reset_cognitive_config():
    """Each test starts from baked-in defaults (flag off)."""
    cognitive_config.reset_cache()
    yield
    cognitive_config.reset_cache()


# ── Step A: schema layer (migration 087 + model) ──────────────────────────


def test_typed_metadata_column_contract():
    """typed_metadata is a NOT-NULL JSONB column defaulting to ``{}`` (R1)."""
    cols = MemoryTrace.__table__.columns
    assert "typed_metadata" in cols, "migration 087 column missing from model"
    tm = cols["typed_metadata"]
    assert isinstance(tm.type, JSONB), "typed_metadata must be JSONB"
    assert tm.nullable is False, "typed_metadata is NOT NULL (defaults to {})"
    assert tm.server_default is not None, "needs a server_default of '{}'::jsonb"
    # Python-side default too — tests build MemoryTrace(...) client-side, where
    # server defaults don't fire (mirrors encoding_signals at memory_trace.py:66).
    assert tm.default is not None and tm.default.is_callable, (
        "typed_metadata needs default=dict so client-side rows are {} not None"
    )


def test_provenance_columns_contract():
    """The four provenance scalars are real, nullable columns (hot-path, R1).

    They land NULL on day one and are populated lazily — ``source_user_id``
    not until E (INC 8). All four are nullable so pre-087 rows stay valid.
    """
    cols = MemoryTrace.__table__.columns
    expected = {
        "source_confidence",
        "source_user_id",
        "source_path",
        "invalidation_source",
    }
    missing = expected - set(cols.keys())
    assert not missing, f"migration 087 provenance columns missing: {missing}"
    for name in expected:
        assert cols[name].nullable is True, f"{name} must be nullable"
    # source_user_id holds a telegram author id — BIGINT, not INTEGER.
    assert isinstance(cols["source_user_id"].type, BigInteger), (
        "source_user_id must be BigInteger (telegram ids exceed int32)"
    )


# ── Step B: store_trace + invalidate_trace plumbing ───────────────────────

_TM = {
    "schema_version": 1,
    "facts": [{"field": "role", "value": "founder", "polarity": "is"}],
    "entities": [{"name": "Misha", "type": "person"}],
    "relationships": [],
}


async def _store(session, **kwargs):
    with patch(
        "kai_memory.store.cognitive_memory.generate_embedding",
        return_value=[0.1] * 1536,
    ):
        svc = CognitiveMemoryService(session)
        return await svc.store_trace(
            content="Misha founded Examplecorp",
            memory_type="fact",
            source_type="conversation",
            **kwargs,
        )


@pytest.mark.asyncio
async def test_store_trace_writes_typed_metadata_when_flag_on():
    cognitive_config.set_for_testing("typed_metadata_write_enabled", 1.0)
    session = AsyncMock()
    session.flush = AsyncMock()
    trace = await _store(session, typed_metadata=_TM, source_path="l2_inline")
    assert trace.typed_metadata == _TM
    assert trace.source_path == "l2_inline"


@pytest.mark.asyncio
async def test_store_trace_ignores_params_when_flag_off():
    # Flag OFF → store_trace must not write the new fields; the row keeps its column
    # defaults ({} / NULL), byte-identical to pre-087. (The CODE default is now ON —
    # write-path fact-loss fix — so this OFF behavior is set explicitly.)
    cognitive_config.set_for_testing("typed_metadata_write_enabled", 0.0)
    session = AsyncMock()
    session.flush = AsyncMock()
    trace = await _store(session, typed_metadata=_TM, source_path="l2_inline")
    # Both left untouched → unset (None client-side; the DB server_default of
    # '{}' fires only at flush). Pins "store_trace wrote neither field".
    assert trace.source_path is None, "flag off must not stamp source_path"
    assert trace.typed_metadata is None, "flag off must not write typed_metadata"


@pytest.mark.asyncio
async def test_store_trace_does_not_thread_source_user_id():
    """INC 1 never populates source_user_id (deferred to E / INC 8)."""
    cognitive_config.set_for_testing("typed_metadata_write_enabled", 1.0)
    session = AsyncMock()
    session.flush = AsyncMock()
    with pytest.raises(TypeError):
        await _store(session, source_user_id=12345)  # kwarg must not exist


@pytest.mark.asyncio
async def test_invalidate_trace_stamps_source_when_flag_on():
    cognitive_config.set_for_testing("typed_metadata_write_enabled", 1.0)
    captured: dict = {}
    session = AsyncMock()

    async def _capture(stmt):
        captured["stmt"] = stmt
        return AsyncMock()

    session.execute = _capture
    svc = CognitiveMemoryService(session)
    await svc.invalidate_trace(uuid.uuid4(), invalidation_source="l2_inline")
    params = captured["stmt"].compile(dialect=postgresql.dialect()).params
    assert params.get("invalidation_source") == "l2_inline"


@pytest.mark.asyncio
async def test_invalidate_trace_omits_source_when_flag_off():
    cognitive_config.set_for_testing("typed_metadata_write_enabled", 0.0)  # code default now ON
    captured: dict = {}
    session = AsyncMock()

    async def _capture(stmt):
        captured["stmt"] = stmt
        return AsyncMock()

    session.execute = _capture
    svc = CognitiveMemoryService(session)
    await svc.invalidate_trace(uuid.uuid4(), invalidation_source="l2_inline")
    params = captured["stmt"].compile(dialect=postgresql.dialect()).params
    assert "invalidation_source" not in params, "flag off must not stamp the edge"


# ── Step C: EntityType `agent` enum + coerce path ─────────────────────────


def test_agent_is_a_valid_entity_type():
    """`agent` joins the EntityType set so Kai-the-assistant has a real type."""
    assert "agent" in VALID_ENTITY_TYPES


def test_entity_type_definitions_stay_in_sync():
    """The three EntityType definitions must enumerate the SAME set.

    `agent` lives in two independent Literals — cognitive_memory_constants
    (the runtime gate VALID_ENTITY_TYPES + its Literal) and memory.types (the
    producer-side Literal). Drift between them would silently re-break
    agent-typing (codex review 2026-06-04). Guard it mechanically.
    """
    import typing

    from kai_memory.store.cognitive_memory_constants import EntityType as ConstType
    from kai_memory.types import EntityType as ProducerType

    const_literal = set(typing.get_args(ConstType))
    producer_literal = set(typing.get_args(ProducerType))
    assert const_literal == VALID_ENTITY_TYPES == producer_literal


def test_prompt_entity_type_list_includes_agent():
    """`agent` is in the entity-type ENUMERATION, not just incidental prose.

    The looser substring guard in test_memory_extraction passes vacuously for
    'agent' because "the agent's memories" already appears in the prompt; this
    pins it to the comma-separated type list so Haiku is actually told to emit it.
    """
    from kai_memory.extraction.memory_extraction import EXTRACTION_PROMPT

    assert "group, agent" in EXTRACTION_PROMPT


@pytest.mark.asyncio
async def test_store_item_agent_entity_passes_through_not_coerced():
    """An unrelated ``agent`` reaches find_or_create_entity as 'agent'.

    Configured self identity now has a separate resolver, tested in DB-869.

    Before INC 1, ``agent`` was unknown → coerced to 'topic' (memory_extraction
    .py:615) / 'person' (:493), collapsing Kai-the-assistant into a person named
    Kai. The resolver keys on (name.lower(), entity_type), so a real 'agent' type
    keeps it distinct.
    """
    session = AsyncMock()
    svc = ExtractionService(session)
    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()
    mock_entity = MagicMock()
    mock_entity.id = uuid.uuid4()
    ctx = ChatContext(chat_id=123, topic_id=None)
    stats = {"traces_created": 0, "entities_found": 0, "contradictions": 0}
    with (
        patch.object(svc.memory_svc, "store_trace", return_value=mock_trace),
        patch.object(svc.memory_svc, "find_or_create_entity", return_value=mock_entity) as find_mock,
        patch.object(svc.memory_svc, "link_trace_to_entity"),
    ):
        await svc._store_item(
            {
                "content": "OtherAssistant helped plan the week",
                "memory_type": "episode",
                "significance": 3,
                "entities": [{"name": "OtherAssistant", "type": "agent"}],
            },
            ctx,
            stats,
        )
    args, kwargs = find_mock.await_args
    # Robust to BOTH positional and kwarg call styles: `args[1]` IndexErrors if
    # find_or_create_entity is ever refactored to be called with entity_type=
    # as a kwarg, so guard the positional access before comparing (CR 3355128378).
    entity_type_arg = args[1] if len(args) > 1 else kwargs.get("entity_type")
    assert entity_type_arg == "agent"


# ── Step D: EXTRACTION_PROMPT + _store_item typed_metadata build ───────────


def test_build_typed_metadata_full_shape():
    """Populated item → the full §1.1 reserved-key blob."""
    from kai_memory.extraction.memory_extraction import _build_typed_metadata

    item = {
        "facts": [{"field": "role", "value": "founder", "polarity": "is"}],
        "relationships": [{"subject": "Misha", "predicate": "works_at", "object": "Examplecorp", "confidence": 0.9}],
        "entities": [{"name": "Misha", "type": "person"}],
        "confidence": 0.8,
        "is_entity_eligible": True,
    }
    tm = _build_typed_metadata(item)
    assert tm["schema_version"] == 3
    assert tm["facts"] == item["facts"]
    assert tm["relationships"] == item["relationships"]
    assert tm["entities"] == item["entities"]
    assert tm["confidence"] == 0.8
    # One model-emitted confidence feeds both D's prior and A's quality floor.
    assert tm["quality"]["extraction_confidence"] == 0.8
    assert tm["quality"]["novelty"] is None  # reserved for A (INC 5)
    assert tm["is_entity_eligible"] is True
    assert tm["query_type_hint"] is None  # reserved for OEC (M4)
    assert tm["events"] == []  # reserved-empty (Trajectory wires it later)


def test_build_typed_metadata_graceful_on_empty_item():
    """Missing keys → empty defaults, never an error (R3)."""
    from kai_memory.extraction.memory_extraction import _build_typed_metadata

    tm = _build_typed_metadata({})
    assert tm["schema_version"] == 3
    assert tm["facts"] == []
    assert tm["events"] == []
    assert tm["relationships"] == []
    assert tm["entities"] == []
    assert tm["confidence"] is None
    assert tm["quality"] == {"extraction_confidence": None, "novelty": None}
    assert tm["is_entity_eligible"] is True  # absent → eligible by default
    assert tm["query_type_hint"] is None


def test_build_typed_metadata_credential_term_marked_ineligible():
    """is_entity_eligible=false survives (credential-term guard for A/INC 5)."""
    from kai_memory.extraction.memory_extraction import _build_typed_metadata

    assert _build_typed_metadata({"is_entity_eligible": False})["is_entity_eligible"] is False


def test_build_typed_metadata_coerces_string_false_eligibility():
    """A STRING 'false' from Haiku must map to False, not bool('false')==True.

    is_entity_eligible is the credential-term guard; mislabeling a secret token
    as eligible (false-negative) is the dangerous direction (codex/CR review).
    """
    from kai_memory.extraction.memory_extraction import _build_typed_metadata

    assert _build_typed_metadata({"is_entity_eligible": "false"})["is_entity_eligible"] is False
    assert _build_typed_metadata({"is_entity_eligible": "False"})["is_entity_eligible"] is False
    assert _build_typed_metadata({"is_entity_eligible": "no"})["is_entity_eligible"] is False
    assert _build_typed_metadata({"is_entity_eligible": "true"})["is_entity_eligible"] is True
    assert _build_typed_metadata({"is_entity_eligible": None})["is_entity_eligible"] is True


def test_build_typed_metadata_coerces_non_list_slots():
    """List-slots emitted as non-lists (str/dict/int) must coerce to [] (CR 3355128362).

    Haiku can emit facts/relationships/entities/events as a bare dict, string,
    or scalar. ``item.get(key) or []`` only catches falsy values — a truthy
    non-list (``facts="x"``, ``relationships={...}``) would persist a malformed
    blob that downstream deterministic aggregation cannot read.
    """
    from kai_memory.extraction.memory_extraction import _build_typed_metadata

    tm = _build_typed_metadata(
        {
            "facts": "role: founder",  # str, not a list
            "relationships": {"subject": "x"},  # dict, not a list
            "entities": 42,  # scalar garbage
            "events": "happened",  # str, not a list
        }
    )
    assert tm["facts"] == []
    assert tm["relationships"] == []
    assert tm["entities"] == []
    assert tm["events"] == []


def test_build_typed_metadata_drops_none_entries_in_list():
    """None entries inside an otherwise-valid list-slot are dropped (normalize garbage)."""
    from kai_memory.extraction.memory_extraction import _build_typed_metadata

    tm = _build_typed_metadata({"facts": [{"field": "role"}, None, {"field": "city"}]})
    assert tm["facts"] == [{"field": "role"}, {"field": "city"}]


def test_build_typed_metadata_none_slots_still_empty_list():
    """Explicit null list-slots stay [] (the original `or []` contract holds)."""
    from kai_memory.extraction.memory_extraction import _build_typed_metadata

    tm = _build_typed_metadata({"facts": None, "entities": None, "relationships": None, "events": None})
    assert tm["facts"] == []
    assert tm["entities"] == []
    assert tm["relationships"] == []
    assert tm["events"] == []


def test_build_typed_metadata_coerces_confidence_to_float_or_none():
    """confidence must persist as float|None (CR 3356... r4).

    Haiku may emit a numeric string ('0.85'), an out-of-range number, a bool,
    or junk text — none of which are a valid float|null shape. Coerce to a
    float clamped to [0,1]; non-numeric / bool / NaN → None. Both the top-level
    slot and quality.extraction_confidence carry the coerced value.
    """
    from kai_memory.extraction.memory_extraction import _build_typed_metadata

    assert _build_typed_metadata({"confidence": "0.85"})["confidence"] == 0.85
    assert _build_typed_metadata({"confidence": 0.5})["confidence"] == 0.5
    assert _build_typed_metadata({"confidence": 1.7})["confidence"] == 1.0  # clamped
    assert _build_typed_metadata({"confidence": -0.2})["confidence"] == 0.0  # clamped
    assert _build_typed_metadata({"confidence": "junk"})["confidence"] is None
    assert _build_typed_metadata({"confidence": True})["confidence"] is None  # bool is not a confidence
    assert _build_typed_metadata({"confidence": None})["confidence"] is None
    tm = _build_typed_metadata({"confidence": "0.9"})
    assert tm["quality"]["extraction_confidence"] == 0.9


def test_scrub_typed_metadata_redacts_nested_secrets():
    """Secrets in the typed_metadata blob are scrubbed before persistence, like
    content (CR 3356167732). Recurses dict/list values; keys and non-strings
    are left intact."""
    from kai_memory.store.cognitive_memory import _scrub_typed_metadata

    blob = {
        "schema_version": 1,
        "facts": [{"field": "token", "value": "SANDBOX_CRED_aabbccddeeff00112233aabb"}],
        "entities": ["plain name"],
        "confidence": 0.8,
    }
    scrubbed = _scrub_typed_metadata(blob)
    assert scrubbed["schema_version"] == 1  # non-string untouched
    assert scrubbed["confidence"] == 0.8  # float untouched
    assert scrubbed["facts"][0]["field"] == "token"  # key/clean value intact
    assert "SANDBOX_CRED_" not in scrubbed["facts"][0]["value"]
    assert "[REDACTED:sandbox_credential]" in scrubbed["facts"][0]["value"]
    assert scrubbed["entities"] == ["plain name"]


@pytest.mark.asyncio
async def test_store_item_passes_typed_metadata_and_l2_source_path():
    session = AsyncMock()
    svc = ExtractionService(session)
    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()
    ctx = ChatContext(chat_id=123, topic_id=None)
    stats = {"traces_created": 0, "entities_found": 0, "contradictions": 0}
    facts = [{"field": "role", "value": "founder", "polarity": "is"}]
    with (
        patch.object(svc.memory_svc, "store_trace", return_value=mock_trace) as store_mock,
        patch.object(svc.memory_svc, "find_or_create_entity", return_value=MagicMock(id=uuid.uuid4())),
        patch.object(svc.memory_svc, "link_trace_to_entity"),
    ):
        await svc._store_item(
            {
                "content": "Misha founded Examplecorp",
                "memory_type": "fact",
                "significance": 3,
                "facts": facts,
                "entities": [{"name": "Misha", "type": "person"}],
            },
            ctx,
            stats,
        )
    _, kwargs = store_mock.await_args
    assert kwargs["source_path"] == "l2_inline"
    assert kwargs["typed_metadata"]["facts"] == facts
    assert kwargs["typed_metadata"]["schema_version"] == 3


@pytest.mark.asyncio
async def test_store_item_stamps_l2_invalidation_source_on_contradiction():
    session = AsyncMock()
    svc = ExtractionService(session)
    new_trace = MagicMock()
    new_trace.id = uuid.uuid4()
    old_id = uuid.uuid4()
    ctx = ChatContext(chat_id=123, topic_id=None)
    stats = {"traces_created": 0, "entities_found": 0, "contradictions": 0}
    with (
        patch.object(svc.memory_svc, "store_trace", return_value=new_trace),
        patch.object(svc.memory_svc, "get_trace", return_value=MagicMock()),
        patch.object(svc.memory_svc, "invalidate_trace") as inval_mock,
    ):
        await svc._store_item(
            {
                "content": "Misha left Examplecorp",
                "memory_type": "status_change",
                "significance": 3,
                "contradicts_memory_id": str(old_id),
                "entities": [],
            },
            ctx,
            stats,
        )
    _, kwargs = inval_mock.await_args
    assert kwargs.get("invalidation_source") == "l2_inline"


def test_prompt_requests_facts_relationships_and_eligibility():
    """The additive prompt asks Haiku for the new typed_metadata fields."""
    from kai_memory.extraction.memory_extraction import EXTRACTION_PROMPT

    assert "is_entity_eligible" in EXTRACTION_PROMPT
    assert "polarity" in EXTRACTION_PROMPT  # facts[] structure
    assert "predicate" in EXTRACTION_PROMPT  # relationships[] structure


# ── Step E: consolidation L3 source_path + invalidation_source ────────────


@pytest.mark.asyncio
async def test_self_reflect_stamps_l3_self_reflect_source_path():
    """L3 self-reflection traces carry source_path='l3_self_reflect' (empty blob)."""
    from kai_memory.consolidation.consolidation import _self_reflect

    session = AsyncMock()
    trace = MagicMock()
    trace.created_at = datetime.now(UTC)
    trace.memory_type = "fact"
    trace.content = "test memory"
    trace.invalidated_at = None
    mock_result = MagicMock()
    mock_result.scalars.return_value = MagicMock(all=MagicMock(return_value=[trace]))
    session.execute = AsyncMock(return_value=mock_result)

    reflection_items = [
        {"dimension": "interaction_quality", "observation": "People respond well", "significance": 3},
    ]
    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()
    with (
        patch("kai_memory.consolidation.consolidation._call_sonnet", return_value=reflection_items),
        patch("kai_memory.consolidation.consolidation.CognitiveMemoryService") as mock_svc_cls,
    ):
        mock_svc = AsyncMock()
        mock_svc.store_trace = AsyncMock(return_value=mock_trace)
        mock_svc_cls.return_value = mock_svc
        await _self_reflect(session)

    assert mock_svc.store_trace.await_count >= 1
    assert mock_svc.store_trace.call_args_list[0].kwargs["source_path"] == "l3_self_reflect"


@pytest.mark.asyncio
async def test_detect_contradictions_stamps_l3_batch_invalidation_source():
    """L3-batch contradiction edges are stamped invalidation_source='l3_batch'."""
    from kai_memory.consolidation.consolidation import _detect_contradictions

    new_id, old_id = uuid.uuid4(), uuid.uuid4()
    new_trace = MagicMock()
    new_trace.id = new_id
    new_trace.content = "Misha left Examplecorp"
    new_trace.invalidated_at = None
    existing = MagicMock()
    existing.id = old_id
    existing.content = "Misha works at Examplecorp"
    existing.invalidated_at = None

    session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value = MagicMock(all=MagicMock(return_value=[new_trace]))
    session.execute = AsyncMock(return_value=mock_result)

    sonnet_item = [{"new_trace_id": str(new_id), "contradicts_trace_id": str(old_id), "confidence": 0.9}]
    with (
        patch(
            "kai_memory.consolidation.consolidation.generate_embeddings_batch",
            AsyncMock(return_value=[[0.1] * 1536]),
        ),
        patch("kai_memory.consolidation.consolidation._call_sonnet", return_value=sonnet_item),
        patch("kai_memory.consolidation.consolidation.CognitiveMemoryService") as mock_svc_cls,
    ):
        mock_svc = AsyncMock()
        mock_svc.search_traces_by_vector = AsyncMock(return_value=[(existing, 0.9)])
        mock_svc.invalidate_trace = AsyncMock()
        mock_svc_cls.return_value = mock_svc
        count = await _detect_contradictions(session)

    assert count == 1
    _, kwargs = mock_svc.invalidate_trace.await_args
    assert kwargs.get("invalidation_source") == "l3_batch"


# ── INC 3 A3: subject attribution (prompt + extraction-time resolution) ─────


def test_extraction_prompt_asks_for_subject_per_fact():
    """The facts spec must request a per-fact subject (entity the fact is about)."""
    from kai_memory.extraction.memory_extraction import EXTRACTION_PROMPT

    assert "ENTITY NAME the fact is" in EXTRACTION_PROMPT
    assert '"subject": "Misha", "field": "role"' in EXTRACTION_PROMPT  # example updated


@pytest.mark.asyncio
async def test_store_item_attributes_fact_subject_id_cross_script():
    """A Cyrillic fact subject resolves to the canonical entity id at extraction
    time (via #201 canonicalization over the resolved entity's aliases)."""
    session = AsyncMock()
    svc = ExtractionService(session)
    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()
    misha = MagicMock()
    misha.id = uuid.uuid4()
    misha.name = "Misha"
    misha.entity_type = "person"
    misha.aliases = ["Миша"]
    ctx = ChatContext(chat_id=123, topic_id=None)
    stats = {"traces_created": 0, "entities_found": 0, "contradictions": 0}
    with (
        patch.object(svc.memory_svc, "store_trace", return_value=mock_trace) as store_mock,
        patch.object(svc.memory_svc, "find_or_create_entity", return_value=misha),
        patch.object(svc.memory_svc, "link_trace_to_entity"),
    ):
        await svc._store_item(
            {
                "content": "Миша is a founder",
                "memory_type": "fact",
                "significance": 3,
                "entities": [{"name": "Миша", "type": "person"}],
                "facts": [{"subject": "Миша", "field": "role", "value": "founder", "polarity": "is"}],
            },
            ctx,
            stats,
        )
    _, kwargs = store_mock.await_args
    stored_fact = kwargs["typed_metadata"]["facts"][0]
    assert stored_fact["subject_id"] == str(misha.id)  # resolved cross-script
    assert stored_fact["subject"] == "Миша"  # raw subject preserved
