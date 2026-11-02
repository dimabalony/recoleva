"""Unit tests for Level 2 memory extraction service."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_test_support import ChatContext
from memory_test_parsing import parse_json_array_response
from kai_memory.extraction.memory_extraction import (
    ExtractionService,
)
from kai_memory.storage.models.social_context import SocialContext


# ── JSON parsing tests ────────────────────────────────────


def test_parse_valid_json():
    raw = json.dumps(
        [
            {"content": "Misha works at Examplecorp", "memory_type": "fact", "significance": 3},
        ]
    )
    result = parse_json_array_response(raw)
    assert len(result) == 1
    assert result[0]["content"] == "Misha works at Examplecorp"


def test_parse_empty_array():
    assert parse_json_array_response("[]") == []


def test_parse_malformed_json_returns_empty():
    assert parse_json_array_response("not json at all") == []


def test_parse_partial_json_extracts_array():
    raw = 'Here are the results:\n```json\n[{"content": "test", "memory_type": "fact"}]\n```'
    result = parse_json_array_response(raw)
    assert len(result) == 1
    assert result[0]["content"] == "test"


def test_parse_json_object_not_array_returns_empty():
    assert parse_json_array_response('{"key": "value"}') == []


def test_parse_truncated_json():
    """Truncated JSON that can't be recovered returns empty."""
    assert parse_json_array_response('[{"content": "test"') == []


def test_parse_json_with_leading_text():
    raw = 'Based on the conversation, I extracted:\n[{"content": "fact one", "memory_type": "fact"}]'
    result = parse_json_array_response(raw)
    assert len(result) == 1


# ── Extraction service tests ─────────────────────────────


def _session_with_savepoints() -> AsyncMock:
    """A mock session whose ``begin_nested()`` is an async context manager, as each
    extracted item is written inside its own savepoint (DB-794)."""
    savepoint = MagicMock()
    savepoint.__aenter__ = AsyncMock(return_value=None)
    savepoint.__aexit__ = AsyncMock(return_value=False)
    session = AsyncMock()
    session.begin_nested = MagicMock(return_value=savepoint)
    return session


def _make_message(role="user", text="hello", metadata=None, sender_name=None, created_at=None):
    """Create a mock ConversationMessage."""
    msg = MagicMock()
    msg.role = role
    msg.text = text
    msg.sender_name = sender_name
    msg.message_metadata = metadata or {}
    msg.created_at = created_at or datetime.now(UTC)
    return msg


@pytest.mark.asyncio
async def test_extract_no_messages():
    session = AsyncMock()
    svc = ExtractionService(session)

    ctx = ChatContext(chat_id=123, topic_id=None)
    with patch.object(svc.conv_manager, "get_messages_since", return_value=[]):
        stats = await svc.extract_from_conversation(ctx, datetime.now(UTC))
    assert stats["traces_created"] == 0


@pytest.mark.asyncio
async def test_extract_too_short_conversation():
    session = AsyncMock()
    svc = ExtractionService(session)

    ctx = ChatContext(chat_id=123, topic_id=None)
    short_msg = _make_message(text="hi")
    with patch.object(svc.conv_manager, "get_messages_since", return_value=[short_msg]):
        stats = await svc.extract_from_conversation(ctx, datetime.now(UTC))
    assert stats["traces_created"] == 0


@pytest.mark.asyncio
async def test_extract_stores_traces():
    session = _session_with_savepoints()
    svc = ExtractionService(session)

    msgs = [
        _make_message(role="user", text="I just got promoted to senior engineer at Examplecorp!", sender_name="Misha"),
        _make_message(role="assistant", text="Congratulations! That's a big milestone."),
    ]

    haiku_response = {
        "memories": [
            {
                "content": "Misha got promoted to senior engineer at Examplecorp",
                "memory_type": "status_change",
                "significance": 4,
                "emotion_valence": 0.8,
                "emotion_arousal": 0.6,
                "entities": ["Misha", "Examplecorp"],
                "contradicts_memory_id": None,
            }
        ],
        "social_experiences": [],
    }

    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()
    mock_entity = MagicMock()
    mock_entity.id = uuid.uuid4()
    mock_link = MagicMock()

    ctx = ChatContext(chat_id=123, topic_id=None)
    with (
        patch.object(svc.conv_manager, "get_messages_since", return_value=msgs),
        patch.object(svc, "_call_haiku", return_value=haiku_response),
        patch.object(svc.memory_svc, "store_trace", return_value=mock_trace),
        patch.object(svc.memory_svc, "find_or_create_entity", return_value=mock_entity),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]),
        patch.object(svc.memory_svc, "link_trace_to_entity", return_value=mock_link),
        patch.object(svc, "_ensure_social_context"),
        patch("kai_memory.extraction.memory_extraction.generate_embeddings_batch", return_value=[[0.1] * 1536]),
    ):
        stats = await svc.extract_from_conversation(ctx, datetime.now(UTC))

    assert stats["traces_created"] == 1
    assert stats["entities_found"] == 2  # Misha + Examplecorp
    assert stats["contradictions"] == 0


@pytest.mark.asyncio
async def test_extract_significance_clamped():
    """Significance must be clamped to 1-5."""
    session = AsyncMock()
    svc = ExtractionService(session)

    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()

    ctx = ChatContext(chat_id=123, topic_id=None)
    with (
        patch.object(svc.memory_svc, "store_trace", return_value=mock_trace) as store_mock,
        patch.object(svc.memory_svc, "find_or_create_entity"),
        patch.object(svc.memory_svc, "link_trace_to_entity"),
    ):
        stats: dict[str, int] = {"traces_created": 0, "entities_found": 0, "contradictions": 0}
        # Test out-of-range significance values
        await svc._store_item(
            {"content": "test low", "significance": -1, "entities": []},
            ctx,
            stats,
        )
        await svc._store_item(
            {"content": "test high", "significance": 10, "entities": []},
            ctx,
            stats,
        )

    calls = store_mock.call_args_list
    assert calls[0].kwargs["significance"] == 1  # clamped from -1
    assert calls[1].kwargs["significance"] == 5  # clamped from 10


@pytest.mark.asyncio
async def test_extract_thinking_blocks_included():
    """Thinking blocks from message metadata should be fed to extraction."""
    session = AsyncMock()
    svc = ExtractionService(session)

    msgs = [
        _make_message(
            role="user",
            text="Maria is coming to visit next week, she'll bring Elena too",
            sender_name="Misha",
        ),
        _make_message(
            role="assistant",
            text="That sounds lovely!",
            metadata={
                "thinking_blocks": [
                    "Maria calling Elena implies a close relationship — possibly mother-in-law",
                    "Should remember Elena for future conversations",
                ]
            },
        ),
    ]

    ctx = ChatContext(chat_id=123, topic_id=None)
    with (
        patch.object(svc.conv_manager, "get_messages_since", return_value=msgs),
        patch.object(svc, "_call_haiku", return_value={"memories": [], "social_experiences": []}) as haiku_mock,
        patch.object(svc, "_ensure_social_context"),
    ):
        await svc.extract_from_conversation(ctx, datetime.now(UTC))

    # Verify thinking blocks were passed to _call_haiku
    call_args = haiku_mock.call_args
    thinking_text = call_args[0][1]  # second positional arg
    assert "mother-in-law" in thinking_text
    assert "Elena" in thinking_text


@pytest.mark.asyncio
async def test_native_reasoning_summary_is_labeled_and_not_a_user_statement():
    session = AsyncMock()
    svc = ExtractionService(session)
    messages = [
        _make_message(role="user", text="Please help me prepare for the family visit next week with Maria and Elena"),
        _make_message(
            role="assistant",
            text="Here is a plan for the visit.",
            metadata={"reasoning_summaries": ["Consider the family relationship when suggesting plans", None, ""]},
        ),
    ]
    with (
        patch.object(svc.conv_manager, "get_messages_since", return_value=messages),
        patch.object(svc, "_call_haiku", return_value={"memories": [], "social_experiences": []}) as extract,
        patch.object(svc, "_ensure_social_context"),
        patch.object(svc, "_load_participant_context", return_value=("", {})),
        patch.object(svc, "_load_existing_context", return_value="(none)"),
    ):
        await svc.extract_from_conversation(ChatContext(chat_id=123), datetime.now(UTC))
    conversation, reasoning = extract.call_args.args[:2]
    assert "Assistant reasoning summary, not a user statement" in reasoning
    assert "Consider the family relationship" in reasoning
    assert "Consider the family relationship" not in conversation


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_key", ["reasoning_summaries", "thinking_blocks"])
async def test_thinking_context_keeps_newest_within_extraction_budget(metadata_key):
    from kai_memory.extraction.memory_extraction import MAX_CONVERSATION_CHARS

    svc = ExtractionService(AsyncMock())
    messages = [
        _make_message(role="user", text="Please remember the family plans for next week and help prepare our visit."),
        _make_message(role="assistant", text="Earlier plan.", metadata={metadata_key: ["obsolete-history-" * 8000]}),
        _make_message(role="assistant", text="Recent plan.", metadata={metadata_key: ["middle-history-" * 8000]}),
        _make_message(role="assistant", text="Latest plan.", metadata={metadata_key: ["latest family context"]}),
    ]
    with (
        patch.object(svc.conv_manager, "get_messages_since", return_value=messages),
        patch.object(svc, "_call_haiku", return_value={"memories": [], "social_experiences": []}) as extract,
        patch.object(svc, "_ensure_social_context"),
        patch.object(svc, "_load_participant_context", return_value=("", {})),
        patch.object(svc, "_load_existing_context", return_value="(none)"),
    ):
        await svc.extract_from_conversation(ChatContext(chat_id=123), datetime.now(UTC))
    thinking = extract.call_args.args[1]
    assert len(thinking) <= MAX_CONVERSATION_CHARS
    assert thinking.endswith("latest family context")
    assert "middle-history-" in thinking
    assert "obsolete-history-" not in thinking
    if metadata_key == "reasoning_summaries":
        assert thinking.startswith("[Assistant reasoning summary, not a user statement]\n")
        assert thinking.count("[Assistant reasoning summary, not a user statement]") == 2


@pytest.mark.asyncio
async def test_extraction_prompt_includes_participants():
    """Extraction prompt should include chat participant list."""

    class Result:
        def scalar_one_or_none(self):
            return SocialContext(
                chat_id=123,
                topic_id=1,
                context_type="group",
                participants=[
                    {"telegram_user_id": 111, "name": "Elena", "username": "natasha_k"},
                    {"telegram_user_id": 222, "name": "Pavel", "username": "pavel_dev"},
                ],
            )

    session = AsyncMock()
    session.execute.return_value = Result()
    svc = ExtractionService(session)

    participant_context, _participant_map = await svc._load_participant_context(
        ChatContext(chat_id=123, topic_id=1, is_group=True)
    )
    with (
        patch("kai_memory.extraction.memory_extraction.get_provider_for_model", return_value=MagicMock()),
        patch(
            "kai_memory.extraction.memory_extraction.call_llm_text", return_value='{"memories":[],"social_experiences":[]}'
        ) as call_mock,
    ):
        await svc._call_haiku(
            "Elena said Pavel is visiting tomorrow",
            "(none)",
            "(none)",
            participant_context=participant_context,
        )

    prompt = call_mock.call_args.args[1]
    assert "natasha_k" in prompt
    assert "Elena" in prompt
    assert "matched_participant_key" in prompt


@pytest.mark.asyncio
async def test_extract_binds_matched_participant_key():
    """A validated matched_participant_key should bind entity to telegram_user_id."""
    session = AsyncMock()
    svc = ExtractionService(session)

    msgs = [
        _make_message(
            role="user",
            text="Elena said Pavel is visiting tomorrow and we should remember the plan",
            sender_name="Misha",
        ),
    ]
    haiku_response = {
        "memories": [
            {
                "content": "Elena said Pavel is visiting tomorrow",
                "memory_type": "plan",
                "significance": 3,
                "entities": [
                    {"name": "Elena", "type": "person", "matched_participant_key": "111"},
                ],
            }
        ],
        "social_experiences": [],
    }
    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()
    mock_entity = MagicMock()
    mock_entity.id = uuid.uuid4()

    ctx = ChatContext(chat_id=123, topic_id=1, is_group=True)
    with (
        patch.object(svc.conv_manager, "get_messages_since", return_value=msgs),
        patch.object(svc, "_call_haiku", return_value=haiku_response),
        patch.object(
            svc,
            "_load_participant_context",
            return_value=(
                "Known chat participants:\n- 111: Elena (@natasha_k)",
                {"111": {"telegram_user_id": 111, "name": "Elena", "username": "natasha_k"}},
            ),
        ),
        patch.object(svc.memory_svc, "store_trace", return_value=mock_trace),
        patch.object(svc.memory_svc, "find_or_create_entity", return_value=mock_entity) as find_mock,
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]),
        patch.object(svc.memory_svc, "link_trace_to_entity"),
        patch.object(svc, "_ensure_social_context"),
        patch("kai_memory.extraction.memory_extraction.generate_embeddings_batch", return_value=[[0.1] * 1536]),
    ):
        await svc.extract_from_conversation(ctx, datetime.now(UTC))

    find_mock.assert_awaited_once_with(
        "Elena", "person", telegram_user_id=111, source_text="Elena said Pavel is visiting tomorrow"
    )


@pytest.mark.asyncio
async def test_l2_extraction_does_not_bind_ungrounded_to_tg_grounded():
    session = _session_with_savepoints()
    svc = ExtractionService(session)

    msgs = [
        _make_message(
            role="user",
            text="Pavel is visiting tomorrow and I need to remember the plan",
            sender_name="Misha",
        )
    ]
    haiku_response = {
        "memories": [
            {
                "content": "Pavel is visiting tomorrow",
                "memory_type": "plan",
                "significance": 3,
                "entities": [{"name": "Pavel", "type": "person"}],
            }
        ],
        "social_experiences": [],
    }
    tg_grounded_pavel = MagicMock()
    tg_grounded_pavel.id = uuid.uuid4()
    tg_grounded_pavel.name = "Pavel"
    tg_grounded_pavel.entity_type = "person"
    tg_grounded_pavel.telegram_user_id = 100
    ungrounded_pavel = MagicMock()
    ungrounded_pavel.id = uuid.uuid4()
    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()

    ctx = ChatContext(chat_id=123, topic_id=None, is_group=True)
    with (
        patch.object(svc.conv_manager, "get_messages_since", return_value=msgs),
        patch.object(svc, "_call_haiku", return_value=haiku_response),
        patch.object(svc, "_load_existing_context", return_value="(none)"),
        patch.object(svc, "_ensure_social_context"),
        patch.object(svc.memory_svc, "store_trace", return_value=mock_trace),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]) as batch_find_mock,
        patch.object(svc.memory_svc, "find_or_create_entity", return_value=ungrounded_pavel) as find_or_create_mock,
        patch.object(svc.memory_svc, "link_trace_to_entity") as link_mock,
        patch("kai_memory.extraction.memory_extraction.generate_embeddings_batch", return_value=[[0.1] * 1536]),
    ):
        await svc.extract_from_conversation(ctx, datetime.now(UTC))

    batch_find_mock.assert_awaited_once_with(["Pavel"], only_ungrounded=True)
    find_or_create_mock.assert_awaited_once_with("Pavel", "person", source_text="Pavel is visiting tomorrow")
    link_mock.assert_awaited_once_with(mock_trace.id, ungrounded_pavel.id, "about")
    assert link_mock.call_args.args[1] != tg_grounded_pavel.id


@pytest.mark.asyncio
async def test_first_message_sender_visible_in_extraction_prompt():
    """Current sender should be registered before participant context is loaded."""
    social_context = SocialContext(
        chat_id=123,
        topic_id=None,
        context_type="group",
        participants=[],
    )
    missing_result = MagicMock()
    missing_result.scalar_one_or_none.return_value = None
    upsert_result = MagicMock()
    select_result = MagicMock()
    select_result.scalar_one_or_none.return_value = social_context
    session = AsyncMock()

    async def execute_side_effect(_stmt, _params=None):
        if session.execute.await_count == 1:
            return missing_result
        if session.execute.await_count == 2:
            social_context.participants = [
                {
                    "telegram_user_id": 12345,
                    "name": "Elena",
                    "username": None,
                    "last_seen_at": datetime.now(UTC).isoformat(),
                }
            ]
            return upsert_result
        return select_result

    session.execute = AsyncMock(side_effect=execute_side_effect)

    svc = ExtractionService(session)
    msgs = [
        _make_message(
            role="user",
            text="Elena said Pavel is visiting tomorrow and this should be remembered now",
            sender_name="Elena",
        )
    ]
    chat_ctx = ChatContext(
        chat_id=123,
        topic_id=None,
        is_group=True,
        sender_user_id=12345,
        sender_name="Elena",
    )

    with (
        patch.object(svc.conv_manager, "get_messages_since", return_value=msgs),
        patch.object(svc, "_load_existing_context", return_value="(none)"),
        patch.object(svc, "_call_haiku", return_value={"memories": [], "social_experiences": []}) as haiku_mock,
    ):
        await svc.extract_from_conversation(chat_ctx, datetime.now(UTC))

    participant_context = haiku_mock.call_args.kwargs["participant_context"]
    assert "Elena" in participant_context
    assert "12345" in participant_context


@pytest.mark.asyncio
async def test_extract_contradiction_invalidates_old():
    """When Haiku flags a contradiction, the old trace should be invalidated."""
    session = _session_with_savepoints()
    svc = ExtractionService(session)

    old_trace_id = uuid.uuid4()
    new_trace = MagicMock()
    new_trace.id = uuid.uuid4()

    haiku_response = {
        "memories": [
            {
                "content": "Misha now works at Google",
                "memory_type": "fact",
                "significance": 4,
                "entities": ["Misha", "Google"],
                "contradicts_memory_id": str(old_trace_id),
            }
        ],
        "social_experiences": [],
    }
    msgs = [
        _make_message(role="user", text="I just switched jobs, I'm at Google now", sender_name="Misha"),
        _make_message(role="assistant", text="Congratulations on the new role!"),
    ]

    mock_entity = MagicMock()
    mock_entity.id = uuid.uuid4()
    mock_link = MagicMock()

    ctx = ChatContext(chat_id=123, topic_id=None)
    with (
        patch.object(svc.conv_manager, "get_messages_since", return_value=msgs),
        patch.object(svc, "_call_haiku", return_value=haiku_response),
        patch.object(svc.memory_svc, "store_trace", return_value=new_trace),
        patch.object(svc.memory_svc, "invalidate_trace") as invalidate_mock,
        patch.object(svc.memory_svc, "find_or_create_entity", return_value=mock_entity),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]),
        patch.object(svc.memory_svc, "link_trace_to_entity", return_value=mock_link),
        patch.object(svc, "_ensure_social_context"),
        patch("kai_memory.extraction.memory_extraction.generate_embeddings_batch", return_value=[[0.1] * 1536]),
    ):
        stats = await svc.extract_from_conversation(ctx, datetime.now(UTC))

    assert stats["contradictions"] == 1
    # Sub-design C INC 1: L2-inline contradictions now stamp the correction-writer
    # so E (INC 8) can separate the L2 vs L3 streams.
    invalidate_mock.assert_awaited_once_with(
        old_trace_id,
        invalidated_by=new_trace.id,
        confidence=0.8,
        invalidation_source="l2_inline",
    )


@pytest.mark.asyncio
async def test_extract_entity_linking():
    """Each entity in extraction should be found/created and linked."""
    session = AsyncMock()
    svc = ExtractionService(session)

    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()
    mock_entity = MagicMock()
    mock_entity.id = uuid.uuid4()
    mock_link = MagicMock()

    ctx = ChatContext(chat_id=123, topic_id=None)
    with (
        patch.object(svc.memory_svc, "store_trace", return_value=mock_trace),
        patch.object(svc.memory_svc, "find_or_create_entity", return_value=mock_entity) as find_mock,
        patch.object(svc.memory_svc, "link_trace_to_entity", return_value=mock_link) as link_mock,
    ):
        stats: dict[str, int] = {"traces_created": 0, "entities_found": 0, "contradictions": 0}
        await svc._store_item(
            {
                "content": "Misha and Maria went hiking",
                "memory_type": "episode",
                "significance": 3,
                "entities": ["Misha", "Maria"],
            },
            ctx,
            stats,
        )

    assert find_mock.await_count == 2
    assert link_mock.await_count == 2
    assert stats["entities_found"] == 2


@pytest.mark.asyncio
async def test_extract_typed_entity_linking():
    """Typed entity dicts should pass the correct entity_type to find_or_create_entity."""
    session = AsyncMock()
    svc = ExtractionService(session)

    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()
    mock_entity = MagicMock()
    mock_entity.id = uuid.uuid4()
    mock_link = MagicMock()

    ctx = ChatContext(chat_id=123, topic_id=None)
    with (
        patch.object(svc.memory_svc, "store_trace", return_value=mock_trace),
        patch.object(svc.memory_svc, "find_or_create_entity", return_value=mock_entity) as find_mock,
        patch.object(svc.memory_svc, "link_trace_to_entity", return_value=mock_link),
    ):
        stats: dict[str, int] = {"traces_created": 0, "entities_found": 0, "contradictions": 0}
        await svc._store_item(
            {
                "content": "Misha works at Examplecorp",
                "memory_type": "fact",
                "significance": 3,
                "entities": [
                    {"name": "Misha", "type": "person"},
                    {"name": "Examplecorp", "type": "organization"},
                ],
            },
            ctx,
            stats,
        )

    assert find_mock.await_count == 2
    calls = find_mock.call_args_list
    assert calls[0].args == ("Misha", "person")
    assert calls[1].args == ("Examplecorp", "organization")
    assert stats["entities_found"] == 2


@pytest.mark.asyncio
async def test_extract_empty_entity_names_skipped():
    """Empty or whitespace entity names should be skipped."""
    session = AsyncMock()
    svc = ExtractionService(session)

    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()

    ctx = ChatContext(chat_id=123, topic_id=None)
    with (
        patch.object(svc.memory_svc, "store_trace", return_value=mock_trace),
        patch.object(svc.memory_svc, "find_or_create_entity") as find_mock,
        patch.object(svc.memory_svc, "link_trace_to_entity"),
    ):
        stats: dict[str, int] = {"traces_created": 0, "entities_found": 0, "contradictions": 0}
        await svc._store_item(
            {
                "content": "test",
                "significance": 3,
                "entities": [
                    "",
                    " ",
                    None,
                    "Valid",
                    {"name": "", "type": "person"},
                    {"name": "AlsoValid", "type": "organization"},
                ],
            },
            ctx,
            stats,
        )

    # "Valid" (string) + "AlsoValid" (dict) should be processed
    assert find_mock.await_count == 2
    assert stats["entities_found"] == 2


@pytest.mark.asyncio
async def test_extract_valid_empty_result_returns_empty_stats():
    """A completed extraction with no memories legitimately returns empty stats."""
    session = AsyncMock()
    svc = ExtractionService(session)

    msgs = [
        _make_message(role="user", text="A long enough conversation for extraction testing here, more words"),
        _make_message(role="assistant", text="Indeed, this is quite a substantial discussion about many topics"),
    ]

    ctx = ChatContext(chat_id=123, topic_id=None)
    with (
        patch.object(svc.conv_manager, "get_messages_since", return_value=msgs),
        patch.object(svc, "_call_haiku", return_value={"memories": [], "social_experiences": []}),
        patch.object(svc, "_ensure_social_context"),
    ):
        stats = await svc.extract_from_conversation(ctx, datetime.now(UTC))
    # _call_haiku returned empty result — no items to store
    assert stats["traces_created"] == 0


@pytest.mark.parametrize("completion", [None, ""])
async def test_call_haiku_without_completion_fails(completion):
    """Missing inference output is a failure, not evidence of no memories."""
    svc = ExtractionService(AsyncMock(), llm_text_fn=AsyncMock(return_value=completion))
    with (
        patch("kai_memory.extraction.memory_extraction.get_settings", return_value=object()),
        patch("kai_memory.extraction.memory_extraction.get_provider_for_model", return_value=object()),
        patch("kai_memory.extraction.memory_extraction.start_generation_span", return_value=None),
        pytest.raises(RuntimeError, match=r"^memory extraction returned no completion$"),
    ):
        await svc._call_haiku("synthetic conversation", "", "")


@pytest.mark.asyncio
async def test_extract_store_item_failure_continues():
    """One item failing to store should not block other items."""
    session = _session_with_savepoints()
    svc = ExtractionService(session)

    msgs = [
        _make_message(role="user", text="Multiple facts in one conversation that should be extracted separately"),
        _make_message(role="assistant", text="I understand, there are several important points to remember"),
    ]

    haiku_response = {
        "memories": [
            {"content": "fact 1", "memory_type": "fact", "significance": 3, "entities": []},
            {"content": "fact 2", "memory_type": "fact", "significance": 3, "entities": []},
        ],
        "social_experiences": [],
    }

    call_count = 0
    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()

    async def store_trace_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("DB error on first item")
        return mock_trace

    ctx = ChatContext(chat_id=123, topic_id=None)
    with (
        patch.object(svc.conv_manager, "get_messages_since", return_value=msgs),
        patch.object(svc, "_call_haiku", return_value=haiku_response),
        patch.object(svc.memory_svc, "store_trace", side_effect=store_trace_side_effect),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]),
        patch.object(svc, "_ensure_social_context"),
        patch(
            "kai_memory.extraction.memory_extraction.generate_embeddings_batch",
            return_value=[[0.1] * 1536, [0.1] * 1536],
        ),
    ):
        stats = await svc.extract_from_conversation(ctx, datetime.now(UTC))

    # First item failed, second succeeded; the failure is counted, not hidden
    assert stats["traces_created"] == 1
    assert stats["store_failures"] == 1


@pytest.mark.asyncio
async def test_extract_episode_vs_fact():
    """Different memory types should be stored with correct type."""
    session = AsyncMock()
    svc = ExtractionService(session)

    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()

    ctx = ChatContext(chat_id=123, topic_id=None)
    with (
        patch.object(svc.memory_svc, "store_trace", return_value=mock_trace) as store_mock,
    ):
        stats: dict[str, int] = {"traces_created": 0, "entities_found": 0, "contradictions": 0}
        await svc._store_item(
            {"content": "Tense deadline conversation", "memory_type": "episode", "significance": 4, "entities": []},
            ctx,
            stats,
        )
        await svc._store_item(
            {"content": "Misha works at Examplecorp", "memory_type": "fact", "significance": 3, "entities": []},
            ctx,
            stats,
        )

    calls = store_mock.call_args_list
    assert calls[0].kwargs["memory_type"] == "episode"
    assert calls[1].kwargs["memory_type"] == "fact"


@pytest.mark.asyncio
async def test_extract_empty_content_skipped():
    """Items with empty content should be skipped."""
    session = AsyncMock()
    svc = ExtractionService(session)

    ctx = ChatContext(chat_id=123, topic_id=None)
    with patch.object(svc.memory_svc, "store_trace") as store_mock:
        stats: dict[str, int] = {"traces_created": 0, "entities_found": 0, "contradictions": 0}
        await svc._store_item({"content": "", "significance": 3, "entities": []}, ctx, stats)
        await svc._store_item({"content": "   ", "significance": 3, "entities": []}, ctx, stats)

    store_mock.assert_not_awaited()
    assert stats["traces_created"] == 0


@pytest.mark.asyncio
async def test_extract_invalid_contradiction_id_ignored():
    """Invalid UUID in contradicts_memory_id should be silently ignored."""
    session = AsyncMock()
    svc = ExtractionService(session)

    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()

    ctx = ChatContext(chat_id=123, topic_id=None)
    with (
        patch.object(svc.memory_svc, "store_trace", return_value=mock_trace),
        patch.object(svc.memory_svc, "invalidate_trace") as invalidate_mock,
    ):
        stats: dict[str, int] = {"traces_created": 0, "entities_found": 0, "contradictions": 0}
        await svc._store_item(
            {"content": "test", "significance": 3, "entities": [], "contradicts_memory_id": "not-a-uuid"},
            ctx,
            stats,
        )

    invalidate_mock.assert_not_awaited()
    assert stats["contradictions"] == 0


@pytest.mark.asyncio
async def test_ensure_social_context_creates_on_first():
    """Social context should be auto-created on first extraction."""
    session = AsyncMock()
    select_result = MagicMock()
    select_result.scalar_one_or_none.return_value = None
    insert_result = MagicMock()
    session.execute = AsyncMock(side_effect=[select_result, insert_result])
    svc = ExtractionService(session)

    ctx = ChatContext(chat_id=123, topic_id=456, is_group=True)
    await svc._ensure_social_context(ctx)
    assert session.execute.await_count == 2

    # Verify the SQL uses ON CONFLICT DO UPDATE
    call_args = session.execute.call_args_list[1]
    stmt = call_args[0][0]
    compiled = stmt.compile(compile_kwargs={"literal_binds": True})
    sql_str = str(compiled)
    assert "ON CONFLICT" in sql_str
    assert "social_contexts" in sql_str


@pytest.mark.asyncio
async def test_ensure_social_context_freshness_short_circuit():
    """Fresh unchanged social context should skip the second UPSERT."""
    session = AsyncMock()
    now = datetime.now(UTC)
    fresh_context = SocialContext(
        chat_id=123,
        topic_id=None,
        context_type="group",
        context_name="Family Group",
        last_active_at=now,
    )
    missing_result = MagicMock()
    missing_result.scalar_one_or_none.return_value = None
    fresh_result = MagicMock()
    fresh_result.scalar_one_or_none.return_value = fresh_context
    insert_result = MagicMock()
    session.execute = AsyncMock(side_effect=[missing_result, insert_result, fresh_result])
    svc = ExtractionService(session)

    ctx = ChatContext(chat_id=123, topic_id=None, is_group=True, chat_title="Family Group")
    with patch("kai_memory.extraction.memory_extraction.utc_now", return_value=now):
        await svc._ensure_social_context(ctx)
        await svc._ensure_social_context(ctx)

    assert session.execute.await_count == 3
    second_call_sql = str(session.execute.call_args_list[1][0][0].compile(compile_kwargs={"literal_binds": True}))
    assert "ON CONFLICT" in second_call_sql


@pytest.mark.asyncio
async def test_upsert_participant_uses_atomic_sql():
    session = AsyncMock()
    session.execute = AsyncMock()
    svc = ExtractionService(session)

    await svc._upsert_participant(
        ChatContext(
            chat_id=123,
            topic_id=None,
            sender_user_id=42,
            sender_name="Misha Petrov",
            sender_username="misha",
        ),
        datetime.now(UTC),
    )

    session.execute.assert_awaited_once()
    stmt, params = session.execute.await_args.args
    sql = str(stmt)
    assert "UPDATE social_contexts" in sql
    assert "jsonb_agg" in sql
    assert "IS DISTINCT FROM" in sql
    assert "topic_id IS NOT DISTINCT FROM :topic_id" in sql
    assert params["tg_id_text"] == "42"
    participant = json.loads(params["new_participant"])
    assert participant["telegram_user_id"] == 42


@pytest.mark.asyncio
async def test_upsert_participant_skips_when_sender_user_id_none():
    session = AsyncMock()
    session.execute = AsyncMock()
    svc = ExtractionService(session)

    await svc._upsert_participant(ChatContext(chat_id=123, sender_user_id=None), datetime.now(UTC))

    session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_upsert_participant_handles_malformed_entry():
    session = AsyncMock()
    session.execute = AsyncMock()
    svc = ExtractionService(session)

    await svc._upsert_participant(ChatContext(chat_id=123, sender_user_id=42), datetime.now(UTC))

    sql = str(session.execute.await_args.args[0])
    assert "elem->>'telegram_user_id' IS DISTINCT FROM" in sql
    assert "::bigint" not in sql


def test_participant_seen_after_missing_last_seen_included():
    assert ExtractionService._participant_seen_after({}, datetime.now(UTC)) is True


@pytest.mark.asyncio
async def test_extracts_social_experiences():
    """Haiku response with social_experiences should persist them."""
    session = _session_with_savepoints()
    svc = ExtractionService(session)

    msgs = [
        _make_message(role="user", text="Don't talk about my salary in this group, please", sender_name="Misha"),
        _make_message(role="assistant", text="Understood, I'll keep that private."),
    ]

    haiku_response = {
        "memories": [],
        "social_experiences": [
            {
                "experience": "Misha asked not to discuss salary in the group chat",
                "learned_norm": "Financial details are private in group contexts",
                "signal_type": "boundary",
            }
        ],
    }

    # Mock session.execute to return a fake context_id
    fake_context_id = uuid.uuid4()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=fake_context_id)
    session.execute = AsyncMock(return_value=mock_result)
    session.flush = AsyncMock()

    ctx = ChatContext(chat_id=123, topic_id=None, is_group=True)
    with (
        patch.object(svc.conv_manager, "get_messages_since", return_value=msgs),
        patch.object(svc, "_call_haiku", return_value=haiku_response),
        patch.object(svc, "_ensure_social_context"),
        patch.object(svc, "_load_existing_context", return_value="(none)"),
    ):
        stats = await svc.extract_from_conversation(ctx, datetime.now(UTC))

    assert stats["social_experiences"] == 1
    # Verify an experience was added to session
    session.add.assert_called()
    added = session.add.call_args[0][0]
    assert added.experience == "Misha asked not to discuss salary in the group chat"
    assert added.learned_norm == "Financial details are private in group contexts"
    assert added.signal_type == "boundary"
    assert added.salience == 0.7


@pytest.mark.asyncio
async def test_social_store_failure_logs_only_exception_type(caplog):
    from sqlalchemy.exc import DBAPIError

    session = _session_with_savepoints()
    svc = ExtractionService(session)
    secret = "private-social-candidate-7f41"
    statement = "INSERT INTO social_experiences (experience) VALUES (:experience)"
    error = DBAPIError(statement, {"experience": secret}, RuntimeError(secret))
    messages = [_make_message(text="A conversation long enough to extract a social experience from this group.")]
    response = {"memories": [], "social_experiences": [{"experience": secret}]}
    with (
        patch.object(svc.conv_manager, "get_messages_since", return_value=messages),
        patch.object(svc, "_call_haiku", return_value=response),
        patch.object(svc, "_ensure_social_context"),
        patch.object(svc, "_load_existing_context", return_value="(none)"),
        patch.object(svc, "_load_participant_context", return_value=("", {})),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]),
        patch.object(svc, "_store_social_experiences", side_effect=error) as store,
    ):
        stats = await svc.extract_from_conversation(ChatContext(chat_id=123, is_group=True), datetime.now(UTC))

    store.assert_awaited_once()
    assert stats["store_failures"] == 1
    assert svc.store_errors == ["DBAPIError"]
    [record] = [r for r in caplog.records if r.getMessage().startswith("Failed to store social experiences")]
    assert record.getMessage() == "Failed to store social experiences type=DBAPIError"
    assert record.exc_info is None and record.exc_text is None
    assert secret not in caplog.text and statement not in caplog.text


@pytest.mark.asyncio
async def test_store_social_experiences_invalid_signal_type():
    """Invalid signal_type should be coerced to None, not rejected."""
    session = AsyncMock()
    svc = ExtractionService(session)

    fake_context_id = uuid.uuid4()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=fake_context_id)
    session.execute = AsyncMock(return_value=mock_result)
    session.flush = AsyncMock()

    ctx = ChatContext(chat_id=123, topic_id=None)
    stats = {"traces_created": 0, "entities_found": 0, "contradictions": 0, "social_experiences": 0}
    experiences = [
        {
            "experience": "valid experience text",
            "learned_norm": None,
            "signal_type": "invalid_type",  # not in _VALID_SIGNAL_TYPES
        }
    ]
    await svc._store_social_experiences(ctx, experiences, stats)

    session.add.assert_called()
    added = session.add.call_args[0][0]
    assert added.signal_type is None
    assert stats["social_experiences"] == 1


# ── Task 4: Type discipline (EntityType expansion + memory_type drift normalizer) ─────


def test_valid_entity_types_has_ten_values():
    """v2a §5.1 (+ Sub-design C INC 1): EntityType 5 → 9 → 10 (added `agent`)."""
    from kai_memory.store.cognitive_memory_constants import VALID_ENTITY_TYPES

    assert VALID_ENTITY_TYPES == {
        "person",
        "animal",
        "place",
        "organization",
        "topic",
        "project",
        "object",
        "event",
        "group",
        "agent",
    }


def test_extraction_prompt_lists_ten_entity_types():
    """v2a §5.3 (+ Sub-design C INC 1): prompt enumerates all 10 entity types."""
    from kai_memory.extraction.memory_extraction import EXTRACTION_PROMPT

    for t in (
        "person",
        "animal",
        "place",
        "organization",
        "topic",
        "project",
        "object",
        "event",
        "group",
        "agent",
    ):
        assert t in EXTRACTION_PROMPT, f"entity_type {t!r} missing from prompt"


def test_extraction_prompt_lists_eleven_memory_types_with_rubric():
    """v2a §5.8: 11-value MemoryType rubric with definitions appears in prompt."""
    from kai_memory.extraction.memory_extraction import EXTRACTION_PROMPT

    for t in (
        "fact",
        "episode",
        "impression",
        "preference",
        "relationship",
        "plan",
        "intention",
        "status_change",
        "decision",
        "social_norm",
        "self_reflection",
    ):
        assert t in EXTRACTION_PROMPT, f"memory_type {t!r} missing from prompt"
    # Anti-invention instruction (v2a §5.8)
    assert "do not invent" in EXTRACTION_PROMPT.lower() or "do not introduce" in EXTRACTION_PROMPT.lower()


def test_memory_type_normalizer_maps_observed_drift():
    """v2a §5.8: normalizer dict maps Haiku drift to canonical types."""
    from kai_memory.extraction.memory_extraction import _MEMORY_TYPE_NORMALIZER

    # Required mappings per design §5.8
    assert _MEMORY_TYPE_NORMALIZER["estimate"] == "fact"
    assert _MEMORY_TYPE_NORMALIZER["observation"] == "impression"
    assert _MEMORY_TYPE_NORMALIZER["insight"] == "impression"
    assert _MEMORY_TYPE_NORMALIZER["boundary"] == "social_norm"
    assert _MEMORY_TYPE_NORMALIZER["rule"] == "social_norm"
    assert _MEMORY_TYPE_NORMALIZER["opinion"] == "preference"
    assert _MEMORY_TYPE_NORMALIZER["recommendation"] == "plan"
    assert _MEMORY_TYPE_NORMALIZER["advice"] == "plan"
    assert _MEMORY_TYPE_NORMALIZER["task"] == "intention"
    assert _MEMORY_TYPE_NORMALIZER["todo"] == "intention"


@pytest.mark.asyncio
async def test_store_item_dict_entity_without_type_defaults_to_topic(caplog):
    """v2a §5.2: dict entity with missing/empty type → 'topic' (NOT 'person')."""
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
                "content": "облачки woke us up",
                "memory_type": "episode",
                "significance": 3,
                "entities": [{"name": "облачки"}],  # no `type`
            },
            ctx,
            stats,
        )

    # Entity created as 'topic', not 'person'
    args, kwargs = find_mock.await_args
    # signature is (name, entity_type, ...)
    assert args[1] == "topic" or kwargs.get("entity_type") == "topic"


@pytest.mark.asyncio
async def test_store_item_string_entity_defaults_to_topic():
    """v2a §5.2: bare string entity → 'topic' (NOT 'person')."""
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
                "content": "Got an S3 error",
                "memory_type": "fact",
                "significance": 2,
                "entities": ["S3"],
            },
            ctx,
            stats,
        )

    args, kwargs = find_mock.await_args
    assert args[1] == "topic" or kwargs.get("entity_type") == "topic"


@pytest.mark.asyncio
async def test_store_item_unknown_entity_type_coerces_to_topic():
    """v2a §5.2: unknown entity_type → 'topic' with warning (NOT 'person')."""
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
                "content": "We adopted a wallaby",
                "memory_type": "fact",
                "significance": 3,
                "entities": [{"name": "wallaby", "type": "creature"}],
            },
            ctx,
            stats,
        )

    args, kwargs = find_mock.await_args
    assert args[1] == "topic" or kwargs.get("entity_type") == "topic"


@pytest.mark.asyncio
async def test_store_item_animal_entity_type_passes_through():
    """v2a §5.1: 'animal' is a valid entity_type now (was coerced to 'person' before)."""
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
                "content": "Got a new dog named Луна",
                "memory_type": "fact",
                "significance": 4,
                "entities": [{"name": "Луна", "type": "animal"}],
            },
            ctx,
            stats,
        )

    args, kwargs = find_mock.await_args
    assert args[1] == "animal" or kwargs.get("entity_type") == "animal"


@pytest.mark.asyncio
async def test_store_item_memory_type_normalized_emits_event(caplog):
    """v2a §5.8: drift type 'estimate' → 'fact' with type_normalized event."""
    import logging

    session = AsyncMock()
    svc = ExtractionService(session)

    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()

    ctx = ChatContext(chat_id=123, topic_id=None)
    stats = {"traces_created": 0, "entities_found": 0, "contradictions": 0}
    with (
        caplog.at_level(logging.INFO, logger="app.services.chat.memory_extraction"),
        patch.object(svc.memory_svc, "store_trace", return_value=mock_trace) as store_mock,
    ):
        await svc._store_item(
            {"content": "Customer count ~50K", "memory_type": "estimate", "significance": 3, "entities": []},
            ctx,
            stats,
        )

    # Stored as 'fact'
    assert store_mock.await_args.kwargs["memory_type"] == "fact"
    # Hotfix 2026-05-01: payload is now JSON-encoded into the message body.
    matching = []
    for r in caplog.records:
        msg = r.getMessage()
        prefix = "memory.encoding.type_normalized "
        if msg.startswith(prefix):
            matching.append(json.loads(msg[len(prefix) :]))
    assert matching, "expected memory.encoding.type_normalized event"
    rec = matching[0]
    assert rec["raw"] == "estimate"
    assert rec["resolved"] == "fact"


@pytest.mark.asyncio
async def test_store_item_memory_type_coerced_emits_event(caplog):
    """v2a §5.8: completely unknown type → 'fact' with type_coerced event."""
    import logging

    session = AsyncMock()
    svc = ExtractionService(session)

    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()

    ctx = ChatContext(chat_id=123, topic_id=None)
    stats = {"traces_created": 0, "entities_found": 0, "contradictions": 0}
    with (
        caplog.at_level(logging.WARNING, logger="app.services.chat.memory_extraction"),
        patch.object(svc.memory_svc, "store_trace", return_value=mock_trace) as store_mock,
    ):
        await svc._store_item(
            {"content": "X", "memory_type": "speculation", "significance": 2, "entities": []},
            ctx,
            stats,
        )

    assert store_mock.await_args.kwargs["memory_type"] == "fact"
    matching = []
    for r in caplog.records:
        msg = r.getMessage()
        prefix = "memory.encoding.type_coerced "
        if msg.startswith(prefix):
            matching.append(json.loads(msg[len(prefix) :]))
    assert matching, "expected memory.encoding.type_coerced event"
    rec = matching[0]
    assert rec["raw"] == "speculation"
    assert rec["resolved"] == "fact"


@pytest.mark.asyncio
async def test_store_item_valid_memory_type_emits_no_drift_event(caplog):
    """Valid types pass through cleanly with no normalize/coerce logs."""
    import logging

    session = AsyncMock()
    svc = ExtractionService(session)

    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()

    ctx = ChatContext(chat_id=123, topic_id=None)
    stats = {"traces_created": 0, "entities_found": 0, "contradictions": 0}
    with (
        caplog.at_level(logging.INFO, logger="app.services.chat.memory_extraction"),
        patch.object(svc.memory_svc, "store_trace", return_value=mock_trace) as store_mock,
    ):
        await svc._store_item(
            {"content": "Misha works at Examplecorp", "memory_type": "fact", "significance": 3, "entities": []},
            ctx,
            stats,
        )

    assert store_mock.await_args.kwargs["memory_type"] == "fact"
    # Hotfix 2026-05-01: drift events live in the message body, not extras.
    drift_prefixes = ("memory.encoding.type_normalized ", "memory.encoding.type_coerced ")
    assert not any(r.getMessage().startswith(p) for r in caplog.records for p in drift_prefixes)


@pytest.mark.asyncio
async def test_store_item_passes_entity_data_to_find_or_create():
    """v2a §5.4: dict entity with `data` passes it through to the service."""
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
                "content": "We adopted Луна the labrador",
                "memory_type": "fact",
                "significance": 4,
                "entities": [
                    {
                        "name": "Луна",
                        "type": "animal",
                        "data": {"breed": "labrador", "color": "white"},
                    }
                ],
            },
            ctx,
            stats,
        )

    find_mock.assert_awaited_once_with(
        "Луна", "animal", data={"breed": "labrador", "color": "white"}, source_text="We adopted Луна the labrador"
    )


# ── Task 7: Temporal grounding ────────────────────────────


def test_extraction_prompt_has_current_timestamp_placeholder():
    """v2a §6.1: prompt template carries a {current_timestamp} slot."""
    from kai_memory.extraction.memory_extraction import EXTRACTION_PROMPT

    assert "{current_timestamp}" in EXTRACTION_PROMPT


def test_extraction_prompt_explains_per_message_timestamps():
    """v2a §6.1: prompt instructs model to use per-message ISO-8601 prefixes."""
    from kai_memory.extraction.memory_extraction import EXTRACTION_PROMPT

    lower = EXTRACTION_PROMPT.lower()
    assert "iso-8601" in lower or "iso 8601" in lower or "[2026-" in EXTRACTION_PROMPT
    # Must mention "timestamp" somewhere
    assert "timestamp" in lower


@pytest.mark.asyncio
async def test_call_haiku_passes_current_timestamp_and_message_prefixes():
    """v2a §6.1: _call_haiku injects current_timestamp; messages get [iso] prefix."""
    session = AsyncMock()
    svc = ExtractionService(session)

    msgs = [
        _make_message(role="user", text="Got the dog vaccinated", sender_name="Misha"),
    ]
    # Set a deterministic timestamp on the message
    msgs[0].created_at = datetime(2026, 4, 17, 9, 0, tzinfo=UTC)
    # Freeze the clock so the rendered current_timestamp is deterministic.
    fixed_now = datetime(2026, 4, 17, 9, 5, tzinfo=UTC)

    ctx = ChatContext(chat_id=123, topic_id=None)
    with (
        patch.object(svc.conv_manager, "get_messages_since", return_value=msgs),
        patch("kai_memory.extraction.memory_extraction.get_provider_for_model", return_value=MagicMock()),
        patch(
            "kai_memory.extraction.memory_extraction.call_llm_text",
            return_value='{"memories":[],"social_experiences":[]}',
        ) as call_mock,
        patch.object(svc, "_ensure_social_context"),
        patch.object(svc, "_load_existing_context", return_value="(none)"),
        patch("kai_memory.extraction.memory_extraction.generate_embeddings_batch", return_value=[]),
        patch("kai_memory.extraction.memory_extraction.utc_now", return_value=fixed_now),
    ):
        await svc.extract_from_conversation(ctx, datetime.now(UTC))

    rendered = call_mock.call_args.args[1]
    # Per-message timestamp prefix
    assert "[2026-04-17T09:00" in rendered, f"missing per-message ISO prefix: {rendered[-300:]}"
    # The {current_timestamp} placeholder must be filled (not literal)
    assert "{current_timestamp}" not in rendered
    # Exact ISO-8601 of the frozen "now" must appear in the rendered prompt.
    assert fixed_now.isoformat() in rendered, f"expected current_timestamp {fixed_now.isoformat()} in rendered prompt"


# ── review-r1 fix 1: source_text plumbing into find_or_create_entity ────


@pytest.mark.asyncio
async def test_store_item_passes_content_as_source_text():
    """v2a §5.5: extraction must thread `source_text` so reverse-source-scan fires."""
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
                "content": "Сегодня встретили Миша в парке",
                "memory_type": "episode",
                "significance": 3,
                "entities": [{"name": "Misha", "type": "person"}],
            },
            ctx,
            stats,
        )

    kwargs = find_mock.await_args.kwargs
    assert kwargs.get("source_text") == "Сегодня встретили Миша в парке", (
        "source_text was not threaded — Task 6 reverse-source-scan is dead in prod"
    )


@pytest.mark.asyncio
async def test_resolve_matched_participant_passes_source_text():
    """`_resolve_matched_participant_entities` should also thread source_text."""
    session = AsyncMock()
    svc = ExtractionService(session)

    mock_entity = MagicMock()
    mock_entity.id = uuid.uuid4()
    items = [
        {
            "content": "Elena сказала привет",
            "entities": [
                {"name": "Elena", "type": "person", "matched_participant_key": "111"},
            ],
        }
    ]
    participant_map = {"111": {"telegram_user_id": 111, "name": "Елена", "username": "n"}}

    with patch.object(svc.memory_svc, "find_or_create_entity", return_value=mock_entity) as find_mock:
        await svc._resolve_matched_participant_entities(items, participant_map)

    kwargs = find_mock.await_args.kwargs
    assert kwargs.get("source_text") == "Elena сказала привет"


# ── review-r1 fix 2: resolver default aligns with _store_item ───


@pytest.mark.asyncio
async def test_matched_participant_without_type_keeps_person_binding():
    """Ultrareview critical: when Haiku omits a 'type' key on a
    matched_participant entity, the resolver must mutate
    raw_entity['type'] to 'person' so _store_item later reads the same
    value. Without this, the resolver's entity_map uses key
    (name, 'person') but _store_item's lookup uses (name, 'topic')
    (the Wave 2 default for missing type) → miss → fresh ungrounded
    entity → telegram binding silently lost.
    """
    session = AsyncMock()
    svc = ExtractionService(session)

    mock_entity = MagicMock()
    mock_entity.id = uuid.uuid4()
    raw_entity = {"name": "Elena", "matched_participant_key": "111"}
    items = [
        {
            "content": "Elena сказала привет",
            "entities": [raw_entity],
        }
    ]
    participant_map = {"111": {"telegram_user_id": 111, "name": "Елена", "username": "n"}}

    with patch.object(svc.memory_svc, "find_or_create_entity", return_value=mock_entity) as find_mock:
        entity_map = await svc._resolve_matched_participant_entities(items, participant_map)

    args, _kwargs = find_mock.await_args
    assert args[1] == "person"
    assert ("elena", "person") in entity_map
    assert raw_entity["type"] == "person"


# ── INC 3: schema-constrained extraction (schema_version 3) ────────────────


def test_schema_version_is_3():
    """TYPED_METADATA_SCHEMA_VERSION must be 3 (INC 3)."""
    from kai_memory.extraction.memory_extraction import TYPED_METADATA_SCHEMA_VERSION

    assert TYPED_METADATA_SCHEMA_VERSION == 3


def test_build_typed_metadata_schema_version_3():
    """_build_typed_metadata must include schema_version == 3."""
    from kai_memory.extraction.memory_extraction import _build_typed_metadata

    result = _build_typed_metadata({"content": "x", "significance": 3, "entities": [], "facts": []})
    assert result["schema_version"] == 3


def test_render_schema_for_prompt_includes_person_slots():
    """render_schema_for_prompt must list the person-type durable slots."""
    from kai_memory.durable_schema import render_schema_for_prompt

    catalog = render_schema_for_prompt()
    assert "person:" in catalog
    for slot_key in ("current_location", "occupation", "employer", "citizenship", "key_relationships"):
        assert slot_key in catalog, f"expected person slot '{slot_key}' in catalog"


def test_render_schema_for_prompt_omits_fieldless_types():
    """Fieldless entity types must NOT appear in the catalog.

    project/agent moved OUT of this set on 2026-07-02 (non-person coverage —
    they carry generic slot groups now); topic/object/event/group stay fieldless.
    """
    from kai_memory.durable_schema import render_schema_for_prompt

    catalog = render_schema_for_prompt()
    for fieldless in ("topic:", "object:", "event:", "group:"):
        assert fieldless not in catalog, f"fieldless type '{fieldless}' must be omitted from catalog"


def test_extraction_prompt_contains_schema_catalog_placeholder():
    """EXTRACTION_PROMPT must have the {schema_catalog} placeholder for the slot list."""
    from kai_memory.extraction.memory_extraction import EXTRACTION_PROMPT

    assert "{schema_catalog}" in EXTRACTION_PROMPT


def test_extraction_prompt_mentions_slot_and_durability():
    """EXTRACTION_PROMPT must instruct Haiku to emit 'slot' and 'durability'."""
    from kai_memory.extraction.memory_extraction import EXTRACTION_PROMPT

    assert '"slot"' in EXTRACTION_PROMPT or "slot:" in EXTRACTION_PROMPT
    assert '"durability"' in EXTRACTION_PROMPT or "durability:" in EXTRACTION_PROMPT


# ── _validate_and_default_fact_slots unit tests ────────────────────────────


def test_validate_defaults_missing_slot_and_durability():
    """Missing slot and durability fields default to 'off_schema' and 'transient'."""
    from kai_memory.extraction.memory_extraction import _validate_and_default_fact_slots

    facts = [{"subject": "Misha", "field": "something", "value": "x", "polarity": "is"}]
    _validate_and_default_fact_slots(facts, [])
    assert facts[0]["slot"] == "off_schema"
    assert facts[0]["durability"] == "transient"


def test_validate_defaults_invalid_durability_to_transient():
    """An invalid durability value (e.g. 'unknown') defaults to 'transient'."""
    from kai_memory.extraction.memory_extraction import _validate_and_default_fact_slots

    facts = [
        {
            "subject": "X",
            "field": "f",
            "value": "v",
            "polarity": "is",
            "slot": "off_schema",
            "durability": "unknown_value",
        }
    ]
    _validate_and_default_fact_slots(facts, [])
    assert facts[0]["durability"] == "transient"


def test_validate_off_schema_durability_computed_transient():
    """Hardening D-1: code-stamped durability is COMPUTED from the resolved slot, not
    trusted from Haiku. An off_schema fact is transient even when Haiku says
    'operational'. R-1: the raw tri-state opinion is preserved as claimed_durability."""
    from kai_memory.extraction.memory_extraction import _validate_and_default_fact_slots

    facts = [
        {
            "subject": "agent",
            "field": "exit_code",
            "value": "0",
            "polarity": "is",
            "slot": "off_schema",
            "durability": "operational",
        }
    ]
    _validate_and_default_fact_slots(facts, [])
    assert facts[0]["durability"] == "transient"  # retention signal (binary)
    assert facts[0]["claimed_durability"] == "operational"  # raw tri-state preserved (R-1)
    assert facts[0]["slot"] == "off_schema"


def test_validate_valid_person_slot_with_resolved_subject_id():
    """A valid schema slot for a person entity with a resolved subject_id passes through."""
    from kai_memory.extraction.memory_extraction import _validate_and_default_fact_slots

    person_id = uuid.uuid4()
    entity = MagicMock()
    entity.id = person_id
    entity.entity_type = "person"

    facts = [
        {
            "subject": "Misha",
            "field": "citizenship",
            "value": "Russian",
            "polarity": "is",
            "slot": "citizenship",
            "durability": "durable",
            "subject_id": str(person_id),
        }
    ]
    _validate_and_default_fact_slots(facts, [entity])
    # citizenship is a valid person slot → must survive
    assert facts[0]["slot"] == "citizenship"
    assert facts[0]["durability"] == "durable"


def test_validate_real_slot_without_subject_id_degrades():
    """A non-off_schema slot without a resolved subject_id must be degraded to 'off_schema'."""
    from kai_memory.extraction.memory_extraction import _validate_and_default_fact_slots

    facts = [
        {
            "subject": "Misha",
            "field": "occupation",
            "value": "founder",
            "polarity": "is",
            "slot": "occupation",
            "durability": "durable",
            "subject_id": None,  # not resolved
        }
    ]
    _validate_and_default_fact_slots(facts, [])
    assert facts[0]["slot"] == "off_schema"


def test_validate_bogus_slot_for_entity_type_degrades():
    """A slot name that is not valid for the subject's entity type must be degraded."""
    from kai_memory.extraction.memory_extraction import _validate_and_default_fact_slots

    org_id = uuid.uuid4()
    entity = MagicMock()
    entity.id = org_id
    entity.entity_type = "organization"

    facts = [
        {
            "subject": "Examplecorp",
            "field": "citizenship",  # citizenship is a PERSON slot, not org
            "value": "Russian",
            "polarity": "is",
            "slot": "citizenship",  # bogus for 'organization'
            "durability": "durable",
            "subject_id": str(org_id),
        }
    ]
    _validate_and_default_fact_slots(facts, [entity])
    assert facts[0]["slot"] == "off_schema"


def test_validate_real_slot_with_unmappable_subject_id_degrades():
    """Precision-first: a subject_id that maps to no known entity type degrades the slot."""
    from kai_memory.extraction.memory_extraction import _validate_and_default_fact_slots

    # subject_id points at an entity not present in resolved_entities → type unknown.
    facts = [
        {
            "subject": "Misha",
            "field": "occupation",
            "value": "founder",
            "polarity": "is",
            "slot": "occupation",
            "durability": "durable",
            "subject_id": str(uuid.uuid4()),
        }
    ]
    _validate_and_default_fact_slots(facts, [])  # no entities → type unmappable
    assert facts[0]["slot"] == "off_schema"


def test_validate_off_schema_fact_is_transient_under_d1():
    """An operational fact (e.g. exit_code) maps to off_schema and, under D-1, transient
    (code-stamped durability computed from slot). R-1: raw tag kept as claimed_durability."""
    from kai_memory.extraction.memory_extraction import _validate_and_default_fact_slots

    facts = [
        {
            "subject": "agent",
            "field": "exit_code",
            "value": "0",
            "polarity": "is",
            "slot": "off_schema",
            "durability": "operational",
            "subject_id": None,
        }
    ]
    _validate_and_default_fact_slots(facts, [])
    assert facts[0]["slot"] == "off_schema"
    assert facts[0]["durability"] == "transient"
    assert facts[0]["claimed_durability"] == "operational"


def test_validate_handles_non_dict_facts_gracefully():
    """Non-dict entries in the facts list must be skipped without raising."""
    from kai_memory.extraction.memory_extraction import _validate_and_default_fact_slots

    facts = [None, "garbage", 42, {"subject": "x", "field": "f", "value": "v", "polarity": "is"}]
    _validate_and_default_fact_slots(facts, [])  # must not raise
    # The dict entry gets defaults
    assert facts[3]["slot"] == "off_schema"
    assert facts[3]["durability"] == "transient"


def test_validate_normalizes_slot_case_when_valid_for_entity_type():
    """A mixed-case slot key ('Citizenship') must be normalized to lower-case and
    preserved when subject_id resolves to a compatible entity type — not lost to
    'off_schema' over a casing mismatch with the lower_snake_case schema keys."""
    from kai_memory.extraction.memory_extraction import _validate_and_default_fact_slots

    person_id = uuid.uuid4()
    entity = MagicMock()
    entity.id = person_id
    entity.entity_type = "person"

    facts = [
        {
            "subject": "Misha",
            "field": "citizenship",
            "value": "Russian",
            "polarity": "is",
            "slot": "Citizenship",  # mixed case, but valid person slot
            "durability": "durable",
            "subject_id": str(person_id),
        }
    ]
    _validate_and_default_fact_slots(facts, [entity])
    assert facts[0]["slot"] == "citizenship"  # normalized + preserved
    assert facts[0]["durability"] == "durable"


# ── D-1: durability computed from slot / P-1: first-party family gate ───────────


def _person(entity_id, tg=None):
    e = MagicMock()
    e.id = entity_id
    e.entity_type = "person"
    e.telegram_user_id = tg
    return e


def test_d1_any_subject_slot_is_durable_regardless_of_haiku():
    """A fact resolving to an any-subject durable slot is durable even when Haiku
    emitted 'transient' — durability is computed from the slot."""
    from kai_memory.extraction.memory_extraction import _validate_and_default_fact_slots

    pid = uuid.uuid4()
    facts = [
        {
            "subject": "P",
            "field": "occupation",
            "value": "eng",
            "polarity": "is",
            "slot": "occupation",
            "durability": "transient",
            "subject_id": str(pid),
        }
    ]
    _validate_and_default_fact_slots(facts, [_person(pid)])
    assert facts[0]["slot"] == "occupation"
    assert facts[0]["durability"] == "durable"


def test_d1_off_schema_is_transient_even_if_haiku_said_durable():
    """A hallucinated 'durable' on a fact that resolves to off_schema cannot floor it."""
    from kai_memory.extraction.memory_extraction import _validate_and_default_fact_slots

    facts = [
        {"subject": "P", "field": "mood", "value": "great", "polarity": "is", "slot": "vibe", "durability": "durable"}
    ]  # 'vibe' is not a schema key → off_schema
    _validate_and_default_fact_slots(facts, [])
    assert facts[0]["slot"] == "off_schema"
    assert facts[0]["durability"] == "transient"  # code-stamped: no floor for junk
    assert facts[0]["claimed_durability"] == "durable"  # R-1: Haiku's raw opinion preserved for the quarantine


def test_p1_family_health_transient_without_first_party():
    """P-1: family/health slots are NOT floored (transient) when the subject is not a
    first-party entity — the default the write path passes today (Kai is ownerless)."""
    from kai_memory.extraction.memory_extraction import _validate_and_default_fact_slots

    pid = uuid.uuid4()
    facts = [
        {
            "subject": "friend's mom",
            "field": "mother",
            "value": "mother: Ann",
            "polarity": "is",
            "slot": "key_relationships",
            "durability": "durable",
            "subject_id": str(pid),
        }
    ]
    _validate_and_default_fact_slots(facts, [_person(pid)])  # first_party_subject_ids defaults empty
    assert facts[0]["slot"] == "key_relationships"  # captured (A2) …
    assert facts[0]["durability"] == "transient"  # … but not floored (P-1)


def test_p1_family_health_durable_for_first_party_subject():
    """P-1 gate mechanism: when the subject IS a first-party entity, family/health is
    durable. Guards the wiring so a confirmed first-party predicate flips it on."""
    from kai_memory.extraction.memory_extraction import _validate_and_default_fact_slots

    pid = uuid.uuid4()
    facts = [
        {
            "subject": "me",
            "field": "mother",
            "value": "mother: Olga",
            "polarity": "is",
            "slot": "key_relationships",
            "durability": "transient",
            "subject_id": str(pid),
        }
    ]
    _validate_and_default_fact_slots(facts, [_person(pid)], first_party_subject_ids=frozenset({str(pid)}))
    assert facts[0]["durability"] == "durable"


def test_p1_third_party_health_captured_but_transient():
    """Refinement 1 (fact-level gate): 'my friend's mom is sick' → the chronic-health
    fact is CAPTURED but stamped transient for a non-first-party subject, so the trace
    earns NO durable marker and the floor never fires for a third party's health."""
    from kai_memory.extraction.memory_extraction import _validate_and_default_fact_slots
    from kai_memory.store.cognitive_memory import _has_durable_fact

    pid = uuid.uuid4()
    facts = [
        {
            "subject": "friend's mom",
            "field": "diagnosis",
            "value": "lupus",
            "polarity": "is",
            "slot": "chronic_health_conditions",
            "durability": "durable",  # Haiku's guess — overridden
            "subject_id": str(pid),
        }
    ]
    _validate_and_default_fact_slots(facts, [_person(pid)])  # no first-party subjects
    assert facts[0]["slot"] == "chronic_health_conditions"  # captured
    assert facts[0]["durability"] == "transient"  # not floored (P-1)
    # marker derivation is free: no durable fact ⇒ store_trace stamps no trace-level marker
    assert _has_durable_fact({"facts": facts}) is False


def test_first_party_subject_ids_from_admin_entities(monkeypatch):
    """P-1 (confirmed): first-party = subject entity whose telegram_user_id is a
    configured admin (the existing trust boundary). Multi-admin ⇒ all are first-party."""
    from kai_memory.extraction import memory_extraction as me

    monkeypatch.setattr(me, "get_settings", lambda: SimpleNamespace(telegram_admin_user_ids="123,456"))
    admin = _person(uuid.uuid4(), tg=123)
    other = _person(uuid.uuid4(), tg=999)  # not an admin
    ids = me.ExtractionService._first_party_subject_ids([admin, other])
    assert ids == frozenset({str(admin.id)})


def test_first_party_subject_ids_empty_when_no_admin_config(monkeypatch):
    """Fail-closed: with no admin config, no subject is first-party (family/health stay
    transient) — mirrors is_admin_user's fail-closed empty behavior."""
    from kai_memory.extraction import memory_extraction as me

    monkeypatch.setattr(me, "get_settings", lambda: SimpleNamespace(telegram_admin_user_ids=""))
    assert me.ExtractionService._first_party_subject_ids([_person(uuid.uuid4(), tg=123)]) == frozenset()


# ── INC 3 write-path integration via _store_item ──────────────────────────


@pytest.mark.asyncio
async def test_store_item_writes_slot_and_durability_in_typed_metadata():
    """_store_item must call _build_typed_metadata after validation so slot/durability
    appear in the stored typed_metadata blob."""
    from kai_memory.extraction.memory_extraction import ExtractionService

    session = AsyncMock()
    svc = ExtractionService(session)

    person_id = uuid.uuid4()
    mock_entity = MagicMock()
    mock_entity.id = person_id
    mock_entity.entity_type = "person"
    mock_entity.name = "Misha"
    mock_entity.aliases = []
    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()

    stored_metadata: dict = {}

    async def capture_store(**kwargs):
        stored_metadata.update(kwargs.get("typed_metadata", {}))
        return mock_trace

    ctx = ChatContext(chat_id=123, topic_id=None)
    stats = {"traces_created": 0, "entities_found": 0, "contradictions": 0}

    with (
        patch.object(svc.memory_svc, "store_trace", side_effect=capture_store),
        patch.object(svc.memory_svc, "find_or_create_entity", return_value=mock_entity),
        patch.object(svc.memory_svc, "link_trace_to_entity"),
    ):
        await svc._store_item(
            {
                "content": "Misha is a Canadian citizen",
                "memory_type": "fact",
                "significance": 3,
                "entities": [{"name": "Misha", "type": "person"}],
                "facts": [
                    {
                        "subject": "Misha",
                        "field": "citizenship",
                        "value": "Russian",
                        "polarity": "is",
                        "slot": "citizenship",
                        "durability": "durable",
                    }
                ],
            },
            ctx,
            stats,
        )

    assert stored_metadata.get("schema_version") == 3
    facts_in_blob = stored_metadata.get("facts", [])
    assert len(facts_in_blob) == 1
    # subject_id was resolved → slot must survive validation
    assert facts_in_blob[0].get("slot") == "citizenship"
    assert facts_in_blob[0].get("durability") == "durable"


@pytest.mark.asyncio
async def test_store_item_counts_third_party_family_transient():
    """P-1 telemetry (refinement 2): a third-party family fact is captured, stamped
    transient, and counted in stats['third_party_family_transient']."""
    from kai_memory.extraction.memory_extraction import ExtractionService

    session = AsyncMock()
    svc = ExtractionService(session)

    pid = uuid.uuid4()
    mock_entity = MagicMock()
    mock_entity.id = pid
    mock_entity.entity_type = "person"
    mock_entity.name = "friend's mom"
    mock_entity.aliases = []
    mock_entity.telegram_user_id = None  # not an admin → not first-party
    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()

    stored_metadata: dict = {}

    async def capture_store(**kwargs):
        stored_metadata.update(kwargs.get("typed_metadata", {}))
        return mock_trace

    ctx = ChatContext(chat_id=123, topic_id=None)
    stats = {"traces_created": 0, "entities_found": 0, "contradictions": 0, "third_party_family_transient": 0}

    with (
        patch.object(svc.memory_svc, "store_trace", side_effect=capture_store),
        patch.object(svc.memory_svc, "find_or_create_entity", return_value=mock_entity),
        patch.object(svc.memory_svc, "link_trace_to_entity"),
    ):
        await svc._store_item(
            {
                "content": "my friend's mom has lupus",
                "memory_type": "fact",
                "significance": 3,
                "entities": [{"name": "friend's mom", "type": "person"}],
                "facts": [
                    {
                        "subject": "friend's mom",
                        "field": "diagnosis",
                        "value": "lupus",
                        "polarity": "is",
                        "slot": "chronic_health_conditions",
                        "durability": "durable",
                        "subject_id": str(pid),
                    }
                ],
            },
            ctx,
            stats,
        )

    assert stats["third_party_family_transient"] == 1
    assert stored_metadata["facts"][0]["durability"] == "transient"  # captured, not floored
    assert stored_metadata.get("durability") != "durable"  # no trace-level marker (via _build_typed_metadata)


@pytest.mark.asyncio
async def test_store_item_degrades_slot_when_subject_id_unresolved():
    """When the subject entity cannot be resolved, a real slot must be degraded to off_schema."""
    from kai_memory.extraction.memory_extraction import ExtractionService

    session = AsyncMock()
    svc = ExtractionService(session)

    # Entity whose name does NOT match the fact's subject → subject_id stays None
    unrelated_entity = MagicMock()
    unrelated_entity.id = uuid.uuid4()
    unrelated_entity.entity_type = "person"
    unrelated_entity.name = "Katya"
    unrelated_entity.aliases = []

    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()

    stored_metadata: dict = {}

    async def capture_store(**kwargs):
        stored_metadata.update(kwargs.get("typed_metadata", {}))
        return mock_trace

    ctx = ChatContext(chat_id=123, topic_id=None)
    stats = {"traces_created": 0, "entities_found": 0, "contradictions": 0}

    with (
        patch.object(svc.memory_svc, "store_trace", side_effect=capture_store),
        patch.object(svc.memory_svc, "find_or_create_entity", return_value=unrelated_entity),
        patch.object(svc.memory_svc, "link_trace_to_entity"),
    ):
        await svc._store_item(
            {
                "content": "Misha lives in Corfu",
                "memory_type": "fact",
                "significance": 3,
                "entities": [{"name": "Katya", "type": "person"}],
                "facts": [
                    {
                        "subject": "Misha",  # does NOT match "Katya"
                        "field": "current_location",
                        "value": "Corfu",
                        "polarity": "is",
                        "slot": "current_location",
                        "durability": "durable",
                    }
                ],
            },
            ctx,
            stats,
        )

    facts_in_blob = stored_metadata.get("facts", [])
    assert len(facts_in_blob) == 1
    # subject_id not resolved → slot degraded
    assert facts_in_blob[0].get("slot") == "off_schema"


@pytest.mark.asyncio
async def test_store_item_preserves_existing_fields_alongside_slot():
    """field/value/polarity/subject must still be present when slot/durability are added."""
    from kai_memory.extraction.memory_extraction import ExtractionService

    session = AsyncMock()
    svc = ExtractionService(session)

    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()

    stored_metadata: dict = {}

    async def capture_store(**kwargs):
        stored_metadata.update(kwargs.get("typed_metadata", {}))
        return mock_trace

    ctx = ChatContext(chat_id=123, topic_id=None)
    stats = {"traces_created": 0, "entities_found": 0, "contradictions": 0}

    with (
        patch.object(svc.memory_svc, "store_trace", side_effect=capture_store),
        patch.object(svc.memory_svc, "link_trace_to_entity"),
    ):
        await svc._store_item(
            {
                "content": "Misha works at Examplecorp",
                "memory_type": "fact",
                "significance": 3,
                "entities": [],
                "facts": [
                    {
                        "subject": "Misha",
                        "field": "employer",
                        "value": "Examplecorp",
                        "polarity": "is",
                        "slot": "off_schema",
                        "durability": "durable",
                    }
                ],
            },
            ctx,
            stats,
        )

    facts_in_blob = stored_metadata.get("facts", [])
    assert len(facts_in_blob) == 1
    f = facts_in_blob[0]
    assert f.get("subject") == "Misha"
    assert f.get("field") == "employer"
    assert f.get("value") == "Examplecorp"
    assert f.get("polarity") == "is"
    assert "slot" in f
    assert "durability" in f


@pytest.mark.parametrize(
    "malformed",
    [None, "not an item", 7, [], {}, {"content": None}, {"content": ""}, {"content": "   "}, {"content": 42}],
)
async def test_extract_malformed_items_preserves_valid_neighbors(malformed):
    svc = ExtractionService(_session_with_savepoints())
    response = {
        "memories": [
            {"content": "first valid fact", "significance": 4},
            malformed,
            {"content": "last valid fact", "significance": 2},
        ],
        "social_experiences": [],
    }
    with (
        patch.object(
            svc.conv_manager,
            "get_messages_since",
            return_value=[
                _make_message(text="A sufficiently long synthetic conversation about two distinct important facts.")
            ],
        ),
        patch.object(svc, "_call_haiku", return_value=response),
        patch.object(svc, "_load_existing_context", return_value=""),
        patch.object(svc, "_ensure_social_context"),
        patch.object(svc, "_load_participant_context", return_value=("", {})),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]),
        patch.object(svc.memory_svc, "store_trace", return_value=SimpleNamespace(id=uuid.uuid4())) as store,
        patch(
            "kai_memory.extraction.memory_extraction.generate_embeddings_batch",
            return_value=[[0.1] * 1536, [0.2] * 1536],
        ) as embed,
    ):
        stats = await svc.extract_from_conversation(ChatContext(chat_id=123), datetime.now(UTC))
    assert stats["malformed_items"] == 1
    assert stats["store_failures"] == 0 and stats["traces_created"] == 2
    assert [call.kwargs["content"] for call in store.await_args_list] == ["first valid fact", "last valid fact"]
    assert [call.kwargs["embedding"] for call in store.await_args_list] == [[0.1] * 1536, [0.2] * 1536]
    embed.assert_awaited_once_with(["first valid fact", "last valid fact"])


@pytest.mark.parametrize(
    "raw,expected",
    [
        (-1, 1),
        (10, 5),
        (4.8, 4),
        ("4", 4),
        ("4.8", 4),
        (None, 3),
        ("invalid", 3),
        (float("nan"), 3),
        (float("inf"), 3),
        (-float("inf"), 3),
        ({}, 3),
        ([], 3),
        (True, 3),
    ],
)
async def test_extract_significance_normalizes_before_storage(raw, expected):
    svc = ExtractionService(_session_with_savepoints())
    with (
        patch.object(
            svc.conv_manager,
            "get_messages_since",
            return_value=[_make_message(text="A sufficiently long synthetic conversation about an important fact.")],
        ),
        patch.object(
            svc,
            "_call_haiku",
            return_value={"memories": [{"content": "valid fact", "significance": raw}], "social_experiences": []},
        ),
        patch.object(svc, "_load_existing_context", return_value=""),
        patch.object(svc, "_ensure_social_context"),
        patch.object(svc, "_load_participant_context", return_value=("", {})),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]),
        patch.object(svc.memory_svc, "store_trace", return_value=SimpleNamespace(id=uuid.uuid4())) as store,
        patch("kai_memory.extraction.memory_extraction.generate_embeddings_batch", return_value=[[0.1] * 1536]),
    ):
        stats = await svc.extract_from_conversation(ChatContext(chat_id=123), datetime.now(UTC))
    assert stats["store_failures"] == 0 and stats["traces_created"] == 1
    assert stats["malformed_items"] == 0  # normalization keeps the fact; only skipped items count
    assert store.await_args.kwargs["significance"] == expected


@pytest.mark.parametrize(
    "field,raw,expected_type",
    [
        *[(field, raw, "fact") for field in ("entities", "facts") for raw in (None, "invalid", {}, 7, True, [None])],
        *[("memory_type", raw, "fact") for raw in (None, [], {}, 7, True)],
        ("memory_type", "preference", "preference"),
        ("memory_type", "observation", "impression"),
        ("memory_type", "unknown", "fact"),
    ],
)
async def test_extract_normalizes_item_fields_before_batch_prepass(field, raw, expected_type):
    svc = ExtractionService(_session_with_savepoints())
    contents = ["first valid fact", "fact with model field drift", "last valid fact"]
    embeddings = [[value] * 1536 for value in (0.1, 0.2, 0.3)]
    response = {
        "memories": [
            {"content": contents[0]},
            {"content": contents[1], field: raw},
            {"content": contents[2]},
        ],
        "social_experiences": [],
    }
    with (
        patch.object(
            svc.conv_manager,
            "get_messages_since",
            return_value=[_make_message(text="A sufficiently long synthetic conversation about three distinct facts.")],
        ),
        patch.object(svc, "_call_haiku", return_value=response),
        patch.object(svc, "_load_existing_context", return_value=""),
        patch.object(svc, "_ensure_social_context"),
        patch.object(svc, "_load_participant_context", return_value=("", {})),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]),
        patch.object(svc.memory_svc, "find_or_create_entity") as create_entity,
        patch.object(svc.memory_svc, "store_trace", return_value=SimpleNamespace(id=uuid.uuid4())) as store,
        patch("kai_memory.extraction.memory_extraction.generate_embeddings_batch", return_value=embeddings) as embed,
    ):
        stats = await svc.extract_from_conversation(ChatContext(chat_id=123), datetime.now(UTC))
    assert stats["malformed_items"] == stats["store_failures"] == stats["entities_found"] == 0
    assert stats["traces_created"] == 3
    assert [call.kwargs["content"] for call in store.await_args_list] == contents
    assert [call.kwargs["embedding"] for call in store.await_args_list] == embeddings
    assert [call.kwargs["memory_type"] for call in store.await_args_list] == ["fact", expected_type, "fact"]
    assert all(call.kwargs["typed_metadata"]["entities"] == [] for call in store.await_args_list)
    assert all(call.kwargs["typed_metadata"]["facts"] == [] for call in store.await_args_list)
    create_entity.assert_not_awaited()
    embed.assert_awaited_once_with(contents)


@pytest.mark.parametrize(
    "raw,expected_type,expected_actor",
    [
        *[(raw, None, None) for raw in (None, True, 42, [], {}, "", "   ")],
        *[({"name": name}, None, None) for name in (None, True, 42, [], {}, ["wrong"], {"wrong": "shape"})],
        *[
            ({"name": "Drift entity", "type": raw_type}, "topic", None)
            for raw_type in (None, True, 42, [], {}, ["person"], {"unexpected": "person"})
        ],
        *[
            ({"name": "Drift entity", "type": raw_type, "matched_participant_key": "111"}, "person", 111)
            for raw_type in (["person"], {"unexpected": "person"})
        ],
        *[
            ({"name": "Drift entity", "type": "person", "matched_participant_key": key}, "person", None)
            for key in ([], {}, True)
        ],
        ({"name": "Drift entity", "type": "person", "matched_participant_key": 111}, "person", 111),
        ({"name": "Drift entity", "type": "topic", "data": ["invalid"]}, "topic", None),
        ("Drift entity", "topic", None),
    ],
)
async def test_extract_nested_entity_shapes_preserve_links_and_attribution(raw, expected_type, expected_actor):
    svc = ExtractionService(_session_with_savepoints())
    contents = ["memory with an entity of uncertain shape", "valid neighboring source fact"]
    traces = [SimpleNamespace(id=uuid.uuid4()), SimpleNamespace(id=uuid.uuid4())]
    created = []

    async def create_entity(name, entity_type, **kwargs):
        entity = SimpleNamespace(
            id=uuid.uuid4(),
            name=name,
            entity_type=entity_type,
            telegram_user_id=kwargs.get("telegram_user_id"),
            data=kwargs.get("data", {}),
        )
        created.append((entity, kwargs))
        return entity

    neighbor = {
        "name": "Synthetic Person",
        "type": "person",
        "matched_participant_key": "111",
        "data": {"label": "retained"},
    }
    response = {
        "memories": [
            {"content": contents[0], "entities": [raw, "Valid topic"]},
            {
                "content": contents[1],
                "entities": [neighbor],
                "facts": [{"subject": "Synthetic Person", "field": "role", "value": "researcher"}],
            },
        ]
    }
    with (
        patch.object(
            svc.conv_manager,
            "get_messages_since",
            return_value=[
                _make_message(text="A sufficiently long synthetic conversation about distinct entity facts.")
            ],
        ),
        patch.object(svc, "_call_haiku", return_value=response),
        patch.object(svc, "_load_existing_context", return_value=""),
        patch.object(svc, "_ensure_social_context"),
        patch.object(
            svc,
            "_load_participant_context",
            return_value=("", {"111": {"name": "Synthetic Person", "telegram_user_id": 111}}),
        ),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]),
        patch.object(svc.memory_svc, "find_or_create_entity", side_effect=create_entity),
        patch.object(svc.memory_svc, "link_trace_to_entity") as link,
        patch.object(svc.memory_svc, "store_trace", side_effect=traces) as store,
        patch(
            "kai_memory.extraction.memory_extraction.generate_embeddings_batch",
            return_value=[[0.1] * 1536, [0.2] * 1536],
        ),
    ):
        stats = await svc.extract_from_conversation(ChatContext(chat_id=123), datetime.now(UTC))
    assert stats["store_failures"] == 0 and stats["traces_created"] == 2
    assert [call.kwargs["content"] for call in store.await_args_list] == contents
    entities = {entity.name: (entity, kwargs) for entity, kwargs in created}
    assert set(entities) == {"Valid topic", "Synthetic Person"} | ({"Drift entity"} if expected_type else set())
    person, person_kwargs = entities["Synthetic Person"]
    assert person.telegram_user_id == 111 and person_kwargs["source_text"] == contents[1]
    topic, topic_kwargs = entities["Valid topic"]
    assert topic.entity_type == "topic" and topic_kwargs["source_text"] == contents[0]
    expected_links = {(traces[0].id, topic.id, "about"), (traces[1].id, person.id, "about")}
    if expected_type:
        drift, drift_kwargs = entities["Drift entity"]
        assert drift.entity_type == expected_type and drift.telegram_user_id == expected_actor
        assert drift_kwargs["source_text"] == contents[0]
        expected_links.add((traces[0].id, drift.id, "about"))
    assert {call.args for call in link.await_args_list} == expected_links
    metadata = store.await_args_list[1].kwargs["typed_metadata"]
    assert metadata["entities"][0]["data"] == {"label": "retained"}
    assert metadata["facts"][0]["subject_id"] == str(person.id)
