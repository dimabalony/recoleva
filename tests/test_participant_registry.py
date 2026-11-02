from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from memory_test_support import ChatContext
from kai_memory.extraction.memory_extraction import ExtractionService

pytestmark = pytest.mark.asyncio


async def test_chat_title_populates_social_context_name():
    # Initial freshness-check SELECT returns None so the upsert path runs.
    initial_select = MagicMock()
    initial_select.scalar_one_or_none.return_value = None
    upsert_result = MagicMock()

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[initial_select, upsert_result])
    svc = ExtractionService(session)

    await svc._ensure_social_context(
        ChatContext(
            chat_id=123,
            topic_id=None,
            is_group=True,
            chat_title="Family Group",
        )
    )

    stmt = session.execute.await_args_list[1].args[0]
    params = stmt.compile(dialect=postgresql.dialect()).params

    assert params["context_name"] == "Family Group"


async def test_sender_fields_add_sender_to_participants():
    initial_select = MagicMock()
    initial_select.scalar_one_or_none.return_value = None
    upsert_result = MagicMock()
    participant_update = MagicMock()

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[initial_select, upsert_result, participant_update])
    svc = ExtractionService(session)

    await svc._ensure_social_context(
        ChatContext(
            chat_id=123,
            topic_id=None,
            is_group=True,
            sender_user_id=42,
            sender_name="Misha Petrov",
            sender_username="misha",
        )
    )

    stmt, params = session.execute.await_args_list[2].args
    sql = str(stmt)
    participant = json.loads(params["new_participant"])

    assert "UPDATE social_contexts" in sql
    assert "jsonb_agg" in sql
    assert params["tg_id_text"] == "42"
    assert participant["telegram_user_id"] == 42
    assert participant["name"] == "Misha Petrov"
    assert participant["username"] == "misha"
    assert "last_seen_at" in participant
