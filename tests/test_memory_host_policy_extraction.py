"""L2 capture honors host teaching policy without Telegram configuration."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kai_memory import ports
from kai_memory.extraction.memory_extraction import ExtractionService
from kai_memory.procedural import rule_capture
from kai_memory.sdk import Chat, Message
from kai_memory.store import cognitive_config


@pytest.mark.parametrize("respect,accepted", [(75, True), (74, False), (100, True)])
async def test_l2_rule_capture_applies_host_respect_with_neutral_settings(monkeypatch, respect, accepted):
    actor = 42
    quote = "Always use concise answers when replying to my requests."
    message = Message(1, None, "example", "user", quote, datetime.now(UTC), sender_user_id=actor)
    candidate = {
        "source_quote": quote,
        "category": "communication",
        "rule": "Use concise answers",
        "trigger": "Replies",
    }
    policy = SimpleNamespace(is_admin=lambda _: False, respect=lambda sender: respect if sender == actor else 0)
    settings = SimpleNamespace()
    calls = []

    async def capture(session, chat, candidates, messages, admin_ids, *, respect_of, **kwargs):
        assert respect_of == policy.respect
        assert admin_ids == ""
        gate = rule_capture.evaluate_candidate(candidates[0], messages, admin_ids, respect_of)
        calls.append(gate)
        return int(gate.accepted)

    monkeypatch.setattr(rule_capture, "capture_rules", capture)
    monkeypatch.setattr(cognitive_config, "get_bool", lambda _: True)
    with ports.scoped(
        {
            "get_settings": lambda: settings,
            "get_host_policy": lambda: policy,
            "ConversationManager": lambda _: None,
            "scrub_secrets": lambda text: text,
        }
    ):
        session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalar_one=lambda: True)))
        service = ExtractionService(session)
        monkeypatch.setattr(service, "_build_served_rules_legend", AsyncMock(return_value=("", {}, {})))
        monkeypatch.setattr(service, "_load_existing_context", AsyncMock(return_value=""))

        async def ensure_social_context(chat):
            session.execute.assert_awaited_once()
            assert "pg_try_advisory_xact_lock" in str(session.execute.await_args.args[0])

        monkeypatch.setattr(service, "_ensure_social_context", ensure_social_context)
        monkeypatch.setattr(service, "_load_participant_context", AsyncMock(return_value=("", {})))
        monkeypatch.setattr(service, "_call_haiku", AsyncMock(return_value={"behavior_rules": [candidate]}))
        monkeypatch.setattr(service, "_resolve_matched_participant_entities", AsyncMock(return_value={}))
        monkeypatch.setattr(service.memory_svc, "find_entities_by_names", AsyncMock(return_value=[]))
        await service._extract_batch(Chat(chat_id=1, venue="example"), [message], service._new_stats(), None)
    assert len(calls) == 1
    assert calls[0].accepted is accepted
    assert calls[0].respect == respect
    if accepted:
        assert calls[0].instructor_id == actor
    else:
        assert calls[0].reason == "untrusted"
