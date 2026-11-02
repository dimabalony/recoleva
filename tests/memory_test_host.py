"""Minimal synthetic defaults for SDK contracts; no evaluation or host runtime."""

from dataclasses import dataclass, field
from datetime import datetime
import json
from uuid import UUID

from kai_memory import AgentIdentity, Chat, MemorySettings, ModelRoles, NoopTelemetry
from kai_memory.configuration import NeutralHostPolicy


@dataclass
class ConversationMessage:
    role: str | None = None
    content_blocks: list = field(default_factory=list)
    message_metadata: dict = field(default_factory=dict)
    sender_user_id: int | None = None
    sender_name: str | None = None
    id: UUID | None = None
    chat_id: int | None = None
    topic_id: int | None = None
    venue: str = "telegram"
    created_at: datetime | None = None

    @property
    def text(self):
        return "".join(block.get("text", "") for block in self.content_blocks or [])


class FixedEmbedder:
    """A valid constant vector for storage contracts, with no semantic quality claim."""

    async def embed(self, texts, *, api_key=None):
        return [[1.0] + [0.0] * 1535 for _ in texts]


async def _unscripted(*args, **kwargs):
    raise RuntimeError("no synthetic response configured")


class UnscriptedEmbedder:
    embed = staticmethod(_unscripted)


class Conversations:
    """Unit tests must explicitly script history calls; no hidden database access."""

    def __init__(self, session):
        self.session = session

    get_messages_since = staticmethod(_unscripted)
    save_message = staticmethod(_unscripted)


async def recover_session(session):
    try:
        await session.rollback()
    except Exception:
        try:
            await session.invalidate()
        except Exception:
            pass


def synthetic_bindings(now: datetime) -> dict:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("synthetic clock must be timezone-aware")
    settings = MemorySettings()
    policy = NeutralHostPolicy()
    return {
        "recover_session": recover_session,
        "get_settings": lambda: settings,
        "clock": lambda: now,
        "get_agent_identity": lambda: AgentIdentity("example-assistant", "Atlas"),
        "get_model_roles": ModelRoles,
        "get_host_policy": lambda: policy,
        "telemetry": NoopTelemetry(),
        "embedder": UnscriptedEmbedder(),
        "ConversationManager": Conversations,
        "ChatContext": Chat,
        "read_conversation_page": _unscripted,
        "senders_seen_before": _unscripted,
        "parse_json_object_response": json.loads,
        "parse_json_array_response": json.loads,
        "get_provider_for_model": lambda *a: None,
        "call_llm_text": _unscripted,
        "record_generation_provenance": lambda *a, **k: None,
        "scrub_secrets": lambda value: value,
        "validate_skill_content": lambda value: [],
        "strip_msg_meta_envelope": lambda value: value,
        "strip_echoed_metadata": lambda value: value or "",
        "redact_urls_for_trace": lambda value: value,
        "is_admin_user": lambda actor, *a: (
            actor is not None and str(actor) in settings.telegram_admin_user_ids.split(",")
        ),
        "invalidate_prompt_cache": lambda *a, **k: None,
        "MODEL_METADATA_KEY": "model",
    }
