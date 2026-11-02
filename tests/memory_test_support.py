"""Synthetic values for adapted unit tests; no application imports."""
from dataclasses import dataclass
from memory_test_host import ConversationMessage

@dataclass
class ChatContext:
    chat_id: int | None = None
    topic_id: int | None = None
    venue: str = "telegram"
    topic_title: str | None = None
    topic_context: str | None = None
    is_group: bool = False
    is_admin_workspace: bool = False
    chat_title: str | None = None
    chat_type: str | None = None
    sender_user_id: int | None = None
    sender_name: str | None = None
    sender_username: str | None = None
    extra_participants: list | None = None

    @classmethod
    def for_event(cls):
        return cls(venue="event", is_admin_workspace=True)

    @classmethod
    def for_heartbeat(cls):
        return cls(venue="heartbeat", is_admin_workspace=True)
