# tests/test_agent_memory_model.py
"""Tests for AgentMemory model."""


from kai_memory.storage.models.agent_memory import AgentMemory


class TestAgentMemoryModel:
    def test_model_has_required_fields(self):
        """AgentMemory has id, category, key, content, metadata, timestamps."""
        memory = AgentMemory(
            category="fact",
            key="user_preference_coding_style",
            content="User prefers functional style over OOP",
        )
        assert memory.category == "fact"
        assert memory.key == "user_preference_coding_style"
        assert memory.content == "User prefers functional style over OOP"

    def test_model_has_optional_fields(self):
        memory = AgentMemory(
            category="fact",
            key="test",
            content="test content",
            priority=8,
            source="conversation",
            memory_metadata={"topic_id": 123},
        )
        assert memory.priority == 8
        assert memory.source == "conversation"
        assert memory.memory_metadata == {"topic_id": 123}

    def test_default_priority_is_5(self):
        """Default priority is 5 — applied via mapped_column(default=5) at DB insert time.

        In-memory construction returns None; the DB insert/server_default provides the value.
        This test verifies the column's default is correctly configured.
        """
        col = AgentMemory.__table__.c.priority
        assert col.default.arg == 5

    def test_tablename(self):
        assert AgentMemory.__tablename__ == "agent_memory"
