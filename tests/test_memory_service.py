"""Tests for MemoryService — CRUD + search for agent memory."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai_memory.store.agent_memory_store import MemoryService, MAX_ENTRIES_PER_CATEGORY, MAX_ENTRIES_TOTAL


class TestMemoryService:
    @pytest.mark.asyncio
    async def test_save_memory(self):
        """Atomic upsert returns 'saved' status for both create and update."""
        session = AsyncMock()
        # First call: existence check (returns None = new entry)
        existence_result = MagicMock()
        existence_result.scalar_one_or_none.return_value = None
        # Second call: category count check
        cat_count_result = MagicMock()
        cat_count_result.scalar.return_value = 0
        # Third call: total count check
        total_count_result = MagicMock()
        total_count_result.scalar.return_value = 0
        # Fourth call: the actual upsert
        upsert_result = MagicMock()
        session.execute = AsyncMock(
            side_effect=[existence_result, cat_count_result, total_count_result, upsert_result, MagicMock()]
        )

        with patch("kai_memory.store.agent_memory_store.generate_embedding", return_value=[0.1] * 1536):
            service = MemoryService(session)
            result = await service.save(category="fact", key="test_key", content="test content")
        assert result["status"] == "saved"
        # existence check + cat count + total count + upsert + embedding update
        assert session.execute.call_count == 5

    @pytest.mark.asyncio
    async def test_save_rejects_oversized_content(self):
        """Content over 20K chars is rejected."""
        session = AsyncMock()
        service = MemoryService(session)
        result = await service.save(category="fact", key="big", content="x" * 20_001)
        assert result["status"] == "error"
        assert "too long" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_get_memory(self):
        existing = MagicMock()
        existing.content = "stored content"
        existing.category = "fact"
        existing.key = "test_key"
        existing.priority = 5
        session = AsyncMock()
        session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=existing)))
        service = MemoryService(session)
        result = await service.get(category="fact", key="test_key")
        assert result["content"] == "stored content"

    @pytest.mark.asyncio
    async def test_get_always_loaded(self):
        """Returns all memory entries that should be in system prompt."""
        memories = [
            MagicMock(category="instruction", key="main", content="Be helpful", priority=10),
            MagicMock(category="persona", key="tone", content="Concise", priority=8),
        ]
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = memories
        session.execute = AsyncMock(return_value=result_mock)
        service = MemoryService(session)
        result = await service.get_always_loaded()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_delete_memory(self):
        existing = MagicMock()
        session = AsyncMock()
        session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=existing)))
        service = MemoryService(session)
        result = await service.delete(category="fact", key="test_key")
        assert result["status"] == "deleted"


class TestMemoryAlwaysLoadedTruncation:
    """Tests for get_always_loaded() 150K char cap."""

    @pytest.mark.asyncio
    async def test_get_always_loaded_truncates_at_150k(self):
        """Memory entries exceeding 150K total chars should be truncated."""
        from kai_memory.store.agent_memory_store import MemoryService

        session = AsyncMock()
        # Create memory entries that exceed 150K total
        big_memory = MagicMock()
        big_memory.category = "instruction"
        big_memory.key = "huge"
        big_memory.content = "x" * 160_000
        big_memory.priority = 10

        small_memory = MagicMock()
        small_memory.category = "instruction"
        small_memory.key = "small"
        small_memory.content = "important note"
        small_memory.priority = 5

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = [big_memory, small_memory]
        session.execute = AsyncMock(return_value=result_mock)

        svc = MemoryService(session)
        entries = await svc.get_always_loaded()

        # big_memory (160K) exceeds 150K cap on first iteration (0 + 160K > 150K),
        # so the loop breaks immediately — neither entry is included
        assert entries == []

    @pytest.mark.asyncio
    async def test_get_always_loaded_includes_entries_within_cap(self):
        """Entries that fit within 150K cap should all be returned."""
        from kai_memory.store.agent_memory_store import MemoryService, MAX_ALWAYS_LOADED_CHARS

        session = AsyncMock()
        entry1 = MagicMock()
        entry1.category = "instruction"
        entry1.key = "first"
        entry1.content = "x" * 100_000
        entry1.priority = 10

        entry2 = MagicMock()
        entry2.category = "instruction"
        entry2.key = "second"
        entry2.content = "y" * 40_000
        entry2.priority = 5

        entry3 = MagicMock()
        entry3.category = "instruction"
        entry3.key = "third"
        entry3.content = "z" * 20_000
        entry3.priority = 3

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = [entry1, entry2, entry3]
        session.execute = AsyncMock(return_value=result_mock)

        svc = MemoryService(session)
        entries = await svc.get_always_loaded()

        # entry1 (100K) + entry2 (40K) = 140K fits, entry3 (20K) would push to 160K
        assert len(entries) == 2
        total_chars = sum(len(e["content"]) for e in entries)
        assert total_chars <= MAX_ALWAYS_LOADED_CHARS


class TestMemorySaveEmbedding:
    """Tests for embedding generation on save."""

    @pytest.mark.asyncio
    async def test_save_generates_embedding(self):
        """Save should attempt to generate and store embedding."""
        session = AsyncMock()
        # Mock for the existing entry check
        existing_result = MagicMock()
        existing_result.scalar_one_or_none.return_value = None
        # Mock for category count
        cat_count_result = MagicMock()
        cat_count_result.scalar.return_value = 0
        # Mock for total count
        total_count_result = MagicMock()
        total_count_result.scalar.return_value = 0
        # Mock for the upsert
        upsert_result = MagicMock()

        session.execute = AsyncMock(
            side_effect=[existing_result, cat_count_result, total_count_result, upsert_result, MagicMock()]
        )
        session.flush = AsyncMock()

        svc = MemoryService(session)
        fake_embedding = [0.1] * 1536
        with patch("kai_memory.store.agent_memory_store.generate_embedding", return_value=fake_embedding) as mock_embed:
            result = await svc.save("fact", "test", "test content")

        assert result["status"] == "saved"
        mock_embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_succeeds_even_if_embedding_fails(self):
        """Save should succeed even if embedding generation fails — and the
        embedding column is explicitly cleared (NULL), never left stale."""
        session = AsyncMock()
        existing_result = MagicMock()
        existing_result.scalar_one_or_none.return_value = None
        cat_count_result = MagicMock()
        cat_count_result.scalar.return_value = 0
        total_count_result = MagicMock()
        total_count_result.scalar.return_value = 0
        upsert_result = MagicMock()
        clear_embedding_result = MagicMock()  # the NULL-clear UPDATE on embedding failure

        session.execute = AsyncMock(
            side_effect=[existing_result, cat_count_result, total_count_result, upsert_result, clear_embedding_result]
        )
        session.flush = AsyncMock()

        svc = MemoryService(session)
        with patch("kai_memory.store.agent_memory_store.generate_embedding", side_effect=Exception("API down")):
            result = await svc.save("fact", "test", "test content")

        assert result["status"] == "saved"
        # Last execute must clear the embedding to NULL (no stale vector survives).
        last_sql = str(session.execute.await_args_list[-1].args[0])
        assert "embedding = NULL" in last_sql


class TestMemoryEntryLimits:
    """Tests for entry count limits."""

    @pytest.mark.asyncio
    async def test_rejects_when_category_full(self):
        """Should reject new entries when category is at limit."""
        session = AsyncMock()
        existing_result = MagicMock()
        existing_result.scalar_one_or_none.return_value = None
        cat_count_result = MagicMock()
        cat_count_result.scalar.return_value = MAX_ENTRIES_PER_CATEGORY

        session.execute = AsyncMock(side_effect=[existing_result, cat_count_result])

        svc = MemoryService(session)
        result = await svc.save("fact", "new_key", "content")
        assert result["status"] == "error"
        assert "limit" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_rejects_when_total_full(self):
        """Should reject new entries when total is at limit."""
        session = AsyncMock()
        existing_result = MagicMock()
        existing_result.scalar_one_or_none.return_value = None
        cat_count_result = MagicMock()
        cat_count_result.scalar.return_value = 0
        total_count_result = MagicMock()
        total_count_result.scalar.return_value = MAX_ENTRIES_TOTAL

        session.execute = AsyncMock(side_effect=[existing_result, cat_count_result, total_count_result])

        svc = MemoryService(session)
        result = await svc.save("fact", "new_key", "content")
        assert result["status"] == "error"
        assert "limit" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_allows_upsert_when_at_limit(self):
        """Should allow updating existing entries even at limit."""
        session = AsyncMock()
        existing_result = MagicMock()
        existing_result.scalar_one_or_none.return_value = MagicMock()  # Existing entry
        upsert_result = MagicMock()

        session.execute = AsyncMock(side_effect=[existing_result, upsert_result, MagicMock()])
        session.flush = AsyncMock()

        svc = MemoryService(session)
        fake_embedding = [0.1] * 1536
        with patch("kai_memory.store.agent_memory_store.generate_embedding", return_value=fake_embedding):
            result = await svc.save("fact", "existing_key", "updated content")

        assert result["status"] == "saved"


class TestMemoryGetNotFound:
    """Tests for get() returning found=False dict instead of None."""

    @pytest.mark.asyncio
    async def test_get_returns_found_false_dict_for_missing_entry(self):
        """get() should return a dict with found=False, not None."""
        session = AsyncMock()
        session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None)))
        svc = MemoryService(session)
        result = await svc.get(category="fact", key="nonexistent")

        assert result is not None, "get() should never return None"
        assert result["found"] is False
        assert result["category"] == "fact"
        assert result["key"] == "nonexistent"

    @pytest.mark.asyncio
    async def test_get_returns_found_true_for_existing_entry(self):
        """get() should return found=True for existing entries."""
        existing = MagicMock()
        existing.content = "stored content"
        existing.category = "fact"
        existing.key = "test_key"
        existing.priority = 5
        session = AsyncMock()
        session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=existing)))
        svc = MemoryService(session)
        result = await svc.get(category="fact", key="test_key")

        assert result["found"] is True
        assert result["content"] == "stored content"


class TestMemoryAlwaysLoadedWarning:
    """Tests for warning when saving to always-loaded categories."""

    @pytest.mark.asyncio
    async def test_save_to_always_loaded_category_includes_warning(self):
        """Saving to an always-loaded category should include a warning in the result."""

        session = AsyncMock()
        existing_result = MagicMock()
        existing_result.scalar_one_or_none.return_value = None
        cat_count_result = MagicMock()
        cat_count_result.scalar.return_value = 0
        total_count_result = MagicMock()
        total_count_result.scalar.return_value = 0
        upsert_result = MagicMock()

        session.execute = AsyncMock(
            side_effect=[existing_result, cat_count_result, total_count_result, upsert_result, MagicMock()]
        )

        svc = MemoryService(session)
        with patch("kai_memory.store.agent_memory_store.generate_embedding", return_value=[0.1] * 1536):
            result = await svc.save(category="instruction", key="test", content="test content")

        assert result["status"] == "saved"
        assert "warning" in result
        assert "always loaded" in result["warning"]

    @pytest.mark.asyncio
    async def test_save_to_non_always_loaded_category_no_warning(self):
        """Saving to a regular category should NOT include a warning."""
        session = AsyncMock()
        existing_result = MagicMock()
        existing_result.scalar_one_or_none.return_value = None
        cat_count_result = MagicMock()
        cat_count_result.scalar.return_value = 0
        total_count_result = MagicMock()
        total_count_result.scalar.return_value = 0
        upsert_result = MagicMock()

        session.execute = AsyncMock(
            side_effect=[existing_result, cat_count_result, total_count_result, upsert_result, MagicMock()]
        )

        svc = MemoryService(session)
        with patch("kai_memory.store.agent_memory_store.generate_embedding", return_value=[0.1] * 1536):
            result = await svc.save(category="fact", key="test", content="test content")

        assert result["status"] == "saved"
        assert "warning" not in result
