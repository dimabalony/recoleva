"""Memory service — persistent knowledge the AI reads and writes each turn."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.storage.models.agent_memory import AgentMemory
from kai_memory.host_calls import utc_now
from kai_memory.store.embeddings import generate_embedding, is_zero_vector

# Pinned to the pre-DB-842 module path: CloudWatch queries, feature-watch criteria and
# log filters match the logger name, which must not follow the module's move.
_LEGACY_LOGGER_NAME = "app.services.agent_memory_store"
logger = logging.getLogger(_LEGACY_LOGGER_NAME)

# Categories always loaded into system prompt
ALWAYS_LOADED_CATEGORIES = {"instruction", "persona", "identity", "tool_notes"}
# Total character cap for always-loaded content (matching OpenClaw's 150K)
MAX_ALWAYS_LOADED_CHARS = 150_000

# Entry count limits to prevent unbounded growth
MAX_ENTRIES_PER_CATEGORY = 50
MAX_ENTRIES_TOTAL = 200


class MemoryService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(
        self,
        category: str,
        key: str,
        content: str,
        priority: int = 5,
        source: str | None = None,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        """Save or update a memory entry. Atomic upsert by (category, key).

        Note: Uses flush() not commit() — the caller's session_scope() manages
        the transaction. This allows multiple operations to be batched in one
        transaction. If a later operation fails, all changes roll back together.
        """
        if len(content) > 20_000:
            return {"status": "error", "message": f"Content too long ({len(content)} chars, max 20,000)"}

        # Scan ALL categories for prompt injection (memory_search loads any category into context)
        from kai_memory.ports import validate_skill_content

        violations = validate_skill_content(content)
        if violations:
            return {"status": "error", "message": f"Memory content contains blocked patterns: {', '.join(violations)}"}

        # Check entry count limits (skip for upserts — existing entries don't increase count)
        from sqlalchemy import func

        existing = await self.session.execute(
            select(AgentMemory).where(AgentMemory.category == category, AgentMemory.key == key)
        )
        if existing.scalar_one_or_none() is None:
            # New entry — check limits (best-effort; atomic upsert below prevents duplicates)
            cat_count = await self.session.execute(select(func.count()).where(AgentMemory.category == category))
            if (cat_count.scalar() or 0) >= MAX_ENTRIES_PER_CATEGORY:
                return {
                    "status": "error",
                    "message": (
                        f"Category '{category}' has reached the limit of"
                        f" {MAX_ENTRIES_PER_CATEGORY} entries. Delete old entries first."
                    ),
                }

            total_count = await self.session.execute(select(func.count()).select_from(AgentMemory))
            if (total_count.scalar() or 0) >= MAX_ENTRIES_TOTAL:
                return {
                    "status": "error",
                    "message": (
                        f"Memory has reached the limit of {MAX_ENTRIES_TOTAL} total entries. Delete old entries first."
                    ),
                }

        from sqlalchemy.dialects.postgresql import insert as pg_insert

        stmt = pg_insert(AgentMemory).values(
            category=category,
            key=key,
            content=content,
            priority=priority,
            source=source,
            memory_metadata=metadata or {},
        )
        # Build update set.  On conflict (upsert), mark source as 'agent'
        # so we know the entry was touched by the AI (not still the original seed).
        update_set: dict[str, Any] = {
            "content": content,
            "priority": priority,
            "source": source or "agent",
            "updated_at": utc_now(),
        }
        if metadata is not None:
            update_set["memory_metadata"] = metadata
        stmt = stmt.on_conflict_do_update(
            index_elements=["category", "key"],
            set_=update_set,
        )
        await self.session.execute(stmt)
        await self.session.flush()

        # Invalidate prompt cache so new memory is reflected immediately
        from kai_memory.ports import invalidate_prompt_cache

        invalidate_prompt_cache()

        # Fire-and-forget embedding generation — failure must not block save.
        # generate_embedding now RAISES on failure rather than returning a zero
        # vector. Crucially this is an upsert: on an UPDATE, a stale embedding
        # from the prior content survives in the column. So whenever we can't
        # compute a valid (non-zero) embedding for the current content, we must
        # explicitly NULL the column — otherwise semantic search indexes the old
        # content under the new text. A NULL is filtered cleanly and backfilled.
        try:
            embedding: list[float] | None = await generate_embedding(content)
        except Exception:
            logger.warning(
                "Failed to generate embedding for memory %s/%s — clearing stale embedding",
                category,
                key,
                exc_info=True,
            )
            embedding = None

        # Guard the embedding write with `content`: the embedding was computed
        # from THIS call's content, so it (or the NULL clear) must only apply
        # while the row still holds that content. If a concurrent save() already
        # wrote newer content, the predicate no longer matches and this stale
        # write is a harmless no-op — it can't clobber the newer embedding.
        if embedding is not None and not is_zero_vector(embedding):
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
            await self.session.execute(
                text(
                    "UPDATE agent_memory SET embedding = CAST(:emb AS vector) "
                    "WHERE category = :cat AND key = :k AND content = :content"
                ),
                {"emb": embedding_str, "cat": category, "k": key, "content": content},
            )
        else:
            if embedding is not None:  # zero-norm vector
                logger.warning("Zero-norm embedding for memory %s/%s — clearing", category, key)
            await self.session.execute(
                text(
                    "UPDATE agent_memory SET embedding = NULL WHERE category = :cat AND key = :k AND content = :content"
                ),
                {"cat": category, "k": key, "content": content},
            )

        result = {"status": "saved", "category": category, "key": key}
        if category in ALWAYS_LOADED_CATEGORIES:
            result["warning"] = (
                f"This modifies the '{category}' category which is always loaded into the AI system prompt."
            )
        return result

    async def get(self, category: str, key: str) -> dict[str, Any]:
        stmt = select(AgentMemory).where(AgentMemory.category == category, AgentMemory.key == key)
        result = await self.session.execute(stmt)
        memory = result.scalar_one_or_none()
        if not memory:
            return {"found": False, "category": category, "key": key}
        return {
            "category": memory.category,
            "key": memory.key,
            "content": memory.content,
            "priority": memory.priority,
            "found": True,
        }

    async def get_always_loaded(self) -> list[dict[str, Any]]:
        """Get all memory entries that should be embedded in every system prompt."""
        stmt = (
            select(AgentMemory)
            .where(AgentMemory.category.in_(ALWAYS_LOADED_CATEGORIES))
            .order_by(AgentMemory.priority.desc(), AgentMemory.category, AgentMemory.key)
        )
        result = await self.session.execute(stmt)
        memories = result.scalars().all()

        entries = []
        total_chars = 0
        for m in memories:
            if total_chars + len(m.content) > MAX_ALWAYS_LOADED_CHARS:
                break
            entries.append({"category": m.category, "key": m.key, "content": m.content})
            total_chars += len(m.content)

        if total_chars > 20_000:
            logger.warning(
                "Always-loaded memory is %d chars (warning at 20K, limit: %d) — consider pruning low-priority entries",
                total_chars,
                MAX_ALWAYS_LOADED_CHARS,
            )
        return entries

    async def delete(self, category: str, key: str) -> dict[str, Any]:
        stmt = select(AgentMemory).where(AgentMemory.category == category, AgentMemory.key == key)
        result = await self.session.execute(stmt)
        existing = result.scalar_one_or_none()
        if not existing:
            return {"status": "not_found"}
        await self.session.delete(existing)
        await self.session.flush()

        # Invalidate prompt cache so deleted memory is reflected immediately
        from kai_memory.ports import invalidate_prompt_cache

        invalidate_prompt_cache()

        return {"status": "deleted", "category": category, "key": key}
