"""Unit tests for the one-time canonicalization data-cleanup helpers.

Covers the NEW decision logic (force-survivor selection, telegram_user_id guard,
soft-close-not-delete, backfill key computation). The actual relationship/link
re-pointing is delegated to the already-tested
``consolidation._dedup_and_transfer_relationships`` and is mocked here; its
real DB behavior is exercised by the integration test
(``tests/integration/test_canonicalization_cleanup_integration.py``).
"""

from __future__ import annotations

import uuid
from contextlib import nullcontext
from unittest.mock import AsyncMock, MagicMock

import pytest

from kai_memory.storage.models.entity_relationship import EntityRelationship
from kai_memory.storage.models.memory_entity import MemoryEntity
from kai_memory import canonicalization_cleanup as cc

pytestmark = pytest.mark.asyncio


def _entity(name, entity_type="person", mention_count=0, telegram_user_id=None, canonical_key=None):
    return MemoryEntity(
        id=uuid.uuid4(),
        name=name,
        entity_type=entity_type,
        aliases=[],
        mention_count=mention_count,
        telegram_user_id=telegram_user_id,
        canonical_key=canonical_key,
        merged_into_id=None,
    )


def _session_with(entities):
    """AsyncMock session whose .get(MemoryEntity, id) returns from a dict."""
    by_id = {e.id: e for e in entities}
    session = AsyncMock()
    session.get = AsyncMock(side_effect=lambda model, _id: by_id.get(_id))
    session.flush = AsyncMock()
    session.no_autoflush = nullcontext()
    return session


# ── force-survivor ──────────────────────────────────────────────────────────


async def test_force_survivor_keeps_designated_survivor_even_with_fewer_mentions(monkeypatch):
    """The agent (m=2) must remain survivor over a person variant (m=20)."""
    transfer = AsyncMock()
    monkeypatch.setattr(cc, "_dedup_and_transfer_relationships", transfer)

    survivor = _entity("Kai", entity_type="agent", mention_count=2)
    absorbed = _entity("Kai", entity_type="person", mention_count=20)
    session = _session_with([survivor, absorbed])

    result = await cc.merge_into_survivor(session, survivor.id, [absorbed.id], reason="cleanup")

    assert survivor.merged_into_id is None  # survivor untouched
    assert absorbed.merged_into_id == survivor.id  # absorbed folded in
    assert result["merged"] == [absorbed.id]
    transfer.assert_awaited_once_with(session, survivor.id, absorbed.id)


async def test_absorbed_name_added_to_survivor_aliases(monkeypatch):
    monkeypatch.setattr(cc, "_dedup_and_transfer_relationships", AsyncMock())
    survivor = _entity("Misha Petrov", mention_count=2783)
    absorbed = _entity("Михаил Петрав", mention_count=11)
    session = _session_with([survivor, absorbed])

    await cc.merge_into_survivor(session, survivor.id, [absorbed.id], reason="cleanup")

    assert "Михаил Петрав" in survivor.aliases
    assert absorbed.merge_reason == "cleanup"
    assert absorbed.merged_at is not None


# ── telegram_user_id guard (mirror resolver: block only distinct non-NULL) ──


async def test_tg_guard_blocks_merge_across_distinct_non_null_tg(monkeypatch):
    transfer = AsyncMock()
    monkeypatch.setattr(cc, "_dedup_and_transfer_relationships", transfer)
    survivor = _entity("Misha", telegram_user_id=100)
    absorbed = _entity("Misha", telegram_user_id=200)
    session = _session_with([survivor, absorbed])

    result = await cc.merge_into_survivor(session, survivor.id, [absorbed.id], reason="cleanup")

    assert absorbed.merged_into_id is None  # NOT merged
    assert result["merged"] == []
    assert absorbed.id in result["skipped_tg_conflict"]
    transfer.assert_not_awaited()


async def test_tg_guard_blocks_merge_when_zero_is_present(monkeypatch):
    """A telegram_user_id of 0 is non-NULL → must still block a conflict
    (truthiness would wrongly let it through)."""
    transfer = AsyncMock()
    monkeypatch.setattr(cc, "_dedup_and_transfer_relationships", transfer)
    survivor = _entity("Misha", telegram_user_id=0)
    absorbed = _entity("Misha", telegram_user_id=1)
    session = _session_with([survivor, absorbed])

    result = await cc.merge_into_survivor(session, survivor.id, [absorbed.id], reason="cleanup")

    assert absorbed.merged_into_id is None  # NOT merged
    assert absorbed.id in result["skipped_tg_conflict"]
    transfer.assert_not_awaited()


async def test_tg_guard_allows_merge_when_absorbed_tg_is_null(monkeypatch):
    monkeypatch.setattr(cc, "_dedup_and_transfer_relationships", AsyncMock())
    survivor = _entity("Misha", telegram_user_id=100)
    absorbed = _entity("Миша", telegram_user_id=None)
    session = _session_with([survivor, absorbed])

    result = await cc.merge_into_survivor(session, survivor.id, [absorbed.id], reason="cleanup")

    assert absorbed.merged_into_id == survivor.id
    assert result["merged"] == [absorbed.id]


# ── edge soft-close (valid_until, never hard-delete) ────────────────────────


async def test_close_relationship_sets_valid_until_and_does_not_delete():
    rel = EntityRelationship(
        id=uuid.uuid4(),
        from_entity_id=uuid.uuid4(),
        to_entity_id=uuid.uuid4(),
        relationship_type="manager_of",
        valid_until=None,
    )
    session = AsyncMock()
    session.get = AsyncMock(return_value=rel)
    session.delete = AsyncMock()
    session.flush = AsyncMock()

    closed = await cc.close_relationship(session, rel.id)

    assert closed is True
    assert rel.valid_until is not None  # soft-closed
    session.delete.assert_not_called()  # NEVER hard-delete


async def test_close_relationship_noop_when_already_closed():
    from kai_memory.storage.models.base import utc_now

    already = utc_now()
    rel = EntityRelationship(
        id=uuid.uuid4(),
        from_entity_id=uuid.uuid4(),
        to_entity_id=uuid.uuid4(),
        relationship_type="manager_of",
        valid_until=already,
    )
    session = AsyncMock()
    session.get = AsyncMock(return_value=rel)
    session.delete = AsyncMock()

    closed = await cc.close_relationship(session, rel.id)

    assert closed is False  # already closed → no-op
    assert rel.valid_until == already  # unchanged
    session.delete.assert_not_called()


# ── backfill canonical_key ──────────────────────────────────────────────────


async def test_backfill_sets_translit_key_skips_empty_leaves_keyed_untouched():
    keyed = _entity("Already", canonical_key="already")  # must not appear (filtered by query)
    cyr = _entity("Миша")
    case = _entity("Checkout", entity_type="project")
    garbage = _entity("!!!", entity_type="object")  # translit_key('!!!') == ''
    null_rows = [cyr, case, garbage]

    result = MagicMock()
    result.scalars.return_value.all.return_value = null_rows
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    session.flush = AsyncMock()

    count = await cc.backfill_canonical_key(session)

    assert cyr.canonical_key == "misha"
    assert case.canonical_key == "checkout"
    assert garbage.canonical_key is None  # empty key → stays NULL
    assert count == 2  # only the two non-empty keys counted
    assert keyed.canonical_key == "already"  # untouched


async def test_operator_merge_preflights_all_absorbed_identity_bindings(monkeypatch):
    from kai_memory.identity import AGENT_IDENTITY_DATA_KEY

    survivor = _entity("Requested survivor")
    earlier = _entity("Ordinary fork")
    bound = _entity("Atlas", entity_type="agent")
    bound.data = {AGENT_IDENTITY_DATA_KEY: "host-agent"}
    session = _session_with([survivor, earlier, bound])
    transfer = AsyncMock()
    monkeypatch.setattr(cc, "_dedup_and_transfer_relationships", transfer)

    with pytest.raises(ValueError, match="cannot absorb bound agent identity"):
        await cc.merge_into_survivor(session, survivor.id, [earlier.id, bound.id], reason="cleanup")

    assert survivor.aliases == earlier.aliases == bound.aliases == []
    assert earlier.merged_into_id is bound.merged_into_id is None
    assert bound.data == {AGENT_IDENTITY_DATA_KEY: "host-agent"}
    session.execute.assert_not_awaited()
    session.flush.assert_not_awaited()
    transfer.assert_not_awaited()


@pytest.mark.parametrize("binding", ["host-agent", "other-agent"])
async def test_operator_merge_rejects_two_bound_uuids(monkeypatch, binding):
    from kai_memory.identity import AGENT_IDENTITY_DATA_KEY

    survivor = _entity("Atlas", entity_type="agent")
    survivor.data = {AGENT_IDENTITY_DATA_KEY: "host-agent"}
    absorbed = _entity("Other", entity_type="agent")
    absorbed.data = {AGENT_IDENTITY_DATA_KEY: binding}
    session = _session_with([survivor, absorbed])
    transfer = AsyncMock()
    monkeypatch.setattr(cc, "_dedup_and_transfer_relationships", transfer)

    with pytest.raises(ValueError, match="two bound agent identities"):
        await cc.merge_into_survivor(session, survivor.id, [absorbed.id], reason="cleanup")

    assert survivor.aliases == absorbed.aliases == []
    assert absorbed.merged_into_id is None
    session.execute.assert_not_awaited()
    session.flush.assert_not_awaited()
    transfer.assert_not_awaited()


async def test_operator_merge_keeps_designated_bound_survivor(monkeypatch):
    from kai_memory.identity import AGENT_IDENTITY_DATA_KEY

    survivor = _entity("Atlas", entity_type="agent", mention_count=1)
    survivor.data = {AGENT_IDENTITY_DATA_KEY: "host-agent"}
    absorbed = _entity("Duplicate", entity_type="agent", mention_count=100)
    session = _session_with([survivor, absorbed])
    transfer = AsyncMock()
    monkeypatch.setattr(cc, "_dedup_and_transfer_relationships", transfer)

    result = await cc.merge_into_survivor(session, survivor.id, [absorbed.id], reason="cleanup")

    assert result["merged"] == [absorbed.id]
    assert absorbed.merged_into_id == survivor.id and survivor.merged_into_id is None
    assert survivor.data == {AGENT_IDENTITY_DATA_KEY: "host-agent"}
    transfer.assert_awaited_once_with(session, survivor.id, absorbed.id)


@pytest.mark.parametrize("telegram_user_id", [0, 42])
async def test_operator_merge_rejects_bound_agent_and_verified_human(monkeypatch, telegram_user_id):
    from kai_memory.identity import AGENT_IDENTITY_DATA_KEY

    survivor = _entity("Atlas", entity_type="agent")
    survivor.data = {AGENT_IDENTITY_DATA_KEY: "host-agent"}
    human = _entity("Human", telegram_user_id=telegram_user_id)
    session = _session_with([survivor, human])
    transfer = AsyncMock()
    monkeypatch.setattr(cc, "_dedup_and_transfer_relationships", transfer)

    with pytest.raises(ValueError, match="Telegram user"):
        await cc.merge_into_survivor(session, survivor.id, [human.id], reason="cleanup")

    assert survivor.aliases == human.aliases == []
    assert human.merged_into_id is None and human.telegram_user_id == telegram_user_id
    session.execute.assert_not_awaited()
    session.flush.assert_not_awaited()
    transfer.assert_not_awaited()


@pytest.mark.parametrize("already_merged", [False, True])
async def test_operator_preflight_preserves_missing_and_unbound_row_handling(monkeypatch, already_merged):
    survivor = _entity("Designated survivor")
    absorbed = _entity("Unbound fork")
    if already_merged:
        absorbed.merged_into_id = uuid.uuid4()
    missing_id = uuid.uuid4()
    session = _session_with([survivor, absorbed])
    transfer = AsyncMock()
    monkeypatch.setattr(cc, "_dedup_and_transfer_relationships", transfer)

    result = await cc.merge_into_survivor(session, survivor.id, [missing_id, absorbed.id], reason="cleanup")

    assert result["skipped_missing"] == [missing_id]
    assert result["merged"] == [absorbed.id]
    assert absorbed.merged_into_id == survivor.id
    transfer.assert_awaited_once_with(session, survivor.id, absorbed.id)
