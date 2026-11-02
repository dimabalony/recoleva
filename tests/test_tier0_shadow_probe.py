"""Unit tests for the Tier-0 shadow probe (task #8) — fail-open + flag-off.

Real-SQL surface-matching + H1-H4 logging live in
tests/integration/test_tier0_shadow_probe.py. Here we drive the mocked execute
sequence to prove the two safety properties: the probe NEVER breaks resolution,
and it does NOT run when the config flag is off.
"""

from __future__ import annotations

import logging
import uuid
from unittest.mock import AsyncMock, MagicMock

from kai_memory.storage.models.memory_entity import MemoryEntity
from kai_memory.store import cognitive_config
from kai_memory.store.cognitive_memory import CognitiveMemoryService

# asyncio_mode = auto (pyproject) auto-detects async tests; no module mark needed
# (a blanket asyncio mark warns on the sync default-flag test in this file).


def _entity(name: str) -> MemoryEntity:
    return MemoryEntity(id=uuid.uuid4(), name=name, entity_type="person", aliases=[], telegram_user_id=None)


def _lookup(entity: MemoryEntity | None) -> MagicMock:
    r = MagicMock()
    r.scalar_one_or_none.return_value = entity
    return r


def _insert(entity: MemoryEntity) -> MagicMock:
    r = MagicMock()
    r.scalar_one.return_value = entity
    return r


async def test_probe_error_is_swallowed_resolution_still_creates(monkeypatch, caplog):
    """FAIL-OPEN: a probe-query error must not escape — the fork is still created."""
    monkeypatch.setitem(cognitive_config._CACHE, "tier0_shadow_probe_enabled", 1.0)
    session = AsyncMock()
    new_entity = _entity("Misha")
    # Sequence on a text-mention miss: main lookup (miss) → PROBE query (raises)
    # → _create_entity insert. The probe error is caught internally; the insert
    # still runs.
    session.execute = AsyncMock(side_effect=[_lookup(None), RuntimeError("probe boom"), _insert(new_entity)])

    svc = CognitiveMemoryService(session)
    with caplog.at_level(logging.WARNING, logger="app.services.cognitive_memory"):
        result = await svc.find_or_create_entity("Misha", "person")

    assert result is new_entity  # resolution succeeded despite the probe failure
    assert session.execute.await_count == 3  # lookup + probe(raised) + insert
    assert any("tier0_shadow probe failed" in r.getMessage() for r in caplog.records)


async def test_probe_not_run_when_flag_off(monkeypatch, caplog):
    """Flag off → no probe query (only lookup + insert), no tier0_shadow log."""
    monkeypatch.setitem(cognitive_config._CACHE, "tier0_shadow_probe_enabled", 0.0)
    session = AsyncMock()
    new_entity = _entity("Misha")
    session.execute = AsyncMock(side_effect=[_lookup(None), _insert(new_entity)])

    svc = CognitiveMemoryService(session)
    with caplog.at_level(logging.INFO, logger="app.services.cognitive_memory"):
        result = await svc.find_or_create_entity("Misha", "person")

    assert result is new_entity
    assert session.execute.await_count == 2  # NO probe query
    assert not any("tier0_shadow" in r.getMessage() for r in caplog.records)


def test_probe_flag_defaults_on_without_a_db_row():
    """On prod the key is NOT migration-seeded, so the code-level default must
    resolve to ON — else the probe ships collecting nothing. get_bool reads the
    baked-in 1.0 default when no DB row overrides it."""
    cognitive_config.reset_cache()  # fresh cache, no rows loaded (mimics prod)
    assert cognitive_config.get_bool("tier0_shadow_probe_enabled") is True
    # Contrast: a genuinely unregistered key defaults to False (0.0) — proving the
    # ON default only holds because the key is registered in _DEFAULTS.
    assert cognitive_config.get_bool("some_unregistered_key") is False


async def test_probe_not_run_for_telegram_bound_mention(monkeypatch):
    """The probe is text-mention-only: a tg-bound incoming mention never probes."""
    monkeypatch.setitem(cognitive_config._CACHE, "tier0_shadow_probe_enabled", 1.0)
    session = AsyncMock()
    new_entity = _entity("Misha")
    # tg-bound path: tg lookup (miss) → name lookup (miss) → insert. No probe
    # (guarded by telegram_user_id is None).
    session.execute = AsyncMock(side_effect=[_lookup(None), _lookup(None), _insert(new_entity)])

    svc = CognitiveMemoryService(session)
    result = await svc.find_or_create_entity("Misha", "person", telegram_user_id=555)

    assert result is new_entity
    assert session.execute.await_count == 3  # tg lookup + name lookup + insert (no probe)
