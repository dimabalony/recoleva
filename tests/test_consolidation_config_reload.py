"""Regression: consolidation must RELOAD cognitive_config from the DB mid-pass,
not reset it to baked-in defaults.

Original bug: a mid-pass `cognitive_config.reset_cache()` reverted the in-process
cache to `_DEFAULTS`, silently flipping DB-tuned flags like
`entityschema_deterministic_enabled` back to 0.0 so the EntitySchema shadow phase
never fired despite the prod flag being 1.0. (The original offender,
`_adjust_retrieval_weights`, was deleted outright as a runaway — see
`test_runaway_weight_adjustment_removed` — leaving the artifact pipeline's
defensive reload as the guarded path.)
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from kai_memory.store import cognitive_config
from kai_memory.consolidation import consolidation

pytestmark = pytest.mark.asyncio


async def test_run_artifact_pipeline_reloads_config_at_start(monkeypatch):
    """Defensive: the pipeline reloads live DB config BEFORE producers gate on flags.

    Isolation trick: ``begin_nested()`` raises a sentinel right after the reload,
    so this asserts the reload ran first without stubbing the whole producer stack.
    On the old code (no reload at the top) the sentinel fires before any
    ``load_config`` call, so ``assert_awaited_once`` fails — a true RED.
    """
    load_mock = AsyncMock()
    monkeypatch.setattr(cognitive_config, "load_config", load_mock)

    class _Sentinel(Exception):
        pass

    session = AsyncMock()
    session.begin_nested = MagicMock(side_effect=_Sentinel)

    with pytest.raises(_Sentinel):
        await consolidation._run_artifact_pipeline(session, uuid.uuid4())

    load_mock.assert_awaited_once_with(session)
