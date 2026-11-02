"""Unit tests for ContrastiveMapProducer (no-op until M3)."""

from __future__ import annotations

import contextlib
import logging
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from kai_memory.artifacts.producer import ProducerSubject
from kai_memory.artifacts.producers.contrastive_map import (
    ContrastiveMapProducer,
)


async def test_discover_subjects_returns_empty_when_table_empty() -> None:
    producer = ContrastiveMapProducer()
    session = MagicMock()
    empty = MagicMock()
    empty.first = MagicMock(return_value=None)
    session.execute = AsyncMock(return_value=empty)

    subjects = await producer.discover_subjects(session, uuid.uuid4())
    assert subjects == []


async def test_discover_subjects_logs_warning_when_table_populated() -> None:
    """When confusion_sets has rows but Wave 2 still ships the no-op,
    the producer must log a warning so the M3 implementer notices."""
    producer = ContrastiveMapProducer()
    session = MagicMock()
    nonempty = MagicMock()
    nonempty.first = MagicMock(return_value=(1,))
    session.execute = AsyncMock(return_value=nonempty)

    import logging

    with _captured_logs("app.services.memory.artifacts.producers.contrastive_map") as records:
        subjects = await producer.discover_subjects(session, uuid.uuid4())

    assert subjects == []
    assert any(r.levelno == logging.WARNING for r in records)


async def test_produce_raises_not_implemented_in_wave_2() -> None:
    """produce() should be unreachable in Wave 2 — discover_subjects always
    returns []. Failing loud here catches a future bug that bypasses the gate."""
    producer = ContrastiveMapProducer()
    subject = ProducerSubject(key="x", payload={})
    with pytest.raises(NotImplementedError):
        await producer.produce(MagicMock(), subject, uuid.uuid4())


# ── Helpers ───────────────────────────────────────────────


@contextlib.contextmanager
def _captured_logs(logger_name: str):
    """Capture log records emitted by ``logger_name`` for the duration of the block."""
    logger = logging.getLogger(logger_name)
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    prior_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior_level)
