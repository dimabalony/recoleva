"""ExtractionCursor serialization (DB-794): the run-row form is stable across time zones."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta, timezone

from kai_memory.extraction_cursor import ExtractionCursor


def test_same_instant_serializes_alike_in_any_time_zone():
    message_id = uuid.uuid4()
    utc = ExtractionCursor(created_at=datetime(2026, 10, 16, 12, 0, tzinfo=UTC), id=message_id)
    local = ExtractionCursor(created_at=utc.created_at.astimezone(timezone(timedelta(hours=8))), id=message_id)

    assert local.to_metadata() == utc.to_metadata()
    assert ExtractionCursor.from_metadata(local.to_metadata()) == utc


def test_legacy_cursor_round_trips_without_an_id():
    legacy = ExtractionCursor(created_at=datetime(2026, 10, 16, 12, 0, tzinfo=UTC), id=None)

    assert legacy.to_metadata()["id"] is None
    assert ExtractionCursor.from_metadata(legacy.to_metadata()) == legacy
    assert ExtractionCursor.from_metadata(None) is None
