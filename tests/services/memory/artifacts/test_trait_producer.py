"""Unit tests for TraitProducer (Sub-design D Wave 2 §3.3).

Covers parse + produce + decay-formula correctness + capacity arithmetic +
retention-staleness eviction + capacity-drop log + noop-path last_reinforced_at
touch. DB-side decay UPDATE and full pipeline integration live in
``tests/integration/test_persona_pipeline.py``.
"""

from __future__ import annotations

import logging
import math
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from kai_memory.store import cognitive_config
from kai_memory.artifacts.producer import ProducerSubject, compute_content_hash
from kai_memory.artifacts.producers.trait_producer import (
    TraitProducer,
    _parse_trait_response,
)
from kai_memory.artifacts.types import ProducedArtifact


# ── parser ─────────────────────────────────────────────────


def test_parse_trait_response_keeps_only_threshold_passing() -> None:
    raw = """[
      {"dimension": "directness", "direction": "increasing", "magnitude": 0.6,
       "guidance_text": "You commit to recommendations without hedging."},
      {"dimension": "warmth", "direction": "stable", "magnitude": 0.1,
       "guidance_text": "You stay warm."},
      {"dimension": "x", "direction": "wrong-direction", "magnitude": 0.5,
       "guidance_text": "g"}
    ]"""
    parsed = _parse_trait_response(raw)
    # warmth dropped (mag < 0.3); x dropped (bad direction); directness kept.
    assert len(parsed) == 1
    assert parsed[0]["dimension"] == "directness"
    assert parsed[0]["magnitude"] == pytest.approx(0.6)


def test_parse_trait_response_handles_string_magnitude() -> None:
    raw = '[{"dimension": "d", "direction": "stable", "magnitude": "0.4", "guidance_text": "g"}]'
    parsed = _parse_trait_response(raw)
    assert len(parsed) == 1
    assert parsed[0]["magnitude"] == pytest.approx(0.4)


def test_parse_trait_response_drops_invalid() -> None:
    raw = (
        '[{"dimension": "", "direction": "stable", "magnitude": 0.5, "guidance_text": "g"},'
        ' {"dimension": "x", "direction": "stable", "magnitude": null, "guidance_text": "g"},'
        ' {"dimension": "y", "direction": "stable", "magnitude": 0.5, "guidance_text": ""}]'
    )
    parsed = _parse_trait_response(raw)
    assert parsed == []


def test_parse_trait_response_clamps_magnitude() -> None:
    raw = '[{"dimension": "d", "direction": "stable", "magnitude": 1.7, "guidance_text": "g"}]'
    parsed = _parse_trait_response(raw)
    assert parsed[0]["magnitude"] == 1.0


# ── compute_strength (decay formula) ───────────────────────


def test_compute_strength_baseline() -> None:
    """day=0, rehearsal=0 → strength == magnitude (no decay, no rehearsal bonus)."""
    s = TraitProducer.compute_strength(
        magnitude=0.6, days_since_last_reinforced=0.0, rehearsal_count=0, decay_lambda=0.0055
    )
    assert s == pytest.approx(0.6)


def test_compute_strength_decays_with_days() -> None:
    """At lambda=0.0055, strength after 30 days ≈ magnitude * exp(-0.165) ≈ 0.847x."""
    s = TraitProducer.compute_strength(
        magnitude=1.0, days_since_last_reinforced=30.0, rehearsal_count=0, decay_lambda=0.0055
    )
    assert s == pytest.approx(math.exp(-0.165), rel=1e-3)


def test_compute_strength_rehearsal_bonus() -> None:
    """rehearsal_count multiplies (1 + 0.2 * count)."""
    s_base = TraitProducer.compute_strength(
        magnitude=0.5, days_since_last_reinforced=0.0, rehearsal_count=0, decay_lambda=0.0055
    )
    s_5x = TraitProducer.compute_strength(
        magnitude=0.5, days_since_last_reinforced=0.0, rehearsal_count=5, decay_lambda=0.0055
    )
    assert s_5x == pytest.approx(s_base * 2.0)


def test_compute_strength_below_threshold_at_long_horizon() -> None:
    """A magnitude=0.3 trait with no rehearsal decays below 0.2 within ~110 days."""
    s = TraitProducer.compute_strength(
        magnitude=0.3, days_since_last_reinforced=120.0, rehearsal_count=0, decay_lambda=0.0055
    )
    assert s < 0.2


def test_compute_strength_clamps_negative_days() -> None:
    """Defensive: future timestamps shouldn't blow up the formula."""
    s_neg = TraitProducer.compute_strength(
        magnitude=0.5, days_since_last_reinforced=-10.0, rehearsal_count=0, decay_lambda=0.0055
    )
    s_zero = TraitProducer.compute_strength(
        magnitude=0.5, days_since_last_reinforced=0.0, rehearsal_count=0, decay_lambda=0.0055
    )
    assert s_neg == pytest.approx(s_zero)


# ── produce() ──────────────────────────────────────────────


async def test_produce_emits_trait_with_stable_hash() -> None:
    """Stable hash on (subject, dimension, direction, magnitude, guidance)."""
    producer = TraitProducer()
    agent_id = uuid.uuid4()
    subject = ProducerSubject(
        key="trait::x",
        payload={
            "agent_self_id": agent_id,
            "dimension": "directness",
            "direction": "increasing",
            "magnitude": 0.6,
            "guidance_text": "You commit without hedging.",
            "rehearsal_count_seed": 4,
        },
    )
    result_a = await producer.produce(MagicMock(), subject, uuid.uuid4())
    # Re-running with the same shape → same content_hash (timestamps not in hash).
    result_b = await producer.produce(MagicMock(), subject, uuid.uuid4())
    assert result_a is not None and result_b is not None
    assert result_a.content_hash == result_b.content_hash
    assert result_a.subject_entity_id == agent_id
    assert result_a.dimension == "directness"
    assert result_a.payload["rehearsal_count"] == 4


async def test_produce_payload_validates_against_schema() -> None:
    """Payload model rejects out-of-bounds magnitude → produce() raises."""
    producer = TraitProducer()
    subject = ProducerSubject(
        key="t",
        payload={
            "agent_self_id": uuid.uuid4(),
            "dimension": "d",
            "direction": "stable",
            "magnitude": 1.5,  # > 1.0
            "guidance_text": "g",
            "rehearsal_count_seed": 0,
        },
    )
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        await producer.produce(MagicMock(), subject, uuid.uuid4())


# ── discover_subjects capacity logic ───────────────────────


async def test_discover_subjects_skips_when_no_kai_self() -> None:
    producer = TraitProducer()

    async def _no_self(session):
        return None

    producer._get_kai_self_entity_id = _no_self  # type: ignore[method-assign]
    # _demote_below_threshold runs first (before the kai-self lookup);
    # _expire_stale_traits would run after but kai_self_id=None short-circuits.
    # Mock execute to return rowcount 0 for the single decay UPDATE.
    session = MagicMock()
    decay_result = MagicMock()
    decay_result.rowcount = 0
    session.execute = AsyncMock(return_value=decay_result)
    subjects = await producer.discover_subjects(session, uuid.uuid4())
    assert subjects == []


async def test_discover_subjects_respects_cap_for_fresh_dimensions() -> None:
    """When cap-of-12 is full, fresh dimensions get filtered; supersedes pass through."""
    producer = TraitProducer()
    agent_id = uuid.uuid4()

    async def _self(session):
        return agent_id

    producer._get_kai_self_entity_id = _self  # type: ignore[method-assign]

    # Decay UPDATE (rowcount 0)
    decay_result = MagicMock()
    decay_result.rowcount = 0

    # Active dimensions query: 12 active (cap = 12 → 0 fresh capacity).
    active_rows = [(f"existing_{i}",) for i in range(12)]
    active_result = MagicMock()
    active_result.all.return_value = active_rows

    # Fetch active patterns (empty)
    patterns_result = MagicMock()
    patterns_result.mappings.return_value.all.return_value = []

    # Fetch recent traces (3+ to pass the gate)
    class _T:
        def __init__(self):
            self.id = uuid.uuid4()
            self.content = "x"
            self.created_at = datetime.now(UTC)

    traces_result = MagicMock()
    traces_result.scalars.return_value.all.return_value = [_T() for _ in range(3)]

    # Order in producer: decay → expire-stale → active_dims → patterns → traces.
    # discover_subjects executes (in order):
    #   1. _demote_below_threshold UPDATE         → decay_result
    #   2. _get_kai_self_entity_id                → mocked to bypass execute
    #   3. _expire_stale_traits UPDATE..RETURNING → expire_result (0 rows)
    #   4. _fetch_active_patterns                 → patterns_result
    #   5. _fetch_recent_traces                   → traces_result
    #   6. _call_sonnet_trait_inference           → mocked
    #   7. _fetch_active_trait_dimensions         → active_result
    expire_result = MagicMock()
    expire_result.all.return_value = []
    session = MagicMock()
    session.execute = AsyncMock(
        side_effect=[decay_result, expire_result, patterns_result, traces_result, active_result]
    )

    async def fake_inference(patterns, traces):
        # Sonnet returns: 1 supersede candidate (existing_0) + 2 fresh.
        return [
            {"dimension": "existing_0", "direction": "stable", "magnitude": 0.5, "guidance_text": "g"},
            {"dimension": "fresh_a", "direction": "stable", "magnitude": 0.5, "guidance_text": "g"},
            {"dimension": "fresh_b", "direction": "stable", "magnitude": 0.5, "guidance_text": "g"},
        ]

    producer._call_sonnet_trait_inference = fake_inference  # type: ignore[method-assign]

    cognitive_config.set_for_testing("persona_max_active_traits", 12.0)
    try:
        subjects = await producer.discover_subjects(session, uuid.uuid4())
    finally:
        cognitive_config.reset_cache()

    # Expect 1 subject — only the supersede candidate (existing_0). Both
    # fresh_a and fresh_b are dropped because capacity_remaining = 12 - 12 = 0.
    dims = [s.payload["dimension"] for s in subjects]
    assert dims == ["existing_0"]


# ── retention-staleness eviction (_expire_stale_traits) ───


async def _run_expire_with_rows(rows: list[dict[str, object]]):
    """Helper mirroring stance_producer's test harness — run _expire_stale_traits
    with a mock session whose UPDATE..RETURNING surfaces ``rows``.
    """
    producer = TraitProducer()

    class _Row:
        def __init__(self, d):
            self.dimension = d["dimension"]
            self.magnitude = d["magnitude"]
            self.last_reinforced_at = d["last_reinforced_at"]

    result = MagicMock()
    result.all = MagicMock(return_value=[_Row(r) for r in rows])
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)
    n = await producer._expire_stale_traits(session, uuid.uuid4())
    return producer, session, n


async def test_expire_stale_traits_stale_row_returned() -> None:
    """``last_reinforced_at`` past retention → row reported as expired."""
    cognitive_config.set_for_testing("persona_trait_retention_days", 7.0)
    try:
        stale_iso = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        _, session, n = await _run_expire_with_rows(
            [{"dimension": "stale_trait", "magnitude": 0.7, "last_reinforced_at": stale_iso}]
        )
    finally:
        cognitive_config.reset_cache()
    assert n == 1
    assert session.execute.await_count == 1


async def test_expire_stale_traits_no_rows_when_all_fresh() -> None:
    """Healthy trait — UPDATE..RETURNING returns empty."""
    cognitive_config.set_for_testing("persona_trait_retention_days", 7.0)
    try:
        _, _, n = await _run_expire_with_rows([])
    finally:
        cognitive_config.reset_cache()
    assert n == 0


async def test_expire_stale_traits_sql_targets_personality_trait() -> None:
    """Lock the SQL shape: WHERE clause filters PersonalityTrait + status='active'
    + ``last_reinforced_at < :cutoff``. A future change that drops the
    artifact-type filter would expire Stance/Pattern rows by accident.
    """
    producer = TraitProducer()
    cognitive_config.set_for_testing("persona_trait_retention_days", 7.0)

    captured: dict[str, str] = {}

    async def capture_execute(stmt, params=None):
        captured["sql"] = str(stmt).lower()
        result = MagicMock()
        result.all = MagicMock(return_value=[])
        return result

    session = MagicMock()
    session.execute = capture_execute
    try:
        await producer._expire_stale_traits(session, uuid.uuid4())
    finally:
        cognitive_config.reset_cache()
    sql = captured["sql"]
    assert "personalitytrait" in sql
    assert "last_reinforced_at" in sql
    assert "status = 'active'" in sql


# ── noop-path last_reinforced_at touch (supersede override) ──


async def test_supersede_noop_touches_last_reinforced_at() -> None:
    """Hash-match noop must UPDATE prev's last_reinforced_at in JSONB.

    Without the touch, retention=7 expires every stable trait after 7 days
    even when Sonnet keeps surfacing it. Mirrors the Stance #137 noop-touch
    pattern.
    """
    producer = TraitProducer()
    agent_id = uuid.uuid4()
    prev_id = uuid.uuid4()
    one_week_ago = (datetime.now(UTC) - timedelta(days=7)).isoformat()
    payload = {
        "dimension": "directness",
        "direction": "increasing",
        "magnitude": 0.6,
        "guidance_text": "You commit without hedging.",
        "last_reinforced_at": one_week_ago,
        "rehearsal_count": 4,
    }
    stable_hash = compute_content_hash(
        {
            "subject_entity_id": str(agent_id),
            "dimension": "directness",
            "direction": "increasing",
            "magnitude": 0.6,
            "guidance_text": "You commit without hedging.",
        }
    )

    prev = MagicMock()
    prev.id = prev_id
    prev.status = "active"
    prev.content_hash = stable_hash
    prev.payload = payload

    new = ProducedArtifact(
        artifact_type="PersonalityTrait",
        schema_version=1,
        subject_entity_id=agent_id,
        dimension="directness",
        payload=payload,
        content_hash=stable_hash,
        human_summary="trait[directness] increasing mag=0.60",
    )

    captured: dict[str, object] = {}

    async def capture_execute(stmt, *args, **kwargs):
        # Capture the UPDATE statement so we can introspect the payload.
        # values() args land in the compiled statement; assert the touch
        # by comparing isoformat timestamps.
        compiled = stmt.compile(compile_kwargs={"literal_binds": False})
        captured["sql"] = str(compiled).lower()
        # Pull the payload bind value.
        params = stmt.compile().params if hasattr(stmt, "compile") else {}
        captured["params"] = params
        return MagicMock()

    session = MagicMock()
    session.execute = capture_execute

    entry = await producer.supersede(session, prev, new, uuid.uuid4())
    assert entry.action == "noop"
    assert entry.artifact_id == prev_id
    sql = captured["sql"]
    assert "update memory_artifacts" in sql
    # The UPDATE writes a new payload dict — verify last_reinforced_at moved
    # forward by inspecting the compiled bind for the payload column.
    bind_payload = captured["params"].get("payload")
    assert bind_payload is not None
    assert bind_payload["last_reinforced_at"] != one_week_ago
    # Touched timestamp should parse and be recent.
    touched = datetime.fromisoformat(bind_payload["last_reinforced_at"])
    assert (datetime.now(UTC) - touched).total_seconds() < 5


async def test_supersede_drift_uses_base_path() -> None:
    """Hash mismatch must NOT trigger the noop touch — falls through to base.

    Verified by patching super().supersede and asserting it's called.
    """
    producer = TraitProducer()
    agent_id = uuid.uuid4()
    prev = MagicMock()
    prev.id = uuid.uuid4()
    prev.status = "active"
    prev.content_hash = "old_hash_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    prev.payload = {"dimension": "directness", "magnitude": 0.5}

    new = ProducedArtifact(
        artifact_type="PersonalityTrait",
        schema_version=1,
        subject_entity_id=agent_id,
        dimension="directness",
        payload={"dimension": "directness", "magnitude": 0.9},
        content_hash="new_hash_yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy",
        human_summary="trait[directness] increasing mag=0.90",
    )

    # If super().supersede is invoked, it'll hit the real machinery — but
    # we can detect it by patching the BaseProducer method.
    from kai_memory.artifacts import producer as producer_module

    called = {"flag": False}

    original = producer_module.BaseProducer.supersede

    async def fake_super(self, session, prev_arg, new_arg, run_id):
        called["flag"] = True
        from kai_memory.artifacts.types import ChangesMadeEntry

        return ChangesMadeEntry(
            action="supersede",
            artifact_id=uuid.uuid4(),
            prev_artifact_id=prev_arg.id if prev_arg else None,
            artifact_type="PersonalityTrait",
            subject_entity_id=new_arg.subject_entity_id,
            human_summary=new_arg.human_summary,
            producer_name="TraitProducer",
            producer_version=1,
        )

    producer_module.BaseProducer.supersede = fake_super  # type: ignore[method-assign]
    try:
        session = MagicMock()
        session.execute = AsyncMock()
        entry = await producer.supersede(session, prev, new, uuid.uuid4())
    finally:
        producer_module.BaseProducer.supersede = original  # type: ignore[method-assign]

    assert called["flag"] is True
    assert entry.action == "supersede"


# ── capacity-drop INFO log ────────────────────────────────


async def test_capacity_drop_emits_info_log(caplog: pytest.LogCaptureFixture) -> None:
    """When the cap blocks fresh dimensions, log them at INFO with the cap
    + active count + a sample of dropped dimension names.

    Regression guard: pre-PR the dropped candidates vanished silently
    (``continue`` in the loop) — operators couldn't diagnose "0 produced"
    nights without re-running the producer locally.
    """
    producer = TraitProducer()
    agent_id = uuid.uuid4()

    async def _self(session):
        return agent_id

    producer._get_kai_self_entity_id = _self  # type: ignore[method-assign]

    decay_result = MagicMock()
    decay_result.rowcount = 0
    expire_result = MagicMock()
    expire_result.all.return_value = []

    # 12 active → 0 fresh capacity remaining.
    active_rows = [(f"existing_{i}",) for i in range(12)]
    active_result = MagicMock()
    active_result.all.return_value = active_rows

    patterns_result = MagicMock()
    patterns_result.mappings.return_value.all.return_value = []

    class _T:
        def __init__(self):
            self.id = uuid.uuid4()
            self.content = "x"
            self.created_at = datetime.now(UTC)

    traces_result = MagicMock()
    traces_result.scalars.return_value.all.return_value = [_T() for _ in range(3)]

    session = MagicMock()
    session.execute = AsyncMock(
        side_effect=[decay_result, expire_result, patterns_result, traces_result, active_result]
    )

    async def fake_inference(patterns, traces):
        # 3 fresh dimensions, 0 supersedes — all dropped at cap=12 / active=12.
        return [
            {"dimension": "fresh_a", "direction": "stable", "magnitude": 0.5, "guidance_text": "g"},
            {"dimension": "fresh_b", "direction": "stable", "magnitude": 0.5, "guidance_text": "g"},
            {"dimension": "fresh_c", "direction": "stable", "magnitude": 0.5, "guidance_text": "g"},
        ]

    producer._call_sonnet_trait_inference = fake_inference  # type: ignore[method-assign]

    cognitive_config.set_for_testing("persona_max_active_traits", 12.0)
    try:
        with caplog.at_level(logging.INFO, logger="app.services.memory.artifacts.producers.trait_producer"):
            subjects = await producer.discover_subjects(session, uuid.uuid4())
    finally:
        cognitive_config.reset_cache()

    # All 3 fresh dropped.
    assert subjects == []
    # Single INFO log with the dropped-dimension trail.
    drop_logs = [
        rec for rec in caplog.records if rec.levelno == logging.INFO and "candidate(s) dropped at cap" in rec.message
    ]
    assert len(drop_logs) == 1
    msg = drop_logs[0].message
    assert "cap=12" in msg
    assert "active=12" in msg
    assert "fresh_a" in msg
    assert "fresh_b" in msg
    assert "fresh_c" in msg


# ── max_tokens budget (P0 truncation fix) ─────────────────


async def test_call_sonnet_uses_8192_max_tokens() -> None:
    """TraitProducer must request max_tokens=8192.

    Bumped together with IdentityRecord and StanceProducer — all three
    used 2048 and showed 0-output on prod from silent JSON truncation.
    """
    from kai_memory.artifacts.producers import trait_producer as tp_module

    captured: dict[str, object] = {}

    class _FakeProvider:
        pass

    async def fake_call_llm_text(provider, prompt, *, max_tokens, span, **kwargs):
        captured["max_tokens"] = max_tokens
        captured["thinking_mode"] = kwargs.get("thinking_mode")
        captured["task_name"] = kwargs.get("task_name")
        return "[]"

    def fake_get_provider(settings, model):
        return _FakeProvider()

    def fake_span(name, model):
        return None

    producer = TraitProducer()
    pattern = MagicMock()
    pattern.id = uuid.uuid4()
    pattern.payload = {"dimension": "directness", "observation": "x", "reinforcement_count": 3}
    trace = MagicMock()
    trace.id = uuid.uuid4()
    trace.created_at = datetime.now(UTC)
    trace.content = "agent reflected on directness"

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(tp_module, "call_llm_text", fake_call_llm_text)
        monkey.setattr(tp_module, "get_provider_for_model", fake_get_provider)
        monkey.setattr(tp_module, "start_generation_span", fake_span)
        await producer._call_sonnet_trait_inference([pattern], [trace])
    finally:
        monkey.undo()

    assert captured["max_tokens"] == 8192
    assert captured["thinking_mode"] == "disabled"
    assert captured["task_name"] == "memory.trait.infer"
