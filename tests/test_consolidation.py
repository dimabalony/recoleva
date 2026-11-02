"""Unit tests for Level 3 nightly consolidation."""

from __future__ import annotations

import inspect
import json
import math
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai_memory.store.cognitive_config import get as _cfg_get
from kai_memory.store import cognitive_config
from kai_memory.consolidation.consolidation import (
    _classify_pair_relationship,
    _compute_spaced_access_bonus,
    _create_relationship_from_classification,
    _dedup_and_transfer_relationships,
    _execute_merge,
    _lock_entities_for_merge,
    _merge_entities,
    _process_new_traces,
    _reassess_significance,
    _resolve_merge_chain,
    _select_releasable_durable_traces,
    _update_relationships,
    consolidate,
)
from kai_memory.storage.models.entity_merge_candidate import EntityMergeCandidate
from kai_memory.storage.models.entity_relationship import EntityRelationship
from kai_memory.storage.models.memory_entity import MemoryEntity
from kai_memory.storage.models.memory_trace import MemoryTrace

BASE_DECAY = _cfg_get("base_decay")


def _attach_pair_entities(pair: dict, a_type: str = "person", b_type: str = "person") -> dict:
    pair["entity_a"] = MemoryEntity(id=pair["a_id"], name=pair.get("name_a", "Entity A"), entity_type=a_type)
    pair["entity_b"] = MemoryEntity(id=pair["b_id"], name=pair.get("name_b", "Entity B"), entity_type=b_type)
    return pair


# ── Decay tests ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_decay_significance_5_no_decay():
    """Significance 5 memories never decay — filtered out in WHERE clauses."""
    from kai_memory.consolidation.consolidation import _apply_decay

    session = AsyncMock()

    # Pass A: bulk SQL UPDATE (raw text) — no rows
    bulk_update_result = MagicMock()
    bulk_update_result.rowcount = 0
    # Pass B: batched SELECT — returns empty batch
    batch_result = MagicMock()
    batch_result.scalars.return_value = MagicMock(all=MagicMock(return_value=[]))
    # sig-5 consolidated_at marking
    sig5_update_result = MagicMock()
    sig5_update_result.rowcount = 1
    # durable-retained observability count (fact-loss fix heartbeat)
    durable_count_result = MagicMock()
    durable_count_result.scalar.return_value = 0
    session.execute = AsyncMock(
        side_effect=[bulk_update_result, batch_result, sig5_update_result, durable_count_result]
    )

    count = await _apply_decay(session)
    assert count == 0  # No sig<5 traces were decayed

    # Third call is the sig-5 marking (not a salience update); fourth is the durable count
    assert session.execute.await_count == 4
    third_call = session.execute.call_args_list[2]
    stmt = third_call[0][0]
    compiled = stmt.compile(compile_kwargs={"literal_binds": True})
    sql_str = str(compiled)
    assert "significance = 5" in sql_str
    assert "consolidated_at" in sql_str

    # Verify the bulk UPDATE filters sig < 5 and access_count = 0
    first_call = session.execute.call_args_list[0]
    first_sql = str(first_call[0][0])
    assert "significance < 5" in first_sql
    assert "access_count = 0" in first_sql


def _decay_trace(**overrides) -> MemoryTrace:
    """A minimal accessed trace for the batched (Python) decay pass."""
    now = datetime.now(UTC)
    fields = {
        "id": uuid.uuid4(),
        "content": "fact",
        "memory_type": "fact",
        "source_type": "conversation",
        "significance": 3,
        "salience": 0.5,
        "initial_salience": 0.5,
        "access_count": 5,
        "access_timestamps": [],  # no spaced bonus — isolates the decay anchor
        "created_at": now - timedelta(days=60),
        "last_accessed_at": None,
        "typed_metadata": {},
    }
    fields.update(overrides)
    return MemoryTrace(**fields)


async def _run_batched_decay(traces: list[MemoryTrace], now: datetime) -> dict:
    """Drive ``_decay_batched_pass`` over one mocked batch; return its UPDATE params."""
    from kai_memory.consolidation.consolidation import _decay_batched_pass

    session = AsyncMock()
    batch_result = MagicMock()
    batch_result.scalars.return_value = MagicMock(all=MagicMock(return_value=traces))
    session.execute = AsyncMock(side_effect=[batch_result, MagicMock()])

    count = await _decay_batched_pass(session, now)
    assert count == len(traces)
    return session.execute.call_args_list[1][0][1]


async def _run_multi_batch_decay(batches: list[list[MemoryTrace]], now: datetime) -> int:
    """Drive ``_decay_batched_pass`` across SEVERAL keyset batches.

    The pass paginates by ``decay_batch_size`` and stops on the first short batch, so
    the caller sets that config and passes batches whose lengths produce the intended
    number of loop iterations. Each iteration consumes one SELECT + one UPDATE.
    """
    from kai_memory.consolidation.consolidation import _decay_batched_pass

    session = AsyncMock()
    side_effect: list = []
    for batch in batches:
        batch_result = MagicMock()
        batch_result.scalars.return_value = MagicMock(all=MagicMock(return_value=batch))
        side_effect.extend([batch_result, MagicMock()])
    session.execute = AsyncMock(side_effect=side_effect)

    count = await _decay_batched_pass(session, now)
    # Every batch was actually consumed — a pass that exited early would silently
    # weaken the multi-batch premise of the tests below.
    assert session.execute.await_count == 2 * len(batches)
    return count


@pytest.mark.asyncio
async def test_decay_reports_broken_last_access_invariant_once_per_pass(caplog: pytest.LogCaptureFixture):
    """[review-r2] The `or created_at` fallback must not fail silently.

    This pass selects ``access_count > 0`` and every writer of ``last_accessed_at``
    bumps ``access_count`` in the same UPDATE, so a NULL here is impossible today. If
    a future writer breaks that, the anchor silently reverts to ``created_at`` — the
    exact drift DB-437 fixes — so it is reported. Aggregated per PASS, not per row and
    not per batch: either would flood CloudWatch if a legacy population ever matched.

    The offenders are split ACROSS TWO keyset batches on purpose. With the default
    batch size of 500 a single-batch fixture cannot tell per-pass logging apart from
    per-batch logging — moving the warning inside the `while` loop would still emit
    exactly one line and the test would pass while the log volume claim broke.
    """
    import logging

    cognitive_config.set_for_testing("decay_batch_size", 2)
    try:
        now = datetime.now(UTC)
        first = [
            _decay_trace(last_accessed_at=None),
            _decay_trace(last_accessed_at=now - timedelta(days=1)),  # healthy — must not be counted
        ]
        second = [_decay_trace(last_accessed_at=None)]  # short batch ends the pass

        with caplog.at_level(logging.WARNING):
            count = await _run_multi_batch_decay([first, second], now)

        assert count == 3  # both batches decayed
        warnings = [r for r in caplog.records if "access_count_without_last_accessed_at" in r.getMessage()]
        assert len(warnings) == 1  # one line per pass — NOT one per batch (which would be 2)
        assert "count=2" in warnings[0].getMessage()  # summed across batches; healthy row excluded
    finally:
        cognitive_config.reset_cache()


@pytest.mark.asyncio
async def test_decay_stays_silent_when_last_access_invariant_holds(caplog: pytest.LogCaptureFixture):
    """[review-r1] No warning on the normal path — the monitor must not be noise."""
    import logging

    now = datetime.now(UTC)
    with caplog.at_level(logging.WARNING):
        await _run_batched_decay([_decay_trace(last_accessed_at=now - timedelta(days=2))], now)

    assert not [r for r in caplog.records if "access_count_without_last_accessed_at" in r.getMessage()]


@pytest.mark.asyncio
async def test_decay_anchors_on_last_access_not_creation():
    """DB-437: retrieval resets the forgetting curve.

    A trace created 60d ago but retrieved yesterday must decay from YESTERDAY,
    not from creation — otherwise a memory the read path keeps fetching still
    falls under the prune line and gets hard-deleted (prod, 2026-07-31).
    """
    now = datetime.now(UTC)
    accessed = _decay_trace(created_at=now - timedelta(days=60), last_accessed_at=now - timedelta(days=1))
    silent = _decay_trace(created_at=now - timedelta(days=60), last_accessed_at=now - timedelta(days=60))

    params = await _run_batched_decay([accessed, silent], now)
    by_id = dict(zip(params["ids"], params["saliences"], strict=True))

    effective_lambda = BASE_DECAY * (6 - 3) / 5
    assert by_id[accessed.id] == pytest.approx(0.5 * math.exp(-effective_lambda * 1), rel=1e-6)
    assert by_id[silent.id] == pytest.approx(0.5 * math.exp(-effective_lambda * 60), rel=1e-6)
    assert by_id[accessed.id] > by_id[silent.id]


@pytest.mark.asyncio
async def test_decay_anchor_falls_back_to_created_at_when_never_accessed():
    """No ``last_accessed_at`` → unchanged pre-DB-437 behaviour (creation anchor).

    Also pins the monotonic guard: a ``last_accessed_at`` BEHIND ``created_at``
    (clock skew / backfill) must not rewind the clock and inflate salience.
    """
    now = datetime.now(UTC)
    never = _decay_trace(created_at=now - timedelta(days=30), last_accessed_at=None)
    skewed = _decay_trace(created_at=now - timedelta(days=30), last_accessed_at=now - timedelta(days=90))

    params = await _run_batched_decay([never, skewed], now)
    by_id = dict(zip(params["ids"], params["saliences"], strict=True))

    expected = 0.5 * math.exp(-(BASE_DECAY * (6 - 3) / 5) * 30)
    assert by_id[never.id] == pytest.approx(expected, rel=1e-6)
    assert by_id[skewed.id] == pytest.approx(expected, rel=1e-6)


@pytest.mark.asyncio
async def test_decay_anchor_never_lifts_durable_floor_or_initial_cap():
    """The new anchor must not change the two existing clamps."""
    now = datetime.now(UTC)
    durable = _decay_trace(
        created_at=now - timedelta(days=200),
        last_accessed_at=now - timedelta(days=200),
        typed_metadata={"durability": "durable"},
    )
    fresh = _decay_trace(created_at=now, last_accessed_at=now)  # zero elapsed → capped at initial

    params = await _run_batched_decay([durable, fresh], now)
    by_id = dict(zip(params["ids"], params["saliences"], strict=True))

    from kai_memory.consolidation.consolidation import DURABLE_SALIENCE_FLOOR

    assert by_id[durable.id] == pytest.approx(DURABLE_SALIENCE_FLOOR)
    assert by_id[fresh.id] == pytest.approx(0.5)  # min(initial_salience, ...) still holds


def test_decay_significance_formula():
    """Verify the decay formula for different significance levels."""
    days = 7

    for sig in [1, 2, 3, 4]:
        effective_lambda = BASE_DECAY * (6 - sig) / 5
        initial = 0.6
        result = initial * math.exp(-effective_lambda * days)
        assert 0 < result < initial, f"sig={sig} should decay from {initial}"

    # sig=1 should decay fastest
    lambda_1 = BASE_DECAY * (6 - 1) / 5
    lambda_4 = BASE_DECAY * (6 - 4) / 5
    assert lambda_1 > lambda_4


def test_spaced_access_bonus_no_timestamps():
    assert _compute_spaced_access_bonus([]) == 0.0
    assert _compute_spaced_access_bonus(["2026-04-01T10:00:00+00:00"]) == 0.0


def test_spaced_access_bonus_clustered():
    """Clustered accesses (same day) give near-zero bonus."""
    base = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    timestamps = [(base + timedelta(minutes=i * 5)).isoformat() for i in range(5)]
    bonus = _compute_spaced_access_bonus(timestamps)
    assert bonus < 0.01


def test_spaced_access_bonus_spaced():
    """Well-spaced accesses (days apart) give significant bonus."""
    base = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    timestamps = [(base + timedelta(days=i * 2)).isoformat() for i in range(5)]
    bonus = _compute_spaced_access_bonus(timestamps)
    assert bonus > 0.05


def test_spaced_access_bonus_capped():
    """Bonus should not exceed 0.2 (20%)."""
    base = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    timestamps = [(base + timedelta(days=i * 30)).isoformat() for i in range(5)]
    bonus = _compute_spaced_access_bonus(timestamps)
    assert bonus <= 0.2


def test_spaced_access_bonus_bad_data():
    """Invalid timestamps should return 0."""
    assert _compute_spaced_access_bonus(["not-a-date", "also-bad"]) == 0.0


# ── Process new traces ────────────────────────────────────


@pytest.mark.asyncio
async def test_process_new_traces_counts():
    session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar.return_value = 42
    session.execute = AsyncMock(return_value=mock_result)

    count = await _process_new_traces(session)
    assert count == 42


# ── Advisory lock / start_run ─────────────────────────────


@pytest.mark.asyncio
async def test_start_run_advisory_lock():
    """pg_try_advisory_lock should be called."""
    from kai_memory.consolidation.consolidation import _start_run

    session = AsyncMock()
    # Lock acquired
    lock_result = MagicMock()
    lock_result.scalar.return_value = True
    # Stale run cleanup (rowcount)
    cleanup_result = MagicMock()
    cleanup_result.rowcount = 0
    session.execute = AsyncMock(side_effect=[lock_result, cleanup_result])
    session.flush = AsyncMock()
    session.add = MagicMock()

    run = await _start_run(session)
    assert run is not None
    # First call should be the advisory lock
    first_call_sql = str(session.execute.call_args_list[0][0][0])
    assert "pg_try_advisory_lock" in first_call_sql


@pytest.mark.asyncio
async def test_consolidation_concurrent_prevention():
    """Second call should return None when lock not acquired."""
    from kai_memory.consolidation.consolidation import _start_run

    session = AsyncMock()
    lock_result = MagicMock()
    lock_result.scalar.return_value = False  # Lock NOT acquired
    session.execute = AsyncMock(return_value=lock_result)

    run = await _start_run(session)
    assert run is None


@pytest.mark.asyncio
async def test_consolidation_stale_run_cleanup():
    """Stale running records (>30 min) should be marked failed."""
    from kai_memory.consolidation.consolidation import _start_run

    session = AsyncMock()
    lock_result = MagicMock()
    lock_result.scalar.return_value = True
    cleanup_result = MagicMock()
    cleanup_result.rowcount = 1  # One stale run cleaned
    session.execute = AsyncMock(side_effect=[lock_result, cleanup_result])
    session.flush = AsyncMock()
    session.add = MagicMock()

    run = await _start_run(session)
    assert run is not None


# ── Complete / Fail run ───────────────────────────────────


@pytest.mark.asyncio
async def test_complete_run_stores_stats():
    from kai_memory.consolidation.consolidation import _complete_run

    session = AsyncMock()
    session.execute = AsyncMock()
    run_id = uuid.uuid4()
    stats = {"new_traces": 5, "decayed": 3}

    await _complete_run(session, run_id, stats)
    assert session.execute.await_count == 2  # UPDATE + advisory_unlock


@pytest.mark.asyncio
async def test_fail_run_stores_error():
    from kai_memory.consolidation.consolidation import _fail_run

    session = AsyncMock()
    session.execute = AsyncMock()
    run_id = uuid.uuid4()

    await _fail_run(session, run_id, "something broke")
    assert session.execute.await_count == 2  # UPDATE + advisory_unlock


# ── Update relationships ──────────────────────────────────


@pytest.mark.asyncio
async def test_update_relationships_invalidated_source():
    session = AsyncMock()
    result_mock = MagicMock()
    result_mock.rowcount = 2
    session.execute = AsyncMock(return_value=result_mock)

    count = await _update_relationships(session)
    assert count == 2


# ── Reassess significance ─────────────────────────────────


@pytest.mark.asyncio
async def test_reassess_significance_upgrades():
    """Traces with 5+ accesses and sig<4 should be upgraded."""
    session = AsyncMock()
    session.execute = AsyncMock()

    await _reassess_significance(session)

    call_args = session.execute.call_args
    stmt = call_args[0][0]
    compiled = stmt.compile(compile_kwargs={"literal_binds": True})
    sql_str = str(compiled)
    assert "access_count" in sql_str
    assert "significance" in sql_str


# ── Full consolidation ────────────────────────────────────


@pytest.mark.asyncio
async def test_consolidation_idempotent():
    """Running consolidation twice should work — second pass finds less."""
    # This is more of a behavioral property than a strict unit test.
    # Verify the function handles zero-work gracefully.
    with (
        patch("kai_memory.consolidation.consolidation.session_scope") as mock_scope,
        patch("kai_memory.consolidation.consolidation._start_run", return_value=None),
    ):
        mock_session = AsyncMock()
        mock_scope.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_scope.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await consolidate()
    assert result["skipped"] is True


# ── Sonnet step tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_detect_contradictions_no_new_traces():
    from kai_memory.consolidation.consolidation import _detect_contradictions

    session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value = MagicMock(all=MagicMock(return_value=[]))
    session.execute = AsyncMock(return_value=mock_result)

    count = await _detect_contradictions(session)
    assert count == 0


@pytest.mark.asyncio
async def test_self_reflect_creates_traces():
    from kai_memory.consolidation.consolidation import _self_reflect

    session = AsyncMock()
    trace = MagicMock()
    trace.created_at = datetime.now(UTC)
    trace.memory_type = "fact"
    trace.content = "test memory"
    trace.invalidated_at = None

    mock_result = MagicMock()
    mock_result.scalars.return_value = MagicMock(all=MagicMock(return_value=[trace]))
    session.execute = AsyncMock(return_value=mock_result)

    reflection_items = [
        {"dimension": "interaction_quality", "observation": "People respond well", "significance": 3},
        {"dimension": "behavioral_patterns", "observation": "Good at connecting topics", "significance": 3},
        {"dimension": "per_person_calibration", "observation": "Misha prefers direct", "significance": 3},
        {"dimension": "goal_alignment", "observation": "Missing calendar access", "significance": 4},
        {"dimension": "social_comfort", "observation": "Comfortable in family group", "significance": 3},
    ]

    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()

    with (
        patch("kai_memory.consolidation.consolidation._call_sonnet", return_value=reflection_items),
        patch("kai_memory.consolidation.consolidation.CognitiveMemoryService") as mock_svc_cls,
    ):
        mock_svc = AsyncMock()
        mock_svc.store_trace = AsyncMock(return_value=mock_trace)
        mock_svc_cls.return_value = mock_svc

        count = await _self_reflect(session)

    assert count == 5
    assert mock_svc.store_trace.await_count == 5
    # Verify memory_type is self_reflection
    first_call = mock_svc.store_trace.call_args_list[0]
    assert first_call.kwargs["memory_type"] == "self_reflection"


def test_runaway_weight_adjustment_removed():
    """Regression: the ``weights_adjusted`` step must never come back.

    ``_adjust_retrieval_weights`` was a positive-feedback runaway — it nudged graph
    +0.01 whenever frequently_accessed > 10 (always true in practice) then
    renormalized (g' = (g + 0.01) / 1.01), so ~267 nightly passes drove graph
    0.25 → 0.945. The 2026-07-02 prod sweep measured the designed defaults beating
    the drifted mix (recall@8 0.75 vs 0.60); the step was deleted and migration 091
    reset the weights. Weight changes are operator decisions now.
    """
    import kai_memory.consolidation.consolidation as consolidation_mod

    src = inspect.getsource(consolidation_mod)
    assert "_adjust_retrieval_weights" not in src
    assert "weights_adjusted" not in src


@pytest.mark.asyncio
async def test_promote_patterns_no_cooccurrences():
    from kai_memory.consolidation.consolidation import _promote_patterns

    session = AsyncMock()
    mock_result = MagicMock()
    mock_result.__iter__ = MagicMock(return_value=iter([]))
    session.execute = AsyncMock(return_value=mock_result)

    count = await _promote_patterns(session)
    assert count == 0


@pytest.mark.asyncio
async def test_promote_patterns_does_not_count_below_floor_rejection():
    from kai_memory.consolidation.consolidation import _promote_patterns

    cognitive_config.set_for_testing("relationship_min_confidence", 0.5)
    try:
        session = AsyncMock()
        session.add = MagicMock()
        pair = _attach_pair_entities(
            {
                "a_id": uuid.uuid4(),
                "b_id": uuid.uuid4(),
                "name_a": "Corfu",
                "name_b": "Cyprus",
                "shared_contents": ["Misha lives in Corfu and registered a company in Cyprus"],
                "earliest_date": datetime.now(UTC),
                "existing_rels_a": [],
                "existing_rels_b": [],
            },
            a_type="place",
            b_type="place",
        )

        with (
            patch("kai_memory.consolidation.consolidation._find_cooccurring_entity_pairs", new=AsyncMock(return_value=[pair])),
            patch("kai_memory.consolidation.consolidation._current_consolidation_run_id", new=AsyncMock(return_value=None)),
            patch(
                "kai_memory.consolidation.consolidation._classify_pair_relationship",
                new=AsyncMock(
                    return_value={
                        "relationship_type": "located_in",
                        "confidence": 0.0,
                        "conflict_action": "none",
                    }
                ),
            ),
        ):
            count = await _promote_patterns(session)

        assert count == 0
        session.add.assert_not_called()
        session.flush.assert_not_awaited()
    finally:
        cognitive_config.reset_cache()


@pytest.mark.asyncio
async def test_promote_patterns_does_not_count_same_person_merge_candidate():
    from kai_memory.consolidation.consolidation import _promote_patterns

    session = AsyncMock()
    session.add = MagicMock()
    pair = _attach_pair_entities(
        {
            "a_id": uuid.uuid4(),
            "b_id": uuid.uuid4(),
            "name_a": "Misha",
            "name_b": "Misha Petrov",
            "shared_contents": ["Misha and Misha Petrov refer to the same person"],
            "earliest_date": datetime.now(UTC),
            "existing_rels_a": [],
            "existing_rels_b": [],
            "source_consolidation_run_id": uuid.uuid4(),
        },
    )

    with (
        patch("kai_memory.consolidation.consolidation._find_cooccurring_entity_pairs", new=AsyncMock(return_value=[pair])),
        patch("kai_memory.consolidation.consolidation._current_consolidation_run_id", new=AsyncMock(return_value=None)),
        patch(
            "kai_memory.consolidation.consolidation._classify_pair_relationship",
            new=AsyncMock(
                return_value={
                    "relationship_type": "same_person",
                    "confidence": 0.95,
                    "conflict_action": "same_person",
                }
            ),
        ),
    ):
        count = await _promote_patterns(session)

    assert count == 0
    added = session.add.call_args.args[0]
    assert isinstance(added, EntityMergeCandidate)


@pytest.mark.asyncio
async def test_classify_pair_includes_existing_relationships():
    rel_id = uuid.uuid4()
    pair = {
        "a_id": uuid.uuid4(),
        "b_id": uuid.uuid4(),
        "name_a": "Pavel",
        "name_b": "Katya",
        "shared_contents": ["Pavel and Katya are together"],
        "earliest_date": datetime.now(UTC),
        "existing_rels_a": [{"type": "partner_of", "other_name": "Natasha", "confidence": 0.88, "id": str(rel_id)}],
        "existing_rels_b": [],
    }

    with patch(
        "kai_memory.consolidation.consolidation._call_sonnet_text",
        new=AsyncMock(
            return_value=json.dumps(
                {
                    "relationship_type": "partner_of",
                    "confidence": 0.99,
                    "conflict_action": "replace",
                    "conflicts_with": [str(rel_id)],
                }
            )
        ),
    ) as mock_sonnet:
        result = await _classify_pair_relationship(pair)

    assert result["conflict_action"] == "replace"
    prompt_arg = mock_sonnet.call_args.args[0]
    assert "Existing relationships for A:" in prompt_arg
    assert "partner_of -> Natasha" in prompt_arg
    assert "Existing relationships for B:" in prompt_arg
    assert "(none)" in prompt_arg


@pytest.mark.asyncio
async def test_create_relationship_conflict_action_none_inserts():
    session = AsyncMock()
    session.add = MagicMock()
    pair = _attach_pair_entities({"a_id": uuid.uuid4(), "b_id": uuid.uuid4(), "earliest_date": datetime.now(UTC)})

    created = await _create_relationship_from_classification(
        session,
        pair,
        {"relationship_type": "friend_of", "confidence": 0.7, "conflict_action": "none"},
    )

    assert created is True
    added = session.add.call_args.args[0]
    assert isinstance(added, EntityRelationship)
    assert added.relationship_type == "friend_of"
    session.execute.assert_not_called()


@pytest.mark.asyncio
async def test_create_relationship_below_confidence_floor_rejected(caplog: pytest.LogCaptureFixture):
    cognitive_config.set_for_testing("relationship_min_confidence", 0.5)
    try:
        session = AsyncMock()
        session.add = MagicMock()
        pair = _attach_pair_entities(
            {
                "a_id": uuid.uuid4(),
                "b_id": uuid.uuid4(),
                "name_a": "Corfu",
                "name_b": "Cyprus",
                "earliest_date": datetime.now(UTC),
            },
            a_type="place",
            b_type="place",
        )

        with caplog.at_level("INFO", logger="app.services.consolidation"):
            created = await _create_relationship_from_classification(
                session,
                pair,
                {"relationship_type": "located_in", "confidence": 0.49, "conflict_action": "none"},
            )

        assert created is False
        session.add.assert_not_called()
        session.flush.assert_not_awaited()
        assert "below confidence floor" in caplog.text
        assert "Corfu" in caplog.text
        assert "Cyprus" in caplog.text
        assert "located_in" in caplog.text
        assert "0.490" in caplog.text
    finally:
        cognitive_config.reset_cache()


@pytest.mark.asyncio
async def test_create_relationship_at_confidence_floor_inserts():
    cognitive_config.set_for_testing("relationship_min_confidence", 0.5)
    try:
        session = AsyncMock()
        session.add = MagicMock()
        pair = _attach_pair_entities({"a_id": uuid.uuid4(), "b_id": uuid.uuid4(), "earliest_date": datetime.now(UTC)})

        created = await _create_relationship_from_classification(
            session,
            pair,
            {"relationship_type": "friend_of", "confidence": 0.5, "conflict_action": "none"},
        )

        assert created is True
        added = session.add.call_args.args[0]
        assert isinstance(added, EntityRelationship)
        assert added.confidence == 0.5
    finally:
        cognitive_config.reset_cache()


@pytest.mark.asyncio
async def test_create_relationship_conflict_action_replace_expires_then_inserts():
    session = AsyncMock()
    session.add = MagicMock()
    conflict_id = uuid.uuid4()
    pair = _attach_pair_entities(
        {
            "a_id": uuid.uuid4(),
            "b_id": uuid.uuid4(),
            "earliest_date": datetime.now(UTC),
            "existing_rels_a": [
                {"type": "partner_of", "other_name": "Natasha", "confidence": 0.8, "id": str(conflict_id)}
            ],
            "existing_rels_b": [],
        }
    )

    await _create_relationship_from_classification(
        session,
        pair,
        {
            "relationship_type": "partner_of",
            "confidence": 0.9,
            "conflict_action": "replace",
            "conflicts_with": [str(conflict_id)],
        },
    )

    assert session.execute.await_count == 1
    stmt = session.execute.call_args.args[0]
    compiled = stmt.compile(compile_kwargs={"literal_binds": True})
    assert "valid_until" in str(compiled)
    assert isinstance(session.add.call_args.args[0], EntityRelationship)


@pytest.mark.asyncio
async def test_create_relationship_conflict_action_same_person_creates_merge_candidate():
    session = AsyncMock()
    session.add = MagicMock()
    pair = _attach_pair_entities(
        {
            "a_id": uuid.uuid4(),
            "b_id": uuid.uuid4(),
            "earliest_date": datetime.now(UTC),
            "source_consolidation_run_id": uuid.uuid4(),
        }
    )

    created = await _create_relationship_from_classification(
        session,
        pair,
        {"relationship_type": "same_person", "confidence": 0.95, "conflict_action": "same_person"},
    )

    assert created is False
    added = session.add.call_args.args[0]
    assert isinstance(added, EntityMergeCandidate)
    assert added.status == "pending"
    assert added.reason == "L3 classified as same person"


@pytest.mark.asyncio
async def test_create_relationship_conflict_action_sequential_expires_at_valid_from():
    session = AsyncMock()
    session.add = MagicMock()
    conflict_id = uuid.uuid4()
    valid_from = datetime(2026, 4, 10, tzinfo=UTC)
    pair = _attach_pair_entities(
        {
            "a_id": uuid.uuid4(),
            "b_id": uuid.uuid4(),
            "earliest_date": valid_from,
            "existing_rels_a": [
                {"type": "partner_of", "other_name": "Natasha", "confidence": 0.8, "id": str(conflict_id)}
            ],
            "existing_rels_b": [],
        }
    )

    await _create_relationship_from_classification(
        session,
        pair,
        {
            "relationship_type": "partner_of",
            "confidence": 0.9,
            "conflict_action": "sequential",
            "conflicts_with": [str(conflict_id)],
        },
    )

    assert session.execute.await_count == 1
    stmt = session.execute.call_args.args[0]
    assert stmt.compile().params["valid_until"] == valid_from
    assert isinstance(session.add.call_args.args[0], EntityRelationship)


@pytest.mark.asyncio
async def test_conflict_scoping_drops_unrelated_ids():
    valid_id = uuid.uuid4()
    random_id = uuid.uuid4()
    valid_rel = EntityRelationship(
        id=valid_id,
        from_entity_id=uuid.uuid4(),
        to_entity_id=uuid.uuid4(),
        relationship_type="partner_of",
    )
    random_rel = EntityRelationship(
        id=random_id,
        from_entity_id=uuid.uuid4(),
        to_entity_id=uuid.uuid4(),
        relationship_type="partner_of",
    )

    class ConflictSession:
        def __init__(self):
            self.relationships = {valid_id: valid_rel, random_id: random_rel}
            self.added = []

        def add(self, rel):
            self.added.append(rel)

        async def flush(self):
            return None

        async def execute(self, stmt):
            params = stmt.compile().params
            ids = []
            for value in params.values():
                if isinstance(value, list):
                    ids.extend(value)
            valid_until = params["valid_until"]
            for rel_id in ids:
                self.relationships[rel_id].valid_until = valid_until
            return MagicMock()

    session = ConflictSession()
    pair = _attach_pair_entities(
        {
            "a_id": uuid.uuid4(),
            "b_id": uuid.uuid4(),
            "earliest_date": datetime.now(UTC),
            "existing_rels_a": [
                {"type": "partner_of", "other_name": "Natasha", "confidence": 0.8, "id": str(valid_id)}
            ],
            "existing_rels_b": [],
        }
    )

    await _create_relationship_from_classification(
        session,  # type: ignore[arg-type]
        pair,
        {
            "relationship_type": "partner_of",
            "confidence": 0.9,
            "conflict_action": "replace",
            "conflicts_with": [str(valid_id), str(random_id)],
        },
    )

    assert valid_rel.valid_until is not None
    assert random_rel.valid_until is None


@pytest.mark.asyncio
async def test_update_social_norms_no_contexts():
    from kai_memory.consolidation.consolidation import _update_social_norms

    session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value = MagicMock(all=MagicMock(return_value=[]))
    session.execute = AsyncMock(return_value=mock_result)

    count = await _update_social_norms(session)
    assert count == 0


@pytest.mark.asyncio
async def test_update_social_norms_below_threshold():
    """Contexts with fewer than threshold new experiences should be skipped."""
    from unittest.mock import patch as _patch

    from kai_memory.consolidation.consolidation import _update_social_norms

    mock_ctx = MagicMock()
    mock_ctx.id = uuid.uuid4()
    mock_ctx.social_norms = None
    mock_ctx.updated_at = None

    session = AsyncMock()
    call_count = {"n": 0}

    async def execute_side_effect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call: return contexts
            r = MagicMock()
            r.scalars.return_value = MagicMock(all=MagicMock(return_value=[mock_ctx]))
            return r
        # Second call: count of experiences = 2 (below threshold of 3)
        r = MagicMock()
        r.scalar = MagicMock(return_value=2)
        return r

    session.execute = AsyncMock(side_effect=execute_side_effect)

    with _patch("kai_memory.consolidation.consolidation.cognitive_config.get_int", return_value=3):
        count = await _update_social_norms(session)

    assert count == 0


@pytest.mark.asyncio
async def test_update_social_norms_threshold_reached():
    """Contexts with threshold+ new experiences should trigger synthesis."""
    from unittest.mock import patch as _patch

    from kai_memory.consolidation.consolidation import _update_social_norms

    mock_ctx = MagicMock()
    mock_ctx.id = uuid.uuid4()
    mock_ctx.social_norms = None
    mock_ctx.updated_at = None
    mock_ctx.context_name = "family"
    mock_ctx.context_type = "group"

    mock_exp = MagicMock()
    mock_exp.experience = "Olga asked for privacy"
    mock_exp.learned_norm = "Privacy matters in family chat"
    mock_exp.signal_type = "boundary"

    session = AsyncMock()
    call_count = {"n": 0}

    async def execute_side_effect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            r = MagicMock()
            r.scalars.return_value = MagicMock(all=MagicMock(return_value=[mock_ctx]))
            return r
        if call_count["n"] == 2:
            # Count query: 3 experiences (meets threshold)
            r = MagicMock()
            r.scalar = MagicMock(return_value=3)
            return r
        if call_count["n"] == 3:
            # Fetch experiences
            r = MagicMock()
            r.scalars.return_value = MagicMock(all=MagicMock(return_value=[mock_exp, mock_exp, mock_exp]))
            return r
        # Update query
        return MagicMock()

    session.execute = AsyncMock(side_effect=execute_side_effect)

    with (
        _patch("kai_memory.consolidation.consolidation.cognitive_config.get_int", return_value=3),
        _patch(
            "kai_memory.consolidation.consolidation._synthesize_norms_for_context",
            new=AsyncMock(return_value="Keep family chat light and respect privacy"),
        ),
    ):
        count = await _update_social_norms(session)

    assert count == 1


# ── Entity merge sweep tests ──────────────────────────────


def _merge_entity(name: str, mention_count: int, telegram_user_id: int | None = None, aliases: list[str] | None = None):
    return MemoryEntity(
        id=uuid.uuid4(),
        name=name,
        entity_type="person",
        mention_count=mention_count,
        telegram_user_id=telegram_user_id,
        aliases=aliases or [],
    )


def _merge_lock_result(*entities: MemoryEntity) -> MagicMock:
    result = MagicMock()
    result.scalars.return_value.all.return_value = list(entities)
    return result


class _AsyncSavepointContext:
    """Mock for `session.begin_nested()` — tracks whether it observed an
    exception so tests can assert the savepoint was used and rolled back
    automatically (no explicit savepoint.rollback() call should be needed)."""

    def __init__(self):
        self.entered = False
        self.exited_with_exc = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # Real SQLAlchemy savepoint context auto-rolls-back on exception and
        # propagates it. Mirror that here (return False = don't suppress).
        if exc_type is not None:
            self.exited_with_exc = True
        return False


@pytest.mark.asyncio
async def test_merge_sweep_processes_candidates_happy_path():
    survivor = _merge_entity("Elena", mention_count=10, aliases=["Natasha"])
    absorbed = _merge_entity("Natasha_old", mention_count=2, aliases=["Nata"])
    candidate = EntityMergeCandidate(
        id=uuid.uuid4(),
        entity_a_id=survivor.id,
        entity_b_id=absorbed.id,
        confidence=1.0,
        reason="same telegram_user_id",
        status="pending",
    )

    result = MagicMock()
    result.scalars.return_value.all.return_value = [candidate.id]
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[result, MagicMock(), _merge_lock_result(survivor, absorbed)])
    session.get = AsyncMock(return_value=candidate)
    session.begin_nested = MagicMock(return_value=_AsyncSavepointContext())

    with patch("kai_memory.consolidation.consolidation._dedup_and_transfer_relationships", new=AsyncMock()):
        count = await _merge_entities(session)

    assert count == 1
    assert absorbed.merged_into_id == survivor.id
    assert absorbed.merge_reason == "same telegram_user_id"
    assert "Natasha_old" in survivor.aliases
    assert "Nata" in survivor.aliases
    assert candidate.status == "merged"


def test_merge_entities_uses_savepoint_not_nested_session():
    source = inspect.getsource(_merge_entities)
    loop_body = source.split("for candidate_id in candidate_ids:", 1)[1]

    # Savepoint is the right pattern; nested session_scope was the deadlock root cause.
    assert "session.begin_nested" in source
    assert "session_scope()" not in loop_body
    # Explicit savepoint.rollback() inside the with-block double-finalizes in
    # SQLAlchemy 2.x (CLOSED savepoint can't be committed by __aexit__). The
    # context manager must own rollback; this guard catches re-introduction.
    assert "savepoint.rollback" not in loop_body
    # Skip-guard must run before begin_nested so we don't issue a SAVEPOINT +
    # RELEASE round-trip per skipped candidate.
    pre_savepoint, _, post_savepoint = loop_body.partition("session.begin_nested")
    assert "candidate.status != MergeCandidateStatus.PENDING" in pre_savepoint
    assert "candidate.status != MergeCandidateStatus.PENDING" not in post_savepoint


@pytest.mark.asyncio
async def test_merge_entities_savepoint_rollback_on_exception():
    candidate_ids = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]
    candidates = [
        EntityMergeCandidate(id=candidate_id, entity_a_id=uuid.uuid4(), entity_b_id=uuid.uuid4(), status="pending")
        for candidate_id in candidate_ids
    ]
    result = MagicMock()
    result.scalars.return_value.all.return_value = candidate_ids

    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    session.get = AsyncMock(
        side_effect=lambda _model, candidate_id: next(
            (candidate for candidate in candidates if candidate.id == candidate_id),
            None,
        )
    )
    savepoints = [_AsyncSavepointContext() for _ in candidate_ids]
    session.begin_nested = MagicMock(side_effect=savepoints)

    async def mark_merged(_session, candidate):
        if candidate.id == candidate_ids[1]:
            raise RuntimeError("merge failed")
        candidate.status = "merged"

    with patch("kai_memory.consolidation.consolidation._execute_merge", new=AsyncMock(side_effect=mark_merged)):
        count = await _merge_entities(session)

    assert count == 2
    assert candidates[0].status == "merged"
    assert candidates[1].status == "rejected"
    assert candidates[1].reviewed_at is not None
    assert candidates[2].status == "merged"
    # Savepoint context manager observed the exception (SQLAlchemy auto-rolls
    # back on __aexit__ — no explicit savepoint.rollback() call inside the body).
    assert savepoints[1].exited_with_exc is True
    assert savepoints[0].exited_with_exc is False
    assert savepoints[2].exited_with_exc is False
    assert await session.get(EntityMergeCandidate, candidate_ids[2]) is candidates[2]


def test_lock_entities_for_merge_sets_lock_timeout():
    source = inspect.getsource(_lock_entities_for_merge)

    assert "SET LOCAL" in source
    assert "lock_timeout" in source


@pytest.mark.asyncio
async def test_merge_survivor_prefers_tg_id_holder():
    higher_mentions = _merge_entity("Natasha", mention_count=20)
    tg_holder = _merge_entity("Elena", mention_count=2, telegram_user_id=123)
    candidate = EntityMergeCandidate(entity_a_id=higher_mentions.id, entity_b_id=tg_holder.id, reason="same person")
    session = AsyncMock()
    session.execute = AsyncMock(return_value=_merge_lock_result(higher_mentions, tg_holder))

    with patch("kai_memory.consolidation.consolidation._dedup_and_transfer_relationships", new=AsyncMock()):
        await _execute_merge(session, candidate)

    assert higher_mentions.merged_into_id == tg_holder.id
    assert tg_holder.telegram_user_id == 123


@pytest.mark.asyncio
async def test_merge_skipped_when_both_have_distinct_tg_ids():
    first = _merge_entity("Pavel", mention_count=20, telegram_user_id=100)
    second = _merge_entity("Pavel B", mention_count=2, telegram_user_id=200)
    candidate = EntityMergeCandidate(entity_a_id=first.id, entity_b_id=second.id, reason="same person")
    session = AsyncMock()
    session.execute = AsyncMock(return_value=_merge_lock_result(first, second))

    with patch("kai_memory.consolidation.consolidation._dedup_and_transfer_relationships", new=AsyncMock()):
        await _execute_merge(session, candidate)

    assert candidate.status == "rejected"
    assert "distinct telegram_user_ids" in candidate.reason
    assert first.merged_into_id is None
    assert second.merged_into_id is None


@pytest.mark.asyncio
async def test_resolve_merge_chain_broken_chain_returns_current():
    current = _merge_entity("Natasha", mention_count=1)
    current.merged_into_id = uuid.uuid4()
    session = AsyncMock()
    session.get = AsyncMock(return_value=None)

    result = await _resolve_merge_chain(session, current)

    assert result is current


@pytest.mark.asyncio
async def test_merge_marks_merged_into_id_before_transfer():
    survivor = _merge_entity("Elena", mention_count=10)
    absorbed = _merge_entity("Natasha", mention_count=1)
    candidate = EntityMergeCandidate(entity_a_id=survivor.id, entity_b_id=absorbed.id, reason="same person")
    session = AsyncMock()
    session.execute = AsyncMock(return_value=_merge_lock_result(survivor, absorbed))
    session.flush = AsyncMock()

    with patch(
        "kai_memory.consolidation.consolidation._dedup_and_transfer_relationships",
        new=AsyncMock(side_effect=RuntimeError("transfer failed")),
    ):
        with pytest.raises(RuntimeError, match="transfer failed"):
            await _execute_merge(session, candidate)

    assert absorbed.merged_into_id == survivor.id
    session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_dedup_relationships_expires_duplicate_active_row():
    survivor_id = uuid.uuid4()
    absorbed_id = uuid.uuid4()
    other_id = uuid.uuid4()
    absorbed_rel = EntityRelationship(
        id=uuid.uuid4(),
        from_entity_id=absorbed_id,
        to_entity_id=other_id,
        relationship_type="partner_of",
        confidence=0.8,
    )
    existing_rel = EntityRelationship(
        id=uuid.uuid4(),
        from_entity_id=survivor_id,
        to_entity_id=other_id,
        relationship_type="partner_of",
        confidence=0.9,
    )

    active_result = MagicMock()
    active_result.scalars.return_value.all.return_value = [absorbed_rel]
    existing_result = MagicMock()
    existing_result.all.return_value = [
        (existing_rel.relationship_type, existing_rel.from_entity_id, existing_rel.to_entity_id)
    ]
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[active_result, existing_result, MagicMock(), MagicMock()])

    await _dedup_and_transfer_relationships(session, survivor_id, absorbed_id)

    assert absorbed_rel.valid_until is not None


@pytest.mark.asyncio
async def test_dedup_and_transfer_uses_single_query():
    survivor_id = uuid.uuid4()
    absorbed_id = uuid.uuid4()
    first_other_id = uuid.uuid4()
    second_other_id = uuid.uuid4()
    active_rels = [
        EntityRelationship(
            id=uuid.uuid4(),
            from_entity_id=absorbed_id,
            to_entity_id=first_other_id,
            relationship_type="partner_of",
            confidence=0.8,
        ),
        EntityRelationship(
            id=uuid.uuid4(),
            from_entity_id=second_other_id,
            to_entity_id=absorbed_id,
            relationship_type="coworker_of",
            confidence=0.8,
        ),
    ]
    active_result = MagicMock()
    active_result.scalars.return_value.all.return_value = active_rels
    existing_result = MagicMock()
    existing_result.all.return_value = []
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[active_result, existing_result, MagicMock(), MagicMock()])

    await _dedup_and_transfer_relationships(session, survivor_id, absorbed_id)

    assert session.execute.await_count == 4
    assert active_rels[0].from_entity_id == survivor_id
    assert active_rels[1].to_entity_id == survivor_id


@pytest.mark.asyncio
async def test_phase_maintain_wires_merge_sweep():
    from kai_memory.consolidation.consolidation import _phase_maintain

    session = AsyncMock()
    stats: dict[str, object] = {}
    failures: list[str] = []
    seen_steps = []

    async def fake_run_step(step_name, func_, _stats, _failures, _session):
        seen_steps.append(step_name)

    with patch("kai_memory.consolidation.consolidation._run_step", new=fake_run_step):
        await _phase_maintain(session, stats, failures)

    assert "entities_merged" in seen_steps
    # Part C release must run, and BEFORE decay so a demoted trace decays the same night.
    assert "durable_released" in seen_steps
    assert seen_steps.index("durable_released") < seen_steps.index("decayed")


# ── Part C: _select_releasable_durable_traces (pure) ────────────────────────


def _durable_fact(slot, sid, durability="durable"):
    return {"slot": slot, "subject_id": sid, "durability": durability}


def _rel_trace(tid, days_ago, facts):
    return {"id": tid, "created_at": datetime.now(UTC) - timedelta(days=days_ago), "facts": facts}


def test_release_supersedes_older_single_slot_trace():
    """Two employer traces for one subject: the older is released, the newest kept."""
    old = _rel_trace("old", 100, [_durable_fact("employer", "S1")])
    new = _rel_trace("new", 1, [_durable_fact("employer", "S1")])
    assert _select_releasable_durable_traces([old, new]) == ["old"]


def test_release_keeps_lone_durable_trace():
    """A single (un-superseded) durable trace keeps its floor."""
    only = _rel_trace("only", 100, [_durable_fact("employer", "S1")])
    assert _select_releasable_durable_traces([only]) == []


def test_release_multi_fact_edge_keeps_partially_superseded_trace():
    """C-1/F-1: a trace whose employer is superseded but whose occupation is still
    current KEEPS its floor (release only when ALL durable facts are superseded)."""
    old = _rel_trace("old", 100, [_durable_fact("employer", "S1"), _durable_fact("occupation", "S1")])
    newer_employer = _rel_trace("emp2", 2, [_durable_fact("employer", "S1")])
    # occupation on `old` is still the newest → `old` must NOT be released.
    assert _select_releasable_durable_traces([old, newer_employer]) == []


def test_release_never_touches_durable_multi_slot():
    """MULTI durable slots (languages) accumulate — never superseded, never released."""
    a = _rel_trace("a", 100, [_durable_fact("languages", "S1")])
    b = _rel_trace("b", 1, [_durable_fact("languages", "S1")])
    assert _select_releasable_durable_traces([a, b]) == []


def test_release_never_touches_off_schema_durable():
    """A durable off-schema fact is retained field-keyed and never supersede-released."""
    a = _rel_trace("a", 100, [_durable_fact("off_schema", "S1")])
    b = _rel_trace("b", 1, [_durable_fact("off_schema", "S1")])
    assert _select_releasable_durable_traces([a, b]) == []


def test_release_scopes_supersession_per_subject():
    """Same slot, different subjects → independent; neither is superseded."""
    s1 = _rel_trace("s1", 100, [_durable_fact("employer", "S1")])
    s2 = _rel_trace("s2", 1, [_durable_fact("employer", "S2")])
    assert _select_releasable_durable_traces([s1, s2]) == []


def test_release_ignores_transient_facts_on_the_trace():
    """A superseded single durable fact + a transient fact → still releasable (transient
    facts don't hold a floor)."""
    old = _rel_trace("old", 100, [_durable_fact("employer", "S1"), _durable_fact("mood", "S1", "transient")])
    new = _rel_trace("new", 1, [_durable_fact("employer", "S1")])
    assert _select_releasable_durable_traces([old, new]) == ["old"]


# ── Pruning step tests ────────────────────────────────────


@pytest.mark.asyncio
async def test_prune_silent_traces():
    from kai_memory.consolidation.consolidation import _prune_silent_traces

    session = AsyncMock()
    count_result = MagicMock()
    count_result.scalar.return_value = 3
    cleanup_result = MagicMock()
    session.execute = AsyncMock(side_effect=[count_result, cleanup_result])

    count = await _prune_silent_traces(session)
    assert count == 3
    # Verify the DELETE SQL uses salience/significance guards with bind params
    delete_call = session.execute.call_args_list[0]
    sql_text = str(delete_call[0][0])
    params = delete_call[0][1]
    assert "salience < :threshold" in sql_text
    assert "significance < 5" in sql_text
    assert "days" in sql_text
    assert params["threshold"] == 0.01
    assert params["age_days"] == 30
    # DB-437: prunable = old AND silent. The recent-access guard reuses :age_days
    # (no new tunable), so a trace retrieved inside the silence window is spared.
    assert "last_accessed_at IS NULL" in sql_text
    assert "last_accessed_at < CAST(:now AS timestamptz) - make_interval(days => :age_days)" in sql_text


@pytest.mark.asyncio
async def test_cleanup_old_runs():
    from kai_memory.consolidation.consolidation import _cleanup_old_runs

    session = AsyncMock()
    result_mock = MagicMock()
    result_mock.rowcount = 5
    session.execute = AsyncMock(return_value=result_mock)

    count = await _cleanup_old_runs(session)
    assert count == 5
    call = session.execute.call_args
    params = call[0][1]
    assert params["days"] == 90


@pytest.mark.asyncio
async def test_cap_access_timestamps():
    from kai_memory.consolidation.consolidation import _cap_access_timestamps

    session = AsyncMock()
    result_mock = MagicMock()
    result_mock.rowcount = 2
    session.execute = AsyncMock(return_value=result_mock)

    count = await _cap_access_timestamps(session)
    assert count == 2
    call = session.execute.call_args
    sql_text = str(call[0][0])
    params = call[0][1]
    assert "jsonb_array_length" in sql_text
    assert params["cap"] == 20


def test_consolidation_rewrite_model_defaults_to_opus(monkeypatch):
    """Guardrail: if the default flips to a non-Opus model the safety margin
    this PR was built on disappears silently. Env var override is still
    allowed; this test pins the *default* only.

    Explicitly clears CONSOLIDATION_REWRITE_MODEL from the environment so
    a CI or dev shell override can't mask a bad default.
    """
    from kai_memory import MemorySettings as Settings

    monkeypatch.delenv("CONSOLIDATION_REWRITE_MODEL", raising=False)
    default_model = Settings().consolidation_rewrite_model
    assert default_model.startswith("opus-"), (
        f"consolidation_rewrite_model default must resolve to an Opus model, got {default_model!r}"
    )


@pytest.mark.parametrize("bound_first", [False, True])
async def test_merge_keeps_bound_agent_uuid_despite_lower_mentions(bound_first):
    from kai_memory.identity import AGENT_IDENTITY_DATA_KEY

    bound = _merge_entity("Atlas", mention_count=1)
    bound.entity_type = "agent"
    bound.data = {AGENT_IDENTITY_DATA_KEY: "host-agent", "role": "assistant"}
    duplicate = _merge_entity("Atlas Assistant", mention_count=100)
    duplicate.entity_type = "agent"
    duplicate.data = {"fact": "preserved"}
    first, second = (bound, duplicate) if bound_first else (duplicate, bound)
    candidate = EntityMergeCandidate(entity_a_id=first.id, entity_b_id=second.id, reason="same agent")
    session = AsyncMock()
    session.execute = AsyncMock(return_value=_merge_lock_result(first, second))
    transfer = AsyncMock()
    with patch("kai_memory.consolidation.consolidation._dedup_and_transfer_relationships", new=transfer):
        await _execute_merge(session, candidate)

    assert candidate.status == "merged"
    assert bound.merged_into_id is None and bound.name == "Atlas"
    assert duplicate.merged_into_id == bound.id
    assert bound.data == {AGENT_IDENTITY_DATA_KEY: "host-agent", "role": "assistant"}
    assert duplicate.data == {"fact": "preserved"}
    transfer.assert_awaited_once_with(session, bound.id, duplicate.id)


@pytest.mark.parametrize("second_binding", ["host-agent", "other-agent"])
async def test_merge_rejects_any_two_bound_uuids_without_entity_mutation(second_binding):
    from kai_memory.identity import AGENT_IDENTITY_DATA_KEY

    first = _merge_entity("Atlas", mention_count=1, aliases=["A"])
    second = _merge_entity("Orion", mention_count=100, aliases=["O"])
    first.entity_type = second.entity_type = "agent"
    first.data = {AGENT_IDENTITY_DATA_KEY: "host-agent"}
    second.data = {AGENT_IDENTITY_DATA_KEY: second_binding}
    candidate = EntityMergeCandidate(entity_a_id=first.id, entity_b_id=second.id, reason="same agent")
    session = AsyncMock()
    session.execute = AsyncMock(return_value=_merge_lock_result(first, second))
    transfer = AsyncMock()
    with patch("kai_memory.consolidation.consolidation._dedup_and_transfer_relationships", new=transfer):
        await _execute_merge(session, candidate)

    assert candidate.status == "rejected" and "two bound agent identities" in candidate.reason
    assert first.merged_into_id is second.merged_into_id is None
    assert first.aliases == ["A"] and second.aliases == ["O"]
    assert first.data == {AGENT_IDENTITY_DATA_KEY: "host-agent"}
    assert second.data == {AGENT_IDENTITY_DATA_KEY: second_binding}
    session.flush.assert_not_awaited()
    transfer.assert_not_awaited()


@pytest.mark.parametrize("telegram_user_id", [0, 42])
async def test_merge_rejects_bound_agent_and_verified_human(telegram_user_id):
    from kai_memory.identity import AGENT_IDENTITY_DATA_KEY

    bound = _merge_entity("Atlas", mention_count=1)
    bound.entity_type = "agent"
    bound.data = {AGENT_IDENTITY_DATA_KEY: "host-agent"}
    human = _merge_entity("Human", mention_count=100, telegram_user_id=telegram_user_id)
    candidate = EntityMergeCandidate(entity_a_id=human.id, entity_b_id=bound.id, reason="same entity")
    session = AsyncMock()
    session.execute = AsyncMock(return_value=_merge_lock_result(human, bound))
    transfer = AsyncMock()
    with patch("kai_memory.consolidation.consolidation._dedup_and_transfer_relationships", new=transfer):
        await _execute_merge(session, candidate)

    assert candidate.status == "rejected" and "Telegram user" in candidate.reason
    assert bound.merged_into_id is human.merged_into_id is None
    assert bound.aliases == human.aliases == []
    assert human.telegram_user_id == telegram_user_id
    session.flush.assert_not_awaited()
    transfer.assert_not_awaited()


async def test_unbound_agent_merge_retains_mention_count_survivor():
    first = _merge_entity("Atlas", mention_count=1)
    second = _merge_entity("Atlas Assistant", mention_count=100)
    first.entity_type = second.entity_type = "agent"
    candidate = EntityMergeCandidate(entity_a_id=first.id, entity_b_id=second.id, reason="same agent")
    session = AsyncMock()
    session.execute = AsyncMock(return_value=_merge_lock_result(first, second))
    transfer = AsyncMock()
    with patch("kai_memory.consolidation.consolidation._dedup_and_transfer_relationships", new=transfer):
        await _execute_merge(session, candidate)
    assert candidate.status == "merged" and first.merged_into_id == second.id
    transfer.assert_awaited_once_with(session, second.id, first.id)
