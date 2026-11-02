"""Adapter unit tests — pure-Python protocol compliance + delegation smoke.

The full equivalence contract (storage round-trip, retrieval parity,
artifact iteration shape) lives in
`tests/integration/test_memory_adapter_contract.py`, which needs the
pgvector test container. These unit tests verify the parts that don't
need a real DB: Protocol structural compliance, delegation wiring, and
the empty-iterator semantics for export_artifacts during M1.

The adapter is imported from its submodule directly to avoid a circular
import — the package `__init__.py` only re-exports the type surface.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai_memory import (
    ArtifactQuery,
    ConsolidationTrigger,
    Cue,
    EncodingSignals,
    Experience,
    FeedbackSignal,
    FeedbackTarget,
    MemoryInterface,
    RetrievalContext,
)
from kai_memory.adapter import CognitiveMemoryAdapter


def test_adapter_satisfies_memory_interface_protocol():
    """Structural Protocol check via runtime_checkable."""
    session = MagicMock()
    adapter = CognitiveMemoryAdapter(session)
    assert isinstance(adapter, MemoryInterface)


@pytest.mark.asyncio
async def test_export_artifacts_returns_empty_async_iterator():
    """v2a §10: export_artifacts yields nothing in M1 (no producers)."""
    session = MagicMock()
    adapter = CognitiveMemoryAdapter(session)
    items = [a async for a in adapter.export_artifacts(ArtifactQuery())]
    assert items == []


@pytest.mark.asyncio
async def test_encode_delegates_to_store_trace_and_links():
    """encode() calls store_trace + link_trace_to_entity per entity_link."""
    session = MagicMock()
    adapter = CognitiveMemoryAdapter(session)

    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()
    entity_a = uuid.uuid4()
    entity_b = uuid.uuid4()

    with (
        patch.object(adapter.memory_svc, "store_trace", AsyncMock(return_value=mock_trace)) as st,
        patch.object(adapter.memory_svc, "link_trace_to_entity", AsyncMock()) as lk,
    ):
        result = await adapter.encode(
            Experience(
                chat_id=42,
                topic_id=None,
                venue="telegram",
                timestamp=datetime.now(UTC),
                memory_type="fact",
                content="Луна is the labrador",
                entity_links=[(entity_a, "subject"), (entity_b, "about")],
                significance=4,
            ),
            EncodingSignals(),
        )

    st.assert_awaited_once()
    assert lk.await_count == 2
    assert result.trace_ids == [mock_trace.id]
    assert set(result.entities_touched) == {entity_a, entity_b}
    assert result.contradictions_detected == 0
    assert result.mode == "defer"


@pytest.mark.asyncio
async def test_feedback_strong_correction_invalidates_trace():
    session = MagicMock()
    adapter = CognitiveMemoryAdapter(session)
    target_id = uuid.uuid4()
    source = uuid.uuid4()

    with patch.object(adapter.memory_svc, "invalidate_trace", AsyncMock()) as inv:
        out = await adapter.feedback(
            FeedbackTarget(kind="trace", id=target_id),
            FeedbackSignal(
                kind="strong_correction",
                evidence_text="that's not right",
                source_message_id=None,
                source_trace_id=source,
            ),
        )

    inv.assert_awaited_once()
    assert out.action == "invalidate"


@pytest.mark.asyncio
async def test_feedback_other_signals_noop_in_m1():
    session = MagicMock()
    adapter = CognitiveMemoryAdapter(session)

    with patch.object(adapter.memory_svc, "invalidate_trace", AsyncMock()) as inv:
        out = await adapter.feedback(
            FeedbackTarget(kind="trace", id=uuid.uuid4()),
            FeedbackSignal(
                kind="affirmation",
                evidence_text="yes exactly",
                source_message_id=None,
                source_trace_id=None,
            ),
        )

    inv.assert_not_awaited()
    assert out.action == "noop"


@pytest.mark.asyncio
async def test_retrieve_uses_prime_with_traces_no_extra_fetch():
    """review-r1 fix 4 + 5: retrieve() consumes prime_with_traces' ranked
    output directly, eliminating the get_traces_batch round-trip, AND
    populates TracePrime.score from the real scoring values.
    """
    session = MagicMock()
    adapter = CognitiveMemoryAdapter(session)

    trace = MagicMock()
    trace.id = uuid.uuid4()
    trace.content = "Луна is the labrador"
    trace.created_at = datetime.now(UTC)
    trace.memory_type = "fact"

    fake_components = {"semantic": 0.84, "graph": 0.0, "recency": 0.62, "access": 0.0, "arousal": 0.0}
    with (
        patch.object(
            adapter.priming_svc,
            "prime_with_traces",
            AsyncMock(return_value=("formatted", [trace.id], [(trace, 0.84, fake_components)])),
        ) as p,
        patch.object(adapter.memory_svc, "get_traces_batch", AsyncMock()) as gtb,
    ):
        result = await adapter.retrieve(
            [Cue(kind="semantic", value="дог Луна")],
            RetrievalContext(chat_id=1, topic_id=None, venue="telegram"),
        )

    p.assert_awaited_once()
    # No second round-trip — adapter consumed the trace from prime_with_traces.
    gtb.assert_not_awaited()

    assert len(result.primed_traces) == 1
    assert result.primed_traces[0].trace_id == trace.id
    assert result.primed_traces[0].score == 0.84  # real score, not 0.0
    # review-r2 fix 3: score_breakdown populated from the same components
    # dict the structured priming log emits.
    assert result.primed_traces[0].score_breakdown == fake_components
    assert result.relevant_artifacts == []
    assert result.total_candidates == 1


@pytest.mark.asyncio
async def test_consolidate_only_wires_nightly_in_m1():
    """consolidate(level=3, reason='nightly') invokes consolidation; others no-op."""
    session = MagicMock()
    adapter = CognitiveMemoryAdapter(session)

    with patch(
        "kai_memory.consolidation.consolidation.consolidate",
        AsyncMock(return_value={"new_traces": 0, "failures": []}),
    ) as cons:
        report = await adapter.consolidate(
            ConsolidationTrigger(level=3, venue="telegram", chat_id=None, topic_id=None, reason="nightly")
        )

    cons.assert_awaited_once()
    assert report.level == 3

    # Eager L2 not yet wired in M1 — call returns a clean empty report.
    with patch("kai_memory.consolidation.consolidation.consolidate", AsyncMock()) as cons2:
        report = await adapter.consolidate(
            ConsolidationTrigger(level=2, venue="telegram", chat_id=42, topic_id=None, reason="silence_timer")
        )

    cons2.assert_not_awaited()
    assert report.level == 2
    assert report.traces_created == 0
