"""Compiled-SQL guards for direct distance ordering and HNSW eligibility."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.dialects import postgresql

from kai_memory.store.cognitive_memory import HNSW_EF_SEARCH, CognitiveMemoryService


def _mock_session() -> AsyncMock:
    session = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = []
    mock_result.tuples.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=mock_result)
    return session


async def _run_search(method: str, session: AsyncMock) -> None:
    svc = CognitiveMemoryService(session)
    if method == "search_traces_by_vector":
        await svc.search_traces_by_vector([0.1] * 1536, limit=5)
        return
    with patch("kai_memory.store.cognitive_memory.generate_embedding", return_value=[0.1] * 1536):
        await getattr(svc, method)("query", limit=5)


def _compiled_search_sql(session: AsyncMock) -> str:
    # Execute order is pin-plan (0), search SELECT (1), restore-defaults (2).
    stmt = session.execute.call_args_list[1].args[0]
    return str(stmt.compile(dialect=postgresql.dialect()))


SEARCH_METHODS = ["search_traces", "search_traces_raw", "search_traces_by_vector"]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", SEARCH_METHODS)
async def test_order_by_is_bare_cosine_distance_operator(method):
    """ORDER BY must be the raw ``<=>`` — the only form the HNSW index serves."""
    session = _mock_session()
    await _run_search(method, session)

    sql = _compiled_search_sql(session)
    assert "ORDER BY" in sql
    order_by = sql.split("ORDER BY", 1)[1]
    assert "<=>" in order_by, f"ORDER BY lost the bare distance operator:\n{sql}"
    # Any wrapping of the distance in ORDER BY — the `1 - …` subtraction
    # (compiled as a bind param: `%(param_1)s - (…)`) or the similarity label —
    # makes the index ineligible → seq scan over every vector.
    assert " - " not in order_by, f"ORDER BY wraps the distance (index-ineligible):\n{sql}"
    assert "similarity" not in order_by, f"ORDER BY references the label (index-ineligible):\n{sql}"


@pytest.mark.asyncio
@pytest.mark.parametrize("method", SEARCH_METHODS)
async def test_similarity_stays_in_select_list(method):
    """Downstream scoring reads ``similarity`` — the wrapper must survive in
    the SELECT list even though the ORDER BY no longer uses it."""
    session = _mock_session()
    await _run_search(method, session)

    sql = _compiled_search_sql(session)
    select_list = sql.split("ORDER BY", 1)[0]
    assert "AS similarity" in select_list
    # The `1 - distance` wrapper compiles as `%(param_1)s - (… <=> …)`.
    assert " - (" in select_list


@pytest.mark.asyncio
@pytest.mark.parametrize("method", SEARCH_METHODS)
async def test_search_plan_pinned_then_restored(method):
    """Each search must pin the plan first (SET LOCAL semantics via
    set_config: hnsw.ef_search recall budget + enable_seqscan/enable_sort off —
    prod's cost model otherwise PREFERS the 5s seq scan / Bitmap+Sort over the
    ~ms idx_mt_embedding scan), then restore the planner toggles right after
    the SELECT so the rest of the transaction (priming Paths 2-4, prompt
    loaders) keeps default plans."""
    session = _mock_session()
    await _run_search(method, session)

    assert session.execute.await_count == 3  # pin, search, restore
    pin_stmt = session.execute.call_args_list[0].args[0]
    pin = str(pin_stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
    assert "hnsw.ef_search" in pin
    assert f"'{HNSW_EF_SEARCH}'" in pin  # bound budget value
    assert "set_config('enable_seqscan', 'off', true)" in pin
    assert "set_config('enable_sort', 'off', true)" in pin

    restore = str(session.execute.call_args_list[2].args[0])
    assert "set_config('enable_seqscan', 'on', true)" in restore
    assert "set_config('enable_sort', 'on', true)" in restore
    assert "hnsw.ef_search" not in restore  # ef_search stays pinned; harmless


@pytest.mark.asyncio
async def test_memory_type_filter_preserved_with_bare_ordering():
    session = _mock_session()
    with patch("kai_memory.store.cognitive_memory.generate_embedding", return_value=[0.1] * 1536):
        svc = CognitiveMemoryService(session)
        await svc.search_traces_raw("query", limit=5, memory_types=["fact"])

    sql = _compiled_search_sql(session)
    # Both the servable allowlist and the caller filter render as IN clauses.
    assert sql.count("memory_traces.memory_type IN") == 2
    assert "<=>" in sql.split("ORDER BY", 1)[1]
