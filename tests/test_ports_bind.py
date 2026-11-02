"""``ports.bind()``: the memory host contract's bootstrap (DB-843)."""

from __future__ import annotations

import asyncio

import pytest

from kai_memory import ports
from kai_memory.store.embeddings import EmbeddingError, generate_embedding, generate_embeddings_batch


@pytest.fixture(autouse=True)
def _clean_bindings(monkeypatch):
    # The test process imported memory already; each test starts and ends unbound.
    monkeypatch.setattr(ports, "_BOUND", {})


class _Embedder:
    name = "fake"

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str], *, api_key: str | None = None) -> list[list[float]]:
        self.calls.append(texts)
        return [list(self.vector) for _ in texts]


def test_bind_overrides_resolution_and_returns_previous_bindings(monkeypatch):
    fallback = object()
    monkeypatch.setattr(ports, "_DEFAULT_RESOLVER", lambda name: fallback if name == "scrub_secrets" else ports.UNBOUND)
    sentinel = object()
    previous = ports.bind(scrub_secrets=sentinel)
    assert previous == {"scrub_secrets": ports.UNBOUND}
    assert ports.scrub_secrets is sentinel
    again = ports.bind(scrub_secrets=ports.UNBOUND)
    assert again == {"scrub_secrets": sentinel}
    assert ports.scrub_secrets is fallback  # back to the explicit host resolver


def test_bind_refuses_names_outside_the_contract():
    with pytest.raises(KeyError, match="not memory host ports"):
        ports.bind(not_a_port=object())
    assert ports._BOUND == {}


def test_call_time_ports_follow_a_bind_made_after_memory_was_imported():
    """The embeddings port is read on every call, so binding it reaches memory that is
    already imported (unlike an eagerly imported name)."""
    embedder = _Embedder([0.5] * 1536)
    ports.bind(embedder=embedder)
    assert asyncio.run(generate_embedding("hello")) == [0.5] * 1536
    assert asyncio.run(generate_embeddings_batch(["a", "b"])) == [[0.5] * 1536, [0.5] * 1536]
    assert embedder.calls == [["hello"], ["a", "b"]]


def test_memory_keeps_the_zero_vector_guard_over_any_embedder():
    ports.bind(embedder=_Embedder([0.0] * 1536))
    with pytest.raises(EmbeddingError, match="zero vector"):
        asyncio.run(generate_embedding("hello"))


def test_a_bind_after_import_reaches_every_memory_module_and_a_rebind_replaces_it():
    """DB-854: no port is captured at import. Memory imported long ago still sees each bind."""
    from datetime import UTC, datetime

    from kai_memory.extraction import memory_extraction
    from kai_memory.retrieval import memory_priming
    from kai_memory.store import cognitive_memory

    first, second = datetime(2030, 1, 1, tzinfo=UTC), datetime(2031, 1, 1, tzinfo=UTC)
    ports.bind(clock=lambda: first, scrub_secrets=lambda text: "scrubbed")
    assert memory_extraction.utc_now() == first
    assert memory_priming.utc_now() == first
    assert cognitive_memory.scrub_secrets("token=abc") == "scrubbed"
    ports.bind(clock=lambda: second)
    assert memory_extraction.utc_now() == second


def test_a_bound_inference_port_answers_memorys_own_call():
    seen: list[str] = []

    async def fake_call(provider, prompt, max_tokens=2048, span=None, **_):
        seen.append(prompt)
        return '{"memories": [], "social_experiences": []}'

    from unittest.mock import MagicMock

    from kai_memory.extraction.memory_extraction import ExtractionService

    ports.bind(call_llm_text=fake_call, get_provider_for_model=lambda settings, model: object())
    out = asyncio.run(ExtractionService(MagicMock())._call_haiku("conv text", "(none)", "(none)"))
    assert out["memories"] == [] and "conv text" in seen[0]


def test_patching_a_memory_module_attribute_still_wins_over_the_port(monkeypatch):
    """The patch-target inventory keeps working: a module attribute patch replaces the wrapper."""
    from datetime import UTC, datetime

    from kai_memory.extraction import memory_extraction

    pinned = datetime(2029, 6, 1, tzinfo=UTC)
    ports.bind(clock=lambda: datetime(2030, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(memory_extraction, "utc_now", lambda: pinned)
    assert memory_extraction.utc_now() == pinned


def test_telemetry_port_supplies_memorys_generation_spans():
    calls: list[tuple[str, str]] = []

    class Spans:
        def span(self, name, model):
            calls.append((name, model))
            return "span"

        def event(self, name, **fields):
            pass

        def counter(self, name, value=1, **tags):
            pass

    from kai_memory.artifacts.producers import entity_schema

    ports.bind(telemetry=Spans())
    assert entity_schema.start_generation_span("memory/x", "haiku-4.5") == "span"
    assert calls == [("memory/x", "haiku-4.5")]


def test_narrow_scoped_override_preserves_outer_adapters_and_restores_after_error():
    class StraySessionError(RuntimeError):
        pass

    def _refusing_session_factory():
        raise StraySessionError("synthetic refusing factory")

    embedder, original_factory = object(), object()
    with ports.scoped({"embedder": embedder, "SessionLocal": original_factory}):
        with pytest.raises(StraySessionError):
            with ports.scoped({"SessionLocal": _refusing_session_factory}, inherit=True):
                assert ports.embedder is embedder
                ports.SessionLocal()
        assert ports.SessionLocal is original_factory
        assert ports.embedder is embedder
