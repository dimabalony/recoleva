from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock
import asyncio
import pytest
from kai_memory import AgentIdentity, HostAdapters, MemoryClient, ModelRoles, NeutralHostPolicy
from kai_memory import ports
from kai_memory.host_calls import get_model_roles, get_host_policy
from kai_memory.identity import get_agent_identity
from kai_memory.store.embeddings import (
    EmbeddingError,
    generate_embedding,
    generate_embeddings_batch,
    _diagnostic_payload,
)


def client(name, model):
    return MemoryClient(
        HostAdapters(
            inference=SimpleNamespace(complete=AsyncMock(return_value="ok")),
            embedder=SimpleNamespace(embed=AsyncMock(return_value=[[1.0] * 1536])),
            history=SimpleNamespace(),
            session_factory=lambda: None,
            identity=AgentIdentity(name, name.title()),
            models=ModelRoles(model, model, model, model),
            overrides={
                "scrub_secrets": lambda s: s,
                "validate_skill_content": lambda s: [],
                "redact_urls_for_trace": lambda s: None,
            },
        )
    )


@pytest.mark.asyncio
async def test_concurrent_scopes_route_models_and_identity():
    a, b = client("atlas", "vendor/a"), client("orion", "other/b")

    async def run(c):
        with c._bound():
            await asyncio.sleep(0)
            assert get_agent_identity() == c.adapters.identity
            assert get_model_roles() == c.adapters.models
            assert get_host_policy() is c.adapters.policy
            provider = ports.get_provider_for_model(None, get_model_roles().extraction)
            assert await ports.call_llm_text(provider, "synthetic") == "ok"
        c.adapters.inference.complete.assert_awaited_once_with(
            "synthetic", model=c.adapters.models.extraction, max_tokens=2048, thinking_mode="adaptive"
        )

    await asyncio.gather(run(a), run(b))
    await a.close()
    await b.close()


def test_neutral_authority_and_compatibility_model_defaults():
    p = NeutralHostPolicy()
    assert not p.is_admin(1) and p.respect(1) == 0
    assert p.may_read(None) and p.may_write(None)
    assert ModelRoles().extraction == ModelRoles().auxiliary == "haiku-5.5"
    assert ModelRoles().artifact == "sonnet-5.5"
    with pytest.raises(ValueError):
        ModelRoles(extraction=" ")


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", [False, True])
async def test_embedding_dimensions_rejected(batch):
    with ports.scoped({"embedder": SimpleNamespace(embed=AsyncMock(return_value=[[1.0] * 3]))}):
        with pytest.raises(EmbeddingError, match="1536"):
            await (generate_embeddings_batch(["private"]) if batch else generate_embedding("private"))


def test_embedding_diagnostics_never_include_content():
    assert "private" not in str(_diagnostic_payload("private", kind="single"))


@pytest.mark.asyncio
@pytest.mark.parametrize("extraction,auxiliary", [("extract/x", "aux/a"), ("haiku-4.5", "haiku-4.5")])
async def test_actual_inference_paths_honor_each_role(extraction, auxiliary):
    from kai_memory.extraction.memory_extraction import ExtractionService
    from kai_memory.consolidation.consolidation import _call_sonnet_text
    from kai_memory.durable_backfill import haiku_slot_classifier
    from kai_memory.slot_summarizer import _curate_via_haiku
    from kai_memory.subject_backfill import haiku_subject_assigner

    from kai_memory.artifacts.producers.trajectory import TrajectoryProducer

    original = client("atlas", "unused")
    c = MemoryClient(
        replace(original.adapters, models=ModelRoles(extraction, "consolidate/y", "artifact/z", auxiliary))
    )
    with c._bound():
        extractor = ExtractionService(SimpleNamespace())
        await extractor._call_haiku("synthetic", "", "")
        await _call_sonnet_text("synthetic", "test")
        await _curate_via_haiku([], "Synthetic", None)
        await haiku_slot_classifier("synthetic", [], [])
        await haiku_subject_assigner("synthetic", [], [])
        await TrajectoryProducer()._call_sonnet_summaries("synthetic")
        await haiku_subject_assigner("synthetic", [], [], provider=SimpleNamespace(model="explicit/provider"))
    assert [call.kwargs["model"] for call in c.adapters.inference.complete.await_args_list] == [
        extraction,
        "consolidate/y",
        auxiliary,
        auxiliary,
        auxiliary,
        "artifact/z",
        "explicit/provider",
    ]
    calls = c.adapters.inference.complete.await_args_list
    assert [calls[index].kwargs["thinking_mode"] for index in (0, 2, 3, 4, 6)] == ["disabled"] * 5
    assert [calls[index].kwargs["max_tokens"] for index in (0, 2, 3, 4)] == [11000, 1400, 2800, 1400]
    await original.close()
    await c.close()
