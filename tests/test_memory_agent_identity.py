"""Configured identity without host imports or inference calls."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from sqlalchemy.dialects import postgresql

from kai_memory import ports
from kai_memory.identity import AGENT_IDENTITY_DATA_KEY, AgentIdentity, ensure_agent_identity, resolve_agent_identity


@pytest.fixture
def identity(monkeypatch):
    value = AgentIdentity("example-assistant", "Atlas", ("A",))
    monkeypatch.setitem(ports.__dict__, "get_agent_identity", lambda: value)
    return value


def row(name="Atlas", *, data=None):
    return SimpleNamespace(id=UUID(int=1), name=name, data=data or {}, aliases=[])


def session_with(*results):
    session = MagicMock()
    session.execute = AsyncMock(side_effect=results)
    session.flush = AsyncMock()
    return session


def result(*, scalar=None, rows=()):
    value = MagicMock()
    value.scalar_one_or_none.return_value = scalar
    value.scalars.return_value.all.return_value = list(rows)
    return value


@pytest.mark.parametrize("kwargs", [{"stable_id": ""}, {"display_name": " "}, {"aliases": ("",)}])
def test_identity_rejects_empty_names(kwargs):
    with pytest.raises(ValueError):
        AgentIdentity(**{"stable_id": "host", "display_name": "Atlas", **kwargs})


def test_identity_binding_uses_reserved_metadata_key():
    assert AGENT_IDENTITY_DATA_KEY == "_agent_identity_id"


def test_identity_recognition_is_separate_from_legacy_adoption():
    identity = AgentIdentity("host", "Atlas", ("A",), ("Previous",))
    assert identity.recognizes("atlas")
    assert identity.recognizes("a")
    assert not identity.recognizes("Previous")


@pytest.mark.asyncio
async def test_lookup_does_not_create_and_filters_tombstones(identity):
    session = session_with(result())
    assert await resolve_agent_identity(session) is None
    session.add.assert_not_called()
    sql = str(session.execute.call_args.args[0].compile(dialect=postgresql.dialect()))
    assert "merged_into_id IS NULL" in sql
    assert "telegram_user_id IS NULL" in sql
    assert "ORDER BY memory_entities.created_at, memory_entities.id" in sql
    assert AGENT_IDENTITY_DATA_KEY in session.execute.call_args.args[0].compile().params.values()


@pytest.mark.asyncio
async def test_explicit_legacy_adoption_keeps_uuid(monkeypatch):
    identity = AgentIdentity(
        "kai",
        "Kai",
        legacy_names=(
            "Kai",
            "Older",
        ),
    )
    monkeypatch.setitem(ports.__dict__, "get_agent_identity", lambda: identity)
    survivor = row("Kai", data={"human_authored": "keep"})
    session = session_with(result(), result(rows=[survivor]))
    assert await resolve_agent_identity(session) is survivor
    assert survivor.data == {"human_authored": "keep", AGENT_IDENTITY_DATA_KEY: "kai"}
    session.add.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_binding_cannot_steal_other_host(monkeypatch):
    monkeypatch.setitem(
        ports.__dict__, "get_agent_identity", lambda: AgentIdentity("one", "Atlas", legacy_names=("Old",))
    )
    session = session_with(result(), result(rows=[row("Old", data={AGENT_IDENTITY_DATA_KEY: "two"})]))
    with pytest.raises(ValueError, match="different stable identity"):
        await resolve_agent_identity(session)
    session.flush.assert_not_awaited()


@pytest.mark.asyncio
async def test_rename_keeps_stored_name_uuid_and_other_data(identity):
    survivor = row("Previous", data={AGENT_IDENTITY_DATA_KEY: identity.stable_id, "fact": "keep"})
    session = session_with(result(scalar=survivor))
    assert await ensure_agent_identity(session) is survivor
    assert survivor.name == "Previous"
    assert survivor.aliases == ["Atlas", "A"]
    assert survivor.data["fact"] == "keep"
    session.add.assert_not_called()


@pytest.mark.asyncio
async def test_explicit_initialization_creates_bound_entity(identity):
    session = session_with(result(), result())
    entity = await ensure_agent_identity(session)
    assert entity.name == "Atlas"
    assert entity.data == {AGENT_IDENTITY_DATA_KEY: "example-assistant"}
    session.add.assert_called_once_with(entity)


@pytest.mark.asyncio
async def test_initialization_does_not_adopt_names_implicitly(identity):
    session = session_with(result(), result(scalar=UUID(int=2)))
    with pytest.raises(ValueError, match="explicit legacy adoption"):
        await ensure_agent_identity(session)
    session.add.assert_not_called()


def test_synthetic_prompt_template_bytes_and_configured_identity_preserved():
    import hashlib
    from kai_memory.extraction.memory_extraction import EXTRACTION_PROMPT

    compatible = EXTRACTION_PROMPT.replace("{agent_display_name}", "Kai")
    assert (
        hashlib.sha256(compatible.encode()).hexdigest()
        == "331f10873fe39d40fdfe21d642a4ff94204109d4640fd79f83e6eec681d08539"
    )
    other = EXTRACTION_PROMPT.replace("{agent_display_name}", "Atlas")
    assert 'agent (e.g. "Atlas", this assistant itself)' in other
    assert '"Misha works at Examplecorp"' in other


@pytest.mark.asyncio
async def test_extraction_reuses_configured_alias_and_preserves_human(identity, monkeypatch):
    from kai_memory.extraction import memory_extraction as extraction

    survivor = row("Previous", data={AGENT_IDENTITY_DATA_KEY: identity.stable_id})
    survivor.entity_type = "agent"
    survivor.aliases = ["A"]
    human = SimpleNamespace(id=UUID(int=2), name="A", entity_type="person", aliases=[], telegram_user_id=None)
    monkeypatch.setitem(ports.__dict__, "is_admin_user", lambda *args: False)
    monkeypatch.setattr(extraction, "ensure_agent_identity", AsyncMock(return_value=survivor))
    monkeypatch.setattr(extraction, "get_settings", lambda: SimpleNamespace(telegram_admin_user_ids=""))
    service = extraction.ExtractionService.__new__(extraction.ExtractionService)
    service.session = MagicMock()
    service.memory_svc = MagicMock()
    service.memory_svc.store_trace = AsyncMock(return_value=SimpleNamespace(id=UUID(int=3)))
    service.memory_svc.link_trace_to_entity = AsyncMock()
    service.memory_svc.find_or_create_entity = AsyncMock()
    item = {
        "content": "A reflected",
        "entities": [
            {"name": "A", "type": "agent", "data": {AGENT_IDENTITY_DATA_KEY: "other-host"}},
            {"name": "A", "type": "person"},
        ],
        "facts": [{"subject": "A", "field": "mood", "value": "calm"}],
    }
    await service._store_item(
        item,
        SimpleNamespace(chat_id=1, topic_id=None),
        {"traces_created": 0, "entities_found": 0},
        entity_map={("a", "person"): human},
    )
    assert survivor.data[AGENT_IDENTITY_DATA_KEY] == identity.stable_id
    extraction.ensure_agent_identity.assert_awaited_once_with(service.session)
    service.memory_svc.find_or_create_entity.assert_not_awaited()
    assert {call.args[1] for call in service.memory_svc.link_trace_to_entity.await_args_list} == {survivor.id, human.id}
    # Homonymous agent and human make this fact ambiguous; do not steal it.
    assert item["facts"][0]["subject_id"] is None


@pytest.mark.asyncio
async def test_extraction_cannot_supply_host_binding_to_other_entity(identity, monkeypatch):
    from kai_memory.extraction import memory_extraction as extraction

    entity = row("Other")
    entity.entity_type = "agent"
    monkeypatch.setitem(ports.__dict__, "is_admin_user", lambda *args: False)
    monkeypatch.setattr(extraction, "get_settings", lambda: SimpleNamespace(telegram_admin_user_ids=""))
    service = extraction.ExtractionService.__new__(extraction.ExtractionService)
    service.session = MagicMock()
    service.memory_svc = MagicMock()
    service.memory_svc.store_trace = AsyncMock(return_value=SimpleNamespace(id=UUID(int=3)))
    service.memory_svc.link_trace_to_entity = AsyncMock()
    service.memory_svc.find_or_create_entity = AsyncMock(return_value=entity)

    await service._store_item(
        {
            "content": "Other is an assistant",
            "entities": [
                {
                    "name": "Other",
                    "type": "agent",
                    "data": {
                        AGENT_IDENTITY_DATA_KEY: identity.stable_id,
                        "role": "assistant",
                    },
                },
            ],
        },
        SimpleNamespace(chat_id=1, topic_id=None),
        {"traces_created": 0, "entities_found": 0},
    )

    service.memory_svc.find_or_create_entity.assert_awaited_once_with(
        "Other",
        "agent",
        data={"role": "assistant"},
        source_text="Other is an assistant",
    )


@pytest.mark.parametrize("lookup", ["name", "alias", "telegram"])
async def test_generic_entity_merge_preserves_host_binding_and_caller_data(identity, lookup):
    from kai_memory.store.cognitive_memory import CognitiveMemoryService

    survivor = row("Atlas", data={AGENT_IDENTITY_DATA_KEY: identity.stable_id, "profile": {"kept": "yes"}})
    survivor.aliases = ["A"]
    survivor.telegram_user_id = 42 if lookup == "telegram" else None
    session = session_with(result(scalar=survivor), result())
    incoming = {AGENT_IDENTITY_DATA_KEY: "other-host", "profile": {"role": "assistant"}, "tags": ["new"]}
    kwargs = {"telegram_user_id": 42} if lookup == "telegram" else {}

    entity = await CognitiveMemoryService(session).find_or_create_entity(
        "A" if lookup == "alias" else "Atlas", "agent", data=incoming, **kwargs
    )

    assert entity is survivor
    assert survivor.data == {
        AGENT_IDENTITY_DATA_KEY: identity.stable_id,
        "profile": {"kept": "yes", "role": "assistant"},
        "tags": ["new"],
    }
    assert incoming == {
        AGENT_IDENTITY_DATA_KEY: "other-host",
        "profile": {"role": "assistant"},
        "tags": ["new"],
    }


@pytest.mark.parametrize("collision", [False, True])
@pytest.mark.parametrize("public_fields", [False, True])
async def test_generic_entity_insert_cannot_forge_host_binding(identity, monkeypatch, collision, public_fields):
    from kai_memory.store import cognitive_config
    from kai_memory.store.cognitive_memory import CognitiveMemoryService

    monkeypatch.setattr(cognitive_config, "get_bool", lambda _name: False)
    existing = row("Atlas")
    existing.telegram_user_id = 10
    created = row("Atlas")
    inserted = result()
    inserted.scalar_one.return_value = created
    session = session_with(*([result(), result(scalar=existing), inserted] if collision else [result(), inserted]))
    public = {"profile": {"role": "assistant"}, "tags": ["new"]} if public_fields else {}
    incoming = {AGENT_IDENTITY_DATA_KEY: identity.stable_id, **public}

    assert (
        await CognitiveMemoryService(session).find_or_create_entity(
            "Atlas", "agent", data=incoming, telegram_user_id=20 if collision else None
        )
        is created
    )

    insert = session.execute.await_args.args[0]
    inserted_data = insert.compile(dialect=postgresql.dialect()).params.get("data")
    assert inserted_data == (public or None)
    assert incoming == {AGENT_IDENTITY_DATA_KEY: identity.stable_id, **public}


def test_synthetic_extraction_examples_keep_structured_retention_and_script_contract():
    import json
    from string import Formatter
    from kai_memory.extraction.memory_extraction import EXTRACTION_PROMPT

    # Replacing example identities must preserve every runtime insertion point.
    names = {name for _, name, _, _ in Formatter().parse(EXTRACTION_PROMPT) if name is not None}
    assert names == {
        "agent_display_name", "slot_durability_rubric", "schema_catalog", "procedural_block",
        "current_timestamp", "existing_memories", "thinking_blocks", "participant_context",
        "conversation", "served_rules_legend", "procedural_response_note",
    }
    rendered = EXTRACTION_PROMPT.format(**dict.fromkeys(names, "synthetic"))
    example = json.loads(rendered.split("\nExample:\n", 1)[1].split("\nIf nothing worth extracting", 1)[0])
    memory = example["memories"][0]
    person, employer = memory["entities"]
    assert person == {"name": "Misha", "type": "person"}
    assert employer == {"name": "Examplecorp", "type": "organization"}
    assert memory["facts"] == [{
        "subject": person["name"], "field": "role", "value": "founder", "polarity": "is",
        "slot": "occupation", "durability": "durable",
    }]
    assert memory["relationships"] == [{
        "subject": person["name"], "predicate": "works_at", "object": employer["name"], "confidence": 0.9,
    }]
    assert 'no "Луна" → "Luna", no "Миша" → "Misha"' in rendered
    assert '(NOT subject "Olga"/value "son: Misha")' in rendered
    assert "a relationships[] edge alone is invisible to durable retention" in rendered
    assert "Never\nstrip timestamps from source material" in rendered
