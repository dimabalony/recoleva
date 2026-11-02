from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from kai_memory.storage.models.memory_entity import MemoryEntity
from kai_memory.store import cognitive_config
from kai_memory.store.cognitive_memory import CognitiveMemoryService

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _disable_tier0_shadow_probe(monkeypatch):
    """Focus these tests on core resolution's exact execute sequences.

    The log-only Tier-0 shadow probe (task #8) adds one query on the text-mention
    miss path; it has its own dedicated tests (tests/test_tier0_shadow_probe.py +
    tests/integration/test_tier0_shadow_probe.py). Disable it here so these
    sequence-exact mocks stay unaffected. ``setitem`` auto-restores after each test.
    """
    monkeypatch.setitem(cognitive_config._CACHE, "tier0_shadow_probe_enabled", 0.0)


def _entity(
    name: str,
    entity_type: str = "person",
    telegram_user_id: int | None = None,
) -> MemoryEntity:
    return MemoryEntity(
        id=uuid.uuid4(),
        name=name,
        entity_type=entity_type,
        aliases=[],
        telegram_user_id=telegram_user_id,
    )


def _lookup_result(entity: MemoryEntity | None) -> MagicMock:
    result = MagicMock()
    result.scalar_one_or_none.return_value = entity
    return result


def _insert_result(entity: MemoryEntity) -> MagicMock:
    result = MagicMock()
    result.scalar_one.return_value = entity
    return result


async def test_telegram_user_id_match_takes_priority():
    session = AsyncMock()
    existing = _entity("Misha", telegram_user_id=100)
    session.execute = AsyncMock(side_effect=[_lookup_result(existing), MagicMock()])

    svc = CognitiveMemoryService(session)
    entity = await svc.find_or_create_entity("Mikhail", "person", telegram_user_id=100)

    assert entity.id == existing.id
    assert entity.name == "Misha"


async def test_exact_name_type_match_without_telegram_user_id_returns_existing():
    session = AsyncMock()
    existing = _entity("Maria", entity_type="person")
    session.execute = AsyncMock(
        side_effect=[
            _lookup_result(None),
            _insert_result(existing),
            _lookup_result(existing),
            MagicMock(),
        ]
    )

    svc = CognitiveMemoryService(session)
    first = await svc.find_or_create_entity("Maria", "person")
    second = await svc.find_or_create_entity("Maria", "person")

    assert first.id == second.id
    assert session.execute.await_count == 4


async def test_same_name_different_entity_type_creates_two_entities():
    session = AsyncMock()
    person = _entity("Python", entity_type="person")
    topic = _entity("Python", entity_type="topic")
    session.execute = AsyncMock(
        side_effect=[
            _lookup_result(None),
            _insert_result(person),
            _lookup_result(None),
            _insert_result(topic),
        ]
    )

    svc = CognitiveMemoryService(session)
    person_entity = await svc.find_or_create_entity("Python", "person")
    topic_entity = await svc.find_or_create_entity("Python", "topic")

    assert person_entity.id != topic_entity.id
    assert person_entity.entity_type == "person"
    assert topic_entity.entity_type == "topic"


async def test_same_name_type_different_telegram_user_id_creates_two_entities():
    session = AsyncMock()
    first_entity = _entity("Alex", telegram_user_id=100)
    second_entity = _entity("Alex", telegram_user_id=200)
    session.execute = AsyncMock(
        side_effect=[
            _lookup_result(None),
            _lookup_result(None),
            _insert_result(first_entity),
            _lookup_result(None),
            _lookup_result(first_entity),
            _insert_result(second_entity),
        ]
    )

    svc = CognitiveMemoryService(session)
    first = await svc.find_or_create_entity("Alex", "person", telegram_user_id=100)
    second = await svc.find_or_create_entity("Alex", "person", telegram_user_id=200)

    assert first.id != second.id
    assert first.telegram_user_id == 100
    assert second.telegram_user_id == 200


async def test_brand_new_entity_sets_fields():
    session = AsyncMock()
    created = _entity("Elena", entity_type="person", telegram_user_id=555)
    session.execute = AsyncMock(side_effect=[_lookup_result(None), _lookup_result(None), _insert_result(created)])

    svc = CognitiveMemoryService(session)
    entity = await svc.find_or_create_entity("Elena", "person", namespace="telegram", telegram_user_id=555)

    insert_stmt = session.execute.await_args_list[2].args[0]
    params = insert_stmt.compile(dialect=postgresql.dialect()).params

    assert entity.name == "Elena"
    assert entity.entity_type == "person"
    assert entity.telegram_user_id == 555
    assert params["name"] == "Elena"
    assert params["entity_type"] == "person"
    assert params["namespace"] == "telegram"
    assert params["telegram_user_id"] == 555


async def test_name_only_lookup_skips_tg_grounded():
    session = AsyncMock()
    tg_grounded = _entity("Pavel", telegram_user_id=100)
    ungrounded = _entity("Pavel", telegram_user_id=None)

    async def execute(stmt):
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        if "telegram_user_id IS NULL" in sql:
            return _lookup_result(ungrounded)
        if "memory_entities" in sql:
            return _lookup_result(tg_grounded)
        return MagicMock()

    session.execute = AsyncMock(side_effect=execute)

    svc = CognitiveMemoryService(session)
    entity = await svc.find_or_create_entity("Pavel")

    assert entity.id == ungrounded.id
    assert entity.telegram_user_id is None


async def test_name_only_lookup_with_only_tg_grounded_creates_new():
    session = AsyncMock()
    tg_grounded = _entity("Pavel", telegram_user_id=100)
    created = _entity("Pavel", telegram_user_id=None)

    async def execute(stmt):
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        if "INSERT INTO memory_entities" in sql:
            return _insert_result(created)
        if "telegram_user_id IS NULL" in sql:
            return _lookup_result(None)
        if "memory_entities" in sql:
            return _lookup_result(tg_grounded)
        return MagicMock()

    session.execute = AsyncMock(side_effect=execute)

    svc = CognitiveMemoryService(session)
    entity = await svc.find_or_create_entity("Pavel")

    assert entity.id == created.id
    assert entity.id != tg_grounded.id
    assert entity.telegram_user_id is None


# ── Task 5: data deep-merge ──────────────────────────────


async def test_deep_merge_nested_dicts():
    """v2a §5.4: nested dicts merge, leaf values from overlay win."""
    from kai_memory.store.cognitive_memory import _deep_merge

    base = {"color": "white", "details": {"breed": "labrador", "age": 3}}
    overlay = {"details": {"age": 4, "weight": "30kg"}, "owner": "Misha"}

    merged = _deep_merge(base, overlay)
    assert merged == {
        "color": "white",
        "details": {"breed": "labrador", "age": 4, "weight": "30kg"},
        "owner": "Misha",
    }
    # Inputs not mutated
    assert base == {"color": "white", "details": {"breed": "labrador", "age": 3}}
    assert overlay == {"details": {"age": 4, "weight": "30kg"}, "owner": "Misha"}


# ── Sub-design C: cross-script canonical_key resolver wiring ──────────────────


async def test_lookup_query_filters_on_canonical_key():
    """The name+type lookup also filters on canonical_key == translit_key(name),
    so an existing 'Миша' (canonical_key 'misha') is reachable when the Latin
    'Misha' is looked up — closing the cross-script fork gap."""
    from kai_memory.canonicalization import translit_key

    session = AsyncMock()
    captured: dict = {}

    async def execute(stmt):
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        if "INSERT INTO memory_entities" in sql:
            return _insert_result(_entity("Misha"))
        captured["sql"] = sql
        captured["params"] = stmt.compile(dialect=postgresql.dialect()).params
        return _lookup_result(None)

    session.execute = AsyncMock(side_effect=execute)

    svc = CognitiveMemoryService(session)
    await svc.find_or_create_entity("Misha", "person")

    # Inspect the WHERE predicate, not the SELECT projection (which now lists the
    # canonical_key column regardless).
    where = captured["sql"].split("WHERE", 1)[1]
    assert "canonical_key =" in where, "lookup WHERE must filter on canonical_key"
    assert translit_key("Misha") in captured["params"].values()


async def test_create_populates_canonical_key():
    """A newly created entity stores its own translit_key as canonical_key."""
    from kai_memory.canonicalization import translit_key

    session = AsyncMock()
    created = _entity("Миша")
    session.execute = AsyncMock(side_effect=[_lookup_result(None), _insert_result(created)])

    svc = CognitiveMemoryService(session)
    await svc.find_or_create_entity("Миша", "person")

    insert_stmt = session.execute.await_args_list[-1].args[0]
    params = insert_stmt.compile(dialect=postgresql.dialect()).params
    assert "canonical_key" in params, "insert must set canonical_key"
    assert params["canonical_key"] == translit_key("Миша")  # "misha"


async def test_cross_script_lookup_returns_existing_entity():
    """Behavioral: looking up the Latin 'Misha' returns the existing Cyrillic
    'Миша' entity (merge) instead of forking a new one."""
    session = AsyncMock()
    existing = _entity("Миша")

    async def execute(stmt):
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        if "INSERT INTO memory_entities" in sql:
            return _insert_result(_entity("Misha"))  # the fork it WOULD create
        # Match on the WHERE predicate, not the SELECT projection (which lists
        # the canonical_key column on every row).
        where = sql.split("WHERE", 1)[-1] if "WHERE" in sql else ""
        if "canonical_key =" in where:
            return _lookup_result(existing)
        return _lookup_result(None)

    session.execute = AsyncMock(side_effect=execute)

    svc = CognitiveMemoryService(session)
    entity = await svc.find_or_create_entity("Misha", "person")

    assert entity.id == existing.id
    assert entity.name == "Миша"


async def test_cross_script_merge_emits_observability_log(caplog):
    """A canonical_key-only recovery (no surface/alias overlap) logs a
    `cognitive.entity.cross_script_merge` INFO line — the prod signal
    feature-watch verifies the feature against."""
    import logging

    session = AsyncMock()
    existing = _entity("Миша")  # no alias overlap with the Latin lookup

    async def execute(stmt):
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        if "INSERT INTO memory_entities" in sql:
            return _insert_result(_entity("Misha"))
        if "canonical_key =" in sql.split("WHERE", 1)[-1]:
            return _lookup_result(existing)
        return _lookup_result(None)

    session.execute = AsyncMock(side_effect=execute)

    svc = CognitiveMemoryService(session)
    with caplog.at_level(logging.INFO):
        await svc.find_or_create_entity("Misha", "person")

    assert any("cross_script_merge" in r.getMessage() for r in caplog.records), (
        f"expected cross_script_merge log; got {[r.getMessage() for r in caplog.records]!r}"
    )


async def test_plain_surface_match_does_not_emit_cross_script_log(caplog):
    """An exact-name match must NOT log a cross-script merge."""
    import logging

    session = AsyncMock()
    existing = _entity("Maria")

    async def execute(stmt):
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        if "INSERT INTO memory_entities" in sql:
            return _insert_result(existing)
        return _lookup_result(existing)

    session.execute = AsyncMock(side_effect=execute)

    svc = CognitiveMemoryService(session)
    with caplog.at_level(logging.INFO):
        await svc.find_or_create_entity("Maria", "person")

    assert not any("cross_script_merge" in r.getMessage() for r in caplog.records)


async def test_garbage_name_does_not_filter_on_empty_canonical_key():
    """A name with empty translit_key (punctuation only) must NOT add a
    `canonical_key == ''` clause — that would mass-match unrelated rows."""
    session = AsyncMock()
    captured: dict = {}

    async def execute(stmt):
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        if "INSERT INTO memory_entities" in sql:
            return _insert_result(_entity("!!!"))
        captured["sql"] = sql
        return _lookup_result(None)

    session.execute = AsyncMock(side_effect=execute)

    svc = CognitiveMemoryService(session)
    await svc.find_or_create_entity("!!!", "topic")

    # The WHERE clause must carry no canonical_key predicate (the SELECT
    # projection lists the column, so check the WHERE portion only).
    where = captured["sql"].split("WHERE", 1)[1]
    assert "canonical_key =" not in where, "must not filter on empty canonical_key"


async def test_deep_merge_lists_replaced_not_concatenated():
    """v2a §5.4: lists are replaced wholesale (avoid unbounded growth)."""
    from kai_memory.store.cognitive_memory import _deep_merge

    base = {"members": ["a", "b"]}
    overlay = {"members": ["c"]}

    merged = _deep_merge(base, overlay)
    assert merged == {"members": ["c"]}


async def test_deep_merge_overlay_replaces_when_types_differ():
    from kai_memory.store.cognitive_memory import _deep_merge

    base = {"x": {"nested": True}}
    overlay = {"x": "scalar"}

    merged = _deep_merge(base, overlay)
    assert merged == {"x": "scalar"}


async def test_find_or_create_entity_accepts_data_kwarg():
    """v2a §5.4: find_or_create_entity accepts data: dict | None."""
    from inspect import signature

    sig = signature(CognitiveMemoryService.find_or_create_entity)
    assert "data" in sig.parameters


async def test_hit_path_telegram_id_deep_merges_data():
    """v2a §5.4 hit path: existing entity by tg_id gets data deep-merged."""
    session = AsyncMock()
    existing = _entity("Луна", telegram_user_id=100)
    existing.data = {"color": "white", "details": {"age": 3}}
    session.execute = AsyncMock(side_effect=[_lookup_result(existing), MagicMock()])

    svc = CognitiveMemoryService(session)
    entity = await svc.find_or_create_entity(
        "Луна",
        "person",
        telegram_user_id=100,
        data={"details": {"weight": "5kg"}, "owner": "Misha"},
    )

    assert entity.id == existing.id
    assert entity.data == {
        "color": "white",
        "details": {"age": 3, "weight": "5kg"},
        "owner": "Misha",
    }


async def test_hit_path_name_match_deep_merges_data():
    """v2a §5.4 hit path: existing entity by name match gets data merged."""
    session = AsyncMock()
    existing = _entity("Луна", entity_type="animal")
    existing.data = {"color": "white"}
    # Two execute calls:
    #   1. tg_id lookup (skipped — no telegram_user_id passed) → name+type
    #   2. _bump_mention (UPDATE)
    session.execute = AsyncMock(side_effect=[_lookup_result(existing), MagicMock()])

    svc = CognitiveMemoryService(session)
    entity = await svc.find_or_create_entity(
        "Луна",
        "animal",
        data={"breed": "labrador"},
    )

    assert entity.id == existing.id
    assert entity.data == {"color": "white", "breed": "labrador"}


async def test_hit_path_no_incoming_data_leaves_existing():
    """No incoming data → existing entity.data unchanged."""
    session = AsyncMock()
    existing = _entity("Луна", entity_type="animal")
    existing.data = {"color": "white"}
    session.execute = AsyncMock(side_effect=[_lookup_result(existing), MagicMock()])

    svc = CognitiveMemoryService(session)
    entity = await svc.find_or_create_entity("Луна", "animal")

    assert entity.data == {"color": "white"}


async def test_hit_path_existing_data_none_assigns_overlay():
    """Existing entity.data is None → assign overlay (no merge needed)."""
    session = AsyncMock()
    existing = _entity("Луна", entity_type="animal")
    existing.data = None
    session.execute = AsyncMock(side_effect=[_lookup_result(existing), MagicMock()])

    svc = CognitiveMemoryService(session)
    entity = await svc.find_or_create_entity("Луна", "animal", data={"color": "white"})

    assert entity.data == {"color": "white"}


async def test_insert_path_passes_data_to_values():
    """v2a §5.4 insert path: new entity's data field populated from kwarg."""
    session = AsyncMock()
    created = _entity("Нора", entity_type="animal")
    session.execute = AsyncMock(
        side_effect=[
            _lookup_result(None),
            _insert_result(created),
        ]
    )

    svc = CognitiveMemoryService(session)
    entity = await svc.find_or_create_entity(
        "Нора",
        "animal",
        data={"color": "grey"},
    )

    # The insert statement should include data
    insert_stmt = session.execute.await_args_list[1].args[0]
    params = insert_stmt.compile(dialect=postgresql.dialect()).params
    assert params.get("data") == {"color": "grey"}
    assert entity.id == created.id


# ── Task 6: romanization wired into find_or_create_entity ──────


async def test_find_or_create_accepts_source_text_kwarg():
    """v2a §5.5: source_text kwarg added for reverse-source-scan."""
    from inspect import signature

    sig = signature(CognitiveMemoryService.find_or_create_entity)
    assert "source_text" in sig.parameters


async def test_insert_path_seeds_aliases_with_romanization():
    """v2a §5.5: non-ASCII name → romanized form added to aliases on INSERT."""
    session = AsyncMock()
    created = _entity("Лиана", entity_type="animal")
    session.execute = AsyncMock(
        side_effect=[
            _lookup_result(None),
            _insert_result(created),
        ]
    )

    svc = CognitiveMemoryService(session)
    await svc.find_or_create_entity("Лиана", "animal")

    insert_stmt = session.execute.await_args_list[1].args[0]
    params = insert_stmt.compile(dialect=postgresql.dialect()).params
    aliases = params.get("aliases") or []
    assert "Liana" in aliases


async def test_insert_path_no_aliases_when_already_ascii():
    """ASCII name + no source_text → no aliases seeded."""
    session = AsyncMock()
    created = _entity("Maria", entity_type="person")
    session.execute = AsyncMock(
        side_effect=[
            _lookup_result(None),
            _insert_result(created),
        ]
    )

    svc = CognitiveMemoryService(session)
    await svc.find_or_create_entity("Maria", "person")

    insert_stmt = session.execute.await_args_list[1].args[0]
    params = insert_stmt.compile(dialect=postgresql.dialect()).params
    aliases = params.get("aliases") or []
    # No source_text → no Cyrillic candidates; ASCII name → no romanization
    # candidate. Aliases should be empty / absent.
    assert aliases == [] or aliases is None or "aliases" not in params


async def test_hit_path_appends_new_alias_candidates():
    """v2a §5.5: existing entity gets new Cyrillic alias when source_text scan finds it."""
    session = AsyncMock()
    existing = _entity("Misha", entity_type="person")
    existing.aliases = []
    session.execute = AsyncMock(side_effect=[_lookup_result(existing), MagicMock()])

    svc = CognitiveMemoryService(session)
    await svc.find_or_create_entity(
        "Misha",
        "person",
        source_text="приехал Миша из Москвы",
    )

    assert "Миша" in (existing.aliases or [])


async def test_hit_path_no_duplicate_alias():
    """Existing alias is not duplicated."""
    session = AsyncMock()
    existing = _entity("Misha", entity_type="person")
    existing.aliases = ["Миша"]
    session.execute = AsyncMock(side_effect=[_lookup_result(existing), MagicMock()])

    svc = CognitiveMemoryService(session)
    await svc.find_or_create_entity(
        "Misha",
        "person",
        source_text="Миша снова",
    )

    assert (existing.aliases or []).count("Миша") == 1


# ── review-r3: transliteration bifurcation regression (M3 ConfusionSetProducer scope) ───


@pytest.mark.xfail(
    strict=False,
    reason=(
        "M3 ConfusionSetProducer scope. The exact-alias lookup + "
        "soft-sign-apostrophe romanization combination cannot bridge "
        "Haiku-transliterated forms ('Luna') to canonical Cyrillic "
        "originals ('Луна') when the Cyrillic alias list lacks the "
        "exact transliteration Haiku chose. Stop-the-bleed for this is "
        "the L2 prompt rule 'always keep entity names in original "
        "script' (review-r3 hotfix); the structural fix is M3's "
        "edit-distance match in ConfusionSetProducer."
    ),
)
async def test_haiku_transliteration_binds_to_existing_cyrillic_entity():
    """Reproduces the 2026-05-02 production bifurcation:
    Луна (animal) exists with aliases [Luna, Luna, S'iuzi, Sьюзи].
    Haiku later returns "Luna" (with trailing E) — find_or_create_entity
    should return the existing Луна entity, not create a duplicate.

    Currently fails because:
      - "Luna" doesn't equal any existing name OR alias (the "Luna"
        alias is missing the trailing E).
      - The reverse-source-scan can't bridge "Луна" (in source) to
        "Luna" (the lookup name): unidecode("Луна")="S'iuzi" which
        normalizes to "s'iuzi", not "luna".

    Marked xfail (strict=False) so the test runs and reports xpassed
    once M3 lands — at which point flip strict=True and remove the marker.
    """
    session = AsyncMock()
    existing = _entity("Луна", entity_type="animal")
    existing.aliases = ["Luna", "Luna", "S'iuzi", "Sьюзи"]
    fresh = _entity("Luna", entity_type="animal")  # what bifurcation creates

    # First execute: name/alias lookup against memory_entities — in prod
    # this MISSES because "Luna" doesn't equal any existing name OR alias
    # of the canonical Луна row. The reverse-source-scan in
    # `candidate_aliases("Luna", "...Луна...")` returns [] because
    # unidecode("Луна")="S'iuzi" doesn't normalize to "luna".
    # Second execute: insert path mints a fresh duplicate.
    session.execute = AsyncMock(side_effect=[_lookup_result(None), _insert_result(fresh)])

    svc = CognitiveMemoryService(session)
    entity = await svc.find_or_create_entity(
        "Luna",
        "animal",
        source_text="Луна и Нора сегодня будут с мамой гулять",
    )

    assert entity.id == existing.id, (
        f"expected existing Луна entity ({existing.id}), got {entity.id} — transliteration bifurcation occurred"
    )
