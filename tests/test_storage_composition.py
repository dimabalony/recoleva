"""SDK metadata composition and index semantics, independent of host schemas."""

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex


def test_migration_indexes_have_exact_expression_and_predicate():
    from kai_memory.storage import metadata

    dialect = postgresql.dialect()
    private = next(i for i in metadata.tables["social_contexts"].indexes if i.name == "idx_social_ctx_private")
    assert str(CreateIndex(private).compile(dialect=dialect)) == (
        "CREATE UNIQUE INDEX idx_social_ctx_private ON social_contexts (chat_id) WHERE topic_id IS NULL"
    )
    durability = next(
        i for i in metadata.tables["memory_traces"].indexes if i.name == "ix_memory_traces_durability_marker"
    )
    expression = str(durability.expressions[0].compile(dialect=dialect, compile_kwargs={"literal_binds": True}))
    predicate = str(
        durability.dialect_options["postgresql"]["where"].compile(
            dialect=dialect, compile_kwargs={"literal_binds": True}
        )
    )

    def normalize(sql):
        return "".join(sql.replace("memory_traces.", "").split()).strip("()")

    assert normalize(expression) == "typed_metadata->>'durability'"
    assert normalize(predicate) == "typed_metadata->>'durability'IN('durable','superseded'"


def test_composition_is_idempotent_without_duplicate_function_hooks():
    from sqlalchemy import Column, Integer, MetaData, Table

    from kai_memory.storage import compose_metadata

    target = MetaData()
    host = Table("host_only", target, Column("id", Integer, primary_key=True))
    assert compose_metadata(target) is target
    first_tables = dict(target.tables)
    entity = target.tables["memory_entities"]
    first_hooks = tuple(entity.dispatch.before_create)
    assert len(first_hooks) == 1
    assert compose_metadata(target) is target
    assert dict(target.tables) == first_tables
    assert target.tables["host_only"] is host
    assert tuple(entity.dispatch.before_create) == first_hooks


def test_composition_rejects_conflict_before_copying_any_tables():
    from sqlalchemy import Column, Integer, MetaData, Table

    from kai_memory.storage import compose_metadata, metadata

    target = MetaData()
    Table("host_only", target, Column("id", Integer, primary_key=True))
    # A late dependency ensures a copy-as-you-go implementation would already
    # have polluted the host registry before discovering this conflict.
    name = metadata.sorted_tables[-1].name
    conflict = Table(name, target, Column("host_owned_id", Integer, primary_key=True))
    before = dict(target.tables)
    with pytest.raises(ValueError, match="Conflicting memory table"):
        compose_metadata(target)
    assert dict(target.tables) == before
    assert target.tables[name] is conflict
    assert list(conflict.columns.keys()) == ["host_owned_id"]
