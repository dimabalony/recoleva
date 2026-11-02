"""Versioned schema contracts without a live database."""

import importlib.util
from pathlib import Path

import pytest
from sqlalchemy import MetaData
from sqlalchemy.dialects.postgresql import dialect

from kai_memory.storage import compose_metadata, metadata
from kai_memory.storage.compatibility import (
    CURRENT_SCHEMA_REVISION,
    M1_SCHEMA_REVISION,
    REPLAY_INDEX,
    StorageCompatibility,
    _shape,
    _sql,
)


@pytest.mark.parametrize(
    "left,right",
    [
        ("status = 'active'", "status = 'ACTIVE'"),
        ("value = 'a b'", "value = 'ab'"),
        ("value = '(a)'", "value = 'a'"),
        ("a = 1 OR b = 2 AND c = 3", "(a = 1 OR b = 2) AND c = 3"),
    ],
)
def test_expression_contract_preserves_literals_and_precedence(left, right):
    assert _sql(left) != _sql(right)


def test_known_postgres_expression_deparsing():
    def expression(sql):
        return _sql(sql, text_columns={"status"}, allow_text_literals=True)

    assert expression("status IN ('active', 'expired')") == expression(
        "((status)::text = ANY (ARRAY['active'::text, 'expired'::text]))"
    )
    assert expression("success_signals->>'assistant_message_id' IS NOT NULL") == expression(
        "((success_signals ->> 'assistant_message_id'::text) IS NOT NULL)"
    )


def test_casts_and_quoted_identifiers_remain_significant():
    assert _sql('"Case"') != _sql('"case"')
    assert _sql("ranking_score") != _sql("ranking_score::text")
    assert _sql("value = 'public.memory_traces.content'", "memory_traces") != _sql("value = 'content'")
    assert _sql("0.5") != _sql("0.5::double precision")


def test_unknown_expression_fails_closed():
    with pytest.raises(ValueError, match="Unsupported"):
        _sql("status ~~ 'active%'")


def test_storage_version_is_distinct_from_sdk_api_and_requires_current():
    assert M1_SCHEMA_REVISION != CURRENT_SCHEMA_REVISION
    assert not StorageCompatibility(M1_SCHEMA_REVISION, ()).compatible
    assert StorageCompatibility(CURRENT_SCHEMA_REVISION, ()).compatible
    assert not StorageCompatibility(None, ("unknown schema",)).compatible
    assert not StorageCompatibility(CURRENT_SCHEMA_REVISION, ("bad index",)).compatible


def test_m1_historical_snapshot_matches_only_one_index_delta():
    path = Path(__file__).parents[1] / "examples/storage_upgrade.py"
    spec = importlib.util.spec_from_file_location("storage_upgrade", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old, commit = module.historical_schema(dialect())
    assert commit == "c08ca8fbb1f6217e0116be390659ec52715c1280"
    for name, table in metadata.tables.items():
        current = _shape(table, dialect())
        if name == "execution_traces":
            current["indexes"].pop(REPLAY_INDEX)
        assert current == _shape(old.tables[name], dialect())


def test_schema_qualification_does_not_change_contract():
    public = MetaData(schema="public")
    for table in metadata.sorted_tables:
        table.to_metadata(public, schema="public")
    for name, table in metadata.tables.items():
        assert _shape(table, dialect()) == _shape(public.tables["public." + name], dialect())


def test_index_method_and_opclass_are_part_of_contract():
    schema = compose_metadata(MetaData())
    table = schema.tables["memory_traces"]
    original = _shape(table, dialect())
    index = next(i for i in table.indexes if i.dialect_options["postgresql"].get("using") == "hnsw")
    index.dialect_options["postgresql"]["using"] = "btree"
    assert _shape(table, dialect()) != original


def test_changed_server_defaults_are_incompatible():
    from sqlalchemy import DefaultClause, text

    schema = compose_metadata(MetaData())
    table = schema.tables["memory_artifacts"]
    original = _shape(table, dialect())
    table.c.status.server_default = DefaultClause(text("'expired'"))
    assert _shape(table, dialect())["server_defaults"] != original["server_defaults"]


def test_changed_foreign_key_deferrability_is_incompatible():
    schema = compose_metadata(MetaData())
    table = schema.tables["memory_artifacts"]
    original = _shape(table, dialect())
    deferred = next(fk for fk in table.foreign_key_constraints if fk.deferrable)
    deferred.deferrable = False
    assert _shape(table, dialect())["foreign_keys"] != original["foreign_keys"]


def test_unsupported_diagnostics_do_not_include_expression_or_default_content():
    secret = "a-private-default-that-must-not-leak"
    with pytest.raises(ValueError) as error:
        _sql(f"content ~~ '{secret}'")
    assert secret not in str(error.value)
    assert str(error.value) == "Unsupported schema expression"


def test_current_primary_and_unique_contracts_are_nondeferrable():
    from sqlalchemy.schema import UniqueConstraint

    for table in metadata.tables.values():
        assert not table.primary_key.deferrable
        for constraint in table.constraints:
            if isinstance(constraint, UniqueConstraint):
                assert not constraint.deferrable


def test_rehearsal_checks_survive_optimized_python():
    import subprocess
    import sys

    path = Path(__file__).parents[1] / "examples/storage_upgrade.py"
    code = (
        "import importlib.util; "
        f"s=importlib.util.spec_from_file_location('rehearsal', {str(path)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "m._require(False, 'invariant remains enforced')"
    )
    result = subprocess.run([sys.executable, "-I", "-O", "-c", code], capture_output=True, text=True)
    assert result.returncode != 0
    assert "AssertionError: invariant remains enforced" in result.stderr
