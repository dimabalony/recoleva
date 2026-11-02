"""Altered physical semantics must fail the public read-only compatibility gate."""

from types import SimpleNamespace

import pytest
from sqlalchemy import MetaData, text
from sqlalchemy.dialects.postgresql import dialect


@pytest.mark.parametrize("mutation", ["status_literal", "json_key", "vector_operator"])
async def test_physical_index_semantics_cannot_hide_behind_same_name(monkeypatch, mutation):
    from kai_memory.storage import compatibility as gate
    from kai_memory.storage import metadata

    reflected = MetaData()
    for table in metadata.sorted_tables:
        table.to_metadata(reflected, schema="public")
    if mutation == "status_literal":
        table = reflected.tables["public.memory_artifacts"]
        index = next(i for i in table.indexes if str(i.dialect_options["postgresql"].get("where")) == "status = 'active'")
        index.dialect_options["postgresql"]["where"] = text("status = 'ACTIVE'")
    elif mutation == "json_key":
        table = reflected.tables["public.execution_traces"]
        index = next(i for i in table.indexes if i.name == "idx_exec_traces_assistant_message")
        index.dialect_options["postgresql"]["where"] = text("success_signals->>'ASSISTANT_MESSAGE_ID' IS NOT NULL")
    else:
        candidates = [(t, i) for t in reflected.tables.values() for i in t.indexes
                      if any(str(op).startswith("vector_") for op in i.dialect_options["postgresql"].get("ops", {}).values())]
        assert candidates, "The storage contract must retain vector-index operator coverage"
        table, index = candidates[0]
        index.dialect_options["postgresql"]["ops"] = {
            key: "vector_l2_ops" if value != "vector_l2_ops" else "vector_cosine_ops"
            for key, value in index.dialect_options["postgresql"]["ops"].items()
        }

    monkeypatch.setattr(gate, "inspect", lambda _conn: SimpleNamespace(get_table_names=lambda **_kw: list(metadata.tables)))
    monkeypatch.setattr(gate, "MetaData", lambda: reflected)
    monkeypatch.setattr(reflected, "reflect", lambda *_args, **_kw: None)
    statements = []

    def execute(statement, _parameters=None):
        statements.append(str(statement))
        return SimpleNamespace(all=lambda: [], scalars=lambda: SimpleNamespace(all=lambda: []))

    sync = SimpleNamespace(dialect=dialect(), execute=execute)

    async def run_sync(callback):
        return callback(sync)

    connection = SimpleNamespace(dialect=dialect(), run_sync=run_sync)
    report = await gate.inspect_storage(connection)
    assert not report.compatible
    assert report.revision is None
    assert any("indexes" in issue and table.name in issue for issue in report.issues), report
    assert statements and all(s.lstrip().upper().startswith("SELECT") for s in statements)
    with pytest.raises(gate.IncompatibleStorageError):
        await gate.require_compatible_storage(connection)
