"""Host policy must be explicit before memory can accept work."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from kai_memory import HostAdapters, MemoryClient
from kai_memory.artifacts.producer import ProducerSubject
from kai_memory.artifacts.producers.pattern_producer import PatternProducer
from kai_memory.store.cognitive_memory import CognitiveMemoryService

POLICY = {
    "scrub_secrets": lambda text: text.replace("synthetic-secret", "[redacted]"),
    "validate_skill_content": lambda text: ["unsafe"] if "unsafe" in text else [],
    "redact_urls_for_trace": lambda text: "[url redacted]" if text else text,
}


def adapters(**kwargs):
    return HostAdapters(inference=object(), embedder=object(), history=object(), session_factory=lambda: None, **kwargs)


@pytest.mark.parametrize("name", POLICY)
@pytest.mark.parametrize("invalid", ["missing", None, object()])
def test_hygiene_policy_is_required_before_any_work(name, invalid):
    policy = dict(POLICY)
    if invalid == "missing":
        policy.pop(name)
    else:
        policy[name] = invalid
    with pytest.raises(ValueError, match=name):
        MemoryClient(adapters(overrides=policy))


async def test_host_policy_changes_the_trace_submitted_to_storage():
    session = MagicMock()
    session.flush = AsyncMock()
    session.execute = AsyncMock()
    client = MemoryClient(adapters(overrides=POLICY))
    try:
        trace = await client.run_owned(
            lambda: asyncio.create_task(
                CognitiveMemoryService(session).store_trace(
                    "unsafe synthetic-secret",
                    "episodic",
                    significance=5,
                    embedding=[1.0],
                )
            )
        )
        assert trace.content == "unsafe [redacted]"
        assert trace.significance == 1
        assert session.add.call_args.args[0] is trace
    finally:
        await client.close()


async def test_pattern_payload_uses_each_clients_clock():
    subject = ProducerSubject(
        key="test", payload={"dimension": "directness", "observation": "speaks clearly", "valid_trace_ids": set()}
    )
    base = adapters(overrides=POLICY)
    for year in (2030, 2040):
        fixed = datetime(year, 1, 2, tzinfo=UTC)
        client = MemoryClient(replace(base, clock=lambda fixed=fixed: fixed))
        try:
            artifact = await client.run_owned(
                lambda: asyncio.create_task(PatternProducer().produce(MagicMock(), subject, None))
            )
            assert datetime.fromisoformat(artifact.payload["observed_at"]) == fixed
        finally:
            await client.close()


@pytest.mark.parametrize("guidance", ["unsafe instruction", "synthetic-secret"])
async def test_host_policy_quarantines_rules_before_rendering(guidance):
    from types import SimpleNamespace

    from kai_memory.procedural.procedural_serving import _render

    client = MemoryClient(adapters(overrides=POLICY))
    try:
        row = SimpleNamespace(id="synthetic", typed_metadata={"procedural": {"guidance": guidance}})
        rendered = client.run_owned(lambda: _render([row]))
        assert "[QUARANTINED]" in rendered
        assert guidance not in rendered
    finally:
        await client.close()


async def test_host_url_policy_protects_persisted_procedural_trace(monkeypatch):
    from kai_memory.storage.models.execution_trace import ExecutionTrace
    from kai_memory import Chat
    from kai_memory.procedural import procedural_recorder

    monkeypatch.setattr(procedural_recorder.cognitive_config, "get_bool", lambda _: True)
    session = MagicMock()
    session.begin_nested.return_value = AsyncMock()
    session.flush = AsyncMock()
    client = MemoryClient(adapters(overrides=POLICY))
    try:
        await client.run_owned(
            lambda: asyncio.create_task(
                procedural_recorder.record_execution_trace(
                    session,
                    Chat(chat_id=1),
                    {"tool_calls": [{"name": "synthetic-secret", "input": {"https://example.test/private": 1}}]},
                    "open https://example.test/private synthetic-secret",
                )
            )
        )
        rows = [call.args[0] for call in session.add.call_args_list]
        execution = next(row for row in rows if isinstance(row, ExecutionTrace))
        assert execution.task_signature == "[url redacted]"
        assert execution.steps == [{"tool": "[url redacted]", "arg_keys": ["[url redacted]"]}]
        assert len(rows) == 2  # both execution trace and cognitive candidate were submitted
    finally:
        await client.close()


def _wall_clock_calls(
    source, *, module="", is_package=False, allowed_functions=(), allowed_dynamic_sql=(), persistence_defaults=False
):
    """Resolve datetime imports including aliases, so spelling cannot evade the gate."""
    import ast
    import re

    tree = ast.parse(source)
    tree.body = [
        node for node in tree.body if not (isinstance(node, ast.FunctionDef) and node.name in allowed_functions)
    ]
    storage_base = "kai_memory.storage.models.base"
    reexports = {
        f"{storage_base}.text": "sqlalchemy.text",
        f"{storage_base}.mapped_column": "sqlalchemy.orm.mapped_column",
    }
    reexports.update(
        {key.replace("kai_memory", "kai_memory"): value for key, value in reexports.copy().items()}
    )
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name] = item.name
        elif isinstance(node, ast.ImportFrom):
            target_module = node.module or ""
            if node.level:
                # Resolve against the scanned module's actual containing package,
                # including __init__.py and imports spanning multiple parents.
                package = module.split(".") if is_package else module.split(".")[:-1]
                if not module or node.level > len(package):
                    continue
                package = package[: len(package) - node.level + 1]
                target_module = ".".join([*package, *([target_module] if target_module else [])])
            for item in node.names:
                aliases[item.asname or item.name] = f"{target_module}.{item.name}"

    def qualified(node):
        if isinstance(node, ast.Name):
            name = aliases.get(node.id, node.id)
        elif isinstance(node, ast.Attribute):
            name = f"{qualified(node.value)}.{node.attr}"
        else:
            return ""
        return reexports.get(name, name)

    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

    def scope(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                return node
        return tree

    def static_sql(node, before=None):
        """Conservatively inspect literals and their local assignments, not execute Python.

        Keep every prior assignment as a possibility (including conditional writes).
        Anything beyond names and string addition fails closed at the text() call.
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = static_sql(node.left), static_sql(node.right)
            return None if left is None or right is None else {a + b for a in left for b in right}
        if not isinstance(node, ast.Name):
            return None
        cutoff = node.lineno if before is None else before
        writes = [
            item
            for item in ast.walk(scope(node))
            if isinstance(item, (ast.Assign, ast.AnnAssign, ast.AugAssign))
            and scope(item) is scope(node)
            and item.lineno < cutoff
            and any(
                isinstance(target, ast.Name) and target.id == node.id
                for target in (item.targets if isinstance(item, ast.Assign) else [item.target])
            )
        ]
        if not writes:
            return None
        values = set()
        for item in writes:
            parts = static_sql(item.value)
            if isinstance(item, ast.AugAssign):
                previous = static_sql(node, before=item.lineno)
                if not isinstance(item.op, ast.Add) or previous is None or parts is None:
                    return None
                parts = {a + b for a in previous for b in parts}
            if parts is None:
                return None
            values.update(parts)
        return values

    banned = {"datetime.datetime.now", "datetime.datetime.utcnow", "datetime.datetime.today", "datetime.date.today"}
    banned.update({"sqlalchemy.func.now", "sqlalchemy.func.current_date", "sqlalchemy.func.current_timestamp"})
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = qualified(node.func)
        if name in banned:
            violations.append(node.lineno)
        elif name == "sqlalchemy.text":
            parent = parents.get(node)
            if (
                persistence_defaults
                and isinstance(parent, ast.keyword)
                and parent.arg == "server_default"
                and isinstance(parents.get(parent), ast.Call)
                and qualified(parents[parent].func) == "sqlalchemy.orm.mapped_column"
            ):
                continue
            if len(node.args) == 1 and not node.keywords:
                sql_argument = node.args[0]
            elif not node.args and len(node.keywords) == 1 and node.keywords[0].arg == "text":
                sql_argument = node.keywords[0].value
            else:
                # Missing, ambiguous, or unpacked arguments cannot be inspected safely.
                violations.append(node.lineno)
                continue
            queries = static_sql(sql_argument)
            if queries is None:
                if (getattr(scope(node), "name", None), ast.unparse(sql_argument)) not in allowed_dynamic_sql:
                    violations.append(node.lineno)
            elif any(
                re.search(r"\bnow\s*\(\s*\)|\b(?:CURRENT_DATE|CURRENT_TIMESTAMP)\b", sql, re.I) for sql in queries
            ):
                violations.append(node.lineno)
    return violations


def _operational_sql_exceptions(relative_path):
    import ast

    checksum = "f\"SELECT count(*), coalesce(string_agg(t::text, '|' ORDER BY t::text), '') FROM {quoted} AS t\""
    expressions = {
        "eval/disposable_db.py": (
            ("create_schema", "ddl"),
            ("_require_empty", 'f"SELECT count(*) FROM {table}"'),
            ("session", 'f"SELECT count(*) FROM {table}"'),
            ("_fingerprint", checksum),
        ),
        "eval/runner.py": (("store_fingerprint", checksum),),
    }
    return {(name, ast.unparse(ast.parse(sql, mode="eval").body)) for name, sql in expressions.get(relative_path, ())}


def test_semantic_memory_time_cannot_bypass_host_clock():
    from pathlib import Path

    root = Path(__file__).parents[1] / "src/kai_memory"
    # SDK's real-clock default and operational benchmark folder names are the only
    # datetime wall-clock exceptions. Elapsed timers/cache TTL use time.time and
    # intentionally continue to advance when a host pins its semantic clock.
    exceptions = {"benchmark_io.py"}
    # Only these operational schema/count/checksum expressions may be dynamic.
    # Exact expressions keep the rest of this eval module under the semantic gate.
    violations = {
        str(path.relative_to(root)): lines
        for path in root.rglob("*.py")
        if str(path.relative_to(root)) not in exceptions
        and (
            lines := _wall_clock_calls(
                path.read_text(),
                module=".".join(path.relative_to(root.parent).with_suffix("").parts[:-1])
                if path.name == "__init__.py"
                else ".".join(path.relative_to(root.parent).with_suffix("").parts),
                is_package=path.name == "__init__.py",
                allowed_functions=("_utc_now",)
                if path.name == "sdk.py"
                else (("utc_now",) if path.relative_to(root).as_posix() == "storage/models/base.py" else ()),
                # Existing ORM persistence defaults remain real UTC; semantic operations use host time.
                persistence_defaults=path.relative_to(root).as_posix().startswith("storage/models/"),
                allowed_dynamic_sql=_operational_sql_exceptions(path.relative_to(root).as_posix()),
            )
        )
    }
    assert not violations, violations


@pytest.mark.parametrize(
    "source",
    [
        "from datetime import datetime as dt\ndt.now()",
        "import datetime as dates\ndates.datetime.utcnow()",
        "from datetime import date\ndate.today()",
        'from sqlalchemy import text\ntext("SELECT now()")',
        "from sqlalchemy import func\nfunc.current_date()",
        'from sqlalchemy import text as query\nquery("SELECT CURRENT_TIMESTAMP")',
        'import sqlalchemy as sa\nsa.text("SELECT now()")',
    ],
)
def test_clock_gate_detects_aliases(source):
    assert _wall_clock_calls(source) == [2]


@pytest.mark.parametrize(
    "body",
    [
        "sql = 'SELECT now()'\ntext(sql)",
        "sql = 'SELECT CURRENT_DATE'\nquery = sql\ntext(query)",
        "sql = 'SELECT ' + 'now()'\ntext(sql)",
        "sql = 'SELECT no'\nsql += 'w()'\ntext(sql)",
        "sql = 'SELECT :now'\nif condition:\n    sql += ', CURRENT_TIMESTAMP'\ntext(sql)",
        "sql = supplied_query()\ntext(sql)",
        "text(f'SELECT {expression}')",
        "sql = 'SELECT :now'\nif condition:\n    sql = supplied_query()\ntext(sql)",
    ],
)
def test_clock_gate_rejects_composed_or_uninspectable_sql(body):
    assert _wall_clock_calls("from sqlalchemy import text\n" + body)


@pytest.mark.parametrize(
    "body",
    [
        "text('SELECT :now')",
        "sql = 'SELECT :now'\ntext(sql)",
        "sql = 'SELECT :now'\nquery = sql\ntext(query)",
        "sql = 'SELECT :now'\nif condition:\n    sql += ' WHERE created_at > :since'\nsql += ' ORDER BY created_at'\ntext(sql)",
    ],
)
def test_clock_gate_preserves_static_bound_sql(body):
    assert not _wall_clock_calls("from sqlalchemy import text\n" + body)


@pytest.mark.parametrize(
    "call",
    [
        "text(text='SELECT now()')",
        "text(text='SELECT CURRENT_DATE')",
        "text(text=supplied_query())",
        "text(text=unknown_query)",
        "text(**kwargs)",
        "text(text='SELECT :now', **kwargs)",
        "text('SELECT :now', **kwargs)",
        "text(query='SELECT :now')",
    ],
)
def test_clock_gate_rejects_unsafe_or_unknown_keyword_sql(call):
    assert _wall_clock_calls("from sqlalchemy import text\n" + call) == [2]


@pytest.mark.parametrize("call", ["text(text='SELECT :now')", "text(text=sql)"])
def test_clock_gate_preserves_static_keyword_sql(call):
    assert not _wall_clock_calls("from sqlalchemy import text\nsql = 'SELECT :now'\n" + call)


@pytest.mark.parametrize("argument", ["{sql}", "text={sql}"])
def test_clock_gate_dynamic_exception_is_scoped_to_function_and_expression(argument):
    source = """
from sqlalchemy import text
async def operational():
    text(ddl)
    text(other_query)
    text("SELECT now()")
async def semantic():
    text(ddl)
"""
    for sql in ("ddl", "other_query", '"SELECT now()"'):
        source = source.replace(f"text({sql})", f"text({argument.format(sql=sql)})")
    assert _wall_clock_calls(source, allowed_dynamic_sql={("operational", "ddl")}) == [5, 6, 8]


@pytest.mark.parametrize(
    "operation",
    [
        "pattern_expiry",
        "trait_decay",
        "trajectory_discovery",
        "composer_traits",
        "composer_stances",
        "composer_patterns",
    ],
)
async def test_semantic_sql_uses_the_clients_clock(operation):
    from kai_memory import Chat
    from kai_memory.artifacts.producers.trait_producer import TraitProducer
    from kai_memory.artifacts.producers.trajectory import TrajectoryProducer
    from kai_memory.persona_composer import PersonaComposer

    fixed = datetime(2040, 2, 3, tzinfo=UTC)
    session = MagicMock()
    result = MagicMock()
    result.rowcount = 0
    result.mappings.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=result)
    composer = PersonaComposer(session)
    operations = {
        "pattern_expiry": lambda: PatternProducer()._expire_stale(session),
        "trait_decay": lambda: TraitProducer()._demote_below_threshold(session),
        "trajectory_discovery": lambda: TrajectoryProducer().discover_subjects(session, None),
        "composer_traits": composer._fetch_active_traits,
        "composer_stances": lambda: composer._fetch_relevant_stances(Chat()),
        "composer_patterns": composer._fetch_recent_patterns,
    }
    client = MemoryClient(adapters(overrides=POLICY, clock=lambda: fixed))
    try:
        await client.run_owned(lambda: asyncio.create_task(operations[operation]()))
        sql, parameters = session.execute.call_args.args
        assert parameters["now"] == fixed
        assert "NOW()" not in str(sql)
        assert "CAST(:now AS timestamptz)" in str(sql)
    finally:
        await client.close()


def test_persistence_clock_exception_is_narrow_and_opt_in():
    source = (
        "from sqlalchemy import text\n"
        "from sqlalchemy.orm import mapped_column\n"
        "mapped_column(server_default=text('NOW()'))\n"
        "other(server_default=text('NOW()'))\n"
        "mapped_column(default=text('NOW()'))\n"
        "text('SELECT now()')\n"
    )
    assert sorted(_wall_clock_calls(source)) == [3, 4, 5, 6]
    assert sorted(_wall_clock_calls(source, persistence_defaults=True)) == [4, 5, 6]


_STORAGE_HELPER_IMPORTS = [
    ("from .base import text, mapped_column", "text", "mapped_column"),
    ("from .base import text as sql, mapped_column as column", "sql", "column"),
    (
        "from kai_memory.storage.models.base import text as sql, mapped_column as column",
        "sql",
        "column",
    ),
    ("from . import base as model", "model.text", "model.mapped_column"),
    (
        "from kai_memory.storage.models import base as model",
        "model.text",
        "model.mapped_column",
    ),
    (
        "import kai_memory.storage.models.base as model",
        "model.text",
        "model.mapped_column",
    ),
    (
        "import kai_memory.storage.models.base",
        "kai_memory.storage.models.base.text",
        "kai_memory.storage.models.base.mapped_column",
    ),
]


@pytest.mark.parametrize("imports,sql,column", _STORAGE_HELPER_IMPORTS)
@pytest.mark.parametrize("argument", ["'SELECT now()'", "text='SELECT now()'"])
def test_storage_reexports_cannot_bypass_clock_guard(imports, sql, column, argument):
    source = f"{imports}\n{sql}({argument})\n"
    context = {"module": "kai_memory.storage.models.example"}
    assert _wall_clock_calls(source, **context) == [2]
    assert _wall_clock_calls(source, persistence_defaults=True, **context) == [2]


@pytest.mark.parametrize("imports,sql,column", _STORAGE_HELPER_IMPORTS)
@pytest.mark.parametrize("argument", ["'now()'", "text='now()'"])
def test_storage_reexport_defaults_keep_only_narrow_opt_in_exception(imports, sql, column, argument):
    source = (
        f"{imports}\n"
        f"{column}(server_default={sql}({argument}))\n"
        f"{column}(default={sql}({argument}))\n"
        f"other(server_default={sql}({argument}))\n"
    )
    context = {"module": "kai_memory.storage.models.example"}
    assert sorted(_wall_clock_calls(source, **context)) == [2, 3, 4]
    assert sorted(_wall_clock_calls(source, persistence_defaults=True, **context)) == [3, 4]


@pytest.mark.parametrize(
    "module,is_package,imports,sql,column",
    [
        ("kai_memory.storage", True, "from .models.base import text, mapped_column", "text", "mapped_column"),
        ("kai_memory.storage.schema", False, "from .models import base as b", "b.text", "b.mapped_column"),
        (
            "kai_memory.storage.models.nested.model",
            False,
            "from ..base import text as sql, mapped_column as column",
            "sql",
            "column",
        ),
        ("kai_memory.storage.models.nested", True, "from .. import base as b", "b.text", "b.mapped_column"),
        (
            "kai_memory.storage.models.nested.deeper.model",
            False,
            "from ...base import text, mapped_column",
            "text",
            "mapped_column",
        ),
    ],
)
@pytest.mark.parametrize("argument", ["'now()'", "text='now()'"])
def test_package_relative_reexports_preserve_clock_guard(module, is_package, imports, sql, column, argument):
    source = (
        f"{imports}\n"
        f"{sql}({argument})\n"
        f"{column}(server_default={sql}({argument}))\n"
        f"{column}(default={sql}({argument}))\n"
        f"other(server_default={sql}({argument}))\n"
    )
    context = {"module": module, "is_package": is_package}
    assert sorted(_wall_clock_calls(source, **context)) == [2, 3, 4, 5]
    assert sorted(_wall_clock_calls(source, persistence_defaults=True, **context)) == [2, 4, 5]


@pytest.mark.parametrize("module,is_package", [("unrelated.model", False), ("unrelated", True)])
@pytest.mark.parametrize(
    "imports,sql,column",
    [
        ("from .base import text, mapped_column", "text", "mapped_column"),
        ("from . import base as b", "b.text", "b.mapped_column"),
    ],
)
def test_unrelated_relative_base_is_not_a_storage_reexport(module, is_package, imports, sql, column):
    context = {"module": module, "is_package": is_package}
    assert _wall_clock_calls(f"{imports}\n{sql}('now()')", **context) == []
    source = f"{imports}\nfrom sqlalchemy import text as sa_text\n{column}(server_default=sa_text('now()'))"
    assert _wall_clock_calls(source, persistence_defaults=True, **context) == [3]


def test_store_checksum_exception_cannot_authorize_other_dynamic_sql():
    source = """
from sqlalchemy import text
async def store_fingerprint():
    text(f"SELECT count(*), coalesce(string_agg(t::text, '|' ORDER BY t::text), '') FROM {quoted} AS t")
    text(f"SELECT count(*) FROM {quoted} WHERE created_at > NOW()")
    text(other_query)
async def semantic():
    text(f"SELECT count(*), coalesce(string_agg(t::text, '|' ORDER BY t::text), '') FROM {quoted} AS t")
"""
    assert _wall_clock_calls(source, allowed_dynamic_sql=_operational_sql_exceptions("eval/runner.py")) == [5, 6, 8]
    assert _wall_clock_calls(source, allowed_dynamic_sql=_operational_sql_exceptions("eval/other.py")) == [4, 5, 6, 8]
