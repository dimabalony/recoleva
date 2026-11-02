"""Read-only physical-schema revisions, independent of SDK/API versions.

Hosts own migrations. Historical unversioned stores are recognized by structure;
inspection never stamps a version or changes host Alembic state.
"""

from __future__ import annotations

import re
from decimal import Decimal

from sqlalchemy import Float, Integer, String
from dataclasses import dataclass

from sqlalchemy import MetaData, inspect, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine
from sqlalchemy.schema import CheckConstraint, UniqueConstraint

from . import metadata

M1_SCHEMA_REVISION = "memory-1"
CURRENT_SCHEMA_REVISION = "memory-2"
REPLAY_INDEX = "idx_exec_traces_assistant_message"


@dataclass(frozen=True)
class StorageCompatibility:
    revision: str | None
    issues: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return self.revision == CURRENT_SCHEMA_REVISION and not self.issues


class IncompatibleStorageError(RuntimeError):
    """The host must explicitly migrate or repair this storage."""


def _sql(value: object, table_name: str | None = None, *, text_columns=frozenset(), allow_text_literals=False):
    """Parse the small expression grammar used by this schema, failing closed.

    PostgreSQL deparses IN as = ANY(ARRAY[]) and adds text casts. Literal bytes
    and operator precedence remain significant; unsupported grammar is rejected.
    """
    if value is None:
        return None
    source = str(value)
    pattern = (
        r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|->>|::|<>|<=|>=|"
        r"[a-zA-Z_][a-zA-Z_0-9.]*|[0-9]+(?:\.[0-9]+)?|[(),\[\]=<>]"
    )
    tokens = re.findall(pattern, source)
    remainder = re.sub(pattern, "", source)
    if remainder.strip():
        raise ValueError("Unsupported schema expression")
    position = 0

    def peek():
        return tokens[position].lower() if position < len(tokens) else ""

    def take(expected=None):
        nonlocal position
        if position == len(tokens) or (expected is not None and peek() != expected):
            raise ValueError("Unsupported schema expression")
        token = tokens[position]
        position += 1
        return token

    def parse(minimum=0):
        token = take()
        if token == "(":
            left = parse()
            take(")")
        elif token.lower() == "array":
            take("[")
            items = [parse()]
            while peek() == ",":
                take(",")
                items.append(parse())
            take("]")
            left = ("list", tuple(items))
        elif token.startswith("'"):
            left = ("literal", token)
        else:
            identifier = token if token.startswith('"') else token.lower()
            if table_name and not token.startswith('"'):
                for prefix in ("public." + table_name + ".", table_name + "."):
                    if identifier.startswith(prefix):
                        identifier = identifier.removeprefix(prefix)
            left = ("quoted identifier" if token.startswith('"') else "atom", identifier)
            if peek() == "(":
                take("(")
                arguments = [] if peek() == ")" else [parse()]
                while peek() == ",":
                    take(",")
                    arguments.append(parse())
                take(")")
                left = ("call", token if token.startswith('"') else token.lower(), tuple(arguments))
        while peek() == "::":
            take("::")
            cast = take().lower()
            if cast == "character":
                take("varying")
            elif cast not in {"text", "double", "jsonb"}:
                raise ValueError("Unsupported schema cast")
            if cast == "double":
                take("precision")
                cast = "double precision"
            array = peek() == "["
            if array:
                take("[")
                take("]")
            harmless_text = cast in {"text", "character"} and (
                left in {("atom", name) for name in text_columns}
                or left[0] == "->>"
                or (allow_text_literals and left[0] == "literal")
                or (allow_text_literals and array and left[0] == "list")
            )
            if not harmless_text:
                left = ("cast " + cast + ("[]" if array else ""), left)
        precedence = {"or": 1, "and": 2, "=": 3, "<>": 3, "<": 3, ">": 3, "<=": 3, ">=": 3, "in": 3, "is": 3, "->>": 4}
        while peek() in precedence and precedence[peek()] >= minimum:
            operator = take().lower()
            if operator == "is":
                negate = peek() == "not"
                if negate:
                    take("not")
                take("null")
                left = ("is not null" if negate else "is null", left)
            elif operator == "in":
                take("(")
                items = [parse()]
                while peek() == ",":
                    take(",")
                    items.append(parse())
                take(")")
                left = ("in", left, tuple(items))
            elif operator == "=" and peek() == "any":
                take("any")
                take("(")
                right = parse()
                take(")")
                if right[0] != "list":
                    raise ValueError("Unsupported ANY expression")
                left = ("in", left, right[1])
            else:
                left = (operator, left, parse(precedence[operator] + 1))
        return left

    result = parse()
    if peek() in {"asc", "desc"}:
        result = (take().lower(), result)
    if position != len(tokens):
        raise ValueError("Unsupported schema expression")
    return result


def _shape(table, dialect):
    def expression(value):
        compiled = (
            value if isinstance(value, str) else value.compile(dialect=dialect, compile_kwargs={"literal_binds": True})
        )
        result = _sql(
            compiled,
            table.name,
            text_columns={c.name for c in table.columns if isinstance(c.type, String)},
            allow_text_literals=True,
        )
        float_columns = {c.name for c in table.columns if isinstance(c.type, Float)}

        def normalize_float_comparison(node):
            if not isinstance(node, tuple):
                return node
            if (
                len(node) == 3
                and node[0] in {"<", ">", "<=", ">=", "=", "<>"}
                and node[1] in {("atom", name) for name in float_columns}
                and node[2][0] == "cast double precision"
                and node[2][1][0] == "atom"
                and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", node[2][1][1])
            ):
                return (node[0], node[1], node[2][1])
            return tuple(normalize_float_comparison(item) for item in node)

        return normalize_float_comparison(result)

    def server_default(column):
        if column.server_default is None:
            return None
        sql = dialect.ddl_compiler(dialect, None).get_column_default_string(column)
        result = _sql(sql, allow_text_literals=isinstance(column.type, String))
        if isinstance(column.type, (Float, Integer)):
            constant = result
            if constant[0] == "cast double precision" and isinstance(column.type, Float):
                constant = constant[1]
            if constant[0] in {"literal", "atom"}:
                try:
                    return ("numeric constant", str(Decimal(constant[1].strip("'")).normalize()))
                except Exception:
                    pass
        return result

    return {
        "columns": {
            c.name: (
                "DOUBLE PRECISION"
                if str(c.type.compile(dialect=dialect)) == "FLOAT"
                else str(c.type.compile(dialect=dialect)),
                c.nullable,
            )
            for c in table.columns
        },
        "server_defaults": {c.name: server_default(c) for c in table.columns},
        "primary_key": tuple(c.name for c in table.primary_key),
        "foreign_keys": {
            (
                tuple(e.parent.name for e in fk.elements),
                fk.referred_table.name,
                fk.referred_table.schema or "public",
                tuple(e.column.name for e in fk.elements),
                fk.ondelete,
                fk.onupdate,
                bool(fk.deferrable),
                fk.initially,
                fk.match,
            )
            for fk in table.foreign_key_constraints
        },
        "unique": {
            (
                tuple(c.name for c in constraint.columns),
                bool(constraint.dialect_options["postgresql"].get("nulls_not_distinct")),
            )
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint)
        },
        "checks": {expression(c.sqltext) for c in table.constraints if isinstance(c, CheckConstraint)},
        "indexes": {
            i.name: (
                bool(i.unique),
                tuple(expression(e) for e in i.expressions),
                expression(i.dialect_options["postgresql"]["where"])
                if i.dialect_options["postgresql"].get("where") is not None
                else None,
                i.dialect_options["postgresql"].get("using") or "btree",
                tuple(sorted(i.dialect_options["postgresql"].get("ops", {}).items())),
                tuple(i.dialect_options["postgresql"].get("include") or ()),
                tuple(sorted(i.dialect_options["postgresql"].get("with", {}).items())),
                i.dialect_options["postgresql"].get("nulls_not_distinct"),
            )
            for i in table.indexes
            if not i.info.get("duplicates_constraint")
        },
    }


def _inspect(conn) -> StorageCompatibility:
    names = set(inspect(conn).get_table_names(schema="public"))
    missing = set(metadata.tables) - names
    issues = [f"missing table: public.{name}" for name in sorted(missing)]
    reflected = MetaData()
    reflected.reflect(conn, schema="public", only=sorted(set(metadata.tables) - missing))
    opclasses = conn.execute(
        text("""
        SELECT t.relname, idx.relname, a.attname, opc.opcname
        FROM pg_index i JOIN pg_class t ON t.oid=i.indrelid
        JOIN pg_namespace n ON n.oid=t.relnamespace
        JOIN pg_class idx ON idx.oid=i.indexrelid
        CROSS JOIN LATERAL unnest(i.indclass::oid[], i.indkey::smallint[]) AS keys(opcoid, attnum)
        JOIN pg_opclass opc ON opc.oid=keys.opcoid
        JOIN pg_attribute a ON a.attrelid=t.oid AND a.attnum=keys.attnum
        WHERE n.nspname='public' AND NOT opc.opcdefault
    """)
    ).all()
    for table_name, index_name, column_name, opclass in opclasses:
        table = reflected.tables.get("public." + table_name)
        if table is not None:
            for index in table.indexes:
                if index.name == index_name:
                    index.dialect_options["postgresql"]["ops"] = {
                        **index.dialect_options["postgresql"].get("ops", {}),
                        column_name: opclass,
                    }
    replay_present = False
    for name, expected in metadata.tables.items():
        if name in missing:
            continue
        try:
            want = _shape(expected, conn.dialect)
            actual = _shape(reflected.tables["public." + name], conn.dialect)
        except ValueError:
            issues.append(f"unsupported expression: public.{name}")
            continue
        if name == "execution_traces":
            replay_present = REPLAY_INDEX in actual["indexes"]
            if not replay_present:
                want["indexes"].pop(REPLAY_INDEX)
        for component in want:
            if want[component] != actual[component]:
                issues.append(f"{component} differ: public.{name}")
    invalid = (
        conn.execute(
            text("""
        SELECT 'index ' || c.relname FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid
        JOIN pg_class t ON t.oid=i.indrelid JOIN pg_namespace n ON n.oid=t.relnamespace
        WHERE n.nspname='public' AND t.relname = ANY(:tables)
          AND (NOT i.indisvalid OR NOT i.indisready)
        UNION ALL
        SELECT 'constraint ' || con.conname FROM pg_constraint con
        JOIN pg_class t ON t.oid=con.conrelid JOIN pg_namespace n ON n.oid=t.relnamespace
        WHERE n.nspname='public' AND t.relname = ANY(:tables)
          AND (NOT con.convalidated OR (con.contype IN ('p', 'u') AND con.condeferrable)
               OR (con.contype='c' AND con.connoinherit))
    """),
            {"tables": list(metadata.tables)},
        )
        .scalars()
        .all()
    )
    issues.extend(f"invalid storage object: public.{name}" for name in invalid)
    revision = None if issues else CURRENT_SCHEMA_REVISION if replay_present else M1_SCHEMA_REVISION
    return StorageCompatibility(revision, tuple(issues))


async def inspect_storage(connection: AsyncEngine | AsyncConnection) -> StorageCompatibility:
    """Inspect public storage without DDL or access to the host migration ledger.

    Extra host tables are allowed; unknown memory columns/constraints/indexes
    fail closed. Shape checks include pgvector dimensions and index validity.
    Stored data and function bodies remain host responsibilities.
    """
    if connection.dialect.name != "postgresql":
        raise ValueError("Memory storage requires PostgreSQL with pgvector")
    if isinstance(connection, AsyncEngine):
        async with connection.connect() as conn:
            return await conn.run_sync(_inspect)
    return await connection.run_sync(_inspect)


async def require_compatible_storage(connection: AsyncEngine | AsyncConnection) -> StorageCompatibility:
    """Require current physical schema; never automatically migrate storage."""
    report = await inspect_storage(connection)
    if not report.compatible:
        raise IncompatibleStorageError(f"Host migration required: {report.revision or 'unknown'}; {report.issues}")
    return report
