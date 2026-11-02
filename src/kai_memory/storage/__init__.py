"""Explicit schema composition and fresh disposable PostgreSQL initialization.

This is not a migration API. Hosts own the engine, sessions and schema upgrades.
Importing this module only constructs metadata; it never opens a connection.
"""

from __future__ import annotations

from sqlalchemy import DDL, MetaData, event, text
from sqlalchemy.ext.asyncio import AsyncEngine

from . import models
from .models.base import Base as MemoryBase
from .models.memory_entity import ALIASES_LOWER_DDL

metadata = MemoryBase.metadata
MEMORY_TABLES = frozenset(metadata.tables)


class UnsafeStorageError(RuntimeError):
    """Disposable initialization requires a fresh database with no user objects."""


def compose_metadata(target: MetaData) -> MetaData:
    """Copy memory tables into a host registry, preserving schema creation hooks.

    Repeated composition into the same registry is safe. Pre-existing definitions
    are rejected: choosing which contract wins must never be implicit.
    """
    if target is metadata:
        return target
    for name in MEMORY_TABLES:
        if name in target.tables and target.tables[name].info.get("memory_schema_source") is not metadata.tables[name]:
            raise ValueError(f"Conflicting memory table definition: {name}")
    for table in metadata.sorted_tables:
        if table.key in target.tables:
            continue
        clone = table.to_metadata(target)
        clone.info["memory_schema_source"] = table
        if table.name == "memory_entities":
            event.listen(clone, "before_create", DDL(ALIASES_LOWER_DDL))
    return target


# Namespace dependencies cover tables, views, sequences, routines, types,
# operators, collations, etc. Follow internal/automatic dependencies of extension
# members too (array types and composite row types are not direct members).
_USER_OBJECTS = text("""
    WITH RECURSIVE extension_objects(classid, objid) AS (
        SELECT 'pg_extension'::regclass, oid FROM pg_extension
        UNION
        SELECT d.classid, d.objid FROM pg_depend d
        JOIN extension_objects e ON d.refclassid = e.classid AND d.refobjid = e.objid
        WHERE d.deptype IN ('e', 'i', 'a')
    ), user_namespaces AS (
        SELECT oid, nspname FROM pg_namespace
        WHERE nspname NOT IN ('pg_catalog', 'information_schema')
          AND nspname NOT LIKE 'pg_toast%'
    )
    SELECT n.nspname || '.' || pg_describe_object(d.classid, d.objid, d.objsubid) AS name
    FROM pg_depend d JOIN user_namespaces n
      ON d.refclassid = 'pg_namespace'::regclass AND d.refobjid = n.oid
    WHERE NOT EXISTS (
        SELECT 1 FROM extension_objects e WHERE e.classid = d.classid AND e.objid = d.objid
    )
    UNION
    SELECT 'schema ' || n.nspname FROM user_namespaces n
    WHERE n.nspname <> 'public' AND NOT EXISTS (
        SELECT 1 FROM extension_objects e WHERE e.classid = 'pg_namespace'::regclass AND e.objid = n.oid
    )
    ORDER BY 1
""")


async def bootstrap_disposable(engine: AsyncEngine, *, schema: MetaData | None = None) -> None:
    """Initialize a fresh disposable PostgreSQL database, atomically.

    ``schema`` may explicitly compose host-owned tables with ``compose_metadata``.
    Existing user objects cause refusal before any DDL. The caller remains owner
    of the engine; this function never disposes it or reads host configuration.
    """
    selected = metadata if schema is None else schema
    if engine.dialect.name != "postgresql":
        raise ValueError("Memory storage requires PostgreSQL with pgvector")
    if not MEMORY_TABLES.issubset(selected.tables):
        raise ValueError("Schema must contain all memory tables")
    async with engine.begin() as conn:
        # Serialize cooperating initializers; a second call rechecks after the first commits.
        await conn.execute(text("SELECT pg_advisory_xact_lock(862741)"))
        objects = (await conn.execute(_USER_OBJECTS)).scalars().all()
        if objects:
            raise UnsafeStorageError(f"Refusing non-empty disposable storage: {', '.join(objects[:5])}")
        await conn.execute(text("SET LOCAL search_path TO public"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public"))
        await conn.run_sync(selected.create_all)


__all__ = [
    "MEMORY_TABLES",
    "MemoryBase",
    "UnsafeStorageError",
    "bootstrap_disposable",
    "compose_metadata",
    "metadata",
    "models",
]
