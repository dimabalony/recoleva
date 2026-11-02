"""Loopback-only URL guard for disposable storage proof and evaluation."""

from __future__ import annotations

from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
# URL query keys that override the host, port, or database validated below.
_CONNECTION_TARGET_KEYS = frozenset({"host", "hostaddr", "port", "service", "database", "dbname"})


class UnsafeDatabaseError(RuntimeError):
    """The eval was pointed at a database it must not touch."""


_POSTGRES_DEFAULT_PORT = 5432


def _database_key(url: str) -> tuple[str, int, str]:
    """(host, port, database) with spelling differences removed: every loopback alias is
    one host and an omitted port is PostgreSQL's default, so two URLs naming the same
    local database compare equal."""
    parsed = make_url(url)
    host = (parsed.host or "").lower()
    if host in LOCAL_HOSTS:
        host = "localhost"
    return (host, parsed.port or _POSTGRES_DEFAULT_PORT, parsed.database or "")


def check_url_is_local(url: str, *, production_url: str | None) -> None:
    """Refuse any URL that is not an explicit local host, or that names the app's database."""
    try:
        parsed = make_url(url)
    except ArgumentError as exc:
        raise UnsafeDatabaseError(f"unparseable database URL: {exc}") from None
    host = (parsed.host or "").lower()
    # Drivers read connection targets from the query string, and those override the
    # host or database checked below (asyncpg: ?host=, ?port=, ?database=).
    overriding = sorted(k for k in parsed.query if k.lower() in _CONNECTION_TARGET_KEYS)
    if overriding:
        raise UnsafeDatabaseError(f"refusing connection-target parameters in the URL query: {overriding}")
    if host.endswith(".rds.amazonaws.com"):
        raise UnsafeDatabaseError(f"refusing an RDS host ({host}); the memory eval runs on a local database only")
    if production_url:
        try:
            same = _database_key(url) == _database_key(production_url)
        except ArgumentError:
            same = False
        if same:
            raise UnsafeDatabaseError("refusing the configured application DATABASE_URL; point the eval elsewhere")
    if host not in LOCAL_HOSTS:
        raise UnsafeDatabaseError(
            f"refusing non-local host {host or '<unix socket>'!r}; allowed: {sorted(LOCAL_HOSTS)}"
        )
