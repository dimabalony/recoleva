"""Explicit local server; every SDK contract gets a newly created database."""

import ipaddress
import os
import socket
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from kai_memory.storage import bootstrap_disposable
from kai_memory.storage.disposable import check_url_is_local


@pytest_asyncio.fixture(loop_scope="session")
async def sdk_empty_db_url():
    url = os.environ.get("MEMORY_SDK_TEST_DATABASE_URL")
    if not url:
        pytest.skip("set MEMORY_SDK_TEST_DATABASE_URL to a disposable local PostgreSQL server")
    check_url_is_local(url, production_url=os.environ.get("DATABASE_URL"))
    admin = create_async_engine(url, isolation_level="AUTOCOMMIT", poolclass=NullPool)
    name = f"memory_sdk_test_{uuid.uuid4().hex}"
    created = False
    try:
        async with admin.connect() as connection:
            await connection.execute(text(f'CREATE DATABASE "{name}"'))
            created = True
        test_url = make_url(url).set(database=name).render_as_string(hide_password=False)
        yield test_url
    finally:
        if created:
            async with admin.connect() as connection:
                await connection.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        await admin.dispose()


@pytest_asyncio.fixture(loop_scope="session")
async def sdk_db_url(sdk_empty_db_url):
    engine = create_async_engine(sdk_empty_db_url, poolclass=NullPool)
    try:
        await bootstrap_disposable(engine)
    finally:
        await engine.dispose()
    return sdk_empty_db_url


@pytest.fixture(autouse=True)
def loopback_only(monkeypatch):
    """Synthetic contracts cannot reach a remote inference or database service."""
    attempts = []
    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex

    def guard(sock, address):
        if sock.family not in (socket.AF_INET, socket.AF_INET6):
            return
        host = address[0]
        try:
            local = host == "localhost" or ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = False
        if not local:
            attempts.append(address)
            raise ConnectionRefusedError("SDK proof attempted a non-loopback connection")

    def guarded_connect(sock, address):
        guard(sock, address)
        return connect(sock, address)

    def guarded_connect_ex(sock, address):
        guard(sock, address)
        return connect_ex(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    yield
    assert not attempts, "SDK proof attempted a non-loopback connection"
