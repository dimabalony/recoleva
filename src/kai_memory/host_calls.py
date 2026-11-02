"""Call-time host calls: memory's typed way to use a function port (DB-854).

A memory module imports these wrappers, never a port value. Each wrapper reads its
port from ``kai_memory.ports`` on every call, so:

* a host may bind its adapters after memory is imported (``ports.bind`` or
  ``MemoryClient``), and every later call sees the binding;
* a test that patches the memory module's attribute (``memory_extraction.utc_now``)
  replaces the wrapper and still takes effect, as it did when the name was an
  eager import. Each wrapper is sync or ``async def`` exactly as its host function
  is, so ``patch()`` still builds the right kind of mock;
* a test that patches the host's function (``app.services.chat.llm_call.call_llm_text``)
  now reaches memory too, because the port is looked up at call time.

Classes and constants (``ChatContext``, ``CancelledByNewMessage``,
``MODEL_METADATA_KEY``) are read as ``ports.X`` where they are used.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from kai_memory import ports

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


# ── time ────────────────────────────────────────────────────────────────────


def utc_now() -> datetime:
    """The current time from the clock port (timezone-aware UTC).

    A host clock that returns a naive datetime fails here, not later in timestamp math.
    """
    now = cast(datetime, ports.clock())
    if now.tzinfo is None:
        raise ValueError("memory clock port returned a naive datetime; it must be timezone-aware UTC")
    return now


# ── configuration ───────────────────────────────────────────────────────────


def get_host_policy() -> Any:
    return ports.get_host_policy()


def get_model_roles() -> Any:
    return ports.get_model_roles()


def get_settings() -> Any:
    return ports.get_settings()


# ── infrastructure ──────────────────────────────────────────────────────────


def session_scope() -> AbstractAsyncContextManager[AsyncSession]:
    """A committing session from the storage port (``async with session_scope() as s``)."""
    return cast("AbstractAsyncContextManager[AsyncSession]", ports.session_scope())


async def recover_session(session: AsyncSession) -> None:
    await ports.recover_session(session)


# ── inference ───────────────────────────────────────────────────────────────


async def call_llm_text(provider: Any, prompt: str, *args: Any, **kwargs: Any) -> str | None:
    # async def, like the host function: ``patch()`` builds an AsyncMock only for one.
    return cast("str | None", await ports.call_llm_text(provider, prompt, *args, **kwargs))


def get_provider_for_model(settings: Any, model_name: str) -> Any:
    return ports.get_provider_for_model(settings, model_name)


# ── observability ───────────────────────────────────────────────────────────


def start_generation_span(name: str, model: str) -> Any:
    """A generation span from the telemetry port (None when telemetry is off)."""
    return ports.telemetry.span(name, model)


# ── parsing ─────────────────────────────────────────────────────────────────


def parse_json_object_response(raw: str) -> dict[str, Any]:
    return cast("dict[str, Any]", ports.parse_json_object_response(raw))


def parse_json_array_response(raw: str) -> list[Any]:
    return cast("list[Any]", ports.parse_json_array_response(raw))


# ── text hygiene ────────────────────────────────────────────────────────────


def scrub_secrets(text: str) -> str:
    return cast(str, ports.scrub_secrets(text))


def validate_skill_content(instructions: str) -> list[str]:
    return cast("list[str]", ports.validate_skill_content(instructions))


def strip_msg_meta_envelope(text: str) -> str:
    return cast(str, ports.strip_msg_meta_envelope(text))


def strip_echoed_metadata(text: str | None) -> str:
    return cast(str, ports.strip_echoed_metadata(text))


def redact_urls_for_trace(text: str | None) -> str | None:
    return cast("str | None", ports.redact_urls_for_trace(text))


# ── trust ───────────────────────────────────────────────────────────────────


def is_admin_user(user_id: int | None, admin_ids_str: str | None) -> bool:
    return bool(ports.is_admin_user(user_id, admin_ids_str))
