"""Host-neutral memory ports.

Resolution is scoped client adapters, explicit process bindings, then an optional
host-installed compatibility resolver. Importing the package never configures a
host. Independent hosts use MemoryClient and HostAdapters.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_HOST_SYMBOLS = frozenset(
    (
        "session_scope",
        "recover_session",
        "SessionLocal",
        "init_engine",
        "get_settings",
        "get_agent_identity",
        "get_host_policy",
        "get_model_roles",
        "call_llm_text",
        "get_provider_for_model",
        "get_provider",
        "record_generation_provenance",
        "clock",
        "telemetry",
        "parse_json_array_response",
        "parse_json_object_response",
        "scrub_secrets",
        "validate_skill_content",
        "strip_msg_meta_envelope",
        "strip_echoed_metadata",
        "redact_urls_for_trace",
        "is_admin_user",
        "ChatContext",
        "ConversationManager",
        "CancelledByNewMessage",
        "MODEL_METADATA_KEY",
        "invalidate_prompt_cache",
        "embedder",
        "read_conversation_page",
        "senders_seen_before",
        "delete_conversation_messages",
    )
)

__all__ = sorted([*_HOST_SYMBOLS, "UNBOUND", "bind", "scoped", "set_default_resolver"])


class _Unbound:
    def __repr__(self) -> str:
        return "UNBOUND"


UNBOUND: Any = _Unbound()
_BOUND: dict[str, Any] = {}
_SCOPED: ContextVar[Mapping[str, Any] | None] = ContextVar("memory_ports_scoped", default=None)


_DEFAULT_RESOLVER: Callable[[str], Any] | None = None


def set_default_resolver(resolver: Callable[[str], Any] | None) -> Callable[[str], Any] | None:
    """Install a host's lazy compatibility fallback; scoped and bound ports always win."""
    global _DEFAULT_RESOLVER
    previous = _DEFAULT_RESOLVER
    _DEFAULT_RESOLVER = resolver
    return previous


def _check_names(symbols: Mapping[str, Any]) -> None:
    unknown = sorted(set(symbols) - set(_HOST_SYMBOLS))
    if unknown:
        raise KeyError(f"not memory host ports: {unknown}; the contract is {sorted(_HOST_SYMBOLS)}")


def bind(**symbols: Any) -> dict[str, Any]:
    """Install the process-default host adapters by contract name; returns each name's previous binding.

    Binding ``UNBOUND`` removes a binding, so ``bind(**previous)`` puts the table back.
    A ``scoped`` block's bindings win over these.
    """
    _check_names(symbols)
    previous = {name: _BOUND.get(name, UNBOUND) for name in symbols}
    for name, value in symbols.items():
        if value is UNBOUND:
            _BOUND.pop(name, None)
        else:
            _BOUND[name] = value
    return previous


@contextmanager
def scoped(bindings: Mapping[str, Any], *, inherit: bool = False) -> Iterator[None]:
    """Resolve ports in this context and its tasks.

    ``inherit`` layers a narrow override onto the enclosing scope; the default
    starts a separate host scope so independent clients never inherit adapters.
    """
    _check_names(bindings)
    token = _SCOPED.set({**(_SCOPED.get() or {}), **bindings} if inherit else bindings)
    try:
        yield
    finally:
        _SCOPED.reset(token)


def __getattr__(name: str) -> Any:
    scope = _SCOPED.get()
    if scope is not None and name in scope:
        return scope[name]
    if name in _BOUND:
        return _BOUND[name]
    if name not in _HOST_SYMBOLS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if _DEFAULT_RESOLVER is not None:
        return _DEFAULT_RESOLVER(name)
    raise RuntimeError(f"Memory host port {name!r} is unbound; supply HostAdapters through MemoryClient")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
