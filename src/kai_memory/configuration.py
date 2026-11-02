"""Host-owned model routing and authority; no service configuration is required."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelRoles:
    """Existing inference routes; hosts select their own model names.

    Procedural judgments share the L2 extraction call and therefore its role.
    Deterministic recording, capture and serving do not add a model call.
    """

    extraction: str = "haiku-5.5"
    consolidation: str = "sonnet-5.5"
    artifact: str = "sonnet-5.5"
    auxiliary: str = "haiku-5.5"

    def __post_init__(self) -> None:
        for value in vars(self).values():
            if not isinstance(value, str) or not value.strip():
                raise ValueError("model roles require nonempty model names")


class HostPolicy(Protocol):
    """Host-verified authority. IDs asserted by inference never grant permission.

    Chat is an authorization context, not a storage or tenant partition. In
    particular, permitting artifact export authorizes the store-wide collection.
    Independent security domains require independent stores and clients.
    """

    def is_admin(self, actor_id: int | None) -> bool: ...
    def respect(self, actor_id: int | None) -> int: ...
    def may_read(self, chat: Any) -> bool: ...
    def may_write(self, chat: Any) -> bool: ...


@dataclass(frozen=True)
class NeutralHostPolicy:
    """One host-owned store: allow its scopes, grant nobody teaching authority.

    Chat identifiers are not a tenant boundary. Multi-tenant hosts must supply
    their own scope checks before exposing a client to callers.
    """

    def is_admin(self, actor_id: int | None) -> bool:
        return False

    def respect(self, actor_id: int | None) -> int:
        return 0

    def may_read(self, chat: Any) -> bool:
        return True

    def may_write(self, chat: Any) -> bool:
        return True
