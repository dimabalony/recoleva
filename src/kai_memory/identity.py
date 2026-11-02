"""Configured agent identity, independent of mutable display names."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory import ports
from kai_memory.storage.models.memory_entity import MemoryEntity

AGENT_IDENTITY_DATA_KEY = "_agent_identity_id"


def identity_merge_survivor(first: MemoryEntity, second: MemoryEntity) -> MemoryEntity | None:
    """Preserve a sole bound UUID; conflicting persisted identities require review."""
    bound = [
        entity
        for entity in (first, second)
        if isinstance(entity.data, dict) and entity.data.get(AGENT_IDENTITY_DATA_KEY)
    ]
    if len(bound) == 2:
        raise ValueError("cannot merge two bound agent identities")
    if not bound:
        return None
    survivor = bound[0]
    other = second if survivor is first else first
    if other.telegram_user_id is not None:
        raise ValueError("cannot merge bound agent identity with a Telegram user")
    return survivor


@dataclass(frozen=True)
class AgentIdentity:
    """Host-owned stable identity and names recognized as the assistant.

    ``legacy_names`` explicitly permits adoption of unbound historical agent rows.
    Names and aliases alone never authorize adoption of another identity.
    """

    stable_id: str
    display_name: str
    aliases: tuple[str, ...] = ()
    legacy_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.aliases, str) or isinstance(self.legacy_names, str):
            raise ValueError("identity aliases and legacy_names must be sequences of names")
        for label, values in (
            ("stable_id", (self.stable_id,)),
            ("display_name", (self.display_name,)),
            ("aliases", self.aliases),
            ("legacy_names", self.legacy_names),
        ):
            if any(not isinstance(value, str) or not value.strip() or value != value.strip() for value in values):
                raise ValueError(f"identity {label} must contain nonempty, trimmed strings")
        if len(self.display_name) > 200:
            raise ValueError("identity display_name exceeds the entity name limit of 200 characters")
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "legacy_names", tuple(self.legacy_names))

    def recognizes(self, name: str) -> bool:
        return name.casefold() in {value.casefold() for value in (self.display_name, *self.aliases)}


def get_agent_identity() -> AgentIdentity:
    """Resolve the calling client's identity at operation time."""
    identity = ports.get_agent_identity()
    if not isinstance(identity, AgentIdentity):
        raise TypeError("host must supply an AgentIdentity")
    return identity


async def resolve_agent_identity(session: AsyncSession) -> MemoryEntity | None:
    """Find or explicitly adopt the deterministic active self entity; never create.

    Artifact subjects remain the original entity UUID. One client owns each store;
    concurrent cross-client initialization is outside the supported topology.
    """
    identity = get_agent_identity()
    active = (
        select(MemoryEntity)
        .where(
            MemoryEntity.entity_type == "agent",
            MemoryEntity.merged_into_id.is_(None),
            MemoryEntity.telegram_user_id.is_(None),
        )
        .order_by(MemoryEntity.created_at, MemoryEntity.id)
    )
    bound = await session.execute(
        active.where(MemoryEntity.data[AGENT_IDENTITY_DATA_KEY].astext == identity.stable_id).limit(1)
    )
    entity = bound.scalar_one_or_none()
    if entity is not None or not identity.legacy_names:
        return entity
    candidates = (await session.execute(active.where(MemoryEntity.name.in_(identity.legacy_names)))).scalars().all()
    if any((row.data or {}).get(AGENT_IDENTITY_DATA_KEY) not in (None, identity.stable_id) for row in candidates):
        raise ValueError("legacy agent name is already bound to a different stable identity")
    if not candidates:
        return None
    entity = candidates[0]
    entity.data = {**(entity.data or {}), AGENT_IDENTITY_DATA_KEY: identity.stable_id}
    await session.flush()
    return entity


async def ensure_agent_identity(session: AsyncSession) -> MemoryEntity:
    """Initialize self identity in the caller's transaction, preserving legacy UUIDs."""
    identity = get_agent_identity()
    entity = await resolve_agent_identity(session)
    if entity is None:
        collision = await session.execute(
            select(MemoryEntity.id)
            .where(
                MemoryEntity.entity_type == "agent",
                MemoryEntity.name == identity.display_name,
                MemoryEntity.telegram_user_id.is_(None),
            )
            .limit(1)
        )
        if collision.scalar_one_or_none() is not None:
            raise ValueError(
                "agent display name already exists; configure explicit legacy adoption or a different name"
            )
        entity = MemoryEntity(
            entity_type="agent",
            name=identity.display_name,
            aliases=list(identity.aliases),
            data={AGENT_IDENTITY_DATA_KEY: identity.stable_id},
        )
        session.add(entity)
    else:
        # Keep the stored name and UUID; new display names are recognition aliases.
        entity.aliases = list(dict.fromkeys([*(entity.aliases or []), identity.display_name, *identity.aliases]))
    await session.flush()
    return entity
