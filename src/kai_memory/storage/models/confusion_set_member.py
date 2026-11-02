"""ConfusionSetMember model — entity membership in a confusion set with contrastive features."""

from __future__ import annotations

import uuid as _uuid

from sqlalchemy import ForeignKey, Index, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class ConfusionSetMember(Base):
    """Membership row pairing an entity with a confusion set.

    `contrastive_features` holds the per-member typed slots used to
    distinguish this entity from the others in the set.
    """

    __tablename__ = "confusion_set_members"

    confusion_set_id: Mapped[_uuid.UUID] = mapped_column(
        ForeignKey("confusion_sets.id", ondelete="CASCADE"), primary_key=True
    )
    entity_id: Mapped[_uuid.UUID] = mapped_column(
        ForeignKey("memory_entities.id", ondelete="CASCADE"), primary_key=True
    )
    contrastive_features: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )

    __table_args__ = (Index("idx_conf_members_entity", "entity_id"),)
