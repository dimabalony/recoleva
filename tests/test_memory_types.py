"""Task 1 — types module unit tests.

Covers dataclass construction, Protocol shape, and Literal value sets per
v2a §3.3.
"""

from __future__ import annotations

import dataclasses
import uuid
from datetime import UTC, datetime
from typing import get_args


def test_imports_from_package() -> None:
    from kai_memory import (
        Artifact,
        ArtifactQuery,
        ConsolidationReport,
        ConsolidationTrigger,
        Cue,
        EncodingResult,
        EncodingSignals,
        Experience,
        FeedbackResult,
        FeedbackSignal,
        FeedbackTarget,
        MemoryInterface,
        Representation,
        RetrievalContext,
        RetrievalResult,
        TracePrime,
    )

    expected = {
        Artifact,
        ArtifactQuery,
        ConsolidationReport,
        ConsolidationTrigger,
        Cue,
        EncodingResult,
        EncodingSignals,
        Experience,
        FeedbackResult,
        FeedbackSignal,
        FeedbackTarget,
        MemoryInterface,
        Representation,
        RetrievalContext,
        RetrievalResult,
        TracePrime,
    }
    assert all(t is not None for t in expected)


def test_memory_type_literal_has_eleven_values() -> None:
    from kai_memory.types import MemoryType

    args = set(get_args(MemoryType))
    assert args == {
        "fact",
        "episode",
        "impression",
        "preference",
        "relationship",
        "plan",
        "intention",
        "status_change",
        "decision",
        "social_norm",
        "self_reflection",
    }


def test_entity_type_literal_has_ten_values() -> None:
    from kai_memory.types import EntityType

    args = set(get_args(EntityType))
    assert args == {
        "person",
        "animal",
        "place",
        "organization",
        "topic",
        "project",
        "object",
        "event",
        "group",
        # D Wave 1 — kai-self entity row uses entity_type='agent'.
        "agent",
    }


def test_artifact_type_literal_has_ten_values() -> None:
    from kai_memory.types import ArtifactType

    args = set(get_args(ArtifactType))
    assert args == {
        # M2 (shipped).
        "EntitySchema",
        "ContrastiveMap",
        "CueMapping",
        "Trajectory",
        "IdentityRecord",
        "TrainingDialog",
        # Sub-design D Wave 1+2.
        "Anchor",
        "PersonalityTrait",
        "Stance",
        "Pattern",
    }


def test_link_role_and_modality_literals() -> None:
    from kai_memory.types import LinkRole, Modality

    assert set(get_args(LinkRole)) == {"subject", "about", "mentioned", "participant"}
    assert set(get_args(Modality)) == {"text", "image", "audio", "transcript", "sensor"}


def test_representation_construction() -> None:
    from kai_memory.types import Representation

    rep = Representation(
        modality="text",
        payload_ref="inline:hello",
        embedding_ref=None,
        provenance="test",
    )
    assert rep.modality == "text"
    assert rep.embedding_ref is None


def test_encoding_signals_defaults() -> None:
    from kai_memory.types import EncodingSignals

    sig = EncodingSignals()
    assert sig.identity == 0.0
    assert sig.correction == 0.0
    assert sig.introduction == 0.0
    assert sig.haiku_novelty is None
    assert sig.haiku_arousal is None
    assert sig.oec_mode is None


def test_cue_with_uuid_value() -> None:
    from kai_memory.types import Cue

    eid = uuid.uuid4()
    c = Cue(kind="entity_id", value=eid)
    assert c.value == eid
    assert c.confidence == 1.0


def test_retrieval_context_defaults() -> None:
    from kai_memory.types import RetrievalContext

    ctx = RetrievalContext(chat_id=None, topic_id=None)
    assert ctx.venue == "telegram"
    assert ctx.limit == 8
    assert ctx.conversation_entities == []


def test_experience_optional_id_and_default_signals() -> None:
    from kai_memory.types import EncodingSignals, Experience

    exp = Experience(
        chat_id=1,
        topic_id=None,
        venue="telegram",
        timestamp=datetime.now(UTC),
        memory_type="fact",
        content="hello",
        entity_links=[],
        significance=3,
    )
    assert exp.id is None
    assert isinstance(exp.signals, EncodingSignals)
    assert exp.representations == []


def test_artifact_query_defaults_status_active() -> None:
    from kai_memory.types import ArtifactQuery

    q = ArtifactQuery()
    assert q.status == "active"
    assert q.artifact_type is None


def test_consolidation_trigger_construction() -> None:
    from kai_memory.types import ConsolidationTrigger

    t = ConsolidationTrigger(
        level=2,
        venue="telegram",
        chat_id=42,
        topic_id=None,
        reason="silence_timer",
    )
    assert t.level == 2


def test_feedback_target_kind() -> None:
    from kai_memory.types import FeedbackTarget

    fid = uuid.uuid4()
    t = FeedbackTarget(kind="trace", id=fid)
    assert t.kind == "trace"
    assert t.id == fid


def test_memory_interface_is_protocol() -> None:
    from kai_memory.types import MemoryInterface

    # typing.Protocol marks classes with _is_protocol=True.
    assert getattr(MemoryInterface, "_is_protocol", False) is True


def test_representation_is_frozen() -> None:
    from kai_memory.types import Representation

    rep = Representation(modality="text", payload_ref="x", embedding_ref=None, provenance="y")
    try:
        rep.modality = "image"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("Representation should be a frozen dataclass")
