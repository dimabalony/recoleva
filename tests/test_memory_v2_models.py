"""Task 3 — Memory v2 model file unit tests.

Each new model can be imported and constructed in-process with required
fields. New JSONB columns on MemoryTrace materialize their Python defaults at
construction time (server defaults don't fire when SQLAlchemy builds a row
client-side, so both must be present per v2a §4.2).
"""

from __future__ import annotations

import uuid as _uuid




def test_confusion_set_tablename_and_pk() -> None:
    from kai_memory.storage.models import ConfusionSet

    assert ConfusionSet.__tablename__ == "confusion_sets"
    cs = ConfusionSet(entity_type="animal", name="dog-confusions")
    assert cs.entity_type == "animal"
    assert cs.name == "dog-confusions"


def test_confusion_set_member_composite_pk() -> None:
    from kai_memory.storage.models import ConfusionSetMember

    assert ConfusionSetMember.__tablename__ == "confusion_set_members"
    cs_id = _uuid.uuid4()
    e_id = _uuid.uuid4()
    m = ConfusionSetMember(
        confusion_set_id=cs_id,
        entity_id=e_id,
        contrastive_features={"color": "white"},
    )
    assert m.confusion_set_id == cs_id
    assert m.entity_id == e_id
    assert m.contrastive_features == {"color": "white"}


def test_trace_inhibition_construction() -> None:
    from kai_memory.storage.models import TraceInhibition

    assert TraceInhibition.__tablename__ == "trace_inhibitions"
    t = TraceInhibition(
        winner_trace_id=_uuid.uuid4(),
        loser_trace_id=_uuid.uuid4(),
        strength=0.2,
    )
    assert t.strength == 0.2


def test_memory_artifact_construction() -> None:
    from kai_memory.storage.models import MemoryArtifact

    assert MemoryArtifact.__tablename__ == "memory_artifacts"
    art = MemoryArtifact(
        artifact_type="IdentityRecord",
        producer_name="identity_producer",
        payload={"k": "v"},
        content_hash="abc123",
        dimension="communication_style",
    )
    assert art.artifact_type == "IdentityRecord"
    assert art.dimension == "communication_style"
    assert art.payload == {"k": "v"}




def test_memory_trace_python_defaults_for_new_columns() -> None:
    """Both Python and server defaults are configured per v2a §4.2.

    SQLAlchemy fires the Python `default=` callable at INSERT time (not at
    construction), which is what populates the row client-side when the test
    DB doesn't materialize server defaults the same way the live DB does.
    """
    from kai_memory.storage.models.memory_trace import MemoryTrace

    cols = {c.name: c for c in MemoryTrace.__table__.columns}

    rep = cols["representations"]
    assert rep.default is not None, "Python default missing on representations"
    assert rep.default.is_callable
    assert rep.server_default is not None
    assert rep.nullable is False

    sig = cols["encoding_signals"]
    assert sig.default is not None, "Python default missing on encoding_signals"
    assert sig.default.is_callable
    assert sig.server_default is not None
    assert sig.nullable is False


def test_memory_trace_columns_present_in_schema() -> None:
    from kai_memory.storage.models.memory_trace import MemoryTrace

    cols = {c.name for c in MemoryTrace.__table__.columns}
    assert "representations" in cols
    assert "encoding_signals" in cols
