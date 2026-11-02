"""Synthetic M5 policy evidence; model quality and DB persistence are separate."""

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

import pytest

from kai_memory import AgentIdentity, ports
from kai_memory.consolidation.consolidation import _validate_relationship_classification
from kai_memory.procedural.rule_capture import _build_rule_metadata, evaluate_candidate
from kai_memory.procedural.procedural_serving import _in_scope
from kai_memory.sdk import Chat, Message
from kai_memory.subject_attribution import attribute_fact_subjects


@pytest.fixture(params=["atlas"])
def host(request):
    policy = SimpleNamespace(respect=lambda actor: {10: 100, 20: 75, 30: 74}.get(actor, 0))
    identity = AgentIdentity(request.param, request.param.title())
    with ports.scoped(
        {
            "get_agent_identity": lambda: identity,
            "scrub_secrets": lambda text: text,
            "validate_skill_content": lambda text: [],
            "strip_msg_meta_envelope": lambda text: text,
        }
    ):
        yield policy, identity


@pytest.mark.parametrize(
    "role,actor,metadata,accepted",
    [
        ("user", 10, {}, True),
        ("user", 20, {}, True),
        ("user", 30, {}, False),
        ("user", 99, {}, False),
        ("assistant", 10, {}, False),
        ("tool", 10, {}, False),
        ("user", 10, {"forwarded": True}, False),
        ("user", 10, {"reply_quote_len": 200}, False),
    ],
    ids=["admin", "teacher", "below-threshold", "stranger", "assistant-echo", "tool-echo", "forward", "quote"],
)
def test_host_teaching_preserves_sender_subject_and_source(host, role, actor, metadata, accepted):
    policy, identity = host
    quote = "Remember: when answering questions about Mira, use short paragraphs."
    message = Message(
        101,
        7,
        "synthetic",
        role,
        quote,
        datetime(2026, 10, 31, tzinfo=UTC),
        sender_user_id=actor,
        message_metadata=metadata,
        id=UUID(int=101),
    )
    candidate = {
        "source_quote": quote,
        "category": "communication",
        "rule": "Use short paragraphs when answering questions about Mira",
        "trigger": "Questions about Mira",
        "directive_strength": "instruction",
        "scope_hint": "this_person",
        "instructor_id": 99,  # Inference cannot replace the verified teacher.
    }
    gate = evaluate_candidate(candidate, [message], "", policy.respect)
    assert gate.accepted is accepted
    if accepted:
        assert gate.instructor_id == actor
        proc = _build_rule_metadata(
            candidate,
            gate,
            now=message.created_at,
            source_message_id=str(message.id),
            source_chat_id=message.chat_id,
            source_topic_id=message.topic_id,
        )
        assert proc["provenance"] == {"taught_by": actor, "respect": policy.respect(actor)}
        assert proc["source_message_ids"] == [str(message.id)]
        assert proc["source_chat_ids"] == ["101:7"]
        assert proc["scope"]["person"] == actor
    # The subject of a fact is neither its sender nor the configured assistant.
    subject = SimpleNamespace(id=UUID(int=202), name="Mira", entity_type="person", aliases=["Мира"])
    assistant = SimpleNamespace(id=UUID(int=303), name=identity.display_name, entity_type="agent", aliases=[])
    facts = [{"subject": "Мира", "field": "preference", "value": "short paragraphs"}]
    attribute_fact_subjects(facts, [subject, assistant])
    assert facts[0]["subject_id"] == str(subject.id)
    assert facts[0]["subject"] == "Мира"


@pytest.mark.parametrize(
    "person,safe,origin,chat_id,topic_id,actor,venue,allowed",
    [
        (None, True, [], 202, 8, 99, "synthetic", True),
        (None, False, ["101:7"], 101, 7, 99, "synthetic", True),
        (None, False, ["101:7"], 101, 8, 99, "synthetic", False),
        (None, False, ["101:7"], 202, 7, 99, "synthetic", False),
        (10, True, ["101:7"], 101, 7, 99, "synthetic", False),
        (10, False, [], 202, 8, 10, "synthetic", True),
        (None, True, [], 202, 8, 10, "other", False),
    ],
    ids=[
        "approved-sharing",
        "origin-group",
        "other-topic",
        "other-group",
        "person-denied",
        "person-approved",
        "venue-denied",
    ],
)
def test_host_sharing_matrix(host, person, safe, origin, chat_id, topic_id, actor, venue, allowed):
    del host  # Run both host bindings against the same SDK scope contract.
    proc = {
        "scope": {"person": person, "venue": "synthetic", "group_safe": safe},
        "source_chat_ids": origin,
    }
    chat = Chat(chat_id=chat_id, topic_id=topic_id, venue=venue, sender_user_id=actor, is_group=True)
    assert _in_scope(proc, chat) is allowed


def test_relationship_allowlist_labeled_precision_recall(host):
    del host
    # Labels are explicit, independent of the implementation allowlist. This is
    # validator precision/recall on eight cases, not extraction-model quality.
    cases = [
        ("person", "organization", "employee_of", True),
        ("organization", "person", "employee_of", True),
        ("person", "person", "friend_of", True),
        ("agent", "person", "assistant_of", True),
        ("person", "project", "employee_of", False),
        ("place", "place", "employee_of", False),
        ("person", "project", "leads", False),
        ("person", "topic", "reminds_about", False),
    ]
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for source, target, predicate, expected in cases:
        left = SimpleNamespace(id=UUID(int=1), name="Synthetic Left", entity_type=source)
        right = SimpleNamespace(id=UUID(int=2), name="Synthetic Right", entity_type=target)
        actual = _validate_relationship_classification(left, right, predicate, 0.9).ok
        counts[("t" if actual == expected else "f") + ("p" if actual else "n")] += 1
    assert counts == {"tp": 4, "fp": 0, "fn": 0, "tn": 4}
    assert counts["tp"] / (counts["tp"] + counts["fp"]) == 1.0
    assert counts["tp"] / (counts["tp"] + counts["fn"]) == 1.0


def test_named_host_persona_composition_is_deterministic(host):
    from kai_memory.persona_composer import PersonaComposer, _AnchorRow, _PatternRow, _StanceRow, _TraitRow

    _, identity = host
    anchor = _AnchorRow(content=f"{identity.display_name} is a synthetic assistant.")
    args = (
        anchor,
        [_TraitRow("Explain uncertainty")],
        [_StanceRow("Prefer evidence")],
        [_PatternRow("Uses examples")],
        [],
    )
    rendered = PersonaComposer._render(*args)
    assert rendered == PersonaComposer._render(*args)
    assert (
        rendered.index(anchor.content)
        < rendered.index("Explain uncertainty")
        < rendered.index("Prefer evidence")
        < rendered.index("Uses examples")
    )
    if identity.stable_id == "atlas":
        assert "Kai" not in rendered
