"""Public memory evidence stays bounded, content-free and stable at completion."""

import json
import logging
from uuid import UUID

import pytest

from kai_memory.retrieval import memory_observation as telemetry


def _events(caplog, name):
    prefix = name + " "
    return [
        json.loads(record.getMessage().removeprefix(prefix))
        for record in caplog.records
        if record.getMessage().startswith(prefix)
    ]


def test_large_turn_keeps_honest_counts_with_bounded_terminal_evidence(caplog):
    caplog.set_level(logging.INFO)
    inputs = [f"input-{index:03d}-" + "a" * 118 for index in range(80)]
    identifiers = [f"trace-{index:03d}-" + "b" * 118 for index in range(90)]
    observation = telemetry.MemoryObservation(model="gpt-6-astra", scope="scope", input_ids=inputs)
    for _ in range(telemetry.MAX_DECISIONS + 7):
        decision = observation.begin("initial")
        decision.complete("retrieved", retrieved=identifiers, selected=identifiers)
    observation.prepared("initial")
    observation.delivery("initial", "accepted", turn_id="turn-1")

    snapshot = observation.finish("completed")

    assert len(json.dumps(snapshot).encode()) <= telemetry.MAX_SNAPSHOT_BYTES
    assert snapshot["canonical_input_count"] == len(inputs)
    assert snapshot["input_ids_truncated"] is True
    assert snapshot["omitted_decisions"] == 7
    assert snapshot["omitted_snapshot_decisions"] > 0
    assert len(snapshot["decisions"]) + snapshot["omitted_snapshot_decisions"] == telemetry.MAX_DECISIONS
    assert snapshot["decisions"]
    for decision in snapshot["decisions"]:
        assert decision["retrieved_count"] == len(identifiers)
        assert decision["selected_count"] == len(identifiers)
        assert decision["ids_truncated"] is True
        assert len(decision["injected_ids"]) <= telemetry.MAX_IDS
    events = _events(caplog, "native.memory.turn")
    assert len(events) == 1
    assert len(json.dumps(events[0]).encode()) <= telemetry.MAX_SNAPSHOT_BYTES + 128


def test_terminal_receipt_freezes_first_outcome_and_identifiers(caplog):
    caplog.set_level(logging.INFO)
    observation = telemetry.MemoryObservation(model="gpt-6-astra", scope="scope", input_ids=["input-1"])
    decision = observation.begin("initial")
    decision.complete("retrieved", retrieved=["memory-1"], selected=["memory-1"])
    observation.prepared("initial")
    first = observation.finish("cancelled")
    frozen = json.dumps(first, sort_keys=True)

    observation.bind(thread_id="late-thread", turn_id="late-turn", trace_id="late-trace")
    observation.delivery("initial", "accepted", turn_id="late-turn")
    observation.begin("steered", input_id="late-input").complete("empty")
    again = observation.finish("completed")

    assert json.dumps(again, sort_keys=True) == frozen
    assert json.dumps(first, sort_keys=True) == frozen
    assert again["terminal_status"] == "cancelled"
    assert len(_events(caplog, "native.memory.turn")) == 1


def test_tool_reply_written_is_not_remote_injection_acceptance():
    observation = telemetry.MemoryObservation(model="gpt-6-astra", scope="scope", input_ids=["input-1"])
    decision = observation.begin("reprise", trigger="reasoning_summary")
    decision.call_id = "tool-1"
    decision.complete("retrieved", retrieved=["memory-1"], selected=["memory-1"])
    observation.prepared("reprise")
    observation.tool_response("tool-1", True)

    snapshot = observation.finish("completed")

    receipt = snapshot["decisions"][0]
    assert receipt["delivery"] == "written_unacknowledged"
    assert receipt["written_ids"] == ["memory-1"]
    assert receipt["injected_ids"] == []


def test_observer_rejects_content_and_unknown_objects_in_identifier_fields(caplog):
    caplog.set_level(logging.INFO)

    class PrivateObject:
        def __str__(self):
            raise AssertionError("Observability must not stringify arbitrary private objects")

    identifier = UUID("39fd85ad-d490-488c-bc58-107a75d6b26f")
    secret = "private family memory must never appear in diagnostic logs"
    observation = telemetry.MemoryObservation(model=secret, scope=secret, input_ids=[secret, "input-1"])
    observation.bind(thread_id=secret, turn_id="x" * 129, trace_id="safe-trace")
    decision = observation.begin("initial")
    decision.call_id = secret + "x" * telemetry.MAX_SNAPSHOT_BYTES
    decision.complete("retrieved", retrieved=[PrivateObject(), secret, identifier], selected=[identifier])
    snapshot = observation.finish("failed")

    assert snapshot["model"] is None
    assert snapshot["scope"] is None
    assert snapshot["native_thread_id"] is None
    assert snapshot["native_turn_id"] is None
    assert snapshot["decisions"][0]["retrieved_ids"] == [str(identifier)]
    assert snapshot["decisions"][0]["call_id"] is None
    assert secret not in json.dumps(snapshot)
    assert secret not in caplog.text


def test_retrieval_latency_is_frozen_before_later_delivery_and_cleanup(monkeypatch):
    observation = telemetry.MemoryObservation(model="gpt-6-astra", scope="scope", input_ids=["input-1"])
    decision = observation.begin("reprise", trigger="reasoning_summary")
    decision.started = 10.0
    clock = [11.5]
    monkeypatch.setattr(telemetry.time, "monotonic", lambda: clock[0])
    decision.retrieval_finished()
    clock[0] = 20.0
    decision.complete("empty")

    assert observation.finish("completed")["decisions"][0]["duration_ms"] == 1500


@pytest.mark.parametrize("failure", [ValueError, RuntimeError])
def test_broken_log_sink_cannot_change_memory_or_terminal_outcome(monkeypatch, failure):
    def unavailable(*_args, **_kwargs):
        raise failure("telemetry sink is unavailable")

    monkeypatch.setattr(telemetry.logger, "info", unavailable)
    observation = telemetry.MemoryObservation(model="gpt-6-astra", scope="scope", input_ids=["input-1"])
    decision = observation.begin("initial")
    decision.complete("retrieved", retrieved=["memory-1"], selected=["memory-1"])
    observation.prepared("initial")
    observation.delivery("initial", "accepted", turn_id="turn-1")

    snapshot = observation.finish("completed")

    assert snapshot["terminal_status"] == "completed"
    assert snapshot["decisions"][0]["injected_ids"] == ["memory-1"]


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("uncertain", [False, True])
def test_abandonment_preserves_full_counts_and_bounded_receipts(monkeypatch, caplog, completed, uncertain):
    caplog.set_level(logging.INFO)
    observation = telemetry.MemoryObservation(model="synthetic", scope="scope", input_ids=["input-1"])
    decision = observation.begin("reprise", trigger="reasoning_summary")
    decision.call_id = "tool-1"
    decision.started = 10.0
    clock = [12.5]
    monkeypatch.setattr(telemetry.time, "monotonic", lambda: clock[0])
    retrieved = [f"retrieved-{index}" for index in range(telemetry.MAX_IDS + 9)]
    selected = retrieved[: telemetry.MAX_IDS + 3] if completed else []
    decision.retrieval_finished(retrieved=retrieved)
    if completed:
        decision.complete("retrieved", retrieved=retrieved, selected=selected)
        observation.prepared("reprise")
    clock[0] = 20.0
    emitted_before = len(_events(caplog, "native.memory.decision"))

    observation.abandon_call("tool-1", uncertain=uncertain)

    receipts = [
        decision.payload(),
        observation.snapshot()["decisions"][0],
        observation.finish("completed")["decisions"][0],
        *_events(caplog, "native.memory.decision")[emitted_before:],
    ]
    assert len(receipts) > 3
    for receipt in receipts:
        assert receipt["retrieved_count"] == len(retrieved)
        assert receipt["selected_count"] == len(selected)
        assert receipt["retrieved_ids"] == retrieved[: telemetry.MAX_IDS]
        assert receipt["selected_ids"] == selected[: telemetry.MAX_IDS]
        assert receipt["ids_truncated"] is True
        assert receipt["duration_ms"] == 2500
        assert receipt["outcome"] == ("retrieved" if completed else "discarded")
        assert receipt["injected_ids"] == receipt["written_ids"] == []
    assert receipts[0]["delivery"] == ("unknown" if uncertain else "not_sent")
    assert receipts[1]["delivery"] == receipts[2]["delivery"] == receipts[0]["delivery"]
