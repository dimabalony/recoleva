"""Unit tests for L1.5 iterative retrieval (Stage A window cues + Stage C/C2 re-priming).

Design of record: backend-docs/ai-workforce/plans/design/2026-07-21-l15-iterative-retrieval-design.md.
System-boundary mocks only (``generate_embedding``, ``session_scope``, the priming
service); the cue math, weighting, caps, dedup, novelty gate, and containment
invariants run for real (repo doctrine: a mock shaped like your code can only confirm
you).
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai_memory.store import cognitive_config
from kai_memory.retrieval import iterative_retrieval as ir
from kai_memory.retrieval.iterative_retrieval import (
    LIVE,
    OFF,
    RECALL_MARKER,
    SHADOW,
    RepriseState,
    build_segment_cues,
    extract_window_user_texts,
    inject_into_tool_results,
)
from memory_test_generation import LLMResponse
from memory_test_generation import record_generation_provenance


@pytest.fixture(autouse=True)
def _reset():
    ir.reset_state_for_testing()
    cognitive_config.reset_cache()
    yield
    ir.reset_state_for_testing()
    cognitive_config.reset_cache()


def _trace(content="a memory", memory_type="fact", trace_id=None):
    return SimpleNamespace(id=trace_id or uuid.uuid4(), content=content, memory_type=memory_type)


# ── Flags ────────────────────────────────────────────────────────────────


def test_flag_states_default_shadow():
    # Defaults come from cognitive_config._DEFAULTS (mirrors migration 099).
    assert ir.context_cues_state() == SHADOW
    assert ir.thinking_cues_state() == SHADOW


def test_flag_state_clamps_out_of_range_to_off():
    cognitive_config.set_for_testing("priming_thinking_cues", 9.0)
    assert ir.thinking_cues_state() == OFF


def test_flag_state_reads_live():
    cognitive_config.set_for_testing("priming_context_cues", 2.0)
    assert ir.context_cues_state() == LIVE


# ── Stage A: window extraction + segment weighting ───────────────────────


def test_extract_window_excludes_assistant_and_tool_results():
    messages = [
        {"role": "user", "content": "first user"},
        {"role": "assistant", "content": [{"text": "Kai reply"}]},
        {"role": "user", "content": [{"text": "second user"}]},
        # tool-result user message: content is toolResult blocks, no plain text
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"json": {}}]}}]},
    ]
    assert extract_window_user_texts(messages) == ["first user", "second user"]


def test_extract_window_ignores_structured_model_metadata():
    messages = [
        {
            "role": "user",
            "content": [{"text": "authored question"}],
            "model_metadata": {
                "message_id": 4821,
                "time": "2026-07-24 10:00",
                "sender": "Misha",
                "reactions": [{"emoji": "👍", "user": "Alice", "user_id": 99}],
            },
        }
    ]
    assert extract_window_user_texts(messages) == ["authored question"]


def test_extract_window_trusts_literal_envelope_in_canonical_sidecar_message():
    literal = '[msg_meta: {"message_id": 99}] — this is authored text'
    messages = [
        {
            "role": "user",
            "content": [{"text": literal}],
            "model_metadata": {"message_id": 42},
        }
    ]
    assert extract_window_user_texts(messages) == [literal]


def test_extract_window_strips_msg_meta_envelope_before_embedding():
    # Pre-cutover pending states can still contain the rendered envelope. It
    # must not reach the cue embedding. Covers string and text-block shapes.
    env = '[msg_meta: {"message_id": 4821, "time": "2026-07-24T10:00:00Z", "sender": "Misha"}]'
    messages = [
        {"role": "user", "content": f"{env} А что она говорила про дедлайн?"},  # noqa: RUF001 — Cyrillic is the register under test
        {"role": "user", "content": [{"text": f"{env} and the second one?"}]},
    ]
    assert extract_window_user_texts(messages) == [
        "А что она говорила про дедлайн?",  # noqa: RUF001
        "and the second one?",
    ]


def test_extract_window_stripper_survives_brace_in_sender_name():
    # #253 lesson: a sender display name (or reactions map) carrying a '}' defeats a regex
    # stripper but NOT the raw_decode structural parser — the envelope is still fully removed.
    env = '[msg_meta: {"message_id": 7, "time": "t", "sender": "misha}", "reactions": {"thumbsup": 1}}]'
    messages = [{"role": "user", "content": f"{env} real content here"}]
    assert extract_window_user_texts(messages) == ["real content here"]


def test_extract_window_leaves_non_envelope_text_untouched():
    # A message that merely mentions the token but isn't a real [msg_meta: {json}] envelope
    # is returned unchanged (no over-stripping).
    messages = [{"role": "user", "content": "what about the [msg_meta] tag as a topic?"}]
    assert extract_window_user_texts(messages) == ["what about the [msg_meta] tag as a topic?"]


def test_extract_window_does_not_strip_non_object_msg_meta_payload():
    # CR #301: only a real [msg_meta: {OBJECT}] envelope is stripped. A syntactically-valid
    # but non-object payload (a bare JSON string/number) is NOT the pipeline envelope, so the
    # user's text must be left fully intact rather than silently altered.
    messages = [{"role": "user", "content": '[msg_meta: "not-the-envelope"] keep this whole thing'}]
    assert extract_window_user_texts(messages) == ['[msg_meta: "not-the-envelope"] keep this whole thing']


def test_empty_window_invariant_single_message():
    """A single-message window ⇒ exactly one segment at weight 1.0 (byte-identical
    to today's last-message cue). The core Stage A safety invariant."""
    segs = build_segment_cues(["what did she say about the deadline?"])
    assert len(segs) == 1
    assert segs[0][0] == "what did she say about the deadline?"
    assert segs[0][1] == 1.0


def test_segment_weighting_newest_brightest_and_decays():
    texts = [f"msg {i}" for i in range(5)]  # oldest..newest
    segs = build_segment_cues(texts)
    weights = [w for _t, w in segs]
    # newest is 1.0 and weights are strictly decreasing back through the window
    assert weights[0] == 1.0
    assert all(weights[i] > weights[i + 1] for i in range(len(weights) - 1))
    assert min(weights) >= ir._SEG_WEIGHT_FLOOR


def test_deep_window_floors_and_collapses_old():
    texts = [f"msg {i}" for i in range(40)]
    segs = build_segment_cues(texts)
    # newest + up to _RECENT_WINDOW-1 recent + 1 collapsed-old chunk
    assert len(segs) <= ir._RECENT_WINDOW + 1
    assert segs[-1][1] == pytest.approx(ir._SEG_WEIGHT_FLOOR)


def test_long_message_splits_into_position_segments():
    long_msg = "x" * (ir._LONG_MSG_CHARS + 6000)
    segs = build_segment_cues([long_msg])
    assert len(segs) > 1  # split into multiple position segments
    # later segments brighter (position ramp), last reaches base weight 1.0
    assert segs[-1][1] == pytest.approx(1.0)
    assert segs[0][1] < segs[-1][1]


def test_event_message_cues_content_not_just_header():
    """Fixes the pre-existing header-only degradation: a transcript's CONTENT
    becomes a cue, not just the [EVENT] header."""
    body = "TRANSCRIPT_BODY " * 400
    event = "[EVENT] title: standup\nparticipants: A, B" + ir._EVENT_CONTENT_SEPARATOR + body
    segs = build_segment_cues([event])
    joined = " ".join(t for t, _w in segs)
    assert "title: standup" in joined  # header present
    assert "TRANSCRIPT_BODY" in joined  # content present too


# ── Embedding cache (A2) ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_embed_cache_single_embed_per_content():
    calls = {"n": 0}

    async def fake_embed(text, api_key=None):
        calls["n"] += 1
        return [1.0, 0.0, 0.0]

    with patch("kai_memory.store.embeddings.generate_embedding", side_effect=fake_embed):
        a = await ir._embed_cached("same text")
        b = await ir._embed_cached("same text")
        c = await ir._embed_cached("other text")
    assert a == b == [1.0, 0.0, 0.0]
    assert c == [1.0, 0.0, 0.0]
    assert calls["n"] == 2  # "same text" embedded once, "other text" once


@pytest.mark.asyncio
async def test_embed_cache_failure_returns_none():
    with patch("kai_memory.store.embeddings.generate_embedding", side_effect=RuntimeError("boom")):
        assert await ir._embed_cached("x") is None


# ── Stage C: novelty gate + read-only retrieval ──────────────────────────


def _make_reprise(state=LIVE, cue_embeddings=None, primed=None):
    return RepriseState(
        chat_ctx=SimpleNamespace(venue="telegram", chat_id=1, topic_id=None),
        cue_embeddings=cue_embeddings if cue_embeddings is not None else [[0.0, 1.0, 0.0]],
        primed_trace_ids=primed or [],
        thinking_state=state,
    )


@pytest.mark.asyncio
async def test_reprime_novelty_gate_returns_none_for_skipped_gate():
    """Thinking too similar to the cue set (cosine >= 0.85) returns None (novelty gate) —
    distinct from [] (retrieved-but-nothing) so collect() can log skipped_gate vs noop."""
    r = _make_reprise(cue_embeddings=[[1.0, 0.0, 0.0]])
    with patch("kai_memory.store.embeddings.generate_embedding", AsyncMock(return_value=[1.0, 0.0, 0.0])):
        out = await r._reprime("thinking identical to cue")
    assert out is None  # novelty gate — collect() maps None → skipped_gate


@pytest.mark.asyncio
async def test_funnel_skipped_gate_and_cancelled_emit_log(caplog):
    """The consideration funnel must be reconstructable from logs: skipped_gate (novelty)
    and cancelled each emit their own memory.reprime.complete line (were counter-only /
    mislabeled as noop before)."""
    import json as _json

    # skipped_gate: task returns None ⇒ collect logs skipped_gate (not noop)
    r = _make_reprise(state=SHADOW)
    r._pending_cue = "some gated thought"
    r._task = asyncio.ensure_future(_completed_task(None))
    with caplog.at_level("INFO"):
        assert await r.collect() == []
    gate = _json.loads(
        next(m for m in (rec.message for rec in caplog.records) if "memory.reprime.complete" in m).split(
            "memory.reprime.complete ", 1
        )[1]
    )
    assert gate["outcome"] == "skipped_gate"
    assert r.counters["skipped_gate"] == 1 and r.counters["noop"] == 0
    caplog.clear()

    # cancelled: cancel() on an in-flight task emits a cancelled line
    r2 = _make_reprise(state=SHADOW)

    async def _never():
        await asyncio.sleep(100)

    r2._task = asyncio.ensure_future(_never())
    with caplog.at_level("INFO"):
        r2.cancel()
    assert any(
        '"outcome": "cancelled"' in rec.message for rec in caplog.records if "memory.reprime.complete" in rec.message
    )
    assert r2.counters["cancelled"] == 1


@pytest.mark.asyncio
async def test_reprime_is_read_only_record_access_false():
    """Stage C retrieval MUST pass record_access=False (containment invariant #1)."""
    r = _make_reprise(cue_embeddings=[[1.0, 0.0, 0.0]])
    top = [(_trace("she said the deadline is Friday"), 0.9, {"semantic": 0.9})]
    fake_svc = MagicMock()
    fake_svc.prime_with_traces = AsyncMock(return_value=("", [], top))
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=MagicMock())
    cm.__aexit__ = AsyncMock(return_value=False)
    with (
        patch("kai_memory.store.embeddings.generate_embedding", AsyncMock(return_value=[0.0, 1.0, 0.0])),
        patch("kai_memory.ports.session_scope", return_value=cm),
        patch("kai_memory.retrieval.memory_priming.MemoryPrimingService", return_value=fake_svc),
    ):
        out = await r._reprime("novel thinking about a deadline")
    assert len(out) == 1
    _tid, line, rel = out[0]
    assert RECALL_MARKER in line
    assert rel == pytest.approx(0.9)
    _args, kwargs = fake_svc.prime_with_traces.call_args
    assert kwargs.get("record_access") is False


@pytest.mark.asyncio
async def test_reprime_dedups_already_primed_and_floors_weak():
    primed_id = uuid.uuid4()
    r = _make_reprise(cue_embeddings=[[1.0, 0.0, 0.0]], primed=[primed_id])
    top = [
        (_trace("already surfaced", trace_id=primed_id), 0.9, {"semantic": 0.9}),  # dedup drop
        (_trace("weak match"), 0.9, {"semantic": 0.1}),  # below floor drop
        (_trace("strong new"), 0.9, {"semantic": 0.8}),  # kept
    ]
    fake_svc = MagicMock()
    fake_svc.prime_with_traces = AsyncMock(return_value=("", [], top))
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=MagicMock())
    cm.__aexit__ = AsyncMock(return_value=False)
    with (
        patch("kai_memory.store.embeddings.generate_embedding", AsyncMock(return_value=[0.0, 1.0, 0.0])),
        patch("kai_memory.ports.session_scope", return_value=cm),
        patch("kai_memory.retrieval.memory_priming.MemoryPrimingService", return_value=fake_svc),
    ):
        out = await r._reprime("novel thinking")
    assert [line for _t, line, _r in out] == [ir._one_line(_trace("strong new"))]


# ── Stage C: consider / kick gating ──────────────────────────────────────


def test_consider_bedrock_no_op_empty_thinking():
    """Bedrock extracts no thinking → consider() sees empty list → no task (fail-open)."""
    r = _make_reprise()
    r.consider([], "tool_use")
    r.consider(None, "tool_use")
    assert r._task is None


def test_consider_skips_non_tool_use_stop_reason():
    r = _make_reprise()
    r.consider(["some thinking"], "end_turn")
    assert r._task is None


def test_consider_off_state_no_op():
    r = _make_reprise(state=OFF)
    r.consider(["thinking"], "tool_use")
    assert r._task is None


@pytest.mark.asyncio
async def test_consider_kicks_task_when_active():
    r = _make_reprise(state=SHADOW)
    with patch.object(r, "_reprime", AsyncMock(return_value=[])):
        r.consider(["novel thinking"], "tool_use")
        assert r._task is not None
        await r.collect()  # drain the task so it doesn't leak


# ── Stage C: collect caps + counters ─────────────────────────────────────


async def _completed_task(found):
    return found


@pytest.mark.asyncio
async def test_collect_live_injects_marker_and_lines():
    r = _make_reprise(state=LIVE)
    found = [("id1", "- line one", 0.9), ("id2", "- line two", 0.8)]
    r._task = asyncio.ensure_future(_completed_task(found))
    lines = await r.collect()
    assert lines[0] == RECALL_MARKER
    assert lines[1:] == ["- line one", "- line two"]
    assert r.counters["fired"] == 1


@pytest.mark.asyncio
async def test_fired_event_carries_would_inject_ids_and_cue_snippet(caplog):
    """The fired memory.reprime.complete must log WHICH memories the thought would surface
    + a truncated cue snippet, so shadow can show real thought→memory pairs (user request)."""
    r = _make_reprise(state=SHADOW)  # shadow: injects nothing, still logs "fired"
    r._pending_cue = "x" * 500  # long thinking cue → must be hard-truncated
    found = [("trace-aaa", "- line one", 0.9), ("trace-bbb", "- line two", 0.8)]
    r._task = asyncio.ensure_future(_completed_task(found))
    with caplog.at_level("INFO"):
        await r.collect()
    line = next(rec.message for rec in caplog.records if "memory.reprime.complete" in rec.message)
    import json as _json

    payload = _json.loads(line.split("memory.reprime.complete ", 1)[1])
    assert payload["outcome"] == "fired"
    assert payload["would_inject_trace_ids"] == ["trace-aaa", "trace-bbb"]
    assert payload["cue_snippet"] == "x" * ir._CUE_SNIPPET_CHARS  # hard-truncated
    assert len(payload["cue_snippet"]) == ir._CUE_SNIPPET_CHARS


@pytest.mark.asyncio
async def test_collect_shadow_injects_nothing_but_counts():
    r = _make_reprise(state=SHADOW)
    found = [("id1", "- line one", 0.9)]
    r._task = asyncio.ensure_future(_completed_task(found))
    lines = await r.collect()
    assert lines == []  # shadow injects nothing
    assert r.counters["fired"] == 1  # but still logs would-have-injected


@pytest.mark.asyncio
async def test_collect_per_injection_line_cap():
    r = _make_reprise(state=LIVE)
    found = [(f"id{i}", f"- line {i}", 0.9) for i in range(6)]
    r._task = asyncio.ensure_future(_completed_task(found))
    lines = await r.collect()
    assert len(lines) - 1 == ir._MAX_LINES_PER_INJECTION  # marker + <=3
    assert r.counters["suppressed_beyond_cap"] == 1  # truncation counted


@pytest.mark.asyncio
async def test_collect_round_cap_and_turn_line_cap():
    r = _make_reprise(state=LIVE)
    # exhaust round cap
    for _ in range(ir._MAX_REPRIME_ROUNDS):
        r._task = asyncio.ensure_future(_completed_task([("x", "- l", 0.9)]))
        await r.collect()
    assert r._injections == ir._MAX_REPRIME_ROUNDS
    # one more round is suppressed
    r._task = asyncio.ensure_future(_completed_task([("y", "- l2", 0.9)]))
    lines = await r.collect()
    assert lines == []
    assert r.counters["suppressed_beyond_cap"] >= 1


@pytest.mark.asyncio
async def test_collect_noop_when_nothing_found():
    r = _make_reprise(state=LIVE)
    r._task = asyncio.ensure_future(_completed_task([]))
    lines = await r.collect()
    assert lines == []
    assert r.counters["noop"] == 1


@pytest.mark.asyncio
async def test_collect_budget_timeout_skips(monkeypatch):
    monkeypatch.setattr(ir, "_REPRIME_BUDGET_S", 0.05)
    r = _make_reprise(state=LIVE)

    async def _slow():
        await asyncio.sleep(1.0)
        return [("id", "- l", 0.9)]

    r._task = asyncio.ensure_future(_slow())
    lines = await r.collect()
    assert lines == []
    assert r.counters["skipped_budget"] == 1


@pytest.mark.asyncio
async def test_cancel_cancels_inflight_task():
    r = _make_reprise(state=LIVE)

    async def _never():
        await asyncio.sleep(100)

    task = asyncio.ensure_future(_never())
    r._task = task
    r.cancel()
    await asyncio.sleep(0)
    assert task.cancelled()
    assert r.counters["cancelled"] == 1
    assert r._task is None


# ── Injection placement — the BLOCKING safety invariant ──────────────────


def test_inject_into_tool_results_is_inside_content_not_sibling():
    tool_results = [
        {"toolResult": {"toolUseId": "t1", "content": [{"json": {"ok": True}}]}},
        {"toolResult": {"toolUseId": "t2", "content": [{"json": {"ok": True}}]}},
    ]
    ok = inject_into_tool_results(tool_results, [RECALL_MARKER, "- recalled thing"])
    assert ok is True
    # injected INSIDE the first toolResult.content, never as a top-level sibling block
    first_content = tool_results[0]["toolResult"]["content"]
    assert first_content[-1] == {"text": f"{RECALL_MARKER}\n- recalled thing"}
    # no top-level (sibling) text block anywhere in the message list
    assert all("text" not in block for block in tool_results)


def test_inject_noop_on_empty_lines():
    tool_results = [{"toolResult": {"toolUseId": "t1", "content": [{"json": {}}]}}]
    assert inject_into_tool_results(tool_results, []) is False


def test_persisted_injection_is_invisible_to_conversation_text():
    """Structural containment test (design §Containment #5): a persisted turn carrying
    an injection yields ZERO top-level text — invisible to ConversationMessage.text →
    invisible to L2 extraction."""
    from memory_test_support import ConversationMessage

    tool_results = [{"toolResult": {"toolUseId": "t1", "content": [{"json": {"ok": True}}]}}]
    inject_into_tool_results(tool_results, [RECALL_MARKER, "- recalled fact about deadline"])
    # This is exactly what gets persisted as ConversationMessage.content_blocks.
    msg = ConversationMessage(content_blocks=tool_results)
    assert RECALL_MARKER not in msg.text
    assert msg.text == ""


def test_l2_marker_skip_strips_recalled_lines():
    """Defense-in-depth: if a marker line ever reaches msg.text, L2 strips it."""
    from kai_memory.extraction.memory_extraction import _strip_recalled_lines

    text = f"real user line\n- {RECALL_MARKER} [fact] Kai association\nanother real line"
    stripped = _strip_recalled_lines(text)
    assert RECALL_MARKER not in stripped
    assert "real user line" in stripped
    assert "another real line" in stripped


# ── C2: recall-and-revise ────────────────────────────────────────────────


def _revise_reprise(state=LIVE):
    r = _make_reprise(state=state, cue_embeddings=[[1.0, 0.0, 0.0]])
    return r


@pytest.mark.asyncio
async def test_c2_shadow_logs_would_revise_no_model_call():
    r = _revise_reprise(state=SHADOW)
    provider = MagicMock()
    provider.converse = AsyncMock()
    with patch.object(r, "_reprime_for_revise", AsyncMock(return_value=[("id", f"- {RECALL_MARKER} [fact] X", 0.9)])):
        out = await r.maybe_revise(
            provider=provider,
            system_prompt=("sys", ""),
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            max_tokens=1000,
            thinking_blocks=["novel thinking"],
            draft="draft answer",
            metadata={},
        )
    assert out == "draft answer"  # unchanged in shadow
    provider.converse.assert_not_called()
    assert r.counters["would_revise"] == 1


@pytest.mark.asyncio
async def test_c2_live_revises_once_and_persists_only_r2():
    r = _revise_reprise(state=LIVE)
    provider = MagicMock()
    provider.converse = AsyncMock(
        return_value=SimpleNamespace(text="R2 revised answer", thinking_blocks=[], usage={"input": 5, "output": 7})
    )
    recall = [("id", f"- {RECALL_MARKER} [fact] the scar is from the morning stitches", 0.9)]
    messages = [{"role": "user", "content": "here is a photo"}]
    metadata: dict = {}
    with patch.object(r, "_reprime_for_revise", AsyncMock(return_value=recall)):
        out = await r.maybe_revise(
            provider=provider,
            system_prompt=("sys", ""),
            messages=messages,
            tools=[],
            max_tokens=1000,
            thinking_blocks=["novel thinking about a scar"],
            draft="R1 draft",
            metadata=metadata,
        )
    # only R2 is returned (R1 + recall never leave the request)
    assert out == "R2 revised answer"
    assert metadata.get("c2_revised") is True
    # the caller's persisted `messages` list is NOT mutated with R1 / the recall block
    assert messages == [{"role": "user", "content": "here is a photo"}]
    # the revise REQUEST (in-memory only) carried R1 draft + the marker recall block
    sent_messages = provider.converse.call_args[0][1]
    assert sent_messages[-2] == {"role": "assistant", "content": [{"text": "R1 draft"}]}
    assert RECALL_MARKER in sent_messages[-1]["content"][0]["text"]


@pytest.mark.parametrize(
    "revision,expected,used",
    [
        ("", "subscription draft", False),
        (" \n\t ", "subscription draft", False),
        ('[msg_meta: {"message_id": 42}]\n', "subscription draft", False),
        ('Conversation info (untrusted metadata):\n{"message_id": 42}\n', "subscription draft", False),
        ('[msg_meta: {"message_id": 42}]\npaid revision', "paid revision", True),
        ("paid revision", "paid revision", True),
    ],
)
async def test_c2_attributes_only_the_revision_that_survives_sanitizing(revision, expected, used):
    r = _revise_reprise(state=LIVE)
    original = {
        "provider": "codex_subscription",
        "model": "gpt-6-astra",
        "billing_mode": "subscription",
        "generation_id": "draft-generation",
    }
    revised = {
        "provider": "anthropic",
        "model": "paid-model",
        "billing_mode": "api",
        "selected_provider": "codex_subscription",
        "selected_model": "gpt-6-astra",
        "fallback_reason": "incompatible",
    }
    response = LLMResponse(
        text=revision,
        stop_reason="end_turn",
        tool_uses=[],
        usage={"input": 5, "output": 7},
        thinking_blocks=["revision thinking"],
        provenance={**revised, "output": "private provider output"},
    )
    provider = SimpleNamespace(converse=AsyncMock(return_value=response))
    messages = [{"role": "user", "content": [{"text": "user question"}], "model_metadata": {"message_id": 42}}]
    metadata = {"input_tokens": 11, "output_tokens": 13, "thinking_blocks": ["draft thinking"]}
    record_generation_provenance(metadata, original)
    priming = SimpleNamespace(
        prime_with_traces=AsyncMock(return_value=("", [], [(_trace("new recalled fact"), 0.99, {"semantic": 0.99})]))
    )
    session = AsyncMock()
    session.__aenter__.return_value = MagicMock()
    with (
        patch("kai_memory.store.embeddings.generate_embedding", AsyncMock(return_value=[0.0, 1.0, 0.0])),
        patch("kai_memory.ports.session_scope", return_value=session),
        patch("kai_memory.retrieval.memory_priming.MemoryPrimingService", return_value=priming),
    ):
        result = await r.maybe_revise(
            provider=provider,
            system_prompt=("system", ""),
            messages=messages,
            tools=[],
            max_tokens=1000,
            thinking_blocks=["novel thinking"],
            draft="subscription draft",
            metadata=metadata,
        )

    actual = revised if used else original
    assert result == expected
    assert metadata["actual_provider"] == actual["provider"]
    assert metadata["actual_model"] == actual["model"]
    assert metadata["billing_mode"] == actual["billing_mode"]
    assert metadata["llm_generations"] == [original, revised]
    assert metadata["c2_revised"] is used
    assert r.counters["revised"] == int(used)
    assert metadata["input_tokens"] == 16
    assert metadata["output_tokens"] == 20
    assert metadata["thinking_blocks"] == ["draft thinking", "revision thinking"]
    provider.converse.assert_awaited_once()
    assert response.text == revision
    assert messages == [{"role": "user", "content": [{"text": "user question"}], "model_metadata": {"message_id": 42}}]


@pytest.mark.asyncio
async def test_c2_rechecks_completion_ownership_after_retrieval_before_provider_call():
    r = _revise_reprise(state=LIVE)
    provider = MagicMock()
    provider.converse = AsyncMock()
    ownership_lost = AsyncMock(return_value=True)
    recall = [("id", f"- {RECALL_MARKER} [fact] X", 0.9)]

    with patch.object(r, "_reprime_for_revise", AsyncMock(return_value=recall)):
        out = await r.maybe_revise(
            provider=provider,
            system_prompt=("sys", ""),
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            max_tokens=1000,
            thinking_blocks=["novel thinking"],
            draft="draft answer",
            metadata={},
            ownership_lost=ownership_lost,
        )

    assert out == "draft answer"
    ownership_lost.assert_awaited_once()
    provider.converse.assert_not_awaited()


@pytest.mark.asyncio
async def test_c2_single_revise_cap_per_turn():
    r = _revise_reprise(state=LIVE)
    provider = MagicMock()
    provider.converse = AsyncMock(
        return_value=SimpleNamespace(text="R2", thinking_blocks=[], usage={"input": 1, "output": 1})
    )
    recall = [("id", f"- {RECALL_MARKER} [fact] X", 0.9)]
    with patch.object(r, "_reprime_for_revise", AsyncMock(return_value=recall)):
        first = await r.maybe_revise(
            provider=provider,
            system_prompt=("s", ""),
            messages=[{"role": "user", "content": "a"}],
            tools=[],
            max_tokens=100,
            thinking_blocks=["t"],
            draft="d1",
            metadata={},
        )
        second = await r.maybe_revise(
            provider=provider,
            system_prompt=("s", ""),
            messages=[{"role": "user", "content": "a"}],
            tools=[],
            max_tokens=100,
            thinking_blocks=["t"],
            draft="d2",
            metadata={},
        )
    assert first == "R2"
    assert second == "d2"  # capped — second turn returns its draft unchanged
    assert provider.converse.await_count == 1


@pytest.mark.asyncio
async def test_c2_streaming_cannot_revise_live():
    """can_revise_live=False (streaming) forces shadow behavior even when flag is LIVE:
    R1 was already streamed, so it cannot be un-sent."""
    r = _revise_reprise(state=LIVE)
    provider = MagicMock()
    provider.converse = AsyncMock()
    recall = [("id", f"- {RECALL_MARKER} [fact] X", 0.9)]
    with patch.object(r, "_reprime_for_revise", AsyncMock(return_value=recall)):
        out = await r.maybe_revise(
            provider=provider,
            system_prompt=("s", ""),
            messages=[{"role": "user", "content": "a"}],
            tools=[],
            max_tokens=100,
            thinking_blocks=["t"],
            draft="streamed draft",
            metadata={},
            can_revise_live=False,
        )
    assert out == "streamed draft"
    provider.converse.assert_not_called()
    assert r.counters["would_revise"] == 1


# ── build_reprise_state + Stage A shadow ─────────────────────────────────


def _ctx():
    return SimpleNamespace(venue="telegram", chat_id=1, topic_id=None)


@pytest.mark.asyncio
async def test_build_window_cues_none_when_both_flags_off():
    cognitive_config.set_for_testing("priming_context_cues", 0.0)
    cognitive_config.set_for_testing("priming_thinking_cues", 0.0)
    with patch("kai_memory.store.embeddings.generate_embedding", AsyncMock()) as emb:
        cues = await ir.build_window_cues([{"role": "user", "content": "hi"}], _ctx())
    assert cues is None
    emb.assert_not_awaited()  # byte-identical: no embeds when both off


@pytest.mark.asyncio
async def test_build_window_cues_single_message_not_windowed():
    """Single-message turn ⇒ is_windowed=False, one weight-1.0 cue (byte-identical path)."""
    cognitive_config.set_for_testing("priming_context_cues", 2.0)  # even LIVE
    with patch("kai_memory.store.embeddings.generate_embedding", AsyncMock(return_value=[1.0, 0.0])):
        cues = await ir.build_window_cues([{"role": "user", "content": "one message only"}], _ctx())
    assert cues is not None
    assert cues.is_windowed is False
    assert len(cues.segments) == 1
    assert cues.segments[0].weight == 1.0


@pytest.mark.asyncio
async def test_build_window_cues_multi_message_windowed():
    cognitive_config.set_for_testing("priming_context_cues", 1.0)  # shadow
    with patch("kai_memory.store.embeddings.generate_embedding", AsyncMock(return_value=[1.0, 0.0])):
        cues = await ir.build_window_cues(
            [
                {"role": "user", "content": "older question about the deadline"},
                {"role": "assistant", "content": [{"text": "Kai reply"}]},
                {"role": "user", "content": "and what about the budget?"},
            ],
            _ctx(),
        )
    assert cues is not None
    assert cues.is_windowed is True
    assert len(cues.segments) == 2  # two user messages
    assert cues.segments[0].weight == 1.0  # newest brightest


@pytest.mark.asyncio
async def test_build_reprise_state_inert_when_both_flags_off():
    cognitive_config.set_for_testing("priming_context_cues", 0.0)
    cognitive_config.set_for_testing("priming_thinking_cues", 0.0)
    state = await ir.build_reprise_state(None, "hi", _ctx(), [])
    assert state._state == OFF
    assert not state.active


@pytest.mark.asyncio
async def test_build_reprise_state_single_cue_logs_last_msg(caplog):
    cognitive_config.set_for_testing("priming_context_cues", 1.0)
    cognitive_config.set_for_testing("priming_thinking_cues", 0.0)
    cues = ir.WindowCues(segments=[ir.SegmentCue("only msg", 1.0, [1.0, 0.0])], is_windowed=False)
    with caplog.at_level("INFO"):
        await ir.build_reprise_state(cues, "only msg", _ctx(), [uuid.uuid4()])
    line = next(rec.message for rec in caplog.records if "memory.window.shadow" in rec.message)
    assert '"cue_mode": "last_msg"' in line
    assert '"window_decisive_rate": null' in line  # no counterfactual for a single cue


@pytest.mark.asyncio
async def test_stage_a_shadow_counterfactual_runs_full_windowed_prime(caplog):
    """The SHADOW counterfactual runs the FULL windowed prime (all 4 paths, read-only,
    own session) vs the ACTUAL live top-8; decisive = sets differ. This is the honest
    measure the narrow semantic-only diff missed."""
    cognitive_config.set_for_testing("priming_context_cues", 1.0)  # shadow
    cognitive_config.set_for_testing("priming_thinking_cues", 0.0)

    live_id = uuid.uuid4()
    windowed_ids = [uuid.uuid4(), uuid.uuid4()]  # different set ⇒ decisive
    prime_svc = MagicMock()
    prime_svc.prime_with_traces = AsyncMock(return_value=("", windowed_ids, []))
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=MagicMock())
    cm.__aexit__ = AsyncMock(return_value=False)
    scope = MagicMock(return_value=cm)

    cues = ir.WindowCues(
        segments=[ir.SegmentCue("newest", 1.0, [1.0, 0.0]), ir.SegmentCue("older", 0.6, [0.0, 1.0])],
        is_windowed=True,
    )
    with (
        patch("kai_memory.ports.session_scope", scope),
        patch("kai_memory.retrieval.memory_priming.MemoryPrimingService", return_value=prime_svc),
        caplog.at_level("INFO"),
    ):
        await ir.build_reprise_state(cues, "and what about the budget?", _ctx(), [live_id])
        # The counterfactual is dispatched as a BACKGROUND task (off the hot path) — drain it.
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        await asyncio.gather(*pending, return_exceptions=True)

    # own session opened; full windowed prime run read-only with the cues
    scope.assert_called()
    _args, kwargs = prime_svc.prime_with_traces.call_args
    assert kwargs.get("record_access") is False
    assert kwargs.get("window_cues") is cues
    assert kwargs.get("emit_log") is False  # no duplicate retrieve.complete for the counterfactual
    line = next(rec.message for rec in caplog.records if "memory.window.shadow" in rec.message)
    assert '"cue_mode": "windowed"' in line
    assert '"window_decisive_rate": 1' in line  # {windowed} != {live}


@pytest.mark.asyncio
async def test_stage_c_disabled_when_no_cue_embeddings():
    """No cue set (empty user text / all embeds failed) ⇒ Stage C is disabled — otherwise
    the novelty gate defaults to 0.0 and fires on every thinking block (CR finding)."""
    cognitive_config.set_for_testing("priming_context_cues", 0.0)
    cognitive_config.set_for_testing("priming_thinking_cues", 1.0)  # Stage C shadow, but no cues
    state = await ir.build_reprise_state(None, None, _ctx(), [])
    assert state._state == OFF  # can't judge novelty without a cue set
    assert not state.active


# ── Stage A LIVE windowed collection (memory_priming) ────────────────────


def _priming_svc():
    from kai_memory.retrieval.memory_priming import MemoryPrimingService

    svc = MemoryPrimingService(MagicMock())
    svc.memory_svc = MagicMock()
    svc.memory_svc.find_entities_by_names = AsyncMock(return_value=[])
    svc.memory_svc.find_entities_by_lowercase_tokens = AsyncMock(return_value=[])
    svc.memory_svc.traverse_entity_graph = AsyncMock(return_value=[])
    svc.memory_svc.get_intention_traces = AsyncMock(return_value=[])
    svc.memory_svc.get_entity_linked_traces_with_entity = AsyncMock(return_value=[])
    return svc


@pytest.mark.asyncio
async def test_windowed_path1_weighted_max_semantic():
    """Path 1: each candidate's semantic = weighted-MAX over segments of weight*sim."""
    svc = _priming_svc()
    a, b = SimpleNamespace(id=uuid.uuid4()), SimpleNamespace(id=uuid.uuid4())

    async def fake_search(embedding, limit=20):
        if list(embedding) == [1.0, 0.0]:  # newest, weight 1.0
            return [(a, 0.8)]
        return [(a, 0.9), (b, 0.7)]  # older, weight 0.6

    svc.memory_svc.search_traces_by_vector = AsyncMock(side_effect=fake_search)
    cues = ir.WindowCues(
        segments=[ir.SegmentCue("nw", 1.0, [1.0, 0.0]), ir.SegmentCue("od", 0.6, [0.0, 1.0])],
        is_windowed=True,
    )
    all_traces, _paths = await svc._collect_candidates_windowed(cues, limit=8)
    # a: max(1.0*0.8, 0.6*0.9) = 0.8 ; b: 0.6*0.7 = 0.42
    assert all_traces[str(a.id)][1]["semantic"] == pytest.approx(0.8)
    assert all_traces[str(b.id)][1]["semantic"] == pytest.approx(0.42)


@pytest.mark.asyncio
async def test_windowed_path1_caps_search_count():
    """Path 1 bounds the per-segment pgvector queries to the highest-weight segments
    (CR: worst-case latency on the 2 GB ARM hot path)."""
    from kai_memory.retrieval import memory_priming as mp

    svc = _priming_svc()
    svc.memory_svc.search_traces_by_vector = AsyncMock(return_value=[])
    # 12 segments (more than the cap); descending weights
    segs = [ir.SegmentCue(f"s{i}", 1.0 - i * 0.05, [float(i), 1.0]) for i in range(12)]
    cues = ir.WindowCues(segments=segs, is_windowed=True)
    await svc._collect_candidates_windowed(cues, limit=8)
    assert svc.memory_svc.search_traces_by_vector.await_count == mp._WINDOWED_MAX_SEARCH_SEGMENTS
    # the highest-weight segment (s0) was searched
    searched = {tuple(call.args[0]) for call in svc.memory_svc.search_traces_by_vector.await_args_list}
    assert (0.0, 1.0) in searched


@pytest.mark.asyncio
async def test_windowed_paths_2_3_entity_weight_scaling():
    """Paths 2/3: entity union across segments; graph = the entity's best segment weight."""
    svc = _priming_svc()
    svc.memory_svc.search_traces_by_vector = AsyncMock(return_value=[])

    async def fake_find(names):
        return [SimpleNamespace(id=f"ent-{n}") for n in names]

    svc.memory_svc.find_entities_by_names = AsyncMock(side_effect=fake_find)
    tx, ty = SimpleNamespace(id=uuid.uuid4()), SimpleNamespace(id=uuid.uuid4())
    svc.memory_svc.get_entity_linked_traces_with_entity = AsyncMock(return_value=[(tx, "ent-Maria"), (ty, "ent-Boris")])
    cues = ir.WindowCues(
        segments=[
            ir.SegmentCue("Maria said hi", 1.0, [1.0, 0.0]),  # newest, weight 1.0
            ir.SegmentCue("Boris asked something", 0.6, [0.0, 1.0]),  # older, weight 0.6
        ],
        is_windowed=True,
    )
    all_traces, paths = await svc._collect_candidates_windowed(cues, limit=8)
    assert all_traces[str(tx.id)][1]["graph"] == pytest.approx(1.0)  # Maria (newest)
    assert all_traces[str(ty.id)][1]["graph"] == pytest.approx(0.6)  # Boris (older)
    assert "entity_link" in paths[str(tx.id)]


@pytest.mark.asyncio
async def test_cue_mode_stamped_on_retrieve_complete(caplog):
    """cue_mode rides on memory.retrieve.complete: windowed when cues fed, else last_msg."""
    svc = _priming_svc()
    svc.memory_svc.search_traces_by_vector = AsyncMock(return_value=[])
    svc.memory_svc.search_traces_raw = AsyncMock(return_value=[])
    cues = ir.WindowCues(
        segments=[ir.SegmentCue("nw", 1.0, [1.0, 0.0]), ir.SegmentCue("od", 0.6, [0.0, 1.0])], is_windowed=True
    )
    with caplog.at_level("INFO"):
        await svc.prime_with_traces("the budget", chat_ctx=_ctx(), record_access=False, window_cues=cues)
        await svc.prime_with_traces("the budget", chat_ctx=_ctx(), record_access=False)  # no cues
    lines = [r.message for r in caplog.records if "memory.retrieve.complete" in r.message]
    assert any('"cue_mode": "windowed"' in ln for ln in lines)
    assert any('"cue_mode": "last_msg"' in ln for ln in lines)


@pytest.mark.asyncio
async def test_windowed_none_uses_single_message_path():
    """window_cues=None must NOT touch the windowed collector (byte-identical guarantee)."""
    svc = _priming_svc()
    svc._collect_candidates_windowed = AsyncMock(side_effect=AssertionError("should not be called"))
    svc.memory_svc.search_traces_raw = AsyncMock(return_value=[])
    # No window_cues ⇒ original _collect_candidates path; windowed collector untouched.
    await svc.prime_with_traces("hello", chat_ctx=_ctx(), record_access=False, window_cues=None)
    svc._collect_candidates_windowed.assert_not_awaited()


def test_primed_set_churn_first_turn_is_none_then_computes():
    ctx = SimpleNamespace(venue="telegram", chat_id=42, topic_id=None)
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    # first turn: no baseline
    assert ir._churn(ctx, [a, b]) is None
    # second turn: dropped `b`, kept `a`, added `c` ⇒ 1 of 2 previous dropped = 0.5
    assert ir._churn(ctx, [a, c]) == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_windowed_path1_spends_one_total_budget(monkeypatch):
    """Deterministic slow searches consume 3 s each; the second gets only 1 s."""
    from kai_memory.retrieval import memory_priming as mp

    svc = _priming_svc()
    svc._recover_session = AsyncMock()
    now = 10.0
    timeouts = []
    seen = []
    survivor = SimpleNamespace(id=uuid.uuid4())
    svc.memory_svc.find_entities_by_names = AsyncMock(return_value=[SimpleNamespace(id="synthetic-person")])

    async def slow_search(embedding, limit):
        seen.append(embedding)
        return [(survivor, 0.9)]

    async def wait_for(operation, timeout):  # noqa: ASYNC109 — deterministic asyncio.wait_for test adapter
        nonlocal now
        timeouts.append(timeout)
        result = await operation
        now += min(3.0, timeout)
        if timeout < 3.0:
            raise TimeoutError
        return result

    monkeypatch.setattr(mp, "time", SimpleNamespace(monotonic=lambda: now))
    monkeypatch.setattr(mp, "asyncio", SimpleNamespace(wait_for=wait_for))
    monkeypatch.setattr(mp, "_SEMANTIC_TIMEOUT_S", 4.0)
    monkeypatch.setattr(mp, "_WINDOWED_TOTAL_BUDGET_S", 4.0)
    svc.memory_svc.search_traces_by_vector = AsyncMock(side_effect=slow_search)
    cues = ir.WindowCues(
        segments=[ir.SegmentCue(f"Person {i}", 1.0 - i * 0.1, [float(i)]) for i in range(3)], is_windowed=True
    )
    before = getattr(mp, "_windowed_budget_skipped", 0)
    all_traces, _ = await svc._collect_candidates_windowed(cues, limit=8)
    assert now == 14.0
    assert timeouts == [4.0, 1.0]
    assert seen == [[0.0], [1.0]]
    assert mp._windowed_budget_skipped - before == 1
    assert str(survivor.id) in all_traces  # arithmetic only; real rollback coverage lives in the PostgreSQL suite
    svc._recover_session.assert_awaited_once()
    svc.memory_svc.get_intention_traces.assert_awaited_once()  # fail-open later paths


@pytest.mark.asyncio
async def test_windowed_path1_exhausted_budget_never_starts_a_search(monkeypatch):
    from kai_memory.retrieval import memory_priming as mp

    svc = _priming_svc()
    svc.memory_svc.search_traces_by_vector = AsyncMock(return_value=[])
    ticks = iter([10.0, 14.0])
    monkeypatch.setattr(mp, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    monkeypatch.setattr(mp, "_WINDOWED_TOTAL_BUDGET_S", 4.0)
    before = getattr(mp, "_windowed_budget_skipped", 0)
    cues = ir.WindowCues(segments=[ir.SegmentCue("x", 1.0, [1.0])], is_windowed=True)
    await svc._collect_candidates_windowed(cues, limit=8)
    svc.memory_svc.search_traces_by_vector.assert_not_awaited()
    assert mp._windowed_budget_skipped - before == 1


@pytest.fixture(autouse=True)
def _synthetic_generation_boundary(monkeypatch):
    # The memory mechanism receives a host-owned response/telemetry adapter.
    monkeypatch.setattr("kai_memory.ports.record_generation_provenance", record_generation_provenance)
    monkeypatch.setattr("kai_memory.ports.MODEL_METADATA_KEY", "model_metadata")
