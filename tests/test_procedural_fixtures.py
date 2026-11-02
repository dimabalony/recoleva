"""Phase C E0/E9 — fixture-driven capture-fidelity + retrieval/scope checks.

Runs the pre-registered E0 capture fixtures through the deterministic trust gate
(0 injection mints is the hard bar) and validates the E9 retrieval set's structure
plus the serving scope firewall against its hard-negative scope-violation query.
"""

from __future__ import annotations

import json
from pathlib import Path

from memory_test_support import ConversationMessage
from memory_test_support import ChatContext
from kai_memory.procedural.procedural_serving import _in_scope
from kai_memory.procedural.rule_capture import evaluate_candidate

_FIX = Path(__file__).parent / "fixtures" / "procedural"
_E0 = json.loads((_FIX / "e0_capture_fixtures.json").read_text(encoding="utf-8"))
_E9 = json.loads((_FIX / "e9_retrieval_set.json").read_text(encoding="utf-8"))


def _build_window(rows: list[dict]) -> list[ConversationMessage]:
    out = []
    for r in rows:
        meta = dict(r.get("meta") or {})
        text = r["text"]
        prefix = r.get("reply_prefix")
        if prefix is not None:  # assemble the real handler shape + span marker
            text = f"{prefix}\n{text}"
            meta["reply_quote_len"] = len(prefix)
        out.append(
            ConversationMessage(
                role=r["role"],
                content_blocks=[{"text": text}],
                sender_user_id=r.get("sender_user_id"),
                message_metadata=meta,
            )
        )
    return out


def test_e0_every_case_matches_expected_verdict():
    admin_ids = _E0["admin_ids"]
    for case in _E0["cases"]:
        window = _build_window(case["window"])
        res = evaluate_candidate(case["candidate"], window, admin_ids)
        exp = case["expect"]
        assert res.accepted is exp["accepted"], f"{case['name']}: accepted mismatch ({res.reason})"
        if not exp["accepted"] and "reason" in exp:
            assert res.reason == exp["reason"], f"{case['name']}: reason {res.reason} != {exp['reason']}"
        if exp.get("status"):
            assert res.status == exp["status"], f"{case['name']}: status {res.status} != {exp['status']}"


def test_e0_zero_injection_mints():
    """The hard E0 bar: NOT ONE injection vector may mint (all four + variants)."""
    admin_ids = _E0["admin_ids"]
    injections = [c for c in _E0["cases"] if c["kind"] == "injection"]
    assert injections, "fixture set must contain injection vectors"
    for case in injections:
        res = evaluate_candidate(case["candidate"], _build_window(case["window"]), admin_ids)
        assert not res.accepted, f"INJECTION MINTED: {case['name']}"


def test_e9_structure_and_referential_integrity():
    rule_ids = {r["id"] for r in _E9["rules"]}
    assert len(rule_ids) == len(_E9["rules"]), "duplicate rule ids in E9 set"
    for q in _E9["queries"]:
        for rid in q.get("intended", []) + q.get("hard_negatives", []):
            assert rid in rule_ids, f"E9 query references unknown rule id {rid}"


def test_e9_person_scope_firewall():
    """The scope-violation guard query: a person-scoped rule must not serve to a
    different sender (E9 zero-scope-violations)."""
    person_rule = next(r for r in _E9["rules"] if r["id"] == "r_person12_ru_only")
    proc = {"scope": person_rule["scope"]}
    stranger_ctx = ChatContext(chat_id=1, venue="telegram", sender_user_id=999)
    owner_ctx = ChatContext(chat_id=1, venue="telegram", sender_user_id=12345)
    assert _in_scope(proc, stranger_ctx) is False
    assert _in_scope(proc, owner_ctx) is True
