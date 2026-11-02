"""Conversation-identity + category helpers shared by the procedural WRITE and READ paths.

``rule_capture`` stamps ``typed_metadata.procedural.source_chat_ids`` with the
conversations that taught a rule; ``procedural_serving`` reads that list to decide
whether a rule may serve in a group (DB-436 origin-conversation serving). Both sides
MUST agree on the key format and on how legacy rows are normalized — if they drift,
a rule silently stops serving in the very conversation that taught it. That is why
this lives in one module instead of being spelled twice.

``CATEGORY_ORDER`` and ``PROC_KINDS`` are the same story for the vocabularies: capture
allowlists the categories, the serve-all block groups by them; every procedural query
filters on the same two kinds. This module deliberately imports nothing from the app
(in particular not ``app.telegram``), so every side can share these freely.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

# The rule-category vocabulary — form/process only, never stance (design §1.4/§2.2).
# ORDERED, because the serve-all standing block groups by category in exactly this
# order and its byte-stability is what keeps the prompt cache hitting (DB-487 §3).
# ``rule_capture.CATEGORY_ALLOWLIST`` is derived from it, so a new category can never
# be allowed on the write side without getting a render position on the read side.
# Ordered mapping: the key order IS the render order, and every category carries its
# display heading here — so a new category cannot be allowlisted at capture without also
# getting a render position and a title (it would otherwise silently land under "Other").
CATEGORY_TITLES: dict[str, str] = {
    "format": "Format",
    "workflow": "Workflow",
    "tool_use": "Tool use",
    "communication": "Communication",
    "scheduling": "Scheduling",
}
CATEGORY_ORDER: tuple[str, ...] = tuple(CATEGORY_TITLES)

# The two ``typed_metadata.procedural.kind`` values that are behavioral rules. Every
# query that selects rules filters on exactly this set — serving, the nightly ledger,
# the liveness check, capture's dedup load — so it lives in one place rather than being
# re-spelled per module (AGENTS.md: shared constants for repeated enum checks).
PROC_KINDS: frozenset[str] = frozenset({"rule", "habit"})

# Rule lifecycle statuses. Only ``active`` serves; ``suspended`` is the ledger's demotion
# state and ``candidate`` survives only on legacy rows the DB-487 backfill did not reach
# (first-teach activation no longer mints candidates). Both of the latter return to
# ``active`` on fresh trusted teaching — that set is spelled once, here.
PROC_STATUS_ACTIVE = "active"
PROC_STATUS_SUSPENDED = "suspended"
PROC_STATUS_CANDIDATE = "candidate"
REACTIVATABLE_STATUSES: frozenset[str] = frozenset({PROC_STATUS_SUSPENDED, PROC_STATUS_CANDIDATE})

# L2 Part-5 rule-compliance verdicts. The PRODUCER (memory_extraction: the judge prompt
# and the persist path) and the CONSUMER (procedural_reinforcement: the nightly ledger)
# must agree on these strings exactly — a rename on one side alone silently stops the
# ledger from ever seeing that verdict, which is the failure this whole PR exists to fix.
VERDICT_FOLLOWED = "followed"
VERDICT_VIOLATED = "violated"
VERDICT_NOT_APPLICABLE = "not_applicable"
# DB-487: wrong-rule-fired. PERSISTED but INERT — the ledger scores only followed and
# violated, so this accumulates as telemetry without becoming a demotion signal. The
# research survey predicts this is serve-all's real failure mode ("many rules ->
# wrong-rule-fired, never rule-not-found"); what it should COST is a decision for real
# data after E1, not a default picked here.
VERDICT_MISAPPLIED = "misapplied"

# Verdicts the ledger acts on.
LEDGER_VERDICTS: frozenset[str] = frozenset({VERDICT_FOLLOWED, VERDICT_VIOLATED})
# Verdicts written onto the served trace. not_applicable writes nothing: it is neither
# success nor failure, and under serve-all it is the majority verdict, so persisting it
# would bloat success_signals for no reader.
PERSISTED_VERDICTS: frozenset[str] = frozenset({VERDICT_FOLLOWED, VERDICT_VIOLATED, VERDICT_MISAPPLIED})
# Every verdict the judge may emit — the counter buckets.
ALL_VERDICTS: tuple[str, ...] = (
    VERDICT_FOLLOWED,
    VERDICT_VIOLATED,
    VERDICT_NOT_APPLICABLE,
    VERDICT_MISAPPLIED,
)


def conv_key(chat_id: int | None, topic_id: int | None) -> str | None:
    """A conversation identity (§2.3). A Telegram supergroup TOPIC is a distinct
    conversation even though it shares the group's ``chat_id``, so the key includes
    BOTH — otherwise cross-topic teaching in one supergroup never satisfies
    "different conversation". NULL chat → no identity."""
    if chat_id is None:
        return None
    return f"{chat_id}:{'' if topic_id is None else topic_id}"


def normalize_conv_keys(raw: Any) -> list[str]:
    """Normalize a stored ``source_chat_ids`` list to canonical conv-key strings.

    Legacy rows hold a bare ``int`` chat id (pre-topic-aware capture); it maps to the
    topic-less key ``"{chat_id}:"`` so a stored ``[100]`` and fresh chat-100 evidence
    resolve to the SAME identity rather than counting as two conversations. Anything
    else is passed through as ``str`` — an unrecognized entry must never crash either
    the capture gate or the serving filter.
    """
    if not isinstance(raw, list):
        return []
    return [f"{v}:" if isinstance(v, int) and not isinstance(v, bool) else str(v) for v in raw]


@dataclass(frozen=True)
class MessageAnchor:
    """A plain (non-tool) assistant turn that served a rule (DB-629).

    What a compliance verdict attaches to when the turn has no execution trace. It lives
    here, with the verdict vocabulary, because the PRODUCER (``memory_extraction`` builds
    it from the window) and the WRITER (``rule_capture`` turns it into a pending ledger
    event) are in modules that must not import each other.

    ``source_user_id`` is the turn's TRIGGERING human, not a sender: an assistant row has
    no sender, so the §4.2 credit boundary resolves through the person whose message the
    reply answers — the same actor ``procedural_recorder`` stamps on an execution trace.
    """

    message_id: str
    source_user_id: int | None
    at: datetime | None


def trusted_actor(source_user_id: int | None, scope_person: int | None, admin_ids: str | None) -> bool:
    """The credit boundary (design §4.2): a user-supplied signal moves a rule's ledger only
    when it comes from an admin OR the rule's own ``scope.person``. Untrusted praise, or a
    stranger's rejected nudge, can never reinforce or demote a rule.

    Spelled ONCE, here, for the same reason the verdict vocabulary is: both the nightly
    ledger and the capture-time compliance writer gate on it, and a copy that drifts on one
    side is a security boundary that quietly stops holding.

    The ``is_admin_user`` import is deliberately LAZY. Importing ``app.telegram.auth`` at
    module load pulls in the ``app.telegram.router`` submodule, which overwrites the
    ``router`` OBJECT re-exported by ``app/telegram/__init__.py`` with the submodule —
    breaking ``app/routes``' ``from ..telegram import router`` at app.main build time
    (order-dependent). Keeping it inside the call is what lets this module stay importable
    from anywhere, which is the whole point of ``procedural_scope``.
    """
    from kai_memory.ports import is_admin_user

    return is_admin_user(source_user_id, admin_ids) or (scope_person is not None and source_user_id == scope_person)
