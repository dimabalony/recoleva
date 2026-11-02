"""Reserved provenance for EPHEMERAL benchmark probes (DB-591).

A benchmark that exercises the REAL write path (today: the capture benchmark's
``run_one_probe``) writes synthetic facts into prod memory and then hard-deletes
them. Every such write is tagged by a reserved synthetic ``chat_id`` — far outside
any real Telegram id — which is the ownership tag cleanup deletes on.

This module is the ONE home for that tag, so the WRITE side can honor the same
boundary the cleanup side already does. It deliberately holds no imports beyond
the stdlib: both ``app.services.chat.memory_extraction`` (write side) and
``app.services.benchmarks.capture_probe`` (probe driver) import it, and a
heavier module here would create an import cycle between them.

WHY THE WRITE SIDE NEEDS IT (the DB-591 incident): the probe's docstring promised
"cleanup can never touch a real trace", and cleanup honored it — but L2 extraction
did not. A probe statement ("my legal name is …") landed in the same ``full_name``
slot as the admin's real legal-name trace, Haiku flagged a contradiction, and the
extraction path obeyed it: a synthetic probe fact INVALIDATED a real durable
memory, which retrieval then filtered out. An ephemeral write must be inert
against non-ephemeral memory in BOTH directions — it may create rows it owns, and
it may never mutate rows it does not.
"""

from __future__ import annotations

# Reserved synthetic chat id for capture probes — far below any real Telegram chat
# id (groups are ~ -1e12..-1e13; users are positive), so a real trace can never
# share it. Combined with per-probe topic ids and captured-id deletes, cleanup is
# doubly scoped.
CAPTURE_PROBE_CHAT_ID = -8_500_000_000_000

# Every reserved chat id whose writes are EPHEMERAL. A frozenset (not a bare
# equality) so a second probe harness gets the same guarantees by adding one line
# here rather than by copying a comparison into the write path.
EPHEMERAL_PROBE_CHAT_IDS: frozenset[int] = frozenset({CAPTURE_PROBE_CHAT_ID})


def is_ephemeral_probe_chat(chat_id: int | None) -> bool:
    """True when writes from this chat are ephemeral benchmark probe writes.

    ``chat_id=None`` (headless venues: event/heartbeat/wake) is NOT ephemeral —
    those are real internal callers whose memory is kept.
    """
    return chat_id is not None and chat_id in EPHEMERAL_PROBE_CHAT_IDS
