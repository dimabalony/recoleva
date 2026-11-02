"""Embedding generation: input truncation, zero-vector guard and error contract (DB-843: a port).

The vectors come from the host's embeddings port, ``ports.embedder``, read on every
call (``await embedder.embed(texts, api_key=None)``). The production host binds the
OpenAI adapter in ``app/services/host_adapters.py``; another host binds its own.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import cast
import time
import traceback

from kai_memory import ports

# Pinned to the pre-DB-842 module path: CloudWatch queries, feature-watch criteria and
# log filters match the logger name, which must not follow the module's move.
_LEGACY_LOGGER_NAME = "app.services.embeddings"
logger = logging.getLogger(_LEGACY_LOGGER_NAME)

# OpenAI text-embedding-3-small produces 1536-dimensional vectors
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
MAX_INPUT_TOKENS = 8191  # Model limit


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails or yields a degenerate vector.

    Replaces the old silent ``[0.0] * 1536`` fallback. A zero vector NaN-sorts
    to the TOP of every pgvector cosine search, so a single stored zero vector
    poisons retrieval corpus-wide. Callers MUST handle this explicitly:
      * storage paths persist a NULL embedding (backfillable), never a zero;
      * retrieval paths degrade to empty semantic results.
    See the 2026-06-03 zero-vector poisoning incident.
    """


def is_zero_vector(vec: list[float]) -> bool:
    """True if ``vec`` is empty or all-zero (i.e. has zero L2 norm).

    A zero-norm embedding is degenerate: cosine distance against it is NaN,
    which sorts ahead of every real result in pgvector. Storage layers use
    this as a last-line guard before persisting.
    """
    return not any(vec)


# Hard upper bound on input size, expressed in UTF-8 bytes.
# cl100k_base tokens decode to at least one byte each, so
# `len(text.encode("utf-8")) <= MAX_INPUT_TOKENS` is a tokenizer-free
# guarantee that the encoded token count stays under the model's 8191
# ceiling. Char-based caps were unsafe for CJK/emoji where a single char
# encodes to 3-4 bytes — 16k chars of CJK is ~50k bytes and triggers the
# 400 we were trying to fix. UTF-8 byte cap is the only way to bound
# tokens without tiktoken.
_MAX_INPUT_BYTES = MAX_INPUT_TOKENS


def _truncate_to_safe_bytes(text: str, max_bytes: int = _MAX_INPUT_BYTES) -> str:
    """Trim ``text`` so its UTF-8 byte length is <= ``max_bytes``.

    ``errors='ignore'`` drops any partial multi-byte char at the boundary
    so we never emit an invalid UTF-8 string back to OpenAI.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


# Diagnostics contain structural metadata only; never input text or provider error bodies.


def _diagnostic_payload(text: str, *, kind: str) -> dict[str, object]:
    """Build a structured log payload pinpointing the offending input.

    ``kind`` is ``"single"`` or ``"batch:N/M"`` (item index in batch) so the
    aggregator can tell apart per-item failures from batch-level ones.
    """
    stack = traceback.extract_stack()
    # Find the most recent frame outside this module — that's the caller
    # we want to attribute. The fallback to "unknown" handles the unusual
    # case where the entire stack is inside this file.
    caller = "unknown"
    for frame in reversed(stack[:-1]):
        if "/embeddings.py" not in frame.filename:
            parts = frame.filename.rsplit("/", 2)
            short = f"{parts[-2]}/{parts[-1]}" if len(parts) >= 2 else frame.filename
            caller = f"{short}:{frame.lineno}:{frame.name}"
            break
    encoded = text.encode("utf-8", errors="replace")
    return {
        "event_type": "embeddings.input_diagnostic",
        "kind": kind,
        "chars": len(text),
        "bytes": len(encoded),
        "sha256_prefix": hashlib.sha256(encoded).hexdigest()[:16],
        "caller": caller,
    }


def _log_embed_timing(*, kind: str, duration_s: float, n_inputs: int, ok: bool, embedder: object = None) -> None:
    """Emit the embed-call stage timing.

    Split-stage triage: this measures ONLY the OpenAI round trip. The pgvector
    query is timed separately at its call site
    (``cognitive_memory.search_traces_raw``, ``embeddings.db_query`` event).
    Previously a "semantic search timed out" line named one 4s budget with no
    way to tell an OpenAI stall from a slow DB scan — the two have opposite
    fixes, so they must be attributable apart.
    """
    logger.info(
        "embeddings.embed_call %s",
        json.dumps(
            {
                "event_type": "embeddings.embed_call",
                "stage": "openai_embed",
                "kind": kind,
                "duration_ms": round(duration_s * 1000, 1),
                "n_inputs": n_inputs,
                "ok": ok,
                "timeout_s": getattr(embedder, "timeout_s", None),
                "max_retries": getattr(embedder, "max_retries", None),
            }
        ),
    )


async def generate_embedding(
    text: str,
    api_key: str | None = None,
) -> list[float]:
    """Generate the embedding vector for one text through the host's embeddings port.

    Args:
        text: Text to embed. Will be UTF-8-byte-truncated to fit the model
            ceiling — see ``_truncate_to_safe_bytes`` for the safety reasoning.
        api_key: Optional per-call key, passed to the embeddings port.

    Returns:
        List of 1536 floats representing the embedding vector.

    Raises:
        EmbeddingError: on API failure OR if the API returns a degenerate
            zero vector. NEVER returns a zero vector — that would poison every
            downstream pgvector search (see ``EmbeddingError``).
    """
    truncated_text = _truncate_to_safe_bytes(text)
    embedder = ports.embedder

    started = time.monotonic()
    try:
        embedding = (await embedder.embed([truncated_text], api_key=api_key))[0]
        _log_embed_timing(kind="single", duration_s=time.monotonic() - started, n_inputs=1, ok=True, embedder=embedder)

    except Exception as e:
        _log_embed_timing(kind="single", duration_s=time.monotonic() - started, n_inputs=1, ok=False, embedder=embedder)
        # Pre-truncation original text gives the most useful diagnostic; the
        # API saw the truncated copy but the caller passed the full thing.
        payload = _diagnostic_payload(text, kind="single")
        payload["error_type"] = type(e).__name__
        logger.error("embeddings.input_diagnostic %s", json.dumps(payload, ensure_ascii=False))
        raise EmbeddingError("embedding generation failed") from None

    # Defense-in-depth: a degenerate zero vector must not propagate even when
    # the API call "succeeds". Treat it as a failure so callers store NULL.
    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise EmbeddingError("embedding dimension must be 1536")
    if is_zero_vector(embedding):
        raise EmbeddingError("embedding API returned a zero vector")
    return cast("list[float]", embedding)


async def generate_embeddings_batch(
    texts: list[str],
    api_key: str | None = None,
) -> list[list[float]]:
    """Generate embedding vectors for multiple texts in one embeddings-port call.

    Args:
        texts: List of texts to embed (each UTF-8-byte-truncated if too long).
        api_key: Optional per-call key, passed to the embeddings port.

    Returns:
        List of embedding vectors (same order as input texts).

    Raises:
        EmbeddingError: on API failure OR if any returned vector is a
            degenerate zero vector. NEVER returns zero vectors — the caller
            must fall back (NULL storage / empty results) rather than persist
            a poison row.
    """
    if not texts:
        return []

    truncated = [_truncate_to_safe_bytes(t) for t in texts]
    embedder = ports.embedder

    started = time.monotonic()
    try:
        # The port returns one vector per text, in input order. A bound embedder that
        # returns another count would pair vectors with the wrong texts downstream.
        embeddings = await embedder.embed(truncated, api_key=api_key)
        if len(embeddings) != len(truncated):
            raise EmbeddingError(f"embedder returned {len(embeddings)} vectors for {len(truncated)} inputs")
        _log_embed_timing(
            kind="batch", duration_s=time.monotonic() - started, n_inputs=len(texts), ok=True, embedder=embedder
        )

    except Exception as e:
        _log_embed_timing(
            kind="batch", duration_s=time.monotonic() - started, n_inputs=len(texts), ok=False, embedder=embedder
        )
        # Batch path: log diagnostic for the LONGEST input plus aggregate
        # stats. OpenAI rejects the whole batch when any one item is too big,
        # so the longest item is overwhelmingly the offender. Aggregate count
        # + total bytes catches the rare case where the offender is the sum
        # of many medium-sized items.
        longest_idx = max(range(len(texts)), key=lambda i: len(texts[i].encode("utf-8")))
        payload = _diagnostic_payload(texts[longest_idx], kind=f"batch:{longest_idx}/{len(texts)}")
        payload["error_type"] = type(e).__name__
        payload["batch_total_bytes"] = sum(len(t.encode("utf-8")) for t in texts)
        payload["batch_count"] = len(texts)
        logger.error("embeddings.input_diagnostic %s", json.dumps(payload, ensure_ascii=False))
        raise EmbeddingError("batch embedding generation failed") from None

    # Defense-in-depth: a zero vector for any item means that item failed.
    # Raise so the caller falls back wholesale rather than persisting a poison
    # row mixed in with good ones.
    if any(len(v) != EMBEDDING_DIMENSIONS for v in embeddings):
        raise EmbeddingError("embedding dimension must be 1536")
    if any(is_zero_vector(v) for v in embeddings):
        raise EmbeddingError("embedding API returned a zero vector in batch")
    return cast("list[list[float]]", embeddings)


# NOTE: ``generate_event_embedding`` and ``generate_topic_embedding`` were
# removed in this PR — both had zero callers across app/, tests/, scripts/
# (verified via grep 2026-05-15). The right embedding surface is now
# ``generate_embedding`` / ``generate_embeddings_batch`` with the caller
# responsible for selecting a sensible source-text shape (e.g. summary, not
# raw transcript — see ``memory_priming.py:embedding_query``).
