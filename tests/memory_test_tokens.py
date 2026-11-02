"""Control tokens for response suppression + output sanitization."""

from __future__ import annotations

import json
import re
from typing import Any, Final

SILENT_REPLY_TOKEN = "NO_REPLY"
HEARTBEAT_OK_TOKEN = "HEARTBEAT_OK"

_SILENT_TOKENS = {SILENT_REPLY_TOKEN, HEARTBEAT_OK_TOKEN}

_MSG_META_HEAD: Final = "[msg_meta:"
_MSG_META_BLOCK_PREFIX: Final = "[msg_meta: "
_LEGACY_METADATA_HEAD: Final = "Conversation info (untrusted metadata):\n"
_NEED_MORE: Final[int] = -1
_MSG_META_KEYS: Final = frozenset({"message_id", "time", "sender", "edited", "reactions"})
_MAX_METADATA_PREFIX_CHARS: Final[int] = 4096
_MODEL_MSG_ID_ALIAS_HEAD: Final = "[msg_mess_id: "
_MODEL_MSG_ID_ALIAS_RE: Final = re.compile(
    r"^\[msg_mess_id: (?P<message_id>[1-9][0-9]{0,18})\]",
)
_MAX_SIGNED_BIGINT: Final[int] = (1 << 63) - 1


def is_msg_meta_payload(value: object) -> bool:
    """Return whether ``value`` has the model-visible transport schema.

    Requiring the discriminator and known keys prevents a user-requested
    literal such as ``[msg_meta: {"example": true}]`` from being mistaken for
    infrastructure output.
    """
    return isinstance(value, dict) and type(value.get("message_id")) is int and set(value).issubset(_MSG_META_KEYS)


def _json_object_end(text: str, start: int) -> int | None:
    """Return the exclusive end of a complete JSON object, or ``None``.

    This scanner only identifies the outer object boundary. ``json.loads``
    validates the completed slice at the caller. Tracking strings explicitly
    keeps braces in sender names or other JSON strings from ending the envelope
    early.
    """
    if start >= len(text) or text[start] != "{":
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return idx + 1
    return None


def _parse_msg_meta_prefix(text: str, *, allow_leading_whitespace: bool) -> tuple[dict[str, Any], int] | None:
    """Parse one leading ``[msg_meta: {...}]`` block and return its end."""
    stripped = text.lstrip() if allow_leading_whitespace else text
    leading_whitespace = len(text) - len(stripped)
    if not stripped.startswith(_MSG_META_HEAD):
        return None

    cursor = leading_whitespace + len(_MSG_META_HEAD)
    while cursor < len(text) and text[cursor] in " \t":
        cursor += 1
    object_end = _json_object_end(text, cursor)
    if object_end is None or object_end >= len(text) or text[object_end] != "]":
        return None
    try:
        payload = json.loads(text[cursor:object_end])
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload, object_end + 1


def strip_msg_meta_envelope(text: str) -> str:
    """Strip structurally valid leading ``[msg_meta: {...}]`` envelope(s).

    This is the shared legacy-input parser for semantic consumers. It accepts
    any JSON-object payload because pre-cutover rows predate the strict current
    transport schema. Malformed or non-object examples remain authored text.
    """
    remainder = text
    while True:
        parsed = _parse_msg_meta_prefix(remainder, allow_leading_whitespace=True)
        if parsed is None:
            return remainder.lstrip()
        _payload, prefix_end = parsed
        remainder = remainder[prefix_end:]


def parse_exact_msg_meta_block(block: Any) -> dict[str, Any] | None:
    """Parse one exact standalone current-schema ``[msg_meta: {...}]`` block."""
    if not isinstance(block, dict):
        return None
    text = block.get("text")
    if not isinstance(text, str) or not text.startswith(_MSG_META_BLOCK_PREFIX):
        return None
    parsed = _parse_msg_meta_prefix(text, allow_leading_whitespace=False)
    if parsed is None:
        return None
    payload, prefix_end = parsed
    if prefix_end != len(text) or not is_msg_meta_payload(payload):
        return None
    return payload


def _consume_model_msg_id_alias(text: str, *, final: bool) -> int | None:
    """Consume the bounded model-invented alias observed in production.

    Sonnet has been observed compressing the canonical
    ``[msg_meta: {"message_id": ...}]`` block into
    ``[msg_mess_id: ...]``. This alias is never rendered by our code, so an
    exact, leading-only, directly-concatenated rule keeps the output backstop
    effective without treating similar bracketed prose as transport metadata.
    """
    if not text.startswith(_MODEL_MSG_ID_ALIAS_HEAD):
        if _MODEL_MSG_ID_ALIAS_HEAD.startswith(text):
            return None if final else _NEED_MORE
        return None

    value_start = len(_MODEL_MSG_ID_ALIAS_HEAD)
    cursor = value_start
    while cursor < len(text) and text[cursor] in "0123456789":
        cursor += 1

    digits = text[value_start:cursor]
    if not digits:
        return _NEED_MORE if cursor == len(text) and not final else None
    if digits[0] == "0" or len(digits) > 19:
        return None
    if cursor == len(text):
        return None if final else _NEED_MORE
    if text[cursor] != "]":
        return None

    match = _MODEL_MSG_ID_ALIAS_RE.match(text)
    if match is None:
        return None
    if int(match.group("message_id")) > _MAX_SIGNED_BIGINT:
        return None

    end = match.end()
    if end == len(text) and not final:
        # The observed alias was glued directly to response prose. Wait for
        # that disambiguating character rather than stripping a standalone
        # authored example.
        return _NEED_MORE
    if end == len(text) or text[end].isspace():
        return None
    return end


def _consume_metadata_prefix(text: str, *, final: bool) -> int | None:
    """Classify and consume one leading metadata envelope.

    Returns the exclusive prefix end, ``_NEED_MORE`` while a streaming buffer
    could still become a valid prefix, or ``None`` when the leading text is
    ordinary assistant content. Leading whitespace is consumed only when a
    complete metadata envelope follows it.
    """
    stripped = text.lstrip()
    leading_whitespace = len(text) - len(stripped)
    if not stripped:
        return None if final else _NEED_MORE

    for head in (_MSG_META_HEAD, _LEGACY_METADATA_HEAD):
        if head.startswith(stripped):
            return None if final else _NEED_MORE

    if stripped.startswith(_MSG_META_HEAD):
        cursor = leading_whitespace + len(_MSG_META_HEAD)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            return None if final else _NEED_MORE
        if text[cursor] != "{":
            return None

        object_end = _json_object_end(text, cursor)
        if object_end is None:
            return None if final else _NEED_MORE
        try:
            payload = json.loads(text[cursor:object_end])
        except (json.JSONDecodeError, TypeError):
            return None
        if not is_msg_meta_payload(payload):
            return None
        if object_end >= len(text):
            return None if final else _NEED_MORE
        if text[object_end] != "]":
            return None

        end = object_end + 1
        if end == len(text) and not final:
            # A delimiter newline may arrive in the next provider delta.
            return _NEED_MORE
        if end < len(text) and text[end] == "\n":
            end += 1
        return end

    # Use the original text: this exact alias must not tolerate leading whitespace.
    alias_end = _consume_model_msg_id_alias(text, final=final)
    if alias_end is not None:
        return alias_end

    if stripped.startswith(_LEGACY_METADATA_HEAD):
        cursor = leading_whitespace + len(_LEGACY_METADATA_HEAD)
        if cursor >= len(text):
            return None if final else _NEED_MORE
        if text[cursor] != "{":
            return None

        object_end = _json_object_end(text, cursor)
        if object_end is None:
            return None if final else _NEED_MORE
        try:
            payload = json.loads(text[cursor:object_end])
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None

        end = object_end
        while end < len(text) and text[end] == "\n":
            end += 1
        if end == len(text) and not final:
            # Legacy envelopes admit a run of delimiter newlines, which may
            # itself be split across provider deltas.
            return _NEED_MORE
        return end

    return None


def _strip_leading_metadata(text: str, *, final: bool) -> tuple[str, bool, bool]:
    """Return ``(remainder, needs_more, was_sanitized)``.

    Modern and legacy envelopes may be stacked. This helper owns their grammar
    for both the batch API and the streaming adapter.
    """
    remainder = text
    was_sanitized = False
    while True:
        prefix_end = _consume_metadata_prefix(remainder, final=final)
        if prefix_end == _NEED_MORE:
            return remainder, True, was_sanitized
        if prefix_end is None:
            return remainder, False, was_sanitized
        remainder = remainder[prefix_end:]
        was_sanitized = True


class MetadataPrefixStreamSanitizer:
    """Statefully remove leading metadata envelopes from streamed text.

    Provider deltas may split ``[msg_meta: {...}]`` at any byte. The sanitizer
    buffers only while the leading text could still be an envelope; once normal
    assistant content is established, later deltas pass through unchanged.
    Instantiate one sanitizer per model generation.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._resolved = False
        self._finished = False
        self.was_sanitized = False

    def feed(self, text: str) -> str:
        """Consume one raw delta and return text safe to expose downstream."""
        if self._finished:
            raise RuntimeError("cannot feed a finished metadata stream sanitizer")
        if not text:
            return ""
        if self._resolved:
            return text

        self._buffer += text
        remainder, needs_more, modified = _strip_leading_metadata(self._buffer, final=False)
        self.was_sanitized = self.was_sanitized or modified
        if needs_more:
            if len(remainder) <= _MAX_METADATA_PREFIX_CHARS:
                self._buffer = remainder
                return ""
            # Availability guard: fail open on an oversized unresolved
            # candidate so streaming cannot stall or repeatedly rescan an
            # ever-growing buffer. This is intentionally not a claim that
            # every possible model-authored envelope fits below the cap.
            self._buffer = ""
            self._resolved = True
            return remainder

        self._buffer = ""
        self._resolved = True
        return remainder

    def finish(self) -> str:
        """Finish the stream and release any buffered non-envelope text."""
        if self._finished:
            return ""
        self._finished = True
        if self._resolved:
            return ""

        remainder, _needs_more, modified = _strip_leading_metadata(self._buffer, final=True)
        self.was_sanitized = self.was_sanitized or modified
        self._buffer = ""
        self._resolved = True
        return remainder


def strip_echoed_metadata(text: str | None) -> str:
    """Remove leading metadata prefixes from text.

    Strips any run of ``[msg_meta: {...}]\\n`` and legacy ``Conversation
    info...`` prefixes from the front of ``text``. Idempotent on clean input.
    """
    if not text:
        return text or ""
    remainder, _needs_more, _modified = _strip_leading_metadata(text, final=True)
    return remainder


def strip_metadata_from_content_blocks(
    content_blocks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Strip metadata from text blocks while preserving non-text blocks.

    Text blocks made empty by sanitization are removed because provider APIs
    reject empty text content. Returns the copied blocks and whether any block
    changed.
    """
    modified = False
    sanitized_blocks: list[dict[str, Any]] = []
    for block in content_blocks:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            original = block["text"]
            sanitized = strip_echoed_metadata(original)
            if sanitized != original:
                modified = True
                if not sanitized.strip():
                    continue
                sanitized_blocks.append({**block, "text": sanitized})
                continue
        sanitized_blocks.append(block)
    return sanitized_blocks, modified


# Matches a leading NO_REPLY / HEARTBEAT_OK token with any leading whitespace.
# The match is unconditional -- whatever follows the token (space, punctuation,
# word char, Unicode letter, nothing) still counts as "the message starts with
# the silent-reply token." This is what makes is_silent_reply() fail-closed:
# the LLM was instructed that NO_REPLY must be the ENTIRE message, so any
# message that begins with the token is treated as an intent to stay silent.
_LEADING_SILENT_TOKEN_RE = re.compile(rf"^\s*(?:{SILENT_REPLY_TOKEN}|{HEARTBEAT_OK_TOKEN})")


def is_silent_reply(text: str | None) -> bool:
    """Check if response should be suppressed (NO_REPLY or HEARTBEAT_OK).

    Fail-closed on leading tokens: if the message starts with NO_REPLY or
    HEARTBEAT_OK in any form -- exact, with punctuation, followed by a space,
    or concatenated with Unicode word chars (e.g. the token glued directly
    onto Cyrillic or CJK content) -- the whole message is treated as silent.
    The prompt instructs the LLM that
    the token must be its ENTIRE message; if the LLM violates that by gluing
    content onto the front, we still respect its original "stay silent"
    signal rather than second-guessing which half to deliver. Callers that
    want to observe the concatenation anomaly can use
    ``strip_leading_silent_token`` to extract the bled-in suffix for logging.

    Single-purpose: does NOT strip leaked ``[msg_meta: ...]`` prefixes. If a
    caller hands in raw assistant text that might contain an echoed prefix,
    it must call ``strip_echoed_metadata`` first. The delivery pipeline in
    ``app/telegram/handlers/chat.py`` composes them in that order so the
    coupling is explicit at the call site instead of hidden here.
    """
    if not text:
        return False
    stripped = text.strip()
    # Leading token -- fail-closed, matches regardless of what follows.
    if _LEADING_SILENT_TOKEN_RE.match(stripped):
        return True
    # Trailing token: "... NO_REPLY"
    for token in _SILENT_TOKENS:
        if re.search(rf"\b{re.escape(token)}\b\W*$", stripped):
            return True
    return False


def strip_leading_silent_token(text: str) -> str:
    """Strip a leading NO_REPLY/HEARTBEAT_OK token from response text.

    Used for observability, not delivery: when ``is_silent_reply()`` fires on
    a message that had real content after the token (the LLM "concatenation"
    failure mode), callers can use this helper to extract the bled-in suffix
    and log it. The delivery path itself stays fail-closed -- see
    ``is_silent_reply`` for the rationale.

    Idempotent: text without a leading silent token is returned unchanged.
    """
    if not text:
        return text
    return _LEADING_SILENT_TOKEN_RE.sub("", text, count=1)
