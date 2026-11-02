"""Shared parsing helpers for LLM JSON responses."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")
_MD_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*\n?", re.IGNORECASE | re.MULTILINE)
_FENCE_CLOSE_RE = re.compile(r"\n?```\s*$")


def _strip_fences(text: str) -> str:
    """Strip markdown fence markers even when the closing fence is missing.

    Haiku sometimes returns ```json\\n{...} without a closing ```, which
    makes _MD_FENCE_RE (non-greedy, needs both delimiters) fail. This
    helper aggressively strips the opening ```json and optional closing
    ``` so downstream json.loads sees raw JSON.
    """
    stripped = _FENCE_OPEN_RE.sub("", text)
    stripped = _FENCE_CLOSE_RE.sub("", stripped)
    return stripped.strip()


def parse_json_object_response(raw: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response, tolerating markdown fences."""
    if not raw:
        return {}
    text = raw.strip()

    # 1. Direct parse (no fences)
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        pass

    # 2. Balanced markdown fence (```json ... ```)
    fence_match = _MD_FENCE_RE.search(text)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1).strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # 3. Aggressive fence strip (handles unclosed fences — common when
    #    Haiku response is truncated by token limit)
    stripped = _strip_fences(text)
    if stripped != text:
        try:
            result = json.loads(stripped)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # 4. Bare {...} regex (last resort)
    obj_match = _JSON_OBJECT_RE.search(text)
    if obj_match:
        try:
            result = json.loads(obj_match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse LLM JSON object response: %.200s", text)
    return {}


def parse_json_array_response(raw: str) -> list[Any]:
    """Parse a JSON array from an LLM response, tolerating markdown fences."""
    if not raw:
        return []
    text = raw.strip()

    # 1. Direct parse
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        pass

    # 2. Balanced markdown fence
    fence_match = _MD_FENCE_RE.search(text)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1).strip())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # 3. Aggressive fence strip (unclosed fences)
    stripped = _strip_fences(text)
    if stripped != text:
        try:
            result = json.loads(stripped)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # 4. Bare [...] regex
    array_match = _JSON_ARRAY_RE.search(text)
    if array_match:
        try:
            result = json.loads(array_match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse LLM JSON array response: %.200s", text)
    return []
