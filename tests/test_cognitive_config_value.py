"""Unit tests for the CognitiveConfig.value type-coercion validator.

Regression for the asyncpg ``DataError: '1.0' must be real number, not str`` +
perpetual-dirty-session crash: an operator flip that assigned the STRING ``'1.0'``
to the ``Float`` ``value`` column poisoned every session that loaded the row.
The model-level ``@validates('value')`` coerces on assignment for ALL callers.
"""

from __future__ import annotations

import pytest

from kai_memory.storage.models.cognitive_config import CognitiveConfig


def test_string_value_coerced_to_float_on_construct():
    c = CognitiveConfig(key="procedural_rule_extraction_enabled", value="1.0", category="memory")
    assert c.value == 1.0
    assert isinstance(c.value, float)


def test_string_value_coerced_on_setattr():
    c = CognitiveConfig(key="k", value=0.0, category="memory")
    c.value = "0.5"  # a normalize/flip write assigning a string
    assert c.value == 0.5 and isinstance(c.value, float)


def test_int_value_coerced():
    c = CognitiveConfig(key="k", value=1, category="memory")
    assert c.value == 1.0 and isinstance(c.value, float)


def test_non_numeric_string_rejected():
    with pytest.raises(ValueError, match="numeric"):
        CognitiveConfig(key="k", value="on", category="memory")


def test_none_rejected():
    with pytest.raises(ValueError, match="None"):
        CognitiveConfig(key="k", value=None, category="memory")
