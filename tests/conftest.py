from datetime import UTC, datetime
from types import SimpleNamespace
import pytest
from kai_memory import MemorySettings, NoopTelemetry, ports
from memory_test_host import synthetic_bindings
from memory_test_support import ChatContext
import memory_test_parsing
import memory_test_tokens
import memory_test_sanitizer
import memory_test_skill

@pytest.fixture(autouse=True)
def synthetic_host():
    for name in ports._HOST_SYMBOLS:
        ports.__dict__.pop(name, None)
    ports._BOUND.clear()
    # Tests may deliberately clear process bindings. A fresh explicit resolver
    # keeps their defaults synthetic without importing application configuration.
    names = synthetic_bindings(datetime.now(UTC))
    names["ChatContext"] = ChatContext
    names["parse_json_object_response"] = memory_test_parsing.parse_json_object_response
    names["parse_json_array_response"] = memory_test_parsing.parse_json_array_response
    names["strip_msg_meta_envelope"] = memory_test_tokens.strip_msg_meta_envelope
    names["strip_echoed_metadata"] = memory_test_tokens.strip_echoed_metadata
    names["scrub_secrets"] = memory_test_sanitizer.scrub_secrets
    names["validate_skill_content"] = memory_test_skill.validate_skill_content
    names["is_admin_user"] = lambda actor, ids=None: actor is not None and actor in {
        int(value.strip()) for value in (ids or "").split(",") if value.strip().isdigit()
    }
    previous = ports.set_default_resolver(lambda name: names.get(name, ports.UNBOUND))
    try:
        yield
    finally:
        ports.set_default_resolver(previous)
        ports._BOUND.clear()
        for name in ports._HOST_SYMBOLS:
            ports.__dict__.pop(name, None)
