"""The deterministic SDK proof owns its adapters and does not import evaluation."""

import ast
from datetime import UTC, datetime
import importlib.util
from pathlib import Path

import pytest

from kai_memory.storage.disposable import UnsafeDatabaseError, check_url_is_local
from memory_test_host import ConversationMessage, FixedEmbedder, synthetic_bindings


def test_proof_sources_do_not_import_evaluation():
    tests = Path(__file__).parent
    example = tests.parent / "examples" / "storage_upgrade.py"
    support = Path(importlib.util.find_spec("memory_test_host").origin)
    guard = Path(importlib.util.find_spec("kai_memory.storage.disposable").origin)
    for path in [example, support, guard, *sorted(tests.rglob("*.py"))]:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            assert all(
                module != "kai_memory.eval"
                and not module.startswith("kai_memory.eval.")
                and module != "app"
                and not module.startswith("app.")
                for module in modules
            ), path


async def test_unscripted_defaults_fail_without_external_calls():
    bindings = synthetic_bindings(datetime.now(UTC))
    for call in (
        bindings["call_llm_text"]("prompt"),
        bindings["embedder"].embed(["text"]),
        bindings["ConversationManager"](None).get_messages_since(1, None, datetime.now(UTC)),
        bindings["read_conversation_page"](None, None, 10),
    ):
        with pytest.raises(RuntimeError, match="no synthetic response configured"):
            await call


async def test_storage_embedder_has_independent_nonzero_vectors():
    vectors = await FixedEmbedder().embed(["first", "second"])
    assert len(vectors) == 2 and all(len(vector) == 1536 and sum(vector) == 1 for vector in vectors)
    vectors[0][0] = 0
    assert vectors[1][0] == 1


def test_conversation_value_preserves_gate_input():
    message = ConversationMessage(
        role="user",
        content_blocks=[{"text": "use "}, {"text": "examples"}],
        sender_user_id=10,
        message_metadata={"quoted": False},
    )
    assert message.text == "use examples"
    assert message.sender_user_id == 10 and message.message_metadata == {"quoted": False}


def test_conversation_omitted_role_preserves_nested_tool_containment():
    message = ConversationMessage(
        content_blocks=[{"toolResult": {"content": [{"text": "nested recalled fact"}]}}],
    )
    assert message.role is None
    assert message.text == ""
    assert ConversationMessage().text == ""


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
def test_loopback_guard_protects_configured_database_under_aliases(host):
    url = f"postgresql+asyncpg://synthetic:synthetic@{host}/memory_sdk"
    check_url_is_local(url, production_url=None)
    with pytest.raises(UnsafeDatabaseError, match="configured application DATABASE_URL"):
        check_url_is_local(url, production_url="postgresql://other:other@localhost:5432/memory_sdk")


@pytest.mark.parametrize("target", ["host", "hostaddr", "port", "service", "database", "dbname", "HostAddr"])
def test_loopback_guard_refuses_target_overrides(target):
    with pytest.raises(UnsafeDatabaseError, match="connection-target parameters"):
        check_url_is_local(f"postgresql+asyncpg://localhost/synthetic?{target}=elsewhere", production_url=None)


@pytest.mark.parametrize(
    "url",
    [
        "not a URL",
        "postgresql+asyncpg:///synthetic",
        "postgresql+asyncpg://db.invalid/synthetic",
        "postgresql+asyncpg://sample.rds.amazonaws.com/synthetic",
    ],
)
def test_loopback_guard_refuses_nonlocal_targets(url):
    with pytest.raises(UnsafeDatabaseError):
        check_url_is_local(url, production_url=None)
