"""The memory SDK's public API: ``MemoryClient`` and the adapters a host supplies (DB-854).

A host builds a ``MemoryClient`` from a ``HostAdapters`` and runs the memory
lifecycle through it. Each client's adapters are in effect only inside its own
operations (``ports.scoped``) and in the work those operations leave behind, so
several clients over different stores can run in one process, and a client may be
created after memory is imported:

    client = MemoryClient(HostAdapters(inference=..., embedder=..., history=...,
                                       session_factory=..., settings=MemorySettings(),
                                       overrides={"scrub_secrets": scrub_secrets,
                                                  "validate_skill_content": validate_skill_content,
                                                  "redact_urls_for_trace": redact_urls_for_trace}))
    await client.record_message(chat, "user", "I started at Example Co last week", sender_id=7)
    await client.extract(chat)                    # L2: write what the conversation says
    recognition = await client.recall("Where does Pat work?", chat)
    recognition.commit()                          # or .discard(); access is written on commit
    await client.consolidate()                    # L3
    await client.close()                          # finishes, then drains or cancels its work

This module, re-exported from ``kai_memory``, is the supported surface;
everything else in the package is internal. ``API_VERSION`` changes on any
incompatible change to it.

Fresh disposable storage is initialized through ``storage.bootstrap_disposable``;
production hosts own migrations. ``begin_turn`` owns staged delivery and read-only
mid-turn recall; ``procedural`` owns canonical execution and teaching evidence.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast, get_args

from kai_memory import ports
from kai_memory.ownership import owned_by
from kai_memory.identity import AgentIdentity
from kai_memory.configuration import HostPolicy, ModelRoles, NeutralHostPolicy

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from kai_memory.recognition import StagedRecognition
    from kai_memory.procedural_api import ProceduralMemory
    from kai_memory.retrieval.memory_observation import MemoryObservation
    from kai_memory.turn import TurnMemory
    from kai_memory.types import Artifact, ArtifactQuery

__version__ = "0.3.0"
API_VERSION = "2"

__all__ = [
    "API_VERSION",
    "Chat",
    "Embedder",
    "History",
    "HostAdapters",
    "Inference",
    "MemoryClient",
    "MemorySettings",
    "Message",
    "NoopTelemetry",
    "Telemetry",
    "__version__",
]


# ── Data the host and memory exchange ───────────────────────────────────────


@dataclass
class Chat:
    """One conversational scope: whose messages memory reads and writes about."""

    chat_id: int | None = None
    topic_id: int | None = None
    venue: str = "telegram"
    is_group: bool = False
    chat_title: str | None = None
    topic_title: str | None = None
    chat_type: str | None = None
    sender_user_id: int | None = None
    sender_name: str | None = None
    sender_username: str | None = None
    extra_participants: list[dict[str, Any]] | None = None


@dataclass
class Message:
    """One stored conversation message, as the history adapter keeps and returns it."""

    chat_id: int | None
    topic_id: int | None
    venue: str
    role: str
    text: str
    created_at: datetime
    sender_user_id: int | None = None
    sender_name: str | None = None
    message_metadata: dict[str, Any] = field(default_factory=dict)
    id: uuid.UUID = field(default_factory=uuid.uuid4)


@dataclass
class MemorySettings:
    """The configuration memory reads. Field names follow the application's settings."""

    telegram_admin_user_ids: str = ""  # comma-separated ids of first-party (trusted) users
    procedural_respect_roster: str = ""
    consolidation_rewrite_model: str = "opus-4.8"
    database_url: str = ""
    aws_region: str = ""
    cost_archive_s3_bucket: str = ""
    cost_archive_s3_prefix: str = ""


# ── Adapters a host supplies ────────────────────────────────────────────────


class Inference(Protocol):
    """Text completion. ``model`` is memory's model name (e.g. ``haiku-5.5``).

    API v2 hosts must accept ``thinking_mode`` and honor ``disabled`` for
    structured calls; Haiku 5.5 otherwise enables adaptive thinking by default.
    """

    async def complete(
        self,
        prompt: str,
        *,
        model: str | None,
        max_tokens: int,
        thinking_mode: Literal["adaptive", "disabled"] = "adaptive",
    ) -> str | None: ...


class Embedder(Protocol):
    """One vector per text, in input order."""

    async def embed(self, texts: list[str], *, api_key: str | None = None) -> list[list[float]]: ...


class History(Protocol):
    """The host's conversation log. Memory reads it and never edits it."""

    async def record(self, message: Message) -> None: ...

    async def read_page(self, chat: Chat, after: Any | None, limit: int) -> list[Message]:
        """Messages of ``chat`` after the cursor ``after`` (``created_at``/``id``), oldest first."""
        ...

    async def senders_seen_before(self, senders: set[int], before: datetime) -> set[int]: ...

    async def delete(self, chat_id: int, topic_id: int | None) -> None: ...


class Telemetry(Protocol):
    def span(self, name: str, model: str) -> Any: ...

    def event(self, name: str, **fields: Any) -> None: ...

    def counter(self, name: str, value: float = 1, **tags: Any) -> None: ...


class NoopTelemetry:
    """The SDK default: no spans, events or counters."""

    def span(self, name: str, model: str) -> Any:
        return None

    def event(self, name: str, **fields: Any) -> None:
        return None

    def counter(self, name: str, value: float = 1, **tags: Any) -> None:
        return None


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class HostAdapters:
    """Everything memory needs from its host.

    ``overrides`` binds any other port by its contract name (``ports._HOST_SYMBOLS``),
    including the required callable hygiene policy: ``scrub_secrets``,
    ``validate_skill_content``, and ``redact_urls_for_trace``. The SDK never
    supplies a pass-through policy implicitly. Other defaults are listed in
    ``MemoryClient``.
    """

    inference: Inference
    embedder: Embedder
    history: History
    session_factory: Callable[[], AsyncSession]
    settings: Any = field(default_factory=MemorySettings)
    clock: Callable[[], datetime] = _utc_now
    telemetry: Telemetry = field(default_factory=NoopTelemetry)
    overrides: Mapping[str, Any] = field(default_factory=dict)
    identity: AgentIdentity = field(default_factory=lambda: AgentIdentity("agent", "Agent"))
    models: ModelRoles = field(default_factory=ModelRoles)
    policy: HostPolicy = field(default_factory=NeutralHostPolicy)


# ── SDK defaults for the ports a host rarely needs to replace ───────────────


@dataclass(frozen=True)
class _ModelHandle:
    """What memory passes around as a "provider": just the model it asked for."""

    model: str | None


class CancelledByNewMessage(Exception):
    """Raised by a host to abandon a turn a newer message superseded."""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _parse_object(raw: str) -> dict[str, Any]:
    text = _strip_fences(raw or "")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        try:
            value = json.loads(text[start : end + 1]) if 0 <= start < end else {}
        except json.JSONDecodeError:
            value = {}
    return value if isinstance(value, dict) else {}


def _parse_array(raw: str) -> list[Any]:
    text = _strip_fences(raw or "")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        try:
            value = json.loads(text[start : end + 1]) if 0 <= start < end else []
        except json.JSONDecodeError:
            value = []
    return value if isinstance(value, list) else []


def _is_admin(user_id: int | None, admin_ids: str | None) -> bool:
    if user_id is None or not admin_ids:
        return False
    return str(user_id) in {part.strip() for part in admin_ids.split(",") if part.strip()}


def _unsupported(name: str) -> Callable[..., Any]:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"memory SDK: {name} is not available to SDK hosts (pass it in overrides)")

    return fail


@dataclass(frozen=True)
class _Since:
    """A ``read_page`` cursor that means "strictly after this time"."""

    created_at: datetime
    id: Any = None


class _HistoryConversations:
    """The conversation reader memory constructs per session, backed by the history adapter."""

    def __init__(self, history: History) -> None:
        self._history = history

    def __call__(self, session: Any) -> _HistoryConversations:
        return self

    async def get_messages_since(
        self,
        chat_id: int | None,
        topic_id: int | None,
        since: datetime,
        limit: int = 200,
        venue: str = "telegram",
    ) -> list[Message]:
        """Messages after ``since``, oldest first, at most ``limit`` (as Kai's reader)."""
        chat = Chat(chat_id=chat_id, topic_id=topic_id, venue=venue)
        return await self._history.read_page(chat, _Since(since), limit)


def _as_context(chat: Chat) -> Any:
    """``Chat`` is what the client binds as memory's ``ChatContext`` port."""
    return chat


# ── The client ──────────────────────────────────────────────────────────────

logger = logging.getLogger("app.services.memory.sdk")

_OPEN, _CLOSING, _CLOSED = "open", "closing", "closed"
T = TypeVar("T")


# Asyncio keeps only weak references to tasks. Root the entire closing owner until
# its actual workers and operations settle, even if the host drops its last reference.
_CLOSING_CLIENTS: set[MemoryClient] = set()


class _BestEffortTelemetry:
    """Keep host observability failures from replacing memory results."""

    def __init__(self, telemetry: Telemetry) -> None:
        self._telemetry = telemetry

    def span(self, name: str, model: str) -> Any:
        try:
            return self._telemetry.span(name, model)
        except Exception as exc:
            logger.debug("Memory SDK telemetry span failed type=%s", type(exc).__name__)
            return None

    def event(self, name: str, **fields: Any) -> None:
        try:
            self._telemetry.event(name, **fields)
        except Exception as exc:
            logger.debug("Memory SDK telemetry event failed type=%s", type(exc).__name__)

    def counter(self, name: str, value: float = 1, **tags: Any) -> None:
        try:
            self._telemetry.counter(name, value, **tags)
        except Exception as exc:
            logger.debug("Memory SDK telemetry counter failed type=%s", type(exc).__name__)


class MemoryClient:
    """The memory lifecycle over one host's adapters.

    Each operation runs with this client's adapters bound (``ports.scoped``) and the
    client as the owner of its work (``ownership``); tasks it starts inherit both.
    Clients are independent: several may be open at once, each on its own store.

    Lifecycle: ``open`` until ``close()``, then ``closing`` (new operations and
    commits are refused, in-flight ones finish, the client's own tasks drain), then
    ``closed``. Work a client started never runs under another client's adapters.

    Ports with no ``HostAdapters`` field get SDK defaults: JSON parsing that tolerates
    code fences; pass-through envelope strippers; neutral host policy with no
    teaching authority; no prompt cache; no generation provenance.
    Replace any of them through ``HostAdapters.overrides``. Hosts must supply
    callable ``scrub_secrets``, ``validate_skill_content``, and
    ``redact_urls_for_trace`` policies there before constructing a client.
    """

    def __init__(self, adapters: HostAdapters) -> None:
        bindings = self._port_bindings(adapters)
        missing = sorted(set(ports._HOST_SYMBOLS) - set(bindings))
        if missing:  # a scoped client never falls through to the process default
            raise RuntimeError(f"MemoryClient binds no adapter for ports {missing}")
        self.adapters = adapters
        self._bindings: Mapping[str, Any] = bindings
        self._extract_locks: dict[tuple[Any, ...], asyncio.Lock] = {}
        self.caches: dict[str, Any] = {}  # memory's between-call state for this client
        self._state = _OPEN
        self._in_flight = 0
        self._operations: dict[asyncio.Task[Any], int] = {}
        self._idle = asyncio.Event()
        self._idle.set()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._cleanup_tasks: set[asyncio.Task[Any]] = set()
        self._staged: list[StagedRecognition] = []
        self._closer: asyncio.Future[bool] | None = None
        self._close_finished = False
        self._turns: list[TurnMemory] = []
        self._procedural: ProceduralMemory | None = None

    def _require_read(self, chat: Chat) -> None:
        if not self._bindings["get_host_policy"]().may_read(chat):
            raise PermissionError("memory host policy denied reading this scope")

    def _require_write(self, chat: Chat) -> None:
        if not self._bindings["get_host_policy"]().may_write(chat):
            raise PermissionError("memory host policy denied writing this scope")

    @staticmethod
    def _port_bindings(adapters: HostAdapters) -> dict[str, Any]:
        ports._check_names(adapters.overrides)
        required_policy = ("scrub_secrets", "validate_skill_content", "redact_urls_for_trace")
        missing_policy = [name for name in required_policy if not callable(adapters.overrides.get(name))]
        if missing_policy:
            raise ValueError(f"MemoryClient requires callable host hygiene policies in overrides: {missing_policy}")
        inference, history, settings = adapters.inference, adapters.history, adapters.settings

        async def call_llm_text(
            provider: Any,
            prompt: str,
            max_tokens: int = 2048,
            span: Any = None,
            *,
            thinking_mode: Literal["adaptive", "disabled"] = "adaptive",
            **_: Any,
        ) -> Any:
            result: str | None = None
            succeeded = False
            failure = "error"
            try:
                result = await inference.complete(
                    prompt, model=getattr(provider, "model", None), max_tokens=max_tokens, thinking_mode=thinking_mode
                )
                succeeded = True
                return result
            except asyncio.CancelledError:
                failure = "cancelled"
                raise
            finally:
                # Like Kai's helper, this call owns the generation span. The neutral
                # inference adapter supplies no token counts, so none are fabricated.
                # Observability must never replace a result or the original failure.
                if span:
                    try:
                        output = (
                            (ports.redact_urls_for_trace(result[:500]) if result else None) if succeeded else failure
                        )
                        span.update(output=output)
                    except Exception as exc:
                        logger.debug("Memory SDK span update failed type=%s", type(exc).__name__)
                    finally:
                        try:
                            span.end()
                        except Exception as exc:
                            logger.debug("Memory SDK span end failed type=%s", type(exc).__name__)

        @asynccontextmanager
        async def session_scope() -> AsyncIterator[Any]:
            session = adapters.session_factory()
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

        async def recover_session(session: Any) -> None:
            try:
                await session.rollback()
            except Exception:
                await session.invalidate()

        async def read_conversation_page(session: Any, chat: Any, after: Any, limit: int) -> list[Message]:
            return await history.read_page(chat, after, limit)

        async def senders_seen_before(session: Any, senders: set[int], before: datetime) -> set[int]:
            return await history.senders_seen_before(senders, before)

        async def delete_conversation_messages(session: Any, *, chat_id: int, topic_id: int | None) -> None:
            await history.delete(chat_id, topic_id)

        bindings: dict[str, Any] = {
            # storage, configuration, time, observability
            "session_scope": session_scope,
            "recover_session": recover_session,
            "SessionLocal": adapters.session_factory,
            "init_engine": lambda: None,
            "get_settings": lambda: settings,
            "get_agent_identity": lambda: adapters.identity,
            "get_model_roles": lambda: adapters.models,
            "get_host_policy": lambda: adapters.policy,
            "clock": adapters.clock,
            "telemetry": adapters.telemetry,
            # inference and parsing
            "call_llm_text": call_llm_text,
            "get_provider_for_model": lambda _settings, model: _ModelHandle(model),
            "get_provider": _unsupported("get_provider"),
            "record_generation_provenance": lambda *a, **k: None,
            "parse_json_object_response": _parse_object,
            "parse_json_array_response": _parse_array,
            # embeddings and history
            "embedder": adapters.embedder,
            "read_conversation_page": read_conversation_page,
            "senders_seen_before": senders_seen_before,
            "delete_conversation_messages": delete_conversation_messages,
            "ConversationManager": _HistoryConversations(history),
            "ChatContext": Chat,
            # text hygiene, trust, cancellation, prompt cache
            "strip_msg_meta_envelope": lambda text: text,
            "strip_echoed_metadata": lambda text: text or "",
            "is_admin_user": lambda actor_id, _admin_ids=None: adapters.policy.is_admin(actor_id),
            "CancelledByNewMessage": CancelledByNewMessage,
            "MODEL_METADATA_KEY": "model",
            "invalidate_prompt_cache": lambda *a, **k: None,
        }
        bindings.update(adapters.overrides)
        bindings["telemetry"] = _BestEffortTelemetry(bindings["telemetry"])
        return bindings

    # ── ownership (``kai_memory.ownership.Owner``) ──

    @property
    def state(self) -> str:
        """``open``, ``closing`` or ``closed``."""
        return self._state

    def accepts_work(self) -> bool:
        return self._state == _OPEN

    @contextmanager
    def _bound(self) -> Iterator[None]:
        with ports.scoped(self._bindings), owned_by(self):
            yield

    def run_owned(self, fn: Callable[[], T]) -> T:
        def bound() -> T:
            with self._bound():
                return fn()

        return contextvars.copy_context().run(bound)

    def _settle_close(self) -> None:
        """Release the retained owner only after every operation and cleanup really finishes."""
        if self._close_finished and self._in_flight == 0 and not self._running():
            self._state = _CLOSED
            _CLOSING_CLIENTS.discard(self)

    def _task_finished(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        self._cleanup_tasks.discard(task)
        if not task.cancelled():
            task.exception()  # consume discarded failures, including cleanup after the deadline
        self._settle_close()

    def adopt(self, task: asyncio.Task[Any]) -> None:
        """Own ``task`` until it finishes. Once closing, a new task is cancelled at once."""
        if task in self._tasks:
            return
        self._tasks.add(task)
        task.add_done_callback(self._task_finished)
        if self._state != _OPEN:
            logger.info("MemoryClient %s: cancelling a task started after close()", self._state)
            self._state = _CLOSING
            _CLOSING_CLIENTS.add(self)
            if not task.cancelling():
                task.cancel()

    def track_staged(self, staged: StagedRecognition) -> None:
        if staged.chat_ctx is not None:
            policy = self._bindings["get_host_policy"]()
            staged.commit_guard = lambda: policy.may_read(staged.chat_ctx) and policy.may_write(staged.chat_ctx)
        if self._state != _OPEN:  # a late result must not survive the initial close discard sweep
            staged.discard()
            return
        self._staged = [s for s in self._staged if s.outcome is None]
        self._staged.append(staged)

    def adopt_cleanup(self, task: asyncio.Task[Any]) -> None:
        """Retain cleanup through close deadlines without cancelling it a second time."""
        if task in self._cleanup_tasks:
            return
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._task_finished)
        if self._state != _OPEN:
            self._state = _CLOSING
            _CLOSING_CLIENTS.add(self)

    @asynccontextmanager
    async def _operation(self) -> AsyncIterator[None]:
        """One client operation: refused unless open, counted while in flight, bound."""
        if self._state != _OPEN:
            raise RuntimeError(f"this MemoryClient is {self._state}")
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("MemoryClient operations require an asyncio task")
        # A count alone cannot keep a caller's suspended Task alive. Keep the
        # actual task until its outermost operation exits, including async cleanup.
        self._operations[task] = self._operations.get(task, 0) + 1
        self._in_flight += 1
        self._idle.clear()
        try:
            with self._bound():
                yield
        finally:
            self._operations[task] -= 1
            if self._operations[task] == 0:
                del self._operations[task]
            self._in_flight -= 1
            if self._in_flight == 0:
                self._idle.set()
            self._settle_close()

    async def _load_config_once(self) -> None:
        """Read this client's store's cognitive_config before its first extract or recall."""
        from kai_memory.host_calls import session_scope
        from kai_memory.store import cognitive_config

        if not cognitive_config.is_loaded():
            async with session_scope() as session:
                await cognitive_config.load_config(session)

    # ── lifecycle ──

    async def record_message(
        self,
        chat: Chat,
        role: str,
        text: str,
        *,
        sender_id: int | None = None,
        sender_name: str | None = None,
        at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Store one message through the history adapter. ``at`` must be timezone-aware."""
        from kai_memory.host_calls import utc_now

        if at is not None and at.tzinfo is None:
            raise ValueError("record_message: `at` must be a timezone-aware datetime")
        async with self._operation():
            self._require_write(chat)
            message = Message(
                chat_id=chat.chat_id,
                topic_id=chat.topic_id,
                venue=chat.venue,
                role=role,
                text=text,
                created_at=at or utc_now(),
                sender_user_id=sender_id,
                sender_name=sender_name,
                message_metadata=dict(metadata or {}),
            )
            await self.adapters.history.record(message)
            return message

    async def extract(self, chat: Chat) -> dict[str, Any]:
        """Run one L2 extraction over the chat's messages after its cursor, and commit it.

        The cursor lives in the store, committed with each run's writes (as on Kai's
        path), so a new client or a restarted host resumes where the last run stopped.
        Calls for one chat on this client run one at a time. Traces that recognitions
        committed for the chat are its contradiction candidates.

        Returns the run's stats with ``completed`` True. ``completed`` is False when
        another run was in progress, capture contention deferred this run, or it
        failed (recorded on its consolidation run and logged). Deferred work keeps
        its cursor and retry budget for a later call; ``backlog_remaining`` asks
        for another call after a completed run.
        """
        from kai_memory.consolidation.consolidation_trigger import extract_now

        key = (chat.venue, chat.chat_id, chat.topic_id)
        async with self._operation():
            self._require_read(chat)
            self._require_write(chat)
            await self._load_config_once()
            async with self._extract_locks.setdefault(key, asyncio.Lock()):
                stats = await extract_now(_as_context(chat))
        result = {**stats, "completed": True} if stats is not None else {"completed": False}
        counts = {k: v for k, v in result.items() if isinstance(v, int)}
        self._bindings["telemetry"].event("memory.extract", chat_id=chat.chat_id, **counts)
        return result

    async def recall(self, query: str, chat: Chat, *, defer_access: bool = True) -> StagedRecognition:
        """Recognise what memory holds for ``query``; settle the result with ``commit()``/``discard()``.

        The result belongs to this client: its ``commit()`` writes through this
        client's adapters, even when called outside the client. ``close()`` discards
        it if it is still unsettled; a commit after that returns False and writes
        nothing.
        """
        from kai_memory.host_calls import session_scope
        from kai_memory.recognition import recognize

        async with self._operation():
            self._require_read(chat)
            if not defer_access:
                self._require_write(chat)
            await self._load_config_once()
            async with session_scope() as session:
                staged = await recognize(session, query, _as_context(chat), defer_access=defer_access)
        self._bindings["telemetry"].counter("memory.recall.traces", len(staged.primed.trace_ids))
        return staged

    async def initialize_identity(self) -> uuid.UUID:
        """Bind the host's stable self identity, explicitly adopting eligible legacy rows."""
        from kai_memory.host_calls import session_scope
        from kai_memory.identity import ensure_agent_identity

        async with self._operation():
            async with session_scope() as session:
                entity = await ensure_agent_identity(session)
                return entity.id

    async def begin_turn(
        self,
        chat: Chat,
        *,
        model: str = "unspecified",
        scope: str = "turn",
        observation: MemoryObservation | None = None,
    ) -> TurnMemory:
        """Own initial/steered recognition and read-only mid-turn delivery for one turn."""
        from kai_memory.turn import TurnMemory

        async with self._operation():
            self._require_read(chat)
            await self._load_config_once()

            @asynccontextmanager
            async def operation() -> AsyncIterator[None]:
                async with self._operation():
                    self._require_read(chat)
                    yield

            turn = TurnMemory(
                operation=operation,
                chat_ctx=chat,
                owner=self,
                model=model,
                scope=scope,
                observation=observation,
                read_guard=lambda: self._bindings["get_host_policy"]().may_read(chat),
            )
            self._turns = [existing for existing in self._turns if existing.snapshot().status != "closed"]
            self._turns.append(turn)
            return turn

    @property
    def procedural(self) -> ProceduralMemory:
        """The public execution, capture, serving and outcome operations owned by this client."""
        from kai_memory.procedural_api import ProceduralMemory

        if self._procedural is None:
            self._procedural = ProceduralMemory(self._operation, self._load_config_once)
        return self._procedural

    async def compose_persona(self, chat: Chat, *, budget_chars: int = 4000) -> str:
        """Read and render the existing artifact/persona mechanisms under host disclosure policy."""
        from kai_memory.host_calls import session_scope
        from kai_memory.persona_composer import PersonaComposer

        async with self._operation():
            self._require_read(chat)
            await self._load_config_once()
            async with session_scope() as session:
                return await PersonaComposer(session).compose(chat, budget_chars=budget_chars)

    async def export_artifacts(self, chat: Chat, query: ArtifactQuery | None = None) -> AsyncIterator[Artifact]:
        """Yield store-wide typed artifacts; never run a producer or model call.

        ``chat`` is the host authorization context, not an artifact partition.
        A policy granting this operation must authorize the store's entire artifact
        collection; use independent stores for isolated tenants. ``query`` narrows
        results for consumers and does not establish an authorization boundary.
        """
        from sqlalchemy import select, tuple_

        from kai_memory.host_calls import session_scope
        from kai_memory.storage.models.memory_artifact import MemoryArtifact
        from kai_memory.types import Artifact, ArtifactQuery, ArtifactType

        query = query or ArtifactQuery()
        if query.produced_after is not None and query.produced_after.tzinfo is None:
            raise ValueError("produced_after must be timezone-aware")
        stmt = (
            select(MemoryArtifact)
            .where(MemoryArtifact.artifact_type.in_(get_args(ArtifactType)))
            .order_by(MemoryArtifact.created_at, MemoryArtifact.id)
        )
        if query.artifact_type is not None:
            stmt = stmt.where(MemoryArtifact.artifact_type == query.artifact_type)
        if query.schema_version_min is not None:
            stmt = stmt.where(MemoryArtifact.schema_version >= query.schema_version_min)
        if query.produced_after is not None:
            stmt = stmt.where(MemoryArtifact.created_at > query.produced_after)
        if query.subject_entity_id is not None:
            stmt = stmt.where(MemoryArtifact.subject_entity_id == query.subject_entity_id)
        if query.status != "any":
            stmt = stmt.where(MemoryArtifact.status == query.status)
        page_size = 200
        cursor = None
        while True:
            async with self._operation():
                self._require_read(chat)
                page_stmt = stmt.limit(page_size)
                if cursor is not None:
                    page_stmt = page_stmt.where(tuple_(MemoryArtifact.created_at, MemoryArtifact.id) > cursor)
                async with session_scope() as session:
                    rows = (await session.execute(page_stmt)).scalars().all()
                    if not rows:
                        return
                    cursor = (rows[-1].created_at, rows[-1].id)
                    has_more = len(rows) == page_size
                    artifacts = []
                    for row in rows:
                        payload = dict(row.payload or {})
                        confidence = payload.get("confidence")
                        if query.min_confidence is not None and confidence is not None:
                            if not isinstance(confidence, (int, float)) or confidence < query.min_confidence:
                                continue
                        artifacts.append(
                            Artifact(
                                id=row.id,
                                artifact_type=cast(ArtifactType, row.artifact_type),
                                schema_version=row.schema_version,
                                producer_name=row.producer_name,
                                producer_version=row.producer_version,
                                subject_entity_id=row.subject_entity_id,
                                payload=payload,
                                content_hash=row.content_hash,
                                status=cast(Literal["active", "superseded", "expired", "permanent"], row.status),
                                supersedes_id=row.supersedes_id,
                                created_at=row.created_at,
                                expires_at=row.expires_at,
                            )
                        )
            for artifact in artifacts:
                # No session or in-flight operation remains borrowed across yield.
                async with self._operation():
                    self._require_read(chat)
                yield artifact
            if not has_more:
                return

    async def forget(self, chat: Chat, trace_id: uuid.UUID) -> bool:
        """Invalidate one trace using the existing correction mechanism; retain its provenance."""
        from kai_memory.host_calls import session_scope
        from kai_memory.store.cognitive_memory import CognitiveMemoryService

        async with self._operation():
            self._require_write(chat)
            async with session_scope() as session:
                service = CognitiveMemoryService(session)
                trace = await service.get_trace(trace_id)
                if trace is None or trace.invalidated_at is not None:
                    return False
                await service.invalidate_trace(trace_id)
                return True

    async def consolidate(self) -> dict[str, Any]:
        """Run L3 consolidation once."""
        from kai_memory.consolidation.consolidation import consolidate

        async with self._operation():
            result = await consolidate()
        self._bindings["telemetry"].event("memory.consolidate", success=result.get("success"))
        return result

    def _running(self) -> set[asyncio.Task[Any]]:
        return {task for task in self._tasks | self._cleanup_tasks if not task.done()}

    async def drain(self, timeout: float = 10.0) -> bool:  # noqa: ASYNC109 — reports leftovers, never cancels
        """Wait until this client owns no running task, for at most ``timeout`` seconds in all.

        Tasks adopted while it waits (a task's own child, a retry) are waited for too.
        True when none remain.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while running := self._running():
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            await asyncio.wait(running, timeout=remaining)
        return True

    async def close(self, timeout: float = 10.0) -> bool:  # noqa: ASYNC109 — one deadline, then cancels
        """Fence new work and wait up to ``timeout`` seconds for this client's work.

        Returns True if operations and tasks finished within the deadline. Otherwise
        returns False and requests cancellation of overdue owned tasks once. Cleanup
        that suppresses cancellation stays strongly owned; the client remains
        ``closing`` until it and any in-flight caller operations really finish. Late
        recognitions are discarded. Inspect ``state`` before disposing borrowed host
        resources: the session factory, engine and adapters may still be in use after
        a False result. The host must keep the event loop running for cleanup to finish.

        Closing runs once, shielded from caller cancellation. Concurrent and later
        calls share the first call's deadline and stable boolean result; they never
        cancel cleanup again. ``state`` can subsequently become ``closed`` even when
        the recorded result is False. Other clients' work is never waited on or cancelled.
        """
        if self._closer is None:
            self._state = _CLOSING
            _CLOSING_CLIENTS.add(self)
            self._closer = asyncio.ensure_future(self._finish_closing(timeout))
        return await asyncio.shield(self._closer)

    async def _finish_closing(self, timeout: float) -> bool:  # noqa: ASYNC109 — see close()
        drained = False
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        try:
            for turn in self._turns:
                turn.begin_close()
                task = self.run_owned(partial(asyncio.create_task, turn.close()))
                self.adopt_cleanup(task)
            try:
                async with asyncio.timeout(timeout):
                    await self._idle.wait()
            except TimeoutError:
                # The caller's own operations cannot be cancelled from here. They stay this
                # client's: fenced (no new task, commit or result survives) and still bound.
                logger.warning("MemoryClient.close: %d operation(s) still in flight at the timeout", self._in_flight)
            unsettled = [s for s in self._staged if s.discard()]
            self._staged.clear()
            if unsettled:
                logger.info("MemoryClient.close: discarded %d unsettled recognition(s)", len(unsettled))
            drained = await self.drain(max(0.0, deadline - loop.time())) and self._in_flight == 0
            if not drained:
                logger.warning("MemoryClient.close: cancelling %d task(s) still running", len(self._running()))
                # Cancellation is a request, not resource completion. Never await a
                # cancellation-resistant task past the deadline or cancel its cleanup again.
                for task in self._tasks:
                    if not task.cancelling():
                        task.cancel()
        finally:
            self._close_finished = True
            self._settle_close()
        return drained

    async def __aenter__(self) -> MemoryClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
