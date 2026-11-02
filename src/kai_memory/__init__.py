"""Memory package.

Public surface:

* the memory SDK (DB-854): ``MemoryClient``, the ``HostAdapters`` a host supplies,
  and their data types, versioned by ``API_VERSION`` (see ``sdk.py``);
* staged turn delivery and procedural evidence types returned by the client.

Everything else in the package is internal.

New hosts use ``MemoryClient``. ``MemoryInterface`` and its data types remain
compatibility contracts for existing backend adapters, not an alternative host
lifecycle. Kai's legacy module aliases resolve to this same installed package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kai_memory.identity import AgentIdentity
from kai_memory.configuration import HostPolicy, ModelRoles, NeutralHostPolicy

from kai_memory.types import (
    Artifact,
    ArtifactId,
    ArtifactQuery,
    ArtifactType,
    ConsolidationReport,
    ConsolidationRunId,
    ConsolidationTrigger,
    Cue,
    EncodingResult,
    EncodingSignals,
    EntityId,
    EntityType,
    Experience,
    FeedbackResult,
    FeedbackSignal,
    FeedbackTarget,
    LinkRole,
    MemoryInterface,
    MemoryType,
    Modality,
    Representation,
    RetrievalContext,
    RetrievalResult,
    TraceId,
    TracePrime,
)

if TYPE_CHECKING:
    from kai_memory.turn import DecisionReceipt, PreparedContext, PreparedRecall, TurnMemory, TurnReceipt
    from kai_memory.procedural_api import (
        CaptureReceipt,
        ExecutionEvidence,
        OutcomeReceipt,
        ProceduralMemory,
        RecordingReceipt,
        RuleCandidate,
        ServingReceipt,
        ToolEvidence,
    )
    from kai_memory.sdk import (
        API_VERSION,
        Chat,
        Embedder,
        History,
        HostAdapters,
        Inference,
        MemoryClient,
        MemorySettings,
        Message,
        NoopTelemetry,
        Telemetry,
        __version__,
    )

_SDK_EXPORTS = frozenset(
    (
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
    )
)

_TURN_EXPORTS = frozenset(("DecisionReceipt", "PreparedContext", "PreparedRecall", "TurnMemory", "TurnReceipt"))
_PROCEDURAL_EXPORTS = frozenset(
    (
        "CaptureReceipt",
        "ExecutionEvidence",
        "OutcomeReceipt",
        "ProceduralMemory",
        "RecordingReceipt",
        "RuleCandidate",
        "ServingReceipt",
        "ToolEvidence",
    )
)


def __getattr__(name: str) -> Any:
    if name in _TURN_EXPORTS | _PROCEDURAL_EXPORTS:
        from importlib import import_module

        module = import_module("kai_memory.turn" if name in _TURN_EXPORTS else "kai_memory.procedural_api")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _SDK_EXPORTS:
        from kai_memory import sdk

        value = getattr(sdk, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = [
    "API_VERSION",
    "AgentIdentity",
    "Artifact",
    "ArtifactId",
    "ArtifactQuery",
    "ArtifactType",
    "CaptureReceipt",
    "Chat",
    "ConsolidationReport",
    "ConsolidationRunId",
    "ConsolidationTrigger",
    "Cue",
    "DecisionReceipt",
    "Embedder",
    "EncodingResult",
    "EncodingSignals",
    "EntityId",
    "EntityType",
    "ExecutionEvidence",
    "Experience",
    "FeedbackResult",
    "FeedbackSignal",
    "FeedbackTarget",
    "History",
    "HostAdapters",
    "HostPolicy",
    "Inference",
    "LinkRole",
    "MemoryClient",
    "MemoryInterface",
    "MemorySettings",
    "MemoryType",
    "Message",
    "Modality",
    "ModelRoles",
    "NeutralHostPolicy",
    "NoopTelemetry",
    "OutcomeReceipt",
    "PreparedContext",
    "PreparedRecall",
    "ProceduralMemory",
    "RecordingReceipt",
    "Representation",
    "RetrievalContext",
    "RetrievalResult",
    "RuleCandidate",
    "ServingReceipt",
    "Telemetry",
    "ToolEvidence",
    "TraceId",
    "TracePrime",
    "TurnMemory",
    "TurnReceipt",
    "__version__",
]
