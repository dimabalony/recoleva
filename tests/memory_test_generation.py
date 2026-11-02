"""Synthetic response/provenance boundary for iterative retrieval tests."""
from types import SimpleNamespace
LLMResponse = SimpleNamespace

_KEYS = frozenset({"provider", "model", "billing_mode", "generation_id",
                   "selected_provider", "selected_model", "fallback_reason"})

def record_generation_provenance(metadata, provenance, *, used_for_response=True):
    safe = {key: value for key, value in (provenance or {}).items()
            if key in _KEYS and isinstance(value, str) and 0 < len(value) <= 256}
    if not safe:
        return
    metadata.setdefault("llm_generations", []).append(safe)
    if used_for_response:
        for source, target in (("provider", "actual_provider"), ("model", "actual_model"),
                               ("billing_mode", "billing_mode")):
            if source in safe:
                metadata[target] = safe[source]
