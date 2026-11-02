"""Level 3 nightly consolidation — deep synthesis, decay, self-reflection."""

from __future__ import annotations

from kai_memory.host_calls import get_model_roles
from kai_memory.identity import identity_merge_survivor

import asyncio
import hashlib
import json
import logging
import math
import re
import uuid as _uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast

from sqlalchemy import func, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.host_calls import (
    call_llm_text,
    get_provider_for_model,
    get_settings,
    parse_json_array_response,
    session_scope,
    start_generation_span,
    utc_now,
)
from kai_memory.storage.models.consolidation_run import ConsolidationRun
from kai_memory.storage.models.entity_merge_candidate import EntityMergeCandidate
from kai_memory.storage.models.entity_relationship import EntityRelationship
from kai_memory.storage.models.memory_entity import MemoryEntity
from kai_memory.storage.models.memory_trace import MemoryTrace
from kai_memory.storage.models.social_context import SocialContext, SocialExperience
from kai_memory.ownership import current_owner
from kai_memory.store import cognitive_config
from kai_memory.store.cognitive_memory import CognitiveMemoryService
from kai_memory.store.cognitive_memory_constants import (  # noqa: F401 — MEMORY_TYPES imported for type validation visibility
    MEMORY_TYPES,
    ConflictAction,
    MergeCandidateStatus,
)
from kai_memory.store.embeddings import EmbeddingError, generate_embeddings_batch
from kai_memory.artifact_pipeline import ArtifactPipeline
from kai_memory.artifacts.producers import (
    CueMappingProducer,
    EntitySchemaProducer,
    IdentityRecordProducer,
    PatternProducer,
    StanceProducer,
    TraitProducer,
    TrajectoryProducer,
)
from kai_memory.durable_schema import DURABLE_SINGLE_SLOTS
from kai_memory.entity_resolution_tier3 import run_tier3_resolution
from kai_memory.procedural_liveness import assert_rule_liveness
from kai_memory.procedural_reinforcement import run_reinforcement

# Pinned to the pre-DB-842 module path: CloudWatch queries, feature-watch criteria and
# log filters match the logger name, which must not follow the module's move.
_LEGACY_LOGGER_NAME = "app.services.consolidation"
logger = logging.getLogger(_LEGACY_LOGGER_NAME)

STALE_RUN_MINUTES = 30
SONNET_MODEL = "sonnet-5.5"
# Durable identity facts are floored ABOVE the prune line (silent_salience_threshold,
# default 0.01) so Ebbinghaus decay can never silently prune them (write-path fact-loss
# fix). This DECAY floor is the load-bearing mechanism — an initial-salience floor alone
# decays away over ~68d. Both decay passes apply it; the prune also spares durable rows.
# `durable ≠ immortal`: every durable guard keeps `invalidated_at IS NULL`, so an L3
# contradiction can still retire a WRONG durable fact via invalidation. Read from the
# persisted `typed_metadata->>'durability'` marker (requires typed_metadata_write on).
DURABLE_SALIENCE_FLOOR = 0.3
# Sentinel for ordering durable traces whose created_at is somehow absent (should
# never happen for a stored row) — keeps Part C's newest-wins comparison total.
_EPOCH_DT = datetime(1970, 1, 1, tzinfo=UTC)
# Identity evolution (the legacy ``_evolve_identity`` step) was removed in
# Sub-design D Wave 1 — see _phase_synthesize for the freeze rationale.
# settings.consolidation_rewrite_model still exists for any future L3 step
# that needs Opus's 128k output ceiling, but no current step routes through it.
ADVISORY_LOCK_ID = int(hashlib.md5(b"consolidation_level_3", usedforsecurity=False).hexdigest()[:15], 16)
MERGE_CANDIDATE_BATCH_LIMIT = 200
MERGE_CHAIN_MAX_DEPTH = 10

# Shared L3 LLM concurrency gate. Named "sonnet" historically because every
# current caller defaults to Sonnet. Identity evolution (the previous Opus
# caller) was removed in Sub-design D Wave 1. If a future step routes to a
# different model and needs separate scheduling, size per-model limits here.
# A memory client gates its own calls (they go to its own inference adapter): ``ownership``.
_SONNET_SEMAPHORE: asyncio.Semaphore | None = None
_SONNET_CONCURRENCY = 5


def _get_sonnet_semaphore() -> asyncio.Semaphore:
    global _SONNET_SEMAPHORE
    owner = current_owner()
    if owner is not None:
        gate: asyncio.Semaphore = owner.caches.setdefault("sonnet_semaphore", asyncio.Semaphore(_SONNET_CONCURRENCY))
        return gate
    if _SONNET_SEMAPHORE is None:
        _SONNET_SEMAPHORE = asyncio.Semaphore(_SONNET_CONCURRENCY)
    return _SONNET_SEMAPHORE


class ConsolidationStepError(Exception):
    """Raised by a consolidation sub-step so consolidate() can aggregate failures."""

    def __init__(self, step: str, original: Exception) -> None:
        super().__init__(f"{step}: {original}")
        self.step = step
        self.original = original


async def _run_step(
    step_name: str,
    func_: Callable[[], Awaitable[Any]],
    stats: dict[str, Any],
    failures: list[str],
    session: AsyncSession | None = None,
) -> None:
    """Invoke a sub-step, routing exceptions into the failures list.

    Wraps the call in a savepoint so a DB error in one step doesn't abort
    the outer transaction. Without this, a failure in step N kills every
    subsequent query with "current transaction is aborted".
    """
    savepoint = None
    if session is not None:
        try:
            savepoint = await session.begin_nested()
        except Exception:
            logger.warning("Failed to open savepoint for step %s", step_name, exc_info=True)
    try:
        result = await func_()
        if result is not None:
            stats[step_name] = result
        if savepoint is not None and savepoint.is_active:
            await savepoint.commit()
    except ConsolidationStepError as e:
        logger.error("Consolidation step %s failed: %s", e.step, e.original)
        failures.append(e.step)
        if savepoint is not None and savepoint.is_active:
            await savepoint.rollback()
    except Exception as e:
        logger.error("Consolidation step %s crashed: %s", step_name, e, exc_info=True)
        failures.append(step_name)
        if savepoint is not None and savepoint.is_active:
            await savepoint.rollback()


# ── Main entry point ──────────────────────────────────────


async def consolidate() -> dict[str, Any]:
    """Level 3 nightly consolidation — runs 3 times per night."""
    async with session_scope() as session:
        await cognitive_config.load_config(session)
        run = await _start_run(session)
        if run is None:
            return {"skipped": True, "reason": "lock not acquired"}

        stats: dict[str, Any] = {}
        failures: list[str] = []

        try:
            await _phase_ingest(session, stats, failures)
            await _phase_synthesize(session, stats, failures, run.id)
            await _phase_maintain(session, stats, failures)

            result_meta = {**stats, "failures": failures}
            await _complete_run(session, run.id, result_meta)

            from kai_memory.ports import invalidate_prompt_cache

            invalidate_prompt_cache()

            logger.info("Level 3 consolidation complete: %s", result_meta)
            return {"success": len(failures) == 0, "failures": failures, **stats}
        except Exception as e:
            await _fail_run(session, run.id, str(e))
            raise


async def _phase_ingest(session: AsyncSession, stats: dict[str, Any], failures: list[str]) -> None:
    """Absorb new L2 traces: count, detect contradictions, update relationships."""
    await _run_step("new_traces", lambda: _process_new_traces(session), stats, failures, session)
    await _run_step("contradictions", lambda: _detect_contradictions(session), stats, failures, session)
    await _run_step("relationships_updated", lambda: _update_relationships(session), stats, failures, session)


async def _phase_synthesize(
    session: AsyncSession,
    stats: dict[str, Any],
    failures: list[str],
    run_id: _uuid.UUID,
) -> None:
    """LLM-powered synthesis: graph, norms, reflection, identity."""
    await _run_step("patterns_promoted", lambda: _promote_patterns(session), stats, failures, session)
    await _run_step("social_updates", lambda: _update_social_norms(session), stats, failures, session)
    await _run_step("reflections", lambda: _self_reflect(session), stats, failures, session)
    # Phase C §4.2 — the nightly procedural reinforcement/demotion ledger. No-op
    # unless procedural_reinforcement_enabled; idempotent; #241-contained (fixed
    # versioned constants, zero contact with retrieval_config / L1 weights / salience).
    await _run_step("rule_scoring", lambda: run_reinforcement(session), stats, failures, session)
    # DB-487 — assert the rule store can actually serve. Capture enabled with zero
    # active rules for >= 3 days is the DB-436 signature and now warns loudly.
    await _run_step("rule_liveness", lambda: assert_rule_liveness(session), stats, failures, session)
    # D Wave 1 — THE FREEZE. The legacy ``persona_updated`` step (which
    # nightly-rewrote agent_memory category=persona key=soul + instruction/main
    # via _evolve_identity) is REMOVED. Persona collapses on 2026-04-30 and
    # 2026-05-04 traced to compression bias in this exact path. Anchor row
    # in memory_artifacts (status='permanent') replaces the rewriteable
    # persona blob; PersonaComposer renders it at prompt-build time. See
    # backend-docs/ai-workforce/plans/design/2026-05-06-memory-v2d-persona-pipeline-design.md §5.1.
    # The legacy nightly retrieval-weight-adjustment step is REMOVED: it was a
    # positive-feedback runaway — graph nudged +0.01 whenever frequently_accessed
    # > 10 (always true in practice) then renormalized (g' = (g + 0.01) / 1.01),
    # so ~267 nightly passes drove graph 0.25 → 0.945. The 2026-07-02 prod sweep
    # measured the designed defaults BEATING that drifted mix (recall@8 0.75 vs
    # 0.60); migration 091 reset the row. Retrieval weights are operator-tuned
    # via retrieval_config now — never self-adjusted. Guarded by
    # tests/test_consolidation.py::test_runaway_weight_adjustment_removed. The
    # designed replacement (benchmark-gated, operator-approved adaptation) is
    # backend-docs/ai-workforce/plans/design/2026-07-02-benchmark-gated-weight-adaptation-design.md.
    # Typed-artifact pipeline (EntitySchema card etc.) — the live replacement for the
    # legacy ``entity.summary`` path. That path was removed as dead: nothing reads
    # ``MemoryEntity.summary`` anymore (the prompt builder serves the artifact cards),
    # yet it was regenerated by a Sonnet call per entity every pass. Pure cost.
    await _run_step(
        "artifacts_produced",
        lambda: _run_artifact_pipeline(session, run_id),
        stats,
        failures,
        session,
    )


async def _phase_maintain(session: AsyncSession, stats: dict[str, Any], failures: list[str]) -> None:
    """Bookkeeping: decay, pruning, growth controls."""
    # Part C — release fully-superseded durable facts (flip their marker) BEFORE decay,
    # so a demoted trace decays the same night instead of staying floored one more cycle.
    await _run_step("durable_released", lambda: _release_superseded_durable(session), stats, failures, session)
    await _run_step("decayed", lambda: _apply_decay(session), stats, failures, session)
    await _run_step("reassessed", lambda: _reassess_significance(session), stats, failures, session)
    await _run_step("pruned_traces", lambda: _prune_silent_traces(session), stats, failures, session)
    # Tier 3 (entity-resolution v2 §3): block active entities + write deterministic
    # merge candidates. Must run BEFORE _merge_entities so that, when auto-merge is
    # enabled, the PENDING candidates it writes execute in the same nightly sweep.
    await _run_step("tier3_candidates", lambda: run_tier3_resolution(session), stats, failures, session)
    await _run_step("entities_merged", lambda: _merge_entities(session), stats, failures, session)
    await _run_step("runs_cleaned", lambda: _cleanup_old_runs(session), stats, failures, session)
    await _run_step("timestamps_capped", lambda: _cap_access_timestamps(session), stats, failures, session)
    await _run_step("ranking_snapshotted", lambda: _snapshot_ranking_score(session), stats, failures, session)


async def _run_artifact_pipeline(session: AsyncSession, run_id: _uuid.UUID) -> dict[str, int]:
    """Run all five M2 Wave 2 producers and persist `changes_made` atomically.

    Returns a dict ``{action: count}`` summary. Producer-level isolation
    is handled inside ``ArtifactPipeline`` (per-producer savepoint).

    The whole step is additionally wrapped in an explicit
    ``session.begin_nested()`` so producer commits and the
    ``consolidation_runs.changes_made`` UPDATE land or roll back together.
    Without this wrapper, ``_run_step``'s outer savepoint setup can fail
    silently (it logs and continues without a savepoint) — producers
    would then commit independently, and a downstream UPDATE failure
    would leave artifact writes orphaned from their audit row.

    Since the Wave 3 cutover, prompt-builder consumes these artifacts via
    ``PersonaComposer.compose()`` (prompt_builder.py) — writes here and
    reads there must stay consistent.
    """
    # Defensive: ensure the pipeline always sees LIVE DB config (flags + tunables),
    # independent of any prior in-pass cache mutation. Producers like
    # EntitySchemaProducer gate on cognitive_config flags read here.
    await cognitive_config.load_config(session)
    async with session.begin_nested():
        pipeline = ArtifactPipeline(
            session,
            run_id,
            phase="synthesize",
            pass_number=1,
            venue="scheduler",
        )
        # Priority order per design §6 Wave 2 — EntitySchema first because
        # CueMapping references its outputs (target_schema_ids).
        pipeline.register(EntitySchemaProducer())
        pipeline.register(TrajectoryProducer())
        pipeline.register(IdentityRecordProducer())
        pipeline.register(CueMappingProducer())
        # ContrastiveMapProducer intentionally NOT registered — its discover_subjects
        # is a hardcoded no-op until M3 (ConfusionSetProducer). Keeping it in the
        # L3 loop showed up as 3x/night "0 produced" noise on the producer-health
        # dashboard. Re-enable when M3 lands.
        # Sub-design D Wave 2 — persona producers (shadow mode).
        # Order matters: PatternProducer first so TraitProducer can read
        # the same-run Pattern outputs when computing slow drift.
        # StanceProducer is independent of the others and can run last.
        # All three write to memory_artifacts; PersonaComposer reads them
        # via compose(), which prompt-builder renders since the D Wave 3
        # cutover (prompt_builder.py).
        pipeline.register(PatternProducer())
        pipeline.register(TraitProducer())
        pipeline.register(StanceProducer())

        entries = await pipeline.run_all()
        # Always persist — empty list ([]) for no-op runs, not NULL — so
        # downstream consumers can rely on ``changes_made`` being a list.
        # The column has a server-default of ``'[]'::jsonb`` but ORM writes
        # elsewhere may have bypassed that, so this UPDATE is the canonical
        # end-of-pass writer.
        await session.execute(
            update(ConsolidationRun)
            .where(ConsolidationRun.id == run_id)
            .values(changes_made=[e.to_dict() for e in entries])
        )

    summary: dict[str, int] = {}
    for entry in entries:
        summary[entry.action] = summary.get(entry.action, 0) + 1
    return summary


async def _snapshot_ranking_score(session: AsyncSession) -> int:
    """Snapshot mention_count into ranking_score for stable cached-prompt ordering."""
    try:
        result = await session.execute(
            text("UPDATE memory_entities SET ranking_score = mention_count WHERE ranking_score <> mention_count")
        )
        return result.rowcount or 0  # type: ignore[attr-defined]
    except Exception as e:
        logger.error("Snapshot ranking_score failed", exc_info=True)
        raise ConsolidationStepError("ranking_snapshotted", e) from e


# ── Run lifecycle ─────────────────────────────────────────


async def _start_run(session: AsyncSession) -> ConsolidationRun | None:
    """Acquire advisory lock, clean stale runs, create new run record."""
    result = await session.execute(text("SELECT pg_try_advisory_lock(:lock_id)"), {"lock_id": ADVISORY_LOCK_ID})
    acquired = result.scalar()
    if not acquired:
        logger.info("Consolidation lock not acquired — another pass running")
        return None

    # Reclaim orphaned running records from crashed processes
    cutoff = utc_now() - timedelta(minutes=STALE_RUN_MINUTES)
    await session.execute(
        update(ConsolidationRun)
        .where(
            ConsolidationRun.level == 3,
            ConsolidationRun.status == "running",
            ConsolidationRun.started_at < cutoff,
        )
        .values(status="failed", completed_at=utc_now(), error_message="stale: orphaned run")
    )

    run = ConsolidationRun(level=3, status="running")
    session.add(run)
    await session.flush()
    return run


async def _complete_run(session: AsyncSession, run_id: Any, stats: dict[str, Any]) -> None:
    await session.execute(
        update(ConsolidationRun)
        .where(ConsolidationRun.id == run_id)
        .values(
            status="completed",
            completed_at=utc_now(),
            traces_processed=stats.get("new_traces", 0),
            run_metadata=stats,
        )
    )
    await session.execute(text("SELECT pg_advisory_unlock(:lock_id)"), {"lock_id": ADVISORY_LOCK_ID})


async def _fail_run(session: AsyncSession, run_id: Any, error: str) -> None:
    try:
        await session.execute(
            update(ConsolidationRun)
            .where(ConsolidationRun.id == run_id)
            .values(
                status="failed",
                completed_at=utc_now(),
                error_message=error[:500],
            )
        )
    except Exception as e:
        logger.warning("Could not mark run as failed: %s", e)
    finally:
        try:
            await session.execute(text("SELECT pg_advisory_unlock(:lock_id)"), {"lock_id": ADVISORY_LOCK_ID})
        except Exception:
            logger.warning("Could not release advisory lock", exc_info=True)


# ── Step 1: Process new traces ────────────────────────────


async def _process_new_traces(session: AsyncSession) -> int:
    """Count unprocessed traces (consolidated_at IS NULL)."""
    try:
        result = await session.execute(
            select(func.count())
            .select_from(MemoryTrace)
            .where(MemoryTrace.consolidated_at.is_(None), MemoryTrace.invalidated_at.is_(None))
        )
        return result.scalar() or 0
    except Exception as e:
        logger.error("Process new traces failed", exc_info=True)
        raise ConsolidationStepError("new_traces", e) from e


# ── Step 2: Contradiction detection (Sonnet) ──────────────


async def _detect_contradictions(session: AsyncSession) -> int:
    """Find new facts contradicting existing facts via Sonnet."""
    try:
        svc = CognitiveMemoryService(session)

        batch_limit = cognitive_config.get_int("contradiction_batch")
        result = await session.execute(
            select(MemoryTrace)
            .where(
                MemoryTrace.consolidated_at.is_(None),
                MemoryTrace.invalidated_at.is_(None),
                MemoryTrace.memory_type.in_(["fact", "status_change", "preference"]),
            )
            .limit(batch_limit)
        )
        new_traces = list(result.scalars().all())
        if not new_traces:
            return 0

        # Batch-embed all new trace contents in one API call. These embeddings
        # are used only for similarity search (not persisted), so if the
        # embedding API is down we skip contradiction detection this pass
        # rather than crash the nightly consolidation — it resumes next run.
        trace_contents = [t.content for t in new_traces]
        try:
            embeddings = await generate_embeddings_batch(trace_contents)
        except EmbeddingError:
            logger.warning(
                "Contradiction detection: embedding API unavailable — skipping this pass (%d traces deferred)",
                len(new_traces),
            )
            return 0

        new_facts = []
        existing_facts = []
        known_new_ids: set = set()
        known_existing_ids: set = set()
        for trace, emb in zip(new_traces, embeddings, strict=True):
            try:
                similar = await svc.search_traces_by_vector(emb, limit=5)
                for existing, _sim in similar:
                    if existing.id != trace.id and existing.invalidated_at is None:
                        existing_facts.append(f"[id={existing.id}] {existing.content}")
                        known_existing_ids.add(existing.id)
            except Exception:
                pass
            new_facts.append(f"[id={trace.id}] {trace.content}")
            known_new_ids.add(trace.id)

        if not existing_facts:
            return 0

        prompt = (
            "You are analyzing memories for contradictions. Given:\n\n"
            "EXISTING FACTS:\n" + "\n".join(existing_facts[:batch_limit]) + "\n\n"
            "NEW FACTS:\n" + "\n".join(new_facts) + "\n\n"
            "Identify NEW facts that CONTRADICT existing facts. A contradiction means "
            "the new fact makes the old fact no longer true — not just different information.\n\n"
            'Return JSON array: [{"new_trace_id": "...", "contradicts_trace_id": "...", '
            '"confidence": 0.0-1.0, "reason": "..."}]\n'
            "Return empty array [] if no contradictions found."
        )

        items = await _call_sonnet(prompt, "contradiction-detection")
        count = 0
        for item in items:
            try:
                new_id = _uuid.UUID(item["new_trace_id"])
                old_id = _uuid.UUID(item["contradicts_trace_id"])
                # Validate: both IDs must be from the set we passed to Sonnet
                if old_id not in known_existing_ids or new_id not in known_new_ids:
                    logger.warning(
                        "Skipping fabricated contradiction — unknown IDs: new=%s old=%s",
                        new_id,
                        old_id,
                    )
                    continue
                confidence = float(item.get("confidence", 0.8))
                # Sub-design C INC 1: tag this as an L3-batch correction so E
                # (INC 8) can separate it from the high-volume L2-inline stream
                # via GROUP BY. Flag-gated in invalidate_trace.
                await svc.invalidate_trace(
                    old_id,
                    invalidated_by=new_id,
                    confidence=confidence,
                    invalidation_source="l3_batch",
                )
                count += 1
            except (ValueError, KeyError):
                pass
        return count
    except Exception as e:
        logger.error("Contradiction detection failed", exc_info=True)
        raise ConsolidationStepError("contradictions", e) from e


# ── Step 3: Update relationships ──────────────────────────


async def _update_relationships(session: AsyncSession) -> int:
    """Expire relationships whose source memory was invalidated."""
    try:
        result = await session.execute(
            update(EntityRelationship)
            .where(
                EntityRelationship.valid_until.is_(None),
                EntityRelationship.source_memory_id.isnot(None),
                EntityRelationship.source_memory_id.in_(
                    select(MemoryTrace.id).where(MemoryTrace.invalidated_at.isnot(None))
                ),
            )
            .values(valid_until=utc_now())
        )
        return result.rowcount
    except Exception as e:
        logger.error("Relationship update failed", exc_info=True)
        raise ConsolidationStepError("relationships_updated", e) from e


# ── Apply decay ──────────────────────────────────────────


async def _decay_bulk_sql_pass(session: AsyncSession, now: datetime) -> int:
    """Pure-SQL decay for traces that have never been accessed.

    Anchoring on ``created_at`` is correct here and needs no DB-437
    retrieval-anchored equivalent: ``last_accessed_at`` is written only by
    ``CognitiveMemoryService.record_access``/``record_access_batch``, which
    bump ``access_count`` in the same UPDATE, so ``access_count = 0`` implies
    ``last_accessed_at IS NULL`` and the two anchors coincide.
    """
    bulk_result = await session.execute(
        text("""
            UPDATE memory_traces SET
                salience = GREATEST(
                    -- durable identity facts never decay below the floor (fact-loss fix)
                    CASE WHEN typed_metadata->>'durability' = 'durable' THEN CAST(:durable_floor AS float) ELSE 0 END,
                    LEAST(
                        initial_salience,
                        initial_salience * exp(
                            -CAST(:base_decay AS float) * (6 - significance) / 5.0
                            * EXTRACT(EPOCH FROM (CAST(:now AS timestamptz) - created_at)) / 86400.0
                        )
                    )
                ),
                consolidated_at = CAST(:now AS timestamptz)
            WHERE significance < 5
              AND invalidated_at IS NULL
              AND access_count = 0
              AND (consolidated_at IS NULL OR consolidated_at < :day_start)
        """),
        {
            "base_decay": cognitive_config.get("base_decay"),
            "durable_floor": DURABLE_SALIENCE_FLOOR,
            "now": now,
            "day_start": now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0),
        },
    )
    return bulk_result.rowcount or 0  # type: ignore[attr-defined]


async def _decay_batched_pass(session: AsyncSession, now: datetime) -> int:
    """Python-loop decay for accessed traces — spaced-repetition bonus applies."""
    batch_size = cognitive_config.get_int("decay_batch_size")
    base_decay = cognitive_config.get("base_decay")
    last_id: _uuid.UUID | None = None
    count = 0
    # DB-437 invariant monitor: this pass selects `access_count > 0`, and the only
    # writers of `last_accessed_at` bump `access_count` in the same UPDATE, so a NULL
    # last-access here is impossible. If a future writer breaks that, the anchor
    # silently reverts to `created_at` — precisely the drift this PR fixes — so the
    # violation is counted and reported once per pass (never per row, which would
    # spam CloudWatch if a legacy population ever matched).
    missing_last_access = 0
    while True:
        query = (
            select(MemoryTrace)
            .where(
                MemoryTrace.significance < 5,
                MemoryTrace.invalidated_at.is_(None),
                MemoryTrace.access_count > 0,
                # Phase A: procedural lifecycle is the utility pass's, not episodic decay's.
                MemoryTrace.memory_type != "procedural",
                or_(
                    MemoryTrace.consolidated_at.is_(None),
                    MemoryTrace.consolidated_at
                    < now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0),
                ),
            )
            .order_by(MemoryTrace.id)
            .limit(batch_size)
        )
        if last_id is not None:
            query = query.where(MemoryTrace.id > last_id)
        batch_result = await session.execute(query)
        batch = list(batch_result.scalars().all())
        if not batch:
            break

        # Collect (id, new_salience) pairs for bulk UPDATE
        updates: list[tuple[_uuid.UUID, float]] = []
        for trace in batch:
            effective_lambda = base_decay * (6 - trace.significance) / 5
            # DB-437: the forgetting curve is anchored on the LAST RETRIEVAL, not on
            # creation. Retrieval is a rehearsal event — an Ebbinghaus curve that never
            # resets means a trace fetched every day still decays to zero on its
            # creation-date schedule and is then hard-deleted by `_prune_silent_traces`
            # (prod, 2026-07-31: a 779-access trace accessed the previous day was pruned).
            # `max()` (not a plain `or`) keeps the anchor monotonic if a clock skew or a
            # backfill ever leaves `last_accessed_at` behind `created_at`.
            if trace.last_accessed_at is None:
                missing_last_access += 1
            decay_anchor = max(trace.created_at, trace.last_accessed_at or trace.created_at)
            days_since = max(0, (now - decay_anchor).total_seconds() / 86400)
            spaced_bonus = _compute_spaced_access_bonus(trace.access_timestamps or [])
            new_salience = trace.initial_salience * math.exp(-effective_lambda * days_since) * (1 + spaced_bonus)
            new_salience = min(trace.initial_salience, new_salience)
            # Durable identity facts never decay below the floor — identical behavior to
            # the bulk SQL pass (both-pass sync). Reads the persisted trace-level marker.
            if (trace.typed_metadata or {}).get("durability") == "durable":
                new_salience = max(new_salience, DURABLE_SALIENCE_FLOOR)
            updates.append((trace.id, new_salience))

        # Single bulk UPDATE per batch using unnest arrays
        if updates:
            ids = [u[0] for u in updates]
            saliences = [u[1] for u in updates]
            await session.execute(
                text("""
                    UPDATE memory_traces SET
                        salience = bulk.new_salience,
                        consolidated_at = :now
                    FROM unnest(CAST(:ids AS uuid[]), CAST(:saliences AS float[])) AS bulk(id, new_salience)
                    WHERE memory_traces.id = bulk.id
                """),
                {"ids": ids, "saliences": saliences, "now": now},
            )
            count += len(updates)

        last_id = batch[-1].id
        if len(batch) < batch_size:
            break
    if missing_last_access:
        logger.warning(
            "memory.decay.access_count_without_last_accessed_at count=%d — decay anchor fell back to created_at",
            missing_last_access,
        )
    return count


async def _apply_decay(session: AsyncSession) -> int:
    """Significance-gated Ebbinghaus decay on memory salience.

    Two passes: bulk SQL for traces with no access history (no spaced bonus),
    then a batched Python loop for traces with access_timestamps.
    """
    try:
        now = utc_now()
        total_count = 0
        total_count += await _decay_bulk_sql_pass(session, now)
        total_count += await _decay_batched_pass(session, now)

        # Mark sig-5 traces as consolidated (no salience update)
        await session.execute(
            update(MemoryTrace)
            .where(
                MemoryTrace.significance == 5,
                MemoryTrace.invalidated_at.is_(None),
                MemoryTrace.consolidated_at.is_(None),
            )
            .values(consolidated_at=now)
        )

        # Observability for the write-path fact-loss fix: how many durable identity
        # facts the retention floor is actively protecting from decay-pruning. A
        # nightly heartbeat (fires even at count=0) so the feature-watch can confirm
        # the retention path is live post-deploy.
        durable_retained = (
            await session.execute(
                text(
                    "SELECT count(*) FROM memory_traces "
                    "WHERE typed_metadata->>'durability' = 'durable' AND invalidated_at IS NULL"
                )
            )
        ).scalar() or 0
        logger.info("memory.decay.durable_retained count=%d", durable_retained)
        return total_count
    except Exception as e:
        logger.error("Decay application failed", exc_info=True)
        raise ConsolidationStepError("decayed", e) from e


# ── Part C: release superseded durable facts ──────────────


def _select_releasable_durable_traces(traces: list[dict[str, Any]]) -> list[Any]:
    """Pure: durable traces whose EVERY durable fact is a superseded SINGLE-slot fact,
    so they can be released to decay (write-path fact-loss fix, Part C / F-1).

    Each trace is ``{"id", "created_at", "facts": [...]}``. Within a durable SINGLE slot
    (newest-wins) the newest ``(subject_id, slot)`` trace is current; older ones are
    superseded. A trace is releasable ONLY when all of its durable facts are superseded
    SINGLE-slot facts — the multi-fact edge (C-1/F-1): a trace KEEPS its floor if it
    carries any still-current SINGLE fact, or any durable MULTI / off-schema fact (those
    accumulate and are never superseded by newest-wins).
    """

    def _order(t: dict[str, Any]) -> tuple[datetime, str]:
        return (t.get("created_at") or _EPOCH_DT, str(t.get("id")))

    # Newest (subject_id, slot) → order key, across all durable SINGLE-slot facts.
    newest: dict[tuple[str, str], tuple[datetime, str]] = {}
    for t in traces:
        ok = _order(t)
        for f in t.get("facts") or []:
            if not isinstance(f, dict) or f.get("durability") != "durable":
                continue
            slot, sid = f.get("slot"), f.get("subject_id")
            if slot in DURABLE_SINGLE_SLOTS and sid:
                key = (str(sid), slot)
                if key not in newest or ok > newest[key]:
                    newest[key] = ok

    releasable: list[Any] = []
    for t in traces:
        ok = _order(t)
        durable_facts = [f for f in (t.get("facts") or []) if isinstance(f, dict) and f.get("durability") == "durable"]
        if not durable_facts:
            continue
        all_superseded = True
        for f in durable_facts:
            slot, sid = f.get("slot"), f.get("subject_id")
            if slot in DURABLE_SINGLE_SLOTS and sid and newest.get((str(sid), slot)) != ok:
                continue  # this SINGLE fact is superseded by a newer trace → releasable so far
            all_superseded = False  # still-current SINGLE, or a durable MULTI/off-schema → keep the floor
            break
        if all_superseded:
            releasable.append(t.get("id"))
    return releasable


async def _release_superseded_durable(session: AsyncSession) -> int:
    """Part C — nightly release of superseded durable facts (write-path fact-loss fix).

    Floor 0.3 > prune 0.01 ⇒ a durable-tagged trace is un-prunable with no release path;
    superseded single-value facts (old employer, previous surname, former city) would
    otherwise stay floored forever → monotonic durable accumulation → retrieval dilution
    (a #241-shape failure: the recall fix eroding recall at scale). This demotes each
    FULLY-superseded durable trace to ``durability='superseded'`` so the decay/prune
    passes stop sparing it and it fades out naturally. ``durable ≠ immortal``;
    ``invalidated_at`` remains the escape hatch for a WRONG (not merely stale) fact.
    """
    try:
        rows = (
            (
                await session.execute(
                    text(
                        "SELECT id, created_at, typed_metadata->'facts' AS facts "
                        "FROM memory_traces "
                        "WHERE typed_metadata->>'durability' = 'durable' AND invalidated_at IS NULL"
                    )
                )
            )
            .mappings()
            .all()
        )

        def _facts(raw: Any) -> list[Any]:
            # asyncpg usually decodes jsonb to a Python list, but a raw text() sub-expression
            # can surface as a JSON string — parse defensively so release never silently no-ops.
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except (ValueError, TypeError):
                    return []
            return raw if isinstance(raw, list) else []

        traces = [{"id": r["id"], "created_at": r["created_at"], "facts": _facts(r["facts"])} for r in rows]
        releasable = _select_releasable_durable_traces(traces)
        if releasable:
            await session.execute(
                text(
                    "UPDATE memory_traces "
                    "SET typed_metadata = typed_metadata || "
                    # CAST(:now AS text): asyncpg can't infer a bind param's type inside
                    # jsonb_build_object's polymorphic ("any") arg — an explicit cast fixes it.
                    # (:ids already carries its own CAST(... AS uuid[]) and was never the
                    # indeterminate param, so it stays untouched — this delta is a pure cast.)
                    "  jsonb_build_object('durability', 'superseded', 'superseded_at', CAST(:now AS text)) "
                    "WHERE id = ANY(CAST(:ids AS uuid[])) "
                    "  AND typed_metadata->>'durability' = 'durable'"  # guard: only still-durable rows
                ),
                {"ids": [str(i) for i in releasable], "now": utc_now().isoformat()},
            )
        logger.info("memory.durable.superseded_released count=%d", len(releasable))
        return len(releasable)
    except Exception as e:
        logger.error("Durable supersede-release failed", exc_info=True)
        raise ConsolidationStepError("durable_released", e) from e


def _compute_spaced_access_bonus(access_timestamps: list) -> float:
    """Boost for spaced repetition — larger gaps between accesses = better retention."""
    if not access_timestamps or len(access_timestamps) < 2:
        return 0.0

    # Parse last 10 timestamps
    recent = access_timestamps[-10:]
    try:
        parsed = sorted(datetime.fromisoformat(ts) if isinstance(ts, str) else ts for ts in recent)
    except (ValueError, TypeError):
        return 0.0

    gaps = []
    for i in range(1, len(parsed)):
        gap_days = (parsed[i] - parsed[i - 1]).total_seconds() / 86400
        gaps.append(gap_days)

    if not gaps:
        return 0.0

    avg_gap = sum(gaps) / len(gaps)
    scale = cognitive_config.get("spaced_bonus_scale")
    cap = cognitive_config.get("spaced_bonus_cap")
    return min(1.0, avg_gap / scale) * cap


# ── Step 6: Promote patterns (Sonnet) ─────────────────────


@dataclass(frozen=True)
class RelationshipPredicateRule:
    allowed_pairs: frozenset[tuple[str, str]]
    canonical_direction: str
    symmetric: bool


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    corrected_from_id: _uuid.UUID | None
    corrected_to_id: _uuid.UUID | None
    predicate: str
    reason: str


_CORE_ENTITY_TYPES = (
    "person",
    "agent",
    "animal",
    "pet",
    "place",
    "organization",
    "topic",
    "project",
    "object",
    "tool",
    "event",
    "group",
)
_SAME_TYPE_PAIRS = frozenset((entity_type, entity_type) for entity_type in _CORE_ENTITY_TYPES)
_COMPARABLE_ENTITY_PAIRS = frozenset(
    (entity_type, entity_type) for entity_type in ("object", "tool", "project", "organization", "topic")
)
_COMPETITOR_ENTITY_PAIRS = frozenset(
    (entity_type, entity_type) for entity_type in ("organization", "object", "tool", "project")
)
_COMPLEMENT_ENTITY_PAIRS = frozenset((entity_type, entity_type) for entity_type in ("topic", "object", "tool"))
_ANIMAL_PET_PAIRS = frozenset({("animal", "animal"), ("pet", "pet"), ("animal", "pet")})
_PLACE_KIND_RANKS = {
    "world": 100,
    "continent": 90,
    "country": 80,
    "nation": 80,
    "state": 65,
    "province": 65,
    "region": 65,
    "island": 60,
    "city": 40,
    "town": 35,
    "village": 30,
    "neighborhood": 20,
    "district": 20,
}
_PLACE_NAME_RANK_HINTS = {
    "indonesia": 80,
    "bali": 60,
}


_RELATIONSHIP_PREDICATE_ALLOWLIST: dict[str, RelationshipPredicateRule] = {
    "partner_of": RelationshipPredicateRule(frozenset({("person", "person")}), "person <-> person", True),
    "friend_of": RelationshipPredicateRule(
        frozenset({("person", "person"), ("person", "agent"), ("agent", "agent")}),
        "person/agent <-> person/agent",
        True,
    ),
    "family_of": RelationshipPredicateRule(frozenset({("person", "person")}), "person <-> person", True),
    "colleague_of": RelationshipPredicateRule(
        frozenset({("person", "person"), ("person", "agent"), ("agent", "agent")}),
        "person/agent <-> person/agent",
        True,
    ),
    "coworker_of": RelationshipPredicateRule(
        frozenset({("person", "person"), ("person", "agent"), ("agent", "agent")}),
        "person/agent <-> person/agent",
        True,
    ),
    "same_person": RelationshipPredicateRule(_SAME_TYPE_PAIRS, "same entity type <-> same entity type", True),
    "alternative_to": RelationshipPredicateRule(
        _COMPARABLE_ENTITY_PAIRS,
        "comparable entity <-> comparable entity",
        True,
    ),
    "competitor_of": RelationshipPredicateRule(
        _COMPETITOR_ENTITY_PAIRS,
        "competing entity <-> competing entity",
        True,
    ),
    "companion_of": RelationshipPredicateRule(
        _ANIMAL_PET_PAIRS,
        "animal/pet <-> animal/pet",
        True,
    ),
    "complement_of": RelationshipPredicateRule(
        _COMPLEMENT_ENTITY_PAIRS,
        "complementary entity <-> complementary entity",
        True,
    ),
    "parent_of": RelationshipPredicateRule(
        frozenset({("person", "person"), ("animal", "animal"), ("pet", "pet")}),
        "parent entity -> child entity",
        False,
    ),
    "child_of": RelationshipPredicateRule(
        frozenset({("person", "person"), ("animal", "animal"), ("pet", "pet")}),
        "child entity -> parent entity",
        False,
    ),
    "manager_of": RelationshipPredicateRule(
        frozenset({("person", "person"), ("person", "agent")}),
        "manager -> report",
        False,
    ),
    "reports_to": RelationshipPredicateRule(
        frozenset({("person", "person"), ("agent", "person")}),
        "report -> manager",
        False,
    ),
    "mentor_of": RelationshipPredicateRule(
        frozenset({("person", "person"), ("person", "agent"), ("agent", "person")}),
        "mentor -> mentee",
        False,
    ),
    "works_at": RelationshipPredicateRule(
        frozenset({("person", "organization"), ("agent", "organization")}),
        "person/agent -> organization",
        False,
    ),
    "employee_of": RelationshipPredicateRule(
        frozenset({("person", "organization"), ("agent", "organization")}),
        "person/agent -> organization",
        False,
    ),
    "member_of": RelationshipPredicateRule(
        frozenset(
            {
                ("person", "organization"),
                ("person", "group"),
                ("person", "project"),
                ("agent", "organization"),
                ("agent", "group"),
                ("agent", "project"),
                ("organization", "organization"),
            }
        ),
        "member -> organization/group",
        False,
    ),
    "leader_of": RelationshipPredicateRule(
        frozenset(
            {
                ("person", "project"),
                ("person", "organization"),
                ("person", "group"),
                ("agent", "project"),
                ("agent", "organization"),
                ("agent", "group"),
            }
        ),
        "leader -> project/organization/group",
        False,
    ),
    "customer_of": RelationshipPredicateRule(
        frozenset(
            {
                ("person", "organization"),
                ("agent", "organization"),
                ("organization", "organization"),
            }
        ),
        "customer -> organization",
        False,
    ),
    "prospective_customer_of": RelationshipPredicateRule(
        frozenset(
            {
                ("person", "organization"),
                ("agent", "organization"),
                ("organization", "organization"),
            }
        ),
        "prospective customer -> organization",
        False,
    ),
    "owner_of": RelationshipPredicateRule(
        frozenset(
            {
                ("person", "animal"),
                ("person", "pet"),
                ("person", "object"),
                ("person", "tool"),
                ("person", "group"),
                ("organization", "object"),
                ("organization", "tool"),
                ("organization", "project"),
            }
        ),
        "owner -> owned entity",
        False,
    ),
    "owned_by": RelationshipPredicateRule(
        frozenset(
            {
                ("animal", "person"),
                ("pet", "person"),
                ("object", "person"),
                ("tool", "person"),
                ("object", "organization"),
                ("tool", "organization"),
                ("project", "organization"),
            }
        ),
        "owned entity -> owner",
        False,
    ),
    "uses": RelationshipPredicateRule(
        frozenset(
            {
                ("person", "tool"),
                ("person", "object"),
                ("agent", "tool"),
                ("agent", "object"),
                ("organization", "tool"),
                ("project", "tool"),
            }
        ),
        "user -> tool/object",
        False,
    ),
    "user_of": RelationshipPredicateRule(
        frozenset(
            {
                ("person", "organization"),
                ("person", "tool"),
                ("person", "object"),
                ("person", "project"),
                ("person", "topic"),
                ("agent", "organization"),
                ("agent", "tool"),
                ("agent", "object"),
                ("agent", "project"),
                ("agent", "topic"),
                ("organization", "organization"),
                ("organization", "tool"),
                ("organization", "object"),
                ("organization", "project"),
                ("organization", "topic"),
            }
        ),
        "user -> service/tool/object/project/topic",
        False,
    ),
    "assistant_of": RelationshipPredicateRule(
        frozenset(
            {
                ("agent", "person"),
                ("agent", "organization"),
                ("tool", "person"),
                ("tool", "organization"),
            }
        ),
        "assistant/tool -> assisted person/organization",
        False,
    ),
    "lives_in": RelationshipPredicateRule(
        frozenset({("person", "place"), ("agent", "place"), ("animal", "place"), ("pet", "place")}),
        "resident -> place",
        False,
    ),
    "resident_of": RelationshipPredicateRule(
        frozenset({("person", "place"), ("agent", "place"), ("animal", "place"), ("pet", "place")}),
        "resident -> place",
        False,
    ),
    "hometown_of": RelationshipPredicateRule(
        frozenset({("place", "person")}),
        "place -> person",
        False,
    ),
    "located_in": RelationshipPredicateRule(
        frozenset(
            {
                ("place", "place"),
                ("organization", "place"),
                ("event", "place"),
                ("group", "place"),
                ("object", "place"),
                ("tool", "place"),
            }
        ),
        "contained place/entity -> containing place",
        False,
    ),
    "part_of": RelationshipPredicateRule(
        frozenset(
            {
                ("place", "place"),
                ("organization", "organization"),
                ("project", "organization"),
                ("topic", "topic"),
                ("object", "object"),
                ("tool", "object"),
                ("event", "project"),
                ("group", "organization"),
            }
        ),
        "part/component -> whole/container",
        False,
    ),
    "business_registered_in": RelationshipPredicateRule(
        frozenset({("organization", "place"), ("person", "place")}),
        "registered entity -> place",
        False,
    ),
    "used_in": RelationshipPredicateRule(
        frozenset(
            {
                ("object", "place"),
                ("object", "project"),
                ("object", "event"),
                ("tool", "place"),
                ("tool", "project"),
                ("tool", "event"),
            }
        ),
        "object/tool -> place/project/event",
        False,
    ),
    "interested_in": RelationshipPredicateRule(
        frozenset(
            {
                ("person", "topic"),
                ("person", "project"),
                ("person", "object"),
                ("person", "place"),
                ("agent", "topic"),
                ("agent", "project"),
            }
        ),
        "interested entity -> subject",
        False,
    ),
    "preference_for": RelationshipPredicateRule(
        frozenset(
            {
                ("person", "topic"),
                ("person", "object"),
                ("person", "tool"),
                ("person", "place"),
                ("agent", "topic"),
                ("agent", "object"),
                ("agent", "tool"),
            }
        ),
        "person/agent -> preferred subject",
        False,
    ),
    "practitioner_of": RelationshipPredicateRule(
        frozenset({("person", "topic"), ("agent", "topic")}),
        "practitioner -> practice/topic",
        False,
    ),
    "patient_with": RelationshipPredicateRule(
        frozenset({("person", "topic")}),
        "patient -> condition/topic",
        False,
    ),
    "patient_of": RelationshipPredicateRule(
        frozenset({("person", "person"), ("person", "organization")}),
        "patient -> practitioner/organization",
        False,
    ),
    "practitioner_at": RelationshipPredicateRule(
        frozenset({("person", "organization")}),
        "practitioner -> organization",
        False,
    ),
    "cardholder_of": RelationshipPredicateRule(
        frozenset({("person", "object"), ("person", "organization")}),
        "cardholder -> card/provider",
        False,
    ),
    "creator_of": RelationshipPredicateRule(
        frozenset(
            {
                ("person", "project"),
                ("person", "object"),
                ("person", "tool"),
                ("person", "topic"),
                ("agent", "project"),
                ("agent", "object"),
                ("agent", "tool"),
                ("agent", "topic"),
                ("organization", "project"),
                ("organization", "object"),
                ("organization", "tool"),
                ("organization", "topic"),
            }
        ),
        "creator -> created entity",
        False,
    ),
    "prospective_buyer_of": RelationshipPredicateRule(
        frozenset(
            {
                ("person", "object"),
                ("person", "tool"),
                ("agent", "object"),
                ("agent", "tool"),
                ("organization", "object"),
                ("organization", "tool"),
            }
        ),
        "prospective buyer -> object/tool",
        False,
    ),
    "renter_of": RelationshipPredicateRule(
        frozenset(
            {
                ("person", "object"),
                ("person", "tool"),
                ("person", "place"),
                ("agent", "object"),
                ("agent", "tool"),
                ("agent", "place"),
                ("organization", "object"),
                ("organization", "tool"),
                ("organization", "place"),
            }
        ),
        "renter -> rented object/tool/place",
        False,
    ),
    "traveling_to": RelationshipPredicateRule(
        frozenset({("person", "place"), ("agent", "place")}),
        "traveler -> destination place",
        False,
    ),
    "visits_regularly": RelationshipPredicateRule(
        frozenset({("person", "place"), ("agent", "place")}),
        "visitor -> place",
        False,
    ),
    "participant_in": RelationshipPredicateRule(
        frozenset(
            {
                ("person", "event"),
                ("person", "group"),
                ("agent", "event"),
                ("agent", "group"),
                ("organization", "event"),
            }
        ),
        "participant -> event/group",
        False,
    ),
    "hosted_by": RelationshipPredicateRule(
        frozenset({("event", "person"), ("event", "organization"), ("event", "group")}),
        "event -> host",
        False,
    ),
}


def get_symmetric_relationship_predicates() -> frozenset[str]:
    """Return predicate names that must be queried direction-agnostically."""
    return frozenset(predicate for predicate, rule in _RELATIONSHIP_PREDICATE_ALLOWLIST.items() if rule.symmetric)


def _relationship_entity_type(entity: MemoryEntity) -> str:
    return str(getattr(entity, "entity_type", "") or "").strip().lower()


def _relationship_entity_label(entity: MemoryEntity) -> str:
    name = getattr(entity, "name", None) or str(getattr(entity, "id", "unknown"))
    return f"{name}({_relationship_entity_type(entity)}:{getattr(entity, 'id', 'unknown')})"


def _place_hierarchy_rank(entity: MemoryEntity) -> int | None:
    if _relationship_entity_type(entity) != "place":
        return None

    data = getattr(entity, "data", None)
    if isinstance(data, dict):
        for key in ("place_kind", "place_type", "kind", "type", "category"):
            value = data.get(key)
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in _PLACE_KIND_RANKS:
                    return _PLACE_KIND_RANKS[normalized]

    name = str(getattr(entity, "name", "") or "").strip().lower()
    return _PLACE_NAME_RANK_HINTS.get(name)


def _should_swap_hierarchical_direction(
    predicate: str,
    from_entity: MemoryEntity,
    to_entity: MemoryEntity,
) -> bool:
    if predicate not in {"located_in", "part_of"}:
        return False

    from_rank = _place_hierarchy_rank(from_entity)
    to_rank = _place_hierarchy_rank(to_entity)
    if from_rank is None or to_rank is None or from_rank == to_rank:
        return False

    return from_rank > to_rank


def _validate_relationship_classification(
    from_entity: MemoryEntity,
    to_entity: MemoryEntity,
    predicate: str,
    confidence: float,
) -> ValidationResult:
    """Validate and canonicalize a Sonnet relationship classification."""
    normalized_predicate = str(predicate or "").strip().lower()
    rule = _RELATIONSHIP_PREDICATE_ALLOWLIST.get(normalized_predicate)
    if rule is None:
        logger.warning(
            "Rejected Sonnet relationship classification: unknown predicate=%r from=%s to=%s confidence=%.3f",
            predicate,
            _relationship_entity_label(from_entity),
            _relationship_entity_label(to_entity),
            confidence,
        )
        return ValidationResult(False, None, None, normalized_predicate, f"unknown predicate: {predicate!r}")

    from_type = _relationship_entity_type(from_entity)
    to_type = _relationship_entity_type(to_entity)
    direct_pair = (from_type, to_type)
    swapped_pair = (to_type, from_type)

    if rule.symmetric:
        if direct_pair not in rule.allowed_pairs and swapped_pair not in rule.allowed_pairs:
            reason = f"{normalized_predicate} not allowed for {from_type}->{to_type}"
            return ValidationResult(False, None, None, normalized_predicate, reason)
        ordered = sorted((from_entity.id, to_entity.id), key=str)
        reason = "valid symmetric relationship"
        if ordered[0] != from_entity.id:
            reason = "canonical symmetric ordering by entity id"
        return ValidationResult(True, ordered[0], ordered[1], normalized_predicate, reason)

    if direct_pair in rule.allowed_pairs:
        if swapped_pair in rule.allowed_pairs and _should_swap_hierarchical_direction(
            normalized_predicate,
            from_entity,
            to_entity,
        ):
            reason = f"swapped inverted hierarchy; expected {rule.canonical_direction}"
            return ValidationResult(True, to_entity.id, from_entity.id, normalized_predicate, reason)
        return ValidationResult(True, from_entity.id, to_entity.id, normalized_predicate, "valid direction")

    if swapped_pair in rule.allowed_pairs:
        reason = f"swapped inverted direction; expected {rule.canonical_direction}"
        return ValidationResult(True, to_entity.id, from_entity.id, normalized_predicate, reason)

    reason = f"{normalized_predicate} not allowed for {from_type}->{to_type} or {to_type}->{from_type}"
    return ValidationResult(False, None, None, normalized_predicate, reason)


async def _relationship_entity_for_pair(
    session: AsyncSession,
    pair: dict[str, Any],
    side: Literal["a", "b"],
) -> MemoryEntity | None:
    embedded = pair.get(f"entity_{side}")
    if embedded is not None:
        return cast(MemoryEntity, embedded)
    return await session.get(MemoryEntity, pair[f"{side}_id"])


async def _find_cooccurring_entity_pairs(session: AsyncSession) -> list[dict[str, Any]]:
    """Return entity pairs with ≥ N shared traces and no existing active relationship.

    The discovery SQL has two structural guarantees that lock the producer
    onto strongest-fresh signal:

    1. ``ORDER BY cnt DESC, entity_a, entity_b`` — without this, pgsql returns any
       ``LIMIT N`` rows of the GROUP BY, which in practice is the same
       low-count pairs every run, leaving stronger fresh pairs undiscovered.
    2. ``NOT EXISTS`` filter against active ``entity_relationships`` —
       previously enforced in Python after fetching all candidates.
       Pushing it into SQL frees the ``LIMIT`` slots to land on truly
       fresh pairs. Diagnostic 2026-05-18: 423/475 qualifying pairs were
       unclassified (89%) because the Python filter kept eating the
       ``LIMIT 10`` results.
    """
    threshold = cognitive_config.get_int("cooccurrence_threshold")
    limit = cognitive_config.get_int("pair_limit")
    result = await session.execute(
        text("""
            WITH co AS (
                SELECT a.entity_id AS entity_a,
                       b.entity_id AS entity_b,
                       COUNT(*)    AS cnt
                FROM memory_entity_links a
                JOIN memory_entity_links b
                  ON a.memory_id = b.memory_id
                 AND a.entity_id < b.entity_id
                GROUP BY a.entity_id, b.entity_id
                HAVING COUNT(*) >= :threshold
            )
            SELECT entity_a, entity_b, cnt
            FROM co
            WHERE NOT EXISTS (
                SELECT 1 FROM entity_relationships er
                WHERE er.valid_until IS NULL
                  AND ((er.from_entity_id = co.entity_a AND er.to_entity_id = co.entity_b)
                    OR (er.from_entity_id = co.entity_b AND er.to_entity_id = co.entity_a))
            )
            ORDER BY cnt DESC, entity_a, entity_b
            LIMIT :pair_limit
        """),
        {"threshold": threshold, "pair_limit": limit},
    )
    cooccurrence_rows = list(result)
    if not cooccurrence_rows:
        return []

    # 1. Batch-fetch all active relationships for every entity with one query
    all_entity_ids = set()
    for row in cooccurrence_rows:
        all_entity_ids.add(row.entity_a)
        all_entity_ids.add(row.entity_b)
    existing_rels_result = await session.execute(
        select(EntityRelationship).where(
            EntityRelationship.valid_until.is_(None),
            or_(
                EntityRelationship.from_entity_id.in_(list(all_entity_ids)),
                EntityRelationship.to_entity_id.in_(list(all_entity_ids)),
            ),
        )
    )
    existing_rels = list(existing_rels_result.scalars().all())

    related_entity_ids = set(all_entity_ids)
    for rel in existing_rels:
        related_entity_ids.add(rel.from_entity_id)
        related_entity_ids.add(rel.to_entity_id)

    # 2. Batch-fetch entity names with one query, including relationship counterparties
    entity_result = await session.execute(
        select(MemoryEntity.id, MemoryEntity.name).where(MemoryEntity.id.in_(list(related_entity_ids)))
    )
    entity_map: dict[Any, str] = {entity_id: name for entity_id, name in entity_result.all()}

    rels_by_entity: dict[Any, list[dict[str, Any]]] = {entity_id: [] for entity_id in all_entity_ids}
    existing_rel_set: set[tuple[Any, Any]] = set()
    for rel in existing_rels:
        existing_rel_set.add((rel.from_entity_id, rel.to_entity_id))
        existing_rel_set.add((rel.to_entity_id, rel.from_entity_id))
        for entity_id, other_id in ((rel.from_entity_id, rel.to_entity_id), (rel.to_entity_id, rel.from_entity_id)):
            if entity_id not in rels_by_entity:
                continue
            other_entity = entity_map.get(other_id)
            rels_by_entity[entity_id].append(
                {
                    "type": rel.relationship_type,
                    "other_name": other_entity if other_entity else str(other_id),
                    "confidence": float(rel.confidence or 0.0),
                    "id": str(rel.id),
                }
            )

    # 3. Filter locally, then per-pair shared traces query
    pairs: list[dict[str, Any]] = []
    for row in cooccurrence_rows:
        a_id, b_id = row.entity_a, row.entity_b
        if (a_id, b_id) in existing_rel_set or (b_id, a_id) in existing_rel_set:
            continue
        name_a = entity_map.get(a_id)
        name_b = entity_map.get(b_id)
        if not name_a or not name_b:
            continue

        # Self-join ensures both entities are linked to the same trace (not either/or)
        shared_traces_sql = text("""
            SELECT mt.content, mt.created_at
            FROM memory_traces mt
            JOIN memory_entity_links la ON la.memory_id = mt.id AND la.entity_id = :ea
            JOIN memory_entity_links lb ON lb.memory_id = mt.id AND lb.entity_id = :eb
            WHERE mt.invalidated_at IS NULL
            ORDER BY mt.created_at ASC
            LIMIT 10
        """)
        shared_result = await session.execute(shared_traces_sql, {"ea": a_id, "eb": b_id})
        shared_rows = list(shared_result)
        pairs.append(
            {
                "a_id": a_id,
                "b_id": b_id,
                "name_a": name_a,
                "name_b": name_b,
                "shared_contents": [r.content for r in shared_rows],
                "earliest_date": shared_rows[0].created_at if shared_rows else utc_now(),
                "existing_rels_a": rels_by_entity.get(a_id, []),
                "existing_rels_b": rels_by_entity.get(b_id, []),
            }
        )
    return pairs


def _format_existing_relationships(rels: list[dict[str, Any]]) -> str:
    if not rels:
        return "(none)"
    return "\n".join(
        f"- {rel['type']} -> {rel['other_name']} (conf={float(rel.get('confidence', 0.0)):.2f}, id={rel['id']})"
        for rel in rels
    )


_DIRECTION_EXAMPLES = [
    "Misha (person) user_of ExampleTool (tool) — NOT ExampleTool user_of Misha",
    "Riga (place) hometown_of Katya (person) — NOT Katya hometown_of Riga",
    "Corfu (place) part_of Greece (place) — smaller place part_of larger place",
    "Olga (person) owner_of щенки (pet) — NOT щенки owner_of Olga",
    "Misha (person) employee_of Examplecorp (organization) — NOT Examplecorp employee_of Misha",
    "Katya (person) customer_of ExampleMarket (organization) — NOT ExampleMarket customer_of Katya",
]


def _render_allowlist_for_prompt() -> str:
    """Generate the canonical predicate + direction list for the classifier prompt.

    Pulled dynamically from ``_RELATIONSHIP_PREDICATE_ALLOWLIST`` so future
    allowlist expansions automatically reach Sonnet without a producer-code
    change. Each line has the shape ``predicate — direction_label (symmetric?)``.
    """
    lines: list[str] = []
    for predicate, rule in _RELATIONSHIP_PREDICATE_ALLOWLIST.items():
        symmetry = " (symmetric)" if rule.symmetric else ""
        lines.append(f"- {predicate} — {rule.canonical_direction}{symmetry}")
    return "\n".join(lines)


async def _classify_pair_relationship(pair: dict[str, Any]) -> dict[str, Any] | None:
    """Ask Sonnet to classify the relationship between a pair of entities.

    Prompt embeds the full ``_RELATIONSHIP_PREDICATE_ALLOWLIST`` so Sonnet
    knows exactly which predicates are accepted and in which direction. The
    validator (added in PR #153) does defense-in-depth on the output; this
    prompt change attacks the root cause — Sonnet was making up predicates
    (transitioning_to_member_of, calibration_reference_for, ...) and emitting
    inverted directions because the prior prompt only listed 4 example
    predicates with no direction guidance.

    Sonnet is explicitly told to return ``"no_relationship"`` (not just
    ``null``) when none of the allowlisted predicates fit. ``null`` is still
    accepted for backward compat with the existing parser at the bottom.
    """
    existing_a = _format_existing_relationships(pair.get("existing_rels_a", []))
    existing_b = _format_existing_relationships(pair.get("existing_rels_b", []))
    allowlist_block = _render_allowlist_for_prompt()
    direction_examples = "\n".join(f"  • {ex}" for ex in _DIRECTION_EXAMPLES)
    prompt = (
        f"Based on these memories, what is the relationship between "
        f"{pair['name_a']} and {pair['name_b']}?\n\n"
        f"Memories mentioning both:\n" + "\n".join(f"- {t}" for t in pair["shared_contents"]) + "\n\n"
        f"Existing relationships for A:\n{existing_a}\n\n"
        f"Existing relationships for B:\n{existing_b}\n\n"
        "Allowed relationship_type values (use ONLY these — emit any other value "
        "and the result is rejected at validation time):\n"
        f"{allowlist_block}\n\n"
        "Direction examples (the FIRST entity in the JSON is the FROM side):\n"
        f"{direction_examples}\n\n"
        "Rules:\n"
        '- Pick ONE predicate from the list above. Do not combine with "/", do not '
        "invent new predicates, do not paraphrase.\n"
        "- The direction must match the predicate's allowed entity-type pairs.\n"
        "- Symmetric predicates (partner_of, family_of, colleague_of, friend_of, "
        "  same_person, alternative_to, competitor_of, companion_of, complement_of) "
        "  can have either order.\n"
        "- If none of the listed predicates fits the actual relationship, return "
        '`{"relationship_type": "no_relationship"}` (NOT a made-up predicate). '
        "  The validator rejects unknown predicates anyway; reject yourself first to "
        "  save the round-trip.\n\n"
        "If the new relationship conflicts with existing relationships, choose conflict_action: "
        "none, replace, same_person, or sequential. Include conflicts_with relationship IDs when applicable.\n\n"
        'Return JSON: {"relationship_type": "<one of the allowed values OR no_relationship>",'
        ' "description": "brief description", "confidence": 0.0-1.0,'
        ' "conflict_action": "none|replace|same_person|sequential", "conflicts_with": []}\n'
        "Return null if no clear relationship (equivalent to no_relationship)."
    )
    # JSON-extraction call — disable adaptive thinking (see _call_sonnet docstring).
    response_text = await _call_sonnet_text(
        prompt, f"pattern-{pair['name_a']}-{pair['name_b']}", thinking_mode="disabled"
    )
    if not response_text or response_text.strip() == "null":
        return None
    text = response_text.strip()
    # Strip markdown fence if present (Sonnet sometimes wraps JSON in ```json ... ```)
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse relationship JSON for %s-%s: %.200s", pair["name_a"], pair["name_b"], text)
        return None
    if not isinstance(data, dict) or not data.get("relationship_type"):
        return None
    # ``no_relationship`` is the self-reject token — treat as null. Validator
    # would reject downstream anyway; short-circuit so the run stat reads
    # ``Sonnet returned null`` instead of an audit-noisy validation rejection.
    if str(data.get("relationship_type", "")).strip().lower() == "no_relationship":
        return None
    return data


async def _create_relationship_from_classification(
    session: AsyncSession, pair: dict[str, Any], classification: dict[str, Any]
) -> bool:
    """Persist a relationship action; return True only when an EntityRelationship is inserted."""
    try:
        conflict_action = ConflictAction(classification.get("conflict_action") or ConflictAction.NONE)
    except ValueError:
        logger.info("Unknown conflict_action %r; defaulting to none", classification.get("conflict_action"))
        conflict_action = ConflictAction.NONE

    try:
        confidence = float(classification.get("confidence", 0.7))
    except (TypeError, ValueError):
        logger.warning("Skipping relationship classification with invalid confidence: %s", classification)
        return False

    min_confidence = float(cognitive_config.get("relationship_min_confidence"))
    if confidence < min_confidence:
        logger.info(
            "Rejected relationship classification below confidence floor: from=%s to=%s type=%s "
            "confidence=%.3f floor=%.3f",
            pair.get("name_a", pair.get("a_id")),
            pair.get("name_b", pair.get("b_id")),
            classification.get("relationship_type", ""),
            confidence,
            min_confidence,
        )
        return False

    from_entity = await _relationship_entity_for_pair(session, pair, "a")
    to_entity = await _relationship_entity_for_pair(session, pair, "b")
    if from_entity is None or to_entity is None:
        logger.warning(
            "Skipping relationship classification for %s-%s: missing entity row(s); sonnet_output=%s",
            pair.get("name_a", pair.get("a_id")),
            pair.get("name_b", pair.get("b_id")),
            classification,
        )
        return False

    validation = _validate_relationship_classification(
        from_entity,
        to_entity,
        classification.get("relationship_type", ""),
        confidence,
    )
    if not validation.ok or validation.corrected_from_id is None or validation.corrected_to_id is None:
        logger.warning(
            "Skipping relationship classification for %s-%s: %s; sonnet_output=%s",
            pair.get("name_a", pair.get("a_id")),
            pair.get("name_b", pair.get("b_id")),
            validation.reason,
            classification,
        )
        return False

    allowed_conflict_ids: set[_uuid.UUID] = set()
    for rel in pair.get("existing_rels_a", []) + pair.get("existing_rels_b", []):
        try:
            raw_id = rel["id"]
            allowed_conflict_ids.add(_uuid.UUID(raw_id) if isinstance(raw_id, str) else raw_id)
        except (KeyError, TypeError, ValueError):
            continue

    conflicts_with = []
    for raw_id in classification.get("conflicts_with") or []:
        try:
            conflicts_with.append(_uuid.UUID(str(raw_id)))
        except ValueError:
            logger.debug("Skipping invalid relationship conflict id: %s", raw_id)

    if conflict_action in {ConflictAction.REPLACE, ConflictAction.SEQUENTIAL}:
        dropped_ids = [rel_id for rel_id in conflicts_with if rel_id not in allowed_conflict_ids]
        if dropped_ids:
            logger.info("Dropped unscoped relationship conflict ids: %s", [str(rel_id) for rel_id in dropped_ids])
        conflicts_with = [rel_id for rel_id in conflicts_with if rel_id in allowed_conflict_ids]

    if conflict_action == ConflictAction.SAME_PERSON:
        session.add(
            EntityMergeCandidate(
                entity_a_id=validation.corrected_from_id,
                entity_b_id=validation.corrected_to_id,
                source_consolidation_run_id=pair.get("source_consolidation_run_id"),
                confidence=confidence,
                reason="L3 classified as same person",
                status=MergeCandidateStatus.PENDING,
            )
        )
        await session.flush()
        return False

    rel = EntityRelationship(
        from_entity_id=validation.corrected_from_id,
        to_entity_id=validation.corrected_to_id,
        relationship_type=validation.predicate,
        description=classification.get("description", ""),
        confidence=confidence,
        valid_from=pair["earliest_date"],
    )
    if conflict_action == ConflictAction.REPLACE and conflicts_with:
        await session.execute(
            update(EntityRelationship)
            .where(EntityRelationship.id.in_(conflicts_with), EntityRelationship.valid_until.is_(None))
            .values(valid_until=utc_now())
        )
    elif conflict_action == ConflictAction.SEQUENTIAL and conflicts_with:
        await session.execute(
            update(EntityRelationship)
            .where(EntityRelationship.id.in_(conflicts_with), EntityRelationship.valid_until.is_(None))
            .values(valid_until=rel.valid_from or utc_now())
        )
    session.add(rel)
    await session.flush()
    return True


async def _promote_patterns(session: AsyncSession) -> int:
    """Find entity pairs co-occurring in ≥N traces and promote them to relationships."""
    try:
        pairs = await _find_cooccurring_entity_pairs(session)
        if not pairs:
            logger.info("No co-occurring entity pairs above threshold")
            return 0

        run_id = await _current_consolidation_run_id(session)
        for pair in pairs:
            pair["source_consolidation_run_id"] = run_id

        logger.info(
            "Found %d co-occurring pairs: %s",
            len(pairs),
            ", ".join(f"{p['name_a']}+{p['name_b']}({len(p['shared_contents'])})" for p in pairs),
        )

        classifications = await asyncio.gather(
            *(_classify_pair_relationship(p) for p in pairs),
            return_exceptions=True,
        )

        count = 0
        for pair, result in zip(pairs, classifications, strict=False):
            if isinstance(result, BaseException):
                logger.warning("Pattern classify failed for %s-%s: %s", pair["name_a"], pair["name_b"], result)
                continue
            if result is None:
                logger.info(
                    "Sonnet returned null for %s-%s (no clear relationship yet)", pair["name_a"], pair["name_b"]
                )
                continue
            logger.info("Classified %s-%s: %s", pair["name_a"], pair["name_b"], result)
            try:
                created = await _create_relationship_from_classification(session, pair, result)
                if created:
                    count += 1
            except Exception:
                logger.warning(
                    "Failed to persist relationship for %s-%s",
                    pair["name_a"],
                    pair["name_b"],
                    exc_info=True,
                )
        return count
    except Exception as e:
        logger.error("Pattern promotion failed", exc_info=True)
        raise ConsolidationStepError("patterns_promoted", e) from e


async def _current_consolidation_run_id(session: AsyncSession) -> _uuid.UUID | None:
    result = await session.execute(
        select(ConsolidationRun.id)
        .where(ConsolidationRun.level == 3, ConsolidationRun.status == "running")
        .order_by(ConsolidationRun.started_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


# ── Step 7: Social norms (Sonnet) ─────────────────────────


async def _synthesize_norms_for_context(social_ctx: SocialContext, exp_text: str) -> str | None:
    """Sonnet call for a single social context — returns updated norms text.

    If current norms exist, refine them rather than rewriting from scratch.
    """
    prompt = (
        "Based on these social experiences in this chat context, describe the "
        "social norms that have emerged. Keep it concise (2-4 short bullet points). "
        "Focus on what the agent should understand about how to behave in this chat.\n\n"
        "If previous norms exist, refine them — don't rewrite from scratch.\n\n"
        f"Chat: {social_ctx.context_name or 'unnamed'} ({social_ctx.context_type})\n"
        f"Current norms: {social_ctx.social_norms or '(none yet)'}\n\n"
        f"Recent experiences:\n{exp_text}\n\n"
        "Write updated norms:"
    )
    return await _call_sonnet_text(prompt, f"social-norms-{social_ctx.id}")


async def _update_social_norms(session: AsyncSession) -> int:
    """Synthesize social norms from accumulated experiences.

    Only synthesizes when a context has >= social_norm_threshold NEW experiences
    since its last norm update. Matches neuroscience: norms form from pattern
    repetition, not single events.
    """
    try:
        result = await session.execute(
            select(SocialContext).where(SocialContext.last_active_at >= utc_now() - timedelta(days=7))
        )
        contexts = list(result.scalars().all())
        if not contexts:
            logger.info("Social norms: no active social contexts")
            return 0

        threshold = cognitive_config.get_int("social_norm_threshold")
        prepared: list[tuple[SocialContext, str]] = []
        below_threshold = 0
        for social_ctx in contexts:
            # Count experiences since last norm update (or all-time if never updated)
            since = social_ctx.updated_at if social_ctx.social_norms else None
            count_query = select(func.count(SocialExperience.id)).where(
                SocialExperience.social_context_id == social_ctx.id,
            )
            if since:
                count_query = count_query.where(SocialExperience.created_at > since)
            new_count = (await session.execute(count_query)).scalar() or 0

            if new_count < threshold:
                below_threshold += 1
                continue

            exp_result = await session.execute(
                select(SocialExperience)
                .where(SocialExperience.social_context_id == social_ctx.id)
                .order_by(SocialExperience.created_at.desc())
                .limit(10)
            )
            experiences = list(exp_result.scalars().all())
            exp_lines = []
            for e in experiences:
                line = f"- [{e.signal_type or 'signal'}] {e.experience}"
                if e.learned_norm:
                    line += f" → tentative: {e.learned_norm}"
                exp_lines.append(line)
            exp_text = "\n".join(exp_lines)
            prepared.append((social_ctx, exp_text))

        if not prepared:
            logger.info(
                "Social norms: %d contexts, %d below threshold (< %d new experiences), 0 synthesized",
                len(contexts),
                below_threshold,
                threshold,
            )
            return 0

        logger.info("Social norms: synthesizing for %d contexts (threshold=%d)", len(prepared), threshold)

        results = await asyncio.gather(
            *(_synthesize_norms_for_context(sc, et) for sc, et in prepared),
            return_exceptions=True,
        )
        count = 0
        for (social_ctx, _), norms_text in zip(prepared, results, strict=False):
            if isinstance(norms_text, BaseException):
                logger.warning("Social norms update failed for context %s: %s", social_ctx.id, norms_text)
                continue
            if not norms_text:
                continue
            await session.execute(
                update(SocialContext).where(SocialContext.id == social_ctx.id).values(social_norms=norms_text.strip())
            )
            count += 1
        return count
    except Exception as e:
        logger.error("Social norms update failed", exc_info=True)
        raise ConsolidationStepError("social_updates", e) from e


# ── Entity merge sweep ────────────────────────────────────


async def _resolve_merge_chain(session: AsyncSession, entity: MemoryEntity) -> MemoryEntity:
    """Walk merged_into_id chain to terminal survivor. Cap depth to prevent cycles."""
    visited = {entity.id}
    current = entity
    for _ in range(MERGE_CHAIN_MAX_DEPTH):
        if current.merged_into_id is None:
            return current
        if current.merged_into_id in visited:
            raise RuntimeError(f"merge cycle detected at {current.id}")
        visited.add(current.merged_into_id)
        next_ent = await session.get(MemoryEntity, current.merged_into_id)
        if next_ent is None:
            return current
        current = next_ent
    return current


async def _lock_entities_for_merge(
    session: AsyncSession, entity_ids: set[_uuid.UUID]
) -> dict[_uuid.UUID, MemoryEntity]:
    ordered_ids = sorted(entity_ids, key=str)
    # SET LOCAL is transaction-scoped, so this 30s cap covers every subsequent
    # lock acquisition in the outer L3 transaction (not just this SELECT). That
    # is intentional defense-in-depth: any future regression that reintroduces
    # a separate-session pattern fails fast instead of hanging the nightly run
    # for hours. L3 phases should never hold a contended lock anywhere near 30s.
    await session.execute(text("SET LOCAL lock_timeout = '30s'"))
    result = await session.execute(
        select(MemoryEntity).where(MemoryEntity.id.in_(ordered_ids)).order_by(MemoryEntity.id).with_for_update()
    )
    return {entity.id: entity for entity in result.scalars().all()}


async def _dedup_and_transfer_relationships(
    session: AsyncSession,
    survivor_id: _uuid.UUID,
    absorbed_id: _uuid.UUID,
) -> None:
    """Move active absorbed relationships to survivor, expiring canonical duplicates."""
    result = await session.execute(
        select(EntityRelationship).where(
            EntityRelationship.valid_until.is_(None),
            or_(
                EntityRelationship.from_entity_id == absorbed_id,
                EntityRelationship.to_entity_id == absorbed_id,
            ),
        )
    )
    active_rels = list(result.scalars().all())
    now = utc_now()
    existing_keys: set[tuple[str, _uuid.UUID]] = set()
    rel_keys_to_check: set[tuple[str, _uuid.UUID]] = set()
    for rel in active_rels:
        other_id = rel.to_entity_id if rel.from_entity_id == absorbed_id else rel.from_entity_id
        if other_id != survivor_id:
            rel_keys_to_check.add((rel.relationship_type, other_id))

    if rel_keys_to_check:
        other_ids = {other_id for _, other_id in rel_keys_to_check}
        rel_types = {rel_type for rel_type, _ in rel_keys_to_check}
        existing_result = await session.execute(
            select(
                EntityRelationship.relationship_type,
                EntityRelationship.from_entity_id,
                EntityRelationship.to_entity_id,
            ).where(
                EntityRelationship.valid_until.is_(None),
                EntityRelationship.relationship_type.in_(rel_types),
                or_(
                    (EntityRelationship.from_entity_id == survivor_id)
                    & (EntityRelationship.to_entity_id.in_(other_ids)),
                    (EntityRelationship.to_entity_id == survivor_id)
                    & (EntityRelationship.from_entity_id.in_(other_ids)),
                ),
            )
        )
        for rel_type, from_entity_id, to_entity_id in existing_result.all():
            other_id = to_entity_id if from_entity_id == survivor_id else from_entity_id
            existing_keys.add((rel_type, other_id))

    for rel in active_rels:
        other_id = rel.to_entity_id if rel.from_entity_id == absorbed_id else rel.from_entity_id
        if other_id == survivor_id:
            rel.valid_until = now
            continue
        if (rel.relationship_type, other_id) in existing_keys:
            rel.valid_until = now
        elif rel.from_entity_id == absorbed_id:
            rel.from_entity_id = survivor_id
        else:
            rel.to_entity_id = survivor_id

    await session.execute(
        text("""
            INSERT INTO memory_entity_links (memory_id, entity_id, role)
            SELECT memory_id, :survivor_id, role
            FROM memory_entity_links
            WHERE entity_id = :absorbed_id
            ON CONFLICT (memory_id, entity_id, role) DO NOTHING
        """),
        {"survivor_id": survivor_id, "absorbed_id": absorbed_id},
    )
    await session.execute(
        text("DELETE FROM memory_entity_links WHERE entity_id = :absorbed_id"),
        {"absorbed_id": absorbed_id},
    )


async def _execute_merge(session: AsyncSession, candidate: EntityMergeCandidate) -> None:
    """Execute a single entity merge."""
    locked = await _lock_entities_for_merge(session, {candidate.entity_a_id, candidate.entity_b_id})
    a = locked.get(candidate.entity_a_id)
    b = locked.get(candidate.entity_b_id)
    now = utc_now()
    if not a or not b:
        candidate.status = MergeCandidateStatus.REJECTED
        candidate.reviewed_at = now
        candidate.reason = f"{candidate.reason or ''} — entity not found".strip()
        return

    a = await _resolve_merge_chain(session, a)
    b = await _resolve_merge_chain(session, b)
    terminal_ids = {a.id, b.id}
    if not terminal_ids.issubset(locked):
        locked.update(await _lock_entities_for_merge(session, terminal_ids))
        a = locked.get(a.id, a)
        b = locked.get(b.id, b)
        a = await _resolve_merge_chain(session, a)
        b = await _resolve_merge_chain(session, b)
    if a.id == b.id:
        candidate.status = MergeCandidateStatus.MERGED
        candidate.reviewed_at = now
        return
    if a.telegram_user_id and b.telegram_user_id and a.telegram_user_id != b.telegram_user_id:
        candidate.status = MergeCandidateStatus.REJECTED
        candidate.reason = (candidate.reason or "") + " — distinct telegram_user_ids"
        candidate.reviewed_at = now
        return

    try:
        survivor = identity_merge_survivor(a, b)
    except ValueError as exc:
        candidate.status = MergeCandidateStatus.REJECTED
        candidate.reason = (candidate.reason or "") + f" — {exc}"
        candidate.reviewed_at = now
        return
    if survivor is not None:
        absorbed = b if survivor is a else a
    else:
        survivor, absorbed = (a, b) if (a.mention_count or 0) >= (b.mention_count or 0) else (b, a)
        if absorbed.telegram_user_id and not survivor.telegram_user_id:
            survivor, absorbed = absorbed, survivor

    aliases = list(survivor.aliases or [])
    if absorbed.name != survivor.name and absorbed.name not in aliases:
        aliases.append(absorbed.name)
    for alias in absorbed.aliases or []:
        if alias != survivor.name and alias not in aliases:
            aliases.append(alias)
    survivor.aliases = aliases

    absorbed.merged_into_id = survivor.id
    absorbed.merged_at = now
    absorbed.merge_reason = candidate.reason
    await session.flush()

    await _dedup_and_transfer_relationships(session, survivor.id, absorbed.id)

    candidate.status = MergeCandidateStatus.MERGED
    candidate.reviewed_at = now
    await session.flush()


async def _merge_entities(session: AsyncSession) -> int:
    """Process pending entity merge candidates."""
    try:
        result = await session.execute(
            select(EntityMergeCandidate.id)
            .where(EntityMergeCandidate.status == MergeCandidateStatus.PENDING)
            .order_by(EntityMergeCandidate.created_at)
            .limit(MERGE_CANDIDATE_BATCH_LIMIT)
        )
        candidate_ids = list(result.scalars().all())
        count = 0
        for candidate_id in candidate_ids:
            # Guard outside the savepoint: skipping inside `begin_nested()` would
            # still cost a SAVEPOINT + RELEASE round-trip per skipped candidate.
            candidate = await session.get(EntityMergeCandidate, candidate_id)
            if candidate is None or candidate.status != MergeCandidateStatus.PENDING:
                continue
            try:
                async with session.begin_nested():
                    await _execute_merge(session, candidate)
                    if candidate.status == MergeCandidateStatus.MERGED:
                        count += 1
            except Exception:
                # Let the savepoint context manager own the rollback. Explicit
                # rollback inside the with-block double-finalizes the savepoint
                # in SQLAlchemy 2.x and raises InvalidRequestError when __aexit__
                # then tries to commit a CLOSED savepoint.
                logger.warning("Merge candidate %s failed", candidate_id, exc_info=True)
                # Re-fetch in the outer transaction since the savepoint discarded
                # any in-memory mutations on the candidate row.
                candidate = await session.get(EntityMergeCandidate, candidate_id)
                if candidate is not None:
                    candidate.status = MergeCandidateStatus.REJECTED
                    candidate.reviewed_at = utc_now()
        return count
    except Exception as e:
        logger.error("Entity merge sweep failed", exc_info=True)
        raise ConsolidationStepError("entities_merged", e) from e


# ── Step 8+9: Self-reflection (Sonnet) ────────────────────


async def _self_reflect(session: AsyncSession) -> int:
    """5-dimension self-reflection creating memory traces."""
    try:
        cutoff = utc_now() - timedelta(days=7)
        result = await session.execute(
            select(MemoryTrace)
            .where(
                MemoryTrace.created_at >= cutoff,
                MemoryTrace.invalidated_at.is_(None),
                MemoryTrace.memory_type != "procedural",  # Phase A: keep candidates out of self-reflection
            )
            .order_by(MemoryTrace.created_at.desc())
            .limit(50)
        )
        recent = list(result.scalars().all())
        if not recent:
            return 0

        by_type: dict[str, list[str]] = {}
        for t in recent:
            by_type.setdefault(t.memory_type, []).append(t.content)
        grouped_text = "\n".join(
            f"[{mtype}]\n" + "\n".join(f"  - {c}" for c in items[:5]) for mtype, items in by_type.items()
        )

        # M2.5 #3 — active-form questions (Goldsmith). Passive-form questions
        # trigger external attribution + observer mode; active-form triggers
        # agency + effort attribution. One-line prompt change, disproportionate
        # behavioural effect — see roadmap §M2.5.
        prompt = (
            "Reflect on your recent interactions (last 7 days). Analyze these memories:\n\n"
            f"{grouped_text}\n\n"
            "Provide observations on 5 dimensions. Active form - focus on effort and agency:\n"
            "1. INTERACTION QUALITY: Are you doing your best to engage well? What worked, what didn't?\n"
            "2. BEHAVIORAL PATTERNS: Are you doing your best on the things you do well? Where did effort fall short?\n"
            "3. PER-PERSON CALIBRATION: Are you doing your best to adapt tone/style for each person?\n"
            "4. GOAL ALIGNMENT: Are you doing your best to help effectively? What capabilities are you missing?\n"
            "5. SOCIAL COMFORT: Are you doing your best to stay grounded in each chat context?\n\n"
            "Return JSON array of 5 observations:\n"
            '[{"dimension": "interaction_quality", "observation": "...", "significance": 3}, ...]\n'
        )

        items = await _call_sonnet(prompt, "self-reflection")
        svc = CognitiveMemoryService(session)
        count = 0
        for item in items:
            try:
                obs = item.get("observation", "").strip()
                if not obs:
                    continue
                sig = max(1, min(5, int(item.get("significance", 3))))
                await svc.store_trace(
                    content=obs,
                    memory_type="self_reflection",
                    significance=sig,
                    source_type="consolidation",
                    # Sub-design C INC 1: stamp the trace origin. L3 self-reflection
                    # has no entity structure, so typed_metadata stays empty ({});
                    # only the origin is recorded. Flag-gated in store_trace.
                    source_path="l3_self_reflect",
                )
                count += 1
            except Exception:
                pass
        return count
    except Exception as e:
        logger.error("Self-reflection failed", exc_info=True)
        raise ConsolidationStepError("reflections", e) from e


# ── Significance reassessment ─────────────────────────────


async def _reassess_significance(session: AsyncSession) -> None:
    """Upgrade significance for frequently accessed traces."""
    try:
        await session.execute(
            update(MemoryTrace)
            .where(
                MemoryTrace.access_count >= 5,
                MemoryTrace.significance < 4,
                MemoryTrace.invalidated_at.is_(None),
            )
            .values(significance=4)
        )
    except Exception as e:
        logger.error("Significance reassessment failed", exc_info=True)
        raise ConsolidationStepError("reassessed", e) from e


# ── Pruning / growth controls ─────────────────────────────


async def _prune_silent_traces(session: AsyncSession) -> int:
    """Delete very-low-salience, old, non-critical traces.

    "Old" is two conditions since DB-437, not one: a trace must be past
    ``silent_age_days`` since creation AND have had no retrieval inside that
    same window. Retrieval is the strongest in-demand signal the read path
    produces, so deleting a recently-fetched trace destroys a memory that is
    demonstrably still in use.

    Explicitly deletes linked rows first — FK constraints don't cascade.

    Skip invalidators: ``memory_traces.invalidated_by`` is a self-FK
    (REFERENCES memory_traces(id) with no ON DELETE clause), so deleting
    a trace that still acts as the ``invalidated_by`` pointer for another
    trace raises ForeignKeyViolationError. The NOT IN exclusion below
    keeps the invalidation audit chain intact — preserved invalidators
    can be cleaned later once their dependents have been pruned through
    a separate policy.

    Symptom pre-fix (2026-05-18 L3, all 3 passes):
        ``update or delete on table "memory_traces" violates foreign key
        constraint "memory_traces_invalidated_by_fkey"``
    4119 prunable traces stayed unpruned every night because one row in
    the batch was an active invalidator.
    """
    try:
        threshold = cognitive_config.get("silent_salience_threshold")
        age_days = cognitive_config.get_int("silent_age_days")
        grace_days = cognitive_config.get_int("durable_supersede_grace_days")
        result = await session.execute(
            text("""
                WITH to_delete AS (
                    SELECT id FROM memory_traces
                    WHERE salience < :threshold
                      AND memory_type != 'procedural'
                      AND invalidated_at IS NULL
                      AND created_at < CAST(:now AS timestamptz) - make_interval(days => :age_days)
                      -- DB-437 recent-access guard: prunable means old AND silent.
                      -- Deliberately REUSES :age_days (no new tunable) — a trace
                      -- retrieved inside the same silence window is by definition not
                      -- silent, so deleting it destroys a memory the read path just
                      -- proved is in demand. NULL = never accessed = prunable as before.
                      AND (
                          last_accessed_at IS NULL
                          OR last_accessed_at < CAST(:now AS timestamptz) - make_interval(days => :age_days)
                      )
                      AND significance < 5
                      -- Spare durable identity facts (belt-and-suspenders — the decay
                      -- floor already keeps them above :threshold). The `invalidated_at
                      -- IS NULL` clause above already EXCLUDES invalidated traces from
                      -- pruning entirely (they persist as soft-deleted audit rows). The
                      -- escape hatch for a WRONG durable fact is invalidation itself — it
                      -- drops the fact from retrieval regardless of this spare — so
                      -- durable ≠ immortal.
                      AND COALESCE(typed_metadata->>'durability', '') <> 'durable'
                      -- Grace period (Codex #3): a trace Part C demoted to 'superseded'
                      -- is spared while it is still within :grace_days of superseded_at,
                      -- so a WRONG newer fact that superseded a correct older one can be
                      -- caught by contradiction-detection before the old value is pruned.
                      -- Past the grace window it prunes normally (bounded accumulation).
                      AND NOT (
                          typed_metadata->>'durability' = 'superseded'
                          AND typed_metadata->>'superseded_at' IS NOT NULL
                          AND (typed_metadata->>'superseded_at')::timestamptz
                              > CAST(:now AS timestamptz) - make_interval(days => :grace_days)
                      )
                      AND id NOT IN (
                          SELECT invalidated_by
                          FROM memory_traces
                          WHERE invalidated_by IS NOT NULL
                      )
                ),
                del_links AS (
                    DELETE FROM memory_entity_links WHERE memory_id IN (SELECT id FROM to_delete)
                ),
                del_rels AS (
                    DELETE FROM entity_relationships WHERE source_memory_id IN (SELECT id FROM to_delete)
                ),
                del_traces AS (
                    DELETE FROM memory_traces WHERE id IN (SELECT id FROM to_delete) RETURNING id
                )
                SELECT COUNT(*) FROM del_traces
            """),
            {"threshold": threshold, "age_days": age_days, "grace_days": grace_days, "now": utc_now()},
        )
        return result.scalar() or 0
    except Exception as e:
        logger.error("Prune silent traces failed", exc_info=True)
        raise ConsolidationStepError("pruned_traces", e) from e


async def _cleanup_old_runs(session: AsyncSession) -> int:
    """Delete old consolidation_runs records beyond retention window."""
    try:
        days = cognitive_config.get_int("run_retention_days")
        result = await session.execute(
            text("""
                DELETE FROM consolidation_runs
                WHERE completed_at < CAST(:now AS timestamptz) - make_interval(days => :days)
                  AND status IN ('completed', 'failed', 'deferred')
            """),
            {"days": days, "now": utc_now()},
        )
        return result.rowcount or 0  # type: ignore[attr-defined]
    except Exception as e:
        logger.error("Cleanup old runs failed", exc_info=True)
        raise ConsolidationStepError("runs_cleaned", e) from e


async def _cap_access_timestamps(session: AsyncSession) -> int:
    """Trim access_timestamps JSONB arrays to the configured cap."""
    try:
        cap = cognitive_config.get_int("timestamp_cap")
        result = await session.execute(
            text("""
                UPDATE memory_traces SET access_timestamps = (
                    SELECT jsonb_agg(elem) FROM (
                        SELECT elem FROM jsonb_array_elements(access_timestamps) elem
                        ORDER BY elem DESC LIMIT :cap
                    ) sub
                )
                WHERE jsonb_array_length(access_timestamps) > :cap
            """),
            {"cap": cap},
        )
        return result.rowcount or 0  # type: ignore[attr-defined]
    except Exception as e:
        logger.error("Cap access_timestamps failed", exc_info=True)
        raise ConsolidationStepError("timestamps_capped", e) from e


# ── Sonnet LLM helpers ────────────────────────────────────


async def _call_sonnet(
    prompt: str,
    span_name: str,
    *,
    thinking_mode: Literal["adaptive", "disabled"] = "disabled",
) -> list[dict[str, Any]]:
    """Call Sonnet and parse JSON array response.

    Defaults to ``thinking_mode="disabled"`` because every current caller
    (contradiction-detection, self-reflection) is a JSON-array extraction
    task — see scripts/verify_disable_thinking.py for the experiment that
    confirms adaptive thinking is a budget-tax with zero quality benefit
    on structured outputs.
    """
    raw = await _call_sonnet_text(prompt, span_name, thinking_mode=thinking_mode)
    return parse_json_array_response(raw) if raw else []


async def _call_sonnet_text(
    prompt: str,
    span_name: str,
    max_tokens: int = 8192,
    model: str | None = None,
    *,
    thinking_mode: Literal["adaptive", "disabled"] = "adaptive",
) -> str | None:
    """Call an LLM via the shared provider chain and return raw text response.

    Name kept historical ("sonnet") because every current caller defaults
    to Sonnet; renaming would churn 5+ call sites for no functional gain.
    The ``model`` parameter is preserved so future callers (e.g. Sub-design
    D Wave 2 producers) can route to Opus or another model when their
    output budget exceeds Sonnet's 8k ceiling.

    Default max_tokens=8192 prevents mid-JSON truncation on L3 prompts
    (contradictions, pattern promotion, social norms, self-reflection) that
    produce 5-15-entry JSON arrays.

    Default thinking_mode="adaptive" preserves the historical behavior for
    free-text callers (e.g. ``_synthesize_norms_for_context``).
    JSON-extraction callers should pass ``thinking_mode="disabled"`` — see
    ``_call_sonnet`` above for that path.
    """
    async with _get_sonnet_semaphore():
        model = model or get_model_roles().consolidation
        provider = get_provider_for_model(get_settings(), model)
        span = start_generation_span(f"consolidation/{span_name}", model)
        return await call_llm_text(
            provider,
            prompt,
            max_tokens=max_tokens,
            span=span,
            thinking_mode=thinking_mode,
            task_name=f"consolidation.{span_name}",
        )
