"""ArtifactPipeline — orchestrates typed-artifact production (M2 Wave 1).

Replaces the procedural step-list in current ``consolidation.py`` for the M2
artifact path. Wave 1 ships the orchestrator; Wave 2 plugs in the five
producers (EntitySchemaProducer, …); Wave 3 wires the consumer + cutover.

Key invariants (design §3.1, §3.2):

1. **Savepoint per producer.** A producer crash rolls back its rows only;
   the pipeline keeps going.
2. **No fallback to no-savepoint execution.** If ``session.begin_nested()``
   itself raises, we SKIP the producer (`pipeline.savepoint_setup_failed`
   log) — running without isolation can corrupt the run by partially
   writing artifacts that other producers later supersede on top of.
3. **Systemic-failure abort.** If >50 % of producers' savepoint setup
   fails, the pipeline aborts with `pipeline.savepoint_systemic_failure`
   — DB / transaction state is broken, investigation required.
4. **ChangesMadeEntry timing.** A producer's local entry buffer is merged
   into the run-aggregated list ONLY after its savepoint commits. Failed
   savepoint = entries discarded (mirrors the DB rollback).

Telemetry — emits the spans the locked contract already defines
(`memory/consolidate`, `memory/artifact-producer.<producer_name>`); does NOT
introduce new namespaces. Wave 3 will add `memory/consolidate-preview` and
friends per the contract update plan in design §9.
"""

from __future__ import annotations

import json
import logging
import time
import uuid as _uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.host_calls import utc_now
from kai_memory.artifacts.producer import ArtifactProducer
from kai_memory.artifacts.types import ArtifactType, ChangesMadeEntry

logger = logging.getLogger("app.services.memory.artifact_pipeline")


# Skipped-producer threshold — above this fraction of savepoint setup
# failures and the pipeline aborts on the assumption the DB session is
# wedged. 0.5 = "half the registered producers couldn't even open a
# savepoint".
_SYSTEMIC_FAILURE_THRESHOLD = 0.5


class ArtifactPipeline:
    """Per-run orchestrator. Construct once per L3 pass, register producers, run."""

    def __init__(
        self,
        session: AsyncSession,
        run_id: _uuid.UUID,
        *,
        phase: str = "synthesize",
        pass_number: int = 1,
        chat_id: int | None = None,
        topic_id: int | None = None,
        venue: str = "scheduler",
    ) -> None:
        self._session = session
        self._run_id = run_id
        self._phase = phase
        self._pass_number = pass_number
        # Span context — required attributes per locked contract §1.
        # Nightly L3 has no chat/topic; venue defaults to "scheduler".
        self._chat_id = chat_id
        self._topic_id = topic_id
        self._venue = venue

        self._producers: list[ArtifactProducer] = []
        self._changes: list[ChangesMadeEntry] = []
        self._producers_run = 0
        self._producers_failed = 0
        self._artifacts_by_type: dict[str, int] = {}

    # ── Public API ────────────────────────────────────────

    def register(self, producer: ArtifactProducer) -> None:
        """Register a producer to run on the next ``run_all()`` call."""
        self._producers.append(producer)

    @property
    def changes(self) -> list[ChangesMadeEntry]:
        """Aggregated entries from successfully-committed savepoints."""
        return list(self._changes)

    async def run_all(self) -> list[ChangesMadeEntry]:
        """Execute every registered producer in savepoint isolation.

        Returns the aggregated ``ChangesMadeEntry`` list (only entries from
        producers whose savepoints committed). Caller is expected to write
        this list to ``consolidation_runs.changes_made`` at end of L3.
        """
        started_at = time.monotonic()
        savepoint_setup_failures = 0
        total_producers = len(self._producers)

        try:
            for producer in self._producers:
                outcome = await self._run_one(producer)
                if outcome == "savepoint_failed":
                    savepoint_setup_failures += 1
                    # Systemic-failure tripwire — cancel the rest of the run.
                    if total_producers > 0 and savepoint_setup_failures / total_producers > _SYSTEMIC_FAILURE_THRESHOLD:
                        self._emit_systemic_failure_log(savepoint_setup_failures, total_producers)
                        break
        finally:
            duration_ms = int((time.monotonic() - started_at) * 1000)
            self._emit_pipeline_log(duration_ms)
        return list(self._changes)

    # ── Per-producer orchestration ─────────────────────────

    async def _run_one(self, producer: ArtifactProducer) -> str:
        """Run one producer inside its own savepoint.

        Returns one of:
          - ``"committed"``       — savepoint committed, entries merged
          - ``"rolled_back"``     — producer raised, savepoint rolled back
          - ``"savepoint_failed"`` — ``begin_nested()`` raised, producer skipped
        """
        producer_started = time.monotonic()
        producer_name = getattr(producer, "name", producer.__class__.__name__)
        producer_version = getattr(producer, "version", 1)

        # Step 1 — open savepoint. If this itself raises, skip the producer.
        try:
            savepoint = await self._session.begin_nested()
        except Exception:
            logger.warning(
                "pipeline.savepoint_setup_failed producer=%s run_id=%s",
                producer_name,
                self._run_id,
                exc_info=True,
            )
            self._emit_producer_log(
                producer=producer,
                subjects_discovered=0,
                artifacts_produced=0,
                duration_ms=int((time.monotonic() - producer_started) * 1000),
                savepoint_outcome="savepoint_failed",
            )
            self._producers_failed += 1
            return "savepoint_failed"

        # Step 2 — execute producer subjects inside the savepoint.
        local_entries: list[ChangesMadeEntry] = []
        artifacts_produced = 0
        subjects_discovered = 0
        outcome: str

        try:
            subjects_discovered, artifacts_produced, local_entries = await self._iterate_subjects(producer)
            if savepoint.is_active:
                await savepoint.commit()
            outcome = "committed"
        except Exception as exc:
            # Structured payload mirrors `_emit_producer_log` so CloudWatch
            # queries on `event_type=pipeline.producer_failed` can read
            # error_type / error_message inline. exc_info=True preserves the
            # Python traceback in the log record so the formatter renders
            # the full stack — root-causing producer failures shouldn't
            # require a docker exec into prod.
            error_payload = {
                "event_type": "pipeline.producer_failed",
                "timestamp": utc_now().isoformat(),
                "run_id": str(self._run_id),
                "producer": producer_name,
                "producer_version": producer_version,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            logger.error(
                "pipeline.producer_failed %s",
                json.dumps(error_payload),
                exc_info=True,
            )
            if savepoint.is_active:
                await savepoint.rollback()
            outcome = "rolled_back"
            local_entries = []  # discard per §3.2 timing spec
            artifacts_produced = 0
            self._producers_failed += 1

        # Step 3 — merge entries ONLY on commit (§3.2 timing).
        if outcome == "committed":
            self._changes.extend(local_entries)
            self._producers_run += 1
            artifact_type: ArtifactType = getattr(producer, "artifact_type", "EntitySchema")
            self._artifacts_by_type[artifact_type] = self._artifacts_by_type.get(artifact_type, 0) + artifacts_produced

        duration_ms = int((time.monotonic() - producer_started) * 1000)
        self._emit_producer_log(
            producer=producer,
            subjects_discovered=subjects_discovered,
            artifacts_produced=artifacts_produced,
            duration_ms=duration_ms,
            savepoint_outcome=outcome,
            producer_name=producer_name,
            producer_version=producer_version,
        )
        return outcome

    async def _iterate_subjects(self, producer: ArtifactProducer) -> tuple[int, int, list[ChangesMadeEntry]]:
        """Discover subjects, produce + supersede each, collect entries."""
        subjects = await producer.discover_subjects(self._session, self._run_id)
        local: list[ChangesMadeEntry] = []
        artifacts_produced = 0

        for subject in subjects:
            produced = await producer.produce(self._session, subject, self._run_id)
            if produced is None:
                continue

            # Producer-supplied current-active resolver if available
            # (BaseProducer subclasses get this via _fetch_active). Stub
            # producers can omit it — supersede() then sees prev=None and
            # treats every produce as a fresh create.
            prev = None
            fetch = getattr(producer, "_fetch_active", None)
            if callable(fetch):
                prev = await fetch(
                    self._session,
                    produced.subject_entity_id,
                    produced.dimension,
                )

            entry = await producer.supersede(self._session, prev, produced, self._run_id)
            local.append(entry)
            if entry.action in ("create", "supersede"):
                artifacts_produced += 1

        return len(subjects), artifacts_produced, local

    # ── Telemetry emission ────────────────────────────────

    def _producer_log_payload(
        self,
        producer: ArtifactProducer,
        *,
        subjects_discovered: int,
        artifacts_produced: int,
        duration_ms: int,
        savepoint_outcome: str,
        producer_name: str | None = None,
        producer_version: int | None = None,
    ) -> dict[str, Any]:
        name = producer_name or getattr(producer, "name", producer.__class__.__name__)
        version = producer_version
        if version is None:
            version = getattr(producer, "version", 1)
        return {
            "event_type": f"memory/artifact-producer.{name}",
            "timestamp": utc_now().isoformat(),
            "run_id": str(self._run_id),
            "chat_id": self._chat_id,
            "topic_id": self._topic_id,
            "venue": self._venue,
            "duration_ms": duration_ms,
            "producer_version": version,
            "subjects_discovered": subjects_discovered,
            "artifacts_produced": artifacts_produced,
            "savepoint_outcome": savepoint_outcome,
        }

    def _emit_producer_log(
        self,
        *,
        producer: ArtifactProducer,
        subjects_discovered: int,
        artifacts_produced: int,
        duration_ms: int,
        savepoint_outcome: str,
        producer_name: str | None = None,
        producer_version: int | None = None,
    ) -> None:
        payload = self._producer_log_payload(
            producer,
            subjects_discovered=subjects_discovered,
            artifacts_produced=artifacts_produced,
            duration_ms=duration_ms,
            savepoint_outcome=savepoint_outcome,
            producer_name=producer_name,
            producer_version=producer_version,
        )
        # Inline-serialize per the project's structured-log convention so the
        # default formatter doesn't drop extra= keys (memory-v2 hotfix
        # 2026-05-01).
        logger.info("%s %s", payload["event_type"], json.dumps(payload))

    def _pipeline_log_payload(self, duration_ms: int) -> dict[str, Any]:
        return {
            "event_type": "memory/consolidate",
            "timestamp": utc_now().isoformat(),
            "run_id": str(self._run_id),
            "phase": self._phase,
            "pass_number": self._pass_number,
            "chat_id": self._chat_id,
            "topic_id": self._topic_id,
            "venue": self._venue,
            "duration_ms": duration_ms,
            "producers_run": self._producers_run,
            "producers_failed": self._producers_failed,
            "artifacts_produced": dict(self._artifacts_by_type),
        }

    def _emit_pipeline_log(self, duration_ms: int) -> None:
        payload = self._pipeline_log_payload(duration_ms)
        logger.info("memory/consolidate %s", json.dumps(payload))

    def _emit_systemic_failure_log(self, failed: int, total: int) -> None:
        payload = {
            "event_type": "pipeline.savepoint_systemic_failure",
            "timestamp": utc_now().isoformat(),
            "run_id": str(self._run_id),
            "savepoint_setup_failures": failed,
            "total_producers": total,
        }
        logger.error("pipeline.savepoint_systemic_failure %s", json.dumps(payload))


__all__ = ["ArtifactPipeline"]
