# Recoleva

A Python cognitive memory SDK with host-owned inference, history, storage and policy.
Distribution `recoleva`, version 0.3.0, API v2; import `kai_memory`.
The import namespace is retained for application compatibility.

Install the wheel from the [versioned release](https://github.com/dimabalony/recoleva/releases/tag/v0.3.0):

```sh
python -m pip install "recoleva @ https://github.com/dimabalony/recoleva/releases/download/v0.3.0/recoleva-0.3.0-py3-none-any.whl"
```

For a verified pin, append `#sha256=<wheel_sha256>` from that release's
`release.json` to the URL. Releases include the sdist, SHA256SUMS and test receipts.
A source checkout can be installed with `pip install .`.

Python 3.13+; PostgreSQL 15 with pgvector; 1536-dimensional embeddings.
Install the tested dependency profile with `python -m pip install -r requirements-test.txt`.
Run the complete exported unit suite with `python -m pytest`.
The SDK excludes comparative evaluation and provider runners.
Run the independent PostgreSQL contract suite against a disposable local server:
`MEMORY_SDK_TEST_DATABASE_URL=postgresql+asyncpg://postgres@127.0.0.1:5432/postgres python -m pytest tests/postgres`.
The suite creates and drops a unique database for each case. An unset URL skips
storage tests and does not qualify persistence. Python 3.13 and PostgreSQL 15 with
pgvector are the rehearsal baseline. Runtime dependency ranges are not a claim
that all permitted versions were tested.

The supported API is `MemoryClient`, `HostAdapters`, `Chat` and their public
contract types, plus `kai_memory.storage` composition and disposable bootstrap.
Internal modules are not stable application entry points.

```python
from kai_memory import MemoryClient, HostAdapters, Chat

async def preview(adapters: HostAdapters, chat: Chat):
    async with MemoryClient(adapters) as memory:
        recognition = await memory.recall("What do we know?", chat)
        context = recognition.primed.priming_context
        recognition.discard()  # Preview only; no access learning.
        return context
```

Create one application-owned client per store and reuse it across operations.
For a delivered turn, call `recognition.commit()` after using its context;
discard unused or rejected context. `begin_turn` supports staged delivery and
read-only mid-turn recall. `record_message` writes through the history port,
`extract` captures memory, and `consolidate` performs maintenance. The context
manager calls `close`, which finishes/drains or cancels owned work; it does not
transfer ownership of the application's engine or history.

`HostAdapters` requires inference, embedder, history and session_factory ports;
provide the scrub_secrets, validate_skill_content and redact_urls_for_trace
hygiene callbacks through `overrides`. Hosts own their async engine/sessions,
canonical conversation history, clock, settings and authorization. Telemetry
is optional and defaults to no output. `AgentIdentity` configures the host's
identity; `ModelRoles` selects extraction, consolidation, artifact and auxiliary
models for its inference adapter. No inference vendor is required by the SDK.
Embedders return one nonzero 1536-dimensional vector per input; storage fixes
this dimension and rejects incompatible vectors before persistence.

`HostPolicy` supplies is_admin, integer respect (0..100), may_read and may_write.
The default NeutralHostPolicy permits the host-owned chat scopes but grants no
teaching authority. Chat IDs do not authorize tenants; multi-tenant applications
must implement those policy checks. See `examples/standalone_host.py` for a
complete synthetic host and executable lifecycle/recovery examples.

Clients on separate stores are supported. Shared-store concurrent writers and
process-kill durability remain unqualified; reopen proof retains host history.
`storage.bootstrap_disposable` is for fresh stores and refuses populated ones.
Production schema migration, backup and previous-version rollback belong to the
application. Rehearse upgrades with `examples/storage_upgrade.py`; these examples
grant no production access. Procedural reinforcement supports 1,000 eligible
rules/habits and refuses the 1,001st before partial writes.

This repository is the canonical SDK source. General evaluation, private host
configuration, transcripts and production fixtures are outside this distribution.
The SDK tests include authored synthetic examples and reviewed generic test
adaptations; they demonstrate contracts, not real-model answer quality, learning
efficacy or human-equivalent memory. No provider calls are needed for CI.

`EXPORT_PROVENANCE.json` preserves the initial private-candidate export receipt;
its historical publication status is not current release metadata.
`BOOTSTRAP_PROVENANCE.json` records the public distribution and license.
Release evidence records public source revision, artifact hashes, installed
payload hashes, executed test counts and scenario checksums.

Licensed under GPL-3.0-or-later; see LICENSE and THIRD_PARTY_NOTICES.md.
Dependencies retain their own licenses, including Unidecode's GPL-2.0-or-later.
Runtime requirements are ranges, not a reproducible lock; release evidence
records the versions installed for proof. See CONTRIBUTING.md and SECURITY.md.
