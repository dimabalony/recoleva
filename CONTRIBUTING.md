# Contributing

This repository owns the SDK implementation, public contracts and deterministic
proof. Applications supply model, embedding, history, clock and authorization
adapters. General evaluation and private data do not belong here. Do not fork SDK
logic into an application.

Use Python 3.13. Install `pip install . -r requirements-test.txt`; run
`python -I -m pytest tests`. PostgreSQL tests skip without an explicit disposable
local server; qualify them with:

```sh
MEMORY_SDK_TEST_DATABASE_URL=postgresql+asyncpg://postgres:eval@127.0.0.1:5432/postgres python -I -m pytest tests/postgres
```

Use PostgreSQL 15 with pgvector and permission to create databases. The fixture
creates/drops unique databases. Do not use production data. CI builds an sdist,
builds the wheel from its unpacked contents, installs it, and runs portable,
PostgreSQL and lifecycle/storage proof. Release evidence rejects empty or skipped
PostgreSQL proof and verifies installed bytes against the exact wheel.

Preserve MemoryClient/HostAdapters, configured authority, commit/discard,
cancellation/replay and borrowed-resource ownership. Add meaningful behavioral
regressions. Keep fixture identities and label scripted inference accurately.
Contributions use GPL-3.0-or-later; include attribution/license evidence for
third-party material. Never relabel private data as synthetic.

Public issues belong on GitHub; accepted private work may carry its existing
tracker reference. See SECURITY.md for confidential reports. Maintainers publish
from main after CI passes. A `vX.Y.Z` tag must match pyproject.toml. Release CI
publishes the already-tested wheel/sdist bytes, refuses existing releases and
includes SHA256SUMS and release.json. No PyPI publishing is configured.
Previous artifact versions/checksums support host rollback; database rollback
remains the application's responsibility.
