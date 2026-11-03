# Recoleva

**Human-inspired memory for AI agents.**

Recoleva is a Python SDK that extracts memories from conversations, stores them in
PostgreSQL, and recalls context for later turns. Its design combines semantic
retrieval, recency, access history and consolidation. Your application supplies
the models, conversation history and access policy.

- **Remember across conversations.** Extract facts and entities into persistent
  memory, with a stored cursor to resume extraction.
- **Recall context for a turn.** Rank memories using semantic similarity, entity
  connections, recency and prior access.
- **Choose what counts as used.** Commit recalled context after delivery, or
  discard a preview without recording an access.
- **Maintain memory over time.** Run consolidation through the same client, with
  APIs for procedural rules, execution evidence and outcomes.
- **Keep control of your stack.** Supply inference and embedding adapters, own
  your PostgreSQL database, and configure identity, model roles and permissions.

## Install

Requires Python 3.13+, PostgreSQL with pgvector, and an embedder that returns
1536-dimensional vectors. The demo below uses PostgreSQL 15.

```sh
python -m pip install "recoleva @ https://github.com/dimabalony/recoleva/releases/download/v0.3.0/recoleva-0.3.0-py3-none-any.whl"
```

The [v0.3.0 release](https://github.com/dimabalony/recoleva/releases/tag/v0.3.0)
includes the wheel, source distribution, checksums and verification receipts.

## Try it locally

Run the included **synthetic demo** against a fresh PostgreSQL database. It uses
scripted inference and token-hash embeddings, so no model account or API key is
needed. It records a conversation, extracts a fact, recalls it, commits the
access, runs consolidation, and reopens the client against the same database
and history.

With Python 3.13 and Docker installed:

```sh
git clone https://github.com/dimabalony/recoleva.git
cd recoleva
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install "recoleva @ https://github.com/dimabalony/recoleva/releases/download/v0.3.0/recoleva-0.3.0-py3-none-any.whl"

docker run --detach --rm --name recoleva-demo \
  -e POSTGRES_PASSWORD=demo \
  -p 127.0.0.1:55432:5432 pgvector/pgvector:pg15
until docker exec recoleva-demo pg_isready -h 127.0.0.1 -U postgres >/dev/null 2>&1; do
  sleep 1
done
docker exec recoleva-demo createdb -U postgres recoleva_demo

python -I examples/standalone_host.py lifecycle \
  --db-url postgresql+asyncpg://postgres:demo@127.0.0.1:55432/recoleva_demo
```

The command validates the lifecycle and prints a JSON report. Selected fields:

```json
{
  "trace_contents": ["Pat Example works as a pharmacist at Example Co"],
  "recognition": {"context_mentions_fact": true},
  "after_restart": {"messages_read": 0}
}
```

Remove the demo container and its disposable database when finished:

```sh
docker rm -f recoleva-demo
```

See [the complete demo](examples/standalone_host.py) for adapter implementations
and additional lifecycle, recovery and retention scenarios.

## Connect your agent

```mermaid
flowchart LR
    App[Your application] --> Memory[Recoleva]
    Memory --> Models[Your inference and embedding adapters]
    Memory --> History[Your conversation history]
    Memory --> Store[Your PostgreSQL + pgvector database]
```

Create a `MemoryClient` with `HostAdapters`: inference, embedder, history and an
async SQLAlchemy session factory. Supply the `scrub_secrets`,
`validate_skill_content` and `redact_urls_for_trace` callbacks through `overrides`,
and use `HostPolicy` to define read, write and teaching permissions.

Reuse one client per store. `record_message()` writes to your history adapter;
`extract()` reads that history into memory. Before generating a response, recall
context and settle it after your application delivers the answer:

```python
from kai_memory import Chat, MemoryClient


async def answer(memory: MemoryClient, chat: Chat, question: str, deliver):
    """deliver(question, context) generates and delivers your application's reply."""
    recognition = await memory.recall(question, chat)
    try:
        reply = await deliver(question, recognition.primed.priming_context or "")
    except BaseException:
        recognition.discard()
        raise
    recognition.commit()
    return reply
```

For previews, call `discard()` after inspecting the context. `begin_turn()` adds
staged delivery and recall within a turn. `consolidate()` runs maintenance. Close
the client with `await memory.close()` or an async context manager to drain its
work; your application manages its database engine and history lifecycle.

## Explore

- [Public API and adapter contracts](src/kai_memory/sdk.py)
- [Storage upgrade example](examples/storage_upgrade.py)
- [Tests and verification](CONTRIBUTING.md) · [CI workflow](.github/workflows/test.yml)
- [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

## License

[GPL-3.0-or-later](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for
dependency licenses and attribution.
