# Operator Guide — Shared graph memory (Track W)

This guide is for the person running an Open Notebook instance. It covers two
things Track W added:

1. **Registering the `surrealdb-mcp` server** in a Claude Code session, so the
   session reads/writes the shared SurrealDB knowledge-graph substrate through
   the graph tools (`search` / `get_node` / `related` / `cite` / `add_note`).
2. **Bringing up + smoke-testing the optional reranker** microservice, and
   verifying the heuristic fallback when it is down.

> **TL;DR**
> - The MCP server runs over **stdio** (local, no auth). It needs `SURREAL_*`
>   env to point at your DB — set `SURREAL_DATABASE=staging` to use the live
>   work DB.
> - The reranker is **off by default** (`/search` only calls it when
>   `rerank=true`). The model (~2 GB) is gated behind an explicit
>   `docker compose build/up reranker`. With the service down, `/search` still
>   works via the zero-dep heuristic fallback.

---

## 1. Register `surrealdb-mcp` in a Claude Code session

The server's entry point is the `surrealdb-mcp` console script
(`packages/surrealdb-service/pyproject.toml` → `surrealdb_service.mcp.server:main`).
Run it over **stdio** — the default, and the only transport that is safe without
auth (the `cite`/`add_note` WRITE tools must not be exposed over HTTP unguarded;
see §3).

### Which env it needs

The server reads its DB connection from `SURREAL_*` (env prefix `SURREAL_`,
`packages/surrealdb-service/src/surrealdb_service/config.py`):

| Env var | Purpose | Default |
|---|---|---|
| `SURREAL_URL` | Full RPC URL (overrides address/port if set) | — |
| `SURREAL_ADDRESS` | Host | `localhost` |
| `SURREAL_PORT` | Port | `8000` |
| `SURREAL_USER` | User | `root` |
| `SURREAL_PASSWORD` (or `SURREAL_PASS`) | Password | `root` |
| `SURREAL_NAMESPACE` | Namespace | `open_notebook` |
| **`SURREAL_DATABASE`** | **Database** — set to **`staging`** for the live work DB | `open_notebook` |

> The config default `open_notebook` is stale; the live work DB is **`staging`**.
> Always set `SURREAL_DATABASE=staging` explicitly.

### `.mcp.json` (project-scoped) — recommended

Add the server to the project's `.mcp.json` so any Claude Code session in this
repo picks it up:

```json
{
  "mcpServers": {
    "surrealdb": {
      "command": "uv",
      "args": [
        "run", "--project", "packages/surrealdb-service",
        "surrealdb-mcp", "--transport", "stdio"
      ],
      "env": {
        "SURREAL_ADDRESS": "localhost",
        "SURREAL_PORT": "8000",
        "SURREAL_USER": "root",
        "SURREAL_PASSWORD": "root",
        "SURREAL_NAMESPACE": "open_notebook",
        "SURREAL_DATABASE": "staging"
      }
    }
  }
}
```

### `claude mcp add` (one-off, equivalent)

```bash
claude mcp add surrealdb \
  --env SURREAL_ADDRESS=localhost \
  --env SURREAL_PORT=8000 \
  --env SURREAL_USER=root \
  --env SURREAL_PASSWORD=root \
  --env SURREAL_NAMESPACE=open_notebook \
  --env SURREAL_DATABASE=staging \
  -- uv run --project packages/surrealdb-service surrealdb-mcp --transport stdio
```

### Verify it is attached

In the session, list MCP servers (`/mcp`) and confirm `surrealdb` is connected,
then call a read tool — e.g. `search` with a keyword (no embedding → lexical
BM25), then `related` on a returned node id, then `get_node` on a neighbour. Two
separate sessions pointed at the same `SURREAL_DATABASE` see the same nodes and
edges — that is the shared substrate.

You can also run the server standalone to sanity-check it starts:

```bash
SURREAL_DATABASE=staging \
  uv run --project packages/surrealdb-service surrealdb-mcp --transport stdio
```

---

## 2. Reranker bring-up + smoke + fallback (gated)

The reranker (`services/reranker/`) loads a local multilingual cross-encoder
(`BAAI/bge-reranker-v2-m3`, ~2 GB on first run). The model download + the CPU
`torch`/`sentence-transformers` install are heavy, so this is an **explicit,
operator-run** step. `torch`/`sentence-transformers` live only in the reranker
container — never in `app-main`.

### a. Build + bring up

```bash
docker compose build reranker
docker compose up -d reranker
docker compose logs -f reranker          # wait for model load (~2 GB first run)
```

### b. Health — expect `model_loaded: true`

```bash
curl -s localhost:8105/health
# {"status":"ok","service":"reranker","model":"BAAI/bge-reranker-v2-m3","model_loaded":true}
```

### c. Dutch `/rerank` smoke — corroborated passages on top

```bash
curl -s -X POST localhost:8105/rerank \
  -H 'content-type: application/json' \
  -d '{"query":"Wat zijn de doelstellingen van de regiodeal?",
       "passages":[
         "De regiodeal richt zich op economische versterking van de regio.",
         "Het weer is vandaag zonnig met een lichte bries.",
         "Doelstellingen van de regiodeal omvatten werkgelegenheid en leefbaarheid."]}'
```

Expect the two regiodeal passages (indices **0** and **2**) ranked **above** the
weather passage (index **1**) — i.e. the `results` list leads with `index` 0/2.

### d. End-to-end through the app

`POST /search {type:"hybrid", rerank:true}` on a Dutch query reorders the top-N
fused hits (each carries a `_rerank_score`):

```bash
curl -s -X POST localhost:5055/search \
  -H 'content-type: application/json' \
  -d '{"query":"doelstellingen regiodeal","type":"hybrid","rerank":true,"limit":10}'
```

### e. Fallback — verify the heuristic takes over when the service is down

```bash
docker compose stop reranker
```

Re-run the same `rerank:true` request: it must still return **200** (no 500),
now reordered by the zero-dep heuristic `retrieval.reranker.Reranker` fallback
(logged). With `rerank:false` (the default) the reranker is never called and the
result is byte-identical to the W.1 hybrid output regardless of service state.

---

## 3. Safety notes

- **stdio only, no auth.** The MCP server has no authentication; this is correct
  for `stdio` (the only caller is the local process). The `--transport sse` /
  `--transport streamable-http` options would expose the `cite`/`add_note`
  WRITE tools over HTTP unauthenticated — **do not** use them without first
  gating write-tool registration to stdio or adding auth. (Tracked as a W.3
  follow-up.)
- **`cite` dedup is read-then-write**, not atomic — unreachable under the
  single-caller stdio transport, but would need a transaction / unique-edge
  guard before any concurrent-writer setup.
- **Reranker is optional.** Nothing depends on it being up; leaving it down just
  means `rerank=true` falls back to the heuristic. `docker-compose.yml` has no
  hard `depends_on` for it.
