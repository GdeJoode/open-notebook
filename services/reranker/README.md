# Reranker service

A small FastAPI service that re-scores `(query, passage)` pairs with a local
cross-encoder, used by `app-main` to rerank the top-N hits from hybrid search.

Keeping `torch` / `sentence-transformers` in this container keeps those heavy
dependencies out of `app-main` — the app calls this service over plain HTTP
(`RERANKER_SERVICE_URL`).

## Endpoints

- `GET /health` — liveness; reports the configured model and whether it is loaded.
- `POST /rerank` — body `{query: str, passages: [str], top_k?: int}`; returns
  `{results: [{index, score}]}` sorted by score descending, where `index` is the
  position in the request's `passages` list.

## Model

- `RERANKER_MODEL` (default `BAAI/bge-reranker-v2-m3`) — multilingual; chosen
  because the corpus is Dutch. Swap for a smaller/faster variant to tune latency.
- `RERANKER_DEVICE` (default empty → auto/CPU) — set `cuda` to use a GPU.
- `HF_HOME` (default `/data/models`) — HuggingFace cache dir; mount a volume so
  the ~2 GB model is downloaded once.
- `RERANKER_SKIP_WARMUP=1` — skip loading the model at startup (load on first
  request); used for fast boots / tests.

## Run locally (deploy / smoke test)

```bash
# Build + start just the reranker container
docker compose build reranker
docker compose up -d reranker

# First start downloads ~2 GB (bge-reranker-v2-m3); watch the logs
docker compose logs -f reranker

# Smoke test once /health reports model_loaded: true
curl -s localhost:8105/health

curl -s -X POST localhost:8105/rerank \
  -H 'content-type: application/json' \
  -d '{
        "query": "Wat zijn de doelstellingen van de regiodeal?",
        "passages": [
          "De regiodeal richt zich op economische versterking van de regio.",
          "Het weer is vandaag zonnig met een lichte bries.",
          "Doelstellingen van de regiodeal omvatten werkgelegenheid en leefbaarheid."
        ]
      }'
```
