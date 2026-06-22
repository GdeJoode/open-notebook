# Operator Guide — Cloud/Local Model Routing (Track J)

This guide is for the person running an Open Notebook instance. It covers the
environment keys, the privacy model, the provider-chain config UI, the fair-use
caution learned from live testing, and how to enable/disable providers.

> **TL;DR**
> - The three LLM stages (entity **extraction**, **summarization**, **chat**)
>   route through cloud-by-default with automatic per-document failover to your
>   **local** model.
> - A document or notebook marked **private** never leaves the box — its LLM
>   stages always run local, and a private flag is **sticky** (can never be
>   un-privated upward).
> - **Embeddings and parsing always stay local** — never routed to cloud. This
>   is a hard invariant (vector-search correctness depends on it).
> - The cloud provider is **NVIDIA NIM** (`NVIDIA_API_KEY`). No key → everything
>   runs local automatically.

---

## 1. Environment keys

Routing reads provider API keys from the environment (set them in `.env`):

| Env var | Provider | Role | If unset |
|---|---|---|---|
| `NVIDIA_API_KEY` | NVIDIA NIM (cloud) | Default cloud provider for all three LLM stages | The provider is **dropped from every chain**; routing runs local-only |
| `NVIDIA_BASE_URL` | NVIDIA NIM | Optional endpoint override (default `https://integrate.api.nvidia.com/v1`) | Uses the default NIM endpoint |
| `OLLAMA_API_BASE` | Ollama (local) | The local last-resort fallback (and the only provider used in PRIVATE mode) | Defaults to `http://localhost:11434` |

NIM is an **OpenAI-compatible** REST endpoint. The default cloud model is
`mistralai/mistral-medium-3.5-128b`. The key is a `Bearer` token; **never commit
it** — keep it in `.env` (gitignored).

**"Configured" semantics.** A provider is considered configured when it is local
**or** its API-key env var is present (`bool(os.environ[api_key_env_var])`). An
unconfigured cloud provider is silently dropped from the resolved chain — there is
no error, the route simply falls through to the next candidate (ultimately local).

Other provider keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …) are recognized by
the provider registry and may be added to a chain via the config UI, but the
shipped default chain is `[nvidia → local]`.

### Telemetry opt-out (operational, not privacy)

`OPEN_NOTEBOOK_DISABLE_METRICS=1` disables the `routing.served` telemetry writes
(and all other metrics). Telemetry payloads contain **no document content** — only
the served provider, the task name, and the failover trail — so this switch is an
operational quiet-mode, not a privacy control. The privacy guarantee below holds
regardless of this flag.

---

## 2. The privacy model — global → notebook → document (sticky-private)

Privacy is **layered**, most-specific-wins:

```
document.private == true   →  PRIVATE   (sticky — wins over everything, never escalates to cloud)
      else notebook.privacy_mode  ("private" | "cloud" | unset→inherit)
      else global default_privacy_mode   (ships as "cloud")
```

- **PRIVATE** mode pins the *entire* LLM path for that document/notebook to your
  **local** provider. No cloud `LanguageModel` is ever constructed for a private
  document — verified by the E2E `private-never-cloud` test.
- **Sticky rule**: once a document is `private`, no notebook or global setting
  can route it to cloud. There is exactly one early-return in the privacy
  resolver encoding this, and a property test asserts no input combination turns
  a private document cloud.
- **CLOUD** mode prefers the cloud provider, then automatically falls back to
  local on outage.

### What leaves the box vs. what never does

| Data path | Cloud-eligible? |
|---|---|
| Entity extraction (public doc, CLOUD mode) | Yes → NIM, fallback local |
| Summarization (public doc, CLOUD mode) | Yes → NIM, fallback local |
| Chat (CLOUD default) | Yes → NIM, fallback local |
| Any stage on a **private** document/notebook | **No — always local** |
| **Embeddings** (vector search) | **No — always local, 768-dim `nomic-embed-text`** |
| **Parsing** (docling / MinerU) | **No — local GPU service, never an LLM routing call** |

The embeddings/parsing-local rule is a **blocking invariant**: cloud embeddings
would change the vector dimensionality (NIM/OpenAI are 1536/3072-dim vs. the
768-dim HNSW index) and silently corrupt vector search. The routing service has
no embedding/parsing task and the embedding service imports nothing from it; a
guardrail test is the canary.

### Making things private

- **A notebook**: open the notebook settings panel → Privacy toggle → `Private`
  (or `Inherit` to follow the global default).
- **A document on upload**: flip the `Private` toggle in the upload dialog; the
  source then shows a "Private" badge.
- **Globally**: Settings → Model Routing → global privacy switch → `Private`. A
  privacy-first operator flips this one setting and everything defaults local.

---

## 3. The provider-chain config UI

`Settings → Model Routing` (`/settings/model-routing`):

- **Per task** (extraction / summarization / chat): an ordered list of providers
  with **up/down** reorder buttons and an **enable/disable** toggle each. The
  order is the failover order — the first available, healthy, enabled provider
  serves; on a transient failure the document fails over to the next; the local
  provider is the last resort.
- **Empty state**: when no chain is persisted the UI shows the built-in default
  chain (`nvidia → local`) read-only with a "Save to customize" affordance.
  Saving writes a `model_route` row you can then edit.
- **Global privacy switch**: the `default_privacy_mode` (cloud/private).
- **Provider health chips**: per provider, `configured` (key present),
  `is_local`, the circuit-breaker state (`closed`/`open`/`half_open`), and a
  recent fallback count (how often it was failed-over-FROM).

The **private chain** is LOCAL-only by contract — the API rejects a cloud
provider in a private chain with a 400.

### Re-enabling / disabling a provider

- **Disable** a provider for a task: toggle it off in the chain editor → it is
  greyed and excluded from the effective chain. Saved via `PUT
  /api/model-routes/{task}`.
- **Disable cloud entirely** (run fully local): either unset `NVIDIA_API_KEY`
  (the provider drops from every chain automatically) or flip the global privacy
  mode to `private`.
- **Re-enable**: set the key back / toggle the provider on / set privacy to
  `cloud`. No restart needed for chain edits; an env-key change needs a process
  restart to be picked up.
- **A circuit-breaker that has tripped** (`open`) recovers automatically: after
  the cooldown it goes `half_open`, probes one request, and closes on success.
  Breaker state is **in-process** (single worker, V1) — it does not persist
  across restarts and diverges per worker in a multi-worker deployment (J-D3).

---

## 4. Fair-use caution (the lesson from the live test)

**Do not route high-volume entity extraction at a rate-limited cloud tier.**

During live testing, NVIDIA NIM (`mistral-medium-3.5-128b`) was:

- **Slow / easily overloaded for high-volume extraction** — extraction issues
  many calls per document (per-chunk, multi-pass), and the cloud tier throttled
  and stalled.
- **Good for summarization** — fewer, larger calls; the cloud model's quality is
  worth the round-trip there.

Recommended posture:

- **Extraction → local** (`llama3.1:8b` fits fully on a ~9 GiB GPU and runs ~9×
  faster than CPU-spilled larger models). Set the extraction chain to local-first,
  or run extraction under PRIVATE mode.
- **Summarization → cloud (NIM)**, with local as the automatic fallback.
- **Chat → local or cloud** to taste.

The routing layer protects the endpoint regardless:

- A conservative **cloud rate limiter** (default **20 req/min**, configurable)
  with **exponential backoff** on 429 caps the request rate.
- Per-document **sequential** processing — no parallel hammering of the endpoint.
- When the cloud limiter saturates, the document fails over to the (unthrottled)
  **local** provider **without** a circuit-breaker penalty — a saturated throttle
  is not evidence the provider is down.

If NIM is overloaded, you can temporarily disable it (unset the key or toggle it
off in the chain) and let extraction + summarization run local until it cools
down.

---

## 5. Live smoke checklist (manual, needs a real key + worker)

CI exercises all three guarantees with **mocked SDKs** (no live NIM). To verify
the live cloud path on a deployed instance:

1. **Cloud path works**: set `NVIDIA_API_KEY`, global mode `cloud`, ingest a
   small public document. Confirm a summary is produced and the routing summary
   (`GET /api/model-routes/summary`) shows a `nvidia` served event.
2. **Forced-outage failover**: temporarily set an invalid `NVIDIA_API_KEY` (or
   block the endpoint), ingest a document. Confirm extraction completes on the
   local provider and the health endpoint shows a fallback count / open circuit
   for `nvidia`.
3. **Private-never-cloud**: with a valid cloud key, upload a document with
   `Private` on. Confirm (via the routing summary / provider health, which should
   show zero cloud served events for that source) that nothing reached cloud.
4. **No-cloud → local**: unset `NVIDIA_API_KEY`, restart, ingest a document.
   Confirm it completes on local and the health endpoint shows `nvidia`
   unconfigured.

---

## 6. Reference

- `ARCHITECTURE.md` §8 — the routing service + the three-stage diagram + the
  embeddings/parsing-local invariant.
- `docs/tracks/J-model-routing/plan.md` — the full per-phase plan + decision
  points (J-D1…J-D5, J-Q1…J-Q7).
- `docs/tracks/J-model-routing/status.md` — the implementation ledger.
