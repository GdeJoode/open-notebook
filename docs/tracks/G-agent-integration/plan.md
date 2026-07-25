# Track G — Agent Integration & Headless Mode — SPRINT PLAN

> **Status**: 📝 PROPOSED (2026-07-24) — **awaiting human approval**. Track-planner
> output; not yet approved for implementation. Grounded against the shipped ingest
> chain (`services/source_pipeline.py`, `services/command_service.py`,
> `handlers.py`), the shipped auth surface (`api/auth.py` `PasswordAuthMiddleware` +
> `api/rate_limit.py` slowapi limiter, Track I.H1), and the existing summarization /
> entity-extraction services.
>
> **Track ID**: `G`. Reference: `docs/FEATURE_ROADMAP.md` § "Track G — Agent
> Integration & Headless Mode (NEW)" (~line 522) and § 3.3.
>
> **Scope of this sprint plan**: the **headless core** — roadmap sub-features **G1
> (Public Agent API)** and **G2 (File-watcher service)**. The heavier Obsidian
> bidirectional-sync (roadmap G3/G4), outbound webhooks + HMAC/retry audit (G5), and
> summary templates (G6) are **deferred to a follow-up sprint** and are summarised in
> § 9, not decomposed here (decision **G-D8**). Track **H** (vision parser) is
> DEFERRED until Track G is complete and is out of scope entirely.

## 1. Context

**Track goal.** Turn open-notebook into a headless backend that external agents
(hermes, claude-code, cursor, custom) can drive over a **versioned, API-key-authed**
HTTP surface (`/api/v1/agents/`) for document/audio/URL processing, summary
generation and KG extraction — plus an always-on **file-watcher** that auto-ingests
files dropped into conventional inbox paths. The agent API is a **thin façade over
the already-shipped ingest chain**, not a parallel pipeline.

**Two in-scope sub-tracks (roadmap):**
- **G1 — Public Agent API**: versioned `/api/v1/agents/` endpoints
  (`process-document` / `process-audio` / `process-url` / `generate-summary` /
  `extract-entities`), a `GET /agents/jobs/{job_id}` status, a per-agent audit log,
  and **API-key auth** (`X-API-Key`, per-key `agent_id` + permissions +
  rate-limit). New `agent_keys` table. Auto OpenAPI at
  `GET /api/v1/agents/openapi.json`. UI: an "API Keys" settings tab.
- **G2 — File-watcher service**: always-on watcher on `~/open-notebook/inbox/` and
  `<notebook_data>/<notebook_id>/inbox/` that debounces, routes by file-type, and
  auto-ingests via the **same** jobs the API façade calls; move-to-`_processed/` on
  success, `_errors/` on failure.

**What already exists that G builds on (do NOT duplicate):**
- **Ingest chain (the façade's engine).** `services/source_pipeline.py`
  (declarative `SOURCE_PIPELINE` + `advance_source`) drives parse→embed→extract→
  graph→insights and is resumable. Sources are created via `_create_source_impl`
  (`api/routers/sources_upload.py`) for the three input kinds already supported:
  `type="upload"` (file path or multipart), `type="link"` (URL), `type="text"`.
  Ingestion is enqueued as the `process_source` command → `JobType.DOCUMENT_PARSE`
  → `handle_process_source` (`handlers.py`). **`process-document` / `process-url` /
  `process-audio` must reuse this exact path**, only wrapping it in an agent-facing
  request/response and an audit-log write.
- **Command/job seam + status.** `services/command_service.py`
  (`CommandService.submit_command_job` maps a command name via `_COMMAND_TO_JOB_TYPE`;
  `get_command_status(job_id)` returns the normalised status dict). **`GET
  /agents/jobs/{job_id}` reuses `get_command_status` verbatim** — no new job store.
- **Text capabilities.** `services/summarization_service.py`
  (`generate_summary(source_id, strategy, config)`) and
  `services/entity_extraction_service.py` (`EntityExtractionService`, multi-schema
  workflow builder). `generate-summary` / `extract-entities` over **raw text** map
  onto these — via a small no-DB "over text" method (see G-D5), not a new pipeline.
- **Auth + rate-limit (must integrate, not bypass).** `api/auth.py`
  `PasswordAuthMiddleware` gates all `/api/*` with the shared `OPEN_NOTEBOOK_PASSWORD`
  bearer when set, with an `excluded_paths` list (already supports `*` wildcards).
  `api/rate_limit.py` exposes a shared slowapi `Limiter` (per-IP, `headers_enabled`);
  `config.py` exposes `RATE_LIMIT_RPM` (Track I.H1). The agent key scheme layers on
  top of these — the password middleware **excludes** the agent routes, and the
  `require_agent_key` dependency is their sole authenticator.
- **Settings UI shell.** `frontend/src/app/(dashboard)/settings/page.tsx` is a
  shadcn `Tabs` shell (General / Vault / Zotero / Advanced). The "API Keys" tab is
  one more `TabsTrigger` + `TabsContent`, mirroring `VaultSync.tsx` /
  `ZoteroSettings.tsx`.
- **Migration/test conventions.** `migrations/NN.surrealql` + `NN_down.surrealql`;
  OVERWRITE-guarded SCHEMAFULL DEFINEs (migration 74/75 drift note); roundtrip is
  CI-gated. Tests split `@requires_docker` (testcontainer) vs pure unit. **Next free
  migration number is 76** (75 = Track F `audit_findings`).

**Dependencies on other tracks:** none hard. G reuses Track PL's ingest chain, the
job seam, and Track I.H1's rate-limiter — all merged.

**Conflicts with other tracks (concurrent-file risk):**
| File | Also touched by | Mitigation |
|---|---|---|
| `packages/shared/src/shared/types/enums.py` (`JobType`) | OKF.2, V.5, Y.3, F.5 already landed | G reuses **existing** `JobType`s (`DOCUMENT_PARSE`, `INSIGHT_EXTRACT`, `ENTITY_EXTRACT`); adds none in G1/G2. A webhook job (deferred G5) would append one. |
| `services/command_service.py` (`_COMMAND_TO_JOB_TYPE`) | F.5, V.5, Y.3 landed | G1/G2 reuse existing commands; no new entry needed for the façade. |
| `api/app.py` (router registration + middleware) | every track adds a router | G.1 adds the agent sub-app mount + `agent_keys` router + one `excluded_paths` entry; small, additive. |
| `api/auth.py` (`excluded_paths`) | none active | G.1 appends `/api/v1/agents/*`; trivial. |
| `frontend/.../settings/page.tsx` | I (design tokens), Vault, Zotero | G.4 adds one tab; append-only. |
| `config.py` | I.H1, others | G appends agent/watcher env keys; additive. |
| Migration numbering | any track adding a migration | G.1 claims **76** on merge; rebase if raced (OVERWRITE guards make it safe). |

## 2. Decision gates (resolve before / during G.1)

Five of these **block G.1**; the rest have recommended defaults that proceed on
autopilot unless contested.

| ID | Question | Recommendation | Blocks |
|---|---|---|---|
| **G-D1** | How does `X-API-Key` coexist with `PasswordAuthMiddleware`? Agents cannot present the shared `OPEN_NOTEBOOK_PASSWORD`. | Mount the agent API as a **FastAPI sub-app** at `/api/v1/agents` (isolates its OpenAPI + `X-API-Key` security scheme), **add `/api/v1/agents/*` to `PasswordAuthMiddleware.excluded_paths`**, and make `require_agent_key` the **sole** authenticator for agent routes — it **always** enforces a valid key, independent of whether `OPEN_NOTEBOOK_PASSWORD` is set. Key-**management** routes (`/api/agent-keys`) stay under the existing session/bearer auth (humans manage keys from the authed UI). | **G.1** |
| **G-D2** | Key storage at rest. | Store a **SHA-256 hash** of the key (never the plaintext); return the plaintext **once** at mint time. Persist a non-secret display prefix (`onk_live_ab12…`) + `label`. Lookup hashes the presented header and matches. | **G.1** (table shape) |
| **G-D3** | Per-key rate-limit mechanism. Existing limiter is per-IP. | A **dedicated agent `Limiter`** keyed by the `X-API-Key` header (falling back to remote address), applied via a **dynamic per-key limit** read from the key record (`rate_limit_rpm`), default `AGENT_RATE_LIMIT_RPM` (60). Reuses slowapi + its `Retry-After` handling — no new infra. | **G.1** |
| **G-D4** | Audit-log shape. | New **append-only** `agent_audit_log` table (in migration 76): `{agent_key, agent_id, method, path, status, ts, latency_ms, job_id?}`, written by the auth dependency on **every** authenticated call. `GET /agents/audit-log` reads the calling key's own entries. **Full webhook/HMAC/DLQ audit is roadmap G5 (deferred).** Retention via `AGENT_AUDIT_RETENTION_DAYS` (90). | **G.1** (table shape) |
| **G-D5** | Raw-text `extract-entities` / `generate-summary` — the shipped services read chunks from the DB by `source_id`. | Add a **no-DB** `extract_from_text(text, ontology)` on `EntityExtractionService` and `summarize_text(text, strategy)` on `SummarizationService` that run the **existing** workflow over a single synthetic chunk and **write nothing** (no source lifecycle). Keeps "no file I/O" literal and the endpoint a pure function. | **G.1** (extract) / G.2 (summary) |
| G-D6 | `process-document` input: server-side path vs multipart upload. | **Both.** Path for co-located agents/file-watcher; multipart for remote agents. Reuse `enforce_upload_guards` + `_create_source_impl` unchanged; the agent supplies `notebook_id` + optional `transformations`. | no (G.3) |
| G-D7 | File-watcher deployment: in-process vs separate container. | **In-process** asyncio task started from the app-main FastAPI lifespan, **default-OFF** (`FILE_WATCHER_ENABLED`), reusing the job seam. A separate `services/file-watcher` container only if horizontal scaling is later needed (revisit; not now). | no (G.5) |
| G-D8 | Are roadmap G3/G4 (Obsidian sync), G5 (webhooks), G6 (templates) in this sprint? | **No — deferred to a follow-up sprint** (§ 9). This plan ships the headless core (G1 + G2) so agents can drive ingestion before the sync/webhook/template surface is built on top. | no (scope) |

## 3. Phases

Convention per `docs/tracks/README.md` (Backend → UI → Integration); each phase =
one PR, own branch `track/g-<phase>`, green tests + AC before the next. Ordered by
dependency; **G.1 is the thin, self-contained auth + scaffolding + first-endpoint
slice** the roadmap sequencing calls for.

**Phase list (one-line goals):**
- **G.1** — `agent_keys` + `agent_audit_log` tables (migration **76**) + key-management API (session-auth) + `X-API-Key` dependency + versioned `/api/v1/agents` sub-app + auto-OpenAPI + first endpoint `POST /agents/extract-entities` (raw text). *(Backend — recommended first PR.)*
- **G.2** — `POST /agents/generate-summary` (raw text + template) façade over the summarization service.
- **G.3** — Ingest façade: `process-document` / `process-url` / `process-audio` over the `process_source` job chain + `GET /agents/jobs/{job_id}` + `GET /agents/audit-log`.
- **G.4** — "API Keys" settings tab (mint / revoke / list + per-key audit-log view).
- **G.5** — File-watcher service (watchdog, debounce, scan-on-startup, type-routing, `_processed`/`_errors`) feeding the same ingest jobs.
- **G.6** — File-watcher UI/config: inbox paths + enable toggle + watcher status in settings.
- **G.7** — Integration: e2e agent flow + OpenAPI-client smoke + inbox-drop e2e + docs + RETRO.

---

### Phase G.1 — Agent auth foundation + first endpoint (Backend)

> **Status (2026-07-24): SHIPPED.** Auth model = the approved **Option A + two
> refinements** (G-D1): agent routes are key-gated by `require_agent_key`
> (fail-closed, password-independent), `/api/v1/agents/` is **prefix-excluded**
> from the shared-password middleware, and key management stays under it.
> Implementation refinement vs the plan: a **prefix-router + prefix-exclusion +
> audit middleware** instead of a mounted FastAPI sub-app — same security model,
> less wiring, no sub-app slowapi/middleware duplication. The per-key
> `rate_limit_rpm` OVERRIDE is deferred (slowapi evaluates before auth resolves
> the key); G.1 buckets per key at the config default. **Migration 77** (was 76;
> renumbered so it applies cleanly after the librarian-fix migration 76) ships
> both tables. A security bug caught in the first review — a naive prefix match
> treating the root `/` as a prefix would have disabled the whole password gate —
> is fixed. **Adversarial review round 1** then hardened it further: agent routes
> are gated at the ROUTER level (per-IP pre-auth throttle → read-scope key check,
> so no future route can be added ungated); `agent_audit_log.agent_key` is
> `option<>` so failed-auth rows are recorded not dropped; `authenticate` uses
> `revoked = false` (drift-safe, not `!= true`); `revoke` is table-scoped (no
> cross-table write). **Adversarial review round 2** then found the throttle fix
> was partial and hardened it (round 3 verified clean): the audit write is skipped
> on a 429 (else a throttled flood is write-amplification), the throttle prunes
> window-expired IPs (its self-prune was dead code), and `revoke` honours its
> idempotent bool contract on a malformed id.
>
> **Known trade-offs / follow-ups (from the review, accepted — not regressions):**
> - The single-mode `extract_from_text` uses the extractor's configured model, not
>   the privacy-routed caller (G-D5 raw-text path).
> - ALL 429s are excluded from the audit trail, including a per-key-limit 429 on a
>   *valid* key. Acceptable (a blocked request performs no action), but a future
>   refinement could audit per-key-limit 429s (authenticated-abuse signal) while
>   still skipping the pre-auth-throttle 429s (unauthenticated flood).
> - The per-IP throttle state is per-process in-memory (`time.monotonic`), so under
>   N uvicorn workers the effective cap is ~N× the configured RPM — same durability
>   as the existing per-IP slowapi limiter; a shared store (redis) would tighten it.
> - `run_deep` has a pre-existing superseded-run TOCTOU race (`latest()` → append);
>   low severity, untouched by the review fixes.

**Goal**: A versioned, API-key-authed agent surface that stands entirely on its own:
mint/revoke keys, authenticate with `X-API-Key` (permission-scoped, rate-limited,
audited), auto-generated OpenAPI, and **one real capability** —
`POST /api/v1/agents/extract-entities` over raw text — proving the whole path end to
end. No ingest coupling, no UI, no file I/O.

**Files to create**:
- `migrations/76.surrealql` + `migrations/76_down.surrealql` — two tables,
  OVERWRITE-guarded SCHEMAFULL (per migration 74/75 drift note):
  ```surql
  DEFINE TABLE OVERWRITE agent_keys SCHEMAFULL;
  DEFINE FIELD OVERWRITE agent_id     ON agent_keys TYPE string;
  DEFINE FIELD OVERWRITE key_hash     ON agent_keys TYPE string;                 -- SHA-256, never plaintext (G-D2)
  DEFINE FIELD OVERWRITE key_prefix   ON agent_keys TYPE string;                 -- non-secret display prefix
  DEFINE FIELD OVERWRITE label        ON agent_keys TYPE option<string>;
  DEFINE FIELD OVERWRITE permission   ON agent_keys TYPE string ASSERT $value IN ["read","write","admin"];
  DEFINE FIELD OVERWRITE rate_limit_rpm ON agent_keys TYPE option<int>;          -- per-key override; NONE -> config default
  DEFINE FIELD OVERWRITE revoked      ON agent_keys TYPE bool DEFAULT false;
  DEFINE FIELD OVERWRITE created      ON agent_keys TYPE datetime DEFAULT time::now();
  DEFINE FIELD OVERWRITE last_used_at ON agent_keys TYPE option<datetime>;
  DEFINE INDEX OVERWRITE ak_hash ON agent_keys FIELDS key_hash UNIQUE;
  DEFINE INDEX OVERWRITE ak_agent ON agent_keys FIELDS agent_id;

  DEFINE TABLE OVERWRITE agent_audit_log SCHEMAFULL;
  DEFINE FIELD OVERWRITE agent_key  ON agent_audit_log TYPE record<agent_keys>;
  DEFINE FIELD OVERWRITE agent_id   ON agent_audit_log TYPE string;
  DEFINE FIELD OVERWRITE method     ON agent_audit_log TYPE string;
  DEFINE FIELD OVERWRITE path       ON agent_audit_log TYPE string;
  DEFINE FIELD OVERWRITE status     ON agent_audit_log TYPE int;
  DEFINE FIELD OVERWRITE latency_ms ON agent_audit_log TYPE option<int>;
  DEFINE FIELD OVERWRITE job_id     ON agent_audit_log TYPE option<string>;
  DEFINE FIELD OVERWRITE ts         ON agent_audit_log TYPE datetime DEFAULT time::now();
  DEFINE INDEX OVERWRITE aal_agent ON agent_audit_log FIELDS agent_id;
  DEFINE INDEX OVERWRITE aal_ts    ON agent_audit_log FIELDS ts;
  ```
  Down: `REMOVE TABLE agent_audit_log; REMOVE TABLE agent_keys;` (faithful inverse — both brand new).
- `apps/app-main/src/app_main/api/agent_auth.py` — `require_agent_key(permission=...)`
  FastAPI dependency factory: read `X-API-Key`, SHA-256-hash, look up a non-revoked
  `agent_keys` row, enforce the permission scope (`read` < `write` < `admin`), stamp
  `last_used_at`, and attach the key context to the request. Plus `hash_key()` /
  `generate_key()` helpers. Mirrors `auth.py`'s location + style.
- `apps/app-main/src/app_main/api/agent_rate_limit.py` — a dedicated slowapi
  `Limiter` keyed by the `X-API-Key` header (fallback `get_remote_address`),
  `headers_enabled=True`; a `agent_key_limit()` dynamic-limit callable reading the
  per-key `rate_limit_rpm` (fallback `AGENT_RATE_LIMIT_RPM`). (G-D3.)
- `apps/app-main/src/app_main/services/agents/__init__.py`
- `apps/app-main/src/app_main/services/agents/key_service.py` — `AgentKeyService`:
  `mint(agent_id, permission, label, rate_limit_rpm) -> (record, plaintext)`,
  `revoke(key_id)`, `list()`, `authenticate(plaintext) -> record | None`.
- `apps/app-main/src/app_main/services/agents/audit_service.py` —
  `AgentAuditService.record(entry)` + `list_for_agent(agent_id, limit)`; a
  best-effort writer used by the dependency (an audit-write failure never fails the
  request).
- `apps/app-main/src/app_main/api/routers/agent_keys.py` — **session-auth** (existing
  bearer/UI) key management: `POST /api/agent-keys` (mint → returns plaintext once),
  `GET /api/agent-keys` (list, no secrets), `DELETE /api/agent-keys/{id}` (revoke).
- `apps/app-main/src/app_main/api/routers/agents.py` — the versioned agent router
  (mounted under the sub-app): `POST /extract-entities` (raw text + ontology →
  typed entities, `read` scope) and `GET /openapi.json` (auto). Every route depends
  on `require_agent_key` + the agent limiter; the audit write wraps the response.
- `packages/shared/src/shared/models/agents.py` — pydantic models: `AgentKeyCreate`,
  `AgentKeyCreated` (plaintext once), `AgentKeyPublic`, `AgentAuditEntry`,
  `ExtractEntitiesRequest{text, ontology_name}`, `ExtractEntitiesResponse{entities}`.
- `apps/app-main/tests/test_agent_auth.py` — dependency unit tests (see Tests).
- `apps/app-main/tests/test_agent_keys_api.py` — mint/list/revoke + auth-integration.
- `apps/app-main/tests/test_agent_migration.py` — migration 76 up/down roundtrip
  (`@requires_docker`).

**Files to modify**:
- `apps/app-main/src/app_main/api/app.py` — create the `/api/v1/agents` FastAPI
  sub-app, mount `agents.router` on it, `application.mount("/api/v1/agents", agent_app)`;
  register `agent_keys.router` under `/api`; wire the agent limiter onto the sub-app.
- `apps/app-main/src/app_main/api/auth.py` — append `/api/v1/agents/*` to
  `PasswordAuthMiddleware.excluded_paths` (G-D1).
- `apps/app-main/src/app_main/config.py` — `AGENT_RATE_LIMIT_RPM` (60),
  `AGENT_AUDIT_RETENTION_DAYS` (90).
- `apps/app-main/src/app_main/services/entity_extraction_service.py` — add
  `extract_from_text(text, ontology_name)` running the existing single-schema
  workflow over one synthetic chunk, **no DB writes** (G-D5).

**Acceptance criteria** (falsifiable):
1. Migration 76 applies (forward) and reverts (down) cleanly on a fresh
   testcontainer AND on a drifted DB (OVERWRITE idempotent) — CI roundtrip green;
   `agent_keys.key_hash` UNIQUE index present.
2. `POST /api/agent-keys` (under existing session auth) returns HTTP 201 with the
   plaintext key **exactly once**; a subsequent `GET /api/agent-keys` never returns
   the plaintext or `key_hash`, only `key_prefix` + metadata.
3. Keys are stored hashed: no row in `agent_keys` contains the plaintext (grep/DB
   assertion); `authenticate(plaintext)` matches by SHA-256 hash.
4. `POST /api/v1/agents/extract-entities` with a valid `read`-scope `X-API-Key`
   returns HTTP 200 with an `entities` array; the same call with **no** key → 401,
   with an **unknown/revoked** key → 401, and (once minted with `read` only) still
   succeeds because extract is `read` scope.
5. A `read`-scope key calling a `write`-scoped route (asserted with a stub
   `write`-guarded test route) → 403; an `admin` key passes all scopes.
6. The agent routes are reachable **regardless** of `OPEN_NOTEBOOK_PASSWORD`: with
   the password set, `/api/v1/agents/*` is excluded from the bearer middleware and
   still requires `X-API-Key` (a valid bearer alone → still 401 without a key).
7. Exceeding the per-key `rate_limit_rpm` (or the config default) → HTTP 429 with a
   `Retry-After` header; a second distinct key is unaffected (per-key, not per-IP).
8. Every authenticated agent call writes exactly one `agent_audit_log` row
   (`agent_id`, `method`, `path`, `status`); an audit-write failure does **not**
   fail the underlying request (fail-soft).
9. `GET /api/v1/agents/openapi.json` returns a valid OpenAPI document listing
   `extract-entities`, declaring the `X-API-Key` (apiKey/header) security scheme, and
   **not** leaking the internal `/api/...` routes.
10. `extract_from_text` performs no DB writes (asserted by a repo-write spy / no-source
    assertion) and returns entities for a seeded text fixture.

**Tests required**:
- Unit (`test_agent_auth.py`): hash round-trip; scope ordering (read<write<admin);
  missing/unknown/revoked key → 401; scope violation → 403; audit-write fail-soft.
- Integration (`test_agent_keys_api.py`, `@requires_docker`): mint → call
  extract-entities with the returned key → 200; revoke → 401; password-set exclusion
  (AC6); per-key 429 (AC7); audit row written (AC8).
- Migration roundtrip (`test_agent_migration.py`).

**PR boundary**: ONE PR titled `feat(agents): agent_keys + X-API-Key auth + extract-entities (G.1)`.
No UI, no ingest façade, no file-watcher.

**Effort estimate**: 4–5 days (roadmap G1 = 1–1.5 week total; this is the auth
foundation half). Reviewer cycle budget: ×1.5.

**Risk mitigations**: sub-app mount isolates the OpenAPI + security scheme and keeps
the parent app untouched except one mount + one exclude; hashed-at-rest keys avoid a
plaintext-leak footgun; audit write is best-effort so it can never brick a request.

---

### Phase G.2 — `generate-summary` over raw text (Backend)

**Goal**: The second text capability — `POST /api/v1/agents/generate-summary` (raw
text + template/strategy name → summary), a thin façade over the summarization
service, proving the `write`-scope + template path.

**Files to create**:
- `apps/app-main/tests/test_agent_generate_summary.py` — endpoint + service-method tests.

**Files to modify**:
- `apps/app-main/src/app_main/api/routers/agents.py` — add `POST /generate-summary`
  (`write` scope), audited + rate-limited like G.1.
- `apps/app-main/src/app_main/services/summarization_service.py` — add
  `summarize_text(text, strategy, config)` running `SummarizationWorkflow` over one
  synthetic `ChunkInput`, **no DB writes** (G-D5); validate the strategy against the
  existing `_IMPLEMENTED` set.
- `packages/shared/src/shared/models/agents.py` — add
  `GenerateSummaryRequest{text, strategy, config?}` / `GenerateSummaryResponse{summary}`.

**Acceptance criteria**:
1. `POST /agents/generate-summary` with a valid `write`-scope key + a supported
   `strategy` returns HTTP 200 with a `summary` string; an unknown strategy → 422
   naming the valid strategies.
2. A `read`-scope key → 403 (write-gated); no key → 401.
3. `summarize_text` writes nothing to the DB (repo-write spy) and appears in the
   `agent_audit_log` like every other call.
4. The endpoint is present in `GET /agents/openapi.json` with its request/response
   schema.

**Tests required**:
- Unit: `summarize_text` over a text fixture → non-empty summary; unknown strategy →
  ValueError → 422; no DB write.
- Integration: mint write key → call → 200; read key → 403.

**PR boundary**: ONE PR titled `feat(agents): generate-summary over raw text (G.2)`.
Assumes G.1 merged.

**Effort estimate**: 1.5–2 days. Reviewer cycle budget: ×1.0.

---

### Phase G.3 — Ingest façade + job status + audit-log read (Backend)

> **Status (2026-07-24): SHIPPED (PR #69, squash `1c662b3`).** `POST /process-url`
> (write, async job → real `job_id`) + `GET /jobs/{job_id}` (read, **ownership-bound**:
> a job the caller didn't enqueue → 404) + `GET /audit-log` (read, self-scoped;
> admin may read another agent's trail). `process-document` / `process-audio`
> (multipart upload + `enforce_upload_guards`) are **deferred to G.3b**. Converged
> clean over **5 adversarial-review rounds** (see follow-ups below).
>
> **Known trade-offs / follow-ups (from the review, accepted — not regressions):**
> - **SSRF on `body.url`** is guarded pre-flight (`_reject_ssrf_url`): non-http(s)
>   schemes + any host that NORMALIZES to loopback/private/link-local/reserved/
>   metadata, including numeric (decimal/octal/hex/short/trailing-dot) AND Unicode
>   homoglyph-dot / fullwidth / outlined-digit encodings. **Deferred to the shared
>   fetch-layer** (affects the password-gated UI equally): hostname→private-IP
>   **DNS-rebinding**, and a public URL that **30x-redirects** to a private target.
> - **Poll-authz is derived from the best-effort audit write** (`agent_owns_job`
>   reads the row `process-url` stamps). Fail-SAFE (a failed audit CREATE → owner
>   404s their own poll, never a leak). A dedicated non-audit ownership store would
>   decouple it — deployment follow-up.
> - `JobStatusResponse.error` now reads `error_message` (pre-existing null bug).

**Goal**: The file/URL/audio capabilities as a **thin façade over the shipped
`process_source` job chain**, plus job polling and the audit-log read endpoint.
This is where an agent actually ingests a document headlessly.

**Files to create**:
- `apps/app-main/tests/test_agent_ingest_facade.py` — enqueue-at-seam + status tests.

**Files to modify**:
- `apps/app-main/src/app_main/api/routers/agents.py` — add:
  - `POST /process-document` (`write`): accepts a server-side `path` **or** a
    multipart upload (G-D6) + `notebook_id` (+ optional `transformations`); calls the
    **same** `_create_source_impl` / `process_source` path as
    `sources_upload.create_source` (`type="upload"`), reusing `enforce_upload_guards`;
    returns `{job_id, status:"queued"}`.
  - `POST /process-url` (`write`): `{url, notebook_id}` → `type="link"` ingest → `{job_id}`.
  - `POST /process-audio` (`write`): audio path/upload → same chain (audio is a
    `process_source` input already); returns `{job_id}`.
  - `GET /jobs/{job_id}` (`read`): returns `CommandService.get_command_status(job_id)`
    verbatim (status + result + error).
  - `GET /audit-log` (`read`): the **calling** key's own `agent_audit_log` entries
    (paginated); an `admin` key may pass `?agent_id=` to read another agent's trail.
- `packages/shared/src/shared/models/agents.py` — add the ingest request/response +
  `JobStatusResponse` models (reuse the `get_command_status` dict shape).

**Acceptance criteria**:
1. `POST /agents/process-url` with a valid `write` key enqueues exactly one
   `process_source` job (asserted at the `submit_command_job` seam) and returns its
   `job_id`; the created source carries the supplied `notebook_id`.
2. `POST /agents/process-document` accepts **both** a server-side path and a multipart
   upload (G-D6); an oversize/over-paged upload is rejected by the reused
   `enforce_upload_guards` (413/422) **before** a job is enqueued.
3. `GET /agents/jobs/{job_id}` returns the same status payload as
   `CommandService.get_command_status` (job_id/status/result/error); an unknown id
   returns the `status:"unknown"` shape (200, not 500).
4. `GET /agents/audit-log` returns the calling key's own entries only; an `admin` key
   with `?agent_id=` can read another agent's; a `read` key with `?agent_id=` other
   than its own is scoped to itself (no cross-agent leak).
5. All four routes are `write`/`read`-scoped correctly (401/403 matrix) and appear in
   the OpenAPI document.
6. The façade adds **no** new `JobType` / `_COMMAND_TO_JOB_TYPE` entry — it reuses
   `process_source` (grep: no new enum member).

**Tests required**:
- Unit: enqueue-at-seam (mock `submit_command_job`) for url/document/audio; upload
  guard rejection before enqueue; job-status passthrough; audit-log scoping.
- Integration (`@requires_docker`): mint key → process-url → poll `jobs/{id}` →
  status transitions; audit-log lists the calls.

**PR boundary**: ONE PR titled `feat(agents): ingest façade + job status + audit-log (G.3)`.
Assumes G.1 merged (independent of G.2).

**Effort estimate**: 3–4 days. Reviewer cycle budget: ×1.5.

**Risk mitigations**: reuses the proven ingest path + guards + job seam (no parallel
pipeline); the only new logic is request-shape mapping + audit, both thin.

---

### Phase G.4 — "API Keys" settings tab (UI)

> **Status (2026-07-24): SHIPPED (PR #70).** Mint / list / revoke + a per-key
> audit-log drawer, with a show-once plaintext reveal. Added a session-authed
> backend read (`GET /api/agent-keys/{key_id}/audit-log`, resolves key→agent_id
> server-side) since the `X-API-Key` agent-router audit-log needs a key the
> operator doesn't hold. Adversarial review APPROVED (1 round): session-gated,
> no IDOR, parameterized SurrealQL, table-scoped, show-once never hits
> cache/DOM/logs; e2e drives revoked-key → 401.

**Goal**: A settings tab to mint, list, and revoke agent keys and view a key's
audit-log — the roadmap's "API Keys tab in settings". Mirrors `VaultSync.tsx` /
`ZoteroSettings.tsx`.

**Files to create**:
- `frontend/src/components/settings/ApiKeys.tsx` — key list (agent_id, label,
  prefix, permission, created, last-used, revoked badge), a "Generate key" dialog
  (agent_id + permission + optional label/rpm), a **show-once** plaintext reveal with
  copy-to-clipboard, per-row Revoke, and a per-key "View audit log" drawer.
- `frontend/src/lib/hooks/use-agent-keys.ts` — data hook over `/api/agent-keys` +
  `/api/v1/agents/audit-log` (or an `admin`-scoped audit read via the session API).
- `frontend/e2e/track-g/api-keys.spec.ts` — mint → reveal-once → revoke flow.

**Files to modify**:
- `frontend/src/app/(dashboard)/settings/page.tsx` — add an `<TabsTrigger value="api-keys">`
  + `<TabsContent>` mounting `<ApiKeys />`.

**Acceptance criteria**:
1. The tab renders all states: default (keys listed), loading, error, and empty
   ("No API keys yet"), not a blank.
2. "Generate key" mints a key and shows the plaintext **exactly once** in a dialog
   with a copy button and an explicit "you won't see this again" warning; closing the
   dialog and refetching never shows the plaintext again.
3. Revoke marks the key revoked (reflected on reload) and a revoked key can no longer
   authenticate (verified against the G.1 dependency in the e2e).
4. "View audit log" for a key lists its recent calls (method/path/status/ts).
5. Keyboard-accessible: Tab reaches Generate, each Revoke, and the audit drawer;
   dialog traps focus and closes on Esc; controls have `aria-label`s.

**Tests required**:
- Component unit: 4 render states; generate-dialog show-once; revoke optimistic
  update; keyboard focus order.
- E2E (`api-keys.spec.ts`): generate → copy plaintext → revoke → key rejected.

**PR boundary**: ONE PR titled `feat(frontend): API Keys settings tab (G.4)`.
Assumes G.1 (+ G.3 for the audit-log drawer) merged.

**Effort estimate**: 2–3 days. Reviewer cycle budget: ×1.5.

---

### Phase G.5 — File-watcher service (Backend)

> **Status (2026-07-25): SHIPPED (PR #73).** Opt-in (`FILE_WATCHER_ENABLED`,
> default OFF) `InboxWatcher` on watchdog, bridged onto the app loop; debounce +
> backlog scan + extension routing + copy-to-uploads + `process_source` enqueue +
> move to `_processed`/`_errors`. Converged over 2 review rounds (a MAJOR
> exactly-once hole — terminal-wait/move outside the guard → duplicate source on
> rescan — was found and fixed). 17 tests incl. a real-Observer integration test.

**Goal**: An always-on (opt-in) watcher on the conventional inbox paths that
debounces bursts, scans a startup backlog, routes by file-type, ingests via the
**same** `process_source` chain the API façade uses, and moves files to
`_processed/` on success or `_errors/` on failure.

**Files to create**:
- `apps/app-main/src/app_main/services/agents/file_watcher.py` — `InboxWatcher`
  built on `watchdog`: a debounced (2–5s, `INBOX_DEBOUNCE_SECONDS`) handler, a
  recursive `scan_on_startup()` for the backlog, a `_route(path)` mapping by
  extension (`.pdf/.docx/... → document`, `.mp3/.m4a/.wav → audio`,
  `.url/.webloc → url`), an ingest call reusing `_create_source_impl` / the
  `process_source` command, and the `_processed`/`_errors` move-after-terminal-state
  pattern. Resolves the target notebook from the path
  (`<notebook_data>/<notebook_id>/inbox/` → that notebook; the global
  `~/open-notebook/inbox/` → a configured default notebook).
- `apps/app-main/tests/test_file_watcher.py` — routing + debounce + move + backlog.

**Files to modify**:
- `apps/app-main/src/app_main/api/app.py` — start/stop the watcher from the FastAPI
  lifespan **only when** `FILE_WATCHER_ENABLED` (G-D7), alongside the job worker.
- `apps/app-main/src/app_main/config.py` — `FILE_WATCHER_ENABLED` (false),
  `INBOX_PATHS` (default `~/open-notebook/inbox`), `INBOX_DEFAULT_NOTEBOOK_ID`
  (for the global inbox), `INBOX_DEBOUNCE_SECONDS` (3).
- `apps/app-main/pyproject.toml` — add `watchdog>=4`.

**Acceptance criteria**:
1. Dropping a `.pdf` into a watched inbox enqueues exactly one `process_source` job
   (asserted at the `submit_command_job` seam) and, on terminal success, moves the
   file to `<inbox>/_processed/`; on failure, to `<inbox>/_errors/`.
2. File-type routing is correct: document/audio/url extensions map to the right
   ingest input; an unknown extension is ignored (not moved, not enqueued) and logged.
3. Debounce clusters a burst of writes to the same file into a single ingest (a rapid
   double-write does not enqueue twice).
4. `scan_on_startup()` ingests a pre-existing backlog file present before the watcher
   started, exactly once (idempotent against `_processed`).
5. A per-notebook inbox (`.../<notebook_id>/inbox/`) routes to that notebook; the
   global inbox routes to `INBOX_DEFAULT_NOTEBOOK_ID`.
6. With `FILE_WATCHER_ENABLED=false` (default) the watcher never starts (no behaviour
   change for existing deployments).

**Tests required**:
- Unit (`test_file_watcher.py`): extension→route table; debounce single-enqueue;
  move-to-`_processed`/`_errors` on mocked terminal states; backlog scan; disabled →
  no-op. Enqueue asserted at the `submit_command_job` seam (per the job-queue
  singleton test pattern).
- Integration (`@requires_docker`): drop a fixture file into a temp inbox → job
  enqueued → file moved.

**PR boundary**: ONE PR titled `feat(agents): inbox file-watcher auto-ingest (G.5)`.
Reuses the G.3 ingest path; land after G.3.

**Effort estimate**: 3–4 days (matches roadmap G2). Reviewer cycle budget: ×1.5.

**Risk mitigations**: default-OFF means zero impact until enabled; move-after-terminal
+ `_processed` idempotency prevents re-ingest loops; reuses the guarded ingest path so
oversize files are rejected the same way as the API.

---

### Phase G.6 — File-watcher config UI (UI)

**Goal**: Surface the watcher in settings — enable toggle, inbox path(s), default
notebook, and a simple last-activity/status readout.

**Files to create**:
- `frontend/src/components/settings/FileWatcher.tsx` — enable toggle, inbox path
  display + default-notebook select, and a recent-activity list (last N ingested /
  errored files, read from the audit log or a small status endpoint).

**Files to modify**:
- `frontend/src/app/(dashboard)/settings/page.tsx` — mount `<FileWatcher />` (either
  its own tab or within the API-Keys/agents tab).
- `apps/app-main/src/app_main/api/routers/agents.py` **or** a small settings route —
  a `GET` watcher-status endpoint (enabled? watched paths? last activity) if the audit
  log is insufficient for the "recent activity" readout.

**Acceptance criteria**:
1. The panel shows whether the watcher is enabled and which paths are watched; all
   states render (default/loading/error/empty).
2. The default-notebook select persists and is reflected on reload.
3. Recent-activity list shows the last processed/errored files with status.
4. Keyboard-accessible toggle + select with `aria-label`s.

**Tests required**:
- Component unit: enabled/disabled + activity states.
- E2E: open settings → watcher panel renders status.

**PR boundary**: ONE PR titled `feat(frontend): file-watcher settings panel (G.6)`.
Assumes G.5 merged.

**Effort estimate**: 1.5–2 days. Reviewer cycle budget: ×1.0.

---

### Phase G.7 — Integration: e2e + OpenAPI client + docs + RETRO (Integration)

**Goal**: End-to-end validation of the headless core and documentation close-out.

**Tasks**:
- E2E agent flow: mint a key (UI) → `extract-entities` (raw text) → `generate-summary`
  → `process-url` → poll `jobs/{id}` to completion → read `audit-log`.
- OpenAPI-client smoke: generate a client from `GET /api/v1/agents/openapi.json`
  (e.g. `openapi-generator`/`datamodel-code-gen`) and assert it round-trips one call —
  proves the spec is externally consumable.
- Inbox-drop e2e: drop a fixture file → confirm auto-ingest + `_processed` move.
- Update `docs/ARCHITECTURE.md` — add the agent API sub-app, `agent_keys` /
  `agent_audit_log`, the `X-API-Key` scheme, and the file-watcher.
- Update `docs/FEATURE_ROADMAP.md` — mark Track G (G1/G2) status; note G3–G6 deferred.
- `docs/tracks/G-agent-integration/status.md` + `RETRO.md`.
- Manual smoke checklist (key mint/reveal-once/revoke, 401/403/429 matrix, inbox drop).

**Acceptance criteria**:
1. The full agent flow works end-to-end on a dev container.
2. A client generated from the published OpenAPI spec successfully calls at least one
   endpoint (spec is valid + consumable).
3. `ARCHITECTURE.md` + `FEATURE_ROADMAP.md` reflect the shipped state (G1/G2 done,
   G3–G6 deferred).
4. Full-suite regression green; track marked accordingly in `status.md`.

**PR boundary**: ONE PR titled `docs(g): agent API + file-watcher integration + RETRO (G.7)`.
Assumes all backend + UI phases merged.

**Effort estimate**: 2–3 days. Reviewer cycle budget: ×1.0.

## 4. Risk assessment

- **Risk: the agent API bypasses or breaks the existing password auth.** Mitigation:
  sub-app mount + explicit `excluded_paths` entry; `require_agent_key` is the sole,
  always-on authenticator for agent routes; AC6 asserts the interaction with
  `OPEN_NOTEBOOK_PASSWORD` set and unset.
- **Risk: API key leakage (plaintext at rest / in list responses).** Mitigation:
  SHA-256 hash at rest (G-D2), show-once at mint, `GET` never returns the secret;
  AC2/AC3 assert it.
- **Risk: a rogue/looping agent exhausts the pipeline.** Mitigation: per-key
  slowapi rate-limit with `Retry-After` (G-D3, AC7) + the reused upload guards
  (I.H1) on the ingest façade.
- **Risk: the façade drifts into a parallel pipeline.** Mitigation: `process-*`
  reuses `_create_source_impl` / `process_source` and `jobs/{id}` reuses
  `get_command_status`; AC (G.3-6) forbids new `JobType`/command entries.
- **Risk: file-watcher re-ingest loops or double-enqueues.** Mitigation:
  move-after-terminal to `_processed`/`_errors` + debounce + startup-scan idempotency
  (G.5 AC1/AC3/AC4); default-OFF.
- **Risk: raw-text capabilities silently persist state.** Mitigation: `extract_from_text`
  / `summarize_text` are no-DB, asserted by repo-write spies (G-D5, G.1 AC10 / G.2 AC3).
- **Risk: migration 76 raced by a concurrent track.** Mitigation: OVERWRITE-guarded
  DEFINEs (idempotent on drift) + rebase-on-merge if the number is taken.

## 5. Test strategy summary

- **Unit test files to create**: `test_agent_auth.py`, `test_agent_keys_api.py`,
  `test_agent_generate_summary.py`, `test_agent_ingest_facade.py`,
  `test_file_watcher.py` (per-scenario positive+negative fixtures; `@requires_docker`
  where a seeded DB / real ingest is needed; enqueue asserted at the
  `submit_command_job` seam per the job-queue-singleton pattern).
- **Migration roundtrip**: `test_agent_migration.py` (76, CI-gated forward/down).
- **Integration**: mint→auth→extract/summary; ingest façade enqueue + `jobs/{id}`
  passthrough; audit-log scoping; per-key 429; inbox drop → job → move.
- **E2E (Playwright)**: `api-keys.spec.ts` (G.4), file-watcher settings (G.6),
  full agent flow + OpenAPI-client smoke (G.7).

## 6. Open questions (escalate to user before G.1)

- **G-D1 (blocks G.1)**: confirm the sub-app mount + `PasswordAuthMiddleware`
  exclusion of `/api/v1/agents/*` + `require_agent_key` as the sole agent
  authenticator (key **management** stays under session auth).
- **G-D2 (blocks G.1)**: confirm SHA-256-hash-at-rest + show-once plaintext + display
  prefix for `agent_keys`.
- **G-D3 (blocks G.1)**: confirm the dedicated per-key slowapi limiter + default
  `AGENT_RATE_LIMIT_RPM=60`.
- **G-D4 (blocks G.1)**: confirm the append-only `agent_audit_log` shape + retention
  default (full webhook/HMAC audit deferred to G5).
- **G-D5 (blocks G.1)**: confirm the no-DB `extract_from_text` / `summarize_text`
  approach over the create-ephemeral-source alternative.
- G-D6 (G.3, default): confirm `process-document` accepts both server-path and
  multipart upload.
- G-D7 (G.5, default): confirm in-process, default-OFF file-watcher over a separate
  container; and the global-inbox default-notebook config.
- **G-D8 (scope)**: confirm G3/G4 (Obsidian sync), G5 (webhooks), G6 (templates) are
  deferred to a follow-up sprint (§ 9), not this one.

## 7. Dependency map

```
G.1 (Backend: tables 76 + auth dep + sub-app + extract-entities)   ← recommended FIRST PR
 ├─→ G.2 (Backend: generate-summary)        — requires G.1 auth + router
 ├─→ G.3 (Backend: ingest façade + jobs + audit-log read)  — requires G.1
 │      └─→ G.5 (Backend: file-watcher)      — reuses G.3 ingest path
 │             └─→ G.6 (UI: watcher settings)
 └─→ G.4 (UI: API Keys tab)                  — requires G.1 (+ G.3 for audit drawer)

G.7 (Integration + docs + RETRO)             — requires all above
```

Cross-track:
- **This track depends on**: none hard (Track PL ingest chain, the job seam, and
  Track I.H1 rate-limiter are all merged).
- **This track blocks**: Track **H** (vision parser) — DEFERRED until G is complete.
  The deferred G3/G4 (Obsidian sync) build on the existing `vault_sync_service.py` /
  `obsidian_export_service.py` and this track's ingest façade.

## 8. Ordering & effort

| Phase | Title | Effort (days) | PR | Migration |
|---|---|---|---|---|
| G.1 | Agent auth foundation + extract-entities | 4–5 | 1 | **76** |
| G.2 | generate-summary over raw text | 1.5–2 | 1 | — |
| G.3 | Ingest façade + job status + audit-log | 3–4 | 1 | — |
| G.4 | API Keys settings tab | 2–3 | 1 | — |
| G.5 | File-watcher service | 3–4 | 1 | — |
| G.6 | File-watcher config UI | 1.5–2 | 1 | — |
| G.7 | Integration + docs + RETRO | 2–3 | 1 | — |
| **Total (critical path)** | | **~18–23 days** | **7 PRs** | |
| **With ×1.5 reviewer budget** | | **~27–35 days** | | |

**Recommended starting PR**: **G.1** — the self-contained auth + versioned-router +
first-endpoint slice (`agent_keys`/`agent_audit_log` on migration **76**, `X-API-Key`
dependency, auto-OpenAPI, `extract-entities` over raw text). It lands the auth
scaffolding all later phases depend on, with no ingest/UI/file-watcher coupling.

## 9. Deferred to a follow-up sprint (roadmap G3–G6)

Out of this plan's detailed decomposition (G-D8); each is a sprint of its own once
the headless core (G1/G2) is live. Summary so the roadmap-to-plan mapping is complete:

| Roadmap | Scope | Notes / dependencies | Effort (roadmap) |
|---|---|---|---|
| **G5 — Webhooks + robust audit** | `POST /agents/webhooks` (register URL + event types), outbound `job.complete`/`job.failed`/… delivery with **HMAC signing**, exponential-backoff retries + dead-letter queue. | New `agent_webhook` table (next migration after 76) + a delivery `JobType`. Builds on the G.1 `agent_audit_log`. | ~1 week |
| **G3 — ObsidianSync (write side)** | Template-driven vault export per Source (`literature_note`, `meeting_notes`), atomic writes. | Builds on existing `obsidian_export_service.py` + G6 templates. | ~1 week |
| **G4 — ObsidianSync (read side)** | Vault→DB sync, content-hash diff, re-extraction on body change, non-blocking conflict-resolution UI. **The hardest component.** | New `sync_state` table; reuses F1 audit on new state; extends `vault_sync_service.py`. | ~2 weeks |
| **G6 — Summary templates** | `literature_note` + `meeting_notes` YAML-driven templates with Writer-Evaluator-Editor enhancer. | `pipelines/summarization/.../templates/*.yaml`; feeds G2/G3. | ~1–1.5 week |
