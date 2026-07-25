# Track G — Agent Integration & Headless Mode — Status

**State: SHIPPED (2026-07-25)** — the headless agent core (G.1–G.6) is merged;
G.7 is this integration + docs close-out. Track G makes every meaningful
capability callable by external agents (hermes, claude-code, …) via a versioned,
API-key-authed REST surface, plus an opt-in inbox file-watcher.

| Phase | Status | PR |
|---|---|---|
| G.1 — Agent auth foundation + `extract-entities` | ✅ | (G.1 stack) |
| G.2 — `generate-summary` over raw text | ✅ | #68 |
| G.3 — Ingest façade: `process-url` + `jobs/{id}` + `audit-log` | ✅ | #69 |
| G.3b — Multipart upload: `process-document` + `process-audio` | ✅ | #71 |
| security — upload filename path-traversal fix (found in G.3b review) | ✅ | #72 |
| G.4 — "API Keys" settings tab | ✅ | #70 |
| G.5 — Inbox file-watcher (opt-in, default OFF) | ✅ | #73 |
| G.6 — File-watcher settings panel (read-only) | ✅ | #74 |
| G.7 — Integration: OpenAPI spec test + docs + RETRO | ✅ (this) | — |

See `RETRO.md` for the deliberate deviations and the open follow-ups.

## Surface (what an agent can call)

Router `/api/v1/agents/*`, gated by a per-IP pre-auth throttle → `X-API-Key`
(`require_agent_key`, fail-closed, scope read<write<admin) → per-key rate-limit,
with an append-only `agent_audit_log`:

- `POST /extract-entities` (read) — typed entities from raw text, no DB.
- `POST /generate-summary` (write) — summarize raw text, no DB.
- `POST /process-url` (write) — headless URL ingest → pollable `job_id`.
- `POST /process-document` · `POST /process-audio` (write) — multipart upload ingest.
- `GET /jobs/{job_id}` (read) — poll a job the caller enqueued (ownership-bound → 404 otherwise).
- `GET /audit-log` (read) — the caller's own call trail (admin may read another's).
- `GET /openapi.json` — the agent surface's spec, for external client generation.

Key management (session-authed, under `/api`, for the operator UI):
`POST/GET/DELETE /api/agent-keys` + `GET /api/agent-keys/{id}/audit-log`;
`GET /api/watcher/status`.

## Migrations consumed

- **76** — `librarian_enabled` on `notebook` (Track F, merged in this window).
- **77** — `agent_keys` + `agent_audit_log` (G.1).

No further migrations — the watcher is filesystem + env driven.

## Deferred / follow-ups (see RETRO)

- **Runtime-reconfigurable watcher** — config is env-managed (read at startup);
  the G.6 panel is read-only by design. An effective toggle needs the watcher on
  `app.state` + start/stop from a settings write.
- **Shared fetch-layer SSRF** — hostname→private-IP DNS-rebinding and
  redirect-to-private are out of the `process-url` pre-flight guard's reach (they
  affect the password-gated UI equally).
- **`transformations` on async ingest** — accepted + validated by the façade but
  not forwarded into the async `process_source` payload (pre-existing; shared with
  the UI async path).
- **Live e2e** — the mint→extract→summarize→process→poll→audit flow and the
  inbox-drop flow are authored as Playwright specs (`frontend/e2e/track-g/`) that
  run against a live stack; they were not executed in this dev environment.
- **Watcher recent-activity scan** — O(N) over `_processed`/`_errors`; fine for a
  single-tenant trickle inbox, a bounded read would harden a huge backlog.
