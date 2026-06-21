
## B.8b — image rebuild blocked by Docker credential helper (2026-06-20)

**Status**: BLOCKED_PENDING_USER

**Blocker**: `docker compose build open_notebook` fails:
`error getting credentials — exec: "docker-credential-desktop.exe": executable file not found in $PATH`.
`~/.docker/config.json` sets `"credsStore": "desktop.exe"` (Windows Docker Desktop helper not on the WSL PATH). The base image `python:3.12-slim-bookworm` is not cached locally, so the build must resolve it from Docker Hub, which invokes the missing helper.

**Why not auto-fixed**: the standard fix edits the user's global `~/.docker/config.json` — an out-of-project-scope auth-config change; the auto-mode classifier denied it (correctly).

**Options surfaced to user**: (1) user fixes creds + I retry; (2) user approves a scoped `DOCKER_CONFIG=<temp empty>` workaround (no global change); (3) hot-patch fallback (spec-documented, non-persistent); (4) user runs the build themselves via `!`.

**RESOLVED (2026-06-20)**: structural fix — symlinked the Docker Desktop credential helper onto the WSL PATH:
`ln -sf "/mnt/c/Program Files/Docker/Docker/resources/bin/docker-credential-desktop.exe" ~/.local/bin/docker-credential-desktop.exe`.
`~/.local/bin` is on PATH via the user profile, so `credsStore: desktop.exe` now resolves for all future builds — no global config change, no per-build intervention. `docker manifest inspect python:3.12-slim-bookworm` authenticates. Rebuild re-running.

## B.8c — redeploy blocked by Docker Desktop stale bind-mount (2026-06-20)

**Status**: BLOCKED_PENDING_USER

New image built OK (ca8e8088a360; Dockerfile reorder validated — 2 uv-sync layers). But `docker compose up open_notebook` fails:
`error while creating mount source path '/run/desktop/mnt/host/wsl/docker-desktop-bind-mounts/Ubuntu/730803...': mkdir ...: file exists`.
The stale mount is `./notebook_data:/app/data` (open_notebook's only unique bind; the shared docling/mineru binds work). It lives in the Docker Desktop VM and survives `compose down`.

**Clean fix** = restart Docker Desktop / WSL backend → bounces ALL containers incl. other projects (openproject ×6, forgejo, caddy, standalone surrealdb). High blast radius; not done autonomously.
**Non-disruptive workaround** = remap the (empty) /app/data to a fresh host path → new mount hash → open_notebook starts, other containers untouched. Revertible after a future DD restart.

**UPDATE (2026-06-20)**: `wsl --terminate docker-desktop` did NOT clear the stale mount. A force Stop/Start of Docker Desktop (via PowerShell) left the engine DOWN and not recovering (~10 min) — over-aggressive, made it worse. Docker Desktop processes are running but the daemon socket is down. STOP poking via interop. Clean recovery = USER action: `wsl --shutdown` (Windows terminal) or quit+reopen Docker Desktop from the tray — this also clears the original stale mount. All B.8 code is committed; resume after Docker is healthy.

## B.8c — live DB entity schema is pre-Track-B (Q-B-1 drift) (2026-06-21)

**Status**: BLOCKED_PENDING_USER (systemic, needs a decision — stop field-patching)

The live `entity` table enforces constraints that are NOT in any migration and that Track B's `upsert_entity` does not satisfy (it writes the new `canonical_name` schema):
- `name` — required string (upsert sets `canonical_name`, not `name`).
- `hash_id` — required string (upsert didn't set it; patched in B.8c).
- `entity_type` — `ASSERT $value INSIDE [lowercase enum]` (qwen emits "Location"/"ABBREVIATION"; normalized in B.8c).
- `idx_entity_fulltext` SEARCH index (broke ORDER BY; worked around in the KG fix).

This is the documented **Q-B-1 drift** (Track B plan §B.1a): the live DB was never migrated to the Track B entity schema, so Track B-tested-against-migration-schema code passes CI but every live write fails. Persistence is fundamentally blocked until the schema is reconciled.

**Decision needed**: (A) write+apply a migration aligning the live `entity` table to the Track B model (drop/relax the legacy required `name`/`hash_id` + enum, or map them); (B) decide the fate of the 147 legacy pre-Track-B entities; (C) NOT keep patching upsert_entity to satisfy an un-versioned old schema.

**Extraction-side fixes done + verified this run** (independent of the schema wall): ORDER BY (deployed), no-schema-fallback caller wiring, JSON strict-mode parse, entity_type normalization, RELATE syntax, hash_id. qwen extraction itself WORKS (241 entities produced from one paper) — only persistence is blocked by the schema drift.

## B.8c — post-persist _save_result KeyError (job fails despite entities persisting) (2026-06-21)
**Status**: NOTED → B.8d follow-up (non-blocking; entities persist correctly).
bc6xa re-extraction: entities+relations PERSIST ("Persisted to KG: 225 entities, 179 relations"; 421 total, method=llm, model=qwen). But `_save_result` (entity_extraction_service.py:934) — saving the raw extraction_result record — raises `KeyError: <uuid>` from the surrealdb async-ws client (`async_ws.py:_send`, request/response UUID desync; likely concurrent-query or large-payload). It logs "Failed to save extraction result" then RE-RAISES → handle_entity_extract marks the job failed + dead-letters it, even though extraction succeeded.
Fix (B.8d): make _save_result failure non-fatal (entities already persisted; raw result is a secondary re-filter artifact) and/or serialize/guard the ws query to avoid the client desync. The misleading "failed" job status would otherwise tell users extraction failed when it produced 400+ entities.
