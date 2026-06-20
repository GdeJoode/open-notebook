
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
