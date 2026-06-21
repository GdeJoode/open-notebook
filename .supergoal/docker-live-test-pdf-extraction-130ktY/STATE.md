# State: Docker live-test — PDF extraction vs own ground-truth

**Status:** IN_PROGRESS (inline execution)
**Started:** 2026-06-20
**Run root:** .supergoal/docker-live-test-pdf-extraction-130ktY

## Phase progress
| # | Phase | Status | Notes |
|---|-------|--------|-------|
| 1 | Bring up Docker stack | in progress | `.dockerignore` fixed (excluded .claude/.supergoal/.serena + parser output dirs — context was 5GB+→small); open_notebook rebuild running (bg b2bvsya69, at uv-sync ML deps). SurrealDB already up. |
| 2 | Build own ground-truth | DONE | 9 files in ground-truth/ — economics (42 ent/27 rel), jcms-cohesion (50/25), centrifugal-state (40/22). |
| 3 | Ingest + KG-extract | pending | needs stack up |
| 4 | Compare & score ≥80% | pending | — |
| 5 | Report & harden | pending | — |

## Notable findings
- 2026-06-20 — PDF #2 (`J of Common Market Studies…Ali…`) is MISLABELED: actual content is "The Centrifugal State" (Rodríguez-Pose) = same paper as PDF #3. Ground-truth built on real content. App output for #2 should be compared vs the Centrifugal-State ground-truth.
- 2026-06-20 — All 3 PDFs are owner-password-restricted (empty user pw). The Read TOOL refuses them, but `pypdfium2` (the app's lib) opens all 3 fine → NOT a blocker for ingestion; no decryption needed.
- 2026-06-20 — Build was pathological: `.dockerignore` lacked `.claude/` → 5GB+ context (agent worktrees w/ venvs). Fixed. (`.dockerignore` change should be committed to main as a real fix.)

## Env
See context.md. main=1a07c91 at start. WSL venv restored. Ollama up w/ all needed models. GPU OK.
