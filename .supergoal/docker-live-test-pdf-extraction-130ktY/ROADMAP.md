# Roadmap: Docker live-test — PDF extraction vs own ground-truth

**Task:** Run the app via Docker, then for each of the 3 `tests/pdfs/` PDFs build my own ground-truth (markdown + concepts/entities/relations) and verify the app's ingested output overlaps it ≥80% (recall of salient items; 1:1 not required).
**Type:** brownfield · infra + validation
**Created:** 2026-06-20
**Total phases:** 5

## Context summary
- Stack: Docker Compose (SurrealDB + open_notebook[API 5055 / frontend 8502] + docling/mineru/whisperx/summarization/extraction). Ollama on host (all needed models pulled). GPU OK.
- Env quirks captured in `context.md` (WSL venv, rev-parse glitch, Ollama, model list, KG-extraction is a separate step).
- Build of the new `open_notebook` image is in progress (bg task `bs69d1v42`).

## Assumptions (correct any that are wrong)
- **80% metric = recall of MY ground-truth's salient items**: ≥80% of my key entities and ≥80% of my key relations are present in the app's output (case-insensitive + alias/semantic fuzzy match), plus markdown content-coverage (≥80% of my major sections/headings present). Precision (app extras) is reported but not gated — the user said exact 1:1 isn't required.
- My ground-truth is built independently from the PDFs (docling skill / Read-PDF), BEFORE inspecting app output.
- KG extraction uses the app's configured Ollama model + an ontology; if the app returns empty entities due to config, that's fixed (not counted as a quality miss) before scoring.
- Ingestion via the app's API (`POST /api/sources`), KG extraction via the `knowledge_graph` API; full GPU stack up (docling needed for parse).

## Risk top 3
1. Stack health (1GB context build; open_notebook depends_on docling/GPU) → wait for build, bring up + verify health/migrations first.
2. Model/ontology config → verify app's KG model/ontology before scoring; pull/select a present model if needed.
3. Fuzzy 80% on two LLM extractions → recall metric with explicit matched/missed lists, transparently reported.

## Phase map
| # | Phase | Depends on | Deliverable |
|---|-------|-----------|-------------|
| 1 | Bring up Docker stack | — | All services healthy; API+frontend reachable; migrations applied |
| 2 | Build own ground-truth | — | `ground-truth/<pdf>.md` + entities/relations per PDF (3 sets) |
| 3 | Ingest + KG-extract via app | 1 | App markdown + entities + relations exported per PDF (3 sets) |
| 4 | Compare & score ≥80% | 2,3 | `comparison/<pdf>.md` with recall metrics; all ≥80% |
| 5 | Report & harden | 1-4 | Consolidated report; mismatches + any config fixes documented |

---

## Phase 1 — Bring up Docker stack
**Why:** Nothing can be live-tested until the rebuilt app + parsers + DB are up and healthy.
**Deliverables:** running stack; `.supergoal/docker-live-test-pdf-extraction-130ktY/phase1-health.txt` capturing `docker compose ps` + health probes.
**Acceptance criteria:**
- `open_notebook` image rebuilt from current main (image CreatedSince < this run; not "2 weeks ago").
- `docker compose up -d` brings up surrealdb + docling + open_notebook (+ mineru/extraction/summarization as needed for ingest); `docker compose ps` shows them Up/healthy.
- API health: `curl -s http://localhost:5055/health` (or `/api/...`) returns 200/healthy.
- Frontend: `curl -s -o /dev/null -w '%{http_code}' http://localhost:8502` returns 200.
- DB migrations applied on startup with no error (app log shows migrations completed; migration 49 / I.F present) — captures I.F's deferred AC1.
**Mandatory commands:** `docker compose up -d`, `docker compose ps`, `curl -s -o /dev/null -w '%{http_code}' http://localhost:8502`, `docker logs open-notebook-open_notebook-1 --tail 50`
**Evidence required:** `docker compose ps` output; API + frontend HTTP codes; the migration log line.

## Phase 2 — Build own ground-truth (independent baseline)
**Why:** The user asked me to first produce my own MD + concepts/entities/relations to test against — built independently so it's a fair baseline.
**Deliverables:** for each of the 3 PDFs: `.supergoal/.../ground-truth/<slug>.md` (markdown) + `<slug>.entities.json` + `<slug>.relations.json` (concepts/entities + subject–predicate–object relations).
**Acceptance criteria:**
- 3 markdown files produced from the PDFs (via docling skill / Read-PDF), each with the document's section structure + body text.
- Each PDF has an entities list (key concepts/people/orgs/places/terms) and a relations list (≥10 entities and ≥5 relations per doc, or all that genuinely exist if fewer).
- Ground-truth is built WITHOUT inspecting the app's output (independence noted in the file header).
**Mandatory commands:** (ground-truth tooling — e.g. docling skill invocation / `Read` the PDFs; no single fixed shell command)
**Evidence required:** the 3 file sets exist; a count of entities/relations per PDF printed.

## Phase 3 — Ingest + KG-extract via the app
**Why:** Produce the app's own markdown + entities + relations for the same PDFs through the real pipeline.
**Deliverables:** per PDF: app markdown + app entities + app relations pulled from the API/DB into `.supergoal/.../app-output/<slug>.*`.
**Acceptance criteria:**
- Each PDF uploaded via `POST /api/sources` (type=upload); processing completes (source status done; chunks persisted).
- KG entity/relation extraction triggered (knowledge_graph API) and completes; entities + relations persisted.
- The app's markdown (full_text), entities, and relations are exported for each PDF.
- If extraction returns empty due to model/ontology config, the config is corrected (documented) and re-run before declaring phase done.
**Mandatory commands:** `curl` against `http://localhost:5055/api/sources` (upload), the knowledge-graph + sources endpoints to read back results; `docker logs` to watch processing.
**Evidence required:** per-PDF source id + status; counts of chunks/entities/relations the app produced.

## Phase 4 — Compare & score ≥80%
**Why:** The core acceptance — app output vs my ground-truth, ≥80% recall of salient info.
**Deliverables:** `.supergoal/.../comparison/<slug>.md` per PDF + a summary table.
**Acceptance criteria:**
- For each PDF: entity recall (matched / my-ground-truth, fuzzy) ≥ 0.80; relation recall ≥ 0.80; markdown content-coverage ≥ 0.80.
- Each comparison file lists matched + missed items (not just a number) so the score is auditable.
- Where < 80%, root-cause is stated (genuine miss vs config/plumbing) and, if a quick config fix lifts it ≥80%, applied + re-scored.
**Mandatory commands:** comparison done with judgment + small scripts (WSL venv python) for fuzzy matching; `cat` the comparison files.
**Evidence required:** per-PDF recall numbers + the matched/missed lists; the summary table showing all three PDFs ≥80% (or an honest account if a metric can't reach 80% and why).

## Phase 5 — Report & harden
**Why:** Consolidate so the user can act; capture fixes + residual gaps.
**Deliverables:** `.supergoal/.../REPORT.md`.
**Acceptance criteria:**
- One report: per-PDF scores, what matched/missed, any config fixes made, how to reproduce, and the URLs (http://localhost:8502) for the user to eyeball.
- Stack left running for the user's own live testing.
- Any genuine quality gaps (< 80% that couldn't be fixed) flagged with a hypothesis + next step.
**Mandatory commands:** `docker compose ps` (confirm still up).
**Evidence required:** REPORT.md written; final stack status.
