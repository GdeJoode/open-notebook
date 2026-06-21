# THINKING — Docker live-test PDF extraction vs own ground-truth

## Goal
Get the full app running via Docker, then validate ingestion QUALITY: for each of the 3 `tests/pdfs/` PDFs, compare the app's output (markdown, entities, relations) against a ground-truth I build myself first. "Found info" must overlap ≥80% (recall of my ground-truth salient items). Exact 1:1 match NOT required.

## Constraints
- App runs the NEW Track-I code only after the `open_notebook` image is rebuilt (in progress).
- Ingestion (docling parse + chunk) and KG extraction (entities/relations) are SEPARATE steps; both must run.
- Two LLM extractions are being compared (mine vs the app's) → fuzzy/semantic matching, not string equality.
- All runs go through WSL venv / Docker per `context.md`.

## Top 3 risks
1. **Build/stack health** — 1GB+ context bloat slows the build; `open_notebook` depends_on `docling` (GPU) so the app won't come up unless docling starts. Mitigation: wait for build; bring up surrealdb+docling+open_notebook; verify health + migrations before anything else.
2. **Model/config mismatch** — the app's KG extraction must use a pulled Ollama model + an ontology; if mis-set, entities/relations come back empty → 0% match (a config problem, not a quality problem). Mitigation: verify the app's configured models/ontology via the API/settings before judging the 80%; pull/select a present model if needed.
3. **80% metric is fuzzy** — comparing two LLM extractions. Mitigation: define recall of MY ground-truth's salient entities/relations (case-insensitive + alias/semantic fuzzy match), and content-coverage for markdown; report the metric transparently with the matched/missed lists, not just a number.

## Non-obvious dependencies
- Ground-truth (phase 2) should be built BEFORE seeing the app's output, so it's a fair independent baseline (the user said "maak eerst zelf"). Build it from the PDFs directly (docling skill / Read-PDF / my reading).
- KG extraction depends on ingest completing (chunks must exist first).
- The comparison depends on both the app output AND the ground-truth existing.

## Tools/skills relied on
- `docling` skill or `firecrawl:firecrawl-parse` or the Read tool (reads PDFs) for my ground-truth markdown.
- WebSearch (available) to sanity-check entity facts if needed.
- Docker + curl for stack + API; WSL venv for any scripts.

## Best practices applied
- Independent ground-truth first (no anchoring on app output).
- Measurable recall threshold with matched/missed lists (falsifiable).
- Distinguish "config/plumbing failure" (fixable) from "genuine quality gap" when a score is low.

## Execution note
This task is tightly coupled to THIS session's live state (running build `bs69d1v42`, env quirks, my extraction tools + judgment). Recommended execution is **inline** (I continue, guided by the ROADMAP), not a fresh `/goal` session that would lose the live build context. Phase specs + this context.md still make a `/goal` handoff possible if preferred.
