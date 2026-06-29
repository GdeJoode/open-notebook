# Track X — Citations to exact source (Docling provenance → answers)

> **Feature 4** of the Constella adoption plan (`docs/constella-features-adoption-research.md`).
> Provenance (file, page, chunk, section) is already STORED on the `chunk` table by the Docling
> ingest, but it is NOT threaded into generated answers. This track passes it through retrieval →
> the answer graphs → a structured citation on every answer. No re-extraction; reuse stored provenance.

## Decisions (locked 2026-06-29)
1. **Structured citations** on answers: `citations: [{source, page, chunk_id, section?}]` (additive to the
   existing answer shape — existing consumers unaffected).
2. **Faithfulness = precision over coverage.** A cited `chunk_id` MUST be one that was actually in the
   retrieval set for that answer; hallucinated citations are dropped/flagged. (Membership check — not a
   semantic-support claim; note that limitation.) Same precision-first discipline as U.3 `cites`.
3. **Reuse stored provenance** — `chunk.physical_page`/`printed_page`/`section_path`/`positions`/`source`
   (`migrations/10.surrealql`); no new extraction. Handle sources lacking a page (`None`) gracefully.

## Existing pieces this track wires (verified — research Area A)
- `chunk` table carries provenance: `physical_page`, `printed_page`, `section_path`, `positions` (bbox), `element_type`, `source`.
- Answer generation: `apps/app-main/src/app_main/graphs/ask.py` (`provide_answer` retrieves via `RetrievalService`, stores `ids` in state but does NOT put chunk/page provenance in the prompt or output) + `source_chat.py` (`ContextService.build_source_context` — source-level `full_text`, no chunk granularity).
- Retrieval: `pipelines/retrieval/src/retrieval/service.py` (`RetrievalService.{text,vector,hybrid}_search`) ↔ `SearchRepository` — the search rows already contain the chunk fields; they're dropped before the prompt.

**Workflow**: track methodology — `implementer` → `adversarial-reviewer` (≤3 → `escalation-handler`).
Main tree, `uv run pytest`, no worktree. Additive/backward-compatible; live writes gated; read-only against `staging` (`SURREAL_DATABASE=staging`).

---

## Phase X.1 — Thread chunk provenance through retrieval (Backend)
**Why**: the answer graph can't cite what retrieval doesn't hand it; today the chunk metadata is dropped.
**Deliverables**: have `RetrievalService.{vector,hybrid,text}_search` surface per-hit provenance
(`chunk_id`/`source`/`physical_page`/`printed_page`/`section_path`/`element_type`) alongside the existing
fields, sourced from the rows the `fn::` functions already return (confirm what's present; add to the SELECT
only if a field is genuinely missing). Additive — existing callers that ignore the new keys are unaffected.
**Acceptance**
1. Each retrieval hit carries the provenance keys (when the underlying chunk has them; `None`/absent handled).
2. Existing callers/tests unchanged (backward-compat; the search/retrieval suites stay green).
3. Unit tests for the surfaced shape + a read-only staging probe showing real page/section on a known source.
**Branch**: `track/x1-retrieval-provenance`. **Depends on**: none.

## Phase X.2 — Cited answers in the ask + source_chat graphs (Backend / LangGraph)
**Why**: deliver the feature — answers that cite the exact source/page/chunk.
**Deliverables**: in `ask.py` (`provide_answer`) and `source_chat.py`, preserve per-hit provenance in the
graph state; inject it into the context/prompt so the model can attribute claims (`[source, p.X]`); and emit
a structured `citations: [{source, page, chunk_id, section?}]` alongside the answer text. `source_chat` gains
chunk-level context (page refs) rather than only `full_text`. Prompt templates updated
(`ask/query_process`/`ask/final_answer` or the source-chat template) to request grounded attribution.
**Acceptance**
1. An `ask` answer carries a `citations` array; each citation's `page` matches the cited chunk's
   `physical_page`; `source` is the real source id.
2. `source_chat` answers cite chunk-level provenance (not just the source title).
3. Existing chat/ask behavior intact (answer text still produced; the citation field is additive); graph tests green.
4. Tests with a seeded source+chunks (known pages) asserting the emitted citations match.
**Branch**: `track/x2-cited-answers`. **Depends on**: X.1.

## Phase X.3 — Faithfulness guard + integration + docs + RETRO (Integration)
**Why**: stop hallucinated citations; close the track.
**Deliverables**: a light post-generation guard that each emitted `chunk_id` was in the retrieval set for
that answer (drop/flag the rest — precision-first); ARCHITECTURE note on the provenance→citation flow + its
limitation (membership, not semantic-support); RETRO. Mark **Track X CLOSED**; FEATURE_ROADMAP entry.
**Acceptance**
1. A citation referencing a chunk NOT in the retrieval set is dropped/flagged (test with a planted bad id).
2. Genuine citations pass through unchanged; the answer never 500s on a missing-page source.
3. ARCHITECTURE note + RETRO written; Track X CLOSED; roadmap updated; suites green.
**Depends on**: X.1–X.2.

---

## Risks & open decisions
1. **LLM citation faithfulness** — the model may attribute a claim to a chunk that doesn't support it. The
   guard checks *membership in the retrieval set* (cheap, deterministic), NOT semantic support; a semantic
   check would need another LLM pass (out of scope — note it). Precision-first.
2. **Answer-schema change** — `citations` is additive; confirm no existing consumer breaks on the new field.
3. **Provenance gaps** — some sources (e.g. audio/WhisperX, plain text) have no `physical_page`; cite what
   exists (`section_path`/`source`) and handle `None` without breaking.
4. **Two answer graphs** — `ask` (cross-source) and `source_chat` (single-source) differ; X.2 covers both but
   their context-builders differ — keep each idiomatic.

## Verification (end-to-end)
- X.1: a staging probe (`SURREAL_DATABASE=staging`) showing retrieval hits with real `physical_page`/`section_path`.
- X.2: ask a question via the `ask` graph on a seeded multi-page source → answer carries `citations` whose pages match the chunks.
- X.3: plant a citation to a chunk not retrieved → it's dropped; a missing-page source → answer still returns.
- `@requires_docker` roundtrips + `uv run --project <pkg> pytest`.
