# Track Y — Auto-link (new note → related → RELATE)

> **Feature 2** of the Constella adoption plan (`docs/constella-features-adoption-research.md`).
> When a note is created, find its most-related notes by embedding similarity and persist the
> links as SurrealDB RELATE edges. The substrate exists (note embeddings, `find_related_by_embedding`
> at source level, RELATE patterns); Y adds the note-level similarity, a note↔note edge table, an
> orchestrator, and the trigger.

## Decisions (locked 2026-06-29)
1. **Trigger** → **on-demand first** (endpoint + MCP tool), **background job later** (the phased choice from the research Q).
2. **Scope** → **note↔note core** (the Constella behavior). The orchestrator + edge model are designed to
   **extend to note↔source** (a documented follow-up / optional Y phase) — same embedding-similarity mechanism.
   *(Open scope choice surfaced to the user; note↔note is the default.)*
3. **Precision / no graph explosion** → a configurable **min-similarity threshold** + top-k cap (reuse the
   R.6 discipline: don't link weak/everything). Default conservative.
4. **Reuse + known gotchas**:
   - Note embedding via `embed_note` (`pipelines/embeddings/.../service.py`); mirror the source-level
     `SourceRepository.find_related_by_embedding` (`source.py:364`).
   - **RELATE is NOT idempotent** (writes a 2nd row for a repeated `(in,out)` — Track W.2/W.3) → clear-before-relate
     or in-run pair-dedup. [[note-embedding-non-optional]]: `note.embedding` is strict `array<float>`, no default.
   - Migrations follow the non-destructive + S.4 backfill rules (migrations 66/67 pattern).

**Workflow**: track methodology — `implementer` → `adversarial-reviewer` (≤3 → `escalation-handler`).
Main tree, `uv run pytest`, no worktree. Additive/reversible; live writes gated; canonical note rows untouched.

---

## Phase Y.1 — Note-level similarity + note↔note edge table (Backend)
**Why**: the orchestrator needs "related notes" + a place to store the links; neither exists at note level.
**Deliverables**:
- `NoteRepository.find_related_by_embedding(note_id, k)` — mirror `SourceRepository.find_related_by_embedding`
  (cosine over `note.embedding`, exclude self, rank desc, return `[{id, title, score}]`). Handle a note with no
  embedding gracefully (empty / needs-embedding signal).
- A **note↔note edge table** migration (`DEFINE TABLE ... TYPE RELATION FROM note TO note`, e.g. `related_note`,
  fields `similarity_score: float`, `method: string` default `"embedding"`, `created_at`) — **fresh-container-safe**
  (non-destructive `DEFINE TABLE OVERWRITE` + null-endpoint-only DELETE + S.4 strict-fields-with-defaults, mirroring
  migrations 66/67). Verify on a fresh migration container (the cites/mentions drift lesson).
- An **idempotent** RELATE helper for the edge (pair-dedup / clear-before-relate, since RELATE isn't idempotent).
**Acceptance**
1. `find_related_by_embedding` ranks other notes by cosine, excludes self, handles the no-embedding note (no crash).
2. The edge table is correctly `TYPE RELATION FROM note TO note` on a FRESH migration container (not just staging);
   strict fields carry defaults.
3. The RELATE helper is idempotent (re-run → identical edge set, no duplicate `(in,out)` rows); no self-edge.
4. Tests (`@requires_docker` for the edge/migration) + canonical `note` rows untouched.
**Branch**: `track/y1-note-similarity`. **Depends on**: none.

## Phase Y.2 — Auto-link orchestrator + on-demand trigger (Backend)
**Why**: deliver the feature on-demand before automating it.
**Deliverables**:
- `note_auto_link_service`: given a note id → ensure embedding (embed if missing) → `find_related_by_embedding`
  → filter by `min_similarity` + top-k → create idempotent `related_note` edges (carry `similarity_score`).
  Reversible; canonical data untouched; returns a summary (created/skipped/below-threshold counts).
- **On-demand triggers**: a `POST /notes/{id}/auto-link` endpoint (+ params `k`, `min_similarity`) AND an MCP tool
  (`auto_link_note`) on the `surrealdb-mcp` server (consistent with the W.3 graph tools; reuse `add_note`'s notebook
  flow if relevant). Stdio-write-tool note from W.3 applies.
**Acceptance**
1. On-demand auto-link on a note with seeded similar/dissimilar notes creates `related_note` edges only above
   `min_similarity`, top-k capped, no self-link, idempotent on re-run.
2. The endpoint + MCP tool both drive the service and return the summary; a note with no embedding is embedded first
   (or a clear needs-embedding response), no 500.
3. Tests for the service (threshold/top-k/idempotency/self-exclusion) + the endpoint + the MCP tool.
**Branch**: `track/y2-autolink-ondemand`. **Depends on**: Y.1.

## Phase Y.3 — Background job (phased trigger) + integration + docs + RETRO (Integration)
**Why**: the "job later" half of the trigger decision; close the track.
**Deliverables**: wire auto-link into the existing job queue, triggered after a note is created+embedded (reuse the
`EMBEDDING_GENERATE` job pattern — auto-link runs once the note has an embedding). Idempotent + best-effort (a job
failure never corrupts the note). ARCHITECTURE note (the auto-link flow + the sync model: re-link on new notes; edit
re-link is a noted follow-up). RETRO. Mark **Track Y CLOSED**; FEATURE_ROADMAP entry. Note the **note↔source
extension** point.
**Acceptance**
1. Creating a note (with content) → after embedding, the job auto-links it; on-demand still works; both idempotent.
2. A job failure is isolated (note still created; logged; no 500/corruption).
3. ARCHITECTURE note + RETRO; Track Y CLOSED; roadmap updated; suites green.
**Depends on**: Y.1–Y.2.

---

## Risks & open decisions
1. **Graph explosion** — every note linking to many others. Mitigate: `min_similarity` + top-k; conservative defaults; reuse R.6 thinking. Decide the defaults at Y.2.
2. **RELATE non-idempotency** — must clear-before-relate / pair-dedup (Track W.2/W.3 lesson), or notes accrue duplicate edges on re-link.
3. **`note.embedding` strict** — writers must pass `[]` not None ([[note-embedding-non-optional]]); auto-link must embed before similarity.
4. **Data reality** — the current corpus is source-heavy with few notes, so auto-link is a built-and-tested MECHANISM that lights up as notes accumulate (like U.3 `cites`). Honest framing.
5. **Sync upkeep** — the job handles NEW notes; re-linking on note EDIT (content/embedding change) is a noted follow-up, not Y core.
6. **Scope (note↔note vs note↔source)** — Y core is note↔note; note↔source is the documented extension (same mechanism, a second edge type). User can promote it.

## Verification (end-to-end)
- Y.1: fresh migration container shows the edge table `TYPE RELATION FROM note TO note`; `find_related_by_embedding` ranks seeded notes.
- Y.2: `POST /notes/{id}/auto-link` + the MCP `auto_link_note` tool on seeded notes → idempotent threshold-respecting edges.
- Y.3: create a note → job auto-links after embedding; `@requires_docker` roundtrips; `uv run --project <pkg> pytest`.
