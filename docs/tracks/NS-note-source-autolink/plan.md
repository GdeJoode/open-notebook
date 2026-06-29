# Track NS — Note↔source auto-link (extend Track Y to link notes to the sources they're about)

> **Extension of Track Y** (auto-link), per the user's choice: a note should also auto-link to the
> **sources** it's about (by embedding similarity), not only to related notes. More immediately visible
> in the current source-heavy corpus. Reuses Y's orchestrator, threshold/top-k discipline, idempotent +
> injection-safe RELATE pattern, and the after-embed job trigger.

## Decisions (locked 2026-06-29)
1. **Scope** → note→**source** edges (a note links to the sources it's most similar to), added alongside the
   existing note→note `related_note` links. Same on-demand + job triggers as Y.
2. **Cross-type similarity** → compare `note.embedding` vs `source.embedding`. **Verify both are the same space**
   (mxbai-embed-large, 1024-dim) — the `array::len(embedding) = array::len($q)` guard already enforces dim-match,
   but confirm a note embedding and a source aggregate embedding are genuinely comparable (same model). If not, stop
   and surface it.
3. **Precision / no explosion** → reuse Y's `min_similarity` + top-k (conservative defaults). Precision-first.
4. **Reuse + gotchas** → mirror `relate_note`/`relate_verdict`: strict `_validate_record_id` on raw ids BEFORE
   interpolation ([[surrealdb-relate-id-injection]]); clear-before-relate idempotency; S.4 fresh-container-safe
   migration; [[note-embedding-non-optional]].

**Workflow**: track methodology — `implementer` → `adversarial-reviewer` (≤3 → `escalation-handler`).
Main tree, `uv run pytest`, no worktree. Additive/reversible; canonical rows untouched.

---

## Phase NS.1 — Note→source similarity + `note_about` edge + helper (Backend)
**Why**: the orchestrator needs "related sources for a note" + a place to store the links.
**Deliverables**:
- `NoteRepository.find_related_sources_by_embedding(note_id, k)` — cosine of `note.embedding` vs every
  `source.embedding`, rank desc, return `[{id, title, score}]`; skip sources/notes without an embedding
  (the `array::len` guard); a note with no embedding → `[]` (graceful). (Mirror `find_related_by_embedding`,
  but cross-type note→source.)
- A **note→source edge table** migration (next number — likely 70), `DEFINE TABLE OVERWRITE note_about
  TYPE RELATION FROM note TO source` (name: `note_about` suggested — "this note is about this source"), fields
  `similarity_score: float` (DEFAULT 0.0), `method: string` (DEFAULT "embedding"), `created_at` (DEFAULT
  time::now()) — strict WITH defaults (S.4), non-destructive (null-endpoint DELETE + OVERWRITE), **verified on a
  FRESH 1..N container**; index on score. Down sane.
- An idempotent, injection-safe `NoteRepository.relate_note_source(note_id, source_id, *, similarity_score, method)`
  — strict `_validate_record_id` on BOTH raw ids before interpolation; refuse self-edge is N/A (different types) but
  validate the `note:`/`source:` prefixes; clear-before-relate per `(in,out)`.
**Acceptance**
1. `find_related_sources_by_embedding` ranks sources by cosine vs the note's embedding, skips no-embedding rows,
   handles the no-embedding note (no crash); the note↔source embedding comparison is validated as same-space.
2. `note_about` is `TYPE RELATION FROM note TO source` on a FRESH container; strict fields default.
3. `relate_note_source` is idempotent (re-relate → one row, latest score), validates ids (injection payload refused,
   tables intact — explicit test), enforces note→source typing.
4. Tests (`@requires_docker`) + canonical rows untouched.
**Branch**: `track/ns1-note-source-similarity`. **Depends on**: none.

## Phase NS.2 — Extend the auto-link orchestrator + triggers (Backend)
**Why**: deliver the feature through the same on-demand + job paths as Y.
**Deliverables**:
- Extend `NoteAutoLinkService.auto_link` (or add a sibling method) to ALSO produce note→source `note_about` edges:
  ensure embedding → `find_related_sources_by_embedding` → `min_similarity` + top-k → idempotent `relate_note_source`
  → fold the counts into the summary (`source_links_created`, etc.). Keep the existing note→note path intact.
- The on-demand endpoint (`POST /notes/{id}/auto-link`) summary includes source links (param to include/limit them);
  the after-embed job links both notes and sources; the MCP `auto_link_note` tool covers sources too (it's repo-direct
  — `find_related_sources_by_embedding` + `relate_note_source` are both in surrealdb-service, so option (a) holds).
- Route-layer validation unchanged (Y.2 discipline: bad id/bounds → 422).
**Acceptance**
1. On-demand auto-link on a note with seeded similar/dissimilar sources creates `note_about` edges only above
   `min_similarity`, top-k capped, idempotent; note→note links still produced; summary reports both.
2. The endpoint + MCP tool + job all link sources too; no 500 on no-embedding; route validation intact.
3. Tests: the source-link path (threshold/top-k/idempotency), endpoint, MCP tool, job.
**Branch**: `track/ns2-orchestrator`. **Depends on**: NS.1.

## Phase NS.3 — Integration + docs + RETRO (CLOSE)
**Deliverables**: ARCHITECTURE note (the note→source layer alongside note→note + the shared orchestrator); RETRO; mark
**Track NS CLOSED**; FEATURE_ROADMAP entry. Note any extension (note→entity, edit-relink). **Acceptance**: docs + RETRO;
CLOSED; roadmap; suites green. **Depends on**: NS.1–NS.2.

---

## Risks & open decisions
1. **Cross-type embedding validity** — note vs source aggregate embedding must be the same model/space (verify in NS.1).
2. **Graph explosion** — note→source over many sources; `min_similarity` + top-k; conservative defaults.
3. **RELATE non-idempotency + injection** — clear-before-relate + strict id validation (the Y.1/relate_cites lessons).
4. **Edge naming / ontology** — reconcile `note_about` with the schema_core ontology edges; pick a clear name.
5. **Data reality** — source-heavy corpus → this lights up immediately (notes link to the convenanten/papers they resemble).

## Verification (end-to-end)
- NS.1: fresh container shows `note_about TYPE RELATION FROM note TO source`; `find_related_sources_by_embedding` ranks seeded sources; `relate_note_source` idempotent + injection-safe.
- NS.2: `POST /notes/{id}/auto-link` + the MCP tool + the job on a seeded note → `note_about` edges above threshold + note→note links, idempotent.
- `@requires_docker` roundtrips + `uv run --project <pkg> pytest`.
