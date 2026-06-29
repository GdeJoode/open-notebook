# Track Z — Contradiction detection (LLM judges related pairs: reinforces / contradicts)

> **Feature 3** (the last) of the Constella adoption plan (`docs/constella-features-adoption-research.md`).
> Take candidate pairs from RELATED sources and have an LLM judge whether they **reinforce / contradict /
> are neutral**, persisting only confident verdicts as edges. Highest-risk feature: a false contradiction
> pollutes the graph, so **precision over recall** throughout (same discipline as U.3 `cites`).

## Decisions (locked 2026-06-29)
1. **Trigger** → **on-demand first** (endpoint + MCP tool); a background job is a **noted follow-up** (deferred
   given the LLM cost + the precision risk — automate only once the judge is trusted).
2. **Scope** → **source↔source verdict** (judge two RELATED sources). Candidates come from the **existing
   source-level related substrate** (Track R: `find_related_hybrid`/`find_related_by_embedding`) so we judge only
   plausibly-related pairs, NOT O(n²). Claim-level contradiction (the app-side `claim`/`contradicts` scaffolding)
   is a **documented extension**, not Z core (claims are sparse).
3. **Precision-first** → persist a `contradicts`/`reinforces` edge ONLY on a confident verdict above a threshold;
   `neutral`/low-confidence → NO edge. A fabricated contradiction is worse than a missing one. Bound LLM cost to
   related pairs + a top-k cap.
4. **LLM judge** → reuse the Track J model-routing: `RoutedLLMCaller` / `call_candidate(json_mode=True)`
   (`apps/app-main/.../services/model_routing/llm_call.py`) → structured `{verdict, confidence, reasoning}`.
5. **Schema first** → the app-side `claim`/`contradicts` tables exist but are NOT migration-managed; Z.1 asserts a
   proper, fresh-container-safe verdict-edge schema before anything writes to it (the cites/mentions drift lesson).
6. **Known gotchas** → RELATE endpoints can't be `$param`-bound + strict-validate ids before interpolation
   ([[surrealdb-relate-id-injection]]); RELATE isn't idempotent → clear-before-relate ([[note-embedding-non-optional]] family).
   Keep the verdict edge distinct from the triage `VERSTERKT` entity-relation predicate (different layer).

**Workflow**: track methodology — `implementer` → `adversarial-reviewer` (≤3 → `escalation-handler`).
Main tree, `uv run pytest`, no worktree. Additive/reversible; live writes gated; canonical source/claim rows untouched.

---

## Phase Z.1 — Verdict edge schema + idempotent helper (Backend)
**Why**: a safe, fresh-container-correct place to store judgments, before any judge writes to it.
**Deliverables**:
- A migration (next number) asserting a **source↔source verdict edge** (e.g. `source_verdict`
  `TYPE RELATION FROM source TO source`), fields `verdict: string` (default `"neutral"`; values
  reinforces/contradicts/neutral), `confidence: float` (default 0.0), `reasoning: string` (default ""),
  `judge_model: string`, `created_at: datetime` (default `time::now()`) — strict WITH defaults (S.4), non-destructive
  (`DEFINE TABLE OVERWRITE` + null-endpoint-only DELETE), **verified on a FRESH 1..N container**. Reconcile with the
  app-side `claim`/`contradicts` scaffolding: either assert their schema too or document why the new edge is separate.
- An **idempotent** verdict-relate helper (clear-before-relate per `(in,out)`; **strict `_validate_record_id` before
  interpolation** — the Y.1 injection lesson; refuse self-edges; carry verdict/confidence/reasoning).
**Acceptance**
1. The verdict edge is `TYPE RELATION FROM source TO source` on a FRESH migration container; strict fields default.
2. The relate helper is idempotent (re-run → one row, latest verdict), refuses self-edges, and is injection-safe
   (a `;`-bearing id is rejected, the table survives — explicit test).
3. Tests (`@requires_docker`) + canonical rows untouched.
**Branch**: `track/z1-verdict-schema`. **Depends on**: none.

## Phase Z.2 — Candidate generation + pairwise LLM judge (Backend)
**Why**: the heart of the feature — judge related pairs, precision-first.
**Deliverables**:
- **Candidate generator**: for a source, use the Track R related substrate (`find_related_hybrid` /
  `find_related_by_embedding`) to get its top related sources; form `(a,b)` pairs (dedup, exclude self, bound by top-k)
  — only plausibly-related pairs are judged.
- **Pairwise judge service** (`contradiction_judge_service`): for a pair, build a compact context (the two sources'
  titles + summaries/`full_text` snippets), call the LLM via `RoutedLLMCaller`/`call_candidate(json_mode=True)` with a
  judge prompt → parse `{verdict ∈ {reinforces,contradicts,neutral}, confidence ∈ [0,1], reasoning}`. **Precision-first**:
  persist a verdict edge ONLY for `contradicts`/`reinforces` with `confidence >= threshold`; `neutral`/low-confidence →
  NO edge. Idempotent persistence (Z.1 helper). A new judge prompt lives in `packages/ontology-manager/.../prompts.py`
  (or the prompts dir).
**Acceptance**
1. The judge returns a structured `{verdict, confidence, reasoning}`; a low-confidence/neutral verdict produces NO edge
   (precision-first — test with a mocked LLM returning each verdict class).
2. Candidates come from the related substrate (not all-pairs); no self-judgment; bounded by top-k.
3. Persistence is idempotent (re-judging a pair → one edge, latest verdict); injection-safe via Z.1.
4. Tests: candidate generation, the judge parse + each verdict class, the threshold gate, idempotency. The LLM is
   MOCKED in tests (no live model needed); a live smoke is optional/gated.
**Branch**: `track/z2-judge`. **Depends on**: Z.1.

## Phase Z.3 — On-demand trigger + integration + docs + RETRO (Integration → CLOSE)
**Why**: deliver on-demand; close the track.
**Deliverables**: an endpoint (`POST /sources/{id}/judge-contradictions` — judge the source's related pairs; params
`k`, `min_confidence`) with route-layer validation, AND an MCP tool (`judge_contradiction` — judge a specific pair, or
a source's related pairs). Returns a summary (judged/contradicts/reinforces/neutral/skipped). ARCHITECTURE note (the
candidate→judge→verdict flow + the precision-first + cost framing + the **background-job + claim-level** extensions).
RETRO. Mark **Track Z CLOSED**; FEATURE_ROADMAP entry.
**Acceptance**
1. On-demand judging over a source's related pairs creates verdict edges only for confident contradicts/reinforces;
   summary returned; route-layer validation (bad id/bounds → 422, not 500).
2. The endpoint + MCP tool both drive the judge; the LLM cost is bounded (top-k related pairs).
3. ARCHITECTURE note + RETRO; Track Z CLOSED; roadmap updated; background-job + claim-level extensions noted; suites green.
**Depends on**: Z.1–Z.2.

---

## Risks & open decisions
1. **Judge precision (the central risk)** — a false `contradicts` pollutes the graph. Precision-first: confident-only,
   a conservative threshold, `reasoning` retained for review. The adversarial review must hammer false positives.
2. **LLM cost** — judging is O(pairs). Bound to related pairs (Track R) + top-k; on-demand only (no auto-job in Z core).
3. **Semantic overlap with `VERSTERKT`** — the triage has a `VERSTERKT` entity-relation predicate; the Z verdict is a
   source↔source judgment (different layer/unit). Keep them distinct; don't conflate.
4. **Data reality** — few sources today; the mechanism is built/tested and lights up as the corpus grows (like U.3).
5. **Scope** — source↔source is Z core; claim-level (the app-side `contradicts source→claim`) is the documented extension.

## Verification (end-to-end)
- Z.1: fresh migration container shows the verdict edge `TYPE RELATION FROM source TO source`; the relate helper is idempotent + injection-safe.
- Z.2: a mocked judge returning contradicts/reinforces/neutral → only confident contradicts/reinforces persist; candidates come from related.
- Z.3: `POST /sources/{id}/judge-contradictions` + the MCP tool on seeded related sources → bounded, precision-first verdict edges + summary.
- `@requires_docker` roundtrips + `uv run --project <pkg> pytest`; LLM mocked.
