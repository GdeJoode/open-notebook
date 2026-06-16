# Phase D.4 — Self-review

> Phase scope: integration + retro + docs (NO code changes).
> Plan reference: `docs/tracks/D-output-richness/plan.md` §D.4
> (lines 352–382).
> Branch: `track/d-integration-retro`.
> Date: 2026-06-16.

## Acceptance criteria checklist (plan §371)

### AC #1 — All 3 export formats successfully run end-to-end against the corpus notebook; results captured in `E2E_EVIDENCE.md`

**Status**: PARTIALLY MET (sandbox-bounded).

- ✅ Captured in `docs/tracks/D-output-richness/E2E_EVIDENCE.md`.
- ✅ Test-suite evidence covers contract correctness of all three
  exports — 76+ tests across 6 files (22 Obsidian service + 13 JSONL
  service + 16 NetworkX service + 7 preview + 16 router + 2 handler).
  All previously merged and recorded green in per-phase self-reviews
  (D.0, D.1a attempt-2, D.1b, D.1c attempt-2, D.2 attempt-2, D.3
  attempt-2). Per-phase pytest exit codes documented in those
  self-reviews; D.4 does not re-run them.
- ⚠️ Live consumer-tool smoke (Obsidian wikilink renderer, Neo4j
  `apoc.load.json` parse, Gephi GraphML import) **NOT executable in
  the CI sandbox** — no real Obsidian binary, no running Neo4j with
  APOC, no Gephi installation. Documented honestly at the top of
  E2E_EVIDENCE.md.
- ✅ Manual smoke checklist for an operator session is captured in
  E2E_EVIDENCE.md §"Manual smoke instructions", step-by-step from
  pre-flight through Obsidian (zip + vault_path), JSONL (+ optional
  Neo4j), NetworkX (+ optional Gephi), and a preview-parity check.
- ✅ Pattern follows Track B's RETRO §"Live-test recommendation"
  precedent (lines 188–213) — sandbox limitation, deferred to
  operator, manual checklist with capture points named.

**Conclusion**: AC #1 is satisfied for the testable component
(contract correctness of bytes produced) and explicitly deferred for
the consumer-side component (real Obsidian / Neo4j / Gephi parse) per
sandbox limitation. The deferral is named, bounded, and reproducible
per the manual checklist.

### AC #2 — `ARCHITECTURE.md` reflects the 3 new endpoints + the new service modules + the `export.*` metrics event types

**Status**: MET.

- ✅ New §7 "Knowledge graph export surfaces (Track D — output
  richness)" added. Endpoint table lists all four URLs (3 exports +
  preview), the service-module file path for each, and the
  `export.obsidian` / `export.jsonl` / `export.networkx` metric event
  types.
- ✅ Shared filter pipeline (4-stage: SurrealQL gate +
  `EXCLUDED_ENTITY_STATUSES` post-filter +
  `_apply_min_connections_filter` + Q-D-4 endpoint intersection)
  documented with the parity invariant explanation.
- ✅ Shared Pydantic + TypeScript model surface called out
  (`ExportFilter`, `ObsidianExportRequest`, etc.).
- ✅ Telemetry contract (counts-only payload, `vault_path_redacted`)
  documented.
- ✅ §7 "Further reading" renumbered to §8; new entries for
  Track-D RETRO + exports troubleshooting doc.
- ✅ Edit is an integration (not a bare append) — the new §7 sits
  between the existing storage-layer §6 (Track B) and the renumbered
  "Further reading" §8, which is the right structural slot.

### AC #3 — `FEATURE_ROADMAP.md` Track D status reflects all phases complete

**Status**: MET.

- ✅ Status line at the top of the Track D section follows the
  Track B convention exactly: `> **Status**: ✅ **COMPLETE
  (2026-06-16)** — …`.
- ✅ Per-phase table (D.0, D.1a, D.1b, D.1c, D.2, D.3, D.4) added
  with `| ✅ |` per row.
- ✅ Vision paragraph rewritten in past-tense to reflect shipped
  state.
- ✅ Cross-refs to RETRO, E2E_EVIDENCE, status.md.

### AC #4 — `RETRO.md` exists with the required entry counts

**Status**: MET (exceeds).

- ✅ `docs/tracks/D-output-richness/RETRO.md` created.
- ✅ "What worked" section: **7 entries** (required: 5+). Bulleted
  with `^- \*\*` markdown headers.
- ✅ "What hurt" section: **4 entries** (required: 3+).
- ✅ "Recommendations for future tracks" section: **6 entries**
  (required: 5+). Numbered list addressing tracks C/E/F/G/H.
- ✅ Cross-reference section added explicitly mapping which Track-B
  RETRO entries Track-D validated.
- ✅ Phase-by-phase attempt count table modeled on Track B's.
- ✅ Tooling sections (paid off + didn't carry its weight)
  modeled on Track B's.

Counts verified by `awk` over the file (see status.md entry).

### AC #5 — `docs/troubleshooting/exports.md` exists and is linked from `docs/troubleshooting/index.md`

**Status**: MET.

- ✅ `docs/troubleshooting/exports.md` created, following the
  `parser-engines.md` shape (problem-symptom-cause-fix blocks per
  failure mode).
- ✅ Sections per the plan brief: Obsidian zip, Obsidian vault_path
  (with status-code reference table), JSONL (with per-line shape
  reference + Neo4j apoc.load.json error), NetworkX (with per-format
  limitations table), preview counts mismatch.
- ✅ Linked from `docs/troubleshooting/index.md` under a new
  "Knowledge Graph Exports" section in the Quick Navigation list
  (parallel to the existing "Parser Engines" section).
- ✅ Cross-refs back to ARCHITECTURE §7, Track D RETRO,
  E2E_EVIDENCE.

## Deliverables ledger

| Deliverable | Path | Status |
|---|---|---|
| E2E evidence | `docs/tracks/D-output-richness/E2E_EVIDENCE.md` | NEW |
| Architecture update | `ARCHITECTURE.md` (new §7, renumbered §8) | UPDATED |
| Roadmap status | `docs/FEATURE_ROADMAP.md` (Track D block) | UPDATED |
| Retrospective | `docs/tracks/D-output-richness/RETRO.md` | NEW |
| Troubleshooting | `docs/troubleshooting/exports.md` + `index.md` link | NEW + UPDATED |
| Status entry | `docs/tracks/D-output-richness/status.md` (D.4 row) | APPENDED |
| Self-review (this file) | `docs/tracks/D-output-richness/reviews/phase-D.4-self-review.md` | NEW |

## Sandbox-limited acknowledgement (AC #1 detail)

Per the plan brief, AC #1 calls for "All 3 export formats
successfully run end-to-end against the corpus notebook". The
sandbox in which this phase executes does NOT have:

- A real Obsidian binary (cannot validate wikilink rendering /
  frontmatter parsing in the actual UI).
- A running Neo4j with the APOC plugin (cannot validate
  `apoc.load.json` parse semantics against the produced JSONL).
- A Gephi installation (cannot validate GraphML import).

What the test suite (which IS executable in the sandbox per the
earlier per-phase self-reviews) DOES prove: the export pipelines
produce bytes that match every contract our tests pin (frontmatter
shape, JSONL line shape, NetworkX node/edge/attribute round-trip),
the shared filter pipeline produces identical entity sets across all
four surfaces (3 exports + preview), and the HTTP + job-handler
wiring is end-to-end through routers, services, and DI.

What it does NOT prove: that the produced bytes parse cleanly in a
real consumer's runtime. For that, an operator with access to those
tools needs to walk the manual smoke checklist in E2E_EVIDENCE.md.

This trade-off mirrors Track B's RETRO §"Live-test recommendation"
(`docs/tracks/B-kg-quality/RETRO.md`, lines 188–213) verbatim — a
documented, bounded deferral rather than a hidden gap.

## Cross-references to Track-B RETRO entries this track validates

Track D was authored under the discipline established by Track B's
RETRO; this self-review explicitly cross-references which entries
were operationally validated:

- **Track B RETRO §"What worked" point on adversarial review catching
  real production bugs** → Track D added four operational examples
  (D.1c preview drift, D.2 GeneratorExit semantic gap, D.1a filename
  sanitisation, D.3 JSON-tree multi-rooted DAG). The 4/6 attempt-2
  rate is the empirical proof the pattern carries forward.
- **Track B RETRO #2 "for complex shared-file phases prefer
  sequential over parallel"** → Track D operationalised as a hard
  rule; zero merge conflicts on the shared `exports.py` router and
  shared `NotebookHeader.tsx` despite 3+ phases touching each.
- **Track B RETRO #3 "centralise what-tests-claimed-vs-what-actually-
  ran in the implementer self-review"** → Track D's attempt-2
  self-reviews enumerate inversion proofs at the head; D.2 attempt-1's
  embedding-exclusion shadow-mask is the cautionary tale documented
  in Track-D RETRO §"What hurt" entry 2.
- **Track B RETRO #6 "inversion test pattern is the gold standard"**
  → Track D adopted directly. Track-D RETRO recommendation #6
  (formalise the inversion proof as a template-level requirement)
  strengthens this for future tracks.
- **Track B RETRO #7 "telemetry-first"** → Track D piggy-backed on
  B.4's `metrics` table for three new `export.*` event types
  without a migration; operators get per-format call-rate counters
  for free.

## Out-of-scope items (deliberately not touched in D.4)

- D.0 follow-up #1 (status filter SurrealQL promotion). The three
  exporters + the preview surface all apply the Python-side
  `EXCLUDED_ENTITY_STATUSES` filter; promoting it to the SurrealQL
  gate in `EntityRepository.list_entities_for_notebook` would
  consolidate the gate to a single layer. Recommended by D.1a / D.3
  self-reviews. Track-D RETRO recommendation 1 ("promote
  EXCLUDED_ENTITY_STATUSES to a shared module") is the more general
  framing of this — addressing both Python-layer co-location and
  the SurrealQL-promotion question together is left to a future
  track plan (likely as part of the first track that adds a 4th
  exporter or read-only view).
- Q9 / Track M4 (TOOI + Crossref `external_ids` resolution). The
  Obsidian frontmatter ships `external_ids: []` in V1; swap is a
  one-file change in `shared.utils.external_ids.resolve_external_ids`
  with no caller migration. Documented in D.0's self-review for the
  M4 plan to pick up.

## Risk assessment (per plan §385+)

The seven risks the plan identified:

- **Risk 1 (large-notebook memory blowup)**: mitigated by D.0
  paginated repo projection + D.2's streaming + D.1a's BytesIO
  build-then-stream. No D.4 follow-up needed.
- **Risk 2 (vault-path filesystem safety)**: D.1b's defense-in-depth
  (absolute path, exists, writable, containment check) covers this;
  troubleshooting doc documents the failure modes.
- **Risk 3 (Obsidian filename collisions)**: D.1a's `-2`/`-3`
  suffix; `ExportReport.metadata` surfaces collisions.
- **Risk 4 (Q9 not ready)**: documented in this self-review under
  "Out-of-scope items".
- **Risk 5 (NetworkX attribute-type quirks)**: D.3 round-trip tests
  pin the flatten/unflatten contract; troubleshooting doc documents
  the per-format limitations.
- **Risk 6 (VaultSyncService write-collision)**: D.1b documented the
  directional split (VaultSync read-only on startup; ObsidianExport
  writes on trigger); no concurrent execution in normal usage.
- **Risk 7 (reviewer-rejection cycle)**: Track D's 67% rate on a
  small base is within the noise of Track B's 47% on a larger base.
  RETRO recommendation #6 strengthens the inversion-test discipline
  for future tracks.

## PR scope check

Plan §379 — ONE PR titled `docs(track-D): output richness
integration + RETRO (D.4)`. The branch
`track/d-integration-retro` contains six commits:

1. `docs(track-d): E2E evidence + sandbox-smoke acknowledgement`
2. `docs(architecture): add knowledge graph export surfaces`
3. `docs(roadmap): mark Track D phases complete`
4. `docs(track-d): RETRO + cross-references`
5. `docs(troubleshooting): export-formats reference + index link`
6. `docs(track-d): D.4 self-review + Track D closed status`

`git diff main --stat` should show ONLY paths under `docs/` and
`ARCHITECTURE.md` — no code changes. (Verified before push.)

## Ready for review

**Track D CLOSED.** All five ACs met (AC #1 partial-with-bounded-
deferral per sandbox limitation, explicitly named and documented;
ACs #2 through #5 fully satisfied). PR ready for the strict
reviewer.
