# Cross-track dashboard — `_status.md`

At-a-glance status across all tracks (see [`README.md`](./README.md) for the
workflow). Normally auto-maintained by the `cross-track-monitor` agent after each
phase merge.

> **Snapshot**: 2026-07-23 — **manually reconstructed** (the file was referenced
> in `README.md` but never created). This is a point-in-time snapshot, **not a
> live auto-update**. Source of truth per track remains its own `status.md` +
> the `docs/FEATURE_ROADMAP.md` status line.
>
> **Baseline**: `main` @ `5735deb` (in sync with `origin/main`); last merge =
> Track UX.6, closing Track UX.

## Legend

| Mark | Meaning |
|---|---|
| ✅ CLOSED | Roadmap/`status.md` declares complete; work merged to `main` |
| 🚢 SHIPPED | Phases merged to `main`, but no formal CLOSED/RETRO marker |
| 🔍 REVIEW | Implemented on a branch, "ready for review", **not merged to `main`** |
| ⏸ PARKED | Deliberately stopped after a decision gate |
| 📋 PLANNED | Plan exists; implementation not started (or not merged) |
| 🕗 NOT STARTED | On the roadmap; no work yet |
| ⚠ RECONCILE | Roadmap status and actual state disagree — needs attention |

## Board

| Track | Scope | Status | In `main`? | Last activity | Notes |
|---|---|---|:---:|---|---|
| **A** — mineru | Ingestion robustness (MinerU + fallback) | ✅ CLOSED | yes | 2026-06-04 | A3 (markitdown) = optional follow-up, not blocking |
| **B** — kg-quality | Multi-schema KG extraction + discipline | ✅ CLOSED | yes | 2026-06-12 | RETRO present |
| **C** — content-quality | Writer-Evaluator-Editor + markdown lint | 🕗 NOT STARTED | no | — | Roadmap entry only |
| **D** — output-richness | Obsidian / TTL / JSONL / NetworkX export | ✅ CLOSED | yes | 2026-06-16 | RETRO present |
| **E** — research-workflows | E1 single-agent + E3 thesis-mode | 🕗 NOT STARTED | no | — | Roadmap entry only |
| **F** — operations | Audit, librarian, resumable jobs | 🕗 NOT STARTED | no | — | Roadmap entry only |
| **G** — agent-integration | Headless mode + REST agent API | 🕗 NOT STARTED | no | — | Gates Track H |
| **H** — vision-parser | Vision-model parser tier | ⏸ DEFERRED | no | 2026-06-05 | Do not start before G is complete |
| **I** — docling-studio | Upload guards + inspect workspace + coord canon | 🚢 SHIPPED | yes | 2026-06-19 | Merged to `main` (I.A/I.B/I.D/I.E/I.H1); `status.md` still says "ready for review" — **docs stale**, no formal close |
| **J** — model-routing | Cloud/local routing (privacy + failover) | ✅ CLOSED | yes | 2026-06-22 | All 6 phases (J.1–J.6) |
| **K** — entity-resolution | NL normalizer + alias tables + fuzzy dedup + UI | 🚢 SHIPPED ⚠ | yes | 2026-06-22 | K.1–K.7a **merged to `main`** (merge commits `7c9e117`…`dd10eb0`); **roadmap still says PLANNED** — docs drift, no code pending |
| **L** — entity-typing | Ontology bridge + NL aliases + schema application | 🚢 SHIPPED ⚠ | yes | 2026-06-23 | L.1–L.4 **merged to `main`** (`5b7cb7f`…`9bcf0ee`); **roadmap still says PLANNED** — docs drift, no code pending |
| **M** — extraction-params | Model-aware chunk packing + RPM caps | 🚢 SHIPPED | yes | 2026-06-23 | M core + M.3 merged (`d2d7e22`, `c3ecf03`); **M.5 shipped 2026-07-25** (hetero-chain regression gate + `chunking_metrics`). Only M.4(a) full re-architecture deferred |
| **NS** — note-source-autolink | Note→source auto-link (extends Y) | ✅ CLOSED | yes | 2026-06-29 | ARCHITECTURE §12a |
| **O** — relation-persist | Relation endpoint resolution + self-heal migration | ✅ CLOSED | yes | 2026-06-23 | Staging healthy @ v62; old live re-verify block resolved |
| **P** — entity-embeddings | Per-entity embeddings + live backfill | ✅ CLOSED | yes | 2026-06-25 | Unblocks K.5 semantic dedup |
| **PL** — source-pipeline | Auto source→KG chain with gates | ✅ CLOSED | yes | 2026-06-30 | RETRO present; ARCHITECTURE §14 |
| **Q** — triage | Triage signals + after-extraction orchestration | 🚢 SHIPPED | yes | ~2026-06-24 | Q.1–Q.5 **merged to `main`** (`fd0e13a`…`ad435f2`); `status.md` says "ready for review" — docs stale |
| **R** — hybrid-search | Embedding foundation → kNN → KG signal → RRF → UI | 🚢 SHIPPED | yes | 2026-06-27 | R.0–R.6 all merged to `main` (tips are ancestors); no formal CLOSED/RETRO marker |
| **S** — drift-repair | Self-healing strict-field drift migration + guard | ✅ CLOSED | yes | ~2026-06-26 | RETROSPECTIVE in status.md |
| **T** — extraction-rescope | Extraction-economy rescope | ⏸ PARKED | n/a | 2026-06-27 | Decision gate: "stop thin" — R.6 owns search cleanup, exports stay rich |
| **U** — document-graph | Documents as first-class graph nodes (`mentions`) | ✅ CLOSED | yes | 2026-06-28 | |
| **UX** — pipeline-alignment | Frontend `processing_stage` spine + lean create flow | ✅ CLOSED | yes | 2026-07-01 | Latest closed track; ARCHITECTURE §14 |
| **V** — reference-extraction | Reference/citation extraction (GROBID engine + GF footnote/Kamerstuk path) | ✅ CLOSED | yes | 2026-07-24 | Shipped via #42 (resolvers + GROBID) + #44 (GF policy footnotes). Style-independent; heuristic evaluated then dropped. Live: 198 academic refs + Kamerstuk footnotes |
| **W** — mcp-graph-memory | MCP graph tools + hybrid search + reranker | ✅ CLOSED | yes | 2026-06-29 | Shared substrate proven over independent connection |
| **X** — citations-to-source | Exact chunk/page provenance in answers | ✅ CLOSED | yes | 2026-06-29 | |
| **Y** — auto-link | New note → related notes → RELATE | ✅ CLOSED | yes | 2026-06-29 | ARCHITECTURE §12 |
| **Z** — contradiction | LLM judges related pairs → verdict edges | ✅ CLOSED | yes | 2026-06-29 | |
| **OKF** — open-knowledge-format | OKF v0.1 export/import + MCP + UI (interchange adapter) | ✅ CLOSED | no | 2026-07-24 | Shipped via #48 (squash of OKF.1–OKF.5). Docs under `OKF-open-knowledge-format/`; RETRO present. Lossy-by-design ledger surfaced via REST header + MCP report + UI. Open follow-ups: async artifact store for large bundles, frontend import dialog |

## Merge-readiness finding (2026-07-23)

Verified with read-only git (`git merge-base --is-ancestor <branch> main`) across
**every** `origin/track/*` branch: **all of them are already ancestors of `main`
(ahead = 0).** Concretely — K.1–K.7a, L.1–L.4, M core+M.3, Q.1–Q.5, R.0–R.6, and
the I.* phases **all have merge commits in `main`**.

**Conclusion: nothing from the internal track workflow is waiting to merge.** The
"review-pending" / `📋 PLANNED` labels on I, K, L, M, Q are **stale
documentation**, not un-landed code. The drift is docs-only.

The only refs NOT in `main` are outside the track workflow:

| Ref | State | Merge-ready? |
|---|---|---|
| `feature/knowledge-graph` | tip 2025-12-17; `main` is **996 ahead**, branch 8 ahead; predates the entire track rebuild | **No** — stale / superseded. Candidate for **deletion**, not merge |
| `upstream/dependabot/uv/*` (pyjwt, pydantic-settings, langsmith) | upstream dependency bumps, ~1000 commits divergent | Fork-sync decision, not track work — **evaluate security bumps separately** |
| `upstream/codex/…security-dependency-audit` | upstream security audit, tip 2026-06-25 | Fork-sync decision — worth a separate review |
| `upstream/fix/*`, `upstream/refactor/i18n`, `upstream/issue-*` | upstream bug/refactor branches | Fork-sync decision, independent of the roadmap |

## Summary

- **Merged to `main` (22)**: A, B, D, I, J, K, L, M, NS, O, P, PL, Q, R, S, U, UX,
  W, X, Y, Z — plus C/E/F/G still to come.
- **Formally CLOSED (roadmap/RETRO)**: A, B, D, J, NS, O, P, PL, S, U, UX, W, X, Y, Z.
- **Merged but docs not closed** ⚠: I, K, L, M, Q, R (code landed; ledgers/roadmap stale).
- **Parked (1)**: T.
- **Planned / not started (6)**: C, E, F, G, H (deferred), V.

## Reconciliation items (docs-only; no code pending)

1. **Roadmap K & L still say `📋 PLANNED`** ⚠ but both are fully merged. Update the
   roadmap status lines to ✅ CLOSED (with merge-commit refs).
2. **Close-out the stale ledgers** — I, K, L, M, Q, R `status.md` files still read
   "ready for review". Mark them closed (and add RETRO.md where the track
   convention wants one; currently only A/B/D/PL have RETROs).
3. **Delete `feature/knowledge-graph`** — 996 commits behind, superseded by the
   track rebuild. Confirm nothing unique remains in its 8 commits, then prune.
4. **Evaluate upstream security syncs separately** — pyjwt / pydantic-settings /
   langsmith bumps + the security-dependency-audit branch. This is a fork-sync
   decision, not part of the internal roadmap.
5. **This dashboard is not auto-maintained** — regenerate it (or wire up the
   `cross-track-monitor` agent) after the next phase merge, otherwise it drifts.

## Open items / backlog (merged ≠ fully built)

The merge-readiness audit proves no *branch* is unmerged. That is NOT the same as
"every planned feature is done". Real open work, evidence-based (2026-07-23):

### A. ~~Genuinely open small track~~ — DONE (2026-07-24)
- ~~**Track V — reference-extraction**~~ — **SHIPPED**. Built end-to-end: a
  style-agnostic **GROBID** CRF service parses academic bibliographies from source
  PDFs (a hand-rolled heuristic was evaluated first, proven citation-style-dependent
  on real docling output, and dropped); the **GF footnote/Kamerstuk path** covers
  policy-document cross-references (Kamerbrieven/convenanten). Merged via #42 + #44.
  See `V-reference-extraction/{grobid-integration-plan,footnote-kamerstuk-plan,live-smoke-results}.md`.

### Follow-up session — 2026-07-24 (10 items shipped, 12 PRs)

A post-ingest follow-up pass off the interactive backlog. Shipped and merged:
OKF → main (#48) + **async download endpoint** (#55) + **frontend import UI**
(#56) — OKF is now fully first-class (export/import/MCP/UI, inline + deferred);
**I.H2 chunk_edit audit-log core** (#52, snapshots→I.H2b); **A.3 markitdown**
parser engine (#53); **reference polish** (#51: GROBID container URL, footnote
whitespace, and **3 live-verified KOOP-resolver bugs** — phrase-vs-`all` query
recall, `issued`→`date` year, dedicated `<dossiernummer>`); a **flaky-test
isolation** fix (#49) and the **live-validation** dashboard update (#50). Still
open: **I.H2b**, A.3 live smoke,
OKF download-polling UI.

### B. Deferred follow-ups inside CLOSED tracks
| Item | What | Size |
|---|---|---|
| ~~**A.3**~~ | ~~markitdown~~ — **SHIPPED** #53 (`parser_engine="markitdown"` + markdown→Document adapter). Follow-up: live corpus smoke vs docling |
| **M.5** | heterogeneous-chain integration test + metrics + docs | ✅ **SHIPPED (2026-07-25)** — `chunking_metrics` + 3-candidate packing gate (no-overflow + call-count divergence + M.4 guard). M.4(a) full re-architecture still deferred |
| ~~**I.H2**~~ | ~~`chunk_edit` table~~ — **audit core SHIPPED** #52 (migration 74 + `ChunkAuditService` + history endpoint). **I.H2b** (snapshots/restore/UI) deferred |
| **I.H1 AC5** | multi-worker shared state (currently in-memory backend) | small |
| **J FU-J4-2** | error-mapping edge: a 400 error with a 5xx number in its body can misclassify as failover-eligible | tiny |
| **D.0 #1** | SurrealQL promotion of a filter | tiny |
| **A e2e workflow** | ~~parked as `.pending`~~ — **already ACTIVE** (2026-07-24): `e2e.yml` triggers on every PR to main. The real open work is that it is **red** — ~13 pre-existing Playwright specs fail (track-i result-tabs / structure-graph "renders without console errors", ux-pipeline-spine disabled-state). Needs a local frontend+compose run to fix; not headless-fixable. | follow-up: fix 13 e2e specs |

### C. Live-verification gaps (largely closed 2026-07-24)
**~54 acceptance criteria** across the ledgers were marked "deferred to a live run"
because the original sandbox had no Docker/SurrealDB — migration apply/revert, live
counts, latency, visual checks.

**Live-validation pass (2026-07-24)** — with Docker/SurrealDB available, the
Docker-gated suite now runs green end-to-end:
- **277/277 `@requires_docker` tests pass** against an isolated testcontainer
  (session-scoped `open_notebook_test` DB spun per run — real staging untouched).
  One cross-file isolation flake was fixed (#49: a `find_related` top-k assertion
  that assumed a clean DB).
- The read-only `requires_staging` hybrid-ranker sanity (`test_source_related_hybrid_staging`)
  **passes against real staging** (0 write-ops, 5 read-ops).
- Migration roundtrip (apply/revert) is green in CI (testcontainers job).

What remains genuinely unproven is the *visual* layer: the Playwright e2e suite has
~13 pre-existing failures in tracks I / UX-pipeline-spine (console-error + disabled-state
assertions), and the `test-build-*` CI jobs reference a missing `Dockerfile.single`
(a workflow bug, also red on `main`). These are infra/e2e debt, not track-code
regressions — see item A (e2e workflow) and the CI-hygiene follow-up.

### D. Not-started / parked / deferred (known, larger)
- **Not started**: C (content quality), E (research workflows),
  G (agent integration). **Deferred**: H (vision parser, after G).
  **Parked**: T.2b (prompt change), RePEc resolver-leg (config-gated, deliberately
  not enabled). **Shipped**: OKF (merged via #48); **F — operations** (audit +
  librarian + failure-provenance; F.1–F.7 via #58–#64, 2026-07-24 — see
  `F-operations/{plan,status,RETRO}.md`; F.7b `chunked`-split deferred).

## Next up (per roadmap dependency order)

- Then the untouched roadmap tracks: **G** (agent integration) — with **H**
  deferred until **G** lands — plus **C** (content quality) and **E** (research
  workflows). (**F** shipped 2026-07-24.)
