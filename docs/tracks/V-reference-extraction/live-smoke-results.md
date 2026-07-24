# Track V — live smoke results

Live validation against the real ingested `staging` corpus. **No image rebuild** —
new code hot-loaded locally via `uv run python` against the live SurrealDB + a live
GROBID container.

## Environment
- Docker up; `surrealdb` 2.6.5 (`:8000`, NS `open_notebook`, DB **`staging`**);
  `grobid/grobid:0.8.2-crf` (`:8070`).
- Corpus already ingested: **14 sources**, 3824 chunks, `cites = 0` initially.

---

## Reference extraction — GROBID engine (G.1–G.3), 2026-07-24

The style-dependent V.1/V.2/V.3 heuristic was replaced by a self-hosted GROBID CRF
service (Apache-2.0). Sources feed their **PDF** (`source.asset.file_path`) to
GROBID `/api/processReferences`; TEI → `ParsedReference`. One engine, no
citation-style code.

### Per-source refs_extracted (live, through the wired V.5 path)

| Source | GROBID | old heuristic | note |
|---|---:|---:|---|
| J of Common Market Studies 2025 — Ali | **73** | 62 | Wiley journal, DOI-rich |
| the_centrifugal_state | **62** | 1 | no-parens `Surname A.B., YYYY.` style |
| 363996567-1994 fiscal-needs | **32** | 27 | mixed |
| Economics_without_equilibrium | **31** | 7 | **two** bibliographies, parens-APA |
| 10 × Convenant Regio Deal … | **0** | 0 | policy docs — see caveats |
| **Total** | **198** | 98 | |

### Verdict
- **GROBID roughly doubles extraction** vs the heuristic and is **style-independent**:
  parens-APA (Economics) and no-parens (centrifugal) both parse through the same code
  path. Fragmented entries reassembled (Schumpeter 2002 → one), straddles split
  (Alesina/Arzaghi), OCR ligatures cleaned ("fi rms" → "firms"). DOIs/years/authors/
  titles parsed cleanly from TEI.
- **Graceful degrade confirmed**: a source with no/absent PDF → `[]`, logged, no crash.

### Convenanten = 0 — two honest reasons
1. **Path resolution**: their `asset.file_path` is `data/uploads/Convenant….pdf`, but
   the files live in `notebook_data/uploads/` → file-not-found → graceful `[]`. A
   data/path-consistency issue to reconcile for a full policy-corpus run.
2. **Wrong reference class**: even with the PDF, convenanten (and Kamerbrieven) cite
   other documents in **footnotes** using Kamerstuk identifiers, not a scholarly
   bibliography — that is the **GF footnote+Kamerstuk path** (separate plan), not
   GROBID's bibliography model.

---

## `cites` materialization (V.5 → U.3)
Whole-corpus `materialize_corpus`: all extracted references are classified **external**
by U.3 (the 14 sources cite external works, not each other), so `edges_materialized = 0`
— the correct result for this corpus (confirms U.1's 0 intra-corpus citations).
Extraction is judged by `refs_extracted` (198), not the edge count. Idempotent; no
failures, no self-edges.

---

## Policy-document finding (drives the GF track)
A live test on the **NPVR voortgangsbrief (30 Jan 2026)** confirmed the policy-citation
gap: GROBID `processReferences` → 0, but `processFulltextDocument` shows the letter
cites other Kamerstukken in **footnotes** — `Kamerstuk 31305-489`, `Motie 36410-111`.
Formats vary (`Kamerstukken II 2024/25, 36410, nr. 111` · `12345-VII-blg-1` · …). See
`footnote-kamerstuk-plan.md`.

---

## Remaining live TODOs
- **GF footnote+Kamerstuk path** — the primary citation type for the policy corpus
  (convenanten + Kamerbrieven); not yet built.
- **Convenant PDF path reconciliation** — `data/uploads/` vs `notebook_data/uploads/`.
- **overheid.nl KOOP record mapping** (V.4) — query valid, record→work mapping needs a
  real dossier.
- **RePEc** — request `REPEC_API_KEY`, then live-verify CitEc.
- **A non-zero `cites` count** needs a corpus that cites itself (a data choice).
