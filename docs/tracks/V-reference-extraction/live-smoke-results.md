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

## GF — footnote + Kamerstuk path (policy documents), 2026-07-24

Built (GF.1–GF.4) and live-validated on the **NPVR voortgangsbrief (30 Jan 2026)**.
Pipeline: GROBID `processFulltextDocument` → footnotes → classify
(government / academic / prose) → government via the tolerant `KamerstukIdentifier`
normalizer + `OverheidResolver`, academic via GROBID `processCitation`.

Live NPVR result — 7 footnotes → **3 government references** (4 prose correctly
dropped):

| footnote | classified | parsed id |
|---|---|---|
| `Kamerstuk 31305-489` | government | dossier 31305, nr 489, kamerstuk |
| `Motie 36410-111` | government | dossier 36410, nr 111, motie |
| `Bij beleids-en investeringslogica … gebiedsgericht …` | government | dossier 29697, nr 158 (inline in prose; 29697 = the NPVR parent dossier — a real reference) |
| 4 explanatory notes | prose | — (correctly no misfire) |

Normalizer is format-tolerant (one structured id from many surface forms;
`12345-VII, blg I` ≡ `12345-VII-blg-1`) and does not fire on a bare year / page /
€ amount (GF-D1: number + government cue). Government refs are recorded regardless;
`OverheidResolver` resolution is best-effort enrichment (GF-D3). Academic-in-footnote
routes to GROBID (GF-D2).

**This closes the policy-citation gap**: the pipeline now picks up references to other
Kamerstukken/moties in footnotes — the primary citation type for the policy corpus.

Minor live-TODO: for a prose-embedded reference the `ParsedReference.raw_text` is the
whole footnote paragraph (the resolver still reads the dossiernummer from it) — could
be trimmed to the identifier + context.

## Remaining live TODOs
- **Convenant PDF path reconciliation** — `data/uploads/` vs `notebook_data/uploads/`
  (so the GF path can run over the convenanten too).
- **overheid.nl KOOP record mapping** (V.4) — query valid, record→work mapping needs a
  real dossier.
- **RePEc** — request `REPEC_API_KEY`, then live-verify CitEc.
- **A non-zero `cites` count** needs a corpus that cites itself (a data choice).
