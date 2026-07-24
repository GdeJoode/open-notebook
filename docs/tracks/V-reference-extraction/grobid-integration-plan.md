# Plan — GROBID reference parsing (replaces V.2/V.3) — PROPOSED

> **Status**: 📝 PROPOSED (2026-07-24), awaiting approval. Supersedes the
> hand-rolled V.2 segmenter + V.3 parser after the live smoke proved they are
> citation-style-dependent and brittle on real docling output. Keeps V.1 (region
> location) and V.5 (orchestration → U.3 `cites`).

## Why

The live pass exposed that hand-rolled segmentation/parsing is a treadmill:

- Parens-APA style (`Baumol, W. J. (2004).`) worked; the no-parens style
  (`Alesina A., Spolaore E., 2003.`) regressed the same doc to 1 reference.
- Fixing it meant per-style regex openers — the user's directive is the opposite:
  **the app must be citation-style-INDEPENDENT** (mostly APA, but not reliably).
- Real docling output also fragments one reference across chunks and straddles two
  references in one chunk, plus OCR artifacts ("fi rms", ligatures).

Style-independent, high-fidelity reference parsing is a **solved problem** via a
purpose-built ML model. **GROBID** (Apache-2.0, self-hosted, CRF) is the right
engine and fits the repo's "GPU/heavy code in its own HTTP service" constraint (§6)
— it sits next to the existing `docling` / `extraction` / `mineru` containers.
Free, local, no paid tier.

## Scope

**Replace** V.2 (`segmenter.py`) + V.3 (`reference_parser.py`) with a GROBID-backed
service. **Keep** V.1 (`region_locator.py` — locate the reference region) and V.5
(`ReferenceExtractionService` — orchestration to U.3). `ParsedReference` (the
V→U.3/V.4 contract) is unchanged — GROBID output is mapped onto it.

`835c13c` (the style-specific V.2 openers) is reverted. The two real chunk fixtures
(`econ_two_sections_chunks.json`, `centrifugal_noparens_chunks.json`) are kept as
GROBID regression data.

## Architecture

- New **`grobid`** service in `docker-compose.yml` — the **lightweight CRF image**
  (`lfoppiano/grobid:<pin>` or `grobid/grobid:<pin>`, ~500 MB, CPU-only; the ~8 GB
  DL image is overkill for reference parsing). Health via `/api/isalive`.
- New **`GrobidReferenceService`** — sends reference text (or the source PDF) to
  GROBID, parses the returned TEI-XML `<biblStruct>` entries → `List[ParsedReference]`
  (authors → surnames, `<title>`, `<date>` → year, `<idno type="DOI">` → `normalize_doi`).

## Decision gates

- **G-D1 (input path)** — GROBID reference endpoints: `/api/processReferences` (a
  **PDF** → GROBID finds + segments + parses the whole bibliography, best quality,
  uses GROBID's own layout parsing so it sidesteps docling's OCR degradation) vs
  `/api/processCitationList` (raw citation strings, needs pre-segmentation) vs
  feeding the V.1 region text. **Rec**: spike `processReferences` on the source PDF
  first (the papers' PDFs exist under `docling_input/` / managed dirs); fall back to
  region-text if PDF access is awkward in the pipeline. Resolve empirically in G.2.
- **G-D2 (fallback)** — keep the heuristic as a degraded fallback, or remove it?
  **Rec**: remove V.2/V.3 heuristic; GROBID is the engine. If GROBID is unreachable,
  V returns `[]` (best-effort, matching the guarded post-ingest hook) — never a crash.
- **G-D3 (image pin)** — pin a specific GROBID version for reproducibility.

## Phases (Infra → Backend → Integration; one PR per phase, `track/g<n>-…`)

### G.1 — GROBID service + smoke (Infra)
**Deliverables**: a `grobid` service in `docker-compose.yml` (CRF image, pinned),
health-gated; a smoke script.
**ACs**: (1) `docker compose up grobid` → `/api/isalive` returns `true`;
(2) a sample reference string through `/api/processCitation` returns TEI with a
parsed author + title + year; (3) image is version-pinned (G-D3).
**Evidence**: curl output of isalive + a parsed sample.
**PR boundary**: infra only, no app code.

### G.2 — `GrobidReferenceService` + TEI→ParsedReference mapping (Backend)
**Deliverables**: `references/grobid_reference_service.py` — client + TEI parser →
`List[ParsedReference]`. Resolve G-D1 here (PDF vs region-text).
**ACs**: (1) on `econ_two_sections_chunks.json` (or the source PDF) → both
bibliographies captured, **≥ 35** references, no style-specific code; (2) on
`centrifugal_noparens_chunks.json` (no-parens style) → **≥ 35** references, WITHOUT
any per-style branch — the same code path handles both styles; (3) DOI/year/authors/
title mapped from TEI; `normalize_doi` reused; (4) `Schumpeter (2002)` fragment and
the Alesina/Arzaghi straddle are handled by GROBID, not our code; (5) CI-offline: a
**recorded GROBID TEI-XML response** fixture drives the mapping test (no live service
in CI); a `@requires_grobid`-gated integration test hits the real container.
**Grep gate**: no identifier/comment naming a citation style (apa/ieee/harvard/…) as
a code path — GROBID owns style handling.
**Evidence**: per-fixture counts vs the heuristic's 7 / 1.
**PR boundary**: service + mapping + tests; not yet wired into V.

### G.3 — Swap into the V extractor (Integration)
**Deliverables**: `reference_extractor.py` routes region → `GrobidReferenceService`
instead of `segment_chunks` + `parse_reference`. Remove the V.2/V.3 heuristic modules
(or demote per G-D2). V.1 region location + V.5 orchestration unchanged.
**ACs**: (1) `extract_source_references` produces references via GROBID; (2) GROBID
unreachable → `[]`, logged, never raises (best-effort); (3) V.5 `materialize_corpus`
unchanged and still feeds U.3; (4) the removed heuristic leaves no dead imports.
**Evidence**: end-to-end on a fixture through the real service (gated).
**PR boundary**: wiring + removal; assumes G.1/G.2 merged.

### G.4 — Live validation + docs + close (Integration)
**Deliverables**: re-run the live pass on `staging`; update `live-smoke-results.md`
with the real GROBID counts; ARCHITECTURE reference-parsing section; close.
**ACs**: (1) live `refs_extracted` recorded per source (expect the papers' counts to
rise toward their true bibliography size vs the heuristic's 98 corpus total);
(2) `cites` behaviour unchanged (still external-dominated on this corpus); (3) docs
updated, track V reference-parsing marked GROBID-backed.
**Evidence**: live counts, before/after.

## Testing strategy
- **CI: offline + deterministic** — a recorded GROBID TEI-XML response fixture drives
  the mapping unit tests; no GROBID container in CI.
- **Integration: `@requires_grobid`-gated** — hits a real container (like the
  existing `@requires_docker` pattern).
- The two real chunk fixtures pin **style-independence** (a per-style regression is
  impossible when GROBID owns parsing).

## Reuse anchors
- `region_locator.py` (V.1) — keep, feeds GROBID the region (if G-D1 = region-text).
- `ReferenceExtractionService` (V.5) — keep, unchanged orchestration.
- `ParsedReference` + `normalize_doi` (`cites_matching.py`) — the output contract.
- compose service pattern (`docling`/`extraction`/`mineru`) — the `grobid` service mirrors it.
- `@requires_docker` test pattern — mirror for `@requires_grobid`.

## Risks & mitigations
1. **OCR-degraded input** (docling "fi rms", mid-word breaks) → prefer feeding GROBID
   the **source PDF** (its own layout parsing) over docling's OCR text (G-D1).
2. **New service weight** → lightweight CRF image (~500 MB, CPU); graceful-degrade so
   V works (returns `[]`) when GROBID is down.
3. **TEI mapping fidelity** → recorded-response fixture tests + the two real fixtures.

## Effort
Small–medium. Thin service, moderate client+mapper, small swap, small live-validate.
No changes to V.1 / V.5 / U.3 / the `cites` schema.

## Open questions
- [ ] G-D1: feed GROBID the **PDF** (best quality) or the V.1 **region text**?
- [ ] G-D2: remove the heuristic entirely (rec) or keep as offline fallback?
- [ ] Run GROBID CRF (fast, ~500 MB) or the DL image (heavier, marginally better)?
