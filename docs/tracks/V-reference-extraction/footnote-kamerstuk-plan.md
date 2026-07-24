# Plan — footnote + Kamerstuk reference path (policy documents) — PROPOSED

> **Status**: 📝 PROPOSED (2026-07-24), awaiting approval. A SEPARATE phase set
> (GF.*) alongside the GROBID academic-reference track (G.*). Found by a live test:
> the NPVR voortgangsbrief (30 Jan 2026) and all 10 Regio Deal convenanten yield **0**
> references from GROBID's bibliography model — because policy documents cite other
> documents in **footnotes**, using government-document identifiers, not scholarly
> bibliographies. For the policy corpus this is the **primary** citation type.

## The finding (live evidence)

- GROBID `processReferences` on the NPVR letter → **0** (HTTP 204): no scholarly
  bibliography section.
- GROBID `processFulltextDocument` → the letter's **footnotes** DO carry the
  cross-references: `Kamerstuk 31305-489`, `Motie 36410-111`. Our pipeline calls
  only `processReferences`, so these are missed entirely.

## Scope

Extract references from the **footnotes** of policy documents and route each by kind:

1. **Government-document reference** (Kamerstuk / Motie / begroting / bijlage) →
   normalize the identifier → resolve via the V.4 `OverheidResolver` (KOOP SRU).
2. **Academic reference** — a footnote may (rarely) hold a scholarly citation →
   route that footnote's text to GROBID `processCitation` and reuse `parse_grobid_tei`.
3. **Explanatory prose** (most footnotes) → drop.

Merges into the source's `ParsedReference` list alongside GROBID's bibliography refs
(G.*). Keeps V.5 orchestration + the `ParsedReference`/U.3 contract unchanged.

## Design principles (from operator guidance — read before coding)

- **Format-tolerant, NOT rigid.** The kamerstuk identifier appears in many surface
  forms; normalize to a structured id, never match one exact string. Known variants
  to cover (non-exhaustive — the detector must generalize, not enumerate):
  - `Tweede Kamer, vergaderjaar 24/25, 12345-VII, blg I`
  - `12345-VII-blg-1`
  - `Kamerstukken II 2024/25, 36410, nr. 111`
  - `Kamerstuk 31305-489`
  - `Motie 36410-111`
  So the identifier core is: **dossiernummer** (4–6 digits) + optional
  **begrotings-hoofdstuk** (Roman numeral, e.g. `-VII`) + a **sub-document** cue
  (`nr.` / `blg` / a trailing `-<n>`) + number. Separators (comma / space / dash)
  and lead words (`Kamerstukken II`, `Tweede Kamer`, `vergaderjaar`, `Motie`) vary.
- **This is NOT "chasing citation styles."** Unlike unbounded academic styles (which
  is why G.* delegates to GROBID), a kamerstuk identifier is a **bounded, stable
  scheme** — recognizing it is like recognizing a DOI or arXiv id. The tolerance is
  in the surface normaliser, not in per-style branches.
- **Mixed footnote content.** A footnote is not always a reference. Classify each:
  government | academic | prose. Do not force a kamerstuk parse on prose, and do not
  send prose to GROBID.
- **Precision over recall, logged.** A footnote that can't be confidently classified
  is dropped and logged, not guessed — mirrors the K.4/K.5 precision-guard posture.

## Architecture

- Input: GROBID `processFulltextDocument` on the source PDF → TEI `<note place="foot">`.
- New `FootnoteReferenceExtractor`:
  1. collect footnote texts (+ their marker ids) from the fulltext TEI;
  2. classify each (government / academic / prose);
  3. government → `KamerstukIdentifier` normalizer → `OverheidResolver` query;
  4. academic → GROBID `processCitation` → `parse_grobid_tei`;
  5. build `ParsedReference` (raw_text = the footnote text always; venue = e.g.
     `Kamerstuk`/`Motie`; the normalized identifier carried for the resolver).

## Decision gates (DECIDED 2026-07-24)

- **GF-D1 (detector tolerance)** — ✅ **DECIDED: number + cue.** Fire only on a
  dossiernummer (4–6 digits) **plus** a government cue (`Kamerstuk`/`Motie`/`blg`/
  `nr.`/`-<Roman>`/`vergaderjaar`/`Tweede Kamer`), so a bare number (page, year,
  € amount, a count) does not misfire.
- **GF-D2 (academic-in-footnote)** — ✅ **DECIDED: gate on a scholarly cue.** Route a
  non-government footnote to GROBID `processCitation` only when it shows a scholarly
  cue (author-year, DOI, a journal shape); else treat as prose.
- **GF-D3 ("resolved" vs "recorded")** — ✅ **DECIDED: edge only after resolve.** A
  government ref is always **recorded** as a `ParsedReference`, but it creates a
  `cites`/stub edge in U.3 **only when `OverheidResolver` (KOOP SRU) resolves it**.
  An unresolved identifier is retained on the reference (for later re-resolution) but
  does not yet produce an edge.

## Phases (Backend → Integration; one PR per phase, `track/gf<n>-…`)

### GF.1 — Footnote extraction (Backend)
**Deliverables**: extend `GrobidReferenceService` with `fulltext_footnotes(pdf) ->
List[str]` (POST `processFulltextDocument`, pull `<note place="foot">` texts);
best-effort (`[]` on failure).
**ACs**: (1) on the NPVR letter → extracts the 7 footnotes incl. `Kamerstuk 31305-489`
and `Motie 36410-111`; (2) a recorded `processFulltextDocument` TEI fixture drives the
offline test; (3) `@requires_grobid`-gated live test.
**Evidence**: the extracted footnote list.

### GF.2 — Kamerstuk identifier normalizer (Backend, pure)
**Deliverables**: `KamerstukIdentifier` + `parse_kamerstuk_identifier(text) ->
KamerstukIdentifier | None` — tolerant parse → `(dossier, hoofdstuk?, subtype?,
nummer?)`.
**ACs**: (1) all five listed variants normalize to the SAME structured id where they
denote the same document (`12345-VII, blg I` ≡ `12345-VII-blg-1`); (2)
`31305-489` → dossier 31305 nr 489; `Motie 36410-111` → dossier 36410 nr 111,
type=motie; (3) does NOT fire on a bare year (`2026`), a page number, or a `€`
amount; (4) a committed fixture of variant strings (incl. malformed) pins it.
**Evidence**: the variant → normalized-id table.

### GF.3 — Footnote classifier + router (Backend, pure)
**Deliverables**: `classify_footnote(text) -> {"government"|"academic"|"prose"}` +
routing.
**ACs**: (1) `Kamerstuk 31305-489` / `Motie 36410-111` → government; (2) an
author-year/DOI footnote → academic; (3) `Een overzicht … vindt u in de bijlage.` →
prose (dropped); (4) a mixed fixture covers all three.
**Evidence**: per-footnote classification of the NPVR set.

### GF.4 — Wire into the extractor + resolvers (Integration)
**Deliverables**: `FootnoteReferenceExtractor` runs GF.1→GF.3, routes government →
`OverheidResolver` (best-effort resolve), academic → GROBID `processCitation`, and
merges the resulting `ParsedReference`s into the source's list (alongside the G.*
bibliography refs) in the V.5 orchestration.
**ACs**: (1) NPVR letter → `Kamerstuk 31305-489` + `Motie 36410-111` appear as
references; (2) `OverheidResolver` resolves them where KOOP has the record, else they
stay recorded with the identifier; (3) a source with only prose footnotes → no spurious
refs; (4) V.5/U.3 contract unchanged.
**Evidence**: end-to-end on the NPVR letter (live).

### GF.5 — Live validation + docs + close (Integration)
Re-run over the policy corpus (convenanten + the NPVR letter); record counts in
`live-smoke-results.md`; ARCHITECTURE note; close.

## Testing strategy
- **Offline/deterministic**: recorded `processFulltextDocument` TEI fixture (NPVR
  footnotes) + a kamerstuk-identifier variant fixture + a mixed-footnote fixture.
- **Live-gated**: `@requires_grobid` for the GROBID calls; `OverheidResolver` uses
  its existing mocked-HTTP tests, plus an optional live KOOP check.

## Reuse anchors
- `OverheidResolver` (V.4, KOOP SRU) — the government-ref resolver.
- `GrobidReferenceService` + `parse_grobid_tei` (G.2) — footnote fulltext + academic
  `processCitation`.
- `ParsedReference` + `normalize_doi` — the contract.
- V.5 orchestration — merges footnote refs with bibliography refs; unchanged shape.

## Risks & mitigations
1. **Format variety beyond the listed variants** → tolerant normalizer + a variant
   fixture; log unparseable identifiers rather than guess (precision over recall).
2. **Number disambiguation** (page / year / amount vs dossiernummer) → require a
   government cue, not a bare number (GF-D1).
3. **Footnote extraction quality on scanned/OCR PDFs** → depends on GROBID's fulltext
   model; degrade to `[]` (best-effort), never crash.
4. **Academic-in-footnote is rare** → keep that path a thin GROBID `processCitation`
   call; don't over-engineer.

## Effort
Small–medium. GF.2/GF.3 are pure parsers (the real new work); GF.1/GF.4 reuse GROBID +
OverheidResolver. No changes to V.5/U.3/the `cites` schema.

## Open questions
- [ ] GF-D1 tolerance confirmed (number + cue)?
- [ ] GF-D2: gate academic-footnote routing on a scholarly cue, or always try GROBID?
- [ ] Should a recorded-but-unresolved government ref still create a `cites`/stub edge
      in U.3, or only when KOOP resolves it?
