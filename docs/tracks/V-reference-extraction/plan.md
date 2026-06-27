# Track V — Reference / footnote extraction (AGENDA — not built yet)

> **This is a NOTE / agenda item, not an active build.** It records the upstream that must
> exist to FILL the `cites` edges. Track U.3 builds the cites *mechanism* (match parsed
> references → `cites` source→source edges); Track V is what FEEDS it: turning the
> documents' reference sections + footnotes into structured references. Captured at the
> user's request ("maak een notitie om die referenties te vullen") so it doesn't fall off
> the radar — it is currently NOT scheduled anywhere else (verified: Track I enrichment
> covers code/formula/picture only; the roadmap's "citations" is entity↔source provenance).

## Why it's needed
Measured on staging (Track U.1 + the 2026-06-27 reference probe):
- Docling produces NO `footnote`/`reference`/`bibliography` element types — only generic
  `list_item`/`text`/`heading`/`section_header`.
- BUT the reference *sections* ARE locatable: there are `section_header`/`heading` nodes
  literally titled "References" (9 found, incl. both papers), and the papers carry real
  bibliographies (~20 + ~12 DOIs in `full_text`).
- So the data is PRESENT but UNSTRUCTURED — nothing parses it into citable references today.

## Proposed approach (the user's idea)
1. **Locate the reference region** — find the `References`/`Referenties`/`Bibliography`
   `section_header` in the doc structure (or the bibliography block in `full_text`), plus
   inline footnotes/endnotes.
2. **A reference-FORM classifier** — recognize the SHAPE of each entry (numbered list,
   author-year, DOI line, footnote marker) and segment the region into individual references.
   Lightweight (regex/heuristics first; an embedding/LLM classifier only on ambiguous forms —
   the same cheap-first / LLM-on-the-margin cascade as `design-thematic-classification.md`).
3. **Parse each reference** into structured fields → `ParsedReference {authors, title, year,
   doi, venue, raw_text}` (the exact interface U.3 consumes).
4. **Targeted source lookup** (optional enrichment) — for an external reference, query the web
   (Crossref/DOI/title search) to resolve/confirm the cited work; create a stub node or fetch
   metadata. (Reuse Track K's external-id reconciliation + a precision guard.)
5. **Feed Track U.3** — hand the `ParsedReference` list per source to the cites mechanism,
   which matches against in-corpus sources (→ `cites` edges) and records external ones (stub).

## Boundary with U.3 (already being built)
- **U.3 (infrastructure, now)**: consumes `List[ParsedReference]` per source → confident
  intra-corpus match → `cites` source→source edges; external refs noted/stubbed. Tested on
  synthetic references; 0 live edges until V exists.
- **V (this note, later)**: produces the `ParsedReference` list from the actual documents.

## Open questions for when V is scheduled
1. Region detection from doc_node structure vs `full_text` regex — which is more robust given Docling's generic node types?
2. Footnote/endnote handling — Docling doesn't tag them; can they be recovered from layout/superscript, or only from `full_text` markers?
3. External lookup scope — DOI-only (cheap, precise) vs title/author web search (broader, noisier)?
4. Build as part of the Docling ingest (Track I-adjacent enrichment) or a post-ingest pass?
