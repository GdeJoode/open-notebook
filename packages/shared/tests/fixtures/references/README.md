# Track V reference fixtures (synthetic, committed)

Small, hand-authored fixtures for the Track V.1–V.3 producer pipeline. They mimic
the SHAPES of the real corpus (economics papers + Regio Deal convenanten) without
depending on the gitignored working data (`data/`, `docling_output/`), so they are
safe as CI fixtures.

| file | shape it exercises |
|------|--------------------|
| `references_apa.txt` | APA author-year bibliography, blank-line separated (wrapped multi-line entries), DOI lines |
| `references_numbered.txt` | IEEE-style numbered `[1]` list with quoted titles + wrapped continuations |
| `references_dutch.txt` | Dutch "Literatuur" section incl. a Kamerstuk citation + a `doi.org` URL |
| `full_text_with_bib.txt` | flat document text with a heading-only `References` line and an `Appendix` terminator after it (full_text fallback path) |
| `full_text_no_refs.txt` | a document body with NO bibliography (the `located_via="none"` case) |

The real papers are for the manual live smoke (`# TODO(V-live)`), not CI.
