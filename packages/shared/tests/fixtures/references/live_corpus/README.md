# Live-corpus fixtures — Track V reference extraction

Real documents committed **for the live smoke**, not for CI.

## Why these are here

The repo's only other committed PDFs (`apps/app-main/tests/fixtures/synthetic_*.pdf`)
are synthetic and carry **no reference section**, so they cannot exercise Track V's
bibliography location / segmentation / parsing. The real working corpus
(`data/`, `docling_input/`, `docling_output/`, `notebook_data/`) is **gitignored**,
so it is not reproducible for anyone else. These six files are a small, deliberately
diverse slice copied in (force-added past `.gitignore`) so the live validation run is
reproducible.

## What they are NOT for

**Do not use these in unit tests.** The offline V.1–V.3 tests run on the small
synthetic text fixtures in `../` (`references_apa.txt`, `references_numbered.txt`,
`references_dutch.txt`, `full_text_with_bib.txt`, `full_text_no_refs.txt`). Keep CI
offline, deterministic and fast — these PDFs are inputs for the manual/live smoke
only.

## The set (2.8 MB total) and what each covers

| File | ~Size | Exercises |
|---|---|---|
| `jcms-2025-ali-cohesion-policy.pdf` | 0.55 MB | DOI-rich Wiley journal → the Crossref / OpenAlex resolver legs |
| `economics-without-equilibrium.pdf` | 0.62 MB | economics paper, mixed DOI + author-year entries |
| `oecd-2008-charbit-blochliger-fiscal-equalisation.pdf` | 0.33 MB | OECD report style, author-year, smallest paper |
| `mapping-innovation-in-space.pdf` | 0.80 MB | regional-innovation journal, different venue shape |
| `convenant-het-hogeland.pdf` | 0.21 MB | Dutch Regio Deal policy doc → `OverheidResolver` (KOOP SRU) leg, non-paper citation shapes |
| `convenant-midden-limburg.pdf` | 0.23 MB | second Dutch policy doc (region variation) |

Bennett_1983 was considered and **rejected**: 28.7 MB, far too large to commit.

## Live-smoke checklist (the deferred ACs)

Run these against a live SurrealDB with the documents ingested, to close the
`# TODO(V-live)` items:

1. **V.1** — region located on each paper (`located_via=structure`, fallback to
   `full_text` where headings are generic); convenanten may legitimately have no
   bibliography → `located_via=none`, clean empty.
2. **V.2** — segmentation on real wrapped, no-blank-line bibliographies; a wrapped
   multi-line entry must stay ONE entry.
3. **V.3** — author parsing across APA/IEEE mixes, particle surnames ("van der"),
   "et al.", corporate authors.
4. **V.4** — provider contracts: arXiv Atom mapping, KOOP SRU record structure,
   ORCID `Accept: application/json`, CitEc contract (only once a RePEc key exists —
   the leg self-skips without one).
5. **V.5** — the headline AC: true end-to-end `cites` edge count. Note U.1 measured
   **0 intra-corpus citations** on the current corpus, so a non-zero count needs a
   corpus that actually cites itself — these six may legitimately yield 0 `cites`
   edges while still proving extraction works (check `refs_extracted`, not only
   `edges_materialized`).
