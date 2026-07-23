# Track V — Reference / footnote extraction — status

> 📝 **PROPOSED (2026-07-23)** — sprint plan in [`plan.md`](./plan.md), **awaiting
> human approval**. Not yet implemented. V is the producer that feeds the already-
> built U.3 `cites` mechanism (which yields 0 live edges until V exists).

| Phase | Title | Status | Branch | Notes |
|---|---|---|---|---|
| V.1 | Reference-region location | proposed | — | structure-first + full_text fallback |
| V.2 | Segmentation + form classifier | proposed | — | cheap-first, LLM on the margin |
| V.3 | Parse → `ParsedReference` | proposed | — | reuses the U.3 contract (`cites_matching.py`) |
| V.4 | External resolution (opt-in) | proposed | — | multi-source `WorkResolver` cascade: OpenAlex + Crossref + RePEc/CitEc + overheid.nl (+ DataCite/arXiv/ORCID/ROR follow-ons); mirrors K.4 precision guard |
| V.5 | Orchestration → feed U.3 | proposed | — | post-ingest pass; live edge-count = deferred live AC |
| V.6 | Docs + RETRO + close | proposed | — | — |

See [`../_status.md`](../_status.md) → Open items / backlog (V was the one
genuinely open small track surfaced by the merge-readiness audit).
