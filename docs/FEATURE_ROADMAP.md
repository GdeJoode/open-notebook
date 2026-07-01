# Open Notebook — Feature Roadmap (v2)

> **Scope**: doorontwikkeling van open-notebook door cherry-picking uit drie referentie-projecten — `myKG` (knowledge graph extractor), `wiki_llm` (Writer-Evaluator-Editor wiki pipeline), en `llm-wiki` (multi-agent research prompt-kit). Plus uitbreiding tot **headless agent-platform** voor externe integraties (hermes, claude-code, etc.). Open-notebook blijft de **substrate** (Web-UI + SurrealDB + multi-notebook + workspace-monorepo); de drie repos leveren bouwstenen; Track G maakt het aanroepbaar door agents.
>
> **Datum**: 2026-06-03 — basis is de post-Phase-7 staat van open-notebook (zie `docs/REFACTOR_PLAN.md`).
>
> **Referentie-analyses**: `docs/MYKG_COMPARISON.md` (myKG diepte), `docs/LLM_WIKI_COMPARISON.md` (llm-wiki + wiki_llm).
>
> **Status**: V2 — alle Q1-Q9 beslispunten zijn opgelost, Track G (Agent Integration) toegevoegd.

## 0. Architecturale constraints

Elke nieuwe feature respecteert:

1. **Workspace structuur**: nieuwe code in `apps/`, `packages/`, `pipelines/`, of `services/` — niet aan de root.
2. **Service-naming-conventions** uit `ARCHITECTURE.md`: `*Service` / `*Orchestrator` / `*Processor` / `*Extractor` / `*Repository`.
3. **DI-container patroon**: alle services via factories in `apps/app-main/src/app_main/dependencies.py`.
4. **SurrealDB als persistence-laag** (geen filesystem-only state behalve voor zware artifacts zoals media output).
5. **Job-queue voor langlopende werk** (zie `packages/job-queue`).
6. **GPU-heavy code in een eigen HTTP service** (zoals docling/whisperx), geen direct linken in app-main.
7. **MIT-licensed code uit referentie-repos** mag direct geport worden mits attribution in de file-header.
8. **Agent-first API surface** — elk meaningful capability is via REST aanroepbaar (Track G); UI is één front, niet de enige.

Wat we **niet** overnemen:
- myKG's pure-CLI / session-dir-state model (in tegenspraak met SurrealDB-persistente notebooks)
- wiki_llm's `content_new/ → content_processed/` filesystem-routing (we hebben job-queue)
- llm-wiki's hub/registry concept buiten de UI (in open-notebook is een notebook al de hub)

## 1. Q1-Q9 + G — opgeloste beslispunten

| # | Decision | Samenvatting finaal |
|---|---|---|
| Q1 | Obsidian export-stijl | myKG-puur: platte vault `vault/<entity>.md`, filter-defaults `min_conn=5, min_conf=0.9`, handmatige overrides. Notebook-overview + cross-notebook merge zijn aparte features. |
| Q2 | Repair / cleanup | Full myKG `orphan_connector` port (~1060 LOC) + prune-lifecycle voor blijvende orphans (recoverable archive). LangGraph + lint komen mee met hun parent-features. |
| Q3 | Quality-gates | Drie lagen: confidence-scoring (always-on, gratis) → Writer-Evaluator-Editor (per-summary opt-in) → Audit (8 concrete checks, 6 LLM-vrij). |
| Q4 | markdown-hero | Adopt + wrapper-module in `packages/shared/utils/markdown.py`. 6 use-cases ontgrendeld. |
| Q5 | BM25 fallback | Skip — bestaande hybrid-search heeft al BM25 via SurrealDB. Wel UI-toggle voor text/vector/hybrid modes. |
| Q6 | Multi-agent research | E1 single-agent + E3 thesis-mode samen; Tavily-first, DuckDuckGo fallback. Plus review-found-sources stap. E2 multi-agent later als optimalisatie. |
| Q7 | Typed collectors | Geen aparte collector-feature. Wel lichte **binary routing layer**: research papers → Zotero (per-notebook collection), andere binaries → managed disk + link. Strict heuristic met flag-for-review bij twijfel. |
| Q8 | Schema review UX | Schema-als-artifact (niet-blocking) + soft nudge na Pass 1. **View + edit in V1**. Pass 1 doet schema-validatie + delta-detection bovenop bestaande detector. **Multi-schema sequentieel top-3** met merge-step. |
| Q9 | Vocabulary stack | TOOI volledig + maandelijks refresh + user-overlay (global + per-notebook) + **Crossref direct** voor research papers. Architecturaal prepared voor ORCID/arXiv/Wikidata. Plus myKG name_normalizer + NL-uitbreiding. |
| G-Q1 | Agent API auth | Simple API-keys (header-based), per-agent. |
| G-Q2 | File-watcher | Always-on op conventional path (configurable). |
| G-Q3 | Conflict resolution | Non-blocking — warning-badge in UI, werk gaat door. |
| G-Q4 | Summary templates v1 | `literature_note` + `meeting_notes` als eerste, anderen op aanvraag. |

---

## 2. Zeven parallelle tracks

### Track legend

- 🅐 = porting uit **myKG**
- 🅑 = porting uit **wiki_llm**
- 🅒 = porting uit **llm-wiki**
- 📦 = nieuwe workspace member

---

## Track A — Ingestion robustness (MinerU naast docling)

> **Status**: ✅ **COMPLETE (2026-06-04)** — A1 (MinerU service +
> dispatcher) + A2 (confidence-fallback + UI badge + slider + health
> chip + per-source override) shipped across six PRs. See
> [`docs/tracks/A-mineru/RETRO.md`](./tracks/A-mineru/RETRO.md) for
> the retrospective and
> [`docs/tracks/A-mineru/threshold-tuning.md`](./tracks/A-mineru/threshold-tuning.md)
> for the default-0.95 calibration decision. A3 (markitdown / ephemeral
> venv reuse) remains as "nice-to-have" follow-up; not blocking.

**Vision**: twee parser-routes in de UI (docling/MinerU) + automatische fallback wanneer docling-confidence te laag is.

### A1 — MinerU als parallelle service 📦

**Implementatie**:

1. Nieuwe service: `services/mineru/`
   - `Dockerfile` op CUDA-base
   - `api.py` met FastAPI endpoint `POST /parse`
   - Volume-mount `/data/input` + `/data/output`
   - Installs `mineru[all]` in een **ephemeral uv-venv** (overnemen van 🅐 myKG's `src/mykg/uv_venv.py::ephemeral_mineru_venv`)
2. Compose service `mineru:` in `docker-compose.yml` met CUDA reservations
3. HTTP client: `apps/app-main/src/app_main/services/mineru_http_client.py`
4. Pipeline integratie: nieuwe content-setting `parser_engine: "docling" | "mineru" | "auto"`
5. UI: dropdown in Settings + per-source override via `processing_overrides`

**Effort**: 2-3 dagen. **Risico**: 🟡 medium (eerste keer tweede GPU-service).

### A2 — Sequentiële MinerU fallback bij lage docling-confidence

**Confidence-metric** voor docling, geconstrueerd uit 6 signalen:

| Signaal | Gewicht |
|---|---|
| OCR-confidence avg | 30% |
| Text density (chars/page) | 20% |
| Heading detection rate | 15% |
| Table parsing success | 15% |
| Image-to-text ratio | 10% |
| Element_type onbekend % | 10% |

Threshold default `0.95`, user-configurable. Module: `apps/app-main/src/app_main/services/parsing/confidence.py`.

Implementatie in `SourceExtractor._process_file` (Phase 3 refactor):
```python
if parser_engine == "auto":
    docling_result = await self._extract_with_docling(...)
    conf = score_docling_extraction(docling_result)
    if conf < settings.docling_min_confidence:
        mineru_result = await self._extract_with_mineru(...)
        return select_best(docling_result, mineru_result)
    return docling_result
```

Persisteren confidence in `Source.metadata.extraction_confidence`; UI toont badge bij fallback.

**Effort**: 3-4 dagen + 1-2 weken threshold-tuning op realistische input.

### A3 — Bestaande docling-improvements (nice-to-have)

- Markitdown als 3e parser-optie (lichtgewicht, geen ML). Bron: 🅑 wiki_llm.
- Ephemeral-venv pattern hergebruiken voor andere optionele heavy deps.

---

## Track B — KG quality (multi-schema + formele discipline)

> **Status**: ✅ **COMPLETE (2026-06-12)** — Multi-schema KG extraction,
> schema-edit UX, telemetry, orphan-lifecycle, cross-notebook merge.
> All 18 PRs merged (17 production sub-phases — B.0, B.1a–B.1f,
> B.2a/B.2b, B.3a–B.3d, B.4, B.5a/B.5b, B.6 — plus B.7 integration/retro).
> See [`docs/tracks/B-kg-quality/RETRO.md`](./tracks/B-kg-quality/RETRO.md)
> for the retrospective and
> **B.8 follow-up (2026-06-21)**: live-validation surfaced + fixed a 7-bug chain blocking KG persistence + the Q-B-1 schema drift (migration 50); qwen extraction verified end-to-end; cross-doc resolution PARTIALLY MET (V1-capped → M4). See [`reviews/phase-B.8-findings.md`](./tracks/B-kg-quality/reviews/phase-B.8-findings.md).
>
> [`docs/tracks/B-kg-quality/status.md`](./tracks/B-kg-quality/status.md)
> for the rolling per-phase log.

| Phase | Title | Status |
|---|---|---|
| B.0 | Testcontainers SurrealDB harness | ✅ |
| B.1a | Entity/Relation models + persistence drift fix + migration 44 | ✅ |
| B.1b | `notebook_schema` + `pass1_results` tables + repos + migration 45 | ✅ |
| B.1c | Pass-1 schema-validation module | ✅ |
| B.1d | Pass-2 typed-extraction module (confidence-everywhere) | ✅ |
| B.1e | Multi-schema orchestrator + merge step | ✅ |
| B.1f | EntityExtractionService rewire + LLMExtractor DI fix | ✅ |
| B.2a | TTL/RDFS exporter fix + roundtrip test | ✅ |
| B.2b | `GET /api/notebooks/{id}/schema.ttl` endpoint | ✅ |
| B.3a | Schema-tab view-only UI | ✅ |
| B.3b | Schema edit operations (rename / merge / split / delete) + migration 46 | ✅ |
| B.3c | Soft-nudge banner + per-notebook pause toggle | ✅ |
| B.3d | Schema-change → re-extract prompt | ✅ |
| B.4 | Confidence display + filter + always-on telemetry + migration 47 | ✅ |
| B.5a | Orphan-connector (co-occurrence + LLM-confirm) | ✅ |
| B.5b | Orphan prune-lifecycle + migration 48 | ✅ |
| B.6 | Cross-notebook graph merge | ✅ |
| B.7 | Track integration + ARCHITECTURE + RETRO + CHANGELOG | ✅ |

**Vision**: KG wordt grounded in echte ontology met validatie. Schema is editable in UI. Multi-schema sequentieel voor cross-domain documenten.

### B1 — Two-pass extractie met multi-schema + extension-detection

**Bron**: 🅐 myKG `src/mykg/pass1.py` + `pass2.py`, aangepast aan open-notebook's multi-schema realiteit.

**Architectuur** (zie Sectie 3.1 voor uitgebreide design):

```
Document ─► Detector (bestaand) ─► Pass 1 (LLM, lichtgewicht)
                                          │
                                          ├─ Confirm/refine schema-keuze
                                          ├─ Coverage-check (% gedekt)
                                          ├─ Propose extensions
                                          ▼
                                  Soft-nudge UI (Q8)
                                          │
                                          ▼
                                   Pass 2 (entity extraction)
```

**Multi-schema mode** (top-3 schemas applicabel):
- Sequential 3× Pass 1
- Soft-nudge cumulatief
- Sequential 3× Pass 2 met respective accepted extensions
- Merge-step: cross-pass entity-dedup (via Q9 name-normalizer) + multi-type tagging

**Implementatie**:
- Module: `pipelines/ontology-extraction/src/ontology_extraction/pass1_schema_validation.py`
- Module: `pipelines/ontology-extraction/src/ontology_extraction/pass2_typed_extraction.py`
- Module: `pipelines/ontology-extraction/src/ontology_extraction/multi_schema_orchestrator.py`
- Storage: nieuwe SurrealDB table `notebook_schema` (per notebook)
- Storage: `pass1_results` table (coverage, extensions, used)

**Effort**: 1.5-2 weken (vereenvoudigd door multi-schema setup t.o.v. myKG pure-induction).

### B2 — TTL/RDFS export met Protégé-compatibiliteit

**Bron**: 🅐 myKG `src/mykg/ttl_validator.py`.

1. Vervang/aanvul bestaande `packages/ontology-manager/src/ontology_manager/rdf_owl_shacl.py` (en fix bestaande rdflib-import bug uit REFACTOR_PLAN follow-up).
2. Port `ttl_validator.py` (rdflib syntax + semantische checks).
3. API endpoint: `GET /api/notebooks/{id}/schema.ttl`.
4. UI: download-knop + "Edit in Protégé" instructie.

**Effort**: 3-4 dagen.

### B3 — Schema review UX (Q8 finalized)

**Implementation aspects**:

1. **Schema-tab** in notebook-UI (altijd zichtbaar):
   - Classes + properties browsen
   - Per-source coverage-stats kolom
   - Pending extensions sectie
   - Edit operations: rename, merge, split, delete (V1)
   - TTL export-knop (via B2)

2. **Soft nudge** na eerste Pass 1:
   - Dismissible notification: "Schema gegenereerd: N classes, M properties. [Review] [Use as-is]"
   - `[Use as-is]` is silent default
   - Per-notebook "don't show again" optie

3. **Optional pause-toggle** (default off):
   - Per-notebook setting `schema_review_required: bool`
   - Als true: workflow stopt na Pass 1, wacht op `[Approve]`

4. **Schema-change → re-extract prompt**:
   - Na schema-edit: "Schema gewijzigd. Bestaande sources re-extracten? [N affected sources]"

**Effort**:
- Schema-tab view-only: 3-4 dagen
- Edit operations (rename/merge/delete/split): 1 week
- Soft-nudge + pause-toggle: 2-3 dagen
- Schema-change → re-extract flow: 2-3 dagen
- TTL roundtrip (export → Protégé edit → import): 3-4 dagen
- **Totaal**: 2.5-3 weken

### B4 — Confidence scoring overal

**Bron**: 🅐 myKG — verspreid over modules.

1. Schema-extensie: `Entity` model krijgt `confidence: float` veld (`packages/shared/.../models/entity.py`).
2. Idem `Relation`, `Attribute`.
3. Pass 2 prompts returnen confidence als structured output veld.
4. UI: progress-bar per entity/relation; filter "Verberg < N".
5. Synergie: Pass 1's `coverage_pct` is een confidence-score → same data-flow.
6. Synergie: A2 MinerU-fallback gebruikt zelfde confidence-paradigma.

**Effort**: 1 week (model + prompts + UI tonen + filter).

### B5 — Orphan-connector + prune-lifecycle (Q2 finalized)

**Bron**: 🅐 myKG `src/mykg/orphan_connector.py` (1060 LOC).

**Implementation**:

1. Nieuwe module: `pipelines/entity-filtering/src/entity_filtering/resolution/orphan_connector.py`.
2. Run na entity-extraction, voor finale dedup.
3. Heuristic: co-occurrence in zelfde chunk.
4. LLM-confirm proposed edges.
5. **Prune-lifecycle**:
   - 1e run: probeer connect → success of fail
   - Bij fail: entity status `pending_reconnect`
   - Bij elke nieuwe source-import in notebook: re-try voor pending entities
   - Na N pogingen (default 3) of M tijd (default 90 dagen) → status `archived`
   - Archive is recoverable, niet hard-delete
6. UI: per-notebook toggle "Auto-reconnect orphans" + dashboard met pending/archived counts.
7. Audit-laag (F1) surfaces long-pending orphans als action items.

**Effort**: 1.5 weken (port + prune-lifecycle + tests).

### B6 — Cross-notebook graph merge

**Bron**: 🅐 myKG `src/mykg/merger.py` + `merge_orchestrator.py`.

1. API: `POST /api/notebooks/merge` met `{ source_ids, target_id }`.
2. Backend port van myKG's merger.
3. UI: "Merge into..." dropdown in notebook-menu.

**Effort**: 1-2 weken.

---

## Track C — Content quality (Writer-Evaluator-Editor + markdown lint)

### C1 — Writer-Evaluator-Editor chain voor summaries

**Bron**: 🅑 wiki_llm `src/pipeline.py` generate-stage.

1. Nieuwe enhancer: `pipelines/summarization/src/summarization/enhancers/writer_evaluator_editor.py`.
2. Configureerbaar via SummarizationConfig: `enhancer: "writer_evaluator_editor"`.
3. Wrapt elke strategy (naive/raptor/treekg/refine): final summary door 3-pass review.
4. **Opt-in** per-call (default off). UI-toggle: "High-quality mode" voor publicatie-rijp content.
5. Vult de bestaande `chain_of_density.py` / `self_correction.py` stubs.

**Effort**: 1 week (wiki_llm's ~150 LOC port + prompt-tuning).

### C2 — markdown-hero (Q4 finalized)

**Bron**: 🅑 wiki_llm dep.

1. Add `markdown-hero` als dep in `packages/shared/pyproject.toml`.
2. Wrapper-module: `packages/shared/src/shared/utils/markdown.py` (façade pattern).
3. 6 integratiepunten over de tijd:
   - Frontmatter parsing in `pipelines/ingestion` (direct)
   - `strip()` voor content-hash IDs (C4)
   - `extract_chunks(purpose="rag")` voor re-imported notes
   - `markdown_merge` voor summary consolidation
   - `lint` voor audit-laag (F1)
   - `word_format` voor .docx export (C3)

**Effort**: 1-1.5 dag wrapper + first integration; andere integraties komen mee met parent-features.

### C3 — Word (.docx) export

Via `markdown_hero.word_format()`.

1. Endpoint: `POST /api/notebooks/{id}/export.docx` (en per-source/summary).
2. UI: "Export as Word" knop op notebook + source detail-pages.

**Effort**: 2-3 dagen.

### C4 — Deterministic UUID van content-hash

Source-IDs en chunk-IDs uit `sha256(stripped_content)[:16]` i.p.v. path-based.

1. Migration: bestaande IDs behouden via lookup-table; nieuwe extractions content-hash.
2. Voorkomt ID-changes bij file-rename.

**Effort**: 1-2 dagen + migration script.

---

## Track D — Output rijkdom (Obsidian / TTL / JSONL / NetworkX)

> **Status**: ✅ **COMPLETE (2026-06-16)** — Obsidian export
> (zip + direct-to-vault), JSONL streaming export, NetworkX 7-format
> export, shared counts-only preview surface, async vault-path job.
> All 8 PRs merged (7 production sub-phases — D.0, D.1a, D.1b, D.1c,
> D.2, D.3 — plus D.4 integration/retro).
> See [`docs/tracks/D-output-richness/RETRO.md`](./tracks/D-output-richness/RETRO.md)
> for the retrospective,
> [`docs/tracks/D-output-richness/E2E_EVIDENCE.md`](./tracks/D-output-richness/E2E_EVIDENCE.md)
> for the smoke evidence (live consumer-tool smoke deferred to
> operator session per sandbox limitation), and
> [`docs/tracks/D-output-richness/status.md`](./tracks/D-output-richness/status.md)
> for the rolling per-phase log.

| Phase | Title | Status |
|---|---|---|
| D.0 | Export contracts + notebook-scoped repo projections (Q-D-1/2/4) | ✅ |
| D.1a | Obsidian zip exporter + service + endpoint | ✅ |
| D.1b | Obsidian direct-write-to-vault + `JobType.EXPORT_OBSIDIAN` handler | ✅ |
| D.1c | Obsidian export UI dialog + `/export-preview` parity surface + E2E | ✅ |
| D.2 | JSONL streaming exporter + endpoint + popover | ✅ |
| D.3 | NetworkX 7-format exporter + endpoint + format dropdown | ✅ |
| D.4 | Track integration + ARCHITECTURE + RETRO + troubleshooting doc | ✅ |

**Vision**: KG wordt downloadable in de drie consumer-formaten die de
roadmap §340 finaliseerde — Obsidian (mens-leesbaar), JSONL (Neo4j +
RAG), NetworkX (Gephi + Python notebooks). Eén filter-pipeline shared
tussen alle exporters + de preview, zodat "X entities will be
exported" in de UI **exact** klopt met wat er in het zip-bestand zit.

### D1 — Obsidian vault export (Q1 finalized)

**myKG-puur, platte vault.**

1. Endpoint: `POST /api/notebooks/{id}/export-obsidian` → zip-download of direct-write naar configured vault-path.
2. Per entity: `<vault>/<entity_name>.md`:
   - **Frontmatter**: `id`, `type`, `confidence`, `external_ids` (TOOI/Crossref URIs), `aliases`, `sources` (citations)
   - **Body**: attributes als bullet list
   - **Wikilinks** naar gerelateerde entities (op basis van relations)
3. Index-bestand `<vault>/README.md` met graph-overzicht.
4. **Filter-defaults voor auto-pipeline**: `min_connections=5`, `min_confidence=0.9`.
5. **Handmatige UI** met sliders voor custom thresholds.

**Effort**: 1 week (port + zip-streaming + UI download-knop + filter-vorm).

### D2 — JSONL export (Neo4j/RAG ready)

Streaming JSONL voor Neo4j / RAG-pipelines.

**Effort**: 2-3 dagen.

### D3 — NetworkX export (7 formaten)

GraphML, GEXF, GML, JSON-tree, edge-list, adjacency-list, pickle.

**Effort**: 1-2 dagen — myKG's exporter direct hergebruikbaar.

---

## Track E — Research workflows (E1 + E3 first, E2 later)

### E1 — Single-agent research

**Bron**: 🅒 llm-wiki `research.md` command (vertaald naar Python orchestrator).

**Architectuur**:
```
User: "research X" of "thesis: Y"
        │
        ▼
job-queue: JobType.RESEARCH
        │
        ▼
handle_research handler:
   1. Parse intent (LLM)
   2. Generate search queries (LLM)
   3. Tavily search → URL list
   4. **Review-stap** (Q7-confirmed):
      User sieve found URLs → keep/discard subset
   5. Per kept URL: `source_processor.process_source(url=...)`
      met **binary routing** (Q7):
        - PDF + paper-heuristic → Zotero collection
        - other → docling/mineru
   6. Synthesis-LLM-call → resulterende `Source` type `research_synthesis`
```

**Web-search**: Tavily-first, DuckDuckGo fallback. Geabstraheerd via `WebSearchClient` interface in `apps/app-main/src/app_main/services/web_search/`.

**UI**: research-modal op notebook-page met query input, source-count slider, depth (quick/standard/deep), mode-select (query/thesis).

### E3 — Thesis-mode

Bovenop E1: query split in for-evidence + against-evidence, parallel searches, verdict-synthesis.

Source-tagging in SurrealDB: `evidence_polarity: "for" | "against" | "neutral"`.

Synthesis-rapport sectie "Voor" / "Tegen" / "Verdict + confidence".

### E2 — Multi-agent (future optimization)

Planning-stap die query opbreekt in 3-7 subqueries → parallelle subagents per subquery → master synthesis.

**Effort gehele Track E**:
- E1 incl. WebSearchClient + review-step + binary routing: 2.5-3 weken
- E3 thesis-mode bovenop: 1 week
- E2 multi-agent (future): 1-2 weken
- **Direct te leveren**: E1+E3 = 3.5-4 weken

---

## Track F — Operations & quality (audit, librarian, resumable)

### F1 — Audit met 8 concrete checks (Q3 finalized)

**Always-on dashboard widget** (6 LLM-vrije checks):
1. Citation completeness — entities zonder source-link (error)
2. Stale source detection — sources > threshold leeftijd (info)
3. Low-confidence survivors — entities net boven filter (warn)
4. Long-pending orphans — Q2 prune-candidates (warn)
5. Community quality — clusters met lage cohesion (info)
6. Schema drift — veel fallback-type entities (info)

**On-demand "Deep audit"** (LLM-calls):
7. Conflicting facts — attribute Y=A in S1 vs Y=B in S2 (warn)
8. Provenance gaps — relations zonder duidelijke source-evidence (warn)

**Optional periodiek background-job** (opt-in per notebook).

**Effort**:
- Phase 3a (6 LLM-vrije checks + widget): 4-5 dagen
- Phase 3b (deep audit + checks 7-8): 1 week
- Phase 3c (background-job): 3-4 dagen
- **Totaal**: ~2 weken

### F2 — Librarian background-task

🅒 llm-wiki librarian-pattern: cron-style periodic checks per notebook. Bovenop F1.

**Effort**: 1 week.

### F3 — Resumable pipeline / step-level recovery

Source.status fine-grained: `extracted`, `chunked`, `embedded`, `entities_extracted`, `summarized`. Resume vanaf laatste-succesvolle stap.

**Effort**: 1 week.

---

## Track G — Agent Integration & Headless Mode (NEW)

**Vision**: open-notebook als headless backend dat externe agents (hermes, claude-code, cursor, custom) via API kunnen aanroepen voor document-processing, summary-generation, en KG-extractie. Bidirectionele sync met Obsidian-vaults voor cross-front edit-flow.

Zie Sectie 3.3 voor architectuur-deep-dive.

### G1 — Public Agent API

**Endpoints** (versioned `/api/v1/agents/`):

| Endpoint | Doel |
|---|---|
| `POST /agents/process-document` | PDF/DOCX/etc. via path of upload → Source met chunks/entities/summary |
| `POST /agents/process-audio` | Audio via path → transcript + summary |
| `POST /agents/process-url` | URL → ingest + summary |
| `POST /agents/generate-summary` | Raw text + template-name → summary |
| `POST /agents/extract-entities` | Raw text + schema → typed entities |
| `GET /agents/jobs/{job_id}` | Job status + result |
| `POST /agents/webhooks` | Register callback URL per agent + event types |
| `GET /agents/audit-log` | Per-agent audit trail |

**Authentication**: API-keys via `X-API-Key` header. Per-key:
- `agent_id`
- Permissions (read-only / write / admin)
- Rate-limits
- Audit-log entries

Storage: nieuwe SurrealDB tabel `agent_keys`.

UI: "API Keys" tab in settings — generate, revoke, view audit-log per key.

**OpenAPI spec / Swagger** auto-generated; endpoint `GET /api/v1/agents/openapi.json`.

**Effort**: 1-1.5 week.

### G2 — File-watcher service (always-on)

**Conventional paths** (defaults, configurable):
- `~/open-notebook/inbox/` — globaal watched
- `<notebook_data>/<notebook_id>/inbox/` — per-notebook watched

**Mechanisme**:
- `watchdog` library voor cross-platform file-events
- Debounced (2-5s) om burst-writes te clusteren
- Recursive scan-on-startup voor backlog
- File-type detection → route naar juiste handler:
  - `.pdf .docx` → handle_process_document
  - `.mp3 .m4a .wav` → handle_process_audio
  - `.url .webloc` → handle_process_url
- Moved-to-processed pattern: na succes, `mv` naar `<inbox>/_processed/`; bij failure naar `<inbox>/_errors/`

**Effort**: 3-4 dagen.

### G3 — ObsidianSyncService — write side

**Templates** (V1 Q7-confirmed: `literature_note`, `meeting_notes`):

```
vault/
├── literature_notes/
│   ├── <slug>.md                    # generated per Source
│   └── ...
├── meeting_notes/
│   ├── <slug>.md
│   └── ...
└── entities/                         # Q1 atlas export
    └── ...
```

**Template-driven rendering** via `markdown-hero` (Q4):

```python
class TemplateRenderer:
    def render_literature_note(self, source: Source) -> str:
        # YAML frontmatter + structured body
        # Sections: Abstract, Key concepts, Entities (with TOOI/Crossref links),
        #           Methodology, Findings, Citations
        ...
```

**Atomic writes**: write to `.tmp`, fsync, rename — voorkomt half-written files.

**Effort**: 1 week.

### G4 — ObsidianSyncService — read side (sync)

**De moeilijkste component.**

Architectuur:
```
File-watcher op vault
       │
       ▼
Change detected (debounced)
       │
       ├─ Parse markdown (markdown-hero)
       ├─ Diff tegen stored versie (content-hash)
       │
       ├─ Body changed?
       │   ├─ Re-extract entities op nieuwe content (Pass 2)
       │   ├─ Diff entity-set vs stored
       │   ├─ Add new, mark removed als pending-prune
       │   └─ Trigger quality-check (F1 audit) op nieuwe state
       │
       ├─ Frontmatter changed?
       │   └─ Update Source-metadata in DB
       │
       └─ Conflict (UI heeft ook gewijzigd sinds last sync)?
           ├─ Status: "needs_merge"
           ├─ UI badge "Conflict pending" (Q7 G-Q3 = non-blocking)
           └─ Surface in conflict-resolution panel
```

**Conflict resolution UI** (non-blocking):
- Lijst van pending conflicts in notebook-sidebar
- Per conflict: diff-view (UI versie vs vault versie)
- User picks: "keep UI", "keep vault", "manual merge"
- Werk gaat door op andere notes terwijl conflicts pending zijn

**State tracking**: nieuwe tabel `sync_state` met `{source_id, last_synced_hash, last_synced_at, last_origin, conflict_status}`.

**Effort**: **2 weken** — moeilijkste deel van Track G.

### G5 — Webhook outbound + audit log

**Event types** waar agents zich op kunnen subscriben:
- `job.complete`
- `job.failed`
- `sync.conflict`
- `quality.warning` (audit-laag)
- `entity.extension_proposed` (Q8 schema-extension)

**Per-agent config**: webhook URL + secret (HMAC sign).

**Retry-policy**: exponential backoff, max 5 retries, dead-letter queue.

**Audit log**: every API-call + every webhook-fire persistent in SurrealDB.

**Effort**: 1 week.

### G6 — Specific summary templates

**V1 (Q-G4 confirmed)**:

1. **`literature_note`** — voor research-papers:
   - Frontmatter: title, authors, year, DOI (uit Crossref), abstract, tags
   - Sections: Key concepts, Methodology, Findings, Critique, Citations
   - Wikilinks naar entities (researchers, organizations, concepts)
   - Implementation: `pipelines/summarization/.../templates/literature_note.py`

2. **`meeting_notes`** — voor audio-transcripts:
   - Frontmatter: date, deelnemers (TOOI-resolved), duration, location
   - Sections: Besluiten, Action items (met assignee), Quotes (met timestamps), Open vragen
   - Wikilinks naar deelnemers + onderwerpen
   - Implementation: `pipelines/summarization/.../templates/meeting_notes.py`

**Future templates** (on-demand): `research_synthesis` (uit E1), `policy_brief`, `executive_summary`, etc.

**Template config**: YAML-driven prompts in `pipelines/summarization/src/summarization/templates/<name>.yaml`.

**Effort**: 1-1.5 week voor 2 templates met Writer-Evaluator-Editor enhancer (Q3 Laag 2).

---

## Track H — Vision-model parser tier (DEFERRED — start na Track G)

**Status**: ⏸ Geplanned, **niet beginnen voor Track G compleet is**. Toegevoegd 2026-06-05 op user-request tijdens Track B.1c.

**Probleem**: Track A's auto-fallback is binair — docling óf MinerU. Drie scenario's vallen tussen wal en schip:

1. **Tabellen** waar zowel docling als MinerU de structuur fout interpreteren (samengevoegde cellen, multi-line headers, nested tables).
2. **Figuren** met embedded text, diagrammen, of charts — beide parsers vlakken die af tot ruwe pixels of niets.
3. **Fallback-van-fallback**: zowel docling als MinerU produceren lage-confidence output (beide onder threshold). Nu wint MinerU per default (Q-A-2 V1-trust). Een derde optie kan helpen.

### H1 — Vision-model parser als parallelle service

**Aanpak** (vergelijkbaar met A1 MinerU-service):

- Nieuwe Docker service (bv. `vision-parser` in `services/vision-parser/`) wraps een multimodaal model. Kandidaten:
  - **Local**: `Qwen2-VL-7B`, `MiniCPM-V`, of `Llama-3.2-Vision` via vLLM
  - **Hosted**: GPT-4o, Claude Sonnet, Gemini 1.5 Pro vision-mode
  - User-configurable via Settings, zoals andere model-keuzes
- HTTP API: `POST /extract` met file + region-of-interest hints (pagina-range, bbox)
- Output: structured table HTML, figure-description, of extracted text

### H2 — Hybride routing (per-element)

- Auto-fallback wordt drie-fase i.p.v. twee:
  1. Docling primary
  2. MinerU als docling.confidence < threshold
  3. **Vision parser** voor specifieke elementen waar MinerU's confidence per-element ook laag is
- Vereist **per-chunk/per-element confidence** uit de note `docs/tracks/A-mineru/CONFIDENCE_GRANULARITY_NOTE.md` (Optie A) — Track H is dus afhankelijk van die feature.
- Result-merge: docling tekst + MinerU layout + vision tables/figuren

### H3 — UI granulariteit

- Parser-engine dropdown krijgt 5e optie: "Vision (tables/figures only)" en "Vision (full fallback)".
- Badge wordt rijker: "docling+mineru+vision (3 fragments)".
- Per-source override in reparse modal: kies welke parser voor welk segment.

**Vereisten voor start van Track H**:

1. Track G volledig gemerged (geen scope-conflict met Agent platform).
2. Per-chunk confidence ingebouwd (Optie A uit CONFIDENCE_GRANULARITY_NOTE).
3. Live-test feedback van Track A bevestigt dat tabellen/figuren een echt probleem zijn (anders YAGNI).
4. Vision-model keuze gemaakt (lokaal vs. hosted; budget-impact).

**Effort schatting**: 2-3 weken (1 nieuwe service, hybride routing, UI rijker, threshold tuning).

**Risico's**:

- Vision-models zijn 5-10× duurder in inference dan docling/MinerU. Threshold en routing moeten conservatief zijn.
- Lokale modellen (Qwen2-VL etc) vereisen GPU met ≥16GB VRAM; user's H100-class is overkill maar niet alle deployments hebben dat.
- Hosted vision modellen brengen data-soevereiniteit issues mee voor users die on-prem draaien.

---

## 3. Architecturale deep-dives

### 3.1 Multi-schema two-pass design

**Probleem**: Ollama-context-budget + multi-domain documents.

**Oplossing**: Pass 1 = **schema-validatie + delta-detection** bovenop bestaande detector, NIET full schema-induction.

```
Document
   │
   ▼
Document detector (bestaand, regel-based)
   → applicable_schemas: [(schema_id, confidence)]
   → top-3 with confidence ≥ 0.3
   │
   ▼
For each applicable schema (sequential bij top-3):
   Pass 1 (LLM, lightweight)
      Input:
        - Compact schema-representation (~500 tokens)
        - Sample chunks van doc (~1500 tokens)
      Output (structured JSON):
        {
          "detected_schema": "fiscal_policy_doc",
          "confidence_in_choice": 0.92,
          "alternative_schemas": [...],
          "coverage_pct": 87,
          "uncovered_concepts": [
            {"surface_form": "...", "suggested_type": "..."},
            ...
          ],
          "proposed_extensions": [...]
        }
   │
   ▼
Soft-nudge decision (cumulatief over schemas):
   coverage > 95%  → silent, proceed
   coverage 80-95% → notification "extend schema?"
   coverage < 80%  → prompt "switch schema?"
   │
   ▼
For each schema (sequential):
   Pass 2 (LLM, focused)
      Input: schema + accepted extensions + chunk
      Output: typed entities + relations per chunk
   │
   ▼
Merge step (in-process, no LLM):
   - Match entities cross-passes (via Q9 vocab-stack + name-normalizer)
   - Multi-type assignment: primary van hoogste-confidence pass
   - Relations dedup op (source, target, rel_type)
   - Cross-schema relations toegestaan
```

**Per-pass token-budget**: 2000-3000 tokens — past in llama3.1:8b context.

**Storage**:
- `notebook_schema` table — per-notebook ontology evolution
- `pass1_results` table — per-source coverage + extension-history
- `entity` table extended met `type_tags: list[str]`, `primary_type: str`

### 3.2 Vocabulary stack — gelaagde entity-resolution

**Probleem**: entities uit verschillende sources hebben verschillende surface-forms. Need authoritative canonicalization.

**Oplossing**: gelaagde stack van vocabularies, geprobeerd in priority-volgorde.

```
For each extracted entity (surface_form, entity_type):
   │
   ▼
Layer 1: TOOI (NL government authoritative)
   - SKOS lookup via SurrealDB cached vocabulary
   - Match: prefLabel + altLabel (case-insensitive + light fuzzy)
   - Hit → use TOOI canonical + URI; STOP
   │
   ▼ (no hit)
Layer 2: User overlay
   - Per-notebook overlay first, dan globaal
   - User-defined canonicals (SKOS-format)
   - Hit → use user-defined canonical; STOP
   │
   ▼ (no hit)
Layer 3: Crossref (papers + authors + DOIs) [Q9 finalized: now]
   - REST API call met cache (avoid repeat-calls)
   - Hit → use Crossref-record (DOI, author-IDs); STOP
   │
   ▼ (no hit)
Layer 4-6: ORCID, arXiv, Wikidata (architecturally prepared)
   - Same pattern; activable via vocabularies-tab
   │
   ▼ (no hit)
Layer 7: name_normalizer (myKG port + NL extensions)
   - Strip honorifics, suffixes, domain tokens
   - Casing-normalize
   - Cluster surface variants
   │
   ▼
Layer 8: Fuzzy dedup (jellyfish)
Layer 9: Embedding dedup (FAISS)
   │
   ▼
Existing notebook entity match?
   yes → link
   no → create new
```

**Module layout**:
```
packages/ontology-manager/src/ontology_manager/vocabularies/
├── __init__.py
├── base.py                  # Abstract VocabularyResolver protocol
├── stack.py                 # Stack orchestrator
├── tooi/                    # Layer 1 — NL gov (always-on)
├── user_overlay/            # Layer 2 — both global + per-notebook
├── crossref/                # Layer 3 — papers + DOIs
├── orcid/                   # Layer 4 — placeholder, architecturally prepared
├── arxiv/                   # Layer 5 — placeholder
└── wikidata/                # Layer 6 — placeholder
```

**Interface** (Protocol):
```python
class VocabularyResolver(Protocol):
    name: str
    priority: int                # lower = higher authority
    domain_types: set[str]       # which entity-types this resolver handles
    
    async def resolve(
        self, surface_form: str, entity_type: str
    ) -> Optional[VocabularyMatch]: ...
```

**Storage**: SurrealDB tables `vocabulary_tooi`, `vocabulary_user_overlay`, `vocabulary_crossref`, etc. Refresh via cron-job (TOOI monthly, Crossref on-demand cache).

**UI**: "Vocabularies" tab — browse + manage user-overlays + enable/disable vocabularies + view refresh status.

### 3.3 Agent platform & bidirectionele sync

```
                ┌───────────────────────────────────┐
                │     SurrealDB + workspace pkgs    │
                │     (single source of truth)      │
                └─────────────────┬─────────────────┘
                                  │
       ┌──────────────────────────┼──────────────────────────┐
       │                          │                          │
       ▼                          ▼                          ▼
┌─────────────┐         ┌──────────────────┐       ┌────────────────┐
│   Web UI    │         │ Agent API (G1)   │       │ Obsidian Sync  │
│  (humans)   │         │ + Webhooks (G5)  │       │ (G3 + G4)      │
│             │         │                  │       │                │
│ Next.js     │         │ hermes /         │       │ Vault watcher  │
│ frontend    │         │ claude-code /    │       │ + writer       │
│             │         │ custom agents    │       │                │
└─────────────┘         └──────────────────┘       └────────────────┘
       │                          │                          │
       └──────────┬───────────────┴──────────┬───────────────┘
                  │                          │
                  ▼                          ▼
            ┌──────────────┐         ┌────────────────────┐
            │ DI container │         │ Job queue + audit  │
            │ (services)   │         │ + webhook dispatch │
            └──────────────┘         └────────────────────┘
```

**Drie fronts, één business-logic**:
- Web UI = humans
- Agent API = machines (hermes, etc.)
- Obsidian Sync = filesystem-driven (human edits via Obsidian, agents drop files)

**Conflict-detectie state machine** voor sync:

```
Source state per (source_id, last_synced_hash, last_origin):
  - origin: "ui" | "vault" | "agent"
  - hash: sha256 of body content
  - timestamp

On vault file change:
  new_hash = sha256(parsed_body)
  if new_hash == last_synced_hash:
      no-op (we wrote this ourselves)
  elif last_origin == "vault":
      vault is canonical, simple update
  elif last_origin == "ui" or "agent":
      both sides changed → mark "needs_merge"
      surface in conflict-resolution panel
      keep work going (non-blocking per G-Q3)
```

---

## 4. Execution roadmap (9-12 maanden)

### 4.1 Maandschema

| Maand | Track | Focus | Voorwaarde |
|---|---|---|---|
| **M1** | A — MinerU | A1 service + A2 confidence-fallback | direct startbaar |
| **M2** | B — KG quality | B1 two-pass + B2 TTL + B4 confidence | direct startbaar |
| **M3** | B + C | B5 orphan-connector + B3 schema-tab + C1 Writer-Eval-Editor | na B1 |
| **M4** | G — Agent platform | G1 API + G2 file-watcher + G3 Obsidian write + Q9 vocabulary stack | na B1, parallel met B3 |
| **M5** | E — Research | E1+E3 incl. review-step + binary routing (Q7) | na G1+G2 |
| **M5** | G + D | G6 templates (literature_note + meeting_notes) + D1 Obsidian export | na G3, na C1 |
| **M6** | G — Sync | G4 bidirectional sync + G5 webhooks | na G3 |
| **M7** | F — Operations | F1 audit (8 checks) + F3 resumable pipeline | na B4 + B5 |
| **M8** | C + D | C2 markdown-hero + C3 docx + D2/D3 JSONL/NetworkX | parallel |
| **M9** | B + F | B6 cross-notebook merge + F2 librarian | parallel |
| **M10-12** | E + polish | E2 multi-agent + tuning + remaining templates | optional / on-demand |

### 4.2 Track-onafhankelijke quick wins (tussendoor)

- A3 markitdown (½ dag)
- C4 deterministic UUID (1-2 dagen)
- Vocabulary architecture prep (ORCID/arXiv/Wikidata stubs): geen werk, alleen interface-skeletons

### 4.3 Effort vs Impact matrix

```
                        IMPACT
                          ▲
                          │
              🟥 hoog     │   🟦 hoog
              • E2 multi  │   • B1 two-pass + multi-schema
              • G4 sync   │   • A2 mineru-fallback
                          │   • C1 Writer-Eval-Editor
                          │   • G1 Agent API
                          │   • Q9 vocab stack (TOOI+Crossref)
                          │
              🟨 medium   │   🟩 medium
              • F2 libra  │   • A1 mineru-alt
              • E4 colle. │   • B2 TTL export
              • B6 merge  │   • B4 confidence
                          │   • D1 obsidian-export
                          │   • G3 Obsidian write
                          │   • G6 templates
                          │
   ◄──────────────────────┼──────────────────────►
   hoog effort            │             laag effort
                          │              
```

**Sweet spot** (rechtsboven):
1. A2 MinerU confidence-fallback
2. B1 two-pass multi-schema
3. Q9 vocabulary stack (TOOI + Crossref)
4. C1 Writer-Evaluator-Editor
5. G1 Agent API

Begin daarmee — hoogste user-facing-impact per ingezet uur.

---

## 5. Appendix

### 5.1 Feature → source mapping (uitgebreid)

| Feature | myKG | wiki_llm | llm-wiki | Most kansrijk |
|---|---|---|---|---|
| Two-pass extraction | ✅ first-class | – | – | **myKG** (adapted naar multi-schema) |
| TTL/RDFS export | ✅ + validator | – | – | **myKG** |
| Schema editing UX | CLI + Protégé | – | – | **myKG-pattern, eigen UI** |
| Multi-schema sequentieel | – | – | – | **eigen design** (open-notebook specific) |
| Confidence scoring | ✅ everywhere | – | – | **myKG** |
| Orphan-connector + prune | ✅ 1060 LOC | LangGraph alt | `/lint --fix` | **myKG** + eigen prune-lifecycle |
| Name normalization | ✅ 202 LOC | markdown-hero dedup | – | **myKG** + NL-extensies |
| Vocabulary stack (TOOI/Crossref/User) | – (SKOS basis) | – | – | **eigen design**, geïnspireerd door myKG SKOS |
| Cross-notebook merge | ✅ 852 LOC | – | hub-merge | **myKG** |
| Writer-Evaluator-Editor | – | ✅ first-class | – | **wiki_llm** |
| markdown-hero | – | ✅ library | – | **wiki_llm** |
| .docx export | – | ✅ via mh | – | **wiki_llm** |
| Deterministic content-hash IDs | – | ✅ | – | **wiki_llm** |
| LangGraph repair | – | ✅ | – | **wiki_llm** (future, na D1) |
| markitdown | – | ✅ | – | **wiki_llm** |
| Multi-agent research | – | – | ✅ first-class | **llm-wiki** (E2 future) |
| Thesis-driven mode | – | – | ✅ | **llm-wiki** (E3) |
| Single-agent research | – | – | ✅ (foundation) | **llm-wiki-pattern, eigen impl** |
| Audit command | – | – | ✅ | **llm-wiki-pattern, 8 concrete checks** |
| Librarian background | – | – | ✅ | **llm-wiki** |
| Obsidian export (atlas) | ✅ entity-centric | – | ✅ topic-centric | **myKG-stijl finalized** |
| Agent API / headless mode | – | – | – | **eigen design**, geen ref-repo |
| Bidirectional vault sync | – | – | – | **eigen design** |

### 5.2 File-path catalog (waar elke feature komt te wonen)

```
apps/app-main/src/app_main/
├── api/
│   ├── routers/
│   │   ├── agents.py                  # G1 Agent API endpoints
│   │   ├── notebooks.py               # extended: schema, export, merge
│   │   └── vocabularies.py            # NEW: user-overlay management
│   └── auth/
│       └── api_keys.py                # G1 API-key auth
├── services/
│   ├── source_extractor.py            # Phase 3, parser-engine routing toegevoegd
│   ├── source_processor.py            # Phase 3, geen wijziging
│   ├── source_summarization_orchestrator.py  # Phase 3 + templates
│   ├── source_embedding_orchestrator.py
│   ├── parsing/
│   │   ├── confidence.py              # A2 docling-confidence-metric
│   │   └── mineru_http_client.py      # A1
│   ├── web_search/
│   │   ├── tavily_client.py           # E1
│   │   ├── duckduckgo_client.py       # E1 fallback
│   │   └── base.py                    # interface
│   ├── obsidian_sync/
│   │   ├── writer.py                  # G3 write-side
│   │   ├── watcher.py                 # G4 read-side
│   │   ├── differ.py                  # G4 diff + conflict detection
│   │   └── conflict_resolver.py       # G4 conflict UI service
│   ├── audit_service.py               # F1 audit checks
│   ├── librarian_service.py           # F2
│   ├── file_watcher_service.py        # G2 always-on watcher
│   └── webhook_dispatcher.py          # G5 outbound webhooks

services/
├── docling/                           # bestaand
├── mineru/                            # NEW (A1)
├── summarization/                     # bestaand
├── whisperx/                          # bestaand
└── extraction/                        # bestaand

packages/
├── shared/
│   └── src/shared/utils/
│       └── markdown.py                # NEW Q4 markdown-hero wrapper
├── ontology-manager/
│   └── src/ontology_manager/
│       ├── rdf_owl_shacl.py           # bestaand, B2 fix + uitbreiding
│       └── vocabularies/              # NEW Q9
│           ├── base.py
│           ├── stack.py
│           ├── tooi/                  # active
│           ├── user_overlay/          # active
│           ├── crossref/              # active
│           ├── orcid/                 # prepared (interface only)
│           ├── arxiv/                 # prepared
│           └── wikidata/              # prepared
└── ... (rest unchanged from Phase 7)

pipelines/
├── entity-filtering/
│   └── src/entity_filtering/
│       ├── deduplication/
│       │   ├── canonical_entities.py  # bestaand Phase 4
│       │   └── name_normalizer.py     # NEW Q9 myKG-port + NL
│       └── resolution/
│           └── orphan_connector.py    # NEW B5 myKG-port + prune
├── ontology-extraction/
│   └── src/ontology_extraction/
│       ├── pass1_schema_validation.py    # NEW B1
│       ├── pass2_typed_extraction.py     # NEW B1
│       └── multi_schema_orchestrator.py  # NEW B1 multi-schema
├── summarization/
│   └── src/summarization/
│       ├── enhancers/
│       │   └── writer_evaluator_editor.py  # NEW C1
│       └── templates/                       # NEW G6
│           ├── literature_note.py
│           ├── meeting_notes.py
│           └── ...                          # future templates
└── retrieval/                          # bestaand
```

### 5.3 Storage additions (SurrealDB tables)

| Table | Purpose | Track |
|---|---|---|
| `notebook_schema` | Per-notebook ontology evolution | B1 |
| `pass1_results` | Coverage + extensions per source | B1 |
| `agent_keys` | API key registry per agent | G1 |
| `agent_audit_log` | All agent actions | G5 |
| `webhook_subscriptions` | Per-agent webhook config | G5 |
| `sync_state` | Last-synced state per Source/file | G4 |
| `vocabulary_tooi` | TOOI cached entries | Q9 |
| `vocabulary_user_overlay` | User-defined canonicals (global+notebook) | Q9 |
| `vocabulary_crossref` | Crossref cache | Q9 |
| `pending_extensions` | Schema extensions awaiting user review | B3 |
| `pending_reconnects` | Orphans awaiting reconnection retry | B5 |
| `archived_entities` | Pruned orphans (recoverable) | B5 |
| `audit_findings` | Last audit-run per notebook | F1 |
| `conflict_pending` | Sync conflicts awaiting resolution | G4 |

### 5.4 Nieuwe externe dependencies

| Dep | Purpose | Track |
|---|---|---|
| `markdown-hero` | Markdown structural ops | Q4 (C2) |
| `tavily-python` (of REST direct) | Web search | E1 |
| `duckduckgo-search` | Web search fallback | E1 |
| `crossref-commons` (of REST direct) | Paper metadata | Q9 |
| `watchdog` | Cross-platform file events | G2 + G4 |
| `mineru[all]` | PDF parsing (alt) | A1 |
| `python-docx` (al via markdown-hero) | Word export | C3 |
| `rank-bm25` | ❌ NIET nodig (Q5 finalized: skip) | - |
| `pyshacl` | SHACL validation (optional) | B2 (already in rdf_owl_shacl) |

---

## 6. Track G beslispunten recap

| # | Decision |
|---|---|
| G-Q1 | API-key authentication (header-based, per-agent) |
| G-Q2 | File-watcher always-on op conventional path (`~/open-notebook/inbox/` + per-notebook) |
| G-Q3 | Conflict-resolution non-blocking (warning-badge, werk gaat door) |
| G-Q4 | Summary templates V1: `literature_note` + `meeting_notes` |

---

## 7. Volgende stap

Met alle Q1-Q9 + G-Q1 t/m G-Q4 finaal, is de roadmap nu actionable. Aanbevolen volgende stap:

**Sprint-plan voor Track A** (eerste track, laagste-risico, hoogste-impact):
- Concrete file-paden + Dockerfile-sketch voor MinerU service
- Per-file effort + PR-grenzen
- Test strategy
- Zelfde detail-niveau als `REFACTOR_PLAN.md`

Of: **alternatief beginnen met Q9 vocabulary-stack** als foundation voor B + G (vocabulary is dependency van entity-resolution overal).

---

## Track J — Cloud/local model routing (privacy + failover) (NEW)

> **Status**: ✅ COMPLETE (2026-06-22). All six phases (J.1-J.6) implemented +
> reviewed + merged. Cloud routing is live-capable: the three LLM stages (entity
> extraction, summarization, chat) route through NVIDIA NIM by default with
> per-document failover to local; `private` documents/notebooks never reach
> cloud; keyless/NIM-down → automatic local fallback. Embeddings + parsing stay
> local-and-fixed (768-dim invariant). Per-task routing telemetry
> (`routing.served`) + provider health + per-notebook routing summary surfaced
> in the UI. Operator runbook at `docs/tracks/J-model-routing/OPERATOR_GUIDE.md`.
> Built on the per-function model-resolution layer from Track B.8
> (`resolve_default_model_id` / `make_default_llm_caller`, `DefaultModels`).
>
> **Phase ledger**: J.1 route resolver · J.2 failover executor + circuit-breaker
> + rate limiter · J.3 layered privacy plumbing + local-embeddings guardrail ·
> J.4 NVIDIA NIM cloud wiring + summarization unification + provenance · J.5
> routing config + privacy toggles + health UI · J.6 chat telemetry (FU-J4-1) +
> per-notebook routing summary + E2E failover/privacy/no-cloud scenarios +
> operator docs. See `docs/tracks/J-model-routing/status.md`.

**Vision**: public documents process via cloud APIs by default (fast, strong
models) with automatic failover across providers and down to local; documents
tagged `private` run entirely on local models; if no cloud is online, local is
the fallback. One privacy switch — layered global → notebook → document — drives
the whole LLM pipeline.

**Scope (resolved decisions)**:
- **J-Q1 Privacy granularity**: layered — global default + per-notebook setting
  + per-document `private` override. Most specific wins; `private` is sticky and
  never escalates to cloud.
- **J-Q2 Pipeline scope**: **LLM stages only** — entity extraction,
  summarization, chat. Parsing (docling/MinerU) and **embeddings stay local and
  fixed**: cloud embeddings change the vector space/dimensions and break HNSW
  search (Track I.G pins 768-dim). Embeddings + parsing are explicitly OUT of
  the cloud-routing path.
- **J-Q3 Provider chain**: proposed default `[Anthropic Claude → OpenAI →
  local]` per LLM function, fully configurable (order + enable/disable per
  provider). Track-planner proposes the concrete default; user tunes.
- **J-Q4 Failover granularity**: **per-document** — a document's LLM stage runs
  on the first available provider; on a provider failure the whole document
  fails over to the next provider; local is the last resort (cloud mode only).
- **J-Q5 Availability**: per-provider health + circuit-breaker; "no cloud online
  / all cloud failing" → local fallback (cloud mode). Private mode never touches
  cloud.
- **J-Q6 Secrets**: cloud API keys via env/secrets per provider; a missing key
  disables that provider (it drops out of the chain).
- **J-Q7 Observability**: telemetry per task — provider served, fallback events,
  latency/cost — surfaced in UI + metrics.

**Builds on / touches**: Track B.8 model-resolution layer
(`apps/app-main/.../entity_extraction_service.py` `make_default_llm_caller` /
`resolve_default_model_id`, `DefaultModels`), the esperanto `LanguageModel`
provider abstraction, summarization + chat services, and the ingestion API.
Does NOT touch embeddings (I.G) or parsing (A) — by design.

**Phase overview** (Backend → UI → Integration):
- J.1 Provider-chain config + privacy-aware route resolver (generalize
  `resolve_default_model_id` to an ordered route per task × privacy mode).
- J.2 Availability/circuit-breaker + per-document failover executor.
- J.3 Privacy-flag plumbing (ingestion `private` flag → source/notebook → LLM
  stages); embeddings/parsing guarded local.
- J.4 Cloud provider integrations (Anthropic + OpenAI via esperanto; latest
  Claude models per the claude-api reference; provider-error → failover mapping).
- J.5 UI — layered privacy toggle (global/notebook/document) + provider-chain
  config + health/fallback status.
- J.6 Integration, telemetry, E2E (forced-outage failover; private-never-cloud;
  no-cloud→local) + operator docs (keys, public-vs-private data-residency note).

See [`docs/tracks/J-model-routing/plan.md`](./tracks/J-model-routing/plan.md).

---

## Track K — Entity resolution & deduplication (NEW)

> **Status**: 📋 PLANNED (2026-06-22, user-request). Implements the resolution
> layer the B.8c assessment proved missing: the V1 `name_normalizer` resolves
> identical surface forms across documents (107 cross-doc entities over the 4
> Regio Deal docs) but **fragments variants** of the same real entity —
> `BZK` / `ministerie van BZK` / `(Ministerie van) Binnenlandse Zaken en
> Koninkrij(k|ks)relaties` split ~8 ways; `"minister"` 23 ways. See
> [`docs/tracks/B-kg-quality/reviews/phase-B.8c-resolution-assessment.md`](./tracks/B-kg-quality/reviews/phase-B.8c-resolution-assessment.md).
> This is the Q9 vocabulary stack + the M4 work, now a concrete track.

**Vision**: the same real-world entity collapses to ONE canonical node across
documents regardless of surface form — abbreviation, role-prefix, spelling
variant, or vocabulary alias. Two layers: cheap NL-aware normalization first
(immediate fragmentation drop), then full vocabulary-backed resolution.

**Scope (two layers)**:
- **K.1–K.2 Quick wins (F1, cheap, high-impact)** — extend the V1
  `name_normalizer` (`packages/shared/src/shared/utils/name_normalizer.py`, the
  swap-point everything already calls): strip leading articles + role-prefixes
  (`De `, `Het `, `Minister(ie) van `, `Gemeente `), a curated NL
  government-org **abbreviation alias table** (BZK ↔ Binnenlandse Zaken en
  Koninkrijksrelaties, VRO ↔ Volkshuisvesting en Ruimtelijke Ordening, …), and
  spelling-variant tolerance (the Koninkrij(k|ks) class). Tighten/measure the
  entity-filtering similarity-dedup (currently threshold-configurable ~0.85).
  Must not over-merge distinct entities (precision guard + a measured
  fragmentation-vs-false-merge metric on the live Convenant set).
- **K.3+ Full resolution (Q9/M4)** — TOOI (Dutch government vocabulary) +
  Crossref (research papers) lookup → canonical IDs + `external_ids`; fuzzy /
  embedding-based candidate dedup with a human-reviewable merge step; per-entity
  `aliases`; global + per-notebook user-overlay; architecturally prepared for
  ORCID/arXiv/Wikidata. A canonicalization/merge operation over already-persisted
  entities (retroactive dedup), not just at-ingest.

**Builds on / touches**: `name_normalizer` (swap-point — entity persistence dedup
key + filtering both call it), `pipelines/entity-filtering` (similarity dedup),
the `entity` table (`canonical_name`/`hash_id`/`aliases`/`external_ids`), and the
Q9 vocabulary decision. Coordinates with B.8's persistence/upsert merge
semantics — must not regress the hash_id/dedup-key contract (migration 50).

**Phase overview** (Backend → measure → UI):
- K.1 NL-aware normalizer (article/role-prefix strip + spelling tolerance) +
  precision guard + measurement harness on the live Convenant set.
- K.2 Government-org abbreviation alias table (+ extensible alias config).
- K.3 Retroactive canonicalization/merge over existing entities (dedup the KG
  already built) with a dry-run + reviewable merges.
- K.4 TOOI + Crossref vocabulary lookup → external_ids + aliases (Q9 core).
- K.5 Fuzzy/embedding candidate dedup + merge step; user-overlay (global +
  per-notebook).
- K.7 Type-safe relation endpoints (Option B) — carry entity type/ID through relations.
- K.8 Office/role & temporal entity resolution (user-requested 2026-06-22) — model roles as temporal entities + time-bounded held_office relations (the "minister van BZK across different holders/time" gap that normalization cannot solve). Track-sized; depends on K.7 + K.4 TOOI functie vocab. — carry entity type/ID through relations so cross-type homographs never mis-attach; unlocks re-enabling the aggressive K.1 prefixes. Sequenced after K.3.
- K.6 UI: review/merge duplicates + alias management + per-entity external IDs.

See [`docs/tracks/K-entity-resolution/plan.md`](./tracks/K-entity-resolution/plan.md).

---

## Track L — Entity typing fidelity (NEW)

> **Status**: 📋 PLANNED (2026-06-22). The KG holds 27% `other` + 44% generic
> concept/topic because persistence FLATTENS the rich Dutch ontology types
> (`Gemeente`/`Ministerie`/`RegioDeal`…) the LLM correctly extracts onto a 20-type
> generic enum with no bridge + an English-only alias map, leaving
> `type_tags`/`primary_type` empty. See `claudedocs/entity-typing-analysis.md`.

**Vision**: the rich ontology type the LLM extracts survives to the KG. Bridge
ontology types → the canonical enum via the EXISTING `parent_type` schema.org
hierarchy (a fixed ~20-entry schema.org→canonical map — language-agnostic),
preserving the original type in `primary_type`/`type_tags`. Apply `policy_themes`
so themes stop degrading to generic concept/topic.

**Language**: the core bridge is **language-agnostic by construction** (typing
flows through ontology-declared types; add a language = add an ontology). The only
language-specific surface is a curated **EN+NL** residual alias map (user: EN+NL =
98% of needs; no LLM/embedding fallback for the 2%).

**Phases**: L.1 ontology→canonical bridge + preserve rich type · L.2 EN+NL residual
aliases + non-silent fallback · L.3 add `programme`/`technology` + enum-relax
migration · L.4 make `policy_themes` fire (notebook-schema default) · L.5
retroactive re-typing (dry-run+reversible, the destructive one — changes
`entity_type` → `hash_id`, routed through the K.3 merge) · L.6 typing-fidelity
metric + per-schema orphan audit.

**Risk**: L.5 re-typing changes `entity_type` → the B.8 `hash_id` — collision-
detection + K.3 merge. **Expected**: `other` 27%→<5%, `primary_type` 0%→>90%.
**Depends on**: B.8 (persistence) + K (better types make the K.7 (name,type) guard
+ K.8 role modeling work). See [`docs/tracks/L-entity-typing/plan.md`](./tracks/L-entity-typing/plan.md).

---

## Track U — Document graph (documents as first-class nodes) (NEW)

> **Status**: ✅ CLOSED (2026-06-28). The KG now contains DOCUMENTS as nodes,
> connected (a) via the entities they share (`mentions`) and (b) directly
> (`cites`). Key finding from U.1: the schema ALREADY existed — the ontology
> defined these edges but they were empty. Track U filled the document-centric
> layer the design anticipated. See [`docs/tracks/U-document-graph/status.md`](./tracks/U-document-graph/status.md).

**Vision**: documents are first-class graph nodes, not just an array on each
entity. Two materialized edge layers make the document graph real (traversable /
drawable / exportable), complementing — not replacing — the Track R computed
search signal (see `ARCHITECTURE.md` §9, computed-vs-materialized).

**Phases**: U.1 design + measurement (decision gate; confirmed 0 intra-corpus
citations on this corpus → `cites` built as mechanism, deferred as data) · U.2
`mentions` (source→entity) regenerated projection of `entity.source_documents`,
R.2-weighted, R.6-filtered (67 edges on staging) · U.3 `cites` (source→source)
citation materialization — precision-first matcher + Track V input contract
(`ParsedReference`), 0 live edges by design until Track V feeds references · U.4
document graph in the KG visualization (bipartite view, entity-layer toggle,
a11y fallback) · U.5 richer NetworkX/JSONL exports (gated document layer) +
computed-vs-materialized architecture note + RETRO.

**Value**: navigation / visualization / export / graph-algorithms + the
genuinely-new citation facts. It does NOT add search value (R.1–R.3 already
compute shared-entity + similarity relatedness on the fly). Scales with corpus
size — the more documents share entities / cite each other, the richer the graph.
**Depends on**: R (computed layer it complements), B.8 (entity persistence).
**Feeds / waits on**: Track V (reference extraction) for live `cites` edges.

---

## Track W — Shared graph memory (MCP graph-tools + hybrid search + reranker) (NEW)

> **Status**: ✅ CLOSED (2026-06-29). The SurrealDB knowledge graph is now a
> shared memory substrate every Claude Code session reads/writes through MCP
> graph tools, hybrid (BM25⊕vector) search is reachable via `/search`, and an
> optional local multilingual reranker can sharpen the top-N. Key finding (as in
> Track U): most of this ALREADY existed — the hybrid fusion chain, the MCP
> server, and the repo backings were present but unwired. Track W exposed and
> tied them together. See [`docs/tracks/W-mcp-graph-memory/status.md`](./tracks/W-mcp-graph-memory/status.md)
> and `ARCHITECTURE.md` §10.

**Vision**: the graph is a shared substrate, not per-session state — two sessions
pointed at the same DB see the same nodes/edges, and notes/citations one writes
are visible to the other. Built the *foundation* (Constella Features 1 + 5);
citations / auto-link / contradiction are separate later tracks.

**Phases**: W.1 expose hybrid (BM25+vector, RRF) search via `/search`
`type=hybrid` — the fusion chain already existed at the repo layer; W.1 added the
router seam + caught a latent fusion bug (scores collapsed to 0 → fixed with
Reciprocal Rank Fusion) · W.2 local cross-encoder reranker as its OWN container
(`services/reranker/`, `bge-reranker-v2-m3`), wired into `/search` behind a
default-off `rerank` flag, with a zero-dep heuristic fallback when the service is
down — keeps `torch`/`sentence-transformers` out of `app-main`; the ~2 GB model
run is operator-gated · W.3 five MCP graph tools (`search`/`get_node`/`related`/
`cite`/`add_note`) on the existing `surrealdb-mcp` server + a unified
`load_all_edges` (relation+mentions+cites) + the `relates_to`→`relation` fix ·
W.4 integration check (model-free `search`→`related`→`get_node` compose) +
`ARCHITECTURE.md` §10 + operator runbook + RETRO.

**Value**: a queryable, writable shared graph memory for agent sessions +
better retrieval quality (hybrid + optional rerank), built lean by reusing the
existing surface. **Depends on**: R (the validated hybrid fusion it mirrors), the
existing `surrealdb-mcp` server + repos. **Feeds**: later Constella features
(citations / auto-link / contradiction) build on this substrate.

---

## Track X — Citations to exact source (Docling provenance → answers) (NEW)

> **Status**: ✅ CLOSED (2026-06-29). Generated answers now cite the **exact
> source / page / chunk** a claim came from. The provenance (file/page/section)
> was already STORED on the `chunk` table by the Docling ingest but never
> threaded into answers; Track X passes it retrieval → answer graphs → a
> structured `citations` array, and guards the LLM's inline prose markers
> against the retrieval set. No re-extraction — pure reuse of stored provenance.
> See [`docs/tracks/X-citations-to-source/status.md`](./tracks/X-citations-to-source/status.md)
> and `ARCHITECTURE.md` §11.

**Vision**: every answer is traceable to the page it stands on — Feature 4 of
the Constella adoption plan. Precision over coverage: a citation is only emitted
for a source/page/chunk that was actually retrieved; hallucinated attributions
are flagged/dropped (membership check, not semantic support — that limitation is
explicit).

**Phases**: X.1 thread chunk provenance through retrieval — `fn::` collapses a
source's matching embeddings to one source-level hit, so the originating chunk's
page is re-attached by a batched `source_embedding ⋈ chunk` cosine-argmax SELECT
(the exact row behind the collapsed `math::max`, verified to 1e-9 on staging);
hydration is opt-in so the `/search` hot path is untouched · X.2 cited answers
in the `ask` + `source_chat` graphs — provenance tags in the prompt + a
structured `citations: [{source, page, chunk_id, section}]` array derived
**deterministically from the context hits** (not parsed from prose) · X.3
faithfulness guard — the real risk is the LLM's inline `[doc, p.X]` prose
markers (the array is context-derived, so already `⊆` the retrieval set);
`guard_answer_citations` membership-checks those markers and flags hallucinated
ones (non-destructive to the answer text), with a defensive array safety-net for
regression insurance + ARCHITECTURE note + RETRO.

**Value**: trustworthy, page-level attribution on answers, reusing provenance
that already existed end-to-end. **Depends on**: the Docling ingest provenance on
`chunk` (X.1 hydration), the `ask` / `source_chat` answer graphs. **Limitation**:
membership-not-semantic — a semantic-support check (does the cited passage
actually back the claim) would need a second LLM pass and is deferred.

---

## Track Y — Auto-link (new note → related notes → RELATE) (NEW)

> **Status**: ✅ CLOSED (2026-06-29). When a note is created and embedded, its
> most-related notes are found by embedding similarity and the links are
> persisted as `related_note` graph edges — Feature 2 of the Constella adoption
> plan. The substrate existed (note embeddings, source-level
> `find_related_by_embedding`, RELATE patterns); Track Y added note-level
> similarity, a note↔note edge table, the orchestrator + precision gate, and
> **both** halves of the phased trigger: on-demand (endpoint + MCP tool) and a
> background job chained off the embed. See
> [`docs/tracks/Y-auto-link/status.md`](./tracks/Y-auto-link/status.md) and
> `ARCHITECTURE.md` §12.

**Vision**: notes self-organise into a related-notes graph by meaning, the same
embedding-similarity mechanism the source layer already used, lifted to the note
level. Precision over coverage: a conservative `min_similarity` + top-`k` keep
the graph sparse and meaningful (no graph explosion), reusing the R.6 discipline.

**Phases**: Y.1 note-level similarity + the `related_note` edge table —
`NoteRepository.find_related_by_embedding` (cosine, self-excluded, no-embedding
graceful) + migration 68 (`TYPE RELATION FROM note TO note`, fresh-container-safe)
+ an idempotent clear-before-relate helper (RELATE is not idempotent; both
endpoints strict-validated before interpolation — RELATE can't bind a `$param` in
the in/out position, a data-destroying-injection lesson) · Y.2 orchestrator +
on-demand trigger — `NoteAutoLinkService.auto_link` (ensure-embedding → rank →
threshold/top-k → idempotent edges; conservative `min_similarity=0.75`, `k=5`),
`POST /notes/{id}/auto-link`, and the embedding-free MCP `auto_link_note` tool ·
Y.3 background job + integration + docs/RETRO — a `NOTE_AUTO_LINK` job chained
off a successful note embed (`_handle_embed_single_item`), best-effort enqueue +
isolated failure (a linking failure never corrupts the already-persisted note),
idempotent.

**Value**: a meaning-organised note graph that grows automatically as notes
accumulate, reusing the existing embedding substrate. **Depends on**: note
embeddings + the embed job (the trigger), the source-level similarity pattern.
**Sync model**: handles NEW notes; re-linking on note EDIT is a noted follow-up
(idempotent `auto_link` makes it a drop-in), not Y core. **Data reality**:
source-heavy corpus / few notes today — a built-and-tested mechanism that lights
up as notes accumulate (like the Track U `cites` edges). **Extension point**:
**note↔source** — same embedding-similarity mechanism, a second edge type; a
documented follow-up in Y, **now built in Track NS** (below).

---

## Track NS — Note→source auto-link (extend Y: notes link to the sources they're about) (NEW)

> **Status**: ✅ CLOSED (2026-06-29). Y's promoted note↔source extension. A note
> auto-links to the **sources it is about** (by embedding similarity), persisted
> as `note_about` graph edges, alongside the existing note→note `related_note`
> links. The orchestrator embeds the note **once** and runs **two passes** over
> that one vector (notes, then sources); every trigger — the
> `POST /notes/{id}/auto-link` endpoint, the `NOTE_AUTO_LINK` job, and the MCP
> `auto_link_note` tool — carries both link types through for free. Chosen as Y's
> first extension because, unlike note↔note in a note-sparse corpus, note→source
> **lights up immediately** in the current source-heavy corpus. See
> [`docs/tracks/NS-note-source-autolink/status.md`](./tracks/NS-note-source-autolink/status.md)
> and `ARCHITECTURE.md` §12a.

**Vision**: a note self-locates against the source corpus it lives in — link a
new note to the convenanten / papers it most resembles — reusing Y's
orchestrator, threshold/top-k discipline, idempotent + injection-safe RELATE, and
the after-embed trigger, with a second edge type and a cross-type ranking.

**Phases**: NS.1 note→source similarity + the `note_about` edge —
`NoteRepository.find_related_sources_by_embedding` (cosine of `note.embedding` vs
every `source.embedding`, no-embedding graceful) + migration 70
(`TYPE RELATION FROM note TO source`, fresh-container-safe) + an idempotent,
injection-safe `relate_note_source` (strict id validation on both endpoints,
clear-before-relate, `note:`→`source:` typing enforced); the cross-type
same-space embedding validity verified as a STOP gate first · NS.2 extend the
orchestrator + triggers — `auto_link` gains the additive second pass
(`find_related_sources_by_embedding` → threshold/top-k → `relate_note_source`)
and a parallel, distinctly-named source counter family; the endpoint, job, and
MCP tool all carry it through · NS.3 integration + docs/RETRO — `ARCHITECTURE.md`
§12a, this entry, RETRO; the two NS.2 minors folded (trigger docstrings refreshed
to describe both link types; the MCP `source_skipped` key aligned to
`source_skipped_existing` to match the HTTP/service layer).

**Value**: in a source-heavy corpus, immediate, visible note→source links from
the moment a note is embedded — the same meaning-organised graph as Y, but
landing on the data that actually exists today. **Depends on**: Track Y
(orchestrator + edge/RELATE pattern + the embed trigger), R.0 source aggregate
embeddings (the same-space other end of the cosine). **Cross-type validity**:
note + source embeddings are the same model/space by construction
(mxbai-embed-large, 1024-dim); the `array::len` guard excludes any mismatch
rather than crashing. **Gotcha**: `array::len(NONE)` raises in this SurrealDB
version, so the `!= NONE` guard before the length check is load-bearing on the
source side. **Extension point**: note→entity (link a note to the entities it
mentions) and edit-relink (re-run on note edit — idempotent `auto_link` makes it
a drop-in), both noted follow-ups, not built in NS.

---

## Track Z — Contradiction detection (LLM judges related pairs) (NEW)

> **Status**: ✅ CLOSED (2026-06-29). An LLM judges whether two *already-related*
> sources reinforce / contradict / are neutral, persisting only confident
> verdicts as `source_verdict` edges — **Feature 3** (the last) of the Constella
> adoption plan. The highest-risk graph-write feature (a false contradiction
> pollutes the graph), so **precision-first** throughout: a fabricated
> contradiction is worse than a missed one. Candidates come from the Track R
> related substrate (top-`k`, NOT O(n²)); the trigger is on-demand
> (`POST /sources/{id}/judge-contradictions`); a background job is a documented
> follow-up. See [`docs/tracks/Z-contradiction/status.md`](./tracks/Z-contradiction/status.md)
> and `ARCHITECTURE.md` §13.
>
> **This CLOSES all five Constella adoption features** — Feature 1 = Track W
> (shared graph memory), Feature 2 = Track Y (auto-link), Feature 3 = Track Z
> (contradiction), Feature 4 = Track X (citations to source), Feature 5 = Track W
> (MCP graph-tools substrate). **5 / 5 done.**

**Vision**: the knowledge graph surfaces *tension*, not just relatedness — which
sources agree and which conflict — without manufacturing conflict from topic
overlap. Precision over recall: a conservative `min_confidence` + a `neutral`-by-
default judge keep the verdict edges trustworthy (the same discipline as the
Track U `cites` membership gate).

**Phases**: Z.1 verdict-edge schema + idempotent helper — migration 69
(`source_verdict` `TYPE RELATION FROM source TO source`, strict fields w/ defaults,
fresh-container-verified; kept distinct from the app-side `claim`/`contradicts`
scaffolding) + `SourceRepository.relate_verdict` (idempotent clear-before-relate;
strict id validation before interpolation — the `relate_cites` injection lesson) ·
Z.2 candidate generation + pairwise judge — `build_candidate_pairs` (related
substrate, self-excluded, deduped, top-`k`) + `ContradictionJudgeService`
(compact context → Track J routed LLM `json_mode` → robust **top-level-object-only**
parse → precision gate; the JSON-array fabrication blocker caught + fixed in
review) · Z.3 on-demand trigger + integration + docs/RETRO —
`POST /sources/{id}/judge-contradictions` (route-layer validation: bad id /
out-of-range bounds → 422, missing source → 404); the MCP `judge_contradiction`
tool **deferred** (the judge needs the app-main LLM layer; the surrealdb-mcp
server is repo-direct — documented in the server docstring + RETRO).

**Value**: a precision-first contradiction/reinforcement layer over the related
graph, lighting up as the corpus grows. **Depends on**: Track R related substrate
(candidates), Track J model routing (the judge LLM), Z.1 verdict schema.
**Data reality**: few sources today — a built-and-tested mechanism that lights up
as the corpus grows (like the Track U `cites` edges). **Extensions** (documented,
not built in Z): a **background job** judging new/edited sources on ingest;
**claim-level** contradiction (source→`claim`, a finer grain) as claims accumulate.

---

## Track PL — Source-ingestion pipeline (auto source→KG with gates) (NEW)

> **Status**: ✅ CLOSED (2026-06-30). Ingest used to auto-chain only
> `parse→chunk→embed(per-chunk)` and **stop** — the entire KG/analytical layer
> (the mean-pool `source.embedding` aggregate, entity/relation extraction, the
> `mentions` document-graph, insights) was manual or orphaned. Track PL makes
> ingest a **fully-automatic, gated, idempotent** pipeline from a raw source to
> its KG end-result, and consolidates the wiring into one declarative
> `SOURCE_PIPELINE` + `advance_source` driver with a per-source
> `processing_stage` status model. See
> [`docs/tracks/PL-source-pipeline/status.md`](./tracks/PL-source-pipeline/status.md),
> [`docs/tracks/PL-source-pipeline/RETRO.md`](./tracks/PL-source-pipeline/RETRO.md),
> and `ARCHITECTURE.md` §14.

**Vision**: a new document becomes a knowledge-graph citizen with **no manual
step** — embedded (+ aggregate), extracted, graphed, complete — while explicit
gates (schema-review, the per-notebook `auto_insights` toggle) keep cost and
correctness in the operator's hands. Idempotent + resumable from
`processing_stage` throughout; contradiction (Z) and `cites` (U.3) stay
on-demand/deferred.

| Phase | Deliverable | Status |
|-------|-------------|--------|
| PL.1 | Fix the orphaned `source.embedding` aggregate (write it in the live embed step; promote the helper out of the backfill script) + migration-safe backfill | ✅ |
| PL.2 | Auto-chain EXTRACT after EMBED + `source.processing_stage` model (migration 71) + schema-review gate | ✅ |
| PL.3 | Auto-chain GRAPH (source-scoped `mentions` refresh) + INSIGHTS parallel branch behind `notebook.auto_insights` toggle (migration 72) | ✅ |
| PL.4 | One declarative `SOURCE_PIPELINE` + `advance_source` driver (thin handlers); `processing_stage` on the source read API | ✅ |
| PL.5 | Bounded configurable worker concurrency (default 1, opt-in) + sync-ingest decoupled from the shared worker + cleanups (dead `USE_MINERU_SERVICE`, one parser-`auto` source of truth) + ARCHITECTURE/RETRO | ✅ |

**The spine**: `DOCUMENT_PARSE → EMBED(+aggregate) → EXTRACT → GRAPH(mentions)
→ complete`, with INSIGHTS parallel off EMBED. **Gates**: schema-review parks
EXTRACT at `awaiting_schema_review`; the per-notebook toggle gates INSIGHTS.
**Invariant preserved**: source-scoped `mentions` is a GLOBAL projection filtered
on write (the R.2×R.6 weights are inherently cross-source). **Worker decision**:
concurrency made bounded + configurable but **default 1 (serial)** — safe because
stage ordering is enforced by the job chain, not by worker serialization; turning
it on for the live deployment is a documented opt-in follow-up. **Deferred**:
contradiction (Z) + `cites` (U.3, blocked on Track V) stay on-demand.

---

## Track UX — Frontend pipeline alignment (surface `processing_stage` as the spine) (NEW)

> **Status**: ✅ CLOSED (2026-07-01). Track PL rebuilt the backend into a full
> auto-chain with a per-source `source.processing_stage`, but the Next.js
> frontend still reflected the pre-auto era: it reconstructed an approximate
> status from output-counts + a coarse job status, in the wrong order, and
> exposed manual "Run X" buttons for steps that now happen automatically. Track
> UX makes `processing_stage` the single source of truth across every source
> surface (list / detail / create), collapses the 9-step creation wizard into a
> lean 4-step flow, relocates parser config to the Reprocess dialog, and
> reframes the leftover manual controls as recovery-only actions. See
> [`docs/tracks/UX-pipeline-alignment/status.md`](./tracks/UX-pipeline-alignment/status.md)
> and `ARCHITECTURE.md` §14.

**Frontend spine**: one canonical `PipelineStatus` component (variants
`live` / `card` / `detail`) renders the runtime-order spine
`Ingest → Embed → Extract → Graph → Complete` (Embed **before** Extract) with
Insights as a parallel branch, driven purely by `processing_stage`. Output
counts are enrichment on `done` nodes, never the status source; the job axis
only drives the current node's spinner. GRAPH / EXTRACT / INSIGHTS run
automatically, so the manual runners are recovery-only.

| Phase | Deliverable | Status |
|-------|-------------|--------|
| UX.1 | `processing_stage` TS union + `useSourcePipeline` stage-polling hook (+ list-schema field + backfill migration) | ✅ |
| UX.2 | Canonical `PipelineStatus` component + pure stage state machine (`derivePipelineNodes`) | ✅ |
| UX.3 | SourceCard 5-segment mini progress bar (bounded per-card polling) | ✅ |
| UX.4 | Source-detail spine + Graph signal + drop stale regenerate guidance | ✅ |
| UX.5 | Lean 4-step creation flow (Input → Organize → Processing → Done) + config relocation | ✅ |
| UX.6 | Recovery-only manual controls + deferred Contradictions stub + orphaned-tab cleanup + docs/e2e | ✅ |

**Deferred**: the Contradictions surface (UX.6) is a data-absent-safe stub —
Track Z ships a `POST /sources/{id}/judge-contradictions` trigger but no
verdicts read API yet, so the panel renders nothing until a GET lands.

---

## Bijlagen

- `docs/REFACTOR_PLAN.md` — voltooide refactor (Phase 0-7)
- `docs/MYKG_COMPARISON.md` — myKG diepte-analyse
- `docs/LLM_WIKI_COMPARISON.md` — llm-wiki + wiki_llm comparison
- `docs/SUMMARIZATION_APPROACHES.md` — bestaande summarization strategies
- `ARCHITECTURE.md` — post-refactor structuur
