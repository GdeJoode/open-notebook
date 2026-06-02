# Refactor Plan — open-notebook

Volgorde, dependencies, validatie, rollback. Geordend van laag-risico/hoge-ROI naar diep-structureel. Elke fase is een aparte PR-grens.

> **Baseline**: gebouwd op de graphify knowledge-graph analyse van commit `a2130b40` (zie `graphify-out/GRAPH_REPORT.md` en `docs/SUMMARIZATION_APPROACHES.md` voor context). Plan opgesteld 2026-06-02.

---

## Uitgangspunten

- **Eén PR per fase** behalve waar expliciet gegroepeerd. Houdt review behapbaar en rollback chirurgisch.
- **Tests-first als safety net**: voor elke fase eerst karakterisatie-tests schrijven die het huidige gedrag vastleggen (geen vertrouwen → geen refactor).
- **Geen scope-creep**: per fase strict bij de fase blijven. Bug-fixes of "while I'm here" tweaks die je tegenkomt → issue maken, niet bundelen.
- **Re-extract graph** na elke fase: `graphify update . && graphify cluster-only . --no-viz` om vooruitgang objectief te zien (nodes/communities/cohesion).
- **Branch-naming**: `refactor/01-cleanup`, `refactor/02-chunk-builder`, etc.

---

## Fase 0 — Pre-flight (½ dag)

**Doel**: safety nets op orde, baselines vastleggen voordat we iets aanraken.

### Stappen
1. **Baseline test-status vastleggen**: `pytest apps/app-main/tests/ -v --tb=line > /tmp/baseline-tests.txt` — noteer welke tests groen zijn. Doel: na elke fase zelfde set groen.
2. **Coverage baseline**: `pytest --cov=apps/app-main/src/app_main/services --cov-report=term apps/app-main/tests/test_source_*` → noteer % voor source_processing.
3. **Karakterisatie-tests voor `SourceProcessingService`** schrijven. Minstens deze paden:
   - `process_source` met file-path input (PDF & markdown)
   - `process_source` met URL input
   - `process_source` met raw-text input
   - `embed_source` happy path
   - `run_summaries` happy path
   - Error path: source_id niet gevonden → ValueError

   Gebruik echte fixtures (geen mocks van de service zelf — alleen externe boundaries: docling-client, embedding-pipeline-client).
4. **Graph-baseline**: huidige `graphify-out/GRAPH_REPORT.md` kopiëren naar `/tmp/graph-baseline.md` om communities/cohesion later te vergelijken.

### Klaar als
- Alle bestaande tests groen
- Minimaal 5 karakterisatie-tests groen voor `SourceProcessingService`
- Branch `refactor/00-preflight` gemerged

### Rollback
- N/A (alleen tests toegevoegd, niets gewijzigd)

---

## Fase 1 — Quick-win cleanup (1 dag, 1 PR)

**Doel**: dood gewicht weg, structurele troep weg. Hoogste ROI per uur.

### Stappen
1. **Verwijder lege carcassen**:
   ```bash
   rmdir open_notebook api  # alleen __pycache__/ erin
   ```
2. **Beslis** over lege workspace-leden (`apps/chat/`, `apps/canvas/`, `pipelines/enrichment/`):
   - Optie A: hou ze als placeholders → voeg `README.md` toe met "Planned: <feature>, see issue #XX"
   - Optie B: `git rm -r` ze, voeg ze later toe wanneer er code komt
   - **Aanbeveling**: Optie B. Lege scaffolds zijn altijd misleidend.
3. **Strip dead params uit `SourceProcessingService.process_source`**:
   - Verwijder uit signature: `apply_transformations`, `embed`, `transformation_ids` (alle 3 zijn "Ignored (kept for backward compat)")
   - Audit eerst: `rg "process_source\(" apps/app-main` — check geen caller gebruikt deze met betekenisvolle waarde
   - Update docstring overeenkomstig
4. **Split `entity_persistence_service.py`** in twee bestanden:
   - `entity_persistence_service.py` → behoud `EntityPersistenceService`
   - Nieuw `resolution_log_service.py` ← `ResolutionLogService` hierheen
   - Update import in `apps/app-main/src/app_main/dependencies.py` en alle callers
   - Tests die deze importeren bijwerken

### Validatie
- `pytest apps/app-main/tests/ -x` → alles groen
- `mypy apps/app-main/src/` → geen nieuwe errors
- `git diff --stat` → checkbare wijziging (~-200 LOC, paar bestanden)

### Rollback
- `git revert <commit>`

---

## Fase 2 — Extract pure utilities uit god (½-1 dag, 1 PR)

**Doel**: de "makkelijke" delen uit `SourceProcessingService` halen — zaken zonder state of repo-deps.

### Stappen

1. **Maak `apps/app-main/src/app_main/services/chunking/` met**:
   - `chunk_builder.py` met module-level functies:
     - `from_document(document, source_id, settings) -> List[Chunk]` ← was `_document_to_chunks`
     - `from_transcription(transcription, source_id) -> List[Chunk]` ← was `_transcription_to_chunks`
     - `prepare_for_db(chunks, source_id) -> List[Dict]` ← was `_prepare_chunks_for_db`
   - **Pure functions**: geen `self`, alleen input → output. Makkelijk te testen.

2. **Maak `apps/app-main/src/app_main/services/ingestion/config_builder.py`**:
   - `build_ingestion_config(settings: ContentSettings, overrides: Optional[Dict]) -> IngestionConfig` ← was `_build_ingestion_config` (80 LOC)
   - Pure function, geen state.

3. **Verplaats `_strip_null_bytes`** naar `packages/shared/src/shared/utils/text.py::strip_null_bytes` (it's already module-level, dit is gewoon move + import update).

4. **Update `SourceProcessingService`** om de nieuwe utilities aan te roepen — class is nu ~260 LOC lichter (van 903 → ~640).

5. **Schrijf unit-tests** voor de geëxtraheerde utilities:
   - `tests/services/chunking/test_chunk_builder.py` (zonder DB, alleen input/output)
   - `tests/services/ingestion/test_config_builder.py`

### Validatie
- Karakterisatie-tests van Fase 0 blijven groen (cruciale assertie!)
- Nieuwe unit-tests groen
- Coverage van `chunk_builder.py` en `config_builder.py` ≥ 80%

### Rollback
- `git revert` deze PR — de utilities zijn standalone, geen DB-state betrokken

### Effect
- SourceProcessingService: 903 → ~640 LOC
- 3 nieuwe testbare units met directe coverage

---

## Fase 3 — Extract specialized services (1-2 dagen, 1 PR)

**Doel**: SourceProcessingService splitsen in 4 services rond duidelijke verantwoordelijkheden.

### Stappen

1. **Nieuwe class: `SourceExtractor`** in `services/source_extractor.py` (~250 LOC)
   - Public: `async def extract(content_state, settings, *, source_id) -> ExtractionResult`
   - Verplaats: `_extract`, `_process_file`, `_process_url`, `_process_text`, `_fetch_url_content`
   - Deps: docling client, ingestion workflow (geen DB-repos nodig hier — extractor is puur over content)
   - **Belangrijke beslissing**: `_process_file` (141 LOC) is nog steeds te lang. Splits intern in private helpers:
     - `_extract_with_ingestion_workflow()` (docling-pad)
     - `_extract_with_whisperx()` (audio/video)
     - `_route_by_extension()` (dispatcher)

2. **Nieuwe class: `SourceEmbeddingOrchestrator`** in `services/source_embedding_orchestrator.py` (~50 LOC)
   - Public: `async def embed(source_id) -> Dict[str, Any]`
   - Verplaats: `embed_source`, `_embed`
   - Deps: `source_repo`, embedding-pipeline-client

3. **Nieuwe class: `SourceSummarizationOrchestrator`** in `services/source_summarization_orchestrator.py` (~100 LOC)
   - Public: `async def run_summaries(source_id, transformation_ids) -> Dict[str, Any]`
   - Verplaats: `run_summaries`, `_run_transformations`
   - Deps: `transformation_repo`, summarization-pipeline-client

4. **`SourceProcessor` (= refactored `SourceProcessingService`)** (~120 LOC)
   - Public: `async def process_source(source_id, content_state, *, notebook_ids=None, processing_overrides=None) -> Dict`
   - Verantwoordelijkheid: **orchestreren** van extractor → update_source → chunks persistence
   - Deps: `source_repo`, `chunk_repo`, `settings_repo`, `SourceExtractor`, `ChunkBuilder`-utilities, `IngestionConfigBuilder`-utilities
   - Verplaats: `process_source` (afgeslankt), `_update_source`

5. **Update DI container** (`apps/app-main/src/app_main/dependencies.py`):
   - Nieuwe factory functies: `get_source_extractor()`, `get_source_embedding_orchestrator()`, `get_source_summarization_orchestrator()`, `get_source_processor()`
   - Update routers die de oude `get_source_processing_service` gebruikten

6. **Naming-besluit**: behoud `SourceProcessingService` als backward-compat shim die intern delegeert naar de nieuwe services, OF doe een hard cutover. **Aanbeveling**: hard cutover — uit `RULES.md`: _"Avoid backwards-compatibility hacks"_. We zitten in een refactor-fase, niet een API-versioning fase.

### Validatie
- **Karakterisatie-tests blijven groen** (zonder aanpassingen — als ze rood worden ben je gedrag aan het veranderen, niet refactoren)
- Nieuwe unit-tests per service (target: 70%+ coverage per nieuwe class)
- Manual smoke test: upload PDF → check source-record + chunks via API

### Rollback
- `git revert` deze PR. Dit is de hoogste-risico stap; daarom alle voorgaande safety nets nodig.

### Effect
- 1 god van 903 LOC → 4 services van 50-250 LOC
- Elke service heeft 1 reden om te veranderen (SRP)
- Routers kunnen targeted services injecteren in plaats van de hele god

---

## Fase 4 — Structurele cleanup (1-2 dagen, 1-2 PRs)

**Doel**: workspace-structuur op orde, root-level orphans weg.

### Stappen

1. **Verifieer of `semantic_layer/` nog actief gebruikt wordt**:
   ```bash
   rg "from semantic_layer|import semantic_layer" --type py
   ```
   Drie uitkomsten:
   - **Geen importers** → `git rm -r semantic_layer/` + commit (dood code)
   - **Importers in apps/packages** → migreren (zie 2)
   - **Alleen geïmporteerd door semantic_layer zelf** → standalone tool, OK om te laten maar verplaats naar `tools/semantic_layer/` of `experiments/`

2. **Als migratie nodig**: maak `packages/semantic-layer/`:
   - Standaard workspace-structuur met `pyproject.toml`, `src/semantic_layer/`, `tests/`
   - Verplaats de 8 .py files
   - Update alle importers
   - Run tests

3. **Re-graph en valideer**: `graphify update . && graphify cluster-only . --no-viz` — vergelijk met baseline. Verwacht:
   - Lagere node-count (lege dirs en orphan-code weg)
   - Communities rond semantic_layer zouden óf moeten verdwijnen óf moeten samenklikken met ontology-manager

### Validatie
- `pytest` van alle packages groen
- Geen import-errors bij `python -c "import semantic_layer"` (als gemigreerd) — werkt via package install
- Graph: minder isolated nodes, semantic_layer communities geïntegreerd

### Rollback
- `git revert` per stap

---

## Fase 5 — Coverage gaps dichten (1 dag, 1 PR per pipeline)

**Doel**: kritieke pipelines met te weinig tests beschermen voordat verdere refactoring ze raakt.

### Prioriteit
1. **`pipelines/retrieval/`** (397 LOC, 1 testfile) — _eerste_ omdat retrieval het hart van RAG is
   - Minstens: vector-search edge cases, KG-query routing, empty-result handling, malformed-query handling
   - Target: 5-10 tests

2. **`pipelines/summarization/`** (2572 LOC, 2 testfiles) — _tweede_
   - Skip de NotImplementedError-strategieën (correct gedrag al gecovered door erover)
   - Focus op Naive/TreeKG/RAPTOR happy + edge paden

### Validatie
- Coverage per pipeline ≥ 60% (start ergens)
- Nieuwe tests onafhankelijk (geen volgorde-afhankelijkheid)

### Rollback
- N/A (tests toegevoegd)

---

## Fase 6 — Splits van overige API-monolieten (optioneel, 1-2 dagen per file)

**Doel**: de andere >500 LOC bestanden splitsen — alleen als ze actief evolueren of pijn opleveren.

### Kandidaten (in volgorde)
- `services/extraction/api.py` (1175 LOC) → splitsen per endpoint-resource (extraction, jobs, health)
- `services/summarization/api.py` (790 LOC) → idem
- `apps/app-main/.../api/routers/sources_crud.py` (683 LOC) → splitsen per HTTP-verb of per resource
- `apps/app-main/.../api/routers/sources_upload.py` (615 LOC) → split upload-flow van URL-fetch
- `apps/app-main/.../api/routers/source_chat.py` (600 LOC)

**Aanpak per file**: gebruik FastAPI's `APIRouter`-per-resource patroon. Voor `services/*` (separate HTTP-services) idem.

**Niet doen als**: het file stabiel is en geen actieve wijzigingen kent. _Premature refactor is ook anti-pattern._

---

## Fase 7 — Documentatie + graph re-baseline (½ dag)

**Doel**: post-refactor state vastleggen voor de volgende ontwikkelaar (incl. jij over 6 maanden).

### Stappen

1. **Schrijf `ARCHITECTURE.md`** in repo-root met:
   - 4-laagse structuur (apps / packages / pipelines / services / frontend)
   - Core abstracties (Ontology, ExtractionResult, NotebookService, SourceService, KGResolver) en hun rol
   - Dependency-injection patroon + verwijzing naar `dependencies.py`
   - Service-naming-conventies (Service vs Orchestrator vs Repository)
2. **Update `docs/SUMMARIZATION_APPROACHES.md`** — markeer welke stubs nog open zijn na deze refactor (waarschijnlijk allemaal — refactor raakt summarization-strategy stubs niet).
3. **Re-extract graph**: `graphify update . && graphify cluster-only . --no-viz`
4. **Vergelijk met `/tmp/graph-baseline.md`** — schrijf bondige conclusie in PR-beschrijving:
   - Nodes: was 9.547 → nu ___
   - Communities: was 430 → nu ___
   - God-nodes top-10: vergelijk
   - Highlight verdwenen "PDF Acrobat" -achtige cruft + nieuwe schonere communities

---

## Tijdschema samenvatting

| Fase | Effort | Risico | Cumulatief |
|---|---|---|---|
| 0 — Pre-flight | ½ d | nul | ½ d |
| 1 — Cleanup | 1 d | laag | 1½ d |
| 2 — Extract utilities | 1 d | laag-medium | 2½ d |
| 3 — Split god-service | 1-2 d | **medium-hoog** | 4-4½ d |
| 4 — Structureel | 1-2 d | medium | 5½-6½ d |
| 5 — Coverage | 1 d (parallel mogelijk) | laag | 6½-7½ d |
| 6 — Optionele splits | 1-2 d p/file | laag-medium | open |
| 7 — Docs + graph | ½ d | nul | +½ d |

**Realistisch totaal voor kerntraject** (Fase 0-5, 7): **7-8 mandagen**. Fase 6 op verzoek toevoegen.

---

## Wat *niet* in dit plan zit (bewust)

- **Open stubs implementeren** (11 summarization-strategieën, MergedGraphAnalyzer): dit is feature-werk, geen refactor. Apart traject met eigen design-doc per strategie.
- **Frontend refactor**: 31k LOC, eigen analyse nodig. Wel: bekijk of de "Frontend Chat/Notes API"-coupling met API-clients herschikt moet.
- **pdfjs-dist npm-migratie**: makkelijk maar buiten scope.
- **Test-infra wijzigingen** (pytest plugins, fixtures): laat staan tenzij ze het refactoren blokkeren.
- **API-versioning of deprecation policy** voor publieke endpoints — dat is een product-vraag.

---

## Definition of Done (overall)

- [ ] `SourceProcessingService` bestaat niet meer; verantwoordelijkheden in 4 helder genoemde services
- [ ] Geen bestand >700 LOC in `services/` directories (richtlijn, geen harde regel)
- [ ] `open_notebook/`, `api/` weg; `semantic_layer/` óf gemigreerd óf bewust gehouden met reden
- [ ] `pipelines/retrieval/` met ≥5 tests
- [ ] `pyproject.toml` per workspace-lid heeft echte content (geen lege scaffolds)
- [ ] Graph re-extract toont **lagere god-node-edges** voor de gesplitste services en **geen** community met >150 nodes (Entity Filtering Workflow nu 175 → te grote brok)
- [ ] `ARCHITECTURE.md` in root, gerefereerd vanuit `README.md`

---

## Appendix — Onderbouwing per fase

### Waarom deze volgorde?

- **Fase 0/1 eerst**: laag risico, hoog moreel — concrete wins zonder gedragsrisico. Bouwt vertrouwen in de aanpak en levert direct schoner werkomgeving op voor latere fases.
- **Fase 2 vóór 3**: pure utilities extracten is reversibel en risico-vrij; je leert tegelijk waar de échte koppeling zit binnen de god-class. Eventuele verrassingen tijdens Fase 2 zijn nog vóór de gevaarlijke splitsing.
- **Fase 3 in het midden**: de gevaarlijkste stap, maar nu beschermd door karakterisatie-tests (Fase 0) en met de utilities al netjes apart (Fase 2). De resterende splitsing draait dus puur om service-grenzen, niet om implementatie-extractie.
- **Fase 4 na 3**: structurele cleanup raakt imports throughout codebase; doe dit ná de god-split zodat je niet imports moet bijwerken in code die zometeen toch wordt verplaatst.
- **Fase 5 parallel mogelijk**: coverage-werk is onafhankelijk van de service-refactor en kan door iemand anders worden gedaan.

### Waarom karakterisatie-tests en niet bestaande tests?

Bestaande tests van `SourceProcessingService` testen waarschijnlijk implementation details (mocks van interne methods). Karakterisatie-tests vangen het *externe gedrag* van `process_source`/`embed_source`/`run_summaries` — wat ná de refactor identiek moet blijven, ook al verandert de interne structuur compleet.

### Wat als Fase 3 mislukt?

Concrete fall-back: behoud `SourceProcessingService` als een **facade** die intern delegeert naar de nieuwe services. Tijdelijke pragmatische exit als de DI-wijzigingen elders te ingrijpend blijken. Tegen `RULES.md` "Avoid backwards-compatibility hacks", maar verdedigbaar als incident-mitigatie. Plan in dat geval een Fase 3.5 om de facade alsnog op te ruimen wanneer de routers één voor één migreren.
