# Docling Studio → Open Notebook — Integratieplan

> Status: plan / niet gestart. Basis = Open Notebook (deze repo). Donor = Docling
> Studio (`scub-france/Docling-Studio`, MIT). Doel: de visuele stijl, de
> responsiviteit en de PDF-inspectie-UX van Docling Studio overnemen, plus een
> aantal bewust gekozen extra capaciteiten — zonder Open Notebook's bredere
> functionaliteit (audio/video/URL/podcast/kennisgraaf) af te breken.

## Kerninzicht (bepaalt de scope)

Open Notebook heeft het grootste deel van Docling Studio's PDF-inspectie **al**:
`frontend/src/components/source/PdfChunkViewer.tsx` (canvas-bbox-overlay,
kleurgecodeerd "matches Docling Studio"), een `PipelineConfigPanel`, server-side
pagina-rasterisatie (`/page-preview`, `/page-count`) en persistente geometrie
(`chunk.positions = [[page,x1,x2,y1,y2]]` in SurrealDB). Dit is dus grotendeels
**stijl-adoptie + polish + correctheid**, geen greenfield-port.

Wat Docling Studio (DS) toevoegt zit in de **visuele/inspectielaag** — niet in de
RAG-pijplijn: ON's RAG is al een ruime superset (provider-agnostische modellen,
hybrid search met instelbare weging, agentic "Ask" met drie kiesbare modellen,
transformations, context-niveaus). Zie de twee onderbouwende vergelijkingen
onderaan.

## Uitgangspunten

- **Open Notebook = basis.** Niets van ON's brede functionaliteit sneuvelt.
- **Geen component-copy.** DS = Vue + vanilla CSS; ON = React + Tailwind 4/shadcn.
  Alleen framework-vrije zaken (design-tokens, pure scaling/kleur-modules,
  JSON-contracten) worden 1:1 overgenomen; de rest wordt herschreven.
- **Meten vóór optimaliseren** (m.n. Fase G).

---

## Fasen

### Fase A — Visuele identiteit (DS-design-tokens)
DS's look = bijna-zwarte surfaces + oranje accent `#f97316` + Inter (UI) / IBM
Plex Mono (cijfers/metadata) + 8px-radii + dunne scrollbars + actieve pills in
`accent-muted`. ~230 regels CSS-variabelen die 1:1 op ON's Tailwind-theme mappen.

- **Bestand:** `frontend/src/app/globals.css` (`@theme inline` + OKLCH-tokens in
  `:root`/`.dark`); fonts via `next/font` in `layout.tsx`.
- **Token-mapping:** `--accent #f97316` → `--primary`/`--ring`/`--sidebar-primary`;
  `--bg/--bg-surface/--bg-elevated` → `--background`/`--card`/`--popover`;
  `--accent-muted` → `--accent` (active); radius 8px → `--radius: 0.5rem`.
- **Effort:** 0,5–1 dag. **Risico:** laag (puur visueel, omkeerbaar).

### Fase B — "Inspect"-workspace
De viewer zit nu geknepen in de Chunks-tab (`SourceDetailContent.tsx`). DS's
kracht is een full-height drie-paneel layout met sleepbare scheiding.

- **Nieuwe route:** `frontend/src/app/(dashboard)/sources/[id]/inspect/page.tsx`
  met `<AppShell>`. Links chunk/structuur-lijst · midden PDF+overlay+paginator ·
  rechts properties/markdown/config.
- **Resizable panels:** voeg `react-resizable-panels` toe (nog niet aanwezig), of
  DS's eigen `mousedown`+document-listener-patroon (~40 regels) zonder dep.
- **Hergebruik:** `PdfChunkViewer`/`BboxOverlay` als middenpaneel; nieuwe Zustand
  `document-workspace-store.ts` (actieve pagina, selectie, panel-sizes).
- **Effort:** 2–3 dagen. **Risico:** middel.

### Fase C — Coördinaat-canonicalisatie (correctheidsfix)
**Bug:** Docling schrijft bboxes in **rauwe PDF-punten**, MinerU in **0–1
genormaliseerd**, beide in hetzelfde `chunk.positions`-veld. De frontend raadt het
formaat (`analyzeChunkFormat`) → fragiel. De Docling y-flip leunt op
`prov.page_height` dat vaak ontbreekt.

- **Backend (canoniek 0–1):** `pipelines/ingestion/src/ingestion/models/document.py`
  → `BoundingBox.from_docling` deelt door pagina-`width`/`height`
  (`doc.pages[page_no].size`) en flipt y correct. Dan klopt de docstring én kan de
  frontend-heuristiek weg.
- **Raakvlakken:** `chunk_builder.from_document` (positions-emit),
  `mineru_layout_parser` (al 0–1), `PdfChunkViewer.tsx` (`analyzeChunkFormat` weg).
- **Effort:** 1–2 dagen + her-ingest/backfill van bestaande sources. **Risico:**
  middel (datamigratie). **Blokkeert betrouwbare overlay + Fase F → vroeg doen.**

### Fase D — Inspectie-sub-features (DS-pariteit)
1. **LayersBar** — chip-rij per element-type (toggle zichtbaarheid). Port
   `elementColors.ts` verbatim; `hiddenTypes: Set<string>` als prop. (½ dag)
2. **Volledige conversie-config (#1)** — breid `PipelineConfigPanel` uit met DS's
   ontbrekende Docling-toggles: `do_code_enrichment`, `do_formula_enrichment`,
   losse `do_picture_classification`, `generate_page_images`, `images_scale`.
   Backend doorgeven aan de docling-service-config. (~1 dag)
3. **Chunk merge/split** — DS heeft `POST .../chunks/merge` en `.../split
   {cursorOffset}`; ON heeft create/update/delete maar geen merge/split. (1 dag)
4. **Result-tabs** — MarkdownViewer (`react-markdown`+`rehype-sanitize`),
   ImageGallery, StructureViewer. (1 dag)
- **Effort:** ~2–3 dagen. **Risico:** laag-middel.

### Fase E — Responsiviteit & polish
DS is volledig fluïde (nul media-queries, `minmax()`-grids, sleepbare panelen);
ON leunt op breakpoints + vaste grids.
- Maak de inspect-workspace fluïde via resizable panels (B) + het `min-h-0`/
  `overflow`-patroon dat `AppShell` al gebruikt. Mono-numerics op page-pills,
  token-counts, bbox-coördinaten.
- **Effort:** 1 dag. **Risico:** laag.

### Fase F — Documentstructuur-graaf (SurrealDB-port)
De DoclingDocument-tree (secties → paragrafen → tabellen → figuren, leesvolgorde,
pagina-binding) als **structurele** graaf in SurrealDB, náást ON's *semantische*
entiteitengraaf. Geeft structurele navigatie + een graaf-view zoals DS's Cytoscape.

**Datamodel** (DS Neo4j → SurrealDB, `migrations/49.surrealql`):

```surql
DEFINE TABLE doc_node SCHEMAFULL;
DEFINE FIELD source       ON doc_node TYPE record<source>;
DEFINE FIELD self_ref     ON doc_node TYPE string;            -- "#/texts/12"
DEFINE FIELD element_type ON doc_node TYPE string;            -- section_header|paragraph|table|picture|page|title|list_item|caption|formula|code
DEFINE FIELD text         ON doc_node TYPE option<string>;
DEFINE FIELD page         ON doc_node TYPE option<int>;
DEFINE FIELD level        ON doc_node TYPE option<int>;
DEFINE FIELD sequence     ON doc_node TYPE int;
DEFINE FIELD bbox         ON doc_node TYPE option<array<float>>;  -- [x1,y1,x2,y2] 0–1 (uit Fase C)
DEFINE INDEX dn_source    ON doc_node FIELDS source;
DEFINE INDEX dn_ref       ON doc_node FIELDS source, self_ref UNIQUE;

DEFINE TABLE parent_of    SCHEMAFULL TYPE RELATION FROM doc_node TO doc_node;  -- PARENT_OF
DEFINE TABLE next_node    SCHEMAFULL TYPE RELATION FROM doc_node TO doc_node;  -- NEXT
DEFINE TABLE derived_from SCHEMAFULL TYPE RELATION FROM chunk   TO doc_node;  -- HAS_CHUNK/DERIVED_FROM
-- ON_PAGE → `page`-veld; HAS_ROOT → top-level nodes = zonder inkomende parent_of.
```

- **Ingestie:** nieuwe `apps/app-main/.../services/graph/doc_graph_builder.py` leest
  de DoclingDocument-JSON (`json_path`), upsert `doc_node` + `RELATE`
  `parent_of`/`next_node`; `derived_from` koppelt chunks aan hun bron-`self_ref`.
  Idempotent (delete-rebuild per source). Hergebruikt Fase C's 0–1-bboxes.
- **API + view:** `GET /api/sources/{id}/structure-graph` (met pagina-limiet, vgl.
  DS 413 @ 200 pagina's); **Structure-tab** in de inspect-workspace via ON's
  bestaande **Sigma.js/graphology**-stack (geen Cytoscape-dep). Node-klik →
  highlight bbox via `self_ref`.
- **Effort:** 3–5 dagen. **Risico:** middel (datavolume → batched RELATE +
  index op `source` + query-limiet; sync via re-ingest).

### Fase G — Vector-search: brute-force → ANN — **Optie 1 gekozen**
ON doet nu `vector::similarity::cosine` als brute-force SELECT over de hele tabel
(`migrations/1.surrealql` `fn::vector_search`); de `embedding`-velden hebben geen
index en geen vaste dimensie. SurrealDB v2 ondersteunt ANN echter native.

- **Optie 1 (GEKOZEN) — native SurrealDB HNSW/MTREE.** In-architectuur, geen extra
  infra.
  ```surql
  -- migrations/49→50.surrealql
  DEFINE INDEX idx_source_embedding_hnsw ON source_embedding
    FIELDS embedding HNSW DIMENSION 768 DIST COSINE TYPE F32 EFC 150 M 12;
  -- idem source_insight, note
  ```
  Query: `WHERE embedding <|K,EF|> $query` + `vector::distance::knn()`.
  - **Dimensie vastpinnen** — kies één embedding-model (bv. `nomic-embed-text`
    768); modelwissel = index droppen/herbouwen. Vastleggen in Models-config.
  - **Geheugen** — HNSW (in-memory, snel) als RAM het toelaat; anders **MTREE**
    (disk, lager geheugen, trager).
  - **Valideren** — top-k geïndexeerd vs. brute-force op eigen corpus vóór je
    brute-force verwijdert.
- **Optie 2 (interim, niet als eindfix):** notebook/source-pre-filter vóór de
  cosine, zodat brute-force over een kleine subset draait. Combineerbaar met 1.
- **Optie 3 (alleen bij bewezen schaal):** externe ANN-store (OpenSearch/faiss/
  Qdrant), DS-stijl. Breekt single-store-eenvoud → niet nu.
- **Effort:** 1–2 dagen + herindexering. **Risico:** middel (jonge feature → meten;
  dimensie-pin is een bewuste inperking).

### Fase H — Overgenomen DS-sterktes (besloten)

#### H1 — Upload-guards + per-IP rate-limiting *(laag risico, vroeg doen)*
- Upload-guards in de source-upload-flow: `MAX_FILE_SIZE_MB` + `MAX_PAGE_COUNT`
  (tel PDF-pagina's via pypdfium — al gebruikt in `/page-count` — en weiger boven
  de limiet) + type-check.
- Per-IP rate-limiting: echte limiter (bv. slowapi) als middleware op
  `apps/app-main/.../api/app.py` (de `RateLimitError`-handler bestaat al), via env
  `RATE_LIMIT_RPM`.
- **Effort:** ~1 dag. **Risico:** laag. → De page-count-guard voorkomt OOM's zoals
  bij grote scanned PDF's.

#### H2 — Chunk-/versie-audit + frozen snapshots
- Nieuwe SurrealDB-tabellen (`migrations/...`): `chunk_edit` (append-only: chunk,
  op insert/update/delete/merge/split, before/after, actor, ts) en
  `document_snapshot` (bevroren: source, kind analysis/chunks, payload-JSON,
  created) — spiegelt DS's `chunk_edits`/`chunk_pushes`/`document_versions`.
- Schrijfpaden haken op de bestaande chunk-CRUD-endpoints; snapshot on-demand of
  bij ingest-completion. API + UI voor history + snapshot/restore.
- **Effort:** 2–3 dagen. **Risico:** middel (schrijfpad-dekking; opslaggroei →
  retentie/soft-delete).

#### H3 — Externe stores-push + stale-tracking *(hoog risico, optioneel/laatst)*
- Nieuwe tabellen: `store` (kind opensearch/neo4j, config, sealed credentials) +
  `document_store_link` (source, store, state Ingested/Stale/Failed, last_pushed).
- Push-service: chunks + embeddings → externe OpenSearch (knn)/Neo4j; markeer links
  `Stale` bij re-ingest. Secret-sealing (DS's Fernet `STORE_SECRET_KEY`-patroon)
  mee porten.
- **Effort:** 4–6 dagen. **Risico:** hoog (externe integraties + secret-management).
- ⚠️ **Afweging:** herintroduceert de externe-infra-last die Fase G's Optie 3 afwees.
  Alleen consistent als je ON echt een bestaand extern zoeksysteem wilt laten voeden.

*Afgewezen:* reasoning-trace export/import — ON's agentic "Ask" is al sterker.

---

## Overzicht & volgorde

| Fase | Inhoud | Effort | Risico |
|---|---|---|---|
| A  | DS-design-tokens → Tailwind-theme | 0,5–1d | laag |
| B  | Inspect-workspace (resizable 3-paneel) | 2–3d | middel |
| C  | Coördinaat-canonicalisatie | 1–2d | middel |
| D  | Sub-features (LayersBar, conversie-config, merge/split, result-tabs) | 2–3d | laag-middel |
| E  | Responsiviteit & polish | 1d | laag |
| F  | Documentstructuur-graaf (SurrealDB) | 3–5d | middel |
| G  | ANN-vectorsearch — Optie 1 (HNSW/MTREE) | 1–2d | middel |
| H1 | Upload-guards + per-IP rate-limiting | ~1d | laag |
| H2 | Chunk-/versie-audit + frozen snapshots | 2–3d | middel |
| H3 | Externe stores-push + stale-tracking | 4–6d | hoog |

**Aanbevolen volgorde:** A → H1 → C → B → D → E → G → F → H2 → H3.
Grove totaalschatting: ~18–28 dagen, gefaseerd opleverbaar (elke fase los waardevol).

## Kernbestanden (referentie)

**Open Notebook (basis):** `frontend/src/components/source/PdfChunkViewer.tsx`,
`SourceDetailContent.tsx`, `sources/[id]/page.tsx`, `lib/api/sources.ts` +
`lib/hooks/use-sources.ts`, `app/globals.css`,
`pipelines/ingestion/src/ingestion/models/document.py` (`BoundingBox.from_docling`),
`apps/app-main/.../services/chunking/chunk_builder.py`,
`apps/app-main/.../services/parsing/mineru_layout_parser.py`,
`apps/app-main/.../api/routers/sources_files.py`, `migrations/1.surrealql`.

**Docling Studio (donor, verbatim te kopiëren):**
`frontend/src/features/document/bboxScaling.ts`, `bboxPercent.ts`,
`elementColors.ts`, `shared/types.ts` (contracten),
`features/document/ui/LayersBar.vue`, `app/App.vue` `<style>` (design-tokens).

---

## Bijlage — onderbouwende vergelijkingen

### Conversie-pipeline (#1): DS vs ON
Geen superset; verschillend gevormd. **DS** = fijnmaziger Docling-micro-knoppen in
de UI (code/formula-enrichment, losse picture-classification, page-images,
images_scale) maar één engine. **ON** = engine-orchestratie (engine-keuze +
auto-fallback Docling↔MinerU, OCR-engine/-talen, VLM-modelkeuze) maar minder fijne
enrichment-toggles. → Fase D-2 dicht het gat aan ON-zijde.

### RAG-pijplijn: DS vs ON
ON is een ruime superset: provider-agnostische modellen per rol (Esperanto),
3 retrieval-types (text/vector/hybrid) met instelbare `text_weight`, agentic "Ask"
met drie kiesbare modellen, granulaire chat-context, transformations,
privacy-routing. DS's RAG is simpeler/opinionated. → DS's RAG niet porten; alleen
de **documentstructuur-graaf** (Fase F) is een complementaire aanvulling.
DS's enige unieke RAG-infra (externe "stores") = optionele Fase H3.
