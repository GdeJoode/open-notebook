# Onderzoek & overnameplan — Constella-features op open-notebook

> **Context.** Je wilt vijf features (afgeleid van Constella) overnemen in je bestaande
> lokale kennispijplijn. Dit is een **onderzoeksdeliverable, geen implementatie** — het brengt
> de huidige code in kaart en levert per feature het aansluitpunt, de inspanning, het risico,
> en een voorgestelde volgorde. Pad: `/mnt/e/repos/private/open-notebook`. SurrealDB blijft de
> enige store; bestaande pijplijn hergebruiken; Python; Constella = gedragsreferentie, geen codebron.
> Twee keuzes bevestigd: **lokaal cross-encoder rerank-model** (feature 5 — lokaal/gratis per query, als eigen service), **gefaseerde trigger** (on-demand → job) voor features 2/3.

---

## CONCLUSIE EERST

**Dit is grotendeels een bedraad-en-afmaak-klus, geen greenfield.** Vier van de vijf features
hebben al substantieel fundament; de meeste "nieuwe" code is het aan elkaar knopen van bestaande
repository-methodes, services en een al draaiende MCP-server.

| # | Feature | Wat er al staat | Inspanning | Risico |
|---|---|---|---|---|
| 1 | **MCP graph-tools** | MCP-server bestaat (`surrealdb-mcp`, FastMCP, 5 tools); alle backing-methodes bestaan op 1 na | **Laag** | Laag |
| 5 | **Hybride search + rerank** | BM25 + vector + lineaire fusie bestaan; alleen niet via de router ontsloten; reranker-stub aanwezig | **Laag→Midden** | Midden (eigen rerank-service) |
| 4 | **Citaties naar bron** | Provenance (pagina/bbox/chunk) wordt opgeslagen, maar niet doorgegeven aan antwoorden | **Midden** | Midden (LLM-trouw) |
| 2 | **Auto-link** | Embeddings + RELATE-patronen + source-level "related" bestaan; note-laag + orchestrator ontbreken | **Midden** | Midden (graph-explosie) |
| 3 | **Contradictiedetectie** | `claim`/`contradicts`-schema + LLM-judge-infra bestaan; de judge zelf ontbreekt | **Midden→Hoog** | **Hoog** (judge-precisie + kosten) |

**Belangrijke correctie op de brief:** embeddings draaien hier **lokaal op Ollama
(`mxbai-embed-large`, 1024-dim)** — niet op NIM. NIM is de **LLM**-laag (extractie/summarisatie/chat)
via de Track J-routing. Dat raakt feature 4/5/3 (LLM-calls = NIM; embeddings = lokaal/gratis).

**Aanbevolen volgorde** (afwijkend van de strikte 1→5 prioriteit, op grond van afhankelijkheid +
hefboom): **1 + 5a samen** (MCP-fundament + hybride-search ontsluiten — feature 1's `search`-tool
leunt op 5) → **5b** (lokale cross-encoder rerank) → **4** (citaties) → **2** (auto-link, on-demand) → **3**
(contradictie, on-demand) → daarna de **achtergrond-job-laag** voor 2/3.

---

## (a) ARCHITECTUURKAART

**Store — SurrealDB (enige store, multi-model).** Werk-DB = `staging` (zie geheugen-notitie; de
config-default `open_notebook` is verouderd). Kernobjecten:
- **Documenten/tekst:** `source` (`full_text`, `metadata`, `embedding`-aggregaat, BM25-geïndexeerd op `title`+`full_text`), `chunk` (`text`, `positions`=bbox `[[page,x1,x2,y1,y2]]`, `physical_page`, `printed_page`, `section_path`, `element_type`, `source`), `source_embedding` (1 rij/chunk, `embedding`, BM25 op `content`), `source_insight`, `note` (`content`, `embedding`, BM25-geïndexeerd).
- **Graph:** `entity` (canonical, `status`, `entity_type`, `source_documents`, `embedding`), edge-tabellen `relation` (entity→entity, ~1466), `mentions` (source→entity, 67 — Track U.2), `cites` (source→source, schema klaar, 0 — Track U.3), plus ontology-edges `contradicts`/`discusses`/`authored_by`/`implements`/… (grotendeels leeg), `claim` (~13 rijen, app-side aangemaakt — **niet** migratie-beheerd).
- **Structuur:** `doc_node` (Docling-structuur, `page`/`bbox`/`self_ref`), `parent_of`/`next_node`/`derived_from`.
- **Full-text:** `DEFINE ANALYZER my_analyzer` + BM25 SEARCH-indexen (`migrations/1.surrealql`); `fn::text_search` (6 targets) + `fn::vector_search` (cosine over chunk/insight/note-embeddings) (`migrations/1.surrealql`, `4.surrealql`).

**Ingest.** Docling → `apps/app-main/.../services/source_processor.py` → `chunk`-tabel (met
provenance) + `source.full_text` + `source.metadata` (`parser_engine_used`, `extraction_confidence`).
WhisperX voor audio. Embeddings via `pipelines/embeddings/.../service.py` (`embed_source`, `embed_note`),
**lokaal Ollama mxbai 1024-dim**.

**LLM & orkestratie.** LangGraph-grafen in `apps/app-main/src/app_main/graphs/`
(`ask.py`, `source_chat.py`, `chat.py`, `transformation.py`, `prompt.py`). LLM-calls via de
**Track J model-routing**: `apps/app-main/.../services/model_routing/llm_call.py`
(`call_candidate(..., json_mode=True)`, failover, providers `nvidia`/NIM + `ollama` + `llamacpp`).

**MCP.** Drie FastMCP-servers (stdio + sse/http): `surrealdb-mcp`
(`packages/surrealdb-service/.../mcp/server.py`), `llm-manager-mcp`, `file-manager-mcp`.
`surrealdb-mcp` heeft nu: `query_database`, `get_record`, `list_sources`, `search_similar` (vector),
`get_entity_graph`. **Geen auth** (stdio-lokaal prima; aandacht bij HTTP-blootstelling).

**Retrieval — twee lagen (belangrijk onderscheid).**
- **Chunk/note-search:** `SearchRepository.{text,vector,hybrid}_search` (`…/repositories/search.py`) ↔ `RetrievalService` (`pipelines/retrieval/.../service.py`) ↔ router `/search`. **Hybrid bestaat (lineaire text+vector-fusie) maar wordt niet via de router ontsloten** (router kiest alleen `text` óf `vector`). Een **`Reranker`** (`pipelines/retrieval/.../reranker.py`, heuristiek score+embed-sim) bestaat maar is **nergens aangeroepen**.
- **Source-level (deze sessie gebouwd — Track R):** `find_related_by_embedding` (R.1, source.py:364), `kg_source_scorer` (R.2), `hybrid_fusion` RRF (R.3), endpoints `/sources/{id}/related[-kg|-hybrid]`. Plus `mentions`/`cites`-materialisatie (U.2/U.3). **Andere vraag** dan de chunk-search ("welke bronnen lijken op deze bron" vs "welke chunks matchen deze query") — geen overlap, wel herbruikbare bouwstenen.

---

## (b) PER FEATURE — aansluitpunt · inspanning · risico

### Feature 1 — Lokale MCP-server als graph-tools (FUNDAMENT) · **Laag** · Risico Laag
**Aansluitpunt.** Breid `packages/surrealdb-service/src/surrealdb_service/mcp/server.py` uit met de 5 tools; bijna alle backing bestaat:

| Tool | Bestaande backing | Status |
|---|---|---|
| `search` | `SearchRepository.hybrid_search` (na 5a ontsloten) | wrappen |
| `get_node` | `get_record`-tool doet dit al via `type::thing($id)` (polymorf) | bestaat |
| `related` | `load_mentions_edges`, `load_cites_edges`, `find_related_by_embedding`, `get_entity_detail` — **4 gespecialiseerd; geünificeerde "alle edges van node" ontbreekt** | 1 nieuwe methode `load_all_edges(node_id)` (~50 r.) of rauwe query |
| `cite` | `SourceRepository.relate_cites` (Track U.3) | bestaat (volledig) |
| `add_note` | `NoteRepository.create_with_embedding` + `add_to_notebook` | bestaat (volledig) |

**Let op:** de bestaande `get_entity_graph`-tool bevraagt een `relates_to`-edge; de echte KG-edge is
`relation` — verifiëren/repareren bij het bouwen.
**Waarom dit het fundament is:** alle Claude Code-sessies praten met dezelfde SurrealDB → het
"gedeeld geheugensubstraat" is inherent zodra de tools er zijn. Run: `surrealdb-mcp --transport stdio`.
**Risico.** Laag. Aandachtspunt: geen auth (alleen relevant bij HTTP-transport).

### Feature 5 — Hybride search + rerank · **Laag→Midden** · Risico Midden
**Aansluitpunt (5a, laag).** `SearchRepository.hybrid_search` combineert al BM25 + vector (lineair).
Ontbreekt: ontsluiting via de router (`apps/app-main/.../api/routers/search.py` doet alleen text|vector)
en het doorgeven van de query-embedding. Wiring = `mode="hybrid"` toevoegen + embedding doorrijgen via
`RetrievalService.hybrid_search`.
**Aansluitpunt (5b, midden — gekozen: lokaal cross-encoder rerank-model).** Na de top-N hybride hits:
herorden met een lokaal cross-encoder rerank-model.
- **Modelkeuze:** een **meertalige** cross-encoder, bv. `bge-reranker-v2-m3` — load-bearing omdat het corpus
  **Nederlandstalig** is; `mxbai-rerank` is sterk Engels-gericht en daarom hier minder geschikt.
- **Plaatsing (architectuur-consistent):** een **eigen microservice** `services/reranker/` (kleine FastAPI die
  de cross-encoder laadt), in `docker-compose.yml` naast `docling`/`mineru`/`whisperx`/`summarization`; `app-main`
  roept 'm aan via `RERANKER_SERVICE_URL` (spiegelt `DOCLING_SERVICE_URL`/`MINERU_SERVICE_URL`). Zo blijft de zware
  dep (`sentence-transformers`/`torch`, die `whisperx` al meebrengt) **buiten `app-main`**. *Lichter maar
  afgeraden:* een in-process `CrossEncoder` in `app-main` (torch in het hoofdproces).
- **Dependency-motivatie (per randvoorwaarde):** een cross-encoder is het kwaliteits-passende rerank-middel;
  isoleren in een eigen service spiegelt de bestaande zware-ML-services (`whisperx`=torch) en houdt `app-main`
  licht; volledig **lokaal → geen per-query-kosten** (consistent met de lokale embeddings). De heuristische
  `Reranker` (`pipelines/retrieval/.../reranker.py`) blijft als zero-dep noodrem beschikbaar.
**Inspanning.** 5a laag (puur bedrading); 5b midden (kleine nieuwe service + model + bedrading + `rerank`-flag).
**Risico.** Midden: nieuwe service/container + model-download/-grootte; cross-encoder-latency op CPU
(mitigatie: alleen top-N reranken, klein/gequantiseerd model, GPU indien beschikbaar). **Geen** per-query
LLM-kosten (lokaal).

### Feature 4 — Citaties naar exacte bron · **Midden** · Risico Midden
**Aansluitpunt.** Provenance wordt **opgeslagen** (`chunk.positions`/`physical_page`/`section_path`/`source`,
`source.metadata`) maar **niet doorgegeven**: `apps/app-main/.../graphs/ask.py` (`provide_answer`) en
`source_chat.py` halen context op maar rijgen chunk-id/pagina/bron **niet** in de prompt of de output.
Te raken: `RetrievalService.vector_search` (chunk-metadata teruggeven) → `ask.py` graph-state +
prompt-templates (`ask/query_process`, `ask/final_answer`: "citeer [bron, pagina, chunk]") →
gestructureerd antwoord-schema (`citations: [{source, page, chunk_id}]`).
**Inspanning.** Midden — leidingwerk door de LangGraph-state + prompt + output-schema; de data bestaat.
**Risico.** Midden — LLM-citatietrouw (citeert het de chunk die de bewering echt staaft?). Mitigatie:
chunk-id's expliciet meesturen + een lichte post-check dat de geciteerde chunk in de retrieval-set zat.

### Feature 2 — Auto-link (nieuwe notitie → verwant → RELATE) · **Midden** · Risico Midden
**Aansluitpunt.** Substraat bestaat: `Note` met `embedding` (`embed_note`), BM25-index op notes,
RELATE-patronen overal (idempotent-upsert in `entity_persistence_service._upsert_relation`), en
`find_related_by_embedding` **op source-niveau**. Ontbreekt: (1) note-niveau
`NoteRepository.find_related_by_embedding` (spiegel van `source.py:364`), (2) een note↔note edge-tabel
(nieuwe migratie — volg het **non-destructieve patroon van migratie 66/67** + de **S.4-preventieregel**),
(3) een orchestrator `note_auto_link_service` (embed → related → drempel → RELATE, idempotent), (4) de
trigger. **Gekozen: on-demand eerst** (endpoint + MCP-tool), **job later** (hergebruik de bestaande
job-queue / `EMBEDDING_GENERATE`-patroon).
**Hergebruik (deze sessie gebouwd):** Track R related-by-embedding, de `mentions`-materialisatie,
de R.6-drempel/saliency-aanpak tegen graph-explosie, de **RELATE-is-niet-idempotent**-les (clear-before-relate / pair-dedup).
**Inspanning.** Midden.
**Risico.** Midden — graph-explosie (drempel + named/saliency-weging hergebruiken), scope (note↔note
vs ook note↔source), sync-onderhoud (her-linken bij bewerken).

### Feature 3 — Contradictiedetectie (LLM beoordeelt verwante paren) · **Midden→Hoog** · Risico **Hoog**
**Aansluitpunt.** Scaffolding leeft: `claim`-tabel (~13 rijen) + `contradicts`-edge + ontology
`CONTRADICTS` (`scholarly.yaml`) + triage-semantiek `VERSTERKT`/`RELATED` (`config/triage_config.json`).
LLM-judge-infra herbruikbaar: `RoutedLLMCaller` + `call_candidate(json_mode=True)`. Ontbreekt: (1) een
**kandidaat-paar-generator** — hergebruik R.1/R.2 "related" om alleen *verwante* paren te beoordelen
(niet O(n²)), (2) de **pairwise judge-service** (`reinforces`/`contradicts`/`neutral` + confidence +
reasoning), (3) edge-persistentie (een `contradicts`/`relation`-edge met `verdict`), (4) de trigger
(on-demand eerst, job later). **Let op:** `claim`/`contradicts` bestaan live maar zijn **niet
migratie-beheerd** → eerst een migratie die hun schema vastlegt (S.4-preventie) vóór je erop bouwt.
**Inspanning.** Midden→Hoog — de judge-kwaliteit is het moeilijke deel.
**Risico.** **Hoog** — judge-precisie (valse tegenstrijdigheden vervuilen de graph; pas dezelfde
**precisie-eerst**-discipline toe als bij U.3 `cites`), kosten (begrens tot verwante paren + drempels),
en de semantische overlap met de bestaande triage `VERSTERKT`.

### Secundair (alleen haalbaarheid)
- **Graph-visualisatie (Constella's canvas):** **bestaat al grotendeels** — `frontend/.../knowledge-graph/`
  met `SigmaGraphView.tsx` (Sigma + forceAtlas2), plus de **"Document Graph"-tab** (Track U.4) en
  `apps/app-main/.../routers/knowledge_graph.py`. Een "kale" canvas is dus een UI-uitbreiding, geen nieuwbouw.
  Haalbaarheid: hoog; valt buiten deze Python-fase.
- **Browser-clip (Constella's overlay): NIET onderdeel van dit plan / de bouwvolgorde.** Per de brief
  ("secundair, alleen kort op haalbaarheid beoordelen, niet uitwerken"). **Genuinely nieuw** — vereist een
  browser-extensie/Electron-overlay (frontend/desktop), buiten de Python-only/geen-frontend-fase.
  Haalbaarheid: midden; hoort thuis als een **aparte toekomstige track**, niet in features 1–5.

---

## (c) VOORGESTELDE VOLGORDE

1. **Feature 1 + 5a samen** — MCP graph-tools + hybride-search ontsluiten. Laagste inspanning, hoogste
   hefboom; `search`-tool leunt op 5a; levert direct het gedeeld-geheugensubstraat.
2. **Feature 5b** — lokaal cross-encoder rerank-model (`bge-reranker-v2-m3`, meertalig; eigen service à la docling/whisperx; achter een `rerank`-flag, top-N).
3. **Feature 4** — citaties: provenance door de antwoord-grafen rijgen (vergroot vertrouwen, onafhankelijk).
4. **Feature 2** — auto-link, **on-demand** (note-similariteit + edge-tabel + orchestrator + endpoint/MCP-tool).
5. **Feature 3** — contradictie, **on-demand**, precisie-eerst (hergebruikt feature 2's verwante-paren + judge-infra).
6. **Achtergrond-job-laag** voor 2/3 (fase 2 van de trigger-keuze), zodra het on-demand-gedrag bevalt.
7. **Secundair** (apart): graph-canvas uitbreiden (bestaat al); browser-clip als eigen track.

---

## AANNAMES & ONZEKERHEDEN (expliciet)
1. **Pad** = `/mnt/e/repos/private/open-notebook` (aangenomen; bevestig indien anders).
2. **Embeddings zijn lokaal (Ollama mxbai 1024-dim), niet NIM** — de brief noemt "Embeddings/LLM: NVIDIA NIM"; in werkelijkheid alleen de LLM-laag is NIM. Geverifieerd in `model_routing` + `pipelines/embeddings`.
3. **`claim`/`contradicts` bestaan live maar zijn niet migratie-beheerd** (app-side aangemaakt). Vóór feature 3: een migratie die hun schema vastlegt (S.4-preventieregel), anders schema-drift zoals bij `mentions`/`cites` (migratie 66/67).
4. **`get_entity_graph`-MCP-tool bevraagt `relates_to`** terwijl de echte edge `relation` is — te verifiëren/repareren in feature 1.
5. **"Notities" vs "bronnen":** Constella is note-centrisch; open-notebook is bron/document-centrisch met óók een `note`-tabel. Auto-link/contradictie kunnen op notes, sources of beide. Aanbeveling: leun op het **al-gebouwde source-level substraat** (Track R) en breid uit naar notes; de exacte scope is een productkeuze.
6. **Kosten:** feature 3 doet LLM-calls (NIM) per beoordeeld paar — begrens via alleen-verwante-paren + drempels. Feature 5b (lokale cross-encoder) is **gratis per query** (lokaal, zoals de embeddings); de prijs is een nieuwe rerank-service/model + cross-encoder-latency. Embeddings blijven lokaal.
7. **Geen frontend in deze fase** (per randvoorwaarde); features 1–5 zijn backend/Python.

---

## VERIFICATIE (hoe je dit straks end-to-end test)
- **Feature 1:** start `surrealdb-mcp --transport stdio`, koppel als MCP-server in een Claude Code-sessie, roep `search`/`get_node`/`related`/`cite`/`add_note` aan; bevestig dat twee sessies dezelfde nodes zien (gedeeld substraat).
- **Feature 5:** `GET /search?mode=hybrid&rerank=true` op een bekende query; vergelijk top-k met/zonder rerank; meet latency.
- **Feature 4:** stel een vraag via de `ask`-graaf; bevestig dat het antwoord `citations: [{source, page, chunk_id}]` draagt en dat de pagina klopt met `chunk.physical_page`.
- **Feature 2:** maak een note, roep de auto-link-tool aan; bevestig idempotente note↔note RELATE-edges + drempelgedrag.
- **Feature 3:** voer een verwant bronpaar in de judge; bevestig `verdict`+confidence+reasoning en een `contradicts`-edge alleen bij hoge zekerheid (precisie-eerst).
- Reuse de bestaande test-conventies (`@requires_docker` roundtrips tegen een testcontainer; `uv run --project <pkg> pytest`); live tegen `staging` met expliciete `SURREAL_DATABASE=staging`.
