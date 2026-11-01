# Verschilanalyse signaalsysteem

Stand: 29 oktober 2026 · stap 1 van `opdracht.md` · branch `docs/signaalsysteem-fo`

Dit document vergelijkt het functioneel ontwerp (FO, `functioneel-ontwerp.md`) met de code van de fork. Het is input voor de ontwerpbesluiten (Architecture Decision Records, ADR's) in stap 2. Er is niets gebouwd of gewijzigd.

## 1. Werkwijze en legenda

- Bronnen: de code in `apps/`, `packages/`, `pipelines/`, `services/`, `migrations/` en `docker-compose.yml`; daarnaast `ARCHITECTURE.md`, `docs/tracks/_status.md` (momentopname van 23 juli 2026) en de plannen en statussen van de latere tracks PC, N, OE, Q en WB in `docs/tracks/`.
- Elke statusclaim verwijst naar een bestand, waar zinvol met regelnummer. Claims die alleen op documentatie berusten, staan als zodanig gemarkeerd. Aannames staan in paragraaf 4.
- Regelnummers gelden voor de stand van deze branch.

| Status | Betekenis |
| --- | --- |
| aanwezig | de fork doet wat het FO vraagt, eventueel met configuratie |
| gedeeltelijk | er is een bruikbare basis, maar er ontbreekt iets wezenlijks |
| ontbreekt | er is niets in de fork dat deze rol vervult |
| conflicteert | de code doet iets anders dan het FO of een besluit in FO paragraaf 14 voorschrijft |

## 2. Overzicht

| FO-onderdeel | Status | Kern | Paragraaf |
| --- | --- | --- | --- |
| Documentextractie (docling-serve) | conflicteert | eigen, synchrone Docling-dienst met eigen verrijkingen; geen docling-serve | 3.1 |
| Transcriptie (WhisperX-container) | gedeeltelijk | eigen WhisperX-dienst; model blijft geladen | 3.1 |
| Planning (schema-trigger of APScheduler) | ontbreekt | geen scheduler; alleen een takenwachtrij en een mapbewaker | 3.2 |
| Verwerkingsrun | gedeeltelijk | `job`-records met status en herhaalpogingen; geen pipelineversie, modellen of koppeling vanuit fragmenten | 3.2, 3.7 |
| Modelpoort en lokale modelserver (llama-swap) | conflicteert | Ollama op de host, deels via Ollama-specifieke API-aanroepen in code | 3.3 |
| Keuzemodel Kev | ontbreekt | geen Kev; documentclassificatie via een gewoon taalmodel | 3.3, 3.6 |
| Modelregister met "toegestaan voor intern" | gedeeltelijk | providerregister met `is_local`; routering per privacymodus (Track J) | 3.3 |
| Pipeline tien stappen (FO 4) | gedeeltelijk | vaste keten parse, embed, extract, graph in de fork; geen registratie, poortwachter of selectie | 3.4 |
| Structuurgebonden fragmentatie | aanwezig | fragmenten met sectiepad, pagina en elementtype uit Docling | 3.4 |
| Entiteitextractie met referentielijsten | gedeeltelijk | aanwezig, niet als LangGraph-module; CBS- en TOOI-lijsten aanwezig, BWB ontbreekt | 3.5 |
| Deduplicatie met review | gedeeltelijk | meerlaagse deduplicatie en reviewwachtrij aanwezig; geen LangGraph-interrupt | 3.5 |
| Labeling met codeboek | gedeeltelijk | ontologieën met beleidsthema's; geen Iv3-taakvelden, geen Kev | 3.5 |
| Bronkaart | ontbreekt | geen tabel of model | 3.6 |
| Poortwachter | ontbreekt | geen regels of model voor route en classificatie | 3.6 |
| Classificatie intern/extern | conflicteert | `private`-vlag met standaard "niet privé" en globale standaard `cloud` | 3.6, 3.8 |
| Bronversie en inhoudshash | ontbreekt | geen versietabel; herverwerking verwijdert fragmenten | 3.7 |
| Herkomst op fragmentniveau | gedeeltelijk | fragment kent bron, pagina en sectie; geen run of model | 3.7 |
| Afgeleid-markering | conflicteert | notities en inzichten (`source_insight`) komen in zoekresultaten | 3.7, 3.9 |
| Classificatiefilter in de API | ontbreekt | zoek- en leesroutes filteren niet op classificatie | 3.8 |
| Toegang met sleutels en rechten | gedeeltelijk | gedeeld wachtwoord voor `/api`; agentsleutels met `read`/`write`/`admin` alleen voor `/api/v1/agents` | 3.8 |
| Versleutelde opslag van API-sleutels | ontbreekt | sleutels via omgevingsvariabelen | 3.8 |
| MCP-server open-notebook-mcp | conflicteert | eigen `surrealdb-mcp` met directe databasetoegang, zonder filter en zonder toegangscontrole | 3.9 |
| Wiki-generatie | ontbreekt | geen code | 3.9 |
| Export naar Obsidian | gedeeltelijk | export per entiteit aanwezig; overschrijft hele bestanden, geen classificatie of gegenereerd blok | 3.9 |
| Workflow-editor Langflow | ontbreekt | geen Langflow | 3.10 |
| LangGraph-checkpoints in lokaal bestand | aanwezig | SQLite-checkpointbestand voor de bestaande grafen | 3.10 |
| Uitrol met Docker Compose | gedeeltelijk | één Compose-bestand; één netwerk, alle poorten op de host, inloggegevens in platte tekst | 3.11 |
| GPU-beheer | conflicteert | vier GPU-afnemers zonder coördinatie | 3.11 |
| Chat, ask, podcasts, interface | aanwezig | blijven staan; recent is de interface juist uitgebreid (Track WB) | 3.12 |
| Ingebouwde bronverwerking vervalt | conflicteert | meerdere actieve innameroutes en een volledig automatische keten (Track PL) | 3.13 |

## 3. Bevindingen per onderdeel

### 3.1 Documentextractie en transcriptie (FO 7, 12, 13.1)

**`services/docling` tegenover docling-serve: conflicteert.**

- De fork heeft een eigen FastAPI-dienst, geen docling-serve. Het eindpunt `POST /process` werkt synchroon op een pad in een gedeeld volume en schrijft uitvoer naar een uitvoermap (`services/docling/api.py:126-136`, `:187-234`). docling-serve werkt volgens het FO met asynchrone taken.
- De parser wordt één keer geladen en blijft in het proces (`services/docling/api.py:118-125`). Er is geen code die het videogeheugen (VRAM) vrijgeeft na een taak; zoeken op `empty_cache` in `pipelines/ingestion/src` en `services/` levert niets op.
- De dienst doet meer dan Docling: tekstaanvulling met PyMuPDF, PDF-annotaties en handschriftherkenning via een vision-language model (VLM) op Ollama (`services/docling/api.py:31-34`; `docker-compose.yml:82-85`). docling-serve kent die verrijkingen niet (aanname A3). Bij vervanging vervallen ze, of ze moeten als aparte bouwsteen terugkomen.
- De dienst gebruikt de parser- en exportcode uit `pipelines/ingestion` (`services/docling/Dockerfile`, regel `COPY pipelines/ingestion/src/ingestion /app/ingestion`).
- Naast Docling heeft de fork MinerU als tweede parser met een terugvalroute (`services/mineru/`, `apps/app-main/src/app_main/services/parsing/engine_dispatcher.py`, `auto_fallback.py`) en GROBID voor literatuurverwijzingen (`docker-compose.yml:35-43`). Het FO noemt deze niet.

**`services/whisperx` tegenover een WhisperX-container onder llama-swap: gedeeltelijk.**

- Eigen dienst met `POST /transcribe`, sprekerherkenning en woordtijdcodes (`services/whisperx/api.py:197-225`; `docker-compose.yml:130-140`). Functioneel dekt dit FO 7.
- Het model wordt bewaard in een procesvariabele (`services/whisperx/api.py:117-121`) en blijft geladen; stoppen na een bundel taken (FO 12.2) is niet ingericht.
- Het Hugging Face-token loopt via een omgevingsvariabele, niet via Docker secrets (`docker-compose.yml:140`).
- Of deze dienst onder llama-swap kan, is niet uit de code af te leiden; dat is proef 1 uit `opdracht.md`.

### 3.2 Takenwachtrij, planning en verwerkingsruns (FO 5, 8)

**Planning: ontbreekt.**

- Er is geen APScheduler, geen cron en geen feedparser in de afhankelijkheden (`apps/app-main/pyproject.toml`; zoeken op `apscheduler` en `feedparser` in alle `pyproject.toml`-bestanden levert niets op).
- `LibrarianService.enqueue_due` is als scheduler-ingang bedoeld (`apps/app-main/src/app_main/services/audit/librarian_service.py:3-10`, `:38`), maar heeft geen aanroeper in `apps/app-main/src`.
- Er is wel een mapbewaker op basis van watchdog (Track G.5): `apps/app-main/src/app_main/services/agents/file_watcher.py`, standaard uit (`apps/app-main/src/app_main/config.py:201`). Die doet inname, geen planning, en bepaalt geen classificatie. Het FO plaatst de mapcontrole als Python-component in Langflow (FO 5).

**`packages/job-queue`: gedeeltelijk als basis voor verwerkingsruns.**

- Een eigen wachtrij met worker, register en repository (`packages/job-queue/src/job_queue/`, 928 regels). De tabel `job` legt type, status, payload, resultaat, foutmelding en herhaalpogingen vast (`migrations/28.surrealql:7-17`, `max_retries` standaard 2). De worker probeert opnieuw en zet daarna op `failed` (`packages/job-queue/src/job_queue/worker.py:296-320`).
- Wat ontbreekt voor een verwerkingsrun volgens FO 8: pipelineversie, gebruikte modellen, logboek per stap, een record ook als er niets nieuws was, en melding aan de beheerder. Fragmenten verwijzen niet naar een job; alleen `source.command` verwijst naar de laatste job (`migrations/29.surrealql`).
- Jobtypen zijn gekoppeld aan de huidige bronverwerking (`packages/shared/src/shared/types/enums.py:144-178`: `DOCUMENT_PARSE`, `AUDIO_TRANSCRIBE`, `EMBEDDING_GENERATE`, `ENTITY_EXTRACT` en andere).
- FO paragraaf 14 besluit: "runs draaien los van de editor; planning via de schema-trigger van Langflow of APScheduler". Het FO zegt niet of de takenwachtrij van de fork blijft voor de stappen die in de fork draaien. Zie open vraag 5.

### 3.3 Modelkoppeling tegenover llama-swap en Kev (FO 6, 9, 12.2; controle 4)

**Hoe modellen nu worden aangeroepen.** Er zijn vier routes naast elkaar:

1. **Esperanto via `ModelManager`** (`packages/llm-manager/src/llm_manager/manager.py:8`, `:16`; `providers/factory.py`). Gebruikt door de LangGraph-grafen, zoeken, samenvatting, voorbewerking en de embeddings-pipeline (`apps/app-main/src/app_main/graphs/utils.py:9`, `pipelines/embeddings/src/embeddings/providers/esperanto_provider.py`). Het providerregister kent onder meer `ollama` en `llamacpp` als lokaal (`packages/llm-manager/src/llm_manager/providers/registry.py:91-101`, `:121-130`) en een `openai-compatible`-route voor NVIDIA en Ollama Cloud (`providers/factory.py:41-62`).
2. **Track J-routering** in `apps/app-main/src/app_main/services/model_routing/`: per taak (extractie, samenvatting, chat) een geordende providerketen met failover. `PrivacyMode` kent `cloud` en `private` (`route_resolver.py:53-61`); standaard cloudketen `["nvidia"]` met Ollama als lokale terugval (`route_resolver.py:84-85`). De privacymodus volgt document, notebook, globaal, met "privé is plakkerig" (`privacy_resolver.py`, docstring).
3. **`packages/shared/src/shared/model_routing.py` met `model_routing.yaml`**, gebruikt door de extractiedienst. Niveaus `public`, `internal`, `confidential`; standaard `internal` (`packages/shared/src/shared/model_routing.yaml:16`). De NVIDIA-provider staat uit (`model_routing.yaml`, blok `providers.nvidia.available: false`). Deze module roept Ollama rechtstreeks aan op `/api/generate` en `/api/embed` (`packages/shared/src/shared/model_routing.py:284`, `:353`).
4. **Directe HTTP-aanroepen naar Ollama** buiten Esperanto: `pipelines/entity-filtering/src/entity_filtering/resolution/llm_matcher.py:349` (`/api/chat`), `packages/semantic-intelligence/src/semantic_intelligence/config.py:64` (`/api/embed`), de handschrift-VLM in de Docling-dienst (`docker-compose.yml:82-84`) en de samenvattingsdienst (`docker-compose.yml:101-108`).

**Tegenover het FO.**

- *Lokale modelserver (FO 14, besloten: llama-swap met llama-server):* **conflicteert.** Alle lokale modellen draaien nu op Ollama op de host, buiten Docker (`docker-compose.yml:84`, `:103`, `:108`, `:154`, `:246-247`). Dat botst ook met harde eis 4.
- *"Esperanto wijst naar llama-swap; dat is configuratie, geen code" (FO 13.2 punt 5):* **conflicteert gedeeltelijk.** Voor route 1 is het configuratie, mits een OpenAI-compatibele provider naar llama-swap wijst. Routes 3 en 4 gebruiken Ollama-specifieke eindpunten (`/api/generate`, `/api/chat`, `/api/embed`). Die vragen codewijzigingen als llama-swap alleen een OpenAI-compatibel eindpunt biedt (aanname A1).
- *Modelregister (FO 9):* **gedeeltelijk.** Het register legt naam, provider en `is_local` vast (`registry.py:24`), maar niet "toegestaan voor intern", versie of aanroepwijze CLI. Modelrijen staan in de database; een migratie die de tabel `model` definieert is niet gevonden in `migrations/` (alleen `model_route`, `migrations/51.surrealql:31`).
- *Claude-CLI als aanroepwijze (FO 9):* **ontbreekt.** Geen code die `claude -p` aanroept.
- *Kev (FO 6, 14):* **ontbreekt.** Geen code of configuratie die Kev noemt. De nauwste verwant is de documentclassificatie in `apps/app-main/src/app_main/services/preprocessing_service.py:23` (een vaste lijst documentsoorten via een gewoon taalmodel), zonder gekalibreerde kansen.
- *Beleid "intern alleen lokaal, geen terugval naar de cloud" (FO 9):* **gedeeltelijk.** Track J weigert cloudproviders in `private`-modus (`route_resolver.py:199-216`). Maar de standaard is `cloud` (zie 3.6), en de routes 3 en 4 lopen niet via deze controle.
- *Embeddings lokaal (FO 9):* **aanwezig.** Embeddings lopen altijd lokaal en vallen buiten de Track J-routering (`ARCHITECTURE.md` paragraaf 8, invariant; test `test_embedding_local_guardrail.py` volgens die paragraaf). De documentatie is niet eenduidig over het model: `ARCHITECTURE.md` paragraaf 8 noemt `nomic-embed-text` (768 dimensies), paragraaf 12a en `packages/shared/src/shared/models/source.py:130` noemen `mxbai-embed-large` (1024 dimensies). Zie open vraag 8.
- *Audit van modelaanroepen (FO 9):* **gedeeltelijk.** Track J schrijft per aanroep een `routing.served`-regel in `metrics` met provider, model, taak, bron en notebook (`apps/app-main/src/app_main/services/model_routing/telemetry.py`; `ARCHITECTURE.md` paragraaf 8). Classificatie en run staan er niet in; routes 3 en 4 schrijven deze regel niet.

### 3.4 Pipelines `ingestion`, `embeddings` en `retrieval` tegenover FO 4 en 7

| FO-stap | Fork | Status | Verwijzing |
| --- | --- | --- | --- |
| 1. Inname | upload, URL, mapbewaker, agent-API, Zotero, OKF-import | aanwezig, maar als ingebouwde inname (zie 3.13) | `apps/app-main/src/app_main/api/routers/sources_upload.py`, `services/agents/file_watcher.py`, `api/routers/agents.py:220-386` |
| 2. Registratie met inhoudshash | geen hash bij inname; `TrackedFile.file_hash` bestaat alleen in het model van `packages/file-manager`, dat niet in de Compose-omgeving draait | ontbreekt | `packages/shared/src/shared/models/file_tracking.py:30`; `docker-compose.yml` |
| 3. Poortwachter | geen | ontbreekt | 3.6 |
| 4. Extractie | Docling, MinerU, markitdown, Playwright of httpx voor URL's, WhisperX | aanwezig, andere componenten dan het FO | `apps/app-main/src/app_main/services/source_extractor.py:6`, `:85-94`, `:164-191` |
| 5. Selectie | geen selectie op kop of sectieclassificatie | ontbreekt | zoeken op selectiebouwsteen levert niets op |
| 6. Fragmentatie | structuurgebonden, met sectiepad, pagina en elementtype; documentstructuur als graaf (`doc_node`) | aanwezig | `pipelines/ingestion/src/ingestion/chunking/extractor.py`, `regrouper.py`; `migrations/10.surrealql:5-38`, `27.surrealql:8-11`, `49.surrealql:36-60` |
| 7. Labeling | ontologiegestuurde extractie met typen | gedeeltelijk (3.5) | `pipelines/ontology-extraction/` |
| 8. Deduplicatie | meerlaags | gedeeltelijk (3.5) | `pipelines/entity-filtering/` |
| 9. Opslag als nieuwe versie | overschrijft fragmenten | conflicteert (3.7) | `apps/app-main/src/app_main/services/source_processor.py:117` |
| 10. Ontsluiting | zoeken, exports, chat, ask | gedeeltelijk (3.9) | `apps/app-main/src/app_main/api/routers/search.py:19`, `exports.py` |

- `pipelines/embeddings`: berekent embeddings per fragment en het gemiddelde per bron (`pipelines/embeddings/src/embeddings/aggregate.py`; `ARCHITECTURE.md` paragraaf 14). Bruikbaar achter een importeindpunt. Embeddings erven geen classificatie (`migrations/1.surrealql:16-20`, `27.surrealql:16`).
- `pipelines/retrieval` en `SearchRepository`: tekst-, vector- en hybride zoeken, met herkomst per fragment (`packages/surrealdb-service/src/surrealdb_service/repositories/search.py:69`, `:202`, `:257`, `:314`). Geen filter op classificatie of nieuwste versie (zie 3.7 en 3.8).
- De keten in de fork is vast in code gedefinieerd (`SOURCE_PIPELINE`, `apps/app-main/src/app_main/services/source_pipeline.py:109-170`), niet als definitiebestand dat een editor bewerkt (FO 2 principe 2).

### 3.5 Entiteitextractie en deduplicatie (FO 7, 11)

**`pipelines/ontology-extraction`: gedeeltelijk.**

- Tweestapsextractie met schemadetectie en meerdere ontologieën (`pipelines/ontology-extraction/src/ontology_extraction/multi_schema_orchestrator.py`, `pass1_schema_validation.py`, `pass2_typed_extraction.py`), kandidaat-ankers en onthouding (Track N: `candidates.py`, `not_a_concept.py`; `docs/tracks/N-evidence-first-extraction/status.md`).
- Validatie tegen referentielijsten: TOOI (Thesauri en Ontologieën voor Overheidsinformatie) en CBS (Centraal Bureau voor de Statistiek) worden opgehaald en gecachet (`services/extraction/vocabulary_manager.py:1-19`; `packages/shared/src/shared/vocabulary/tooi_provider.py`; tabel `reference_entity`, `packages/surrealdb-service/src/surrealdb_service/repositories/reference_entity.py:1-5`). BWB (Basiswettenbestand) ontbreekt: zoeken op `bwb` en `basiswettenbestand` in `.py` en `.yaml` levert niets op.
- Herkomst per vermelding: `chunk_id`, grounding en context per bron, plus het extractiemodel (`apps/app-main/src/app_main/services/entity_persistence_service.py:640-675`, Track PC.3).
- Niet als LangGraph-module: LangGraph wordt alleen gebruikt in `apps/app-main/src/app_main/graphs/` (ask, chat, source_chat, transformation, prompt). Zoeken op `StateGraph` in `pipelines/` levert niets op. De lus "per entiteit toetsen en alleen afgekeurde voorstellen opnieuw" (FO 11) is niet als lus met stopconditie gebouwd.
- Extractie bestaat twee keer: in app-main (`EntityExtractionService`) en in de aparte extractiedienst (`services/extraction/api.py:634`), die rechtstreeks naar SurrealDB schrijft (`docker-compose.yml:162-166`).

**`pipelines/entity-filtering`: gedeeltelijk.**

- Lagen voor ruisfilter, normalisatie, fuzzy matching, embedding-deduplicatie, LLM-matching (large language model) en kennisgraafresolutie (`pipelines/entity-filtering/src/entity_filtering/workflow.py:1-30`; `resolution/`). Dat dekt de drie lagen uit FO 7.
- Review bij twijfel: kandidaatvoorstellen en een triagewachtrij bestaan (`apps/app-main/src/app_main/services/entity_resolution/candidate_dedup_service.py`; `migrations/60.surrealql`, tabel `triage_queue`; Track Q). Die wachtrij blokkeert niet; een LangGraph-interrupt die de lus pauzeert en hervat (FO 11) is er niet.
- Handmatige correcties: samenvoegingen komen in `resolution_log` met methode en tijdstip, maar zonder auteur (`apps/app-main/src/app_main/services/entity_resolution/merge_audit.py:1-20`; `migrations/36.surrealql`, `85.surrealql`). `entity.manual_override` beschermt een handmatige status (`migrations/59.surrealql`). Optimistische vergrendeling ontbreekt; `packages/surrealdb-service/src/surrealdb_service/repositories/notebook_schema.py:295` noemt het alleen als ontbrekend.

**Labeling met codeboek: gedeeltelijk.** Ontologieën met beleidsthema's en brede welvaart staan in `packages/ontology-manager/ontologies/` (`policy_themes.yaml`, `regiodeal.yaml`, `deals.yaml`, `instruments.yaml`), met versiebeheer in de database (`migrations/86.surrealql:21-29`, Track OE). Iv3-taakvelden (Informatie voor derden) ontbreken als lijst; `rdf_owl_shacl.py:608` noemt ze alleen als voorbeeld. Een kans per label uit een keuzemodel ontbreekt; entiteiten en relaties hebben wel een `confidence` (`migrations/39.surrealql:23`).

### 3.6 Bronkaart, poortwachter en classificatie (FO 5, 6)

- **Bronkaart: ontbreekt.** Geen tabel, model of API voor bronkaartregels. Het dichtst bij komen `source_folder` (map per bron, `migrations/26.surrealql:8-41`) en de inboxmappen van de mapbewaker, die per map een notebook kiezen (`apps/app-main/src/app_main/services/agents/file_watcher.py`). Velden als schema, route, selectie, labelschema, eigenaar en "inhoud ophalen" komen niet voor.
- **Poortwachter: ontbreekt.** Er is geen stap die route en classificatie bepaalt. De voorbewerking herkent een documentsoort (`apps/app-main/src/app_main/services/preprocessing_service.py:23`), maar beslist niet over route of vertrouwelijkheid en draait na de extractie.
- **Classificatie: conflicteert.** Wat er is:
  - `source.private`, een booleaanse vlag met standaard `false` (`migrations/52.surrealql:30`);
  - `notebook.privacy_mode` (`migrations/52.surrealql:32`);
  - een globale standaard `default_privacy_mode = 'cloud'` (`migrations/52.surrealql:34`, `53.surrealql:19`);
  - het uploadformulier zet `private` standaard op `"false"` (`apps/app-main/src/app_main/api/routers/sources_upload.py:199`);
  - de extractiedienst gebruikt een eigen indeling `public`/`internal`/`confidential` met standaard `internal` (`packages/shared/src/shared/model_routing.yaml:16`).

  Het FO eist "onbekend is intern" en overerving van bron naar versie, fragment en embedding (FO 9). De huidige standaard is het omgekeerde (extern), er zijn twee woordenlijsten, fragmenten en embeddings hebben geen classificatieveld (`migrations/10.surrealql`, `27.surrealql`), en de classificatie werkt alleen op de modelkeuze, niet op zoeken of lezen.
- **Herclassificatie alleen door de beheerder, met logregel (FO 14):** **ontbreekt.** `SourceUpdate` bevat geen `private`-veld (`apps/app-main/src/app_main/api/schemas.py:610-614`); wijzigen na aanmaak via de API is niet gevonden, en er is geen logregel.

### 3.7 Datamodel (FO 8; controle 5)

Vergelijking van de negen objecttypen uit het datamodel (diagram `diagrammen/08-datamodel.png`) met `migrations/` en `packages/surrealdb-service`.

| Objecttype FO | Fork | Status | Verwijzing |
| --- | --- | --- | --- |
| Bron (type, locatie, eigenaar, classificatie, route, schema) | `source` met titel, tekst, `private`, `processing_stage`, metadata | gedeeltelijk | `migrations/1.surrealql:2`, `43`, `52:30`, `71.surrealql` |
| Bronversie (inhoudshash, ophaalmoment, origineel, opvolger) | geen | ontbreekt | zoeken op `content_hash`, `file_hash`, `source_version` in `migrations/` levert niets op |
| Pipelinedefinitie (versie onder Git) | `SOURCE_PIPELINE` in code; `pipeline_cache` met `version`, `is_current` en `config_hash` per stap, als cache van tussenuitvoer | gedeeltelijk | `apps/app-main/src/app_main/services/source_pipeline.py:109`; `migrations/26.surrealql:46-64` |
| Verwerkingsrun (tijdstip, status, modellen, logboek) | `job` (status, fout, pogingen); `audit_findings.run_id` alleen voor de kwaliteitsaudit | gedeeltelijk | `migrations/28.surrealql:7-17`; `75.surrealql:24` |
| Model (naam, versie, lokaal of cloud, mag intern) | providerregister met `is_local`; modelrijen in de database | gedeeltelijk | `packages/llm-manager/src/llm_manager/providers/registry.py:24` |
| Fragment (tekst, sectiepad, pagina, embedding, classificatie) | `chunk` met tekst, volgorde, fysieke en gedrukte pagina, hoofdstuk, sectiepad, elementtype, posities; embedding in `source_embedding` met verwijzing naar `chunk` | gedeeltelijk: geen classificatie, geen run, geen bronversie | `migrations/10.surrealql:5-38`; `27.surrealql:8-22`; `38.surrealql:2` |
| Entiteit (canonieke naam, aliassen, sleutel, relaties) | `entity`, `entity_alias`, `relation`, `reference_entity`; externe id's als `external_ids` | aanwezig | `migrations/39.surrealql:15-82`; `packages/surrealdb-service/src/surrealdb_service/repositories/reference_entity.py` |
| Codeboekterm (methode en zekerheid) | typen in ontologieën; `type_tags`, `primary_type`, `confidence` op entiteit | gedeeltelijk | `migrations/44.surrealql`; `86.surrealql` |
| Kennisproduct (afgeleid, niet terug in retrieval) | `note`, `source_insight`, samenvattingen; geen afgeleid-markering | conflicteert | `migrations/1.surrealql:22-42`; `9.surrealql:21-46` |

Aanvullend:

- **Herkomst volgens PROV-O:** **ontbreekt** als model. `provenance_chain` op `entity` en `relation` is een vrij array zonder schema (`migrations/39.surrealql:24`, `:79`). Relaties houden `relation_sources` bij (`ARCHITECTURE.md` paragraaf 6, Track N.5c).
- **Fragment naar documentstructuur:** `derived_from` koppelt een fragment aan Docling-knopen (`migrations/49.surrealql:60`). Dat is herkomst binnen het document, geen herkomst naar run of model.
- **Herverwerking: conflicteert.** `SourceProcessor` verwijdert alle fragmenten van een bron en maakt ze opnieuw aan (`apps/app-main/src/app_main/services/source_processor.py:114-123`; `packages/surrealdb-service/src/surrealdb_service/repositories/source.py:1158-1176`). FO 2 principe 6 en FO 8 eisen een nieuwe versie die niets overschrijft.
- **Handmatige fragmentbewerking:** fragmenten kunnen worden samengevoegd en gesplitst (`apps/app-main/src/app_main/services/chunking/chunk_mutator.py`; `api/schemas.py:23-35`). Dat staat op gespannen voet met "fragmenten horen bij één bronversie" en herhaalbaarheid. Zie open vraag 11.
- **Migratieconventie:** recente migraties zijn idempotent (`IF NOT EXISTS` of `OVERWRITE`) met een `_down`-variant (`migrations/86.surrealql`, `86_down.surrealql`). Oudere, zoals `migrations/10.surrealql:5`, zijn dat niet. De hoogste migratie is 87 (`migrations/87.surrealql`).

### 3.8 Toegangsbeveiliging en classificatiefilter (FO 9, 10; controle 2)

**Huidige opbouw (feiten).**

- Een middleware met één gedeeld wachtwoord beschermt alle routes onder `/api`, maar alleen als `OPEN_NOTEBOOK_PASSWORD` is gezet; zonder wachtwoord laat hij alles door (`apps/app-main/src/app_main/api/auth.py:28-29`). In het lokale `docker.env` staat deze variabele niet (gecontroleerd op aanwezigheid van de sleutel, niet op waarden; het bestand valt onder `.gitignore`).
- Uitgezonderd van het wachtwoord zijn onder meer `/api/config`, `/api/auth/status`, alle PDF-routes `/api/sources/*/pdf` en het hele voorvoegsel `/api/v1/agents/` (`apps/app-main/src/app_main/api/app.py:278-292`; `auth.py:53-57`).
- `require_agent_key` is een dependency die een `X-API-Key` controleert met rechten `read` < `write` < `admin`; zonder geldige sleutel volgt altijd 401 (`apps/app-main/src/app_main/api/agent_auth.py:46-103`; tabel `agent_keys`, `migrations/77.surrealql:20-30`). Alleen `apps/app-main/src/app_main/api/routers/agents.py` gebruikt deze dependency (`:164`, `:200`, `:226`, `:336`, `:366`, `:393`, `:429`). De agentroutes zijn extract-entities, generate-summary, process-url, process-document, process-audio, jobs en audit-log; zoeken of lezen van fragmenten zit er niet bij.
- Geen enkele andere router onder `/api` (42 routers in `apps/app-main/src/app_main/api/app.py`) heeft een eigen auth-dependency. Dat geldt dus ook voor `notebooks`, `sources`, `notes`, `search`, `entity_resolution` en `knowledge_graph` (`app.py:350`, `:355`, `:363`, `:370`). Dit sluit aan bij de bevinding in `docs/tracks/PC-pipeline-coherence/plan.md:392-395`.
- De extractiedienst heeft geen authenticatie. `POST /extract` neemt een willekeurig `file_path`, leest dat en schrijft ernaast (`services/extraction/api.py:569`, `:634-641`, `:736`). Poort 8103 is op de host gepubliceerd (`docker-compose.yml:147-148`), en app-main stuurt verzoeken door via `services_proxy` (`apps/app-main/src/app_main/api/routers/services_proxy.py:134-138`). PC.7, dat dit moet afbakenen, staat op "NOT STARTED" (`docs/tracks/PC-pipeline-coherence/plan.md:380-399`).
- API-sleutels van cloudaanbieders komen uit omgevingsvariabelen (`packages/llm-manager/src/llm_manager/providers/registry.py`, velden `api_key_env_var`). Versleuteling ontbreekt: zoeken op `encrypt`, `fernet` en `cryptography` in `apps/` en `packages/` levert niets op. FO 13 gaat uit van "toegangsbeveiliging en versleutelde sleutels: aanwezig"; voor deze fork klopt dat niet (aanname A4 over upstream).

**Tegenover het FO.**

- *Sleutel per gebruiker en agent met rechten "lezen extern, lezen intern, beheren" (FO 9):* **gedeeltelijk.** Het sleutelmechanisme bestaat, maar de rechten gaan over schrijven en niet over classificatie, en het geldt alleen voor de agent-API.
- *Classificatiefilter in de API (FO 9, 10, 13.2 punt 3; harde eis 1):* **ontbreekt.** `fn::vector_search` en `fn::text_search` filteren niet op `private` (laatste definities: `migrations/9.surrealql:4-57` en `4.surrealql:5`); `SearchRepository` evenmin (`packages/surrealdb-service/src/surrealdb_service/repositories/search.py`, geen enkele vermelding van `private`). Het filter moet op elke leesroute gelden, ook op `/api/sources`, `/api/notes`, de exports en de kennisgraafroutes.
- *Gevolg voor harde eis 1:* zolang het wachtwoord leeg kan zijn en de extractiedienst en SurrealDB rechtstreeks bereikbaar zijn (`docker-compose.yml:10-11`, `:147-148`), kan een filter in app-main worden omzeild. Het netwerk is daarom de eerste verdedigingslijn (FO 9, drie plekken).

### 3.9 Kennisontsluiting: MCP-server, wiki en Obsidian (FO 10; controle 3)

**MCP-server: conflicteert.**

- De fork heeft zelf twee MCP-servers, geen open-notebook-mcp:
  - `surrealdb-mcp` (Track W): `packages/surrealdb-service/src/surrealdb_service/mcp/server.py`. FastMCP, standaard via stdio, optioneel via SSE of streamable HTTP (`:714-729`). Tools: `query_database` (willekeurige SurrealQL, `:77-83`), `get_record`, `list_sources`, `search_similar`, `search`, `get_node`, `related`, `cite`, `add_note`, `auto_link_note`, `get_entity_graph`, `export_okf`, `import_okf`. Geen authenticatie (`:11-13`), en schrijftools zijn altijd geregistreerd (`:15-17`).
  - Een MCP-server in `packages/llm-manager/src/llm_manager/mcp/server.py` voor modelbeheer.
- `surrealdb-mcp` praat rechtstreeks met de database ("repo-direct", `ARCHITECTURE.md` paragraaf 10), niet via de REST-API. Een classificatiefilter in de API zou deze server niet raken; met `query_database` is elk record leesbaar.
- open-notebook-mcp zit niet in de repository: zoeken op `open-notebook-mcp` en `open_notebook_mcp` levert alleen de FO-documenten en een agentdefinitie op. Het projectbestand `.mcp.json` registreert alleen context7. Welke API-paden en welke authenticatie open-notebook-mcp verwacht, is daarom niet in code te controleren (aanname A2).
- FO 14 besluit: open-notebook-mcp, met het filter in de API van de fork. Dat besluit staat; het conflict zit in de bestaande `surrealdb-mcp`, die het filter omzeilt.

**Afgeleide teksten (harde eis 3): conflicteert.**

- `fn::vector_search` zoekt altijd ook in `source_insight` (door een model gegenereerde inzichten) en, met `show_notes`, in `note` (`migrations/9.surrealql:21-29`, `:38-46`). `SearchRepository.hybrid_search` heeft `include_notes=True` als standaard (`packages/surrealdb-service/src/surrealdb_service/repositories/search.py:320`).
- De MCP-tool `add_note` schrijft notities die daarna via `search` terugkomen (`packages/surrealdb-service/src/surrealdb_service/mcp/server.py:337`; `ARCHITECTURE.md` paragraaf 10).

**Wiki-generatie: ontbreekt.** Geen code; zoeken op `wiki` in `apps/`, `packages/` en `pipelines/` levert alleen verwijzingen in exports en woordenlijsten op. De citaatcontrole uit Track X toetst lidmaatschap, niet of een fragment de bewering draagt (`apps/app-main/src/app_main/graphs/citations.py`; `ARCHITECTURE.md` paragraaf 11, "Limitation"). Dat is een bruikbare eerste controle, niet de tweede controle uit FO 10.

**Export naar Obsidian: gedeeltelijk.**

- Aanwezig: één Markdown-bestand per entiteit met frontmatter en `id` (`apps/app-main/src/app_main/services/obsidian_export_service.py:1001-1005`), atomair schrijven (`:811`), padcontrole (`:766-772`), en een leeszijde die aliassen uit de vault terugleest (`apps/app-main/src/app_main/services/vault_sync_service.py:1-8`).
- Wijkt af: elk bestand wordt volledig vervangen met `os.replace` (`obsidian_export_service.py:811`), dus handmatige tekst blijft niet staan; er is geen gegenereerd blok. De standaardmap is één map `Entities` (`apps/app-main/src/app_main/api/routers/vault.py:48`), geen indeling naar personen, organisaties en gebieden. Classificatie en fragment-id's staan niet in de frontmatter. Het FO plaatst de export als Langflow-flow met eigen sjabloon; in de fork is het een app-main-service.

### 3.10 Beheeromgeving en orkestratie (FO 11)

- **Langflow: ontbreekt.** Geen code, image of configuratie (zoeken op `langflow` in `.py`, `.yml`, `.yaml` en `.toml` levert niets op).
- **Pipelinedefinitie als bestand onder versiebeheer:** **ontbreekt.** De keten staat in Python (`apps/app-main/src/app_main/services/source_pipeline.py:109`).
- **LangGraph en checkpoints:** **aanwezig** voor de bestaande grafen. Checkpoints staan in een lokaal SQLite-bestand (`apps/app-main/src/app_main/config.py:11-13`; `graphs/chat.py:44-50`). LangGraph is al een afhankelijkheid (`apps/app-main/pyproject.toml:21-22`). Dit past bij FO 11.
- **Runoverzicht:** **gedeeltelijk**, in de interface van de fork (Track WB, `docs/tracks/WB-pipeline-workbench/plan.md`; router `apps/app-main/src/app_main/api/routers/attention.py`), niet in Langflow.

### 3.11 Uitrol met Docker en GPU-beheer (FO 12)

- **Compose:** **gedeeltelijk.** Eén `docker-compose.yml` met surrealdb, grobid, docling, summarization, whisperx, extraction, mineru, reranker en open_notebook (`docker-compose.yml:6-277`). Langflow, llama-swap, Kev, docling-serve, open-notebook-mcp en claude-cli ontbreken.
- **Netwerk:** **conflicteert** met FO 12.1. Er is één bridge-netwerk `web` (`docker-compose.yml:1-4`), geen netwerk met `internal: true`. Alle diensten publiceren hun poort op de host, ook SurrealDB (`:10-11`) en de extractiedienst (`:147-148`). De diensten bereiken Ollama op de host via `host.docker.internal` (`:84`, `:87`, `:154`, `:246`).
- **Geheimen:** **conflicteert.** De SurrealDB-inloggegevens staan in platte tekst in het Compose-bestand (`docker-compose.yml:24`, `:165-166`). Tokens en sleutels lopen via omgevingsvariabelen (`:140`, `:172`); er is geen `secrets:`-blok.
- **Vaste imageversies:** **gedeeltelijk.** `surrealdb/surrealdb:v2` met `pull_policy: always` (`docker-compose.yml:8`, `:25`) is geen vaste versie; GROBID is wel vastgezet (`:36`). De overige diensten worden lokaal gebouwd.
- **GPU-beheer:** **conflicteert.** docling, whisperx en mineru reserveren elk de GPU (`docker-compose.yml:55-61`, `:123-129`, `:192-198`), en Ollama op de host gebruikt dezelfde kaart (aanname A5). Er is geen coördinatie of wachtrij voor de GPU; wel staat in `docker-compose.yml:73-77` een bekend geheugenprobleem (GPU-geheugenfout bij grote gescande PDF's). De worker van de takenwachtrij draait standaard één taak tegelijk (`packages/job-queue/src/job_queue/worker.py:53`; `ARCHITECTURE.md` paragraaf 14), wat in de praktijk gelijktijdige GPU-taken vanuit app-main beperkt, maar niet vanuit de extractiedienst of de samenvattingsdienst.
- **Offline-instellingen:** **ontbreekt.** `HF_HUB_OFFLINE` en `DO_NOT_TRACK` komen niet voor in `docker-compose.yml`.

### 3.12 Onderdelen die volgens het FO niet meer worden gebruikt (FO 13.2)

| Onderdeel | Locatie | Status tegenover FO 13.2 punt 7 |
| --- | --- | --- |
| Chat en bronchat | `apps/app-main/src/app_main/graphs/chat.py`, `source_chat.py`; routers `chat.py`, `source_chat.py` | aanwezig; kan blijven staan |
| Ask | `apps/app-main/src/app_main/graphs/ask.py`; `api/routers/search.py:157`, `:213` | aanwezig; kan blijven staan |
| Podcasts (generatie) | `apps/app-main/src/app_main/api/routers/podcasts.py`, `services/podcast_service.py` | aanwezig; kan blijven staan |
| Transformaties en inzichten | `graphs/transformation.py`; `services/insight_service.py` | aanwezig; de inzichten komen wel in zoekresultaten (3.9), dat hindert |
| Interface | `frontend/` (Next.js 15, `frontend/package.json:55`) | aanwezig en recent uitgebreid (Tracks UX en WB, afgerond 30 september 2026 volgens `docs/tracks/WB-pipeline-workbench/plan.md:3`) |

"Laten staan zolang ze niet hinderen" is uitvoerbaar voor chat, ask en podcasts. Twee onderdelen hinderen wel: de inzichten in de zoekindex (harde eis 3) en de chat, die via Track J standaard een cloudmodel kiest (`route_resolver.py:84-85`) en met de huidige standaard "niet privé" interne inhoud naar de cloud kan sturen. De interface toont daarnaast alle fragmenten zonder filter.

### 3.13 Ingebouwde bronverwerking tegenover recente tracks (FO 13.2 punt 1; controle 1)

FO 14 besluit: "eigen pipeline is de enige inname". FO 13.2 punt 1: "de ingebouwde bronverwerking vervalt". In de fork is de ingebouwde bronverwerking niet de LangGraph-graaf uit het oorspronkelijke project (er is geen `graphs/source.py` in `apps/app-main/src/app_main/graphs/`), maar een eigen keten die de afgelopen maanden is uitgebouwd. Per track:

| Track | Wat het levert | Herbruikbaar achter een importeindpunt | Vervalt bij inname via Langflow |
| --- | --- | --- | --- |
| PL, bronpipeline (`ARCHITECTURE.md` paragraaf 14; `apps/app-main/src/app_main/services/source_pipeline.py`) | keten parse, embed, extract, graph, insights met `processing_stage` en `advance_source` | de stappen na de extractie: EMBED (fragment-embeddings en bronvector, `pipelines/embeddings/src/embeddings/aggregate.py`), EXTRACT, GRAPH; `processing_stage` als status per bron | de stap INGEST (`DOCUMENT_PARSE`) en het automatisch starten ervan bij upload; INSIGHTS levert afgeleide tekst en botst met harde eis 3 |
| PC, pipelinesamenhang (`docs/tracks/PC-pipeline-coherence/plan.md`) | curatorwachtrij, één identiteit, kijken naar de bestaande graaf, herkomst per vermelding, configuratie die één intentie uitdrukt | vrijwel alles: het werkt op extractie en resolutie, niet op inname; PC.3-herkomst (`entity_persistence_service.py:640-675`) is een basis voor herkomst | niets; PC.7 (grenzen van de extractiedienst) is juist voorwaarde voor harde eis 1 |
| N, evidence-first extractie (`docs/tracks/N-evidence-first-extraction/status.md`) | kandidaat-ankers, onthouding, niet-een-begrip-filter, regressiepoort | ja, als onderdeel van de entiteitextractie; de regressiepoort (`scripts/n_extraction_gate.py`, volgens `ARCHITECTURE.md` paragraaf 6) past bij een stopconditie buiten het model | niets |
| OE, ontologie-editor (`docs/tracks/OE-ontology-editor/plan.md`) | ontologieën verliesvrij opslaan met versies in `ontology_version`, concepten, publiceren | de opslag met versies als bron voor het labelschema of codeboek (`migrations/86.surrealql`, `87.surrealql`) | het bewerken via de interface valt onder "interface niet gebruikt" (vraag 14) |
| Q, triage (`docs/tracks/Q-triage/plan.md`) | na extractie matchen, samenvoegen, status `active`/`reference`, reviewwachtrij | ja, als reviewwachtrij voor twijfelgevallen bij deduplicatie (`migrations/60.surrealql`) | niets; wel: de wachtrij blokkeert niet, terwijl FO 11 een pauzerende lus beschrijft |
| WB, werkbak (`docs/tracks/WB-pipeline-workbench/plan.md`) | interface als werkbak voor de pijplijn: aandachtsoverzicht, stages, wachtrijen | het aandachtsoverzicht (`api/routers/attention.py`) kan de wachtrij van de beheerder tonen | valt onder "interface niet gebruikt"; botst met de richting van een recent afgeronde track (conflict 13) |
| G, agent-API en mapbewaker | `process-document`, `process-url`, `process-audio` met sleutelauthenticatie; inboxbewaker | het sleutelmechanisme als basis voor de rechten in FO 9; de router als plek voor een importeindpunt (`api/routers/agents.py:157`) | de `process-*`-routes en de inboxbewaker zijn ingebouwde inname |
| A en H, MinerU en parsingroutering | tweede parser, terugval op lage zekerheid | eventueel als Docker-container-bouwsteen in Langflow | de routering in app-main (`services/parsing/`) |
| V, verwijzingen (GROBID) | literatuurverwijzingen en `cites` | ja, als nabewerking op een geïmporteerde bron | niets |

Samengevat: wat vóór de fragmenten zit (parsing, fragmentatie, inname-routes) vervalt of verhuist naar Langflow-componenten. De code in `pipelines/ingestion` (Docling-parser, fragmentatie) is als Python-code herbruikbaar binnen een Langflow-component. Wat ná de fragmenten zit (embeddings, extractie, resolutie, triage, documentgraaf) blijft bruikbaar, mits het start vanaf een importeindpunt in plaats van vanaf `DOCUMENT_PARSE`. Er is nu geen eindpunt dat kant-en-klare fragmenten aanneemt: `api/schemas.py` kent alleen verzoeken voor samenvoegen en splitsen van fragmenten (`:23`, `:35`), en de OKF-import neemt entiteiten, relaties, notities en bronnen aan, geen fragmenten (`ARCHITECTURE.md` paragraaf 7).

## 4. Aannames

Deze punten zijn niet in de code van de fork te controleren.

- **A1.** llama-swap biedt een OpenAI-compatibel eindpunt en geen Ollama-eigen eindpunten als `/api/generate`, `/api/chat` en `/api/embed`. Bron: FO 12.2 ("één OpenAI-compatibel eindpunt"); niet getest.
- **A2.** open-notebook-mcp spreekt de REST-paden van het oorspronkelijke Open Notebook aan (zoals `/api/notebooks`, `/api/sources`, `/api/notes`, `/api/search`) en authenticeert met het gedeelde wachtwoord. De fork heeft routers met deze namen (`apps/app-main/src/app_main/api/app.py:350-370`), maar of vorm en antwoorden overeenkomen met wat open-notebook-mcp verwacht, is niet nagegaan.
- **A3.** docling-serve heeft geen ingebouwde equivalent van de PyMuPDF-aanvulling, de annotatie-extractie en de handschrift-VLM van de eigen dienst.
- **A4.** Het oorspronkelijke Open Notebook heeft versleutelde opslag van API-sleutels (FO 13 noemt dit "aanwezig"); de fork heeft die niet overgenomen.
- **A5.** Ollama draait op de host op dezelfde GPU als de containers. Dat volgt uit `host.docker.internal:11434` in `docker-compose.yml`, maar de host zelf is niet gecontroleerd.
- **A6.** De productiedatabase wijkt mogelijk af van de migraties: `entity_persistence_service.py:672-674` meldt dat `entity` in productie schemaloos is, terwijl `migrations/39.surrealql` de tabel strikt definieert. Deze analyse gaat uit van de migraties.
- **A7.** Kev heeft een eigen server met een eigen API (FO 6). Hoe llama-swap die aanroept, is niet bekend; dat is proef 1.

## 5. Conflicten

1. **Lokale modelserver.** FO 14 besluit llama-swap met llama-server; de fork draait alle lokale modellen op Ollama op de host, buiten Docker (`docker-compose.yml:84`, `:103`, `:108`, `:154`, `:246-247`). Dat botst ook met harde eis 4.
2. **"Modelkoppeling is configuratie, geen code" (FO 13.2 punt 5).** Meerdere modules roepen Ollama-eigen eindpunten rechtstreeks aan (`packages/shared/src/shared/model_routing.py:284`, `:353`; `pipelines/entity-filtering/src/entity_filtering/resolution/llm_matcher.py:349`; `packages/semantic-intelligence/src/semantic_intelligence/config.py:64`). Omzetten naar llama-swap vraagt codewijzigingen (onder aanname A1).
3. **Standaardclassificatie.** FO 9 eist "onbekend is intern"; de fork zet bronnen standaard op niet-privé en de globale standaard op `cloud` (`migrations/52.surrealql:30`, `:34`; `53.surrealql:19`; `sources_upload.py:199`).
4. **Twee classificatie-indelingen en het verkeerde niveau.** Track J gebruikt `cloud`/`private` per document en notebook (`route_resolver.py:53-61`), de extractiedienst `public`/`internal`/`confidential` (`model_routing.yaml:16`). Het FO kent één classificatie intern/extern op de bron, met overerving naar versie, fragment, embedding en kennisproduct. Fragmenten en embeddings hebben geen classificatieveld.
5. **MCP-server.** FO 14 besluit open-notebook-mcp via de REST-API. De fork heeft `surrealdb-mcp` met directe databasetoegang, een tool voor willekeurige queries en schrijftools, zonder authenticatie (`packages/surrealdb-service/src/surrealdb_service/mcp/server.py:11-17`, `:77-83`). Die server omzeilt elk filter in de API.
6. **Toegangsbeveiliging als basis.** FO 9 en 13 gaan uit van bestaande toegangsbeveiliging met versleutelde sleutels. In de fork laat het gedeelde wachtwoord zonder instelling alles door (`auth.py:28-29`), hebben de routers onder `/api` geen eigen auth-dependency, is de extractiedienst onbeveiligd met willekeurige padtoegang (`services/extraction/api.py:634-641`; PC.7 niet gestart), en worden sleutels niet versleuteld opgeslagen.
7. **Herverwerking overschrijft.** FO 2 principe 6 en FO 8 eisen een nieuwe versie zonder overschrijven; `SourceProcessor` verwijdert en vervangt alle fragmenten (`source_processor.py:117`).
8. **Afgeleide teksten in zoekresultaten.** Harde eis 3 en FO 13.2 punt 4; de zoekfuncties nemen `source_insight` altijd en notities standaard mee (`migrations/9.surrealql:21-46`; `search.py:320`), en de MCP-tool `add_note` voegt zoekbare notities toe.
9. **Meerdere actieve innameroutes.** FO 14 besluit "eigen pipeline is de enige inname". De fork heeft upload, URL, mapbewaker, agent-API, Zotero en OKF-import, met een automatische keten tot en met extractie (Track PL). Laten staan volgens FO 13.2 punt 7 betekent een tweede inname zonder bronkaart en poortwachter, en daarmee zonder classificatie.
10. **Documentextractie.** FO 13.1 vervangt de eigen Docling-worker door docling-serve. De eigen dienst is synchroon en padgebaseerd, houdt het model geladen en heeft eigen verrijkingen (`services/docling/api.py:31-34`, `:118-125`) die bij vervanging vervallen.
11. **GPU-beheer.** FO 12.2 laat alle GPU-diensten via llama-swap lopen. In de fork reserveren docling, whisperx en mineru elk de GPU en gebruikt Ollama op de host dezelfde kaart, zonder coördinatie (`docker-compose.yml:55-61`, `:123-129`, `:192-198`).
12. **Netwerk en geheimen.** FO 12.1 vraagt een netwerk zonder internet, alleen editor en MCP-server bereikbaar, en geheimen via Docker secrets. De fork heeft één netwerk, publiceert alle poorten (ook SurrealDB) en heeft de database-inloggegevens in platte tekst in `docker-compose.yml:24` en `:165-166`.
13. **Interface en beheeromgeving.** FO 13.2 zegt dat de interface niet wordt gebruikt en FO 11 maakt Langflow de beheeromgeving. Track WB, afgerond op 30 september 2026, heeft de interface juist ingericht als werkbak voor de pijplijn, met de reviewwachtrijen (`docs/tracks/WB-pipeline-workbench/plan.md:3-20`).
14. **Obsidian-export.** FO 10 eist een idempotente export die alleen in een gegenereerd blok schrijft, met classificatie en fragment-id's in de frontmatter en een vaste mappenindeling. De fork vervangt het hele bestand (`obsidian_export_service.py:811`), schrijft naar één map (`vault.py:48`) en neemt classificatie niet op.

## 6. Open vragen

1. Welke authenticatie gebruikt open-notebook-mcp richting de API, en sluiten de API-paden en antwoorden van de fork aan op wat het verwacht (aanname A2)? Moet het classificatiefilter afhangen van de sleutel waarmee een client zich meldt, en krijgt open-notebook-mcp een sleutel met alleen "lezen extern"?
2. Wat gebeurt er met `surrealdb-mcp` (Track W): uitzetten voor het signaalsysteem, beperken tot lokaal gebruik met intern-rechten, of het tool `query_database` en de schrijftools verwijderen? Zelfde vraag voor de MCP-server in `packages/llm-manager`.
3. Blijven de bestaande innameroutes (upload, mapbewaker, agent-API `process-*`, Zotero, OKF-import) actief? Zo ja: hoe krijgen ze een bronkaartregel en een classificatie; zo nee: uitzetten via configuratie of verwijderen, gezien FO 13.2 punt 7?
4. Docling: komen de eigen verrijkingen (PyMuPDF-tekst, annotaties, handschrift-VLM) terug als aparte bouwsteen naast docling-serve, of vervallen ze? Horen MinerU, GROBID, de reranker, de samenvattingsdienst en de extractiedienst in `docker-compose.signaalsysteem.yml`?
5. Blijft `packages/job-queue` de uitvoerder van de stappen binnen de fork (embedding, extractie, graaf na een import), of start Langflow elke stap? Wordt een verwerkingsrun een nieuwe tabel, of een uitbreiding van `job`?
6. Bestaande classificatievelden (`source.private`, `notebook.privacy_mode`, `default_privacy_mode`, de niveaus in `model_routing.yaml`): vervangen door één veld intern/extern, of migreren? Wat gebeurt er met de bestaande gegevens in de lokale database: weggooien (FO 14: "nog niet in productie") of migreren met standaard intern?
7. Wordt de Track J-routering (resolver, failover, circuit breaker) de basis van de modelpoort, of ligt de modelkeuze alleen in de Langflow-componenten (FO 9)? Waar staat het modelregister met "toegestaan voor intern": in de bestaande modelrijen, in `model_route` of in een nieuwe tabel?
8. Welk embeddingmodel draait onder llama-swap, en blijft de vectorruimte gelijk (`mxbai-embed-large`, 1024 dimensies volgens `packages/shared/src/shared/models/source.py:130`)? Zo niet, dan moet de hele index opnieuw. `ARCHITECTURE.md` noemt in paragraaf 8 en 12a twee verschillende modellen.
9. Worden de bestaande entiteitextractie (`EntityExtractionService`) en deduplicatie (`FilteringWorkflow`) in een LangGraph-lus verpakt (FO 11, 13.2 punt 6), of blijven ze zoals ze zijn en komt alleen de pauzerende review erbij? Welke van de twee extractiepaden, app-main of `services/extraction`, is leidend?
10. Waar komen de referentielijsten voor Iv3-taakvelden en BWB-id's vandaan, en in welke vorm: als ontologie met versies (Track OE, `ontology_version`) of als woordenlijst naast TOOI en CBS (`services/extraction/vocabulary_manager.py`)?
11. Hoe verhouden handmatig samengevoegde of gesplitste fragmenten (`chunk_mutator.py`) zich tot "fragmenten horen bij één bronversie" en herhaalbaarheid? Vervalt die functie, of wordt een handmatige wijziging een eigen versie met auteur?
12. Hoe roept llama-swap Kev aan, en past Kev met docling-serve en WhisperX in de groepsindeling (aanname A7; proeven 1 en 3)?
13. Wordt de bestaande `metrics`-tabel het auditlogboek van modelaanroepen (uitgebreid met run en classificatie), of komt er een aparte tabel? Routes buiten Track J schrijven nu geen auditregel.
14. Waar handelt de beheerder zijn wachtrijen af (bron zonder bronkaartregel, twijfel van de poortwachter, deduplicatie, verouderde wiki-pagina's): in Langflow, in de werkbak van Track WB of elders? Het FO noemt de reviewwachtrij, maar niet waar die wordt bediend.
15. Optimistische vergrendeling en auteur bij handmatige wijzigingen (FO 8): uitbreiden van `resolution_log` en `entity.manual_override`, of een algemeen mechanisme per tabel?
