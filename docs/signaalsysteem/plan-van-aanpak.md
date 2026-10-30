# Plan van aanpak signaalsysteem

Stand: 30 oktober 2026 · stap 4 van `opdracht.md` · branch `docs/signaalsysteem-fo`

Dit plan deelt de fasen 2 tot en met 5 uit het functioneel ontwerp (FO, paragraaf 15) op in taken. Het steunt op het FO, de verschilanalyse, de ontwerpbesluiten (Architecture Decision Records, ADR's) 0001–0019 en de proefuitkomsten P0–P9 in `proeven.md`. Er is geen code gewijzigd.

Inhoud: 1 werkwijze · 2 taken per fase · 3 harde eisen naar taken · 4 Compose-opzet · 5 interfaces · 6 risico's · 7 aannames en conflicten · 8 open vragen en besluiten vóór fase 2.

## 1. Werkwijze

- **Agent.** Elke taak bouwt `signaal-bouwer`, één taak tegelijk; daarna toetst `signaal-reviewer` de taak aan FO, ADR's en harde eisen. Een taak is af als de stopconditie slaagt en de reviewer geen blokkerende bevinding heeft. Daarna volgt één regel in `voortgang.md` en een kleine commit.
- **Volgorde.** Fase 2 begint met de taken die de harde eisen borgen (F2-01 tot en met F2-06): afscherming, classificatie en bronversie met run en herkomst. Daarna modelpoort en omgeving, en pas dan de inname.
- **Tests.** Tests staan naast de code (`ARCHITECTURE.md` §5) en draaien met `uv run --project <lid> pytest <lid>/tests/<bestand>`. Op de werkplek van Geert gaat dat via de `.venv` in WSL (Windows Subsystem for Linux). Tests over de hele repository staan in `tests/` in de root, zoals het bestaande `tests/test_compose_env_is_consumed.py`.
- **Migraties.** Elke migratie is idempotent (`DEFINE … OVERWRITE` of `IF NOT EXISTS`) en heeft een `_down`-variant. De hoogste migratie is nu 87 (`migrations/87.surrealql`); de nummers hieronder zijn een verwachting op volgorde van de taken. Elke migratietaak heeft een test `packages/surrealdb-service/tests/test_migration_<n>_<onderwerp>.py` die heen en terug draait, zoals `test_migrations_roundtrip.py`.
- **Verwijdertaken.** Een taak die bestaande code verwijdert, staat apart en heeft altijd de **verwijdercontrole (V)**:
  1. de volledige testsuite van elk geraakt workspace-lid slaagt, na het verwijderen van de tests van de verwijderde code;
  2. `python -c "import app_main.api.app"` slaagt, en de API start in de Compose-omgeving met `/health` HTTP 200;
  3. een zoektest `tests/test_geen_verwijzingen_<onderwerp>.py` faalt zodra een verwijderd module- of routenaam nog voorkomt in `apps/`, `packages/`, `pipelines/`, `services/` of `frontend/src/`;
  4. raakt de taak routes die de interface gebruikt: `npm --prefix frontend run test` en `npm --prefix frontend run build` slagen.
- **Bevestiging.** Een taak die bestaande gegevens weggooit, draagt de markering **vraagt expliciete bevestiging van Geert**. De bouwer start die taak pas na die bevestiging in het gesprek, en maakt vooraf een reservekopie van de database.
- **Testdata.** Alleen openbare bronnen (tabel 1.6 in `proeven.md`) of synthetische tekst. Grote bestanden worden gedownload met controle van de SHA-256 en niet in Git gezet.
- **Werkplek.** Een stopconditie die de GPU (grafische kaart) nodig heeft, is een script onder `scripts/signaalsysteem/controle/` of `scripts/signaalsysteem/toetsen/`, met dezelfde exitcodes als de proeven (0 geslaagd, 1 criterium niet gehaald, 2 omgeving ongeschikt, 3 scriptfout). Zulke taken zijn "uit te voeren op de werkplek".
- **Taak-id.** `F<fase>-<nr>`, zoals de startopdracht `/signaalsysteem taak F2-03` verwacht.

## 2. Taken

### Fase 2: minimale keten

Doel volgens FO 15: PDF en lokale map, docling-serve, fragmentatie, importeindpunt, opslag met versie, herkomst en classificatie, en de export naar Obsidian, vanaf het begin in Docker met llama-swap. *Toets (F2-24):* elk fragment is herleidbaar tot bronversie en run, en twee gelijktijdige GPU-taken leiden niet tot een geheugenfout.

#### A. Harde eisen borgen

**F2-01 Sleutels en rechten per route, standaard weigeren**
- *Wat:* `PasswordAuthMiddleware` en `OPEN_NOTEBOOK_PASSWORD` vervallen. Eén sleutelmiddleware voor alles onder `/api`, met `Authorization: Bearer`; `X-API-Key` vervalt. Rechten `lezen_extern` < `lezen_intern` < `beheren`. Elke route declareert haar minimale recht met `require_right(...)`. `AgentAuditMiddleware` geldt voor alle routes. Uitgezonderd zijn alleen `/health` en `/api/auth/status`. Een migratie zet `agent_keys.permission` op de nieuwe waarden en trekt bestaande sleutels in. Een script maakt de eerste sleutel met `beheren` en schrijft die als geheimbestand weg, zonder log.
- *Hangt af van:* —
- *Bestanden:* `apps/app-main/src/app_main/api/auth.py`, `agent_auth.py`, `app.py`; alle routers in `apps/app-main/src/app_main/api/routers/` (dependency per route); `apps/app-main/src/app_main/services/agents/key_service.py`; `migrations/88.surrealql` en `88_down.surrealql`; nieuw `scripts/signaalsysteem/maak_beheersleutel.py`.
- *ADR:* 0002 A, B.
- *Stopconditie:* `apps/app-main/tests/test_route_rechten.py` loopt alle `application.routes` af en faalt bij een route zonder rechtendeclaratie; `apps/app-main/tests/test_sleutel_verplicht.py`: elk verzoek zonder geldige sleutel geeft HTTP 401, behalve de twee uitzonderingen; `packages/surrealdb-service/tests/test_migration_88_agent_key_permission.py`: `read` wordt geweigerd, bestaande sleutels zijn ingetrokken, `_down` laat ze ingetrokken. Bestaand `test_agent_auth.py` aangepast en groen.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.
- *Let op:* de interface kan tot F2-20 niet inloggen.

**F2-02 Verwijderen: routes en diensten die het filter omzeilen**
- *Wat:* weg gaan de router `services_proxy`, de drie MCP-servers (Model Context Protocol) in de fork en de extractiedienst. Daarmee vervalt PC.7. Vooraf verhuist `services/extraction/vocabulary_manager.py` naar `packages/shared/src/shared/vocabulary/` (ADR 0012). De querylogica achter de MCP-tools blijft bestaan in de repositories (`SearchRepository`, `GraphRepository`; `ARCHITECTURE.md` §10), voor de herkomsteindpunten in F3-09.
- *Hangt af van:* F2-01.
- *Bestanden:* `apps/app-main/src/app_main/api/routers/services_proxy.py` en de registratie in `app.py`; `frontend/src/lib/api/services.ts` (roept `/api/services` aan); `packages/surrealdb-service/src/surrealdb_service/mcp/` met script-ingang in `packages/surrealdb-service/pyproject.toml` en de tests `test_mcp_*.py`; `packages/llm-manager/src/llm_manager/mcp/` en `pyproject.toml`; `packages/file-manager/src/file_manager/mcp/` en `pyproject.toml`; `services/extraction/` en de dienst `extraction` in `docker-compose.yml`; `ARCHITECTURE.md` §1, §6 en §10.
- *ADR:* 0002 D, E; 0012.
- *Stopconditie:* verwijdercontrole V, met `tests/test_geen_verwijzingen_omzeilroutes.py` op `services_proxy`, `surrealdb_service.mcp`, `llm_manager.mcp`, `file_manager.mcp` en `services/extraction`; een test dat `/api/services/...` HTTP 404 geeft; de woordenlijsttests draaien op de nieuwe plek.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-03 Classificatie intern/extern, standaard intern, met herclassificatie**
- *Wat:* veld `classification` (`ASSERT $value IN ['intern','extern']`, `DEFAULT 'intern'`) op `source`, `chunk`, `source_embedding`, `note`, `source_insight` en `notebook`; alle bestaande records worden `intern`. Bij schrijven neemt de repositorylaag de waarde van de bron over; een lossere waarde dan de bovenliggende geeft HTTP 409, een ontbrekende waarde geldt als `intern`. Nieuwe `ClassificationService` met `POST /api/v1/sources/{id}/classification` (recht `beheren`, reden verplicht) en tabel `classification_log` die alleen aanvult. De oude velden `private`, `privacy_mode` en `default_privacy_mode` blijven tot F2-09 staan, zodat de code tussen de taken blijft werken.
- *Hangt af van:* F2-01.
- *Bestanden:* `migrations/89.surrealql` en `_down`; `packages/shared/src/shared/models/source.py`; `apps/app-main/src/app_main/api/schemas.py`; `packages/surrealdb-service/src/surrealdb_service/repositories/source.py`, `notebook.py`; nieuw `apps/app-main/src/app_main/services/classification_service.py`, nieuw `api/routers/classification.py`; `dependencies.py`.
- *ADR:* 0001, 0019.
- *Stopconditie:* `test_migration_89_classification.py`: per tabel het veld met standaard `intern`; de test schrijft een waarde en leest die terug, omdat een SCHEMAFULL-tabel een onbekend veld stil weglaat (P4); een record zonder waarde is `intern`; `_down` zet `private = true` op alle bronnen. `apps/app-main/tests/test_classification_service.py`: import of poortwachter kan intern niet naar extern zetten (409); herclassificatie alleen met `beheren` en reden, en schrijft precies één logregel; er is geen pad om een logregel te wijzigen of te verwijderen.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-04 Bronversie, verwerkingsrun en herkomst**
- *Wat:* nieuwe tabellen `source_version` (met `content_hash`, `original`, `predecessor`, `classification`, `created_by_run`, `current_run`, unieke index op `source` + `content_hash`) en `processing_run` (status `bezig`/`geslaagd`/`mislukt`/`niets_nieuws`, trigger, `pipeline_id`, `pipeline_version`, `models`, `log`, `error`, `attempts`, `notified_at`); `source.current_version`; op `chunk` de velden `source_version`, `run` (verplicht voor nieuwe records), `extractor` en `classification`; op `source_embedding` de velden `run` en `embedding_model`; `run` op `job` en op de entiteitvermeldingen (PC.3). Per bestaande bron één bronversie met `pre_signaalsysteem` en één run met `pipeline_version = 'pre-signaalsysteem'`, gekoppeld aan de bestaande fragmenten. De strengste classificatie wint over alle versies. `SourceVersionRepository`, `ProcessingRunRepository`, `ProcessingRunService` en de run-eindpunten (§5.4). De bouwer inventariseert de lezers van `provenance_chain` (voor F3-08).
- *Hangt af van:* F2-03.
- *Bestanden:* `migrations/90.surrealql` en `_down`; nieuw `packages/shared/src/shared/models/source_version.py`, `processing_run.py`; nieuw `packages/surrealdb-service/src/surrealdb_service/repositories/source_version.py`, `processing_run.py`; `repositories/source.py`; `packages/job-queue/src/job_queue/` (model en repository van `job`); `apps/app-main/src/app_main/services/entity_persistence_service.py`; nieuw `apps/app-main/src/app_main/services/processing_run_service.py`, nieuw `api/routers/runs.py`.
- *ADR:* 0003, 0001.
- *Stopconditie:* `test_migration_90_source_version_run.py`: een nieuw `chunk` zonder `source_version` of `run` wordt geweigerd; de unieke index weigert een dubbele hash; na de migratie heeft elk bestaand fragment een bronversie en een run met `pre-signaalsysteem`; `_down` draait terug. `apps/app-main/tests/test_processing_run_api.py`: een run zonder nieuwe inhoud eindigt als `niets_nieuws`; `PATCH` op een afgesloten run geeft 409. `apps/app-main/tests/test_classificatie_strengste_wint.py`: een interne versie maakt bron, versies, fragmenten en embeddings intern, in één transactie.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-05 Zoeken en lezen: classificatiefilter, nieuwste versie, alleen fragmenten**
- *Wat:* `fn::vector_search` en `fn::text_search` opnieuw gedefinieerd: ze lezen alleen `chunk` en `source_embedding`, filteren in de query op `$classifications` en standaard op de huidige bronversie en run (`$history` voor oudere). `$show_notes` vervalt. `is_derived` (`DEFAULT true`, `ASSERT $value = true`) op `note` en `source_insight`. `ClassificationScope` volgt uit het recht van de sleutel en is een verplicht argument zonder standaard in de leesmethoden van `SearchRepository`, `SourceRepository`, `NoteRepository` en `NotebookRepository`. Een gefilterd record geeft HTTP 404. `include_notes` en `SearchRequest.search_notes` vervallen; `ContextService` neemt geen notities en inzichten meer mee. `SourceResponse` kan een bron met `source_insight` teruggeven (nu HTTP 500, P4 K3).
- *Hangt af van:* F2-03, F2-04.
- *Bestanden:* `migrations/91.surrealql` en `_down`; nieuw `packages/shared/src/shared/models/classification.py`; `packages/surrealdb-service/src/surrealdb_service/repositories/search.py`, `source.py`, `notebook.py`; `apps/app-main/src/app_main/services/search_service.py`, `context_service.py`; `apps/app-main/src/app_main/graphs/ask.py`; `apps/app-main/src/app_main/api/routers/search.py`, `sources.py`, `sources_crud.py`, `notes.py`, `notebooks.py`; `api/schemas.py`.
- *ADR:* 0002 C, 0003, 0004.
- *Stopconditie:* `packages/surrealdb-service/tests/test_search_classificatiefilter.py`: met scope extern 0 interne records bij tekst-, vector- en hybride zoeken, ook met `history=true`; een leesmethode zonder `scope` geeft `TypeError`; het aantal treffers met filter wordt gemeten en vastgelegd (volledigheid, ADR 0002 C). `apps/app-main/tests/test_contract_lezen_extern.py`: synthetische interne en externe gegevens; elke route voor `lezen_extern` aangeroepen; geen intern id of interne tekst in een antwoord; een intern id geeft 404; een bron met `source_insight` geeft 200. `apps/app-main/tests/test_zoeken_geen_afgeleide_teksten.py`: een notitie en een inzicht met de zoektekst komen niet terug; elk resultaat verwijst naar `chunk` of `source_embedding`.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-06 Optimistische vergrendeling en handmatige run**
- *Wat:* `rev`, `modified_by` en `modified_at` op `entity`, `entity_alias`, `relation` en de besluiten in `triage_queue`; één hulpfunctie in de basisrepository; een wijziging met een verouderde `rev` geeft 409. `manual_override` blokkeert een run, die het conflict logt. `resolution_log` krijgt `author`. `ChunkMutator` wordt een dienst die een run met `trigger = handmatig` en nieuwe fragmenten maakt; een automatische run verzet `current_run` niet zolang die naar een handmatige run wijst, en meldt dat in de wachtrij.
- *Hangt af van:* F2-04.
- *Bestanden:* `migrations/92.surrealql` en `_down`; `packages/surrealdb-service/src/surrealdb_service/repositories/base.py`, `entity.py`; `apps/app-main/src/app_main/services/chunking/chunk_mutator.py`; `api/routers/sources_crud.py` (`:1077`, `:1110`), `api/schemas.py` (`:23-35`); `services/entity_resolution/merge_audit.py`; `api/routers/triage.py`, `entity_resolution.py`.
- *ADR:* 0003, 0005.
- *Stopconditie:* `packages/surrealdb-service/tests/test_optimistische_vergrendeling.py`; `apps/app-main/tests/test_handmatige_run.py`: samenvoegen geeft een handmatige run met nieuwe fragmenten, het aantal fragmenten van de oude run blijft gelijk, een automatische run verzet `current_run` niet en maakt een melding; `resolution_log` heeft een auteur.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

#### B. Modelpoort

**F2-07 Modelregister, `ModelGate` en auditlog**
- *Wat:* register als YAML-bestand (YAML Ain't Markup Language) in Git met naam, versie, aanroepwijze, locatie, toegestaan voor intern (standaard `nee`) en taken. `llm_manager.gate.ModelGate` zoekt het model op, weigert een niet-toegestaan model bij interne invoer zonder terugval, probeert bij HTTP 502 van llama-swap opnieuw (pauze, maximum) en schrijft een regel in `model_call` (alleen aanvulbaar, nooit prompt- of antwoordtekst). `POST /api/v1/model-calls` voor Langflow; `/api/models` wordt een alleen-lezen weergave van het register.
- *Hangt af van:* F2-01, F2-04. *Besluit vooraf:* het werkmodel (§8, B4).
- *Bestanden:* nieuw `config/modelregister.yaml`; nieuw `packages/llm-manager/src/llm_manager/gate.py`, `register.py`; `migrations/93.surrealql` en `_down`; `apps/app-main/src/app_main/services/model_routing/telemetry.py` (schrijver van `model_call`); nieuw `api/routers/model_calls.py`; `api/routers/models.py`.
- *ADR:* 0006, 0007, 0009 (502).
- *Stopconditie:* `packages/llm-manager/tests/test_model_gate.py`: cloudmodel geweigerd bij interne invoer, ook als het lokale model faalt; onbekend model geweigerd; 502 leidt tot een nieuwe poging tot het maximum; elke aanroep één auditregel. `packages/llm-manager/tests/test_modelregister_schema.py` valideert het register. `test_migration_93_model_call.py`: de tabel heeft geen tekstveld voor prompt of antwoord.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-08 Modelaanroepen via `ModelGate` en llama-swap**
- *Wat:* Esperanto gebruikt `openai-compatible` met `base_url=http://llama-swap:8080/v1` en model-id's uit de llama-swap-configuratie. `llm_call.py` wordt de dunne aanroeper van `ModelGate`; chat gaat via de poort naar een lokaal model. `llm_matcher.py` gebruikt de chat van de poort; `semantic_intelligence/config.py` en `pipelines/embeddings` gebruiken `/v1/embeddings` (mxbai-embed-large, 1 024 dimensies). Fragmenten boven 512 tokens: de bouwer kiest opdelen of afkappen en legt dat vast (ADR 0008). De aanroepers van `packages/shared/src/shared/model_routing.py` in app-main (`preprocessing_service.py`, `app.py`, `config_coherence.py`) gaan over op de poort.
- *Hangt af van:* F2-07.
- *Bestanden:* `packages/llm-manager/src/llm_manager/providers/factory.py`; `apps/app-main/src/app_main/services/model_routing/llm_call.py`; `apps/app-main/src/app_main/graphs/utils.py`; `apps/app-main/src/app_main/services/preprocessing_service.py`; `packages/shared/src/shared/config_coherence.py`; `pipelines/entity-filtering/src/entity_filtering/resolution/llm_matcher.py`; `packages/semantic-intelligence/src/semantic_intelligence/config.py`; `pipelines/embeddings/src/embeddings/providers/`.
- *ADR:* 0006, 0008.
- *Stopconditie:* `tests/test_geen_ollama_eindpunten.py` faalt bij `/api/generate`, `/api/chat`, `/api/embed`, `11434` of `host.docker.internal` in `apps/`, `packages/` of `pipelines/` (met als enige tijdelijke uitzondering `packages/shared/src/shared/model_routing.py`, dat in F2-09 vervalt); `apps/app-main/tests/test_embedding_local_guardrail.py` aangepast en groen; `pipelines/embeddings/tests/test_lange_fragmenten.py` toetst de 512-tokenregel; de bestaande tests met 1 024 dimensies blijven groen.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-09 Verwijderen: Track J-routering, `model_route` en oude privacyvelden**
- *Wat:* weg gaan `route_resolver.py` (de weigerlogica zit al in `ModelGate`), `failover_executor.py`, `circuit_breaker.py`, `rate_limiter.py`, `error_mapping.py`, `privacy_resolver.py`, `seed.py`, `seed_chain_models.py`, de router `model_routes`, de tabel `model_route`, `packages/shared/src/shared/model_routing.py` en `.yaml`, de cloudaanbieders met `api_key_env_var` in `registry.py`, de modelrijen in de database na omzetten van hun lezers, en de velden `private`, `privacy_mode` en `default_privacy_mode` met het formulierveld `private`.
- *Hangt af van:* F2-08, F2-03.
- *Bestanden:* `apps/app-main/src/app_main/services/model_routing/` (genoemde bestanden); `api/routers/model_routes.py`; `packages/surrealdb-service/src/surrealdb_service/repositories/model_route.py`, `model.py`; `packages/shared/src/shared/model_routing.py`, `model_routing.yaml`; `packages/llm-manager/src/llm_manager/providers/registry.py`; `apps/app-main/src/app_main/api/routers/sources_upload.py` (`:199`); `migrations/94.surrealql` en `_down` (de `_down` zet `private = true`); `ARCHITECTURE.md` §8.
- *ADR:* 0006, 0001.
- *Stopconditie:* verwijdercontrole V met `tests/test_geen_verwijzingen_trackj.py` (`PrivacyMode`, `privacy_mode`, `default_privacy_mode`, `model_route`, `shared.model_routing`); `tests/test_geen_ollama_eindpunten.py` zonder uitzondering; `test_migration_94_*.py`.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

#### C. Omgeving

**F2-10 Compose-basis: netwerken, geheimen, SurrealDB, API en werkbak**
- *Wat:* `docker-compose.signaalsysteem.yml` volgens §4: netwerken `verwerking` en `beheer` (`internal: true`), `toegang` en `uitgaand`; `surrealdb` en `open-notebook` alleen op `verwerking`, zonder poort; `werkbak-toegang` als omgekeerde proxy op `127.0.0.1`; `secrets:` met bestanden in `secrets/`; `secrets/` in `.gitignore` (nu niet aanwezig). De image-build van de fork krijgt vaste basis-tags (`python:3.12.16-slim-bookworm`, `ghcr.io/astral-sh/uv:0.11.13`, zoals in P4). `root`/`root` en `extra_hosts` vervallen.
- *Hangt af van:* F2-01. *Besluit vooraf:* database en SurrealDB-versie (§8, B3).
- *Bestanden:* nieuw `docker-compose.signaalsysteem.yml`; `Dockerfile`; `.gitignore`; nieuw `scripts/signaalsysteem/controle/netwerk.sh` (opzet van P8); nieuw `tests/test_compose_signaalsysteem.py`.
- *ADR:* 0002 F, 0016.
- *Stopconditie:* `docker compose -f docker-compose.signaalsysteem.yml config --quiet` exitcode 0. `tests/test_compose_signaalsysteem.py`: geen image zonder vaste tag of digest, geen `pull_policy: always`; alleen `werkbak-toegang`, `mcp-toegang` en `langflow` publiceren een poort, en alleen op `127.0.0.1`; geen dienst hangt aan `verwerking` én een gewoon netwerk, behalve de proxy's en `langflow`; geen letterlijke wachtwoorden of sleutels in het bestand. `netwerk.sh` exitcode 0: `open-notebook` en `surrealdb` hebben geen internet en geen bereikbare poort; de werkbakproxy op `127.0.0.1` geeft de interface; een API-aanroep zonder sleutel geeft 401.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-11 llama-swap met socket-proxy, embeddingmodel, werkmodel en docling-serve**
- *Wat:* afgeleid image (llama-swap plus Docker-client), `socket-proxy` met configuratie B op `beheer`, `docling-serve` als vooraf aangemaakte container zonder `restart`, configuratie `services/llama-swap/config.yaml` met groepen `zwaar` en `licht` (§4.3), `ttl` van docling 300 s, `checkEndpoint` per dienst. Geheimen als bestand, zodat ze ook bij `docker start` door llama-swap aanwezig zijn. `NVIDIA_DISABLE_REQUIRE=1` en `LD_LIBRARY_PATH` zolang de driver CUDA 13.3 levert. Modellen in een alleen-lezen volume, met een downloadscript dat de SHA-256 en revisies uit `proeven.md` 1.7 controleert.
- *Hangt af van:* F2-10, F2-07.
- *Bestanden:* nieuw `services/llama-swap/Dockerfile`, `config.yaml`, `wacht_op_dienst.sh` (uit `scripts/signaalsysteem/proeven/p1/`); nieuw `scripts/signaalsysteem/modellen_downloaden.py` (uit `p1/modellen_downloaden.py`); `docker-compose.signaalsysteem.yml`; nieuw `packages/llm-manager/tests/test_modelregister_consistent.py`.
- *ADR:* 0006, 0008, 0009, 0014.
- *Stopconditie:* `test_modelregister_consistent.py`: elk model met aanroepwijze `llama-swap` in het register staat in `config.yaml`, en omgekeerd. Op de werkplek `scripts/signaalsysteem/controle/gpu.sh` exitcode 0: `docling-serve` start en stopt via llama-swap (`exited` binnen `ttl` + 15 s); de API-sleutel van docling is aanwezig na `docker start`; embedding van 1 024 dimensies; een externe stop gevolgd door een aanvraag eindigt via de herhaalpoging in HTTP 200; vrij werkgeheugen van de host en piek-VRAM (videogeheugen) worden vastgelegd. Een socket-proxytest herhaalt K2 van P7 (`create`, `exec`, `inspect` geven 403).
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`. Uit te voeren op de werkplek.

**F2-12 Langflow-inrichting**
- *Wat:* afgeleid Langflow-image met `llm_manager` en de componenten; automatisch aanmelden uit; opstartscript dat flows uit Git laadt (`POST /api/v1/flows/upload/`) en triggers aanzet (`POST /api/v1/triggers/{id}/enable`); exportscript (`POST /api/v1/flows/download/`); waarschuwing `niet_vastgelegd` voor een flow die niet in Git staat. MCP-server van Langflow uit, `DO_NOT_TRACK=true`, `LITELLM_LOCAL_MODEL_COST_MAP=True`, `LANGFLOW_SECRET_KEY` als Fernet-sleutel uit een geheim.
- *Hangt af van:* F2-10.
- *Bestanden:* nieuw `services/langflow/Dockerfile`, `services/langflow/scripts/flows_laden.py`, `flows_exporteren.py`, map `services/langflow/flows/`, `services/langflow/components/signaalsysteem/`, `services/langflow/tests/`; `docker-compose.signaalsysteem.yml`.
- *ADR:* 0017, 0002 D.
- *Stopconditie:* `services/langflow/tests/test_flows_rondgang.py`: laden en daarna exporteren levert dezelfde flow-JSON (JavaScript Object Notation) op, op volgorde van sleutels na. `scripts/signaalsysteem/controle/langflow.sh` exitcode 0: aanmelden zonder wachtwoord faalt; het MCP-eindpunt van Langflow geeft geen 200; het aantal uitgaande verbindingspogingen van LiteLLM is 0, geteld zoals in P2 met Langflow alleen op `verwerking`. Blijven er pogingen over, dan stopt de taak en volgt de terugval uit ADR 0017 (ophalen van bronnen in een eigen container).
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

#### D. Inname en verwerking

**F2-13 Bronkaart, innamebatch en innamewachtrij in de fork**
- *Wat:* tabellen en API voor bronkaartregels (schema §5.2), batches van een bulkupload (classificatie en route per batch) en de innamewachtrij (documenten uit een watchfolder en losse uploads ter bevestiging, bronnen zonder regel). Standaardclassificatie per innameroute volgens ADR 0019; zonder keuze `intern`.
- *Hangt af van:* F2-03, F2-04. *Besluit vooraf:* plaats van de bronkaart (§8, B2).
- *Bestanden:* `migrations/95.surrealql` en `_down`; nieuw `packages/shared/src/shared/models/source_map.py`; nieuw `packages/surrealdb-service/src/surrealdb_service/repositories/source_map.py`, `intake.py`; nieuw `apps/app-main/src/app_main/services/source_map_service.py`, `intake_service.py`; nieuw `api/routers/source_map.py`, `intake.py`.
- *ADR:* 0019, 0001, 0010; FO 5.
- *Stopconditie:* `test_migration_95_source_map.py` (schemacontrole, ook op de standaardwaarden); `apps/app-main/tests/test_source_map_service.py`: een regel voor route API, MCP, feed of website stelt `extern` voor en vraagt bevestiging; een batch zonder keuze is `intern`; een watchfolder-item heeft geen standaard en staat op `wacht`.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-14 Importeindpunt**
- *Wat:* `POST /api/v1/import` met `ImportService` (§5.1): bron, bronversie, classificatie, `run_id`, Docling-structuur als `doc_node`, fragmenten met labels. Bekende `source` + `content_hash` geeft `ongewijzigd` en de run `niets_nieuws`; hetzelfde verzoek met dezelfde `run_id` geeft dezelfde id's; een lagere classificatie dan de bestaande geeft 409. Hergebruik van `_create_source_impl`, het schrijven van fragmenten in `repositories/source.py` en de opslag als `doc_node`.
- *Hangt af van:* F2-04, F2-05, F2-13.
- *Bestanden:* nieuw `apps/app-main/src/app_main/services/import_service.py`, nieuw `api/routers/imports.py`; `api/schemas.py`; `repositories/source.py`; `dependencies.py`.
- *ADR:* 0010, 0019, 0003, 0001.
- *Stopconditie:* `apps/app-main/tests/test_import_service.py`: twee keer hetzelfde verzoek geeft hetzelfde resultaat; een bekende hash maakt geen nieuwe versie; 409 bij een lagere classificatie; elk geschreven fragment heeft `source_version`, `run` en `extractor`. `apps/app-main/tests/test_enige_schrijfroute.py`: geen route behalve `/api/v1/import` (en de handmatige run uit F2-06) roept de schrijfmethoden voor `source` of `chunk` aan.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-15 Stap-eindpunten en `RunStepOrchestrator`**
- *Wat:* `POST` en `GET /api/v1/runs/{run_id}/steps/{stap}` (§5.4) voor `embed`, `entity_extraction`, `entity_dedup` en `graph`. De fork zet een job met `job.run` in `packages/job-queue`; de worker doet de herhaalpogingen; na het maximum gaat de run op `mislukt` met een melding. `processing_stage` blijft als statusveld. De handlers voor EMBED, EXTRACT en GRAPH in `handlers.py` blijven, en starten vanuit een stap. Embeddings berekent de fork via `pipelines/embeddings` en `ModelGate`.
- *Hangt af van:* F2-04, F2-08.
- *Bestanden:* nieuw `apps/app-main/src/app_main/services/run_step_orchestrator.py`; `api/routers/runs.py`; `apps/app-main/src/app_main/handlers.py`; `packages/job-queue/src/job_queue/`.
- *ADR:* 0011, 0003.
- *Stopconditie:* `apps/app-main/tests/test_run_step_orchestrator.py`: een stap maakt een job met `run`; een mislukte job wordt opnieuw geprobeerd tot `max_retries` en zet daarna run en stap op `mislukt` met een melding; een stap op een onbekende run geeft 404; embeddings van de stap hebben `run` en `embedding_model`.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-16 Basiscomponenten in Langflow**
- *Wat:* gedeelde Python-componenten: run aanmaken en afsluiten (met Git-commit en hash van de flow), stap starten en volgen, modelkeuze met registercontrole (een cloudmodel achter een interne of onbekende bron geeft een fout bij verbinden en bij uitvoeren), herhaalbeleid bij 502, 429 en 5xx, en een auditregel per modelaanroep via `POST /api/v1/model-calls`. Langflow praat alleen met de API van de fork en met llama-swap, nooit met de database.
- *Hangt af van:* F2-12, F2-07, F2-14, F2-15.
- *Bestanden:* nieuw `services/langflow/components/signaalsysteem/run.py`, `stap.py`, `model.py`, `fork_client.py`; `services/langflow/tests/`.
- *ADR:* 0006, 0007, 0009, 0011, 0017.
- *Stopconditie:* `services/langflow/tests/test_model_component.py`: cloudmodel achter een interne of onbekende bron geweigerd, zonder terugval; `test_stap_component.py` tegen een nagebootste fork: 502 leidt tot een nieuwe poging, `wacht_op_review` wordt herkend (§5.4); `test_run_component.py`: de run legt commit en flow-hash vast.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-17 Docling-component met fragmentatie**
- *Wat:* de component stuurt een document naar `/upstream/docling/v1/convert/file/async` (`to_formats` met `json`, `ocr_lang` `nl` en `en`), vraagt elke 10 s de status op, dient bij HTTP 404 opnieuw in (met maximum en logregel) en haalt het resultaat op. Het adres is een instelling per aanroepwijze; vóór elke aanroep toetst de component de classificatie tegen het register. Het DoclingDocument gaat naar `ChunkExtractor` en `regrouper.py`. De OCR- en tabelopties staan in de flow, niet in Compose.
- *Hangt af van:* F2-16, F2-11.
- *Bestanden:* nieuw `services/langflow/components/signaalsysteem/docling.py`, `fragmentatie.py`; `pipelines/ingestion/src/ingestion/chunking/extractor.py`, `regrouper.py` (hergebruik, alleen aanhaakpunten).
- *ADR:* 0014, 0009, 0019, 0010.
- *Stopconditie:* `services/langflow/tests/test_docling_component.py` tegen een nagebootste docling-serve: 404 tijdens het pollen leidt tot opnieuw indienen; een intern document naar een adres dat niet toegestaan is voor intern wordt geweigerd, zonder terugval. Op de werkplek `scripts/signaalsysteem/controle/docling_fragmentatie.sh` exitcode 0: D2 (Wageningen, `proeven.md` 1.6) door docling-serve en `ChunkExtractor`; elk fragment heeft sectiepad en paginanummer, en de tekst "weerstandsvermogen" komt voor.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`. Deels uit te voeren op de werkplek.

**F2-18 Poortwachter, regeldeel**
- *Wat:* component die brontype en route uit de bronkaart bepaalt en interne kenmerken zoekt met regels (rubriceringsmarkering, conceptstatus, interne afzender, map en bestandskenmerken). Een treffer zet het document op `intern` en in de wachtrij. De poortwachter kan nooit van intern naar extern. Het modeldeel met Kev volgt in F3-04.
- *Hangt af van:* F2-16, F2-13.
- *Bestanden:* nieuw `services/langflow/components/signaalsysteem/poortwachter.py`; nieuw `config/poortwachter-regels.yaml`.
- *ADR:* 0001, 0019; FO 6.
- *Stopconditie:* `services/langflow/tests/test_poortwachter_regels.py` met synthetische documenten: een externe batch met een rubriceringsmarkering wordt `intern` en komt in de wachtrij; invoer `intern` geeft nooit uitvoer `extern`; zonder regel en zonder besluit `intern`.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-19 Innameflows voor PDF en lokale map**
- *Wat:* drie flows: losse upload (voorstel van de poortwachter, start na bevestiging), bulkupload (één classificatie en route voor de batch) en mapcontrole (watchdog-logica uit `file_watcher.py`; nieuwe documenten gaan naar de innamewachtrij en wachten op de beheerder). Elke flow: registratie met inhoudshash, poortwachter, docling, fragmentatie, import en de stappen `embed`, `entity_extraction` en `graph`. Originelen komen in het volume `bronarchief`.
- *Hangt af van:* F2-17, F2-18, F2-14, F2-15.
- *Bestanden:* nieuw `services/langflow/flows/pdf_upload.json`, `pdf_bulkupload.json`, `mapcontrole.json`; nieuw `services/langflow/components/signaalsysteem/mapcontrole.py`, `registratie.py`.
- *ADR:* 0010, 0019, 0011.
- *Stopconditie:* `services/langflow/tests/test_mapcontrole.py`: een nieuw bestand levert precies één wachtrij-item en geen import; `test_registratie.py`: dezelfde inhoud geeft dezelfde hash. De integratie toetst F2-24.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-20 Werkbak als reviewomgeving**
- *Wat:* inloggen met een sleutel (Bearer), PDF-weergave met header, verbergen van menu-items op vervallen routes, scherm voor herclassificatie (met reden), en wachtrijen 1 (innamewachtrij: bron zonder regel, watchfolder, losse upload), 2 (treffer of twijfel van de poortwachter) en 7 (mislukte runs) op het patroon van het aandachtsoverzicht.
- *Hangt af van:* F2-01, F2-03, F2-13, F2-15.
- *Bestanden:* `frontend/src/lib/api/client.ts`, de PDF-weergave en het menu in `frontend/src/`; `apps/app-main/src/app_main/api/routers/attention.py`; `intake.py`.
- *ADR:* 0016, 0002, 0019.
- *Stopconditie:* `npm --prefix frontend run test` met nieuwe tests voor aanmelden met sleutel en de drie wachtrijen; `npm --prefix frontend run build`; `apps/app-main/tests/test_attention_wachtrijen.py` voor de eindpunten.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-21 Obsidian-export als Langflow-flow**
- *Wat:* `GET /api/v1/export/obsidian/entities?since=` (recht `lezen_intern`) met id, canonieke naam, aliassen, type, codeboekkoppelingen, herkomst en classificatie. Component met sjabloon: indeling naar FO 10, notitie vinden op `id` in de frontmatter, schrijven alleen in het blok `<!-- signaalsysteem:begin -->` … `<!-- signaalsysteem:end -->`, atomair schrijven en padcontrole hergebruikt uit `obsidian_export_service.py`. Een nieuwe persoon gaat naar de wachtrij in plaats van naar een bestand. Alleen `langflow` koppelt de vault.
- *Hangt af van:* F2-16, F2-05. *Besluit vooraf:* map voor programma's (§8, B8).
- *Bestanden:* nieuw `apps/app-main/src/app_main/api/routers/export_obsidian.py` en dienst; nieuw `services/langflow/components/signaalsysteem/obsidian_export.py`, `services/langflow/templates/obsidian_entiteit.md`, `services/langflow/flows/obsidian_export.json`.
- *ADR:* 0018, 0001, 0003.
- *Stopconditie:* `services/langflow/tests/test_obsidian_export.py` op een testvault met alleen openbare gegevens: twee runs na elkaar geven byte voor byte dezelfde bestanden; tekst buiten het blok blijft gelijk; een interne entiteit krijgt `classificatie: intern`; een nieuwe persoon geeft een reviewitem en geen bestand. `apps/app-main/tests/test_export_obsidian_api.py`: classificatie intern zodra één onderliggend fragment intern is.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

#### E. Opruimen en toets

**F2-22 Verwijderen: ingebouwde inname en keten**
- *Wat:* weg gaan de routes in `sources_upload.py` die bronnen maken of opnieuw verwerken, `process-url`, `process-document` en `process-audio` in `agents.py`, `file_watcher.py` met de router `watcher` en de schakelaar in `config.py`, de router `zotero` (de client in `packages/zotero-integration` blijft voor F4-05), de OKF-import, de stappen INGEST en INSIGHTS en `advance_source` in `source_pipeline.py`, `SourceExtractor`, `SourceProcessor` met het verwijderpad voor fragmenten (`repositories/source.py:1158-1176`), `services/parsing/`, de jobtypen `DOCUMENT_PARSE` en `AUDIO_TRANSCRIBE` met hun handlers, en `LibrarianService.enqueue_due`.
- *Hangt af van:* F2-19, F2-14, F2-15.
- *Bestanden:* genoemde bestanden in `apps/app-main/src/app_main/`; `packages/shared/src/shared/types/enums.py`; `packages/surrealdb-service/src/surrealdb_service/repositories/source.py`; `frontend/src/` (upload, Zotero, mapbewaker); `ARCHITECTURE.md` §14.
- *ADR:* 0010, 0003, 0004.
- *Stopconditie:* verwijdercontrole V met `tests/test_geen_verwijzingen_inname.py`; `test_enige_schrijfroute.py` uit F2-14 blijft groen.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-23 Verwijderen: oude GPU-diensten, oude export en `docker-compose.yml`**
- *Wat:* weg gaan `services/docling/`, `services/whisperx/`, `services/mineru/`, `services/summarization/`; in `pipelines/ingestion` `parsers/docling_parser.py`, `workflow.py`, `cli.py`, `audio/transcriber.py` en de exporters als niets ze meer gebruikt; `obsidian_export_service.py`, `vault_sync_service.py` en de schrijvende routes in `vault.py`. `docker-compose.signaalsysteem.yml` vervangt `docker-compose.yml`; er blijven geen twee Compose-bestanden. Volumes en gegevens blijven staan.
- *Hangt af van:* F2-21, F2-22, F2-24 geslaagd. *Besluit vooraf:* vervangen van de bestaande omgeving (§8, B6).
- *Bestanden:* genoemde mappen en bestanden; `docker-compose.yml`; `ARCHITECTURE.md` §1 en §4.
- *ADR:* 0014, 0015, 0018, 0002.
- *Stopconditie:* verwijdercontrole V met `tests/test_geen_verwijzingen_oude_diensten.py`; `tests/test_compose_env_is_consumed.py` aangepast aan het nieuwe Compose-bestand en groen.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-24 Toets fase 2**
- *Wat:* script dat de keten van begin tot eind draait op de werkplek: D2 en een tweede openbare PDF via bulkupload (extern), D1 via de watchfolder met een besluit van de beheerder via de API (intern), en de stappen tot en met `graph`. Daarna de GPU-toets met een lopende asynchrone docling-taak (D1) en tegelijk een aanvraag voor het werkmodel en voor embeddings.
- *Hangt af van:* F2-19, F2-20, F2-08, F2-09.
- *Bestanden:* nieuw `scripts/signaalsysteem/toetsen/fase2.sh`, `fase2_herleidbaarheid.surql`.
- *ADR:* 0003, 0009.
- *Stopconditie:* `fase2.sh` exitcode 0: (a) 0 fragmenten van nieuwe runs zonder `source_version`, `run`, `run.pipeline_version` of `extractor`, en 0 embeddings zonder `run` en `embedding_model`; (b) 0 geheugenfouten in de logs van llama-swap en de diensten, 0 tijdvakken met twee zware diensten tegelijk (`docker events`), de docling-taak eindigt in `success`, eventueel na opnieuw indienen; (c) vrij werkgeheugen van de host blijft boven de grens uit risico R1.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`. Uit te voeren op de werkplek.

**F2-25 Omweg voor de CUDA-driver laten vervallen**
- *Wat:* Geert werkt de NVIDIA-driver bij naar CUDA 13.4 of hoger. Daarna gaan `NVIDIA_DISABLE_REQUIRE=1` en `LD_LIBRARY_PATH` (ADR 0006, ADR 0009, `proeven.md` §8 punt 23) uit `docker-compose.signaalsysteem.yml`. Het criterium van P0 wordt aangescherpt van CUDA ≥ 13.0 naar ≥ 13.4, de eis van de vastgezette llama-swap-tag.
- *Hangt af van:* F2-11; driverupdate door Geert.
- *Bestanden:* `docker-compose.signaalsysteem.yml`; `scripts/signaalsysteem/proeven/p0_omgeving.sh`; `tests/test_compose_signaalsysteem.py`.
- *ADR:* 0006, 0009.
- *Stopconditie:* `scripts/signaalsysteem/proeven/p0_omgeving.sh --gpu` exitcode 0 en meldt CUDA ≥ 13.4; `llama-swap` start zonder de twee variabelen en laadt het werkmodel op de GPU (`scripts/signaalsysteem/controle/gpu.sh` exitcode 0); `tests/test_compose_signaalsysteem.py` faalt als een van beide variabelen nog in het Compose-bestand staat.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`. Uit te voeren op de werkplek.

### Fase 3: labeling en MCP

*Toets (F3-12):* precisie en recall zijn gemeten op de referentieset, en geen intern fragment verschijnt in een externe uitvraag (F3-11).

**F3-01 Kev-4B onder llama-swap**
- *Wat:* image `services/kev/Dockerfile` (uit `p1/kev.Dockerfile`: tag `kev-1.0`, `uv sync --extra serve`), adapter en basismodel op vaste revisie in een volume, `--host 0.0.0.0`, `KEV_API_KEY` als geheimbestand. In llama-swap groep `zwaar`, `checkEndpoint: /openapi.json`, proxy naar Kev. Een Kev-client in `llm_manager` voor `/upstream/kev-4b/v1/systemone`, via `ModelGate`. Register: lokaal, toegestaan voor intern.
- *Hangt af van:* F2-11, F2-07.
- *Bestanden:* nieuw `services/kev/Dockerfile`; `services/llama-swap/config.yaml`; nieuw `packages/llm-manager/src/llm_manager/kev_client.py`; `config/modelregister.yaml`; `docker-compose.signaalsysteem.yml`.
- *ADR:* 0009, 0006.
- *Stopconditie:* `packages/llm-manager/tests/test_kev_client.py` tegen een nagebootste server. Op de werkplek `scripts/signaalsysteem/controle/kev.sh` exitcode 0: HTTP 200 en kansen per `choice`-vraag 1 ± 0,01; 50 vragen zonder geheugenfout met het embeddingmodel geladen; piek-VRAM met nulmeting vastgelegd en ≤ 15 279 MiB. Faalt dat, dan legt de bouwer Kev-0,8B voor aan Geert.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`. Uit te voeren op de werkplek.

**F3-02 Referentieset opbouwen**
- *Wat:* 200 tot 300 fragmenten uit openbare bronnen (jaarstukken, raadsstukken, Kamerstukken), met per fragment de gouden labels: route en documentsoort, Iv3-taakvelden (Informatie voor derden), entiteiten met referentie-id (CBS-gemeentecode, BWB-id) en, voor de poortwachter, synthetische interne kenmerken op een deel van de fragmenten. Tooling maakt kandidaten uit de pipeline van fase 2; Geert labelt. Kan starten zodra F2-17 werkt.
- *Hangt af van:* F2-17, F3-03.
- *Bestanden:* nieuw `tests/signaalsysteem/referentieset/bronnen.yaml` (URL, SHA-256, licentie), `fragmenten.jsonl`; nieuw `scripts/signaalsysteem/referentieset/kandidaten.py`; nieuw `tests/signaalsysteem/test_referentieset_schema.py`.
- *ADR:* FO 7; 0008, 0013.
- *Stopconditie:* `test_referentieset_schema.py`: 200 ≤ aantal ≤ 300; elk fragment verwijst naar een bron met openbare URL en SHA-256; elk label bestaat in codeboekversie Iv3 2025 of in de referentielijst; geen dubbele fragmenten.
- *Agent:* `signaal-bouwer` voor de tooling, labels door Geert; daarna `signaal-reviewer`.

**F3-03 Codeboek Iv3 2025 en BWB-woordenlijst**
- *Wat:* laadscript voor de Iv3-taakvelden 2025 naar een ontologie met versie in `ontology_version`; BWB-leverancier naast `tooi_provider.py` die `reference_entity` vult; eindpunt `GET /api/v1/codeboek/{schema}/{versie}` voor de keuzelijst van de Kev-component. De vindplaats van beide openbare bronnen zoekt de bouwer uit (ADR 0013).
- *Hangt af van:* F2-01.
- *Bestanden:* nieuw `scripts/signaalsysteem/laad_iv3_taakvelden.py`; nieuw `packages/shared/src/shared/vocabulary/bwb_provider.py`; `packages/ontology-manager/`; nieuw `apps/app-main/src/app_main/api/routers/codeboek.py`.
- *ADR:* 0013.
- *Stopconditie:* `packages/shared/tests/test_bwb_provider.py` op een klein openbaar voorbeeld; `tests/signaalsysteem/test_iv3_laden.py`: het aantal geladen taakvelden is gelijk aan het aantal in het bronbestand (SHA-256 vastgelegd); twee keer laden geeft één versie.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F3-04 Poortwachter, modeldeel met Kev**
- *Wat:* Kev beoordeelt wat de regels openlaten: documentsoort, route en interne kenmerken, met een gekalibreerde kans per keuze. Onder de drempel gaat de bron naar wachtrij 2. De drempel volgt uit de meting op de referentieset en staat in `config/poortwachter-regels.yaml`.
- *Hangt af van:* F2-18, F3-01, F3-02.
- *Bestanden:* `services/langflow/components/signaalsysteem/poortwachter.py`; `config/poortwachter-regels.yaml`; nieuw `scripts/signaalsysteem/toetsen/poortwachter_referentieset.py`.
- *ADR:* 0001, 0019, 0009; FO 6.
- *Stopconditie:* `services/langflow/tests/test_poortwachter_kev.py` met een nagebootste Kev: kans onder de drempel geeft een wachtrij-item; invoer `intern` geeft nooit `extern`. `poortwachter_referentieset.py` levert precisie en recall per route en het aandeel naar review; de gekozen drempel staat in de configuratie.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F3-05 Labeling met Kev**
- *Wat:* component die per fragment Kev vraagt met de keuzelijst uit de codeboekversie in de bronkaartregel, en labels met methode `kev`, kans en model meegeeft aan de import. De run legt de versie van het labelschema vast.
- *Hangt af van:* F3-01, F3-03, F2-14.
- *Bestanden:* nieuw `services/langflow/components/signaalsysteem/labeling.py`; `apps/app-main/src/app_main/services/import_service.py` (labels opslaan).
- *ADR:* 0013, 0003; FO 7.
- *Stopconditie:* `services/langflow/tests/test_labeling.py`: alleen termen uit de keuzelijst; `apps/app-main/tests/test_import_labels.py`: label met methode, kans en model opgeslagen; de run bevat de labelschemaversie.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F3-06 Graaf `entity_extraction`**
- *Wat:* LangGraph-graaf met knopen extraheren (`EntityExtractionService` en de orchestrator uit `pipelines/ontology-extraction`), valideren tegen `reference_entity`, alleen afgekeurde voorstellen opnieuw met een maximum aantal rondes, en wegschrijven met `run`. Stap `entity_extraction` via `RunStepOrchestrator`. De lezers van `provenance_chain` gaan over op de vermeldingen met run.
- *Hangt af van:* F2-15, F3-03.
- *Bestanden:* nieuw `apps/app-main/src/app_main/graphs/entity_extraction.py`; `services/entity_extraction_service.py`, `entity_persistence_service.py`; `services/run_step_orchestrator.py`; de lezers van `provenance_chain` (onder meer `recanonicalization_service.py`, `notebook_merge_service.py`, `okf_export_service.py`, `semantic_intelligence/reasoning.py`).
- *ADR:* 0012, 0003.
- *Stopconditie:* `apps/app-main/tests/test_graph_entity_extraction.py`: een afgekeurd voorstel gaat opnieuw, een goedgekeurd niet; stop na het maximum; elke vermelding heeft `run`. De regressiepoort van Track N (`scripts/n_extraction_gate.py` tegen `tests/regression/n_extraction_baseline.json`) slaagt.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F3-07 Graaf `entity_dedup` met pauze bij twijfel**
- *Wat:* de lagen van `FilteringWorkflow` als knopen; bij twijfel een item in `triage_queue` en `interrupt()`; het triagebesluit via de router `triage` hervat de graaf met hetzelfde `thread_id`; checkpoints in het bestaande SQLite-bestand. De stap staat intussen op `wacht_op_review`.
- *Hangt af van:* F3-06, F2-06.
- *Bestanden:* nieuw `apps/app-main/src/app_main/graphs/entity_dedup.py`; `pipelines/entity-filtering/src/entity_filtering/workflow.py` (alleen aanhaakpunten); `api/routers/triage.py`; `services/run_step_orchestrator.py`.
- *ADR:* 0012, 0011.
- *Stopconditie:* `apps/app-main/tests/test_graph_entity_dedup.py`: bij twijfel stopt de graaf en staat de stap op `wacht_op_review`; na een triagebesluit hervat hij met dezelfde tussenstand; een verouderde `rev` op het besluit geeft 409.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F3-08 Verwijderen: `provenance_chain`** — vraagt expliciete bevestiging van Geert
- *Wat:* het veld `provenance_chain` op `entity` en `relation` vervalt met een migratie. De inhoud gaat verloren; de herkomst staat dan in de vermeldingen met run.
- *Hangt af van:* F3-06.
- *Bestanden:* `migrations/<n>.surrealql` en `_down`; `packages/shared/src/shared/models/entity.py`; `packages/surrealdb-service/src/surrealdb_service/repositories/entity.py`.
- *ADR:* 0003.
- *Stopconditie:* verwijdercontrole V met `tests/test_geen_verwijzingen_provenance_chain.py`; reservekopie van de database aanwezig vóór de migratie.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F3-09 Herkomst- en versie-eindpunten**
- *Wat:* `GET /api/v1/chunks/{id}/provenance` (bron, bronversie, run, pipelineversie, modellen) en `GET /api/v1/sources/{id}/versions`, met classificatiebereik. Deze komen op de lijst voor `lezen_extern`. Entiteiten blijven voor `lezen_extern` afgesloten tot het besluit in §8, B10.
- *Hangt af van:* F2-05, F2-04.
- *Bestanden:* nieuw `apps/app-main/src/app_main/api/routers/provenance.py`; `packages/surrealdb-service/src/surrealdb_service/repositories/graph.py`, `source_version.py`.
- *ADR:* 0002 B, D; 0003; FO 10.
- *Stopconditie:* `apps/app-main/tests/test_provenance_api.py`: voor elk fragment een volledige keten tot run en model; een intern fragment geeft 404 bij `lezen_extern`; `test_contract_lezen_extern.py` uitgebreid met deze routes.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F3-10 open-notebook-mcp, `mcp-toegang` en de reranker in Compose**
- *Wat:* container met `open-notebook-mcp==0.3.0` en `mcp==1.30.0`, een startscript (uit `p4/start_mcp.py`) dat de sleutel `lezen_extern` uit het geheimbestand leest en `host`, `port` en `path` via `mcp.settings` zet. `mcp-toegang` stuurt alleen `/mcp` door en publiceert alleen op `127.0.0.1`. De bridge-optie `enable_ip_masquerade: "false"` op `toegang` alleen als de proef laat zien dat de poort bereikbaar blijft en internet wegvalt. De reranker als CPU-dienst op `verwerking`, via `ModelGate` (ADR 0015 5c).
- *Hangt af van:* F2-10, F2-05.
- *Bestanden:* nieuw `services/open-notebook-mcp/Dockerfile`, `start_mcp.py`; `docker-compose.signaalsysteem.yml`; `services/reranker/`; `scripts/signaalsysteem/controle/netwerk.sh`.
- *ADR:* 0002 D, F; 0015.
- *Stopconditie:* `netwerk.sh` uitgebreid, exitcode 0: `open-notebook-mcp` heeft geen internet en geen poort; `127.0.0.1:<poort>/mcp` antwoordt; een ander pad via de proxy geeft 404. `tests/test_compose_signaalsysteem.py`: beide pakketten met `==` vastgezet.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F3-11 Acceptatietest van het filter: P4**
- *Wat:* `p4_mcp_classificatiefilter.sh` aangepast aan de nieuwe stand: een sleutel `lezen_extern` in plaats van het wachtwoord, `classification` in plaats van `private`, en het image van de fork na F2-05. Deelstap 4c met vector- en hybride zoeken via llama-swap.
- *Hangt af van:* F3-10, F2-08, F2-11.
- *Bestanden:* `scripts/signaalsysteem/proeven/p4_mcp_classificatiefilter.sh`, `p4/vul_testdatabase.py`, `p4/mcp_lektest.py`.
- *ADR:* 0001, 0002.
- *Stopconditie:* `scripts/signaalsysteem/proeven/p4_mcp_classificatiefilter.sh` exitcode 0: K1 nul treffers `PRF-INT-`; K2 positieve controle geldig; K3 HTTP 404 op een intern id; K4 HTTP 403 op elke schrijftool en op `ask_simple` (falen door een fout telt niet); K5 0 verwerkingsfouten; K6 HTTP 401 zonder sleutel; 4c uitgevoerd met dezelfde uitkomst.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F3-12 Toets fase 3: meting op de referentieset**
- *Wat:* meetscript voor precisie en recall van labels, entiteiten (met referentie-id) en de route van de poortwachter. Optioneel de embeddingvergelijking uit ADR 0008: een kandidaat moet naast Kev-4B passen.
- *Hangt af van:* F3-02, F3-04, F3-05, F3-06, F3-07, F3-11.
- *Bestanden:* nieuw `scripts/signaalsysteem/toetsen/fase3_referentieset.py`.
- *ADR:* FO 7, 15; 0008.
- *Stopconditie:* het script draait over alle fragmenten van de set en schrijft per onderdeel precisie, recall en aantallen naar `docs/signaalsysteem/meting-fase3.md`; F3-11 is geslaagd. De toets vraagt een meting, geen drempel; Geert beslist op basis van de uitkomst over fase 4.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

### Fase 4: bronnen en planning

*Toets (F4-12):* een geplande run verwerkt alleen nieuwe of gewijzigde inhoud.

**F4-01 Planning volgens de bronkaart**
- *Wat:* een synchronisatiescript maakt per bronkaartregel met schema een trigger via de API van Langflow (cron, `Europe/Amsterdam`, `coalesce` of `skip`), met het id van de regel in de payload; een verwijderde of inactieve regel zet de trigger uit.
- *Hangt af van:* F2-12, F2-13.
- *Bestanden:* nieuw `services/langflow/scripts/schema_synchroniseren.py`; `services/langflow/tests/`.
- *ADR:* 0017 C.
- *Stopconditie:* `scripts/signaalsysteem/controle/planning.sh` exitcode 0, met de criteria K2–K4 van P2 op een regel van elke minuut; elke run heeft `trigger = schema` en het id van de regel.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F4-02 Feedcontrole voor YouTube en podcasts, Spotify naar RSS**
- *Wat:* component op basis van `p6/feedcontrole.py` met `feedparser==6.0.14`: detectie op video-id of GUID, elk nieuw item een eigen bron; de fork beantwoordt of een extern id al bestaat. Het veld Inhoud ophalen wordt gevolgd: van YouTube standaard alleen metadata. Spotify-link via iTunes Search en oEmbed naar de RSS-feed (Really Simple Syndication).
- *Hangt af van:* F4-01, F2-14.
- *Bestanden:* nieuw `services/langflow/components/signaalsysteem/feedcontrole.py`, `spotify_naar_rss.py`; `apps/app-main/src/app_main/api/routers/sources.py` (opzoeken op extern id).
- *ADR:* FO 5; 0019.
- *Stopconditie:* `services/langflow/tests/test_feedcontrole.py` op ingekorte opnamen van D3 en D4 (openbaar): criteria K1–K4 van P6 als test; `test_spotify_naar_rss.py` met opgenomen antwoorden: criterium K5.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F4-03 WhisperX onder llama-swap en transcriptiecomponent**
- *Wat:* container `whisperx` (variant A) als vooraf aangemaakte container in groep `zwaar`, met `checkEndpoint: /ping`, `OPT_OUT_TRACKING=true`, `HF_HUB_OFFLINE=1` en volumes voor Whisper, uitlijnmodellen en NLTK `punkt_tab`. Component zet segmenten met tijdcodes en sprekers om in fragmenten, met het model uit `pipelines/ingestion/src/ingestion/models/transcription.py`.
- *Hangt af van:* F2-11, F2-16.
- *Bestanden:* `docker-compose.signaalsysteem.yml`; `services/llama-swap/config.yaml`; nieuw `services/langflow/components/signaalsysteem/transcriptie.py`; `scripts/signaalsysteem/modellen_downloaden.py`.
- *ADR:* 0014, 0009.
- *Stopconditie:* op de werkplek `scripts/signaalsysteem/controle/whisperx.sh` exitcode 0: D7 en D8 zonder `language` geven taal `nl` en `en` met woordtijden; D4 ≥ 2 sprekers; geen internet op `verwerking`; de niet gemeten combinatie embed + WhisperX gemeten en ≤ 15 279 MiB; vrij werkgeheugen van de host vastgelegd. `services/langflow/tests/test_transcriptie.py` op een opgenomen antwoord.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`. Uit te voeren op de werkplek.

**F4-04 Website-connector met wijzigingsdetectie**
- *Wat:* pagina of sitemap ophalen in Langflow, HTML via docling-serve, wijzigingsdetectie op inhoudshash.
- *Hangt af van:* F2-17, F4-01.
- *Bestanden:* nieuw `services/langflow/components/signaalsysteem/website.py`, `services/langflow/flows/website.json`.
- *ADR:* FO 5, 7.
- *Stopconditie:* `services/langflow/tests/test_website.py` tegen een lokale testserver met een openbare pagina als opname: ongewijzigd geeft `niets_nieuws`, een wijziging één nieuwe bronversie.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F4-05 API-connector en Zotero als API-bron**
- *Wat:* connector die API-velden naar tekst en metadata zet en bijlagen als document doorgeeft, bijvoorbeeld voor raadsmoties; de Zotero-client uit `packages/zotero-integration` als tweede connector.
- *Hangt af van:* F2-17, F4-01.
- *Bestanden:* nieuw `services/langflow/components/signaalsysteem/api_connector.py`, `zotero.py`.
- *ADR:* 0010; FO 5.
- *Stopconditie:* `services/langflow/tests/test_api_connector.py` en `test_zotero.py` op opgenomen openbare antwoorden: velden en bijlagen komen in de import; een tweede run zonder wijziging geeft `niets_nieuws`.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F4-06 MCP-bron**
- *Wat:* component met een MCP-client naar een externe server; het resultaat wordt een document. Classificatie volgens de regel; een intern systeem krijgt `intern`.
- *Hangt af van:* F2-17, F4-01.
- *Bestanden:* nieuw `services/langflow/components/signaalsysteem/mcp_bron.py`.
- *ADR:* 0019; FO 5.
- *Stopconditie:* `services/langflow/tests/test_mcp_bron.py` tegen een nagebootste MCP-server: resultaat als document; een regel `intern` geeft geen cloudmodel in de rest van de flow.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F4-07 Video: transcript en slides**
- *Wat:* transcript als bij audio; bij elke scènewissel een beeld, tekst via OCR (Optical Character Recognition) in docling-serve, gekoppeld aan het transcript via de tijdcode. *Besluit vooraf:* de tool voor scènedetectie en de plek waar die draait (§8, B11). Bouw er niet op door zonder ADR.
- *Hangt af van:* F4-03, F2-17, ADR voor scènedetectie.
- *Bestanden:* nieuw `services/langflow/components/signaalsysteem/video.py`; eventueel een container volgens die ADR.
- *ADR:* nieuw (voorgesteld); FO 7.
- *Stopconditie:* `services/langflow/tests/test_video.py` met een openbare video (Wikimedia Commons): elk slidefragment heeft een tijdcode die binnen een transcriptsegment valt.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F4-08 Selectieve extractie**
- *Wat:* eerst zoeken op kop in de Docling-structuur; geen treffer, dan sectieclassificatie door Kev met de secties als keuzelijst; nog geen treffer, dan wachtrij 3 in de werkbak.
- *Hangt af van:* F2-17, F3-01, F2-20.
- *Bestanden:* nieuw `services/langflow/components/signaalsysteem/selectie.py`; `frontend/src/` (wachtrij 3); `api/routers/attention.py`.
- *ADR:* FO 7; 0016.
- *Stopconditie:* `services/langflow/tests/test_selectie.py` op de Docling-JSON van D1 en D2: alleen de paragraaf weerstandsvermogen en risicobeheersing blijft over; een document zonder treffer geeft een wachtrij-item. Frontendtest voor wachtrij 3.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F4-09 PyMuPDF-aanvulling, annotaties en GROBID**
- *Wat:* component na docling-serve voor tekstaanvulling en PDF-annotaties met PyMuPDF, zonder GPU. GROBID (`grobid/grobid:0.8.2-crf`) op `verwerking`, aangeroepen voor wetenschappelijke PDF's; de verwijzingen gaan via een stap `cites` naar de bestaande code van Track V.
- *Hangt af van:* F2-17, F2-15.
- *Bestanden:* nieuw `services/langflow/components/signaalsysteem/pymupdf_aanvulling.py`, `grobid.py`; `docker-compose.signaalsysteem.yml`; `services/run_step_orchestrator.py`.
- *ADR:* 0015.
- *Stopconditie:* `services/langflow/tests/test_pymupdf_aanvulling.py` met een synthetische PDF met annotatie; `test_grobid.py` met een openbaar wetenschappelijk artikel (open licentie): verwijzingen geparsed en `cites` aangemaakt.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F4-10 Bestaande bronnen opnieuw inlezen**
- *Wat:* script dat de bronnen met een `pre_signaalsysteem`-versie toont met hun oorspronkelijke locatie; de beheerder leest ze in via de nieuwe flows of sluit ze uitdrukkelijk uit.
- *Hangt af van:* F4-02 tot en met F4-06, F4-09.
- *Bestanden:* nieuw `scripts/signaalsysteem/herinname_overzicht.py`.
- *ADR:* 0003, 0010.
- *Stopconditie:* het script meldt 0 bronnen waarvan de huidige versie `pre_signaalsysteem` is, op de bronnen na die op de lijst met uitsluitingen van de beheerder staan.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F4-11 Opruimen van de runs `pre-signaalsysteem`** — vraagt expliciete bevestiging van Geert
- *Wat:* apart script, geen migratie. Standaard een proefdraai met een lijst van wat verdwijnt; verwijderen alleen met `--bevestig` en na een reservekopie.
- *Hangt af van:* F4-10.
- *Bestanden:* nieuw `scripts/signaalsysteem/opruimen_pre_signaalsysteem.py`.
- *ADR:* 0003.
- *Stopconditie:* `tests/signaalsysteem/test_opruimen_pre_signaalsysteem.py` op een testdatabase: de proefdraai verandert niets; na verwijderen 0 runs met `pre-signaalsysteem` en geen fragment zonder run. Reservekopie aanwezig vóór de echte uitvoering.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F4-12 Toets fase 4**
- *Wat:* geplande runs op een vaste testfeed en een testwebsite op het testnetwerk: eerste run, tweede run zonder wijziging, derde run met één gewijzigd en één nieuw item.
- *Hangt af van:* F4-01, F4-02, F4-04.
- *Bestanden:* nieuw `scripts/signaalsysteem/toetsen/fase4.sh`.
- *ADR:* 0003, 0017; FO 15.
- *Stopconditie:* `fase4.sh` exitcode 0: tweede run `niets_nieuws` met 0 nieuwe bronversies en fragmenten; derde run precies 2 nieuwe bronversies.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

### Fase 5: kennisproducten

*Toets (F5-06):* elke bewering in een wiki-pagina verwijst naar een fragment dat haar draagt.

*Besluit vooraf:* waar de wiki-generatie draait (§7, conflict C5; §8, B12). De taken hieronder gaan uit van het voorstel daar.

**F5-01 `claude-cli`-container en aanroepwijze `cli`**
- *Wat:* container met een vaste versie van de Claude-CLI (Command Line Interface), alleen op `uitgaand`, zonder volumes met brongegevens, met een lege tijdelijke werkmap per aanroep. Een kleine aanroepdienst in de container voert `claude -p --output-format json --model <model>` uit, zonder tools en zonder hervatten van een sessie, en geeft CLI-versie en exitcode terug. `ModelGate` en de Langflow-modelcomponent kennen de aanroepwijze `cli` als cloudmodel. CLI-login als geheim.
- *Hangt af van:* F2-16, F2-10.
- *Bestanden:* nieuw `services/claude-cli/Dockerfile`, `aanroep.py`; `packages/llm-manager/src/llm_manager/gate.py`; `config/modelregister.yaml`; `docker-compose.signaalsysteem.yml`.
- *ADR:* 0006; FO 9.
- *Stopconditie:* `services/claude-cli/tests/test_aanroep.py`: elke aanroep in een nieuwe lege werkmap, zonder hervat-opties, met versie en exitcode; `packages/llm-manager/tests/test_model_gate.py` uitgebreid: `cli` geweigerd bij interne invoer; `tests/test_compose_signaalsysteem.py`: `claude-cli` hangt niet aan `verwerking` en heeft geen volumes behalve tmpfs.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F5-02 Wiki-pagina als kennisproduct**
- *Wat:* tabel voor wiki-pagina's met `is_derived = true`, classificatie (intern zodra één fragment intern is of de herkomst ontbreekt), beweringen met fragment-id's, status (`concept`, `review`, `gepubliceerd`, `verouderd`), run en model; leesroutes met classificatiebereik.
- *Hangt af van:* F2-05.
- *Bestanden:* `migrations/<n>.surrealql` en `_down`; nieuw `packages/shared/src/shared/models/wiki_page.py`; nieuw repository, dienst en router.
- *ADR:* 0004, 0001.
- *Stopconditie:* `test_migration_<n>_wiki_page.py`; `apps/app-main/tests/test_wiki_page_classificatie.py`; `test_zoeken_geen_afgeleide_teksten.py` uitgebreid: een pagina met de zoektekst komt niet terug.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F5-03 Wiki-generatie met controle**
- *Wat:* lus volgens FO 10 en 11: onderwerp kiezen, fragmenten ophalen (alleen fragmenten), pagina schrijven met een verwijzing per bewering, een tweede, onafhankelijke controle per bewering, stopconditie en maximum aantal rondes; afgekeurd naar wachtrij 5. Intern alleen met een lokaal model; de pagina krijgt dan `intern`.
- *Hangt af van:* F5-02, F5-01, F3-09.
- *Bestanden:* volgens het besluit in §8, B12; bij het voorstel nieuw `apps/app-main/src/app_main/graphs/wiki_generation.py` en een Langflow-component voor de cloudaanroep.
- *ADR:* 0004, 0006; FO 10, 11; nieuwe ADR volgens B12.
- *Stopconditie:* `apps/app-main/tests/test_graph_wiki_generation.py` met een nagebootst model: een bewering zonder dragend fragment leidt tot een nieuwe ronde en na het maximum tot review; een pagina op een intern fragment gebruikt nooit `api` of `cli`.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F5-04 Verouderde pagina's markeren**
- *Wat:* bij een nieuwe bronversie worden pagina's die op fragmenten van de oude versie steunen `verouderd`, met een item in wachtrij 5.
- *Hangt af van:* F5-02, F2-20.
- *Bestanden:* nieuwe dienst in `apps/app-main/src/app_main/services/`; `frontend/src/` (wachtrij 5).
- *ADR:* FO 10; 0003, 0016.
- *Stopconditie:* `apps/app-main/tests/test_wiki_verouderd.py`; frontendtest voor wachtrij 5.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F5-05 Uitbreiding van de Obsidian-export**
- *Wat:* wiki-pagina's in de export, programma's gekoppeld aan het codeboek, wachtrij 6 voor nieuwe personen in de werkbak. De vault-beheer-agent toetst de export buiten de fork.
- *Hangt af van:* F2-21, F5-02, F3-03.
- *Bestanden:* `services/langflow/components/signaalsysteem/obsidian_export.py`, `services/langflow/templates/`; `apps/app-main/src/app_main/api/routers/export_obsidian.py`; `frontend/src/` (wachtrij 6).
- *ADR:* 0018, 0016.
- *Stopconditie:* de tests van F2-21 uitgebreid met wiki-pagina's en programma's (byte voor byte gelijk bij herhaling, label intern in de frontmatter); frontendtest voor wachtrij 6.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F5-06 Toets fase 5**
- *Wat:* twee delen. Deterministisch: elke bewering van elke pagina heeft ten minste één fragment-id dat bestaat, binnen de classificatie van de pagina valt en bij de huidige of een uitdrukkelijk genoemde bronversie hoort. Door een mens: Geert beoordeelt een aselecte steekproef van beweringen op "het fragment draagt de bewering".
- *Hangt af van:* F5-03, F5-04.
- *Bestanden:* nieuw `scripts/signaalsysteem/toetsen/fase5_wiki.py`; `docs/signaalsysteem/meting-fase5.md` (uitkomst van de steekproef).
- *ADR:* FO 10, 15.
- *Stopconditie:* `fase5_wiki.py` exitcode 0 op het deterministische deel (100 %); steekproef van 50 beweringen met 0 niet-dragende fragmenten. De omvang van de steekproef is een voorstel (§8, B13).
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

### Aantallen en kritiek pad

| Fase | Taken | Waarvan verwijdertaken | Werkplek (GPU) |
| --- | --- | --- | --- |
| 2 | 25 | 4 (F2-02, F2-09, F2-22, F2-23) | F2-11, F2-17 (deels), F2-24, F2-25 |
| 3 | 12 | 1 (F3-08, met bevestiging) | F3-01 |
| 4 | 12 | 1 (F4-11, met bevestiging) | F4-03 |
| 5 | 6 | 0 | — |

Kritiek pad, met de taken die het meeste werk achter zich laten wachten:

- **Fase 2:** F2-01 → F2-03 → F2-04 → F2-13 → F2-14 → F2-16 → F2-17 → F2-19 → F2-24. Een tweede keten loopt parallel en sluit aan bij F2-16: F2-01 → F2-10 → F2-11 en F2-12. F2-07 en F2-08 moeten klaar zijn vóór F2-16 en F2-24.
- **Fase 3:** F3-01 → F3-04 → F3-12, met de referentieset F3-02 als lange pool: die hangt af van labelwerk door Geert. F3-02 kan starten zodra F2-17 werkt.
- **Fase 4:** F4-01 → F4-02 → F4-12. F4-07 wacht op een ADR.
- **Fase 5:** het besluit B12 → F5-01 → F5-03 → F5-06.

## 3. Harde eisen naar taken

Elke eis uit `opdracht.md` §1 met de taken waarvan de stopconditie haar toetst.

| Eis | Onderdeel | Taken |
| --- | --- | --- |
| 1. Intern blijft lokaal; geen intern fragment naar een cloudmodel of via MCP | Docker-netwerk | F2-10, F2-11, F3-10 (`netwerk.sh`, Compose-test) |
| | modelkeuze in de Langflow-componenten | F2-07, F2-16, F2-17, F3-04, F4-06, F5-01, F5-03 |
| | filter in de API van de fork | F2-03, F2-05, F3-09, F3-11 (P4) |
| | omzeilroutes dicht | F2-01, F2-02 |
| 2. Herkomst: elk fragment herleidbaar tot bronversie, run en model | | F2-04, F2-06, F2-14, F2-15, F3-05, F3-06, F3-09, F2-24 |
| 3. Afgeleide teksten tellen nooit als bron | | F2-05, F2-22 (INSIGHTS weg), F5-02, F5-03 |
| 4. Intern altijd lokaal in Docker; GPU via llama-swap; extern mag naar een externe dienst | | F2-08, F2-09 (geen Ollama), F2-11, F2-17 (adres per classificatie), F2-25, F3-01, F4-03, F2-24 |
| Geen interne documenten in repository of tests | | F2-05, F3-02 (schema met openbare URL), F2-21, F4-09 |
| Geen geheimen in code, Compose of commits | | F2-01, F2-10 (Compose-test op letterlijke geheimen), F3-10 |

## 4. Compose-opzet (schets)

Dit is een schets voor `docker-compose.signaalsysteem.yml`, nog geen echt bestand. Poorten tussen `<…>` en paden zijn voorstellen voor de bouwtaak. Diensten met een fase erachter komen in die fase.

### 4.1 Diensten en netwerken

| Dienst | Image of build | Netwerken | Poort op de host | GPU | Fase |
| --- | --- | --- | --- | --- | --- |
| `surrealdb` | `surrealdb/surrealdb:<vaste versie, B3>` | verwerking | — | nee | 2 |
| `open-notebook` | build `Dockerfile` (basis-tags vast) | verwerking | — | nee | 2 |
| `werkbak-toegang` | omgekeerde proxy, vaste tag (keuze bouwtaak) | verwerking, toegang | `127.0.0.1:<werkbak>` | nee | 2 |
| `langflow` | build `services/langflow` op `langflowai/langflow:1.13.0@sha256:<volledige digest>` | verwerking, uitgaand, toegang | `127.0.0.1:<editor>` | nee | 2 |
| `socket-proxy` | `tecnativa/docker-socket-proxy:v0.5.0` | beheer | — | nee | 2 |
| `llama-swap` | build `services/llama-swap` op `ghcr.io/mostlygeek/llama-swap:v268-cuda13-b11977` plus client uit `docker:29.9.2-cli` | verwerking, beheer | — | ja | 2 |
| `docling-serve` | `quay.io/docling-project/docling-serve-cu128:v1.38.2` | verwerking | — | ja | 2 |
| `kev` | build `services/kev` (`kev-1.0` op `python:3.13.18-slim-bookworm`) | verwerking | — | ja | 3 |
| `open-notebook-mcp` | build `services/open-notebook-mcp` (`python:3.12.16-slim-bookworm`, `open-notebook-mcp==0.3.0`, `mcp==1.30.0`) | verwerking | — | nee | 3 |
| `mcp-toegang` | omgekeerde proxy, vaste tag | verwerking, toegang | `127.0.0.1:<mcp>`, plus Tailscale op de host | nee | 3 |
| `reranker` | build `services/reranker` (CPU) | verwerking | — | nee | 3 |
| `whisperx` | `public.ecr.aws/deep-learning-containers/whisperx:3.8.6-cu128-amzn2023` | verwerking | — | ja | 4 |
| `grobid` | `grobid/grobid:0.8.2-crf` | verwerking | — | nee | 4 |
| `claude-cli` | build `services/claude-cli` (vaste CLI-versie) | uitgaand | — | nee | 5 |

`proeven.md` 1.7 noemt de Langflow-digest ingekort (`sha256:ff2716e0…e303f`); de bouwtaak zet de volledige digest vast.

### 4.2 Schets

```yaml
# docker-compose.signaalsysteem.yml (schets)
name: signaalsysteem

networks:
  verwerking: { internal: true }     # geen internet, geen bereikbare poort (P8 K1, K2)
  beheer:     { internal: true }     # alleen llama-swap en socket-proxy (P1)
  toegang:    {}                     # proxy's en langflow; enable_ip_masquerade "false" alleen na proef (ADR 0002 F)
  uitgaand:   {}                     # langflow (bronnen ophalen) en claude-cli

secrets:                             # bestanden in secrets/ (in .gitignore); bron file, niet environment (P1 §8.24)
  surreal_gebruiker:        { file: ./secrets/surreal_gebruiker }
  surreal_wachtwoord:       { file: ./secrets/surreal_wachtwoord }
  sleutel_beheren:          { file: ./secrets/sleutel_beheren }         # Langflow naar de fork
  sleutel_lezen_extern:     { file: ./secrets/sleutel_lezen_extern }    # open-notebook-mcp
  langflow_superuser:       { file: ./secrets/langflow_superuser }
  langflow_secret_key:      { file: ./secrets/langflow_secret_key }     # Fernet-sleutel
  docling_api_key:          { file: ./secrets/docling_api_key }
  kev_api_key:              { file: ./secrets/kev_api_key }             # fase 3
  claude_api_key:           { file: ./secrets/claude_api_key }          # alleen langflow
  claude_cli_login:         { file: ./secrets/claude_cli_login }        # fase 5

volumes:
  surreal_data: {}        # B3: bestaand volume of nieuw
  langflow_data: {}       # SQLite van Langflow (ADR 0017 D)
  checkpoints: {}         # LangGraph-checkpoints (SQLite)
  bronarchief: {}         # originelen van bronversies
  modellen: {}            # GGUF-bestanden, alleen-lezen voor llama-swap
  docling_artifacts: {}
  kev_hf: {}              # fase 3
  whisperx_hf: {}         # fase 4
  whisperx_torch: {}
  whisperx_nltk: {}

services:
  surrealdb:
    image: surrealdb/surrealdb:<vaste versie>
    networks: [verwerking]
    secrets: [surreal_gebruiker, surreal_wachtwoord]
    volumes: [surreal_data:/data]

  open-notebook:
    build: { context: ., dockerfile: Dockerfile }
    networks: [verwerking]
    environment:
      SURREAL_URL: ws://surrealdb:8000
      LLAMA_SWAP_URL: http://llama-swap:8080/v1
    secrets: [surreal_gebruiker, surreal_wachtwoord]
    volumes: [./config:/app/config:ro, checkpoints:/data/checkpoints]

  werkbak-toegang:                   # alleen localhost, niet via Tailscale (ADR 0016)
    image: <proxy>:<vaste tag>
    networks: [verwerking, toegang]
    ports: ["127.0.0.1:<werkbak>:80"]    # naar open-notebook:8502

  langflow:
    build: services/langflow
    networks: [verwerking, uitgaand, toegang]
    ports: ["127.0.0.1:<editor>:7860"]
    environment:
      DO_NOT_TRACK: "true"
      LANGFLOW_AUTO_LOGIN: "false"
      LANGFLOW_MCP_SERVER_ENABLED: "false"
      LANGFLOW_ADD_PROJECTS_TO_MCP_SERVERS: "false"
      LANGFLOW_COMPONENTS_PATH: /app/componenten
      LITELLM_LOCAL_MODEL_COST_MAP: "True"
      LANGFLOW_VARIABLES_TO_GET_FROM_ENVIRONMENT: CLAUDE_API_KEY
    secrets: [sleutel_beheren, langflow_superuser, langflow_secret_key, claude_api_key]
    volumes:
      - langflow_data:/app/data
      - bronarchief:/bronarchief
      - <inbox>:/inbox:ro
      - <watchfolder>:/watchfolder:ro
      - <vault>:/vault                  # alleen langflow schrijft in de vault
      - ./config:/config:ro             # modelregister, poortwachterregels

  socket-proxy:
    image: tecnativa/docker-socket-proxy:v0.5.0
    networks: [beheer]
    environment: { POST: 1, ALLOW_START: 1, ALLOW_STOP: 1, CONTAINERS: 0 }   # configuratie B (P7)
    volumes: ["/var/run/docker.sock:/var/run/docker.sock:ro"]

  llama-swap:
    build: services/llama-swap
    networks: [verwerking, beheer]
    environment:
      DOCKER_HOST: tcp://socket-proxy:2375
      NVIDIA_DISABLE_REQUIRE: "1"                                         # omweg zolang de driver CUDA 13.3 levert; vervalt in F2-25
      LD_LIBRARY_PATH: /usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
    volumes: [modellen:/modellen:ro, ./services/llama-swap/config.yaml:/app/config.yaml:ro]
    deploy: { resources: { reservations: { devices: [{ capabilities: [gpu] }] } } }

  docling-serve:                    # aangemaakt met --profile door-llama-swap create; start en stop door llama-swap
    image: quay.io/docling-project/docling-serve-cu128:v1.38.2
    container_name: signaal-docling
    profiles: [door-llama-swap]
    networks: [verwerking]
    environment:
      DOCLING_SERVE_ARTIFACTS_PATH: /artifacts
      DOCLING_SERVE_ENABLE_REMOTE_SERVICES: "false"
      DOCLING_SERVE_ENG_LOC_NUM_WORKERS: "1"
      DOCLING_SERVE_LOAD_MODELS_AT_BOOT: "false"
      DOCLING_SERVE_OPTIONS_CACHE_SIZE: "1"
      HF_HUB_OFFLINE: "1"
    secrets: [docling_api_key]      # startscript leest /run/secrets/docling_api_key
    volumes: [docling_artifacts:/artifacts:ro]
    deploy: { resources: { reservations: { devices: [{ capabilities: [gpu] }] } } }
    # geen restart

  # kev (fase 3), whisperx (fase 4): zelfde patroon als docling-serve
  # whisperx: OPT_OUT_TRACKING=true, HF_HUB_OFFLINE=1, volumes op /opt/models/hf, /opt/models/torch, /root/nltk_data
  # open-notebook-mcp, mcp-toegang, reranker (fase 3); grobid (fase 4); claude-cli (fase 5, alleen uitgaand, tmpfs)
```

### 4.3 llama-swap: groepen en geheugen

```yaml
# services/llama-swap/config.yaml (schets)
models:
  embed:      # mxbai-embed-large-v1 Q8_0, ChristianAzinn@3ec6d46a…, 1 024 dimensies
    cmd: llama-server --embeddings -c 512 -m /modellen/mxbai-embed-large-v1.Q8_0.gguf --port ${PORT}
  werkmodel:  # Qwen3.5-9B Q4_K_M (B4)
    cmd: llama-server -c 8192 -m /modellen/Qwen3.5-9B-Q4_K_M.gguf --port ${PORT}
    ttl: 60
  docling:
    cmd: /app/wacht_op_dienst.sh signaal-docling http://signaal-docling:5001/health
    cmdStop: docker stop signaal-docling
    proxy: http://signaal-docling:5001
    checkEndpoint: /health
    ttl: 300            # ruim boven de pollpauze van 10 s (ADR 0009)
  kev-4b:     # fase 3
    cmd: /app/wacht_op_dienst.sh signaal-kev http://signaal-kev:<poort>/openapi.json
    cmdStop: docker stop signaal-kev
    proxy: http://signaal-kev:<poort>
    checkEndpoint: /openapi.json
    ttl: 60
  whisperx:   # fase 4
    checkEndpoint: /ping
    ttl: 60
groups:
  zwaar: { swap: true, exclusive: true, members: [werkmodel, kev-4b, docling, whisperx] }
  licht: { persistent: true, members: [embed] }     # geen ttl: blijft geladen
```

De `ttl` van 60 s voor werkmodel, Kev en WhisperX is die van P1; de bouwtaak meet of die past bij de bundeling.

Videogeheugen per dienst (P3, 30-10; MiB; nulmeting 0; grens 16 303 − 1 024 = 15 279 MiB):

| Dienst | Piek | Rust | Met mxbai | Past |
| --- | --- | --- | --- | --- |
| mxbai-embed-large Q8_0, `-c 512` | 584 | 584 | — | ja |
| werkmodel Qwen3.5-9B Q4_K_M, context 8 192 | 5 696 | 5 696 | 6 279, gemeten | ja |
| werkmodel Qwen3.5-9B Q4_K_M, context 32 768 | 6 488 | 6 488 | 7 072, berekend | ja |
| Kev-4B bf16 | 14 574 | 14 574 | 15 157, gemeten | ja, marge 121 |
| docling-serve | 2 438–2 502 | 2 438–2 502 | 3 021, gemeten (1 herhaling) | ja |
| WhisperX variant A (`diarize=true`) | 10 200–10 328 | 8 874 | 10 912, berekend | ja |

Niet gemeten: mxbai + WhisperX (F4-03) en docling + WhisperX (komt in `zwaar` niet voor).

## 5. Interfaces

Alle eindpunten vragen `Authorization: Bearer <sleutel>`. Recht tussen haakjes. Fouten: 401 zonder geldige sleutel, 403 bij te weinig recht, 404 bij een onbekend of gefilterd record, 409 bij een conflict (classificatie, `rev`, afgesloten run), 422 bij een ongeldig verzoek.

### 5.1 Importeindpunt: `POST /api/v1/import` (beheren)

Verzoek:

```json
{
  "run_id": "processing_run:…",
  "source_map_rule_id": "source_map_rule:…",
  "batch_id": null,
  "source": { "external_id": "…", "type": "pdf", "location": "watchfolder/…", "title": "…" },
  "source_version": {
    "content_hash": "sha256:…",
    "fetched_at": "2026-11-02T09:00:00+01:00",
    "original": "bronarchief/…",
    "feed_item_id": null
  },
  "classification": {
    "value": "intern",
    "basis": "bronkaartregel | batch | beheerder | poortwachter | standaard",
    "gatekeeper": {
      "decision": "intern",
      "internal_markers": ["rubricering"],
      "method": "regel | kev",
      "probability": 0.97,
      "model": "kev-4b@6cfce5c2…"
    }
  },
  "extraction": { "extractor": "docling-serve", "version": "v1.38.2", "options": { "ocr_lang": ["nl", "en"] } },
  "structure": { "format": "docling_document", "content": { } },
  "chunks": [
    {
      "seq": 0,
      "text": "…",
      "section_path": ["Paragrafen", "Weerstandsvermogen en risicobeheersing"],
      "page": 12,
      "printed_page": "10",
      "element_type": "paragraph",
      "positions": [ { "page": 12, "bbox": [0, 0, 0, 0] } ],
      "doc_ref": "#/texts/12",
      "labels": [ { "schema": "iv3", "schema_version": "2025", "term": "0.4", "method": "kev", "probability": 0.83, "model": "kev-4b" } ]
    }
  ]
}
```

Antwoord `201` (nieuw) of `200` (ongewijzigd of herhaald verzoek):

```json
{
  "status": "nieuw | ongewijzigd",
  "source_id": "source:…",
  "source_version_id": "source_version:…",
  "version_no": 3,
  "run_id": "processing_run:…",
  "classification": "intern",
  "chunk_ids": ["chunk:…"]
}
```

Regels:

- `run_id` moet een run met status `bezig` zijn. Hetzelfde verzoek met dezelfde `run_id` geeft hetzelfde antwoord (idempotent).
- Bekende `source` + `content_hash`: geen nieuwe versie, `status = ongewijzigd`, de run wordt `niets_nieuws` bij afsluiten.
- Classificatie: `extern` alleen als de regel of batch `extern` zegt en de poortwachter geen interne kenmerken vond; anders `intern`. Een import met `extern` voor een bestaande interne bron geeft 409. Ontbreekt de classificatie, dan geldt `intern`.
- Elk fragment krijgt de classificatie, bronversie, run en extractor van het verzoek. Embeddings zitten niet in het verzoek: die berekent de stap `embed` (ADR 0011).
- `labels` mag leeg zijn (fase 2).

### 5.2 Bronkaartregel

Velden volgens FO 5, met de standaarden uit ADR 0019. De opslag in de fork is een voorstel (§8, B2).

| Veld | Type | Verplicht | Standaard en regels |
| --- | --- | --- | --- |
| `id` | record-id | ja | |
| `name` | tekst | ja | |
| `route_type` | `api`, `mcp`, `feed`, `website`, `bulkupload`, `watchfolder`, `upload` | ja | bepaalt de standaard hieronder |
| `source_type` | `pdf`, `map`, `website`, `api`, `video`, `podcast`, `mcp` | ja | |
| `location` | tekst | ja | pad, URL of API-eindpunt |
| `schedule` | `{ cron, timezone, catch_up }` of leeg | nee | `timezone` `Europe/Amsterdam`; `catch_up` `coalesce` of `skip` |
| `classification` | `intern`, `extern` | ja | `api`, `mcp`, `feed`, `website`: voorstel `extern`, de beheerder bevestigt; `bulkupload`: keuze per batch, zonder keuze `intern`; `watchfolder`: geen standaard, tot besluit `intern`; `upload`: voorstel van de poortwachter, zonder keuze `intern` |
| `classification_confirmed_by`, `_at` | sleutelhouder, tijd | bij `extern` | `extern` zonder bevestiging geldt als `intern` |
| `pipeline_start` | `automatisch`, `na_bevestiging`, `wachtrij` | afgeleid | `api`/`mcp`/`feed`/`website` automatisch; `bulkupload` volgens de batchroute; `upload` na bevestiging; `watchfolder` wachtrij |
| `document_type` | lijst | nee | bijvoorbeeld jaarrekening, begroting |
| `route` | flow-id in Git | ja | |
| `selection` | lijst | nee | leeg is alles |
| `label_schema` | `{ schema, version }` | nee | verwijst naar een codeboekversie (ADR 0013) |
| `owner` | tekst | ja | standaard beheerder |
| `fetch_content` | `metadata`, `transcript`, `volledige_media`, `volledig_document` | ja | YouTube standaard `metadata` (FO 5) |
| `active` | boolean | ja | `true` |

Een batch (`intake_batch`) heeft `classification` (standaard `intern`), `route`, `created_by` en de lijst met bestanden. Een wachtrij-item (`intake_item`) heeft pad, hash, voorstel van de poortwachter, status (`wacht`, `besloten`, `verwerkt`) en het besluit van de beheerder (route en classificatie).

### 5.3 Classificatiefilter in de API

| Recht | `ClassificationScope` | Ziet | Routes (kern) |
| --- | --- | --- | --- |
| `lezen_extern` | `EXTERN` | alleen `extern` | `GET /api/notebooks[/{id}]`, `GET /api/sources[/{id}]`, `GET /api/notes[/{id}]`, `POST /api/search`, en vanaf F3-09 de herkomst- en versie-eindpunten |
| `lezen_intern` | `ALLES` | alles | dezelfde leesroutes zonder filter, plus `GET /api/v1/export/obsidian/entities` |
| `beheren` | `ALLES` | alles | de rest, waaronder import, runs, stappen, herclassificatie, sleutels, triage, chat, ask, podcasts |

- De middleware leidt het bereik af uit de sleutel; de router geeft het door. In de repositorylaag is het een verplicht argument zonder standaard, bijvoorbeeld `async def get(self, source_id: str, *, scope: ClassificationScope) -> Source`. Een aanroep zonder bereik faalt.
- `fn::text_search($query, $limit, $classifications, $history)` en `fn::vector_search($vector, $limit, $min_score, $classifications, $history)` lezen alleen `chunk` en `source_embedding` en filteren in de `WHERE`-clausule op `classification IN $classifications` en, zonder `$history`, op de huidige bronversie en run.
- Een gefilterd record geeft 404, niet 403. Schrijftools van open-notebook-mcp krijgen 403.
- Herclassificatie: `POST /api/v1/sources/{id}/classification` (beheren), `{ "to": "extern", "reason": "…" }`, één regel in `classification_log`.

### 5.4 Contract tussen Langflow en de fork

Langflow gebruikt één sleutel met `beheren` en praat alleen met de API van de fork (`http://open-notebook:5055`) en met llama-swap (`http://llama-swap:8080`).

| Eindpunt | Verzoek | Antwoord |
| --- | --- | --- |
| `POST /api/v1/runs` | `trigger` (`schema`, `mapcontrole`, `feed`, `handmatig`, `herverwerking`), `pipeline_id`, `pipeline_version` (`{ git_commit, flow_hash }`), `source_map_rule_id`, `label_schema` | `{ run_id, status: "bezig" }` |
| `PATCH /api/v1/runs/{id}` | `status`, `ended_at`, `log[]` (tijd, stap, status, melding), `models[]` (naam, versie, aanroepwijze; bij `cli` ook versie en exitcode), `error` | run; 409 als de run al is afgesloten |
| `POST /api/v1/import` | §5.1 | §5.1 |
| `POST /api/v1/runs/{run_id}/steps/{stap}` | `stap` ∈ `embed`, `entity_extraction`, `entity_dedup`, `graph` (fase 4 ook `cites`); `{ source_version_id }` | `{ step_id, job_id, status: "wachtrij" }`; dezelfde stap op dezelfde versie in dezelfde run geeft dezelfde `step_id` |
| `GET /api/v1/runs/{run_id}/steps/{stap}` | — | `{ status: wachtrij | bezig | geslaagd | mislukt | wacht_op_review, attempts, error, review_item }` |
| `POST /api/v1/model-calls` | tijd, `run_id`, model, aanroepwijze, taak, classificatie, tokens, duur, status | 201; nooit prompt- of antwoordtekst |
| `POST /api/v1/intake/items`, `GET /api/v1/intake/items?status=besloten`, `PATCH /api/v1/intake/items/{id}` | wachtrij-item uit de mapcontrole of een losse upload; ophalen van besloten items; status `verwerkt` | item |
| `GET /api/v1/source-map-rules` | — | bronkaartregels voor planning en route |
| `GET /api/v1/codeboek/{schema}/{versie}` | — | keuzelijst voor Kev (fase 3) |
| `GET /api/v1/export/obsidian/entities?since=` | — | exportgegevens (ADR 0018) |

Gedrag:

- **Volgorde per bron:** run aanmaken → registratie (hash) → poortwachter → extractie → fragmentatie → (fase 3: labels) → import → stappen → run afsluiten. Ook bij `ongewijzigd` wordt de run afgesloten, als `niets_nieuws`.
- **Volgen:** de component vraagt de stapstatus op met oplopende pauze (beginnend bij 5 s). Herhaalpogingen van de job doet de worker van de fork; de component start een mislukte stap niet zelf opnieuw.
- **Herhaalbeleid van de component:** bij 502, 429, 503 en verbindingsfouten opnieuw met pauze, tot een maximum; daarna `mislukt` met een logregel. Bij een geheugenfout van llama-swap zet de component de taak opnieuw in de rij en logt de fout (FO 12.2).
- **`wacht_op_review` (voorstel, B5):** de component stopt met volgen en sluit de flow af met een logregel; de run blijft `bezig`. Na het triagebesluit hervat de fork de graaf (ADR 0012). Een geplande flow `hervatten` haalt runs op waarvan een stap van `wacht_op_review` naar `geslaagd` ging, en start de resterende stappen. Zo blijft Langflow elke stap starten (ADR 0011).
- **Innamewachtrij (voorstel):** Langflow meldt nieuwe watchfolder-documenten en losse uploads aan als wachtrij-item, en haalt besloten items op. De fork roept Langflow niet aan.
- **Modelkeuze:** de component leest `config/modelregister.yaml` (alleen-lezen gekoppeld) en weigert een model met `toegestaan voor intern = nee` bij een interne of onbekende bron, zowel bij verbinden als bij uitvoeren.

## 6. Risico's

| Nr | Risico | Gevolg | Maatregel | Taak |
| --- | --- | --- | --- | --- |
| R1 | **Werkgeheugen van de host.** P3 stopte na 28 van 36 metingen bij 1,2 GB vrij (WSL-VM 13,2 GB); per dienst niet gemeten. | Containers of de VM vallen weg bij zware combinaties; metingen onvolledig. | Geheugengrens van de WSL-VM vastleggen; vrij werkgeheugen meten in elke GPU-controle met een ondergrens (voorstel 2 GB); `mem_limit` per dienst; de ontbrekende combinaties alsnog meten. | F2-11, F2-24, F4-03 |
| R2 | **Marge van 121 MiB naast Kev-4B.** Kev-4B met mxbai 15 157 MiB, grens 15 279 MiB, bij een nulmeting van 0. | Het bureaublad van Windows of een andere toepassing vreet de marge op; geheugenfout. | Nulmeting in elke controle; reranker niet op de GPU (ADR 0015); terugval Kev-0,8B na besluit Geert. | F3-01 |
| R3 | **CUDA-driver 13.3 tegenover 13.4.** Het llama-swap-image eist `cuda>=13.4`; driver 610.88 levert 13.3. | Werkt alleen met `NVIDIA_DISABLE_REQUIRE=1` en `LD_LIBRARY_PATH`; een driver- of image-update kan dat breken. | Geert werkt de driver bij naar CUDA ≥ 13.4; F2-25 haalt de omweg weg en toetst dat. Tot dan beide variabelen in Compose met commentaar, en een GPU-controle na elke driver- of image-update. | F2-11, F2-25 |
| R4 | **`ttl` van docling.** llama-swap telt `ttl` vanaf de laatste aanvraag; een gestopte docling-serve verliest zijn taken (P1). Niet beproefd: een aanvraag voor een andere zware dienst tijdens een lopende docling-taak. | Taken gaan verloren, of docling stopt bij een wissel. | Pollen elke 10 s, `ttl` 300 s, opnieuw indienen bij 404 met maximum; bundelen per dienst; de toets van fase 2 doet precies deze combinatie. | F2-17, F2-24 |
| R5 | **Langflow belt naar buiten.** LiteLLM deed 21 uitgaande pogingen (P2); Langflow heeft bewust internet en ziet interne inhoud vóór de poortwachter. | Gegevens kunnen het systeem verlaten via een component of bibliotheek. | `LITELLM_LOCAL_MODEL_COST_MAP=True` en `DO_NOT_TRACK=true`; telling 0 als stopconditie; blijven er pogingen, dan ophalen in een eigen container en Langflow zonder internet (FO 12.1). | F2-12 |
| R6 | **MCP-versiebinding.** open-notebook-mcp 0.3.0 vraagt `mcp>=1.23.2` zonder bovengrens; het officiële startpunt faalt met `mcp==1.30.0`; het pakket is van een derde partij. | Een update breekt de server of verandert tools. | Beide pakketten met `==`; eigen startscript; elke update eerst P4. | F3-10, F3-11 |
| R7 | **Referentieset van 200–300 fragmenten.** Bestaat nog niet; vraagt labelwerk door Geert; alleen openbare bronnen, terwijl de poortwachter interne kenmerken moet herkennen. | De toets van fase 3 en de drempel van de poortwachter schuiven op. | F3-02 vroeg starten; synthetische interne kenmerken op openbare tekst; schemacontrole als stopconditie. | F3-02, F3-04, F3-12 |
| R8 | **Socket-proxy valt later uit.** ADR 0009 optie 2 is besloten (Geert, 30-10); optie 3 (geen Docker-toegang) blijft alleen de terugval. | Bij terugval: docling-serve en WhisperX altijd aan, 2,4 + 8,9 GB in rust; Kev-4B (14,6 GB) past dan niet; gedeelde vergrendeling in de Langflow-componenten nodig. | Alleen bij uitval: F2-11 zonder socket-proxy, een extra taak voor de vergrendeling, Kev-0,8B in F3-01, WhisperX in F4-03 buiten llama-swap; FO 12.2 en het plan aanpassen. | F2-11, F3-01, F4-03 |
| R9 | **Restrisico socket-proxy.** Elke container is op naam te starten en te stoppen, ook `surrealdb` (P7 K2); 502 na een externe stop. | Verstoring, geen toegang tot gegevens. | Socket-proxy alleen op `beheer`; herhaalpoging bij 502 in `ModelGate` en componenten. | F2-07, F2-11, F2-16 |
| R10 | **Migraties op bestaande gegevens.** `entity` is mogelijk schemaloos in productie (aanname A6 verschilanalyse); de huidige SurrealDB-versie achter `v2` is onbekend. | Migratie 88 en verder falen of geven andere uitkomsten dan in de test. | Eerst op een kopie draaien; reservekopie (er is een kopie van 25-09-2026); versie vastzetten na B3. | F2-01 t/m F2-06 |
| R11 | **Verborgen afhankelijkheden bij verwijderen.** De interface en tests gebruiken routes die vervallen (bijvoorbeeld `frontend/src/lib/api/services.ts`). | Breuk buiten de taak. | Verwijdercontrole V, inclusief frontendbuild. | F2-02, F2-09, F2-22, F2-23 |
| R12 | **Volledigheid van vector-zoeken met filter.** Filter en rangschikking in één `WHERE`; met een index op dichtstbijzijnde buren kan het filter na de index komen. | Minder treffers; raakt eis 1 niet. | Meting in F2-05. | F2-05 |
| R13 | **Fragmenten boven 512 tokens** bij mxbai. | Afgekapte embeddings. | Keuze opdelen of afkappen, met test. | F2-08 |
| R14 | **OCR op de CPU** in docling-serve (RapidOCR zonder CUDA). | Grote gescande documenten zijn traag; grotere kans op R4. | `ttl` en pollen volgens R4; externe docling alleen voor extern en pas met een eigen ADR (ADR 0019). | F2-17 |
| R15 | **Ophalen van images:** `public.ecr.aws` gaf anoniem `toomanyrequests`. | Vertraging bij installatie. | Herhaalpogingen in het downloadscript; images lokaal bewaren. | F2-11, F4-03 |

## 7. Aannames en conflicten

### 7.1 Aannames

Feiten in dit plan komen uit de code van de fork (gecontroleerd op 30-10: bestandspaden, `migrations/87.surrealql` als hoogste, `config/` met alleen `triage_config.json`, geen `secrets/` in `.gitignore`, `Dockerfile` met `EXPOSE 8502 5055` en zwevende basis-tags), uit de ADR's of uit `proeven.md`. Niet geverifieerd:

- **A1.** Compose-geheimen met bron `file` blijven aanwezig als llama-swap een container met `docker start` start. P1 gebruikte bron `environment`, die dat niet doet. Te toetsen in F2-11.
- **A2.** De interface draait in de container `open-notebook` op poort 8502 en is via een omgekeerde proxy bereikbaar.
- **A3.** Langflow-componenten, flows en scripts staan in `services/langflow/`, en het modelregister en de poortwachterregels in `config/`. ADR 0017 laat de plek aan dit plan.
- **A4.** De bronkaart, batches en de innamewachtrij staan in de fork (SurrealDB), met API. Geen ADR beslist dit; ADR 0017 C gaat er voor de terugval van uit ("leest de schema's uit de bronkaart via de API van de fork").
- **A5.** De triggers-API van Langflow 1.13.0 kan per bronkaartregel een trigger aanmaken (P2 toonde aanzetten via de API, niet aanmaken via de API).
- **A6.** Het effect van `enable_ip_masquerade: "false"` onder Docker Desktop is onbekend (ADR 0002 F).
- **A7.** De vindplaats en vorm van de openbare Iv3-lijst 2025 en van de BWB-open data zijn niet nagegaan (ADR 0013).
- **A8.** Een kleine aanroepdienst in de container `claude-cli` is nodig, omdat Langflow geen Docker-toegang heeft om de CLI zelf te starten.

### 7.2 Conflicten en inconsistenties

Vastgelegd, niet opgelost.

- **C1.** FO 11 noemt nog `LANGFLOW_LOAD_FLOWS_PATH` als laadmechanisme; ADR 0017 A2 (besloten) laat dat vervallen. Het FO is op dit punt niet bijgewerkt.
- **C2.** ADR 0001 verwijdert `private`, `privacy_mode` en `default_privacy_mode` in dezelfde migratie die `classification` toevoegt. Het plan verdeelt dat over F2-03 en F2-09, zodat de code tussen de taken blijft werken. De eindstand is gelijk.
- **C3.** FO 13.2 punt 7 laat chat en podcasts ongewijzigd; ADR 0006 laat chat via `ModelGate` naar een lokaal model gaan. Chat verandert dus wel.
- **C4.** ADR 0002 D wil de querylogica van de MCP-tools hergebruiken voor de herkomsteindpunten; F2-02 verwijdert de tools in fase 2 en F3-09 bouwt de eindpunten in fase 3. Volgens `ARCHITECTURE.md` §10 zit de logica van `search` en `related` in `SearchRepository` en `GraphRepository`; die blijven. Voor `cite` en `get_node` is dat niet nagegaan.
- **C5.** FO 11 plaatst de wiki-generatie als LangGraph-lus in de fork, en FO 9 noemt de Claude-CLI geschikt voor wiki-generatie. Volgens ADR 0006 en FO 12.1 heeft de fork geen internet en roept hij alleen modellen via llama-swap aan; de Claude-API en -CLI zijn alleen vanuit Langflow bereikbaar. Een wiki-lus in de fork kan dus geen cloudmodel gebruiken. Dit vraagt een ADR vóór F5-03 (B12).
- **C6.** FO 15 noemt voor fase 2 de lokale map; volgens ADR 0019 wacht een watchfolder op de beheerder. Fase 2 heeft daardoor de werkbak met de innamewachtrij nodig (F2-20). Dat is een gevolg, geen tegenspraak.

## 8. Open vragen en besluiten

### 8.1 Vóór fase 2 te beslissen door Geert

| Nr | Vraag | Voorstel | Blokkeert |
| --- | --- | --- | --- |
| B1 | ADR 0009, Docker-toegang optie 2 met configuratie B: **besloten** (Geert, 30-10), met de restrisico's uit R9. Geert werkt de driver bij naar CUDA ≥ 13.4 (F2-25). | — | — |
| B2 | Waar staat de bronkaart: tabel in de fork met API, of YAML in Git zoals het modelregister? Geen ADR beslist dit. | Tabel in de fork (A4), vast te leggen in een nieuwe ADR. | F2-13, F2-14, F4-01 |
| B3 | Database: de bestaande gegevens migreren in het bestaande volume (ADR 0001 en 0003 gaan daarvan uit), of een nieuwe lege database? En welke SurrealDB-versie wordt vastgezet (nu zwevend `v2`)? | Bestaand volume, na een kopie; versie gelijk aan wat nu draait. | F2-01, F2-03, F2-10 |
| B4 | Werkmodel en context voor gebruik, niet alleen voor de proef. | Qwen3.5-9B Q4_K_M, context 8 192 (P1, P3). | F2-07, F2-11 |
| B5 | Akkoord op het contractvoorstel voor `wacht_op_review` en de richting van de innamewachtrij (§5.4). | Akkoord. | F2-15, F2-16, F3-07 |
| B6 | Vervangen van `docker-compose.yml` aan het eind van fase 2 (ADR 0002): de bestaande omgeving stopt dan; volumes blijven. | Akkoord, na een geslaagde F2-24. | F2-23 |
| B7 | Plaats van de Langflow-code en de configuratie (A3). | `services/langflow/` en `config/`. | F2-12 |
| B8 | Map voor programma's in de vault (ADR 0018 laat dat aan de vault-beheer-agent). | Vóór F2-21 vastleggen. | F2-21 |

### 8.2 Later te beslissen

| Nr | Vraag | Vóór |
| --- | --- | --- |
| B9 | Wie labelt de referentieset, uit welke openbare bronnen, en hoe worden interne kenmerken nagebootst? | F3-02 |
| B10 | Wanneer is een entiteit zichtbaar voor `lezen_extern` (open punt ADR 0002)? | F3-09 |
| B11 | Tool en plek voor de scènedetectie bij video (nieuwe ADR). | F4-07 |
| B12 | Waar draait de wiki-generatie, gegeven C5? Voorstel: de lus en de controle als LangGraph in de fork met lokale modellen; een generatie met de Claude-API of -CLI voor externe onderwerpen als Langflow-component die de lus van buiten aanroept (nieuwe ADR). | F5-01, F5-03 |
| B13 | Omvang van de steekproef in de toets van fase 5. Voorstel: 50 beweringen. | F5-06 |
| B14 | Plek van de reranker onder llama-swap (ADR 0015); tot een meting blijft hij op de processor (CPU). | na F3-10 |
