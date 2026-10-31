# Plan van aanpak signaalsysteem

Stand: 30 oktober 2026, bijgewerkt na de review · stap 4 van `opdracht.md` · branch `docs/signaalsysteem-fo`

Dit plan deelt de fasen 2 tot en met 5 uit het functioneel ontwerp (FO, paragraaf 15) op in taken. Het steunt op het FO, de verschilanalyse, de ontwerpbesluiten (Architecture Decision Records, ADR's) 0001–0022 en de proeven P0–P10 in `proeven.md`. Er is geen code gewijzigd. Verwerkt zijn de besluiten van Geert van 30-10 (B2–B8, Kev-4B gekwantiseerd, de referentieset), de reviewbevindingen en de ADR-wijzigingen van 30-10 (commit `1e42b05e`): rechten `pijplijn`, `ophalen` en `beheerder`, een netwerk per proxy, de afscherming van Langflow (ADR 0022) en de proef met SurrealDB v3 (ADR 0021, P10).

Inhoud: 1 werkwijze · 2 taken per fase · 3 harde eisen en FO-dekking naar taken · 4 Compose-opzet · 5 interfaces · 6 risico's · 7 aannames en conflicten · 8 open vragen en besluiten.

## 1. Werkwijze

- **Agent.** Elke taak bouwt `signaal-bouwer`, één taak tegelijk; daarna toetst `signaal-reviewer` de taak aan FO, ADR's en harde eisen. Een taak is af als de stopconditie slaagt en de reviewer geen blokkerende bevinding heeft. Daarna volgt één regel in `voortgang.md` en een kleine commit.
- **Volgorde.** Fase 2 begint met de proef P10 (F2-26) en de taken die de harde eisen borgen (F2-01 tot en met F2-06): afscherming, classificatie en bronversie met run en herkomst. Daarna modelpoort en omgeving, en pas dan de inname.
- **Tests.** Tests staan naast de code (`ARCHITECTURE.md` §5) en draaien met `uv run --project <lid> pytest <lid>/tests/<bestand>`. Op de werkplek van Geert gaat dat via de `.venv` in WSL (Windows Subsystem for Linux). Tests over de hele repository staan in `tests/` in de root, zoals het bestaande `tests/test_compose_env_is_consumed.py`.
- **Migraties.** Elke migratie is idempotent (`DEFINE … OVERWRITE` of `IF NOT EXISTS`) en heeft een `_down`-variant. De nieuwe omgeving start met een lege database (B3): migraties definiëren schema's en zetten geen oude gegevens om. De hoogste migratie is nu 87 (`migrations/87.surrealql`); de nummers hieronder zijn een verwachting op volgorde van de taken. Elke migratietaak heeft een test `packages/surrealdb-service/tests/test_migration_<n>_<onderwerp>.py` die heen en terug draait, zoals `test_migrations_roundtrip.py`. De syntaxis volgt de SurrealDB-versie uit F2-26 (ADR 0021).
- **Verwijdertaken.** Een taak die bestaande code verwijdert, staat apart en heeft altijd de **verwijdercontrole (V)**:
  1. de volledige testsuite van elk geraakt workspace-lid slaagt, na het verwijderen van de tests van de verwijderde code;
  2. `python -c "import app_main.api.app"` slaagt, en de API start in de Compose-omgeving met `/health` HTTP 200;
  3. een zoektest `tests/test_geen_verwijzingen_<onderwerp>.py` faalt zodra een verwijderd module- of routenaam nog voorkomt in `apps/`, `packages/`, `pipelines/`, `services/` of `frontend/src/`;
  4. raakt de taak routes die de interface gebruikt: `npm --prefix frontend run test` en `npm --prefix frontend run build` slagen.
- **Bevestiging.** Een taak die bestaande gegevens weggooit, draagt de markering **vraagt expliciete bevestiging van Geert**. De bouwer start die taak pas na die bevestiging in het gesprek, en maakt vooraf een reservekopie van de database.
- **Voorwaardelijke taken.** Een taak met de markering **voorwaardelijk** start alleen als de genoemde voorwaarde is vervuld (F2-25, F2-32).
- **Testdata.** Alleen openbare bronnen (tabel 1.6 in `proeven.md`) of synthetische tekst. Grote bestanden worden gedownload met controle van de SHA-256 en niet in Git gezet.
- **Werkplek.** Een stopconditie die de GPU (grafische kaart) nodig heeft, is een script onder `scripts/signaalsysteem/controle/` of `scripts/signaalsysteem/toetsen/`, met dezelfde exitcodes als de proeven (0 geslaagd, 1 criterium niet gehaald, 2 omgeving ongeschikt, 3 scriptfout). Zulke taken zijn "uit te voeren op de werkplek".
- **Rechten.** In dit plan: `lezen_extern`, `lezen_intern`, `pijplijn` (Langflow), `ophalen` (de container `ophalen`) en `beheerder` (een persoon), volgens ADR 0002 B. Het vroegere `beheren` bestaat niet meer.
- **Taak-id.** `F<fase>-<nr>`, zoals de startopdracht `/signaalsysteem taak F2-03` verwacht. Nieuwe taken na de review krijgen een volgnummer achteraan in hun fase; de volgorde van uitvoeren volgt de afhankelijkheden, niet het nummer.

## 2. Taken

### Fase 2: minimale keten

Doel volgens FO 15: PDF en lokale map, docling-serve, fragmentatie, importeindpunt, opslag met versie, herkomst en classificatie, en de export naar Obsidian (op een tijdelijke testvault, B8), vanaf het begin in Docker met llama-swap. *Toets (F2-24):* elk fragment is herleidbaar tot bronversie en run, en twee gelijktijdige GPU-taken leiden niet tot een geheugenfout.

#### 0. Databaseversie

**F2-26 Proef P10: SurrealDB v3**
- *Wat:* het script `scripts/signaalsysteem/proeven/p10_surrealdb_v3.sh` met hulpscripts in `p10/` schrijven en uitvoeren volgens `proeven.md` P10: alle migraties en hun `_down`-varianten en de databasetests van de fork op een lege `surrealdb/surrealdb:v3.3.5`, met `v2.7.0` als nulmeting, eerst zonder en daarna met alleen mechanische vervangingen uit de migratiegids, op een kopie in de runmap. De repository verandert niet. Daarna beslist Geert volgens de beslisregel van ADR 0021.
- *Hangt af van:* — . Vooraf stelt Geert de grens in K5 vast (§8, B18).
- *Bestanden:* nieuw `scripts/signaalsysteem/proeven/p10_surrealdb_v3.sh`, `scripts/signaalsysteem/proeven/p10/`; `docs/signaalsysteem/proeven.md` (uitkomst); `docs/signaalsysteem/adr/0021-surrealdb-v3.md` (besluit, door de ontwerper na Geert).
- *ADR:* 0021.
- *Stopconditie:* `p10_surrealdb_v3.sh` eindigt met exitcode 0 of 1 (2 of 3 telt niet als uitgevoerd) en schrijft `uitkomst.json` met de meetwaarden van K1–K6 en `p10_patronen.csv`; K1 (nulmeting op v2.7.0) geslaagd; de uitkomst staat in `proeven.md` P10; het besluit van Geert staat in ADR 0021.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-27 Overstap naar SurrealDB v3** — na F2-24
- *Voorwaarde:* P10 is niet geslaagd (30-10, ADR 0021): `v2.7.0` blijft. Deze taak is de aparte overstaptaak na F2-24, met de lijst uit P10 (`p10_patronen.csv`, 41 migratiebestanden met niet-mechanische aanpassingen); daarna P10 opnieuw.
- *Wat:* de mechanische vervangingen uit P10 10d in `migrations/` en in de Python-code (`type::thing` naar `type::record`, `type::is::*` naar `type::is_*`, `string::is::*` naar `string::is_*`, `SEARCH ANALYZER` naar `FULLTEXT ANALYZER`, `LET`), ook in de migraties die F2-01 al heeft toegevoegd; de niet-mechanische aanpassingen uit P10; `surrealdb` in `packages/surrealdb-service/pyproject.toml` op de SDK-versie waarmee P10 slaagde. Zie conflict C9 over de vorm (eigen migratie per vervanging tegenover aanpassen van bestaande migraties).
- *Hangt af van:* F2-26, F2-24.
- *Bestanden:* `migrations/*.surrealql` volgens de lijst uit P10; de Python-bestanden uit `p10_patronen.csv` (39 bestanden, waarvan 12 broncode, telling ADR 0021); `packages/surrealdb-service/pyproject.toml`; `uv.lock`; nieuw `tests/test_geen_surrealql_v2_patronen.py`.
- *ADR:* 0021.
- *Stopconditie:* `test_migrations_roundtrip.py` slaagt tegen `v3.3.5` (alle migraties, alle `_down`-varianten, en opnieuw alle migraties); de volledige testsuites van `packages/surrealdb-service` en `apps/app-main` slagen tegen `v3.3.5`; `tests/test_geen_surrealql_v2_patronen.py` faalt bij `SEARCH ANALYZER`, `type::thing(`, `type::is::`, `string::is::` of `MTREE` in `migrations/`, `apps/`, `packages/` of `pipelines/`.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

#### A. Harde eisen borgen

**F2-01 Sleutels en rechten per route, standaard weigeren**
- *Wat:* `PasswordAuthMiddleware` en `OPEN_NOTEBOOK_PASSWORD` vervallen. Eén sleutelmiddleware voor alles onder `/api`, met `Authorization: Bearer`; `X-API-Key` vervalt. Rechten `lezen_extern`, `lezen_intern`, `pijplijn`, `ophalen` en `beheerder` volgens ADR 0002 B; `beheerder` heeft een verplicht veld `person`. `ophalen` mag alleen de ophaalcontrole (F4-13) en heeft een leeg bereik. Elke route declareert haar minimale recht met `require_right(...)`. `AgentAuditMiddleware` geldt voor alle routes. Uitgezonderd zijn alleen `/health` en `/api/auth/status`. Nieuw eindpunt `GET /api/v1/auth/scope` (elke geldige sleutel behalve `ophalen`, ADR 0002 B) dat alleen het recht en het `ClassificationScope` van de aanroepende sleutel teruggeeft, voor de startcontrole van open-notebook-mcp (F3-10) en de gateway (F5-07). Een migratie definieert `agent_keys.permission` met de vijf waarden en `person`; er zijn geen bestaande sleutels om in te trekken (lege database, B3). Een script maakt de eerste sleutel met `beheerder` en de naam van de persoon, en toont die één keer, zonder log. Hetzelfde script maakt op verzoek de dienstsleutels (`pijplijn`, `lezen_extern`, later `ophalen` en die van de gateway) als geheimbestand in `secrets/`.
- *Hangt af van:* —
- *Bestanden:* `apps/app-main/src/app_main/api/auth.py`, `agent_auth.py`, `app.py`; alle routers in `apps/app-main/src/app_main/api/routers/` (dependency per route); `apps/app-main/src/app_main/services/agents/key_service.py`; `migrations/88.surrealql` en `88_down.surrealql`; nieuw `scripts/signaalsysteem/maak_sleutel.py`.
- *ADR:* 0002 A, B; 0001.
- *Stopconditie:* `apps/app-main/tests/test_route_rechten.py` loopt alle `application.routes` af en faalt bij een route zonder rechtendeclaratie, en toetst dat een sleutel `ophalen` op elke route behalve de ophaalcontrole HTTP 403 krijgt; `apps/app-main/tests/test_sleutel_verplicht.py`: elk verzoek zonder geldige sleutel geeft HTTP 401, behalve de twee uitzonderingen; `apps/app-main/tests/test_rechten_pijplijn.py` (nieuw, groeit mee met F2-03, F2-13 en F2-30): een sleutel `pijplijn` krijgt HTTP 403 op sleutelbeheer (`/api/agent-keys`) en op het triagebesluit (`api/routers/triage.py`); `apps/app-main/tests/test_auth_scope.py`: het eindpunt geeft per recht het juiste bereik en geen andere gegevens; `packages/surrealdb-service/tests/test_migration_88_agent_key_permission.py`: `read` wordt geweigerd, de vijf nieuwe waarden worden aanvaard, `beheerder` zonder `person` wordt geweigerd, `_down` draait terug en de migratie is twee keer na elkaar uit te voeren. Bestaand `test_agent_auth.py` aangepast en groen.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.
- *Let op:* de interface kan tot F2-20 niet inloggen.

**F2-02 Verwijderen: routes en diensten die het filter omzeilen**
- *Wat:* weg gaan de router `services_proxy`, de drie MCP-servers (Model Context Protocol) in de fork en de extractiedienst. Daarmee vervalt PC.7. Vooraf verhuist `services/extraction/vocabulary_manager.py` naar `packages/shared/src/shared/vocabulary/` (ADR 0012). De querylogica achter de MCP-tools blijft bestaan in de repositories (`SearchRepository`, `GraphRepository`; `ARCHITECTURE.md` §10), voor de herkomsteindpunten in F3-09.
- *Hangt af van:* F2-01.
- *Bestanden:* `apps/app-main/src/app_main/api/routers/services_proxy.py` en de registratie in `app.py`; `frontend/src/lib/api/services.ts` (roept `/api/services` aan); `packages/surrealdb-service/src/surrealdb_service/mcp/` met script-ingang in `packages/surrealdb-service/pyproject.toml` en de tests `test_mcp_*.py`; `packages/llm-manager/src/llm_manager/mcp/` en `pyproject.toml`; `packages/file-manager/src/file_manager/mcp/` en `pyproject.toml`; `services/extraction/` en de dienst `extraction` in `docker-compose.yml`; `ARCHITECTURE.md` §1, §6 en §10.
- *ADR:* 0002 D, E; 0012.
- *Stopconditie:* verwijdercontrole V, met `tests/test_geen_verwijzingen_omzeilroutes.py` op `services_proxy`, `surrealdb_service.mcp`, `llm_manager.mcp`, `file_manager.mcp` en `services/extraction`; een test dat `/api/services/...` HTTP 404 geeft; de woordenlijsttests draaien op de nieuwe plek.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-03 Classificatie intern/extern, standaard intern, met herclassificatie en afgeleide classificatie**
- *Wat:* veld `classification` (`ASSERT $value IN ['intern','extern']`, `DEFAULT 'intern'`) op `source`, `chunk`, `source_embedding`, `doc_node`, `note`, `source_insight` en `notebook` (`source_version` volgt in F2-04). Er zijn geen bestaande records om om te zetten (B3). Bij schrijven neemt de repositorylaag de waarde van de bron over; een fragment, embedding of `doc_node` met een lossere waarde dan de bovenliggende geeft HTTP 409, een ontbrekende waarde geldt als `intern`. **Afgeleide classificatie** (ADR 0001): een vermelding (PC.3-herkomst) neemt de waarde van haar fragment over; `entity`, `entity_alias`, `relation` en `triage_queue` zijn `intern` zodra één onderliggende vermelding intern is, en `intern` zonder vermelding; de fork berekent dat bij elk schrijven van een vermelding en bij elke herclassificatie. Nieuwe `ClassificationService` met `POST /api/v1/sources/{id}/classification` (recht `beheerder`, reden verplicht): werkt bron, fragmenten, embeddings en `doc_node` in één transactie bij en berekent de afgeleide records opnieuw. Tabel `classification_log` (bron, van, naar, reden, sleutel-id, persoon, tijdstip) die alleen aanvult; een sleutel zonder persoon kan geen regel schrijven. De oude velden `private`, `privacy_mode` en `default_privacy_mode` blijven tot F2-09 staan, zodat de code tussen de taken blijft werken.
- *Hangt af van:* F2-01, F2-26.
- *Ook:* herstel van de bestaande parsefout in `migrations/5_down.surrealql` regel 2 (`SCHEMAFULL` weg uit `REMOVE TABLE`), gevonden in P10 (ADR 0021).
- *Bestanden:* `migrations/5_down.surrealql`; `migrations/89.surrealql` en `_down`; `packages/shared/src/shared/models/source.py`; `apps/app-main/src/app_main/api/schemas.py`; `packages/surrealdb-service/src/surrealdb_service/repositories/source.py`, `notebook.py`, `entity.py`; `apps/app-main/src/app_main/services/entity_persistence_service.py` (vermeldingen); nieuw `apps/app-main/src/app_main/services/classification_service.py`, nieuw `api/routers/classification.py`; `dependencies.py`.
- *ADR:* 0001 ("Uitwerking in F2-03"; B25 besloten), 0019, 0002 B.
- *Stopconditie:* `test_migration_89_classification.py`: per tabel het veld met standaard `intern`; de test schrijft een waarde en leest die terug, omdat een SCHEMAFULL-tabel onder v2.7.0 een onbekend veld stil weglaat (P4); een record zonder waarde is `intern`; `_down` zet `private = true` op alle bronnen. `apps/app-main/tests/test_classification_service.py`: import of poortwachter kan intern niet naar extern zetten (409); herclassificatie alleen met `beheerder` en reden, werkt fragmenten, embeddings en `doc_node` in één transactie bij en schrijft precies één logregel met persoon; er is geen pad om een logregel te wijzigen of te verwijderen. `apps/app-main/tests/test_afgeleide_classificatie.py`: een entiteit, alias, relatie en triage-item met één interne vermelding zijn `intern`; zonder vermelding `intern`; na herclassificatie van de enige interne bron worden ze opnieuw berekend. `test_rechten_pijplijn.py` uitgebreid: `pijplijn` krijgt 403 op herclassificatie.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-04 Bronversie, verwerkingsrun, herkomst en controle van de flow**
- *Wat:* nieuwe tabellen `source_version` (met `content_hash`, `original`, `predecessor`, `classification`, `created_by_run`, `current_run`, unieke index op `source` + `content_hash`) en `processing_run` (status `bezig`/`geslaagd`/`mislukt`/`niets_nieuws`, trigger, `pipeline_id`, `pipeline_version`, `models`, `log`, `error`, `attempts`, `notified_at`); `source.current_version`; op `chunk` de velden `source_version`, `run`, `extractor` en `classification`, en op `doc_node` de velden `source_version` en `run`; `source_version` en `run` zijn verplicht voor alle records van `chunk` en `doc_node` (lege database, B3); op `source_embedding` de velden `run` en `embedding_model`; `run` op `job` en `run` met `classification` op de entiteitvermeldingen (PC.3); `source_map_rule_id` en `source_map_rule_rev` op `processing_run` (ADR 0020; gevuld vanaf F2-13). De classificatie van een vermelding wordt opgeslagen (in F2-03 berekend, ADR 0001 "Uitwerking in F2-03"). De graafrelatie `mentions` (bron naar entiteit, `migrations/66.surrealql:58`) krijgt een afgeleide classificatie: die van haar bron, en `intern` zonder waarde. De strengste classificatie wint over alle versies. `SourceVersionRepository`, `ProcessingRunRepository`, `ProcessingRunService` en de run-eindpunten (§5.4). **Controle van de flow** (ADR 0022 E, ADR 0017 B): bij `POST /api/v1/runs` vergelijkt de fork `pipeline_version` (Git-commit en hash van de flow-JSON) met de flows in de repository, die alleen-lezen in de container `open-notebook` zijn gekoppeld; wijkt die af en is de invoer intern of onbekend, dan HTTP 409 en de run wordt `mislukt`; bij externe invoer een waarschuwing `niet_vastgelegd` in het runlogboek. De controle bij de import volgt in F2-14. De bouwer inventariseert de lezers van `provenance_chain` (voor F3-08).
- *Hangt af van:* F2-03.
- *Bestanden:* `migrations/90.surrealql` en `_down`; nieuw `packages/shared/src/shared/models/source_version.py`, `processing_run.py`; nieuw `packages/surrealdb-service/src/surrealdb_service/repositories/source_version.py`, `processing_run.py`; `repositories/source.py`; `packages/job-queue/src/job_queue/` (model en repository van `job`); `apps/app-main/src/app_main/services/entity_persistence_service.py`; nieuw `apps/app-main/src/app_main/services/processing_run_service.py`, nieuw `flow_register.py`, nieuw `api/routers/runs.py`.
- *ADR:* 0003 ("Uitwerking in F2-04"; B26 voorgesteld), 0001, 0020, 0022 E.
- *Stopconditie:* `test_migration_90_source_version_run.py`: een `chunk` of `doc_node` zonder `source_version` of `run` wordt geweigerd; de unieke index weigert een dubbele hash; `_down` draait terug. `apps/app-main/tests/test_processing_run_api.py`: een run zonder nieuwe inhoud eindigt als `niets_nieuws`; `PATCH` op een afgesloten run geeft 409; een `mentions`-relatie van een interne bron is `intern`, ook na herclassificatie van een externe bron naar intern. `apps/app-main/tests/test_flow_niet_in_git.py`: een run met een flow-hash die niet in de repository staat en interne of onbekende invoer geeft 409 en status `mislukt`; met externe invoer 201 en een logregel `niet_vastgelegd`. `apps/app-main/tests/test_classificatie_strengste_wint.py`: een interne versie maakt bron, versies, fragmenten, `doc_node` en embeddings intern, in één transactie. Uit de herreview van F2-03: de pijplijn weigert een vermelding waarvan `grounding[bron].chunk_id` niet bij die bron hoort, zodat een entiteit niet extern blijft als een andere bron intern wordt; test met twee externe bronnen waarvan er één intern wordt.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-05 Zoeken en lezen: classificatiefilter, nieuwste versie, alleen fragmenten, foutmeldingen zonder inhoud**
- *Wat:* `fn::vector_search` en `fn::text_search` opnieuw gedefinieerd: ze lezen alleen `chunk` en `source_embedding`, filteren in de query op `$classifications` en standaard op de huidige bronversie en run (`$history` voor oudere). `$show_notes` vervalt. `is_derived` (`DEFAULT true`, `ASSERT $value = true`) op `note` en `source_insight`. `ClassificationScope` volgt uit het recht van de sleutel en is een verplicht argument zonder standaard in de leesmethoden van `SearchRepository`, `SourceRepository`, `NoteRepository` en `NotebookRepository`. Een gefilterd record geeft HTTP 404. `include_notes` en `SearchRequest.search_notes` vervallen; `ContextService` neemt geen notities en inzichten meer mee. `SourceResponse` kan een bron met `source_insight` teruggeven (nu HTTP 500, P4 K3). `source.full_text` vervalt (ADR 0001, 0003): de bouwer zet de lezers om naar tekst die is samengesteld uit de fragmenten van de huidige versie, met het classificatiefilter. **Foutmeldingen:** `detail=str(e)` en vergelijkbare constructies in de routers (93 keer, telling ADR 0002 C) worden een generieke melding met een fout-id; de volledige fout gaat alleen naar het lokale log. Een router met `except Exception` laat `ClassificationConflictError` door als HTTP 409 en maakt er geen 500 van. **Classificatie in antwoorden en graaf:** `classification` komt in `SourceResponse`, `NotebookResponse` en de andere response-schema's van records met een classificatie; het filter geldt ook voor de graafrelatie `mentions` (F2-04).
- *Hangt af van:* F2-03, F2-04.
- *Bestanden:* `migrations/91.surrealql` en `_down`; nieuw `packages/shared/src/shared/models/classification.py`; `packages/surrealdb-service/src/surrealdb_service/repositories/search.py`, `source.py`, `notebook.py`; `apps/app-main/src/app_main/services/search_service.py`, `context_service.py`; `apps/app-main/src/app_main/graphs/ask.py`; `apps/app-main/src/app_main/api/routers/search.py`, `sources.py`, `sources_crud.py`, `notes.py`, `notebooks.py` en alle routers met `str(e)` in `detail`; `api/schemas.py`; de lezers van `full_text` (inventaris door de bouwer); nieuw `apps/app-main/src/app_main/api/fouten.py`.
- *ADR:* 0002 C, 0003, 0004, 0001.
- *Stopconditie:* `packages/surrealdb-service/tests/test_search_classificatiefilter.py`: met scope extern 0 interne records bij tekst-, vector- en hybride zoeken, ook met `history=true`; een leesmethode zonder `scope` geeft `TypeError`; het aantal treffers met filter wordt gemeten en vastgelegd (volledigheid, ADR 0002 C). `apps/app-main/tests/test_contract_lezen_extern.py` (de lektest uit ADR 0002): synthetische interne en externe gegevens voor bronnen, fragmenten, notities, entiteiten, aliassen, relaties, `doc_node`, inzichten (`source_insight`) en notebooknamen en -beschrijvingen; elke route voor `lezen_extern` aangeroepen; ook de foutpaden (ongeldige id's, ongeldige invoer, een geforceerde fout in de repositorylaag); geen intern id of interne tekst in een antwoord of foutmelding; een intern id geeft 404; een bron met `source_insight` geeft 200; een `lezen_extern`-uitvraag toont geen `mentions` van een interne bron. `apps/app-main/tests/test_classificatie_in_antwoord.py`: `SourceResponse`, `NotebookResponse` en de andere response-schema's geven `classification` terug. `apps/app-main/tests/test_classificatieconflict_409.py`: een geforceerde `ClassificationConflictError` achter een router met `except Exception` geeft 409, geen 500. `tests/test_geen_uitzonderingstekst_in_detail.py`: zoektest die faalt bij `str(e)`, `repr(e)` of een f-string met een uitzondering in `detail=` in `apps/app-main/src/app_main/api/`. `apps/app-main/tests/test_zoeken_geen_afgeleide_teksten.py`: een notitie en een inzicht met de zoektekst komen niet terug; elk resultaat verwijst naar `chunk` of `source_embedding`. Schemacontrole in `test_migration_91_*.py`: `source.full_text` bestaat niet meer.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-06 Optimistische vergrendeling en handmatige run voor samenvoegen en splitsen**
- *Wat:* `rev`, `modified_by` en `modified_at` op `entity`, `entity_alias`, `relation` en de besluiten in `triage_queue`; één hulpfunctie in de basisrepository; een wijziging met een verouderde `rev` geeft 409. `manual_override` blokkeert een run, die het conflict logt. `resolution_log` krijgt `author`. `ChunkMutator` wordt een dienst die een run met `trigger = handmatig` en nieuwe fragmenten maakt; een automatische run verzet `current_run` niet zolang die naar een handmatige run wijst, en meldt dat in de wachtrij. De overige schrijfroutes voor fragmenten en bronnen volgen in F2-29. Uit de review van F2-04: `ChunkMutator.split` schrijft nu een nieuw fragment onder de oude, mogelijk al afgesloten run; dat krijgt hier een eigen handmatige run.
- *Hangt af van:* F2-04.
- *Bestanden:* `migrations/92.surrealql` en `_down`; `packages/surrealdb-service/src/surrealdb_service/repositories/base.py`, `entity.py`; `apps/app-main/src/app_main/services/chunking/chunk_mutator.py`; `api/routers/sources_crud.py` (`:1077`, `:1110`), `api/schemas.py` (`:23-35`); `services/entity_resolution/merge_audit.py`; `api/routers/triage.py`, `entity_resolution.py`.
- *ADR:* 0003, 0005.
- *Stopconditie:* `packages/surrealdb-service/tests/test_optimistische_vergrendeling.py`; `apps/app-main/tests/test_handmatige_run.py`: samenvoegen geeft een handmatige run met nieuwe fragmenten, het aantal fragmenten van de oude run blijft gelijk, een automatische run verzet `current_run` niet en maakt een melding; `resolution_log` heeft een auteur.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-29 Overige schrijfroutes voor fragmenten en bronnen: handmatige run en vervalmarkering**
- *Wat:* de routes uit ADR 0005 in `apps/app-main/src/app_main/api/routers/sources_crud.py` die fragmenten of bronnen ter plekke wijzigen, worden omgebouwd: `PATCH /api/sources/{source_id}/chunks/{chunk_id}` (`:964`), `DELETE /api/sources/{source_id}/chunks/{chunk_id}` (`:998`) en `POST /api/sources/{source_id}/chunks` (`:1027`) worden een handmatige run met een volledige nieuwe fragmentenset voor dezelfde bronversie, via de dienst uit F2-06; elk nieuw fragment krijgt de classificatie van de bronversie en `extractor = handmatig`. `DELETE /api/sources/{source_id}` (`:475`) vervalt; in de plaats komt `POST /api/v1/sources/{id}/retire` (recht `beheerder`, reden verplicht) met `retired_at`, `retired_by` en `retired_reason`. Zoeken en de leesroutes laten een vervallen bron standaard weg; met `history=true` blijft ze raadpleegbaar, met het filter. De schermen voor fragmentbewerking en het verwijderen van een bron roepen de nieuwe routes aan (ADR 0016). `test_enige_schrijfroute.py` (uit F2-14) wordt uitgebreid. Uit de herreview van F2-03: een rechtstreekse `MERGE {classification: NONE}` zet een externe bron terug op intern (strenger, maar verlies van het besluit); de schrijfroutes laten `classification` weg in plaats van `NONE` te sturen, met een test.
- *Hangt af van:* F2-06, F2-14.
- *Bestanden:* `apps/app-main/src/app_main/api/routers/sources_crud.py`; `apps/app-main/src/app_main/services/chunking/chunk_mutator.py`; `packages/surrealdb-service/src/surrealdb_service/repositories/source.py`, `search.py`; `migrations/96.surrealql` en `_down` (velden voor de vervalmarkering); `api/schemas.py`; `frontend/src/` (fragmentbewerking, bron verwijderen); `apps/app-main/tests/test_enige_schrijfroute.py`.
- *ADR:* 0005, 0003, 0001, 0016.
- *Stopconditie:* `apps/app-main/tests/test_handmatige_fragmentroutes.py`: na wijzigen, verwijderen en toevoegen is het aantal fragmenten van de vorige run gelijk en hun tekst ongewijzigd, en bestaat een handmatige run met de nieuwe set; `apps/app-main/tests/test_bron_vervallen.py`: na `retire` staat de bron niet in een zoekresultaat, wel met `history=true`, en haar fragmenten, versies en embeddings staan nog in de database; `DELETE /api/sources/{id}` geeft 404 of 405. `test_enige_schrijfroute.py`: geen route behalve `/api/v1/import` en de handmatige run wijzigt of verwijdert een record van `chunk`, `source_embedding`, `doc_node` of `source_version`, of maakt er een aan. `test_migration_96_*.py`; `npm --prefix frontend run test` en `npm --prefix frontend run build`.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-30 Herclassificatie van notebooks en notities**
- *Wat:* `POST /api/v1/notebooks/{id}/classification` en `POST /api/v1/notes/{id}/classification` (recht `beheerder`, reden verplicht), met een regel in `classification_log` (met het type van het record). Regel volgens aanname A9: een notitie die naar een intern fragment verwijst, kan niet naar `extern` (409); een notebook beschermt alleen zijn eigen naam en beschrijving (ADR 0001) en verandert niets aan zijn bronnen.
- *Hangt af van:* F2-03. Vooraf vastgelegd door de ontwerper in ADR 0001 (§8, B22).
- *Bestanden:* `apps/app-main/src/app_main/services/classification_service.py`; `api/routers/classification.py`; `packages/surrealdb-service/src/surrealdb_service/repositories/notebook.py`; `migrations/<n>.surrealql` en `_down` (type in `classification_log`).
- *ADR:* 0001 (aanvulling, voorgesteld), 0002 B.
- *Stopconditie:* `apps/app-main/tests/test_herclassificatie_notebook_notitie.py`: met `beheerder` en reden één logregel met persoon en type; zonder reden 422; een notitie met een verwijzing naar een intern fragment naar `extern` geeft 409; `test_rechten_pijplijn.py` uitgebreid: `pijplijn` krijgt 403 op beide routes; `test_contract_lezen_extern.py`: een intern notebook is voor `lezen_extern` niet zichtbaar (404).
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

#### B. Modelpoort

**F2-07 Modelregister, `ModelGate` en auditlog**
- *Wat:* register als YAML-bestand (YAML Ain't Markup Language) in Git met naam, versie, aanroepwijze, locatie, toegestaan voor intern (standaard `nee`) en taken. `llm_manager.gate.ModelGate` zoekt het model op, weigert een niet-toegestaan model bij interne invoer zonder terugval, probeert bij HTTP 502 van llama-swap opnieuw (pauze, maximum) en schrijft een regel in `model_call` (alleen aanvulbaar, nooit prompt- of antwoordtekst), met `run` als de aanroep bij een run hoort (ADR 0007). `POST /api/v1/model-calls` (recht `pijplijn`) voor Langflow; `/api/models` wordt een alleen-lezen weergave van het register.
- *Hangt af van:* F2-01, F2-04. Werkmodel volgens B4 (besloten): Qwen3.5-9B Q4_K_M, context 8 192.
- *Bestanden:* nieuw `config/modelregister.yaml`; nieuw `packages/llm-manager/src/llm_manager/gate.py`, `register.py`; `migrations/93.surrealql` en `_down`; `apps/app-main/src/app_main/services/model_routing/telemetry.py` (schrijver van `model_call`); nieuw `api/routers/model_calls.py`; `api/routers/models.py`.
- *ADR:* 0006, 0007, 0009 (502).
- *Stopconditie:* `packages/llm-manager/tests/test_model_gate.py`: cloudmodel geweigerd bij interne invoer, ook als het lokale model faalt; onbekend model geweigerd; 502 leidt tot een nieuwe poging tot het maximum; elke aanroep één auditregel, met `run` als die is meegegeven. `packages/llm-manager/tests/test_modelregister_schema.py` valideert het register. `test_migration_93_model_call.py`: de tabel heeft geen tekstveld voor prompt of antwoord.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-08 Modelaanroepen via `ModelGate` en llama-swap**
- *Wat:* Esperanto gebruikt `openai-compatible` met `base_url=http://llama-swap:8080/v1` en model-id's uit de llama-swap-configuratie. `llm_call.py` wordt de dunne aanroeper van `ModelGate`; chat gaat via de poort naar een lokaal model. `llm_matcher.py` gebruikt de chat van de poort; `semantic_intelligence/config.py` en `pipelines/embeddings` gebruiken `/v1/embeddings` (mxbai-embed-large, 1 024 dimensies). Fragmenten boven 512 tokens: de bouwer kiest opdelen of afkappen en legt dat vast (ADR 0008). De aanroepers van `packages/shared/src/shared/model_routing.py` in app-main (`preprocessing_service.py`, `app.py`, `config_coherence.py`) gaan over op de poort.
- *Hangt af van:* F2-07.
- *Bestanden:* `packages/llm-manager/src/llm_manager/providers/factory.py`; `apps/app-main/src/app_main/services/model_routing/llm_call.py`; `apps/app-main/src/app_main/graphs/utils.py`; `apps/app-main/src/app_main/services/preprocessing_service.py`; `packages/shared/src/shared/config_coherence.py`; `pipelines/entity-filtering/src/entity_filtering/resolution/llm_matcher.py`; `packages/semantic-intelligence/src/semantic_intelligence/config.py`; `pipelines/embeddings/src/embeddings/providers/`.
- *ADR:* 0006, 0008.
- *Stopconditie:* `tests/test_geen_ollama_eindpunten.py` faalt bij `/api/generate`, `/api/chat`, `/api/embed`, `11434` of `host.docker.internal` in `apps/`, `packages/` of `pipelines/` (met als enige tijdelijke uitzondering `packages/shared/src/shared/model_routing.py`, dat in F2-09 vervalt); `apps/app-main/tests/test_embedding_local_guardrail.py` aangepast en groen; `pipelines/embeddings/tests/test_lange_fragmenten.py` toetst de 512-tokenregel; de bestaande tests met 1 024 dimensies blijven groen.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-09 Verwijderen: Track J-routering, `model_route` en oude privacyvelden**
- *Wat:* weg gaan `route_resolver.py` (de weigerlogica zit al in `ModelGate`), `failover_executor.py`, `circuit_breaker.py`, `rate_limiter.py`, `error_mapping.py`, `privacy_resolver.py`, `seed.py`, `seed_chain_models.py`, de router `model_routes`, de tabel `model_route`, `packages/shared/src/shared/model_routing.py` en `.yaml`, de cloudaanbieders met `api_key_env_var` in `registry.py`, de code die modelrijen in de database schrijft of leest (na omzetten van de lezers; in de lege database staan geen rijen om op te ruimen, B3), en de velden `private`, `privacy_mode` en `default_privacy_mode` met het formulierveld `private`.
- *Hangt af van:* F2-08, F2-03.
- *Bestanden:* `apps/app-main/src/app_main/services/model_routing/` (genoemde bestanden); `api/routers/model_routes.py`; `packages/surrealdb-service/src/surrealdb_service/repositories/model_route.py`, `model.py`; `packages/shared/src/shared/model_routing.py`, `model_routing.yaml`; `packages/llm-manager/src/llm_manager/providers/registry.py`; `apps/app-main/src/app_main/api/routers/sources_upload.py` (`:199`); `migrations/94.surrealql` en `_down` (verwijdert alleen de velddefinities, zonder gegevens om te zetten; de `_down` zet `private = true`); `ARCHITECTURE.md` §8.
- *ADR:* 0006, 0001.
- *Stopconditie:* verwijdercontrole V met `tests/test_geen_verwijzingen_trackj.py` (`PrivacyMode`, `privacy_mode`, `default_privacy_mode`, `model_route`, `shared.model_routing`); `tests/test_geen_ollama_eindpunten.py` zonder uitzondering; `test_migration_94_*.py`.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

#### C. Omgeving

**F2-10 Compose-basis: netwerken, geheimen, SurrealDB, API en werkbak**
- *Wat:* `docker-compose.signaalsysteem.yml` volgens §4: interne netwerken (`internal: true`) `opslag`, `verwerking`, `beheer`, `werkbak-intern` en `editor-intern` (en vooraf gedefinieerd, nog zonder leden: `mcp-intern`, `ophalen-intern`, `gateway-intern`, `cli-intern`); gewone netwerken `toegang` (alleen de proxy's) en `uitgaand` (nog zonder leden; vanaf fase 4 alleen `ophalen` en `cloud-gateway`). `surrealdb` alleen op `opslag`, zonder poort; `open-notebook` op `opslag`, `verwerking` en `werkbak-intern`, zonder poort, met de flows uit `services/langflow/flows/` alleen-lezen gekoppeld (F2-04); `werkbak-toegang` als omgekeerde proxy op `werkbak-intern` en `toegang`, alleen op `127.0.0.1`. `secrets:` met bestanden in `secrets/`; `secrets/` in `.gitignore` (nu niet aanwezig). De image-build van de fork krijgt vaste basis-tags (`python:3.12.16-slim-bookworm`, `ghcr.io/astral-sh/uv:0.11.13`, zoals in P4). `root`/`root` en `extra_hosts` vervallen. Database volgens B3 en ADR 0021: `surrealdb/surrealdb:v3.3.5` met vaste digest als F2-26 tot v3 leidt, anders `v2.7.0`, met een eigen, nieuw volume `signaal_surreal_data`; de bestaande map `./surreal_data` van `docker-compose.yml` blijft onaangeroerd staan als archief en wordt niet gekoppeld. Bij de eerste start draaien alle migraties op de lege database.
  - **Verplichte proef masquerade** (ADR 0002 F; was aanname A6): met de opzet van P8 de bridge-optie `com.docker.network.bridge.enable_ip_masquerade: "false"` op `toegang`. Criterium: de gepubliceerde poort van `werkbak-toegang` blijft bereikbaar vanaf de host, en vanuit de proxy faalt een verbinding met internet. Slaagt de proef, dan staat de optie in Compose. Slaagt ze niet, dan wordt elke proxy gehard: een minimaal image zonder shell, `read_only: true`, `cap_drop: [ALL]`, `security_opt: [no-new-privileges:true]`, een gebruiker zonder rootrechten, geen secrets en geen volumes, en een configuratie die alleen het eigen pad naar de eigen doeldienst doorstuurt. De uitkomst geldt ook voor `editor-toegang` (F2-12) en `mcp-toegang` (F3-10).
- *Hangt af van:* F2-01, F2-26 (versie). B3 besloten (Geert, 30-10).
- *Bestanden:* nieuw `docker-compose.signaalsysteem.yml`; `Dockerfile`; `.gitignore`; nieuw `services/proxy/` (configuratie van de proxy's); nieuw `scripts/signaalsysteem/controle/netwerk.sh` (opzet van P8); nieuw `tests/test_compose_signaalsysteem.py`.
- *ADR:* 0002 F, 0016, 0021, 0022 A.
- *Stopconditie:* `docker compose -f docker-compose.signaalsysteem.yml config --quiet` exitcode 0. `tests/test_compose_signaalsysteem.py`: geen image zonder vaste tag of digest, geen `pull_policy: always`; `surrealdb` gebruikt de versie uit F2-26 en het volume `signaal_surreal_data`, en geen dienst koppelt `./surreal_data`; `surrealdb` hangt alleen aan `opslag`, en op `opslag` hangen alleen `surrealdb` en `open-notebook`; op elk proxynetwerk (`werkbak-intern`, `editor-intern`, `mcp-intern`) hangen precies de proxy en haar doeldienst; alleen de proxy's publiceren een poort, en alleen op `127.0.0.1`; aan `toegang` hangen alleen proxy's; aan `uitgaand` hangen alleen `ophalen` en `cloud-gateway`; geen dienst hangt aan een intern én een gewoon netwerk, behalve de proxy's, `ophalen` en `cloud-gateway`; geen letterlijke wachtwoorden of sleutels in het bestand. `netwerk.sh` exitcode 0: `open-notebook` en `surrealdb` hebben geen internet en geen bereikbare poort; `surrealdb` is alleen bereikbaar vanuit `open-notebook`; de API start op een leeg volume en geeft `/health` HTTP 200 (alle migraties gedraaid); de werkbakproxy op `127.0.0.1` geeft de interface; een API-aanroep zonder sleutel geeft 401; de proef masquerade is uitgevoerd en vastgelegd, en bij een geharde proxy toetst het script `read_only`, `cap_drop` en de gebruiker via `docker inspect`. Classificatie end-to-end in de draaiende omgeving (open punt F2-03): een import of poortwachter die een lossere classificatie probeert, geeft via HTTP 409; een ongeldige waarde geeft 422; geen router maakt van `ClassificationConflictError` een 500 (doorlaten volgens F2-05).
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-28 Referentiegegevens na een lege database: inventaris en laadscript**
- *Wat:* de bouwer inventariseert welke gegevens de code leest die niet uit een migratie, een import of een run komen, en dus na een lege database ontbreken (ADR 0021, akkoord Geert 30-10; R10). Bekend: de TOOI-woordenlijsten (`packages/shared/src/shared/vocabulary/tooi_bulk.py`, `tooi_provider.py`), de gemeentecodes van het Centraal Bureau voor de Statistiek (CBS) (aanname A10: via TOOI of een eigen bron), de gedeelde ontologieën (`packages/ontology-manager/ontologies/*.yaml`; tabel `Ontology`, `ARCHITECTURE.md` §2) en wat de bouwer verder vindt. Per set: tabel, bron, openbare vindplaats, SHA-256 en laadwijze in `config/referentiegegevens.yaml`. Eén laadscript laadt alle sets idempotent; het draait in de container `open-notebook` met dezelfde verbinding als de migraties. Iv3 en het Basiswettenbestand (BWB) volgen in F3-03 met hetzelfde script. Het archief `./surreal_data` wordt niet gekoppeld; wijzigingen aan gedeelde ontologieën die alleen in de oude database staan, meldt de bouwer aan Geert (aanname A11).
- *Hangt af van:* F2-10, F2-01.
- *Bestanden:* nieuw `config/referentiegegevens.yaml`; nieuw `scripts/signaalsysteem/laad_referentiegegevens.py`; nieuw `tests/signaalsysteem/test_referentiegegevens_schema.py`; nieuw `scripts/signaalsysteem/controle/referentiegegevens.sh`.
- *ADR:* 0021, 0013.
- *Stopconditie:* `test_referentiegegevens_schema.py` valideert de inventaris (elk item met tabel, bron, SHA-256 en laadwijze). `referentiegegevens.sh` exitcode 0 op een nieuw, leeg volume: na de migraties en het laadscript is per set het aantal records gelijk aan het aantal in het bronbestand; een tweede run verandert niets (zelfde aantallen, geen nieuwe records); de extractietest `scripts/n_extraction_gate.py` vindt de ontologieën die hij gebruikt.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-11 llama-swap met socket-proxy, embeddingmodel, werkmodel en docling-serve**
- *Wat:* afgeleid image (llama-swap plus Docker-client), `socket-proxy` met configuratie B op `beheer`, `docling-serve` als vooraf aangemaakte container zonder `restart`, configuratie `services/llama-swap/config.yaml` met groepen `zwaar` en `licht` (§4.3), `ttl` van docling 300 s, `checkEndpoint` per dienst, `timeouts.responseHeader` op de standaard 0 of boven `DOCLING_SERVE_MAX_SYNC_WAIT`. docling-serve krijgt `DOCLING_SERVE_MAX_SYNC_WAIT` (voorstel 3 600 s, §8 B21) voor het synchrone eindpunt. Geheimen als bestand, zodat ze ook bij `docker start` door llama-swap aanwezig zijn. `NVIDIA_DISABLE_REQUIRE=1` en `LD_LIBRARY_PATH` zolang de driver CUDA 13.3 levert. Modellen in een alleen-lezen volume, met een downloadscript dat de SHA-256 en revisies uit `proeven.md` 1.7 controleert. docling-serve staat in het modelregister (aanroepwijze `llama-swap`, toegestaan voor intern).
- *Hangt af van:* F2-10, F2-07.
- *Bestanden:* nieuw `services/llama-swap/Dockerfile`, `config.yaml`, `wacht_op_dienst.sh` (uit `scripts/signaalsysteem/proeven/p1/`); nieuw `scripts/signaalsysteem/modellen_downloaden.py` (uit `p1/modellen_downloaden.py`); `docker-compose.signaalsysteem.yml`; `config/modelregister.yaml`; nieuw `packages/llm-manager/tests/test_modelregister_consistent.py`.
- *ADR:* 0006, 0008, 0009, 0014.
- *Stopconditie:* `test_modelregister_consistent.py`: elk model met aanroepwijze `llama-swap` in het register staat in `config.yaml`, en omgekeerd. Op de werkplek `scripts/signaalsysteem/controle/gpu.sh` exitcode 0: `docling-serve` start en stopt via llama-swap (`exited` binnen `ttl` + 15 s); de API-sleutel van docling is aanwezig na `docker start`; `DOCLING_SERVE_MAX_SYNC_WAIT` staat in de omgeving van de gestarte container; embedding van 1 024 dimensies; een externe stop gevolgd door een aanvraag eindigt via de herhaalpoging in HTTP 200; vrij werkgeheugen van de host en piek-VRAM (videogeheugen) worden vastgelegd. Een socket-proxytest herhaalt K2 van P7 (`create`, `exec`, `inspect` geven 403).
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`. Uit te voeren op de werkplek.

**F2-12 Langflow-inrichting en afscherming**
- *Wat:* afgeleid Langflow-image met `llm_manager` (inclusief `llm_manager.gate` uit F2-07) en de componenten; `langflow` alleen op `verwerking` en `editor-intern`, nooit op `uitgaand` of `toegang`; de editor via de proxy `editor-toegang` op `editor-intern` en `toegang`, alleen op `127.0.0.1`, volgens de uitkomst van de masquerade-proef (F2-10). Automatisch aanmelden uit; opstartscript dat flows uit Git laadt (`POST /api/v1/flows/upload/`) en triggers aanzet (`POST /api/v1/triggers/{id}/enable`); exportscript (`POST /api/v1/flows/download/`). **Componentenlijst** (ADR 0022 D): `config/langflow-componenten.yaml` met de toegestane typen (de eigen componenten en een kleine set kerncomponenten zonder netwerk- of codefunctie, vastgelegd door de bouwer); het opstartscript toetst elke flow vóór het laden en weigert een flow met een type dat niet op de lijst staat, met een foutcode. **Instellingen:** `LANGFLOW_ALLOW_CUSTOM_COMPONENTS=false`, `LANGFLOW_SUBSTITUTE_OUTDATED_COMPONENT_CODE=false`, `LANGFLOW_BLOCK_CODE_INTERPRETER_COMPONENTS=true`, `LANGFLOW_AGENTIC_EXPERIENCE=false`, `LANGFLOW_ALLOW_COMPONENTS_PATHS_OVERRIDE=true` met de componentenmap alleen-lezen gekoppeld, `LANGFLOW_DEACTIVATE_TRACING=true`, `DO_NOT_TRACK=true`, `LITELLM_LOCAL_MODEL_COST_MAP=True`, MCP-server van Langflow uit, `LANGFLOW_SECRET_KEY` als Fernet-sleutel uit een geheim. **Verboden variabelen** (ADR 0022 F): het startscript van de container weigert te starten als een variabele met voorvoegsel `LANGCHAIN_`, `LANGSMITH_`, `LANGFUSE_`, `LANGWATCH_`, `ARIZE_`, `OPIK_`, `PHOENIX_`, `TRACELOOP_`, `OPENLAYER_` of `OTEL_EXPORTER_OTLP_` is gezet. Langflow krijgt alleen de geheimen `sleutel_pijplijn`, `langflow_superuser` en `langflow_secret_key`; geen Claude-sleutel, in geen enkele fase.
- *Hangt af van:* F2-10, F2-07. Plaats volgens B7 (besloten): code in `services/langflow/`, configuratie in `config/`.
- *Bestanden:* nieuw `services/langflow/Dockerfile`, `services/langflow/scripts/start.sh` (controle verboden variabelen), `flows_laden.py`, `flows_exporteren.py`, map `services/langflow/flows/`, `services/langflow/components/signaalsysteem/`, `services/langflow/tests/`; nieuw `config/langflow-componenten.yaml`; `docker-compose.signaalsysteem.yml`; `services/proxy/` (editor).
- *ADR:* 0017, 0022 A, D, F; 0002 D, F.
- *Stopconditie:* `services/langflow/tests/test_flows_rondgang.py`: laden en daarna exporteren levert dezelfde flow-JSON (JavaScript Object Notation) op, op volgorde van sleutels na. `services/langflow/tests/test_componentenlijst.py`: een flow met een type dat niet op de lijst staat, wordt niet geladen en het script eindigt met een foutcode; elke flow in `services/langflow/flows/` gebruikt alleen typen van de lijst. `services/langflow/tests/test_verboden_variabelen.py`: één test per voorvoegsel; het startscript weigert te starten. `scripts/signaalsysteem/controle/langflow.sh` exitcode 0: aanmelden zonder wachtwoord faalt; het MCP-eindpunt van Langflow geeft geen 200; `langflow` bereikt geen internet (P8-controle); `docker inspect` van `langflow` toont geen Claude-sleutel en geen verboden variabele; de editor is alleen via `editor-toegang` op `127.0.0.1` bereikbaar. Het aantal uitgaande verbindingspogingen van LiteLLM wordt geteld zoals in P2 en vastgelegd (doel 0 volgens ADR 0017; het netwerk blokkeert ze hoe dan ook).
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

#### D. Inname en verwerking

**F2-13 Bronkaart, innamebatch en innamewachtrij in de fork**
- *Wat:* tabellen `source_map_rule`, `intake_batch` en `intake_item` in SurrealDB, beheerd met de migraties van de fork, met API (ADR 0020): bronkaartregels (schema §5.2), batches van een bulkupload (classificatie en route per batch) en de innamewachtrij (documenten uit een watchfolder en losse uploads ter bevestiging, bronnen zonder regel). Standaardclassificatie per innameroute volgens ADR 0019; zonder keuze `intern`. Rechten (ADR 0020): lezen met `pijplijn` of `beheerder`; `lezen_extern` en `lezen_intern` krijgen 403; aanmaken, wijzigen, op `extern` zetten en bevestigen van regels en batches, en het besluit over een wachtrij-item, alleen met `beheerder`; `extern` zonder bevestiging (`classification_confirmed_by`, `_at`) geldt als `intern`; `pijplijn` mag regels en batches lezen, wachtrij-items aanmelden, besloten items ophalen en een item op `verwerkt` zetten. **Geen locatie in de vault of de exportmap** (ADR 0004, 0020): een regel waarvan `location` binnen de vault (`D:\# Vault` of het koppelpunt daarvan in een container) of binnen de exportmap van de export naar Obsidian ligt, geeft HTTP 422; beide paden zijn een instelling van de fork; de controle vergelijkt genormaliseerde paden, ook na het oplossen van symbolische koppelingen. `source_map_rule` krijgt `rev`, `modified_by` en `modified_at`; een nieuwe run neemt id en `rev` van zijn regel over.
- *Hangt af van:* F2-03, F2-04, F2-06 (vergrendeling). B2 besloten (ADR 0020).
- *Bestanden:* `migrations/95.surrealql` en `_down`; nieuw `packages/shared/src/shared/models/source_map.py`; nieuw `packages/surrealdb-service/src/surrealdb_service/repositories/source_map.py`, `intake.py`; nieuw `apps/app-main/src/app_main/services/source_map_service.py`, `intake_service.py`; nieuw `api/routers/source_map.py`, `intake.py`; `apps/app-main/src/app_main/config.py` (paden van vault en exportmap).
- *ADR:* 0020, 0019, 0001, 0004, 0010; FO 5.
- *Stopconditie:* `test_migration_95_source_map.py` (schemacontrole, ook op de standaardwaarden; `_down` draait terug); `apps/app-main/tests/test_source_map_service.py`: een regel voor route API, MCP, feed of website stelt `extern` voor en vraagt bevestiging; een batch zonder keuze is `intern`; een watchfolder-item heeft geen standaard en staat op `wacht`; een wijziging met een verouderde `rev` geeft 409; een run bevat id en `rev` van zijn regel; een regel met een locatie in de vault, in de exportmap of via een symbolische koppeling daarnaartoe geeft 422. `apps/app-main/tests/test_source_map_api.py`: een sleutel `lezen_extern` of `lezen_intern` krijgt op elke bronkaart-, batch- en wachtrijroute 403. `test_rechten_pijplijn.py` uitgebreid: `pijplijn` krijgt 403 op aanmaken, wijzigen en bevestigen van een regel of batch (extern bevestigen) en op het besluit over een wachtrij-item. `test_contract_lezen_extern.py` uitgebreid met bronkaartregels in de testgegevens: geen naam of locatie van een regel in een antwoord voor `lezen_extern`.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-14 Importeindpunt**
- *Wat:* `POST /api/v1/import` (recht `pijplijn`) met `ImportService` (§5.1): bron, bronversie, classificatie, `run_id`, Docling-structuur als `doc_node` (met `source_version`, `run` en `classification`), fragmenten met labels. Bekende `source` + `content_hash` geeft `ongewijzigd` en de run `niets_nieuws`; hetzelfde verzoek met dezelfde `run_id` geeft dezelfde id's; een lagere classificatie dan de bestaande geeft 409. Een verzoek waarvan de ruwe inhoud of een fragment `<!-- signaalsysteem:begin -->` of `<!-- signaalsysteem:end -->` bevat, geeft 422 met een regel in het runlogboek (ADR 0004). De controle van de flow uit F2-04 geldt ook hier: een run met een flow die niet in Git staat en interne of onbekende invoer geeft 409. Hergebruik van `_create_source_impl`, het schrijven van fragmenten in `repositories/source.py` en de opslag als `doc_node`. Uit de review van F2-04 (ADR 0003 "Uitwerking in F2-04"): de bronkaartregel van de run wordt gebonden aan de werkelijke bron, niet door de aanroeper gekozen; `content_hash` is de hash van de ruwe inhoud, ook bij een URL (niet van geëxtraheerde tekst); een bekende hash die niet de huidige versie is, volgt B26.
- *Hangt af van:* F2-04, F2-05, F2-13.
- *Bestanden:* nieuw `apps/app-main/src/app_main/services/import_service.py`, nieuw `api/routers/imports.py`; `api/schemas.py`; `repositories/source.py`; `dependencies.py`.
- *ADR:* 0010, 0019, 0003, 0001, 0004, 0022 E.
- *Stopconditie:* `apps/app-main/tests/test_import_service.py`: twee keer hetzelfde verzoek geeft hetzelfde resultaat; een bekende hash maakt geen nieuwe versie; 409 bij een lagere classificatie; elk geschreven fragment en elke `doc_node` heeft `source_version`, `run` en de classificatie van de versie, en elk fragment `extractor`; een verzoek met de markering van het blok geeft 422 en een logregel; een import binnen een run met een niet-vastgelegde flow en interne invoer geeft 409; de invoerclassificatie van de run komt uit de bronkaartregel van de werkelijke bron, ook als de aanroeper een andere regel opgeeft; de inhoudshash van een URL-bron is die van de opgehaalde ruwe inhoud, niet van de geëxtraheerde tekst; inhoud A → B → A volgt B26 (bij het voorstel: `current_version` wijst weer naar v1, één logregel, geen nieuwe versie). `apps/app-main/tests/test_enige_schrijfroute.py` (eerste versie; uitgebreid in F2-29): geen route behalve `/api/v1/import` (en de handmatige run uit F2-06) roept de schrijfmethoden voor `source` of `chunk` aan.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-15 Stap-eindpunten en `RunStepOrchestrator`**
- *Wat:* `POST` en `GET /api/v1/runs/{run_id}/steps/{stap}` (§5.4, recht `pijplijn`) voor `embed`, `entity_extraction`, `entity_dedup` en `graph`. De fork zet een job met `job.run` in `packages/job-queue`; de worker doet de herhaalpogingen; na het maximum gaat de run op `mislukt` met een melding. `processing_stage` blijft als statusveld. De handlers voor EMBED, EXTRACT en GRAPH in `handlers.py` blijven, en starten vanuit een stap. Embeddings berekent de fork via `pipelines/embeddings` en `ModelGate`.
- *Hangt af van:* F2-04, F2-08.
- *Bestanden:* nieuw `apps/app-main/src/app_main/services/run_step_orchestrator.py`; `api/routers/runs.py`; `apps/app-main/src/app_main/handlers.py`; `packages/job-queue/src/job_queue/`.
- *ADR:* 0011, 0003.
- *Stopconditie:* `apps/app-main/tests/test_run_step_orchestrator.py`: een stap maakt een job met `run`; een mislukte job wordt opnieuw geprobeerd tot `max_retries` en zet daarna run en stap op `mislukt` met een melding; een stap op een onbekende run geeft 404; embeddings van de stap hebben `run` en `embedding_model`.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-16 Basiscomponenten in Langflow en `run.models` uit `model_call`**
- *Wat:* gedeelde Python-componenten: run aanmaken en afsluiten (met Git-commit en hash van de flow), stap starten en volgen, modelkeuze met registercontrole (een cloudmodel achter een interne of onbekende bron geeft een fout bij verbinden en bij uitvoeren), herhaalbeleid bij 502, 429 en 5xx, en een auditregel per modelaanroep via `POST /api/v1/model-calls`. Langflow praat alleen met de API van de fork en met llama-swap, nooit met de database; cloudaanroepen lopen vanaf fase 5 alleen via `cloud-gateway` (F5-07). **In de fork** (ADR 0003, 0007): `processing_run.models` wordt afgeleid uit de regels in `model_call` met dezelfde run, bij het afsluiten van de run en bij elke latere regel; een lijst `models[]` in `PATCH` wordt niet overgenomen maar vergeleken, en een verschil komt als waarschuwing in het runlogboek.
- *Hangt af van:* F2-12, F2-07, F2-14, F2-15.
- *Bestanden:* nieuw `services/langflow/components/signaalsysteem/run.py`, `stap.py`, `model.py`, `fork_client.py`; `services/langflow/tests/`; `apps/app-main/src/app_main/services/processing_run_service.py`.
- *ADR:* 0006, 0007, 0009, 0011, 0017, 0003.
- *Stopconditie:* `services/langflow/tests/test_model_component.py`: cloudmodel achter een interne of onbekende bron geweigerd, zonder terugval; `test_stap_component.py` tegen een nagebootste fork: 502 leidt tot een nieuwe poging, `wacht_op_review` wordt herkend (§5.4); `test_run_component.py`: de run legt commit en flow-hash vast. `apps/app-main/tests/test_run_models_uit_model_call.py`: `processing_run.models` is gelijk aan de modellen in `model_call` van die run, ook als `PATCH` een andere lijst meestuurt; het verschil staat als waarschuwing in het logboek.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-17 Docling-component met fragmentatie, synchroon eindpunt**
- *Wat:* de component stuurt een document naar het synchrone eindpunt `/upstream/docling/v1/convert/file` (`to_formats` met `json`, `ocr_lang` `nl` en `en`), zodat de aanvraag openstaat zolang de conversie loopt (ADR 0009, 0014). Een verloren of afgebroken aanvraag dient de component hooguit drie keer opnieuw in; daarna wordt de stap `mislukt`, met een regel in het runlogboek en een melding in de wachtrij van de beheerder. Het adres is een instelling per aanroepwijze; vóór elke aanroep toetst de component de classificatie tegen het register. Elke aanroep van docling-serve schrijft een auditregel via `POST /api/v1/model-calls` (run, dienst, versie, classificatie, duur, status; FO 9, audit). Het DoclingDocument gaat naar `ChunkExtractor` en `regrouper.py`. De OCR- en tabelopties staan in de flow, niet in Compose. **Toets op de werkplek:** tijdens een synchrone conversie van D1 een aanvraag voor het werkmodel en het verlopen van `ttl`. Faalt die toets, dan volgt de terugvaltaak F2-32.
- *Hangt af van:* F2-16, F2-11.
- *Bestanden:* nieuw `services/langflow/components/signaalsysteem/docling.py`, `fragmentatie.py`; `pipelines/ingestion/src/ingestion/chunking/extractor.py`, `regrouper.py` (hergebruik, alleen aanhaakpunten); nieuw `scripts/signaalsysteem/controle/docling_sync.sh`.
- *ADR:* 0014, 0009, 0019, 0010, 0007.
- *Stopconditie:* `services/langflow/tests/test_docling_component.py` tegen een nagebootste docling-serve: een afgebroken aanvraag leidt tot opnieuw indienen, na drie mislukte pogingen is de stap `mislukt` met een logregel en een melding; een intern document naar een adres dat niet toegestaan is voor intern wordt geweigerd, zonder terugval; elke aanroep geeft precies één auditregel, zonder tekst. Op de werkplek `docling_sync.sh` exitcode 0: de conversie van D1 eindigt met HTTP 200 terwijl het werkmodel wordt aangevraagd en `ttl` verloopt, en docling-serve stopt pas daarna (`docker events`). Op de werkplek `scripts/signaalsysteem/controle/docling_fragmentatie.sh` exitcode 0: D2 (Wageningen, `proeven.md` 1.6) door docling-serve en `ChunkExtractor`; elk fragment heeft sectiepad en paginanummer, en de tekst "weerstandsvermogen" komt voor.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`. Deels uit te voeren op de werkplek.

**F2-32 Terugval docling: asynchroon met bundeling en vergrendeling** — voorwaardelijk
- *Voorwaarde:* `docling_sync.sh` uit F2-17 faalt: llama-swap stopt docling-serve ook met een openstaande aanvraag (ADR 0009, terugval).
- *Wat:* de component gebruikt `/upstream/docling/v1/convert/file/async` met pollen elke 10 s; Langflow-componenten bundelen taken per dienst en houden een vergrendeling, zodat geen andere zware dienst wordt aangevraagd zolang een docling-taak loopt; bij HTTP 404 op een taak hooguit drie keer opnieuw indienen, daarna `mislukt` met melding.
- *Hangt af van:* F2-17.
- *Bestanden:* `services/langflow/components/signaalsysteem/docling.py`; nieuw `services/langflow/components/signaalsysteem/vergrendeling.py`; `services/langflow/tests/`; `scripts/signaalsysteem/controle/docling_sync.sh`.
- *ADR:* 0009, 0014.
- *Stopconditie:* `services/langflow/tests/test_vergrendeling.py`: zolang een docling-taak loopt, wacht een aanvraag voor het werkmodel of Kev tot de taak klaar is; 404 tijdens het pollen leidt tot opnieuw indienen, na drie keer `mislukt`. Op de werkplek `docling_sync.sh --async` exitcode 0: de taak van D1 eindigt in `success` met een gelijktijdige aanvraag voor het werkmodel.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`. Uit te voeren op de werkplek.

**F2-18 Poortwachter, regeldeel, op lokaal geëxtraheerde tekst**
- *Wat:* component die brontype en route uit de bronkaart bepaalt en interne kenmerken zoekt met regels (rubriceringsmarkering, conceptstatus, interne afzender, map en bestandskenmerken). De poortwachter werkt op tekst die lokaal is gemaakt (ADR 0019): de tekstlaag van een PDF via PyMuPDF, of anders (bijvoorbeeld bij een scan) de lokale docling-serve via llama-swap, vóór elke externe verwerking. Is geen lokale tekst te maken, dan blijft het document `intern` en gaat het naar de wachtrij. Een treffer zet het document op `intern` en in de wachtrij. De poortwachter kan nooit van intern naar extern. Pas na een bevestiging `extern` mag een externe verwerkingsdienst het document zien. Het modeldeel met Kev volgt in F3-04.
- *Hangt af van:* F2-16, F2-13, F2-17.
- *Bestanden:* nieuw `services/langflow/components/signaalsysteem/poortwachter.py`, `lokale_tekst.py`; nieuw `config/poortwachter-regels.yaml`.
- *ADR:* 0001, 0019; FO 4, 6.
- *Stopconditie:* `services/langflow/tests/test_poortwachter_regels.py` met synthetische documenten: een externe batch met een rubriceringsmarkering wordt `intern` en komt in de wachtrij; invoer `intern` geeft nooit uitvoer `extern`; zonder regel en zonder besluit `intern`; een PDF zonder tekstlaag met een onbereikbare lokale docling-serve wordt `intern` en komt in de wachtrij; in een flow met een nagebootste externe verwerkingsdienst komt geen aanroep naar dat adres vóór het besluit van de poortwachter (volgorde van aanroepen vastgelegd en getoetst).
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-19 Innameflows voor PDF en lokale map**
- *Wat:* drie flows: losse upload (voorstel van de poortwachter, start na bevestiging), bulkupload (één classificatie en route voor de batch) en mapcontrole (watchdog-logica uit `file_watcher.py`; nieuwe documenten gaan naar de innamewachtrij en wachten op de beheerder). Elke flow: registratie met inhoudshash, poortwachter, docling, fragmentatie, import en de stappen `embed`, `entity_extraction` en `graph`. Originelen komen in het volume `bronarchief`.
- *Hangt af van:* F2-17, F2-18, F2-14, F2-15; F2-32 als die nodig is.
- *Bestanden:* nieuw `services/langflow/flows/pdf_upload.json`, `pdf_bulkupload.json`, `mapcontrole.json`; nieuw `services/langflow/components/signaalsysteem/mapcontrole.py`, `registratie.py`.
- *ADR:* 0010, 0019, 0011.
- *Stopconditie:* `services/langflow/tests/test_mapcontrole.py`: een nieuw bestand levert precies één wachtrij-item en geen import; `test_registratie.py`: dezelfde inhoud geeft dezelfde hash; de drie flows staan in `services/langflow/flows/` en gebruiken alleen typen van de componentenlijst (`test_componentenlijst.py`). De integratie toetst F2-24.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-31 Toets herhaalbaarheid**
- *Wat:* FO 2, principe 6: dezelfde bronversie met dezelfde pipelineversie geeft hetzelfde resultaat. Een script verwerkt D2 twee keer met dezelfde flow (zelfde Git-commit en flow-hash), de tweede keer als run met `trigger = herverwerking` op dezelfde bronversie (ADR 0003), en vergelijkt de uitkomst van beide runs.
- *Hangt af van:* F2-19.
- *Bestanden:* nieuw `scripts/signaalsysteem/toetsen/herhaalbaarheid.py`, `scripts/signaalsysteem/toetsen/herhaalbaarheid.sh`.
- *ADR:* 0003, 0005; FO 2.
- *Stopconditie:* `herhaalbaarheid.sh` exitcode 0 op de werkplek: fragmenten (tekst, sectiepad, pagina, positie, volgorde) en `doc_node`-records van beide runs zijn gelijk (hash over de gesorteerde velden); de embeddings van overeenkomstige fragmenten verschillen per component ten hoogste 1·10⁻⁶ (voorstel; de bouwer legt de gemeten maximale afwijking vast); de fragmenten van de eerste run zijn na de tweede run ongewijzigd.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`. Uit te voeren op de werkplek.

**F2-20 Werkbak als reviewomgeving**
- *Wat:* inloggen met een persoonlijke sleutel `beheerder` (Bearer), PDF-weergave met header, verbergen van menu-items op vervallen routes, scherm voor herclassificatie (met reden), en wachtrijen 1 (innamewachtrij: bron zonder regel, watchfolder, losse upload), 2 (treffer of twijfel van de poortwachter) en 7 (mislukte runs) op het patroon van het aandachtsoverzicht.
- *Hangt af van:* F2-01, F2-03, F2-13, F2-15.
- *Bestanden:* `frontend/src/lib/api/client.ts`, de PDF-weergave en het menu in `frontend/src/`; `apps/app-main/src/app_main/api/routers/attention.py`; `intake.py`.
- *ADR:* 0016, 0002, 0019.
- *Stopconditie:* `npm --prefix frontend run test` met nieuwe tests voor aanmelden met sleutel en de drie wachtrijen; `npm --prefix frontend run build`; `apps/app-main/tests/test_attention_wachtrijen.py` voor de eindpunten.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-21 Obsidian-export als Langflow-flow**
- *Wat:* `GET /api/v1/export/obsidian/entities?since=` (recht `lezen_intern` of `pijplijn`, ADR 0002) met id, canonieke naam, aliassen, soort, codeboekkoppelingen, herkomst en classificatie. Component met sjabloon volgens de vaultregels van `D:\# Vault` (FO 10, ADR 0018): alleen in bestaande notities onder `20 Atlas/28 Entities/` (programma's in `28.06 Programma's`); soort herkend aan de tag (`type/entity/organisation`, `type/entity/region`, `type/entity/government_programme`, `type/person` of `type/contact`) in een alleen-lezen index; notitie vinden op het id in het blok, anders op een unieke overeenkomst van titel of `aliases`; schrijven alleen in het blok `<!-- signaalsysteem:begin -->` … `<!-- signaalsysteem:end -->`, met id, classificatie, herkomst en codeboekkoppelingen in het blok; frontmatter nooit gewijzigd; atomair schrijven en padcontrole hergebruikt uit `obsidian_export_service.py`. Een nieuwe entiteit gaat naar de wachtrij in plaats van naar een bestand (ADR 0018, optie c; §8 B16). Geen mapnamen in code of sjabloon. **Alleen op een tijdelijke testvault:** in deze taak koppelt geen container `D:\# Vault` en schrijft niets daarin; het aansluiten op de echte vault is een latere fase (F5-05).
- *Hangt af van:* F2-16, F2-05. B8 besloten (Geert, 30-10).
- *Bestanden:* nieuw `apps/app-main/src/app_main/api/routers/export_obsidian.py` en dienst; nieuw `services/langflow/components/signaalsysteem/obsidian_export.py`, `services/langflow/templates/obsidian_entiteit.md`, `services/langflow/flows/obsidian_export.json`.
- *ADR:* 0018, 0001, 0003.
- *Stopconditie:* `services/langflow/tests/test_obsidian_export.py` op een tijdelijke testvault (in `tmp_path`, door de test opgebouwd met de map- en veldnamen uit de vaultregels en alleen openbare gegevens): twee runs na elkaar geven byte voor byte dezelfde bestanden; frontmatter en tekst buiten het blok van een bestaande notitie blijven gelijk; een interne entiteit krijgt `classificatie: intern` in het blok; een nieuwe entiteit geeft een reviewitem en geen bestand; een wikilink in het blok verwijst alleen naar een bestaande notitie; geen schrijfactie buiten `20 Atlas/28 Entities/` en geen nieuw, verplaatst of verwijderd bestand. `tests/test_compose_signaalsysteem.py`: geen dienst koppelt `D:\# Vault` of een andere vault. `apps/app-main/tests/test_export_obsidian_api.py`: classificatie intern zodra één onderliggend fragment intern is.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

#### E. Opruimen en toets

**F2-22 Verwijderen: ingebouwde inname en keten**
- *Wat:* weg gaan de routes in `sources_upload.py` die bronnen maken of opnieuw verwerken, `process-url`, `process-document` en `process-audio` in `agents.py`, `file_watcher.py` met de router `watcher` en de schakelaar in `config.py`, de router `zotero` (de client in `packages/zotero-integration` blijft voor F4-05), de OKF-import, de stappen INGEST en INSIGHTS en `advance_source` in `source_pipeline.py`, `SourceExtractor`, `SourceProcessor` met het verwijderpad voor fragmenten (`repositories/source.py:1158-1176`), `services/parsing/`, de jobtypen `DOCUMENT_PARSE` en `AUDIO_TRANSCRIBE` met hun handlers, en `LibrarianService.enqueue_due`.
- *Hangt af van:* F2-19, F2-14, F2-15, F2-29.
- *Bestanden:* genoemde bestanden in `apps/app-main/src/app_main/`; `packages/shared/src/shared/types/enums.py`; `packages/surrealdb-service/src/surrealdb_service/repositories/source.py`; `frontend/src/` (upload, Zotero, mapbewaker); `ARCHITECTURE.md` §14.
- *ADR:* 0010, 0003, 0004.
- *Stopconditie:* verwijdercontrole V met `tests/test_geen_verwijzingen_inname.py`; `test_enige_schrijfroute.py` uit F2-14 en F2-29 blijft groen.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-23 Verwijderen: oude GPU-diensten, oude export en `docker-compose.yml`**
- *Wat:* weg gaan `services/docling/`, `services/whisperx/`, `services/mineru/`, `services/summarization/`; in `pipelines/ingestion` `parsers/docling_parser.py`, `workflow.py`, `cli.py`, `audio/transcriber.py` en de exporters als niets ze meer gebruikt; `obsidian_export_service.py`, `vault_sync_service.py` en de schrijvende routes in `vault.py`. `docker-compose.signaalsysteem.yml` vervangt `docker-compose.yml`; er blijven geen twee Compose-bestanden. Volumes en gegevens blijven staan; `./surreal_data` blijft het archief van de oude database (B3).
- *Hangt af van:* F2-21, F2-22, F2-24 geslaagd. B6 besloten: de bestaande omgeving stopt na een geslaagde F2-24.
- *Bestanden:* genoemde mappen en bestanden; `docker-compose.yml`; `ARCHITECTURE.md` §1 en §4.
- *ADR:* 0014, 0015, 0018, 0002.
- *Stopconditie:* verwijdercontrole V met `tests/test_geen_verwijzingen_oude_diensten.py`; `tests/test_compose_env_is_consumed.py` aangepast aan het nieuwe Compose-bestand en groen.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F2-24 Toets fase 2**
- *Wat:* script dat de keten van begin tot eind draait op de werkplek: D2 en een tweede openbare PDF via bulkupload (extern), D1 via de watchfolder met een besluit van de beheerder via de API (intern), en de stappen tot en met `graph`, op een database die alleen de migraties en de referentiegegevens uit F2-28 bevat. Daarna de GPU-toets met een lopende synchrone docling-conversie (D1) en tegelijk een aanvraag voor het werkmodel en voor embeddings. Tot slot de contracttest voor harde eis 1 en de netwerkcontrole.
- *Hangt af van:* F2-19, F2-20, F2-08, F2-09, F2-02, F2-22, F2-28, F2-29, F2-30, F2-31.
- *Bestanden:* nieuw `scripts/signaalsysteem/toetsen/fase2.sh`, `fase2_herleidbaarheid.surql`.
- *ADR:* 0003, 0009, 0002, 0001.
- *Stopconditie:* `fase2.sh` exitcode 0: (a) 0 fragmenten en 0 `doc_node`-records zonder `source_version`, `run`, `run.pipeline_version` of (bij fragmenten) `extractor`, 0 embeddings zonder `run` en `embedding_model`, en elke run met modelaanroepen heeft een niet-lege `models`; (b) 0 geheugenfouten in de logs van llama-swap en de diensten, 0 tijdvakken met twee zware diensten tegelijk (`docker events`), de docling-conversie eindigt met HTTP 200, eventueel na opnieuw indienen; (c) vrij werkgeheugen van de host blijft boven de grens uit risico R1; (d) zolang F2-25 niet is uitgevoerd: llama-swap start met de CUDA-omweg (`NVIDIA_DISABLE_REQUIRE=1` en `LD_LIBRARY_PATH`, ADR 0006) en laadt werkmodel en embeddingmodel op de GPU; het script meldt de CUDA-versie van de driver; (e) **contracttest eis 1:** `test_contract_lezen_extern.py` en `test_rechten_pijplijn.py` slagen tegen de draaiende omgeving met de gegevens van de keten: geen intern id of interne tekst in een antwoord voor `lezen_extern`, en `pijplijn` krijgt 403 op herclassificatie (bron, notebook, notitie), extern bevestigen, het besluit over een wachtrij-item, het triagebesluit en sleutelbeheer; (f) `netwerk.sh` en `langflow.sh` slagen; (g) de zoektests van F2-02 en F2-22 slagen (geen omzeilroutes, geen oude inname).
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`. Uit te voeren op de werkplek.

**F2-25 Omweg voor de CUDA-driver laten vervallen** — voorwaardelijk
- *Voorwaarde:* Geert werkt de driver voorlopig niet bij (30-10). De taak start pas als NVIDIA een driver met CUDA ≥ 13.4 heeft uitgebracht en Geert die heeft geïnstalleerd. Tot dan blijft de omweg uit ADR 0006 staan en toetst F2-24 dat llama-swap met de omweg start.
- *Wat:* na de driverupdate gaan `NVIDIA_DISABLE_REQUIRE=1` en `LD_LIBRARY_PATH` (ADR 0006, ADR 0009, `proeven.md` §8 punt 23) uit `docker-compose.signaalsysteem.yml`. Het criterium van P0 wordt aangescherpt van CUDA ≥ 13.0 naar ≥ 13.4, de eis van de vastgezette llama-swap-tag.
- *Hangt af van:* F2-11; een uitgebrachte driver met CUDA ≥ 13.4, geïnstalleerd door Geert.
- *Bestanden:* `docker-compose.signaalsysteem.yml`; `scripts/signaalsysteem/proeven/p0_omgeving.sh`; `tests/test_compose_signaalsysteem.py`.
- *ADR:* 0006, 0009.
- *Stopconditie:* `scripts/signaalsysteem/proeven/p0_omgeving.sh --gpu` exitcode 0 en meldt CUDA ≥ 13.4; `llama-swap` start zonder de twee variabelen en laadt het werkmodel op de GPU (`scripts/signaalsysteem/controle/gpu.sh` exitcode 0); `tests/test_compose_signaalsysteem.py` faalt als een van beide variabelen nog in het Compose-bestand staat.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`. Uit te voeren op de werkplek.

### Fase 3: labeling en MCP

*Toets (F3-12):* precisie en recall zijn gemeten op de referentieset, en geen intern fragment verschijnt in een externe uitvraag (F3-11).

**F3-01 Kev-4B onder llama-swap, opnieuw gemeten**
- *Wat:* image `services/kev/Dockerfile` (uit `p1/kev.Dockerfile`: tag `kev-1.0`, `uv sync --extra serve`), adapter en basismodel op vaste revisie in een volume, `--host 0.0.0.0`, `KEV_API_KEY` als geheimbestand. In llama-swap groep `zwaar`, `checkEndpoint: /openapi.json`, proxy naar Kev. Een Kev-client in `llm_manager` voor `/upstream/kev-4b/v1/systemone`, via `ModelGate`. Register: lokaal, toegestaan voor intern. **Opnieuw meten** (ADR 0009, na de review): Kev-4B met fragmenten van ten minste 512 tokens (de langste fragmenten van D1 en D2 uit de pipeline van fase 2; openbaar) en met de grootste keuzelijst (alle Iv3-taakvelden uit codeboek 2025, F3-03), samen met mxbai geladen, terwijl het bureaublad van Windows in gewoon gebruik is; de nulmeting met het actieve bureaublad wordt vastgelegd.
- *Hangt af van:* F2-11, F2-07, F2-17, F3-03.
- *Bestanden:* nieuw `services/kev/Dockerfile`; `services/llama-swap/config.yaml`; nieuw `packages/llm-manager/src/llm_manager/kev_client.py`; `config/modelregister.yaml`; `docker-compose.signaalsysteem.yml`; nieuw `scripts/signaalsysteem/controle/kev.sh`.
- *ADR:* 0009, 0006.
- *Stopconditie:* `packages/llm-manager/tests/test_kev_client.py` tegen een nagebootste server. Op de werkplek `kev.sh` exitcode 0: HTTP 200 en kansen per `choice`-vraag 1 ± 0,01; 50 vragen met fragmenten van ≥ 512 tokens en de volledige Iv3-keuzelijst zonder geheugenfout met het embeddingmodel geladen; nulmeting met actief bureaublad en piek-VRAM vastgelegd, piek plus nulmeting ≤ 15 279 MiB. Faalt dat, dan legt de bouwer Kev-0,8B of een gekwantiseerde variant (F3-13) voor aan Geert.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`. Uit te voeren op de werkplek.

**F3-02 Referentieset uit de vault: NOVEX en NPVR**
- *Wat:* 200 tot 300 fragmenten met gouden labels (besluit Geert, 30-10). Een script leest alleen `D:\# Vault\50 Resources\51 Raw\51.01 Sources\NOVEX` en `D:\# Vault\50 Resources\51 Raw\51.01 Sources\NPVR`, zonder schrijfactie, en neemt de labels uit de metadata in die mappen; labels uit de vault zijn metadata. **Alleen openbare bronnen:** het script neemt alleen een bron op waarvan de metadata een openbare vindplaats noemt, en slaat de rest over met een telling (harde eis "geen interne documenten in de repository of in tests"); de bouwer legt vast hoe "openbaar" wordt vastgesteld. **Kopie buiten de vault** (ADR 0004): het script kopieert de originelen naar een lokale map van de set buiten de vault en buiten de repository (pad uit een instelling), met SHA-256 per bestand; de pipeline van fase 2 (bulkupload met classificatie `extern`, docling-serve en fragmentatie) verwerkt die kopieën. Zo wijst geen bronkaartregel naar de vault (F2-13), en komt geen fragment uit de tekst van een vaultnotitie (harde eis 3). Iv3-taakvelden (Informatie voor derden) zijn niet af te leiden en krijgen een handmatige steekproef. Geert controleert een steekproef van ongeveer 50 fragmenten. Voor de poortwachter krijgt een deel van de fragmenten synthetische interne kenmerken (B9). **Buiten Git:** de set (labels, kopieën, fragmenten) staat lokaal op de werkplek en komt niet in de repository. In de repository staan alleen het script, het schema en een schematest met een synthetisch voorbeeld.
- *Hangt af van:* F2-17, F3-03.
- *Bestanden:* nieuw `scripts/signaalsysteem/referentieset/uit_vault.py`, `steekproef.py`; nieuw `tests/signaalsysteem/referentieset/schema.json`, `voorbeeld_synthetisch.jsonl`; nieuw `tests/signaalsysteem/test_referentieset_schema.py`; `.gitignore` (pad van de lokale set).
- *ADR:* FO 7; 0004, 0008, 0013, 0018 (vaultregels, alleen lezen).
- *Stopconditie:* `test_referentieset_schema.py` op het synthetische voorbeeld: schema geldig; `uit_vault.py` op een tijdelijke testvault opent geen bestand om te schrijven, leest niets buiten de twee mappen, schrijft alleen in de doelmap buiten de vault, en neemt geen tekst uit een notitie als fragmenttekst. Op de werkplek: `uit_vault.py --controleer <set>` exitcode 0: 200 ≤ aantal ≤ 300; elk fragment verwijst naar een kopie buiten de vault met SHA-256 en naar een bron met openbare vindplaats; geen bronkaartregel van de set heeft een locatie in de vault; elk label bestaat in de vaultregels, in een codeboekversie (F3-03, F3-15) of in de referentielijst; geen dubbele fragmenten; de telling van overgeslagen bronnen is vastgelegd; de steekproef van Geert (ongeveer 50 fragmenten) en de Iv3-steekproef zijn vastgelegd. `git status` toont de set niet.
- *Agent:* `signaal-bouwer` voor de tooling, steekproeven door Geert; daarna `signaal-reviewer`. Deels uit te voeren op de werkplek.

**F3-13 Proef: Kev-4B gekwantiseerd (8-bit en 4-bit) naast bf16**
- *Wat:* Kev-4B in 8-bit en in 4-bit naast bf16 (besluit Geert, 30-10). Doel: ruimte naast Kev voor het embeddingmodel en eventueel de reranker op de GPU (ADR 0015 5b). Per variant: piek- en rustgeheugen gemeten zoals in P3 (nulmeting, samen met mxbai), en de kalibratie op de referentieset tegen bf16: ECE (Expected Calibration Error), Brier-score en precisie en recall per label. Er is geen officiële gekwantiseerde Kev-4B; wel een communityconversie in W4A16 ([bowmanslayer/kev-4b-W4A16](https://huggingface.co/bowmanslayer/kev-4b-W4A16), met een eigen regressietest van 6 van 72 gewijzigde keuzes) en GGUF-varianten alleen voor 0,8B ([mys/kev-0.8b-GGUF](https://huggingface.co/mys/kev-0.8b-GGUF)). Kev-4B is een LoRA-adapter met keuzekop op Qwen3.5-4B-Base ([OpenRouter](https://openrouter.ai/jaredpalmer/kev-4b)). De bouwer legt per variant vast hoe die is gemaakt (conversie of eigen kwantisering van het basismodel) en met welke revisie; een variant die niet te maken is, staat als "niet gemeten" in de tabel.
- *Grens (besloten, Geert, 30-10; ADR 0009):* ECE ten hoogste 0,02 hoger dan bij bf16, en precisie en recall per label elk ten hoogste 0,02 lager. De grens staat in `config/kev-kwantisering.yaml`, gecommit vóór de meting. De kleinste variant binnen de grens wordt gekozen; haalt geen variant de grens, dan blijft bf16.
- *Hangt af van:* F3-01, F3-02. De drempels van de poortwachter wachten op de uitkomst.
- *Bestanden:* nieuw `scripts/signaalsysteem/toetsen/kev_kwantisering.py`, `scripts/signaalsysteem/controle/kev_kwantisering.sh`; nieuw `config/kev-kwantisering.yaml`; eventueel `services/kev/Dockerfile` en `services/llama-swap/config.yaml` (alleen als een variant wordt gekozen).
- *ADR:* 0009 (GPU-budget, Kev-4B, grens), 0015 (plek van de reranker, 5b); de uitkomst wordt in beide verwerkt.
- *Stopconditie:* `kev_kwantisering.sh` exitcode 0 op de werkplek en een tabel in `docs/signaalsysteem/meting-kev-kwantisering.md` met per variant (bf16, 8-bit, 4-bit): piek- en rustgeheugen, met mxbai, ECE, Brier, precisie en recall per label, en het verschil met bf16; plus de keuze volgens de vooraf vastgelegde grens. `git log` toont `config/kev-kwantisering.yaml` vóór de commit met de meting. De tabel bevat geen fragmenttekst (de set staat buiten Git).
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`. Uit te voeren op de werkplek.

**F3-03 Codeboek Iv3 2025 en BWB-woordenlijst**
- *Wat:* laadscript voor de Iv3-taakvelden 2025 naar een ontologie met versie in `ontology_version`; BWB-leverancier naast `tooi_provider.py` die `reference_entity` vult; beide als set in `config/referentiegegevens.yaml` en via het laadscript uit F2-28; eindpunt `GET /api/v1/codeboek/{schema}/{versie}` (recht `pijplijn` of `beheerder`) voor de keuzelijst van de Kev-component. De vindplaats van beide openbare bronnen zoekt de bouwer uit (ADR 0013).
- *Hangt af van:* F2-01, F2-28.
- *Bestanden:* nieuw `scripts/signaalsysteem/laad_iv3_taakvelden.py` (aangeroepen door `laad_referentiegegevens.py`); nieuw `packages/shared/src/shared/vocabulary/bwb_provider.py`; `packages/ontology-manager/`; `config/referentiegegevens.yaml`; nieuw `apps/app-main/src/app_main/api/routers/codeboek.py`.
- *ADR:* 0013, 0021.
- *Stopconditie:* `packages/shared/tests/test_bwb_provider.py` op een klein openbaar voorbeeld; `tests/signaalsysteem/test_iv3_laden.py`: het aantal geladen taakvelden is gelijk aan het aantal in het bronbestand (SHA-256 vastgelegd); twee keer laden geeft één versie; `referentiegegevens.sh` uit F2-28 slaagt met Iv3 en BWB in de inventaris.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F3-15 Codeboeken brede welvaart en foundational thema's**
- *Wat:* de twee andere codeboeken uit FO 7 (labels): de brede-welvaartthema's en de foundational thema's voor signaalweging, elk als ontologie met versie, met hetzelfde laadscript en eindpunt als F3-03. De bron van beide lijsten is niet vastgelegd (§8, B23; aanname A12); de vault noemt `20 Atlas/28 Entities/28.04 Thema's` als plek van het codeboek (FO 10).
- *Hangt af van:* F3-03. Vooraf het besluit over de bron (B23).
- *Bestanden:* nieuw `scripts/signaalsysteem/laad_codeboek.py`; `config/referentiegegevens.yaml`; `packages/ontology-manager/`.
- *ADR:* 0013 (uitbreiding, voorgesteld); FO 7.
- *Stopconditie:* `tests/signaalsysteem/test_codeboeken_laden.py`: per codeboek is het aantal geladen termen gelijk aan het aantal in het bronbestand (SHA-256 vastgelegd); twee keer laden geeft één versie; `GET /api/v1/codeboek/{schema}/{versie}` geeft de lijst voor beide; de repository bevat geen inhoud uit de vault (zoektest op het bronbestand als dat uit de vault komt).
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F3-04 Poortwachter, modeldeel met Kev**
- *Wat:* Kev beoordeelt wat de regels openlaten: documentsoort, route en interne kenmerken, met een gekalibreerde kans per keuze, op lokaal geëxtraheerde tekst (F2-18). Onder de drempel gaat de bron naar wachtrij 2. De drempel volgt uit de meting op de referentieset en staat in `config/poortwachter-regels.yaml`.
- *Hangt af van:* F2-18, F3-01, F3-02, F3-13 (gekozen Kev-variant).
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
- *Wat:* LangGraph-graaf met knopen extraheren (`EntityExtractionService` en de orchestrator uit `pipelines/ontology-extraction`), valideren tegen `reference_entity`, alleen afgekeurde voorstellen opnieuw met een maximum aantal rondes, en wegschrijven met `run` en de afgeleide classificatie (F2-03). Stap `entity_extraction` via `RunStepOrchestrator`. De lezers van `provenance_chain` gaan over op de vermeldingen met run.
- *Hangt af van:* F2-15, F3-03, F2-28.
- *Bestanden:* nieuw `apps/app-main/src/app_main/graphs/entity_extraction.py`; `services/entity_extraction_service.py`, `entity_persistence_service.py`; `services/run_step_orchestrator.py`; de lezers van `provenance_chain` (onder meer `apps/app-main/src/app_main/services/entity_resolution/recanonicalization_service.py`, `notebook_merge_service.py`, `okf_export_service.py`, `semantic_intelligence/reasoning.py`).
- *ADR:* 0012, 0003, 0001.
- *Stopconditie:* `apps/app-main/tests/test_graph_entity_extraction.py`: een afgekeurd voorstel gaat opnieuw, een goedgekeurd niet; stop na het maximum; elke vermelding heeft `run` en `classification`. De regressiepoort van Track N (`scripts/n_extraction_gate.py` tegen `tests/regression/n_extraction_baseline.json`) slaagt.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F3-14 Kwaliteitsborging entiteiten: NPVR-regio's en canonieke naam**
- *Wat:* FO 7, kwaliteitsborging. (1) Een regio uit het Nationaal Programma Vitale Regio's (NPVR) verwijst ook naar de onderliggende gemeenten: een relatie van de regio naar elke gemeente, met de CBS-gemeentecode, uit een openbare lijst als set in `config/referentiegegevens.yaml` (vindplaats zoekt de bouwer uit). (2) Bij fusie of opvolging geldt de meest recente naam: de entiteit krijgt de naam van de opvolger als canonieke naam en de eerdere naam als alias, op basis van de opvolgingsgegevens in TOOI (aanname A10); een opvolging die niet uit de referentiegegevens volgt, gaat naar de triage.
- *Hangt af van:* F3-06, F2-28.
- *Bestanden:* `apps/app-main/src/app_main/services/entity_resolution/recanonicalization_service.py`; `packages/shared/src/shared/vocabulary/tooi_provider.py`; `config/referentiegegevens.yaml`; nieuw `scripts/signaalsysteem/laad_npvr_regios.py`.
- *ADR:* 0013, 0012; FO 7.
- *Stopconditie:* `apps/app-main/tests/test_npvr_regio_gemeenten.py`: na het laden heeft elke NPVR-regio precies de gemeenten uit de openbare lijst (aantal en CBS-codes gelijk, SHA-256 van de lijst vastgelegd); `apps/app-main/tests/test_canonieke_naam_opvolging.py` met een openbaar voorbeeld van een gemeentelijke herindeling uit TOOI: een vermelding van de opgeheven gemeente landt op de opvolger, met de oude naam als alias; een opvolging zonder referentiegegevens geeft een triage-item.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F3-07 Graaf `entity_dedup` met pauze bij twijfel**
- *Wat:* de lagen van `FilteringWorkflow` als knopen; bij twijfel een item in `triage_queue` en `interrupt()`; het triagebesluit (recht `beheerder`) via de router `triage` hervat de graaf met hetzelfde `thread_id`; checkpoints in het bestaande SQLite-bestand. De stap staat intussen op `wacht_op_review`.
- *Hangt af van:* F3-06, F2-06.
- *Bestanden:* nieuw `apps/app-main/src/app_main/graphs/entity_dedup.py`; `pipelines/entity-filtering/src/entity_filtering/workflow.py` (alleen aanhaakpunten); `api/routers/triage.py`; `services/run_step_orchestrator.py`.
- *ADR:* 0012, 0011, 0002 B.
- *Stopconditie:* `apps/app-main/tests/test_graph_entity_dedup.py`: bij twijfel stopt de graaf en staat de stap op `wacht_op_review`; na een triagebesluit hervat hij met dezelfde tussenstand; een verouderde `rev` op het besluit geeft 409; `test_rechten_pijplijn.py`: het hervattende triagebesluit geeft 403 met `pijplijn`.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F3-08 Verwijderen: `provenance_chain`** — vraagt expliciete bevestiging van Geert
- *Wat:* het veld `provenance_chain` op `entity` en `relation` vervalt met een migratie. De inhoud gaat verloren; de herkomst staat dan in de vermeldingen met run.
- *Hangt af van:* F3-06.
- *Bestanden:* `migrations/<n>.surrealql` en `_down`; `packages/shared/src/shared/models/entity.py`; `packages/surrealdb-service/src/surrealdb_service/repositories/entity.py`.
- *ADR:* 0003.
- *Stopconditie:* verwijdercontrole V met `tests/test_geen_verwijzingen_provenance_chain.py`; reservekopie van de database aanwezig vóór de migratie.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F3-09 Herkomst-, versie- en fragmenteindpunten**
- *Wat:* `GET /api/v1/chunks/{id}/provenance` (bron, bronversie, run, pipelineversie, modellen), `GET /api/v1/sources/{id}/versions` en de leesroute voor fragmenten `GET /api/v1/chunks/{id}` (tekst, classificatie, bronversie en run), met classificatiebereik. Deze komen op de lijst voor `lezen_extern`. De fragmentroute is de route waarmee `cloud-gateway` elk fragment ophaalt (ADR 0022 C, F5-07). Entiteiten blijven voor `lezen_extern` afgesloten tot het besluit in §8, B10.
- *Hangt af van:* F2-05, F2-04.
- *Bestanden:* nieuw `apps/app-main/src/app_main/api/routers/provenance.py`, `chunks.py`; `packages/surrealdb-service/src/surrealdb_service/repositories/graph.py`, `source_version.py`, `search.py`.
- *ADR:* 0002 B, D; 0003; 0022 C; FO 10.
- *Stopconditie:* `apps/app-main/tests/test_provenance_api.py`: voor elk fragment een volledige keten tot run en model; een intern fragment geeft 404 bij `lezen_extern` op alle drie de routes; `GET /api/v1/chunks/{id}` geeft voor een extern fragment de tekst en de classificatie; `test_contract_lezen_extern.py` uitgebreid met deze routes.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F3-10 open-notebook-mcp, `mcp-toegang` en de reranker in Compose**
- *Wat:* container met `open-notebook-mcp==0.3.0` en `mcp==1.30.0` op `verwerking` en `mcp-intern`, een startscript (uit `p4/start_mcp.py`) dat de sleutel `lezen_extern` uit het geheimbestand leest en `host`, `port` en `path` via `mcp.settings` zet. **Startcontrole** (ADR 0002 D): vóór het starten vraagt het script `GET /api/v1/auth/scope` (F2-01); meldt de API niet precies `EXTERN`, of is de API niet bereikbaar, dan start de server niet en eindigt het script met een foutcode. `mcp-toegang` op `mcp-intern` en `toegang` stuurt alleen `/mcp` door en publiceert alleen op `127.0.0.1` (en via Tailscale op de host), volgens de uitkomst van de masquerade-proef (F2-10). De reranker als CPU-dienst op `verwerking`, via `ModelGate` (ADR 0015 5c).
- *Hangt af van:* F2-10, F2-05, F2-01.
- *Bestanden:* nieuw `services/open-notebook-mcp/Dockerfile`, `start_mcp.py`; nieuw `services/open-notebook-mcp/tests/test_start_mcp.py`; `docker-compose.signaalsysteem.yml`; `services/proxy/` (MCP); `services/reranker/`; `scripts/signaalsysteem/controle/netwerk.sh`.
- *ADR:* 0002 D, F; 0015.
- *Stopconditie:* `services/open-notebook-mcp/tests/test_start_mcp.py`: het startscript weigert te starten met een sleutel `lezen_intern`, `pijplijn` of `beheerder`, en als de API niet bereikbaar is; het start met `lezen_extern`. `netwerk.sh` uitgebreid, exitcode 0: `open-notebook-mcp` heeft geen internet en geen poort; op `mcp-intern` hangen alleen `mcp-toegang` en `open-notebook-mcp`; `127.0.0.1:<poort>/mcp` antwoordt; een ander pad via de proxy geeft 404. `tests/test_compose_signaalsysteem.py`: beide pakketten met `==` vastgezet.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F3-11 Acceptatietest van het filter: P4**
- *Wat:* `p4_mcp_classificatiefilter.sh` aangepast aan de nieuwe stand: een sleutel `lezen_extern` in plaats van het wachtwoord, `classification` in plaats van `private`, het image van de fork na F2-05, en de SurrealDB-versie uit F2-26 (bij v3 herhaalt dit P4 op v3, ADR 0021). Deelstap 4c met vector- en hybride zoeken via llama-swap. De testgegevens dekken ook entiteiten, relaties, `doc_node`, inzichten, notebooknamen en bronkaartregels (ADR 0002, Gevolgen).
- *Hangt af van:* F3-10, F2-08, F2-11.
- *Bestanden:* `scripts/signaalsysteem/proeven/p4_mcp_classificatiefilter.sh`, `p4/vul_testdatabase.py`, `p4/mcp_lektest.py`.
- *ADR:* 0001, 0002, 0021.
- *Stopconditie:* `scripts/signaalsysteem/proeven/p4_mcp_classificatiefilter.sh` exitcode 0: K1 nul treffers `PRF-INT-`; K2 positieve controle geldig; K3 HTTP 404 op een intern id; K4 HTTP 403 op elke schrijftool en op `ask_simple` (falen door een fout telt niet); K5 0 verwerkingsfouten; K6 HTTP 401 zonder sleutel; 4c uitgevoerd met dezelfde uitkomst.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F3-12 Toets fase 3: meting op de referentieset**
- *Wat:* meetscript voor precisie en recall van labels (alle codeboeken), entiteiten (met referentie-id, inclusief NPVR-regio's en opvolging) en de route van de poortwachter. Optioneel de embeddingvergelijking uit ADR 0008: een kandidaat moet naast Kev-4B passen.
- *Hangt af van:* F3-02, F3-04, F3-05, F3-06, F3-07, F3-11, F3-14, F3-15.
- *Bestanden:* nieuw `scripts/signaalsysteem/toetsen/fase3_referentieset.py`.
- *ADR:* FO 7, 15; 0008.
- *Stopconditie:* het script draait over alle fragmenten van de set en schrijft per onderdeel precisie, recall en aantallen naar `docs/signaalsysteem/meting-fase3.md`; F3-11 is geslaagd. De toets vraagt een meting, geen drempel; Geert beslist op basis van de uitkomst over fase 4.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

### Fase 4: bronnen en planning

*Toets (F4-12):* een geplande run verwerkt alleen nieuwe of gewijzigde inhoud.

Langflow heeft geen internet (ADR 0022). Elke component die een externe bron ophaalt, doet dat via de container `ophalen` (F4-13).

**F4-13 Container `ophalen` en de ophaalcontrole**
- *Wat:* een kleine container `ophalen` op `ophalen-intern` en `uitgaand` (ADR 0022 B). Langflow vraagt hem een bron op te halen met het id van een bronkaartregel en een adres. Per aanvraag vraagt hij de fork `GET /api/v1/source-map-rules/{id}/ophaalcontrole?url=…` (recht `ophalen`); de fork antwoordt alleen ja of nee: ja als de regel actief is, `extern` bevestigd is, en het adres de `location` van de regel is of op dezelfde host ligt. Bij nee weigert `ophalen` met 403. Alleen `GET` en `HEAD`. Hij heeft alleen de sleutel `ophalen`, geen leesrechten. Hij legt elke aanvraag vast (tijd, regel, adres, status, grootte), zonder inhoud. `langflow` en `open-notebook` komen ook aan `ophalen-intern`.
- *Hangt af van:* F2-01, F2-10, F2-13.
- *Bestanden:* nieuw `services/ophalen/Dockerfile`, `ophalen.py`, `services/ophalen/tests/`; nieuw `apps/app-main/src/app_main/api/routers/ophaalcontrole.py` (of in `source_map.py`); `docker-compose.signaalsysteem.yml`; `scripts/signaalsysteem/controle/netwerk.sh`; nieuw `services/langflow/components/signaalsysteem/ophalen_client.py`.
- *ADR:* 0022 B, 0002 B, F; 0020.
- *Stopconditie:* `services/ophalen/tests/test_ophalen.py`: een adres dat niet bij een actieve, bevestigd externe regel hoort, geeft 403; een regel `intern`, een inactieve regel of een andere host geeft 403; `POST` geeft 405; het logboek heeft geen veld voor inhoud. `apps/app-main/tests/test_ophaalcontrole.py`: het antwoord bevat alleen ja of nee; een sleutel `ophalen` krijgt op elke andere route 403. `netwerk.sh` exitcode 0: `ophalen` bereikt internet; `langflow` en `open-notebook` nog steeds niet; op `ophalen-intern` hangen alleen `langflow`, `ophalen` en `open-notebook`; `docker inspect` van `ophalen` toont alleen het geheim `sleutel_ophalen`.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F4-01 Planning volgens de bronkaart**
- *Wat:* een synchronisatiescript maakt per bronkaartregel met schema een trigger via de API van Langflow (cron, `Europe/Amsterdam`, `coalesce` of `skip`), met het id van de regel in de payload; een verwijderde of inactieve regel zet de trigger uit.
- *Hangt af van:* F2-12, F2-13.
- *Bestanden:* nieuw `services/langflow/scripts/schema_synchroniseren.py`; `services/langflow/tests/`.
- *ADR:* 0017 C.
- *Stopconditie:* `scripts/signaalsysteem/controle/planning.sh` exitcode 0, met de criteria K2–K4 van P2 op een regel van elke minuut; elke run heeft `trigger = schema` en het id van de regel.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F4-02 Feedcontrole voor YouTube en podcasts, Spotify naar RSS**
- *Wat:* component op basis van `p6/feedcontrole.py` met `feedparser==6.0.14`, die feeds via `ophalen` haalt: detectie op video-id of GUID, elk nieuw item een eigen bron; de fork beantwoordt of een extern id al bestaat. Het veld Inhoud ophalen wordt gevolgd: van YouTube standaard alleen metadata. Spotify-link via iTunes Search en oEmbed naar de RSS-feed (Really Simple Syndication).
- *Hangt af van:* F4-01, F2-14, F4-13.
- *Bestanden:* nieuw `services/langflow/components/signaalsysteem/feedcontrole.py`, `spotify_naar_rss.py`; `apps/app-main/src/app_main/api/routers/sources.py` (opzoeken op extern id).
- *ADR:* FO 5; 0019, 0022 B.
- *Stopconditie:* `services/langflow/tests/test_feedcontrole.py` op ingekorte opnamen van D3 en D4 (openbaar): criteria K1–K4 van P6 als test; elke ophaalactie loopt via een nagebootste `ophalen`; `test_spotify_naar_rss.py` met opgenomen antwoorden: criterium K5.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F4-03 WhisperX onder llama-swap en transcriptiecomponent**
- *Wat:* container `whisperx` (variant A) als vooraf aangemaakte container in groep `zwaar`, met `checkEndpoint: /ping`, `OPT_OUT_TRACKING=true`, `HF_HUB_OFFLINE=1` en volumes voor Whisper, uitlijnmodellen en NLTK `punkt_tab`. Component zet segmenten met tijdcodes en sprekers om in fragmenten, met het model uit `pipelines/ingestion/src/ingestion/models/transcription.py`. Elke aanroep van WhisperX schrijft een auditregel via `POST /api/v1/model-calls` (FO 9, audit); WhisperX staat in het modelregister.
- *Hangt af van:* F2-11, F2-16.
- *Bestanden:* `docker-compose.signaalsysteem.yml`; `services/llama-swap/config.yaml`; `config/modelregister.yaml`; nieuw `services/langflow/components/signaalsysteem/transcriptie.py`; `scripts/signaalsysteem/modellen_downloaden.py`.
- *ADR:* 0014, 0009, 0007.
- *Stopconditie:* op de werkplek `scripts/signaalsysteem/controle/whisperx.sh` exitcode 0: D7 en D8 zonder `language` geven taal `nl` en `en` met woordtijden; D4 ≥ 2 sprekers; geen internet op `verwerking`; de niet gemeten combinatie embed + WhisperX gemeten en ≤ 15 279 MiB; vrij werkgeheugen van de host vastgelegd. `services/langflow/tests/test_transcriptie.py` op een opgenomen antwoord, met precies één auditregel per aanroep en zonder tekst.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`. Uit te voeren op de werkplek.

**F4-04 Website-connector met wijzigingsdetectie**
- *Wat:* pagina of sitemap ophalen via `ophalen`, HTML via docling-serve, wijzigingsdetectie op inhoudshash.
- *Hangt af van:* F2-17, F4-01, F4-13.
- *Bestanden:* nieuw `services/langflow/components/signaalsysteem/website.py`, `services/langflow/flows/website.json`.
- *ADR:* FO 5, 7; 0022 B.
- *Stopconditie:* `services/langflow/tests/test_website.py` tegen een lokale testserver met een openbare pagina als opname: ongewijzigd geeft `niets_nieuws`, een wijziging één nieuwe bronversie.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F4-05 API-connector en Zotero als API-bron**
- *Wat:* connector die API-velden naar tekst en metadata zet en bijlagen als document doorgeeft, bijvoorbeeld voor raadsmoties; de Zotero-client uit `packages/zotero-integration` als tweede connector. Beide halen via `ophalen` (alleen `GET` en `HEAD`; zie conflict C7).
- *Hangt af van:* F2-17, F4-01, F4-13.
- *Bestanden:* nieuw `services/langflow/components/signaalsysteem/api_connector.py`, `zotero.py`.
- *ADR:* 0010, 0022 B; FO 5.
- *Stopconditie:* `services/langflow/tests/test_api_connector.py` en `test_zotero.py` op opgenomen openbare antwoorden: velden en bijlagen komen in de import; een tweede run zonder wijziging geeft `niets_nieuws`.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F4-06 MCP-bron**
- *Wat:* component met een MCP-client naar een externe server; het resultaat wordt een document. Classificatie volgens de regel; een intern systeem krijgt `intern`. *Besluit vooraf:* hoe de MCP-client de externe server bereikt, gegeven dat `ophalen` alleen `GET` en `HEAD` doorlaat (conflict C7, §8 B24).
- *Hangt af van:* F2-17, F4-01, F4-13.
- *Bestanden:* nieuw `services/langflow/components/signaalsysteem/mcp_bron.py`.
- *ADR:* 0019, 0022 B; FO 5.
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
- *Wat:* component na docling-serve voor tekstaanvulling en PDF-annotaties met PyMuPDF, zonder GPU, met hergebruik van `lokale_tekst.py` uit F2-18. GROBID (`grobid/grobid:0.8.2-crf`) op `verwerking`, aangeroepen voor wetenschappelijke PDF's; de verwijzingen gaan via een stap `cites` naar de bestaande code van Track V.
- *Hangt af van:* F2-17, F2-15, F2-18.
- *Bestanden:* nieuw `services/langflow/components/signaalsysteem/pymupdf_aanvulling.py`, `grobid.py`; `docker-compose.signaalsysteem.yml`; `services/run_step_orchestrator.py`.
- *ADR:* 0015.
- *Stopconditie:* `services/langflow/tests/test_pymupdf_aanvulling.py` met een synthetische PDF met annotatie; `test_grobid.py` met een openbaar wetenschappelijk artikel (open licentie): verwijzingen geparsed en `cites` aangemaakt.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F4-10 Bestaande bronnen opnieuw inlezen** — vervallen (B3, 30-10)
- De nieuwe omgeving start met een lege database; er zijn geen bronversies `pre_signaalsysteem`. Wie bronnen uit de oude omgeving terug wil, leest de originelen in via de gewone flows (F2-19, F4-02 tot en met F4-06).

**F4-11 Opruimen van de runs `pre-signaalsysteem`** — vervallen (B3, 30-10)
- Er zijn geen runs `pre-signaalsysteem` om op te ruimen. Het oude datavolume `./surreal_data` blijft onaangeroerd staan als archief; geen taak raakt het aan.

**F4-12 Toets fase 4**
- *Wat:* geplande runs op een vaste testfeed en een testwebsite op het testnetwerk, via `ophalen`: eerste run, tweede run zonder wijziging, derde run met één gewijzigd en één nieuw item.
- *Hangt af van:* F4-01, F4-02, F4-04, F4-13.
- *Bestanden:* nieuw `scripts/signaalsysteem/toetsen/fase4.sh`.
- *ADR:* 0003, 0017, 0022; FO 15.
- *Stopconditie:* `fase4.sh` exitcode 0: tweede run `niets_nieuws` met 0 nieuwe bronversies en fragmenten; derde run precies 2 nieuwe bronversies; `langflow` heeft tijdens de toets geen internet (`netwerk.sh`).
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

### Fase 5: kennisproducten

*Toets (F5-06):* elke bewering in een wiki-pagina verwijst naar een fragment dat haar draagt.

*Besluit vooraf:* waar de wiki-generatie draait (§7, conflict C5; §8, B12), en of de gateway zo streng blijft als ADR 0022 C (§8, B20; conflict C8). De taken hieronder gaan uit van de voorstellen daar.

**F5-01 `claude-cli`-container en aanroepwijze `cli`**
- *Wat:* container met een vaste versie van de Claude-CLI (Command Line Interface), alleen op `cli-intern`, zonder volumes met brongegevens, met een lege tijdelijke werkmap per aanroep. Een kleine aanroepdienst in de container voert `claude -p --output-format json --model <model>` uit, zonder tools en zonder hervatten van een sessie, en geeft CLI-versie en exitcode terug. Alleen `cloud-gateway` roept hem aan (ADR 0022 C); het uitgaande verkeer van de CLI loopt via de gateway. De CLI-login staat als geheim alleen in deze container, gelezen door een startscript en niet als omgevingsvariabele. `ModelGate` en de Langflow-modelcomponent kennen de aanroepwijze `cli` als cloudmodel.
- *Hangt af van:* F2-16, F2-10.
- *Bestanden:* nieuw `services/claude-cli/Dockerfile`, `aanroep.py`, `start.sh`; `packages/llm-manager/src/llm_manager/gate.py`; `config/modelregister.yaml`; `docker-compose.signaalsysteem.yml`.
- *ADR:* 0006, 0022 C, F; FO 9.
- *Stopconditie:* `services/claude-cli/tests/test_aanroep.py`: elke aanroep in een nieuwe lege werkmap, zonder hervat-opties, met versie en exitcode; `packages/llm-manager/tests/test_model_gate.py` uitgebreid: `cli` geweigerd bij interne invoer; `tests/test_compose_signaalsysteem.py`: `claude-cli` hangt alleen aan `cli-intern` en heeft geen volumes behalve tmpfs; `docker inspect` van `claude-cli` toont de login niet als omgevingsvariabele.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F5-07 Container `cloud-gateway`**
- *Wat:* een kleine container `cloud-gateway` op `gateway-intern`, `cli-intern` en `uitgaand` (ADR 0022 C). Hij laat alleen verkeer naar het domein van Anthropic door; de bouwer legt de hostnamen vast uit de documentatie van Anthropic. Langflow stuurt per aanroep de run, de fragment-id's, het id van het sjabloon en het model. De gateway haalt elk fragment op met een eigen sleutel `lezen_extern` via `GET /api/v1/chunks/{id}` (F3-09); geeft de fork voor één fragment 404, dan weigert hij de hele aanroep. Hij bouwt de prompt zelf uit het sjabloon (uit Git, gecontroleerd op hash) en de opgehaalde tekst; vrije tekst van Langflow gaat niet mee. Voor de CLI roept hij de aanroepdienst van `claude-cli` aan na dezelfde controle, en is hij de uitgaande proxy van de CLI met dezelfde domeinbeperking. Bij het starten controleert hij met `GET /api/v1/auth/scope` dat zijn sleutel precies `EXTERN` is. De Claude-API-sleutel staat alleen hier, gelezen door een startscript uit `/run/secrets/` en alleen aan het gatewayproces gegeven. Hij geeft model, tokens, duur en status terug aan Langflow, dat de regel in `model_call` schrijft, en houdt een eigen logboek zonder tekst. `langflow` en `open-notebook` komen aan `gateway-intern`.
- *Hangt af van:* F5-01, F3-09, F2-16, F2-07.
- *Bestanden:* nieuw `services/cloud-gateway/Dockerfile`, `gateway.py`, `start.sh`, `services/cloud-gateway/tests/`; nieuw `services/langflow/templates/cloud/` (sjablonen); nieuw `services/langflow/components/signaalsysteem/cloud_model.py`; `docker-compose.signaalsysteem.yml`; `scripts/signaalsysteem/controle/netwerk.sh`.
- *ADR:* 0022 C, F; 0002 F; 0006; 0007.
- *Stopconditie:* `services/cloud-gateway/tests/test_gateway.py`: een aanroep met één intern of onbekend fragment wordt geweigerd; een verzoek met vrije tekst wordt geweigerd; een sjabloon waarvan de hash niet in Git staat, wordt geweigerd; het startscript weigert met een sleutel die niet `EXTERN` is; het eigen logboek en de regels in `model_call` komen overeen (vergelijkingstest). `netwerk.sh` exitcode 0: de gateway krijgt geen verbinding met een host buiten het domein van Anthropic; `claude-cli` bereikt internet alleen via de gateway; `langflow` heeft nog steeds geen internet. `docker inspect` van `cloud-gateway` toont de Claude-sleutel niet als omgevingsvariabele, en `docker inspect` van `langflow` toont geen Claude-sleutel.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F5-02 Wiki-pagina als kennisproduct**
- *Wat:* tabel voor wiki-pagina's met `is_derived = true`, classificatie (intern zodra één fragment intern is of de herkomst ontbreekt), beweringen met fragment-id's, status (`concept`, `review`, `gepubliceerd`, `verouderd`), run en model; leesroutes met classificatiebereik.
- *Hangt af van:* F2-05.
- *Bestanden:* `migrations/<n>.surrealql` en `_down`; nieuw `packages/shared/src/shared/models/wiki_page.py`; nieuw repository, dienst en router.
- *ADR:* 0004, 0001.
- *Stopconditie:* `test_migration_<n>_wiki_page.py`; `apps/app-main/tests/test_wiki_page_classificatie.py`; `test_zoeken_geen_afgeleide_teksten.py` uitgebreid: een pagina met de zoektekst komt niet terug.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F5-03 Wiki-generatie met controle**
- *Wat:* lus volgens FO 10 en 11: onderwerp kiezen, fragmenten ophalen (alleen fragmenten), pagina schrijven met een verwijzing per bewering, een tweede, onafhankelijke controle per bewering, stopconditie en maximum aantal rondes; afgekeurd naar wachtrij 5. Intern alleen met een lokaal model; de pagina krijgt dan `intern`. Een cloudaanroep loopt alleen via `cloud-gateway`, met sjabloon en fragment-id's.
- *Hangt af van:* F5-02, F5-07, F3-09.
- *Bestanden:* volgens het besluit in §8, B12; bij het voorstel nieuw `apps/app-main/src/app_main/graphs/wiki_generation.py` en de Langflow-component `cloud_model.py` voor de cloudaanroep.
- *ADR:* 0004, 0006, 0022 C; FO 10, 11; nieuwe ADR volgens B12.
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
- *Wat:* wiki-pagina's in de export, programma's gekoppeld aan het codeboek, wachtrij 6 voor nieuwe personen en nieuwe programma's in de werkbak. Daarna, als laatste stap en pas na een besluit van Geert, het aansluiten op de echte vault: `langflow` koppelt alleen `D:\# Vault\20 Atlas\28 Entities`, en de export schrijft alleen in het gegenereerde blok van bestaande notities daaronder (ADR 0018). Vóór de eerste schrijvende run akkoord van Geert op de paragraaf voor `D:\# Vault\CLAUDE.md` (ADR 0018, "Voorstel voor de vaultregels"; §8 B17). Nieuwe notities na review via de Obsidian-CLI op de host, met de OSK-skills en -sjablonen (ADR 0018, optie c; §8 B16); eerst toetsen of de keuzevragen van Templater via de CLI te vullen zijn. De vault-beheer-agent toetst de export buiten de fork.
- *Hangt af van:* F2-21, F5-02, F3-03; voor het aansluiten op de echte vault een besluit van Geert in het gesprek.
- *Bestanden:* `services/langflow/components/signaalsysteem/obsidian_export.py`, `services/langflow/templates/`; `apps/app-main/src/app_main/api/routers/export_obsidian.py`; `frontend/src/` (wachtrij 6); `docker-compose.signaalsysteem.yml` (koppeling van de vault, pas na het besluit).
- *ADR:* 0018, 0016.
- *Stopconditie:* de tests van F2-21 uitgebreid met wiki-pagina's en programma's op de tijdelijke testvault (byte voor byte gelijk bij herhaling, label intern in het blok, frontmatter van bestaande notities ongewijzigd); frontendtest voor wachtrij 6. Voor het aansluiten: een proefdraai op de echte vault die alleen een lijst van voorgenomen wijzigingen geeft en niets schrijft; Geert keurt die lijst goed vóór de eerste schrijvende run.
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

**F5-06 Toets fase 5**
- *Wat:* twee delen. Deterministisch: elke bewering van elke pagina heeft ten minste één fragment-id dat bestaat, binnen de classificatie van de pagina valt en bij de huidige of een uitdrukkelijk genoemde bronversie hoort. Door een mens: Geert beoordeelt een aselecte steekproef van beweringen op "het fragment draagt de bewering".
- *Hangt af van:* F5-03, F5-04.
- *Bestanden:* nieuw `scripts/signaalsysteem/toetsen/fase5_wiki.py`; `docs/signaalsysteem/meting-fase5.md` (uitkomst van de steekproef).
- *ADR:* FO 10, 15.
- *Stopconditie:* `fase5_wiki.py` exitcode 0 op het deterministische deel (100 %); steekproef van 50 beweringen met 0 niet-dragende fragmenten. De omvang van de steekproef is een voorstel (§8, B13).
- *Agent:* `signaal-bouwer`, daarna `signaal-reviewer`.

### Aantallen en kritiek pad

| Fase | Taken | Waarvan voorwaardelijk | Waarvan verwijdertaken | Werkplek (GPU) |
| --- | --- | --- | --- | --- |
| 2 | 32 | 2 (F2-25, F2-32) | 4 (F2-02, F2-09, F2-22, F2-23) | F2-11, F2-17 (deels), F2-24, F2-25, F2-31, F2-32 |
| 3 | 15 | 0 | 1 (F3-08, met bevestiging) | F3-01, F3-02 (deels), F3-13 |
| 4 | 11 (F4-10 en F4-11 vervallen, B3) | 0 | 0 | F4-03 |
| 5 | 7 | 0 | 0 | — |

De afhankelijkheden zijn met een script gecontroleerd (30-10): elke genoemde taak bestaat en er zijn geen kringen. Voorwaardelijke afhankelijkheden (F2-32 vóór F2-19) zijn als afhankelijkheid meegeteld.

Kritiek pad, met de taken die het meeste werk achter zich laten wachten:

- **Fase 2:** F2-26 → F2-03 → F2-04 → F2-06 → F2-13 → F2-14 → F2-16 → F2-17 → F2-18 → F2-19 → F2-22 (en parallel F2-31) → F2-24 → F2-23. F2-13 hangt af van F2-06 (vergrendeling); F2-22 hangt af van F2-29 (schrijfroutes), dat na F2-14 komt. P10 is niet geslaagd (30-10): F2-27 valt van dit pad en komt na F2-24; het pad begint bij F2-26 → F2-03. Een tweede keten loopt parallel en sluit aan bij F2-16 en F2-24: F2-01 → F2-10 → F2-11, F2-12 en F2-28. Even lang als het stuk F2-04 → F2-06 → F2-13 → F2-14 → F2-16 is de keten F2-04 → F2-07 → F2-08 → F2-15 → F2-16; F2-07 moet ook klaar zijn vóór F2-11 en F2-12. Is F2-32 nodig, dan komt die tussen F2-17 en F2-19.
- **Fase 3:** F3-03 → F3-02 → F3-13 → F3-04 → F3-12, met de referentieset F3-02 als lange pool: die hangt af van de steekproeven door Geert. F3-01 loopt parallel (F3-03 → F3-01 → F3-13). F3-02 kan starten zodra F2-17 en F3-03 klaar zijn.
- **Fase 4:** F4-01 en F4-13 → F4-02 → F4-12. F4-07 wacht op een ADR, F4-06 op het besluit B24.
- **Fase 5:** de besluiten B12 en B20 → F5-01 → F5-07 → F5-03 → F5-06.

## 3. Harde eisen en FO-dekking naar taken

### 3.1 Harde eisen

Elke eis uit `opdracht.md` §1 met de taken waarvan de stopconditie haar toetst.

| Eis | Onderdeel | Taken |
| --- | --- | --- |
| 1. Intern blijft lokaal; geen intern fragment naar een cloudmodel of een externe verwerkingsdienst, of via MCP | Docker-netwerk | F2-10 (`opslag`, een netwerk per proxy, masquerade-proef of geharde proxy), F2-11, F2-12 (`langflow` zonder internet), F3-10 (`mcp-intern`), F4-13 (`ophalen`), F5-01, F5-07 (gateway, alleen Anthropic), F2-24 (`netwerk.sh`) |
| | modelkeuze in de Langflow-componenten | F2-07, F2-16, F2-17 (adres per classificatie), F2-18 (lokale tekst vóór externe verwerking), F3-04, F4-06, F5-01, F5-03, F5-07 (controle per fragment, geen vrije tekst) |
| | filter in de API van de fork | F2-03, F2-05 (lektest inclusief foutpaden, geen `str(e)`), F2-13 (bronkaart niet leesbaar voor `lezen_*`), F2-30, F3-09, F3-10 (startcontrole `EXTERN`), F3-11 (P4), F2-24 (contracttest eis 1) |
| | omzeilroutes dicht | F2-01 (rechten, `pijplijn` 403), F2-02, F2-04 (flow buiten Git), F2-12 (componentenlijst, geen eigen code, geen tracing), F2-14, F2-22, F2-24 |
| 2. Herkomst: elk fragment herleidbaar tot bronversie, run en model | | F2-04, F2-06, F2-14, F2-15, F2-16 (`models` uit `model_call`), F2-17 en F4-03 (auditregel docling en WhisperX), F2-29, F3-05, F3-06, F3-09, F2-24 |
| 3. Afgeleide teksten tellen nooit als bron | | F2-05, F2-13 (geen locatie in vault of exportmap), F2-14 (markering van het blok), F2-22 (INSIGHTS weg), F3-02 (kopie buiten de vault), F5-02, F5-03 |
| 4. Intern altijd lokaal in Docker; GPU via llama-swap; extern mag naar een externe dienst | | F2-08, F2-09 (geen Ollama), F2-11, F2-17 (adres per classificatie), F2-18 (pas na `extern` naar buiten), F2-25, F3-01, F4-03, F2-24 |
| Geen interne documenten in repository of tests | | F2-05, F2-21 (tijdelijke testvault), F3-02 (alleen openbare bronnen uit NOVEX en NPVR, set buiten Git), F3-15, F4-09 |
| Geen geheimen in code, Compose of commits | | F2-01, F2-10 (Compose-test op letterlijke geheimen), F2-12 (geen Claude-sleutel in `langflow`), F3-10, F5-01, F5-07 |

### 3.2 Dekking van het FO (reviewbevinding 21)

| FO-onderdeel | Taak | Toelichting |
| --- | --- | --- |
| FO 7, kwaliteitsborging: NPVR-regio naar gemeenten; canonieke naam bij fusie of opvolging | F3-14 | nieuw |
| FO 7, labels: brede-welvaartthema's en foundational thema's | F3-15 | nieuw; Iv3 in F3-03 |
| FO 2, principe 6: herhaalbaar | F2-31 | nieuw; labels herhaald in F3-12 niet apart, omdat Kev dezelfde keuzelijst en hetzelfde model gebruikt en de fragmenten al gelijk moeten zijn |
| FO 9, audit: aanroepen van docling en WhisperX | F2-17, F4-03 | geen aparte taak: de auditregel hoort bij de component die de aanroep doet; beide stopcondities toetsen één regel per aanroep |
| FO 14, herclassificatie: notebooks en notities | F2-30 | nieuw; bronnen in F2-03 |

## 4. Compose-opzet (schets)

Dit is een schets voor `docker-compose.signaalsysteem.yml`, nog geen echt bestand. Poorten tussen `<…>` en paden zijn voorstellen voor de bouwtaak. Diensten met een fase erachter komen in die fase.

### 4.1 Diensten en netwerken

| Dienst | Image of build | Netwerken | Poort op de host | GPU | Fase |
| --- | --- | --- | --- | --- | --- |
| `surrealdb` | `surrealdb/surrealdb:v2.7.0@sha256:<digest>` (B3, ADR 0021) | opslag | — | nee | 2 |
| `open-notebook` | build `Dockerfile` (basis-tags vast) | opslag, verwerking, werkbak-intern; ophalen-intern (4), gateway-intern (5) | — | nee | 2 |
| `werkbak-toegang` | omgekeerde proxy, vaste tag (keuze bouwtaak) | werkbak-intern, toegang | `127.0.0.1:<werkbak>` | nee | 2 |
| `langflow` | build `services/langflow` op `langflowai/langflow:1.13.0@sha256:<volledige digest>` | verwerking, editor-intern; ophalen-intern (4), gateway-intern (5) | — | nee | 2 |
| `editor-toegang` | omgekeerde proxy, vaste tag | editor-intern, toegang | `127.0.0.1:<editor>` | nee | 2 |
| `socket-proxy` | `tecnativa/docker-socket-proxy:v0.5.0` | beheer | — | nee | 2 |
| `llama-swap` | build `services/llama-swap` op `ghcr.io/mostlygeek/llama-swap:v268-cuda13-b11977` plus client uit `docker:29.9.2-cli` | verwerking, beheer | — | ja | 2 |
| `docling-serve` | `quay.io/docling-project/docling-serve-cu128:v1.38.2` | verwerking | — | ja | 2 |
| `kev` | build `services/kev` (`kev-1.0` op `python:3.13.18-slim-bookworm`) | verwerking | — | ja | 3 |
| `open-notebook-mcp` | build `services/open-notebook-mcp` (`python:3.12.16-slim-bookworm`, `open-notebook-mcp==0.3.0`, `mcp==1.30.0`) | verwerking, mcp-intern | — | nee | 3 |
| `mcp-toegang` | omgekeerde proxy, vaste tag | mcp-intern, toegang | `127.0.0.1:<mcp>`, plus Tailscale op de host | nee | 3 |
| `reranker` | build `services/reranker` (CPU) | verwerking | — | nee | 3 |
| `ophalen` | build `services/ophalen` (vaste basis-tag) | ophalen-intern, uitgaand | — | nee | 4 |
| `whisperx` | `public.ecr.aws/deep-learning-containers/whisperx:3.8.6-cu128-amzn2023` | verwerking | — | ja | 4 |
| `grobid` | `grobid/grobid:0.8.2-crf` | verwerking | — | nee | 4 |
| `cloud-gateway` | build `services/cloud-gateway` (vaste basis-tag) | gateway-intern, cli-intern, uitgaand | — | nee | 5 |
| `claude-cli` | build `services/claude-cli` (vaste CLI-versie) | cli-intern | — | nee | 5 |

`proeven.md` 1.7 noemt de Langflow-digest ingekort (`sha256:ff2716e0…e303f`); de bouwtaak zet de volledige digest vast. De drie proxy's volgen de uitkomst van de masquerade-proef in F2-10: de optie op `toegang`, of een geharde proxy.

### 4.2 Schets

```yaml
# docker-compose.signaalsysteem.yml (schets)
name: signaalsysteem

networks:
  opslag:         { internal: true }  # alleen surrealdb en open-notebook (ADR 0002 F)
  verwerking:     { internal: true }  # de keten; geen internet, geen bereikbare poort (P8 K1, K2)
  beheer:         { internal: true }  # alleen llama-swap en socket-proxy (P1)
  werkbak-intern: { internal: true }  # werkbak-toegang en open-notebook
  editor-intern:  { internal: true }  # editor-toegang en langflow
  mcp-intern:     { internal: true }  # mcp-toegang en open-notebook-mcp (fase 3)
  ophalen-intern: { internal: true }  # langflow, ophalen, open-notebook (fase 4, ADR 0022 B)
  gateway-intern: { internal: true }  # langflow, cloud-gateway, open-notebook (fase 5, ADR 0022 C)
  cli-intern:     { internal: true }  # cloud-gateway en claude-cli (fase 5)
  toegang:                            # alleen de drie proxy's
    driver_opts:
      com.docker.network.bridge.enable_ip_masquerade: "false"   # alleen als de proef in F2-10 slaagt; anders geharde proxy's
  uitgaand: {}                        # alleen ophalen (fase 4) en cloud-gateway (fase 5)

secrets:                              # bestanden in secrets/ (in .gitignore); bron file, niet environment (P1 §8.24)
  surreal_gebruiker:        { file: ./secrets/surreal_gebruiker }
  surreal_wachtwoord:       { file: ./secrets/surreal_wachtwoord }
  sleutel_pijplijn:         { file: ./secrets/sleutel_pijplijn }        # langflow naar de fork
  sleutel_lezen_extern:     { file: ./secrets/sleutel_lezen_extern }    # open-notebook-mcp (fase 3)
  sleutel_ophalen:          { file: ./secrets/sleutel_ophalen }         # ophalen (fase 4)
  sleutel_gateway:          { file: ./secrets/sleutel_gateway }         # cloud-gateway, recht lezen_extern (fase 5)
  langflow_superuser:       { file: ./secrets/langflow_superuser }
  langflow_secret_key:      { file: ./secrets/langflow_secret_key }     # Fernet-sleutel
  docling_api_key:          { file: ./secrets/docling_api_key }
  kev_api_key:              { file: ./secrets/kev_api_key }             # fase 3
  claude_api_key:           { file: ./secrets/claude_api_key }          # alleen cloud-gateway (fase 5)
  claude_cli_login:         { file: ./secrets/claude_cli_login }        # alleen claude-cli (fase 5)

volumes:
  signaal_surreal_data: {}  # B3: nieuw, leeg; ./surreal_data van docker-compose.yml blijft archief
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
    image: surrealdb/surrealdb:v2.7.0@sha256:<digest>
    networks: [opslag]
    secrets: [surreal_gebruiker, surreal_wachtwoord]
    volumes: [signaal_surreal_data:/data]

  open-notebook:
    build: { context: ., dockerfile: Dockerfile }
    networks: [opslag, verwerking, werkbak-intern]   # fase 4: + ophalen-intern; fase 5: + gateway-intern
    environment:
      SURREAL_URL: ws://surrealdb:8000
      LLAMA_SWAP_URL: http://llama-swap:8080/v1
    secrets: [surreal_gebruiker, surreal_wachtwoord]
    volumes:
      - ./config:/app/config:ro
      - ./services/langflow/flows:/app/flows:ro      # controle van de flow-hash (F2-04, ADR 0022 E)
      - checkpoints:/data/checkpoints

  werkbak-toegang:                   # alleen localhost, niet via Tailscale (ADR 0016)
    image: <proxy>:<vaste tag>
    networks: [werkbak-intern, toegang]
    ports: ["127.0.0.1:<werkbak>:80"]    # naar open-notebook:8502
    # bij een geharde proxy: read_only, cap_drop [ALL], no-new-privileges, user zonder root (F2-10)

  langflow:
    build: services/langflow
    networks: [verwerking, editor-intern]          # nooit uitgaand of toegang (ADR 0022 A); fase 4: + ophalen-intern; fase 5: + gateway-intern
    environment:
      DO_NOT_TRACK: "true"
      LANGFLOW_AUTO_LOGIN: "false"
      LANGFLOW_MCP_SERVER_ENABLED: "false"
      LANGFLOW_ADD_PROJECTS_TO_MCP_SERVERS: "false"
      LANGFLOW_COMPONENTS_PATH: /app/componenten
      LANGFLOW_ALLOW_CUSTOM_COMPONENTS: "false"
      LANGFLOW_SUBSTITUTE_OUTDATED_COMPONENT_CODE: "false"
      LANGFLOW_BLOCK_CODE_INTERPRETER_COMPONENTS: "true"
      LANGFLOW_AGENTIC_EXPERIENCE: "false"
      LANGFLOW_ALLOW_COMPONENTS_PATHS_OVERRIDE: "true"
      LANGFLOW_DEACTIVATE_TRACING: "true"
      LITELLM_LOCAL_MODEL_COST_MAP: "True"
      # het startscript weigert LANGCHAIN_*, LANGSMITH_*, LANGFUSE_*, LANGWATCH_*, ARIZE_*, OPIK_*,
      # PHOENIX_*, TRACELOOP_*, OPENLAYER_* en OTEL_EXPORTER_OTLP_* (ADR 0022 F)
    secrets: [sleutel_pijplijn, langflow_superuser, langflow_secret_key]   # geen Claude-sleutel, in geen enkele fase
    volumes:
      - langflow_data:/app/data
      - ./services/langflow/components:/app/componenten:ro
      - bronarchief:/bronarchief
      - <inbox>:/inbox:ro
      - <watchfolder>:/watchfolder:ro
      # geen vault-koppeling tot F5-05 (B8); daarna alleen "D:\# Vault\20 Atlas\28 Entities":/vault/entities, alleen voor langflow (ADR 0018)
      - ./config:/config:ro             # modelregister, poortwachterregels, componentenlijst

  editor-toegang:
    image: <proxy>:<vaste tag>
    networks: [editor-intern, toegang]
    ports: ["127.0.0.1:<editor>:80"]     # naar langflow:7860

  socket-proxy:
    image: tecnativa/docker-socket-proxy:v0.5.0
    networks: [beheer]
    environment: { POST: 1, ALLOW_START: 1, ALLOW_STOP: 1, CONTAINERS: 0 }   # configuratie B (P7)
    volumes: ["/var/run/docker.sock:/var/run/docker.sock:ro"]   # :ro beperkt de API niet (ADR 0009)

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
      DOCLING_SERVE_MAX_SYNC_WAIT: "3600"   # synchroon eindpunt (ADR 0009, 0014); waarde is een voorstel (B21)
      HF_HUB_OFFLINE: "1"
    secrets: [docling_api_key]      # startscript leest /run/secrets/docling_api_key
    volumes: [docling_artifacts:/artifacts:ro]
    deploy: { resources: { reservations: { devices: [{ capabilities: [gpu] }] } } }
    # geen restart

  # kev (fase 3), whisperx (fase 4): zelfde patroon als docling-serve
  # whisperx: OPT_OUT_TRACKING=true, HF_HUB_OFFLINE=1, volumes op /opt/models/hf, /opt/models/torch, /root/nltk_data
  # open-notebook-mcp [verwerking, mcp-intern] met sleutel_lezen_extern; mcp-toegang [mcp-intern, toegang]; reranker [verwerking] (fase 3)
  # ophalen [ophalen-intern, uitgaand] met sleutel_ophalen; grobid [verwerking] (fase 4)
  # cloud-gateway [gateway-intern, cli-intern, uitgaand] met sleutel_gateway en claude_api_key (fase 5)
  # claude-cli [cli-intern] met claude_cli_login, alleen tmpfs (fase 5)
```

### 4.3 llama-swap: groepen en geheugen

```yaml
# services/llama-swap/config.yaml (schets)
# timeouts.responseHeader: standaard 0 (geen limiet), of boven DOCLING_SERVE_MAX_SYNC_WAIT (ADR 0009)
models:
  embed:      # mxbai-embed-large-v1 Q8_0, ChristianAzinn@3ec6d46a…, 1 024 dimensies
    cmd: llama-server --embeddings -c 512 -m /modellen/mxbai-embed-large-v1.Q8_0.gguf --port ${PORT}
  werkmodel:  # Qwen3.5-9B Q4_K_M (B4)
    cmd: llama-server -c 8192 -m /modellen/Qwen3.5-9B-Q4_K_M.gguf --port ${PORT}
    ttl: 60
  docling:    # synchroon eindpunt /v1/convert/file; de aanvraag staat open tijdens de conversie
    cmd: /app/wacht_op_dienst.sh signaal-docling http://signaal-docling:5001/health
    cmdStop: docker stop signaal-docling
    proxy: http://signaal-docling:5001
    checkEndpoint: /health
    ttl: 300            # F2-17 toetst of llama-swap een dienst met een openstaande aanvraag laat staan
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
| Kev-4B bf16 | 14 574 | 14 574 | 15 157, gemeten met korte vragen; opnieuw te meten in F3-01 | ja, marge 121 (ondergrens die niet vaststaat) |
| docling-serve | 2 438–2 502 | 2 438–2 502 | 3 021, gemeten (1 herhaling) | ja |
| WhisperX variant A (`diarize=true`) | 10 200–10 328 | 8 874 | 10 912, berekend | ja |

Niet gemeten: mxbai + WhisperX (F4-03) en docling + WhisperX (komt in `zwaar` niet voor).

## 5. Interfaces

Alle eindpunten vragen `Authorization: Bearer <sleutel>`. Recht tussen haakjes. Fouten: 401 zonder geldige sleutel, 403 bij te weinig recht, 404 bij een onbekend of gefilterd record, 409 bij een conflict (classificatie, `rev`, afgesloten run, flow buiten Git), 422 bij een ongeldig verzoek (ook een locatie in de vault of de markering van het blok). Een foutmelding bevat nooit uitzonderingstekst, alleen een generieke melding met een fout-id (ADR 0002 C).

### 5.1 Importeindpunt: `POST /api/v1/import` (pijplijn)

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
      "text_source": "pymupdf | docling-lokaal",
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
- Classificatie: `extern` alleen als de regel of batch `extern` zegt, die keuze door een `beheerder` is bevestigd, en de poortwachter geen interne kenmerken vond; anders `intern`. Een import met `extern` voor een bestaande interne bron geeft 409. Ontbreekt de classificatie, dan geldt `intern`.
- Elk fragment krijgt de classificatie, bronversie, run en extractor van het verzoek. `structure` wordt opgeslagen als `doc_node`-records met dezelfde bronversie, run en classificatie. Embeddings zitten niet in het verzoek: die berekent de stap `embed` (ADR 0011).
- Bevat de ruwe inhoud of een fragment `<!-- signaalsysteem:begin -->` of `<!-- signaalsysteem:end -->`, dan 422 en een regel in het runlogboek (ADR 0004).
- Hoort de run bij een flow die niet in Git staat en is de invoer intern of onbekend, dan 409 (ADR 0022 E).
- `labels` mag leeg zijn (fase 2).

### 5.2 Bronkaartregel

Velden volgens FO 5, met de standaarden uit ADR 0019. Opslag als tabel `source_map_rule` in de fork, alleen via de API (ADR 0020, B2), met daarnaast `rev`, `modified_by` en `modified_at`. Lezen met `pijplijn` of `beheerder`; `lezen_extern` en `lezen_intern` krijgen 403. Aanmaken, wijzigen, op `extern` zetten en bevestigen alleen met `beheerder`.

| Veld | Type | Verplicht | Standaard en regels |
| --- | --- | --- | --- |
| `id` | record-id | ja | |
| `name` | tekst | ja | |
| `route_type` | `api`, `mcp`, `feed`, `website`, `bulkupload`, `watchfolder`, `upload` | ja | bepaalt de standaard hieronder |
| `source_type` | `pdf`, `map`, `website`, `api`, `video`, `podcast`, `mcp` | ja | |
| `location` | tekst | ja | pad, URL of API-eindpunt; niet binnen de vault of de exportmap (422, ADR 0004, 0020) |
| `schedule` | `{ cron, timezone, catch_up }` of leeg | nee | `timezone` `Europe/Amsterdam`; `catch_up` `coalesce` of `skip` |
| `classification` | `intern`, `extern` | ja | `api`, `mcp`, `feed`, `website`: voorstel `extern`, de beheerder bevestigt; `bulkupload`: keuze per batch, zonder keuze `intern`; `watchfolder`: geen standaard, tot besluit `intern`; `upload`: voorstel van de poortwachter, zonder keuze `intern` |
| `classification_confirmed_by`, `_at` | persoon van een sleutel `beheerder`, tijd | bij `extern` | `extern` zonder bevestiging geldt als `intern` |
| `pipeline_start` | `automatisch`, `na_bevestiging`, `wachtrij` | afgeleid | `api`/`mcp`/`feed`/`website` automatisch; `bulkupload` volgens de batchroute; `upload` na bevestiging; `watchfolder` wachtrij |
| `document_type` | lijst | nee | bijvoorbeeld jaarrekening, begroting |
| `route` | flow-id in Git | ja | |
| `selection` | lijst | nee | leeg is alles |
| `label_schema` | `{ schema, version }` | nee | verwijst naar een codeboekversie (ADR 0013) |
| `owner` | tekst | ja | standaard beheerder |
| `fetch_content` | `metadata`, `transcript`, `volledige_media`, `volledig_document` | ja | YouTube standaard `metadata` (FO 5) |
| `active` | boolean | ja | `true` |

Een batch (`intake_batch`) heeft `classification` (standaard `intern`), `route`, `created_by` en de lijst met bestanden. Een wachtrij-item (`intake_item`) heeft pad, hash, voorstel van de poortwachter, status (`wacht`, `besloten`, `verwerkt`) en het besluit van de beheerder (route en classificatie).

### 5.3 Classificatiefilter en rechten in de API

| Recht | `ClassificationScope` | Ziet | Routes (kern) |
| --- | --- | --- | --- |
| `lezen_extern` | `EXTERN` | alleen `extern` | `GET /api/notebooks[/{id}]`, `GET /api/sources[/{id}]`, `GET /api/notes[/{id}]`, `POST /api/search`, en vanaf F3-09 de herkomst-, versie- en fragmenteindpunten; geen bronkaart, batches of wachtrij (403) |
| `lezen_intern` | `ALLES` | alles | dezelfde leesroutes zonder filter, plus `GET /api/v1/export/obsidian/entities`; geen bronkaart, batches of wachtrij (403) |
| `pijplijn` | `ALLES` | alles | de leesroutes van `lezen_intern`; import, runs, stappen, `model-calls`; bronkaartregels en batches lezen; wachtrij-items aanmelden, besloten items ophalen, status `verwerkt`; exportgegevens en codeboek. **403** op herclassificatie (bron, notebook, notitie), aanmaken, wijzigen en bevestigen van regels en batches (extern bevestigen), het besluit over een wachtrij-item, het triagebesluit, sleutels, modellen en instellingen |
| `ophalen` | leeg | niets | alleen `GET /api/v1/source-map-rules/{id}/ophaalcontrole` (fase 4), antwoord ja of nee |
| `beheerder` | `ALLES` | alles | alles, waaronder herclassificatie, bronkaartregels en batches met hun bevestiging, besluiten in de wachtrijen en de triage, `POST /api/v1/sources/{id}/retire`, sleutels, modellen, instellingen, chat, ask, podcasts; verplicht veld `person` |

Elke geldige sleutel behalve `ophalen` (ADR 0002 B) mag `GET /api/v1/auth/scope` aanroepen; dat geeft alleen het recht en het bereik van die sleutel (startcontrole van open-notebook-mcp en de gateway).

- De middleware leidt het bereik af uit de sleutel; de router geeft het door. In de repositorylaag is het een verplicht argument zonder standaard, bijvoorbeeld `async def get(self, source_id: str, *, scope: ClassificationScope) -> Source`. Een aanroep zonder bereik faalt.
- `fn::text_search($query, $limit, $classifications, $history)` en `fn::vector_search($vector, $limit, $min_score, $classifications, $history)` lezen alleen `chunk` en `source_embedding` en filteren in de `WHERE`-clausule op `classification IN $classifications` en, zonder `$history`, op de huidige bronversie en run en op niet-vervallen bronnen.
- Een gefilterd record geeft 404, niet 403. Schrijftools van open-notebook-mcp krijgen 403.
- Herclassificatie: `POST /api/v1/sources/{id}/classification`, `POST /api/v1/notebooks/{id}/classification` en `POST /api/v1/notes/{id}/classification` (beheerder), `{ "to": "extern", "reason": "…" }`, één regel in `classification_log` met de persoon van de sleutel.

### 5.4 Contract tussen Langflow en de fork

Langflow gebruikt één sleutel met `pijplijn` en praat alleen met de API van de fork (`http://open-notebook:5055`), met llama-swap (`http://llama-swap:8080`), vanaf fase 4 met `ophalen` en vanaf fase 5 met `cloud-gateway`.

| Eindpunt | Verzoek | Antwoord |
| --- | --- | --- |
| `POST /api/v1/runs` | `trigger` (`schema`, `mapcontrole`, `feed`, `handmatig`, `herverwerking`), `pipeline_id`, `pipeline_version` (`{ git_commit, flow_hash }`), `source_map_rule_id`, `label_schema` | `{ run_id, status: "bezig" }`; de fork legt de `rev` van de regel vast (ADR 0020); 409 bij een flow buiten Git met interne of onbekende invoer (ADR 0022 E) |
| `PATCH /api/v1/runs/{id}` | `status`, `ended_at`, `log[]` (tijd, stap, status, melding), `models[]` (naam, versie, aanroepwijze; bij `cli` ook versie en exitcode), `error` | run; 409 als de run al is afgesloten. `models[]` wordt niet overgenomen: de fork leidt `models` af uit `model_call` en logt een verschil als waarschuwing (ADR 0003) |
| `POST /api/v1/import` | §5.1 | §5.1 |
| `POST /api/v1/runs/{run_id}/steps/{stap}` | `stap` ∈ `embed`, `entity_extraction`, `entity_dedup`, `graph` (fase 4 ook `cites`); `{ source_version_id }` | `{ step_id, job_id, status: "wachtrij" }`; dezelfde stap op dezelfde versie in dezelfde run geeft dezelfde `step_id` |
| `GET /api/v1/runs/{run_id}/steps/{stap}` | — | `{ status: wachtrij | bezig | geslaagd | mislukt | wacht_op_review, attempts, error, review_item }` |
| `POST /api/v1/model-calls` | tijd, `run_id`, model, aanroepwijze, taak, classificatie, tokens, duur, status; ook voor docling-serve en WhisperX | 201; nooit prompt- of antwoordtekst |
| `POST /api/v1/intake/items`, `GET /api/v1/intake/items?status=besloten`, `PATCH /api/v1/intake/items/{id}` | wachtrij-item uit de mapcontrole of een losse upload; ophalen van besloten items; status `verwerkt` | item |
| `GET /api/v1/source-map-rules` | — | bronkaartregels voor planning en route |
| `GET /api/v1/codeboek/{schema}/{versie}` | — | keuzelijst voor Kev (fase 3) |
| `GET /api/v1/export/obsidian/entities?since=` | — | exportgegevens (ADR 0018) |

Gedrag:

- **Volgorde per bron:** run aanmaken → registratie (hash) → lokale tekst (PyMuPDF of lokale docling-serve) → poortwachter → extractie → fragmentatie → (fase 3: labels) → import → stappen → run afsluiten. Ook bij `ongewijzigd` wordt de run afgesloten, als `niets_nieuws`. Een externe verwerkingsdienst komt pas na een bevestiging `extern` door de poortwachter (ADR 0019).
- **docling-serve:** synchroon `POST /upstream/docling/v1/convert/file`; een verloren of afgebroken aanvraag hooguit drie keer opnieuw, daarna `mislukt` met melding (ADR 0009). Terugval asynchroon met vergrendeling alleen na F2-32.
- **Volgen:** de component vraagt de stapstatus op met oplopende pauze (beginnend bij 5 s). Herhaalpogingen van de job doet de worker van de fork; de component start een mislukte stap niet zelf opnieuw.
- **Herhaalbeleid van de component:** bij 502, 429, 503 en verbindingsfouten opnieuw met pauze, tot een maximum; daarna `mislukt` met een logregel. Bij een geheugenfout van llama-swap zet de component de taak opnieuw in de rij en logt de fout (FO 12.2).
- **`wacht_op_review` (besloten, B5):** de component stopt met volgen en sluit de flow af met een logregel; de run blijft `bezig`. Na het triagebesluit hervat de fork de graaf (ADR 0012). Een geplande flow `hervatten` haalt runs op waarvan een stap van `wacht_op_review` naar `geslaagd` ging, en start de resterende stappen. Zo blijft Langflow elke stap starten (ADR 0011).
- **Innamewachtrij (besloten, B5):** Langflow meldt nieuwe watchfolder-documenten en losse uploads aan als wachtrij-item, en haalt besloten items op. De fork roept Langflow niet aan.
- **Modelkeuze:** de component leest `config/modelregister.yaml` (alleen-lezen gekoppeld) en weigert een model met `toegestaan voor intern = nee` bij een interne of onbekende bron, zowel bij verbinden als bij uitvoeren. Een cloudmodel gaat alleen via `cloud-gateway`, met run, fragment-id's, sjabloon-id en model, zonder vrije tekst (ADR 0022 C).
- **Ophalen (fase 4):** een externe bron haalt Langflow via `ophalen` met het id van de bronkaartregel en het adres; `ophalen` vraagt de fork per aanvraag de ophaalcontrole (ADR 0022 B).

## 6. Risico's

| Nr | Risico | Gevolg | Maatregel | Taak |
| --- | --- | --- | --- | --- |
| R1 | **Werkgeheugen van de host.** P3 stopte na 28 van 36 metingen bij 1,2 GB vrij (WSL-VM 13,2 GB); per dienst niet gemeten. | Containers of de VM vallen weg bij zware combinaties; metingen onvolledig. | Geheugengrens van de WSL-VM vastleggen; vrij werkgeheugen meten in elke GPU-controle met een ondergrens (voorstel 2 GB); `mem_limit` per dienst; de ontbrekende combinaties alsnog meten. | F2-11, F2-24, F4-03 |
| R2 | **Marge van 121 MiB naast Kev-4B.** Kev-4B met mxbai 15 157 MiB, grens 15 279 MiB, bij een nulmeting van 0 en met korte vragen. | Langere fragmenten, de volledige keuzelijst of het bureaublad van Windows vreten de marge op; geheugenfout. | F3-01 meet opnieuw met fragmenten van ≥ 512 tokens, de volledige Iv3-keuzelijst en een actief bureaublad; reranker niet op de GPU (ADR 0015); terugval Kev-0,8B of een gekwantiseerde variant (F3-13) na besluit Geert. | F3-01, F3-13 |
| R3 | **CUDA-driver 13.3 tegenover 13.4.** Het llama-swap-image eist `cuda>=13.4`; driver 610.88 levert 13.3. | Werkt alleen met `NVIDIA_DISABLE_REQUIRE=1` en `LD_LIBRARY_PATH`; een driver- of image-update kan dat breken. | Geert werkt de driver voorlopig niet bij (30-10); de omweg blijft staan, met beide variabelen in Compose met commentaar. F2-24 toetst dat llama-swap met de omweg start; een GPU-controle volgt na elke driver- of image-update. F2-25 haalt de omweg pas weg als NVIDIA een driver met CUDA ≥ 13.4 uitbrengt en Geert die heeft geïnstalleerd. Een nieuwere llama-swap-tag die nog hogere CUDA eist, wordt niet overgenomen zonder die controle. | F2-11, F2-24, F2-25 |
| R4 | **docling-serve en llama-swap.** llama-swap telt `ttl` vanaf de laatste aanvraag; een gestopte docling-serve verliest zijn taken (P1). Niet gedocumenteerd is of llama-swap een dienst met een openstaande synchrone aanvraag laat staan bij `ttl` of bij een wissel in `zwaar`. | Conversies gaan verloren, of docling stopt bij een wissel. | Synchroon eindpunt met `DOCLING_SERVE_MAX_SYNC_WAIT`; hooguit drie keer opnieuw, daarna `mislukt`; toets in F2-17; bij falen de terugval F2-32 (asynchroon met bundeling en vergrendeling); de toets van fase 2 doet precies deze combinatie. | F2-17, F2-32, F2-24 |
| R5 | **Langflow belt naar buiten.** Grotendeels vervallen door ADR 0022: `langflow` heeft in geen enkele fase uitgaand netwerk of een Claude-sleutel; ophalen en cloudaanroepen lopen via `ophalen` en `cloud-gateway` met een eigen controle bij de fork. Restrisico: de blokkade van eigen code noemt Langflow een bètafunctie; een fout in `ophalen` of de gateway. | Bij een fout in de eigen containers kan iets naar buiten. | Netwerkcontrole in elke fase; componentenlijst en verboden variabelen; flow buiten Git met interne invoer geeft 409; tests op `ophalen` en de gateway. | F2-12, F2-04, F4-13, F5-07 |
| R6 | **MCP-versiebinding.** open-notebook-mcp 0.3.0 vraagt `mcp>=1.23.2` zonder bovengrens; het officiële startpunt faalt met `mcp==1.30.0`; het pakket is van een derde partij. | Een update breekt de server of verandert tools. | Beide pakketten met `==`; eigen startscript met controle van het bereik; elke update eerst P4. | F3-10, F3-11 |
| R7 | **Referentieset van 200–300 fragmenten.** Bestaat nog niet; alleen bronnen uit de mappen NOVEX en NPVR, en alleen openbare; Iv3-labels vragen een handmatige steekproef; de poortwachter moet interne kenmerken herkennen terwijl de set alleen openbaar is; twee mappen met openbare bronnen kunnen te weinig fragmenten of een eenzijdige set geven (vooral regio- en programmadocumenten). | De toets van fase 3, de keuze van de Kev-variant (F3-13) en de drempel van de poortwachter schuiven op, of meten op een eenzijdige set. | F3-02 vroeg starten; telling van overgeslagen bronnen; kopie buiten de vault; synthetische interne kenmerken op openbare tekst; schemacontrole als stopconditie; is de set kleiner dan 200, dan legt de bouwer dat voor aan Geert. | F3-02, F3-13, F3-04, F3-12 |
| R8 | **Socket-proxy valt later uit.** ADR 0009 optie 2 is besloten (Geert, 30-10); optie 3 (geen Docker-toegang) blijft alleen de terugval. | Bij terugval: docling-serve en WhisperX altijd aan, 2,4 + 8,9 GB in rust; Kev-4B (14,6 GB) past dan niet; gedeelde vergrendeling in de Langflow-componenten nodig. | Alleen bij uitval: F2-11 zonder socket-proxy, de vergrendeling uit F2-32, Kev-0,8B in F3-01, WhisperX in F4-03 buiten llama-swap; FO 12.2 en het plan aanpassen. | F2-11, F3-01, F4-03 |
| R9 | **Restrisico socket-proxy.** Elke container is op naam te starten en te stoppen, ook `surrealdb` (P7 K2); 502 na een externe stop. | Verstoring, geen toegang tot gegevens. | Socket-proxy alleen op `beheer`; herhaalpoging bij 502 in `ModelGate` en componenten. | F2-07, F2-11, F2-16 |
| R10 | **Lege database en databaseversie (B3, ADR 0021).** Gegevens die alleen in de oude database staan en niet uit een migratie of laadscript komen, ontbreken in de nieuwe omgeving; wijzigingen aan gedeelde ontologieën die in de oude database via de editor zijn gemaakt, staan mogelijk niet in `packages/ontology-manager/ontologies/*.yaml`. Op v2.7.0 blijven advisories open die in v3.3.0 zijn opgelost (ADR 0021); de overstap naar v3 kan meer werk zijn dan mechanisch (strengere schemacontrole, onbekende SDK-ondersteuning). | Een functie werkt niet op de lege database; een bekend lek in de databaseversie; vertraging van fase 2. | F2-26 (P10) vóór F2-10 en F2-03; bij v3 F2-27, anders een aparte overstaptaak na fase 2 en een controle per kwartaal op een 2.x-uitgave met oplossingen; F2-28 inventariseert en laadt de referentiegegevens; F2-10 start op een leeg volume en toetst `/health`; SurrealDB alleen op `opslag`, zonder poort, één namespace en database. Het archief `./surreal_data` blijft staan. | F2-26, F2-27, F2-28, F2-10, F3-03 |
| R11 | **Verborgen afhankelijkheden bij verwijderen.** De interface en tests gebruiken routes die vervallen (bijvoorbeeld `frontend/src/lib/api/services.ts`, `DELETE /api/sources/{id}`), en lezers van `source.full_text`. | Breuk buiten de taak. | Verwijdercontrole V, inclusief frontendbuild; inventaris van de lezers van `full_text` in F2-05. | F2-02, F2-05, F2-09, F2-22, F2-23, F2-29 |
| R12 | **Volledigheid van vector-zoeken met filter.** Filter en rangschikking in één `WHERE`; met een index op dichtstbijzijnde buren kan het filter na de index komen. | Minder treffers; raakt eis 1 niet. | Meting in F2-05. | F2-05 |
| R13 | **Fragmenten boven 512 tokens** bij mxbai. | Afgekapte embeddings. | Keuze opdelen of afkappen, met test. | F2-08 |
| R14 | **OCR op de CPU** in docling-serve (RapidOCR zonder CUDA). | Grote gescande documenten zijn traag; een synchrone aanvraag kan `DOCLING_SERVE_MAX_SYNC_WAIT` overschrijden. | Wachttijd volgens B21 en R4; externe docling alleen voor extern en pas met een eigen ADR (ADR 0019). | F2-17 |
| R15 | **Ophalen van images:** `public.ecr.aws` gaf anoniem `toomanyrequests`. | Vertraging bij installatie. | Herhaalpogingen in het downloadscript; images lokaal bewaren. | F2-11, F4-03 |

## 7. Aannames en conflicten

### 7.1 Aannames

Feiten in dit plan komen uit de code van de fork (gecontroleerd op 30-10: bestandspaden, `migrations/87.surrealql` als hoogste, `config/` met alleen `triage_config.json`, geen `secrets/` in `.gitignore`, `Dockerfile` met `EXPOSE 8502 5055` en zwevende basis-tags, de routes in `sources_crud.py` op `:475`, `:964`, `:998` en `:1027`, `recanonicalization_service.py` onder `apps/app-main/src/app_main/services/entity_resolution/`, de gedeelde ontologieën als YAML in `packages/ontology-manager/ontologies/`, TOOI in `packages/shared/src/shared/vocabulary/`), uit de ADR's of uit `proeven.md`. Niet geverifieerd:

- **A1.** Compose-geheimen met bron `file` blijven aanwezig als llama-swap een container met `docker start` start. P1 gebruikte bron `environment`, die dat niet doet. Te toetsen in F2-11.
- **A2.** De interface draait in de container `open-notebook` op poort 8502 en is via een omgekeerde proxy bereikbaar.
- **A3.** Besloten (B7, 30-10): Langflow-componenten, flows en scripts staan in `services/langflow/`, en het modelregister en de poortwachterregels in `config/`.
- **A4.** Besloten (B2, 30-10, ADR 0020): de bronkaart, batches en de innamewachtrij staan in de fork (SurrealDB), alleen via de API.
- **A5.** De triggers-API van Langflow 1.13.0 kan per bronkaartregel een trigger aanmaken (P2 toonde aanzetten via de API, niet aanmaken via de API).
- **A6.** Geen aanname meer: het effect van `enable_ip_masquerade: "false"` onder Docker Desktop is een verplichte proef in F2-10, met de geharde proxy als uitkomst bij falen (ADR 0002 F).
- **A7.** De vindplaats en vorm van de openbare Iv3-lijst 2025 en van de BWB-open data zijn niet nagegaan (ADR 0013).
- **A8.** Een kleine aanroepdienst in de container `claude-cli` is nodig, omdat de gateway de CLI over `cli-intern` aanroept en geen Docker-toegang heeft om de CLI zelf te starten.
- **A9.** Voor de herclassificatie van notebooks en notities (F2-30) geldt dezelfde regel als voor bronnen (beheerder, reden, logregel met persoon); een notitie die naar een intern fragment verwijst, kan niet naar `extern`. ADR 0001 regelt dit alleen voor bronnen.
- **A10.** De TOOI-gegevens bevatten CBS-gemeentecodes en opvolgingsrelaties bij gemeentelijke herindeling (F2-28, F3-14). Niet nagegaan in `tooi_provider.py` of `tooi_bulk.py`.
- **A11.** De gedeelde ontologieën in de oude database zijn gelijk aan de YAML-bestanden in `packages/ontology-manager/ontologies/`. Wijzigingen via de editor (OE.5) staan mogelijk alleen in de oude database; die terughalen vraagt een besluit van Geert, omdat het archief onaangeroerd blijft.
- **A12.** De brede-welvaartthema's en de foundational thema's bestaan als lijst met een vaste vindplaats; FO 10 noemt `28.04 Thema's` in de vault als plek van het codeboek.
- **A13.** De fork kan de Git-commit van een flow vergelijken zonder Git in de container, met de flow-JSON uit de alleen-lezen gekoppelde map en een commit-label op het image (F2-04). De bouwer legt de vorm vast.

### 7.2 Conflicten en inconsistenties

Vastgelegd, niet opgelost.

- **C1.** FO 11 noemde nog `LANGFLOW_LOAD_FLOWS_PATH` als laadmechanisme; ADR 0017 A2 (besloten) laat dat vervallen. Opgelost: FO 11 is bijgewerkt (30-10).
- **C2.** ADR 0001 verwijdert `private`, `privacy_mode` en `default_privacy_mode` in dezelfde migratie die `classification` toevoegt. Het plan verdeelt dat over F2-03 en F2-09, zodat de code tussen de taken blijft werken. De eindstand is gelijk.
- **C3.** FO 13.2 punt 7 laat chat en podcasts ongewijzigd; ADR 0006 laat chat via `ModelGate` naar een lokaal model gaan. Chat verandert dus wel.
- **C4.** Opgelost (30-10): ADR 0002 D claimt geen hergebruik meer; de herkomsteindpunten worden nieuw gebouwd in F3-09.
- **C5.** FO 11 plaatst de wiki-generatie als LangGraph-lus in de fork, en FO 9 noemt de Claude-CLI geschikt voor wiki-generatie. Volgens ADR 0006 en FO 12.1 heeft de fork geen internet en roept hij alleen modellen via llama-swap aan; cloudaanroepen lopen alleen via `cloud-gateway` (ADR 0022). Een wiki-lus in de fork kan dus geen cloudmodel gebruiken. Dit vraagt een ADR vóór F5-03 (B12).
- **C6.** FO 15 noemt voor fase 2 de lokale map; volgens ADR 0019 wacht een watchfolder op de beheerder. Fase 2 heeft daardoor de werkbak met de innamewachtrij nodig (F2-20). Dat is een gevolg, geen tegenspraak.
- **C7.** ADR 0022 B laat `ophalen` alleen `GET` en `HEAD` doen. Een MCP-bron (F4-06) gebruikt over HTTP `POST` (streamable HTTP), en sommige API-bronnen (F4-05) vragen `POST`. FO 5 noemt beide als brontype. Zonder aanvulling op ADR 0022 is F4-06 niet te bouwen (B24).
- **C8.** ADR 0022 C laat de gateway alleen een prompt uit een sjabloon en opgehaalde externe fragmenten maken, zonder vrije tekst van Langflow. FO 10 en 11 vragen bij de wiki-generatie een tweede, onafhankelijke controle per bewering; als die controle een cloudmodel gebruikt, moet de tekst van de bewering mee, en die is gegenereerde tekst, geen fragment. Dat is strenger dan het FO en raakt B12 (B20).
- **C9.** ADR 0021 vraagt bij een overstap naar v3 "een eigen migratie per vervanging (idempotent, met `_down`)". Op een lege v3-database faalt `migrations/1.surrealql` (`SEARCH ANALYZER`) al vóór een latere migratie kan draaien; een nieuwe migratie kan een oude die faalt niet herstellen. F2-27 moet dus bestaande migraties aanpassen, wat afwijkt van de werkwijze dat migraties onveranderlijk zijn. De ontwerper legt de vorm vast na P10.

## 8. Open vragen en besluiten

### 8.1 Vóór fase 2 te beslissen door Geert

| Nr | Vraag | Voorstel | Blokkeert |
| --- | --- | --- | --- |
| B1 | ADR 0009, Docker-toegang optie 2 met configuratie B: **besloten** (Geert, 30-10), met de restrisico's uit R9. Geert werkt de driver voorlopig niet bij (30-10); de CUDA-omweg blijft staan en F2-25 is voorwaardelijk (R3). | — | — |
| B2 | Plaats van de bronkaart: **besloten** (Geert, 30-10). Tabel in SurrealDB, beheerd met de migraties van de fork, alleen via de API van de fork (sleutel en classificatiefilter); niet als YAML in Git, omdat de bronkaart vaak verandert door watchfolder-besluiten en bulkbatches. ADR 0020. | — | — (F2-13, F2-14, F4-01 aangepast) |
| B3 | Database: **besloten** (Geert, 30-10). Start met een lege database, zonder migratie van bestaande gegevens; het bestaande volume `./surreal_data` blijft onaangeroerd als archief; de nieuwe omgeving krijgt een eigen volume `signaal_surreal_data`. Versie `v2.7.0` (ADR 0021, besloten na P10, 30-10); de overstap naar v3 is F2-27, na F2-24. Referentiegegevens komen uit een laadscript (F2-28). Gevolgen in ADR 0001 en 0003; F2-01, F2-03, F2-04, F2-09 en F2-10 vereenvoudigd; F4-10 en F4-11 vervallen. | — | F2-10 (versie, via F2-26) |
| B4 | Werkmodel: **besloten** (Geert, 30-10) volgens voorstel: Qwen3.5-9B Q4_K_M, context 8 192 (P1, P3). | — | — (F2-07, F2-11) |
| B5 | Contract voor `wacht_op_review` en de richting van de innamewachtrij (§5.4): **besloten** (Geert, 30-10) volgens voorstel. | — | — (F2-15, F2-16, F3-07) |
| B6 | Vervangen van `docker-compose.yml` aan het eind van fase 2: **besloten** (Geert, 30-10) volgens voorstel, na een geslaagde F2-24; volumes blijven. | — | — (F2-23) |
| B7 | Plaats van de Langflow-code en de configuratie: **besloten** (Geert, 30-10) volgens voorstel: `services/langflow/` en `config/`. | — | — (F2-12) |
| B8 | Obsidian: **besloten** (Geert, 30-10), gecorrigeerd op 30-10: de vault is `D:\# Vault`, niet `D:\# Obsidian`. De export volgt de vaultregels van `D:\# Vault`; alleen in bestaande notities onder `20 Atlas/28 Entities/`, programma's in `28.06 Programma's`; schrijven alleen in het gegenereerde blok, met classificatie, id en herkomst in het blok. Nieuwe notities via de Obsidian-CLI (B16). Het schrijven naar de echte vault is een latere fase (F5-05); tot dan schrijft niets in `D:\# Vault` en gebruiken tests een tijdelijke testvault. FO 10 en ADR 0018 bijgewerkt. | — | — (F2-21, F5-05) |
| B16 | ADR 0018, nieuwe vaultnotities: **besloten** (Geert, 30-10), optie c: na review via de Obsidian-CLI op de host, met de OSK-skills en -sjablonen. | — | — (F2-21, F5-05) |
| B18 | ADR 0021, grens K5 van P10: **besloten** (Geert, 30-10) volgens voorstel; na P10 het besluit v3.3.5 of v2.7.0 volgens de beslisregel. | — | — (P10 niet geslaagd; `v2.7.0`, ADR 0021) |
| B19 | Het extra recht `ophalen` (ADR 0002 B, 0022 B): **besloten** (Geert, 30-10) volgens voorstel; het recht komt in de migratie van F2-01. | — | — (F2-01) |
| B20 | Gateway strenger dan gevraagd (ADR 0022 C, F): **besloten** (Geert, 30-10) voor fase 2 tot en met 4; voor fase 5 besluiten samen met B12 (C8). | — | — (F2-12; F5-03, F5-07) |
| B21 | `DOCLING_SERVE_MAX_SYNC_WAIT`: **besloten** (Geert, 30-10), 3 600 s. | — | — (F2-11, F2-17) |
| B25 | ADR 0001 "Uitwerking in F2-03": **besloten** (Geert, 31-10) volgens voorstel: (a) een relatie is intern zodra één van haar twee entiteiten intern is; (b) herclassificatie van notebooks en notities in F2-30. | — | — (F2-30) |
| B26 | ADR 0003 "Uitwerking in F2-04": een bekende inhoudshash die niet de huidige versie is (inhoud A → B → A). Nu vindt `register` v1, blijft `current_version` op v2 en toont zoeken B als actueel. Alternatief: een nieuwe versie met dezelfde hash; dan vervalt de unieke index op bron en hash. | `current_version` gaat terug naar de bestaande versie v1, met een logregel in de run; geen nieuwe versie, de unieke index blijft. | F2-14, F2-05 (zoeken op de huidige versie) |

### 8.2 Later te beslissen

| Nr | Vraag | Vóór |
| --- | --- | --- |
| B9 | Referentieset: **besloten** (Geert, 30-10). Bronnen alleen uit `D:\# Vault\50 Resources\51 Raw\51.01 Sources\NOVEX` en `\NPVR`, alleen openbare; labels uit de metadata (alleen lezen); het script kopieert de originelen naar een map buiten de vault en de pipeline verwerkt de kopieën; Iv3 via een handmatige steekproef; controle door Geert op ongeveer 50 fragmenten; set buiten Git (F3-02). Open: hoe interne kenmerken worden nagebootst (voorstel: synthetisch op openbare tekst). | F3-02 |
| B10 | Wanneer is een entiteit zichtbaar voor `lezen_extern` (open punt ADR 0002)? | F3-09 |
| B11 | Tool en plek voor de scènedetectie bij video (nieuwe ADR). | F4-07 |
| B12 | Waar draait de wiki-generatie, gegeven C5 en C8? Voorstel: de lus en de controle als LangGraph in de fork met lokale modellen; een generatie met de Claude-API of -CLI voor externe onderwerpen als Langflow-component die via `cloud-gateway` loopt en de lus van buiten aanroept (nieuwe ADR). | F5-01, F5-03 |
| B13 | Omvang van de steekproef in de toets van fase 5. Voorstel: 50 beweringen. | F5-06 |
| B14 | Plek van de reranker onder llama-swap (ADR 0015); tot een meting blijft hij op de processor (CPU). De uitkomst van F3-13 kan ruimte op de GPU geven. | na F3-10 en F3-13 |
| B15 | Kev-4B gekwantiseerd: **besloten** (Geert, 30-10) als proef F3-13, met de grens **besloten** (Geert, 30-10; ADR 0009): ECE ten hoogste 0,02 hoger dan bij bf16, precisie en recall per label elk ten hoogste 0,02 lager. | — (F3-13) |
| B17 | ADR 0018: de tekst van de paragraaf "Signaalsysteem block" voor `D:\# Vault\CLAUDE.md` (voorstel in ADR 0018). Het signaalsysteem wijzigt de vault niet zelf; Geert voegt de tekst toe. | F5-05 (vóór de eerste schrijvende run) |
| B22 | Herclassificatie van notebooks en notities: regel volgens aanname A9, vast te leggen in ADR 0001. | F2-30 |
| B23 | Bron van de codeboeken brede welvaart en foundational thema's (A12): een openbare lijst, of `28.04 Thema's` in de vault (alleen lezen, met dezelfde voorwaarden als F3-02). | F3-15 |
| B24 | Hoe een MCP-bron of een API-bron met `POST` de externe server bereikt, gegeven C7: `ophalen` uitbreiden met `POST` voor een bevestigd externe regel, of een eigen kleine container per bron op het patroon van ADR 0022. | F4-05 (alleen voor `POST`), F4-06 |
