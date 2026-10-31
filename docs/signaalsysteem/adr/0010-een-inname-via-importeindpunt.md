# ADR 0010: Eén inname via het importeindpunt

- **Status:** besloten
- **Datum:** 2026-10-29; 2026-10-31 uitwerking in F2-14
- **Raakt FO-paragraaf:** 4, 5, 11 (bouwstenen), 13.2 punt 1, 14 (inname en Open Notebook)
- **Lost op:** verschilanalyse conflict 9, open vraag 3
- **Gewijzigd door:** ADR 0019 (classificatie en pipelinestart per innameroute)

## Context

Het functioneel ontwerp (FO) zegt:

- de eigen pipeline is de enige inname; Open Notebook is niet in productie, dus een overgang is niet nodig (FO 14);
- de ingebouwde bronverwerking vervalt;
- een importeindpunt neemt verwerkte fragmenten aan, met Docling-structuur, sectiepad, labels en herkomst (FO 13.2 punt 1);
- uploadvenster, inboxmap, mapcontrole en feedcontrole zijn bouwstenen in Langflow (FO 5, FO 11);
- een bron zonder bronkaartregel gaat naar de wachtrij van de beheerder (FO 5).

De fork heeft meerdere actieve innameroutes en een automatische keten (verschilanalyse 3.13):

- `POST /api/sources`, `/api/sources/json` en `/api/sources/{id}/retry` (`apps/app-main/src/app_main/api/routers/sources_upload.py:261`, `:609`, `:627`);
- `process-url`, `process-document` en `process-audio` in `apps/app-main/src/app_main/api/routers/agents.py:220-386`;
- de mapbewaker (`apps/app-main/src/app_main/services/agents/file_watcher.py`, router `watcher`, schakelaar in `apps/app-main/src/app_main/config.py:201`);
- Zotero (router `zotero`, `packages/zotero-integration`) en de OKF-import (Open Knowledge Format; `apps/app-main/src/app_main/services/okf_import_service.py`, via `api/routers/exports.py`);
- de keten `SOURCE_PIPELINE` (`apps/app-main/src/app_main/services/source_pipeline.py:109-170`) met INGEST, EMBED, EXTRACT, GRAPH en INSIGHTS;
- de extractielaag `SourceExtractor` (`services/source_extractor.py`), `SourceProcessor` (`services/source_processor.py`) en de parserroutering (`services/parsing/engine_dispatcher.py`, `auto_fallback.py`).

open-notebook-mcp kent bovendien een tool `create_source` die `POST /api/sources` aanroept (ADR 0002).

## Opties

Het FO maakt de keuze. Een route laten staan "voor de compatibiliteit" valt af: dat is een tweede inname zonder bronkaart en poortwachter, en dus zonder classificatie. Over de uitvoering:

1. **Het importeindpunt onder de bestaande agentrouter** (`/api/v1/agents`).
   - Voor: de sleutelauthenticatie bestaat al.
   - Tegen: die router hoort bij de ingebouwde inname, en de authenticatie wordt toch algemeen (ADR 0002).
2. **Een eigen router `imports`** met een `ImportService`, op `/api/v1/import`. Gekozen.

## Voorstel

Optie 2.

**Het importeindpunt**

- Pad: `POST /api/v1/import` in de API (Application Programming Interface) van de fork, met recht `pijplijn` (ADR 0002). Langflow gebruikt een eigen sleutel met dat recht.
- Het eindpunt weigert een verzoek met de markering van het gegenereerde blok van de export naar Obsidian in de inhoud (HTTP 422; ADR 0004).
- Inhoud van het verzoek:
  - id van de bronkaartregel;
  - bron: extern id, type, locatie en titel;
  - bronversie: `content_hash`, `fetched_at` en de verwijzing naar het origineel in het bronvolume;
  - classificatie met de beslissing van de poortwachter (ADR 0001);
  - `run_id` (ADR 0003);
  - de Docling-structuur als JSON (JavaScript Object Notation) van het DoclingDocument of als knopenboom voor `doc_node` (`migrations/49.surrealql`);
  - fragmenten: tekst, sectiepad, pagina, elementtype, posities en labels met methode en kans;
  - het extractiemodel met versie.
  
  Het precieze schema legt het plan van aanpak vast (stap 4).
- Gedrag:
  - is de combinatie (`source`, `content_hash`) bekend, dan maakt het eindpunt geen nieuwe versie en meldt het `ongewijzigd`; de run krijgt dan `niets_nieuws`;
  - hetzelfde verzoek nogmaals met dezelfde `run_id` geeft dezelfde id's terug;
  - classificatieregels volgens ADR 0001, met HTTP 409 bij een poging een bron lager in te delen.
- Hergebruik:
  - het aanmaken van een bronrecord uit `_create_source_impl` (`sources_upload.py`);
  - het schrijven van fragmenten in `packages/surrealdb-service/src/surrealdb_service/repositories/source.py`;
  - het opslaan van de documentstructuur als `doc_node` (`migrations/49.surrealql`).
- Na de import start de verdere verwerking zoals ADR 0011 beschrijft: embeddings, extractie, deduplicatie en graaf.

**Wat naar Langflow verhuist, als code die een component importeert**

- De fragmentatie: `pipelines/ingestion/src/ingestion/chunking/extractor.py` (`ChunkExtractor`, `extract_chunks`) en `regrouper.py`. Die werken op een `DoclingDocument` en passen dus op de JSON-uitvoer van docling-serve (ADR 0014).
- De watchdog-logica uit `file_watcher.py`, als mapcontrole.
- De Zotero-client uit `packages/zotero-integration`, als connector voor een API-bron.

**Wat vervalt in de fork**

- De routes in `sources_upload.py` die bronnen maken of opnieuw verwerken.
- `process-url`, `process-document` en `process-audio` in `agents.py`. De overige agentroutes vallen onder het algemene rechtenmodel (ADR 0002); routes die een model starten, krijgen recht `beheerder`.
- `file_watcher.py`, de router `watcher` en de schakelaar in `config.py`.
- De router `zotero`, en de OKF-import in `okf_import_service.py` met zijn route. De OKF-export blijft, met recht `beheerder`.
- In `SOURCE_PIPELINE`:
  - de stap INGEST en het automatisch starten bij upload;
  - de stap INSIGHTS, die afgeleide tekst maakt (ADR 0004).
  
  EMBED, EXTRACT en GRAPH blijven als stappen na de import.
- `SourceExtractor`, `SourceProcessor`, `services/parsing/`, de jobtypen `DOCUMENT_PARSE` en `AUDIO_TRANSCRIBE` (`packages/shared/src/shared/types/enums.py:144-178`) en de handlers ervoor in `apps/app-main/src/app_main/handlers.py`.
- `LibrarianService.enqueue_due` (`apps/app-main/src/app_main/services/audit/librarian_service.py`), dat geen aanroeper heeft. De planning ligt bij Langflow of APScheduler (FO 5).

**Bestaande gegevens:** zie ADR 0003. De nieuwe omgeving start met een lege database (B3, 30-10); bronnen uit de oude omgeving worden zo nodig opnieuw ingelezen via de nieuwe pipeline.

## Gevolgen

- **Code:** nieuwe `ImportService` en router `imports`. Veel code wordt verwijderd in `apps/app-main`. `pipelines/ingestion` houdt alleen de fragmentatie en de modellen over (zie ADR 0014 voor de parser).
- **Tests (stopcondities):**
  - een test dat een import twee keer hetzelfde resultaat geeft;
  - een test dat een bekende hash geen nieuwe versie maakt;
  - een routetest dat geen route behalve `/api/v1/import` een `source` of `chunk` kan aanmaken. De test zoekt in de router- en servicecode naar aanroepen van de schrijfmethoden.
- **Interface:** de upload, de Zotero-koppeling en de mapbewaker in de interface werken niet meer (ADR 0016).
- **ARCHITECTURE.md:** paragraaf 14 (Track PL) wordt herschreven.

## Uitwerking in F2-14 (31-10)

De bouw (`migrations/94.surrealql`, `ImportService`, `InnameBinding`) werkt het voorstel hierboven als volgt uit, of wijkt ervan af. De classificatie van regel, batch en item volgt ADR 0019 en 0020, "Uitwerking in F2-13".

**Besloten** (review F2-14, 31-10; in uitvoering bij de bouwer):

- **Recht.** `POST /api/v1/import` is alleen voor `pijplijn`. Een beheerder corrigeert via de handmatige run (ADR 0003, "Uitwerking in F2-06").
- **Batch en item.** Het verzoek heeft `intake_batch_id` en `intake_item_id`. De run legt `intake_batch` en `intake_item` vast.
- **Binding.**
  - De locatie van de bron moet onder `regel.location` vallen: bij een URL hetzelfde schema, dezelfde host en een padprefix; bij een pad `ligt_onder`.
  - Een bestand van een batch en een item worden gebonden aan een hash die de fork zelf berekent.
  - Regel, batch en item van het verzoek moeten die van de run zijn (409 `niet_gebonden`).
- **Extern.** Een import is alleen `extern` als alle vier gelden:
  - de regel, de batch of het item is extern;
  - de aanroeper geeft `extern` op;
  - de poortwachter zegt `extern`, zonder interne kenmerken;
  - dezelfde inhoud (hash) is in de fork niet al bekend als intern of onbeslist.

  In alle andere gevallen geldt `intern`.
- **Markering van het blok** (ADR 0004). Na normalisatie (NFKC, HTML-entiteiten) herkent een reguliere expressie het gegenereerde blok in alle tekstvelden van het verzoek en in de aaneengeschakelde tekst van de fragmenten.
  - Restrisico: in een gecomprimeerde PDF-stroom is de markering niet te zien.
- **Structuur.**
  - `doc_node` wordt in dezelfde transactie gebouwd uit de fragmenten met hun sectiepad.
  - `structure.content` van Docling wordt alleen op de markering gecontroleerd en niet ingelezen. Dat wijkt af van plan van aanpak §5.1 ("wordt opgeslagen als `doc_node`").
- **`feed_item_id`** loopt via `source.external_id`. `external_id` is uniek over alle regels, zodat twee feeds met dezelfde GUID elkaar met 409 blokkeren. Dat is veilig, maar beperkt.
- **Origineel.**
  - De fork hasht het origineel zelf uit het bronarchief (`SIGNAAL_BRONARCHIEF`, alleen-lezen) en kopieert het naar een eigen archief, met de hash als naam (`SIGNAAL_FORK_ARCHIEF`).
  - `source_version.original` verwijst naar die kopie. De koppelingen staan in plan van aanpak §4 en F2-10.

**Open punt**

- **Herverwerking van een bekende versie.** Een nieuwe run op een bestaande bronversie (trigger `herverwerking`, ADR 0003) kan nog niet via de import: een bekende hash geeft `ongewijzigd` en schrijft geen fragmenten. Geen taak in het plan lost dit nu op; F2-31 (toets herhaalbaarheid) heeft het nodig. Te beslissen: in welke taak, vóór F2-31.
