# ADR 0004: Afgeleide teksten uit de zoekresultaten

- **Status:** besloten
- **Datum:** 2026-10-29
- **Raakt FO-paragraaf:** 2 (principe 5), 8 (kennisproduct), 10 (wiki-pagina is afgeleid), 13.2 punt 4
- **Lost op:** verschilanalyse conflict 8

## Context

Harde eis 3: afgeleide teksten, zoals notities en wiki-pagina's, tellen nooit als bron. Het functioneel ontwerp (FO) zegt dat door modellen gegenereerde tekst gemarkeerd is als afgeleid en niet terugkeert als bronmateriaal (FO 2, principe 5). Notities en wiki-pagina's tellen niet als bron in zoekresultaten (FO 13.2 punt 4).

In de fork:

- `fn::vector_search` zoekt naast `source_embedding` ook altijd in `source_insight`, zodra `$sources` waar is. Met `$show_notes` zoekt het ook in `note` (`migrations/9.surrealql:21-52`);
- `fn::text_search` (`migrations/4.surrealql`) en `SearchRepository.hybrid_search` met `include_notes=True` als standaard (`packages/surrealdb-service/src/surrealdb_service/repositories/search.py:320`) doen hetzelfde;
- `SearchRequest.search_notes` staat standaard op `True` (`apps/app-main/src/app_main/api/schemas.py:109`). De MCP-server (Model Context Protocol) open-notebook-mcp stuurt dat veld niet mee (zie ADR 0002), dus elke MCP-zoekvraag neemt nu notities mee;
- `ContextService` neemt notities mee in de context voor modellen (`apps/app-main/src/app_main/services/context_service.py:50`);
- de pipelinestap INSIGHTS maakt automatisch `source_insight` aan (`apps/app-main/src/app_main/services/source_pipeline.py`; `ARCHITECTURE.md` paragraaf 14);
- de MCP-tool `add_note` schrijft notities die via `search` terugkomen (`packages/surrealdb-service/src/surrealdb_service/mcp/server.py:337`).

## Opties

Het FO maakt de keuze. Over de uitvoering:

1. **Een vlag `is_derived` en filteren op die vlag.** Werkt alleen zolang elke zoekroute de vlag controleert.
2. **Zoeken leest structureel alleen fragmenten, en kennisproducten dragen daarnaast `is_derived`.** Zoeken raakt de tabellen met afgeleide tekst niet, dus een vergeten filter kan niets lekken. De vlag maakt het onderscheid zichtbaar in export en API (Application Programming Interface). Gekozen.

## Voorstel

Optie 2.

- Een nieuwe migratie definieert `fn::vector_search` en `fn::text_search` opnieuw, met `_down`. Ze lezen alleen `source_embedding` en `chunk`. De delen voor `source_insight` en `note` en de parameter `$show_notes` vervallen. Het classificatiefilter (ADR 0002) en het filter op de nieuwste versie (ADR 0003) komen in dezelfde migratie.
- In `SearchRepository`, `SearchService` en `api/routers/search.py` vervalt de parameter `include_notes`. `SearchRequest.search_notes` vervalt. Clients die het veld nog sturen, krijgen geen fout, want Pydantic negeert onbekende velden standaard. De parameter heeft dan geen effect meer.
- `note`, `source_insight` en later de wiki-pagina's krijgen `is_derived` met `DEFAULT true` en `ASSERT $value = true`. Ze krijgen ook een classificatie (ADR 0001) en verwijzingen naar de fragmenten waarop ze steunen. Ze blijven leesbaar via hun eigen routes, zoals `/api/notes`, maar komen nooit in een zoekresultaat.
- Kennisproducten kunnen modellen ook als context bereiken. Retrieval die context voor een model opbouwt, zoals de wiki-generatie of een LangGraph-module, gebruikt alleen fragmenten. In `ContextService` vervalt het meenemen van notities en inzichten.
- De pipelinestap INSIGHTS vervalt (ADR 0010). `insight_service.py` en de transformaties blijven staan (FO 13.2 punt 7), maar hun uitvoer is afgeleid en niet zoekbaar.
- De MCP-tool `add_note` vervalt met `surrealdb-mcp` (ADR 0002). `create_note` via open-notebook-mcp krijgt HTTP 403, omdat de sleutel alleen `lezen_extern` heeft.
- Koppelingen van notitie naar notitie en van notitie naar bron (Tracks Y en NS: `related_note`, `note_about`; `ARCHITECTURE.md` paragraaf 12 en 12a) blijven. Ze leggen een relatie vanuit een kennisproduct, maar voeren dat product niet terug als bron.
- Bestaande gegevens: de migratie zet `is_derived = true` op alle bestaande `note`- en `source_insight`-records.

## Gevolgen

- **Code:** de zoekfuncties, `SearchRepository.hybrid_search`, `SearchService`, de zoekrouter, `SearchRequest` en `ContextService` veranderen. De graaf `ask` (`apps/app-main/src/app_main/graphs/ask.py`) gebruikt dezelfde zoekfunctie en neemt daarna ook geen notities meer mee.
- **Tests (stopcondities):**
  - een test met een notitie en een inzicht waarvan de tekst gelijk is aan de zoekvraag: geen van beide komt in het resultaat, voor tekst, vector en hybride zoeken;
  - een test dat elk zoekresultaat naar een `chunk` of `source_embedding` verwijst;
  - een schemacontrole op `is_derived`.
- **Interface:** zoeken in de interface toont geen notities meer. Dat past bij FO 13.2.
