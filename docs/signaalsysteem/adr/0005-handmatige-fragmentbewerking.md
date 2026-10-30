# ADR 0005: Handmatige fragmentbewerking

- **Status:** besloten (optie 2, Geert, 2026-10-29)
- **Datum:** 2026-10-29; bijgewerkt 2026-10-30 na de review (alle schrijfroutes, bron verwijderen)
- **Raakt FO-paragraaf:** 2 (principe 6), 8 (versiebeheer)
- **Lost op:** verschilanalyse open vraag 11

## Context

Het functioneel ontwerp (FO) zegt: fragmenten horen bij één bronversie, en dezelfde bronversie met dezelfde pipelineversie geeft hetzelfde resultaat (FO 2, principe 6; FO 8). Handmatige correcties worden vastgelegd met auteur en tijdstip. Het FO noemt als voorbeeld het samenvoegen van entiteiten; over het handmatig bewerken van fragmenten zegt het niets.

De fork kan fragmenten samenvoegen en splitsen: `apps/app-main/src/app_main/services/chunking/chunk_mutator.py` (`ChunkMutator`), aangeroepen vanuit `apps/app-main/src/app_main/api/routers/sources_crud.py:1077` en `:1110`, met verzoekschema's in `apps/app-main/src/app_main/api/schemas.py:23-35` en een audit via `ChunkAuditService` (`apps/app-main/src/app_main/dependencies.py:137-139`). Dat wijzigt fragmenten ter plekke. Dat botst met herhaalbaarheid en met "niets overschrijven" (ADR 0003).

De functie zo laten is geen optie: ze overschrijft fragmenten en breekt harde eis 2.

## Opties

1. **Verwijderen.** `ChunkMutator`, de twee routes, de schema's en de bijbehorende tests vervallen. Een slechte fragmentatie herstel je in de pipeline, met andere instellingen voor Docling of de fragmentatiebouwsteen. Dat levert een nieuwe run op (ADR 0003).
   - Voor: eenvoudig (FO 2, principe 8); herhaalbaar; geen tweede schrijfpad naast het importeindpunt.
   - Tegen: een eenmalige correctie aan één document kost een aanpassing in de pipeline.
2. **Ombouwen tot handmatige run.** Samenvoegen of splitsen maakt een nieuwe `processing_run` met `trigger = handmatig` en de beheerder als actor. Die run levert nieuwe fragmenten voor dezelfde bronversie; de oude blijven staan. `source_version.current_run` wijst naar de handmatige run.
   - Voor: correcties per document blijven mogelijk, met herkomst.
   - Tegen: een volgende automatische run op dezelfde bronversie vervangt de handmatige stand als "huidige". Dan is een regel nodig die handmatige runs beschermt, in de lijn van de optimistische vergrendeling uit ADR 0003. Het is ook een tweede schrijfpad voor fragmenten naast de import (ADR 0010).

## Voorstel

Optie 1. Het FO kiest voor een pipeline die herhaalbaar is en voor één inname. Correcties per document horen daarom in de pipelinedefinitie, bijvoorbeeld als selectie of als parameter in de bronkaartregel. Komt er later behoefte aan correcties per document, dan kan optie 2 alsnog, als eigen bouwsteen die via het importeindpunt schrijft.

Geert beslist.

## Besluit

Optie 2, besloten door Geert op 2026-10-29: handmatig samenvoegen en splitsen blijft mogelijk als handmatige run met nieuwe fragmenten. De regel die handmatige runs beschermt, staat in ADR 0003.

**Alle schrijfroutes voor fragmenten en bronnen** (aanvulling na de review, 30-10). Naast samenvoegen en splitsen wijzigen ook deze routes in `apps/app-main/src/app_main/api/routers/sources_crud.py` fragmenten of bronnen ter plekke:

| Route | Regel | Wordt |
| --- | --- | --- |
| `PATCH /api/sources/{source_id}/chunks/{chunk_id}` (wijzigen) | `:964` | handmatige run: een nieuw fragment met de gewijzigde tekst; het oude blijft staan |
| `DELETE /api/sources/{source_id}/chunks/{chunk_id}` (verwijderen) | `:998` | handmatige run: de nieuwe fragmentenset zonder dat fragment; het oude record blijft staan |
| `POST /api/sources/{source_id}/chunks` (handmatig toevoegen) | `:1027` | handmatige run: de nieuwe fragmentenset met het toegevoegde fragment |
| `DELETE /api/sources/{source_id}` (bron verwijderen) | `:475` | een vervalmarkering op de bron, geen wissen |

- Wijzigen, verwijderen en toevoegen volgen dezelfde regel als samenvoegen en splitsen: een handmatige run (`trigger = handmatig`, de persoon van de sleutel `beheerder` als actor) met een volledige nieuwe fragmentenset voor dezelfde bronversie; `source_version.current_run` wijst daarna naar die run, beschermd volgens ADR 0003. Elk nieuw fragment krijgt de classificatie van de bronversie (ADR 0001) en `extractor = handmatig`. Er is geen route meer die een bestaand `chunk`-record wijzigt of wist.
- **Bron verwijderen** wordt `POST /api/v1/sources/{id}/retire` (recht `beheerder`, verplichte reden): de bron krijgt `retired_at`, `retired_by` en `retired_reason`. Bronversies, runs, fragmenten, embeddings en herkomst blijven staan. Zoeken en de leesroutes laten een vervallen bron standaard weg; met `history=true` blijft ze raadpleegbaar, met het classificatiefilter. De bronkaartregel van zo'n bron wordt niet vanzelf uitgezet; dat beslist de beheerder. `DELETE /api/sources/{source_id}` vervalt.
- Alle routes vragen het recht `beheerder` (ADR 0002).

## Gevolgen

- `ChunkMutator` wordt omgebouwd tot een dienst die een handmatige run en een nieuwe fragmentenset maakt, voor samenvoegen, splitsen, wijzigen, verwijderen en toevoegen. De regel die handmatige runs beschermt, staat in ADR 0003.
- **Tests (stopcondities):** na elke handmatige bewerking is het aantal fragmenten van de vorige run ongewijzigd en hun tekst gelijk; een test dat geen route een `chunk`-record wijzigt of verwijdert (uitbreiding van de routetest uit ADR 0010); een test dat een vervallen bron niet in een zoekresultaat staat, maar haar fragmenten wel in de database.
- **Interface:** de schermen voor fragmentbewerking en het verwijderen van een bron roepen de nieuwe routes aan (ADR 0016).
