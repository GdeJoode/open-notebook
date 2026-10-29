# ADR 0005: Handmatige fragmentbewerking

- **Status:** voorgesteld
- **Datum:** 2026-10-29
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

## Gevolgen

- **Optie 1:** de genoemde code en tests worden verwijderd. Het scherm voor het samenvoegen en splitsen van fragmenten in de interface werkt niet meer (ADR 0016). Er is geen migratie nodig: al bewerkte fragmenten vallen onder de migratie `pre-signaalsysteem` van ADR 0003.
- **Optie 2:** `ChunkMutator` wordt omgebouwd tot een dienst die een handmatige run en nieuwe fragmenten maakt. De regel die handmatige runs beschermt, komt bij in ADR 0003.
