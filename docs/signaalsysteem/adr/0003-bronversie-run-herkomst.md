# ADR 0003: Bronversie, verwerkingsrun en herkomst; herverwerking zonder overschrijven

- **Status:** besloten
- **Datum:** 2026-10-29
- **Raakt FO-paragraaf:** 2 (principe 5 en 6), 4 (stap 2 en 9), 5 (planning), 8
- **Lost op:** verschilanalyse conflict 7, open vraag 5 (de vraag of de run een nieuwe tabel wordt) en open vraag 15

## Context

Harde eis 2 vraagt dat elk fragment herleidbaar is tot bronversie, verwerkingsrun en model. Het functioneel ontwerp (FO) legt vast:

- een gewijzigde bron levert een nieuwe bronversie op, en de vorige blijft bewaard;
- fragmenten horen bij één bronversie;
- zoeken toont standaard de nieuwste versie;
- herverwerking maakt een nieuwe run en nieuwe fragmenten, en de oude blijven traceerbaar;
- elke taak levert een run met status en logboek op, ook als er niets nieuws was;
- handmatige correcties krijgen auteur en tijdstip, met optimistische vergrendeling;
- de herkomst volgt de begrippen van PROV-O (PROV Ontology), zodat een export naar het Resource Description Framework (RDF) mogelijk blijft (FO 8).

In de fork:

- er is geen bronversie en geen inhoudshash (verschilanalyse 3.7);
- `SourceProcessor` verwijdert alle fragmenten en maakt ze opnieuw aan (`apps/app-main/src/app_main/services/source_processor.py:114-123`; `packages/surrealdb-service/src/surrealdb_service/repositories/source.py:1158-1176`);
- `job` legt status, fout en pogingen vast (`migrations/28.surrealql:7-17`), maar geen pipelineversie, modellen of logboek. Fragmenten verwijzen niet naar een job;
- vermeldingen van entiteiten hebben wel `chunk_id` en het extractiemodel (`apps/app-main/src/app_main/services/entity_persistence_service.py:640-675`);
- `provenance_chain` is een vrij array zonder schema (`migrations/39.surrealql:24`, `:79`);
- `resolution_log` heeft geen auteur (`migrations/36.surrealql`, `85.surrealql`);
- `entity.manual_override` bestaat (`migrations/59.surrealql`), maar optimistische vergrendeling ontbreekt (`packages/surrealdb-service/src/surrealdb_service/repositories/notebook_schema.py:295`).

## Opties

Het FO bepaalt wát er vastligt. De opties gaan over de vorm:

1. **De run als uitbreiding van `job`.** Hergebruikt de tabel, maar een job is één uitvoerstap in de wachtrij van de fork. Een run omvat ook stappen in Langflow en ook runs zonder werk. Dat vermengt twee begrippen.
2. **Een nieuwe tabel `processing_run`, waarnaar `job` verwijst.** De run is het PROV-O-begrip activiteit; de job blijft de uitvoereenheid. Gekozen.
3. **Herkomst als aparte relatietabellen per PROV-O-relatie.** Precies, maar elke leesroute heeft dan extra joins nodig. Velden met vaste namen die één op één op PROV-O passen, geven hetzelfde met minder werk.

## Voorstel

Optie 2, met velden die op PROV-O passen.

**Nieuwe tabellen** (idempotente migraties met `_down`, na die van ADR 0001 en 0002)

- `source_version`:
  - `source`, `version_no`, `content_hash` (SHA-256 van de ruwe inhoud), `fetched_at`;
  - `original` (verwijzing naar het bestand in het alleen-lezen bronvolume), `predecessor`;
  - `classification` (ADR 0001), `created_by_run`, `current_run`;
  - unieke index op (`source`, `content_hash`).

  Voor feeds is de sleutel het item-id (video-id of aflevering-GUID; FO 5). Dat wordt een eigen bron met eigen versies.
- `processing_run`:
  - `started_at`, `ended_at`;
  - `status`: `bezig`, `geslaagd`, `mislukt` of `niets_nieuws`;
  - `trigger`: schema, mapcontrole, feed, handmatig of herverwerking;
  - `pipeline_id` en `pipeline_version` (Git-commit en hash van de flow in JSON, JavaScript Object Notation);
  - `models`: per model naam, versie en aanroepwijze, en bij de Claude-CLI (Command Line Interface) ook CLI-versie en exitcode;
  - `log`: per stap tijdstip, status en melding;
  - `error`, `attempts`, `notified_at`.

  Langflow maakt de run aan met `POST /api/v1/runs` en sluit hem af met `PATCH`. Dat gebeurt ook als de hash al bekend was.
- `source.current_version`: verwijst naar de nieuwste bronversie.

**Bestaande tabellen uitbreiden**

- `chunk`: `source_version` en `run` (beide verplicht), `extractor` (bijvoorbeeld docling-serve met versie) en `classification`.
- `source_embedding`: `run` en `embedding_model` (naam en versie).
- vermeldingen van entiteiten (de PC.3-herkomst in `entity_persistence_service.py`): `run` erbij.
- `job`: `run` (optioneel) erbij. De wachtrij blijft de uitvoerder van stappen in de fork. Wie die stappen start, bepaalt ADR 0011.

**Herverwerking zonder overschrijven**

- Het verwijderpad voor fragmenten in `SourceProcessor` en `SourceRepository` vervalt. `SourceProcessor` zelf vervalt met ADR 0010.
- Een nieuwe pipelineversie op dezelfde bronversie geeft een nieuwe run met nieuwe fragmenten. `source_version.current_run` wijst daarna naar die run. De oude fragmenten blijven staan.
- Zoeken toont standaard alleen fragmenten waarvoor geldt: de `source_version` is `source.current_version`, en de `run` is `source_version.current_run`. Met de parameter `history=true` komen oudere versies erbij, nog steeds met het classificatiefilter (ADR 0002).

**PROV-O-afbeelding**

| PROV-O | Fork |
| --- | --- |
| entiteit | `source_version`, `chunk`, `source_embedding`, kennisproduct |
| activiteit | `processing_run` |
| actor | model (register, ADR 0007), beheerder (sleutelhouder, ADR 0002) |
| `wasDerivedFrom` | `chunk.source_version`, `source_version.predecessor` |
| `wasGeneratedBy` | `chunk.run`, `source_version.created_by_run` |
| `wasAssociatedWith` | `processing_run.models` |

`provenance_chain` op `entity` en `relation` vervalt zodra de lezers ervan naar de vermeldingen met run zijn omgezet. De bouwtaak inventariseert eerst wie het veld leest.

**Handmatige correcties en optimistische vergrendeling (open vraag 15)**

- Er komt één algemeen mechanisme, niet per tabel iets eigens. Tabellen die de beheerder handmatig wijzigt (`entity`, `entity_alias`, `relation` en de besluiten in `triage_queue`) krijgen `rev` (geheel getal), `modified_by` (sleutelhouder) en `modified_at`.
- Een wijziging via de API (Application Programming Interface) stuurt de verwachte `rev` mee. Bij een afwijking volgt HTTP 409.
- Een hulpfunctie in de basisrepository van `packages/surrealdb-service` doet dit voor alle tabellen.
- Runs schrijven via dezelfde functie. Een record met `manual_override = true` (hergebruik van `migrations/59.surrealql`) laat een run niet wijzigen; de run legt het conflict vast in zijn logboek.
- `resolution_log` krijgt `author`.
- Handmatige runs (`trigger = handmatig`, ADR 0005) zijn beschermd. Een automatische run op dezelfde bronversie maakt wel nieuwe fragmenten, maar verzet `source_version.current_run` niet zolang die naar een handmatige run wijst. De run legt dit vast in zijn logboek en zet een melding in de wachtrij van de beheerder. Alleen de beheerder kiest welke run de huidige wordt.

**Bestaande gegevens**

- De signaalsysteemomgeving start met een lege database (besluit Geert, 30-10, plan van aanpak §8 B3). Er is geen migratie van bestaande bronnen, fragmenten of embeddings, en dus geen bronversie of run met `pre-signaalsysteem` en geen opruimscript daarvoor.
- Omdat er geen oude records zijn, zijn `chunk.source_version` en `chunk.run` voor alle records verplicht, niet alleen voor nieuwe.
- De migraties blijven idempotent en hebben een `_down`-variant.
- Het oude datavolume blijft onaangeroerd staan als archief; de nieuwe omgeving krijgt een eigen volume. Wie bronnen uit de oude omgeving terug wil, leest de originelen opnieuw in via de nieuwe pipeline.

## Gevolgen

- **Code:**
  - nieuwe repositories `SourceVersionRepository` en `ProcessingRunRepository`;
  - een `ProcessingRunService` met de run-eindpunten;
  - aanpassing van de schrijfpaden voor fragmenten en embeddings, en van `fn::vector_search` en `fn::text_search` voor de standaard "nieuwste versie";
  - de vergrendelingshulp in `packages/surrealdb-service`.
- **Opslag:** fragmenten en embeddings stapelen zich op bij elke herverwerking. Opschonen van oude runs is een bewuste beheeractie, geen automatisme.
- **Tests (stopcondities):**
  - een schemacontrole dat `chunk.source_version` en `chunk.run` verplicht zijn;
  - een test dat herverwerking het aantal fragmenten van de oude run ongewijzigd laat;
  - een test dat een run zonder nieuwe inhoud een record `niets_nieuws` oplevert;
  - een test dat een wijziging met een verouderde `rev` HTTP 409 geeft;
  - de toets van fase 2 (FO 15): elk fragment is herleidbaar tot bronversie en run.
- **Afhankelijkheden:** het importeindpunt (ADR 0010) schrijft deze velden. Het auditlog van modelaanroepen (ADR 0007) verwijst naar `processing_run`.
