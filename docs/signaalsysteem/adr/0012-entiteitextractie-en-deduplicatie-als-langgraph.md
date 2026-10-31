# ADR 0012: Entiteitextractie en deduplicatie als LangGraph-modules; één extractiepad

- **Status:** besloten
- **Datum:** 2026-10-29
- **Raakt FO-paragraaf:** 7 (entiteiten, deduplicatie), 11 (wat LangGraph toevoegt), 13.2 punt 6
- **Lost op:** verschilanalyse open vraag 9

## Context

Het functioneel ontwerp (FO) zegt:

- entiteitextractie met validatie tegen referentielijsten, en deduplicatie met review, zijn LangGraph-modules in de fork (FO 7, FO 13.2 punt 6);
- bij entiteitextractie wordt elke entiteit getoetst, en worden alleen de afgekeurde voorstellen opnieuw gedaan, met een stopconditie (FO 11);
- deduplicatie pauzeert bij twijfel tot de beheerder beslist (interrupt) en gaat daarna verder waar de lus stopte (FO 11);
- checkpoints staan in een eigen lokaal bestand (FO 11).

In de fork:

- entiteitextractie bestaat twee keer:
  - in app-main: `EntityExtractionService` in `apps/app-main/src/app_main/services/entity_extraction_service.py:573`, met `pipelines/ontology-extraction`;
  - in de aparte extractiedienst: `services/extraction/api.py:634`. Die schrijft rechtstreeks naar SurrealDB (`docker-compose.yml:162-166`), heeft geen authenticatie (ADR 0002) en roept Ollama aan via `packages/shared/src/shared/model_routing.py` (ADR 0006);
- deduplicatie zit in `FilteringWorkflow` (`pipelines/entity-filtering/src/entity_filtering/workflow.py:57`). De triagewachtrij (`migrations/60.surrealql`; Track Q) blokkeert niet;
- LangGraph wordt alleen in `apps/app-main/src/app_main/graphs/` gebruikt, met een SQLite-checkpoint (`apps/app-main/src/app_main/config.py:11-13`);
- de referentielijstcode voor TOOI (Thesauri en Ontologieën voor Overheidsinformatie) en CBS (Centraal Bureau voor de Statistiek) staat deels in de extractiedienst (`services/extraction/vocabulary_manager.py`) en deels in `packages/shared/src/shared/vocabulary/tooi_provider.py`.

## Opties

Het FO maakt de keuze voor LangGraph-modules in de fork. Over de uitvoering:

1. **Beide extractiepaden houden.** Valt af: de extractiedienst omzeilt de API (Application Programming Interface) van de fork en het filter, en schrijft zonder run.
2. **Het pad in app-main is leidend en wordt in LangGraph-grafen verpakt. De extractiedienst vervalt.** Gekozen.

## Voorstel

Optie 2.

- **Graaf `entity_extraction`** in `apps/app-main/src/app_main/graphs/`. Plek volgens `ARCHITECTURE.md` paragraaf 4. De knopen:
  - extraheren met `EntityExtractionService` en de orchestrator voor meerdere schema's uit `pipelines/ontology-extraction`;
  - valideren tegen de referentielijsten (`reference_entity`, de woordenlijstcode);
  - alleen afgekeurde voorstellen opnieuw, met een maximum aantal rondes;
  - wegschrijven via `entity_persistence_service.py`, met `run` (ADR 0003).
  
  De kandidaat-ankers en de onthouding uit Track N blijven onderdeel van de extractie.
- **Graaf `entity_dedup`**: de lagen van `FilteringWorkflow` (ruisfilter, normalisatie, fuzzy matching, embedding, matching met een taalmodel) zijn knopen.
  - Bij twijfel maakt de graaf een item in `triage_queue` en roept `interrupt()` aan.
  - Het triagebesluit via de API (router `triage`) hervat de graaf met hetzelfde `thread_id`.
  - Checkpoints staan in het bestaande SQLite-bestand.
- Beide grafen zijn stappen in de vorm van ADR 0011, en alle modelaanroepen gaan via `ModelGate` (ADR 0006).
- **De extractiedienst wordt verwijderd:** `services/extraction/`, de dienst in Compose en `services_proxy` (ADR 0002).
  - Code die alleen daar staat, verhuist eerst. `vocabulary_manager.py` gaat naar `packages/shared/src/shared/vocabulary/`, naast `tooi_provider.py`.
  - `packages/shared/src/shared/model_routing.py` en `model_routing.yaml` vervallen met deze dienst en met ADR 0006.
- De regressiepoort uit Track N (`scripts/n_extraction_gate.py`, volgens `ARCHITECTURE.md` paragraaf 6) blijft de stopconditie voor de extractiekwaliteit.

## Gevolgen

- **Code:** twee nieuwe grafen met tests. `EntityExtractionService` en `FilteringWorkflow` worden aangeroepen vanuit knopen; hun inwendige werking verandert niet.
- **Tests (stopcondities):**
  - een test dat een afgekeurd voorstel opnieuw gaat en een goedgekeurd niet;
  - een test dat de deduplicatiegraaf bij twijfel stopt, en na een triagebesluit hervat met dezelfde tussenstand;
  - de regressiepoort van Track N.
- **Planning:** labels met Kev (FO 7) zijn een aparte bouwsteen in Langflow en vallen buiten deze ADR.
- **ARCHITECTURE.md:** paragraaf 1 (`services/extraction`) en paragraaf 6 worden bijgewerkt.

**Uitwerking in F2-02 (31-10).** `vocabulary_manager.py` is verhuisd naar `packages/shared/src/shared/vocabulary/`. `entity_validator.py` is niet verhuisd: de cachelaag zit al in `ReferenceEntityRepository` (`lookup_by_name`, `lookup_by_alias`) en `vocabulary_reconcile_service`, en de module deed externe aanroepen (ROR, OpenAlex) met entiteitnamen, die met harde eis 1 niet zonder meer terug mogen komen. De knoop "valideren tegen `reference_entity`" in F3-06 gebruikt de heuristieken (fuzzy naamvergelijking, personen, generieke namen) uit `22a9f182:services/extraction/entity_validator.py` als bron, zonder de externe aanroepen.

**Uitwerking in F2-09 (01-11).** De contextpacking van de oude extractie (Track M, `services/extraction_chunking/`) las het contextvenster uit een modelrij en werkt sinds F2-08/F2-09 niet meer; het pakket en zijn tests gaan weg. De graaf `entity_extraction` (F3-06) bepaalt zelf hoe fragmenten per aanroep worden samengenomen, met het contextvenster uit het modelregister (ADR 0007). Tot F3-06 gaat de oude extractie per fragment, met de vaste grens voor Pass 2.
