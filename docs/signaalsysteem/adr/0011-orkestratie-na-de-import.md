# ADR 0011: Orkestratie van de stappen na de import

- **Status:** besloten (Geert, 2026-10-30)
- **Datum:** 2026-10-29
- **Raakt FO-paragraaf:** 2 (principe 2), 5 (planning), 11 (rolverdeling, wat LangGraph toevoegt), 12.2 (geheugenfout), 14 (orchestratie en scheduler)
- **Lost op:** verschilanalyse open vraag 5 (wie de stappen in de fork start; de runtabel staat in ADR 0003)

## Context

Het functioneel ontwerp (FO) bepaalt de rollen:

- Langflow bepaalt welke bouwstenen een bron doorloopt;
- LangGraph draait de lussen als Python-code in de fork, en Langflow roept elke lus aan als Python-component;
- runs draaien los van de editor (FO 11, FO 14);
- een mislukte taak wordt opnieuw geprobeerd en daarna gemeld (FO 5);
- bij een geheugenfout zet Langflow de taak opnieuw in de rij (FO 12.2).

Het FO zegt niet of de takenwachtrij van de fork (`packages/job-queue`) blijft. Het zegt ook niet wie de embeddings van fragmenten berekent: de Langflow-component of de fork.

In de fork:

- `packages/job-queue` heeft een worker met herhaalpogingen (`packages/job-queue/src/job_queue/worker.py:296-320`), standaard één taak tegelijk (`:53`);
- `advance_source` en `processing_stage` sturen de keten EMBED, EXTRACT en GRAPH (`apps/app-main/src/app_main/services/source_pipeline.py`; `ARCHITECTURE.md` paragraaf 14);
- embeddings voor fragmenten en zoekvragen komen uit dezelfde code (`pipelines/embeddings`).

## Opties

1. **De fork start de stappen zelf na een import.** Het importeindpunt zet EMBED, EXTRACT en GRAPH in de wachtrij; Langflow wacht tot de run klaar is.
   - Voor: hergebruik van `advance_source` zoals het is.
   - Tegen: de volgorde van de stappen staat dan in code van de fork, niet in de flow. Dat wijkt af van FO 2, principe 2 ("een pipeline is een definitiebestand").
2. **Langflow start elke stap via de API (Application Programming Interface) van de fork, en de wachtrij van de fork voert ze uit.** Een Langflow-component per stap (embedden, entiteitextractie, deduplicatie, graaf) roept `POST /api/v1/runs/{run_id}/steps/{stap}` aan. De fork zet een job in `job-queue` met `job.run` (ADR 0003), de component volgt de status. Herhaalpogingen doet de worker; na het maximum zet de fork de run op `mislukt` en meldt hem in de wachtrij van de beheerder.
   - Voor: de flow is de volledige pipelinedefinitie; de wachtrij en de worker worden hergebruikt; embeddings voor fragmenten en zoekvragen komen uit dezelfde code.
   - Tegen: een component moet wachten op een job. Daar is een eenvoudig volgmechanisme voor nodig.
3. **Langflow-componenten importeren de fork-code en draaien in het Langflow-proces.** Valt af: Langflow zou dan zelf de database benaderen. Dat omzeilt het API-filter (ADR 0002). En LangGraph hoort volgens FO 11 in de fork.

## Voorstel

Optie 2.

- `advance_source` en de automatische doorloop van `processing_stage` vervallen. `processing_stage` blijft als statusveld, en de stap-eindpunten zetten het.
- De LangGraph-modules (ADR 0012) zijn stappen in dezelfde vorm. Een stap die op review wacht, krijgt de status `wacht_op_review` en wordt hervat na het besluit van de beheerder.
- Embeddings van fragmenten berekent de fork via `pipelines/embeddings` en `ModelGate` (ADR 0006), niet de Langflow-component. Zo gebruiken fragmenten en zoekvragen gegarandeerd hetzelfde model.

Geert beslist.

## Besluit

Optie 2 (Geert, 30-10): Langflow start elke stap via de API van de fork, en de wachtrij van de fork voert ze uit.

## Gevolgen

- **Code:**
  - nieuwe stap-eindpunten en een `RunStepOrchestrator` in `apps/app-main`;
  - `job` krijgt `run`;
  - de handlers in `apps/app-main/src/app_main/handlers.py` voor EMBED, EXTRACT en GRAPH blijven, maar starten vanuit een stap;
  - in Langflow komt een gedeelde basiscomponent voor "stap starten en volgen".
- **Bij optie 1:** het importeindpunt start de keten, en de flow eindigt bij de import.
