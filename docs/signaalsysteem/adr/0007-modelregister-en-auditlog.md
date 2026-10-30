# ADR 0007: Plaats van het modelregister en het auditlog van modelaanroepen

- **Status:** besloten (Geert, 2026-10-30)
- **Datum:** 2026-10-29
- **Raakt FO-paragraaf:** 9 (modelregister, beleidsregel audit), 11 (validatie bij het verbinden)
- **Lost op:** verschilanalyse open vraag 7 (de plaats van het register) en open vraag 13

## Context

Het functioneel ontwerp (FO) bepaalt wat het modelregister per model vastlegt:

- naam en versie;
- aanroepwijze: llama-swap, API of CLI (Command Line Interface);
- locatie: lokaal of cloud;
- toegestaan voor intern: ja of nee;
- geschikte taken.

Elke modelaanroep wordt gelogd met run, model en classificatie. Interne inhoud komt niet in logs buiten de lokale omgeving (FO 9). De Langflow-editor geeft een foutmelding als een cloudmodel achter een interne bron wordt gezet (FO 11). Het FO zegt niet wáár register en log staan.

In de fork:

- het providerregister staat in code (`packages/llm-manager/src/llm_manager/providers/registry.py:24`, met `is_local`);
- modelrijen staan in de database zonder migratie die de tabel definieert (verschilanalyse 3.3);
- er zijn routes onder `/api/models` (`apps/app-main/src/app_main/api/routers/models.py`);
- Track J schrijft per aanroep een regel `routing.served` in `metrics` (`apps/app-main/src/app_main/services/model_routing/telemetry.py`). Classificatie en run staan daar niet in, en routes buiten Track J schrijven geen regel.

`ModelGate` (ADR 0006) leest het register, zowel in de fork als in de Langflow-componenten.

## Opties

**Register**

1. **Tabel `model` in SurrealDB**, met een migratie, door Langflow te lezen via de API (Application Programming Interface) van de fork.
   - Voor: één plek, te beheren via de API.
   - Tegen: configuratie belandt in de SPOT (single point of truth). Een run kan alleen naar de toestand van het register op dat moment verwijzen als het register zelf versies heeft.
2. **YAML-bestand in Git**, naast de llama-swap-configuratie en de flows, bijvoorbeeld `config/modelregister.yaml`, alleen-lezen gekoppeld in de fork en in Langflow.
   - Voor: declaratief en onder versiebeheer (FO 2, principe 2). De run legt de Git-commit al vast (ADR 0003), dus de stand van het register is herleidbaar. Een test kan het bestand controleren tegen de model-id's in de llama-swap-configuratie.
   - Tegen: wijzigen vraagt een commit en een herstart. Voor één beheerder is dat geen bezwaar.
3. **In code** (`registry.py`). Valt af: een modelwissel wordt dan een codewijziging.

**Auditlog**

1. **De regel `routing.served` in `metrics` uitbreiden** met run en classificatie. `metrics` is een algemene tabel voor telemetrie, met een ander bewaarbeleid en andere lezers.
2. **Nieuwe tabel `model_call`**: alleen aanvulbaar en schemavast, met tijdstip, run, model, aanroepwijze, taak, classificatie, aantal tokens, duur, status en bron van de aanroep (fork of Langflow). De tabel bevat nooit prompt- of antwoordtekst. Langflow-componenten schrijven via `POST /api/v1/model-calls` (recht beheren, ADR 0002).

## Voorstel

- **Register: optie 2.** `/api/models` wordt een alleen-lezen weergave van het bestand; de schrijfroutes vervallen. De modelrijen in de database en het providerregister in `registry.py` vervallen, nadat hun lezers zijn omgezet. Het veld `toegestaan voor intern` staat standaard op `nee`: een model dat het register niet uitdrukkelijk toestaat, krijgt geen interne invoer.
- **Auditlog: optie 2.** `telemetry.py` wordt de schrijver voor `model_call`. De bestaande `routing.served`-regels blijven historisch staan in `metrics` en worden niet omgezet.
- Validatie bij het verbinden in Langflow (FO 11): een eigen modelcomponent leest hetzelfde register en weigert een model met `toegestaan voor intern = nee` zodra de bron intern of onbekend is.

Geert beslist.

## Besluit

Volgens het voorstel (Geert, 30-10): register optie 2 (YAML-bestand in Git), auditlog optie 2 (nieuwe tabel `model_call`), en validatie bij het verbinden in Langflow met hetzelfde register.

## Gevolgen

- **Code:**
  - een registerlader in `packages/llm-manager`, gebruikt door `ModelGate`;
  - een migratie voor `model_call`, met `_down`;
  - `/api/models` wordt alleen-lezen;
  - een test die het register naast de llama-swap-configuratie legt.
- **Interface:** het scherm voor modelbeheer kan alleen nog lezen.
- **Bij optie 1 voor het register:** een migratie voor de tabel `model`, met versie per wijziging, en de run legt de registerversie vast.
