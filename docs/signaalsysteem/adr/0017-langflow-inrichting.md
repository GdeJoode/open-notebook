# ADR 0017: Inrichting van Langflow: flows uit Git, aanmelding, planning en MCP

- **Status:** voorgesteld
- **Datum:** 2026-10-29
- **Raakt FO-paragraaf:** 2 (principe 2), 5 (planning), 11 (Langflow als editor), 12.1, 13.1, 14 (workflow-editor, orchestratie en scheduler)
- **Lost op:** geen nummer uit de verschilanalyse; keuzes die het FO openlaat en die bij het controleren van de documentatie naar boven kwamen

## Context

Het functioneel ontwerp (FO) zegt over Langflow:

- flows worden als JSON geëxporteerd en vanuit Git geladen (`LANGFLOW_LOAD_FLOWS_PATH`);
- eigen componenten staan in de repository (`LANGFLOW_COMPONENTS_PATH`);
- Langflow als MCP-server (Model Context Protocol) staat uit;
- de anonieme telemetrie staat uit (`DO_NOT_TRACK=true`);
- wijzigingen die agents in de definitie maken, zijn zichtbaar in de editor, en omgekeerd (FO 11, FO 12.1);
- de editor is bereikbaar op localhost of via Tailscale (FO 12.1);
- planning loopt via de schema-trigger van Langflow. Zolang die niet in een uitgebrachte versie zit, doet APScheduler het (FO 5, FO 14).

**Gecontroleerd (29-10-2026)**

- **Versie.** De nieuwste Langflow-versie is v1.13.0, van 22 oktober volgens de releasepagina; de pagina noemt geen jaar.
- **Schema-trigger.** PR #14958 is op 18 september 2026 samengevoegd in `release-1.13.0`. Die PR voegt een basiscomponent voor triggers toe, plus een schema-trigger met cron-expressie, tijdzone en inhaalbeleid. Volgens de beschrijving verbergt een latere commit het onderdeel Triggers achter een vlag `ENABLE_TRIGGERS`, die standaard uit staat "until the enable/state UI ships in TRG-7". De releasenotities van 1.13.0 noemen de schema-trigger. Of de vlag in 1.13.0 nog nodig is en of de trigger daar werkt, is niet vastgesteld; dat is proef 2.
- **Flows laden.** `LANGFLOW_LOAD_FLOWS_PATH` laadt JSON-flows bij het opstarten en vereist dat automatisch aanmelden aan staat. Met `LANGFLOW_AUTO_LOGIN=true` is de editor zonder wachtwoord te gebruiken. De API kent `POST /api/v1/flows/upload/` om te importeren en `POST /api/v1/flows/download/` om te exporteren (ZIP met flow-JSON). De voorbeelden sturen `x-api-key` mee.
- **MCP.** `LANGFLOW_MCP_SERVER_ENABLED` staat standaard aan; met `false` start Langflow geen MCP-servers.
- **Geheimen.** Globale variabelen worden versleuteld opgeslagen in de database van Langflow.

De editor kan alles: interne bronnen lezen, cloudmodellen aanroepen en in de vault schrijven. Aanmelden zonder wachtwoord via Tailscale geeft die macht aan elk apparaat in het privénetwerk.

Externe bronnen (geraadpleegd 29-10-2026):

- [Langflow, releases](https://github.com/langflow-ai/langflow/releases)
- [Langflow, PR #14958](https://github.com/langflow-ai/langflow/pull/14958)
- [Langflow, omgevingsvariabelen](https://docs.langflow.org/environment-variables)
- [Langflow, API voor flows](https://docs.langflow.org/api-flows)
- [Langflow als MCP-server](https://docs.langflow.org/mcp-server)
- [Langflow, globale variabelen](https://docs.langflow.org/configuration-global-variables)

## Opties

**A. Flows uit Git laden**

1. **`LANGFLOW_LOAD_FLOWS_PATH` met automatisch aanmelden.** Zoals het FO het beschrijft. De editor zit dan zonder wachtwoord open, op localhost en via Tailscale.
2. **Automatisch aanmelden uit.** Een opstartscript laadt de flows uit Git via `POST /api/v1/flows/upload/`, met een sleutel uit een Docker secret. Een exportscript haalt ze via `POST /api/v1/flows/download/` terug naar Git.
   - Voor: de editor vraagt een wachtwoord; de pipelinedefinitie blijft in Git.
   - Tegen: twee kleine scripts in plaats van één omgevingsvariabele.
3. **Optie 1 achter een reverse proxy met aanmelding.**
   - Tegen: nog een container, en de API van Langflow blijft zonder aanmelding bereikbaar voor wie de proxy omzeilt op het netwerk.

**B. Synchronisatie tussen editor en Git** (FO 11: "en omgekeerd")

1. Het exportscript draait handmatig na een wijziging in de editor; een agent die de JSON wijzigt, laat het opstartscript opnieuw laden.
2. Automatisch exporteren bij opslaan. Daarvoor is geen ingebouwde functie gevonden.

**C. Planning**

1. De schema-trigger van Langflow, als proef 2 slaagt in de vastgezette versie, eventueel met `ENABLE_TRIGGERS`.
2. Terugval: APScheduler in een kleine eigen container `planner`. Die leest de schema's uit de bronkaart via de API van de fork en start flows via de API van Langflow.

**D. Database van Langflow**

1. SQLite in een volume (de standaard).
2. PostgreSQL. Valt af voor één beheerder: extra container zonder winst.

## Voorstel

- **A: optie 2.** Aanmelden zonder wachtwoord geeft elk apparaat in het privénetwerk toegang tot interne bronnen. Dat staat op gespannen voet met harde eis 1, ook al gebruikt het FO zelf `LANGFLOW_LOAD_FLOWS_PATH`. De flow-JSON in Git blijft de pipelinedefinitie; alleen het laden verloopt anders.
- **B: optie 1.** Elke run legt de Git-commit en de hash van de flow-JSON vast (ADR 0003). Een run met een flow die niet in Git staat, krijgt in het runlogboek een waarschuwing `niet_vastgelegd`.
- **C:** het FO heeft dit besloten. Proef 2 bepaalt of het optie 1 of 2 wordt. Voor de terugval is de container `planner` het voorstel.
- **D: optie 1.**
- **Vast volgens het FO:**
  - `LANGFLOW_MCP_SERVER_ENABLED=false` en `LANGFLOW_ADD_PROJECTS_TO_MCP_SERVERS=false`;
  - `DO_NOT_TRACK=true`;
  - `LANGFLOW_COMPONENTS_PATH` naar een map met componenten in de repository.
- **Vaste versie:** v1.13.0, of de versie waarmee proef 2 slaagt.

Geert beslist over A. Over C beslist proef 2.

## Gevolgen

- **Code:**
  - een opstartscript en een exportscript voor flows;
  - een map voor Langflow-componenten in de repository, met een plek die het plan van aanpak vastlegt;
  - bij terugval op APScheduler de container `planner`.
- **FO:** bij optie A2 vervalt in FO 11 de verwijzing naar `LANGFLOW_LOAD_FLOWS_PATH` als laadmechanisme. Het principe "flows als JSON uit Git" blijft.
- **Proef 2:** vastgezette versie, vlag `ENABLE_TRIGGERS`, een cron-schema dat twee keer achter elkaar op tijd afgaat, en gedrag na een herstart (inhaalbeleid).
