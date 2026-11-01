# ADR 0022: Afscherming van Langflow

- **Status:** besloten (Geert, 2026-10-30: akkoord met het voorstel uit de review)
- **Datum:** 2026-10-30
- **Raakt FO-paragraaf:** 9 (beleid op drie plekken, Claude-API en -CLI), 11 (Langflow als editor), 12.1 (netwerken, volledig lokaal)
- **Wijzigt:** ADR 0002 F (netwerken, cloudsleutels), ADR 0017 B (flow die niet in Git staat), FO 12.1 ("wie het strikter wil")

## Context

Harde eis 1 wordt op drie plekken afgedwongen: het Docker-netwerk, de modelkeuze in de Langflow-componenten en het filter in de API (Application Programming Interface) van de fork (FO 9). Langflow ziet interne inhoud voordat de poortwachter heeft beslist, en schrijft in de vault. In de opzet van FO 12.1 had `langflow` bovendien uitgaand internet (bronnen ophalen) en de Claude-sleutel. Dan beschermt het netwerk niet meer: elke component, elke bibliotheek en elke wijziging in de editor kan gegevens naar buiten sturen. P2 liet dat zien: LiteLLM in Langflow probeerde 21 keer uit zichzelf een verbinding met GitHub (ADR 0017). De review (30-10) merkte dit aan als blokkerend.

**Gecontroleerd in de documentatie van Langflow 1.13.x** (geraadpleegd 30-10-2026; de pagina's noemen zelf versie 1.13.x):

- **Standaardcomponenten.** Langflow levert kerncomponenten, gegroepeerd naar doel (onder meer *Inputs and Outputs* en *Data*, met bijvoorbeeld *API Request*), en bundels met componenten voor diensten van derden, per aanbieder. Verouderde componenten zijn standaard verborgen, maar dat is geen toegangscontrole. Een lijst met toegestane componenten of een manier om bundels uit te zetten, beschrijft de pagina niet ([componenten](https://docs.langflow.org/concepts-components)).
- **Eigen code in de editor.** `LANGFLOW_ALLOW_CUSTOM_COMPONENTS=false` blokkeert het maken van eigen componenten, het wijzigen van code in de editor en het uitvoeren van ingebouwde componenten die code uitvoeren (*Python Interpreter*, de verouderde *Python REPL*). Bij het bouwen van een flow voert Langflow dan nooit de code uit die in de flow staat, maar de eigen servercopie van de component, gekozen op type; een knooppunt met een onbekend type wordt geweigerd. Componenten uit `LANGFLOW_COMPONENTS_PATH` blijven werken, tenzij `LANGFLOW_ALLOW_COMPONENTS_PATHS_OVERRIDE=false`. Met `LANGFLOW_SUBSTITUTE_OUTDATED_COMPONENT_CODE=false` weigert Langflow een flow waarvan de opgeslagen code afwijkt. De pagina noemt de instelling een bètafunctie die niet de enige beveiliging mag zijn ([eigen componenten blokkeren](https://docs.langflow.org/deployment-block-custom-components)). Eigen code in de editor kan dus worden uitgezet.
- **Andere instellingen** ([omgevingsvariabelen](https://docs.langflow.org/environment-variables)): `LANGFLOW_BLOCK_CODE_INTERPRETER_COMPONENTS` (standaard `False`) blokkeert de ingebouwde componenten die code uitvoeren; `LANGFLOW_AGENTIC_EXPERIENCE` (standaard `True`) zet Langflow Assistant aan, met een eigen tool voor het Model Context Protocol (MCP); `LANGFLOW_DEACTIVATE_TRACING` (standaard `False`); `OTEL_EXPORTER_OTLP_ENDPOINT` zet export via OpenTelemetry aan; `DO_NOT_TRACK` dekt alleen de anonieme telemetrie.
- **Tracing.** De tracingdienst van 1.13.0 start tracers voor LangSmith, LangWatch, Langfuse, Arize Phoenix, Opik, Traceloop, Openlayer en een eigen tracer. Met `LANGFLOW_DEACTIVATE_TRACING=true` start hij er geen ([`service.py` van v1.13.0](https://raw.githubusercontent.com/langflow-ai/langflow/v1.13.0/src/backend/base/langflow/services/tracing/service.py)). Welke variabelen een tracer aanzetten:
  - LangSmith: `LANGSMITH_TRACING`, `LANGSMITH_ENDPOINT`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`; oudere versies `LANGCHAIN_API_KEY` ([LangSmith](https://docs.langflow.org/integrations-langsmith));
  - Arize en Phoenix: `ARIZE_API_KEY` met `ARIZE_SPACE_ID`, of `PHOENIX_API_KEY`, of `PHOENIX_COLLECTOR_ENDPOINT` met een lokaal adres ([`arize_phoenix.py`](https://raw.githubusercontent.com/langflow-ai/langflow/v1.13.0/src/backend/base/langflow/services/tracing/arize_phoenix.py));
  - Opik: `OPIK_API_KEY` of `OPIK_URL_OVERRIDE`, met `OPIK_WORKSPACE` ([`opik.py`](https://raw.githubusercontent.com/langflow-ai/langflow/v1.13.0/src/backend/base/langflow/services/tracing/opik.py));
  - Traceloop: `TRACELOOP_API_KEY`, met `TRACELOOP_BASE_URL` ([`traceloop.py`](https://raw.githubusercontent.com/langflow-ai/langflow/v1.13.0/src/backend/base/langflow/services/tracing/traceloop.py));
  - Openlayer: `OPENLAYER_API_KEY` met een pipeline-id (`OPENLAYER_INFERENCE_PIPELINE_ID`, `OPENLAYER_PIPELINE_<flow>` of `OPENLAYER_LANGFLOW_MAPPING`) ([`openlayer.py`](https://raw.githubusercontent.com/langflow-ai/langflow/v1.13.0/src/backend/base/langflow/services/tracing/openlayer.py));
  - LangWatch en Langfuse: de tracers bestaan; hun variabelen zijn niet per stuk nagekeken.

## Opties

1. **Langflow met uitgaand internet, zoals FO 12.1.** Valt af: het netwerk is dan geen afscherming voor Langflow, en harde eis 1 hangt alleen nog aan de componenten en het filter.
2. **Langflow zonder uitgaand netwerk; ophalen en cloudaanroepen via kleine eigen containers met een eigen controle.** De strengere variant die FO 12.1 al noemde. Gekozen.

## Besluit

Optie 2, in zes onderdelen.

**A. Netwerk**

- `langflow` hangt alleen aan interne netwerken (ADR 0002 F): `verwerking` en `editor-intern`, in fase 4 ook `ophalen-intern`, in fase 5 ook `gateway-intern`. Nooit aan `uitgaand` of `toegang`. De editor is bereikbaar via de proxy `editor-toegang`.
- Een controle zoals P8 toetst in elke fase dat `langflow` geen internet heeft.

**B. Bronnen ophalen (fase 4): container `ophalen`**

- Een eigen kleine container `ophalen` op `ophalen-intern` en `uitgaand`. Langflow vraagt hem een bron op te halen met het id van een bronkaartregel en een adres.
- `ophalen` haalt alleen op voor externe routes. Per aanvraag vraagt hij de fork of het mag: `GET /api/v1/source-map-rules/{rule_id}/ophaalcontrole?url=…`. De fork antwoordt alleen ja of nee: ja als de regel actief is, `extern` bevestigd is (ADR 0020), en het adres de `location` van de regel is of op dezelfde host ligt (een item uit de feed van die regel). Bij nee weigert `ophalen` (HTTP 403). Alleen `GET` en `HEAD`.
- `ophalen` heeft daarvoor een eigen sleutel met het recht `ophalen`, dat alleen dit eindpunt mag aanroepen (ADR 0002 B). Hij heeft geen sleutel met leesrechten: een container met internet ziet zo geen gegevens van de fork. `open-notebook` hangt daarom ook aan `ophalen-intern`.
- Interne routes (watchfolder, uploads) halen niets op; ze lezen lokale mappen.
- Hij legt elke aanvraag vast (tijd, regel, adres, status, grootte), zonder inhoud.

**C. Cloudaanroepen (fase 5): container `cloud-gateway`**

- Een kleine container `cloud-gateway` op `gateway-intern` en `uitgaand`. Hij laat alleen verkeer naar het domein van Anthropic door; de bouwtaak legt de hostnamen vast uit de documentatie van Anthropic. Elke andere bestemming wordt geweigerd.
- Langflow stuurt per aanroep de run, de fragment-id's, het id van het sjabloon en het model. De gateway controleert per aanroep de classificatie bij de fork: hij haalt elk fragment op met een eigen sleutel `lezen_extern`, via de leesroute voor fragmenten die met de herkomsteindpunten komt (F3-09, ADR 0002 B). Geeft de fork voor één fragment HTTP 404 (intern of onbekend), dan weigert de gateway de hele aanroep. De gateway bouwt de prompt zelf uit het sjabloon (uit Git, gecontroleerd op hash) en de opgehaalde tekst. Vrije tekst van Langflow gaat niet mee; zo kan geen interne tekst die Langflow kent, de cloud bereiken.
- `claude-cli` hangt alleen aan `cli-intern`, samen met de gateway. De gateway roept `claude -p` aan na dezelfde controle; het uitgaande verkeer van de CLI loopt via de gateway, met dezelfde beperking tot het domein van Anthropic.
- De gateway geeft het gebruik (model, tokens, duur, status) terug aan Langflow, dat de regel in `model_call` schrijft (ADR 0007). De gateway houdt daarnaast een eigen logboek zonder tekst; een test vergelijkt beide.

**D. Componenten**

- Een vaste lijst toegestane componenttypen staat in Git (`config/langflow-componenten.yaml`): de eigen componenten uit `services/langflow/components/` en een kleine, door de bouwtaak vastgelegde set kerncomponenten zonder netwerk- of codefunctie. Componenten voor modelaanbieders, *API Request*, webzoeken, MCP-tools en *Python Interpreter* staan er niet op.
- Het opstartscript (ADR 0017 A) toetst elke flow uit Git vóór het laden: elk knooppunttype moet op de lijst staan. Een flow met een ander type laadt het niet; het script meldt de flow en het type en eindigt met een foutcode.
- Instellingen van Langflow: `LANGFLOW_ALLOW_CUSTOM_COMPONENTS=false`, `LANGFLOW_SUBSTITUTE_OUTDATED_COMPONENT_CODE=false`, `LANGFLOW_BLOCK_CODE_INTERPRETER_COMPONENTS=true` en `LANGFLOW_AGENTIC_EXPERIENCE=false`. `LANGFLOW_ALLOW_COMPONENTS_PATHS_OVERRIDE` blijft `true`, zodat de eigen componenten uit `LANGFLOW_COMPONENTS_PATH` laden; die map is alleen-lezen gekoppeld uit de repository. Omdat de documentatie de blokkade een bètafunctie noemt, zijn het netwerk (A) en de controle in de fork (E) de eigenlijke afscherming.

**E. Flow buiten Git**

- Een run met een flow die niet in Git staat, mislukt zodra de invoer intern of onbekend is (ADR 0017 B). De fork vergelijkt bij `POST /api/v1/runs` en bij de import de Git-commit en de hash van de flow-JSON met de flows in de repository, en weigert anders (HTTP 409). Bij externe invoer blijft het een waarschuwing `niet_vastgelegd`.
- **Restrisico** (review F2-04, 31-10). Commit en hash zijn een opgave van Langflow zelf. Wie de hash van het vastgelegde bestand opgeeft, komt door de controle, welke flow er ook echt draait. Beperking:
  - de fork leest de hashes uit Git (`HEAD`), niet uit de werkmap;
  - een map met flows met niet-vastgelegde wijzigingen telt als niet vastgelegd;
  - een opgegeven commit die niet gelijk is aan `HEAD` telt als niet vastgelegd.

  Wat blijft: een Langflow die liegt over de flow die hij draait. De echte afscherming daarvoor is het netwerk en de vaste lijst van componenten (A–D), niet deze controle.

**F. Tracing en sleutels**

- Verboden in de omgeving van `langflow`: alle variabelen die beginnen met `LANGCHAIN_`, `LANGSMITH_`, `LANGFUSE_`, `LANGWATCH_`, `ARIZE_` of `OPIK_`, en daarnaast, volgens de tracers van 1.13.0, `PHOENIX_`, `TRACELOOP_`, `OPENLAYER_` en `OTEL_EXPORTER_OTLP_`. Het startscript van de container weigert te starten als er één gezet is. Daarnaast `LANGFLOW_DEACTIVATE_TRACING=true` en `DO_NOT_TRACK=true`.
- Langflow heeft in geen enkele fase een Claude-sleutel. De Claude-API-sleutel staat alleen in `cloud-gateway`: een startscript leest hem uit het Docker secret in `/run/secrets/` en geeft hem alleen aan het gatewayproces, niet als omgevingsvariabele van de container. De CLI-login staat op dezelfde manier alleen in `claude-cli`.
- Langflow heeft alleen de sleutel `pijplijn` voor de fork, zijn eigen `LANGFLOW_SECRET_KEY` en het superuserwachtwoord, alle drie als bestand in `/run/secrets/`.

## Gevolgen

- **ADR 0002 F:** de netwerken `editor-intern`, `ophalen-intern`, `gateway-intern` en `cli-intern`; `uitgaand` alleen voor `ophalen` en `cloud-gateway`; de Claude-sleutel alleen in de gateway (bijgewerkt 30-10).
- **ADR 0017:** onderdeel B aangescherpt; de uitgaande aanvragen van LiteLLM worden door het netwerk geblokkeerd, in elke fase (bijgewerkt 30-10).
- **ADR 0006:** de aanroepwijzen `api` en `cli` lopen via de gateway (bijgewerkt 30-10).
- **FO 12.1:** de tabel met containers en de alinea over uitgaand internet (bijgewerkt 30-10).
- **Plan van aanpak** (de planner werkt het bij): `langflow` zonder `uitgaand` in de Compose-schets; nieuwe diensten `ophalen` (fase 4) en `cloud-gateway` (fase 5); de toets van de componentenlijst en de verboden variabelen in het opstartscript (F2-12); de controle van de flow-hash in de fork (F2-04 of F2-16); risico R5.
- **Tests (stopcondities):**
  - `langflow` bereikt geen internet, in elke fase (P8-controle);
  - het opstartscript weigert een flow met een type dat niet op de lijst staat;
  - het startscript weigert te starten met een verboden variabele, één test per voorvoegsel;
  - een run met een flow die niet in Git staat en interne invoer geeft HTTP 409 en wordt `mislukt`;
  - `ophalen` weigert een adres dat niet bij een actieve, externe regel hoort (HTTP 403);
  - de gateway weigert een aanroep met één intern fragment, weigert vrije tekst, en krijgt geen verbinding met een host buiten het domein van Anthropic;
  - `docker inspect` van `langflow` toont geen Claude-sleutel en geen verboden variabele.

**Uitwerking in F2-12 (01-11).** Langflow 1.13.0 hangt MCP- en agentinterfaces ook in als de instellingen ze uitzetten (`/api/v1/mcp`, `/api/v2/mcp`, `/api/v1/agentic/mcp`, A2A, OpenAI Responses). Het afgeleide image haalt die routes weg (de build faalt als dat niet lukt) en de editorproxy geeft er 404 op; L2 toetst dat aangemeld. Uitgaande aanvragen bij starten en gebruik (pypi-versiecontrole, models.dev, MCP Composer, Store) staan uit; L7 telt de DNS-vragen. De componentenlijst geldt voor de lader én als blokkeerlijst in Langflow zelf (catalog-policy); de laatste kan de superuser terugdraaien en is dus een extra laag naast het netwerk. Restrisico: de editor in de browser van de host haalt externe adressen op (GitHub-sterren, Discord-teller); er gaat geen interne inhoud mee.
- **Herreview F2-12.** Ook flows uitvoeren buiten de editor om is afgesloten: de editorproxy geeft 404 op `/run`, `/webhook`, `/process`, `/predict`, de publieke build, `/api/v2/workflows` en de voice-websocket; flows moeten `PRIVATE` zijn; CORS staat alleen `127.0.0.1:8530` toe en de proxy controleert de Host-header (geen DNS-rebinding). Restrisico: een proces op de host met toegang tot `secrets/` (wachtwoord van de superuser) kan de editor-API gebruiken en via `/api/v1/build` flows draaien en hun uitvoer lezen; na F2-16 kan dat interne inhoud zijn. De bescherming daartegen is de host zelf: `secrets/` is alleen voor Geert.
