# ADR 0009: GPU-beheer: docling-serve, WhisperX en Kev onder llama-swap, en Docker-toegang

- **Status:** voorgesteld
- **Datum:** 2026-10-29
- **Raakt FO-paragraaf:** 6 (keuzemodel), 12.1, 12.2, 13.1 (proeven), 14 (implementatie GPU-beheerder)
- **Lost op:** verschilanalyse conflict 11, open vraag 12

## Context

Het functioneel ontwerp (FO) laat alle GPU-diensten (grafische kaart) via llama-swap lopen, met één wachtrij en groepsregels voor de RTX 5070 Ti met 16 GB videogeheugen (VRAM) (FO 12.2). Verder:

- docling-serve en WhisperX stoppen na een bundel taken, zodat het CUDA-geheugen vrijkomt;
- llama-swap kan containers starten en stoppen, maar heeft daarvoor toegang tot Docker nodig. Dat vraagt "een bewuste afweging, omdat die toegang ruime rechten geeft";
- lukt het niet, dan gebruiken de Langflow-componenten een gedeelde vergrendeling (FO 12.2).

Harde eis 4 sluit uit dat llama-swap buiten Docker op de host draait.

In de fork reserveren `docling`, `whisperx` en `mineru` elk de hele GPU (`docker-compose.yml:55-61`, `:123-129`, `:192-198`), en Ollama op de host gebruikt dezelfde kaart (aanname A5). Coördinatie ontbreekt.

**Gecontroleerd gedrag**

- llama-swap:
  - `cmd` en `cmdStop` samen ondersteunen Docker en Podman. `cmdStop` is nodig om een container netjes te stoppen;
  - `ttl` ontlaadt na inactiviteit; `ttl: 0` ontlaadt nooit;
  - groepen hebben `swap` (één lid tegelijk), `exclusive` (ontlaadt alle andere groepen) en `persistent` (andere groepen mogen dit lid niet ontladen);
  - `matrix` legt toegestane combinaties vast; v267 voegde een router op basis van capaciteit toe, met resourcepools en `evict_cost`;
  - `concurrencyLimit` geeft HTTP 429 boven de limiet; volgens de documentatie is er een wachtrij met prioriteit;
  - `/upstream/:model_id` geeft directe toegang tot een upstream-server.
  
  Niet gevonden: hoe llama-swap een upstream zonder `model`-veld in het verzoek laadt. Dat geldt voor docling-serve, en voor Kev via `/v1/systemone`.
- docling-serve: de configuratie kent geen optie om het videogeheugen na een taak vrij te geven of modellen te ontladen; wel `DOCLING_SERVE_LOAD_MODELS_AT_BOOT`. Stoppen van de container is dus de route (FO 12.2, proef 5).
- Kev:
  - bestaat: Jared Palmer, Apache-2.0, gewichten op Hugging Face in 0,8B, 4B, 9B en 27B;
  - de server is Python (`python -m kev.serve`), met een eigen API (Application Programming Interface): `POST /v1/systemone` en `GET /v1/models`, compatibel met de System One-API van TypeSafe en niet OpenAI-compatibel;
  - er wordt geen Docker-image en geen GGUF genoemd;
  - de README noemt voor Kev-9B ongeveer 17 GB GPU-geheugen bij serveren. Dat is meer dan de 16 GB van de RTX 5070 Ti;
  - kansen zijn standaard gekalibreerd met een temperatuur per checkpoint;
  - de server luistert standaard op 127.0.0.1, en met `KEV_API_KEY` vraagt hij een sleutel.
- docker-socket-proxy (Tecnativa): toegang per API-onderdeel, alles standaard dicht. `ALLOW_START` en `ALLOW_STOP` staan starten en stoppen van containers toe. Filteren op containernaam beschrijft de documentatie niet.

Externe bronnen (geraadpleegd 29-10-2026):

- [llama-swap, README](https://github.com/mostlygeek/llama-swap)
- [llama-swap, voorbeeldconfiguratie](https://raw.githubusercontent.com/mostlygeek/llama-swap/main/docs/config.example.yaml)
- [llama-swap, releases](https://github.com/mostlygeek/llama-swap/releases)
- [docling-serve, configuratie](https://raw.githubusercontent.com/docling-project/docling-serve/main/docs/configuration.md)
- [Kev](https://github.com/jaredpalmer/kev)
- [artikel over Kev](https://runtimewire.com/article/jared-palmer-kev-qwen35-decision-models)
- [docker-socket-proxy](https://github.com/Tecnativa/docker-socket-proxy)

## Opties

Voor de toegang van llama-swap tot Docker:

1. **De Docker-socket in de llama-swap-container.** `cmd` start de container met `docker run --rm --gpus all ...`, `cmdStop` stopt hem met `docker stop`.
   - Voor: llama-swap regelt alles.
   - Tegen: de socket staat gelijk aan beheerrechten op de host. Llama-swap is bereikbaar voor de fork en Langflow; wie llama-swap misbruikt, kan containers met hostmappen maken.
2. **Een socket-proxy met alleen starten en stoppen.** Compose maakt de containers voor docling-serve, WhisperX en Kev vooraf aan (`docker compose create`). llama-swap start en stopt ze via `DOCKER_HOST=tcp://socket-proxy:2375`, met `CONTAINERS=1`, `ALLOW_START=1`, `ALLOW_STOP=1` en verder alles dicht.
   - Voor: geen containers maken, geen `exec`, geen mappen koppelen.
   - Tegen: elke container is te starten en te stoppen, ook `surrealdb`. Dat geeft verstoring, maar geen toegang tot gegevens.
   - Te beproeven: of `ALLOW_START` werkt zonder `POST=1`, of het llama-swap-image een Docker-client bevat, en hoe llama-swap een container volgt die met `docker start` is gestart. Bijvoorbeeld met een `cmd` dat wacht tot de container stopt, en `checkEndpoint` voor de gereedmelding.
3. **Geen Docker-toegang.** llama-swap beheert alleen llama-server-modellen, en Kev als proces in een eigen image op basis van llama-swap. docling-serve en WhisperX blijven altijd draaien, met een gedeelde vergrendeling in de Langflow-componenten (de terugval uit FO 12.2).
   - Voor: geen ruime rechten.
   - Tegen: docling-serve geeft het videogeheugen niet vrij (zie hierboven). Dat geheugen telt dus altijd mee, en stoppen na een bundel kan niet zonder Docker-toegang op een andere plek.
4. **llama-swap op de host.** Valt af: strijdig met harde eis 4.

Voor Kev: de grootte volgt uit het GPU-budget (FO 6). Kev-9B past volgens de eigen README niet in 16 GB. Kev-4B is de kandidaat, Kev-0,8B de terugval. Een gekwantiseerde variant wordt niet genoemd.

## Voorstel

- **Docker-toegang: optie 2**, als proef 1 slaagt. Slaagt die niet, dan valt Geert terug op optie 1 met aanvullende maatregelen, of op optie 3. De proef bepaalt welke.
- **Groepen (beginindeling, bij te stellen na proef 3):** groep `zwaar` met `swap: true` en `exclusive: true` voor het werkmodel, Kev, docling-serve en WhisperX. Groep `licht` met `persistent: true` en `ttl: 0` voor het embeddingmodel (ADR 0008). Als de capaciteitsrouter uit v267 bij de metingen eenvoudiger blijkt, vervangt die de groepen.
- **Kev:** Kev-4B, achter llama-swap via `/upstream/<model-id>/v1/systemone` of via een eigen `proxy`-instelling. `ModelGate` (ADR 0006) roept Kev aan als aanroepwijze `llama-swap`. Kev luistert alleen op het interne netwerk en vraagt `KEV_API_KEY` uit een Docker secret.
- **Bundelen:** de Langflow-componenten groeperen taken per dienst. Een geheugenfout breekt de taak af; Langflow zet hem opnieuw in de rij en schrijft de fout in het runlogboek (FO 12.2; ADR 0003).
- **Opruimen:** `deploy.resources.reservations` voor de GPU blijft alleen bestaan op containers die llama-swap beheert. Ollama op de host gaat uit gebruik (ADR 0006). Of MinerU blijft, regelt ADR 0015; zo ja, dan komt het ook onder llama-swap.

Geert beslist over de Docker-toegang, na proef 1.

## Gevolgen

- **Proeven:**
  - proef 1: starten en stoppen via optie 2, en Kev via `/upstream`;
  - proef 3: geheugen per dienst, onder meer Kev-4B, docling-serve, WhisperX, werkmodel en embedding;
  - proef 5: vrijgeven van het videogeheugen door de container te stoppen.
- **Compose:** extra container `socket-proxy` op een eigen intern netwerk, alleen bereikbaar voor `llama-swap`. De containers die llama-swap beheert, hebben geen `restart: always`.
- **Toets fase 2 (FO 15):** twee gelijktijdige GPU-taken leiden niet tot een geheugenfout.
- **FO:** FO 6 noemt Kev in 0,8B, 4B en 9B. De README noemt ook 27B, en voor 9B een geheugengebruik boven het budget. Dat kan in het FO worden bijgewerkt.
