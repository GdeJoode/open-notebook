# ADR 0009: GPU-beheer: docling-serve, WhisperX en Kev onder llama-swap, en Docker-toegang

- **Status:** besloten (Geert, 2026-10-30: Docker-toegang optie 2, socket-proxy met alleen starten en stoppen; Kev-4B)
- **Datum:** 2026-10-29; bijgewerkt 2026-10-30 met P1, P3, P5 en P7
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
- docker-socket-proxy (Tecnativa): toegang per API-onderdeel, alles standaard dicht. `ALLOW_START` en `ALLOW_STOP` staan starten en stoppen van containers toe. Filteren op containernaam beschrijft de documentatie niet. Volgens `haproxy.cfg` weigert de eerste regel elke methode behalve `GET` zolang `POST` niet aan staat; de regels voor `ALLOW_START` en `ALLOW_STOP` komen daarna (proeven.md §5, punt 1).

**Proeven (30-10-2026)**

- **Docker-toegang (P7).**
  - Het image `ghcr.io/mostlygeek/llama-swap:v268-cuda13-b11977` heeft geen Docker-client (wel `curl` en `sh`). Een afgeleid image met `COPY --from=docker:29.9.2-cli /usr/local/bin/docker /usr/local/bin/docker` heeft een werkende client (K1).
  - Rechten per configuratie van `tecnativa/docker-socket-proxy:v0.5.0`:

    | Configuratie | Instellingen | Uitkomst |
    | --- | --- | --- |
    | A | `POST=0`, `ALLOW_START=1`, `ALLOW_STOP=1` | alles 403 behalve `_ping`: `ALLOW_START` werkt niet zonder `POST=1` (K3) |
    | B | `POST=1`, `ALLOW_START=1`, `ALLOW_STOP=1`, `CONTAINERS=0` | `start` en `stop` 204; containerlijst, `inspect`, `create`, `exec`, `images` en `volumes` 403 (K2) |
    | C | `POST=1`, `ALLOW_START=1`, `ALLOW_STOP=1`, `CONTAINERS=1` (de eerdere tekst van optie 2) | `create` 201: een container is aan te maken; lijst en `inspect` 200 (K3) |

  - Elke container is in configuratie B op naam te starten: ook de tweede proefcontainer `dummy2` gaf 204 (K2).
  - llama-swap start een container zonder `model`-veld in het verzoek via `/upstream/<model-id>/`, volgt hem met een `cmd` die blijft lopen zolang de gezondheidscontrole slaagt (`p1/wacht_op_dienst.sh`), en stopt hem na `ttl` met `cmdStop`. Na een externe `docker stop` geeft de eerste aanvraag HTTP 502, tot de gezondheidslus (elke 5 s) de stop opmerkt; 20 s later weer 200 (K4).
- **Echte diensten (P1).** docling-serve, WhisperX en Kev-4B starten en stoppen onder llama-swap, met configuratie B en het afgeleide image. In groep `zwaar` (`swap: true`, `exclusive: true`) draaiden nooit twee zware diensten tegelijk, ook niet bij gelijktijdige aanvragen; piek 14 536 MiB, geen geheugenfout (K2, K5, K7). Kev heeft geen gezondheidseindpunt, maar `/openapi.json` en `/docs` geven zonder sleutel 200 en pas als het model geladen is; `/v1/models` geeft zonder sleutel 401 (K4). Bevinding: llama-swap stopte docling-serve op `ttl` (60 s) midden in een asynchrone taak, omdat tussen twee statusvragen 61 s zat; de takenrij van docling-serve staat in het geheugen, dus de taak was weg en `status/poll` gaf daarna 404 (1a).
- **Vrijgave (P5).** docling-serve houdt na een taak 2,1 tot 2,5 GB vast, ook na `GET /v1/clear/converters` en `/v1/clear/results`; na `docker stop` is het 0 MiB. Stoppen is dus de enige route.
- **Image en driver (P1, proeven.md §8 punt 23).** De tag `v268-cuda13-b11977` eist `cuda>=13.4`; driver 610.88 levert CUDA 13.3. De container start met `NVIDIA_DISABLE_REQUIRE=1` en `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64`; llama-server gebruikt dan de libcuda van de driver.

Externe bronnen (geraadpleegd 29-10-2026):

- [llama-swap, README](https://github.com/mostlygeek/llama-swap)
- [llama-swap, voorbeeldconfiguratie](https://raw.githubusercontent.com/mostlygeek/llama-swap/main/docs/config.example.yaml)
- [llama-swap, releases](https://github.com/mostlygeek/llama-swap/releases)
- [docling-serve, configuratie](https://raw.githubusercontent.com/docling-project/docling-serve/main/docs/configuration.md)
- [Kev](https://github.com/jaredpalmer/kev)
- [artikel over Kev](https://runtimewire.com/article/jared-palmer-kev-qwen35-decision-models)
- [docker-socket-proxy](https://github.com/Tecnativa/docker-socket-proxy)
- [docker-socket-proxy, `haproxy.cfg`](https://raw.githubusercontent.com/Tecnativa/docker-socket-proxy/master/haproxy.cfg)
- uitkomsten P1, P3, P5 en P7 in `proeven.md` (30-10-2026)

## Opties

Voor de toegang van llama-swap tot Docker:

1. **De Docker-socket in de llama-swap-container.** `cmd` start de container met `docker run --rm --gpus all ...`, `cmdStop` stopt hem met `docker stop`.
   - Voor: llama-swap regelt alles.
   - Tegen: de socket staat gelijk aan beheerrechten op de host. Llama-swap is bereikbaar voor de fork en Langflow; wie llama-swap misbruikt, kan containers met hostmappen maken.
2. **Een socket-proxy met alleen starten en stoppen.** Compose maakt de containers voor docling-serve, WhisperX en Kev vooraf aan (`docker compose create`). llama-swap start en stopt ze via `DOCKER_HOST=tcp://socket-proxy:2375`, met configuratie B: `POST=1`, `ALLOW_START=1`, `ALLOW_STOP=1`, `CONTAINERS=0` en verder alles dicht. De eerdere tekst noemde `CONTAINERS=1`; dat laat `create` toe (P7, configuratie C, 30-10).
   - Voor: geen containers maken, geen `exec`, geen `inspect`, geen mappen koppelen (P7, K2).
   - Tegen: elke container is op naam te starten en te stoppen, ook `surrealdb` (P7, K2). Dat geeft verstoring, maar geen toegang tot gegevens.
   - Beproefd (P7, P1, 30-10): `ALLOW_START` werkt alleen met `POST=1`; het llama-swap-image heeft geen Docker-client, dus een afgeleid image; llama-swap volgt de container met een `cmd` die blijft lopen zolang de dienst gezond is, en `checkEndpoint` voor de gereedmelding.
3. **Geen Docker-toegang.** llama-swap beheert alleen llama-server-modellen, en Kev als proces in een eigen image op basis van llama-swap. docling-serve en WhisperX blijven altijd draaien, met een gedeelde vergrendeling in de Langflow-componenten (de terugval uit FO 12.2).
   - Voor: geen ruime rechten.
   - Tegen: docling-serve geeft het videogeheugen niet vrij (zie hierboven; bevestigd in P5, 30-10). Dat geheugen telt dus altijd mee, en stoppen na een bundel kan niet zonder Docker-toegang op een andere plek.
4. **llama-swap op de host.** Valt af: strijdig met harde eis 4.

Voor Kev: de grootte volgt uit het GPU-budget (FO 6). Kev-9B past volgens de eigen README niet in 16 GB. Kev-4B is de kandidaat, Kev-0,8B de terugval. Een gekwantiseerde variant wordt niet genoemd.

## Voorstel

- **Docker-toegang: optie 2 met configuratie B** (P7, P1, 30-10). Proef 1 en 7 zijn geslaagd op dit punt; terugval op optie 1 of 3 is niet nodig.
  - `socket-proxy` (`tecnativa/docker-socket-proxy:v0.5.0`) met `POST=1`, `ALLOW_START=1`, `ALLOW_STOP=1`, `CONTAINERS=0`, de rest dicht, en de Docker-socket alleen-lezen gekoppeld.
  - llama-swap draait in een afgeleid image: het vastgezette llama-swap-image plus de Docker-client uit `docker:29.9.2-cli` (P7, K1).
  - Per dienst: `cmd` start de vooraf aangemaakte container met `docker start` en blijft lopen zolang de gezondheidscontrole slaagt; `cmdStop: docker stop <container>`; `proxy` naar de containernaam; `checkEndpoint` op een pad dat zonder sleutel 200 geeft. `docker start -a` en `docker wait` werken niet, omdat de proxy de paden onder `/containers` dichthoudt (proeven.md, P1 stap 2).
  - **Restrisico's**, die Geert met dit besluit aanvaardt:
    - elke container op de host is via de proxy op naam te starten en te stoppen, ook `surrealdb` en containers buiten het signaalsysteem (P7, K2). Dat geeft verstoring, geen toegang tot gegevens of mappen;
    - na een stop van buiten llama-swap geeft de eerste aanvraag HTTP 502, tot de gezondheidslus het merkt (P7, K4). De afnemer (de Langflow-component en `ModelGate`) probeert het bij 502 opnieuw, na een korte pauze en met een maximum aantal pogingen.
- **Groepen** (P1 en P3, 30-10):

  | Groep | Instellingen | Leden |
  | --- | --- | --- |
  | `zwaar` | `swap: true`, `exclusive: true` | werkmodel, Kev-4B, docling-serve, WhisperX |
  | `licht` | `persistent: true`, geen `ttl` (ontlaadt nooit) | embeddingmodel mxbai-embed-large Q8_0 (ADR 0008) |

  In P1 hield `zwaar` de zware diensten uit elkaar, ook bij gelijktijdige aanvragen (K5). De reranker staat niet in `licht`: naast Kev-4B en mxbai is maar 121 MiB marge (zie de tabel hieronder); zijn plek regelt ADR 0015. De capaciteitsrouter uit v267 is niet nodig gebleken.
- **`ttl` van docling-serve.** llama-swap telt `ttl` vanaf de laatste aanvraag, niet vanaf het einde van een asynchrone taak, en een gestopte docling-serve verliest zijn lopende taken (P1, 1a, 30-10). Daarom:
  - de `ttl` van docling ligt ruim boven de langste pauze tussen twee statusvragen van de component. Voorstel: component pollt elke 10 s, `ttl` 300 s;
  - geeft `/v1/status/poll/{task_id}` HTTP 404, dan is de taak verloren en dient de component hem opnieuw in, met een maximum aantal pogingen en een regel in het runlogboek (ADR 0003).
  - Niet beproefd: een aanvraag voor een andere zware dienst tijdens een lopende asynchrone docling-taak. Volgens dezelfde redenering stopt llama-swap docling-serve dan ook, omdat de taak voor llama-swap geen lopende aanvraag is. De bundeling per dienst (hieronder) beperkt dat; de bouwtaak voor de docling-component toetst het.
- **Kev:** Kev-4B, achter llama-swap via `/upstream/kev-4b/v1/systemone` (P1, K4). `ModelGate` (ADR 0006) roept Kev aan als aanroepwijze `llama-swap`. Kev heeft geen eigen image: de bouwtaak bouwt het zoals `p1/kev.Dockerfile` (Git-tag `kev-1.0`, `uv sync --extra serve`), met de adapter en het basismodel `Qwen/Qwen3.5-4B-Base` op vaste revisies in een volume (proeven.md 1.7). Kev start met `--host 0.0.0.0`, luistert alleen op het interne netwerk en vraagt `KEV_API_KEY` uit een Docker secret. Gereedheidscontrole: `checkEndpoint: /openapi.json`; dat pad geeft zonder sleutel pas 200 als het model geladen is (P1, 1d). Met `checkEndpoint: none` stuurde llama-swap de aanvraag door voordat Kev luisterde (502).
- **Geheugen per dienst** (P3, 30-10; MiB; nulmeting 0, drie herhalingen; grens volgens de beslisregel van P3: 16 303 − 1 024 = 15 279 MiB):

  | Dienst | Piek | Rust | Samen met mxbai | Past |
  | --- | --- | --- | --- | --- |
  | mxbai-embed-large Q8_0, `-c 512` | 584 | 584 | — | ja |
  | werkmodel Qwen3.5-9B Q4_K_M, context 8 192 | 5 696 | 5 696 | 6 279, gemeten | ja |
  | werkmodel Qwen3.5-9B Q4_K_M, context 32 768 | 6 488 | 6 488 | 7 072, berekend | ja |
  | Kev-4B bf16 | 14 574 | 14 574 | 15 157, gemeten | ja, marge 121 |
  | docling-serve | 2 438–2 502 | 2 438–2 502 | 3 021, gemeten (1 herhaling) | ja |
  | WhisperX variant A (`diarize=true`) | 10 200–10 328 | 8 874 | 10 912, berekend | ja |

  Na het stoppen is elke dienst 0 MiB. De combinaties mxbai + WhisperX en docling-serve + WhisperX zijn niet gemeten (P3 is na 28 van 36 metingen gestopt, zie Gevolgen); in `zwaar` draaien docling-serve en WhisperX toch nooit samen.
- **Kev-4B en het budget.** Kev-4B gebruikt in bf16 14,6 GB, veel meer dan zijn 9,3 GB aan gewichten. Met mxbai blijft 1 146 MiB van de kaart vrij, 121 MiB boven de grens. De meting had een nulmeting van 0 MiB. Gebruikt het Windows-bureaublad of een andere toepassing videogeheugen, dan gaat dat van die marge af. Geeft Kev-4B in gebruik een geheugenfout, dan is de terugval Kev-0,8B (zoals hiervoor besloten).
- **Bundelen:** de Langflow-componenten groeperen taken per dienst. Een geheugenfout breekt de taak af; Langflow zet hem opnieuw in de rij en schrijft de fout in het runlogboek (FO 12.2; ADR 0003).
- **Opruimen:** `deploy.resources.reservations` voor de GPU blijft alleen bestaan op containers die llama-swap beheert. Ollama op de host gaat uit gebruik (ADR 0006). Of MinerU blijft, regelt ADR 0015; zo ja, dan komt het ook onder llama-swap.

Geert beslist over de Docker-toegang. De proeven laten zien dat optie 2 met configuratie B werkt (P1, P7, 30-10); open is alleen het akkoord op de twee restrisico's hierboven. Zonder dat akkoord blijft optie 3 (geen Docker-toegang, docling-serve en WhisperX altijd aan, met vergrendeling) de terugval; die kost volgens P3 en P5 blijvend 2,4 GB plus 8,9 GB videogeheugen in rust, en dan past Kev-4B niet meer.

## Gevolgen

- **Proeven:** uitgevoerd op 30-10.
  - proef 1: starten en stoppen via optie 2, en Kev via `/upstream`: geslaagd, met de bevinding over `ttl` en asynchrone docling-taken;
  - proef 3: geheugen per dienst: deels uitgevoerd (28 van 36 metingen; zie de tabel). Niet gemeten: mxbai + WhisperX en docling-serve + WhisperX. P3 is gestopt doordat de host nog 1,2 GB werkgeheugen vrij had (WSL-VM 13,2 GB). Het werkgeheugen van de host is dus ook een grens; dat is niet per dienst gemeten;
  - proef 5: docling-serve geeft het videogeheugen alleen vrij door de container te stoppen: geslaagd;
  - proef 7: rechten van de socket-proxy en het volgen van een container: configuratie B werkt; K4 mislukt alleen op de eerste aanvraag na een externe stop (502).
- **Compose:**
  - extra container `socket-proxy` op een eigen intern netwerk `beheer` (`internal: true`), alleen bereikbaar voor `llama-swap`, zoals in P1. De containers die llama-swap beheert, hebben geen `restart: always`;
  - Compose zet secrets met bron `environment` alleen bij `compose up`, niet bij een `docker start` door llama-swap; docling-serve stopte daardoor op een lege sleutel (proeven.md §8 punt 24). De bouwtaak gebruikt secrets die ook bij `docker start` aanwezig zijn (bestand of volume) en toetst dat;
  - het llama-swap-image eist `cuda>=13.4` en driver 610.88 levert 13.3 (zie Context). Zolang de driver niet is bijgewerkt, zet Compose `NVIDIA_DISABLE_REQUIRE=1` en `LD_LIBRARY_PATH` zoals in P1. Een driver met CUDA ≥ 13.4 maakt dat overbodig.
- **Toets fase 2 (FO 15):** twee gelijktijdige GPU-taken leiden niet tot een geheugenfout. In P1 (K5) gold dat voor Kev en het werkmodel; de toets herhaalt het met een lopende asynchrone docling-taak (zie Voorstel).
- **FO:** FO 6 noemt nu Kev in 0,8B, 4B, 9B en 27B en het gemeten geheugen van Kev-4B; FO 12.2 de groepsindeling, het geheugenbudget en de `ttl`-regel voor docling-serve (bijgewerkt 30-10).
