# Proefplan signaalsysteem

Stand: 30 oktober 2026 · stap 3 van `opdracht.md`: plan (3a) en scripts met eerste uitvoering (3b) · branch `docs/signaalsysteem-fo`

Dit plan beschrijft de proeven uit het functioneel ontwerp (FO) paragraaf 13.1 en de open punten die de ontwerpbesluiten (Architecture Decision Records, ADR's) naar de proeven verwijzen. De bouwagent (`signaal-bouwer`) schrijft de scripts en voert ze uit; hij vult per proef het veld "Uitkomst". Stap 3b: alle scripts staan er; P0, P8, P6, P2 en P4 zijn uitgevoerd, P1, P3, P5 en P7 nog niet (zie paragraaf 8).

## 1. Gemeenschappelijke opzet

### 1.1 Uitvoeringsplek

Deze machine is de werkplek van Geert: Windows 11, Docker Desktop met WSL2 (Windows Subsystem for Linux), NVIDIA RTX 5070 Ti met 16 GB videogeheugen (VRAM). Elk script controleert dat zelf (zie P0). Twee soorten uitvoering:

- **Hier.** De bouwagent kan de proef zelfstandig draaien: geen grafische kaart (GPU) nodig en geen handeling van Geert.
- **Uit te voeren op de werkplek.** De proef heeft de GPU exclusief nodig, of een handeling of akkoord van Geert (Docker-socket, bestaande GPU-containers stoppen, token). De bouwagent start pas na een uitdrukkelijk akkoord van Geert in het gesprek.

### 1.2 Isolatie: geen bestaande data of containers raken

- Alle proeven draaien in één eigen Compose-project: `docker compose -p signaal-proef -f scripts/signaalsysteem/proeven/compose.proef.yml --profile <pN> ...`. Het script roept Compose alleen via een hulpfunctie aan die project en bestand vastzet. De bestaande `docker-compose.yml` wordt niet gelezen, niet gestart en niet gewijzigd.
- Geen bind mounts naar `surreal_data`, `notebook_data`, `notebook_data_live`, `docling_input`, `docling_output` of andere mappen in de repository. Het script leest `docker.env` niet. Data van de proeven staat in benoemde volumes met het voorvoegsel `signaal-proef_`, of in een werkmap buiten de repository (`${SIGNAAL_PROEF_DIR:-$HOME/.cache/signaal-proef}`).
- Proef 4 gebruikt een eigen SurrealDB-container in geheugenmodus (`start memory`), zonder volume. Die staat dus los van de bestaande database.
- Vóór en na elke proef legt het script `docker ps -a`, `docker volume ls`, `docker network ls` en `docker image ls` vast. Voor alles buiten het project `signaal-proef` moet het verschil leeg zijn; anders eindigt het script met exitcode 3.
- Verboden in de scripts: `docker system prune`, `docker volume prune`, `docker image prune`, `docker compose down` zonder `-p signaal-proef`, en het ophalen van zwevende tags die de bestaande omgeving gebruikt (zoals `surrealdb/surrealdb:v2`). Een `docker pull` van `v2` zou de bestaande stack bij een volgende start ongemerkt van versie laten wisselen.
- Gepubliceerde poorten alleen op `127.0.0.1` en in de reeks 18000–18999, nooit 5055, 8000, 8502 of andere poorten die de bestaande omgeving gebruikt. De meeste proeven publiceren niets: een clientcontainer op het proefnetwerk doet de aanroepen.
- Opruimen met `p9_opruimen.sh`: `docker compose -p signaal-proef down`, met `--volumes` alleen als dat expliciet wordt meegegeven. De modelvolumes blijven standaard staan, zodat een herhaling niet opnieuw downloadt.

### 1.3 Geheimen

- Wachtwoorden en sleutels voor de proef (SurrealDB, Langflow, `KEV_API_KEY`, `DOCLING_SERVE_API_KEY`) maakt het script per run aan met `openssl rand -hex 24`. Ze staan alleen in het geheugen van het script en gaan via een Compose-secret met bron `environment` naar de container. Volgens de Compose-documentatie maakt die bron een secret uit de waarde van een omgevingsvariabele op de host, zonder bestand.
- Een Hugging Face-token (alleen nodig voor WhisperX-variant B in P1) of een API-sleutel (Podcast Index, YouTube Data API; beide optioneel in P6) leest het script alleen uit een omgevingsvariabele die Geert in zijn eigen shell zet. Het script schrijft geen geheim naar een bestand of log; logregels met `token`, `key` of `password` worden gemaskeerd.

### 1.4 Uitkomst en exitcodes

Elk script schrijft `uitkomst.json` in zijn runmap buiten de repository: datum, commit, versies en digests van de images, meetwaarden per criterium, en de exitcode. De bouwagent neemt de kern over in het veld "Uitkomst" hieronder. Ruwe logs blijven buiten de repository.

| Exitcode | Betekenis |
| --- | --- |
| 0 | alle succescriteria gehaald |
| 1 | minstens één criterium niet gehaald (een inhoudelijke uitkomst, geen fout van het script) |
| 2 | omgeving ongeschikt of voorwaarde ontbreekt (geen GPU, geen akkoord, geen token, te weinig schijfruimte) |
| 3 | scriptfout, of een wijziging buiten het project `signaal-proef` vastgesteld |

### 1.5 Meten van het videogeheugen

- Het script meet op de host met `nvidia-smi.exe --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -lms 500` en schrijft een CSV-bestand. Valt dat weg, dan meet het in een container met `--gpus all`; die ziet de hele kaart.
- Onder WSL2 heeft `nvidia-smi` volgens NVIDIA een beperkte functieset. Geheugen per proces is daarom geen betrouwbare maat. De proeven meten het totaal van de kaart en draaien de diensten één voor één, met een nulmeting vooraf.
- Het Windows-bureaublad gebruikt zelf videogeheugen. Het script legt de nulmeting vast en waarschuwt als die boven 2 GB ligt.
- Een GPU-proef start niet (exitcode 2) als een container buiten `signaal-proef` met een GPU-reservering draait (`HostConfig.DeviceRequests`), zoals `docling`, `whisperx` of `mineru` uit de bestaande omgeving. Ook Ollama op de host mag dan geen model geladen hebben (`ollama ps` leeg). Het script stopt die zelf niet; Geert doet dat.

### 1.6 Testdata uit openbare bronnen

Het script downloadt de bestanden naar de werkmap buiten de repository, legt de SHA-256 vast bij de eerste download en controleert die bij elke herhaling. Er komt geen testbestand in Git.

| Id | Soort | URL | Gebruik |
| --- | --- | --- | --- |
| D1 | PDF, gemeente Ridderkerk, jaarstukken 2025 (7,5 MB, volledig document met paragraaf weerstandsvermogen en risicobeheersing) | `https://ridderkerk.begrotingsapp.nl/Data/416de56b-57da-4d12-8484-3b5fc1ec689e/Public/jaarstukken-2025.pdf` | P1, P3, P5: zware Docling-taak |
| D2 | PDF, gemeente Wageningen, jaarstukken 2025, paragraaf 1 weerstandsvermogen en risicobeheersing (2,8 MB, 5 pagina's met tabellen) | `https://wageningen.begrotingsapp.nl/Data/f9c877a9-64e0-4403-a24b-cc1825aa0bba/Public/jaarstukken-2025%20-%20paragraaf-1-weerstandsvermogen--risicobeheersing.pdf` | P1, P5: korte taak, ook met `force_ocr` |
| D3 | YouTube-kanaalfeed van de Tweede Kamer (kanaal-id van de eigen socialemediapagina van de Tweede Kamer; 15 items bij controle) | `https://www.youtube.com/feeds/videos.xml?channel_id=UCEqXS8PoH57XaizXKMTL_UQ` | P6: alleen metadata, geen video of ondertitels |
| D4 | Podcastfeed NOS De Dag | `https://podcast.npo.nl/feed/de-dag.xml` | P6; audio van de nieuwste aflevering (circa 19 MB, 20 minuten, meerdere sprekers) voor WhisperX in P1 en P3 |
| D5 | Spotify-link van dezelfde show | `https://open.spotify.com/show/2Oeh3L5u0y9R54jEz8Mgty` | P6: omzetten naar D4 |
| D6 | Synthetische fragmenten, door het script gemaakt | geen URL | P4: nooit echte interne documenten |
| D7 | Audio, Nederlands: gesproken Wikipedia-artikel "Fractievoorzitter" (Wikimedia Commons, CC BY-SA 3.0; 75 s, 0,77 MB, SHA-1 `893cc979…ac075`) | `https://upload.wikimedia.org/wikipedia/commons/5/58/Nl-Fractievoorzitter.ogg` | P1, deelstap 1h (nl) en voorbereiden (uitlijnmodel nl) |
| D8 | Audio, Engels: gesproken Wikipedia-artikel "Elongate carpet shark" (Wikimedia Commons, CC BY-SA 4.0; 64 s, 0,91 MB, SHA-1 `9ff5baa0…e55e361`) | `https://upload.wikimedia.org/wikipedia/commons/4/47/Spoken_Wikipedia_en_Elongate_carpet_shark.ogg` | P1, deelstap 1h (en) en voorbereiden (uitlijnmodel en) |

Het audiobestand van D4 is een openbare aflevering uit de openbare feed. Het wordt alleen lokaal getranscribeerd en daarna met de runmap verwijderd. Van YouTube haalt geen proef iets anders op dan de feed (FO 5).

### 1.7 Vastgezette versies

Gecontroleerd op 29-10-2026. Grootte is de downloadgrootte volgens de genoemde bron; "meten" betekent dat het script de grootte vastlegt.

| Onderdeel | Vaste versie | Grootte | Proef | Bron |
| --- | --- | --- | --- | --- |
| llama-swap met llama-server, CUDA 13 | `ghcr.io/mostlygeek/llama-swap:v268-cuda13-b11977` | meten | P1, P3, P7 | GitHub-pakketpagina; README |
| docling-serve, CUDA 12.8 | `quay.io/docling-project/docling-serve-cu128:v1.38.2` | 11,4 GB | P1, P3, P5 | README, releases |
| WhisperX, variant A | `public.ecr.aws/deep-learning-containers/whisperx:3.8.6-cu128-amzn2023` | meten | P1, P3 | AWS-documentatie |
| WhisperX, variant B (alleen als A afvalt) | `Nyralei/whisperx-api-server`, lokaal gebouwd uit een vaste commit; geen uitgebrachte versie | meten | P1 | projectpagina |
| Kev | Git-tag `kev-1.0` van `jaredpalmer/kev`, lokaal gebouwd op `python:3.13.18-slim-bookworm` | meten | P1, P3 | README Kev |
| Langflow | `langflowai/langflow:1.13.0` (digest `sha256:ff2716e0…e303f`) | 785 MB gecomprimeerd | P2 | Docker Hub |
| docker-socket-proxy | `tecnativa/docker-socket-proxy:v0.5.0` | ca. 20 MB | P7, P1 | Docker Hub |
| Docker-client (in afgeleid llama-swap-image en als testclient) | `docker:29.9.2-cli` | 67 MB | P7, P1 | Docker Hub |
| HTTP-testclient | `curlimages/curl:8.23.0` | 11 MB | alle | Docker Hub |
| SurrealDB voor proef 4 | `surrealdb/surrealdb:v2.7.0` | 32 MB | P4 | Docker Hub |
| Python voor hulpscripts | `python:3.12.16-slim-bookworm` | meten | P2, P4, P6, P7, P8 | Docker Hub |
| API van de fork | lokaal gebouwd uit de huidige commit met de bestaande `Dockerfile`, tag `signaal-proef/open-notebook:<commit>` | meten | P4 | repository |
| open-notebook-mcp | PyPI `open-notebook-mcp==0.3.0` (22-12-2025, nieuwste) | klein | P4 | PyPI |
| MCP-client | PyPI `mcp==1.30.0` (`2.3.0` is de nieuwste, maar een nieuwe hoofdversie; afwijken alleen met vermelding in de uitkomst) | klein | P4 | PyPI |
| feedparser | PyPI `feedparser==6.0.14` | klein | P6 | PyPI |
| APScheduler (terugval) | PyPI `APScheduler==3.11.3`; geen 4.0-alfa | klein | P2 | PyPI |

| Model | Vaste revisie | Grootte | Proef |
| --- | --- | --- | --- |
| Kev-4B (LoRA-adapter) | `jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52` | 160 MB | P1, P3 |
| Basismodel van Kev-4B | `Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b` | 9,32 GB (safetensors) | P1, P3 |
| Werkmodel (voorstel, Geert kiest) | `unsloth/Qwen3.5-9B-GGUF`, `Qwen3.5-9B-Q4_K_M.gguf`, SHA-256 `03b74727…b7e8` | 5,68 GB | P1, P3 |
| Embedding, standaard (ADR 0008) | `ChristianAzinn/mxbai-embed-large-v1-gguf@3ec6d46a…`, `Q8_0` | 358 MB | P1, P3 |
| Embedding, meertalige kandidaat (voorstel) | `gpustack/bge-m3-GGUF`, `bge-m3-Q8_0.gguf`, SHA-256 `950f4a8e…7821167` | 635 MB | P3 |
| Reranker | `gpustack/bge-reranker-v2-m3-GGUF`, `bge-reranker-v2-m3-Q8_0.gguf`, SHA-256 `a43c7c9b…aaa1d3` | 636 MB | P1 |
| Whisper | `large-v3` (faster-whisper), in `HF_HOME` (`/opt/models/hf`) | meten | P1, P3 |
| Uitlijnmodel Nederlands | standaard van WhisperX voor `nl`: `jonatasgrosman/wav2vec2-large-xlsr-53-dutch` (Hugging Face), in `HF_HOME` | meten | P1, P3 |
| Uitlijnmodel Engels | standaard van WhisperX voor `en`: torchaudio-pijplijn `WAV2VEC2_ASR_BASE_960H`, in `TORCH_HOME` (`/opt/models/torch`) | meten | P1 |
| Sprekerherkenning | `pyannote/speaker-diarization-community-1` (CC-BY-4.0); in variant A al in het image | in image | P1, P3 |
| Docling-modellen | door docling-serve bij de voorbereiding gedownload naar het volume `signaal-proef_docling_artifacts` | meten | P1, P3, P5 |

Alle genoemde Hugging Face-modellen zijn niet afgeschermd ("gated: false"), behalve `pyannote/speaker-diarization-community-1` ("gated: auto"). Totaal te downloaden: ruim 30 GB aan images en modellen, plus de lokaal gebouwde images (de fork, Kev met PyTorch). P0 eist minstens 60 GB vrije ruimte in de Docker-schijf van Docker Desktop.

Downloaden gebeurt in een aparte voorbereidingsstap (`--voorbereiden`) met internet. De proef zelf draait daarna op een netwerk met `internal: true` en met `HF_HUB_OFFLINE=1`. Zo toetst elke GPU-proef ook de offline-instellingen uit FO 12.1.

## 2. Overzicht

Aanbevolen volgorde: P0, P8, P7, P6, P2, P4, P5, P1, P3. P7 moet slagen voordat P1 de socket-proxy gebruikt; P3 gebruikt de configuratie uit P1.

| Proef | Onderwerp | Beslist | GPU | Plek | Script |
| --- | --- | --- | --- | --- | --- |
| P0 | Omgevingscontrole | voorwaarde voor alle proeven | controle | hier | `p0_omgeving.sh` |
| P1 | docling-serve, WhisperX en Kev als containers onder llama-swap | ADR 0009, 0014, 0015; FO 12.2 | ja | werkplek | `p1_llama_swap_containers.sh` |
| P2 | Schema-trigger van Langflow 1.13.0, met APScheduler als terugval | ADR 0017 C; FO 5, 14 | nee | hier | `p2_langflow_schema_trigger.sh` |
| P3 | Videogeheugen per dienst | ADR 0008, 0009 (groepen); FO 12.2 | ja | werkplek | `p3_vram_per_dienst.sh` |
| P4 | open-notebook-mcp: geen intern fragment in een antwoord | ADR 0001, 0002; harde eis 1 | nee | hier | `p4_mcp_classificatiefilter.sh` |
| P5 | docling-serve geeft het videogeheugen vrij | ADR 0009, 0014; FO 12.2 | ja | werkplek | `p5_docling_vram_vrijgave.sh` |
| P6 | Feedcontrole YouTube-kanaal en Spotify-podcast | FO 5 | nee | hier | `p6_feedcontrole.sh` |
| P7 | Docker-toegang voor llama-swap: socket-proxy en Docker-client | ADR 0009 (Docker-toegang) | nee | werkplek (akkoord Docker-socket) | `p7_docker_toegang.sh` |
| P8 | Netwerkafscherming met `internal: true` | ADR 0002 F; FO 12.1 | nee | hier | `p8_netwerkafscherming.sh` |
| P9 | Opruimen van het project `signaal-proef` | — | nee | hier | `p9_opruimen.sh` |

Alle scripts staan onder `scripts/signaalsysteem/proeven/`. Gedeelde onderdelen: `lib/gemeenschappelijk.sh` (Compose-aanroep, momentopnamen, exitcodes, maskeren), `lib/vram_meter.sh` (meting 1.5), `compose.proef.yml` (één bestand met een profiel per proef) en per proef een submap `pN/` met configuratie en hulpscripts.

Toewijzing van de open punten uit de ADR's:

| Open punt | Waar |
| --- | --- |
| llama-swap laadt een upstream zonder `model`-veld (docling-serve; Kev via `/v1/systemone`) | P7 (mechanisme, zonder GPU) en P1 (echte diensten) |
| socket-proxy met `ALLOW_START`/`ALLOW_STOP` zonder `POST=1` | P7 |
| Docker-client in het llama-swap-image | P7 |
| llama-swap geeft `/v1/rerank` door | P1, deelstap 1f |
| poortpublicatie op een netwerk met `internal: true` geblokkeerd | P8 |
| `ENABLE_TRIGGERS` in Langflow 1.13 | P2, deelstap 2b |
| embeddingmodellen vergelijken (ADR 0008) | geheugen in P3; kwaliteit niet als proef in fase 1, zie paragraaf 4 |

## 3. Proeven

### P0. Omgevingscontrole

**Doel.** Vaststellen dat de machine geschikt is, en de beginstand van Docker vastleggen.

**Aanpak.**

1. Besturingssysteem en shell: Windows 11, Git Bash of WSL2.
2. `docker info`: Docker Desktop, Linux-containers, serverversie, vrije ruimte (`docker system df`).
3. GPU in een container: `docker run --rm --gpus all ghcr.io/mostlygeek/llama-swap:v268-cuda13-b11977 nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader`, plus de regel "CUDA Version" uit `nvidia-smi`.
4. Draaiende containers buiten `signaal-proef` met een GPU-reservering, en `ollama ps` op de host (alleen lezen).
5. Momentopname van containers, volumes, netwerken en images.

**Benodigdheden.** Het llama-swap-image (ook nodig voor P1, P3 en P7). GPU: alleen ter controle.

**Succescriterium.**

- GPU-naam bevat `RTX 5070 Ti`; `memory.total` ≥ 15 900 MiB; `compute_cap` = 12.0 (Blackwell).
- "CUDA Version" ≥ 13.0. Dat is nodig voor het image `cuda13`: volgens de README dekt het image `unified-cuda` (CUDA 12) alleen Pascal tot en met Ada, en `cuda13` ook Blackwell.
- Ten minste 60 GB vrij.
- Voor GPU-proeven bovendien: geen GPU-container buiten het project actief en `ollama ps` leeg. Zo niet, dan exitcode 2 met de lijst, en wordt alleen het niet-GPU-deel als geslaagd gemeld.

**Verwachte uitkomst.** Geslaagd, op de bestaande containers van de eigen stack na als die draaien.

**Script.** `scripts/signaalsysteem/proeven/p0_omgeving.sh`. Elk ander script roept P0 eerst aan.

**Plek.** Hier.

**Uitkomst.**

| Datum | Commit | Exitcode | Meetwaarden | Conclusie |
| --- | --- | --- | --- | --- |
| 29-10-2026 | b31c4857 | 0 | Windows 11 (10.0.22621), Git Bash; Docker Desktop 29.7.2, Linux-containers; NVIDIA GeForce RTX 5070 Ti, 16 303 MiB, compute_cap 12.0, driver 610.88; `CUDA UMD Version: 13.3`; 674 GB vrij in de Docker-VM, 737 GB op E:; geen GPU-container buiten het project, `ollama ps` leeg (de bestaande stack was door Geert gestopt). GPU in een container gezien via `python:3.12.16-slim-bookworm` met `--gpus all`: het llama-swap-image is niet opgehaald (geen akkoord voor deze download) | Geslaagd. Alle criteria gehaald. Voorwaarde voor GPU-proeven voldaan op het moment van meten; P1, P3 en P5 controleren dat opnieuw. Eerste run gaf exitcode 1 door een fout in het script (de driver meldt `CUDA UMD Version`), hersteld. |
| 30-10-2026 | 360937d9 | 0 | `p0_omgeving.sh --gpu` als controle vóór de GPU-proeven: RTX 5070 Ti, 16 303 MiB, compute_cap 12.0, driver 610.88, `CUDA UMD Version: 13.3`; 653 GB vrij in de Docker-VM; geen GPU-container buiten het project (de zeven containers van `open-notebook` gestopt); `ollama ps` leeg; `nvidia-smi.exe` 0 MiB in gebruik | Geslaagd. GPU vrij voor P1, P5 en P3. |

### P1. docling-serve, WhisperX en Kev als containers onder llama-swap

**Doel.** Aantonen dat llama-swap de drie zware diensten als container start, op volgorde in de rij zet en weer stopt, via de socket-proxy uit P7, en dat de diensten via llama-swap bereikbaar zijn. Daarnaast: de keuze van de WhisperX-container en de doorgifte van `/v1/rerank`.

**Beslist.** ADR 0009: Docker-toegang optie 2 of terugval, en de beginindeling van de groepen. ADR 0014: welke WhisperX-container. ADR 0015: reranker onder llama-swap (5b) of als CPU-dienst (5c). FO 12.2: "of docling-serve en WhisperX zo onder llama-swap passen".

**Aanpak.**

1. **Voorbereiden (met internet).**
   - Images ophalen; Kev bouwen (`p1/kev.Dockerfile`: tag `kev-1.0`, `uv sync --extra serve`, PyTorch met CUDA 12.8 of hoger). Kev heeft geen eigen Docker-image en staat niet op PyPI (`pip install kev` installeert een ander pakket, volgens de README).
   - Het afgeleide image `signaal-proef/llama-swap:v268-cuda13-b11977-dockercli` bouwen als P7 laat zien dat de Docker-client ontbreekt.
   - Modellen downloaden naar volumes: GGUF-bestanden met controle van de SHA-256; Kev en het basismodel naar `HF_HOME` van Kev; Whisper en het uitlijnmodel naar `/opt/models/hf` van variant A (de AWS-documentatie zegt dat je daar een volume moet koppelen, niet over `/opt/models`); de Docling-modellen door één keer docling-serve te starten met `DOCLING_SERVE_ARTIFACTS_PATH` op het volume.
   - `docker compose create` voor `docling-serve`, `whisperx` en `kev`: aangemaakt, niet gestart, zonder `restart`.
2. **Configuratie** `p1/llama-swap.yaml`:
   - `werkmodel`, `embed` en `rerank` als llama-server-processen in de llama-swap-container (`--embeddings`, of `--reranking` voor de reranker);
   - `docling`, `whisperx` en `kev-4b` met:
     - een `cmd` die de container start met `docker start signaal-proef-<dienst>-1` en daarna blijft lopen zolang de gezondheidscontrole van de dienst slaagt;
     - `cmdStop: docker stop signaal-proef-<dienst>-1`;
     - `proxy` naar de containernaam;
     - `checkEndpoint` (docling `/health`, WhisperX-A `/ping`, Kev zie stap 1d).

   De werkwijze met wachten is nodig omdat `docker start -a` en `docker wait` paden onder `/containers` gebruiken die de proxy dichthoudt (P7).

   Groepen: `zwaar` (`swap: true`, `exclusive: true`) met werkmodel, Kev, docling en WhisperX; `licht` (`persistent: true`) met embed en rerank. `ttl` 60 s voor de zware diensten. Netwerken: `verwerking` (`internal: true`) voor alle diensten, `beheer` (`internal: true`) alleen voor llama-swap en de socket-proxy. Geen gepubliceerde poorten; de testclient draait op `verwerking`.
3. **Deelstappen.** Elke aanroep gaat van de testclient naar `http://llama-swap:8080`.
   - **1a, docling-serve zonder `model`-veld.** `POST /upstream/docling/v1/convert/file/async` met D2 (`files`, `to_formats=json`, `ocr_lang=["nl","en"]`), volgen via `/upstream/docling/v1/status/poll/{task_id}`, ophalen via `/upstream/docling/v1/result/{task_id}`. Daarna hetzelfde met D1. Aanwijzing dat `/upstream` laadt: de configuratievoorbeelden noemen `upstream.ignorePaths` om "swaps te voorkomen". Uitdrukkelijk bevestigd is dat niet.
   - **1b, WhisperX.** `POST /v1/audio/transcriptions` (multipart) met `model=whisperx`, `language=nl`, `diarize=true`, `timestamp_granularities[]=word`, `response_format=verbose_json`, met audio uit D4. Volgens de AWS-documentatie is `model` geen veld van variant A: llama-swap gebruikt het voor de route, de server negeert het. Dat is te bevestigen.
   - **1c, WhisperX-keuze.** De criteria uit ADR 0014 per variant afvinken: OpenAI-compatibel eindpunt, sprekers en woordtijden, vaste imageversie, offline met `HF_HUB_OFFLINE=1`, licentie. Variant B alleen proberen als A op een criterium afvalt. B vraagt een Hugging Face-token en akkoord met de voorwaarden van pyannote.
   - **1d, Kev zonder OpenAI-eindpunt.** `POST /upstream/kev-4b/v1/systemone` met één `choice`- en één `noul`-vraag op een synthetische zin. Kev vraagt met `KEV_API_KEY` een sleutel op `/v1/*`, en heeft volgens de README geen apart gezondheidseindpunt. `checkEndpoint` kan geen header meesturen en verwacht HTTP 200. Vast te stellen: of er een pad zonder sleutel is dat 200 geeft. Zo niet, dan `checkEndpoint: none` plus een gereedheidscontrole in de `cmd`, en dat als bevinding vastleggen. Kev start met `--host 0.0.0.0` (standaard luistert hij op 127.0.0.1).
   - **1e, wisselen en wachtrij.** Achter elkaar docling (D1), Kev, WhisperX, werkmodel (`/v1/chat/completions`), elk twee keer, met daartussen twee gelijktijdige aanvragen voor verschillende zware diensten. Op de host `docker events --filter label=com.docker.compose.project=signaal-proef` volgen (alleen lezen), en het videogeheugen meten.
   - **1f, `/v1/rerank`.** `POST /v1/rerank` met `model=rerank`, één vraag en vijf documenten. De README noemt `v1/rerank`, `v1/reranking` en `/rerank` als doorgegeven paden.
   - **1g, ontladen en herstel.** Na `ttl` + 15 s moeten de zware containers gestopt zijn. Daarna van buitenaf `docker stop` op de Kev-container van de proef, en een nieuwe Kev-aanvraag doen.
   - **1h, Nederlands en Engels (aanvulling Geert, 29-10).** Transcriptie moet lokaal werken voor beide talen. `POST /v1/audio/transcriptions` met D7 (nl) en D8 (en), zonder `language`, zodat de taaldetectie meetelt, met `timestamp_granularities[]=word` en `response_format=verbose_json`. De uitlijnmodellen voor beide talen (zie 1.7) worden bij het voorbereiden opgehaald door D7 en D8 één keer met internet te transcriberen (`whisperx`-CLI in de container `whisperx-voorbereiden`); de proef zelf draait offline.
   - **1h, gebruiksmeldingen van variant A.** De container draait met `OPT_OUT_TRACKING=true`. Het script controleert met `docker inspect` dat de variabele staat, en vanuit de container dat `https://example.org` en `https://aws.amazon.com` op het netwerk `verwerking` (`internal: true`) geen verbinding geven; logregels met "track", "telemetry" of "usage" worden geteld. Wat de documentatie zegt: de pagina "Monitoring and Usage Tracking" van AWS Deep Learning Containers meldt dat gebruiksregistratie standaard aan staat, dat ze instance-id, frameworks en versies, containertype en Python-versie verzamelt en geen opdrachten, en dat `OPT_OUT_TRACKING=true` haar uitzet. Een eindpunt noemt de pagina niet. De WhisperX-pagina's (configuratie, EC2-uitrol) noemen tracking en `OPT_OUT_TRACKING` niet; of het WhisperX-image de registratie bevat, is dus niet gedocumenteerd en blijkt uit deze controle.

**Benodigdheden.** Alle images en modellen uit 1.7 voor P1; D1, D2, D4, D7 en D8; GPU ja; Docker-socket via de socket-proxy (akkoord Geert); het resultaat van P7.

**Succescriterium.**

| Nr | Criterium | Meting |
| --- | --- | --- |
| K1 | 1a: HTTP 200 met `task_id`, takenstatus `success`, JSON met ≥ 1 tabel en de tekst "weerstandsvermogen" | HTTP-status, telling in JSON |
| K2 | 1a, 1b en 1d: de container staat vóór de aanvraag op `exited` en tijdens de aanvraag op `running` | `docker inspect -f '{{.State.Status}}'` |
| K3 | 1b: HTTP 200, ≥ 2 verschillende sprekerlabels, woordtijden aanwezig, ≥ 1 segment | telling in JSON |
| K4 | 1d: HTTP 200, per `choice`-vraag tellen de kansen op tot 1 ± 0,01 | rekenkundige controle |
| K5 | 1e: nul tijdvakken waarin twee zware diensten tegelijk draaien; piek `memory.used` < `memory.total` − 512 MiB; geen geheugenfout (CUDA out of memory) in de logs | `docker events`, CSV, `grep -c` |
| K6 | 1f: HTTP 200 en 5 resultaten met een score | telling |
| K7 | 1g: zware containers `exited` binnen `ttl` + 15 s; na de externe stop geeft de volgende Kev-aanvraag HTTP 200 | `docker inspect`, HTTP-status |
| K8 | geen proefcontainer op `verwerking` bereikt internet tijdens de proef | P8-controle vanuit `whisperx` en `kev` |
| K9 | 1h, per taal (nl, en): HTTP 200, een niet-leeg transcript met woordtijden, en de gedetecteerde taal is `nl` respectievelijk `en` | HTTP-status, telling, veld `language` |
| K10 | 1h: `OPT_OUT_TRACKING=true` staat in de container, en de container krijgt op `verwerking` geen uitgaande verbinding | `docker inspect`, verbindingsproef |

**Verwachte uitkomst.**

- K1, K3 en K6 slagen waarschijnlijk.
- K2 en K7 hangen af van P7: hoe llama-swap een container volgt die met `docker start` is gestart, is nergens beschreven.
- K4 hangt af van de gezondheidscontrole van Kev (1d).
- K5: groep `zwaar` met `exclusive` zou overlap moeten voorkomen. De wachtrij voor gelijktijdige aanvragen is beschreven, maar niet beproefd.
- Kev-4B is een LoRA-adapter op `Qwen3.5-4B-Base` met 9,3 GB aan gewichten in bf16. Of hij samen met het embeddingmodel binnen 16 GB past, volgt uit P3.
- K9: waarschijnlijk geslaagd; large-v3 herkent beide talen. K10: de variabele staat; of het image zonder internet tracking probeert, blijkt uit de logregels.

**Script.** `scripts/signaalsysteem/proeven/p1_llama_swap_containers.sh`, met `p1/llama-swap.yaml`, `p1/kev.Dockerfile`, `p1/llama-swap-dockercli.Dockerfile` en `p1/wacht_op_dienst.sh` (de `cmd` die blijft lopen zolang de dienst gezond is). De diensten staan in `compose.proef.yml`, profiel `p1`.

**Plek.** Uit te voeren op de werkplek: GPU exclusief, akkoord voor de Docker-socket, bestaande GPU-containers en Ollama-modellen gestopt door Geert. Alleen voor variant B: `HF_TOKEN` in de shell van Geert.

**Uitkomst.**

| Datum | Commit | Exitcode | Meetwaarden | Conclusie |
| --- | --- | --- | --- | --- |
| 29-10-2026 | 013704dd | niet uitgevoerd | — | Niet uitgevoerd: vraagt GPU exclusief, ruim 30 GB aan downloads en de Docker-socket in `socket-proxy`; Geert beslist nog (opdracht 3b). Voorwaarde: P7 geslaagd. Script geschreven, `bash -n` geslaagd. |

### P2. Schema-trigger van Langflow 1.13.0, met APScheduler als terugval

**Doel.** Vaststellen of de schema-trigger in Langflow 1.13.0 een flow op tijd start, ook na een herstart, en hoe hij aan te zetten is. Anders aantonen dat de terugval met APScheduler werkt.

**Beslist.** ADR 0017 onderdeel C (schema-trigger of container `planner`) en de vaste Langflow-versie. FO 5 en FO 14 (planning).

**Gecontroleerd vooraf.**

- PR #14958 is op 18-09-2026 samengevoegd in `release-1.13.0`, en 1.13.0 is uitgebracht op 22-10.
- Volgens de PR is `ENABLE_TRIGGERS` een vlag die standaard uit staat en de categorie `triggers` verwijdert in `applyFeatureFlagFilters`. Dat is filtering in de frontend, en de vlag staat niet in de documentatie van de omgevingsvariabelen.
- Een trigger die bij het opslaan ontstaat, krijgt status `pending`. Volgens de PR kan de interface hem pas aanzetten na TRG-7. De backend is ongewijzigd: de triggers-API (`/api/v1/triggers`, `POST /triggers/{id}/enable`), de reconciliatie en de dispatcher werken voor triggers die via de API zijn aangemaakt en voor geïmporteerde flows.
- Inhaalbeleid: `coalesce` (één keer starten, gemiste momenten in de payload) of `skip`. Beide reiken niet verder terug dan een replayvenster, waarvan de lengte niet is gedocumenteerd.

**Aanpak.**

1. Langflow 1.13.0 op een netwerk met `internal: true`, SQLite in een proefvolume. Instellingen:
   - `DO_NOT_TRACK=true`;
   - `LANGFLOW_MCP_SERVER_ENABLED=false` en `LANGFLOW_ADD_PROJECTS_TO_MCP_SERVERS=false`;
   - `LANGFLOW_AUTO_LOGIN=false`, met `LANGFLOW_SUPERUSER`, `LANGFLOW_SUPERUSER_PASSWORD` en `LANGFLOW_SECRET_KEY` per run gegenereerd (1.3);
   - `LANGFLOW_COMPONENTS_PATH` naar `p2/components/`, met een component `TickLogger` die per uitvoering een JSON-regel schrijft naar een proefvolume (gepland moment, werkelijke tijd, payload).
2. **2a.** Inloggen via `POST /api/v1/login` en een API-sleutel maken via `POST /api/v1/api_key/`.
3. **2b, `ENABLE_TRIGGERS`.**
   - Drie keer starten: zonder vlag, met `ENABLE_TRIGGERS=true` en met `LANGFLOW_ENABLE_TRIGGERS=true`.
   - Per keer vastleggen: of de schema-component in `GET /api/v1/all` staat; wat `GET /api/v1/config` meldt over feature flags; hoe vaak `ENABLE_TRIGGERS` voorkomt in de meegeleverde frontendbundel, en of die bundel de vlag uit een runtimeconfiguratie leest of bij het bouwen vastlegt.
4. **2c.** Een flow `schema → TickLogger` maken. Het knooppunt komt uit het sjabloon in `GET /api/v1/all`, niet uit een geraden JSON. Instellingen: cron `* * * * *`, tijdzone `Europe/Amsterdam`, inhaalbeleid `coalesce`. Uploaden met `POST /api/v1/flows/upload/`, daarna `GET /api/v1/triggers` en `POST /api/v1/triggers/{id}/enable`.
5. **2d.** Vier minuten laten lopen.
6. **2e, herstart.** Langflow 150 s stoppen (minstens twee gemiste momenten), weer starten, twee minuten wachten. Herhalen met `skip`.
7. **2f, tijdzone.** Een tweede trigger met `0 8 * * 1-5`. Als de API het volgende moment toont: dat moet 07:00 UTC zijn (wintertijd na 25-10-2026).
8. **2g, terugval.** Python-container met `APScheduler==3.11.3`, `CronTrigger` elke minuut, `coalesce=True`, `misfire_grace_time=300` en een SQLite-jobstore in een proefvolume. Die start dezelfde flow zonder triggerknooppunt via `POST /api/v1/run/{flow_id}` met `x-api-key`. Ook de herstartproef uit stap 2e. Stap 2g draait altijd, zodat de terugval ook bewezen is als de schema-trigger slaagt.

**Benodigdheden.** `langflowai/langflow:1.13.0`, `python:3.12.16-slim-bookworm` met APScheduler, `curlimages/curl:8.23.0`. Geen GPU, geen modellen, geen internet na het ophalen van de images.

**Succescriterium.**

| Nr | Criterium | Meting |
| --- | --- | --- |
| K1 | 2b vastgelegd: waar de vlag werkt (runtime, bouwtijd of niet aanwezig) | vaststelling, geen geslaagd of mislukt |
| K2 | 2c: trigger aangemaakt; `enable` geeft HTTP 2xx; status daarna `active` | HTTP-status, veld `state` |
| K3 | 2d: ≥ 3 regels in `ticks.jsonl`; elk gepland moment uniek; werkelijke tijd − gepland moment ≤ 10 s | telling, tijdsverschil |
| K4 | 2e met `coalesce`: precies 1 inhaalrun met ≥ 2 gemiste momenten in de payload, of vastgelegd dat het replayvenster korter is; met `skip`: 0 inhaalruns | telling |
| K5 | 2g: ≥ 3 runs met HTTP 200, elk ≤ 10 s na het geplande moment; na de herstart precies 1 inhaalrun | telling, HTTP-status |

Beslisregel voor ADR 0017 C:

- K2 tot en met K4 geslaagd: de schema-trigger, via de API aangezet als de interface het niet kan;
- anders: de container `planner` met APScheduler, mits K5 slaagt.

**Verwachte uitkomst.**

- K1: waarschijnlijk alleen een frontendvlag zonder runtimeschakelaar.
- K2 en K3: waarschijnlijk geslaagd via de API, omdat de backend volgens de PR werkt.
- K4: onzeker, omdat het replayvenster niet is gedocumenteerd.
- K5: geslaagd.

**Script.** `scripts/signaalsysteem/proeven/p2_langflow_schema_trigger.sh`, met `p2/components/tick_logger.py` en `p2/apscheduler_terugval.py`. Compose-profiel `p2`.

**Plek.** Hier.

**Uitkomst.**

| Datum | Commit | Exitcode | Meetwaarden | Conclusie |
| --- | --- | --- | --- | --- |
| 30-10-2026 | 013704dd | 0 | K1 (vastgelegd): in alle drie de starts (zonder vlag, `ENABLE_TRIGGERS=true`, `LANGFLOW_ENABLE_TRIGGERS=true`) staat `ScheduleTrigger` in `/api/v1/all` (categorie `triggers`); de meegeleverde frontendbundel bevat de tekst `ENABLE_TRIGGERS` nergens (0 bestanden); `/api/v1/config` kent geen triggervlag; de backend kent ook geen `LANGFLOW_FEATURE_`-vlag voor triggers. K2: trigger ontstaat bij het opslaan van de flow met status `pending`; `POST /api/v1/triggers/{id}/enable` HTTP 200, daarna `active`. K3: 4 ticks in 240 s, elk gepland moment uniek, grootste vertraging 1,0 s. K4: coalesce na 150 s stilstand precies 1 inhaalrun met 2 gemiste momenten in `missed_ticks` (gepland 23:48:00 UTC, uitgevoerd 22 s later, na het opstarten); met skip 0 inhaalruns. Replayvenster: `trigger_replay_window_days`, standaard 7 dagen (instelling in `lfx`, niet in de documentatie). 2f: `next_fire_at` van `0 8 * * 1-5` in Europe/Amsterdam is 2026-10-30T07:00:00 (UTC, wintertijd). K5: terugval 4 runs met HTTP 200, grootste vertraging 0,4 s; na 150 s stilstand precies 1 inhaalrun. Langflow probeerde vanaf `verwerking` 21 keer een uitgaande verbinding (LiteLLM haalt `model_prices_and_context_window.json` op GitHub), geblokkeerd door het netwerk | Geslaagd. Beslisregel ADR 0017 C: de schema-trigger van Langflow 1.13.0, via de API aangezet (de interface kan dat niet; `ENABLE_TRIGGERS` bestaat in deze versie niet als vlag). De terugval met APScheduler werkt ook. Voor ADR 0017: Langflow doet uit zichzelf uitgaande aanvragen (LiteLLM-kostentabel); op `verwerking` is dat afgevangen. Eerdere runs faalden op fouten in het script (Fernet-sleutel, schrijfrechten op het tickvolume, telling van de inhaalrun van APScheduler); hersteld en volledig opnieuw gedraaid. |

### P3. Videogeheugen per dienst op de RTX 5070 Ti

**Doel.** Per dienst het videogeheugen meten, als basis voor de groepsindeling in FO 12.2 en ADR 0009 en voor de keuze van het embeddingmodel in ADR 0008.

**Beslist.** De groepen of `matrix` in de llama-swap-configuratie; of Kev-4B past (besloten door Geert, maar onder voorbehoud van het budget, ADR 0009); de geheugenpost van het embeddingmodel.

**Aanpak.**

1. P0 met GPU-voorwaarden. Configuratie uit P1, maar de diensten worden rechtstreeks gestart (`docker compose start <dienst>` of een eigen llama-server). Zo meet P3 de dienst zelf en niet het wisselgedrag.
2. Per dienst, drie keer:
   - nulmeting van 30 s;
   - start en eerste aanvraag; piek en rustwaarde 30 s na de aanvraag meten;
   - stop; de waarde 15 s na de stop meten.

   | Dienst | Belasting |
   | --- | --- |
   | `embed` (mxbai Q8_0) | 256 fragmenten van 512 tokens, batch van 32 |
   | `embed` (bge-m3 Q8_0) | idem |
   | `werkmodel` (Qwen3.5-9B Q4_K_M) | context 8 192 en 32 768, één aanvraag van 2 000 tokens invoer en 500 uitvoer |
   | `kev-4b` (bf16) | 50 vragen in één run, met de batchinstelling die Kev standaard gebruikt |
   | `docling-serve` | D1 (OCR (Optical Character Recognition, tekstherkenning) aan, standaard) en D2 met `force_ocr=true`; `DOCLING_SERVE_ENG_LOC_NUM_WORKERS=1` (ADR 0014) |
   | `whisperx` (variant uit P1) | aflevering D4, `diarize=true`, batch 16 |

3. Combinaties: `embed` blijvend geladen plus elk van de zware diensten afzonderlijk; daarnaast `docling-serve` plus `whisperx` (de enige zware combinatie die mogelijk past).
4. Uitvoer: `p3_vram.csv` met dienst, herhaling, nulmeting, piek, rust en na stop (MiB), en een afgeleid voorstel voor de groepen.

**Benodigdheden.** Als P1, plus het model bge-m3. GPU ja, exclusief.

**Succescriterium.**

- K1: voor elke dienst drie herhalingen; spreiding van de piek ≤ 10 %.
- K2: elke combinatie gemeten, met vermelding van een eventuele geheugenfout.
- K3: beslisregel toegepast. Een combinatie mag samen draaien als nulmeting + som van de pieken ≤ `memory.total` − 1 024 MiB.

Het resultaat is een tabel; de proef "slaagt" als de metingen volledig zijn, niet als een bepaalde indeling eruit komt.

**Verwachte uitkomst.** Indicatief, niet gemeten:

- het embeddingmodel past naast elke zware dienst;
- Kev-4B (9,3 GB gewichten in bf16) en het werkmodel (5,7 GB gewichten plus KV-cache) passen niet samen;
- docling-serve en WhisperX misschien wel.

Geeft Kev-4B samen met het embeddingmodel een geheugenfout, dan legt de bouwagent dat voor aan Geert. Terugval volgens ADR 0009: Kev-0,8B.

**Script.** `scripts/signaalsysteem/proeven/p3_vram_per_dienst.sh`, met `lib/vram_meter.sh`. Compose-profiel `p1`.

**Plek.** Uit te voeren op de werkplek: GPU exclusief; Geert sluit toepassingen die veel videogeheugen gebruiken en stopt bestaande GPU-containers en Ollama-modellen.

**Uitkomst.**

| Datum | Commit | Exitcode | Meetwaarden | Conclusie |
| --- | --- | --- | --- | --- |
| 29-10-2026 | 013704dd | niet uitgevoerd | — | Niet uitgevoerd: vraagt GPU exclusief en de voorbereiding van P1; Geert beslist nog. Script geschreven, `bash -n` geslaagd. |

### P4. open-notebook-mcp met het classificatiefilter: geen intern fragment in een antwoord

**Doel.** Met een echte client van open-notebook-mcp aantonen dat geen intern fragment, geen interne bron en geen interne notitie in een antwoord verschijnt (harde eis 1). Daarnaast vaststellen of de antwoorden van de fork passen bij wat open-notebook-mcp verwacht (aanname A2).

**Beslist.** De uitvoering van ADR 0001 en 0002 (filter in de repositorylaag, sleutel `lezen_extern`); de toets van fase 3 (FO 15). Eerste uitvoering in fase 1 is een nulmeting: volgens verschilanalyse 3.8 heeft de fork het filter nog niet. Een lek in de nulmeting bewijst dat de proef een lek kan zien. Hetzelfde script is na de bouwtaken voor ADR 0001 en 0002 de acceptatietest.

**Aanpak.**

1. **Testdatabase.** `surrealdb/surrealdb:v2.7.0` met `start memory`, zonder volume, alleen op het netwerk `verwerking` (`internal: true`); namespace en database `signaal_proef`; inloggegevens per run gegenereerd. Er is dus geen verbinding met `surreal_data` of `notebook_data_live`.
2. **API van de fork.** Image bouwen uit de huidige commit (`signaal-proef/open-notebook:<commit>`; bestaande images en containers blijven ongemoeid). Starten zonder `docker.env`, met alleen:
   - `SURREAL_URL=ws://surrealdb:8000`, `SURREAL_USER`, `SURREAL_PASSWORD`, `SURREAL_NAMESPACE` en `SURREAL_DATABASE`;
   - `OPEN_NOTEBOOK_PASSWORD`, gegenereerd. Zolang ADR 0002 niet is gebouwd, is dit het gedeelde wachtwoord; daarna een sleutel met `lezen_extern`.

   De migraties draaien bij het opstarten (`apps/app-main/src/app_main/api/app.py:131-137`). Het script wacht op `/health`.
3. **Vullen** met `p4/vul_testdatabase.py` (Python-container op `verwerking`, SurrealQL via de HTTP-interface van de proefdatabase):
   - twee notebooks: "Proef extern" en "Proef gemengd";
   - tien externe en tien interne bronnen over een verzonnen gemeente "Proefstad", elk met drie fragmenten. De teksten komen uit een vast sjabloon met gewone zoekwoorden (fietspaden, begroting, weerstandsvermogen). Elke titel en elk fragment draagt een uniek kenmerk: `PRF-INT-<8 hex>` bij intern, `PRF-EXT-<8 hex>` bij extern;
   - interne bronnen krijgen `private = true` (huidig schema) én `classification = 'intern'` (schema van ADR 0001). Weigert het schema het onbekende veld, dan legt het script dat vast en gaat het verder met `private`;
   - één interne en één externe notitie, en één `source_insight` bij een interne bron, omdat de zoekfuncties die meenemen (verschilanalyse 3.9);
   - de lijst met kenmerken en id's gaat naar de runmap als referentie.
4. **open-notebook-mcp.** Container op basis van `python:3.12.16-slim-bookworm` met `open-notebook-mcp==0.3.0` (bouwen met internet, draaien op `verwerking`), met `MCP_TRANSPORT=streamable-http`, `HOST=0.0.0.0`, `PORT=8000`, `OPEN_NOTEBOOK_URL=http://api:5055` en het wachtwoord uit stap 2.
5. **Lektest** `p4/mcp_lektest.py` (client `mcp==1.30.0`, transport streamable HTTP op `http://mcp:8000/mcp`):
   - `list_tools`;
   - elke leestool aanroepen: `list_notebooks`, `get_notebook` per id, `list_sources`, `get_source` per id (ook de interne id's rechtstreeks), `list_notes`, `get_note` per id, en `search` met `type=text` voor elk kenmerk en voor de gewone zoekwoorden (`limit=50`);
   - `search` met `type=vector` en `hybrid`: dat is de standaard van de zoektool (`type="vector"` in `server.py`). Zonder embeddingmodel geeft dat een fout; die wordt als "niet uitvoerbaar" vastgelegd;
   - `ask_simple`, `get_settings` en `list_models`: alleen de HTTP-status;
   - schrijftools (`create_note`, `create_source`) met een synthetische tekst, waarbij het script het aantal records vóór en na telt.

   Alle antwoordteksten worden samengevoegd en met een reguliere expressie geteld.
6. **Rechtstreeks.** Een aanroep van de API zonder wachtwoord moet HTTP 401 geven.
7. **4c, optioneel, na P1 en de bouwtaak voor ADR 0006.** Dezelfde test met vector- en hybride zoekopdrachten, met het embeddingmodel via llama-swap.

**Benodigdheden.** Geen GPU, geen modellen, geen echte documenten. Internet alleen voor het bouwen van de images.

**Succescriterium.**

| Nr | Criterium | Meting |
| --- | --- | --- |
| K1 | aantal treffers `PRF-INT-` in alle antwoorden van open-notebook-mcp = 0 | telling |
| K2 | positieve controle: ≥ 1 treffer `PRF-EXT-` in `list_sources`, in `get_source` en in `search`. Ontbreekt die, dan is de run ongeldig (exitcode 2), omdat een leeg antwoord K1 anders vanzelf haalt | telling |
| K3 | `get_source` met een intern id geeft een fout (na ADR 0002: HTTP 404) en geen inhoud | status, telling |
| K4 | schrijftools en `ask_simple` veranderen niets: aantal records vóór = na (na ADR 0002: HTTP 403) | telling |
| K5 | aanname A2: aantal leestools met een fout bij het verwerken van het antwoord = 0 | telling |
| K6 | API zonder wachtwoord of sleutel geeft HTTP 401 | HTTP-status |

**Verwachte uitkomst.**

- In de huidige fork faalt K1 (exitcode 1): `list_sources`, `get_source` en `search` tonen interne kenmerken, omdat geen leesroute filtert (verschilanalyse 3.8).
- K4 faalt: schrijven lukt.
- K6 slaagt alleen omdat het wachtwoord is gezet.
- K5 is onbekend.
- Na de bouwtaken voor ADR 0001 en 0002 moeten alle criteria slagen.

**Script.** `scripts/signaalsysteem/proeven/p4_mcp_classificatiefilter.sh`, met `p4/vul_testdatabase.py`, `p4/mcp_lektest.py` en `p4/mcp.Dockerfile`. Compose-profiel `p4`.

**Plek.** Hier.

**Uitkomst.**

| Datum | Commit | Exitcode | Meetwaarden | Conclusie |
| --- | --- | --- | --- | --- |
| 30-10-2026 | 921c406b (fork f7bb6d55) | 1 | K1: 38 verschillende `PRF-INT-`-kenmerken in antwoorden van open-notebook-mcp, via `list_sources`, `get_source` (ook met interne id's), `list_notes`, `get_note` en `search` (`text` en `hybrid`). K2: positieve controle geldig (`PRF-EXT-` in `list_sources` 20, `get_source` 40, `search` 30). K3: `get_source` met een intern id gaf in 9 van 10 gevallen de inhoud; 1 keer een fout, maar alleen omdat het antwoordmodel `SourceResponse` de bron met een `source_insight` niet kon valideren (HTTP 500). K4: aantal records gelijk, maar alleen doordat de schrijfacties faalden: `create_note` HTTP 500, `create_source` HTTP 422 (veld `type` ontbreekt: open-notebook-mcp stuurt JSON, de fork verwacht een formulier), `ask_simple` faalt op `No module named 'ai_prompter'`. K5: 0 leesaanroepen met een verwerkingsfout. K6: API zonder wachtwoord HTTP 401. Vector-zoeken niet uitvoerbaar: geen embeddingmodel ingesteld (HTTP 400); hybride zoeken valt terug op tekst en lekt mee. Het veld `classification` wordt door het schema (SCHEMAFULL) stil weggelaten; de proef werkte met `private = true`. Officieel startpunt van open-notebook-mcp 0.3.0 faalt met `mcp==1.30.0`: `FastMCP.run() got an unexpected keyword argument 'host'` | Mislukt, zoals verwacht als nulmeting: de fork filtert in geen enkele leesroute op `private` of classificatie (harde eis 1 niet gehaald). De proef ziet het lek, dus de test werkt. Bevindingen voor A2 en ADR 0002: (1) open-notebook-mcp 0.3.0 start niet met mcp 1.30.0 zonder eigen startbestand; (2) `create_source` past niet bij het importeindpunt van de fork; (3) K4 toetst nu alleen dat schrijven faalt, niet dat het geweigerd wordt (na ADR 0002: HTTP 403); (4) `SourceResponse` faalt op een bron met inzicht. |

### P5. docling-serve geeft het videogeheugen vrij

**Doel.** Vaststellen of docling-serve het videogeheugen na een taak vrijgeeft, en of stoppen van de container het volledig vrijgeeft (FO 12.2: "stoppen na een bundel taken").

**Beslist.** ADR 0009 en 0014: of stoppen de enige route is, en daarmee of Docker-toegang voor llama-swap nodig is. De documentatie van docling-serve kent geen instelling om modellen te ontladen; wel `DOCLING_SERVE_LOAD_MODELS_AT_BOOT` (standaard aan) en `DOCLING_SERVE_OPTIONS_CACHE_SIZE` (standaard 2; het aantal converters dat met hun modellen in de cache blijft).

**Aanpak.**

1. P0 met GPU-voorwaarden. docling-serve `cu128:v1.38.2` op `verwerking`, met:
   - `DOCLING_SERVE_ARTIFACTS_PATH` op het volume uit P1 (alleen-lezen);
   - `DOCLING_SERVE_ENABLE_REMOTE_SERVICES=false` en `HF_HUB_OFFLINE=1`;
   - `DOCLING_SERVE_ENG_LOC_NUM_WORKERS=1`;
   - `DOCLING_SERVE_API_KEY` per run gegenereerd.
2. Meten:
   - na de nulmeting de container starten en na `/health` meten (M1, modellen bij het opstarten);
   - D1 via `/v1/convert/file/async`, de piek meten (M2) en 60 s na `success` meten (M3);
   - D2 met `force_ocr=true` (M4, OCR-modellen erbij) en 60 s later (M5).
3. Nagaan in `GET /openapi.json` of er een eindpunt is om de cache te legen (bijvoorbeeld onder `/v1/clear/`). Zo ja: aanroepen en na 15 s meten (M6). Dat eindpunt staat niet in de geraadpleegde documentatie.
4. `docker stop` en na 15 s meten (M7).
5. Herhalen met `DOCLING_SERVE_LOAD_MODELS_AT_BOOT=false` (M1′) en `DOCLING_SERVE_OPTIONS_CACHE_SIZE=1`.

**Benodigdheden.** docling-serve-image en Docling-modellen, D1 en D2, GPU ja.

**Succescriterium.**

- K1: beide conversies hebben status `success`, en de JSON van D1 bevat de tekst "weerstandsvermogen".
- K2: M7 − nulmeting ≤ 256 MiB. Dat is het criterium voor de route "container stoppen".
- K3: M3, M5 en, als het eindpunt bestaat, M6 vastgelegd. Is M3 − nulmeting ≤ 256 MiB, dan geeft docling-serve het geheugen zelf vrij en is stoppen niet nodig. Dat is een vaststelling, geen geslaagd of mislukt.

**Verwachte uitkomst.** K1 en K2 slagen. M3 en M5 blijven hoog: de converters blijven met hun modellen in de cache. Stoppen is dus de route, zoals ADR 0009 aanneemt.

**Script.** `scripts/signaalsysteem/proeven/p5_docling_vram_vrijgave.sh`, met `lib/vram_meter.sh`. Compose-profiel `p5`.

**Plek.** Uit te voeren op de werkplek: GPU exclusief; bestaande GPU-containers en Ollama-modellen gestopt door Geert.

**Uitkomst.**

| Datum | Commit | Exitcode | Meetwaarden | Conclusie |
| --- | --- | --- | --- | --- |
| 29-10-2026 | 013704dd | niet uitgevoerd | — | Niet uitgevoerd: vraagt GPU exclusief en de Docling-modellen uit de voorbereiding van P1; Geert beslist nog. Script geschreven, `bash -n` geslaagd. |

### P6. Feedcontrole voor een YouTube-kanaal en een Spotify-podcast

**Doel.** Aantonen dat de feedcontrole uit FO 5 nieuwe items herkent op id, zonder dubbele meldingen, en dat een Spotify-link is om te zetten naar de openbare RSS-feed van dezelfde show.

**Beslist.** FO 5: detectie op id, het schema korter dan 15 uploads, Spotify naar RSS. De logica wordt later de Langflow-component voor de feedcontrole. Deze proef toetst alleen de logica, buiten Langflow.

**Aanpak.** `p6/feedcontrole.py` in een Python-container met `feedparser==6.0.14`, op een gewoon netwerk met internet.

1. **6a, YouTube (D3).**
   - Ophalen; HTTP-status, `Content-Type`, `ETag` en `Last-Modified` vastleggen.
   - Uitlezen: per item `yt:videoId`, titel, publicatiedatum en beschrijving.
   - De id's opslaan als toestand. Na 60 s opnieuw ophalen en het verschil bepalen.
   - Kunstmatig: de *k* nieuwste id's uit de toestand verwijderen (k = 1, 3) en opnieuw bepalen.
   - Geen download van video, audio of ondertitels.
2. **6b, podcast (D4).** Als 6a, op `guid`, met de bijlage (`enclosure`: URL, type en lengte). Ook een voorwaardelijke aanvraag met `If-None-Match` of `If-Modified-Since`.
3. **6c, Spotify naar RSS (D5).**
   - Shownaam uit de titel van de Spotify-pagina ("De Dag | Listen to the Podcast on Spotify").
   - Titel van de nieuwste aflevering uit `https://open.spotify.com/oembed?url=<showlink>`. Bij controle gaf oEmbed de titel van de nieuwste aflevering en niet de shownaam.
   - Kandidaatfeeds zoeken met de iTunes Search API (`https://itunes.apple.com/search?term=<shownaam>&entity=podcast&country=NL`, zonder sleutel).
   - Een kandidaat geldt als gevonden als de oEmbed-titel overeenkomt met een van de drie nieuwste itemtitels in die feed (genormaliseerd: kleine letters, zonder leestekens).
   - Optioneel, alleen als `PODCASTINDEX_KEY` en `PODCASTINDEX_SECRET` in de omgeving staan: dezelfde zoekvraag via de Podcast Index, die het FO als voorbeeld noemt.
4. **6d, optioneel.** Alleen als `YOUTUBE_API_KEY` in de omgeving staat: de uploads-playlist (`UU…`) via de YouTube Data API, één aanroep van één quota-eenheid.

**Benodigdheden.** Internet. Geen GPU, geen sleutel (6c met Podcast Index en 6d optioneel).

**Succescriterium.**

| Nr | Criterium | Meting |
| --- | --- | --- |
| K1 | 6a: HTTP 200; 1 ≤ aantal items ≤ 15; elk item heeft `videoId` en publicatiedatum | HTTP-status, telling |
| K2 | 6a en 6b: tweede ophaling geeft 0 nieuwe id's, of alleen id's met een publicatiedatum na de eerste ophaling | telling |
| K3 | 6a en 6b: na het verwijderen van *k* id's precies *k* nieuwe | telling |
| K4 | 6b: elk item heeft een `guid` en een bijlage van het type `audio/*` | telling |
| K5 | 6c: de gevonden feed is precies `https://podcast.npo.nl/feed/de-dag.xml`, en de titelcontrole slaagt | stringvergelijking |

**Verwachte uitkomst.** Alle criteria slagen vanaf de werkplek. FO 5 waarschuwt dat de YouTube-feed aan IP-adressen van datacenters soms een foutmelding geeft; vanaf een thuisverbinding wordt dat niet verwacht. De bruikbaarheid van een voorwaardelijke aanvraag wordt alleen vastgelegd.

**Script.** `scripts/signaalsysteem/proeven/p6_feedcontrole.sh`, met `p6/feedcontrole.py`. Compose-profiel `p6`.

**Plek.** Hier.

**Uitkomst.**

| Datum | Commit | Exitcode | Meetwaarden | Conclusie |
| --- | --- | --- | --- | --- |
| 29-10-2026 | b31c4857 | 0 | K1: HTTP 200, 15 items, alle met `videoId` en datum. K2: tweede ophaling na 60 s 0 nieuwe id's (YouTube en podcast). K3: na verwijderen van 1 en 3 id's precies 1 en 3 nieuw (beide feeds). K4: podcastfeed HTTP 200, 2 357 items, alle met `guid` en `audio/mpeg`-bijlage. K5: Spotify-paginatitel "De Dag \| Listen to the Podcast on Spotify", oEmbed-titel = nieuwste aflevering, iTunes-kandidaat `https://podcast.npo.nl/feed/de-dag.xml` met titelovereenkomst. Geen van beide feeds stuurt `ETag` of `Last-Modified`, dus een voorwaardelijke aanvraag is niet bruikbaar. 6c met Podcast Index en 6d overgeslagen: geen sleutels | Geslaagd. Detectie op id werkt voor beide feeds; Spotify-link is zonder sleutel om te zetten naar de RSS-feed. Voor de Langflow-component: elke controle haalt de hele feed op (De Dag: 2 357 items). |

### P7. Docker-toegang voor llama-swap: socket-proxy en Docker-client

**Doel.** De drie open punten van optie 2 uit ADR 0009 beproeven zonder GPU:

- welke rechten de socket-proxy echt geeft;
- of het llama-swap-image een Docker-client bevat;
- hoe llama-swap een container volgt die met `docker start` is gestart, ook zonder `model`-veld in het verzoek.

**Beslist.** ADR 0009, Docker-toegang: optie 2, of terugval op optie 1 of 3.

**Gecontroleerd vooraf.** In `haproxy.cfg` van docker-socket-proxy (master) is de eerste regel `http-request deny unless METH_GET || { env(POST) -m bool }`. Pas daarna komen de regels voor `ALLOW_RESTARTS`, `ALLOW_START` en `ALLOW_STOP`, en daarna die voor `CONTAINERS` (paden die met `/containers` beginnen, dus ook `create`). Volgens die volgorde werkt `ALLOW_START` niet zonder `POST=1`. Met `POST=1` en `CONTAINERS=0` blijven `create`, `exec`, `inspect` en de lijst van containers dicht, omdat de laatste regel alles weigert. De combinatie `CONTAINERS=1` met `POST=1` uit ADR 0009 zou ook `POST /containers/create` toelaten. P7 toetst dit tegen het vastgezette image `v0.5.0`.

**Aanpak.**

1. **7a, Docker-client.** `docker run --rm --entrypoint sh ghcr.io/mostlygeek/llama-swap:v268-cuda13-b11977 -c 'command -v docker; command -v curl; command -v wget; command -v sh'`. Ontbreekt de client, dan `p1/llama-swap-dockercli.Dockerfile` bouwen (`FROM` het llama-swap-image, `COPY --from=docker:29.9.2-cli /usr/local/bin/docker /usr/local/bin/docker`) en de controle herhalen.
2. **Opzet voor 7b en 7c.**
   - Twee proefcontainers, `signaal-proef-dummy-1` en `signaal-proef-dummy2-1`: `python:3.12.16-slim-bookworm` met `python -m http.server 8000`, aangemaakt maar niet gestart.
   - `socket-proxy` op het netwerk `beheer` (`internal: true`), met de host-socket `/var/run/docker.sock` alleen-lezen gekoppeld. Dat is het enige punt waar een proef de Docker-socket gebruikt.
   - Testclient `docker:29.9.2-cli` met `DOCKER_HOST=tcp://socket-proxy:2375`, plus `curlimages/curl:8.23.0` voor de ruwe HTTP-status.
   - Alle aanroepen in 7b gaan naar de proefcontainers. Ook de controle "elke naam is te starten" gebruikt `dummy2`, nooit een container buiten het project.
3. **7b, rechtenmatrix.** Per configuratie de HTTP-status vastleggen van: `_ping`, `POST /containers/<dummy>/start`, `POST /containers/<dummy>/stop`, `GET /containers/json`, `GET /containers/<dummy>/json`, `POST /containers/create` (met een synthetische specificatie zonder koppelingen), `POST /containers/<dummy>/exec`, `GET /images/json`, `GET /volumes`, `POST /containers/<dummy2>/start`.

   | Configuratie | Instellingen |
   | --- | --- |
   | A | `POST=0`, `ALLOW_START=1`, `ALLOW_STOP=1` |
   | B | `POST=1`, `ALLOW_START=1`, `ALLOW_STOP=1`, `CONTAINERS=0` |
   | C | `POST=1`, `ALLOW_START=1`, `ALLOW_STOP=1`, `CONTAINERS=1` (zoals ADR 0009 nu beschrijft) |

4. **7c, llama-swap volgt een container.** llama-swap (image uit 7a) op `verwerking` en `beheer`, met configuratie B en model `dummy`:
   - `cmd: p1/wacht_op_dienst.sh` (`docker start signaal-proef-dummy-1`; wachten tot `http://dummy:8000/` HTTP 200 geeft, hooguit 60 s; daarna blijven lopen zolang dat zo blijft);
   - `cmdStop: docker stop signaal-proef-dummy-1`;
   - `proxy: http://dummy:8000`, `checkEndpoint: /`, `ttl: 30`.

   Uitvoeren:
   - `GET /upstream/dummy/` zonder `model`-veld, de toestand van de container volgen, na `ttl` opnieuw kijken;
   - van buitenaf `docker stop` op `dummy` en daarna een nieuwe aanvraag;
   - `GET /running` of het vergelijkbare beheereindpunt van llama-swap, voor de toestand die llama-swap zelf meldt.

**Benodigdheden.** Images uit 1.7 voor P7. Geen GPU. Akkoord van Geert om de Docker-socket in de container `socket-proxy` te koppelen.

**Succescriterium.**

| Nr | Criterium | Meting |
| --- | --- | --- |
| K1 | 7a: na stap 7a is er een image met een werkende `docker`-client (`docker version --format '{{.Client.Version}}'`, exitcode 0); vastgelegd of het basisimage de client al had | exitcode |
| K2 | 7b, configuratie B: `start` en `stop` HTTP 204 (of 304); `create`, `exec`, beide `GET /containers`, `images` en `volumes` HTTP 403 | HTTP-status |
| K3 | 7b, configuraties A en C vastgelegd. Verwacht bij A: `start` 403; bij C: `create` toegestaan | HTTP-status |
| K4 | 7c: de eerste aanvraag zonder `model`-veld geeft HTTP 200 en zet `dummy` op `running`; binnen `ttl` + 15 s staat `dummy` op `exited`; na de externe stop geeft de volgende aanvraag weer HTTP 200 | HTTP-status, `docker inspect` |
| K5 | momentopname: geen container buiten `signaal-proef` van toestand veranderd | verschil van de momentopnamen |

Beslisregel voor ADR 0009: K2 en K4 geslaagd betekent dat optie 2 uitvoerbaar is met configuratie B. Die configuratie legt de bouwtaak vast in plaats van `CONTAINERS=1`. Het restrisico blijft dat elke container op naam te starten en te stoppen is (K3, `dummy2`).

**Verwachte uitkomst.**

- K2 en K3 volgens de volgorde in `haproxy.cfg`: bij A weigert de proxy `start`, bij B werkt alleen starten en stoppen, en C is te ruim.
- K1: of de client in het image zit, is niet gedocumenteerd. Verwacht wordt van niet, dus een afgeleid image.
- K4: waarschijnlijk geslaagd met de werkwijze met wachten. Niet gedocumenteerd.

**Script.** `scripts/signaalsysteem/proeven/p7_docker_toegang.sh`, met `p1/wacht_op_dienst.sh`, `p1/llama-swap-dockercli.Dockerfile` en `p7/llama-swap.yaml`. Compose-profiel `p7`.

**Plek.** Uit te voeren op de werkplek, alleen vanwege het akkoord voor de Docker-socket; geen GPU nodig.

**Uitkomst.**

| Datum | Commit | Exitcode | Meetwaarden | Conclusie |
| --- | --- | --- | --- | --- |
| 29-10-2026 | 013704dd | niet uitgevoerd | — | Niet uitgevoerd: vraagt het akkoord van Geert om de Docker-socket alleen-lezen in `socket-proxy` te koppelen (`SIGNAAL_PROEF_AKKOORD_SOCKET=ja`), en het llama-swap-image (download). Script geschreven, `bash -n` geslaagd. |
| 30-10-2026 | 7153db3c | 1 | K1: het basisimage `v268-cuda13-b11977` heeft geen `docker` en geen `wget`, wel `curl` en `sh`; het afgeleide image (`COPY --from=docker:29.9.2-cli`) heeft een werkende client (`docker --version` exitcode 0, 29.9.2). K2, configuratie B (`POST=1`, `ALLOW_START=1`, `ALLOW_STOP=1`, `CONTAINERS=0`): `_ping` 200, `start` 204, `stop` 204; `GET /containers/json`, `GET /containers/<id>/json`, `create`, `exec`, `images`, `volumes` alle 403; `start` van `dummy2` 204. K3: configuratie A (`POST=0`): alles 403 behalve `_ping`, dus `ALLOW_START` werkt niet zonder `POST=1`; configuratie C (`CONTAINERS=1`): `create` 201 (container aangemaakt), lijst en inspect 200, `exec` 409 (alleen omdat de container niet draaide). K4: eerste aanvraag op `/upstream/dummy/` zonder `model`-veld HTTP 200, container `running`, `/running` meldt `ready`; na `ttl` + 15 s `exited`; direct na een externe `docker stop` HTTP 502 (llama-swap houdt het model voor `ready` tot `wacht_op_dienst.sh` de stop opmerkt; log: `proxy error: dial tcp: lookup dummy`), 20 s later weer HTTP 200. K5: geen wijziging buiten het project | Mislukt op K4, deels: het mechanisme werkt (starten zonder `model`-veld, volgen, ontladen na `ttl`, herstel na een externe stop), maar de eerste aanvraag na een externe stop geeft 502, tot de gezondheidslus (elke 5 s) het merkt. Voor ADR 0009: optie 2 is uitvoerbaar met configuratie B en een afgeleid image met de Docker-client; `CONTAINERS=1` (C) is te ruim, want `create` werkt dan. Restrisico K3: elke container is op naam te starten en te stoppen (`dummy2`). De afnemer moet een 502 na een externe stop opvangen met een nieuwe poging. De eerste run (360937d9) telde K1 als mislukt doordat `docker version` zonder daemon exitcode 1 geeft; script hersteld. |

### P8. Netwerkafscherming met `internal: true`

**Doel.** Vaststellen of een dienst op een netwerk met `internal: true` geen internet heeft en geen bereikbare poort op de host publiceert, onder Docker Desktop met WSL2, en wat een tweede, gewoon netwerk daaraan verandert.

**Beslist.** ADR 0002 F (netwerken `verwerking`, `uitgaand` en `toegang`); FO 12.1. De Compose-documentatie zegt alleen dat `internal: true` "an externally isolated network" maakt; over poortpublicatie zegt ze niets. Gemeenschapsbronnen melden dat gepubliceerde poorten op zo'n netwerk niet bereikbaar zijn. Voor Docker Desktop is niets gevonden.

**Aanpak.** Drie containers `python:3.12.16-slim-bookworm` met `python -m http.server 8000`.

| Container | Netwerken | Poort |
| --- | --- | --- |
| `alleen_intern` | `verwerking` (`internal: true`) | `127.0.0.1:18081:8000` |
| `dubbel` | `verwerking` en `toegang` (gewoon) | `127.0.0.1:18082:8000` |
| `controle` | `toegang` | `127.0.0.1:18083:8000` |

1. Vanaf de host (Git Bash en WSL2): `curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:1808x/`.
2. Vanuit elke container: `https://example.org` ophalen (Python `urllib`, time-out 5 s) en een DNS-opvraging van `example.org`.
3. Vanuit `alleen_intern`: `http://dubbel:8000/` (bereikbaarheid binnen het netwerk).
4. De waarschuwingen van `docker compose up` vastleggen.

**Benodigdheden.** Alleen `python:3.12.16-slim-bookworm`. Geen GPU.

**Succescriterium.**

| Nr | Criterium | Meting |
| --- | --- | --- |
| K1 | host naar `alleen_intern`: geen HTTP-antwoord (curl-exitcode 7, 28 of 52) | exitcode |
| K2 | `alleen_intern` naar internet: mislukt (geen DNS of geen verbinding) | exitcode |
| K3 | host naar `dubbel` en `controle`: HTTP 200 (positieve controle) | HTTP-status |
| K4 | `alleen_intern` naar `dubbel`: HTTP 200 | HTTP-status |
| K5 | `dubbel` naar internet: vastgelegd | exitcode |

**Verwachte uitkomst.**

- K1 tot en met K4 slagen.
- K5: verwacht wordt dat `dubbel` via `toegang` wél internet heeft. Dan heeft ook `open-notebook-mcp` in de opzet van ADR 0002 uitgaand internet. Dat is voor te leggen aan Geert, bijvoorbeeld met een omgekeerde proxy op `toegang` en open-notebook-mcp alleen op `verwerking`.
- Onder Docker Desktop is K1 onzeker; daarom deze proef.

**Script.** `scripts/signaalsysteem/proeven/p8_netwerkafscherming.sh`. Compose-profiel `p8`.

**Plek.** Hier.

**Uitkomst.**

| Datum | Commit | Exitcode | Meetwaarden | Conclusie |
| --- | --- | --- | --- | --- |
| 29-10-2026 | b31c4857 | 0 | K1: host (Git Bash) naar `alleen_intern` curl-exitcode 7, ook vanuit WSL2 (exitcode 7). K2: `alleen_intern` naar internet: geen DNS (`gaierror`) en geen verbinding. K3: host naar `dubbel` en `controle` HTTP 200 (ook vanuit WSL2). K4: `alleen_intern` naar `dubbel` HTTP 200. K5: `dubbel` en `controle` hebben internet (DNS ok, example.org antwoordt met HTTP 403 op de Python-client, dus wel verbinding). `docker compose up` gaf geen waarschuwing | Geslaagd. Onder Docker Desktop publiceert een container met alleen een `internal: true`-netwerk geen bereikbare poort. Een container die ook aan een gewoon netwerk hangt (`dubbel`) heeft uitgaand internet: voor ADR 0002 F betekent dit dat open-notebook-mcp niet tegelijk aan `verwerking` en `toegang` mag hangen zonder extra maatregel (zie de verwachte uitkomst). Eerste run telde een HTTP-foutstatus als "geen verbinding"; script hersteld. |

### P9. Opruimen

`scripts/signaalsysteem/proeven/p9_opruimen.sh` stopt en verwijdert de containers en netwerken van het project `signaal-proef`. Met `--volumes` verwijdert het ook de volumes met het label `com.docker.compose.project=signaal-proef`, en met `--images` de lokaal gebouwde images met het voorvoegsel `signaal-proef/`. Het script controleert achteraf met de momentopname (1.2) dat niets buiten het project is veranderd. Plek: hier.

**Uitkomst.** 30-10-2026, commit 921c406b, `p9_opruimen.sh --volumes`, exitcode 0: geen containers, netwerken of volumes van het project meer; momentopname zonder afwijking. Containers, volumes en netwerken zijn gelijk aan de beginstand van P0 (de zeven gestopte containers van project `open-notebook` ongemoeid). Blijven staan, bewust: de lokaal gebouwde images `signaal-proef/*` (waaronder de fork-API van 18,7 GB, nodig voor de acceptatietest van P4 na ADR 0001 en 0002) en de opgehaalde images uit 1.7 (Langflow 1.13.0, SurrealDB v2.7.0, python 3.12.16-slim). Verwijderen kan met `p9_opruimen.sh --images`; de build-cache van de fork-API ruimt het script niet op (`docker builder prune` valt onder de verboden opdrachten).

## 4. Embeddingmodellen vergelijken (ADR 0008)

Een kwaliteitsvergelijking is in fase 1 geen zinvolle proef. Het meetcriterium uit ADR 0008 is het zoekdeel van de referentieset uit FO 7, en die set bestaat nog niet. FO 7 zegt zelf: zonder die set is een verbetering in precisie of recall niet aantoonbaar. Een vergelijking op een geïmproviseerde set zou een schijnzekerheid geven.

Wel in fase 1:

- het videogeheugen van mxbai-embed-large en een meertalige kandidaat in P3;
- in P1 een controle dat het embeddingmodel via llama-swap vectoren van 1 024 dimensies teruggeeft. Dat kost niets extra en raakt de tests die 1 024 vastleggen.

De kwaliteitsvergelijking hoort in het plan van aanpak, als taak in fase 3 na het opbouwen van de referentieset.

## 5. Bevindingen bij het opstellen die ADR's raken

Deze punten volgen uit de gecontroleerde documentatie. De proeven bevestigen of weerleggen ze; de ADR's zijn niet aangepast.

1. **ADR 0009, socket-proxy.** `ALLOW_START` werkt volgens `haproxy.cfg` niet zonder `POST=1`. De veilige combinatie is `POST=1` met `CONTAINERS=0`, niet `CONTAINERS=1` zoals de ADR nu noemt (P7).
2. **ADR 0006 en 0009, image.** Voor de RTX 5070 Ti (Blackwell) is een CUDA 13-image van llama-swap nodig. Volgens de README dekt `unified-cuda` (CUDA 12) alleen Pascal tot en met Ada (P0).
3. **ADR 0014, WhisperX.**
   - `Nyralei/whisperx-api-server` heeft geen uitgebrachte versie en geen image in een registry, en vraagt voor sprekerherkenning een Hugging Face-token.
   - De AWS-container heeft een vaste tag (`3.8.6-cu128-amzn2023`), pyannote `speaker-diarization-community-1` in het image (geen token nodig volgens de AWS-documentatie), en instellingen voor offline gebruik. Hij kent geen `model`-veld per verzoek.
   - Op het criterium "vaste imageversie" scoort de AWS-container daarom beter dan de voorkeur in de ADR (P1, deelstap 1c).
4. **ADR 0009, Kev.**
   - Kev-4B is een LoRA-adapter op `Qwen3.5-4B-Base` (9,3 GB in bf16).
   - Kev heeft geen Docker-image en volgens de README geen gezondheidseindpunt.
   - Met `KEV_API_KEY` geven paden onder `/v1/*` zonder sleutel geen HTTP 200, zodat `checkEndpoint` van llama-swap daar niet werkt (P1, deelstap 1d).
5. **ADR 0017, schema-trigger.**
   - `ENABLE_TRIGGERS` lijkt een frontendvlag, geen gedocumenteerde omgevingsvariabele.
   - Triggers starten als `pending`; de interface kan ze in 1.13.0 waarschijnlijk niet aanzetten, de API wel (P2).
6. **ADR 0002, open-notebook-mcp.** De zoektool gebruikt standaard `type="vector"`, dus het embeddingmodel moet voor open-notebook-mcp bereikbaar zijn via de API van de fork. Een container die ook aan het gewone netwerk `toegang` hangt, heeft waarschijnlijk uitgaand internet (P8, K5).

## 6. Wat de uitvoering van Geert vraagt

| Proef | Handeling of besluit |
| --- | --- |
| P1, P7 | Akkoord om de Docker-socket alleen-lezen in de container `socket-proxy` van het proefproject te koppelen |
| P1, P3, P5 | Bestaande GPU-containers (`docling`, `whisperx`, `mineru`) stoppen en `ollama ps` leeg maken; toepassingen sluiten die veel videogeheugen gebruiken. Het script controleert dit, maar doet het niet zelf |
| P1 (alleen variant B) | Hugging Face-token in de omgevingsvariabele `HF_TOKEN`, en op Hugging Face akkoord met de voorwaarden van `pyannote/speaker-diarization-community-1` ("gated: auto"). Voor variant A niet nodig |
| P3 | Keuze van het werkmodel (voorstel: Qwen3.5-9B Q4_K_M) en van de meertalige embeddingkandidaten (voorstel: bge-m3; tweede kandidaat open, ADR 0008) |
| P6 (optioneel) | `PODCASTINDEX_KEY` en `PODCASTINDEX_SECRET`, of `YOUTUBE_API_KEY`, in de omgeving. Zonder deze sleutels draait P6 volledig |
| alle | Ruim 60 GB vrije ruimte op de Docker-schijf; akkoord om de API van de fork lokaal als apart image te bouwen |

## 7. Bronnen

Alle geraadpleegd op 29-10-2026.

- llama-swap:
  - [README](https://github.com/mostlygeek/llama-swap)
  - [voorbeeldconfiguratie](https://raw.githubusercontent.com/mostlygeek/llama-swap/main/docs/config.example.yaml): `cmdStop`, `checkEndpoint`, `ttl`, groepen, `upstream.ignorePaths`, `maxRequestBodyMB`
  - [releases](https://github.com/mostlygeek/llama-swap/releases): v268 van 28-10
  - [imagetags](https://github.com/mostlygeek/llama-swap/pkgs/container/llama-swap/versions?filters%5Bversion_type%5D=tagged): `v268-cuda13-b11977`
- docling-serve:
  - [README](https://github.com/docling-project/docling-serve): images en grootte
  - [configuratie](https://raw.githubusercontent.com/docling-project/docling-serve/main/docs/configuration.md)
  - [gebruik](https://raw.githubusercontent.com/docling-project/docling-serve/main/docs/usage.md)
  - [releases](https://github.com/docling-project/docling-serve/releases): v1.38.2 van 27-10
  - [uitrol](https://docling-project.github.io/docling/usage/api_server/deployment/)
- WhisperX:
  - [AWS Deep Learning Containers, WhisperX](https://aws.github.io/deep-learning-containers/whisperx/)
  - [configuratie](https://aws.github.io/deep-learning-containers/whisperx/configuration/)
  - [EC2-uitrol](https://aws.github.io/deep-learning-containers/whisperx/deployment/ec2/)
  - [whisperx-api-server](https://github.com/Nyralei/whisperx-api-server) en [`.env.example`](https://raw.githubusercontent.com/Nyralei/whisperx-api-server/main/.env.example)
  - [pyannote community-1 op Hugging Face](https://huggingface.co/api/models/pyannote/speaker-diarization-community-1)
- Kev:
  - [README](https://github.com/jaredpalmer/kev)
  - [kev-4b](https://huggingface.co/api/models/jaredpalmer/kev-4b?blobs=true) en [adapterconfiguratie](https://huggingface.co/jaredpalmer/kev-4b/resolve/main/adapter_config.json)
  - [Qwen3.5-4B-Base](https://huggingface.co/api/models/Qwen/Qwen3.5-4B-Base?blobs=true)
- Langflow:
  - [PR #14958](https://github.com/langflow-ai/langflow/pull/14958)
  - [releases](https://github.com/langflow-ai/langflow/releases): 1.13.0 van 22-10
  - [omgevingsvariabelen](https://docs.langflow.org/environment-variables)
  - [image 1.13.0](https://hub.docker.com/v2/repositories/langflowai/langflow/tags/1.13.0)
- open-notebook-mcp:
  - [PyPI](https://pypi.org/pypi/open-notebook-mcp/json): 0.3.0
  - [`server.py`](https://raw.githubusercontent.com/Epochal-dev/open-notebook-mcp/main/src/open_notebook_mcp/server.py)
- docker-socket-proxy:
  - [README](https://github.com/Tecnativa/docker-socket-proxy)
  - [`haproxy.cfg`](https://raw.githubusercontent.com/Tecnativa/docker-socket-proxy/master/haproxy.cfg)
  - [tags](https://hub.docker.com/v2/repositories/tecnativa/docker-socket-proxy/tags?page_size=30&name=v)
- Docker:
  - [Compose, netwerken](https://docs.docker.com/reference/compose-file/networks/)
  - [Compose, secrets](https://docs.docker.com/reference/compose-file/secrets/)
  - [Docker Engine 28, netwerkwijzigingen](https://www.docker.com/blog/docker-engine-28-hardening-container-networking-by-default/)
  - [gemeenschapsbron: poorten op een intern netwerk](https://forums.docker.com/t/internal-network-between-containers-without-external-network-access/41751.md)
- NVIDIA:
  - [CUDA op WSL](https://docs.nvidia.com/cuda/archive/13.3.0/wsl-user-guide/index.html)
  - [CUDA 13.0, releasenotes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/)
- Modellen:
  - [Qwen3.5-9B GGUF](https://huggingface.co/api/models/unsloth/Qwen3.5-9B-GGUF?blobs=true)
  - [mxbai-embed-large-v1 GGUF](https://huggingface.co/api/models/ChristianAzinn/mxbai-embed-large-v1-gguf?blobs=true)
  - [bge-m3 GGUF](https://huggingface.co/api/models/gpustack/bge-m3-GGUF?blobs=true)
  - [bge-reranker-v2-m3 GGUF](https://huggingface.co/api/models/gpustack/bge-reranker-v2-m3-GGUF?blobs=true)
- Python-pakketten:
  - [feedparser](https://pypi.org/pypi/feedparser/json)
  - [mcp](https://pypi.org/pypi/mcp/json)
  - [APScheduler](https://pypi.org/pypi/APScheduler/json)
- Testdata:
  - [Tweede Kamer op sociale media](https://www.tweedekamer.nl/contact-en-bezoek/volg-de-tweede-kamer-online/de-tweede-kamer-op-sociale-media): kanaal-id
  - [iTunes Search API, De Dag](https://itunes.apple.com/search?term=NOS+De+Dag&entity=podcast&country=NL&limit=3): feed-URL
  - [Spotify-link van De Dag](https://www.podengine.ai/podcasts/de-dag)
  - [oEmbed van Spotify](https://open.spotify.com/oembed?url=https://open.spotify.com/show/2Oeh3L5u0y9R54jEz8Mgty)
  - de PDF-URL's in tabel 1.6

## 8. Aanpassingen bij het uitwerken van de scripts (stap 3b)

Stand 29-10-2026. De scripts volgen het plan; deze punten wijken af of vullen aan.

1. **Bestanden die het plan niet noemt.** Geen bind mounts naar mappen in de repository (1.2) betekent dat scripts en configuratie in kleine lokale images zitten: `p2/langflow.Dockerfile` (TickLogger in Langflow), `p2/hulp.Dockerfile` en `p2/langflow_api.py` (API-aanroepen en analyse van de ticks), `p4/start_mcp.py`, `p6/feedcontrole.Dockerfile`, `p1/modellen.Dockerfile` en `p1/modellen_downloaden.py` (voorbereiden). Losse `docker run`-containers krijgen het label `com.docker.compose.project=signaal-proef`, zodat ze tot het project tellen.
2. **Momentopname van images.** Nieuwe images zijn toegestaan als ze in 1.7 staan, `signaal-proef/*` heten of `<none>` zijn. Een verdwenen of gewijzigd image buiten die reeks geeft exitcode 3.
3. **P0, GPU in een container.** Het llama-swap-image staat niet in het akkoord voor deze ronde. P0 gebruikt het als het lokaal staat of met `--voorbereiden`; anders `python:3.12.16-slim-bookworm` met `--gpus all` (Docker Desktop koppelt `nvidia-smi` in elke GPU-container), en als laatste `nvidia-smi.exe` op de host. De bron staat in de uitkomst. Niet-GPU-proeven roepen P0 aan met `--als-voorwaarde` en eisen alleen Docker, Linux-containers en ruimte.
4. **P0, CUDA-versie.** Driver 610 meldt `CUDA UMD Version: 13.3` in plaats van `CUDA Version`; het script herkent beide.
5. **P4, image van de fork.** Gebouwd uit `git archive` van de laatste commit die de code van de fork raakt (buiten `docs/` en `scripts/signaalsysteem/`), niet uit de werkmap, zodat ongetrackte mappen als `notebook_data_live` niet in de buildcontext komen. In die kopie van de `Dockerfile` zijn alleen de zwevende basis-tags vervangen: `python:3.12-slim-bookworm` door `python:3.12.16-slim-bookworm` en `ghcr.io/astral-sh/uv:latest` door `ghcr.io/astral-sh/uv:0.11.13` (de versie van uv in de WSL-omgeving). Zo wisselen de lokale tags van de bestaande omgeving niet van versie. De API start met `uvicorn` rechtstreeks, zonder supervisord en frontend.
6. **P4, open-notebook-mcp.** Het officiële startpunt (`open_notebook_mcp.main`) geeft `host`, `port` en `path` mee aan `FastMCP.run`; in `mcp==1.30.0` kent `run` alleen `transport` en `mount_path`. Het script legt de uitkomst van het officiële startpunt vast en start de server met `p4/start_mcp.py`, dat dezelfde waarden via `mcp.settings` zet en de localhost-beperking op de Host-header uitzet (wat `host=0.0.0.0` in het origineel ook doet). Aan de tools verandert niets.
7. **P4, testdata.** Fragmenten staan als `chunk` en `source_embedding` (met een willekeurige vector van 1 024 dimensies), notities via `artifact`, bronnen via `reference` aan een notebook. Recordverwijzingen gaan via `type::thing`.
8. **P8.** Een HTTP-foutstatus (bijvoorbeeld 403 van example.org) telt als "wel verbinding": het antwoord kwam van internet.
9. **P6.** Kopregels worden hoofdletterongevoelig gelezen.
10. **P2.** Componenten uit `LANGFLOW_COMPONENTS_PATH` staan in 1.13.0 in `/api/v1/all` onder een sleutel als `ext:proef:TickLogger@extra`; die sleutel is ook het knooptype in de flow. De uitvoertypen van de randen komen uit het sjabloon (in 1.13.0 `JSON`, niet `Data`). Elke run begint met lege P2-volumes, omdat het superuser-wachtwoord per run nieuw is.
11. **Wachttijden instelbaar.** P2 kent `P2_STOP_S`, `P2_NA_HERSTART_S` en `P2_LOOPTIJD_S` (standaard 150, 120 en 240 s); alleen de standaardwaarden tellen voor de uitkomst.
12. **Akkoord Docker-socket.** P1 en P7 starten alleen met `SIGNAAL_PROEF_AKKOORD_SOCKET=ja` in de omgeving; anders exitcode 2.
13. **Anonieme volumes.** Het image van SurrealDB declareert `VOLUME /data /logs`. Compose maakt daar anonieme volumes voor zonder projectlabel; de momentopname zag die als wijziging buiten het project. Het opruimen per profiel verwijdert nu de anonieme volumes van de eigen containers.
14. **P2, inhaalrun van de terugval.** APScheduler meldt bij `coalesce` het laatst gemiste moment als gepland moment. Een run waarvan dat moment vóór het startcommando ligt, telt als inhaalrun; het herstartmoment wordt vóór het starten gemeten. Bij de schema-trigger telt `missed_ticks` in de payload.
15. **P2, Langflow-sleutel.** `LANGFLOW_SECRET_KEY` moet een Fernet-sleutel zijn (32 bytes, url-veilig base64); met een hex-waarde geeft `POST /api/v1/api_key/` HTTP 400.
16. **Open in de modellenlijst (1.7).** De revisie van `ChristianAzinn/mxbai-embed-large-v1-gguf` staat ingekort (`3ec6d46a…`); `modellen_downloaden.py` gebruikt die korte waarde. Slaagt dat niet, dan is de volledige commit-hash nodig. Hetzelfde geldt voor de ingekorte SHA-256-waarden: het script controleert begin en einde.

Aanvullingen bij de GPU-ronde van 30-10-2026:

17. **Revisie mxbai.** De volledige revisie is `3ec6d46af11ba2b982d8c6a1e11183c4995f76a7` (Hugging Face API, `sha` van `ChristianAzinn/mxbai-embed-large-v1-gguf`); `modellen_downloaden.py` gebruikt die nu.
18. **Basismodel van Kev.** `huggingface_hub` leest `HF_HOME` bij het importeren; het zetten van `HF_HOME` daarna had geen effect, en het basismodel zou buiten het volume terechtkomen. Het script geeft nu `cache_dir="/hf/hub"` mee. Kev laadt het basismodel op `base_revision` `1001bb4d…` uit `training_config.json` van de adapter; dat is dezelfde revisie als in 1.7.
19. **P7, Docker-client.** `docker version` geeft zonder bereikbare daemon (`--network none`) altijd exitcode 1, ook met een werkende client. K1 toetst nu met `docker --version`.
20. **P7 en P1, herstel na een externe stop.** Naast de directe aanvraag (K4 in P7, K7 in P1) legt het script ook een aanvraag 20 s later vast. Het criterium blijft de directe aanvraag.
21. **P1, K10.** Geslaagd alleen als beide adressen "geen verbinding" geven; een mislukte controle (bijvoorbeeld geen `python` in de container) telde eerder als geslaagd.
22. **P3 en P5, Docling.** P3 gebruikt het asynchrone eindpunt met pollen in plaats van `/v1/convert/file`: het synchrone eindpunt kapt af na `DOCLING_SERVE_MAX_SYNC_WAIT`. In P5 gaf een ontbrekende `task_id` een scriptfout in plaats van een mislukt criterium.
23. **llama-swap-image en de driver.** `v268-cuda13-b11977` eist `cuda>=13.4` (`NVIDIA_REQUIRE_CUDA`, runtime 13.4.1); driver 610.88 levert CUDA 13.3. Docker weigert de container (`unsatisfied condition: cuda>=13.4`). Met `NVIDIA_DISABLE_REQUIRE=1` start hij, maar de meegeleverde compat-libcuda 13.4 ziet onder WSL2 geen GeForce-kaart (`no CUDA-capable device is detected`). Met `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64` laadt llama-server de libcuda van de driver en draait de 13.4-runtime via minor version compatibility (werkmodel geladen, 5,7 GB). P1 (`compose.proef.yml`) en P3 zetten beide variabelen; P0 legt de eis van het image vast. Het criterium van P0 (CUDA ≥ 13.0) was te ruim voor deze tag.
24. **Sleutels van docling en kev.** Compose brengt secrets aan bij `compose up`, niet bij `docker start` door llama-swap; docling-serve stopte op een lege `DOCLING_SERVE_API_KEY`. De sleutels per run staan nu in het volume `signaal-proef_geheimen` (alleen-lezen op `/run/secrets` van docling en kev), dat het script na `create` vult en na de proef verwijdert.
25. **NLTK voor WhisperX.** De uitlijning van WhisperX (variant A) haalt bij elk verzoek NLTK `punkt_tab` op; offline gaf dat HTTP 500. Volume `whisperx_nltk` op `/root/nltk_data`, gevuld bij het voorbereiden (15 MB).
26. **Voorbereiden.** `public.ecr.aws` gaf anoniem `toomanyrequests`; het script probeert drie keer met 60 s ertussen. `docling-voorbereiden` draait als root, omdat het image als uid 1001 draait en niet in het nieuwe volume kon schrijven.
27. **Git Bash.** `curl` is daar een Windows-programma en kreeg door `MSYS2_ARG_CONV_EXCL` een POSIX-pad (curl-fout 23); `testdata` zet het pad nu om. Git Bash kent geen `pkill`: na een scriptfout liepen `nvidia-smi.exe` en `docker events` door. `kill_boom` stopt nu de hele procesboom, en elk einde van een proef stopt alle kindprocessen. Tellingen met `grep` in een pijplijn geven bij nul treffers geen scriptfout meer (`zoek`). De logs van het profiel worden bewaard vóór `compose down`.
