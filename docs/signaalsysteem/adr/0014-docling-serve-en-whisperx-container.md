# ADR 0014: Documentextractie met docling-serve en transcriptie met een bestaande WhisperX-container

- **Status:** besloten; WhisperX-container gekozen op grond van P1 (30-10): variant A
- **Datum:** 2026-10-29; bijgewerkt 2026-10-30 met P1, P3 en P5
- **Raakt FO-paragraaf:** 7, 11 (zware stappen in aparte containers), 12.1 (offline-instellingen), 12.2, 13.1
- **Lost op:** verschilanalyse conflict 10
- **Gewijzigd door:** ADR 0019 (classificatie en pipelinestart per innameroute)

## Context

Het functioneel ontwerp (FO) zegt:

- documentextractie gebeurt met docling-serve, met asynchrone taken en een Compose-bestand voor NVIDIA-GPU's (grafische kaarten). Dat vervangt de eigen Docling-worker;
- transcriptie gebeurt met WhisperX in een bestaande container met een API (Application Programming Interface). Welke container, wordt in fase 1 gekozen;
- zware stappen zijn aparte containers die een Langflow-component aanroept (FO 7, FO 11, FO 13.1);
- offline: modellen vooraf in een vaste map, externe diensten uit, en `HF_HUB_OFFLINE=1` na het downloaden (FO 12.1).

In de fork:

- `services/docling` is een eigen FastAPI-dienst. `POST /process` werkt synchroon op een pad (`services/docling/api.py:126-136`, `:187-234`). Het model blijft geladen (`:118-125`). De dienst heeft eigen verrijkingen (`:31-34`) en gebruikt de parsercode uit `pipelines/ingestion` (`services/docling/Dockerfile`);
- `services/whisperx` is een eigen dienst, met het model in een procesvariabele (`services/whisperx/api.py:117-121`) en het Hugging Face-token via een omgevingsvariabele (`docker-compose.yml:140`);
- de taal voor tekstherkenning (Optical Character Recognition, OCR) staat op Engels (`docker-compose.yml:67`), terwijl het corpus Nederlands is.

**docling-serve (gecontroleerd)**

- Nieuwste versie: v1.38.2, van 27 oktober volgens de releasepagina; de pagina noemt geen jaar.
- CUDA-images `docling-serve-cu128` en `-cu130`, alleen met expliciete versietags, zonder `latest`.
- Eindpunten:
  - `POST /v1/convert/file/async` en `POST /v1/convert/source/async`;
  - `GET /v1/status/poll/{task_id}`;
  - `GET /v1/result/{task_id}`.
- Instellingen:
  - `DOCLING_SERVE_ARTIFACTS_PATH` (modelgewichten uit een vaste map);
  - `DOCLING_SERVE_ENABLE_REMOTE_SERVICES`, standaard `false`;
  - `DOCLING_SERVE_API_KEY` (header `X-Api-Key`);
  - `DOCLING_SERVE_ENG_LOC_NUM_WORKERS`, standaard 2;
  - `DOCLING_SERVE_SINGLE_USE_RESULTS`, standaard `true`;
  - `DOCLING_SERVE_LOAD_MODELS_AT_BOOT`.
- OCR-opties zijn onder meer `do_ocr`, `force_ocr` en `ocr_lang` (BCP-47-taalcodes).
- Er is geen instelling die het videogeheugen vrijgeeft (zie ADR 0009).

**WhisperX-containers (gevonden, niet beproefd)**

- `Nyralei/whisperx-api-server`: OpenAI-compatibel op `/v1/audio/transcriptions`, met sprekerlabels en woordtijden, Compose-profielen voor CUDA, licentie GPL-3.0. Het laad- en ontlaadgedrag staat niet op de projectpagina.
- De WhisperX-container uit de AWS Deep Learning Containers: `/v1/audio/transcriptions` met `diarize`, `min_speakers` en `max_speakers`. Welke imagetags er zijn, kon niet worden vastgesteld.
- `tijszwinkels/whisperX-api`: sprekerherkenning, Hugging Face-token nodig; eindpunten niet nagegaan.

Aanvulling bij het proefplan (proeven.md §5 punt 3): `Nyralei/whisperx-api-server` heeft geen uitgebrachte versie en geen image in een registry, en vraagt voor sprekerherkenning een Hugging Face-token. De AWS-container heeft een vaste tag, `public.ecr.aws/deep-learning-containers/whisperx:3.8.6-cu128-amzn2023`, met `pyannote/speaker-diarization-community-1` in het image. Hij kent geen `model`-veld per verzoek.

**Proeven (30-10-2026)**

- **WhisperX, variant A (P1).** De AWS-container haalt alle criteria; variant B is niet nodig gebleken:
  - `/v1/audio/transcriptions` via llama-swap met `model=whisperx`: llama-swap routeert op het veld, de server negeert het (K3);
  - sprekers en woordtijden: D4 gaf 3 sprekers en 459 segmenten met woordtijden (K3);
  - Nederlands en Engels zonder `language`: taal `nl` respectievelijk `en` gedetecteerd, met woordtijden (K9);
  - offline met `HF_HUB_OFFLINE=1` op het netwerk `verwerking`, zonder Hugging Face-token (K3, K8, K9). Voorwaarde: NLTK `punkt_tab` staat vooraf in een volume op `/root/nltk_data`; anders haalt de uitlijning dat bij elk verzoek op en geeft ze offline HTTP 500 (proeven.md §8 punt 25);
  - gebruiksregistratie: `OPT_OUT_TRACKING=true` staat in de container, `example.org` en `aws.amazon.com` gaven geen verbinding, en de logs bevatten 0 regels met track, telemetry of usage (K10). De AWS-documentatie zegt dat die variabele de gebruiksregistratie van Deep Learning Containers uitzet ([logging and monitoring](https://aws.github.io/deep-learning-containers/security/logging-and-monitoring/), geraadpleegd 30-10-2026);
  - videogeheugen: piek 10 200–10 328 MiB, in rust 8 874 MiB, na stop 0 MiB (P3).
- **docling-serve (P5, P1, P3).**
  - Het videogeheugen komt alleen vrij door de container te stoppen. Na een taak blijft 2 066–2 450 MiB bezet, ook na `GET /v1/clear/converters` en `/v1/clear/results`, die wel bestaan (P5, K3).
  - Met `DOCLING_SERVE_LOAD_MODELS_AT_BOOT=false` en `DOCLING_SERVE_OPTIONS_CACHE_SIZE=1` is het beslag 364 MiB lager en is de container in rust leeg tot de eerste taak (P5).
  - RapidOCR draait op de CPU: `CUDAExecutionProvider` is niet beschikbaar in het image `docling-serve-cu128:v1.38.2` (P5). Een groot document met OCR is daardoor traag; D1 (Ridderkerk, 7,5 MB) liep zo lang dat de pauze tussen twee statusvragen boven de `ttl` van 60 s kwam (P1, 1a; zie ADR 0009).
  - De wachtrij van docling-serve staat in het geheugen: een stop van de container verliest de lopende taken, en `status/poll` geeft daarna 404 (P1, 1a).
- **Ophalen.** `public.ecr.aws` gaf anoniem `toomanyrequests`; opnieuw proberen na 60 s hielp (proeven.md §8 punt 26).

Externe bronnen (geraadpleegd 29-10-2026):

- [docling-serve](https://github.com/docling-project/docling-serve)
- [docling-serve, configuratie](https://raw.githubusercontent.com/docling-project/docling-serve/main/docs/configuration.md)
- [docling-serve, gebruik](https://raw.githubusercontent.com/docling-project/docling-serve/main/docs/usage.md)
- [docling-serve, releases](https://github.com/docling-project/docling-serve/releases)
- [whisperx-api-server](https://github.com/Nyralei/whisperx-api-server)
- [WhisperX in AWS Deep Learning Containers](https://aws.github.io/deep-learning-containers/whisperx/)
- [whisperX-api](https://github.com/tijszwinkels/whisperX-api/blob/main/README.md)

## Opties

Het FO kiest docling-serve en een bestaande WhisperX-container. Open is alleen welke WhisperX-container. Die keuze valt in proef 1, op deze criteria:

- OpenAI-compatibel `/v1/audio/transcriptions` met een `model`-veld, zodat llama-swap op het model kan routeren;
- sprekerherkenning en woordtijden;
- een vaste imageversie;
- offline te draaien met `HF_HUB_OFFLINE=1`;
- een licentie die past bij eigen gebruik.

Op basis van de gevonden documentatie voldoet `Nyralei/whisperx-api-server` het best. De AWS-container is het alternatief.

Uitkomst van proef 1 (30-10): de AWS-container (variant A) haalt alle criteria. Het `model`-veld is voor llama-swap genoeg: hij routeert erop en de server negeert het. Variant B is niet beproefd, omdat A op geen criterium afviel.

## Voorstel

**docling-serve**

- Een container met een vaste versie, bijvoorbeeld `docling-serve-cu128:v1.38.2` of de versie die in proef 5 is getest.
- Instellingen:
  - `DOCLING_SERVE_ARTIFACTS_PATH` op een alleen-lezen volume met vooraf gedownloade modellen;
  - `DOCLING_SERVE_ENABLE_REMOTE_SERVICES=false`;
  - `DOCLING_SERVE_API_KEY` uit een Docker secret;
  - `DOCLING_SERVE_ENG_LOC_NUM_WORKERS=1`, één taak tegelijk op de GPU;
  - `HF_HUB_OFFLINE=1`;
  - `DOCLING_SERVE_LOAD_MODELS_AT_BOOT=false` en `DOCLING_SERVE_OPTIONS_CACHE_SIZE=1`: 364 MiB minder videogeheugen (P5, 30-10).
- Starten en stoppen via llama-swap (ADR 0009). Stoppen is de enige manier om het videogeheugen vrij te geven (P5, 30-10).
- Een Langflow-component stuurt het bestand naar `/v1/convert/file/async`, met `to_formats` met `json` en `ocr_lang` `nl` en `en`. Hij volgt de taak via `/v1/status/poll/{task_id}` en haalt het resultaat op. De pauze tussen twee statusvragen blijft ruim onder de `ttl` van docling in llama-swap; bij HTTP 404 op een taak dient de component hem opnieuw in (P1, 1a, 30-10; regel in ADR 0009).
- **Adres instelbaar (ADR 0019).** Het adres van docling-serve is een instelling van de bouwsteen, per aanroepwijze lokaal of extern. Standaard is het lokale adres via llama-swap (`/upstream/docling/...`). Vóór elke aanroep controleert de bouwsteen de classificatie: een intern document gaat alleen naar een adres dat in het register (ADR 0007) is toegestaan voor intern, en bij een fout valt hij niet terug op een extern adres. Een externe docling-dienst wordt nu niet ingericht; dat vraagt een eigen ADR (ADR 0019).
- OCR met RapidOCR draait in dit image op de CPU (P5, 30-10). Of OCR op de GPU kan, is geen voorwaarde voor fase 2; de keuze van de OCR-engine valt buiten deze ADR.
- Het DoclingDocument uit de JSON-uitvoer (JavaScript Object Notation) gaat naar `ChunkExtractor` (`pipelines/ingestion/src/ingestion/chunking/extractor.py:121`, `extract_chunks` op `:577`) en `regrouper.py`. Die code wordt hergebruikt, niet herschreven.
- De instellingen voor OCR en tabelstructuur die nu in `pipelines/ingestion/src/ingestion/parsers/docling_parser.py` en in `docker-compose.yml:62-81` staan, worden opties in het verzoek. Ze komen in de pipelinedefinitie te staan, niet in Compose.

**WhisperX**

- De AWS-container `public.ecr.aws/deep-learning-containers/whisperx:3.8.6-cu128-amzn2023` (variant A, P1, 30-10), met `OPT_OUT_TRACKING=true`, op het netwerk `verwerking` zonder internet.
- Modellen in volumes: Whisper `large-v3` en het Nederlandse uitlijnmodel in `HF_HOME` (`/opt/models/hf`), het Engelse uitlijnmodel in `TORCH_HOME` (`/opt/models/torch`), NLTK `punkt_tab` in `/root/nltk_data`. Ze worden één keer met internet opgehaald; daarna `HF_HUB_OFFLINE=1`.
- Een Hugging Face-token is voor variant A niet nodig: het sprekermodel zit in het image en de andere modellen zijn niet afgeschermd (P1; proeven.md 1.7). De eerdere regel over een token uit een Docker secret vervalt daarmee.
- Een Langflow-component zet segmenten met tijdcodes en sprekers om in fragmenten. Daarbij wordt het model uit `pipelines/ingestion/src/ingestion/models/transcription.py` hergebruikt.

**Wat vervalt**

- `services/docling/`, de dienst `docling` in Compose, de volumes `docling_input` en `docling_output`, en `DOCLING_SERVICE_URL` en `USE_DOCLING_SERVICE` bij `open_notebook` (`docker-compose.yml:248-249`, `:274-276`).
- `services/whisperx/` en de dienst `whisperx` in Compose.
- In `pipelines/ingestion`:
  - `parsers/docling_parser.py`, `workflow.py`, `cli.py` en `audio/transcriber.py`;
  - de exporters, als niets anders ze nog gebruikt.
  
  Wat blijft: `chunking/`, `models/` en `config.py`, voor zover nog gebruikt.
- De eigen verrijkingen (PyMuPDF-aanvulling, annotaties, handschrift-VLM) vervallen met de dienst. Of ze als aparte bouwsteen terugkomen, beslist ADR 0015.

## Gevolgen

- **Proeven:** uitgevoerd op 30-10.
  - proef 1: keuze van de WhisperX-container (variant A), en starten en stoppen via llama-swap: geslaagd;
  - proef 5: vrijgeven van het videogeheugen: alleen door te stoppen.
- **Tests (stopcondities):**
  - een test die een PDF-bestand uit een openbare bron door docling-serve en `ChunkExtractor` haalt, en sectiepad en paginanummer per fragment controleert;
  - een vergelijkbare test met een openbaar audiobestand voor tijdcodes en sprekers.
- **ARCHITECTURE.md:** paragraaf 1 en 4 (`services/docling`, `services/whisperx`) worden bijgewerkt.
