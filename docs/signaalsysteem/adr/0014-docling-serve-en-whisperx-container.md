# ADR 0014: Documentextractie met docling-serve en transcriptie met een bestaande WhisperX-container

- **Status:** besloten
- **Datum:** 2026-10-29
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

## Voorstel

**docling-serve**

- Een container met een vaste versie, bijvoorbeeld `docling-serve-cu128:v1.38.2` of de versie die in proef 5 is getest.
- Instellingen:
  - `DOCLING_SERVE_ARTIFACTS_PATH` op een alleen-lezen volume met vooraf gedownloade modellen;
  - `DOCLING_SERVE_ENABLE_REMOTE_SERVICES=false`;
  - `DOCLING_SERVE_API_KEY` uit een Docker secret;
  - `DOCLING_SERVE_ENG_LOC_NUM_WORKERS=1`, één taak tegelijk op de GPU;
  - `HF_HUB_OFFLINE=1`.
- Starten en stoppen via llama-swap (ADR 0009).
- Een Langflow-component stuurt het bestand naar `/v1/convert/file/async`, met `to_formats` met `json` en `ocr_lang` `nl` en `en`. Hij volgt de taak via `/v1/status/poll/{task_id}` en haalt het resultaat op.
- Het DoclingDocument uit de JSON-uitvoer (JavaScript Object Notation) gaat naar `ChunkExtractor` (`pipelines/ingestion/src/ingestion/chunking/extractor.py:121`, `extract_chunks` op `:577`) en `regrouper.py`. Die code wordt hergebruikt, niet herschreven.
- De instellingen voor OCR en tabelstructuur die nu in `pipelines/ingestion/src/ingestion/parsers/docling_parser.py` en in `docker-compose.yml:62-81` staan, worden opties in het verzoek. Ze komen in de pipelinedefinitie te staan, niet in Compose.

**WhisperX**

- De gekozen container, met een vaste versie.
- Het Hugging Face-token komt alleen bij de eerste download binnen, uit een Docker secret. Daarna `HF_HUB_OFFLINE=1`.
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

- **Proeven:**
  - proef 1: keuze van de WhisperX-container, en starten en stoppen via llama-swap;
  - proef 5: vrijgeven van het videogeheugen.
- **Tests (stopcondities):**
  - een test die een PDF-bestand uit een openbare bron door docling-serve en `ChunkExtractor` haalt, en sectiepad en paginanummer per fragment controleert;
  - een vergelijkbare test met een openbaar audiobestand voor tijdcodes en sprekers.
- **ARCHITECTURE.md:** paragraaf 1 en 4 (`services/docling`, `services/whisperx`) worden bijgewerkt.
