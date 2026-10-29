# ADR 0015: Eigen verrijkingen en overige diensten in de signaalsysteemomgeving

- **Status:** voorgesteld
- **Datum:** 2026-10-29
- **Raakt FO-paragraaf:** 2 (principe 7 en 8), 7, 11 (bouwsteen Docker-container), 12.1, 12.2
- **Lost op:** verschilanalyse open vraag 4

## Context

Het functioneel ontwerp (FO) noemt voor de extractie alleen docling-serve en WhisperX (FO 7, FO 13.1). Elke taak die de GPU (grafische kaart) gebruikt, loopt via llama-swap (FO 2, principe 3), en elke modelaanroep via de modelpoort. Een Docker-container is een toegestane bouwsteen in Langflow (FO 11).

De fork heeft meer dan het FO noemt. Wat er gebeurt met de eigen verrijkingen en de andere diensten, zegt het FO niet.

| Onderdeel | Locatie | Gebruikt GPU of model |
| --- | --- | --- |
| PyMuPDF-tekstaanvulling | `services/docling/api.py:31-34` | nee |
| Extractie van PDF-annotaties | idem | nee |
| Handschriftherkenning via een vision-language model (VLM) | idem; Ollama `granite3.2-vision` (`docker-compose.yml:82-85`) | ja, via Ollama |
| MinerU met terugvalroute | `services/mineru/`; `apps/app-main/src/app_main/services/parsing/` | ja (`docker-compose.yml:192-198`) |
| GROBID voor literatuurverwijzingen (Track V) | `docker-compose.yml:35-43`, vaste versie `0.8.2-crf` | nee (CPU) |
| Reranker `bge-reranker-v2-m3` (Track W) | `services/reranker/`; `docker-compose.yml:212-233` | standaard op de processor (CPU) |
| Samenvattingsdienst | `services/summarization/`; `docker-compose.yml:89-113` | ja, via Ollama |

Met ADR 0014 vallen de eerste drie weg, samen met `services/docling`. Volgens de configuratiedocumentatie kan docling-serve plaatjes laten beschrijven via een OpenAI-compatibel eindpunt. Dat vraagt `DOCLING_SERVE_ENABLE_REMOTE_SERVICES=true`, en de API-optie daarvoor heet in de gebruiksdocumentatie verouderd. Plaatjesbeschrijving is geen handschriftherkenning.

llama.cpp-server kent `/v1/rerank`, met de vlag `--reranking`. Of llama-swap dat pad doorgeeft, is niet gevonden in de documentatie.

Externe bronnen (geraadpleegd 29-10-2026):

- [docling-serve, configuratie](https://raw.githubusercontent.com/docling-project/docling-serve/main/docs/configuration.md)
- [docling-serve, gebruik](https://raw.githubusercontent.com/docling-project/docling-serve/main/docs/usage.md)
- [llama.cpp, reranking](https://instagit.com/ggml-org/llama.cpp/how-to-implement-reranking-models-for-search-applications)

## Opties

Per onderdeel:

1. **PyMuPDF-aanvulling en annotaties**
   - a) Vervallen.
   - b) Terug als Python-component in Langflow na docling-serve, zonder GPU.
2. **Handschrift-VLM**
   - a) Vervallen.
   - b) Later als bouwsteen achter llama-swap.
3. **MinerU**
   - a) Vervallen.
   - b) Behouden als Docker-container-bouwsteen onder llama-swap, voor documenten waar Docling slecht scoort.
4. **GROBID**
   - a) Vervallen.
   - b) Behouden als Docker-container-bouwsteen op het netwerk `verwerking`, na de extractie van wetenschappelijke PDF's. De code voor `cites` uit Track V blijft dan nodig.
5. **Reranker**
   - a) Vervallen.
   - b) Onder llama-swap, als llama-swap `/v1/rerank` doorgeeft.
   - c) Behouden als CPU-dienst op het interne netwerk, als modelaanroep via `ModelGate` (ADR 0006).
6. **Samenvattingsdienst**
   - a) De dienst vervalt; `pipelines/summarization` blijft als bibliotheek.
   - b) De dienst behouden.

## Voorstel

Uitgangspunt is eenvoud eerst (FO 2, principe 8): in fase 2 alleen wat de minimale keten nodig heeft. Wat nu vervalt, kan later terugkomen als bouwsteen met een eigen proef.

| Onderdeel | Voorstel | Reden |
| --- | --- | --- |
| PyMuPDF-aanvulling, annotaties | 1b, in fase 4 | geen GPU, klein; annotaties kunnen signalen bevatten |
| Handschrift-VLM | 2a | zeldzaam in de brontypen van het FO; nog een GPU-afnemer |
| MinerU | 3a | tweede parser zonder rol in het FO; een GPU-afnemer minder |
| GROBID | 4b | vaste versie, CPU, bestaande code die werkt |
| Reranker | 5c, met 5b als llama-swap het pad doorgeeft | zoekkwaliteit blijft gelijk; geen GPU nodig |
| Samenvattingsdienst | 6a | samenvattingen zijn afgeleid (ADR 0004); wiki-generatie in fase 5 gebruikt de bibliotheek via `ModelGate` |

Geert beslist per onderdeel.

## Gevolgen

- **Bij het voorstel vervallen:**
  - `services/mineru/`, `services/summarization/` en hun diensten in Compose;
  - `apps/app-main/src/app_main/services/parsing/` (al met ADR 0010);
  - de routes voor samenvatten in `services_proxy` (al met ADR 0002);
  - het volume `mineru_models`.
- **GROBID en de reranker** komen in `docker-compose.signaalsysteem.yml` op het netwerk `verwerking`, zonder gepubliceerde poort. De reranker krijgt `HF_HUB_OFFLINE=1` na de eerste download.
- **Proef 1 of een bouwtaak:** nagaan of llama-swap `/v1/rerank` doorgeeft.
