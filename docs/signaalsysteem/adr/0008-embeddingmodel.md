# ADR 0008: Embeddingmodel onder llama-swap

- **Status:** voorgesteld
- **Datum:** 2026-10-29
- **Raakt FO-paragraaf:** 7 (kwaliteitsborging, referentieset), 9 (lokale embeddings), 12.2 (embeddingmodel licht en vast geladen)
- **Lost op:** verschilanalyse open vraag 8

## Context

Het functioneel ontwerp (FO) schrijft voor:

- embeddings worden lokaal berekend (FO 9);
- het embeddingmodel draait in llama-swap in een lichte, vaste groep (FO 12.2).

Welk model het wordt, zegt het FO niet.

De fork gebruikt in de praktijk `mxbai-embed-large` met 1024 dimensies:

- `packages/shared/src/shared/models/source.py:130`;
- `packages/shared/src/shared/model_routing.yaml:102`;
- `packages/semantic-intelligence/src/semantic_intelligence/config.py:30`;
- `pipelines/embeddings/tests/test_service_ollama_context.py:32-33`.

`ARCHITECTURE.md` paragraaf 8 noemt `nomic-embed-text` met 768 dimensies. Dat is verouderde documentatie. De migraties leggen geen dimensie vast (zoeken op `DIMENSION` in `migrations/` levert niets op).

Onder llama-swap draait het model als GGUF in llama-server, met `/v1/embeddings`. Van `mxbai-embed-large-v1` bestaan GGUF-bestanden uit de gemeenschap, onder meer van ChristianAzinn. Een andere runtime en kwantisatie geven niet precies dezelfde vectoren als Ollama. De bestaande index moet dus hoe dan ook opnieuw worden berekend. Onder ADR 0003 is dat een nieuwe run, en de oude embeddings blijven staan.

Niet geverifieerd:

- welke pooling dit model in llama-server nodig heeft;
- of `mxbai-embed-large` goed werkt voor Nederlandse tekst.

Externe bronnen (geraadpleegd 29-10-2026):

- [llama.cpp, embeddings](https://www.mintlify.com/ggml-org/llama.cpp/inference/embeddings)
- [mxbai-embed-large-v1 als GGUF](https://huggingface.co/ChristianAzinn)

## Opties

1. **`mxbai-embed-large-v1` als GGUF** (1024 dimensies).
   - Voor: code en tests gaan uit van 1024 dimensies; bekend gedrag in de fork.
   - Tegen: geschikt voor Nederlands niet aangetoond; GGUF alleen uit de gemeenschap.
2. **Een meertalig model** met GGUF-ondersteuning in llama-server.
   - Voor: mogelijk betere zoekresultaten op een Nederlands corpus.
   - Tegen: kandidaat en dimensie nog te kiezen; tests en drempels (`minimum_score`, `apps/app-main/src/app_main/api/schemas.py:110`) moeten opnieuw worden afgesteld.
3. **`nomic-embed-text`** (768 dimensies).
   - Tegen: geen voordeel tegenover optie 1, en wel een dimensiewissel.

## Voorstel

Begin met optie 1 als standaard in fase 2, omdat de code daarvan uitgaat. Vergelijk optie 1 en twee meertalige kandidaten op het zoekdeel van de referentieset (FO 7) en meet het videogeheugen in proef 3. Kies daarna. Omdat de index toch opnieuw wordt berekend, kost een wissel na de meting alleen rekentijd.

Geert beslist over de kandidaten en het meetcriterium.

## Gevolgen

- Het register (ADR 0007) en de llama-swap-configuratie (ADR 0006) krijgen één embeddingmodel. Dat model wordt gebruikt voor fragmenten én zoekvragen; dezelfde code in `pipelines/embeddings` doet beide.
- `ARCHITECTURE.md` paragraaf 8 wordt rechtgezet.
- Bij een andere dimensie dan 1024 worden de tests die 1024 vastleggen aangepast, en wordt de index opnieuw berekend als nieuwe run.
- Input voor proef 3: het videogeheugen van het gekozen embeddingmodel in de vaste groep.
