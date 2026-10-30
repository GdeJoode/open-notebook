# ADR 0006: llama-swap als lokale modelserver en één modelpoort

- **Status:** besloten
- **Datum:** 2026-10-29
- **Raakt FO-paragraaf:** 2 (principe 3), 9, 11 (modelcomponenten), 12.1, 12.2, 13.2 punt 5, 14 (lokale modelserver)
- **Lost op:** verschilanalyse conflict 1 en 2, open vraag 7 (de rol van de Track J-routering)

## Context

Het functioneel ontwerp (FO) legt vast:

- elke modelaanroep loopt via één modelpoort (FO 2, principe 3);
- lokale modellen draaien via llama-swap met llama-server (GGUF), en Kev via zijn eigen server (FO 12.2, FO 14);
- Open Notebook en Langflow praten alleen met llama-swap, nooit rechtstreeks met een modelserver (FO 12.2);
- "Esperanto wijst naar llama-swap; dat is configuratie, geen code" (FO 13.2 punt 5);
- de container `open-notebook` hangt alleen aan het interne netwerk, en alleen `langflow` en `claude-cli` krijgen uitgaand internet (FO 12.1). Dat betekent: de fork kán geen cloudmodel bereiken;
- de modelkeuze in de Langflow-componenten is een van de drie plekken waar het beleid wordt afgedwongen (FO 9).

Harde eis 4: alles draait in Docker, en GPU-diensten (grafische kaart) lopen via llama-swap.

De fork roept modellen langs vier routes aan (verschilanalyse 3.3):

1. Esperanto via `ModelManager` (`packages/llm-manager/src/llm_manager/manager.py`; `providers/factory.py`, met een `openai-compatible`-route op `:41-62`).
2. De Track J-routering in `apps/app-main/src/app_main/services/model_routing/`: resolver, failover, circuit breaker, rate limiter, privacyresolver en telemetrie. De standaard is de cloudketen `["nvidia"]` (`route_resolver.py:84-85`).
3. `packages/shared/src/shared/model_routing.py` met Ollama-eigen eindpunten (`:284` `/api/generate`, `:353` `/api/embed`). Aanroepers zijn `route_resolver.py`, `apps/app-main/src/app_main/services/preprocessing_service.py`, `api/routers/services_proxy.py`, `api/app.py`, `packages/shared/src/shared/config_coherence.py` en de extractiedienst.
4. Directe HTTP-aanroepen naar Ollama:
   - `pipelines/entity-filtering/src/entity_filtering/resolution/llm_matcher.py:349` (`/api/chat`);
   - `packages/semantic-intelligence/src/semantic_intelligence/config.py:64` (`/api/embed`);
   - de handschrift-VLM (vision-language model) in de Docling-dienst;
   - de samenvattingsdienst (`docker-compose.yml:82-84`, `:101-108`).

Ollama draait op de host (`docker-compose.yml:84`, `:103`, `:108`, `:154`, `:246-247`).

**llama-swap (gecontroleerd).** De README noemt als eindpunten `v1/completions`, `v1/chat/completions`, `v1/responses`, `v1/embeddings` en `v1/models`, eindpunten voor audio en beeld, en Anthropic-eindpunten. Ollama-eigen eindpunten (`/api/generate`, `/api/chat`, `/api/embed`) staan er niet bij; de enige `/api/`-paden zijn beheerpaden van llama-swap zelf. Aanname A1 klopt daarmee voor zover de documentatie het zegt. llama-swap werkt met elke OpenAI-compatibele server. Volgens de releasepagina is v268 van 28 oktober de nieuwste versie; de pagina noemt geen jaar.

Externe bronnen (geraadpleegd 29-10-2026):

- [llama-swap, README](https://github.com/mostlygeek/llama-swap)
- [llama-swap, releases](https://github.com/mostlygeek/llama-swap/releases)

## Opties

Het FO kiest llama-swap. Open was alleen of de Track J-routering de basis van de modelpoort wordt (open vraag 7):

1. **Track J houden als modelpoort in de fork.** De cloudketen, failover en rate limiting zijn gebouwd voor cloudaanroepen vanuit de fork. Die zijn onder FO 12.1 onmogelijk. Wat overblijft, is een privacycontrole op een pad dat alleen lokaal kan zijn.
2. **Eén modelpoort als Python-module in `packages/llm-manager`, gebruikt door de fork én door de Langflow-componenten.** De beleidscontrole uit Track J wordt hergebruikt; de cloudketen vervalt. Gekozen.

## Voorstel

Optie 2.

**llama-swap in Compose**

- Container `llama-swap` op het netwerk `verwerking` (ADR 0002). Het image ligt vast op een uitgebrachte versie met CUDA 13: voor de RTX 5070 Ti (Blackwell) dekt `unified-cuda` (CUDA 12) volgens de README alleen Pascal tot en met Ada (proeven.md §5 punt 2). Beproefd is `ghcr.io/mostlygeek/llama-swap:v268-cuda13-b11977`, in een afgeleid image met de Docker-client (ADR 0009; P1, P7, 30-10). Die tag eist `cuda>=13.4`; met driver 610.88 (CUDA 13.3) start hij alleen met `NVIDIA_DISABLE_REQUIRE=1` en `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64` (P1, proeven.md §8 punt 23). De modelbestanden staan in een alleen-lezen volume. De configuratie staat onder versiebeheer in de repository, op een plek die de bouwtaak kiest, bijvoorbeeld `services/llama-swap/config.yaml`.
- De groepen en het laden van docling-serve, WhisperX en Kev staan in ADR 0009; het embeddingmodel in ADR 0008.

**Modelpoort**

- Nieuwe module `llm_manager.gate` met een klasse `ModelGate`. Alle modelaanroepen uit de fork en uit de Langflow-componenten gaan hierdoorheen. De componenten importeren het pakket; het pakket wordt in het Langflow-image geïnstalleerd.
- Per aanroep doet de poort drie dingen:
  - het model opzoeken in het modelregister (ADR 0007);
  - de classificatie van de invoer toetsen: is die intern en is het model niet toegestaan voor intern, dan weigert de poort, zonder terugval;
  - een auditregel schrijven (ADR 0007).
- Hergebruikt wordt de weigerlogica voor cloudproviders in `route_resolver.py:199-216` en de telemetrie in `telemetry.py`, die in ADR 0007 opgaat.
- Aanroepwijzen: `llama-swap` (OpenAI-compatibel, via Esperanto `openai-compatible` met `base_url=http://llama-swap:<poort>/v1`), `api` (Claude-API, alleen vanuit Langflow) en `cli` (`claude -p` via de container `claude-cli`, FO 9; uitwerking in fase 5).
- In de fork staan alleen modellen met aanroepwijze `llama-swap` in de configuratie. Een poging om vanuit de fork een cloudmodel aan te roepen faalt op het netwerk (ADR 0002) én in de poort.

**Ombouw van de vier routes**

- Route 1 (Esperanto): de provider `ollama` maakt plaats voor `openai-compatible` naar llama-swap. De modelnamen worden model-id's uit de llama-swap-configuratie.
- Route 2 (Track J): verwijderd worden:
  - `route_resolver.py` (behalve de hergebruikte weigerlogica), `failover_executor.py`, `circuit_breaker.py`, `rate_limiter.py`, `error_mapping.py`, `privacy_resolver.py`, `seed.py` en `seed_chain_models.py`;
  - de router `model_routes`;
  - de tabel `model_route` (`migrations/51.surrealql:31`), via een nieuwe migratie met `_down`.
  
  `llm_call.py` wordt de dunne aanroeper van `ModelGate`. `LLMTask` blijft als taaknaam in de auditregel. Chat (`provision_langchain_model`) gaat via de poort naar een lokaal model.
- Route 3: `packages/shared/src/shared/model_routing.py` en `model_routing.yaml` vervallen. Hun aanroepers gaan over op `ModelGate`. De extractiedienst vervalt helemaal (ADR 0012).
- Route 4:
  - `llm_matcher.py` gebruikt de chat van `ModelGate` (`/v1/chat/completions`);
  - `semantic_intelligence/config.py` gebruikt de embeddingdienst uit `pipelines/embeddings` (`/v1/embeddings`);
  - de handschrift-VLM en de samenvattingsdienst vallen onder ADR 0015.
- Ollama verdwijnt uit de keten: alle verwijzingen naar `host.docker.internal:11434` en `extra_hosts` vervallen.

## Gevolgen

- **Code:** grote opruiming in `apps/app-main/src/app_main/services/model_routing/` en `packages/shared`; nieuwe `ModelGate` in `packages/llm-manager`; kleine wijzigingen in `pipelines/entity-filtering` en `packages/semantic-intelligence`. FO 13.2 punt 5 ("alleen configuratie") klopt voor route 1; routes 2 tot en met 4 vragen code. Dat volgt uit de fork, niet uit het FO.
- **Interface:** het scherm voor providerketens (Track J, `docs/tracks/J-model-routing/OPERATOR_GUIDE.md`) werkt niet meer (ADR 0016).
- **Tests (stopcondities):**
  - een test die faalt als de code onder `apps/`, `packages/` of `pipelines/` nog `/api/generate`, `/api/chat`, `/api/embed` of `11434` bevat;
  - een test dat `ModelGate` een cloudmodel weigert bij invoer met classificatie intern, ook als het lokale model faalt;
  - de bestaande `test_embedding_local_guardrail.py` blijft en wordt aangepast aan de nieuwe poort.
- **Documentatie:** `ARCHITECTURE.md` paragraaf 8 wordt herschreven.
