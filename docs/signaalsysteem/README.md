# Signaalsysteem

Deze map bevat het ontwerp en de werkopdracht voor het signaalsysteem: een modulair systeem dat bronnen automatisch binnenhaalt, verwerkt tot gelabelde fragmenten en opslaat met versie en herkomst. De fork van Open Notebook vormt de kennislaag.

## Inhoud

| Bestand | Inhoud |
| --- | --- |
| `functioneel-ontwerp.md` | het functioneel ontwerp (FO), met platen in `diagrammen/` |
| `opdracht.md` | de werkopdracht voor Claude Code: uitgangspunten, werkwijze en de stappen van fase 1 |
| `voortgang.md` | stand van het werk, bijgewerkt na elke taak |
| `adr/` | ontwerpbesluiten (Architecture Decision Records) |

De agentdefinities staan in `.claude/agents/` (`signaal-planner`, `signaal-ontwerper`, `signaal-bouwer`, `signaal-reviewer`). De startopdracht is de skill `.claude/skills/signaalsysteem/`.

## Ontwerpbesluiten

| ADR | Onderwerp | Status |
| --- | --- | --- |
| [0001](adr/0001-classificatie-intern-extern.md) | Classificatie intern/extern, standaard intern, met overerving | besloten |
| [0002](adr/0002-afscherming-api-mcp-compose.md) | Afscherming van de API, de MCP-routes en de Compose-omgeving | besloten |
| [0003](adr/0003-bronversie-run-herkomst.md) | Bronversie, verwerkingsrun en herkomst; herverwerking zonder overschrijven | besloten |
| [0004](adr/0004-afgeleide-teksten-uit-zoekresultaten.md) | Afgeleide teksten uit de zoekresultaten | besloten |
| [0005](adr/0005-handmatige-fragmentbewerking.md) | Handmatige fragmentbewerking | besloten |
| [0006](adr/0006-llama-swap-en-modelkoppeling.md) | llama-swap als lokale modelserver en één modelpoort | besloten |
| [0007](adr/0007-modelregister-en-auditlog.md) | Plaats van het modelregister en het auditlog van modelaanroepen | besloten |
| [0008](adr/0008-embeddingmodel.md) | Embeddingmodel onder llama-swap | besloten (mxbai-embed-large Q8_0) |
| [0009](adr/0009-gpu-beheer-en-docker-toegang.md) | GPU-beheer: docling-serve, WhisperX en Kev onder llama-swap, en Docker-toegang | besloten |
| [0010](adr/0010-een-inname-via-importeindpunt.md) | Eén inname via het importeindpunt | besloten |
| [0011](adr/0011-orkestratie-na-de-import.md) | Orkestratie van de stappen na de import | besloten |
| [0012](adr/0012-entiteitextractie-en-deduplicatie-als-langgraph.md) | Entiteitextractie en deduplicatie als LangGraph-modules; één extractiepad | besloten |
| [0013](adr/0013-referentielijsten-iv3-en-bwb.md) | Referentielijsten voor Iv3-taakvelden en BWB-id's | besloten |
| [0014](adr/0014-docling-serve-en-whisperx-container.md) | Documentextractie met docling-serve en transcriptie met een bestaande WhisperX-container | besloten |
| [0015](adr/0015-verrijkingen-en-overige-diensten.md) | Eigen verrijkingen en overige diensten in de signaalsysteemomgeving | besloten |
| [0016](adr/0016-interface-en-reviewwachtrijen.md) | Interface en de plek waar de beheerder zijn wachtrijen afhandelt | besloten |
| [0017](adr/0017-langflow-inrichting.md) | Inrichting van Langflow: flows uit Git, aanmelding, planning en MCP | besloten |
| [0018](adr/0018-obsidian-export-als-langflow-flow.md) | Obsidian-export als Langflow-flow met een gegenereerd blok | besloten |
| [0019](adr/0019-classificatie-per-innameroute.md) | Classificatie en pipelinestart per innameroute; externe verwerking voor extern | besloten |
| [0020](adr/0020-bronkaart-als-tabel-in-de-fork.md) | De bronkaart als tabel in de fork, alleen via de API | besloten |
| [0021](adr/0021-surrealdb-v3.md) | SurrealDB v3 | besloten: `v2.7.0`, overstap later (na P10) |
| [0022](adr/0022-afscherming-van-langflow.md) | Afscherming van Langflow | besloten |

`adr/0000-sjabloon.md` is het sjabloon.

## Starten in Claude Code

Open de repository in Claude Code en typ:

```
/signaalsysteem
```

Zonder toevoeging voert Claude Code fase 1 uit: verschilanalyse, ontwerpbesluiten, proeven en plan van aanpak. Daarna stopt het werk tot Geert akkoord geeft. Een latere fase of taak start met een toevoeging, bijvoorbeeld `/signaalsysteem fase 2` of `/signaalsysteem taak F2-03`.

De vraag "werk het signaalsysteem uit" werkt ook: de skill wordt dan op basis van zijn beschrijving ingezet.

## Beheer

- **Starten en stoppen:** alleen met `scripts/signaalsysteem/start.sh` en `scripts/signaalsysteem/stop.sh`, nooit tegelijk met de oude stack.
- **Referentiegegevens laden:** `bash scripts/signaalsysteem/laad_referentiegegevens.sh`, met de omgeving gestart. De stap downloadt op de host de registers uit `config/referentiegegevens.yaml` en controleert hun SHA-256. Daarna zet hij ze in de container op `/app/data/referentiegegevens` en laadt hij ze in `reference_entity`. Herhalen kan: de aantallen blijven gelijk. Doe dit na een lege database en na een wijziging van de inventaris. Zonder deze stap geeft `POST /api/vocabulary/refresh` een 503, omdat de container geen internet heeft.
- **Controle:** `bash scripts/signaalsysteem/controle/referentiegegevens.sh` draait de stap twee keer en toetst de aantallen. Gebruik daarvoor een toetsdatabase.
- **Modellen laden:** `bash scripts/signaalsysteem/modellen_laden.sh`, vóór de eerste start. De stap vult de volumes `signaal_modellen` (werkmodel en embeddingmodel als GGUF) en `signaal_docling_artifacts` (Docling-modellen). Een GGUF-bestand wordt van Hugging Face gehaald op de vaste revisie in `scripts/signaalsysteem/modellen_downloaden.py`, met controle van de SHA-256. De Docling-modellen komen uit `docling-tools models download` in het vaste docling-serve-image, gecontroleerd met één hash over de hele boom. Met `--van-proef` kopieert de stap de modellen uit de proefvolumes, die alleen-lezen worden gekoppeld. Een bestaand bestand met een andere hash wordt niet overschreven. De volumes zijn extern, dus `stop.sh` en `down -v` laten ze staan. `start.sh` draait elke keer `modellen_laden.sh --controle` (zonder netwerk) en start niets als een hash afwijkt. Ruim een afwijkend bestand dan zelf op en draai de stap opnieuw. Een nieuwe modelversie vraagt een nieuwe revisie en hash in het downloadscript, en een nieuwe versie in `config/modelregister.yaml`; `test_modelregister_consistent.py` toetst dat die bij elkaar passen.

## Stand

Alle twaalf ontwerpkeuzes in paragraaf 14 van het FO zijn besloten. Fase 1 loopt; zie `voortgang.md`.
