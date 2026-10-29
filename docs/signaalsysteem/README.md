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
| [0005](adr/0005-handmatige-fragmentbewerking.md) | Handmatige fragmentbewerking | voorgesteld |
| [0006](adr/0006-llama-swap-en-modelkoppeling.md) | llama-swap als lokale modelserver en één modelpoort | besloten |
| [0007](adr/0007-modelregister-en-auditlog.md) | Plaats van het modelregister en het auditlog van modelaanroepen | voorgesteld |
| [0008](adr/0008-embeddingmodel.md) | Embeddingmodel onder llama-swap | voorgesteld |
| [0009](adr/0009-gpu-beheer-en-docker-toegang.md) | GPU-beheer: docling-serve, WhisperX en Kev onder llama-swap, en Docker-toegang | voorgesteld |
| [0010](adr/0010-een-inname-via-importeindpunt.md) | Eén inname via het importeindpunt | besloten |
| [0011](adr/0011-orkestratie-na-de-import.md) | Orkestratie van de stappen na de import | voorgesteld |
| [0012](adr/0012-entiteitextractie-en-deduplicatie-als-langgraph.md) | Entiteitextractie en deduplicatie als LangGraph-modules; één extractiepad | besloten |
| [0013](adr/0013-referentielijsten-iv3-en-bwb.md) | Referentielijsten voor Iv3-taakvelden en BWB-id's | voorgesteld |
| [0014](adr/0014-docling-serve-en-whisperx-container.md) | Documentextractie met docling-serve en transcriptie met een bestaande WhisperX-container | besloten |
| [0015](adr/0015-verrijkingen-en-overige-diensten.md) | Eigen verrijkingen en overige diensten in de signaalsysteemomgeving | voorgesteld |
| [0016](adr/0016-interface-en-reviewwachtrijen.md) | Interface en de plek waar de beheerder zijn wachtrijen afhandelt | voorgesteld |
| [0017](adr/0017-langflow-inrichting.md) | Inrichting van Langflow: flows uit Git, aanmelding, planning en MCP | voorgesteld |
| [0018](adr/0018-obsidian-export-als-langflow-flow.md) | Obsidian-export als Langflow-flow met een gegenereerd blok | besloten |

`adr/0000-sjabloon.md` is het sjabloon.

## Starten in Claude Code

Open de repository in Claude Code en typ:

```
/signaalsysteem
```

Zonder toevoeging voert Claude Code fase 1 uit: verschilanalyse, ontwerpbesluiten, proeven en plan van aanpak. Daarna stopt het werk tot Geert akkoord geeft. Een latere fase of taak start met een toevoeging, bijvoorbeeld `/signaalsysteem fase 2` of `/signaalsysteem taak F2-03`.

De vraag "werk het signaalsysteem uit" werkt ook: de skill wordt dan op basis van zijn beschrijving ingezet.

## Stand

Alle twaalf ontwerpkeuzes in paragraaf 14 van het FO zijn besloten. Fase 1 loopt; zie `voortgang.md`.
