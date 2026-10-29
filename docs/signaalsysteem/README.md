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

## Starten in Claude Code

Open de repository in Claude Code en typ:

```
/signaalsysteem
```

Zonder toevoeging voert Claude Code fase 1 uit: verschilanalyse, ontwerpbesluiten, proeven en plan van aanpak. Daarna stopt het werk tot Geert akkoord geeft. Een latere fase of taak start met een toevoeging, bijvoorbeeld `/signaalsysteem fase 2` of `/signaalsysteem taak F2-03`.

De vraag "werk het signaalsysteem uit" werkt ook: de skill wordt dan op basis van zijn beschrijving ingezet.

## Stand

Alle twaalf ontwerpkeuzes in paragraaf 14 van het FO zijn besloten. Fase 1 is nog niet gestart.
