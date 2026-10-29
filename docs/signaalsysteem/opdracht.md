# Opdracht: signaalsysteem uitwerken

Deze opdracht zet het functioneel ontwerp (FO) in `functioneel-ontwerp.md` om in een uitvoerbaar plan en, na akkoord van Geert, in werkende code. Starten kan met `/signaalsysteem` of met de vraag het signaalsysteem uit te werken.

## 1. Uitgangspunten

- **Het FO is leidend.** De besluiten in paragraaf 14 van het FO staan vast. Heropen ze alleen met een voorstel aan Geert.
- **De fork wijkt af van het oorspronkelijke Open Notebook.** `ARCHITECTURE.md` beschrijft de huidige opbouw: een monorepo met `apps/`, `packages/`, `pipelines/` en `services/`. Volg de conventies daaruit: naamgeving Service, Orchestrator, Processor en Repository; idempotente migraties met een `_down`-variant; tests naast de code.
- **Hergebruik gaat boven zelf bouwen** (FO paragraaf 2 en 13). Bouw alleen wat geen bestaande software en geen bestaande code in de fork al doet.
- **Harde eisen.** Deze worden nooit afgezwakt:
  1. Intern blijft lokaal. Een fragment met label intern bereikt nooit een cloudmodel en verschijnt nooit in een uitvraag via de MCP-server (Model Context Protocol). Dit wordt op drie plekken afgedwongen: het Docker-netwerk, de modelkeuze in de Langflow-componenten en het filter in de API van de fork.
  2. Herkomst: elk fragment is herleidbaar tot bronversie, verwerkingsrun en model.
  3. Afgeleide teksten, zoals notities en wiki-pagina's, tellen nooit als bron.
  4. Alles draait lokaal in Docker; GPU-diensten lopen via llama-swap.
- **Geen interne documenten** in de repository of in tests. Testdata komt alleen uit openbare bronnen.
- **Geen geheimen** in code, Compose-bestanden of commits.

## 2. Werkwijze

- Werk op een eigen branch, nooit op `main`. Maak kleine commits met een duidelijke boodschap.
- Elke taak start met verse context. Voortgang staat in `voortgang.md` in deze map, niet in het gesprek.
- Elke taak heeft een stopconditie buiten het model: een test, een schemacontrole of een meting.
- Documentatie is in het Nederlands, zakelijk en neutraal. Afkortingen staan bij eerste gebruik voluit, daarna alleen de afkorting.
- Bij twijfel over het FO of een conflict met bestaande code: leg het vast als ontwerpbesluit (Architecture Decision Record, ADR) met status "voorgesteld" en vraag Geert. Bouw er niet op door.
- Proeven die een GPU of de werkplek van Geert nodig hebben en hier niet kunnen draaien: schrijf ze uit als reproduceerbaar script met verwachte uitkomst en markeer ze als "uit te voeren op de werkplek".

## 3. Fase 1: plan van aanpak en proeven

Voer de stappen in volgorde uit. Elke stap levert een bestand op in deze map.

| Stap | Agent | Resultaat |
| --- | --- | --- |
| 1. Verschilanalyse | `signaal-planner` | `verschilanalyse.md` |
| 2. Ontwerpbesluiten | `signaal-ontwerper` | `adr/NNNN-<onderwerp>.md`, één per besluit |
| 3. Proeven | `signaal-ontwerper`, daarna `signaal-bouwer` | `proeven.md` en scripts in `scripts/signaalsysteem/proeven/` |
| 4. Plan van aanpak | `signaal-planner` | `plan-van-aanpak.md` |
| 5. Onafhankelijke review | `signaal-reviewer` | bevindingen, verwerkt in de stukken hierboven |

### Stap 1. Verschilanalyse

Vergelijk elk onderdeel van het FO met de fork. Geef per onderdeel aan: aanwezig, gedeeltelijk, ontbreekt of conflicteert, met bestandsverwijzingen. Kijk in elk geval naar:

- `services/docling` en `services/whisperx` tegenover docling-serve en een WhisperX-container onder llama-swap (FO 12 en 13.1);
- `packages/job-queue` tegenover de planning in Langflow (schema-trigger of APScheduler) en de verwerkingsruns (FO 5 en 8);
- `packages/llm-manager` en de huidige modelkoppeling tegenover llama-swap en Kev (FO 9 en 12.2);
- `pipelines/ingestion`, `pipelines/embeddings` en `pipelines/retrieval` tegenover de pipeline uit FO 4 en 7;
- `pipelines/entity-filtering` en `pipelines/ontology-extraction` tegenover entiteitextractie en deduplicatie als LangGraph-modules (FO 7 en 11);
- het datamodel in `migrations/` en `packages/surrealdb-service` tegenover FO 8: bronversie, verwerkingsrun, herkomst en classificatie;
- de toegangsbeveiliging en de API tegenover het classificatiefilter (FO 9 en 10);
- onderdelen die volgens het FO niet meer gebruikt worden, zoals chat, ask, podcasts en de interface: die blijven staan en worden niet uitgebreid (FO 13.2).

Sluit af met een lijst van conflicten en open vragen.

### Stap 2. Ontwerpbesluiten

Schrijf voor elk conflict, en voor elke keuze die het FO openlaat, een ADR met context, opties, voorstel, gevolgen en status. Gebruik status "besloten" alleen als het FO de keuze al maakt; anders "voorgesteld".

### Stap 3. Proeven

Werk de proeven uit FO 13.1 uit, elk met aanpak, succescriterium en uitkomst:

1. docling-serve, WhisperX en Kev als containers die llama-swap start en stopt, inclusief de afweging of llama-swap toegang tot Docker mag krijgen;
2. de schema-trigger van Langflow in de gekozen versie, met APScheduler als terugval;
3. het geheugengebruik per dienst op de RTX 5070 Ti (16 GB), als basis voor de groepsindeling in FO 12.2;
4. open-notebook-mcp met het classificatiefilter in de API: geen intern fragment in een antwoord;
5. een docling-serve-taak die het videogeheugen na afloop vrijgeeft;
6. de feedcontrole voor een YouTube-kanaal en voor een Spotify-podcast via de openbare RSS-feed (FO 5).

### Stap 4. Plan van aanpak

`plan-van-aanpak.md` bevat:

- de fasen 2 tot en met 5 uit FO 15, opgedeeld in taken met id, beschrijving, afhankelijkheden, te wijzigen bestanden, stopconditie en agent;
- de Compose-opzet als schets van `docker-compose.signaalsysteem.yml`: diensten, een netwerk zonder internettoegang voor de verwerking, volumes en geheimen;
- de interfaces: het importeindpunt van de fork, het schema van de bronkaart, het classificatiefilter en het contract tussen de Langflow-componenten en de fork;
- risico's en open vragen.

### Stap 5. Review

`signaal-reviewer` kent de bouwcontext niet en toetst de stukken aan het FO en de harde eisen. Verwerk de bevindingen.

### Afronding van fase 1

Leg het resultaat voor aan Geert, met een korte samenvatting en de open vragen. Begin niet aan fase 2 zonder zijn akkoord.

## 4. Latere fasen

Na akkoord voert `signaal-bouwer` de taken uit het plan van aanpak uit, één taak tegelijk. `signaal-reviewer` controleert elke afgeronde taak. Werk `voortgang.md` bij na elke taak.
