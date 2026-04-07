# Open Notebook — Applicatiebeschrijving

## Wat is Open Notebook?

Open Notebook is een privacy-first platform voor documentintelligentie. Het stelt gebruikers in staat om documenten, audio- en video-opnames te uploaden, te organiseren in notebooks, en er vervolgens gestructureerde kennis uit te onttrekken. De applicatie is gebouwd als alternatief voor cloud-gebonden tools zoals Google NotebookLM, met als kernprincipe dat alle verwerking lokaal plaatsvindt — inclusief GPU-versnelde documentparsing en spraaktranscriptie.

De applicatie combineert drie kernfuncties: (1) het omzetten van ongestructureerde bronnen naar doorzoekbare, geïndexeerde content, (2) het extraheren van gestructureerde kennis in de vorm van entiteiten en relaties, en (3) het bieden van een conversationele interface waarmee gebruikers vragen kunnen stellen over hun verzamelde bronnen.

## Architectuur

Open Notebook is opgezet als een modulair monolith binnen een UV workspace. De architectuur bestaat uit vier lagen die elk een helder afgebakende verantwoordelijkheid hebben.

**Frontend (Next.js 15)** — Een React 19-applicatie met Tailwind CSS en Radix UI-componenten. De gebruikersinterface biedt een stapsgewijze wizard voor het verwerken van bronnen: van upload, via extractie en entiteitherkenning, tot samenvatting en opslag. Realtime voortgang wordt getoond via Server-Sent Events. De frontend draait op poort 8502.

**Backend API (FastAPI)** — Een async-first Python API op poort 5055 die alle businesslogica orkestreert. Routes zijn georganiseerd rond bronnen, notebooks, chat, zoekopdrachten en extractie. Langlopende taken zoals documentparsing, embedding-generatie en entiteitextractie worden via een command queue asynchroon verwerkt door achtergrondworkers.

**Gespecialiseerde Pipelines** — Onafhankelijke verwerkingspijplijnen voor specifieke taken:
- *Ingestion*: documentparsing via IBM's GraniteDocling VLM (een compact 258M-parameter vision-language model dat layout-analyse, OCR, tabelherkenning en beeldextractie in één pass afhandelt) en audiotranscriptie via WhisperX met speaker diarization.
- *Embeddings*: vectorgeneratie voor semantisch zoeken via Sentence-Transformers of cloud-providers.
- *Ontology-extraction*: gestructureerde kennisextractie gestuurd door ontologieën (zie hieronder).
- *Summarization*: geautomatiseerde samenvattingen en inzichten.

**Gedeelde Packages** — Herbruikbare bibliotheken die door zowel de API als de pipelines worden gebruikt: `shared` (Pydantic v2-modellen en enums), `surrealdb-service` (database-abstractie), `llm-manager` (routering naar 16+ LLM-providers waaronder OpenAI, Claude, Gemini, Groq, Mistral en lokale Ollama-modellen), `ontology-manager` (ontologiebeheer en promptgeneratie), `file-manager` en `job-queue`.

**Database (SurrealDB)** — Een multi-model database op poort 8000 die zowel relationele, document- als grafenqueries ondersteunt. Dit maakt het mogelijk om bronnen, chunks, embeddings, extractieresultaten en kennisgrafen in één systeem op te slaan en te bevragen.

## Verwerkingsflow

Wanneer een gebruiker een document uploadt, doorloopt het de volgende stappen:

1. **Bronregistratie** — De frontend stuurt het bestand naar de API, die een Source-record aanmaakt in SurrealDB en een verwerkingscommando op de queue plaatst.
2. **Ingestion** — Het document wordt geparseerd: tekst, tabellen en afbeeldingen worden geëxtraheerd. Voor audio en video wordt WhisperX ingezet met speaker diarization op de lokale GPU.
3. **Chunking** — De geëxtraheerde content wordt opgedeeld in semantische chunks met behoud van elementtype (paragraaf, tabel, afbeelding, koptekst) en positie-informatie.
4. **Embedding** — Elke chunk krijgt een vectorrepresentatie voor semantisch zoeken.
5. **Entiteit- en relatieextractie** — Chunks worden geanalyseerd op entiteiten en hun onderlinge relaties, gestuurd door een ontologie.
6. **Verrijking** — Samenvattingen, kernpunten en inzichten worden gegenereerd. De bron is nu volledig doorzoekbaar en bevraagbaar via chat.

De frontend toont elke stap als een tabblad in een drievoudig paneel: bestandslijst links, voortgangslogboek in het midden, en resultaten rechts.

## Ontologie-gestuurde entiteit- en relatieextractie

### Waarom ontologie-gestuurde extractie?

Documenten bevatten impliciete kennisstructuren die niet zichtbaar worden door alleen tekst te indexeren. Een beleidsdocument vermeldt organisaties, personen, regio's en programma's — maar de *relaties* tussen deze entiteiten (wie financiert wat, welke organisatie is verantwoordelijk voor welk programma, welke regio valt onder welk beleidskader) blijven verborgen in doorlopende tekst. Traditionele Named Entity Recognition (NER) herkent weliswaar entiteiten, maar mist de domeinspecifieke context om relaties te typeren en te kwalificeren.

Open Notebook lost dit op door extractie te sturen met een **ontologie**: een formele beschrijving van de entiteittypen en relatietypen die relevant zijn voor een specifiek domein. De ontologie definieert niet alleen *wat* er geëxtraheerd moet worden (bijvoorbeeld `Organisatie`, `Persoon`, `Regio`, `Programma`), maar ook *hoe* entiteiten zich tot elkaar verhouden (`FINANCIERT`, `VERANTWOORDELIJK_VOOR`, `GELEGEN_IN`). Dit geeft het extractieproces een semantisch kader dat de precisie en relevantie van de resultaten aanzienlijk verhoogt ten opzichte van domein-agnostische methoden.

### Hoe werkt het?

De extractie wordt uitgevoerd door een pluggable architectuur met twee extractors die dezelfde interface delen:

**LLM Extractor** (standaard) — Gebruikt een willekeurig groot taalmodel (via de `llm-manager`) in combinatie met door de ontologie gegenereerde prompts. De `ontology-manager` vertaalt de ontologiedefinitie naar een systeemprompt die het model instrueert om alleen entiteiten en relaties te extraheren die passen binnen het opgegeven schema. Per chunk genereert het model een gestructureerd JSON-antwoord met entiteiten (tekst, type, betrouwbaarheidsscore) en relaties (bron, doel, relatietype, betrouwbaarheidsscore). Een configureerbare betrouwbaarheidsdrempel filtert ruis uit de resultaten.

**LangExtract Extractor** (alternatief) — Een snellere extractiemethode via de `langextract`-bibliotheek, die schema-gestuurde extractie uitvoert met lokale modellen (Ollama, Qwen). Deze extractor ondersteunt parallelle verwerking met meerdere workers, YAML-gebaseerde voorbeeldpatronen, en optionele HTML-visualisatie van de resultaten.

Beide extractors taggen elk resultaat met het bron-chunk-ID, zodat entiteiten en relaties altijd herleidbaar zijn naar de oorspronkelijke passage in het document. De resultaten worden opgeslagen in een dedicated `extraction_result`-tabel in SurrealDB en zijn direct beschikbaar voor de kennisgraaf-visualisatie in de frontend.

### Meerwaarde voor de gebruiker

Door ontologie-gestuurde extractie wordt een verzameling losse documenten omgezet in een bevraagbare kennisgraaf. Gebruikers kunnen niet alleen zoeken op trefwoorden, maar ook navigeren door relaties: "Welke organisaties zijn betrokken bij programma X?", "Welke personen zijn verbonden aan regio Y?". De interactieve graafvisualisatie in de frontend maakt deze verbanden direct zichtbaar — entiteiten als knooppunten, relaties als verbindingen, gekleurd op type en gewogen op betrouwbaarheid.

De keuze voor LLM-gestuurde extractie boven regelgebaseerde systemen is bewust: grote taalmodellen kunnen omgaan met de ambiguïteit en variatie in natuurlijke taal die kenmerkend is voor beleidsdocumenten, vergaderverslagen en onderzoeksrapporten. De ontologie houdt het model gefocust op wat relevant is, terwijl het model de flexibiliteit biedt om entiteiten te herkennen ongeacht formulering of context. Dit maakt het systeem inzetbaar voor uiteenlopende domeinen — van beleidsanalyse en juridische documenten tot medische literatuur en technische specificaties — door simpelweg een andere ontologie te selecteren.
