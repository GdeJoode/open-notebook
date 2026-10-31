# ADR 0019: Classificatie en pipelinestart per innameroute; externe verwerking voor extern

- **Status:** besloten (Geert, 2026-10-30)
- **Datum:** 2026-10-30; bijgewerkt 2026-10-30 na de review (poortwachter op lokale tekst); 2026-10-31 uitwerking in F2-13
- **Raakt FO-paragraaf:** 2 (principe 4), 4, 5 (brontypen, bronkaartregel), 6, 9 (beleidsregel "onbekend is intern"), 12.1 (volledig lokaal); opdracht.md §1, harde eis 4
- **Wijzigt:** ADR 0001 (standaardclassificatie), ADR 0010 (inname), ADR 0014 (adres van docling-serve)

## Context

Het FO behandelt elke bron zonder classificatie als intern (FO 9). Het laat alle verwerking lokaal draaien (FO 12.1, harde eis 4). ADR 0001 werkt dat uit met standaard intern voor alle bronnen.

Geert verfijnde dit op 2026-10-30:

1. Wat via een API, een MCP-bron, een feed (RSS, YouTube, podcast) of een website binnenkomt, is in de regel openbaar. Die bronnen zijn standaard extern, en hun pipeline start automatisch.
2. Een bulkupload krijgt één classificatie en één route voor de hele batch.
3. Documenten uit een watchfolder wachten in de wachtrij. De beheerder kiest per document de pipeline en de classificatie.
4. Documentextractie mag later voor externe documenten bij een externe dienst draaien, bijvoorbeeld docling-serve op Google Cloud of AWS, om de eigen pc te ontlasten. Harde eis 4 wordt daarvoor afgezwakt.

## Opties

1. **Alles standaard intern (ADR 0001 zoals besloten).** Veilig, maar elke openbare bron vraagt een handmatig besluit. Dat werkt niet bij feeds met veel items.
2. **Standaard per innameroute, vastgelegd in de bronkaartregel, met de controle van de poortwachter aan.** Openbare routes lopen automatisch door. Een vertrouwelijk document op een openbare route valt nog op door zijn kenmerken.
3. **Standaard per innameroute, zonder controle van de poortwachter.** Eenvoudiger, maar dan is het label van de route de enige bescherming.

## Besluit

Optie 2.

**Standaard per route**

| Innameroute | Classificatie | Pipelinestart |
| --- | --- | --- |
| API, MCP-bron, feed (RSS, YouTube, podcast), website | extern, als voorstel in de bronkaartregel; de beheerder bevestigt het bij het aanmaken van de regel | automatisch, volgens de route en het schema van de regel |
| Bulkupload | één keuze voor de hele batch: intern of extern; zonder keuze intern | volgens de route die bij de batch is gekozen |
| Watchfolder | geen standaard; tot het besluit van de beheerder geldt intern | geen; het document wacht in de wachtrij tot de beheerder de pipeline en de classificatie kiest |
| Losse handmatige upload | voorstel van de poortwachter; zonder keuze intern (FO 5) | na bevestiging |

- De classificatie hoort bij de bronkaartregel of de batch, niet bij de route alleen. Een API of MCP-server van een intern systeem krijgt bij het aanmaken van de regel het label intern.
- De poortwachter blijft ook externe bronnen controleren op interne kenmerken (FO 6). Een treffer zet het document op intern en in de wachtrij. Van intern naar extern kan alleen de beheerder, met een logregel (FO 14).
- **De poortwachter werkt op lokaal geëxtraheerde tekst** (aanvulling na de review, 30-10). Om interne kenmerken te vinden, moet hij de inhoud lezen voordat iets naar buiten gaat. Hij gebruikt daarom altijd tekst die lokaal is gemaakt: de tekstlaag van een PDF via PyMuPDF, of anders (bijvoorbeeld bij een scan) de lokale docling-serve via llama-swap. Pas als de poortwachter `extern` bevestigt, mag een externe verwerkingsdienst het document zien. Een document waarvoor geen lokale tekst te maken is, blijft `intern` en gaat naar de wachtrij.
- "Onbekend is intern" (FO 9) blijft de terugval: een document zonder bronkaartregel, batchkeuze of besluit van de beheerder is intern.

**Externe verwerking**

- Harde eis 4 wordt: *intern draait altijd lokaal in Docker; GPU-diensten lopen via llama-swap. Een extern document mag ook door een externe verwerkingsdienst gaan.*
- Een externe verwerkingsdienst valt onder dezelfde regel als een cloudmodel (FO 9, harde eis 1). De bouwsteen weigert een intern document en valt bij een fout niet terug op een extern adres.
- De docling-bouwsteen krijgt een instelbaar adres per aanroepwijze, lokaal of extern. Het register uit ADR 0007 legt per adres vast of het is toegestaan voor intern.
- Een externe docling-dienst wordt nu niet ingericht. Dat gebeurt pas als de lokale verwerking aantoonbaar te traag is (P1 en P5). Het vraagt dan een eigen ADR over de aanbieder, de authenticatie, de versleuteling onderweg en het bewaren van bestanden bij de aanbieder.

## Gevolgen

- **ADR 0001:** de standaardclassificatie volgt deze tabel in plaats van "alles intern". Er zijn geen bestaande records om te migreren: de omgeving start met een lege database (B3, 30-10).
- **ADR 0010:**
  - het importeindpunt ontvangt de classificatie uit de bronkaartregel of de batch;
  - de bulkupload krijgt een batchveld voor classificatie en route;
  - de mapcontrole zet nieuwe documenten in de wachtrij in plaats van ze te verwerken.
- **ADR 0014:** het adres van docling-serve is een instelling, en de bouwsteen controleert de classificatie voor de aanroep.
- **ADR 0002:**
  - een externe verwerkingsdienst vraagt uitgaand verkeer vanuit een eigen, kleine container, op het patroon van de gateway uit ADR 0022;
  - de verwerkingscontainers op het netwerk `verwerking` blijven zonder internet.
- **FO 4 en 6:** de poortwachter leest lokaal geëxtraheerde tekst (bijgewerkt 30-10).
- **FO:** §5 (bronkaartregel en brontypen), §9 (beleidsregels) en §12.1 (volledig lokaal) worden bijgewerkt.
- **Opdracht:** harde eis 4 in `opdracht.md` §1 wordt bijgewerkt.
- **Proef P4:** de lektest blijft gelijk, want het filter werkt op het label, niet op de route.

## Uitwerking in F2-13 (31-10)

De bouw (`migrations/93.surrealql`, `SourceMapService`) werkt de tabel "Standaard per route" als volgt uit.

**Besloten** (akkoord reviewer, 31-10):

- **Effectief extern.** Een bronkaartregel is alleen effectief `extern` als hij `extern`, bevestigd en actief is, en alleen op een openbare route: API, MCP, feed of website. Voor watchfolder, losse upload en bulkupload is een regel nooit extern; daar beslist het wachtrij-item (watchfolder) of de batch (bulkupload).
- **Logregel.** De logregel volgt de effectieve classificatie. Elke overgang van intern naar extern en elke intrekking wordt gelogd, met de persoon.
- **Bevestiging vervalt.** Elke inhoudelijke wijziging van een regel trekt de bevestiging in, ook het weer aanzetten van een regel. Alleen een wijziging van naam, eigenaar of een beschrijvend veld laat de bevestiging staan.
- **Bulkupload.** Op regelniveau is `pipeline_start = na_bevestiging`. De route komt van de batch.
- **Herkomst `zonder_regel`.** Een wachtrij-item kan ook de herkomst `zonder_regel` hebben. Geen regel betekent intern.
