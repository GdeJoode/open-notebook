# ADR 0020: De bronkaart als tabel in de fork, alleen via de API

- **Status:** besloten (Geert, 2026-10-30)
- **Datum:** 2026-10-30; bijgewerkt 2026-10-30 na de review (rechten, leesrecht, locatie); 2026-10-31 uitwerking in F2-13
- **Raakt FO-paragraaf:** 5 (bronkaart, bronkaartregel), 6 (regels eerst), 11 (planning en routes in Langflow)
- **Lost op:** plan van aanpak §8, besluitpunt B2 en aanname A4

## Context

De bronkaart bepaalt per bron wat het systeem ophaalt, hoe vaak, langs welke route en met welke classificatie (FO 5). ADR 0019 voegt daar batches van een bulkupload en een innamewachtrij voor watchfolder-documenten en losse uploads aan toe. Het plan van aanpak beschrijft de velden in §5.2.

Geen ADR legde vast waar de bronkaart staat. Het plan ging uit van een tabel in de fork (aanname A4). ADR 0017 C gaat er voor de terugval van de planning van uit dat de schema's "uit de bronkaart via de API van de fork" komen. Het modelregister staat wel als YAML (YAML Ain't Markup Language) in Git (ADR 0007).

De bronkaart verandert vaak: bij elk besluit over een watchfolder-document, bij elke bulkbatch en bij elke nieuwe of uitgezette bronkaartregel. Die wijzigingen doet de beheerder in de werkbak (ADR 0016), niet in een editor.

De bronkaart wordt gelezen door:

- Langflow: de planning (F4-01), de routes en de poortwachter (F2-18);
- het importeindpunt: dat toetst de classificatie van een import aan de regel of de batch (ADR 0010, 0019);
- de werkbak: de innamewachtrij (F2-20).

## Opties

1. **YAML-bestand in Git, zoals het modelregister.**
   - Voor: versiebeheer en review per wijziging.
   - Tegen: elk besluit over een watchfolder-document of een batch wordt een commit en een herstart. Batches en wachtrij-items zijn toestand, geen configuratie, en moeten toch in de database staan. De bevestiging van `extern` door een sleutelhouder (ADR 0019) is in een bestand niet af te dwingen.
2. **Tabellen in SurrealDB, beheerd met de migraties van de fork, alleen bereikbaar via de API van de fork.**
   - Voor: één plek voor regels, batches en wachtrij; wijzigen via de werkbak zonder herstart; rechten, classificatiefilter en optimistische vergrendeling (ADR 0002, 0003) gelden ook hier.
   - Tegen: geen Git-historie. Een run moet daarom zelf vastleggen welke stand van de regel hij gebruikte.
3. **Mengvorm: regels in YAML, batches en wachtrij in de database.**
   - Tegen: twee plekken voor één begrip; de import moet beide raadplegen; de vaak wisselende regels houden het nadeel van optie 1.

## Voorstel

Optie 2. Besloten door Geert op 30-10.

- **Tabellen:** `source_map_rule`, `intake_batch` en `intake_item`, met de velden uit plan van aanpak §5.2. Ze zijn SCHEMAFULL en worden gedefinieerd in een idempotente migratie met `_down` (F2-13). Er zijn geen oude gegevens om te zetten (B3, lege database).
- **Alleen via de API:** lezen en schrijven gaat via `/api/v1/source-map-rules` en `/api/v1/intake/...` in de API van de fork. Geen andere dienst praat met de database (ADR 0002).
  - Lezen vraagt `pijplijn` of `beheerder` (aangepast na de review, 30-10). Een sleutel met `lezen_extern` of `lezen_intern` heeft geen leesrecht op de bronkaart, de batches en de wachtrij: de bronkaart is configuratie, geen kennis, en de naam of locatie van een bron hoeft geen MCP-client te zien.
  - Regels en batches aanmaken, wijzigen en op `extern` zetten vraagt `beheerder`. `extern` vraagt een bevestiging door de persoon van die sleutel (`classification_confirmed_by`, `_at`); zonder bevestiging geldt `intern` (ADR 0019). Een besluit over een wachtrij-item (route en classificatie) vraagt ook `beheerder`.
  - `pijplijn` (Langflow) mag regels en batches lezen, wachtrij-items aanmelden, besloten items ophalen en een item op `verwerkt` zetten. Meer niet.
  - **Geen locatie in de vault of de exportmap** (ADR 0004). De API weigert een regel waarvan `location` binnen de vault (`D:\# Vault`, of het koppelpunt daarvan in een container) of binnen de exportmap van de export naar Obsidian ligt (HTTP 422). Beide paden zijn een instelling van de fork; de controle vergelijkt genormaliseerde paden.
- **Herleidbaarheid zonder Git:** `source_map_rule` krijgt `rev`, `modified_by` en `modified_at` (hetzelfde mechanisme als ADR 0003). Een run legt het id én de `rev` van de regel vast, zoals ADR 0007 dat bij een register in de database zou vragen.
- **Reservekopie:** de bronkaart valt onder de reservekopie van de database; er is geen apart bestand.

## Gevolgen

- **Code:** migratie, model, repository, dienst en router voor bronkaart, batch en wachtrij (F2-13); `processing_run` krijgt `source_map_rule_rev` naast `source_map_rule_id` (F2-04, §5.4).
- **Langflow:** leest de regels via `GET /api/v1/source-map-rules` met de sleutel `pijplijn`; de planning (F4-01) en de terugval met APScheduler (ADR 0017 C) gebruiken dezelfde route.
- **Tests (stopcondities, F2-13):** schemacontrole met de standaardwaarden; een sleutel `lezen_extern` of `lezen_intern` krijgt op elke bronkaart-, batch- en wachtrijroute HTTP 403; een sleutel `pijplijn` kan geen regel of batch aanmaken, wijzigen of bevestigen (HTTP 403); een regel met een locatie in de vault of de exportmap geeft HTTP 422; een wijziging met een verouderde `rev` geeft HTTP 409; een run bevat id en `rev` van zijn regel.
- **Plan van aanpak:** B2 besloten; aanname A4 vervalt; §5.2 noemt deze ADR.

## Uitwerking in F2-13 (31-10)

De bouw (`migrations/93.surrealql`, `SourceMapService`) wijkt op de volgende punten af van het voorstel hierboven, of vult het aan. De regels voor de classificatie staan in ADR 0019, "Uitwerking in F2-13".

**Besloten** (akkoord reviewer, 31-10):

- **Migratienummer.** De migratie is 93, niet 95.
- **`fetch_content`.** Standaard `metadata` voor video, anders `volledig_document`.
- **Wachtrij.** Een besluit over een wachtrij-item ligt vast zodra het item niet meer op `wacht` staat.
- **Locaties.**
  - Een padlocatie moet binnen een toegestane hoofdmap liggen (allowlist `SIGNAAL_INNAME_PADEN`). Is die lijst leeg, dan wordt elk pad geweigerd.
  - Daarnaast worden geweigerd: de vault en de exportmap, Windows-omwegen (`\\?\`, UNC, 8.3-namen, een punt of spatie aan het eind, alternatieve datastromen) en `obsidian://`.
  - **Restrisico.** `subst` en junctions zijn vanuit de fork niet te zien. `realpath` lost koppelingen op in de container van de fork, niet in die van Langflow. De echte grens is wat Compose koppelt (F2-10).
- Restrisico (herreview F2-13): de database dwingt de intrekking van een bevestiging bij een gewijzigde locatie niet af; alleen de API van de fork doet dat. Dat is aanvaard zolang alleen de fork schrijft (B2). URL-locaties: alleen `http` en `https`; loopback-, private en link-local adressen, `host.docker.internal` en hosts zonder punt worden geweigerd tenzij ze op `SIGNAAL_INNAME_URLS` staan, zodat een externe regel niet via een lokale dienst (zoals een REST-plug-in van Obsidian) afgeleide tekst uit de vault binnenhaalt.
