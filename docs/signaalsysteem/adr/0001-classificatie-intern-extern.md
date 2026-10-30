# ADR 0001: Classificatie intern/extern, standaard intern, met overerving

- **Status:** besloten
- **Datum:** 2026-10-29; bijgewerkt 2026-10-30 na de review (rechten `pijplijn` en `beheerder`, afgeleide classificatie)
- **Raakt FO-paragraaf:** 5 (bronkaartregel), 6, 9 (beleidsregels), 14 (herclassificatie)
- **Lost op:** verschilanalyse conflict 3 en 4, open vraag 6
- **Gewijzigd door:** ADR 0019 (classificatie en pipelinestart per innameroute)

## Context

Het functioneel ontwerp (FO) kent één classificatie, intern of extern, op de bron. Onbekend geldt als intern. De classificatie gaat over van bron op versie, fragment en embedding; een kennisproduct is intern zodra één bronfragment intern is (FO 9). De poortwachter kan een bron niet van intern naar extern zetten; dat kan alleen de beheerder, met logregel (FO 6, FO 14). Harde eis 1 steunt hierop.

De fork doet het omgekeerde en kent meerdere woordenlijsten:

- `source.private`, booleaans, standaard `false` (`migrations/52.surrealql:30`); het uploadformulier zet `"false"` (`apps/app-main/src/app_main/api/routers/sources_upload.py:199`);
- `notebook.privacy_mode` (`migrations/52.surrealql:32`) en de globale standaard `default_privacy_mode = 'cloud'` (`migrations/52.surrealql:34`, `53.surrealql:19`), samen gelezen door `apps/app-main/src/app_main/services/model_routing/privacy_resolver.py`;
- een eigen indeling `public`/`internal`/`confidential` in `packages/shared/src/shared/model_routing.yaml:16`;
- geen classificatieveld op `chunk` en `source_embedding` (`migrations/10.surrealql`, `27.surrealql`);
- geen manier om de classificatie na aanmaak te wijzigen, en geen logregel (`apps/app-main/src/app_main/api/schemas.py:610-614`).

Het FO maakt de inhoudelijke keuze. Deze ADR legt vast hoe de fork die keuze uitvoert.

## Opties

Over de uitvoering, niet over de regel zelf:

1. **Eén veld `classification` op elk niveau, bij het schrijven overgenomen van de bron.** Elk record draagt zijn eigen waarde, zodat elke query direct kan filteren, ook de vectorzoekopdracht. Herclassificatie moet alle afgeleide records in één transactie bijwerken.
2. **Alleen op de bron, bij elke query via een join afleiden.** Geen kopieën, maar elke leesroute moet de join doen; één vergeten join lekt interne fragmenten. De vectorzoekopdracht wordt trager en complexer.
3. **`source.private` houden en vertalen.** Behoudt de verkeerde standaard en twee woordenlijsten naast elkaar. Valt af: dat zwakt harde eis 1 af.

## Voorstel

Optie 1.

**Datamodel**

- Veld `classification` met `ASSERT $value IN ['intern', 'extern']` en `DEFAULT 'intern'` op `source`, `source_version` (nieuw, ADR 0003), `chunk`, `source_embedding`, `doc_node`, `note`, `source_insight` en `notebook`. Kennisproducten die later komen (wiki-pagina's, ADR 0004) krijgen hetzelfde veld.
- Bij het schrijven neemt de API (Application Programming Interface) van de fork de waarde over van de bronversie. Een fragment, embedding of `doc_node` met een lossere waarde dan zijn bronversie wordt geweigerd (HTTP 409). Ontbreekt de waarde, dan geldt `intern`.
- `doc_node` (de documentstructuur, `migrations/49.surrealql`) krijgt ook `source_version` en `run`, net als `chunk` (ADR 0003). Zo hoort de structuur bij één bronversie en volgt ze haar classificatie.
- `source.full_text` hoort bij geen enkele versie. Het veld vervalt; de tekst per versie staat in de fragmenten en in `doc_node`. Een lezer die de volledige tekst nodig heeft, stelt die samen uit de fragmenten van de huidige versie, met het classificatiefilter.
- **Afgeleide classificatie** voor records die uit fragmenten volgen: `entity`, `entity_alias`, `relation`, de vermeldingen (PC.3-herkomst, ADR 0003) en `triage_queue`. Een vermelding neemt de waarde over van haar fragment. De andere records zijn `intern` zodra één onderliggende vermelding intern is; anders `extern`. De fork berekent de waarde bij elk schrijven van een vermelding en bij elke herclassificatie; een record zonder vermelding is `intern`.
- Een kennisproduct krijgt `intern` als één van de fragmenten waarnaar het verwijst intern is, of als de herkomst ontbreekt.
- Een notebook is een weergave (FO 13). De eigen waarde beschermt de naam en beschrijving; welke bronnen een client ziet, bepaalt het filter (ADR 0002).

**Wie mag wat**

- Een nieuwe bronversie is alleen `extern` als de bronkaartregel of de batch `extern` zegt, die keuze door een sleutel met het recht `beheerder` is bevestigd (ADR 0020), en de poortwachter geen interne kenmerken vindt. In alle andere gevallen is ze `intern`.
- De strengste waarde wint: wordt één versie intern, dan worden de bron en al haar versies, fragmenten, embeddings en `doc_node`-records intern, en daarmee ook de afgeleide records hierboven.
- **Rechten** (ADR 0002 B):
  - `pijplijn` is de sleutel van Langflow. Die importeert met de classificatie van de regel of de batch, maar kan geen regel of batch op `extern` zetten of bevestigen, geen batch aanmaken met `extern` en niet herclassificeren.
  - `beheerder` is een persoonlijke sleutel van een mens. Alleen die kan een bronkaartregel of batch op `extern` zetten en bevestigen, een besluit nemen over een watchfolder-document, en herclassificeren.
  - Van intern naar extern kan alleen via `POST /api/v1/sources/{id}/classification` met een sleutel met het recht `beheerder` en een verplichte reden. Het importeindpunt (ADR 0010) kan dit niet; een import met `extern` voor een interne bron geeft HTTP 409.
- Elke wijziging schrijft een regel in een nieuwe tabel `classification_log` (bron, van, naar, reden, sleutel-id, persoon, tijdstip). Die tabel is alleen aanvulbaar. Een sleutel met `beheerder` heeft een verplicht veld `person` (de naam van de sleutelhouder); de regel legt die persoon vast, zodat elke herclassificatie tot een mens herleidbaar is. Een sleutel zonder persoon kan geen regel schrijven.

**Wat vervalt**

- `source.private`, `notebook.privacy_mode` en de instelling `default_privacy_mode`.
- `privacy_resolver.py` en het gebruik van `PrivacyMode` in `route_resolver.py`. De fork roept na ADR 0006 alleen lokale modellen aan, dus een privacymodus voor de modelkeuze is daar niet meer nodig.
- De niveaus in `model_routing.yaml`; het bestand vervalt met ADR 0006 en 0012.
- Het formulierveld `private` vervalt met de uploadroute (ADR 0010).

**Migratie**

- De signaalsysteemomgeving start met een lege database (besluit Geert, 30-10, plan van aanpak §8 B3). Bestaande records worden niet gemigreerd; het oude datavolume blijft onaangeroerd staan als archief.
- Een nieuwe idempotente migratie (vanaf `migrations/88.surrealql`, met `_down`) definieert de velden met `DEFINE FIELD OVERWRITE`. Ze zet geen bestaande records om.
- Daarna verwijdert een migratie `private`, `privacy_mode` en `default_privacy_mode`.
- De `_down`-variant zet de oude velden terug met `private = true` voor alle bronnen. Zo maakt ook een terugdraaiactie niets extern.

## Gevolgen

- **Code:** schema's in `apps/app-main/src/app_main/api/schemas.py` en `packages/shared/src/shared/models/source.py` krijgen `classification` in plaats van `private`. De repositories in `packages/surrealdb-service` zetten het veld bij het schrijven. Er komt een nieuwe `ClassificationService` met het herclassificatie-eindpunt en `classification_log`.
- **Tests:** een schemacontrole dat elke genoemde tabel `classification` heeft met standaard `intern`. Zolang de migratie het veld niet definieert, laat een SCHEMAFULL-tabel een geschreven `classification` stil weg (P4, 30-10); de test leest de waarde daarom terug na het schrijven; een test dat een record zonder waarde `intern` wordt; een test dat import of poortwachter intern niet naar extern kan zetten; een test dat een sleutel `pijplijn` geen regel of batch op `extern` kan zetten en niet kan herclassificeren (HTTP 403); een test dat herclassificatie alle versies, fragmenten, embeddings en `doc_node`-records in één transactie bijwerkt, de afgeleide records opnieuw berekent en precies één logregel met persoon schrijft; een test dat een entiteit met één interne vermelding `intern` is.
- **Interface:** schermen die `private` of `privacy_mode` tonen of zetten, werken niet meer (zie ADR 0016).
- **Afhankelijkheden:** het filter dat deze waarde gebruikt, staat in ADR 0002; bronversie en run in ADR 0003.
