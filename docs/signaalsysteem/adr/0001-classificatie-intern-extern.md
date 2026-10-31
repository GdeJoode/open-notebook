# ADR 0001: Classificatie intern/extern, standaard intern, met overerving

- **Status:** besloten
- **Datum:** 2026-10-29; bijgewerkt 2026-10-30 na de review (rechten `pijplijn` en `beheerder`, afgeleide classificatie); 2026-10-31 uitwerking in F2-03 (B25 besloten, Geert 31-10)
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

## Uitwerking in F2-03 (31-10)

De bouw (`migrations/89.surrealql`) wijkt op de volgende punten af van het voorstel hierboven, of vult het aan.

**Besloten** (akkoord reviewer, 31-10):

- **Regels in het databaseschema.** De regels staan in het schema: `ASSERT`, `DEFAULT ALWAYS` en events, niet alleen in de repositorylaag. Zo gelden ze voor alle schrijfpaden, ook die buiten de repositories. De vlag `$herclassificatie`, die intern naar extern toelaat, mag alleen `SourceRepository.reclassify` zetten; de verbindingslaag weigert hem elders.
- **Vermelding berekend.** De classificatie van een vermelding wordt berekend, als het strengste van fragment en vermelde bron (`fn::vermelding_classificatie`), en nog niet opgeslagen. F2-04 slaat haar op.
- **Inzichten.** Bij herclassificatie naar extern wordt een inzicht (`source_insight`) niet losser; het blijft intern. Wordt de bron intern, dan gaat het inzicht mee (FO 9, harde eis 3).
- **`DEFAULT ALWAYS`.** Nodig onder SurrealDB v2.7.0: een gewone `DEFAULT` geldt alleen bij aanmaken, een expliciete `None` telt niet als waarde, en `VALUE` draait niet op een ontbrekend veld.
- **Logboek.** `classification_log` kan alleen aangevuld worden; een event weigert wijzigen en verwijderen. `89_down` wist het logboek niet.
- **Persoon.** Het logboek controleert de persoon niet tegen de sleutel. Aanvaard, zolang alleen de fork in de database schrijft.

**Besloten door Geert, 31-10, volgens voorstel** (plan van aanpak §8.1, B25):

- **Relatie telt haar entiteiten mee.** Een relatie telt naast haar vermeldingen ook de classificatie van haar twee entiteiten mee. Dat is strenger dan het voorstel hierboven.
  - Voor: geen extern zichtbare relatie tussen entiteiten waarvan er één intern is.
  - Tegen: de externe graaf wordt kleiner.
  - Voorstel: zo houden.
- **Notitie en notebook.** `note` en `notebook` kunnen nu niet van intern naar extern: het schema eist daarvoor `$herclassificatie`, en er is geen herclassificatiepad dat die vlag voor hen zet. Voorstel: het herclassificatie-eindpunt uitbreiden naar notebook en notitie in F2-30 (B22).

**Open:** de graafrelatie `mentions` (bron naar entiteit, `migrations/66.surrealql:58`) heeft nog geen classificatie; zie plan van aanpak F2-04 en F2-05.

**B27 (Geert, 31-10).** Herclassificatie van een bron naar extern zet alle fragmenten op extern. Heeft de bron fragmenten die vóór een eerdere herclassificatie naar intern zelf intern waren, dan ziet de beheerder eerst hun aantal en moet hij bevestigen; zonder bevestiging volgt 409. De eigen classificatie per fragment wordt niet apart bewaard. Bouw in F2-30.
- **Strengste wint per inhoud (herreview F2-14, 31-10).** Komt dezelfde inhoud (hash) als intern binnen, via een interne import of een intern besluit over een wachtrij-item, dan worden alle bronnen met die hash in hun huidige versie intern, in dezelfde transactie, met een `beheer_melding` per bron. Omgekeerd weigert de import `extern` voor een hash die al intern of onbeslist bekend is (ADR 0010).
- **Strengste wint per inhoud, aanvulling (derde review F2-14, 31-10).** Alle versies met die hash worden intern, niet alleen de huidige (zie hierboven: één interne versie maakt de bron en al haar versies intern). Dat werkt transitief: een bron die zo intern wordt, maakt via haar oudere versies ook andere bronnen met dezelfde inhoud intern. Ook een batch die na de import niet meer effectief extern is, maakt de bronnen met haar bestandshashes intern. Dit is strenger en dus veilig.
- **Tabel `inhoud` (vierde review F2-14, 31-10).** Per inhoudshash één record `inhoud:<hash>` met de strengste bekende classificatie. Het wordt alleen intern bij een expliciet intern besluit (interne import, intern itembesluit, herclassificatie naar intern; een interne batch maakt bestaande externe bronnen met die hash intern en telt mee in de controle, maar zet het record niet, zodat ze na bevestiging naar extern nog kan importeren); onbevestigd extern en onbeslist tellen alleen mee in de controle `fn::inhoud_intern_bekend`. Een externe import schrijft het record en toetst in dezelfde transactie, zodat een gelijktijdige interne actie een schrijfconflict wordt. Het record wordt alleen losser bij een herclassificatie van intern naar extern door de beheerder, en alleen als geen andere bron, versie, item of batch met die hash intern of onbeslist is; anders 409 `inhoud_elders_intern` met die records (B27). Het intern maken van een groep bronnen gebeurt zonder recursie: eerst de hele groep bepalen, dan in vaste statements bijwerken.
