# ADR 0021: SurrealDB v3

- **Status:** besloten: `v2.7.0` voor nu, overstap naar v3 later (Geert, 30-10, na P10)
- **Datum:** 2026-10-30
- **Raakt FO-paragraaf:** 8 (datamodel), 12.1 (`surrealdb`, vaste imageversies), 14 (database van de SPOT)
- **Wijzigt bij besluit:** plan van aanpak §8, besluitpunt B3 (SurrealDB vast op `v2.7.0`)

## Context

De signaalsysteemomgeving start met een lege database op `surrealdb/surrealdb:v2.7.0`, de versie van P4 (besluit Geert, B3, 30-10). Geert wil "als het even kan meteen overstappen naar de volgende versie". Omdat de database leeg begint, is er geen datamigratie nodig; de vraag is of de migraties, de code en de tests van de fork op v3 draaien.

De fork:

- gebruikt de Python-SDK (software development kit) `surrealdb==1.0.6` (`packages/surrealdb-service/pyproject.toml:10`);
- draait de migraties bij het opstarten van de API (`apps/app-main/src/app_main/api/app.py:131-137`);
- gebruikt in de migraties `SEARCH ANALYZER` (`migrations/1.surrealql:67-72`) en `type::is::array` (`migrations/82.surrealql:28`);
- gebruikt `type::thing`, `type::is::*` of `string::is::*` op 111 plaatsen in 39 Python-bestanden onder `apps/`, `packages/` en `pipelines/`, waarvan 12 broncodebestanden en de rest tests (telling 30-10).

**Wat de actuele documentatie zegt** (alles geraadpleegd op 30-10-2026):

- **Versies.** De nieuwste uitgave is v3.3.5 (29 oktober, "Latest"); daarvoor v3.3.1 tot en met v3.3.4 en v3.2.6 (19 oktober). v2.7.0 (28 september) is de enige v2-uitgave op de releasepagina en de nieuwste v2 ([GitHub-releases](https://github.com/surrealdb/surrealdb/releases); de pagina noemt geen jaar). v3.3.0 is van 24 september 2026 ([releasebot](https://releasebot.io/updates/surrealdb)).
- **Advisories die 2.7.0 raken.**
  - GHSA-vx2p-7hhm-wv62 (Critical, Common Vulnerability Scoring System (CVSS) 9,6): toegang tot een andere namespace en database via de headers `Surreal-NS`/`Surreal-DB` met Basic-authenticatie. Getroffen: `< 3.3.0`; opgelost in 3.3.0; geen 2.x-oplossing genoemd. Raakt geen omgeving met één tenant of aanroepen met rootbereik ([advisory](https://github.com/surrealdb/surrealdb/security/advisories/GHSA-vx2p-7hhm-wv62)).
  - GHSA-w5v9-jc5f-5rmc (Critical): dezelfde soort toegang voor systeemgebruikers via de headers en de methode `use` van de remote procedure call (RPC). Getroffen: `< 3.3.0`; opgelost in 3.3.0; geen 2.x-oplossing ([advisory](https://github.com/surrealdb/surrealdb/security/advisories/GHSA-w5v9-jc5f-5rmc)).
  - GHSA-pm86-cc75-q463 (High): use-after-free in de opslaglagen voor sleutels en waarden (KV-backends). Getroffen volgens de advisory `< 3.3.0`, maar de reeks is voorlopig ("We have not yet mapped the released version range") ([advisory](https://github.com/surrealdb/surrealdb/security/advisories/GHSA-pm86-cc75-q463)).
  - De releasenotities van 2.7.0 zeggen volgens releasebot dat meerdere advisories die in 3.3.0 zijn opgelost, waarvan drie Critical, alle eerdere versies raken, ook 2.7.0; 2.7.0 zelf bevat geen beveiligingsoplossingen ([releasebot](https://releasebot.io/updates/surrealdb)). De lijst met advisories telt acht pagina's ([advisories](https://github.com/surrealdb/surrealdb/security/advisories)); alleen de drie hierboven zijn per stuk nagekeken.
  
  Conclusie: de bekende advisories zijn in v3 (vanaf 3.3.0) opgelost en in v2 niet. Een 2.x-uitgave met de oplossingen is niet gevonden.
- **Breaking changes in SurrealQL van 2.x naar 3.x** ([migratiegids 2.x naar 3.x](https://surrealdb.com/docs/build/migrating/from-old-surrealdb-versions/2x-to-3x.md)). Relevant voor de fork:
  - `SEARCH ANALYZER` wordt `FULLTEXT ANALYZER` in indexdefinities (raakt `migrations/1.surrealql`);
  - hernoemde functies, onder meer `type::thing` naar `type::record`, `type::is::*` naar `type::is_*`, `string::is::*` naar `string::is_*`, `time::from::*` en `duration::from::*` naar `*_from_*`, `rand::guid()` naar `rand::id()`;
  - `LET` is verplicht bij het toekennen van een parameter;
  - `GROUP` en `SPLIT` mogen niet meer samen;
  - de like-operatoren (`~`, `!~`, `?~`, `*~`) en `MTREE` (vervangen door `HNSW`) zijn weg;
  - futures worden `COMPUTED`-velden;
  - op een SCHEMAFULL-tabel geeft een onbekend veld een fout in plaats van dat het stil wordt weggelaten, en een `SELECT` op een niet-bestaande tabel geeft een fout. In P4 liet v2.7.0 `classification` stil weg; onder v3 was dat een fout geweest;
  - gewijzigd gedrag van de idioom `.*`, van optionele ketens (`?` wordt `.?`), van `SET` (leest de stand vóór de opdracht) en van numerieke record-id's.
  
  Gegevens: de 3.x-binary leest geen 2.x-gegevens; voor een overstap met gegevens is er `surreal v2 export --v3` (vanaf 3.0.3). Dat speelt hier niet, want de database start leeg. Het oude volume `./surreal_data` blijft een v2-archief.
- **Graafmogelijkheden.** Wat er volgens de releasenotities bij komt:
  - 3.2.0 (1 juli 2026): de graafquerytaal ISO GQL (Graph Query Language, in SurrealDB OpenGQL) met `MATCH`, `OPTIONAL MATCH`, aggregaties en padzoeken (`SHORTEST`, `ALL`, `ANY`), via `POST /gql` of de RPC-methode `gql`. Dit is experimenteel, staat standaard uit (`--allow-experimental opengql`) en "should not be relied on in production". Verder: `WHERE`-filters op graafverplaatsingen worden in de scan van de relaties uitgevoerd, en twee fouten in graafbewerkingen zijn hersteld ([release 3.2](https://surrealdb.com/releases/3.2)).
  - 3.3.0 (24 september 2026): de buren van een record staan in compacte blokken, wat verplaatsingen vanaf records met veel relaties sneller maakt; `count(->edge)` laadt de relaties niet meer. Nieuwe schemaopties: `LIGHTWEIGHT`-relaties (alleen `in` en `out`), `INLINE`-velden op een relatie, en `INLINE EDGES` en `INLINE REFERENCES` ([releasebot](https://releasebot.io/updates/surrealdb)).
  - Een functie voor het kortste pad buiten de experimentele GQL is niet gevonden. Recursieve graafverplaatsing staat op de platformpagina, zonder versie; of die nieuw is in v3, is niet vastgesteld ([platform](https://surrealdb.com/platform)).
  
  Voor het FO is geen van deze functies nodig: het datamodel (FO 8) gebruikt relaties en verplaatsingen die in v2 al werken. Wat v3 voor de graaf oplevert, is vooral snelheid. De reden om over te stappen is de beveiliging, niet de graaf.
- **Python-SDK.** De nieuwste stabiele `surrealdb` op PyPI is 2.0.0 (23 april 2026); 3.0.0b10 (8 oktober 2026) is een voorversie ([PyPI](https://pypi.org/pypi/surrealdb/json)). De notities van de 3.0-bèta's testen tegen SurrealDB 2.1.9 en 3.0.5, en noemen gedrag dat alleen met 3.x of met 3.3.0 en later werkt ([releases surrealdb.py](https://github.com/surrealdb/surrealdb.py/releases)). Welke serverversies 1.0.6 en 2.0.0 ondersteunen, staat niet op de geraadpleegde pagina's; de pagina van 1.0.5 noemt SurrealDB 2.0.0 tot en met 2.3.6 ([PyPI 1.0.5](https://pypi.org/project/surrealdb/1.0.5/)). Of de fork met `1.0.6` of `2.0.0` tegen v3 werkt, is dus niet bekend.

## Opties

1. **v2.7.0, zoals in B3.**
   - Voor: beproefd in P4; geen aanpassingen.
   - Tegen: de advisories hierboven blijven open, zonder bekende 2.x-oplossing. Het risico is beperkt: één namespace en database, en alleen `open-notebook` bereikt de database (netwerk `opslag`, ADR 0002 F). Een latere overstap kost hetzelfde werk, plus een export van gegevens.
2. **Direct v3.3.5.**
   - Voor: advisories opgelost; strengere schemacontrole, die stil weggelaten velden zoals in P4 voorkomt; een overstap zonder gegevens is nu het goedkoopst.
   - Tegen: migraties en code moeten worden aangepast; de omvang en de werking van de SDK zijn onbekend.
3. **Eerst een proef, dan kiezen.** P10 draait alle migraties en de testset van de fork tegen v3 op een lege database. Lukt dat zonder grote aanpassingen, dan v3; anders v2.7.0, met de overstap naar v3 als aparte taak.

## Voorstel

Optie 3.

- **P10** (`proeven.md`, script `scripts/signaalsysteem/proeven/p10_surrealdb_v3.sh`) draait op een lege database met `surrealdb/surrealdb:v3.3.5` en, ter vergelijking, met `v2.7.0`. Aanpassingen voor de proef gebeuren op een kopie in de runmap, niet in de repository.
- **"Zonder grote aanpassingen"** betekent (voorstel, Geert stelt de grens vast vóór de proef):
  - alle migraties, ook de `_down`-varianten, draaien foutloos op v3 na alleen mechanische vervangingen uit de migratiegids (functienamen, `SEARCH ANALYZER` naar `FULLTEXT ANALYZER`, `LET`);
  - geen test faalt op v3 die op v2.7.0 slaagt, na diezelfde mechanische vervangingen;
  - een stabiele SDK-versie (`1.0.6` of `2.0.0`) werkt; een bètaversie telt niet;
  - wat niet mechanisch is, raakt ten hoogste tien bestanden en vraagt geen herontwerp van een tabel.
- **Uitkomst:**
  - P10 geslaagd: SurrealDB vast op `v3.3.5` (of de dan nieuwste 3.3-uitgave met vaste digest). De mechanische vervangingen worden een bouwtaak in fase 2, vóór F2-03, met een eigen migratie per vervanging (idempotent, met `_down`).
  - P10 niet geslaagd: `v2.7.0` blijft (B3). De overstap naar v3 wordt een aparte taak met de lijst uit P10. Het netwerk `opslag` en één namespace en database beperken tot dan het risico van de advisories.
- **Referentiegegevens** (akkoord Geert, 30-10): bij een lege database ontbreken gegevens die niet uit een migratie of een laadscript komen, zoals ontologieën en woordenlijsten. De bouwer inventariseert die in fase 2 en maakt voor elk een laadscript (F3-03 doet dat voor Iv3 en het Basiswettenbestand (BWB), ADR 0013). Dat geldt voor v2 en v3.

Geert beslist, na P10.

## Besluit

`v2.7.0` blijft (Geert, 30-10). P10 (runmap `p10/20261030T205153`, exitcode 1) haalt de beslisregel niet:

- K1 geslaagd op v2.7.0, na correctie van een bestaande parsefout in `migrations/5_down.surrealql` (`REMOVE TABLE IF EXISTS transformation SCHEMAFULL;`, ook op v2 fout). Geert accepteert die correctie in de nulmeting; het herstel in de repository hoort bij F2-03.
- K2 tot en met K5 mislukt: na 126 mechanische vervangingen falen op v3.3.5 nog 41 migratiebestanden (grens 10). Vooral `FLEXIBLE TYPE x` naar `TYPE x FLEXIBLE` (25 bestanden), `FLEXIBLE` alleen bij `object` (16 bestanden met `FLEXIBLE TYPE array`) en fouten bij `REMOVE`, `UPDATE` of `DELETE` op iets dat niet bestaat (17 bestanden). Geen tabel vraagt herontwerp. De SDK `1.0.6` maakt verbinding met v3.3.5.
- K6: v3 weigert een onbekend veld op een SCHEMAFULL-tabel, v2.7.0 laat het stil weg.

**Restrisico.** GHSA-vx2p-7hhm-wv62 en GHSA-w5v9-jc5f-5rmc blijven open. Ze vragen meer dan één namespace of database, of aanroepen zonder rootbereik met de headers; het signaalsysteem gebruikt één namespace en database, en alleen `open-notebook` bereikt SurrealDB, op het netwerk `opslag` zonder poort en zonder internet. GHSA-pm86-cc75-q463 heeft nog geen bevestigde versiereeks.

**Overstap later.** F2-27 wordt de aparte overstaptaak na F2-24, met de lijst uit P10 (`p10_patronen.csv`). Daarna draait P10 opnieuw; pas dan wordt de testset zelf op v3 gemeten.

## Gevolgen

- **Proeven:** P10 in `proeven.md`, plek hier (geen GPU).
- **Bij v3:**
  - Compose: `surrealdb/surrealdb:v3.3.5` met vaste digest in plaats van `v2.7.0`;
  - P4 wordt herhaald op v3 (acceptatietest van ADR 0001 en 0002);
  - code en tests: de vervangingen uit P10; `surrealdb` in `packages/surrealdb-service/pyproject.toml` op de SDK-versie waarmee P10 slaagde;
  - het oude volume `./surreal_data` blijft een v2-archief. Wie het later wil lezen, gebruikt een v2-container of `surreal v2 export --v3`.
- **Bij v2.7.0:** een aparte taak "overstap naar SurrealDB v3" in het plan, na fase 2, met de lijst uit P10; een controle per kwartaal op een 2.x-uitgave met beveiligingsoplossingen.
- **Plan van aanpak:** B3, R10 en F2-10 noemen deze ADR; de planner werkt ze bij.
- **Memory-opslag in tests (derde review F2-14, 31-10).** De memory-opslag van SurrealDB v2.7.0 dwingt een UNIQUE-index niet af bij gelijktijdige transacties; rocksdb (productie) wel. Gelijktijdigheidstests draaien daarom op een rocksdb-fixture. Dit raakt ook `idx_source_external_id` en `dn_run_ref` in tests.
- **Planner-fout in v2.7.0 (vijfde review F2-14, 31-10).** Voor `veld INSIDE $lijst` of `veld IN $lijst` kiest de planner een samengestelde index die met dat veld begint (bijvoorbeeld `idx_source_version_hash (source, content_hash)`) en geeft dan stil 0 rijen. Gelijkheid (`veld = $x`) werkt wel. Mogelijk geraakt: `chunk WHERE source IN $ids` (`repositories/source.py`) en `WHERE source IN $sources` (`repositories/search.py`), door `idx_chunk_order (source, order)`. Te toetsen in een aparte taak; opnieuw bekijken bij de overstap naar v3 (F2-27).

- **Afgeleid image (F2-10).** `signaal/surrealdb:v2.7.0` bouwt op de vaste digest van `surrealdb/surrealdb:v2.7.0` met een statische busybox, alleen om de geheimen in te lezen; dat brengt een shell in de databasecontainer. Het basisimage declareert `VOLUME /logs`; dat krijgt een benoemd volume of tmpfs, zodat er geen anoniem volume achterblijft.
