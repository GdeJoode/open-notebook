# ADR 0018: Obsidian-export als Langflow-flow met een gegenereerd blok

- **Status:** besloten
- **Datum:** 2026-10-29
- **Raakt FO-paragraaf:** 10 (export naar Obsidian), 12.1 (alleen `langflow` schrijft in de vault)
- **Lost op:** verschilanalyse conflict 14

## Context

Het functioneel ontwerp (FO) eist voor de export naar Obsidian:

- entiteiten landen in `20 Atlas/28 Entities`: personen in `28.01 People/Contacts`, organisaties in `28.03 Organisaties`, gebieden in `28.05 Gebieden`;
- programma's worden eigen entiteiten, gekoppeld aan het codeboek;
- wikilinks gebruiken de canonieke naam, en aliassen staan in de frontmatter;
- de frontmatter bevat herkomst (bron, versie, fragment-id's) en classificatie;
- persoonsnamen worden alleen gekoppeld aan bestaande persoonsnotities; nieuwe personen gaan naar review;
- de export is idempotent op een id in de frontmatter, en het systeem schrijft alleen in een gemarkeerd gegenereerd blok;
- de export is een Langflow-flow met een eigen sjabloon (FO 10);
- interne inhoud mag in de export, met label intern in de frontmatter (FO 10);
- alleen `langflow` mag in de vault schrijven (FO 12.1).

In de fork is de export een dienst in app-main:

- één bestand per entiteit, met frontmatter en `id` (`apps/app-main/src/app_main/services/obsidian_export_service.py:1001-1005`);
- atomair schrijven met `os.replace`, waarbij het hele bestand wordt vervangen (`:811`);
- een padcontrole (`:766-772`);
- een standaardmap `Entities` (`apps/app-main/src/app_main/api/routers/vault.py:48`);
- een leeszijde die aliassen uit de vault terugleest (`apps/app-main/src/app_main/services/vault_sync_service.py:1-8`);
- geen classificatie en geen fragment-id's.

## Opties

Het FO maakt de keuze. Over de uitvoering:

1. **De Langflow-component leest de database zelf.** Valt af: dat omzeilt de API (Application Programming Interface) van de fork en het filter (ADR 0002).
2. **De fork levert de exportgegevens via de API, en de Langflow-component vult het sjabloon en schrijft het bestand.** De bestaande render- en schrijfcode wordt hergebruikt in de component. Gekozen.

## Voorstel

Optie 2.

**Fork**

- Een nieuw eindpunt `GET /api/v1/export/obsidian/entities`, met de parameter `since` en recht `lezen_intern` (ADR 0002). Per entiteit levert het:
  - id, canonieke naam, aliassen en type (persoon, organisatie, gebied, programma);
  - koppelingen aan het codeboek;
  - herkomst: bron, bronversie en fragment-id's (ADR 0003);
  - classificatie: intern zodra één onderliggend fragment intern is, net als bij een kennisproduct (FO 9; ADR 0001).
- Vervallen: de schrijvende route en de synchronisatieroute in `vault.py`, `obsidian_export_service.py` en `vault_sync_service.py` in app-main. Hun bruikbare delen verhuizen naar de component (zie hieronder). De container `open-notebook` krijgt geen koppeling met de vault.

**Langflow-component**

- Hergebruik, verplaatst naar een Python-module die de component importeert:
  - het opbouwen van de frontmatter;
  - het atomair schrijven;
  - de padcontrole;
  - het teruglezen van aliassen uit de vault.
- **Indeling:** het type bepaalt de map volgens FO 10. Programma's krijgen een map die de vault-beheer-agent bevestigt; het FO noemt die niet.
- **Een bestaande notitie vinden:** op `id` in de frontmatter, via een index van de vault die de component bij elke run opbouwt. Bestaat de notitie, dan worden alleen de velden in de frontmatter die het systeem beheert en het gegenereerde blok bijgewerkt.
- **Gegenereerd blok:** tussen `<!-- signaalsysteem:begin -->` en `<!-- signaalsysteem:end -->`. Die opmerkingen zijn niet zichtbaar in de leesweergave van Obsidian. Tekst buiten het blok blijft onaangeroerd. Ontbreekt het blok in een bestaande notitie, dan wordt het onderaan toegevoegd.
- **Personen:** zonder bestaande persoonsnotitie wordt geen bestand gemaakt; de persoon gaat naar de reviewwachtrij (ADR 0016).
- **Sjabloon:** een sjabloonbestand in de repository naast de component, onder versiebeheer.
- **Vault:** alleen `langflow` koppelt de vault, met schrijfrechten.

**Bestaande exports:** bestanden die de oude dienst in `Entities` schreef, worden niet automatisch verplaatst. De eerste run meldt ze in zijn logboek; de beheerder beslist.

## Gevolgen

- **Code:**
  - een nieuw exporteindpunt in de fork;
  - de oude exportdienst en de synchronisatie in app-main vervallen;
  - een Langflow-component met sjabloon.
- **Tests (stopcondities):**
  - twee runs na elkaar met dezelfde gegevens geven byte-voor-byte dezelfde bestanden;
  - handmatige tekst buiten het blok blijft gelijk;
  - een interne entiteit krijgt `classificatie: intern` in de frontmatter;
  - een nieuwe persoon levert geen bestand op, maar een reviewitem;
  - testvaults bevatten alleen gegevens uit openbare bronnen.
- **Buiten de fork:** de vault-beheer-agent toetst de export aan de vaultregels (FO 10).
