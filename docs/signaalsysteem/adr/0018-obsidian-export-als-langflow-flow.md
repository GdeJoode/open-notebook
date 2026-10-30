# ADR 0018: Obsidian-export als Langflow-flow met een gegenereerd blok

- **Status:** besloten
- **Datum:** 2026-10-29; bijgewerkt 2026-10-30 (besluit Geert over B8: vaultregels, programma's, latere fase)
- **Raakt FO-paragraaf:** 10 (export naar Obsidian), 12.1 (alleen `langflow` schrijft in de vault)
- **Lost op:** verschilanalyse conflict 14; plan van aanpak §8, besluitpunt B8

## Context

Het functioneel ontwerp (FO) eist voor de export naar Obsidian:

- notities volgens de vaultregels; de bron van die regels is `D:\# Obsidian\CLAUDE.md` (FO 10);
- programma's worden eigen notities, gekoppeld aan het codeboek;
- wikilinks gebruiken de canonieke naam, en aliassen staan in `aliases`;
- herkomst (bron, versie, fragment-id's) en classificatie gaan mee;
- persoonsnamen worden alleen gekoppeld aan bestaande persoonsnotities; nieuwe personen gaan naar review;
- de export is idempotent, en het systeem schrijft alleen in een gemarkeerd gegenereerd blok;
- de export is een Langflow-flow met een eigen sjabloon (FO 10);
- interne inhoud mag in de export, met label intern (FO 10);
- alleen `langflow` mag in de vault schrijven (FO 12.1).

Eerder noemde het FO de mappen `20 Atlas/28 Entities`, `28.01 People/Contacts`, `28.03 Organisaties` en `28.05 Gebieden`. Die bestaan niet in de vault en vervallen (Geert, 30-10).

**Vaultregels** (`D:\# Obsidian\CLAUDE.md`, gelezen op 30-10; alleen map- en veldnamen overgenomen):

- `1000 Atlas (resources)/` is de kennisbasis; het domein werk staat in `1000 Atlas (resources)/Werk/` (`domain: work`);
- `0500 Bronnen/` wordt nooit gewijzigd;
- bestanden worden nooit verwijderd, alleen gearchiveerd in `3000 Efforts/3004 Archief/`;
- verplichte velden voor een Atlas-notitie: `type`, `maturity`, `domain`, `topics`;
- `type` heeft vaste waarden, waaronder `person`, `entity` en `program`;
- bij `type: entity` hoort `entity_kind`: `region`, `municipality`, `province`, `organisation`, `commission`, `network`, `institution` of `company`;
- bij `type: program` horen minimaal `status` en `lead_organisation`;
- `aliases` voor synoniemen en `external_ids` (onder meer `wikidata`, `tooi`) voor koppelingen;
- nieuwe notities beginnen als `maturity: seedling`; de eigenaar bepaalt de verdere rijpheid;
- `log.md` is het logboek van de vaultworkflows.

Submappen onder `Werk/` noemt het bestand niet. De export legt ze daarom niet vast, maar zoekt ze op (zie hieronder).

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
  - id, canonieke naam, aliassen en type (persoon, organisatie, gebied, programma), met het `entity_kind` als dat bekend is;
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
- **Plaats:** alleen onder `1000 Atlas (resources)/Werk/`. De padcontrole weigert elk ander pad, en in elk geval `0500 Bronnen/`. De component verwijdert, verplaatst en hernoemt geen bestanden.
- **Notitietype:** persoon wordt `type: person`; organisatie en gebied worden `type: entity` met een `entity_kind` uit de lijst hierboven; programma wordt `type: program`. Zonder bekend `entity_kind` maakt de component geen bestand en gaat het item naar review.
- **Submap:** de component bouwt bij elke run een index van de vault, alleen lezend: per notitie pad, titel, `type`, `entity_kind`, `aliases` en het id in het blok. Een nieuwe notitie komt in de map waar de bestaande notities met hetzelfde `type` en `entity_kind` staan. Is dat niet precies één map, dan gaat het item naar review. Er staan geen mapnamen in code, sjabloon of configuratie.
- **Programma's:** een bestaande notitie met `type: program` krijgt het gegenereerde blok. Een nieuw programma gaat naar review (wachtrij 6, ADR 0016), omdat `status` en `lead_organisation` niet uit fragmenten volgen. Na goedkeuring maakt de component de notitie in de map van de bestaande programmanotities, volgens de regel voor de submap.
- **Een bestaande notitie vinden:** op het id in het gegenereerde blok. Zonder blok op een unieke overeenkomst van titel of `aliases` met de canonieke naam of een alias; bij meer dan één overeenkomst gaat het item naar review.
- **Gegenereerd blok:** tussen `<!-- signaalsysteem:begin -->` en `<!-- signaalsysteem:end -->`. Die opmerkingen zijn niet zichtbaar in de leesweergave van Obsidian. Het blok bevat het id van de entiteit, de classificatie, de herkomst en de codeboekkoppelingen. De component schrijft in een bestaande notitie alleen in dit blok: frontmatter en tekst erbuiten blijven onaangeroerd. Ontbreekt het blok, dan wordt het onderaan toegevoegd.
- **Nieuwe notitie:** alleen voor organisaties en gebieden, en voor goedgekeurde programma's. De component schrijft eenmalig frontmatter met `type`, `maturity: seedling`, `domain: work`, `topics: []`, bij een entiteit `entity_kind`, en `aliases`; daarna alleen nog het blok.
- **Personen:** zonder bestaande persoonsnotitie wordt geen bestand gemaakt; de persoon gaat naar de reviewwachtrij (ADR 0016).
- **Logboek:** de component schrijft niet in `log.md`; het logboek is de run (ADR 0003).
- **Sjabloon:** een sjabloonbestand in de repository naast de component, onder versiebeheer.
- **Vault:** alleen `langflow` koppelt de vault.

**Latere fase**

- Schrijven naar de echte vault (`D:\# Obsidian`) is een latere fase, na een besluit van Geert. Tot dan schrijft niets in die map, en koppelt geen container haar.
- Bouw en tests (F2-21) gebruiken een tijdelijke testvault, die de test zelf aanmaakt met dezelfde vaultregels en alleen openbare gegevens.
- De repository bevat geen inhoud uit de vault; alleen map- en veldnamen uit `D:\# Obsidian\CLAUDE.md`.

**Bestaande exports:** bestanden die de oude dienst in `Entities` schreef, worden niet automatisch verplaatst. De eerste run op de echte vault meldt ze in zijn logboek; de beheerder beslist.

## Gevolgen

- **Code:**
  - een nieuw exporteindpunt in de fork;
  - de oude exportdienst en de synchronisatie in app-main vervallen;
  - een Langflow-component met sjabloon.
- **Tests (stopcondities), op een tijdelijke testvault:**
  - twee runs na elkaar met dezelfde gegevens geven byte-voor-byte dezelfde bestanden;
  - frontmatter en tekst buiten het blok van een bestaande notitie blijven gelijk;
  - een interne entiteit krijgt `classificatie: intern` in het blok;
  - een nieuwe persoon en een nieuw programma leveren geen bestand op, maar een reviewitem;
  - een nieuwe notitie komt in de map van de bestaande notities met hetzelfde `type` en `entity_kind`; zonder zo'n map, of met meer dan één, volgt een reviewitem;
  - geen schrijfactie buiten `1000 Atlas (resources)/Werk/` van de testvault;
  - testvaults bevatten alleen gegevens uit openbare bronnen.
- **FO 10:** het label intern staat in het gegenereerde blok, niet in de frontmatter, omdat de export frontmatter van een bestaande notitie niet wijzigt.
- **Open:** de vaultregels kennen het gegenereerde blok nog niet. Of `D:\# Obsidian\CLAUDE.md` het blok en de reviewregels gaat noemen, beslist Geert; het signaalsysteem wijzigt dat bestand niet.
- **Buiten de fork:** de vault-beheer-agent toetst de export aan de vaultregels (FO 10).
