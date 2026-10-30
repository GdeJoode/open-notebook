# ADR 0018: Obsidian-export als Langflow-flow met een gegenereerd blok

- **Status:** besloten; onderdeel "Nieuwe notitie" voorgesteld (30-10, na de correctie van de vault)
- **Datum:** 2026-10-29; bijgewerkt 2026-10-30 (besluit Geert over B8: programma's, latere fase) en 2026-10-30 (correctie: de vault is `D:\# Vault`)
- **Raakt FO-paragraaf:** 10 (export naar Obsidian), 12.1 (alleen `langflow` schrijft in de vault)
- **Lost op:** verschilanalyse conflict 14; plan van aanpak §8, besluitpunt B8

## Context

Het functioneel ontwerp (FO) eist voor de export naar Obsidian:

- notities volgens de vaultregels (FO 10);
- programma's worden eigen notities, gekoppeld aan het codeboek;
- wikilinks gebruiken de canonieke naam, en aliassen staan in `aliases`;
- herkomst (bron, versie, fragment-id's) en classificatie gaan mee;
- persoonsnamen worden alleen gekoppeld aan bestaande persoonsnotities; nieuwe personen gaan naar review;
- de export is idempotent, en het systeem schrijft alleen in een gemarkeerd gegenereerd blok;
- de export is een Langflow-flow met een eigen sjabloon (FO 10);
- interne inhoud mag in de export, met label intern (FO 10);
- alleen `langflow` mag in de vault schrijven (FO 12.1).

**Correctie (30-10).** In de vorige ronde zijn de vaultregels gelezen uit `D:\# Obsidian\CLAUDE.md`. Dat is de verkeerde vault. Het signaalsysteem hoort bij `D:\# Vault` (Geert, 30-10). De paden en velden die uit `D:\# Obsidian` waren afgeleid, vervallen. De oorspronkelijke FO-paden onder `20 Atlas/28 Entities/` kloppen daarmee grotendeels.

**Vaultregels van `D:\# Vault`** (`CLAUDE.md`, `AGENTS.md`, `10 Meta/11 Personal/Tagging & Architecture Conventions.md`, de folder notes onder `20 Atlas/28 Entities/` en de sjablonen in `50 Resources/54 Templates/Templater/`; gelezen op 30-10, alleen map- en veldnamen overgenomen):

- De vault volgt de Obsidian Starter Kit (OSK). Notitietypen staan in een register dat `osk-cli` bevraagt; de map volgt uit het type ("choose the type and folder follows"). Paden worden bij het uitvoeren opgezocht, niet in code vastgelegd.
- **OSK-first:** notities worden gemaakt en bewerkt met de OSK-skills of de Templater-sjablonen. Een directe schrijfactie mag alleen als daarvoor geen OSK-gereedschap bestaat, en dat wordt dan uitdrukkelijk gemeld.
- Entiteiten staan onder `20 Atlas/28 Entities/`:

  | Map | Inhoud | Herkenbaar aan |
  | --- | --- | --- |
  | `28.01 People` | personen; persoonlijke contacten in `Contacts` | tag `type/person` of `type/contact` |
  | `28.02 Concepts` | begrippen | — |
  | `28.03 Organisaties` | organisaties, per submap één waarde van `categorie` (bijvoorbeeld `Gemeenten`, `Provincies`, `Rijk`) | tag `type/entity/organisation`; bestandsnaam `<Naam> (Organisation)` |
  | `28.04 Thema's` | het codeboek | — |
  | `28.05 Gebieden` | regio-indelingen, per submap één `indeling` (bijvoorbeeld `NPVR`, `Regio Deals`, `NOVEX`) | tag `type/entity/region`; bestandsnaam `<INDELING> - <regio> (Region)` |
  | `28.06 Programma's` | programma's als entiteiten, in submappen als `Nationaal`, `Gebiedsagendas` en `Archief` | tag `type/entity/government_programme`; bestandsnaam `<Naam> (Gov Programme)` |

- Frontmattervelden van entiteiten: `type` met een schema.org-waarde (onder meer `schema:Organization`, `schema:GovernmentOrganization`, `schema:AdministrativeArea`, `schema:Project`), `title`, `description`, `aliases`, `status` (nieuw: `draft`), `tags` (met `zone/atlas` en het entiteittype), `created` en `updated`. Per soort komen er velden bij, zoals `categorie` (organisaties), `indeling`, `regiocode` en `gemeenten` (gebieden), en `programmasoort`, `eigenaar`, `programmastatus`, `onderwerp_primair` en `onderwerpen_secundair` (programma's, met codes uit het codeboek).
- Sjablonen voor nieuwe entiteiten: *TPL Person*, *TPL Organisation* en *TPL Region*. Die werken met keuzevragen van Templater en kunnen buiten Obsidian niet worden uitgevoerd. Voor programma's in `28.06 Programma's` is geen sjabloon gevonden; *TPL Programme* is voor eigen programma's buiten de Atlas.
- De canonieke naam is de meest recente naam; eerdere namen en afkortingen staan als alias.
- Een wikilink verwijst alleen naar een notitie die bestaat.
- Na een nieuwe notitie komt een link in de dagnotitie; wijzigingen in een entiteitmap staan in een onderhoudslog in `10 Meta/14 Maintenance/`.
- Ruwe bronnen staan in `50 Resources/51 Raw/`.

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

Voor nieuwe notities (zie Voorstel):

- a) **De export maakt geen nieuwe notities.** Elke nieuwe entiteit gaat naar review; de beheerder maakt de notitie met het OSK-sjabloon in Obsidian, en de volgende run vult het blok. Volgt OSK-first zonder uitzondering.
- b) **De export maakt een nieuwe notitie zelf**, met de frontmatter van het sjabloon. Dat is een directe schrijfactie buiten OSK om; de vaultregels moeten die uitzondering dan noemen. Velden die het sjabloon met een keuzevraag vult (`categorie`, `indeling`), moeten uit de exportgegevens komen.

## Voorstel

Optie 2.

**Fork**

- Een nieuw eindpunt `GET /api/v1/export/obsidian/entities`, met de parameter `since` en recht `lezen_intern` of `pijplijn` (ADR 0002). Per entiteit levert het:
  - id, canonieke naam, aliassen en soort (persoon, organisatie, gebied, programma);
  - koppelingen aan het codeboek;
  - herkomst: bron, bronversie en fragment-id's (ADR 0003);
  - classificatie: intern zodra één onderliggend fragment intern is (FO 9; ADR 0001).
- Vervallen: de schrijvende route en de synchronisatieroute in `vault.py`, `obsidian_export_service.py` en `vault_sync_service.py` in app-main. Hun bruikbare delen verhuizen naar de component (zie hieronder). De container `open-notebook` krijgt geen koppeling met de vault.

**Langflow-component**

- Hergebruik, verplaatst naar een Python-module die de component importeert:
  - het atomair schrijven;
  - de padcontrole;
  - het teruglezen van aliassen uit de vault.
- **Koppeling:** `langflow` koppelt alleen `20 Atlas/28 Entities/` van de vault, niet de hele vault. Het pad is een instelling, niet een vaste waarde in code of sjabloon.
- **Plaats:** de padcontrole laat alleen schrijven toe in bestaande notities onder die koppeling. De component verwijdert, verplaatst en hernoemt geen bestanden, en schrijft niet in de dagnotitie of een onderhoudslog.
- **Soort herkennen:** aan de tag in de frontmatter van een bestaande notitie: `type/entity/organisation` (organisatie), `type/entity/region` (gebied), `type/entity/government_programme` (programma), `type/person` of `type/contact` (persoon). De component bouwt bij elke run een index, alleen lezend: per notitie pad, titel, tags, `aliases` en het id in het blok.
- **Een bestaande notitie vinden:** op het id in het gegenereerde blok. Zonder blok op een unieke overeenkomst van titel of `aliases` met de canonieke naam of een alias, binnen de notities van dezelfde soort. Bij meer dan één overeenkomst gaat het item naar review.
- **Gegenereerd blok:** tussen `<!-- signaalsysteem:begin -->` en `<!-- signaalsysteem:end -->`. Die opmerkingen zijn niet zichtbaar in de leesweergave van Obsidian. **De export schrijft alleen in dit blok.** Frontmatter en tekst buiten het blok blijven byte voor byte gelijk. Ontbreekt het blok, dan wordt het onderaan toegevoegd.
- **Inhoud van het blok:** het id van de entiteit, de classificatie, de herkomst (bron, versie, fragment-id's) en de codeboekkoppelingen. Classificatie, id en herkomst staan in het blok en niet in de frontmatter (akkoord Geert, 30-10). Een wikilink in het blok verwijst alleen naar een bestaande notitie, bijvoorbeeld een thema in `28.04 Thema's` of een gebied; anders staat de naam als platte tekst.
- **Programma's** landen in `28.06 Programma's` (Geert, 30-10). Een bestaande programmanotitie krijgt het blok.
- **Personen:** alleen gekoppeld aan een bestaande notitie in `28.01 People`; een nieuwe persoon gaat naar review (wachtrij 6, ADR 0016).
- **Nieuwe notitie — voorgesteld, Geert beslist:** optie a. Elke nieuwe entiteit (organisatie, gebied, programma, persoon) gaat naar review, met de voorgestelde canonieke naam, aliassen, soort en herkomst. De beheerder maakt de notitie met *TPL Organisation*, *TPL Region* of *TPL Person*, of voor een programma op de manier van de bestaande programmanotities. De volgende run vindt de notitie op titel of alias en vult het blok. Reden: de vaultregels vragen OSK-first, de sjablonen werken alleen in Obsidian, en velden als `categorie` en `indeling` bepalen de map; die volgen niet betrouwbaar uit fragmenten. Kiest Geert optie b, dan schrijft de component eenmalig de frontmatter van het sjabloon (`type`, `title`, `description`, `aliases`, `status: draft`, `tags`, `created`, `updated` en de velden van de soort), in de map waar de bestaande notities met dezelfde soort en dezelfde `categorie` of `indeling` staan; is dat niet precies één map, dan volgt review. Dan moet ook de paragraaf hieronder de uitzondering op OSK-first noemen.
- **Logboek:** het logboek van de export is de run (ADR 0003).
- **Sjabloon:** een sjabloonbestand voor het blok in de repository naast de component, onder versiebeheer.

**Latere fase**

- Schrijven naar de echte vault (`D:\# Vault`) is een latere fase, na een besluit van Geert. Tot dan schrijft niets in die map, en koppelt geen container haar.
- Bouw en tests (F2-21) gebruiken een tijdelijke testvault, die de test zelf aanmaakt met de map- en veldnamen hierboven en alleen openbare gegevens.
- De repository bevat geen inhoud uit de vault; alleen map- en veldnamen uit de vaultregels van `D:\# Vault`.

**Voorstel voor de vaultregels.** Geert gaf op 30-10 toestemming dat er later een korte paragraaf over het blok in `D:\# Vault\CLAUDE.md` komt. De tekst wordt pas toegevoegd als de export echt gaat draaien, na het akkoord van Geert op de tekst; het signaalsysteem wijzigt de vault niet zelf. Voorstel, in het Engels zoals de rest van dat bestand:

```markdown
## Signaalsysteem block

Some entity notes contain a block between `<!-- signaalsysteem:begin -->` and `<!-- signaalsysteem:end -->`. The signaalsysteem export (Langflow) writes it; it holds the entity id, classification (intern/extern), provenance (source, version, fragment ids) and codebook links.

- Do not edit the block by hand; the next export run overwrites it. Corrections go through the signaalsysteem review queue.
- The export writes only inside this block. It never changes frontmatter or text outside it, and never creates, moves, renames or deletes notes.
- A block marked `classificatie: intern` contains internal information. Do not copy it to cloud tools or published notes.
- Text in the block is derived, not a source. Never use a note with this block as a source for the signaalsysteem.
```

**Bestaande exports:** bestanden die de oude dienst in `Entities` schreef, worden niet automatisch verplaatst. De eerste run op de echte vault meldt ze in zijn logboek; de beheerder beslist.

## Gevolgen

- **Code:**
  - een nieuw exporteindpunt in de fork;
  - de oude exportdienst en de synchronisatie in app-main vervallen;
  - een Langflow-component met sjabloon.
- **Tests (stopcondities), op een tijdelijke testvault:**
  - twee runs na elkaar met dezelfde gegevens geven byte voor byte dezelfde bestanden;
  - frontmatter en tekst buiten het blok van een bestaande notitie blijven byte voor byte gelijk;
  - een interne entiteit krijgt `classificatie: intern` in het blok;
  - een nieuwe entiteit levert geen bestand op, maar een reviewitem (optie a);
  - een wikilink in het blok verwijst alleen naar een notitie die in de testvault bestaat;
  - geen schrijfactie buiten de koppeling van `20 Atlas/28 Entities/`, en geen nieuw, verplaatst of verwijderd bestand;
  - testvaults bevatten alleen gegevens uit openbare bronnen.
- **Afgeleide tekst:** een bronkaartregel mag niet naar de vault wijzen, en het importeindpunt weigert inhoud met het blok (ADR 0004, 0020).
- **FO 10:** bijgewerkt op `D:\# Vault` (30-10).
- **Open voor Geert:** optie a of b voor nieuwe notities; akkoord op de tekst voor `D:\# Vault\CLAUDE.md` vóór de eerste schrijvende run (F5-05).
- **Buiten de fork:** de vault-beheer-agent toetst de export aan de vaultregels (FO 10).
