# ADR 0016: Interface en de plek waar de beheerder zijn wachtrijen afhandelt

- **Status:** besloten (Geert, 2026-10-30)
- **Datum:** 2026-10-29
- **Raakt FO-paragraaf:** 5, 6, 7, 10, 11, 12.1, 13.2 punt 7
- **Lost op:** verschilanalyse conflict 13, open vraag 14

## Context

Het functioneel ontwerp (FO) zegt:

- de interface van Open Notebook wordt niet gebruikt, maar blijft staan zolang ze niet hindert (FO 13.2);
- Langflow is de beheeromgeving (FO 11);
- alleen de Langflow-editor en de MCP-server (Model Context Protocol) zijn bereikbaar (FO 12.1).

Tegelijk noemt het FO zeven wachtrijen waarin de beheerder beslist:

1. bronnen zonder bronkaartregel (FO 5);
2. twijfel van de poortwachter (FO 6);
3. selectie zonder treffer (FO 7);
4. twijfel bij deduplicatie, als pauzerende lus (FO 7, FO 11; ADR 0012);
5. wiki-pagina's die de controle niet halen, of verouderd zijn (FO 10);
6. nieuwe personen bij de export naar Obsidian (FO 10);
7. mislukte runs (FO 5).

Waar de beheerder die afhandelt, zegt het FO niet. Langflow heeft geen reviewscherm.

De fork heeft de interface met Track WB (afgerond 30 september 2026) ingericht als werkbak voor de pijplijn: aandachtsoverzicht, stadia en wachtrijen (`docs/tracks/WB-pipeline-workbench/plan.md`; `apps/app-main/src/app_main/api/routers/attention.py`; router `triage`; `migrations/60.surrealql`).

Na ADR 0001, 0002, 0005, 0006 en 0010 werken delen van de interface niet meer:

- uploaden, Zotero en de mapbewaker;
- privacyschakelaars;
- providerketens;
- samenvoegen en splitsen van fragmenten;
- de PDF-weergave zonder header.

Chat, ask en podcasts werken nog wel, met een sleutel met recht `beheerder` en een lokaal model.

Een MCP-route via Claude valt af voor de wachtrijen: interne items komen nooit via MCP (harde eis 1).

## Opties

1. **Alles in Langflow.** Reviewcomponenten of aparte flows per wachtrij, met invoer van de beheerder die de LangGraph-lus hervat.
   - Voor: past letterlijk bij FO 11 en 12.1.
   - Tegen: Langflow is geen reviewomgeving, dus er moet een reviewscherm worden gebouwd. Dat gaat in tegen hergebruik boven zelf bouwen (FO 2, principe 7).
2. **De werkbak van Track WB als reviewomgeving, alleen op localhost.**
   - De interface blijft, en wordt bereikbaar op `127.0.0.1` (niet via Tailscale) met een sleutel met recht `beheerder` (ADR 0002).
   - Menu-items die op vervallen routes steunen, worden verborgen. Er komen wachtrijen bij voor 1, 2, 3, 5 en 6, op het bestaande patroon van het aandachtsoverzicht.
   - Voor: hergebruik van recent en getest werk.
   - Tegen: wijkt af van FO 12.1 (een derde bereikbare dienst) en van FO 13.2 (interface niet gebruikt). Het FO moet daarvoor worden aangepast. Kleine wijzigingen in de interface zijn nodig: inloggen met een sleutel en de PDF-weergave met header.
3. **Wachtrijen als notities in de Obsidian-vault.** Langflow schrijft per item een notitie, de beheerder beslist in de frontmatter, en Langflow leest het besluit terug.
   - Voor: de beheerder werkt al in Obsidian, en interne items mogen in de vault, met label (FO 10).
   - Tegen: een omweg met bestanden voor besluiten die direct in de database horen; foutgevoelig bij het teruglezen; de vault raakt vervuild met werkvoorraad.

## Voorstel

Optie 2. De interface hindert dan niet: hij is alleen lokaal bereikbaar, hij gebruikt dezelfde API met sleutel en filter, en de werkbak doet precies wat de wachtrijen vragen. Dit vraagt een aanpassing van FO 12.1 en 13.2; Geert heeft daarover besloten (zie Besluit).

## Besluit

Optie 2 (Geert, 30-10): de werkbak van Track WB wordt de reviewomgeving, alleen bereikbaar op `127.0.0.1`, niet via Tailscale, met een persoonlijke sleutel met het recht `beheerder` (ADR 0002 B, na de review van 30-10; eerder "beheren"). FO 12.1 en 13.2 punt 7 zijn daarop aangepast (30-10).

## Gevolgen

- **Bij optie 2:**
  - het FO wordt aangepast: in 12.1 komt de interface als dienst op localhost, en in 13.2 punt 7 komt "interface alleen als reviewomgeving";
  - in `frontend/` veranderen inloggen met een sleutel en de PDF-weergave met header, en menu-items worden verborgen;
  - de wachtrijen 1, 2, 3, 5 en 6 krijgen API-eindpunten en schermen;
  - de interface wordt bereikbaar op `127.0.0.1` via de omgekeerde proxy `werkbak-toegang` op `werkbak-intern` en `toegang`, net als open-notebook-mcp (ADR 0002 F). Op `werkbak-intern` hangen alleen die proxy en `open-notebook` (aangepast na de review, 30-10). De container `open-notebook` zelf hangt niet aan `toegang`: een container aan een intern én een gewoon netwerk heeft internet (P8, K5, 29-10).
- **Bij optie 1:** de interface krijgt geen gepubliceerde poort meer, en er komen reviewcomponenten in Langflow.
- **In beide gevallen:** de deduplicatielus hervat via de triage-API (ADR 0012). De plek van de bediening verandert daar niets aan.
