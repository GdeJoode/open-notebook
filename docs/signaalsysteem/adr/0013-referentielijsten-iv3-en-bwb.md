# ADR 0013: Referentielijsten voor Iv3-taakvelden en BWB-id's

- **Status:** besloten (Geert, 2026-10-30)
- **Datum:** 2026-10-29
- **Raakt FO-paragraaf:** 5 (labelschema in de bronkaartregel), 7 (entiteiten, labels, codeboek)
- **Lost op:** verschilanalyse open vraag 10

## Context

Het functioneel ontwerp (FO) koppelt entiteiten aan referentielijsten, zoals gemeentecodes van het Centraal Bureau voor de Statistiek (CBS) en id's uit het Basiswettenbestand (BWB). Labels komen uit een codeboek met de taakvelden Informatie voor derden (Iv3), brede-welvaartthema's en thema's voor signaalweging. Labels worden toegekend door Kev, met een vaste keuzelijst (FO 7). Waar de lijsten vandaan komen en in welke vorm ze worden opgeslagen, zegt het FO niet.

In de fork:

- TOOI (Thesauri en Ontologieën voor Overheidsinformatie) en CBS worden opgehaald en gecachet:
  - `services/extraction/vocabulary_manager.py` (verhuist met ADR 0012 naar `packages/shared/src/shared/vocabulary/`);
  - `packages/shared/src/shared/vocabulary/tooi_provider.py`;
  - tabel `reference_entity`;
- ontologieën hebben versies in `ontology_version` (`migrations/86.surrealql`, `87.surrealql`; Track OE). Beleidsthema's en brede welvaart staan in `packages/ontology-manager/ontologies/`;
- Iv3 ontbreekt als lijst (`rdf_owl_shacl.py:608` noemt het alleen als voorbeeld), en BWB ontbreekt helemaal.

De lijsten verschillen van aard:

- BWB-id's en gemeentecodes zijn **identificaties**: een entiteit (wet, gemeente) krijgt een sleutel;
- Iv3-taakvelden zijn **codeboektermen**: een fragment krijgt een label uit een vaste lijst.

Niet geverifieerd in deze stap: de precieze vindplaats en het formaat van de openbare Iv3-taakveldenlijst en van de open data van het BWB. Dat zoekt de bouwtaak uit, uit openbare bronnen.

## Opties

1. **Beide als ontologie met versies** (Track OE).
   - Voor: één mechanisme met versiebeheer.
   - Tegen: een wettenlijst met tienduizenden id's is geen ontologie om te bewerken, en de ontologie-editor valt onder de niet-gebruikte interface.
2. **Beide als woordenlijst** naast TOOI en CBS (`reference_entity`).
   - Voor: past bij identificaties.
   - Tegen: het codeboek voor Kev vraagt een vaste, geversioneerde keuzelijst per labelschema. Die moet een run kunnen vastleggen (ADR 0003).
3. **Per aard.** BWB als woordenlijst in `reference_entity`, net als CBS-gemeentecodes. Iv3-taakvelden als onderdeel van het codeboek, met versies in `ontology_version`, zodat een run de versie van het labelschema vastlegt.

## Voorstel

Optie 3. Een identificatie wordt opgezocht, een codeboekterm wordt gekozen. Die twee hebben verschillende eisen. Het labelschema in de bronkaartregel (FO 5) verwijst naar een codeboekversie.

Geert beslist. Hij bevestigt ook welke Iv3-versie (het jaar van de taakveldenlijst) het codeboek volgt.

## Besluit

Optie 3 (Geert, 30-10): BWB-id's als woordenlijst in `reference_entity`; Iv3-taakvelden als onderdeel van het codeboek, met versies in `ontology_version`.

Iv3-versie 2025: het codeboek volgt de taakveldenlijst 2025.

## Gevolgen

- **Code:**
  - een BWB-leverancier naast `tooi_provider.py`;
  - een laadscript voor de Iv3-taakvelden naar een ontologie met versie;
  - de validatieknoop in de graaf `entity_extraction` (ADR 0012) gebruikt `reference_entity`;
  - de Kev-component leest de keuzelijst uit de codeboekversie.
- **Testdata:** alleen uit openbare bronnen.
- **Run:** de run legt de versie van het labelschema vast in `processing_run.log` of in een eigen veld (ADR 0003).
