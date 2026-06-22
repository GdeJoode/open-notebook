# Entity-typing analysis — extraction output vs the schemas (2026-06-22)

Analysis of the live KG (2,357 active entities, the Regio-Deal / Dutch-gov corpus)
against the ontologies that drive extraction. Triggered by the K.7b finding that
`VRO` and `minister van VRO` were both typed `other`, defeating the `(name,type)`
resolution guard.

## 1. What the extractor actually produced

| entity_type | count | % | note |
|---|---|---|---|
| concept | 819 | 35% | generic — no raw label; LLM emitted "concept" literally |
| **other** | **628** | **27%** | the mistyped bucket — rich types destroyed here |
| organization | 279 | 12% | |
| person | 259 | 11% | |
| topic | 219 | 9% | generic — no raw label |
| location | 94 | 4% | |
| product | 36 | 2% | |
| event | 19 | <1% | |
| legislation | 3 | — | |
| dataset | 1 | — | |

**Top raw labels dumped into `other`** (what the LLM *correctly* emitted, then lost):
`RegioDeal` (59), `ABBREVIATION` (55), `Technology`/`Technologie` (59), `Gemeente`
(39), `Persoon` (33), `Organisatie`/`Organisation` (37), `Ministerie` (27),
`Publication` (17), `Programma` (17), `Document` (14), `Sector` (10), `Wethouder`
(7), `Thema` (6), `Regio` (5), `Project` (4), `Amount` (4).

## 2. What the schemas make explicit (they're GOOD)

The ontologies (`packages/ontology-manager/ontologies/`) are well-designed,
layered, and Dutch-aware:

- **schema_core** — schema.org base: `Thing`, `CreativeWork`, `Article`, …
- **government** (extends schema_core) — `Gemeente`, `Provincie`, `FunctioneleRegio`,
  `Wijk`, `Grensregio`, `Land`, `BuitenlandseRegio`, `Ministerie`, `Uitvoeringsorganisatie`.
- **instruments** (extends government) — `Wet`, beleidsinstrumenten.
- **deals** (extends instruments) — `Deal`, `RegioDeal` (with `tranche`, `themas`, `pijlers`, budgets).
- **policy** (extends schema_core) — `PolicyBrief`, `Legislation`, …
- **policy_themes** — `BeleidsThema`, `BeleidsPijler`, `Indicator`, `BredeWelvaart`,
  `MaterieleWelvaart`, `Gezondheid`, `WonenLeefomgeving`, `Vergrijzing`,
  `BevolkingsDaling`, `VoorzieningenNiveau`, `Leefbaarheid`, …
- Plus `scholarly`, `social_profiles`, `general`.

And the extraction prompt (`ontology_manager/prompts.py:339`) **explicitly tells the
LLM**: *"PRIORITY: Use domain-specific types (e.g., Gemeente, Ministerie, RegioDeal)
over generic types (e.g., Organization, Place)."* The LLM obeys.

## 3. The three disconnects (the actual problems — NOT the schemas)

### 3a. CRITICAL — the persistence enum destroys the rich ontology types
`entity_persistence_service.py::_normalize_entity_type` flattens every LLM label
onto a fixed 20-type generic enum (`_ALLOWED_ENTITY_TYPES`: person, organization,
government_organization, administrative_area, …). It has **no bridge from the
ontology vocabulary**: `Gemeente`→lowercased→not in enum + not in the (English-only)
alias map → **`other`**. Confirmed: those rows have **empty `type_tags` and null
`primary_type`** — the rich type is not preserved anywhere; it is gone.
- The system fights itself: the prompt demands rich types, the LLM produces them,
  persistence deletes them. 27% of the KG.
- The enum DOES have `government_organization` + `administrative_area` — `Ministerie`
  and `Gemeente` simply never reach them (no mapping).

### 3b. The alias map is English-only on a Dutch corpus
`_ENTITY_TYPE_ALIASES` (B.8c) covers `org/per/loc/gov/...` (English/NER). It is
missing every Dutch term the LLM emits: `Persoon`→person, `Organisatie`→organization,
`Ministerie`/`Uitvoeringsorganisatie`→government_organization, `Gemeente`/`Provincie`/
`Regio`→administrative_area, `Wethouder`→person, `Thema`→topic, `Technologie`→(no slot),
`Programma`/`RegioDeal`→(no slot). And `ABBREVIATION` (55) is not a type at all —
the LLM used the *form* as the label.

### 3c. Abstract entities fall to generic `concept`/`topic`, ignoring `policy_themes`
44% of the KG is generic `concept` (819) + `topic` (219) with no raw label — the LLM
emitted them literally as "concept"/"topic" rather than the rich `policy_themes`
types (`BeleidsThema`, `BeleidsPijler`, `BredeWelvaart`, `Leefbaarheid`, …) that exist
for exactly the Regio-Deal themes/pijlers. And **`notebook_schema` is empty** — there
is no per-notebook schema configured, so extraction relies on `detect_applicable_schemas`
auto-detection; `government`+`deals` clearly fired (their types appear), but
`policy_themes` likely did NOT, so themes degrade to generic.

### 3d. Missing canonical types
`Technology` (59), `Programma`/`RegioDeal`/`Project`/`Deal` (programmes/instruments,
80+), `Document`/`Publication` (31) have no enum slot even semantically.
`creative_work`/`policy_document` exist but aren't mapped from `Document`/`Publication`.

## 4. Recommendations (prioritized)

**P1 — stop flattening: bridge ontology types → canonical, and PRESERVE the rich type.**
- Add an **ontology-type → canonical-enum map** (single source, derived from the
  ontologies or a curated table): `Gemeente`/`Provincie`/`Wijk`/`FunctioneleRegio`/
  `Grensregio`/`BuitenlandseRegio`/`Regio`→`administrative_area`;
  `Ministerie`/`Uitvoeringsorganisatie`→`government_organization`; `Wet`→`legislation`;
  `PolicyBrief`/`Document`/`Publication`→`policy_document`/`creative_work`;
  `BeleidsThema`/`BeleidsPijler`/`Thema`→`topic`; `Indicator`/`BredeWelvaart`/…→a theme type.
- **Populate `primary_type` + `type_tags` with the ORIGINAL ontology type** (the B.1a
  fields are empty today) so the rich type survives; `entity_type` becomes the coarse
  canonical projection (for dedup/filter), the ontology type stays for display/semantics.
  This is the architecturally-correct split.
- The cleanest place: the persistence `_normalize_entity_type` consults the ontology
  type-map + stamps `primary_type`/`type_tags`.

**P2 — Dutch aliases + drop non-type labels.**
- Extend `_ENTITY_TYPE_ALIASES` with the Dutch set above (cheap, immediate ~150-entity
  recovery from `other`). Map `ABBREVIATION`/`Amount` to a sensible fallback (these are
  extraction-noise — `ABBREVIATION` should re-prompt for the real type).

**P3 — add the missing canonical types** the domain needs: `programme` (Deal/RegioDeal/
Programma/Project), `technology`. Either extend `_ALLOWED_ENTITY_TYPES` (+ a migration
to relax the live enum, like migration 50) or — better — make `entity_type` accept the
ontology type directly and keep a coarse `primary_type`.

**P4 — apply `policy_themes` (+ configure notebook schemas).**
- The 44% generic concept/topic is a schema-APPLICATION gap, not a schema-quality gap.
  Ensure `detect_applicable_schemas` selects `policy_themes` for Regio-Deal docs, or set
  a per-notebook `notebook_schema` (base=`deals`/`policy_themes`) so themes get their
  rich types. Re-measure the concept/topic share after.

**Per-schema notes** (the schemas themselves are sound; these are refinements):
- **government / deals / instruments / policy / policy_themes**: high quality, correctly
  Dutch + layered. Their ONLY gap is they don't declare a `maps_to:`/canonical mapping
  per `entity_type` — add a `canonical: <enum>` field per type so the persistence bridge
  is data-driven (not a hand-maintained map).
- **schema_core**: the schema.org base is good; `Thing` is the catch-all — entities
  landing on bare `Thing`/`concept` are a signal the domain schema wasn't applied (P4).
- **scholarly / social_profiles / general**: not the bottleneck for this corpus (gov/policy
  docs); revisit when the corpus shifts.

## 5. Bottom line
The schemas are **good and the LLM uses them correctly**. The 27%-`other` + 44%-generic
problem is **downstream**: a persistence flattening layer (the B.8c enum + English alias
map, originally a workaround for the live `entity_type` SCHEMAFULL constraint) that
discards the rich types, plus a schema-application gap for themes. Fix the bridge +
preserve `primary_type`/`type_tags` + Dutch aliases + apply `policy_themes`, and the
typing — and therefore the resolution `(name,type)` guard, the K.7 relation safety, and
the eventual K.8 role modeling — all get materially better.
