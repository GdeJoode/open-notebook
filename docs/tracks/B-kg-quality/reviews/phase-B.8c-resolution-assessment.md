# B.8c — Cross-document entity-resolution assessment (2026-06-21)

Live qwen2.5:14b extraction over 4 related **Regio Deal Convenant** PDFs (same
programme → should share entities), measuring how well the V1 resolver merges
shared entities across documents.

## Setup
- Model: `qwen2.5:14b-instruct-q5_K_M` (the independent `default_extraction_model`, B.8a-2).
- Sources (notebook `t6639pcftyishmguonmc`), per-doc persisted entities:
  - Het Hogeland 200 · Noord-Holland Noord 333 · Midden-Limburg 250 · Zuidwest-Friesland 274
  - (plus the bc6xa academic-paper proof run: 421)
- Resolver under test: V1 `name_normalizer` (lowercase + strip-punct + whitespace-collapse) → upsert by `(canonical_name, entity_type)` UNIQUE, + embedding-dedup@0.90 in filtering. Full TOOI/Crossref + NL normalization is **deferred to Q9 / Track M4**.

## Metric
- **897 distinct canonical entities** across the 4 docs.
- **107 entities span ≥2 docs** — cross-document resolution demonstrably works.
- Shared across **all 4 docs** (single canonical entity each): `Regio Deal`,
  `Regio Envelop`, `ministerie van BZK`, `Minister van Volkshuisvesting en
  Ruimtelijke Ordening`, `J.J.M. Uitermark` (person/signatory), `Regiokassier`,
  `Artikel 14`, `VRO`, `Actieagenda Sterk Bestuur`,
  `Volkshuisvesting en Ruimtelijke Ordening`.

## Verdict: PARTIALLY MET — works for consistent surface forms, capped by V1 on variants

Cross-document resolution **works**: the programme-level entities that should be
shared across these related docs (the programme, the funding instrument, the
ministry, the signing minister, key persons) **do** resolve to single canonical
entities. The mechanism (normalized-name + type match) is sound and the qwen
extraction produces consistent enough surface forms for the obvious cases.

But the V1 exact-normalized matcher **fragments variant forms** of the same
real-world entity — the documented Q9/M4 ceiling. Concrete fragmentation found:

| Real entity | Fragments (distinct canonical rows) | Cause |
|---|---|---|
| Ministry of BZK | `BZK`, `ministerie van BZK` + 6× `(De )?(Minister(ie)? van )?Binnenlandse Zaken en Koninkrij(k\|ks)relaties` | abbreviation↔full; role-prefix; **spelling variant** (Koninkrij**k**relaties vs Koninkrij**ks**relaties) |
| Min. of Volkshuisvesting/RO | 6 forms (`Minister van…`, `Ministerie van…`, `De Minister van…`, bare, `…(VRO)`) | role-prefix; abbreviation |
| Regio Deal | 16 forms (`De Regio Deal`, `Oost-Groningen Regio Deal`, tranche-`Besluit`en, `Partij van de Regio Deal`) | qualifiers; sub-concepts conflated |
| "minister" (role) | 23 forms — each ministry's minister a separate entity | no role/org canonicalization |

So a user asking "show everything about BZK across these deals" would see the
entity split ~8 ways. **High resolution between related docs is achieved for
identical surface forms but not for abbreviations, role-prefixed names, or
spelling variants** — exactly what the V1 stub cannot do.

## Recommendation
This confirms the Track B plan's premise: real cross-document resolution needs
**Q9 / Track M4** — TOOI (Dutch government vocabulary) + Crossref, NL-aware
normalization (abbreviation expansion: BZK ↔ Binnenlandse Zaken; role/org
canonicalization: "Minister van X" → the org), and fuzzy/spelling-tolerant
matching. Pull M4 forward if cross-doc resolution quality matters; the
swap-point hook (`from shared.utils.name_normalizer import
normalize_entity_name`) means M4 drops in without rewiring extraction.

Quick partial wins available before full M4: (a) spelling-variant tolerance
(the Koninkrij(k|ks) typo); (b) strip leading articles/role-prefixes
("De ", "Minister(ie) van ") before the match key; (c) an abbreviation alias
table for the common govt orgs. These are cheap and would collapse a large
share of the fragmentation observed here.

## Caveats on this run
- All four extraction jobs report **"failed"** despite persisting entities — the
  separate post-persist `_save_result` KeyError (B.8d follow-up); entities +
  relations persisted correctly, only the raw-result save + job status are
  affected.
- Counts are post-filtering persisted entities; the embedding-dedup@0.90 in
  filtering already merged some near-duplicates before this measurement.
