# D2 experiment — GLinker against TOOI, on the ministry rows

**Question.** For D2 ("load TOOI and give `reconcile_entity` a caller"): does
GLinker (Knowledgator, entity linking on GLiNER) resolve this corpus's ministries
to the right TOOI organisation, and what does its scoring model add over a
lookup?

**Answer, for ministries only.** Its scoring model adds nothing net.
- GLinker's own string lookup (exact hit first, else fuzzy top-1) gets 14 of 18
  with 2 wrong. The scoring model on top gets the same 14 and 2: it fixes one
  error and makes another.
- A decoration-aware exact lookup (a simulation of D1) also gets 14 of 18, with
  **no** wrong link among the 18. It does link one of the 2 rows that should not
  link.
- The cheap, checkable fixes lie in the TOOI loader and the K.4 lookup, not in a
  model.

Script `scripts/d2_glinker_experiment.py` (`fetch`, `baseline`, `glinker`,
`score`). Data under `claudedocs/d2-glinker/`:
- `tooi_records.json`: 1,438 organisations via the repo's own `TOOIBulkFetcher`.
- `mentions.json`: the 20 active `government_organization` rows.
- `gold.json`: stated to be written before scoring. That is not verifiable from
  git, because gold, script and results arrived in one commit.
- `baseline.json`, `glinker.json` and `results.json`.

## Setup

- **Mentions.** All 20 active `government_organization` rows in `staging`.
  - 18 have a correct TOOI answer. 2 must **not** link: a string listing four
    ministries, and "De Staatssecretaris van IenW", an office. PC.3 recorded that
    one as a relation.
  - **Gold rule for party designations.** Covenants introduce each signatory as a
    person ("De Staatssecretaris van …, hierna te noemen: IenW"). The gold is
    decided by the *extracted name*: "IenW (…)" and "HG/BZK (…)" name the party
    and link to the ministry; "De Staatssecretaris van … (IenW)" names the office
    and does not. This rule penalises only the D1 simulation, which links the
    office row.
  - **EZ.** The gold accepts both "Economische Zaken" records. TOOI does say
    which is right: mnre1040 ended 2010-10-13, and the corpus (the Schoof cabinet)
    points to mnre1045. The double gold is kept as written: returning exactly the
    accepted records counts as correct. This credits two rows, D1 and the L2 exact
    lookup on the text form, which both return both records. Under a strict single
    gold (mnre1045) both answers become ambiguous, which `reconcile_entity`
    refuses: D1 goes from 14 to 13, and L2 exact on the text form from 12 to 11.
- **Context.** The first chunk containing the name, or its part before " (". 16
  of 20 found one, often a single line. The other 4 use the name as text.
- **Mechanisms.**
  - **(a) K.4 as shipped.** `TOOIProvider.lookup` over an in-memory stand-in for
    `reference_entity`'s two queries, `canonical_name = $name` and
    `$alias IN aliases`. Both are case-sensitive: migrations 41, 55 and 57 define
    no collation. `reconcile_entity` passes `entity.canonical_name`, as the
    baseline does.
  - **(b) GLinker L2** (DictLayer). Exact is lower-cased over label and aliases.
    Fuzzy is rapidfuzz ratio ≥ 0.6 with prefix_length 1. It is run on the stored
    name, and separately on the form in the text ("VRO"). The *pipeline* row is L2
    as GLinker runs it: an exact hit first, else the fuzzy top-1.
  - **(c) GLinker L3**, `gliner-linker-{base,large}-v1.0`. Template
    `{label}: {description}` is GLinker's builder default. `threshold=0.0` was
    set, so the model always answers; the rows at GLinker's default of 0.5 are
    also given.
    - **Candidates come from the stored name**, but L3 scores the **span in the
      text** ("VRO"). That mix is a choice of this script.
    - `over_l2` scores L2's candidates (max 5).
    - `over_all_ministries` offers **17** labels, not 19. Two TOOI records share
      the label of another ("Economische Zaken" mnre1040/1045, "Landbouw, Natuur
      en Voedselkwaliteit" mnre1150/1153), and the label de-duplication drops one
      of each.
  - **D1 simulation.** Not shipped code, and it says more than "exact lookup":
    - It lower-cases (casefold).
    - It strips "ministeries van het Rijk:", "(het) ministerie van" and
      "(Ministerie)", and splits "ABBR (gloss)". That is a wider scope than
      `component-decisions.md`'s D1, which counted `X (Ministerie)` and
      `ABBR (X)` for 7 of 20. The list-prefix strip matches 2 of these strings
      and gains exactly 1.
    - It answers when every piece that *hits* agrees, and it ignores pieces that
      miss. That is why it links the Staatssecretaris row through its "(IenW)"
      gloss. The affix rule forbids this (an organ of X is not X); a real D1 must
      treat the role prefix as blocking, not as a miss.

## Result

Of the 18 that should link; the last column is the 2 that should not.

| Mechanism | correct | wrong | refused | no-link rows declined |
|---|---|---|---|---|
| (a) K.4 exact, case-sensitive | 2 | 0 | 16 | 2/2 |
| (b) L2 exact, stored name | 4 | 0 | 14 | 2/2 |
| (b) L2 exact, form in text | 12 | 0 | 6 | 2/2 |
| (b) L2 fuzzy top-1, stored name ¹ | 10 | 2 | 6 | 2/2 |
| **(b) L2 pipeline, stored name** | **14** | **2** | 2 | 2/2 |
| (c) L3 base or large over L2, threshold 0 | 14 | 2 | 2 | 2/2 ² |
| (c) L3 base or large over L2, threshold 0.5 | 12 | 2 | 4 | 2/2 ² |
| (c) L3 base over 17 ministry labels ³ | 6 | 12 | 0 | 0/2 |
| (c) L3 large over 17 ministry labels ³ | 8 | 10 | 0 | 0/2 |
| **D1 simulation + exact** | **14** | **0** | 4 | 1/2 |

¹ Excludes exact hits by construction, so 4 of its 6 refusals (IenW, JenV, EZK,
VWS) are artefacts; the pipeline row is the fair one.
² Only because L2 returned no candidates.
³ Confounded by stale labels (see the next section), so no claim is made here
about the model's own ability.

**L3 over L2 against L2 alone.** Identical totals, for different reasons:
- L3 fixes "VRO (VROM, …)", where L2's top-1 was VROM.
- L3 breaks "Volkshuisvesting en Ruimtelijke Ordening (Ministerie)" → VROM
  (mnre0180), at 0.92 for base and 0.88 for large. 0.92 is the highest base score
  in that row set, so no threshold removes it.
- The other shared error, "VRO (Ministerie van VWS)" → VWS, is **L2's**. Fuzzy
  matching on the stored name offered one candidate, VWS, and at threshold 0 L3
  could only return it.

**D1's 4 refusals.**
- 1 is the corpus spelling "Koninkrij**k**relaties": "… (Ministerie)".
- 2 are slash compounds ("HG/BZK (…)", "…/Herstel Groningen"). They are not in
  the index, whatever the spelling.
- 1 is "VRO (… VWS)", whose pieces disagree. It is scored as a refusal; the gold
  is mnre1171.

## The TOOI loader: the real cause of the stale labels

TOOI marks the end of an organisation with `ont/einddatum`, and the end of one of
its *names* with `ont/einddatumHV`. `tooi_bulk._node_has_einddatum` checks only
`einddatum`. So every historical name counts as current, and the record's label
is its first, oldest name. `IenW` resolves to the right URI (mnre1130), labelled
"Infrastructuur en Milieu"; `JenV` to mnre1058, labelled "Justitie".
Source: the immutable register dump
`https://repository.officiele-overheidspublicaties.nl/waardelijsten/rwc_ministeries_compleet/6/json/rwc_ministeries_compleet_6.json`.
There, IenM's historical version `ministerie/hv_06784634` carries `einddatumHV`
2017-12-31, `ministerie/mnre1171` carries `einddatum` 2026-02-22, and
`ministerie/mnre1040` carries `einddatum` 2010-10-13. `tooi_records.json` does not
contain these dates, because the loader drops them.
`test_tooi_bulk.py` puts `einddatum` on its synthetic historical nodes, a shape
TOOI does not produce. So the test passes on data that does not occur.

**Consequences.**
- The L3 rows over 17 labels were offered "Infrastructuur en Milieu" and
  "Justitie" for two of the golds (mnre1130 and mnre1058), and one stale non-gold
  label ("Landbouw, Natuur en Voedselkwaliteit", mnre1153). The reviewer
  counted 4 of base's 12 and 2 of large's 10 wrong answers on those golds. With
  current labels the model might do much better, so that row says nothing about
  the model.
- **"Prefer organisations that still exist" is wrong.** TOOI ends mnre1171 (VRO)
  on 2026-02-22, and VRO is the gold for 6 of the 18 mentions. The corpus is from
  a cabinet in which VRO existed. The checkable filter is **valid on the
  document's date** (`begindatum`/`einddatum` against a document date), and that
  needs a document date the pipeline does not yet carry.

## Side finding

GLinker fails silently without rapidfuzz. `DictLayer.search_fuzzy` prints a
warning *on every call* and returns nothing when `rapidfuzz` is absent, and
`glinker` 0.1.1 does not declare it as a dependency. The first run of this
experiment measured nothing for that reason; the script now refuses to run
without it.

## What this says about D2

**For ministries:**
- A decoration-aware exact lookup matches GLinker's full pipeline (14 of 18) with
  no wrong link among the 18.
- Its one error, the office row, is a rule violation of the simulation, fixable
  by treating role prefixes as blocking.
- The scoring model contributes nothing net here.

**Checkable fixes, in order of evidence:**
1. Read `einddatumHV` in `tooi_bulk`, and fix the test to use the shape TOOI
   publishes. This fixes the labels.
2. Make the K.4 lookup case-insensitive. That alone takes it from 2 to 4, as (b)
   shows.
3. Use validity **on the document's date**, not "still exists", once a document
   date is available.

**Not measured.** Everything beyond ministries:
- TOOI as loaded holds 605 gemeenten, 12 provincies and 32 waterschappen.
- PC.3 counts 53 active `administrative_area` rows (`phase-PC.3-plan.md`). Those
  are the rows D2 would reach next, and they bring the "Bergen" kind of ambiguity
  and the question of a place against its government.
- Topics (429 of 531 active entities, per `component-decisions.md`) have no
  authority at all.

**Limits.** 20 mentions, one class, one corpus, context often one line. Whether
any model weakness is Dutch or the model cannot be separated here.
