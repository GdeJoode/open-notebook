# D2 experiment — GLinker against TOOI, on the ministry rows

**Question.** For D2 ("load TOOI and give `reconcile_entity` a caller"): does
GLinker (Knowledgator, entity linking on GLiNER) resolve this corpus's ministries
to the right TOOI organisation, and what does its scoring model add over a
checkable lookup?

**Answer.** Almost nothing for this class. Stripping the decoration (D1) and
then doing an exact lookup resolves 13 of 18 with **zero** wrong links. GLinker's
scoring model resolves 14, but at the price of 2 wrong links at high score (0.92
and 0.76 for base; 0.88 and 0.55 for large), which no threshold separates from
correct ones. Left to choose among all 19 ministries on its own, the model is
wrong more often than right, and never declines.

Script `scripts/d2_glinker_experiment.py` (`fetch`, `baseline`, `glinker`,
`score`). Data under `claudedocs/d2-glinker/`:
- `tooi_records.json`: 1,438 organisations via the repo's own `TOOIBulkFetcher`.
- `mentions.json`: the 20 active `government_organization` rows, with a context
  line each.
- `gold.json`: written by hand before any mechanism was scored.
- `baseline.json`, `glinker.json` and `results.json`.

## Setup

- **Mentions.** All 20 active `government_organization` rows in `staging`.
  - 18 have a correct TOOI answer.
  - 2 must **not** link: a list of four ministries in one string, and "De
    Staatssecretaris van IenW", an office rather than the ministry. PC.3 recorded
    that one as a relation.
  - One gold answer accepts either of two records: TOOI holds two "Economische
    Zaken" records (mnre1040 and mnre1045), and nothing in the loaded data says
    which is current.
- **Context.** The first chunk containing the name, or its part before " (".
  Many chunks are a single line, for example "De Minister van … hierna te noemen:
  VRO;". 16 of 20 have one; the other 4 use the name itself as text.
- **Mechanisms.**
  - **(a) K.4 as shipped.** `TOOIProvider.lookup` over an in-memory stand-in for
    `reference_entity` that implements its two queries exactly: case-sensitive
    equality on the name or an alias.
  - **(b) GLinker L2** (DictLayer). Exact lookup is case-insensitive over label
    and aliases. Fuzzy uses rapidfuzz ratio ≥ 0.6, top-1. It is run on the name
    as stored, and separately on the form in the text ("VRO").
  - **(c) GLinker L3**, `gliner-linker-{base,large}-v1.0`, template
    `{label}: {description}`, restricted to the mention's span.
    - `over_l2`: scores L2's candidates (exact + fuzzy, max 5).
    - `over_all_ministries`: scores all 19.
  - **D1 simulation.** Not shipped code. It strips "ministeries van het Rijk:",
    "Ministerie van", "(Ministerie)", splits "ABBR (gloss)", and runs the exact
    lookup on each piece. An answer only when every piece that hits agrees.

## Result

Of the 18 that should link; the last column is the 2 that should not.

| Mechanism | correct | wrong | refused | ambiguous (contains gold) | the 2 no-link cases refused |
|---|---|---|---|---|---|
| (a) K.4 exact | 2 | 0 | 16 | 0 | 2/2 |
| (b) L2 exact, stored name | 4 | 0 | 14 | 0 | 2/2 |
| (b) L2 exact, form in text | 11 | 0 | 6 | 1 | 2/2 |
| (b) L2 fuzzy, top-1 | 10 | 2 | 6 | 0 | 2/2 |
| (c) L3 base over L2 | 14 | 2 | 2 | 0 | 2/2 ¹ |
| (c) L3 large over L2 | 14 | 2 | 2 | 0 | 2/2 ¹ |
| (c) L3 base over all 19 | 6 | 12 | 0 | 0 | **0/2** |
| (c) L3 large over all 19 | 8 | 10 | 0 | 0 | **0/2** |
| **D1 sim + exact** | **13** | **0** | 4 | 1 | 1/2 |

¹ Only because L2 returned no candidates. The model itself never declines.

**L3's two wrong links** are identical for base and large:
- "Volkshuisvesting en Ruimtelijke Ordening (Ministerie)" → VROM (mnre0180), at
  0.92 for base and 0.88 for large.
- "VRO (Ministerie van Volksgezondheid, Welzijn en Sport)" → VWS (mnre1025), at
  0.76 and 0.55. The model followed a gloss the extractor invented; the text says
  "De Minister van Volkshuisvesting en Ruimtelijke Ordening … hierna te noemen:
  VRO".

**What every mechanism fails on:**
- **TOOI's ended organisations.** VROM and Verkeer en Waterstaat are the most
  frequent wrong answers. The fetcher drops `einddatum` (`tooi_bulk.py`), so
  nothing can prefer an organisation that still exists. That filter would be a
  checkable fix, not a score.
- **A spelling variant.** The corpus writes "Koninkrij**k**relaties" without the
  s. This causes 3 of the 4 D1 refusals.
- **Glosses the extractor invented.** "VRO (VROM, …)" and "VRO (… VWS)". D1
  refuses the second correctly, because its pieces disagree.
- **Offices.** The Staatssecretaris row is D1's only wrong answer, via its
  "(IenW)" gloss. Office vs ministry is a typing question, not a lookup one.

## Side findings

1. **K.4's label is the oldest name.** `IenW` resolves to the right URI
   (mnre1130), labelled "Infrastructuur en Milieu"; `JenV` resolves to mnre1058,
   labelled "Justitie". The URI is right, and a curator reading the label would
   think it wrong.
2. **GLinker fails silently without rapidfuzz.** Its fuzzy search prints one WARN
   and returns nothing when `rapidfuzz` is absent, and `pip install glinker`
   (0.1.1) does not install it. The first run of this experiment measured nothing
   for that reason; the script now refuses to run without it.

## What this says about D2

- **Resolution.** D1 then exact lookup against TOOI is sufficient for named
  government organisations here, and it produces no wrong links.
- **Two cheap checkable improvements come before any model:**
  - Keep `einddatum`, and prefer organisations that still exist.
  - Make the lookup case-insensitive. K.4's case-sensitive equality is why (a)
    gets 2 where (b)-exact gets 4.
- **Where GLinker's L3 fits, if anywhere.** A proposer for the mentions the lookup
  refuses, routed to the curator queue with its score. It should never write an
  identity.

**Limits.** 20 mentions of one entity class, one corpus, context often one line.
Whether the model's weakness is Dutch or the model itself cannot be separated
here. Topics (429 of 531 active entities, no authority) are untouched by all of
this.
