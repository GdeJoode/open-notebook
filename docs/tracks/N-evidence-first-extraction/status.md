# Track N — status

Evidence-first extraction & abstention. Five phases, of which N.4 split into four
sub-phases under a type-boundary decision (D-N4-12) and N.5 was re-planned after
a live pipeline review moved most of its findings to Track PC.

## Phase status

| Phase | What it is | Status |
|---|---|---|
| N.1 | Candidate anchors (spaCy noun-chunks + TF-IDF, regex fallback) | SHIPPED |
| N.2 | Hearst `is_a` miner with a per-chunk precision gate | SHIPPED, **default-off since N.5b** |
| N.3 | Deterministic not-a-concept gate + LLM judge + abstention counters | SHIPPED |
| N.4a–d | Concept alignment, the type boundary, schema projection, the gap loop | SHIPPED (reports in `reviews/`) |
| N.5a | N.3's observability survives the multi-schema merge | SHIPPED |
| N.5b | `is_a` declared; the miner ships explicitly off | SHIPPED |
| N.5c | Carried residuals R2, C2–C5 | SHIPPED |
| N.5d | Regression gate + docs | SHIPPED |

## What N.5 actually found

The re-plan expected four small debts. Three of the four were already closed by
earlier phases — verified rather than assumed — and the two that were live turned
out to be the same shape as each other: **a measurement that reads as a
statement.**

**N.5a.** `_merge_results` rebuilt metadata from four keys, so every counter
`run_pass2` emitted was discarded on the multi-schema path. The plan named three
counters; it was five, and the two it missed (`entities_extracted`,
`chunk_count`) are the ones that silently zeroed both RATES. A two-pass fixture
that culled 14 entities to 5 and abstained on 9 of 20 chunk-passes measured as
`over_generation_rate` 0.00 and `abstain_rate` 0.00 — not a gap, a claim. The
`Bennett_test.pdf` case (ten chunks, zero entities) is now answerable from its own
record: `per_schema` distinguishes "the model found nothing" from "the gate
removed all seven".

**N.5b.** The review's finding was not that Hearst mining is bad but that N.2
shipped a producer whose output survived by accident. Measured before deciding:
220 raw pairs over 3823 chunks, 138 distinct, and **zero** `is_a` edges in a graph
holding 1895 relations across 100 types. The per-chunk precision gate is why; even
under a notebook-wide reading only 15 distinct pairs survive, and those include
`banken is_a voedselketen` and `PD is_a Control variables`. The decision (the
user's, on that evidence) was to declare the predicate in the three root
ontologies and ship the miner explicitly off. Neither half works alone: declaring
alone leaves a producer that produces nothing; switching off alone leaves an
undeclared predicate waiting for a flag to delete it.

**N.5c.** C2, C4 and C5 were closed in N.4c/N.4d; C3 became moot when D-N4-12
deleted the tier that carried it. R2 was live and was reproduced before being
fixed: the relation merge kept max confidence and dropped the loser's
`relation_source`, which falsified the reversibility claim the provenance exists
for. Not `is_a`-specific — it applied to any relation carrying provenance.

**N.5d.** The gate splits into a pure comparison (unit-tested, runs every suite)
and an opt-in measurement (Ollama, database, minutes per document). Its central
design decision is that a dimension with no baseline value is SKIPPED and never
PASSED, because the two metrics this track added are exactly the ones the baseline
cannot contain — the merge was discarding them when it was measured.

## Where the numbers live

- `tests/regression/n_extraction_baseline.json` — 124 entities over seven
  documents, six of which produced entities. Carries a provenance block saying
  what it predates and what its two null dimensions mean.
- `claudedocs/extraction-pipeline-review.md` — the live review that re-planned N.5.
- `scripts/n_pipeline_review_run.py` — the harness; `scripts/n_extraction_gate.py`
  compares its output against the baseline.

## Open, and where it went

Most of what the review found was not Track N's. The curator-queue writer,
cross-document identity, canonicalisation stability, the alias-policy
contradiction, the gap/proposal read path and default-configuration coherence all
moved to **Track PC — pipeline coherence**, whose PC.1 has since shipped.

Still open inside this track's code, recorded rather than fixed:

- `workflow.py:836` uses `assert self._llm_matcher is not None` on a production
  path — the same shape as C5, guarded by a caller check at line 436, but outside
  the five residuals N.5c owned.
- The baseline predates PC.1's applicability-sample fix and N.5a's counters, so
  its entity counts come from runs that mostly took the legacy single-schema path.
  Re-measuring is a live run; until then the gate's two cost dimensions skip.

## Found while closing N.5, outside this track

`pipelines/entity-filtering/tests/test_llm_matcher.py::TestMatchPair::test_calls_ollama_for_unknown_pair`
fails, and has failed since before Track PC.1 — verified by running it at
`b8a5238f`, so it is neither N.5's nor PC.1's. Not fixed here because it belongs
to Track K's code, but worth recording because the *shape* is one this track has
paid for repeatedly.

The test builds its matcher with `LLMMatcher.__new__` and sets seven attributes.
Production `__init__` sets four more (`_agentic_enabled`, `_context_provider`,
`_agentic_lower`, `_agentic_upper`), so `match_pair` raises `AttributeError` — and
the broad `except Exception` around the whole LLM call converts that programming
error into a business verdict:

```
{'match': False, 'confidence': 0.0,
 'reasoning': "error: 'LLMMatcher' object has no attribute '_agentic_enabled'"}
```

Two separate problems. The fixture is D-N4-14 rule 2 exactly — a fixture must
build the PRODUCTION argument set — and it is what makes the test fail. The
`except` is the more interesting one: any future defect inside that `try` block
will be reported as "these two entities are not the same, confidence 0.0" rather
than as an error, and a resolution run would carry on quietly.
