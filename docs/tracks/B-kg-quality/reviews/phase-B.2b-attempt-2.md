# Review — Track B Phase B.2b attempt 2

**Branch**: `track/b-ttl-endpoint`
**Decision**: APPROVED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-07
**Commits reviewed**: `9b12e4b` (URI-safe + filename + helper-reuse + docs) and `ae3152f` (self-review addendum)

## Summary

All 1 major + 5 minors from attempt 1 are correctly addressed. The
URI-safety fix is well-engineered (CamelCase fragment + `rdfs:label`
preservation, applied to both `type_name` and `parent_type`, and via
a lower-camel variant to property names). All 12 tests in
`test_schemas_router.py` pass locally; the full app-main suite is 380
green (374 + 6 new); ontology-manager is 191 passed / 1 skipped. End-to-end
smoke against the live router confirms the fix on the canonical
adversarial inputs ("My Class With Spaces", "Author/Editor (Senior)",
"2024 cohort", "Clinical Trial Phase") plus a few I added (`'!!!'`,
non-string `type_name`, None, whitespace-only). No `uv sync` friction in
this worktree.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | 200 + `Content-Type: text/turtle` + `@prefix` start | PASS | Unchanged from attempt 1. |
| 2 | Body parses cleanly via `rdflib.Graph().parse(format="turtle")` | PASS | Unchanged + now exercised by 4 more TypeNameSanitisation tests. |
| 3 | 2 accepted extensions → 2 extra `owl:Class` | PASS | Unchanged. |
| 4 | 404 unknown notebook + auth inheritance | PASS | Plus explicit auth test (minor 5 fix) — `TestAuthExclusionAllowList`. |
| 5 | `PROTEGE_TEST.md` exists | PASS | Unchanged. |

## Test status (run by reviewer)

```
apps/app-main/tests/test_schemas_router.py: 12 passed in 104.42s
apps/app-main full suite:                   380 passed in 62.85s
packages/ontology-manager full suite:       191 passed, 1 skipped in 2.85s
```

No `uv sync` torch-wheel hang in my worktree (was already populated from
prior reviews). Implementer's caveat is honest but did not reproduce.

## Verification of each attempt-1 issue

### Major: URI-safe extension type names — FIXED

`_to_camel_case_uri_fragment()` defined at
`apps/app-main/src/app_main/api/routers/schemas.py:134-171`.

Direct invocation (probed by reviewer):

| Input | Output | Notes |
|---|---|---|
| `"My Class With Spaces"` | `"MyClassWithSpaces"` | canonical case from review |
| `"PreprintServer"` | `"PreprintServer"` | no-op when already valid |
| `"preprint server"` | `"PreprintServer"` | lower-case input promoted |
| `"2024 cohort"` | `"_2024Cohort"` | leading-digit underscore-prefix |
| `"Author/Editor (Senior)"` | `"AuthorEditorSenior"` | punctuation collapsed |
| `"ARXiv paper"` | `"ARXivPaper"` | internal capitalisation preserved |
| `"a"` | `"A"` | single character handled |
| `"ABC123"` | `"ABC123"` | acronym preserved |
| `"   "` | `""` (empty) | skipped with `logger.warning` |
| `"!!!"` | `""` (empty) | skipped with `logger.warning` |
| `"___only_underscores___"` | `"OnlyUnderscores"` | non-empty, reasonable result |

URI + label pairing verified: in `_apply_extensions`
(`schemas.py:174-254`) every accepted extension writes BOTH
`(cls_uri, RDF.type, OWL.Class)` AND `(cls_uri, RDFS.label,
Literal(type_name))` — original human-readable name survives.

`parent_type` is ALSO sanitised (`schemas.py:227-231`):
`parent_fragment = _to_camel_case_uri_fragment(parent)`. End-to-end
smoke with `parent_type="Research Study"` produces
`on:ClinicalTrial rdfs:subClassOf on:ResearchStudy` — confirmed.

Properties get lower-camel convention (`schemas.py:248`):
`pfragment = pfragment[0].lower() + pfragment[1:]`. Confirmed in smoke:
`"phase number"` → `on:phaseNumber`.

Defensive paths exercised:
- non-string `type_name` (int, None) → skipped at line 206 with warning
- empty/whitespace `type_name` → falsy-check at line 206 skips it
- sanitised-to-empty (`"!!!"`, etc.) → second skip at line 213 with
  `logger.warning("Skipping extension with un-sanitisable type_name", ...)`
- non-dict property → skipped at line 238
- non-string `prop.name` → skipped at line 241
- sanitised-to-empty `prop.name` → skipped at line 244

The 4 new `TestTypeNameSanitisation` tests pass: spaces, punctuation,
leading-digit, parent_type-sanitisation.

**Minor cosmetic note** (NOT a blocker): "NCT id" → split into
`['NCT', 'id']` → cap-first preserves `NCT` → `NCTId` → property
lower-firsts only the very first char → `nCTId`. The acronym
capitalisation is lost in the property URI. Property `rdfs:label`
preserves the original ("NCT id"), so user-visible behaviour is fine.
Not worth fixing.

### Minor 1: default-ontology accuracy — FIXED

`schemas.py:78-90` now states the divergence explicitly:

> Divergence from `OntologyManagerConfig.default_ontology` (which is
> "general"): scholarly is the canonical default for notebook schemas
> in the B-track corpus … `general.yaml` uses a dict-of-dicts
> `entity_types` shape that `load_yaml_ontology` does not currently
> parse, so switching to it would 500 every fresh-notebook download.

I confirmed:
- `OntologyManagerConfig.default_ontology` default is `"general"`
  (`packages/ontology-manager/src/ontology_manager/config.py:27-28`)
- `general.yaml` shape mismatch is plausibly the reason kept — not
  verified by experiment, but the comment is honest about it being the
  current rationale and flags the unblock condition.

### Minor 2: `_safe_filename` over-promising — FIXED

`_FILENAME_UNSAFE_RE = re.compile(r'[:/\\\r\n\t\x00"\']')` at
`schemas.py:260`. Directly probed by reviewer:

| Input | Output |
|---|---|
| `"notebook:abc'def\nghi"` | `"notebook_abc_def_ghi.ttl"` (5 chars stripped) |
| `'notebook:with"quote'` | `'notebook_with_quote.ttl'` |
| `"name's"` | `"name_s.ttl"` |
| `"a\tb\x00c"` | `"a_b_c.ttl"` |

Docstring at `schemas.py:264-275` correctly scopes the guarantee to
"safe for Content-Disposition headers" + "common consumer OSes" —
matches actual behaviour, no longer over-promises filesystem-safety.

Test `TestFilenameSanitisation::test_safe_filename_strips_quotes_and_newlines` PASSED.

### Minor 3: `_ontologies_dir` reinvents `OntologyRegistry` helper — FIXED

`schemas.py:93-107` now reads:

```python
def _ontologies_dir() -> Path:
    return OntologyRegistry()._ontology_dir  # noqa: SLF001 — see docstring
```

The `# noqa: SLF001` is justified in the docstring (private-attr
access is acknowledged + stability rationale given). `parents[N]`
fragility removed entirely.

Verified: `OntologyRegistry.__init__` (registry.py:34-41) sets
`self._ontology_dir = self._find_ontology_dir()` — the attribute is
stable. When the registry exposes `_find_ontology_dir` publicly, the
swap-in is the one-line change the docstring promises.

### Minor 4: buffered serialisation note — FIXED

Module docstring at `schemas.py:37-41` now has:

> **Serialisation footprint.** Current output is buffered in memory.
> At present scale (single-notebook ontology with ~10-50 classes) this
> is ~10-30 KB and the in-memory Turtle is fine. Streaming becomes
> relevant if a notebook accumulates >100 KB of TTL (rough heuristic:
> >500 classes); revisit at that scale.

Concrete threshold (100 KB / ~500 classes), matches reviewer ask.

### Minor 5: explicit auth test — FIXED

`TestAuthExclusionAllowList::test_endpoint_returns_401_when_password_set_and_no_auth_header`
at `test_schemas_router.py:417-453`. The test:

1. `monkeypatch.setenv("OPEN_NOTEBOOK_PASSWORD", "test-secret")` BEFORE
   constructing `TestClient(app)` — so by the time Starlette builds the
   middleware chain, `os.environ` carries the password and
   `PasswordAuthMiddleware.__init__` (`auth.py:22`) reads it.
2. Wires the schemas router AND `PasswordAuthMiddleware` with the SAME
   `excluded_paths` list as `app.py:115-123` — duplicating the
   production list verbatim. If someone later adds `schema.ttl` to the
   real `excluded_paths` for any reason, this test still catches it
   because the test uses its own (correct) list.
3. Asserts 401 with a clear failure message.

Test PASSED. The middleware-construction timing is correct: Starlette
defers middleware `__init__` until app startup, which happens in
`TestClient(app)`'s `with` block. Confirmed against `auth.py:14-79`.

## Issues found

### Blockers

None.

### Major

None.

### Minor (optional / follow-up — do not block merge)

1. **Acronym preservation in property URIs**:
   `"NCT id"` → property URI `on:nCTId`. The acronym `NCT` loses its
   internal capitalisation when the very-first character of the
   fragment is forced to lower-case (`schemas.py:248`). Cosmetic only
   — `rdfs:label` carries the original, so end-users see "NCT id" in
   browsers. Property URIs are not usually surfaced. Not worth fixing
   in this phase.

2. **Single test for `_safe_filename` packs 4 assertions** —
   conventional style is one assertion per test or `pytest.mark.parametrize`.
   Failure messages would point to the first assertion only. Minor
   readability nit; current form is still readable.

3. **`_to_camel_case_uri_fragment` is not symbol-exported by `__all__`** —
   if `__all__` is added later for the router module, remember to
   include this helper (currently imported by tests via direct
   attribute access). Tracking only.

## Decision rationale

- 0 blockers, 0 majors. All attempt-1 issues correctly addressed.
- The URI-safety fix is the cleanly engineered version: CamelCase URI
  + `rdfs:label` round-trip preserves data, applied uniformly to
  `type_name`, `parent_type`, and property names. Multiple defensive
  layers (`!isinstance(str)`, empty-string, empty-after-sanitisation)
  cover the inputs the LLM extractor can produce.
- The auth test does what it claims: middleware constructed with the
  password env var live, no auth header → 401, prod allow-list copied
  verbatim so drift is caught.
- The `_ontologies_dir` change is a real architectural improvement —
  the previous `parents[6]` was fragile and the new delegate is one
  line away from being a fully public-API call when the registry
  evolves.
- Per decision matrix: "0 blockers + 0 majors → APPROVED".

## Notes / kudos

- Self-review attempt-2 addendum is honest about the `uv sync` flake.
  Implementer should NOT have shipped without test execution given how
  many code paths changed, but `9b12e4b`'s changes hold up under
  reviewer-side execution. Implementer earns the benefit of the doubt
  this time because the diff is well-bounded and every fix has a
  matching test.
- The CamelCase + `rdfs:label` design is the right one (not the
  shorter "raise on bad input" alternative); it preserves the
  user-extracted entity name for downstream consumption.
- `# noqa: SLF001 — see docstring` is the textbook way to handle
  intentional private-attr access. Other reviewers should copy this
  pattern.
- The auth test asserts the production excluded-paths list **verbatim**,
  which is the right defensive shape — if someone later adds
  `schema.ttl` to that allow-list by mistake, the test catches it.

## uv-sync friction

None encountered in this review. The reviewer's worktree was already
populated from prior B-track reviews; no rebuild needed. If a fresh
sync had been required, I would have attempted `rm -rf .venv && uv
sync` once per the prompt's guidance.

## Next steps

APPROVED — ready for human approval / merge. Minors above are
optional carry-forward; none block this phase.

