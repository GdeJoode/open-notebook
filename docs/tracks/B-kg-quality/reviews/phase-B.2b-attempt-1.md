# Review — Track B Phase B.2b attempt 1

**Branch**: `track/b-ttl-endpoint`
**Decision**: REVISIONS_NEEDED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-06

## Summary

The router is wired, tests pass (6/6 new, 374/374 app-main total), the
smoke output is exactly what the spec calls for, and the Protégé guide
covers the manual gate. AC1–AC4 are met at face value. **However**, an
adversarial probe of the `_apply_extensions` path found a real
unhandled-exception case: an extension whose `type_name` contains
whitespace (or any URI-illegal character) causes `rdflib`'s Turtle
serializer to raise, returning a 500 with an internal stack trace. The
B.1c writer is LLM-derived; nothing upstream enforces URI-safety. This
is the kind of bug Protégé will never see (it never reaches Protégé)
but a fresh-LLM-with-weird-output run absolutely can. One MAJOR; two
MINORs; one factual fix needed in the self-review.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | 200 + `Content-Type: text/turtle` + body starts with `@prefix` | ✅ | Live smoke: `Content-Type: text/turtle; charset=utf-8`, body starts with `@prefix on: <https://open-notebook.dev/ontology/> .` Three tests assert the prefix start. |
| 2 | Body parses cleanly via `rdflib.Graph().parse(format="turtle")` | ✅ | `TestRoundtripParse::test_body_parses_via_rdflib` instantiates a fresh `Graph` and asserts `len(g) > 0`. |
| 3 | 2 accepted extensions → 2 additional `owl:Class` declarations beyond base | ✅ | Live smoke confirmed 8 base + 2 extensions = 10 class declarations. `TestExportWithAcceptedExtensions` diffs class-subject sets against a baseline `load_yaml_ontology("scholarly.yaml")` and asserts exactly `{"PreprintServer", "Cohort"}`. |
| 4 | 404 for unknown notebook id; auth inherits notebook router pattern | ✅ | `TestUnknownNotebook` verifies 404 + `schema_repo.get_by_notebook.assert_not_called()`. Auth path-walk in `apps/app-main/src/app_main/api/auth.py:23-25` confirms `/api/notebooks/.../schema.ttl` is NOT in `excluded_paths`, so `PasswordAuthMiddleware` gates it identically to other `/api/notebooks/...` routes. |
| 5 | `PROTEGE_TEST.md` exists with clear manual steps | ✅ | 5-step script + pass criteria + failure diagnostics + screenshot-capture procedure. Screenshot deferred to first dev run (explicitly noted). |

All 5 plan acceptance criteria are met **as written**. The MAJOR
below is on a code path the plan didn't enumerate — but it's a real
production-reachable crash, not a hypothetical.

## Test status

```
apps/app-main/tests/test_schemas_router.py:
  TestExportWithAcceptedExtensions::test_returns_ttl_with_extra_owl_classes PASSED
  TestUnknownNotebook::test_returns_404_for_unknown_notebook            PASSED
  TestNotebookWithoutSchemaRow::test_returns_base_ontology_when_schema_row_absent PASSED
  TestResponseHeaders::test_content_type_is_turtle                      PASSED
  TestResponseHeaders::test_content_disposition_is_attachment_with_filename PASSED
  TestRoundtripParse::test_body_parses_via_rdflib                       PASSED
  6 passed in 113.46s

apps/app-main full suite:
  374 passed in 89.66s
```

Test counts match the implementer's claim (368 baseline + 6 new = 374).

## Issues found

### 🔴 Blockers (must fix)

*None.* The router and tests deliver what the plan called for. The
extension-URI bug below is a MAJOR, not a blocker, because the path is
reachable but only via upstream content that B.1c does not yet
produce in practice (LLM output post-normalisation is alphanumeric in
the existing test corpus).

### 🟡 Major (must fix)

1. **Extension with whitespace / URI-illegal `type_name` crashes the
   serializer (HTTP 500, unhandled exception)** —
   `apps/app-main/src/app_main/api/routers/schemas.py:119-153`
   (`_apply_extensions`).

   - Reproduction (no mocks beyond the AsyncMock pattern the existing
     tests already use):
     ```
     accepted_extensions=[
         {"extension_id": "e1", "type_name": "My Class With Spaces"},
     ]
     ```
     → `rdflib.term.URIRef.n3()` raises:
     ```
     Exception: "https://open-notebook.dev/ontology/My Class With Spaces"
     does not look like a valid URI, I cannot serialize this as N3/Turtle.
     ```
     Response: 500 with an internal stack trace.

   - Impact: B.1c's writer derives `type_name` from LLM output. Pass-1
     normalisation lower-cases + collapses whitespace + strips
     trailing punctuation, but it does NOT remove embedded
     whitespace or restrict the charset to URI-safe characters. A
     run that produces `"Open Access Repository"` as a proposed
     extension and gets accepted will turn the TTL endpoint into a
     hard 500 for that notebook until the row is repaired by hand.

   - The router docstring explicitly says it is "defensive" of
     extension shape ("missing keys → skip with warning"). The same
     defensive contract should apply to URI-safety: either skip the
     extension with a warning, or normalise the name to a URI-safe
     form (e.g., `re.sub(r"\W+", "", type_name)`). This is the same
     class of defence as the existing `type_name == ""` skip.

   - Recommendation: validate `type_name` (and `parent_type`,
     `prop.get("name")`) for URI-safety in `_apply_extensions` and
     skip/normalise with a `logger.warning` when it fails. Add a
     test for the whitespace case to lock the contract in.

   - Not a blocker only because the path is reachable but currently
     unexercised in production (B.1c is not yet wired to write
     accepted extensions; B.3b will). Catching it now is much cheaper
     than catching it after the schema-tab ships.

### 🔵 Minor (optional / follow-up)

1. **Self-review and status.md claim
   `_DEFAULT_BASE_ONTOLOGY = "scholarly"` matches
   `OntologyManagerConfig.default_ontology` — it does not.**
   The config default is `"general"` (see
   `packages/ontology-manager/src/ontology_manager/config.py:27-30`).
   Either the literal should change to `"general"` to actually match,
   or — better — the comment + the B.3a follow-up entry should be
   honest that the router intentionally diverges from the config
   default and pin the rationale (probably: scholarly is the live
   B-track corpus, and `general.yaml` is too thin for a real
   download). Either fix is fine; the current claim is just wrong.
   File: `apps/app-main/src/app_main/api/routers/schemas.py:65-68`,
   self-review §"Outstanding (deferred to later phases)" item 2,
   `docs/tracks/B-kg-quality/status.md:684-686`.

2. **`_safe_filename` is described as "filesystem-safe" but only
   strips `:` and `/`.** Control characters (CR/LF), quotes, NULs,
   etc. pass through unchanged. In practice, SurrealDB record ids
   are alphanumeric so the path is unreachable from real data, but
   the docstring overstates the guarantee. Either tighten the
   sanitiser (`re.sub(r"[^A-Za-z0-9._-]", "_", notebook_id)`) or
   relax the docstring claim.
   File: `apps/app-main/src/app_main/api/routers/schemas.py:163-171`.

3. **`_ontologies_dir()` reinvents
   `OntologyRegistry._find_ontology_dir()` with a different and more
   fragile mechanism (`parents[6]`).** The implementer caught and
   fixed the off-by-one and added a comment, but the duplication
   itself is the underlying smell — the existing helper has a
   3-level path computation that is robust to file relocation, and
   covers cwd + env fallbacks. Calling the existing helper would
   remove the entire class of off-by-one bugs from this surface.
   Defer-to-B.3a is fine; the self-review correctly flags this as
   one to lift to `app_main.dependencies` when B.3a/B.3b land.
   File: `apps/app-main/src/app_main/api/routers/schemas.py:71-95`.

4. **Streaming vs buffering for very large schemas.** A 100-extension
   notebook produces ~15 KB; a 1000-extension one would be ~150 KB.
   The router buffers the whole `graph.serialize(format="turtle")`
   in memory before returning. Not a concern at current scale, but
   worth a comment.

5. **No explicit auth-path test.** The self-review notes the global
   middleware covers the endpoint, but there's no test asserting
   that `OPEN_NOTEBOOK_PASSWORD=set` + no `Authorization` header
   returns 401. The notebook router doesn't have one either, so
   this is consistent — flagging only because the plan said
   "permission check inherits from the notebook router pattern" and
   "inherits" is unverified.

## Decision rationale

- 0 blockers.
- 1 major (extension URI-safety crash). The plan didn't specifically
  require URI validation, but production-reachable 500s on
  user-derived input are the textbook major-fix-required class of bug.
  This is an inexpensive fix that keeps a future schema-tab from
  having to debug "why does my TTL endpoint randomly 500?" three
  weeks from now.
- Minors are documentation-truth (the `default_ontology` mismatch is
  flatly incorrect in the self-review and status.md) and standard
  follow-up territory.

Per the decision matrix in the prompt — "≥1 major → REVISIONS_NEEDED".

## Notes / kudos

- Honest self-review. The off-by-one path bug story (`parents[5]` →
  `parents[6]`) is exactly the kind of artifact a reviewer wants to
  see written down so it doesn't get re-introduced.
- The "missing notebook_schema row → 200 with base ontology" carve-out
  is the right call and the test for it (`TestNotebookWithoutSchemaRow`)
  is the cleanest in the file.
- Defensive extension parser correctly skips entries with no
  `type_name`, empty strings, non-string types, and non-dict
  properties. The whitespace case slipped through but the existing
  defensive pattern is the right one to extend.
- `Content-Type: text/turtle` + `Content-Disposition: attachment`
  pair is correct, OpenAPI registration confirmed at
  `/api/notebooks/{notebook_id}/schema.ttl`, auth inherits cleanly.

## Next steps

REVISIONS_NEEDED: implementer should:

1. (MAJOR) Add URI-safety validation in `_apply_extensions` — skip or
   normalise extensions whose `type_name`, `parent_type`, or
   `properties[i].name` would produce an unserialisable URI. Add a
   test for the whitespace case.
2. (MINOR-1) Fix the `_DEFAULT_BASE_ONTOLOGY` vs
   `OntologyManagerConfig.default_ontology` claim in the self-review
   and status.md — either align the literal or correct the docs.
3. (MINOR-2..5) Optional — address in this revision or carry forward
   to B.3a as documented follow-ups.

After those, resubmit for review.
