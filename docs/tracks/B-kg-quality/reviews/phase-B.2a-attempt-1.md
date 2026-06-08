# Review — Track B Phase B.2a attempt 1

**Branch**: `track/b-ttl-exporter-fix` (HEAD `030a81d`)
**Decision**: APPROVED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-06

## Summary

Bug accurately characterized and cleanly fixed using the sentinel pattern called
out in the plan. Tests assert real triple-set equality (not "no error") over two
ontologies, and the pyshacl parsability check actually runs and passes when
pyshacl is installed. All six public functions raise a clear `ImportError` when
`RDFLIB_AVAILABLE=False`. Test counts (192 / 128 / 368) reproduce.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | Existing tests in `packages/ontology-manager/tests/` continue to pass | PASS | 192 passed (188 pre-existing + 4 new), zero regressions. |
| 2 | Roundtrip test asserts triple-set equality for `scholarly.yaml` AND `policy.yaml` | PASS | `set(g1.triples((None, None, None))) == set(g2.triples((None, None, None)))` literally — exact match to plan wording. Both fixtures exist, both produce non-empty graphs, both roundtrip identically. |
| 3 | `rdflib` import errors no longer surface at module load when rdflib IS installed | PASS | `test_rdflib_imports_succeed_at_module_load` asserts `RDFLIB_AVAILABLE is True`, `ON`/`ONR` non-None, `DTYPE_MAP` populated, `HAS_RDFLIB` alias resolves. Verified independently via fresh `python -c "from ontology_manager.rdf_owl_shacl import ..."`. |

## Test status

```
cd packages/ontology-manager && uv run pytest -q
192 passed in 0.31s
  # includes test_ttl_roundtrip.py: 4 passed when pyshacl present, 3 passed + 1 skipped when not

cd packages/shared && uv run pytest -q
128 passed in 0.21s

cd apps/app-main && uv run pytest -q
368 passed in 12.22s
```

All numbers reproduce the implementer's claims exactly.

## Issues found

### Blockers (must fix)

None.

### Major (must fix)

None.

### Minor (optional)

1. **`validate_entities` lacks `_require_rdflib()` guard** — `packages/ontology-manager/src/ontology_manager/rdf_owl_shacl.py:408-456`
   - Issue: every other public function calls `_require_rdflib()` first, but `validate_entities` only checks `HAS_PYSHACL`. If pyshacl is somehow present while rdflib is missing, the function will `NameError` on `Graph()` at line 440 rather than raise the clean `ImportError`. The implementer explicitly flagged `validate_entities` as out-of-scope for B.2a (untested, downstream work for B.2c/B.3), so this is consistent with the declared scope — but worth adding a 1-line guard for hygiene alongside the rest.
   - Recommendation: add `_require_rdflib()` at the top of `validate_entities` next time someone touches it.

2. **`create_brede_welvaart_skos` has no top-level guard** — `packages/ontology-manager/src/ontology_manager/rdf_owl_shacl.py:568`
   - Issue: relies transitively on `create_skos_scheme` to raise. Works (verified — the test probe surfaced `ImportError: rdflib required for TTL operations…`), but the error message becomes confusingly distant from the call site, and any future direct rdflib reference added to this function would silently break it. Defensive only.
   - Recommendation: optional follow-up alongside the `validate_entities` cleanup.

3. **`_demo` hardcodes `/mnt/e/Repos/Private/open-notebook`** — `packages/ontology-manager/src/ontology_manager/rdf_owl_shacl.py:625`
   - Issue: this default `PROJECT_ROOT` is user-specific. Documented in status.md as a follow-up. Not in B.2a scope.

4. **`load_all_ontologies` swallows per-file exceptions silently** — `packages/ontology-manager/src/ontology_manager/rdf_owl_shacl.py:286-289`
   - Issue: `except Exception as e: logger.error(...)` masks YAML parse errors as a single log line. Documented in status.md as a follow-up.

## Kudos

- The exact-name rename `HAS_RDFLIB → RDFLIB_AVAILABLE` with a backwards-compat alias is good hygiene — won't break any downstream consumer (zero such consumers exist today, but defensive).
- `_require_rdflib()` is called at the TOP of every in-scope public function (`load_yaml_ontology`, `load_all_ontologies`, `export_ontology`, `generate_shacl_shapes`, `create_skos_scheme`). Failures surface at the call site, not deep in a `Graph()` constructor.
- Test 4 (`test_rdflib_imports_succeed_at_module_load`) is a real regression guard: it asserts `RDFLIB_AVAILABLE is True` AND that the sentinels are populated. If a future refactor accidentally re-introduces the original bug by moving constants outside the `if RDFLIB_AVAILABLE:` block while rdflib happens to still be installed, the test still fires because the sentinel path would resolve `ON = None` and the assertion `rdf_owl_shacl.ON is not None` would fail. Strong.
- The compound bug ("rdflib was never declared as a dep") is real: confirmed via `git show main:packages/ontology-manager/pyproject.toml` — rdflib was not in `[project.dependencies]` on main. No other workspace package pulls it transitively (grep for `rdflib` across all `.toml` returns nothing on main). `rdf_owl_shacl.py` had zero callers in code (only docstring mentions in `semantic_intelligence/__init__.py`), so the latent NameError never surfaced at runtime — confirming the implementer's analysis that the bug was dead-code-level.
- `uv.lock` was regenerated alongside the manifest change; rdflib 7.6.0 + transitive deps are now pinned. pyshacl 0.31.0 lands behind the `dev` marker.
- pyshacl test uses `pytest.importorskip` cleanly so the skip path is deterministic when dev extras are not installed; verified locally that the test goes from `SKIPPED` to `PASSED` once `uv sync --extra dev` runs in the package.
- Set-equality is exactly what the plan asked for, and the helper rationale ("blank-node isomorphism is 100x slower, not needed here") is sound and documented in the test docstring.

## Decision rationale

All 3 plan acceptance criteria are met. Numbers reproduce. The fix correctly applies the sentinel pattern from the RETRO; all in-scope public functions are guarded; the regression guard test would catch a future re-introduction. The one minor I considered escalating (`validate_entities` not guarded) is explicitly out-of-scope for B.2a per the implementer's self-review and the plan only names the import-bug fix + the two roundtrip tests as deliverables — extending guards to all five untested functions would be scope creep.

No blockers. No majors. Approving.

## Next steps

APPROVED — ready for human approval / merge.

Suggested follow-up issue (whoever picks up B.2c or the SHACL test gap):

> "Extend `_require_rdflib()` guard to `validate_entities` + `create_brede_welvaart_skos`. Replace `_demo` hardcoded `/mnt/e/Repos/Private/open-notebook` PROJECT_ROOT with `Path(__file__).resolve().parents[3]` or similar. Promote `load_all_ontologies`'s silent per-file `except` to a structured error report so CLI users see what failed and why."
