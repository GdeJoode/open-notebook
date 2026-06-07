# Protégé import smoke test for `GET /api/notebooks/{id}/schema.ttl`

Manual verification script for Phase B.2b. Run this once per release that
touches the TTL exporter (`apps/app-main/src/app_main/api/routers/schemas.py`
or `packages/ontology-manager/src/ontology_manager/rdf_owl_shacl.py`).

Automated coverage already asserts the Turtle parses cleanly via
`rdflib.Graph().parse(format="turtle")` (see `test_schemas_router.py::TestRoundtripParse`).
Protégé is a stricter consumer (it loads the same Turtle through OWLAPI),
so this script is the canonical "does the export work in the wild" check.

## Prerequisites

- Protégé 5.6.0 or later (Mac, Linux, or Windows build from
  <https://protege.stanford.edu/products.php#desktop-protege>).
- `app-main` running locally with at least one notebook present (any
  notebook id is fine — the endpoint returns the base ontology even if
  the notebook has no `notebook_schema` row yet).
- `curl` or a browser.

## Steps

1. Pick a notebook id from the app UI (Notebooks list → copy the record
   id from the URL, e.g. `notebook:abc123`).

2. Download the schema:

   ```bash
   curl -o my-schema.ttl \
        "http://localhost:5055/api/notebooks/notebook:abc123/schema.ttl"
   ```

   Confirm:
   - HTTP 200
   - File begins with `@prefix`
   - File size > 1 KB

3. Open Protégé and **File → Open…** → select `my-schema.ttl`.

   Confirm:
   - No "Parse error" dialog appears.
   - The **Classes** tab populates with the ontology's classes nested
     under `owl:Thing`.
   - For a notebook with `accepted_extensions`, the extension class
     names appear in the tree (likely under the matching `parent_type`).

4. **Reasoner → Start reasoner** (HermiT default).

   Confirm:
   - No "Inconsistent ontology" warning.
   - Reasoner finishes within a few seconds.

5. **File → Save as…** → Turtle format → write to a new file
   (`roundtrip.ttl`). Confirm Protégé writes without errors.

## Pass criteria

All four "confirm" blocks pass.

## On failure

- "Parse error" → re-check the raw Turtle in a text editor. Common
  culprits are unescaped characters in `rdfs:comment` literals.
- "Inconsistent ontology" → an extension probably declared a class as
  both `subClassOf` and itself, or referenced a parent that doesn't
  exist in the base ontology. Capture the extension dict from
  `notebook_schema.accepted_extensions` and file an issue.
- File saves but the class tree is empty under `Thing` → the IRI
  namespace on the bundled YAML may have drifted from
  `https://open-notebook.dev/ontology/`. See `ON` constant in
  `rdf_owl_shacl.py`.

## Capture

Once a first live run succeeds, take a screenshot of the populated
class tree in Protégé and commit it next to this file as
`docs/tracks/B-kg-quality/protege-class-tree.png`. Reference it from
this script. (Skipped on first cut — no live notebook available yet.)
