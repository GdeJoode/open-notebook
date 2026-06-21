SUPERGOAL_PHASE_START
Phase: 2 of 4 — B.8b Deploy + UI/KG verification (Integration/UI)
Task: Ship the ORDER BY fix + B.8a via image rebuild; verify entities + per-document KG render; check default filters.
Type: brownfield, deploy, ui, integration
Mandatory commands: docker compose build open_notebook, docker compose up -d open_notebook, docker ps
Acceptance criteria: 9
Evidence required: merge, build success, in-container code proof, entities endpoint non-empty, per-doc KG, filter-threshold finding, no ORDER BY error, reviewer APPROVED, ledger entry
Depends on phases: 1

## Why
The committed WITH NOINDEX KG fix and B.8a aren't in the running 18GB image (user chose a rebuild). And the default KG filters min_conf=0.9 / min_conn=5 (Q1) may hide fresh entities independent of the query bug — must be verified, not assumed.

## Work
- Branch `track/b8b-deploy-verify`. Merge `fix/kg-entity-order-by-name` + `track/b8a-model-provenance` into the working branch so both fixes are in the build context.
- `df -h /mnt/e` (need headroom), then `docker compose build open_notebook` and `docker compose up -d open_notebook`.
- Prove the running container has the new code (grep `WITH NOINDEX` in the deployed entity.py inside the container).
- Verify `GET /api/knowledge-graph/entities` and `/graph` return data with no "No iterator has been found" in logs.
- Verify the PER-DOCUMENT KG/structure view returns nodes/entities for a source that has them (Track I.F structure-graph and/or per-source entities endpoint).
- Determine whether the default min_conf=0.9 / min_conn=5 filters hide freshly-extracted entities: query entity confidence distribution vs the threshold; confirm how the UI exposes overrides. Record as a finding.
- adversarial-reviewer on the branch until APPROVED (max 3); append B.8b status.md ledger row.

## Acceptance criteria (all must pass — verify each in transcript)
- Both fixes present in the rebuilt image (in-container grep proof).
- `docker compose build open_notebook` exits 0; container healthy on new image (`docker ps`).
- `GET /api/knowledge-graph/entities` returns HTTP 200 with a non-empty body (shown).
- No "No iterator has been found" in app logs during an entities request post-rebuild.
- The per-document KG endpoint returns nodes/entities for a source that has them (body shown).
- A documented finding on whether default filters (min_conf=0.9/min_conn=5) hide entities, with the confidence distribution as evidence.
- No new errors introduced by the deploy (log check).
- adversarial-reviewer returns APPROVED.
- A B.8b row is appended to docs/tracks/B-kg-quality/status.md.

## Mandatory commands (run each, surface last ~10 lines + exit code)
- docker compose build open_notebook
- docker compose up -d open_notebook
- docker ps

## Evidence required in transcript
- Build tail + image id/date; in-container `WITH NOINDEX` grep.
- entities + per-doc KG endpoint JSON.
- Filter-threshold finding with confidence distribution.
- reviewer APPROVED + ledger row.

## Notes
If the 18GB rebuild stalls/fails, hot-patch the container to finish UI verification and record `REBUILD: needs retry` in the ledger — do not block the run. Frontend "renders" can be evidenced via the API the component calls if no headless browser is drivable.
