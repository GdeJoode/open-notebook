# ROADMAP — Track B.8: KG live-validation + provenance hardening

**Framing:** Follow-up sub-track under **Track B (KG quality — COMPLETE 2026-06-12)**. Not greenfield. Artifacts live under `docs/tracks/B-kg-quality/` per the track methodology (`docs/tracks/README.md`).

**Task:** Validate the live multi-schema extraction pipeline, fix 2 real provenance/error-surfacing bugs, ship the ORDER BY KG fix, and produce an honest cross-document entity-resolution assessment over the 12 related Regio Deal / Convenant PDFs.

**Decisions (intake):** model = `qwen2.5:14b-instruct` · scope = all 11 Convenant PDFs · deploy = rebuild image · execution = **Supergoal /goal + adversarial-reviewer gate + status.md ledger per phase** · resolution = **assess + recommend only** (no resolver change; full TOOI/Crossref = deferred Track M4).

**Corrections vs first draft (after reconciling with Track B):**
- "No entities" is OPERATIONAL (manual `run-entities` trigger / schema-review pause), not a pipeline bug.
- The multi-schema LLM orchestrator is already the live path; spaCy entities are Dec-2025 legacy (dead path).
- Resolution = documented V1 `name_normalizer` stub; full resolution is Q9/Track M4 (deferred).
- Real bugs in scope: `extraction_method` never populated; (noted-only) upsert atomicity B.1e-deferred.
- UI: ORDER BY fix (done, deploy) + default filters min_conf=0.9/min_conn=5 may hide entities.

**Stack:** Python (uv) · SurrealDB v2 · Docker Compose · local Ollama · ruff + pytest.

## Phases (each ends with adversarial-reviewer APPROVED + a status.md ledger row)

### B.8a — Model + provenance fix (Backend)
Switch EXTRACTION_MODEL→qwen2.5:14b; fix `extraction_method` propagation; surface swallowed extraction errors. Tests. Branch `track/b8a-model-provenance`. Depends: none.

### B.8b — Deploy + UI/KG verification (Integration/UI)
Merge ORDER BY fix + B.8a; rebuild image; verify entities + per-document KG render; assess default-filter (min_conf 0.9/min_conn 5) hiding. Branch `track/b8b-deploy-verify`. Depends: B.8a.

### B.8c — Live validation + 11-doc resolution assessment (Integration)
Trigger extraction on the 2 parsed docs (0→>0); ingest+extract 11 Convenant docs (qwen, sequential); measure cross-doc resolution; honest verdict + M4 recommendation. Branch `track/b8c-live-validation`. Depends: B.8a, B.8b.

### B.8d — Polish, Harden + track ledger
Lint/tests green; consolidate findings; codify schema-drift decision; record deferred items; update status.md + FEATURE_ROADMAP; memory. Branch `track/b8d-polish`. Depends: B.8a-c.

## Key assumptions (correct any that are wrong)
- Extraction runs sequentially (so B.1e upsert-atomicity isn't hit — noted not fixed).
- Resolution success = honest measurement + ceiling report (NOT achieving high resolution, which V1 caps).
- The 12 Convenant docs go in a fresh dedicated notebook.
- qwen tag is `qwen2.5:14b-instruct-q5_K_M` (B.8a reconfirms).

## Top risks → mitigations
1. Resolution capped by V1 normalizer → assess-only; report ceiling + M4 recommendation honestly (no fake pass).
2. 12× qwen14b extraction slow / Ollama OOM → sequential + per-doc checkpoint; fall back + flag.
3. 18GB rebuild slow/fails → 231 GB free; hot-patch fallback to finish verification.
4. adversarial-reviewer rejects 3× on a phase → escalation-handler per docs/tracks/README.md (GitHub issue + chat summary); pause for user.

## Escalation policy (binding for every phase)
A phase advances ONLY on adversarial-reviewer APPROVED. On a blocking finding:
- **Fixable defect** → revise + re-review, max 3 attempts.
- **Needs-a-decision blocker** (approach wrong / product call / out-of-scope e.g. pulling Track M4 forward / data-loss or irreversible op / can't satisfy a criterion without changing scope) → ESCALATE IMMEDIATELY, do not burn the 3 attempts.
- Terminal (3rd rejection OR immediate-escalate) → escalation-handler: GitHub issue + docs/tracks/B-kg-quality/escalations.md + chat summary; status BLOCKED_PENDING_USER; print FAILURE_HANDOFF; STOP (no SUPERGOAL_RUN_COMPLETE). Dependency chain means a hard B.8a/b blocker halts the whole run.
- Never self-rationalize past a blocker to keep the chain moving. Final merge to main always requires user sign-off, independent of APPROVED.
