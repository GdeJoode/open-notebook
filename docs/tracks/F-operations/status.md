# Track F — Operations & quality — Status

**State: SHIPPED (2026-07-24)** — F.1–F.7 merged (#58–#64). F.8 = this docs
close-out. Two scoped deferrals: F.7b (`chunked` split) and the e2e specs.

| Phase | Status | PR |
|---|---|---|
| F.1 — Audit engine (findings table + 6 checks + API) | ✅ | #58 |
| F.2 — Audit widget | ✅ | #59 |
| F.3 — Deep audit (conflicting facts + provenance gaps) | ✅ | #60 |
| F.4 — Deep-audit UI trigger | ✅ | #61 |
| F.5 — Librarian periodic job (opt-in) | ✅ | #62 |
| F.6 — Librarian UI (toggle + last-run) | ✅ | #63 |
| F.7 — Resumable: failure provenance | ✅ | #64 |
| F.7b — Resumable: `chunked` sub-stage split | ⏸ deferred | — |
| F.8 — Integration docs + RETRO | ✅ (this) | — |

See `RETRO.md` for the deliberate deviations (F.3 LLM-free by reuse; F.7
provenance-only) and the open follow-ups.

**Migrations consumed**: 75 (`audit_findings`). No others — `librarian_enabled`
and `failed_stage` ride existing schemaless/flexible fields.
