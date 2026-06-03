# Track Workflow

This directory holds per-track sprint plans, status, and audit-trails for the
implementation of `docs/FEATURE_ROADMAP.md`. The work is driven by a 5-agent
multi-track pipeline; see `.claude/agents/` for the agent definitions.

## Directory layout per track

```
docs/tracks/
├── README.md                       # this file
├── _status.md                      # cross-track dashboard (cross-track-monitor)
└── <id>-<slug>/                    # one folder per track
    ├── plan.md                     # sprint plan (track-planner output)
    ├── status.md                   # current phase + progress
    ├── escalations.md              # audit trail of escalations
    ├── escalations/                # per-escalation issue bodies (if gh not auth)
    └── reviews/                    # per-phase review reports
        ├── phase-X.1-attempt-1.md
        ├── phase-X.1-attempt-2.md
        └── ...
```

## Agent roles

See `.claude/agents/` for full definitions.

| Agent | Role |
|---|---|
| `track-planner` | Translates roadmap track → sprint plan with phases (Backend → UI → Integration) |
| `implementer` | Executes one phase: code + tests + docs + commit + push |
| `adversarial-reviewer` | Reviews implementation with intent to find problems; APPROVED or REVISIONS_NEEDED |
| `escalation-handler` | Surfaces blockers to user via GitHub issue + chat summary |
| `cross-track-monitor` | Detects conflicts between parallel tracks; updates `_status.md` |

## Workflow per track

```
1. User: "Plan Track <id>"
   → track-planner produces docs/tracks/<id>-<slug>/plan.md
   → User reviews plan, approves or requests changes

2. User: "Implement Track <id>" (one phase at a time, or autopilot)
   For each phase in plan:
      a. implementer creates branch, writes code + tests, pushes
      b. adversarial-reviewer reviews
         ├─ APPROVED → next phase
         └─ REVISIONS_NEEDED → implementer revises (max 3 attempts)
      c. After 3rd rejection or impasse → escalation-handler
   → All phases complete → ready for human approval to merge to main

3. After phase merge:
   → cross-track-monitor updates _status.md
   → Conflicts detected → recommend merge order or trigger escalation
```

## Quality gates

- **Backend phase**: tests pass, types check, acceptance criteria met
- **UI phase**: a11y verified, responsive on desktop, design-consistent
- **Integration phase**: E2E flow tested, docs updated

## Branches

Per-phase branches follow the pattern:
- `track/<id>-<phase>` — e.g. `track/a-mineru-backend`

Merged via PR to `main` after reviewer approval AND user sign-off (first track at least).

## Escalation channel

Escalations go to:
1. **GitHub issue** (canonical record) — created by `escalation-handler` via `gh issue create`
2. **Chat-ready summary** in the conversation where the user is working
3. **Audit trail** in `docs/tracks/<id>-<slug>/escalations.md`

Track status set to `BLOCKED_PENDING_USER` until user responds.

## Cross-track dashboard

`_status.md` is the at-a-glance view across all tracks. Auto-updated by `cross-track-monitor` after each phase merge or on user request.
