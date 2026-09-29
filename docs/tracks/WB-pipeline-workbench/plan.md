# Track WB — Pipeline workbench (frontend review)

**Status: APPROVED 2026-09-29.** Owner accepted every recommended option
(WB-D1 (a) Werkbak home, WB-D2 (a) sources primary, WB-D3 (a) desktop first,
WB-D4 (a) a collapsed "More" group, WB-D5 (a) English).

Originally proposed 2026-09-29. A UI-design review of the frontend. It
answers the owner's request: "more pipeline focused … a logical way to proceed
through the steps … when I add files via API endpoints it should be clear where
I need to be when I open the app again … layout, font sizes, responsiveness
could be better."

Scope follows the product decision: open-notebook is for **extracting source
material and administering it**. Writing, research and content checks live in
Obsidian + Claude (Tracks C and E are deferred). The frontend should therefore
be a workbench for a pipeline, not a notebook/chat app with a pipeline attached.

Evidence:
- screenshots of the live app at 1440 px, taken 2026-09-29;
- a code survey of `frontend/src`;
- API timings.

Track UX (closed 2026-07-01) made `processing_stage` the spine of the source
views and shortened the creation flow. WB builds on that spine and does not
redo it.

## What the review found

### 1. There is no "where am I needed" view
- The app lands on a plain list. Two pages compete for `/`:
  - `app/page.tsx` redirects to `/notebooks`;
  - `app/(dashboard)/page.tsx` redirects to `/sources`.

  Neither says what needs attention.
- Work is waiting in several places, and none of it is visible from the nav.
  Live examples:
  - The **triage queue has 1,238 open rows**. It can only be reached through
    a button on the Knowledge Graph page, and nothing anywhere shows a count.
  - Sources that stopped at **"Embed done"** (the regio-deal convenants in
    `live-test`) look the same as finished ones. The Sources list has no stage
    column at all, even though the list API returns `processing_stage`.
  - **Pending extensions** (Pass-1 proposals such as `RegioDealAgreement`)
    are only shown on the notebook's Schema tab, one notebook at a time.
  - **Merge candidates**, **failed sources** and **paused extraction** each
    have their own page or banner, or none.
- Anything that arrives through the API or MCP (a source added, a job
  finished, an item gated for review) leaves no trace until the owner goes
  looking. Toasts (`sonner`) are transient. The only unread model,
  `notebook_events`, is scoped per notebook.
- The nav omits exactly the review pages: Triage, Resolution, notebook Schema
  and Model routing.

### 2. The pipeline does not lead to its next step
- **Create flow:**
  - "View in Background" goes to `/sources`, which cannot show the job's
    progress.
  - "Done" offers "Create another" and "View source". It never points to the
    notebook, the entities, triage or resolution.
- **Gated path dead-ends:**
  - `awaiting_schema_review` links to the source's Entities tab;
  - that tab has no link to where the proposal is actually accepted (the
    notebook's Schema tab, and the ontology editor after OE.5).
- **Review pages link back, never onward:**
  - Triage and Resolution link only back to the Knowledge Graph.
  - Ontologies links to nothing.
- **Nav groups don't match the pipeline.** The groups are Collect / Process /
  Analyze / Create / Configure:
  - Notebooks and "Ask and Search" sit under *Process*;
  - Podcasts, Summaries and Transformations sit at the same level as the
    pipeline;
  - Resolution is reached two ways: a tab inside `/knowledge-graph` and a
    separate page.

### 3. The data on screen is hard to use
- Source titles carry an upload hash prefix
  (`410a65bba3f8__Convenant Regio Deal…`). They look like near-duplicates and
  bury the real name.
- **Knowledge Graph table:**
  - it opens alphabetically on noise (`## Introduction`, quoted fragments,
    mostly *Archived* rows);
  - the default status filter shows everything.
- The triage table shows display types (`Organization`, `Journal/ Publication`)
  while the KG table shows canonical types (`organization`, `concept`).
- The notebook Schema tab says "No entity types in the effective schema yet",
  which reflects the per-notebook layer OE.5 retires.

### 4. Speed and failure states
- Pages render nothing (a blank background) until their data arrives. Measured
  timings:
  - `/api/notebooks`: 76 s, then more than 60 s;
  - `/api/entity-resolution/candidates`: 42 s;
  - one source: 17 s.

  While the database is slow the app flips to a full-screen **"Database
  Connection Failed"** page.
- **Cause** (outside the frontend, see "Prerequisite"): signing in to SurrealDB
  takes 10–14 s per new connection, while the query itself takes about 50 µs.
  The API's 2 s health check therefore reports the database "offline".

### 5. Layout, type, responsiveness
- **Type is small.** In `.tsx` files:
  - `text-xs` 385 and `text-sm` 423 uses, against `text-base` 19;
  - plus 38 `text-[10px]`.

  Body copy, table cells and badges are 12–14 px. Labels in the pipeline spine
  and triage reasons are 10–11 px.
- **No layout system.** Pages use `max-w-3xl`, `max-w-4xl`, `max-w-5xl`,
  `max-w-6xl`, or no max width. Some are centred and most are left-aligned.
  Padding varies (`p-6`, `px-6 py-6`, `pt-6 pb-4 px-6`). Page headers differ
  too:
  - "All Sources" in `text-3xl`;
  - Podcasts in `font-semibold tracking-tight`;
  - everything else in `text-2xl font-bold`.
- **The source detail page has no sidebar** (it doesn't use `AppShell`), so the
  owner loses the navigation exactly where the pipeline is shown.
- **Not responsive:**
  - the sidebar is a fixed `w-64` with no drawer;
  - the Sources table has `min-w-[800px]`;
  - the KG detail panel is a fixed `w-96`;
  - dialogs are 480–1080 px wide;
  - the codebase has 60 `sm:`, 33 `md:`, 12 `lg:` and 1 `xl:` prefixes.
- **The notebook page stacks three equal columns** (Sources / Notes / Chat).
  Chat and Notes take two thirds of the width, although content work lives
  outside the app.
- **The header is overloaded:** the notebook header holds six buttons (Export
  Obsidian, Export OKF, Import OKF, Export JSONL, Archive, Delete) above a
  privacy radio group.
- **Dead or unused config:**
  - `tailwind.config.ts` is v3-style and unused under Tailwind v4;
  - `/advanced` is a dead redirect.

## Decisions for the owner

- **WB-D1 — Landing page.**
  - **(a) A "Werkbak / Inbox" home that lists what needs you (recommended).**
  - (b) Keep landing on Notebooks, with badges in the nav.
- **WB-D2 — Primary unit.**
  - **(a) Sources and the pipeline are primary; notebooks become a filter or
    collection (recommended, fits "extracting + administering").**
  - (b) Notebooks stay the main workspace.
- **WB-D3 — Screen sizes.**
  - **(a) Desktop and laptop first (≥1024 px), usable on a tablet (≥768 px),
    read-only on a phone (recommended).**
  - (b) Full mobile.
- **WB-D4 — Features outside the product scope** (Podcasts, Transformations,
  Summaries strategies, the notebook Chat/Notes columns).
  - **(a) Move them to a "More" group, collapsed; nothing is deleted
    (recommended).**
  - (b) Remove them from the nav.
  - (c) Leave as is.
- **WB-D5 — UI language.**
  - **(a) Stay English (recommended; the code, the docs and the ontology UI
    are English).**
  - (b) Dutch.

## Phases (assuming the recommended options)

**WB.0 — Prerequisite: make the database answer fast** (backend, ~0.5d)
- Find why SurrealDB sign-in takes 10–14 s: auth hashing, the root user
  definition, or the RocksDB store.
- Make the API reuse pooled, already signed-in connections for health checks
  and requests.
- **AC:**
  - `/api/notebooks`, `/api/sources/` and the triage queue answer in under
    1 s on the live data;
  - the "Database Connection Failed" page no longer appears while the
    database is up.
- **WB.0 built** (branch `feature/wb0-db-responsiveness`):
  - The hang was the pool, not SurrealDB. A failed or cancelled sign-in
    leaked a slot, and the 2 s health check cancels slow sign-ins; after
    ten leaks every request waited forever. Slots are now semaphore
    permits handed back on every failure path. Sign-in, the reuse ping and
    the wait for a slot are bounded, and the pool raises
    `DatabaseUnavailableError` (a 503 handler exists). A connection whose
    request was cancelled or hit a transport error is closed.
  - The 503 reaches the client only where no `except Exception` turns it
    into a 500. That holds for `/api/notebooks`, `/api/sources/` and
    `/api/triage/queue` (tested); most other routers still answer 500.
  - Health check: the probe outlives the 2 s check and is shared, so a
    slow sign-in finishes into the pool. A timeout reports online unless
    the last finished probe failed.
  - Trade-off: a database that accepts connections but never answers
    reads "online" until a probe fails, i.e. up to the pool's 30 s
    checkout deadline plus the check interval (about 30–35 s). A refused
    connection reads offline at once.
  - Slow sign-in not reproduced: root sign-in now takes 0.02 s (argon2id,
    m=19 MiB, t=2). SurrealDB's `/health` was 250× slower in the slow
    phase too, so the whole process was starved by host load, not auth.
  - `/api/entity-resolution/candidates`: 16 s → 0.7 s on a snapshot of
    the live graph, same report. It was pure-Python Levenshtein over 94k
    pairs, on the event loop.
  - Live: `/api/notebooks` 0.03 s, `/api/sources/` 0.11 s, triage queue
    0.23–0.35 s.
- **Review round 1** (REVISIONS_NEEDED, 1 major, 8 minors), all fixed:
  - M1: a checkout cancelled during the reuse ping left the candidate
    open and off the queue. It is now closed.
  - m2: a `CancelledError` from the SDK (server closed the socket) is told
    apart from our own cancellation with `Task.cancelling()`. During the
    ping the candidate counts as dead; inside a request it becomes
    `DatabaseUnavailableError`.
  - m3: a probe that ended cancelled counts as a failure; `/api/config`
    answers offline instead of raising.
  - m4: `/api/sources/` re-raises `DatabaseUnavailableError`; the note
    above no longer claims a general 503.
  - m5: `acquire_timeout` now bounds the whole checkout (slot, pings,
    sign-in).
  - m6: `close_all` bounds each close; connections returned afterwards
    are closed.
  - m7: the online-while-down window is recorded above.
  - m8: candidate scoring runs in `asyncio.to_thread`.
  - Live timings are verified after merge and rebuild.

> **Carried from WB.0 review round 2 (approved; minors, not blocking):**
> 1. An SDK-caused `CancelledError` during `signin`/`use` (socket closed by a
>    restarting server) still escapes `_create_connection` as a raw cancel;
>    the m2 handling covers the ping and the body only.
> 2. `/api/config` decides by `probe.cancelled()` alone: a request cancelled
>    in the same tick as a cancelled shared probe answers "offline" and
>    swallows its own cancellation.
> 3. The SDK's `close()` swallows a cancel while awaiting its receive task, so
>    a deadline firing during `_close_quietly` can be missed (bounded by the
>    next wait). Pre-existing.
> - The `to_thread` test's last assert checks nothing; the thread-id assert is
>   the guard.

**WB.1 — The Werkbak (attention inbox)** (~2d)
- **Backend:** one `GET /api/attention` that counts and lists, across all
  notebooks:
  - sources by stage that are stuck: failed, `awaiting_schema_review`, and
    stopped short of `complete` for longer than N minutes;
  - paused extraction;
  - pending Pass-1 proposals;
  - open triage rows;
  - merge candidates;
  - retype pending (OE.4 dry run not empty);
  - unfinished jobs.

  Plus "arrived since your last visit": sources created by the API or MCP, and
  jobs finished. Each item carries the link to where it is resolved.
- `?stage=` filter on `GET /sources/`.
- **Frontend:** `/` becomes the Werkbak, with the same groups as cards, each
  showing a count and one primary action ("Review 12 proposals →").
  - The nav shows the same counts as badges.
  - "Last visit" is stored per browser.
- **AC:**
  - a source added through the API, and a job that fails while the app is
    closed, both appear on the Werkbak on the next open, each with a link
    that lands on the page where it is resolved;
  - every count equals its source endpoint (a test compares them).
- **WB.1 built** (branch `feature/wb1-werkbak`):
  - `GET /api/attention` (`AttentionService`) returns ten groups: failed,
    gated and stopped sources, paused extraction, Pass-1 proposals, triage,
    merge candidates, retype, failed jobs, jobs in progress. With `?since=`
    it adds new sources and finished jobs. Each group reads through its
    page's own service. Each group and item carries the route where it is
    resolved, and a `nav` for the sidebar badge.
  - Groups run concurrently with a 5 s deadline each. A failing or slow
    group returns `error`; the others still answer.
  - "Stopped" means: short of `complete`, no queued or running job for the
    source, and the last write more than 15 minutes old. With the job check
    in place, the only normal quiet gap is between a stage write and the
    next enqueue.
  - A failed job counts until a later job of the same type for the same
    source exists, or until its source is deleted (live: 3 of 22). Exposed as
    `GET /commands/jobs?unresolved=true`.
  - Retype counts changes + flag-only, as the retype panel does; collisions
    show as detail.
  - `?stage=` (repeatable) on `GET /sources/`; the Sources page reads it.
  - `/` is the Werkbak; `app/page.tsx` and the middleware redirect are gone.
    The sidebar has a Werkbak entry with the total, and badges on Sources,
    Notebooks, Knowledge Graph and Ontologies. "Last visit" is the
    `generated_at` of the report marked as seen (localStorage). It moves
    only on "Mark as seen".
  - Live, read-only: warm 1.15–1.2 s, bounded by merge candidates (1.05 s
    on their own at the time). Every other group takes 0.06–0.56 s. A cold
    process needs 5.8 s for the pool sign-ins. Live counts: 8 stopped
    sources, 79 proposals, 1,238 triage rows, 48 merge candidates, 3 failed
    jobs.
  - Limits:
    - the < 1 s target is missed by the merge-candidates read;
    - `JobRepository.list_jobs` and `NotebookSchemaRepository.get_by_notebook`
      swallow read errors, so those groups read 0 instead of reporting an
      error;
    - no jobs page exists; a job without a source or notebook links to
      `/podcasts`, `/settings` or `/notebooks` by type;
    - `design-tokens.spec.ts` fails without a backend: 32 console errors
      from endpoints it does not mock. `/api/attention` is now mocked, so
      none of them come from WB.1.
- **Review round 1** (REVISIONS_NEEDED, 2 majors, 8 minors), all addressed:
  - M1: "stopped" now has one predicate (`services/stopped_sources.py`).
    Both the group and `GET /sources/?stalled=true` use it, and the group
    links there. The Sources page shows "Showing only: stopped sources".
    Tests use the real 15 min threshold: they backdate `updated` and leave
    out a busy source and a freshly written one.
  - M2: the paused-jobs read moved to `services/paused_extraction.py`; the
    router and the Werkbak both call it.
  - m3: the new `counted` field goes into `total` and the badges, so each
    thing to act on counts once. A source listed in a source group owns its
    paused extraction and its failed jobs. Proposals count in full. A group
    that fails owns nothing.
  - m4: the mutations that resolve an item invalidate `['attention']`.
  - m9: the sidebar reads the cached report with no refetch on mount or
    focus; the Werkbak page always reads anew.
  - m5: without a mark, arrivals cover the last 7 days and are labelled so
    (`since_default`).
  - m6: gated sources, paused extraction and proposals show one item per
    notebook, each linked to that notebook's page, with no cap. Failed jobs
    say plainly that there is no jobs page.
  - m7: a stopped source without chunks reads "No text extracted:
    reprocess or delete" and links to its page, where Reprocess is.
  - m8: merge candidates include the Resolution page's plan clusters.
  - m10: a failure is keyed by type, source and the payload's
    `notebook_id`. The later jobs are read one row per key, for the
    failures' keys only.
  - Live, read-only after the round: 1.19–1.47 s, still bounded by merge
    candidates (1.18 s). Total 1,375: the 3 failed jobs count 2, because one
    belongs to a listed source. 8 stopped sources, all "Stopped at
    embedded".
  - Limits:
    - the < 1 s target is still missed;
    - the scoring runs in `asyncio.to_thread`, so a merge-candidates timeout
      does not stop it: the thread runs to the end;
    - `list_jobs` and `get_by_notebook` still swallow read errors.

**WB.2 — Pipeline navigation** (~1.5d)
- **Nav regrouped in pipeline order:**
  1. Werkbak
  2. Sources (stage filter)
  3. Review:
     - Schema proposals
     - Triage
     - Duplicates
     - Retype
  4. Knowledge
     - Graph
     - Ontologies
     - Search
  5. Collections (Notebooks)
  6. Settings (Models, Model routing, Settings)
  7. More (WB-D4)
- One Resolution entry (the separate page), with the tab inside `/knowledge-graph`
  removed or linked to it.
- **Every review page ends with "next":**
  - Triage empty → Duplicates;
  - Duplicates empty → Werkbak;
  - Create flow Done → the source's next stage, or the Werkbak if it is gated;
  - `awaiting_schema_review` → the proposal where it is accepted (the notebook
    Schema tab until OE.5, then the ontology editor).
- The source detail page uses `AppShell`.
- **AC:**
  - from the Werkbak, each group reaches its resolution page in one click and
    returns in one click;
  - an e2e test walks upload → gated → accept → complete without typing a URL.

**WB.3 — Sources as the pipeline board** (~1.5d)
- **Sources list:**
  - a stage column (the `PipelineStatus` card variant);
  - stage filter chips with counts;
  - "Add source" in the header;
  - the notebook as a column or filter;
  - sorted by "needs you" first.
- **Titles:** strip the upload hash prefix for display, keeping the file name
  in Details. Optionally a backfill of the `title` field, behind a dry run.
- **Knowledge Graph table:**
  - defaults to Active entities and sort by degree;
  - uses canonical types everywhere, with display labels only as secondary
    text.
- **AC:**
  - a source stopped at `embedded` is visibly different from a `complete` one,
    and is filterable;
  - no displayed title starts with a 12-hex prefix.

**WB.4 — Layout system and typography** (~2d)
- **One `PageHeader`:**
  - title, optional subtitle, and actions on the right;
  - overflow actions go into a menu (for example, the notebook's six export
    buttons become one "Export ▾");
  - one `PageBody` width scale: `max-w-7xl` for tables and boards,
    `max-w-3xl` for forms.
- **Type scale:**
  - base 15–16 px for body and table cells;
  - 13 px as the minimum for metadata;
  - no `text-[10px]`;
  - a lint rule (eslint `no-restricted-syntax` on `text-[`) keeps it.
- Remove the unused `tailwind.config.ts` and the `/advanced` page after
  checking nothing imports them (the settings tab reuses its components, so
  move those first).
- **AC:**
  - every dashboard page uses `PageHeader` / `PageBody`;
  - a visual snapshot per page at 1440 and 1024 px;
  - no `text-[10px]` left in `src/`.

**WB.5 — Responsive** (~1.5d, per WB-D3)
- **Sidebar:**
  - collapses to icons below `xl`;
  - becomes a drawer below `md`.
- Tables become cards below `md` (Sources, Triage, the KG table).
- The KG detail panel becomes a sheet below `lg`.
- Dialogs use `max-w-[min(…, 100vw-2rem)]`.
- The notebook page becomes Sources first, with Notes and Chat as tabs.
- **AC:**
  - no horizontal page scroll at 1024, 768 and 390 px on the Werkbak, Sources,
    source detail, triage and the notebook page (checked by Playwright:
    `scrollWidth <= innerWidth`);
  - tap targets are at least 40 px at 390 px.

Rough total: ~9 days. WB.0 comes first, because a slow database makes every
page look broken. WB.1 and WB.2 deliver most of what the owner asked for.
WB.4 and WB.5 are polish that can run alongside the other phases.

## Rules carried over
- A producer names its consumer: every Werkbak item links to where it is
  resolved.
- Derive, don't sample: counts come from the endpoints, never recomputed in
  the UI.
- A test double reproduces the real contract: e2e mocks follow the real
  `/api/attention`.
- Nothing writes to the live graph without a dry run the owner has seen (the
  title backfill in WB.3).
