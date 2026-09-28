# Track OE — Ontology editor

**Status: IN PROGRESS.** Owner, 2026-09-28: "follow the recommendations,
start with OE.0". Every decision below is taken as its recommended option:
OE-D1 (a) DB versions, OE-D2 (a) retire the per-notebook layer, OE-D3 (a)
retype only after a confirmed dry run, OE-D4 entity + relationship types in
v1, OE-D5 (a) the extraction service goes through the registry or stops
reading ontologies.

> **OE.0 built 2026-09-28** (branch `feature/oe0-lossless-storage`, review
> round 2).
> - Every ontology model keeps the keys it has no field for (`extra='allow'`).
> - `merge_with` combines those keys the way their one reader
>   (`services/extraction`) does: lists joined parent-first without
>   duplicates, dicts merged child-first.
> - `Ontology.to_storage_dict` / `to_yaml_string` write every field that is SET
>   or differs from its default, plus the extra keys. That covers an in-place
>   edit of a collection the YAML never set (a type added to `regiodeal`), and
>   writes no defaults the YAML never said. `to_yaml` uses the same function.
> - `register` stores a NEW immutable version (unique `(name, version)`;
>   re-storing is refused and changes nothing) in one transaction via
>   `execute_transaction`.
> - `activate_version` refuses an unknown version before touching the active
>   one.
> - Both drop the whole cache, so ontologies that `extends` the changed one
>   follow.
> - The cache key names whether inheritance was resolved.
> - Migration 86 defines `ontology_version` exactly as production already had
>   it (a table no repo migration created; same field and index names), plus
>   `note` / `created_by`.
> - Known: a collection written as a YAML list is stored as a mapping (same
>   content). The OE.2 export back to YAML will have to choose its form.
> - Tests: `test_ontology_storage_lossless.py`.
>   - Every shipped YAML stores and reloads identically, through the
>     serialisation `register` uses.
>   - Against SurrealDB, every shipped YAML goes through `register` and
>     `get(version=…)`.
>   - In-place edits, re-storing refused, `activate_version` reaching
>     `regiodeal`, and an unknown version refused.
>
> **Carried into OE.2 (review minors, not blocking):**
> 1. Extra lists from `includes` come out in a different order than the
>    extraction service produces; the set is the same.
> 2. Two concurrent `register(activate=True)` calls on a name with no stored
>    rows can leave two active versions. OE.2's publish must serialise.
> 3. `updated_at` is set on create only. `activate_version` should set it.
> 4. The down migration leaves stored `note` / `created_by` values in place.
> 5. `merge_with` passes the parent's extra-key objects into the merged
>    ontology by reference. OE.2 must deep-copy before editing a merged
>    ontology in place.


## Why

open-notebook extracts source material and administers it (owner, 2026-09-28).
The ontologies decide how every extracted entity is typed, and there is one set
of them, **shared by every notebook** (owner, 2026-09-28). Today they can only
be changed by editing YAML in the repo. PC.4 showed what that costs:
- 15 types were quietly wrong (`Person` → `concept`, `Legislation` →
  `creative_work`) because a `schema_org:` URL is ignored by the loader;
- `Project` sat under the wrong parent;
- only a test caught it.

The editor must make changing the vocabulary safe: validated before it takes
effect, previewed against the real graph, undoable, and followed through on the
rows already extracted.

## What exists (inventory 2026-09-28)

| Piece | State |
|---|---|
| Read API `GET /api/ontologies[/{name}[/entity-types\|/relationship-types]]` + read-only page `/ontologies` | Works. `schema_org_type` is not returned. |
| Registry: DB-first (`ontology_version`), then YAML; `register` / `activate_version` / `list_versions` | Written, never called. **No migration defines `ontology_version`** and there is no unique index, so its `ON DUPLICATE KEY` has nothing to key on. |
| Registry cache | Per process, never expires. `activate_version` clears only `name:*`, so ontologies that `extends` it keep the old merged parent. Worker and API share the process, so one clear covers both. |
| Ontology model | Every model has `extra='ignore'`. Keys used by the YAML but lost on a save: `schema_org:` (125×), `dcterms:` (33×), `graph_relevance` (43×), property `required` / `validation_rules` / `enum_values` (37–43×), `metadata.namespaces`, `extraction_rules`, `abstract`. `register()` round-trips through the model, so saving an ontology today would delete all of these. |
| `services/extraction` (separate container) | Reads the YAML files itself, with its own cache, including keys the model ignores (`graph_relevance`, `extraction_rules`). An edit in the DB would never reach it. |
| Per-notebook edit layer (Schema tab) | See "The per-notebook layer" below. |
| Guards | `test_typing_is_stable.py` (no label with two answers; a base-named type reaches its base; the type is the same under every selection; literals stay out). `would_cycle`, `known_schema_org_base` and orphan detection via `resolve_ontology_type`. The L.6 orphan audit was planned and never built. |
| Data follow-through | `scripts/pc4b_retype_entities.py`: dry run, guarded apply, rollback, refuses identity collisions. |

### What an edit changes (consumers)

| Edited field | Changes |
|---|---|
| `parent_type`, `schema_org_type`, `aliases` | Typing (`canonical_bridge`), and so the `entity_type` of NEW rows. Old rows keep theirs, so the identity `(name_key, entity_type)` would split. |
| type name, `description` | The Pass-1 / Pass-2 prompts (descriptions cut at 160 characters) and the detection score. |
| `extraction_hints`, concepts (`indicators`, `related_concepts`) | The detection score only. This changes which ontologies a document selects. |
| relationship types (domain/range/description) | The Pass-2 prompt. |
| properties | `OntologyValidator`, and the TTL export. |

The TTL export reads the base YAML directly (no registry, no `extends`), looks
for a key `relation_types` that no YAML uses (so relationships are never
exported), and iterates `entity_types` as a list while `general.yaml` uses a
dict.

## The per-notebook layer

The Schema tab stores edits per notebook in `notebook_schema`. That contradicts
"one shared ontology". It also has defects of its own:
- `rename`/`merge`/`split` are recorded but never applied to typing. Worse,
  they leak into the Pass-2 prompt as "accepted extension types" under their
  new name.
- `split.criterion` / `split.into` are never read.
- `excluded_types` ("delete type") is not used by extraction: a deleted type is
  still extracted.
- No op retypes existing rows.

Only two ops take real effect: accepted extensions (new types from Pass 1) and
re-parents. **Decision OE-D2** settles what happens to the layer.

## Decisions for the owner

- **OE-D1 — Where the truth lives.**
  - *(a, recommended)* **DB versions.** The editor writes a new
    `ontology_version`; the YAML files are the seed and a one-click export back
    to YAML lets the owner commit it to git. Changes take effect without a
    rebuild, and every version stays restorable.
  - *(b)* **YAML in the repo.** The editor writes the files, which needs a
    writable bind mount, and git is the history. Simpler, but it only works
    where the repo is mounted, and there is no undo in the app.
- **OE-D2 — The per-notebook layer.**
  - *(a, recommended)* **Retire it into the shared ontology.**
    - Pass-1 proposals ("new type found") stay per notebook as a queue.
    - Accepting one adds the type to the SHARED ontology through the editor
      (with its validation).
    - Rename/merge/split/re-parent/delete become editor operations.
    - The existing notebook edits (0 on the live DB today) are migrated or
      dropped.
  - *(b)* Keep both layers. Not recommended: two places where a type can be
    defined is how PC.4's "type depends on context" came about.
- **OE-D3 — After a publish that changes types:** the retype (dry run → apply)
  *(a, recommended)* runs only after an explicit confirmation showing the dry
  run, or *(b)* runs automatically.
- **OE-D4 — What is editable in v1.**
  - *Recommended:* entity types (name, description, parent, schema.org base,
    aliases, hints, examples) and relationship types.
  - Properties and concepts later: they only reach validation and detection.
- **OE-D5 — The separate extraction service.** It reads the YAML itself.
  - *(a, recommended)* Point it at the registry, or retire its ontology reading
    (overlaps PC.7, which reviews that service anyway).
  - *(b)* Accept that it does not see edits.

## Phases (assuming the recommended options)

**OE.0 — Make an ontology storable without loss** (~1d)
- The model keeps what the YAML says: the ignored keys become fields
  (`schema_org` url, `graph_relevance`, property `required` / `enum_values` /
  `validation_rules`, `namespaces`, `extraction_rules`, `abstract`, `dcterms`),
  or unknown keys are carried through verbatim.
- A migration defines `ontology_version` (name, version, yaml_content,
  is_active, created_at, created_by, note) with a unique index on
  `(name, version)` and at most one active version per name.
- The registry cache is invalidated as a whole, including ontologies that
  `extends` the changed one.
- **AC:**
  - every shipped YAML round-trips (load → register → load) to an identical
    model AND an identical key set; a test compares it against the raw YAML;
  - activating a version of `schema_core` changes what `regiodeal` resolves.

**OE.1 — Validation gate and impact preview** (~1.5d)

> **OE.1 built 2026-09-28** (branch `feature/oe1-validation-preview`, review
> round 3).
> - **Structural rules** (`ontology_manager.validation`), decided from the
>   content alone:
>   - `base_named_type_points_elsewhere` (a base-named type sent to a
>     DIFFERENT canonical) and `unmapped_schema_org_type` (catches a typo like
>     `schema:Persn`);
>   - `parent_cycle`, `inheritance_cycle` (found once, from the graph),
>     `unknown_parent_ontology`;
>   - `alias_collision` (over names, dict keys and aliases),
>     `literal_declared_as_type`, `relationship_endpoint_unknown`;
>   - `unknown_key` at every level (warn only).
>
>   Inherited problems are reported once, where they are defined; `includes`
>   merges only with `extends`, as in the registry.
> - **Typing rules** (`app_main.services.ontology_validation_service`),
>   modelled on production typing: `_resolve_entity_type` over
>   `typing_vocabulary`. A label is typed under every ordered selection (up to
>   three, as detection) of one representative per group of ontologies that
>   define the names on its chain differently. That is the exact set of
>   outcomes a selection can produce, and matches a brute-force run over all
>   1,111 orders for every shipped label. One leader at a time was not enough:
>   a label and its parent each defined twice type differently under a PAIR.
>   The first version modelled its own idea of typing, missed the PC.4 bug
>   class, and blocked 553 of 935 legitimate re-parents. Rules:
>   - `selection_dependent_type`, `base_named_type_misses_its_base`;
>   - `orphan` (a chain that ends on no base at all);
>   - `subtype_disagrees_with_parent`: a child that types differently from
>     the type its `parent_type` names. It catches `parent_type: "grant"`
>     (the bridge's base lookup is case-sensitive) and children of `Product`
>     (they fall to `concept` while `Product` types `product`).
> - `classify`: a finding the active set already has, at the same or a higher
>   `weight`, warns; a new or worse one blocks.
>   - The shipped set is clean against itself.
>   - Its 5 standalone findings (2 deliberate unmapped bases, 3 relationship
>     endpoints naming no type) are pinned.
>   - A test over the whole re-parent space (935 edits) checks every block,
>     and every miss, against production typing brute-forced independently.
> - `retype_planner` holds the retype plan, lifted out of the PC.4b script and
>   shared with it.
>   - It is conservative: every row's current identity stays held for the
>     whole plan, so no two rows can end up with one identity, and the plan can
>     be written in any order.
>   - A chain (a row wanting what another row vacates) completes on the next
>     run.
>   - Releasing identities within one plan (review round 1 m11) was tried and
>     reverted: a refused row kept its identity while another was moved onto
>     it.
> - `ontology_preview_service.preview(candidate, active)` returns:
>   - findings;
>   - label changes, with entity counts;
>   - the retype a publish would offer (changes, collisions, flag-only, and
>     how many changes and collisions this edit causes);
>   - notebook edits, which it does not project;
>   - the documents whose selection changes: the detection replayed per
>     notebook the source belongs to, with that notebook's default. The
>     history fallback is not replayed.
>
>   The count matches the script's own dry run through the script's own
>   vocabulary path. Read-only.
> - **For OE.2:**
>   - The preview is synchronous: about 4–6 s on 14 sources, most of it the
>     detection replay. On a larger corpus the replay should become a
>     background job.
>   - The replay also reads every chunk into memory at once; read only the
>     sampled windows instead.


**OE.2 — Write API** (~1.5d)

> **OE.2 built 2026-09-28** (branch `feature/oe2-write-api`, review round 2).
> - `OntologyEditService` plus routes under `/api/ontologies/{name}/…`:
>   - the draft: `GET` / `DELETE draft`;
>   - edits: `PUT` / `DELETE draft/entity-types/{t}`,
>     `POST draft/entity-types/{t}/rename`, `PUT` / `DELETE
>     draft/relationship-types/{r}`, `POST draft/relationship-types/{r}/rename`;
>   - `POST draft/validate`, `POST draft/preview`, `POST draft/publish`;
>   - `GET versions`, `POST rollback/{version}`, `GET export.yaml`.
> - One draft per ontology (migration 87 `ontology_draft`), holding its own
>   content and `base_version`.
>   - A `PUT` replaces the type WHOLE: the editor sends the stored type back
>     with its changes.
>   - A rename keeps the old name as an alias and moves this ontology's own
>     `parent_type` and relationship references. Other ontologies' references
>     resolve through the alias.
> - Serialisation: edits, discard, publish and rollback share one lock per
>   event loop. An edit made during a publish lands in a new draft on the new
>   version instead of being lost, and a draft is published exactly once.
> - Publish:
>   - refuses a stale base (409, someone published meanwhile) and a blocking
>     finding (422);
>   - stores the YAML seed whenever no stored version is active, even if
>     inactive ones exist;
>   - stores the draft as `{base}+{UTC timestamp with microseconds}` and
>     returns the warnings plus the retype counts (including collisions this
>     edit causes);
>   - a failed store is a 500 and an existing version a 409; a publish that
>     took effect but could not remove the draft says so as a warning.
> - Registry: every `clear_cache` bumps a generation, and a `get` that was
>   under way during a publish does not write the pre-publish ontology back
>   into the cache. A race test fails without it. Rollback goes through the
>   same gate, and `activate_version` stamps `updated_at`.
> - Tests: `test_ontology_edit_service.py` (19), against a real SurrealDB
>   through the database-first read:
>   - with a warm cache, every declared label types as the candidate says
>     after a publish, and exactly as before after a rollback;
>   - many edits to one draft; an edit during a publish; a read in flight
>     during a publish;
>   - the HTTP surface: every route, 404 / 409 / 422.
>
>   Each guard was mutation-checked: removing it fails its test.
> - **Limits, carried:**
>   - The lock and the base check are in-process: one uvicorn process, as
>     now. With several processes the last publish wins; it needs a
>     compare-and-set inside the `register` transaction.
>   - `created_by` is always `ontology-editor` (no authentication; PC.7).
>   - Readers that bypass the registry do not see a publish: the Schema tab
>     tree and TTL export (`schemas.py` read the YAML), and
>     `services/extraction` (OE-D5 / OE.6).
>   - After an ontology's first publish, later YAML edits in the repo no
>     longer reach it; a new seed has to be published.
>   - `export.yaml` writes list-form collections as mappings (same content).
>   - The `includes` order of extra keys (OE.0 minor 1); the preview is
>     synchronous (OE.1 note).
>   - From OE.2 review round 2:
>     - The seed check reads the registry's private `_load_from_db`, which
>       hides DB errors. After a transient error an active stored version can
>       be stored again, labelled as a seed.
>     - A relationship rename keeps no alias.
>     - Endpoints kept under the extra keys `source_types` / `target_types` are
>       not validated.


**OE.3 — Editor UI** (~2d)

> **OE.3 built 2026-09-28** (branch `feature/oe3-editor-ui`, review round 3).
> The owner added drag-and-drop re-parenting to the scope, and Ctrl+X / Ctrl+V
> as the keyboard way to do it ("no need to make it very complicated").
> - The `/ontologies` detail page gets an **Edit** tab (`OntologyEditor`):
>   - a draft bar showing the draft's base version next to the ACTIVE
>     version, with Preview, Publish with a note, and Discard (confirmed);
>   - a stale draft (after a rollback, or a publish elsewhere) is warned
>     about and cannot be published;
>   - the gate's findings, with a blocking count on the type concerned in
>     the tree;
>   - the type tree (`TypeTree`) next to a form for the selected type
>     (`TypeForm`; delete confirmed);
>   - relationship types (`RelationshipTypesEditor`);
>   - the preview, dropped as soon as the draft changes;
>   - versions with a confirmed rollback that warns an open draft will go
>     stale; export to YAML.
> - **The tree** shows own types and the types inherited through the whole
>   chain (`extends` and every `includes`, at every level). Each is labelled
>   with the ontology that DEFINES it: the one that introduces or redefines
>   it, not the parent it arrives through (`inheritedTypes`; the browse API's
>   detail now includes `includes`).
>   - Parents are resolved by name or alias, case-insensitively, as in the
>     backend.
>   - Parents defined nowhere show as "not defined" targets.
>   - A parent cycle is broken for display at a type ON the cycle, marked
>     "parent cycle", so every type stays reachable and repairable. A type
>     below the cycle keeps its parent.
>   - Until the inherited ontologies have loaded, a notice replaces the tree.
> - **Re-parenting**, three ways, all ending in the same draft edit
>   (`dropEdit`):
>   - drag onto the new parent, or onto "No parent";
>   - Ctrl+X, then Ctrl+V (Cmd on a Mac; Escape cancels; a refused move says
>     why);
>   - the drag handle's own keyboard mode.
>
>   Only OWN types move. No drop that would make a cycle or change nothing is
>   offered.
> - **Nothing is silently lost:**
>   - every edit sends the stored type back whole;
>   - relationships keep the endpoint spelling they use (`source_types` /
>     `target_types`, about 80 shipped ones), which the first version wiped;
>     every spelling present is updated;
>   - the type form keeps only the fields the user edited and saves them onto
>     the LATEST stored type, so a drag or paste of the selected type is not
>     undone by the next save. It starts over only on a discard, publish or
>     rollback, never on the edit that creates the draft;
>   - "Add" refuses a name that exists, which the first version replaced;
>   - rows and the form follow the draft without discarding what is being
>     typed.
> - Tests:
>   - `ontology-tree.test.ts` (18): tree incl. cycles, `includes` and
>     aliases; drops; name clashes; both endpoint spellings;
>   - `OntologyEditor.test.tsx` (10): the review cases, each in its state;
>   - e2e `ontology-editor.spec.ts` (8), against a stateful mock that follows
>     the OE.2 contract (source_types spelling, cycle findings, 409 / 422, a
>     rollback leaving the draft stale, a rename appending the alias and
>     moving references): drag, keyboard drag, cut/paste, "No parent" plus
>     Escape, rename → preview → publish → rollback, relationship spelling,
>     parent cycle, stale draft after a rollback.
>
> **Carried from OE.3 review round 3 (not blocking):**
> - A refused save (e.g. a 422) clears what was typed, because the form clears
>   its edited fields before the server answers.
> - A redefinition that changes only a key the browse API does not return
>   (e.g. `graph_relevance`) is labelled with the deeper ontology.


**OE.4 — Data follow-through** (~1d)
- After a publish that changes types, the preview's retype plan is offered:
  dry run shown, apply on confirmation (OE-D3). Collisions are listed for the
  curator's merge (PC.5b's `/apply`).
- The retype becomes a service used by the API. The script remains as a thin
  CLI over it.
- **AC:** after publish + apply, the retype dry run is empty and no new
  identity split exists.

**OE.5 — Retire the per-notebook layer** (~1.5d, OE-D2)
- Pass-1 proposals stay a per-notebook queue. "Accept" publishes the type to
  the shared ontology through OE.2 (validated, previewed).
- The Schema tab's rename/merge/split/re-parent/delete route to the editor.
- `_project_notebook_edits` / `_typing_schemas_for` fall away. Typing is the
  registry's active set, nothing per notebook.
- The leaks are closed: nothing of the old layer reaches the Pass-2 prompt.
- **AC:** the same label types identically in every notebook (extends PC.4's
  test from "every selection" to "every notebook").

**OE.6 — Exports and the extraction service** (~0.5–1d, OE-D5)
- The TTL export goes through the registry (with `extends`) and exports
  relationship types.
- `services/extraction` reads through the registry, or stops reading
  ontologies.

Rough total ~9–10 days. OE.0 and OE.1 are worth doing first on their own: even
without a UI they make every YAML change safe (the validator can run in CI),
and they remove the silent data loss that would otherwise come with the first
save.

## Rules carried over

From Track PC:
- derive the space, don't sample it;
- verify a guard in the state it prevents;
- a producer names its consumer;
- a test double reproduces the real contract;
- nothing writes to the live graph without a dry run the owner has seen.
