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
> round 2).
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
>   `typing_vocabulary`, with each ontology leading and with none. The first
>   version modelled its own idea of typing, missed the PC.4 bug class, and
>   blocked 553 of 935 legitimate re-parents. Rules:
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
>   - A test over the whole re-parent space (935 edits) checks every block
>     against production typing, computed independently.
> - `retype_planner` holds the retype plan, lifted out of the PC.4b script and
>   shared with it. A row that moves releases the identity it held.
> - `ontology_preview_service.preview(candidate, active)` returns:
>   - findings;
>   - label changes, with entity counts;
>   - the retype a publish would offer (changes, collisions, flag-only, and
>     how many changes and collisions this edit causes);
>   - notebook edits, which it does not project;
>   - the documents whose selection changes (the detection replayed with the
>     notebook default; the history fallback is not replayed).
>
>   The count matches the script's own dry run through the script's own
>   vocabulary path. Read-only.
> - **For OE.2:** the preview is synchronous (about 4–6 s on 14 sources, most
>   of it the detection replay); on a larger corpus the replay should become a
>   background job.


**OE.2 — Write API** (~1.5d)
- An ontology has a **draft**. Edits change the draft; publish, discard and
  rollback act on versions.
  - Edit ops: `create`, `update`, `rename`, `delete`, `reparent` for entity
    types and relationship types.
  - `POST …/validate`, `POST …/preview`, `POST …/publish` (validate + activate
    + invalidate caches), `GET …/versions`, `POST …/rollback/{version}`,
    `GET …/export.yaml`.
- A publish records who, when and why (a `note`).
- A publish with blocking findings is refused (422, with the findings).
- `rename` of a type keeps the old name as an alias, so rows and labels that
  use it keep resolving.
- **AC:**
  - a publish takes effect for the next extraction without a restart;
  - a rollback restores the previous typing exactly;
  - the registry's DB-first read is covered by a test against a real SurrealDB.

**OE.3 — Editor UI** (~2d)
- The existing read-only `/ontologies` page becomes the editor.
  - A type tree per ontology, showing its inheritance chain (`extends`) and
    which parent defines what.
  - A form per type for the v1 fields.
  - Validation findings inline, and a preview panel before publish (labels
    that change type, entities affected, collisions).
  - A version history with rollback.
- **AC:** an e2e test performs rename → preview → publish → rollback.

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
