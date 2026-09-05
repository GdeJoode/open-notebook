"""
Repository for entity resolution operations against the knowledge graph.

Provides lookup and alias management for canonical entity matching,
supporting the KG entity resolution pipeline.
"""

import hashlib
from typing import Any, Dict, List, Optional

from loguru import logger
from shared.models.entity import Entity, Relation
from shared.models.export import ExportFilter
from shared.utils.name_normalizer import normalize_entity_name

from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import (
    ensure_record_id,
    execute_query,
    get_pool,
)


def _union_preserve_order(
    existing: List[Any], incoming: List[Any]
) -> List[Any]:
    """Return ``existing + (incoming \\ existing)``, preserving order.

    Mirrors what ``array::union`` does in SurrealDB (dedup with stable order
    of first appearance). We implement it in Python because ``upsert_entity``
    pre-fetches the row and performs the merge client-side — see that
    docstring for rationale.

    Nested dicts/lists inside provenance_chain are JSON-comparable but not
    hashable, so we fall back to an O(n*m) scan rather than a set.
    """
    out = list(existing)
    for item in incoming:
        if item not in out:
            out.append(item)
    return out


#: Databases whose `entity` table has been confirmed to declare `name_key`, keyed
#: by the namespace/database the CONNECTION reports — not by the config object.
#: One check per process per database, not per write.
_IDENTITY_COLUMN_CHECKED: set = set()


class IdentityColumnMissing(RuntimeError):
    """The database predates migration 79, so `upsert_entity` cannot identify.

    Raised rather than degraded. Without `name_key` the lookup matches nothing
    and every upsert falls through to CREATE — so a re-ingest of the same
    document silently DOUBLES its entities. On a database between migrations 39
    and 79 the old `idx_entity_name_type` turns that into a confusing index
    error; below 39 there is no index and nothing complains at all. Migration 79
    removes that index, which is correct and also removes the accident that was
    catching this.
    """


async def _assert_identity_column(config: Any) -> None:
    """Fail loudly, once per database, when `name_key` is not declared.

    THE CONNECTION IS ASKED, NOT THE CONFIG — and asked in-process. An earlier
    version keyed this on `config.namespace/config.database`, which review showed
    to be unsound: `get_pool(config)` builds the global pool once and ignores the
    argument afterwards, so `execute_query(q, p, config)` does NOT go where
    `config` says. The check inspected whichever database the pool was bound to,
    reported a different one, and cached the verdict under a third — so one
    healthy database could vouch for a migration-31 one.

    The fix after that asked the database (`RETURN $session.…`), which was correct
    and cost a network round trip on EVERY upsert — ~20 ms each, so ~10 s on a
    500-entity document — because the query ran before the cache was consulted.
    Review measured it. `ConnectionPool.__init__` does `self.config = config or
    get_config()`, so the pool already knows which database it is on and the same
    answer is available without leaving the process.

    PC.6's rule applied to a repository: a step that cannot do its job says why
    rather than doing something else.
    """
    pool = get_pool(config)
    connected = getattr(pool, "config", None)
    if connected is None:  # a pool shape this function does not understand
        raise IdentityColumnMissing(
            "cannot determine which database this connection is on, so the "
            "identity column cannot be verified. Refusing rather than assuming — "
            "an unverified write is how a re-ingest silently doubles a document."
        )
    key = f"{connected.namespace}/{connected.database}"
    if key in _IDENTITY_COLUMN_CHECKED:
        return

    info = await execute_query("INFO FOR TABLE entity;", None, config)
    table = info if isinstance(info, dict) else (info[0] if info else {})
    fields = table.get("fields") or {}
    if "name_key" not in fields:
        raise IdentityColumnMissing(
            f"`entity.name_key` is not declared on `{key}` (the database this "
            f"connection is actually on). This code identifies entities by "
            f"`name_key` (migration 79); without it every upsert misses its "
            f"lookup and creates a duplicate instead of updating. Run the "
            f"migrations against this database — and note that migration 79 "
            f"refuses until `scripts/backfill_name_key.py` has run, because the "
            f"key cannot be computed in SurrealQL."
        )
    _IDENTITY_COLUMN_CHECKED.add(key)


class EntityRepository:
    """Repository for entity resolution operations.

    Handles entity lookup by alias, type-based candidate retrieval,
    alias registration, and embedding access for the knowledge graph
    entity resolution pipeline.

    This repository does not inherit from BaseRepository because
    there is no corresponding Entity model in the shared models package.
    It operates directly against the entity and entity_alias tables
    using raw SurrealQL queries.
    """

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        """Initialize the entity repository.

        Args:
            config: Optional SurrealDB configuration. Uses the global
                configuration if not provided.
        """
        self.config = config

    async def find_by_alias(self, alias_text: str) -> Optional[Dict[str, Any]]:
        """Find a canonical entity by exact alias text match.

        Queries the entity_alias table for an exact match on the alias text
        and returns the associated canonical entity information.

        **Ordering is part of the contract** (PC.2). Nothing stops two rows from
        binding the same ``alias_text`` to two different canonical entities —
        ``entity_alias`` has no uniqueness constraint on it, and `register_alias`
        writes a row per resolution. With a bare ``LIMIT 1`` the caller then gets
        whichever row the storage engine happened to return, so the same surface
        form can resolve to a different entity between two runs over identical
        input. That is an identity failure the pipeline cannot detect downstream:
        both answers look equally confident.

        The order encodes a policy, not a preference:

        1. ``verified DESC`` — a human decision outranks a machine one. This is
           also the disagreement this method was on the wrong side of:
           `vault_sync_service` exports only ``verified = true`` aliases, while
           this reader accepted any. An unverified alias is a machine guess, and
           using it as tier-1 ground truth for the NEXT resolution is a feedback
           loop that hardens guesses into identity.
        2. ``similarity_score DESC`` — among equally-trusted rows, the strongest.
        3. ``id ASC`` — a stable tie-break, so the result is reproducible rather
           than merely usually-the-same.

        Unverified aliases are still returned when nothing better exists: filtering
        them out entirely would make tier 1 inert, since every writer today writes
        ``verified = false``. Ranking rather than filtering keeps the recall and
        removes the non-determinism.

        Args:
            alias_text: The alias text to search for (exact match).

        Returns:
            A dictionary with keys ``id``, ``name``, ``match_type``,
            ``similarity_score`` and ``verified`` for the resolved canonical
            entity, or None if no match is found.
        """
        if not alias_text:
            return None

        try:
            result = await execute_query(
                "SELECT canonical_entity.id AS id, "
                "canonical_entity.name AS name, "
                "match_type, similarity_score, verified "
                "FROM entity_alias "
                "WHERE alias_text = $alias_text "
                "ORDER BY verified DESC, similarity_score DESC, id ASC "
                "LIMIT 1",
                {"alias_text": alias_text},
                self.config,
            )
            if result:
                return result[0]
            return None
        except Exception as e:
            logger.error(f"Failed to find entity by alias '{alias_text}': {e}")
            return None

    async def find_by_type(
        self, entity_type: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get candidate entities for matching by entity type.

        Retrieves entities of a given type along with their embeddings,
        which can be used for similarity-based entity resolution.

        **Retired rows are excluded (PC.3).** This used to select every row of
        the type regardless of status, which made most of what the resolver
        compared against rows triage had deliberately removed. Measured on
        `staging`:

            type                     live   ALL   the capped 100 held
            topic                     785  1408    31 active rows
            concept                   574  1892     0 active rows

        For `concept` the resolver could not reach a single one of the three
        active entities — every candidate it was offered was archived or merged,
        so a correct match was structurally impossible. `("active", "reference")`
        is the live set, and it is not a new rule: `audit_service` and
        `deep_audit_service` already use exactly that pair to decide what counts
        as graph content.

        **The capped slice is now DECLARED rather than incidental.** `LIMIT` with
        no `ORDER BY` is not guaranteed to return any particular rows. Measured,
        rather than assumed: on SurrealDB v2 with 20 rows the fetch returned id
        order with the clause and without it, so removing it changes nothing
        today and the behavioural test cannot catch its removal — which is why
        `test_the_query_states_both_rules` reads the query text instead, and says
        so. The clause earns its place for two reasons: the ordering becomes a
        real choice the moment `weight` carries centrality (it is uniformly 0.0
        on this corpus), and an ordering nobody wrote down is one nobody can rely
        on. The cap still truncates either way, and the resolver counts
        `capped_fetches` so that stays visible.

        Args:
            entity_type: The entity type to filter by (e.g. "PERSON", "ORG").
            limit: Maximum number of entities to return. Defaults to 100.

        Returns:
            A list of dictionaries, each containing ``id``, ``name``,
            ``embedding``, and ``weight`` for a matching entity. Returns
            an empty list on failure.
        """
        try:
            return await execute_query(
                "SELECT id, name, embedding, weight "
                "FROM entity "
                "WHERE entity_type = $entity_type "
                "AND status IN ['active', 'reference'] "
                "ORDER BY weight DESC, id ASC "
                "LIMIT $limit",
                {"entity_type": entity_type, "limit": limit},
                self.config,
            )
        except Exception as e:
            logger.error(
                f"Failed to find entities by type '{entity_type}': {e}"
            )
            return []

    async def upsert_entity(self, entity: Entity) -> str:
        """Upsert a canonical entity, returning its record ID.

        Lookup is by ``(canonical_name, entity_type)`` — the unique-index pair
        declared in migration 39 (``idx_entity_name_type``). When a row with
        the same name+type already exists, the upsert merges (Python-side
        merge of provenance/scoring fields, then a single UPDATE):

        - confidence: ``max`` of existing vs new (monotonic, never drops)
        - source_documents: union (dedup-preserving)
        - properties: dict overlay — new keys win, existing keys retained
        - type_tags: union (multi-type accumulation from B1 merge)
        - primary_type: replaced with the new value when supplied
        - provenance_chain: union

        All other fields (``status``, ``embedding``, ``description``, ...) keep
        the existing row's value on update. New rows take all fields from
        ``entity`` directly, including the required ``embedding`` (which has
        no DB-side default — see migration 39 line 30).

        Why we merge in Python rather than in SurrealQL: ``object::extend`` /
        ``object::merge`` are not available on SurrealDB v2.x at the time of
        writing — calls produce
        ``Parse error: Invalid function/constant path``. Pre-fetching the
        existing row (one short SELECT) and applying the merge in Python keeps
        the contract identical while sidestepping the SurrealQL gap.

        Args:
            entity: An ``Entity`` model populated with the canonical-schema
                field names. ``embedding`` MUST be a list (use ``[]`` when
                no vector is available).

        Returns:
            The record ID of the upserted entity, e.g. ``"entity:abc123"``.

        Raises:
            RuntimeError: If the query fails.

        Note:
            The SELECT-then-UPDATE flow is not atomic — two concurrent
            ``upsert_entity`` calls for the same
            ``(canonical_name, entity_type)`` can both observe "no
            existing row" and then race on CREATE; the migration-39
            ``idx_entity_name_type`` UNIQUE index will reject one of
            them. This is acceptable today because all writers go
            through the single-process ``EntityPersistenceService``.
            **B.1e must wrap this in a per-canonical-name lock or move
            the merge into a SurrealDB transaction** before introducing
            parallel writers.
        """
        # PC.3: identity is `name_key`, not the display name. Looking up on the
        # RAW `canonical_name` is why `Brede Welvaart` and `brede welvaart` were
        # two rows — the lookup missed, so the create path ran twice. Derived
        # here rather than taken from the caller: a key the caller could supply
        # is a key the caller could get wrong, and migration 79's UNIQUE index
        # would then reject the write with nothing to say about why.
        name_key = normalize_entity_name(entity.canonical_name)

        # Refuse on a database that cannot express identity — see
        # `_assert_identity_column`. Once per process per database.
        await _assert_identity_column(self.config)

        try:
            existing_rows = await execute_query(
                "SELECT * FROM entity "
                "WHERE name_key = $name_key "
                "AND entity_type = $entity_type LIMIT 1",
                {
                    "name_key": name_key,
                    "entity_type": entity.entity_type,
                },
                self.config,
            )
        except Exception as e:
            logger.exception(
                f"upsert_entity lookup failed for "
                f"'{entity.canonical_name}' ({entity.entity_type}): {e}"
            )
            raise

        if existing_rows:
            existing = existing_rows[0]
            merged_confidence = max(
                float(existing.get("confidence", 0.0) or 0.0),
                entity.confidence,
            )
            merged_sources = _union_preserve_order(
                existing.get("source_documents") or [],
                list(entity.source_documents),
            )
            merged_type_tags = _union_preserve_order(
                existing.get("type_tags") or [],
                list(entity.type_tags),
            )
            merged_provenance = _union_preserve_order(
                existing.get("provenance_chain") or [],
                list(entity.provenance_chain),
            )
            merged_properties: Dict[str, Any] = dict(
                existing.get("properties") or {}
            )
            # PC.3: `grounding` merges by SOURCE ID; every other key overlays.
            #
            # It maps `source_id -> {chunk_id, grounding, context}`, one entry per
            # document a mention came from. Under the plain overlay below, the
            # second document to mention an entity would replace where the first
            # one found it — and with cross-document resolution on that is the
            # normal case, not an edge one. The union is what lets a curator ask
            # "why is this one entity" and get an answer per source.
            incoming_grounding = entity.properties.get("grounding")
            if isinstance(incoming_grounding, dict):
                merged_grounding = dict(merged_properties.get("grounding") or {})
                merged_grounding.update(incoming_grounding)
            else:
                merged_grounding = None

            merged_properties.update(entity.properties)
            if merged_grounding:
                merged_properties["grounding"] = merged_grounding

            update_payload = {
                "id": existing["id"],
                "confidence": merged_confidence,
                "source_documents": merged_sources,
                "properties": merged_properties,
                "type_tags": merged_type_tags,
                "primary_type": entity.primary_type
                if entity.primary_type is not None
                else existing.get("primary_type"),
                "provenance_chain": merged_provenance,
                # manual_override (migration 59) is OPERATOR state, never
                # extraction state: re-upserting an entity from a new document
                # must NOT clear an operator's pin. Carry the existing value
                # through verbatim (the triage decision is the only writer of
                # this flag, via ``set_status``). NULL-coalesce so a pre-59 row
                # lacking the column reads as False.
                "manual_override": bool(existing.get("manual_override") or False),
            }
            try:
                await execute_query(
                    """
                    UPDATE type::thing($id) SET
                        confidence = $confidence,
                        source_documents = $source_documents,
                        properties = $properties,
                        type_tags = $type_tags,
                        primary_type = $primary_type,
                        provenance_chain = $provenance_chain,
                        manual_override = $manual_override,
                        updated_at = time::now();
                    """,
                    update_payload,
                    self.config,
                )
            except Exception as e:
                logger.exception(
                    f"upsert_entity update failed for "
                    f"'{entity.canonical_name}' ({entity.entity_type}): {e}"
                )
                raise
            return str(existing["id"])

        # No existing row — fresh CREATE.
        # ``hash_id`` is a required string on the live DB (schema drift — not in
        # any migration, so the testcontainer schema doesn't enforce it and the
        # roundtrip tests pass while live writes fail with "Found NONE for field
        # hash_id"). Derive it deterministically from the EXACT dedup identity
        # (canonical_name + entity_type) used by the existing-row lookup and the
        # case-sensitive ``idx_entity_name_type``. Do NOT lower-case/strip here:
        # the dedup key is case-sensitive, so folding case would make distinct
        # entities ("BZK" vs "bzk") collide on the UNIQUE ``idx_entity_hash``.
        _hash_basis = f"{entity.canonical_name}|{entity.entity_type}"
        create_payload: Dict[str, Any] = {
            "canonical_name": entity.canonical_name,
            # PC.3: identity, and the UNIQUE index's column. Note it is derived
            # from the SAME string the lookup used, so a create can only be
            # reached when that key genuinely had no row.
            "name_key": name_key,
            "entity_type": entity.entity_type,
            "hash_id": hashlib.md5(_hash_basis.encode("utf-8")).hexdigest(),
            "description": entity.description,
            "source_documents": list(entity.source_documents),
            "extraction_method": entity.extraction_method,
            "confidence": entity.confidence,
            "provenance_chain": list(entity.provenance_chain),
            "properties": dict(entity.properties),
            "embedding": list(entity.embedding),
            "status": entity.status,
            # manual_override (migration 59) — operator-pinned status guard. A
            # fresh entity is automation-owned unless explicitly set; the model
            # defaults to False so this is safe for entities built before the
            # field existed.
            "manual_override": bool(getattr(entity, "manual_override", False)),
            "type_tags": list(entity.type_tags),
            "primary_type": entity.primary_type,
        }

        try:
            result = await execute_query(
                """
                CREATE entity SET
                    canonical_name = $canonical_name,
                    name = $canonical_name,
                    name_key = $name_key,
                    entity_type = $entity_type,
                    hash_id = $hash_id,
                    description = $description,
                    source_documents = $source_documents,
                    extraction_method = $extraction_method,
                    confidence = $confidence,
                    provenance_chain = $provenance_chain,
                    properties = $properties,
                    embedding = $embedding,
                    status = $status,
                    manual_override = $manual_override,
                    type_tags = $type_tags,
                    primary_type = $primary_type;
                """,
                create_payload,
                self.config,
            )
        except Exception as e:
            logger.exception(
                f"upsert_entity create failed for "
                f"'{entity.canonical_name}' ({entity.entity_type}): {e}"
            )
            raise

        if not result:
            raise RuntimeError(
                f"CREATE entity returned no rows for "
                f"'{entity.canonical_name}' ({entity.entity_type})"
            )
        return str(result[0]["id"])

    async def get_entity(self, record_id: str) -> Optional[Entity]:
        """Fetch a single entity by record ID, returning a typed ``Entity``.

        Selects all migration-39/44 fields and parses the result into the
        Pydantic model. Used by B.1e's merge step to pick canonical winners
        by recency (and elsewhere wherever a typed handle is preferred over
        a raw dict).

        Args:
            record_id: Entity record ID (e.g. ``"entity:abc123"``).

        Returns:
            An ``Entity`` instance, or ``None`` if no row exists.
        """
        if not record_id:
            return None
        try:
            rid = ensure_record_id(record_id)
            rows = await execute_query(
                "SELECT * FROM entity WHERE id = $id LIMIT 1",
                {"id": rid},
                self.config,
            )
        except Exception as e:
            logger.error(f"Failed to get entity '{record_id}': {e}")
            return None
        if not rows:
            return None
        # ``execute_query`` already converts RecordIDs to strings.
        return Entity(**rows[0])

    # ------------------------------------------------------------------
    # Track Q Phase Q.3 — triage status assignment
    # ------------------------------------------------------------------

    async def set_status(
        self,
        entity_id: str,
        status: str,
        *,
        respect_override: bool = True,
    ) -> bool:
        """Set an entity's ``status``, honouring the operator override pin.

        This is the single DB-side writer of the triage-derived ``status``. The
        ``manual_override`` guard (migration 59) is the load-bearing contract of
        Track Q: when an operator has pinned the status by hand, the triage
        automation MUST NEVER overwrite it. With ``respect_override=True`` (the
        default, used by all automation) the UPDATE is conditioned on
        ``manual_override = false`` in a single round-trip — so the no-op is
        race-free, not a read-then-write check the operator could slip between.

        The operator-decision path (Q.4's queue decision) calls this with
        ``respect_override=False`` to set both the status and, separately, the
        override flag — automation never takes that branch.

        Args:
            entity_id: Record ID of the entity (e.g. ``"entity:abc"``).
            status: The new status value (e.g. ``"active"`` / ``"reference"``).
                ``entity.status`` is a free string (migration 39, no ASSERT),
                so ``"reference"`` is accepted without a schema change.
            respect_override: When True, the write is a no-op if the entity is
                operator-pinned (``manual_override = true``).

        Returns:
            True if the status was written; False if it was suppressed because
            the entity is operator-pinned (or the id was invalid / not found).
        """
        if not entity_id:
            return False
        try:
            rid = ensure_record_id(entity_id)
        except Exception as e:
            logger.error(f"set_status: bad id {entity_id!r}: {e}")
            return False

        # Condition the UPDATE on the override flag in the query itself so the
        # check and the write are one atomic statement. ``UPDATE`` returns the
        # affected rows, so an empty result means the WHERE excluded the row
        # (operator-pinned) — that is the suppressed-write signal.
        where = "id = $id"
        if respect_override:
            where += " AND (manual_override = false OR manual_override = NONE)"
        try:
            rows = await execute_query(
                f"UPDATE entity SET status = $status, updated_at = time::now() "
                f"WHERE {where} RETURN AFTER;",
                {"id": rid, "status": status},
                self.config,
            )
        except Exception as e:
            logger.exception(f"set_status failed for {entity_id} -> {status}: {e}")
            raise
        return bool(rows)

    async def set_manual_override(
        self,
        entity_id: str,
        *,
        status: Optional[str] = None,
        manual_override: bool = True,
    ) -> bool:
        """Pin an entity by hand (Q.4 operator decision): set status + override.

        This is the ONLY writer of ``manual_override = true`` — the operator's
        decision path (the triage-queue decision endpoint), NEVER automation. It
        is UNconditional (no ``respect_override`` guard): an operator can always
        re-pin. When ``status`` is given it is written in the same statement so
        the pin and the chosen status land atomically; pass ``status=None`` to
        flip the override flag alone.

        Args:
            entity_id: Record ID of the entity to pin.
            status: The status to pin (e.g. ``"active"``); ``None`` leaves the
                current status untouched.
            manual_override: The override value to write (default ``True``;
                pass ``False`` to release a pin back to automation).

        Returns:
            True if the entity was updated; False on a bad/missing id.
        """
        if not entity_id:
            return False
        try:
            rid = ensure_record_id(entity_id)
        except Exception as e:
            logger.error(f"set_manual_override: bad id {entity_id!r}: {e}")
            return False

        sets = ["manual_override = $override", "updated_at = time::now()"]
        params: Dict[str, Any] = {"id": rid, "override": bool(manual_override)}
        if status is not None:
            sets.insert(0, "status = $status")
            params["status"] = status
        try:
            rows = await execute_query(
                f"UPDATE entity SET {', '.join(sets)} WHERE id = $id RETURN AFTER;",
                params,
                self.config,
            )
        except Exception as e:
            logger.exception(
                f"set_manual_override failed for {entity_id} "
                f"(status={status}, override={manual_override}): {e}"
            )
            raise
        return bool(rows)

    # ------------------------------------------------------------------
    # Track K Phase K.3 — retroactive merge primitives
    # ------------------------------------------------------------------
    #
    # These helpers are the DB-side seam the ``RecanonicalizationService``
    # composes into one logical merge. They are ID-based (never name-based)
    # so a relation pointing at "the org named X" can never be re-attached to
    # "the person named X" — the cross-type ambiguity the K.7 plan calls out.

    async def repoint_relations(
        self, loser_id: str, winner_id: str
    ) -> int:
        """Re-point every ``relation`` edge incident on ``loser`` onto ``winner``.

        SurrealDB RELATE edges anchor their ``in`` / ``out`` record links at
        creation; mutating them in place is unreliable on v2.x. So this does a
        **delete-and-recreate**: for each edge touching the loser, RELATE an
        equivalent edge with the loser endpoint swapped for the winner, then
        delete the original edge. Edges that would become self-loops on the
        winner (``winner -> winner``) are dropped rather than recreated.

        **Field completeness**: the recreated edge carries EVERY data field the
        original held — every ``DEFINE FIELD ... ON relation`` from migration 39:
        ``relation_type``, ``properties``, ``source_documents``, ``extracted_at``,
        ``extraction_method``, ``confidence``, ``provenance_chain``, ``status``,
        and ``created_at``. Both timestamp fields are provenance and are copied
        through verbatim — without that, migration 39's
        ``DEFAULT time::now()`` on ``extracted_at`` / ``created_at`` would
        silently re-stamp every re-pointed edge with the merge time, destroying
        the original extraction/creation provenance. The only field that may
        legitimately differ on the recreate is the edge's own record ``id``
        (RELATE mints a fresh one); no other data is lost or mutated.

        **Parallel-edge dedup**: before recreating an edge, we probe for an
        existing ``(in, out, relation_type)`` triple on the winner; if one
        already exists we skip the RELATE (the original is still deleted), so
        re-pointing two parallel loser edges onto the winner collapses to one.

        Idempotency: when ``loser`` has no incident edges (e.g. a re-run after
        a prior merge already moved them) this is a no-op returning 0.

        Args:
            loser_id: Record ID of the entity being merged away.
            winner_id: Record ID of the surviving canonical entity.

        Returns:
            The number of edges deleted from the loser (re-pointed or dropped).
        """
        if not loser_id or not winner_id:
            return 0
        try:
            loser = ensure_record_id(loser_id)
            winner = ensure_record_id(winner_id)
        except Exception as e:
            logger.error(
                f"repoint_relations: bad id loser={loser_id!r} "
                f"winner={winner_id!r}: {e}"
            )
            return 0

        try:
            edges = await execute_query(
                "SELECT id, in, out, relation_type, properties, "
                "source_documents, extracted_at, extraction_method, "
                "confidence, provenance_chain, status, created_at "
                "FROM relation WHERE in = $loser OR out = $loser",
                {"loser": loser},
                self.config,
            )
        except Exception as e:
            logger.exception(
                f"repoint_relations: failed to load edges for "
                f"loser={loser_id}: {e}"
            )
            raise

        if not edges:
            return 0

        winner_str = str(winner)
        loser_str = str(loser)
        moved = 0
        for edge in edges:
            src = str(edge.get("in"))
            tgt = str(edge.get("out"))
            new_src = winner_str if src == loser_str else src
            new_tgt = winner_str if tgt == loser_str else tgt

            edge_id = edge.get("id")
            # Drop self-loops that the merge would create.
            if new_src == new_tgt:
                await self._delete_relation(edge_id)
                moved += 1
                continue

            rel_type = edge.get("relation_type") or ""
            # Parallel-edge dedup: skip RELATE if the winner already carries
            # an equivalent (in, out, relation_type) edge.
            existing = await execute_query(
                "SELECT id FROM relation "
                "WHERE in = type::thing($src) AND out = type::thing($tgt) "
                "AND relation_type = $rt LIMIT 1",
                {"src": new_src, "tgt": new_tgt, "rt": rel_type},
                self.config,
            )
            if not existing:
                # RELATE arrow syntax needs bare record-id literals in the
                # source/target positions (SurrealDB issue #4232); the ids are
                # ``entity:<alnum>`` record ids straight from the DB (no user
                # input), so interpolation is safe. Metadata is bound as params.
                # Copy EVERY relation data field through the recreate so the
                # re-pointed edge is byte-for-byte equivalent to the original
                # save for its fresh record id. ``extracted_at`` / ``created_at``
                # MUST be passed explicitly: omitting them lets migration 39's
                # ``DEFAULT time::now()`` re-stamp the edge with the merge time,
                # silently destroying the original extraction/creation
                # provenance (the K.3 BLOCKER). They round-trip as datetime
                # objects from the SELECT, so SurrealDB stores them verbatim.
                set_clauses = [
                    "relation_type = $rt",
                    "properties = $properties",
                    "source_documents = $source_documents",
                    "extraction_method = $extraction_method",
                    "confidence = $confidence",
                    "provenance_chain = $provenance_chain",
                    "status = $status",
                ]
                params: Dict[str, Any] = {
                    "rt": rel_type,
                    "properties": edge.get("properties") or {},
                    "source_documents": edge.get("source_documents") or [],
                    "extraction_method": edge.get("extraction_method") or "llm",
                    "confidence": float(edge.get("confidence") or 1.0),
                    "provenance_chain": edge.get("provenance_chain") or [],
                    "status": edge.get("status") or "active",
                }
                # Preserve provenance timestamps only when the original carried
                # them; on a (theoretical) null we let the DEFAULT apply rather
                # than write an explicit null into a non-optional datetime field.
                extracted_at = edge.get("extracted_at")
                if extracted_at is not None:
                    set_clauses.append("extracted_at = $extracted_at")
                    params["extracted_at"] = extracted_at
                created_at = edge.get("created_at")
                if created_at is not None:
                    set_clauses.append("created_at = $created_at")
                    params["created_at"] = created_at

                await execute_query(
                    f"RELATE {new_src}->relation->{new_tgt} SET "
                    + ", ".join(set_clauses)
                    + ";",
                    params,
                    self.config,
                )
            await self._delete_relation(edge_id)
            moved += 1

        logger.info(
            "repoint_relations: loser={loser} winner={winner} moved={moved}",
            loser=loser_id,
            winner=winner_id,
            moved=moved,
        )
        return moved

    async def _delete_relation(self, edge_id: Any) -> None:
        """Delete a single relation edge by record id (helper for repoint)."""
        if edge_id is None:
            return
        await execute_query(
            "DELETE type::thing($id);",
            {"id": str(edge_id)},
            self.config,
        )

    async def mark_merged(self, loser_id: str, winner_id: str) -> bool:
        """Soft-merge ``loser`` into ``winner`` (status + merged_into).

        Sets ``status='merged'`` and ``merged_into=<winner id>`` on the loser.
        The loser row is **never hard-deleted** — the merge is reversible via
        ``merged_into``. Idempotent: re-marking an already-merged loser is a
        cheap no-op UPDATE.

        Args:
            loser_id: Record ID of the entity being merged away.
            winner_id: Record ID of the surviving canonical entity.

        Returns:
            ``True`` when the update ran (even if nothing changed),
            ``False`` on a transport-level failure or bad id.
        """
        if not loser_id or not winner_id:
            return False
        try:
            rid = ensure_record_id(loser_id)
        except Exception as e:
            logger.error(f"mark_merged: invalid loser_id '{loser_id}': {e}")
            return False
        try:
            await execute_query(
                "UPDATE type::thing($id) SET "
                "status = 'merged', merged_into = $winner, "
                "updated_at = time::now();",
                {"id": rid, "winner": str(winner_id)},
                self.config,
            )
        except Exception as e:
            logger.error(
                f"mark_merged failed for loser='{loser_id}' "
                f"winner='{winner_id}': {e}"
            )
            return False
        return True

    async def update_external_ids(
        self,
        entity_id: str,
        external_ids: List[str],
        aliases: Optional[List[str]] = None,
    ) -> bool:
        """Union the reconciled external_ids (+ aliases) onto an entity (K.4).

        The vocabulary reconciler (K.4) calls this after linking an entity to a
        single high-confidence ``reference_entity``. The write is dedup-preserving
        union, NOT a replace, so a later reconcile against another vocabulary
        (e.g. ORCID after TOOI) accrues rather than clobbers. Never touches
        ``canonical_name`` / ``entity_type`` / ``hash_id`` (B.8 contract intact).

        Args:
            entity_id: Record id of the entity to annotate.
            external_ids: URIs to union onto ``entity.external_ids``.
            aliases: Optional canonical/alias surface forms to union onto
                ``entity.aliases``.

        Returns:
            ``True`` when the update ran, ``False`` on a bad id / transport error.
        """
        if not entity_id:
            return False
        try:
            rid = ensure_record_id(entity_id)
        except Exception as e:
            logger.error(f"update_external_ids: invalid id '{entity_id}': {e}")
            return False
        try:
            existing_rows = await execute_query(
                "SELECT external_ids, aliases FROM type::thing($id) LIMIT 1;",
                {"id": rid},
                self.config,
            )
            existing = existing_rows[0] if existing_rows else {}
            merged_ids = _union_preserve_order(
                existing.get("external_ids") or [], list(external_ids or [])
            )
            merged_aliases = _union_preserve_order(
                existing.get("aliases") or [], list(aliases or [])
            )
            await execute_query(
                "UPDATE type::thing($id) SET "
                "external_ids = $ext, aliases = $aliases, "
                "updated_at = time::now();",
                {"id": rid, "ext": merged_ids, "aliases": merged_aliases},
                self.config,
            )
        except Exception as e:
            logger.error(f"update_external_ids failed for '{entity_id}': {e}")
            return False
        return True

    async def list_active_entities(
        self, source_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Load ``status='active'`` entities for retroactive-merge planning.

        Returns the fields the planner needs to compute the new canonical form,
        group collisions, and pick a winner: identity, the provenance bag, the
        winner-selection inputs (``confidence``, ``source_documents``,
        ``created_at``), plus ``aliases`` so a re-plan after a partial merge
        carries them forward. ``merged`` / ``archived`` rows are excluded.

        Args:
            source_ids: Optional list of source record ids — when supplied, only
                entities whose ``source_documents`` intersect this set are
                returned (the per-notebook scope; the caller resolves the
                notebook → sources mapping). ``None`` = global.

        Returns:
            A list of active-entity row dicts. Empty on failure / no matches.
        """
        try:
            if source_ids is not None:
                if not source_ids:
                    return []
                return await execute_query(
                    "SELECT id, canonical_name, entity_type, confidence, "
                    "source_documents, type_tags, primary_type, "
                    "provenance_chain, properties, aliases, status, "
                    "created_at "
                    "FROM entity "
                    "WHERE status = 'active' "
                    "AND source_documents ANYINSIDE $source_ids",
                    {"source_ids": source_ids},
                    self.config,
                )
            return await execute_query(
                "SELECT id, canonical_name, entity_type, confidence, "
                "source_documents, type_tags, primary_type, "
                "provenance_chain, properties, aliases, status, created_at "
                "FROM entity WHERE status = 'active'",
                {},
                self.config,
            )
        except Exception as e:
            logger.error(f"list_active_entities failed: {e}")
            return []

    async def list_active_entities_with_embeddings(
        self, source_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Like :meth:`list_active_entities` but also projects ``embedding``.

        The K.5 candidate dedup service needs the per-entity ``embedding`` vector
        to run the embedding resolver. Kept SEPARATE from
        :meth:`list_active_entities` so the K.3 row contract (which the
        recanonicalization service depends on) stays byte-identical. The
        embedding is a top-level ``entity`` field; rows without one come back
        with ``embedding=[]`` so the dedup service falls back to fuzzy-only.

        Args:
            source_ids: Optional notebook scope (same semantics as
                :meth:`list_active_entities`).

        Returns:
            Active-entity rows including ``embedding``. Empty on failure.
        """
        try:
            if source_ids is not None:
                if not source_ids:
                    return []
                return await execute_query(
                    "SELECT id, canonical_name, entity_type, confidence, "
                    "source_documents, aliases, status, created_at, embedding "
                    "FROM entity "
                    "WHERE status = 'active' "
                    "AND source_documents ANYINSIDE $source_ids",
                    {"source_ids": source_ids},
                    self.config,
                )
            return await execute_query(
                "SELECT id, canonical_name, entity_type, confidence, "
                "source_documents, aliases, status, created_at, embedding "
                "FROM entity WHERE status = 'active'",
                {},
                self.config,
            )
        except Exception as e:
            logger.error(f"list_active_entities_with_embeddings failed: {e}")
            return []

    async def list_active_entities_missing_embedding(
        self, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load ``status='active'`` entities whose ``embedding`` is empty/missing.

        Powers the P.1 backfill: the corpus persisted before the forward fix
        stored ``embedding=[]`` for every entity, which disables K.5's semantic
        dedup band. This returns the minimal projection the backfill needs to
        embed and update each one — ``id`` (to update) + ``canonical_name`` (the
        text to embed; identical to the ``text`` the forward path embeds, since
        the persist boundary sets ``canonical_name = text``).

        The filter matches both the legacy empty-array form (``embedding = []``)
        and any row where the field is NONE/unset, so the backfill is robust to
        schema drift. Rows that ALREADY carry a vector are excluded — that is
        what makes a re-run idempotent (it only ever touches the not-yet-done
        remainder).

        Args:
            limit: Optional cap on the number of rows returned (for batched /
                resumable runs). ``None`` returns the full remainder.

        Returns:
            A list of ``{"id", "canonical_name"}`` dicts. Empty on failure or
            when every active entity already has a vector.
        """
        # ``array::len(embedding ?? []) == 0`` is true for both [] and NONE; the
        # null-coalesce guards the unset case without a separate IS NONE branch.
        where = (
            "status = 'active' "
            "AND array::len(embedding ?? []) == 0"
        )
        # SurrealDB v2 requires the ORDER BY idiom to appear in the projection,
        # so ``created_at`` is selected even though the backfill only reads
        # ``id`` / ``canonical_name``. Ordering by creation gives a stable,
        # resumable cursor across batched runs.
        query = (
            f"SELECT id, canonical_name, created_at FROM entity WHERE {where} "
            "ORDER BY created_at"
        )
        if limit is not None:
            query += " LIMIT $limit"
        try:
            return await execute_query(
                query,
                {"limit": limit} if limit is not None else {},
                self.config,
            )
        except Exception as e:
            logger.error(
                f"list_active_entities_missing_embedding failed: {e}"
            )
            return []

    async def update_entity_embedding(
        self, entity_id: str, embedding: List[float]
    ) -> bool:
        """Set the ``embedding`` vector on a single entity (P.1 backfill).

        Writes ONLY the embedding (+ ``updated_at``); never touches
        ``canonical_name`` / ``entity_type`` / ``hash_id`` (the B.8 dedup-key
        contract) or any provenance field. Idempotent at the row level — calling
        it again with the same vector is a cheap no-op UPDATE.

        Args:
            entity_id: Record id of the entity to annotate (e.g. ``entity:abc``).
            embedding: The vector to store (the configured model's output —
                the I.G 768-dim pin; not validated/re-shaped here).

        Returns:
            ``True`` when the update ran, ``False`` on a bad id / transport error
            or an empty vector (an empty write would be a no-op that leaves the
            row still "missing", so it is rejected as a caller error).
        """
        if not entity_id or not embedding:
            return False
        try:
            rid = ensure_record_id(entity_id)
        except Exception as e:
            logger.error(f"update_entity_embedding: invalid id '{entity_id}': {e}")
            return False
        try:
            await execute_query(
                "UPDATE type::thing($id) SET "
                "embedding = $embedding, updated_at = time::now();",
                {"id": rid, "embedding": list(embedding)},
                self.config,
            )
        except Exception as e:
            logger.error(
                f"update_entity_embedding failed for '{entity_id}': {e}"
            )
            return False
        return True

    async def merge_into_winner(
        self, winner_id: str, losers: List[Dict[str, Any]]
    ) -> bool:
        """Fold loser provenance/scoring fields into the winner row.

        Reuses the shared ``_union_preserve_order`` helper (not a
        reimplementation). The union/max math is **identical to
        ``upsert_entity``**:

        - ``confidence``: ``max`` over winner + all losers (monotonic)
        - ``source_documents``: union (dedup-preserving)
        - ``type_tags``: union
        - ``provenance_chain``: union
        - ``aliases``: union of winner aliases + each loser's canonical_name
          and prior aliases (the denormalized surface-form list, migration 54)

        **One intentional difference from ``upsert_entity``** — the
        ``properties`` overlay direction is *reversed*:

        - ``upsert_entity``: existing is the base, the INCOMING (caller-supplied)
          row overlays it, so on a key conflict the incoming value wins.
        - ``merge_into_winner``: the losers form the base, the WINNER (the
          surviving canonical row) overlays them, so on a key conflict the
          winner's own value wins.

        This is deliberate: in a retroactive merge the winner is authoritative,
        so its property values must survive the fold rather than be clobbered by
        a loser. Loser-only keys still fill gaps. Every other merged field uses
        identical semantics to ``upsert_entity``.

        The winner's own ``canonical_name`` / ``entity_type`` / ``hash_id`` are
        left untouched (the B.8 dedup-key contract). Idempotent: re-folding the
        same losers re-derives the identical union (no growth).

        Args:
            winner_id: Record ID of the surviving canonical entity.
            losers: Loser entity rows (each a dict with the provenance fields).

        Returns:
            ``True`` when the update ran, ``False`` on failure / bad id.
        """
        if not winner_id:
            return False
        try:
            rid = ensure_record_id(winner_id)
            existing_rows = await execute_query(
                "SELECT * FROM entity WHERE id = $id LIMIT 1",
                {"id": rid},
                self.config,
            )
        except Exception as e:
            logger.error(f"merge_into_winner: load winner '{winner_id}': {e}")
            return False
        if not existing_rows:
            logger.warning(f"merge_into_winner: winner '{winner_id}' not found")
            return False

        winner = existing_rows[0]
        merged_confidence = float(winner.get("confidence", 0.0) or 0.0)
        merged_sources = list(winner.get("source_documents") or [])
        merged_type_tags = list(winner.get("type_tags") or [])
        merged_provenance = list(winner.get("provenance_chain") or [])
        merged_aliases = list(winner.get("aliases") or [])
        # Winner keys must win the overlay, so start from losers and let the
        # winner's own properties overwrite last. This is the REVERSE of
        # upsert_entity (which lets the incoming row win): in a merge the winner
        # is the authoritative survivor, so its values must not be clobbered.
        merged_properties: Dict[str, Any] = {}

        for loser in losers:
            merged_confidence = max(
                merged_confidence, float(loser.get("confidence", 0.0) or 0.0)
            )
            merged_sources = _union_preserve_order(
                merged_sources, list(loser.get("source_documents") or [])
            )
            merged_type_tags = _union_preserve_order(
                merged_type_tags, list(loser.get("type_tags") or [])
            )
            merged_provenance = _union_preserve_order(
                merged_provenance, list(loser.get("provenance_chain") or [])
            )
            # Loser surface form + its own prior aliases become winner aliases.
            loser_surface = loser.get("canonical_name")
            incoming_aliases = list(loser.get("aliases") or [])
            if loser_surface:
                incoming_aliases = [loser_surface, *incoming_aliases]
            merged_aliases = _union_preserve_order(
                merged_aliases, incoming_aliases
            )
            merged_properties.update(dict(loser.get("properties") or {}))

        # Winner's own properties overlay last so they win on key conflict.
        merged_properties.update(dict(winner.get("properties") or {}))
        # Never let the winner alias itself.
        winner_name = winner.get("canonical_name")
        merged_aliases = [a for a in merged_aliases if a and a != winner_name]

        try:
            await execute_query(
                "UPDATE type::thing($id) SET "
                "confidence = $confidence, "
                "source_documents = $source_documents, "
                "type_tags = $type_tags, "
                "provenance_chain = $provenance_chain, "
                "properties = $properties, "
                "aliases = $aliases, "
                "updated_at = time::now();",
                {
                    "id": rid,
                    "confidence": merged_confidence,
                    "source_documents": merged_sources,
                    "type_tags": merged_type_tags,
                    "provenance_chain": merged_provenance,
                    "properties": merged_properties,
                    "aliases": merged_aliases,
                },
                self.config,
            )
        except Exception as e:
            logger.error(f"merge_into_winner update '{winner_id}': {e}")
            return False
        return True

    async def register_alias(
        self,
        canonical_entity_id: str,
        alias_text: str,
        match_type: str,
        similarity_score: float,
        method: str = "",
    ) -> bool:
        """Register a new alias for a canonical entity.

        Creates a mapping from the given alias text to the specified
        canonical entity. If the alias already exists for this entity,
        the operation is skipped and True is returned.

        Args:
            canonical_entity_id: The record ID of the canonical entity
                (e.g. "entity:abc123").
            alias_text: The alias text to register.
            match_type: How the match was determined (e.g. "exact",
                "fuzzy", "embedding").
            similarity_score: Confidence score of the match (0.0 to 1.0).
            method: Optional description of the resolution method used.

        Returns:
            True if the alias was registered or already exists, False on
            failure.
        """
        if not canonical_entity_id or not alias_text:
            return False

        try:
            entity_id = ensure_record_id(canonical_entity_id)

            # Check if alias already exists for this entity
            existing = await execute_query(
                "SELECT id FROM entity_alias "
                "WHERE canonical_entity = $entity_id "
                "AND alias_text = $alias_text LIMIT 1",
                {"entity_id": entity_id, "alias_text": alias_text},
                self.config,
            )
            if existing:
                return True

            # Insert the new alias
            await execute_query(
                "INSERT INTO entity_alias { "
                "canonical_entity: $entity_id, "
                "alias_text: $alias_text, "
                "match_type: $match_type, "
                "similarity_score: $similarity_score, "
                "method: $method, "
                "verified: false "
                "}",
                {
                    "entity_id": entity_id,
                    "alias_text": alias_text,
                    "match_type": match_type,
                    "similarity_score": similarity_score,
                    "method": method,
                },
                self.config,
            )
            return True
        except Exception as e:
            logger.error(
                f"Failed to register alias '{alias_text}' "
                f"for entity '{canonical_entity_id}': {e}"
            )
            return False

    async def list_entities(
        self,
        limit: int = 50,
        offset: int = 0,
        entity_type: Optional[str] = None,
        status: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List entities with pagination and optional type/status/source filters.

        The projection surfaces the triage fields the Q.5 KG table renders:
        ``status`` and ``manual_override`` (operator-pin state) plus
        ``source_documents`` so the caller can derive a cross-document count. It
        also carries ``canonical_name`` and ``primary_type`` so a source-scoped
        view can show the resolved name/type alongside the raw ``entity_type``.
        Effective structural degree is NOT computed here (it needs the relation
        table + the triage predicate config) — the service layer joins it on.

        Args:
            limit: Maximum number of entities to return.
            offset: Number of entities to skip.
            entity_type: Optional entity type filter.
            status: Optional triage-status filter (``"active"`` / ``"reference"``
                / ``"archived"``). Backed by ``idx_entity_status`` (migration 39),
                so the filtered scan is cheap. ``None`` returns all statuses.
            source_id: Optional source-scoped filter. When set, only entities
                whose ``source_documents`` array contains this source id are
                returned. ``source_documents`` stores STRINGIFIED record ids
                (see :meth:`upsert_entity`), so the value is compared in string
                space. ``None`` returns entities from every source (unchanged).

        Returns:
            A list of entity dictionaries.
        """
        # NOTE: ``WITH NOINDEX`` is required here. ``entity.name`` is covered by
        # the full-text SEARCH index ``idx_entity_fulltext`` (name, description).
        # In SurrealDB v2 the query planner cannot satisfy ``ORDER BY name`` from
        # a SEARCH index and aborts the transaction with "No iterator has been
        # found." Forcing a no-index scan + in-memory sort avoids that planner
        # path. Entity tables are small enough that the full scan is acceptable.
        projection = (
            "SELECT id, name, canonical_name, entity_type, primary_type, "
            "weight, confidence, status, manual_override, source_documents "
            "FROM entity WITH NOINDEX"
        )
        clauses: List[str] = []
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if entity_type:
            clauses.append("entity_type = $entity_type")
            params["entity_type"] = entity_type
        if status:
            clauses.append("status = $status")
            params["status"] = status
        if source_id:
            clauses.append("source_documents CONTAINS $source_id")
            params["source_id"] = str(source_id)
            # Source-scoped views show CURRENT entities; hide the archived/merged
            # history of earlier re-extractions unless a status is asked for.
            if not status:
                clauses.append("(status = 'active' OR status = 'reference')")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        try:
            return await execute_query(
                f"{projection}{where} "
                "ORDER BY name LIMIT $limit START $offset",
                params,
                self.config,
            )
        except Exception as e:
            logger.error(f"Failed to list entities: {e}")
            return []

    async def count_entities(
        self,
        entity_type: Optional[str] = None,
        status: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> int:
        """Count entities with optional type/status/source filters.

        Args:
            entity_type: Optional entity type filter.
            status: Optional triage-status filter (mirrors
                :meth:`list_entities`) so the paginated total matches the
                filtered page.
            source_id: Optional source-scoped filter (mirrors
                :meth:`list_entities`). Compared in string space against the
                stringified ids in ``source_documents``.

        Returns:
            Total count of matching entities.
        """
        clauses: List[str] = []
        params: Dict[str, Any] = {}
        if entity_type:
            clauses.append("entity_type = $entity_type")
            params["entity_type"] = entity_type
        if status:
            clauses.append("status = $status")
            params["status"] = status
        if source_id:
            clauses.append("source_documents CONTAINS $source_id")
            params["source_id"] = str(source_id)
            # Source-scoped views show CURRENT entities; hide the archived/merged
            # history of earlier re-extractions unless a status is asked for.
            if not status:
                clauses.append("(status = 'active' OR status = 'reference')")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        try:
            result = await execute_query(
                f"SELECT count() AS total FROM entity{where} GROUP ALL",
                params,
                self.config,
            )
            if result:
                return result[0].get("total", 0)
            return 0
        except Exception as e:
            logger.error(f"Failed to count entities: {e}")
            return 0

    async def get_entity_detail(
        self, entity_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a single entity with its relations.

        Args:
            entity_id: The entity record ID.

        Returns:
            Entity dict with a ``relations`` list, or None.
        """
        if not entity_id:
            return None
        try:
            eid = ensure_record_id(entity_id)
            result = await execute_query(
                "SELECT * FROM entity WHERE id = $id",
                {"id": eid},
                self.config,
            )
            if not result:
                return None

            entity = result[0]

            # Get relations where this entity is source or target. Confidence is
            # surfaced so the KG UI can render a per-relation confidence bar
            # (B.4 frontend).
            relations = await execute_query(
                "SELECT id, in AS source, out AS target, relation_type, confidence "
                "FROM relation WHERE in = $id OR out = $id",
                {"id": eid},
                self.config,
            )
            entity["relations"] = relations or []
            return entity
        except Exception as e:
            logger.error(f"Failed to get entity detail '{entity_id}': {e}")
            return None

    async def get_entity_types_summary(self) -> List[Dict[str, Any]]:
        """Get entity counts grouped by type.

        Returns:
            List of dicts with ``entity_type`` and ``count`` keys.
        """
        try:
            return await execute_query(
                "SELECT entity_type, count() AS count "
                "FROM entity GROUP BY entity_type ORDER BY count DESC",
                {},
                self.config,
            )
        except Exception as e:
            logger.error(f"Failed to get entity types summary: {e}")
            return []

    async def search_entities(
        self, query: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search entities by name using string containment.

        Args:
            query: Search text.
            limit: Maximum results.

        Returns:
            Matching entity dicts.
        """
        try:
            return await execute_query(
                "SELECT id, name, entity_type, weight, confidence "
                "FROM entity WHERE string::contains(string::lowercase(name), "
                "string::lowercase($query)) LIMIT $limit",
                {"query": query, "limit": limit},
                self.config,
            )
        except Exception as e:
            logger.error(f"Failed to search entities for '{query}': {e}")
            return []

    async def get_all_entities_and_relations(
        self,
        entity_type: Optional[str] = None,
        limit: int = 500,
    ) -> Dict[str, Any]:
        """Get raw nodes and edges for graph visualization.

        Edge-first + active-only selection. The previous implementation
        selected an arbitrary ``LIMIT`` slice of *all* entities (archived
        and merged included) and then asked for relations whose *both*
        endpoints happened to fall inside that arbitrary slice. With a
        large KG (~1.2k entities / ~0.9k relations) the odds of both
        endpoints landing in the same arbitrary 500 are ~0, so the
        endpoint returned **0 edges**.

        This rewrite picks the *active* relation core first and derives
        the node set from the relation endpoints, which guarantees every
        returned edge has both endpoints present:

        1. Fetch active relations (``status='active'``) among entities
           that are part of the live KG (``status IN ['active',
           'reference']`` — archived / merged are excluded). When
           ``entity_type`` is given, both endpoints must match the type.
        2. The node set is the union of the relation endpoints, padded
           with the highest-``weight`` active/reference entities (so an
           edge-less but important node still shows, and so the graph is
           non-empty for a KG with no relations yet).

        Args:
            entity_type: Optional type filter (applied to nodes *and*
                both endpoints of every edge).
            limit: Soft cap on the number of nodes returned. Edge
                endpoints are always kept (so the visible edges render);
                the remainder of the budget is filled with the
                highest-weight active entities.

        Returns:
            Dict with ``nodes`` and ``edges`` lists. ``nodes`` carry
            ``id``/``name``/``entity_type``/``weight``; ``edges`` carry
            ``source``/``target``/``relation_type``.
        """
        try:
            relation_limit = max(limit, 500)
            if entity_type:
                edges = await execute_query(
                    "SELECT id, in AS source, out AS target, relation_type "
                    "FROM relation "
                    "WHERE status = 'active' "
                    "AND in.status IN ['active', 'reference'] "
                    "AND out.status IN ['active', 'reference'] "
                    "AND in.entity_type = $entity_type "
                    "AND out.entity_type = $entity_type "
                    "LIMIT $relation_limit",
                    {"entity_type": entity_type, "relation_limit": relation_limit},
                    self.config,
                )
            else:
                edges = await execute_query(
                    "SELECT id, in AS source, out AS target, relation_type "
                    "FROM relation "
                    "WHERE status = 'active' "
                    "AND in.status IN ['active', 'reference'] "
                    "AND out.status IN ['active', 'reference'] "
                    "LIMIT $relation_limit",
                    {"relation_limit": relation_limit},
                    self.config,
                )
            edges = edges or []

            # Endpoint ids must always be in the node set so their edges
            # render. SurrealDB returns record ids as RecordID objects;
            # normalise to strings for de-duplication and the INSIDE probe.
            endpoint_ids: List[str] = []
            seen_endpoints = set()
            for edge in edges:
                for key in ("source", "target"):
                    raw = edge.get(key)
                    if raw is None:
                        continue
                    sid = str(raw)
                    if sid not in seen_endpoints:
                        seen_endpoints.add(sid)
                        endpoint_ids.append(sid)

            # Fill the remaining node budget with the most important
            # (highest weight) active/reference entities so edge-less hubs
            # and KGs with no relations still render something.
            pad = max(limit - len(endpoint_ids), 0)
            pad_nodes: List[Dict[str, Any]] = []
            if pad > 0:
                if entity_type:
                    pad_nodes = await execute_query(
                        "SELECT id, name, entity_type, weight "
                        "FROM entity "
                        "WHERE status IN ['active', 'reference'] "
                        "AND entity_type = $entity_type "
                        "ORDER BY weight DESC LIMIT $pad",
                        {"entity_type": entity_type, "pad": pad},
                        self.config,
                    )
                else:
                    pad_nodes = await execute_query(
                        "SELECT id, name, entity_type, weight "
                        "FROM entity "
                        "WHERE status IN ['active', 'reference'] "
                        "ORDER BY weight DESC LIMIT $pad",
                        {"pad": pad},
                        self.config,
                    )
                pad_nodes = pad_nodes or []

            # Hydrate the endpoint ids into full node rows. Endpoints came
            # from active relations whose endpoints were already filtered
            # to active/reference, so they are guaranteed live entities.
            endpoint_nodes: List[Dict[str, Any]] = []
            if endpoint_ids:
                endpoint_nodes = await execute_query(
                    "SELECT id, name, entity_type, weight "
                    "FROM entity WHERE id INSIDE $ids "
                    "ORDER BY weight DESC",
                    {"ids": endpoint_ids},
                    self.config,
                )
                endpoint_nodes = endpoint_nodes or []

            # Merge node sets, endpoints first, de-duplicated by id.
            nodes: List[Dict[str, Any]] = []
            seen_nodes = set()
            for node in [*endpoint_nodes, *pad_nodes]:
                nid = str(node.get("id"))
                if nid in seen_nodes:
                    continue
                seen_nodes.add(nid)
                nodes.append(node)

            return {"nodes": nodes, "edges": edges}
        except Exception as e:
            logger.error(f"Failed to get graph data: {e}")
            return {"nodes": [], "edges": []}

    async def list_orphans_for_source(
        self, source_id: str
    ) -> List[Dict[str, Any]]:
        """Return entities of this source with no incoming/outgoing relations.

        An "orphan" for the orphan-connector (B.5a) is an entity that:

        - lists *source_id* in its ``source_documents`` array (the
          provenance bag populated by ``upsert_entity``), AND
        - participates in zero rows of the ``relation`` RELATE table
          (no edge has ``in = entity.id`` and none has
          ``out = entity.id``).

        The query is implemented in two steps because SurrealDB's
        sub-select-count syntax for RELATE tables is awkward — we lift
        the per-entity edge probe to the Python side and reject any
        entity with non-zero degree. The probe uses ``LIMIT 1`` so each
        round-trip costs at most one row.

        Cross-source semantics (Minor-5, B.5a attempt 2):
            The edge-probe is GLOBAL — it considers every relation row
            regardless of which source produced it. An entity that
            appears in multiple sources is therefore "orphan" only when
            it has zero edges across ALL sources. This is intentional:
            a relation imported from source X already connects the
            entity to the graph, so re-running the orphan-connector for
            source Y has nothing to add.

        Args:
            source_id: Record ID of the source (e.g. ``"source:abc"``).

        Returns:
            A list of entity-row dicts (``id``, ``canonical_name``,
            ``entity_type``, ``source_documents``, ``properties``). Empty
            when the source has no entities or every entity is
            connected. Always shaped — never ``None``.
        """
        if not source_id:
            logger.warning(
                "list_orphans_for_source called with empty source_id"
            )
            return []

        try:
            entities = await execute_query(
                "SELECT id, canonical_name, entity_type, "
                "source_documents, properties "
                "FROM entity "
                "WHERE $source_id IN source_documents",
                {"source_id": source_id},
                self.config,
            )
        except Exception as e:
            logger.error(
                f"Failed to list source entities for orphan-detection "
                f"on '{source_id}': {e}"
            )
            return []

        if not entities:
            return []

        orphans: List[Dict[str, Any]] = []
        for entity in entities:
            eid = entity.get("id")
            if eid is None:
                continue
            try:
                edges = await execute_query(
                    "SELECT id FROM relation "
                    "WHERE in = $eid OR out = $eid LIMIT 1",
                    {"eid": eid},
                    self.config,
                )
            except Exception as e:
                logger.error(
                    f"Failed to count edges for entity '{eid}' "
                    f"in orphan-detection: {e}"
                )
                continue
            if not edges:
                orphans.append(entity)

        logger.info(
            "list_orphans_for_source: source={source_id} "
            "total_entities={te} orphans={no}",
            source_id=source_id,
            te=len(entities),
            no=len(orphans),
        )
        return orphans

    async def list_orphans_with_status(
        self,
        notebook_id: str,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List entities in a notebook filtered by ``orphan_status``.

        Powers the per-notebook orphans dashboard (B.5b). An entity
        "belongs" to a notebook when one of its ``source_documents``
        was imported via that notebook. Because source ↔ notebook is
        stored on the ``reference`` RELATE edge (NOT a column on
        ``source`` -- see migration 1), the query is a two-step:

        1. Resolve every source RecordID linked to the notebook via the
           ``reference`` edge.
        2. SELECT entities whose ``source_documents`` array intersects
           that set AND ``orphan_status`` matches the requested status.

        The ``source_documents`` field stores stringified record ids
        (see ``EntityRepository.upsert_entity`` / ``ExtractedEntity``)
        rather than ``record<source>`` values. The fix vs the original
        broken query (B.5b attempt 1):

        * The source-list step now traverses ``reference`` -- the table
          ``source`` carries no ``notebook`` column, so the previous
          ``WHERE notebook = $notebook_id`` always returned 0 rows.
        * Source ids are stringified before being handed to the entity
          query so the ``ANYINSIDE`` comparison happens in string space
          (matches how ``source_documents`` actually stores its values).

        Args:
            notebook_id: Record ID of the notebook
                (e.g. ``"notebook:abc"``).
            status: Optional ``orphan_status`` filter. When ``None``,
                returns entities with ANY non-NONE status
                (``pending_reconnect`` or ``archived``). When supplied,
                returns only entities matching that exact value.

        Returns:
            A list of entity rows shaped for the dashboard:
            ``id``, ``canonical_name``, ``entity_type``,
            ``orphan_status``, ``reconnect_attempts``,
            ``first_orphaned_at``, ``last_reconnect_attempt_at``,
            ``source_documents``. Empty on failure or no matches.
        """
        if not notebook_id:
            logger.warning(
                "list_orphans_with_status called with empty notebook_id"
            )
            return []

        try:
            # The ``reference`` edge runs source -> notebook (migration
            # 1, line 54-56); ``in`` is the source RecordID, ``out`` is
            # the notebook RecordID. Same projection-style query used by
            # ``SourceRepository.list_sources`` for the per-notebook view.
            source_rows = await execute_query(
                "SELECT VALUE in FROM reference WHERE out = type::thing($notebook_id)",
                {"notebook_id": notebook_id},
                self.config,
            )
        except Exception as e:
            logger.error(
                f"Failed to list sources for orphan dashboard on "
                f"'{notebook_id}': {e}"
            )
            return []

        if not source_rows:
            return []

        # ``SELECT VALUE in`` returns a flat list of RecordID values;
        # stringify so the comparison below happens against the same
        # representation ``source_documents`` stores (see upsert_entity).
        source_ids = [str(row) for row in source_rows if row]
        if not source_ids:
            return []

        try:
            if status is not None:
                rows = await execute_query(
                    "SELECT id, canonical_name, entity_type, "
                    "orphan_status, reconnect_attempts, first_orphaned_at, "
                    "last_reconnect_attempt_at, source_documents "
                    "FROM entity "
                    "WHERE orphan_status = $status "
                    "AND source_documents ANYINSIDE $source_ids "
                    "ORDER BY first_orphaned_at DESC",
                    {"status": status, "source_ids": source_ids},
                    self.config,
                )
            else:
                rows = await execute_query(
                    "SELECT id, canonical_name, entity_type, "
                    "orphan_status, reconnect_attempts, first_orphaned_at, "
                    "last_reconnect_attempt_at, source_documents "
                    "FROM entity "
                    "WHERE orphan_status != NONE "
                    "AND source_documents ANYINSIDE $source_ids "
                    "ORDER BY first_orphaned_at DESC",
                    {"source_ids": source_ids},
                    self.config,
                )
        except Exception as e:
            logger.error(
                f"Failed to list orphans-with-status for notebook "
                f"'{notebook_id}' (status={status}): {e}"
            )
            return []
        return rows or []

    async def update_orphan_status(
        self,
        entity_id: str,
        status: Optional[str],
        *,
        increment_attempts: bool = False,
        set_first_orphaned_at: bool = False,
        set_last_reconnect_attempt_at: bool = False,
    ) -> bool:
        """Update an entity's orphan-prune lifecycle fields.

        Single helper that drives all three lifecycle transitions in
        B.5b (mark pending, retry attempt, archive). The flags let the
        caller compose the exact field-set rather than running multiple
        round-trips per transition.

        Args:
            entity_id: Record ID of the entity to update.
            status: New ``orphan_status`` value. ``None`` clears the
                field (entity returns to "healthy" state).
            increment_attempts: When ``True``, ``reconnect_attempts``
                is incremented atomically. Used on retry-attempt rows.
            set_first_orphaned_at: When ``True`` AND
                ``first_orphaned_at`` is currently NONE, sets it to
                ``time::now()``. Idempotent — never overwrites an
                existing timestamp.
            set_last_reconnect_attempt_at: When ``True``,
                ``last_reconnect_attempt_at`` is set to ``time::now()``.

        Returns:
            ``True`` when the update query ran successfully (even if
            no row matched), ``False`` on transport-level failure.
        """
        if not entity_id:
            logger.warning(
                "update_orphan_status called with empty entity_id"
            )
            return False

        try:
            rid = ensure_record_id(entity_id)
        except Exception as e:
            logger.error(
                f"update_orphan_status: invalid entity_id '{entity_id}': {e}"
            )
            return False

        # Build the SET clause dynamically so we only touch the fields
        # the caller asked for. Each branch is independently testable.
        set_clauses = ["orphan_status = $status"]
        params: Dict[str, Any] = {"id": rid, "status": status}
        if increment_attempts:
            # Truthy-coalesce ``reconnect_attempts OR 0`` -- guards
            # against the NONE-on-legacy-row case where the migration
            # ``DEFAULT 0`` did not apply (rare, but cheap to defend).
            # Note (attempt 2 Minor-2): an earlier comment claimed
            # ``math::max([..., 0])`` here; rewritten to match the
            # actual SurrealQL OR-fallback above.
            set_clauses.append(
                "reconnect_attempts = "
                "(reconnect_attempts OR 0) + 1"
            )
        if set_first_orphaned_at:
            # Idempotent: only writes when the field is currently NONE.
            set_clauses.append(
                "first_orphaned_at = IF first_orphaned_at = NONE "
                "THEN time::now() ELSE first_orphaned_at END"
            )
        if set_last_reconnect_attempt_at:
            set_clauses.append(
                "last_reconnect_attempt_at = time::now()"
            )

        query = (
            "UPDATE type::thing($id) SET "
            + ", ".join(set_clauses)
            + ";"
        )

        try:
            await execute_query(query, params, self.config)
        except Exception as e:
            logger.error(
                f"update_orphan_status failed for '{entity_id}' "
                f"(status={status}): {e}"
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Track Q Phase Q.2 — effective structural degree (signals)
    # ------------------------------------------------------------------

    async def effective_degree_for_entities(
        self,
        ids: List[str],
        structural_predicates: List[str],
        weak_predicates: List[str],
        weak_promotion_min_docs: int,
    ) -> Dict[str, int]:
        """Batch-count each entity's EFFECTIVE structural degree (Q.2).

        Effective degree = number of *active* relation edges incident on the
        entity (either direction) that either:

        * carry a **structural** predicate (count from a single document), OR
        * carry a **weak** predicate (``RELATED``) that has PROMOTED — i.e. its
          cumulative ``source_documents`` (unioned per edge by Q.2a) has reached
          ``weak_promotion_min_docs`` distinct documents.

        A single-doc weak edge does NOT count; a multi-doc weak edge DOES. The
        promotion threshold is applied **in the query** via
        ``array::len(source_documents) >= $min_docs`` so a stray single-doc
        ``RELATED`` can never inflate the degree client-side.

        One round-trip for the whole ``ids`` batch (NOT one query per node):
        a single SELECT pulls every counting edge incident on any id in the
        batch, then the per-id tally is summed in Python. An edge is counted
        once for EACH of its endpoints that is in ``ids`` (Q.2 AC5: a structural
        or promoted-weak edge counts toward both endpoints' degree); a self-loop
        on a single batched id therefore contributes 1 to that id's degree.

        The predicate lists and threshold are passed in by the caller (read from
        the Q.1 triage config) — this repo helper never hardcodes them.

        Edges whose predicate is neither explicitly structural nor explicitly
        weak default to **weak** (matching the Q.1 config's
        ``_default_for_unmapped_predicate``): the SQL counts an edge when it is
        structural OR (NOT structural AND promoted), so an unmapped predicate
        recurring across enough documents promotes exactly like ``RELATED``.
        ``weak_predicates`` is accepted for interface symmetry / future
        weak-only signals but is not needed to gate promotion (NOT-structural is
        the weak set).

        Args:
            ids: Entity record-id strings to score.
            structural_predicates: Predicates whose edges count unconditionally.
            weak_predicates: Explicit weak predicates (informational; promotion
                is gated by NOT-structural, see above).
            weak_promotion_min_docs: Distinct-doc count a weak edge needs to
                promote (``>=`` is the promotion test).

        Returns:
            ``{id: effective_degree}`` for every id in ``ids`` (ids with no
            counting edge map to ``0``). Empty dict on failure or empty input.
        """
        if not ids:
            return {}

        # ``ids`` are DB record ids (no user input); coerce to record links so
        # the ``INSIDE`` / endpoint comparisons run in record space, and keep a
        # string set for the Python-side tally (endpoints round-trip as strings).
        id_set = {str(i) for i in ids}
        record_ids = [ensure_record_id(i) for i in ids]

        # SQL-side promotion filter: an edge counts when it is structural OR
        # (NOT structural AND its source_documents has reached the promotion
        # threshold). Treating "weak" as "not structural" means an unmapped
        # predicate (config-default weak) promotes too. ``in INSIDE $ids OR out
        # INSIDE $ids`` restricts the scan to edges touching the batch in one
        # round-trip. ``source_documents ?? []`` guards the (theoretical) NONE
        # case so ``array::len`` never errors.
        query = (
            "SELECT in, out, relation_type FROM relation "
            "WHERE status = 'active' "
            "AND (in INSIDE $ids OR out INSIDE $ids) "
            "AND ("
            "  relation_type INSIDE $structural "
            "  OR (relation_type NOT IN $structural "
            "      AND array::len(source_documents ?? []) >= $min_docs)"
            ")"
        )
        try:
            rows = await execute_query(
                query,
                {
                    "ids": record_ids,
                    "structural": list(structural_predicates),
                    "min_docs": weak_promotion_min_docs,
                },
                self.config,
            )
        except Exception as e:
            logger.error(f"effective_degree_for_entities failed: {e}")
            return {}

        degree: Dict[str, int] = {i: 0 for i in id_set}
        for row in rows or []:
            src = str(row.get("in"))
            tgt = str(row.get("out"))
            # Count once per batched endpoint. A self-loop (src == tgt, both in
            # the batch) is the same id twice → contributes 1 (set semantics).
            for endpoint in {src, tgt}:
                if endpoint in degree:
                    degree[endpoint] += 1
        return degree

    async def triage_rows_for_entities(
        self, ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Batch-load the triage-relevant projection for ``ids`` (Q.4).

        The triage pipeline needs, per batch entity: ``id`` (the affected key),
        ``primary_type`` (tier lookup), ``canonical_name`` / ``entity_type`` (the
        queue read-model), ``status`` + ``manual_override`` (status-assignment
        guard), and ``source_documents`` (doc-count signal). This pulls all of
        them in ONE round-trip rather than ``get_entity`` per id, so the hook
        stays a constant number of DB calls regardless of batch size.

        Args:
            ids: Entity record-id strings.

        Returns:
            One row dict per matched id (RecordIDs already stringified by
            ``execute_query``). Ids with no row are simply absent. Empty list on
            empty input or failure (the caller logs-and-continues — triage is
            non-blocking).
        """
        if not ids:
            return []
        record_ids = [ensure_record_id(i) for i in ids]
        try:
            return await execute_query(
                "SELECT id, canonical_name, entity_type, primary_type, "
                "status, manual_override, source_documents "
                "FROM entity WHERE id INSIDE $ids",
                {"ids": record_ids},
                self.config,
            )
        except Exception as e:
            logger.error(f"triage_rows_for_entities failed: {e}")
            return []

    async def relations_for_entities(
        self, ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Batch-load active relation edges incident on ``ids`` (Q.4 backbone).

        The backbone check (B6) needs, per affected well-connected node, every
        edge touching it with its ``relation_type`` and per-edge
        ``source_documents`` so it can flag those resting on a WEAK predicate.
        One round-trip pulls all edges incident on the batch (either endpoint);
        the caller partitions by node and applies the config predicate-strength
        at read-time (NO schema change — predicate strength is config, not DB).

        Args:
            ids: Entity record-id strings (the affected, degree>=threshold set).

        Returns:
            Edge rows with ``in`` / ``out`` (stringified endpoints),
            ``relation_type`` and ``source_documents``. Empty on empty/failure.
        """
        if not ids:
            return []
        record_ids = [ensure_record_id(i) for i in ids]
        try:
            return await execute_query(
                "SELECT in, out, relation_type, source_documents "
                "FROM relation WHERE status = 'active' "
                "AND (in INSIDE $ids OR out INSIDE $ids)",
                {"ids": record_ids},
                self.config,
            )
        except Exception as e:
            logger.error(f"relations_for_entities failed: {e}")
            return []

    async def get_entity_with_embedding(
        self, entity_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a single entity with its embedding vector.

        Args:
            entity_id: The record ID of the entity to retrieve
                (e.g. "entity:abc123").

        Returns:
            A dictionary with keys ``id``, ``name``, ``entity_type``, and
            ``embedding``, or None if the entity is not found.
        """
        if not entity_id:
            return None

        try:
            result = await execute_query(
                "SELECT id, name, entity_type, embedding "
                "FROM entity WHERE id = $id",
                {"id": ensure_record_id(entity_id)},
                self.config,
            )
            if result:
                return result[0]
            return None
        except Exception as e:
            logger.error(
                f"Failed to get entity with embedding '{entity_id}': {e}"
            )
            return None

    # ------------------------------------------------------------------
    # Track D Phase D.0 — notebook-scoped export projections
    # ------------------------------------------------------------------
    #
    # ``list_entities_for_notebook`` / ``list_relations_for_notebook``
    # power every Track D exporter (D.1 Obsidian, D.2 JSONL, D.3 NetworkX).
    # They reuse the notebook→source→entity traversal proven by
    # ``list_orphans_with_status`` (line 688) but project away the
    # ``embedding`` field per Q-D-1 to bound memory — 10K entities * 768
    # floats * 8 bytes ≈ 60MB the exporter never needs.

    # Field list mirrors the Entity model **minus** ``embedding``. Kept
    # as a module-private constant so both list_* methods (and any future
    # export-side projection) share the same projection.
    _ENTITY_EXPORT_FIELDS = (
        "id, created_at, updated_at, canonical_name, entity_type, "
        "description, source_documents, extracted_at, extraction_method, "
        "confidence, provenance_chain, properties, "
        "pagerank, betweenness, community_id, status, merged_into, "
        "type_tags, primary_type, "
        # K.4 vocabulary reconciliation. The Track D exporters call
        # resolve_external_ids(entity) (obsidian_export_service.py:1049),
        # which reads entity.external_ids; aliases ride along so the
        # reconciler-populated alt-names survive into exports too. Without
        # these in the projection the rehydrated Entity defaults both to []
        # and every reconciled URI is silently dropped on the export path.
        "external_ids, aliases, "
        # B.5b orphan-prune lifecycle. Selected so the WHERE filter can
        # see them; Entity.model_validate silently ignores extras.
        "orphan_status, reconnect_attempts, first_orphaned_at, "
        "last_reconnect_attempt_at"
    )

    async def list_entities_for_notebook(
        self,
        notebook_id: str,
        filter: ExportFilter,
    ) -> List[Entity]:
        """Notebook-scoped projection of the full Track-B entity field set.

        Reuses the reference RELATE edge pattern proven by
        ``list_orphans_with_status`` (line 688) for the
        notebook→source→entity traversal: source ↔ notebook lives on the
        ``reference`` RELATE edge (migration 1), so we resolve the
        notebook's source-ids first and then filter entities whose
        ``source_documents`` array intersects that set.

        Projects away the ``embedding`` field per Q-D-1 (Track D plan)
        to bound memory at scale — 10K entities × 768-float embeddings
        × 8 bytes ≈ 60MB that Obsidian/JSONL/NetworkX exporters never
        need. The Entity Pydantic model defaults ``embedding=[]`` so the
        rehydration is type-safe even without the column.

        Filters applied in SurrealQL (NOT in Python — keep the projection
        cheap when the notebook is large):

        * ``filter.min_confidence`` → ``confidence >= $min_confidence``
        * ``filter.entity_types`` → ``primary_type INSIDE $entity_types``
          (B.1a multi-type tagging: the "best" type is the gating one)
        * ``filter.include_orphans = False`` →
          ``orphan_status != 'pending_reconnect'``
        * ``filter.include_archived = False`` →
          ``orphan_status != 'archived'``

        ``min_connections`` is NOT enforced in this method -- counting
        an entity's degree across the ``relation`` table is a join the
        export pipeline does once via ``list_relations_for_notebook``.
        Callers (D.1/D.2/D.3) post-filter on degree there.

        Args:
            notebook_id: Record ID of the notebook (e.g. ``"notebook:abc"``).
            filter: ``ExportFilter`` from ``shared.models.export``.

        Returns:
            List of ``Entity`` rows matching the filter. Empty on
            failure, empty notebook, or fully-filtered output.
        """
        if not notebook_id:
            logger.warning(
                "list_entities_for_notebook called with empty notebook_id"
            )
            return []

        # Step 1: resolve notebook → sources via the reference edge.
        # Identical pattern to ``list_orphans_with_status`` (line 744).
        try:
            source_rows = await execute_query(
                "SELECT VALUE in FROM reference "
                "WHERE out = type::thing($notebook_id)",
                {"notebook_id": notebook_id},
                self.config,
            )
        except Exception as e:
            logger.error(
                f"Failed to list sources for notebook export "
                f"'{notebook_id}': {e}"
            )
            return []

        if not source_rows:
            return []

        # ``source_documents`` stores stringified record ids (see
        # ``upsert_entity``), so compare in string space.
        source_ids = [str(row) for row in source_rows if row]
        if not source_ids:
            return []

        # Step 2: build the WHERE clause incrementally so the planner
        # sees the most-selective predicate (source-ids intersection)
        # first. Unbound knobs (None/include_*=True) don't emit a clause.
        clauses: List[str] = [
            "source_documents ANYINSIDE $source_ids",
            "confidence >= $min_confidence",
        ]
        params: Dict[str, Any] = {
            "source_ids": source_ids,
            "min_confidence": filter.min_confidence,
        }

        if filter.entity_types is not None:
            clauses.append("primary_type INSIDE $entity_types")
            params["entity_types"] = filter.entity_types

        if not filter.include_orphans:
            # ``!=`` returns true when the field is NONE (healthy row),
            # so the negation works cleanly without an explicit IS NONE
            # branch.
            clauses.append("orphan_status != 'pending_reconnect'")

        if not filter.include_archived:
            clauses.append("orphan_status != 'archived'")

        query = (
            f"SELECT {self._ENTITY_EXPORT_FIELDS} FROM entity "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY canonical_name"
        )

        try:
            rows = await execute_query(query, params, self.config)
        except Exception as e:
            logger.error(
                f"list_entities_for_notebook failed for notebook "
                f"'{notebook_id}': {e}"
            )
            return []

        if not rows:
            return []

        # ``execute_query`` already stringifies RecordIDs; Entity model
        # tolerates the extra orphan_* fields (no ``extra="forbid"``).
        entities: List[Entity] = []
        for row in rows:
            try:
                entities.append(Entity.model_validate(row))
            except Exception as e:
                logger.warning(
                    f"list_entities_for_notebook: skipping malformed "
                    f"row id={row.get('id')!r}: {e}"
                )
        return entities

    async def list_relations_for_notebook(
        self,
        notebook_id: str,
        filter: ExportFilter,
    ) -> List[Relation]:
        """Relations between entities in the notebook (after entity filter).

        Two-phase pattern:

        1. Resolve the notebook's entity-id set via
           ``list_entities_for_notebook(notebook_id, filter)``.
        2. SELECT relations where BOTH ``in`` AND ``out`` belong to that
           set, with relation-confidence filtered by
           ``min_relation_confidence`` (or ``min_confidence`` when the
           caller left it ``None`` -- the shared-knob fallback documented
           in ``ExportFilter``).

        Per Q-D-4 (Track D plan), relations whose target was filtered
        out are silently dropped. The two-phase set-intersection
        implements that: an edge into a filtered entity disappears
        because that entity isn't in the id set.

        Args:
            notebook_id: Record ID of the notebook.
            filter: ``ExportFilter`` -- the same instance passed to
                ``list_entities_for_notebook`` for consistency.

        Returns:
            List of ``Relation`` rows. Empty on failure, empty notebook,
            or zero edges among filtered entities.
        """
        entities = await self.list_entities_for_notebook(notebook_id, filter)
        if not entities:
            return []

        # ``in`` / ``out`` on the relation table are ``record<entity>``
        # columns. SurrealQL compares them in RecordID space -- passing
        # plain strings here causes ``INSIDE`` to return zero rows, which
        # silently drops every edge. Coerce via ``ensure_record_id``.
        entity_ids: List[Any] = []
        for e in entities:
            if not e.id:
                continue
            try:
                entity_ids.append(ensure_record_id(e.id))
            except Exception as exc:
                logger.warning(
                    f"list_relations_for_notebook: bad entity id "
                    f"{e.id!r}: {exc}"
                )
        if not entity_ids:
            return []

        # ``min_relation_confidence`` inherits from ``min_confidence``
        # when None -- single-knob ergonomics documented on the model.
        min_rel_conf = (
            filter.min_relation_confidence
            if filter.min_relation_confidence is not None
            else filter.min_confidence
        )

        try:
            rows = await execute_query(
                "SELECT id, in, out, relation_type, properties, "
                "source_documents, extracted_at, extraction_method, "
                "confidence, provenance_chain, status, created_at "
                "FROM relation "
                "WHERE in INSIDE $entity_ids "
                "AND out INSIDE $entity_ids "
                "AND confidence >= $min_rel_conf",
                {
                    "entity_ids": entity_ids,
                    "min_rel_conf": min_rel_conf,
                },
                self.config,
            )
        except Exception as e:
            logger.error(
                f"list_relations_for_notebook failed for notebook "
                f"'{notebook_id}': {e}"
            )
            return []

        if not rows:
            return []

        relations: List[Relation] = []
        for row in rows:
            try:
                # ``in``/``out`` come back as RecordIDs (already string-
                # ified by execute_query); Relation has alias fields so
                # the raw row works.
                relations.append(Relation.model_validate(row))
            except Exception as e:
                logger.warning(
                    f"list_relations_for_notebook: skipping malformed "
                    f"row id={row.get('id')!r}: {e}"
                )
        return relations

    async def audit_ambiguous_relation_endpoints(self) -> Dict[str, Any]:
        """READ-ONLY audit (K.7a): count relation endpoints that are ambiguous.

        An endpoint is *ambiguous* when its entity's ``canonical_name`` maps to
        more than one ACTIVE typed entity (a cross-type homograph). Under the
        conservative K.1 normalizer no such collisions exist, so this is
        expected to report ~0 — proving no existing relation was mis-attached
        before K.7a. It becomes the canary once the aggressive K.7b prefixes
        land: a non-zero count there means an endpoint genuinely needs the
        type-aware resolution K.7a provides.

        This NEVER mutates: it only SELECTs and aggregates in Python, then logs
        a summary. Returns the audit result for callers/tests.

        Returns:
            ``{"ambiguous_names": [...], "ambiguous_name_count": N,
            "affected_relation_count": M}`` where ``ambiguous_names`` lists the
            canonical names that resolve to >1 active typed entity, and
            ``affected_relation_count`` is the number of active relation edges
            whose ``in`` or ``out`` endpoint carries such a name.
        """
        # 1. Active canonical_names that map to >1 distinct entity_type.
        try:
            active = await execute_query(
                "SELECT canonical_name, entity_type FROM entity "
                "WHERE status = 'active';",
                None,
                self.config,
            )
        except Exception as e:
            logger.error(f"audit_ambiguous_relation_endpoints: entity scan failed: {e}")
            return {
                "ambiguous_names": [],
                "ambiguous_name_count": 0,
                "affected_relation_count": 0,
            }

        types_by_name: Dict[str, set] = {}
        for row in active or []:
            name = row.get("canonical_name")
            etype = row.get("entity_type")
            if not name:
                continue
            types_by_name.setdefault(name, set()).add(etype)
        ambiguous_names = sorted(
            name for name, types in types_by_name.items() if len(types) > 1
        )

        # 2. Count active relation edges whose endpoint carries an ambiguous
        #    name. Resolve endpoint names via the edge's in/out ids.
        affected = 0
        if ambiguous_names:
            try:
                edges = await execute_query(
                    "SELECT in.canonical_name AS src_name, "
                    "out.canonical_name AS tgt_name FROM relation "
                    "WHERE status = 'active';",
                    None,
                    self.config,
                )
            except Exception as e:
                logger.error(
                    f"audit_ambiguous_relation_endpoints: relation scan failed: {e}"
                )
                edges = []
            ambiguous_set = set(ambiguous_names)
            for edge in edges or []:
                if (
                    edge.get("src_name") in ambiguous_set
                    or edge.get("tgt_name") in ambiguous_set
                ):
                    affected += 1

        result = {
            "ambiguous_names": ambiguous_names,
            "ambiguous_name_count": len(ambiguous_names),
            "affected_relation_count": affected,
        }
        if ambiguous_names:
            logger.warning(
                "audit_ambiguous_relation_endpoints: {n} canonical name(s) map "
                "to >1 active typed entity, affecting {m} relation edge(s): {names}",
                n=len(ambiguous_names),
                m=affected,
                names=ambiguous_names[:20],
            )
        else:
            logger.info(
                "audit_ambiguous_relation_endpoints: 0 ambiguous endpoints "
                "(no cross-type homograph under the current normalizer)"
            )
        return result

    # ------------------------------------------------------------------
    # Track R Phase R.2 — KG-proximity retrieval signal
    # ------------------------------------------------------------------

    async def load_active_entity_source_map(self) -> List[Dict[str, Any]]:
        """Load the active entity → source mapping for KG-proximity scoring (R.2).

        Projects, for every ``status='active'`` canonical entity, the minimal
        fields the pure :func:`shared.retrieval.score_related_sources` scorer
        needs: ``id``, ``canonical_name``, ``entity_type`` (drives type
        salience) and ``source_documents`` (the provenance bag that yields both
        the source membership and the per-entity document frequency / IDF
        rarity). The scorer derives ``df`` itself from these bags, so NO grouping
        is done here — one flat scan, not the N+1 a per-source loop would incur.

        Archived / merged / reference entities are excluded at the query
        (``status='active'``), so the generic-bucket down-weight in the scorer
        is the only remaining filter on noise (``topic``/``other`` are kept but
        weighted low; they are not torn out — R.6 owns the projection trim).

        ``source_documents`` stores STRINGIFIED record ids (see
        :meth:`upsert_entity`); they are returned as-is so the scorer compares
        them in the same string space the rest of the codebase uses.

        Returns:
            A list of ``{"id", "canonical_name", "entity_type",
            "source_documents"}`` dicts. Empty on failure / no active entities.
        """
        try:
            return await execute_query(
                "SELECT id, canonical_name, entity_type, source_documents "
                "FROM entity WHERE status = 'active'",
                {},
                self.config,
            )
        except Exception as e:
            logger.error(f"load_active_entity_source_map failed: {e}")
            return []

    async def load_active_relations(self) -> List[Dict[str, Any]]:
        """Load all active relation edges for the R.2 1-hop expansion (optional).

        Only used when KG retrieval runs with relation expansion enabled (gated
        off by default in the service — the 1-hop path multiplies through the
        duplicate-predicate noise R.6 has not yet trimmed). Projects the edge
        endpoints + predicate so the pure scorer can build entity adjacency.

        Returns:
            A list of ``{"in", "out", "relation_type"}`` dicts (endpoints
            stringified by ``execute_query``). Empty on failure / no relations.
        """
        try:
            return await execute_query(
                "SELECT in, out, relation_type FROM relation "
                "WHERE status = 'active'",
                {},
                self.config,
            )
        except Exception as e:
            logger.error(f"load_active_relations failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Track U Phase U.2 — `mentions` (source → entity) edge projection
    # ------------------------------------------------------------------
    #
    # `mentions` is a DERIVED, regenerated view of ``entity.source_documents``
    # (migration 66 defines it ``TYPE RELATION FROM source TO entity``). These
    # methods are the DB seam the ``MentionsProjectionService`` composes into one
    # idempotent regenerate; the canonical entity/source rows are NEVER mutated —
    # only the `mentions` edge table is written. Re-running clears then recreates
    # so the edge set is byte-identical with no duplicates (the U.2 idempotency
    # contract).

    async def clear_mentions(self) -> int:
        """Delete every ``mentions`` edge (the clear half of regeneration).

        Returns the number of edges removed so the regenerator can report the
        before/after delta. A bare ``DELETE mentions`` is correct here precisely
        because the table is a regenerated projection — unlike migration 66
        (a schema repair that must preserve any healthy edges), the regenerator
        OWNS the edge set and rebuilds it from ``entity.source_documents`` on the
        next step, so clearing all of it is the intended, reversible operation.

        Returns:
            The count of edges deleted (0 on an already-empty table / failure).
        """
        try:
            before = await self.count_mentions()
            await execute_query("DELETE mentions;", {}, self.config)
            return before
        except Exception as e:
            logger.error(f"clear_mentions failed: {e}")
            return 0

    async def clear_mentions_for_source(self, source_id: str) -> int:
        """Delete only the ``mentions`` edges incident on one source (PL.3).

        The source-scoped half of the incremental refresh: where
        :meth:`clear_mentions` wipes the whole projection (the global
        regenerate), this removes ONLY the edges whose ``in`` endpoint is
        ``source_id``. The PL.3 auto-chain rebuilds just this source's edges
        after an extract, so it must not disturb the rest of the corpus's
        ``mentions`` — clearing all of them would orphan every other source's
        graph until the next global regenerate.

        Idempotent: a second call (after the rebuild) clears the freshly-written
        edges so a re-run yields the same edge set, no duplicates.

        Args:
            source_id: The ``source:...`` whose incident ``mentions`` edges are
                removed.

        Returns:
            The count of edges deleted (0 on a malformed id / failure / no edges).
        """
        if not source_id:
            return 0
        try:
            src = ensure_record_id(source_id)
        except Exception as e:
            logger.error(f"clear_mentions_for_source: bad id {source_id!r}: {e}")
            return 0
        try:
            before = await execute_query(
                "SELECT count() AS total FROM mentions WHERE in = $src GROUP ALL",
                {"src": src},
                self.config,
            )
            count = int(before[0].get("total", 0)) if before else 0
            await execute_query(
                "DELETE mentions WHERE in = $src;", {"src": src}, self.config
            )
            return count
        except Exception as e:
            logger.error(f"clear_mentions_for_source failed for {source_id}: {e}")
            return 0

    async def count_mentions(self) -> int:
        """Count rows in the ``mentions`` edge table."""
        try:
            result = await execute_query(
                "SELECT count() AS total FROM mentions GROUP ALL", {}, self.config
            )
            if result:
                return int(result[0].get("total", 0))
            return 0
        except Exception as e:
            logger.error(f"count_mentions failed: {e}")
            return 0

    async def relate_mention(
        self,
        source_id: str,
        entity_id: str,
        *,
        weight: float,
        concept_name: str = "",
        concept_type: str = "",
        document_frequency: int = 0,
    ) -> bool:
        """RELATE one ``source -> mentions -> entity`` edge with its weight.

        Mirrors the RELATE discipline of :meth:`repoint_relations`: the arrow
        positions need bare record-id literals (SurrealDB issue #4232 — params in
        the ``in``/``out`` slots are rejected), and the ids are ``source:<alnum>``
        / ``entity:<alnum>`` record ids straight from the DB (no user input), so
        interpolating them is safe; all metadata is bound as parameters.

        The edge data fields are the migration-66 shape: ``weight`` (the U.2
        projection weight) plus the per-edge "why" (``concept_name`` /
        ``concept_type`` / ``document_frequency``). ``projection_method`` /
        ``created_at`` take their migration-66 DEFAULTs.

        Args:
            source_id: The ``source:...`` endpoint (edge ``in``).
            entity_id: The ``entity:...`` endpoint (edge ``out``).
            weight: The projected edge weight (R.2 salience × rarity).
            concept_name: The shared concept's surface name (the "why").
            concept_type: The concept's unified entity type.
            document_frequency: The concept's df (distinct-source count).

        Returns:
            True on success; False when an endpoint id is missing or malformed.

        Raises:
            RuntimeError: On a transport-level / SurrealQL failure of the RELATE
                (mirrors the other write helpers in this repo). The regenerator
                wraps this call and counts a raised edge as ``failed`` rather
                than aborting the whole rebuild.
        """
        if not source_id or not entity_id:
            return False
        # Validate the ids by parsing them (rejects malformed input) but RELATE
        # needs the literal string form in the arrow positions.
        try:
            src = str(ensure_record_id(source_id))
            tgt = str(ensure_record_id(entity_id))
        except Exception as e:
            logger.error(
                f"relate_mention: bad id source={source_id!r} "
                f"entity={entity_id!r}: {e}"
            )
            return False
        try:
            await execute_query(
                f"RELATE {src}->mentions->{tgt} SET "
                "weight = $weight, concept_name = $concept_name, "
                "concept_type = $concept_type, "
                "document_frequency = $document_frequency;",
                {
                    "weight": float(weight),
                    "concept_name": concept_name or "",
                    "concept_type": concept_type or "",
                    "document_frequency": int(document_frequency),
                },
                self.config,
            )
            return True
        except Exception as e:
            logger.exception(
                f"relate_mention failed for {source_id}->{entity_id}: {e}"
            )
            raise

    async def load_mentions_edges(
        self,
        *,
        min_weight: float = 0.0,
        source_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load materialized ``mentions`` edges for viz / traversal / export (U.4).

        Returns the document↔entity edges with their endpoints and weight/why
        metadata. ``in`` / ``out`` are stringified record ids (``source:...`` /
        ``entity:...``) by ``execute_query``.

        Args:
            min_weight: Optional per-edge weight floor (default 0.0 — return all).
                The full graph is small; the caller (or a UI slider) can raise
                this to the 0.3 named-only preset for a clean overview.
            source_id: Optional scope to edges incident on a single source
                (the source-detail / single-document neighbourhood view).

        Returns:
            A list of ``{"id", "source", "target", "weight", "concept_name",
            "concept_type", "document_frequency"}`` dicts. Empty on failure.
        """
        clauses: List[str] = []
        params: Dict[str, Any] = {}
        if min_weight is not None and min_weight > 0.0:
            clauses.append("weight >= $min_weight")
            params["min_weight"] = float(min_weight)
        if source_id:
            try:
                params["src"] = ensure_record_id(source_id)
                clauses.append("in = $src")
            except Exception as e:
                logger.error(f"load_mentions_edges: bad source_id {source_id!r}: {e}")
                return []
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        try:
            return await execute_query(
                "SELECT id, in AS source, out AS target, weight, "
                "concept_name, concept_type, document_frequency "
                f"FROM mentions{where} ORDER BY weight DESC",
                params,
                self.config,
            )
        except Exception as e:
            logger.error(f"load_mentions_edges failed: {e}")
            return []
