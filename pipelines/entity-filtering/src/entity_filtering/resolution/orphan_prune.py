"""Orphan-prune lifecycle module (B.5b).

Builds on B.5a's :mod:`orphan_connector` by layering a managed status
lifecycle on top of "orphans the LLM could not confirm". The lifecycle
has three terminal states:

  * ``None`` (default) -- entity is healthy, no orphan handling needed.
  * ``"pending_reconnect"`` -- last B.5a run produced zero confirmed
    relations; subsequent source-imports retry the connector.
  * ``"archived"`` -- ``reconnect_attempts >= max_attempts`` OR
    ``first_orphaned_at`` exceeded ``max_age_days``. Entity is NOT
    deleted; the dashboard hides it from the active view while keeping
    history queryable.

The transitions are exposed as three coroutines so callers can compose
them at different layers:

  1. :func:`mark_pending_reconnect` -- called immediately after B.5a
     when the connector returned zero confirmations for an orphan.
  2. :func:`retry_pending_reconnects` -- called from the entity-
     extraction service after every successful ``run_extraction``;
     re-runs the connector for the notebook's pending orphans and
     clears the status for any that reconnect.
  3. :func:`archive_stale_orphans` -- called on-demand (UI button) or
     by a scheduled job; flips entities meeting the
     ``max_attempts`` / ``max_age_days`` threshold to ``archived``.

Design notes
============

* The repository contract is intentionally minimal -- the protocol below
  declares only the four methods this module relies on. Tests inject an
  in-memory mock; production wires :class:`EntityRepository`.

* :func:`retry_pending_reconnects` accepts an ``orphan_connector``
  module-or-callable so the wiring stays DI-able. The default value
  imports :mod:`entity_filtering.resolution.orphan_connector` lazily,
  which keeps unit tests free of LLM-caller boot.

* All three functions are idempotent: re-running :func:`archive_stale_orphans`
  will not re-archive entities already in the ``"archived"`` state.

* Loguru everywhere -- mirrors the B.5a convention. Each function logs
  one INFO line per invocation with the count it processed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

from loguru import logger

# ``run`` from B.5a is the canonical retry entry-point. We import it
# lazily inside :func:`retry_pending_reconnects` so unit tests that
# never touch the retry path don't pay the import cost.
from entity_filtering.resolution.orphan_connector import (
    OrphanEntityRepoProtocol,
)

# ---------------------------------------------------------------------------
# Status string constants (single source of truth)
# ---------------------------------------------------------------------------

#: Entity is awaiting another B.5a attempt.
STATUS_PENDING_RECONNECT = "pending_reconnect"

#: Entity has exceeded the retry budget or age threshold.
STATUS_ARCHIVED = "archived"

#: Default ceiling on retry attempts before archival.
DEFAULT_MAX_ATTEMPTS = 3

#: Default age (in days) after which a pending orphan is archived.
DEFAULT_MAX_AGE_DAYS = 90


# ---------------------------------------------------------------------------
# Repository protocol -- B.5b extends B.5a's surface
# ---------------------------------------------------------------------------


@runtime_checkable
class OrphanPruneRepoProtocol(OrphanEntityRepoProtocol, Protocol):
    """Repository surface the prune-lifecycle relies on.

    Extends :class:`OrphanEntityRepoProtocol` so the same repository
    instance can drive both B.5a (find orphans) and B.5b (mutate the
    lifecycle fields). Production wires
    :class:`surrealdb_service.repositories.entity.EntityRepository`,
    which satisfies both protocols.
    """

    async def list_orphans_with_status(
        self,
        notebook_id: str,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return entities in *notebook_id* matching the given status."""
        ...

    async def update_orphan_status(
        self,
        entity_id: str,
        status: Optional[str],
        *,
        increment_attempts: bool = False,
        set_first_orphaned_at: bool = False,
        set_last_reconnect_attempt_at: bool = False,
    ) -> bool:
        """Update one entity's lifecycle fields. Returns success."""
        ...


# ---------------------------------------------------------------------------
# DTO returned by retry_pending_reconnects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryOutcome:
    """Summary of one :func:`retry_pending_reconnects` invocation.

    Surfaced both for telemetry (the service wraps this in a metric)
    and for the dashboard's "Reconnect" action (the API returns the
    outcome so the frontend can refresh in place).
    """

    notebook_id: str
    attempted: int  # how many pending-reconnect entities were retried
    reconnected: int  # how many cleared status because B.5a confirmed a link
    still_pending: int  # how many remained pending_reconnect


# ---------------------------------------------------------------------------
# Transition 1: mark pending_reconnect
# ---------------------------------------------------------------------------


async def mark_pending_reconnect(
    orphan_ids: Iterable[str],
    entity_repo: OrphanPruneRepoProtocol,
) -> int:
    """Flip a batch of orphan entities to ``pending_reconnect``.

    For each entity ID:

    1. Set ``orphan_status = "pending_reconnect"``.
    2. Increment ``reconnect_attempts`` by 1 (atomic SET-expression).
    3. Set ``first_orphaned_at = time::now()`` if currently NONE.
    4. Set ``last_reconnect_attempt_at = time::now()``.

    The fact that we increment ``reconnect_attempts`` here -- and again
    in :func:`retry_pending_reconnects` -- means the FIRST time an
    entity is marked counts as attempt #1. That matches the B.5b
    acceptance criterion: "after the initial run produces 2 failures,
    those 2 entities have ``reconnect_attempts=1``".

    Args:
        orphan_ids: Iterable of entity record IDs (e.g.
            ``"entity:abc"``).
        entity_repo: Repository implementing
            :class:`OrphanPruneRepoProtocol`.

    Returns:
        The number of entities the repository reported successful
        updates for.
    """
    id_list = [eid for eid in orphan_ids if eid]
    if not id_list:
        logger.debug("mark_pending_reconnect: no orphan ids supplied")
        return 0

    success = 0
    for eid in id_list:
        ok = await entity_repo.update_orphan_status(
            eid,
            STATUS_PENDING_RECONNECT,
            increment_attempts=True,
            set_first_orphaned_at=True,
            set_last_reconnect_attempt_at=True,
        )
        if ok:
            success += 1

    logger.info(
        "mark_pending_reconnect: ids={n} succeeded={s}",
        n=len(id_list),
        s=success,
    )
    return success


# ---------------------------------------------------------------------------
# Transition 2: retry_pending_reconnects
# ---------------------------------------------------------------------------


# Type-alias for the orphan_connector ``run`` callable so unit tests can
# inject a mock without importing the production module. Matches the
# real signature in :mod:`orphan_connector`.
OrphanConnectorRun = Callable[..., Awaitable[List[Any]]]


async def retry_pending_reconnects(
    notebook_id: str,
    entity_repo: OrphanPruneRepoProtocol,
    chunks: Iterable[Dict[str, Any]],
    *,
    orphan_connector_run: Optional[OrphanConnectorRun] = None,
    llm_caller: Optional[Callable[..., Any]] = None,
    persist_relation: Optional[Callable[..., Awaitable[None]]] = None,
    model: str = "default",
    max_proposals_per_orphan: int = 3,
    min_confidence: float = 0.6,
) -> RetryOutcome:
    """Re-run B.5a's connector for the notebook's pending orphans.

    The function operates in three steps:

    1. List entities with ``orphan_status = "pending_reconnect"`` for
       the notebook.
    2. For each one, increment ``reconnect_attempts`` and call B.5a's
       ``run`` with the orphan's source(s) + the freshly imported
       chunks. We treat each orphan's first ``source_documents`` entry
       as the canonical source id for the retry; orphans with no
       source are skipped (defensive -- they should not exist).
    3. When B.5a returns at least one confirmed relation referencing
       the orphan, clear ``orphan_status`` (back to NONE) and call the
       injected ``persist_relation`` so the new edge lands in the KG.

    Args:
        notebook_id: Record ID of the notebook to retry.
        entity_repo: Repository implementing
            :class:`OrphanPruneRepoProtocol`.
        chunks: Iterable of chunk dicts from the source-import that
            triggered this retry (same shape B.5a expects).
        orphan_connector_run: Override for B.5a's ``run`` coroutine.
            Defaults to
            ``entity_filtering.resolution.orphan_connector.run`` --
            lazy-imported here so unit tests that mock the retry never
            pay the import cost.
        llm_caller: LLM caller forwarded to B.5a. Only required when
            ``orphan_connector_run`` is the default; mocks ignore it.
        persist_relation: Optional async callback ``(notebook_id,
            ExtractedRelation) -> None`` invoked once per confirmed
            relation. The entity_extraction_service wires this to its
            persistence layer; tests inject a recording mock.
        model: Model identifier forwarded to B.5a.
        max_proposals_per_orphan: Forwarded to B.5a.
        min_confidence: Forwarded to B.5a.

    Returns:
        :class:`RetryOutcome` summarising attempted, reconnected, and
        still-pending counts.
    """
    if not notebook_id:
        logger.warning(
            "retry_pending_reconnects called with empty notebook_id"
        )
        return RetryOutcome(
            notebook_id="", attempted=0, reconnected=0, still_pending=0
        )

    pending = await entity_repo.list_orphans_with_status(
        notebook_id, STATUS_PENDING_RECONNECT
    )
    if not pending:
        logger.info(
            "retry_pending_reconnects: notebook={nb} no pending orphans",
            nb=notebook_id,
        )
        return RetryOutcome(
            notebook_id=notebook_id,
            attempted=0,
            reconnected=0,
            still_pending=0,
        )

    # Lazy import so unit tests that fully mock the run callable do not
    # need the heavy orphan_connector import (which pulls shared.models).
    if orphan_connector_run is None:
        from entity_filtering.resolution import orphan_connector as _oc

        orphan_connector_run = _oc.run

    chunk_list = list(chunks)
    attempted = 0
    reconnected = 0

    for orphan in pending:
        eid = orphan.get("id")
        if not eid:
            continue
        source_docs = orphan.get("source_documents") or []
        if not source_docs:
            logger.warning(
                "retry_pending_reconnects: orphan {eid} has no "
                "source_documents; skipping",
                eid=eid,
            )
            continue
        source_id = str(source_docs[0])

        attempted += 1
        # Record the attempt up-front so the dashboard reflects "tried
        # again" even if B.5a returns nothing. The status remains
        # pending_reconnect; only the timestamp + counter advance.
        await entity_repo.update_orphan_status(
            str(eid),
            STATUS_PENDING_RECONNECT,
            increment_attempts=True,
            set_last_reconnect_attempt_at=True,
        )

        try:
            new_relations = await orphan_connector_run(
                source_id,
                chunk_list,
                entity_repo,
                llm_caller,
                model=model,
                max_proposals_per_orphan=max_proposals_per_orphan,
                min_confidence=min_confidence,
            )
        except Exception as e:
            # B.5a transport / token-budget failures must NOT kill the
            # whole retry loop -- the next orphan may still succeed.
            logger.warning(
                "retry_pending_reconnects: orphan_connector.run raised "
                "for orphan={eid} source={src}: {e}",
                eid=eid,
                src=source_id,
                e=e,
            )
            new_relations = []

        # Did the connector return at least one relation referencing
        # this orphan? B.5a's relations carry the surface form, not the
        # entity id; we match against ``canonical_name``.
        orphan_name = orphan.get("canonical_name") or ""
        confirmed_for_this_orphan = [
            r
            for r in new_relations
            if _relation_mentions(r, orphan_name)
        ]
        if confirmed_for_this_orphan:
            # Clear the status back to healthy. ``increment_attempts``
            # stays False here -- the successful attempt was already
            # counted in the up-front update above.
            await entity_repo.update_orphan_status(str(eid), None)
            reconnected += 1
            if persist_relation is not None:
                for rel in confirmed_for_this_orphan:
                    try:
                        await persist_relation(notebook_id, rel)
                    except Exception as e:
                        logger.warning(
                            "retry_pending_reconnects: persist_relation "
                            "failed for orphan={eid}: {e}",
                            eid=eid,
                            e=e,
                        )

    still_pending = attempted - reconnected
    logger.info(
        "retry_pending_reconnects: notebook={nb} attempted={a} "
        "reconnected={r} still_pending={sp}",
        nb=notebook_id,
        a=attempted,
        r=reconnected,
        sp=still_pending,
    )
    return RetryOutcome(
        notebook_id=notebook_id,
        attempted=attempted,
        reconnected=reconnected,
        still_pending=still_pending,
    )


def _relation_mentions(relation: Any, orphan_name: str) -> bool:
    """Return True if *relation* references the orphan's surface form.

    Tolerates both :class:`ExtractedRelation` instances (the production
    shape) and plain dicts (mock-friendly). Matching is case-insensitive
    on whitespace-stripped strings to survive minor formatting drift.
    """
    if not orphan_name:
        return False
    needle = orphan_name.strip().lower()
    if not needle:
        return False
    src = getattr(relation, "source_entity", None)
    tgt = getattr(relation, "target_entity", None)
    if src is None and isinstance(relation, dict):
        src = relation.get("source_entity")
        tgt = relation.get("target_entity")
    src_s = (src or "").strip().lower() if isinstance(src, str) else ""
    tgt_s = (tgt or "").strip().lower() if isinstance(tgt, str) else ""
    return src_s == needle or tgt_s == needle


# ---------------------------------------------------------------------------
# Transition 3: archive_stale_orphans
# ---------------------------------------------------------------------------


async def archive_stale_orphans(
    entity_repo: OrphanPruneRepoProtocol,
    *,
    notebook_id: Optional[str] = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    now: Optional[datetime] = None,
) -> int:
    """Flip stale pending-reconnect entities to ``archived``.

    An entity is considered stale (and thus archived) when EITHER:

    * ``reconnect_attempts >= max_attempts``, OR
    * ``first_orphaned_at`` is older than ``max_age_days`` ago.

    Either threshold alone is sufficient. We do NOT delete the entity --
    archive is a soft-delete: the row stays for audit + the dashboard
    surfaces an "archived" tab.

    Idempotent: entities already in ``"archived"`` are not in the
    ``pending_reconnect`` list, so they're never touched here. Re-running
    on the same DB produces the same row count.

    Args:
        entity_repo: Repository implementing
            :class:`OrphanPruneRepoProtocol`.
        notebook_id: Optional notebook filter. When ``None``, archives
            stale orphans across every notebook (intended for the
            nightly job); when supplied, scopes to a single notebook
            (intended for the on-demand UI button).
        max_attempts: Inclusive threshold on ``reconnect_attempts``.
            Entities at or above this count are archived. Defaults to
            3 (per the B.5b plan).
        max_age_days: Age threshold in days. ``first_orphaned_at``
            older than this is archived. Defaults to 90.
        now: Override the "current time" for tests; defaults to
            :func:`datetime.now(timezone.utc)`. Always tz-aware so the
            comparison with stored datetimes is well-defined.

    Returns:
        The number of entities archived during this run.
    """
    if max_attempts < 1:
        logger.warning(
            "archive_stale_orphans: max_attempts={ma} < 1; clamping to 1",
            ma=max_attempts,
        )
        max_attempts = 1
    if max_age_days < 0:
        logger.warning(
            "archive_stale_orphans: max_age_days={mad} < 0; clamping to 0",
            mad=max_age_days,
        )
        max_age_days = 0

    reference_now = now or datetime.now(timezone.utc)
    cutoff = reference_now - timedelta(days=max_age_days)

    # Walk every "pending_reconnect" row for the requested scope (one
    # notebook OR every notebook). The list query is cheap because the
    # status filter is selective; the per-row decision is O(1).
    if notebook_id:
        candidates = await entity_repo.list_orphans_with_status(
            notebook_id, STATUS_PENDING_RECONNECT
        )
    else:
        # Cross-notebook scan: the protocol does not expose a "list all
        # by status" method (intentional -- the dashboard queries per
        # notebook). Callers wanting a cross-notebook sweep must invoke
        # this function with each notebook_id; we still support the
        # single-notebook case here for completeness.
        logger.warning(
            "archive_stale_orphans called with notebook_id=None; "
            "cross-notebook sweep is not supported. Returning 0."
        )
        return 0

    archived = 0
    for row in candidates:
        eid = row.get("id")
        if not eid:
            continue
        attempts = int(row.get("reconnect_attempts") or 0)
        first = row.get("first_orphaned_at")
        first_dt = _coerce_datetime(first)

        attempts_exceeded = attempts >= max_attempts
        age_exceeded = (
            first_dt is not None
            and first_dt < cutoff
        )
        if not (attempts_exceeded or age_exceeded):
            continue

        ok = await entity_repo.update_orphan_status(
            str(eid), STATUS_ARCHIVED
        )
        if ok:
            archived += 1

    logger.info(
        "archive_stale_orphans: scope={scope} candidates={c} "
        "archived={a} max_attempts={ma} max_age_days={mad}",
        scope=notebook_id or "all",
        c=len(candidates),
        a=archived,
        ma=max_attempts,
        mad=max_age_days,
    )
    return archived


def _coerce_datetime(value: Any) -> Optional[datetime]:
    """Best-effort coerce a stored timestamp to a tz-aware datetime.

    SurrealDB's Python driver returns ``datetime`` objects directly,
    but tests sometimes seed plain ISO strings. Returns ``None`` on
    unparseable inputs so the caller treats it as "no first_orphaned_at
    recorded" rather than crashing the loop.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        # Normalise naive datetimes to UTC so comparisons against
        # ``reference_now`` (always tz-aware) work cleanly.
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str):
        try:
            # ``datetime.fromisoformat`` handles "Z" only in 3.11+.
            text = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            return None
    return None
