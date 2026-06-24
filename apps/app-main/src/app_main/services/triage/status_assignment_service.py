"""Deterministic triage status assignment (Track Q.3).

Turns a per-entity signal triple — ``tier`` x ``effective_degree`` x
``manual_override`` — into a :class:`StatusDecision` (``status`` / ``reason`` /
``queue_flags``) via the rules the operator encoded in
``config/triage_config.json``'s ``status_rules``. The mapping is a PURE function:
no I/O, no clock, no DB. That purity is what makes the status idempotent under
re-runs — the same inputs always yield the same decision, so re-triaging an
unchanged entity is a guaranteed no-op.

The load-bearing constraint of the whole track lives here:
``manual_override == true`` ALWAYS wins. When an operator has pinned a status by
hand, :meth:`StatusAssignmentService.assign` returns their status untouched and
:meth:`StatusAssignmentService.apply` writes nothing and logs nothing — the
automation never overwrites an operator decision, ever, on any run.

Thresholds (``core_active_min_degree``, ``well_connected_threshold``) and the
tier vocabulary come from the Q.1 :class:`~app_main.services.triage.triage_config.TriageConfig`;
nothing here is hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger

from app_main.services.triage.status_change_log import StatusChangeLogRepository
from app_main.services.triage.triage_config import Tier, TriageConfig

# Resolved triage statuses. ``entity.status`` is a free string (migration 39, no
# enum ASSERT), so these are convention, not a DB constraint.
STATUS_ACTIVE = "active"
STATUS_REFERENCE = "reference"

# Review-queue flag strings. These are the EXACT strings the acceptance criteria
# assert and Q.4's queue keys on, so they are defined once here.
FLAG_ISOLATED_CORE = "isolated core actor / possible extraction gap"
FLAG_CANDIDATE_ACTIVE = "candidate active"
FLAG_UNSURE_REVIEW = "unsure — operator decides"


@dataclass(frozen=True)
class StatusDecision:
    """The outcome of assigning a status to one entity.

    Frozen because a decision is a value object: it is computed once from the
    signals and then either applied or discarded; it should never be mutated in
    place.

    Attributes:
        entity_id: Record ID of the entity the decision is about.
        status: The derived status (``active`` / ``reference``), or the
            operator's pinned status when ``manual_override`` is set.
        reason: Human-readable description of the rule that fired (logged on an
            actual change).
        queue_flags: Review-queue flags raised by this decision (Q.4 consumes
            them; here they are just returned). Empty when none apply.
        manual_override: Whether the entity is operator-pinned. When True,
            ``status`` echoes the operator's current status and :meth:`apply`
            is a pure no-op.
        current_status: The entity's status BEFORE this decision, used by
            :meth:`apply` to detect an actual change (and so to decide whether
            to write + log).
    """

    entity_id: str
    status: str
    reason: str
    queue_flags: List[str] = field(default_factory=list)
    manual_override: bool = False
    current_status: Optional[str] = None

    @property
    def changes_status(self) -> bool:
        """True iff applying this decision would actually change the status.

        Automation only writes + logs when this is True (and the entity is not
        operator-pinned), which is what makes a re-run idempotent.
        """
        return (not self.manual_override) and self.status != self.current_status


class StatusAssignmentService:
    """Derive and (optionally) persist triage statuses.

    :meth:`assign` is the pure decision function (no side effects). :meth:`apply`
    is the only side-effecting method: it writes the status via
    ``EntityRepository.set_status`` (which itself re-checks the override guard)
    and appends one :class:`StatusChangeLogRepository` row — but ONLY when the
    status genuinely changed and the entity is not operator-pinned.
    """

    def __init__(
        self,
        config: TriageConfig,
        entity_repository,
        status_change_log: Optional[StatusChangeLogRepository] = None,
    ) -> None:
        """Initialise the service.

        Args:
            config: The loaded Q.1 triage config (source of thresholds + tiers).
            entity_repository: An ``EntityRepository`` exposing ``set_status``.
            status_change_log: The audit-log repository; constructed with the
                repository's config when omitted.
        """
        self._config = config
        self._entities = entity_repository
        self._log = status_change_log or StatusChangeLogRepository(
            getattr(entity_repository, "config", None)
        )

    def assign(
        self,
        entity: object,
        tier: Tier,
        effective_degree: int,
    ) -> StatusDecision:
        """Derive a :class:`StatusDecision` from tier x degree x override.

        PURE: depends only on its arguments and the immutable config. No I/O.

        Args:
            entity: The entity (any object exposing ``id``/``status``/
                ``manual_override``, e.g. a shared ``Entity`` or a dict-like
                row). Read-only here.
            tier: The entity's tier from ``TriageConfig.tier_for`` —
                ``"active"`` / ``"reference"`` / ``"unsure_review"``.
            effective_degree: The Q.2 effective structural degree (structural
                edges + promoted weak edges).

        Returns:
            A :class:`StatusDecision`. When ``manual_override`` is set the
            decision echoes the operator's status with no queue flags; otherwise
            it follows the ``status_rules`` truth table.
        """
        entity_id = _attr(entity, "id")
        current_status = _attr(entity, "status")
        manual_override = bool(_attr(entity, "manual_override", False))

        # Rule 6 (highest precedence): an operator-pinned entity is returned
        # verbatim. Automation neither re-derives nor queues it.
        if manual_override:
            return StatusDecision(
                entity_id=entity_id,
                status=current_status,
                reason="manual_override — operator-pinned, automation skipped",
                queue_flags=[],
                manual_override=True,
                current_status=current_status,
            )

        status, reason, flags = self._derive(tier, effective_degree)
        return StatusDecision(
            entity_id=entity_id,
            status=status,
            reason=reason,
            queue_flags=flags,
            manual_override=False,
            current_status=current_status,
        )

    def _derive(
        self, tier: Tier, effective_degree: int
    ) -> tuple[str, str, List[str]]:
        """Apply the ``status_rules`` truth table (no override considered).

        Thresholds are read from the config, never hard-coded.
        """
        min_active = self._config.core_active_min_degree
        well_connected = self._config.well_connected_threshold

        if tier == "active":
            if effective_degree >= min_active:
                return (
                    STATUS_ACTIVE,
                    f"active-tier & degree {effective_degree} >= "
                    f"{min_active} -> active",
                    [],
                )
            # active-tier but isolated: still active, but flagged as a possible
            # extraction gap (a core actor with no edges is suspicious).
            return (
                STATUS_ACTIVE,
                f"active-tier & degree {effective_degree} < {min_active} "
                f"-> active (isolated)",
                [FLAG_ISOLATED_CORE],
            )

        if tier == "reference":
            if effective_degree >= well_connected:
                # Well-connected reference: stays reference but surfaced as a
                # candidate for promotion to active.
                return (
                    STATUS_REFERENCE,
                    f"reference-tier & degree {effective_degree} >= "
                    f"{well_connected} -> reference (candidate active)",
                    [FLAG_CANDIDATE_ACTIVE],
                )
            return (
                STATUS_REFERENCE,
                f"reference-tier & degree {effective_degree} -> reference",
                [],
            )

        # unsure_review (and any unexpected tier, defensively): reference by
        # default and ALWAYS queued for the operator.
        return (
            STATUS_REFERENCE,
            "unsure_review-tier -> reference (operator decides)",
            [FLAG_UNSURE_REVIEW],
        )

    async def apply(self, decision: StatusDecision, batch_id: str) -> bool:
        """Persist a decision's status and log the change — when it changed.

        The two side effects (write + log) are gated on the SAME condition,
        :attr:`StatusDecision.changes_status`:

        * ``manual_override`` set -> no write, no log (operator wins).
        * status unchanged -> no write, no log (idempotent re-run).
        * status changed -> ``set_status`` + ONE ``status_change_log`` row.

        ``set_status`` itself re-checks the override flag at the DB layer, so
        even a stale decision cannot overwrite a freshly-pinned entity.

        Args:
            decision: The decision from :meth:`assign`.
            batch_id: Identifier of the current triage batch (recorded in the
                log row so a whole run is queryable together).

        Returns:
            True if the status was written (and a log row appended); False if
            the apply was a no-op (override or unchanged).
        """
        if not decision.changes_status:
            return False

        wrote = await self._entities.set_status(
            decision.entity_id, decision.status, respect_override=True
        )
        if not wrote:
            # The DB-level override guard suppressed the write (the entity was
            # pinned after the decision was computed). Do NOT log a change that
            # did not happen.
            logger.debug(
                "apply: status write suppressed for {} (override at DB layer)",
                decision.entity_id,
            )
            return False

        await self._log.append(
            decision.entity_id,
            old=decision.current_status or "",
            new=decision.status,
            reason=decision.reason,
            batch_id=batch_id,
        )
        return True


def _attr(obj: object, name: str, default: object = None) -> object:
    """Read ``name`` from an object or a mapping, with a default.

    Triage callers may pass either a typed ``Entity`` (attribute access) or a
    raw DB row dict (key access); supporting both keeps the service decoupled
    from how the caller loaded the entity.
    """
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
