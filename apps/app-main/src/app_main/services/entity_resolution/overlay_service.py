"""User-overlay service for entity resolution (K.5).

The overlay is the human escape hatch over the automated fuzzy/embedding
matcher. Two rule kinds, each scoped ``global`` or ``notebook``:

* **force-split** ``(A, B)`` — A and B must NEVER be proposed as a merge, even
  above the auto-threshold. This is the ultimate over-merge backstop: a hard
  veto the matcher cannot beat (AC4).
* **force-merge** ``(A, B)`` — A and B are treated as the same entity (a hard
  include) that feeds the K.3 reviewable-merge machinery. A notebook-scoped
  force-merge applies ONLY within that notebook (AC5, per-notebook isolation).

Precedence (K-D4): most-specific wins. A ``notebook`` rule overrides a
conflicting ``global`` rule for that notebook. The pair-matching is
order-insensitive and **type-aware** — a rule only binds two entities OF THE
SAME ``entity_type``, so it can never bind a person to an org (the recurring
Track-K lesson).

The service also bridges force-merge rules into the K.2 ``alias_overrides`` DB
seam (``set_db_overlay``) so a user force-merge flows into ``normalize_entity_name``
for future ingests, not just the candidate proposer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from loguru import logger
from shared.config.alias_overrides import set_db_overlay
from shared.utils.name_normalizer import normalize_entity_name
from surrealdb_service.repositories.alias_overlay import AliasOverlayRepository


@dataclass(frozen=True)
class OverlayRule:
    """A resolved overlay rule (normalized pair, ready for matching)."""

    id: str
    kind: str  # "merge" | "split"
    scope: str  # "global" | "notebook"
    name_a: str
    name_b: str
    entity_type: str
    notebook_id: Optional[str] = None

    @property
    def pair(self) -> frozenset[str]:
        """Order-insensitive normalized name pair for matching."""
        return frozenset(
            {normalize_entity_name(self.name_a), normalize_entity_name(self.name_b)}
        )

    def matches(
        self, norm_a: str, norm_b: str, entity_type: str
    ) -> bool:
        """True when this rule binds the given normalized, typed pair.

        Type-aware: the rule's ``entity_type`` must match (an empty rule type is
        a wildcard for back-compat, but the two endpoints must still share their
        own type — enforced by the caller comparing same-typed entities only).
        """
        if self.entity_type and self.entity_type != entity_type:
            return False
        return self.pair == frozenset({norm_a, norm_b})


class OverlayService:
    """CRUD + resolution over the ``alias_overlay`` rules."""

    def __init__(self, repo: Optional[AliasOverlayRepository] = None) -> None:
        self.repo = repo or AliasOverlayRepository()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def add_rule(
        self,
        *,
        kind: str,
        name_a: str,
        name_b: str,
        entity_type: str = "",
        scope: str = "global",
        notebook_id: Optional[str] = None,
    ) -> str:
        """Create one overlay rule; returns its id. Validates kind/scope/names."""
        rid = await self.repo.create_overlay(
            kind=kind,
            name_a=name_a,
            name_b=name_b,
            entity_type=entity_type,
            scope=scope,
            notebook_id=notebook_id,
        )
        logger.info(
            "overlay rule added: kind={kind} scope={scope} type={t}",
            kind=kind,
            scope=scope,
            t=entity_type or "*",
        )
        return rid

    async def remove_rule(self, overlay_id: str) -> bool:
        """Delete one overlay rule by id. ``True`` when a row was removed."""
        return await self.repo.delete_overlay(overlay_id)

    async def list_rules(
        self, notebook_id: Optional[str] = None
    ) -> List[OverlayRule]:
        """Resolve the rules in effect for a scope (global + that notebook)."""
        rows = await self.repo.list_overlays(notebook_id=notebook_id)
        rules: List[OverlayRule] = []
        for row in rows:
            nb = row.get("notebook")
            rules.append(
                OverlayRule(
                    id=str(row.get("id")),
                    kind=str(row.get("kind") or ""),
                    scope=str(row.get("scope") or "global"),
                    name_a=str(row.get("name_a") or ""),
                    name_b=str(row.get("name_b") or ""),
                    entity_type=str(row.get("entity_type") or ""),
                    notebook_id=str(nb) if nb else None,
                )
            )
        return rules

    # ------------------------------------------------------------------
    # Resolution helpers (consumed by candidate_dedup_service)
    # ------------------------------------------------------------------

    async def split_pairs(
        self, notebook_id: Optional[str] = None
    ) -> set[frozenset[str]]:
        """Normalized force-SPLIT pairs in effect (the hard-veto set).

        Per-notebook scope wins: a notebook split rule is included alongside the
        global split rules when ``notebook_id`` is supplied.
        """
        rules = await self.list_rules(notebook_id=notebook_id)
        return {r.pair for r in rules if r.kind == "split"}

    async def merge_rules(
        self, notebook_id: Optional[str] = None
    ) -> List[OverlayRule]:
        """Force-MERGE rules in effect for the given scope (hard includes)."""
        rules = await self.list_rules(notebook_id=notebook_id)
        return [r for r in rules if r.kind == "merge"]

    # ------------------------------------------------------------------
    # K.2 alias_overrides DB seam
    # ------------------------------------------------------------------

    async def sync_alias_overrides(
        self, notebook_id: Optional[str] = None
    ) -> dict[str, str]:
        """Push force-merge rules into the K.2 ``alias_overrides`` DB overlay.

        A force-merge ``(A, B)`` becomes an alias mapping ``A -> B`` (both
        normalized) so ``normalize_entity_name`` collapses A onto B for FUTURE
        ingests — not just the retroactive proposer. Returns the injected map.
        Force-split rules are NOT pushed here (they are a proposer-time veto, not
        a normalization rule).
        """
        merges = await self.merge_rules(notebook_id=notebook_id)
        overlay: dict[str, str] = {}
        for rule in merges:
            a = normalize_entity_name(rule.name_a)
            b = normalize_entity_name(rule.name_b)
            if a and b and a != b:
                overlay[a] = b
        set_db_overlay(overlay)
        logger.info(
            "alias_overrides DB overlay synced: {n} force-merge rules",
            n=len(overlay),
        )
        return overlay


__all__ = ["OverlayService", "OverlayRule"]
