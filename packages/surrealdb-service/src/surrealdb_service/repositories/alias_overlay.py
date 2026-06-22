"""Repository for the ``alias_overlay`` table (Phase K.5).

``alias_overlay`` (migration 56) is the human escape hatch over the automated
fuzzy/embedding matcher: a ``split`` rule force-vetoes a pair (the over-merge
backstop) and a ``merge`` rule force-includes one. Each rule is scoped either
``global`` (applies everywhere) or ``notebook`` (applies only within one
notebook). Per-notebook is the most-specific scope and wins on conflict (K-D4).

The repo is intentionally thin: CRUD + a scoped ``list_overlays`` the
``CandidateDedupService`` and the ``alias_overrides`` DB seam consult. Rule
pairs are stored order-as-given but matched order-insensitively by the service.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query

_VALID_SCOPES = ("global", "notebook")
_VALID_KINDS = ("merge", "split")


class AliasOverlayRepository:
    """CRUD + scoped lookup over the ``alias_overlay`` table."""

    table_name = "alias_overlay"

    def __init__(self, config: Optional[SurrealDBConfig] = None) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    async def create_overlay(
        self,
        *,
        kind: str,
        name_a: str,
        name_b: str,
        entity_type: str = "",
        scope: str = "global",
        notebook_id: Optional[str] = None,
    ) -> str:
        """Create one overlay rule and return its record id.

        Args:
            kind: ``'merge'`` (force-include) or ``'split'`` (force-veto).
            name_a, name_b: the two surface forms the rule binds (must be
                non-empty; order-insensitive at read time).
            entity_type: the shared entity type the rule binds — the type
                discipline that stops a person/org homograph from being bound.
            scope: ``'global'`` or ``'notebook'``.
            notebook_id: required when ``scope='notebook'``; ignored otherwise.

        Raises:
            ValueError: invalid kind/scope, empty names, or a notebook scope
                without a notebook id.
        """
        if kind not in _VALID_KINDS:
            raise ValueError(f"invalid overlay kind {kind!r}")
        if scope not in _VALID_SCOPES:
            raise ValueError(f"invalid overlay scope {scope!r}")
        if not name_a or not name_a.strip() or not name_b or not name_b.strip():
            raise ValueError("overlay name_a and name_b must be non-empty")
        if scope == "notebook" and not notebook_id:
            raise ValueError("notebook scope requires a notebook_id")

        payload: Dict[str, Any] = {
            "scope": scope,
            "kind": kind,
            "name_a": name_a.strip(),
            "name_b": name_b.strip(),
            "entity_type": entity_type or "",
            "notebook": ensure_record_id(notebook_id)
            if (scope == "notebook" and notebook_id)
            else None,
        }
        rows = await execute_query(
            "CREATE alias_overlay CONTENT $data RETURN AFTER",
            {"data": payload},
            self.config,
        )
        rid = str(rows[0]["id"])
        logger.info(
            "alias_overlay created: id={id} kind={kind} scope={scope}",
            id=rid,
            kind=kind,
            scope=scope,
        )
        return rid

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def list_overlays(
        self, notebook_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return the overlay rules in effect for a scope.

        When ``notebook_id`` is ``None`` only ``global`` rules are returned.
        When it is supplied, ``global`` rules PLUS the rules scoped to that
        notebook are returned (the union the dedup service evaluates; precedence
        is resolved by the caller, most-specific-wins).
        """
        try:
            if notebook_id is None:
                return await execute_query(
                    "SELECT * FROM alias_overlay WHERE scope = 'global'",
                    {},
                    self.config,
                )
            return await execute_query(
                "SELECT * FROM alias_overlay "
                "WHERE scope = 'global' "
                "OR (scope = 'notebook' AND notebook = $nb)",
                {"nb": ensure_record_id(notebook_id)},
                self.config,
            )
        except Exception as exc:
            logger.error("alias_overlay list_overlays failed: {e}", e=exc)
            return []

    async def get_overlay(self, overlay_id: str) -> Optional[Dict[str, Any]]:
        """Fetch one overlay rule by id, or ``None`` if absent."""
        try:
            rows = await execute_query(
                "SELECT * FROM $id",
                {"id": ensure_record_id(overlay_id)},
                self.config,
            )
        except Exception as exc:
            logger.error("alias_overlay get_overlay failed: {e}", e=exc)
            return None
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_overlay(self, overlay_id: str) -> bool:
        """Delete one overlay rule by id. Returns ``True`` when a row was removed."""
        try:
            await execute_query(
                "DELETE $id",
                {"id": ensure_record_id(overlay_id)},
                self.config,
            )
            return True
        except Exception as exc:
            logger.error("alias_overlay delete_overlay failed: {e}", e=exc)
            return False


__all__ = ["AliasOverlayRepository"]
