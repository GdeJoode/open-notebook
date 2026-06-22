"""Unit tests for the K.5 OverlayService.

Covers the CRUD seam and the resolution semantics the candidate dedup service
relies on:

* global vs per-notebook scoping (a notebook rule only surfaces for that
  notebook; most-specific wins) — AC5;
* force-merge vs force-split classification;
* the K.2 ``alias_overrides`` DB-overlay bridge (force-merge rules flow into the
  normalizer; force-split rules do not).

DB-free: a fake repo records created rows and answers scoped reads.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest
from app_main.services.entity_resolution.overlay_service import OverlayService


class FakeOverlayRepo:
    """In-memory ``alias_overlay`` store with scope-aware list semantics."""

    def __init__(self) -> None:
        self._rows: Dict[str, Dict[str, Any]] = {}
        self._seq = 0

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
        # Mirror the repo's validation so service tests exercise it.
        if kind not in ("merge", "split"):
            raise ValueError(f"invalid kind {kind!r}")
        if scope not in ("global", "notebook"):
            raise ValueError(f"invalid scope {scope!r}")
        if not name_a.strip() or not name_b.strip():
            raise ValueError("empty names")
        if scope == "notebook" and not notebook_id:
            raise ValueError("notebook scope requires notebook_id")
        self._seq += 1
        rid = f"alias_overlay:{self._seq}"
        self._rows[rid] = {
            "id": rid,
            "kind": kind,
            "scope": scope,
            "name_a": name_a.strip(),
            "name_b": name_b.strip(),
            "entity_type": entity_type or "",
            "notebook": notebook_id if scope == "notebook" else None,
        }
        return rid

    async def list_overlays(
        self, notebook_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        out = []
        for row in self._rows.values():
            if row["scope"] == "global":
                out.append(dict(row))
            elif notebook_id is not None and row.get("notebook") == notebook_id:
                out.append(dict(row))
        return out

    async def delete_overlay(self, overlay_id: str) -> bool:
        return self._rows.pop(overlay_id, None) is not None


def _service() -> OverlayService:
    return OverlayService(repo=FakeOverlayRepo())


# --- scoping ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_notebook_rule_isolated_from_other_notebooks() -> None:
    """A notebook-scoped rule surfaces only for its notebook (AC5)."""
    svc = _service()
    await svc.add_rule(
        kind="merge",
        name_a="Alfa",
        name_b="Alpha",
        entity_type="organization",
        scope="notebook",
        notebook_id="notebook:nb1",
    )
    # Visible for nb1.
    nb1 = await svc.list_rules(notebook_id="notebook:nb1")
    assert len(nb1) == 1
    # Not visible for nb2.
    nb2 = await svc.list_rules(notebook_id="notebook:nb2")
    assert nb2 == []
    # Not visible at global scope.
    glob = await svc.list_rules(notebook_id=None)
    assert glob == []


@pytest.mark.asyncio
async def test_global_rule_visible_everywhere() -> None:
    """A global rule surfaces for every scope."""
    svc = _service()
    await svc.add_rule(
        kind="split", name_a="BZK", name_b="EZK", entity_type="organization"
    )
    assert len(await svc.list_rules(notebook_id=None)) == 1
    assert len(await svc.list_rules(notebook_id="notebook:nb1")) == 1


@pytest.mark.asyncio
async def test_notebook_scope_unions_global_and_notebook() -> None:
    """A notebook view returns the global rules PLUS the notebook's own."""
    svc = _service()
    await svc.add_rule(
        kind="split", name_a="BZK", name_b="EZK", entity_type="organization"
    )
    await svc.add_rule(
        kind="merge",
        name_a="Alfa",
        name_b="Alpha",
        entity_type="organization",
        scope="notebook",
        notebook_id="notebook:nb1",
    )
    nb1 = await svc.list_rules(notebook_id="notebook:nb1")
    assert len(nb1) == 2


# --- classification -----------------------------------------------------------


@pytest.mark.asyncio
async def test_split_and_merge_partitioned() -> None:
    """``split_pairs`` and ``merge_rules`` partition the rules by kind."""
    svc = _service()
    await svc.add_rule(
        kind="split", name_a="BZK", name_b="EZK", entity_type="organization"
    )
    await svc.add_rule(
        kind="merge", name_a="Alfa", name_b="Alpha", entity_type="organization"
    )
    splits = await svc.split_pairs()
    merges = await svc.merge_rules()
    assert len(splits) == 1
    assert len(merges) == 1
    assert merges[0].kind == "merge"


# --- validation ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_notebook_scope_requires_notebook_id() -> None:
    svc = _service()
    with pytest.raises(ValueError):
        await svc.add_rule(
            kind="merge", name_a="A", name_b="B", scope="notebook"
        )


@pytest.mark.asyncio
async def test_invalid_kind_rejected() -> None:
    svc = _service()
    with pytest.raises(ValueError):
        await svc.add_rule(kind="bogus", name_a="A", name_b="B")


# --- delete -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remove_rule() -> None:
    svc = _service()
    rid = await svc.add_rule(
        kind="merge", name_a="Alfa", name_b="Alpha", entity_type="organization"
    )
    assert await svc.remove_rule(rid) is True
    assert await svc.list_rules() == []
    # Removing a missing rule is False.
    assert await svc.remove_rule("alias_overlay:999") is False


# --- K.2 alias_overrides bridge -----------------------------------------------


@pytest.mark.asyncio
async def test_sync_alias_overrides_pushes_force_merges_only() -> None:
    """Force-merge rules flow into the normalizer overlay; splits do not."""
    from shared.config import alias_overrides

    svc = _service()
    await svc.add_rule(
        kind="merge",
        name_a="Min AZ",
        name_b="Algemene Zaken",
        entity_type="organization",
    )
    await svc.add_rule(
        kind="split", name_a="BZK", name_b="EZK", entity_type="organization"
    )
    overlay = await svc.sync_alias_overrides()
    # The merge rule produced a normalized alias mapping; the split did not.
    assert overlay  # non-empty
    assert all("bzk" not in k and "ezk" not in k for k in overlay)
    # Cleanup: clear the global DB overlay we injected so other tests are clean.
    alias_overrides.set_db_overlay({})
