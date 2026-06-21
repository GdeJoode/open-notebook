"""Unit tests for the Track J.3 layered privacy-mode resolver.

Covers the 4 precedence acceptance criteria from the J.3 plan §3 plus the
sticky-never-escalates property:

1. document private=True over a cloud notebook -> PRIVATE (sticky wins).
2. notebook privacy_mode='private' over global-cloud -> PRIVATE.
3. no overrides + global 'cloud' -> CLOUD.
4. no overrides + global 'private' -> PRIVATE.
5. PROPERTY: no input combination with source_private=True ever yields CLOUD.

The resolver reads the notebook + settings repos via the DI factories; here
those are monkeypatched with fakes so no DB is touched.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import pytest
from app_main.services.model_routing.privacy_resolver import resolve_privacy_mode
from app_main.services.model_routing.route_resolver import PrivacyMode

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeNotebookRepo:
    """Returns a notebook with the configured ``privacy_mode`` by id."""

    def __init__(self, privacy_mode: Optional[str], exists: bool = True):
        self._privacy_mode = privacy_mode
        self._exists = exists

    async def get(self, notebook_id: str):
        if not self._exists:
            return None
        return SimpleNamespace(id=notebook_id, privacy_mode=self._privacy_mode)


class _FakeSettingsRepo:
    """Returns a settings singleton with the configured global default."""

    def __init__(self, default_privacy_mode: Optional[str]):
        self._default = default_privacy_mode

    async def get(self):
        return SimpleNamespace(default_privacy_mode=self._default)


@pytest.fixture
def wire(monkeypatch):
    """Wire the notebook + settings repos with fakes via the DI factories."""

    def _wire(
        notebook_privacy_mode: Optional[str] = None,
        global_default: Optional[str] = "cloud",
        notebook_exists: bool = True,
    ) -> None:
        monkeypatch.setattr(
            "app_main.dependencies.get_notebook_repo",
            lambda: _FakeNotebookRepo(notebook_privacy_mode, exists=notebook_exists),
            raising=True,
        )
        monkeypatch.setattr(
            "app_main.dependencies.get_settings_repo",
            lambda: _FakeSettingsRepo(global_default),
            raising=True,
        )

    return _wire


# ---------------------------------------------------------------------------
# AC1-AC4: precedence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ac1_document_private_wins_over_cloud_notebook(wire):
    """AC1: source_private=True over a cloud notebook -> PRIVATE (sticky)."""
    wire(notebook_privacy_mode="cloud", global_default="cloud")
    mode = await resolve_privacy_mode(
        source_private=True, notebook_id="notebook:abc"
    )
    assert mode == PrivacyMode.PRIVATE


@pytest.mark.asyncio
async def test_ac2_notebook_private_beats_global_cloud(wire):
    """AC2: notebook privacy_mode='private' over global-cloud -> PRIVATE."""
    wire(notebook_privacy_mode="private", global_default="cloud")
    mode = await resolve_privacy_mode(
        source_private=False, notebook_id="notebook:abc"
    )
    assert mode == PrivacyMode.PRIVATE


@pytest.mark.asyncio
async def test_ac3_global_cloud_default(wire):
    """AC3: no overrides + global 'cloud' -> CLOUD."""
    wire(notebook_privacy_mode=None, global_default="cloud")
    mode = await resolve_privacy_mode(source_private=False, notebook_id=None)
    assert mode == PrivacyMode.CLOUD


@pytest.mark.asyncio
async def test_ac4_global_private_default(wire):
    """AC4: no overrides + global 'private' -> PRIVATE."""
    wire(notebook_privacy_mode=None, global_default="private")
    mode = await resolve_privacy_mode(source_private=False, notebook_id=None)
    assert mode == PrivacyMode.PRIVATE


# ---------------------------------------------------------------------------
# Extra precedence corners
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_notebook_cloud_overrides_global_private(wire):
    """A notebook set to 'cloud' beats a global 'private' default (most-specific
    wins is symmetric — only ``source.private`` is sticky)."""
    wire(notebook_privacy_mode="cloud", global_default="private")
    mode = await resolve_privacy_mode(
        source_private=False, notebook_id="notebook:abc"
    )
    assert mode == PrivacyMode.CLOUD


@pytest.mark.asyncio
async def test_unknown_notebook_mode_falls_through_to_global(wire):
    """A typo'd notebook mode degrades to the global default, not a crash."""
    wire(notebook_privacy_mode="cloudy", global_default="private")
    mode = await resolve_privacy_mode(
        source_private=False, notebook_id="notebook:abc"
    )
    assert mode == PrivacyMode.PRIVATE


@pytest.mark.asyncio
async def test_unset_global_defaults_to_cloud(wire):
    """An unset global default resolves to CLOUD (Track J's ship default)."""
    wire(notebook_privacy_mode=None, global_default=None)
    mode = await resolve_privacy_mode(source_private=False, notebook_id=None)
    assert mode == PrivacyMode.CLOUD


# ---------------------------------------------------------------------------
# AC5: property — a private document NEVER goes cloud
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("notebook_privacy_mode", [None, "cloud", "private", "garbage"])
@pytest.mark.parametrize("global_default", [None, "cloud", "private", "garbage"])
@pytest.mark.parametrize("notebook_id", [None, "notebook:abc"])
async def test_sticky_private_never_yields_cloud(
    wire, notebook_privacy_mode, global_default, notebook_id
):
    """PROPERTY: across the full cross-product of notebook + global settings,
    ``source_private=True`` always resolves to PRIVATE — never CLOUD."""
    wire(
        notebook_privacy_mode=notebook_privacy_mode,
        global_default=global_default,
    )
    mode = await resolve_privacy_mode(
        source_private=True, notebook_id=notebook_id
    )
    assert mode == PrivacyMode.PRIVATE
    assert mode != PrivacyMode.CLOUD
