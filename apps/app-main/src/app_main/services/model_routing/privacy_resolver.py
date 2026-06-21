"""Layered privacy-mode resolver (Track J.3).

Computes the :class:`PrivacyMode` to hand to :func:`resolve_route` from the
three layered sources, most-specific-wins, with **sticky ``private``**:

    document ``source.private == True``   (STICKY — always wins, never overridden)
        else notebook ``privacy_mode``    ('cloud' | 'private' when set)
            else global ``default_privacy_mode``

The sticky rule is the load-bearing invariant of the whole track: a document
flagged ``private`` must NEVER be routed to a cloud provider, regardless of the
notebook or global setting. It is encoded here as a SINGLE early-return at the
top of :func:`resolve_privacy_mode` so there is no code path that can turn a
private document into CLOUD. A property-style test asserts no input combination
with ``source_private=True`` yields CLOUD.

This module computes the MODE only; it does not pick a model or run anything —
that is :mod:`route_resolver` (J.1) and the failover executor (J.2). J.4 wires
the resolved mode into the actual cloud dispatch; in J.3 the mode is computed +
threaded but extraction execution is unchanged (see the seam comment in
``entity_extraction_service.run_extraction``).
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from app_main.services.model_routing.route_resolver import PrivacyMode


def _coerce_mode(value: Optional[str]) -> Optional[PrivacyMode]:
    """Parse a stored privacy-mode string into a :class:`PrivacyMode`.

    Returns ``None`` for an unset / blank / unrecognized value so the caller
    falls through to the next layer. We intentionally do not raise on an
    unknown string: a typo'd notebook setting should degrade to "inherit the
    next layer", not crash extraction.
    """
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized == PrivacyMode.PRIVATE.value:
        return PrivacyMode.PRIVATE
    if normalized == PrivacyMode.CLOUD.value:
        return PrivacyMode.CLOUD
    logger.warning(
        f"privacy_resolver: unrecognized privacy mode {value!r}; "
        "falling through to the next layer"
    )
    return None


async def _notebook_mode(notebook_id: Optional[str]) -> Optional[PrivacyMode]:
    """Read the per-notebook ``privacy_mode`` layer, or ``None`` to inherit."""
    if not notebook_id:
        return None
    try:
        from app_main.dependencies import get_notebook_repo

        notebook = await get_notebook_repo().get(notebook_id)
    except Exception as e:  # noqa: BLE001 — degrade to the next layer, never crash
        logger.warning(
            f"privacy_resolver: failed to load notebook {notebook_id!r} "
            f"({e}); inheriting the global privacy mode"
        )
        return None
    if notebook is None:
        return None
    return _coerce_mode(getattr(notebook, "privacy_mode", None))


async def _global_mode() -> PrivacyMode:
    """Read the global ``default_privacy_mode`` setting.

    Defaults to CLOUD (Track J's "cloud by default") when the setting is
    unreadable or unset, so a settings hiccup never silently forces the whole
    instance private (the surprising direction) — and never leaks a *private*
    document to cloud either, because this branch is only reached when no
    document/notebook override applied.
    """
    try:
        from app_main.dependencies import get_settings_repo

        settings = await get_settings_repo().get()
    except Exception as e:  # noqa: BLE001 — degrade to the cloud default
        logger.warning(
            f"privacy_resolver: failed to load settings ({e}); "
            "defaulting to CLOUD"
        )
        return PrivacyMode.CLOUD
    mode = _coerce_mode(getattr(settings, "default_privacy_mode", None))
    return mode or PrivacyMode.CLOUD


async def resolve_privacy_mode(
    *,
    source_private: Optional[bool],
    notebook_id: Optional[str],
) -> PrivacyMode:
    """Resolve the effective :class:`PrivacyMode` for an LLM stage.

    Precedence (most-specific wins, ``private`` sticky):

    1. ``source_private is True`` -> :attr:`PrivacyMode.PRIVATE` — STICKY single
       early-return. No later layer can override this; a private document is
       never routed to cloud.
    2. else the notebook's ``privacy_mode`` when set ('cloud' | 'private').
    3. else the global ``default_privacy_mode`` (defaults to 'cloud').

    Args:
        source_private: The document's ``source.private`` flag. ``True`` forces
            PRIVATE unconditionally; ``False``/``None`` defer to the lower layers.
        notebook_id: The notebook the source belongs to, used to read the
            per-notebook ``privacy_mode``. ``None`` skips the notebook layer.

    Returns:
        The resolved :class:`PrivacyMode`.
    """
    # STICKY RULE — the single early-return that guarantees a private document
    # can never escalate to cloud. Keep this first; do not add any branch above
    # it that could return CLOUD for a private document.
    if source_private is True:
        return PrivacyMode.PRIVATE

    notebook_mode = await _notebook_mode(notebook_id)
    if notebook_mode is not None:
        return notebook_mode

    return await _global_mode()
