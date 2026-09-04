"""The coherence check runs at startup, and refuses (Track PC.6).

`packages/shared/tests/test_config_coherence.py` proves the checks are correct.
This file proves they are REACHED — PC.1b's lesson, paid for once already: a rule
that runs nowhere is not a rule, and a check that is only unit-tested is a rule
that runs nowhere.
"""

from __future__ import annotations

import pytest

from app_main.api.app import _check_configuration_coherence
from shared.config_coherence import ConfigurationError


@pytest.mark.asyncio
async def test_startup_refuses_a_configuration_that_cannot_work(monkeypatch) -> None:
    """Alignment on, KG resolution off — startup must not proceed.

    Driven through the real startup function with the real checks; only the two
    flags and the defaults row are steered. Before PC.6 this configuration
    started fine and logged one warning.
    """
    monkeypatch.setattr(
        "app_main.config.get_concept_alignment_enabled", lambda: True
    )
    monkeypatch.setattr(
        "entity_filtering.config.KGResolutionConfig",
        lambda *a, **k: type("C", (), {"enabled": False})(),
    )

    with pytest.raises(ConfigurationError) as excinfo:
        await _check_configuration_coherence()

    codes = {f.code for f in excinfo.value.findings}
    assert "alignment-without-resolution" in codes
    # The message a developer will actually read must name the fix.
    assert "ENABLE_CONCEPT_ALIGNMENT=false" in str(excinfo.value)


@pytest.mark.asyncio
async def test_startup_accepts_the_shipped_configuration() -> None:
    """The counterweight, and the one that would have caught an over-eager check.

    The configuration this repository actually ships — alignment off, KG
    resolution off, NVIDIA retired but only on unreachable `public` routes — must
    start. A check that refuses the default configuration is worse than no check:
    the first thing anyone does is disable it.
    """
    await _check_configuration_coherence()


@pytest.mark.asyncio
async def test_an_unreadable_defaults_row_does_not_manufacture_a_refusal(
    monkeypatch,
) -> None:
    """A database error is not a verdict about the configuration.

    If reading `default_models` fails, the judge is assumed configured. Treating
    the error as "no model" would turn a transient DB blip into a startup refusal
    — inventing a BLOCK from an absence of information.
    """
    monkeypatch.setattr(
        "app_main.config.get_concept_alignment_enabled", lambda: True
    )
    monkeypatch.setattr(
        "entity_filtering.config.KGResolutionConfig",
        lambda *a, **k: type("C", (), {"enabled": True})(),
    )

    def _boom():
        raise RuntimeError("connection refused")

    monkeypatch.setattr("app_main.dependencies.get_default_models_repo", _boom)

    await _check_configuration_coherence()
