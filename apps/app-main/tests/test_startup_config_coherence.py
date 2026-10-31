"""The coherence check runs at startup, and refuses (Track PC.6).

`packages/shared/tests/test_config_coherence.py` proves the checks are correct.
This file proves they are REACHED — PC.1b's lesson, paid for once already: a rule
that runs nowhere is not a rule, and a check that is only unit-tested is a rule
that runs nowhere.
"""

from __future__ import annotations

import pytest

from app_main.api.app import _check_configuration_coherence, lifespan
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
    resolution off — must
    start. A check that refuses the default configuration is worse than no check:
    the first thing anyone does is disable it.
    """
    await _check_configuration_coherence()


@pytest.mark.asyncio
async def test_the_judge_model_comes_from_the_register(monkeypatch) -> None:
    """F2-08: the judge is a model from the register, not a `default_models` row.

    Alignment and its judge on, KG resolution on: with a chat model in the
    register startup proceeds; without one it refuses.
    """
    from llm_manager.register import ModelRegister, RegisterModel

    monkeypatch.setattr(
        "app_main.config.get_concept_alignment_enabled", lambda: True
    )
    monkeypatch.setattr(
        "entity_filtering.config.KGResolutionConfig",
        lambda *a, **k: type("C", (), {"enabled": True})(),
    )
    monkeypatch.setattr(
        "entity_filtering.config.ConceptAlignmentConfig",
        lambda *a, **k: type("C", (), {"judge_enabled": True})(),
    )

    def _register(taak: str) -> ModelRegister:
        return ModelRegister(versie=1, modellen=[RegisterModel(
            naam="m", versie="v1", aanroepwijze="llama-swap", locatie="lokaal",
            toegestaan_intern="ja", taken=[taak])])

    monkeypatch.setattr("llm_manager.register.get_register", lambda: _register("chat"))
    await _check_configuration_coherence()

    monkeypatch.setattr("llm_manager.register.get_register", lambda: _register("embedding"))
    with pytest.raises(ConfigurationError):
        await _check_configuration_coherence()


@pytest.mark.asyncio
async def test_the_lifespan_actually_calls_the_check_and_lets_it_refuse(
    monkeypatch,
) -> None:
    """The wiring, not the function — found by mutation.

    Deleting `await _check_configuration_coherence()` from the lifespan left every
    test above green, because they all call the function directly. That is the
    same defect the whole track keeps producing: something is built and tested,
    and the caller never uses it. PC.1b's rule in one line — a check that runs
    nowhere is not a check.

    Asserts two things at once: the call site exists, and a refusal PROPAGATES.
    The second matters independently — wrapping the call in the best-effort
    try/except that surrounds the seeds directly below it would silence every
    BLOCK while leaving the call in place.
    """
    calls: list[int] = []

    class _Sentinel(RuntimeError):
        pass

    async def _spy() -> None:
        calls.append(1)
        raise _Sentinel("refusing")

    # Migrations run first and need a database; they are not what is under test.
    class _NoMigrations:
        async def get_current_version(self):
            return 99

        async def needs_migration(self):
            return False

    monkeypatch.setattr("app_main.api.app.AsyncMigrationManager", _NoMigrations)
    monkeypatch.setattr("app_main.api.app._check_configuration_coherence", _spy)

    with pytest.raises(_Sentinel):
        async with lifespan(object()):  # pragma: no cover — must not be entered
            pass

    assert calls == [1], "the lifespan did not call the coherence check"
