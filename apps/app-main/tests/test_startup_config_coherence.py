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


@pytest.mark.asyncio
async def test_the_context_check_is_actually_called(monkeypatch) -> None:
    """The wiring for `check_ollama_context`, not the function — M7.

    Review found this reachable but unasserted: deleting the whole `try:` block
    that calls it would have left every suite green, and on this deployment it
    checks zero rows (the `model` table holds one NVIDIA row), so nothing observed
    it in production either. That is the same hole M6 had, in the check built
    BECAUSE the M.4 guard was inert.

    Asserts it is called with what the model rows actually provide — a name/window
    mapping — rather than merely that something ran.
    """
    seen: list = []

    def _spy(declared, **_kwargs):
        seen.append(declared)
        return []

    class _Row:
        name = "llama3.1:8b-instruct-q4_0"
        context_window = 32768

    class _Repo:
        async def get_by_provider(self, provider):
            assert provider == "ollama"
            return [_Row()]

    monkeypatch.setattr("shared.config_coherence.check_ollama_context", _spy)
    monkeypatch.setattr("app_main.dependencies.get_model_repo", lambda: _Repo())

    await _check_configuration_coherence()

    assert seen == [{"llama3.1:8b-instruct-q4_0": 32768}], (
        "the startup path did not call check_ollama_context with the model rows"
    )
