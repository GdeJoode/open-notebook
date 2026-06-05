"""Non-Docker safety nets for the ``testing`` subpackage.

These tests run on every ``pytest -q`` invocation (no marker) and guard
against path-drift bugs in ``fixtures.py`` that would otherwise only surface
the first time CI tries to boot a real testcontainer. The off-by-one in
attempt 1 (``parents[4]`` resolving to ``packages/`` instead of the repo
root) is the canonical example of what this file is here to catch.
"""

from __future__ import annotations

from pathlib import Path

from surrealdb_service.testing import fixtures as fx


def test_migrations_dir_resolves_to_real_directory() -> None:
    """``_MIGRATIONS_DIR`` must point at the actual workspace migrations dir.

    Regression guard for the attempt-1 off-by-one bug. If this fails after a
    repo restructure, fix ``_find_migrations_dir`` rather than this assertion.
    """
    assert fx._MIGRATIONS_DIR.is_dir(), (
        f"_MIGRATIONS_DIR does not resolve to a directory: {fx._MIGRATIONS_DIR}"
    )


def test_migrations_dir_contains_known_baseline_files() -> None:
    """The resolved directory must contain the migration-39+ canary files.

    Stronger than ``is_dir()`` alone: guarantees we've landed on the *right*
    directory rather than some unrelated ``migrations/`` folder a future
    restructure might introduce. We pin a few specific files rather than the
    whole 1..43 range because the migration sequence has historical gaps
    (1..10 + 26..43) and asserting contiguity would be wrong.
    """
    must_exist = {"1.surrealql", "39.surrealql", "43.surrealql"}
    present = {p.name for p in fx._MIGRATIONS_DIR.iterdir() if p.is_file()}
    missing = must_exist - present
    assert not missing, (
        f"_MIGRATIONS_DIR={fx._MIGRATIONS_DIR} is missing canary files: "
        f"{sorted(missing)}"
    )


def test_migrations_dir_is_under_repo_root_marker() -> None:
    """Sanity: the parent of ``_MIGRATIONS_DIR`` should have a workspace marker.

    A ``pyproject.toml`` declaring ``[tool.uv.workspace]`` lives at the repo
    root. We don't parse the TOML here — presence is enough to catch the
    ``packages/migrations`` mis-resolution from attempt 1.
    """
    parent = fx._MIGRATIONS_DIR.parent
    marker = parent / "pyproject.toml"
    assert marker.is_file(), (
        f"Expected pyproject.toml next to migrations/ at {parent}; not found."
    )
    content = marker.read_text(encoding="utf-8")
    assert "[tool.uv.workspace]" in content, (
        f"{marker} does not declare [tool.uv.workspace] — _MIGRATIONS_DIR may "
        f"be pointing at the wrong directory."
    )


def test_find_migrations_dir_returns_same_path() -> None:
    """Calling the resolver directly must agree with the module-level cache."""
    assert fx._find_migrations_dir() == fx._MIGRATIONS_DIR


def test_docker_available_is_callable_lazy() -> None:
    """``docker_available()`` is a plain function, not a module-import side effect.

    Regression guard for Minor #7: attempt 1 evaluated ``DOCKER_AVAILABLE`` at
    import time which forced a ~2s Docker SDK ping on every ``pytest`` run,
    even ``-m "not requires_docker"``. We now defer.
    """
    # No DOCKER_AVAILABLE constant should be exported.
    assert not hasattr(fx, "DOCKER_AVAILABLE"), (
        "DOCKER_AVAILABLE constant should be removed — use docker_available() "
        "to defer the SDK ping until the fixture actually runs."
    )
    assert callable(fx.docker_available)


def test_fixture_module_has_no_stale_live_surrealdb_async() -> None:
    """``live_surrealdb_async`` was unused dead code — must not return."""
    assert not hasattr(fx, "live_surrealdb_async"), (
        "live_surrealdb_async was removed as YAGNI in attempt 2. Re-introduce "
        "only with a real downstream caller AND a test that exercises it."
    )


def test_migrations_dir_path_is_absolute() -> None:
    """The resolved path must be absolute to survive ``cwd`` changes."""
    assert isinstance(fx._MIGRATIONS_DIR, Path)
    assert fx._MIGRATIONS_DIR.is_absolute()
