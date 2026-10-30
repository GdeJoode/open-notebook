"""De bewaking in conftest: een test die 401/403 verwacht zonder de markering
``echte_sleutels`` faalt (F2-01). Gedraaid in een eigen pytest-run."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]
# Deze test bevat zelf `== 401`; hij heeft de testsleutel niet nodig.
pytestmark = pytest.mark.echte_sleutels

_CONFTEST = Path(__file__).with_name("conftest.py")


def test_401_zonder_markering_faalt(pytester):
    # Laad de echte conftest van app-main en gebruik alleen deze fixture.
    pytester.makeconftest(
        textwrap.dedent(
            f"""
            import importlib.util

            _spec = importlib.util.spec_from_file_location(
                "app_main_conftest", {str(_CONFTEST)!r}
            )
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _testsleutel_beheerder = _mod._testsleutel_beheerder
            """
        )
    )
    pytester.makepyfile(
        textwrap.dedent(
            """
            import pytest

            def test_verwacht_401():
                assert 1 + 400 == 401

            @pytest.mark.echte_sleutels
            def test_met_markering():
                assert 1 + 400 == 401

            def test_gewoon():
                assert True

            def test_ander_getal_401_telt_niet():
                assert str(401) == "401"
            """
        )
    )
    uitkomst = pytester.runpytest_inprocess("-p", "no:cacheprovider")
    uitkomst.assert_outcomes(passed=3, errors=1)
    uitkomst.stdout.fnmatch_lines(["*echte_sleutels*"])
