"""Track V.5: the ``ENABLE_REFERENCE_EXTRACTION`` config gate (read at call time).

The flag decides whether the guarded post-ingest reference pass is enqueued. Its
DEFAULT is the safe value (OFF) so an un-flagged ingest path is byte-for-byte the
pre-V.5 path. Parsed leniently (a mis-set value never crashes ingest).
"""

from __future__ import annotations

import pytest
from app_main.config import get_reference_extraction_enabled


def test_default_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENABLE_REFERENCE_EXTRACTION", raising=False)
    assert get_reference_extraction_enabled() is False


def test_blank_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_REFERENCE_EXTRACTION", "")
    assert get_reference_extraction_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on", " On "])
def test_truthy_values_enable(
    monkeypatch: pytest.MonkeyPatch, val: str
) -> None:
    monkeypatch.setenv("ENABLE_REFERENCE_EXTRACTION", val)
    assert get_reference_extraction_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "garbage"])
def test_non_truthy_values_stay_off(
    monkeypatch: pytest.MonkeyPatch, val: str
) -> None:
    monkeypatch.setenv("ENABLE_REFERENCE_EXTRACTION", val)
    assert get_reference_extraction_enabled() is False
