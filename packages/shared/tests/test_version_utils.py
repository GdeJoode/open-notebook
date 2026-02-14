"""Tests for shared.utils.version."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from shared.utils.version import (
    compare_versions,
    get_installed_version,
    get_version_from_github,
)


class TestCompareVersions:
    def test_less_than(self):
        assert compare_versions("0.1.0", "0.2.0") == -1

    def test_equal(self):
        assert compare_versions("1.0.0", "1.0.0") == 0

    def test_greater_than(self):
        assert compare_versions("2.0.0", "1.9.9") == 1

    def test_prerelease_less_than_release(self):
        assert compare_versions("1.0.0a1", "1.0.0") == -1


class TestGetInstalledVersion:
    def test_existing_package(self):
        # 'packaging' is a direct dependency of shared, so always present
        version = get_installed_version("packaging")
        assert isinstance(version, str)
        assert len(version) > 0

    def test_missing_package(self):
        with pytest.raises(Exception):
            get_installed_version("nonexistent-package-xyz-12345")


class TestGetVersionFromGithub:
    def test_non_github_url_raises(self):
        # Make httpx available in sys.modules so the import doesn't fail
        mock_httpx = MagicMock()
        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            with pytest.raises(ValueError, match="Not a GitHub URL"):
                get_version_from_github("https://gitlab.com/owner/repo")

    def test_invalid_path_raises(self):
        mock_httpx = MagicMock()
        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            with pytest.raises(ValueError, match="Invalid GitHub"):
                get_version_from_github("https://github.com/owner")

    def test_fetches_standard_version(self):
        """Standard project.version in pyproject.toml."""
        toml_content = '[project]\nname = "test"\nversion = "1.2.3"\n'

        mock_response = MagicMock()
        mock_response.text = toml_content
        mock_response.raise_for_status = MagicMock()

        mock_httpx = MagicMock()
        mock_httpx.get.return_value = mock_response

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            result = get_version_from_github(
                "https://github.com/owner/repo", "main"
            )

        assert result == "1.2.3"

    def test_fetches_poetry_version(self):
        """Poetry-style pyproject.toml (tool.poetry.version)."""
        toml_content = '[tool.poetry]\nname = "test"\nversion = "3.4.5"\n'

        mock_response = MagicMock()
        mock_response.text = toml_content
        mock_response.raise_for_status = MagicMock()

        mock_httpx = MagicMock()
        mock_httpx.get.return_value = mock_response

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            result = get_version_from_github(
                "https://github.com/owner/repo", "main"
            )

        assert result == "3.4.5"
