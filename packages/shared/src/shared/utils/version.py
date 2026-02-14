"""
Version utilities.

Fetches remote versions from GitHub, reads installed package versions,
and compares semantic version strings.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from urllib.parse import urlparse

from packaging.version import parse as parse_version


def get_version_from_github(repo_url: str, branch: str = "main") -> str:
    """Fetch the version from ``pyproject.toml`` in a public GitHub repo.

    Uses *httpx* (lazy-imported) to avoid adding a hard dependency on
    ``requests``.

    Raises:
        ValueError: If *repo_url* is not a GitHub URL.
        httpx.HTTPStatusError: On HTTP errors.
        KeyError: If version is not found in pyproject.toml.
    """
    import tomllib

    import httpx

    parsed_url = urlparse(repo_url)
    if "github.com" not in parsed_url.netloc:
        raise ValueError("Not a GitHub URL")

    path_parts = parsed_url.path.strip("/").split("/")
    if len(path_parts) < 2:
        raise ValueError("Invalid GitHub repository URL")

    owner, repo = path_parts[0], path_parts[1]
    raw_url = (
        f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/pyproject.toml"
    )

    response = httpx.get(raw_url, timeout=10)
    response.raise_for_status()

    pyproject_data = tomllib.loads(response.text)

    # Try poetry-style first, then standard project.version
    try:
        return pyproject_data["tool"]["poetry"]["version"]
    except KeyError:
        pass
    try:
        return pyproject_data["project"]["version"]
    except KeyError:
        raise KeyError("Version not found in pyproject.toml")


def get_installed_version(package_name: str) -> str:
    """Return the version of an installed package.

    Raises:
        PackageNotFoundError: If the package is not installed.
    """
    try:
        return _pkg_version(package_name)
    except PackageNotFoundError:
        raise PackageNotFoundError(f"Package '{package_name}' not found")


def compare_versions(version1: str, version2: str) -> int:
    """Compare two semantic version strings.

    Returns:
        -1 if *version1* < *version2*, 0 if equal, 1 if greater.
    """
    v1 = parse_version(version1)
    v2 = parse_version(version2)

    if v1 < v2:
        return -1
    elif v1 > v2:
        return 1
    return 0
