"""Config router - application health and version information."""

import asyncio
import time
import tomllib
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from loguru import logger

from surrealdb_service.connection import execute_query

router = APIRouter(tags=["config"])

# In-memory cache for version check results
_version_cache: dict = {
    "latest_version": None,
    "has_update": False,
    "timestamp": 0,
    "check_failed": False,
}

# Cache TTL in seconds (24 hours)
VERSION_CACHE_TTL = 24 * 60 * 60


def get_version() -> str:
    """Read version from pyproject.toml."""
    try:
        # Navigate from routers/ -> api/ -> app_main/ -> src/ -> app-main/ -> pyproject.toml
        pyproject_path = (
            Path(__file__).parent.parent.parent.parent.parent / "pyproject.toml"
        )
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)
            return pyproject.get("project", {}).get("version", "unknown")
    except Exception as e:
        logger.warning(f"Could not read version from pyproject.toml: {e}")
        return "unknown"


def get_latest_version_cached(
    current_version: str,
) -> tuple[Optional[str], bool]:
    """Check for the latest version from GitHub with caching.

    Returns:
        Tuple of (latest_version, has_update). latest_version is None if check failed.
    """
    global _version_cache

    cache_age = time.time() - _version_cache["timestamp"]
    if _version_cache["timestamp"] > 0 and cache_age < VERSION_CACHE_TTL:
        return _version_cache["latest_version"], _version_cache["has_update"]

    try:
        from open_notebook.utils.version_utils import (
            compare_versions,
            get_version_from_github,
        )

        latest_version = get_version_from_github(
            "https://github.com/lfnovo/open-notebook", "main"
        )
        has_update = compare_versions(current_version, latest_version) < 0

        _version_cache.update(
            {
                "latest_version": latest_version,
                "has_update": has_update,
                "timestamp": time.time(),
                "check_failed": False,
            }
        )
        return latest_version, has_update

    except Exception as e:
        logger.warning(f"Version check failed: {e}")
        _version_cache.update(
            {
                "latest_version": None,
                "has_update": False,
                "timestamp": time.time(),
                "check_failed": True,
            }
        )
        return None, False


async def check_database_health() -> dict:
    """Check if database is reachable using a lightweight query."""
    try:
        result = await asyncio.wait_for(
            execute_query("RETURN 1"),
            timeout=2.0,
        )
        if result:
            return {"status": "online"}
        return {"status": "offline", "error": "Empty result"}
    except asyncio.TimeoutError:
        logger.warning("Database health check timed out after 2 seconds")
        return {"status": "offline", "error": "Health check timeout"}
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        return {"status": "offline", "error": str(e)}


@router.get("/config")
async def get_config():
    """Get frontend configuration including version and health status."""
    current_version = get_version()

    latest_version = None
    has_update = False
    try:
        latest_version, has_update = get_latest_version_cached(current_version)
    except Exception as e:
        logger.error(f"Unexpected error during version check: {e}")

    db_health = await check_database_health()
    db_status = db_health["status"]

    if db_status == "offline":
        logger.warning(
            f"Database offline: {db_health.get('error', 'Unknown error')}"
        )

    return {
        "version": current_version,
        "latestVersion": latest_version,
        "hasUpdate": has_update,
        "dbStatus": db_status,
    }
