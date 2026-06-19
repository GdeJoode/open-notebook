"""
Application configuration for app-main.
"""

import os

# ROOT DATA FOLDER
DATA_FOLDER = os.environ.get("DATA_FOLDER", "./data")

# LANGGRAPH CHECKPOINT FILE
sqlite_folder = os.path.join(DATA_FOLDER, "sqlite-db")
os.makedirs(sqlite_folder, exist_ok=True)
LANGGRAPH_CHECKPOINT_FILE = os.path.join(sqlite_folder, "checkpoints.sqlite")

# UPLOADS FOLDER
UPLOADS_FOLDER = os.path.join(DATA_FOLDER, "uploads")
os.makedirs(UPLOADS_FOLDER, exist_ok=True)

# TIKTOKEN CACHE FOLDER
TIKTOKEN_CACHE_DIR = os.path.join(DATA_FOLDER, "tiktoken-cache")
os.makedirs(TIKTOKEN_CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Upload guards + rate limiting (Track I, Phase I.H1)
# ---------------------------------------------------------------------------
# These knobs are intentionally module-level functions rather than constants
# because tests need to be able to monkeypatch env at runtime *after* the
# `create_app()` factory and routers have already imported the module.
# Reading from env at call time also matches the existing DATA_FOLDER pattern
# above and avoids the need to wire Pydantic Settings just for three integers.


def _int_env(name: str, default: int) -> int:
    """Read an integer env var, falling back to ``default`` on missing/invalid.

    Used for upload guards + rate limiting. Kept lenient (no crash on garbage
    input) because mis-set ops env should not take the API down — fall back
    to the safe default instead.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def get_max_file_size_mb() -> int:
    """Maximum accepted upload size in megabytes (default 500)."""
    return _int_env("MAX_FILE_SIZE_MB", 500)


def get_max_page_count() -> int:
    """Maximum accepted PDF page count (default 500)."""
    return _int_env("MAX_PAGE_COUNT", 500)


def get_rate_limit_rpm() -> int:
    """Per-IP requests-per-minute for guarded endpoints (default 120).

    DS upstream uses 60; ON gets a little slack per Q-I-H1-1.
    """
    return _int_env("RATE_LIMIT_RPM", 120)


# Module-level constants for convenience / introspection. Prefer the getters
# above in hot code paths so env changes (e.g. in tests) take effect without
# re-importing.
MAX_FILE_SIZE_MB = get_max_file_size_mb()
MAX_PAGE_COUNT = get_max_page_count()
RATE_LIMIT_RPM = get_rate_limit_rpm()
