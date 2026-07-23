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


# ---------------------------------------------------------------------------
# Reference-extraction post-ingest pass (Track V.5)
# ---------------------------------------------------------------------------
# The V.5 orchestration (extract a source's references -> feed U.3's ``cites``
# materialization) hangs off ingest as a GUARDED, best-effort, OPT-IN post-ingest
# stage. It is OFF by default so the ingest path is byte-for-byte the same as
# before (only a guarded, config-gated no-op is added). Ops opt in by setting
# ``ENABLE_REFERENCE_EXTRACTION=true``. Read at call time (like the upload guards
# above) so tests can monkeypatch the env after import.

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _bool_env(name: str, default: bool) -> bool:
    """Read a boolean env var, falling back to ``default`` on missing/blank.

    Lenient like :func:`_int_env`: an unset/blank value returns ``default``; any
    other value is truthy only when it is one of ``1/true/yes/on`` (case-
    insensitive). A mis-set flag never crashes ingest — it falls back to the safe
    default instead.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in _TRUE_VALUES


def get_reference_extraction_enabled() -> bool:
    """Whether the V.5 reference-extraction post-ingest pass is enabled.

    Default ``False`` (the safe value): an un-flagged ingest never enqueues the
    reference pass, keeping the pipeline behaviour identical to pre-V.5. Set
    ``ENABLE_REFERENCE_EXTRACTION=true`` to turn the guarded pass on.
    """
    return _bool_env("ENABLE_REFERENCE_EXTRACTION", False)
