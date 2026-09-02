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


def get_concept_alignment_enabled() -> bool:
    """Whether Track N.4's concept-alignment stage runs (D-N4-8).

    Default ``False``, the safe value: an un-flagged run behaves exactly as it
    did before the stage existed. Set ``ENABLE_CONCEPT_ALIGNMENT=true`` to turn
    it on.

    The flag exists because the stage was otherwise unreachable in every real
    run — the previous judge tier shipped behind a config default nothing set,
    which is the failure D-N4-8 was written to prevent. Reading it at call time
    (like the upload guards) lets ops flip it without a rebuild.
    """
    return _bool_env("ENABLE_CONCEPT_ALIGNMENT", False)


# ---------------------------------------------------------------------------
# Audit thresholds (Track F.1 — operations & quality)
# ---------------------------------------------------------------------------
# Tunable defaults for the six LLM-free audit checks. Read at call time (like
# the upload guards) so tests + ops can override via env without a rebuild. A
# @dataclass snapshot keeps the service body free of numeric literals (AC7).


def _float_env(name: str, default: float) -> float:
    """Read a float env var, falling back to ``default`` on missing/invalid."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class AuditThresholds:
    """Resolved audit thresholds. Instantiate per run so env overrides apply."""

    def __init__(self) -> None:
        #: stale_sources: a source older than this many days is stale (info).
        self.stale_days: int = _int_env("AUDIT_STALE_DAYS", 180)
        #: low_confidence_survivors: the extraction filter floor a survivor sits on.
        self.filter_min_confidence: float = _float_env(
            "AUDIT_FILTER_MIN_CONFIDENCE", 0.9
        )
        #: ...and the band width just above it that counts as "barely survived".
        self.low_confidence_band: float = _float_env(
            "AUDIT_LOW_CONFIDENCE_BAND", 0.05
        )
        #: long_pending_orphans: a pending orphan older than this many days (warn).
        self.orphan_days: int = _int_env("AUDIT_ORPHAN_DAYS", 30)
        #: community_cohesion: min members before a community is judged.
        self.cohesion_min_size: int = _int_env("AUDIT_COHESION_MIN_SIZE", 3)
        #: ...and the internal-edge-density floor below which cohesion is low.
        self.cohesion_floor: float = _float_env("AUDIT_COHESION_FLOOR", 0.15)
        #: schema_drift: flag when the fallback-type entity ratio exceeds this.
        self.drift_ratio: float = _float_env("AUDIT_DRIFT_RATIO", 0.30)


def get_audit_thresholds() -> "AuditThresholds":
    """A fresh :class:`AuditThresholds` snapshot (env overrides applied)."""
    return AuditThresholds()


# ---------------------------------------------------------------------------
# Agent integration (Track G.1)
# ---------------------------------------------------------------------------


def get_agent_rate_limit_rpm() -> int:
    """Default per-key requests-per-minute for the agent API (default 60).

    A key's own ``rate_limit_rpm`` overrides this; the config value is the
    fallback when a key has none. Read at call time so ops can retune via env.
    """
    return _int_env("AGENT_RATE_LIMIT_RPM", 60)


def get_agent_audit_retention_days() -> int:
    """How long to keep agent_audit_log rows (default 90 days)."""
    return _int_env("AGENT_AUDIT_RETENTION_DAYS", 90)


# ---------------------------------------------------------------------------
# Inbox file-watcher (Track G.5)
# ---------------------------------------------------------------------------
# An opt-in watcher on conventional inbox folders that auto-ingests dropped
# files through the SAME process_source chain as the API. OFF by default, so an
# existing deployment sees zero behaviour change until it is explicitly enabled.


def get_file_watcher_enabled() -> bool:
    """Whether the inbox file-watcher runs (default False — opt-in).

    When False the watcher never starts from the lifespan, so there is no new
    background thread / observer and behaviour is identical to pre-G.5.
    """
    return _bool_env("FILE_WATCHER_ENABLED", False)


def get_inbox_paths() -> list[str]:
    """Watched inbox folders (default: ``<DATA_FOLDER>/inbox``).

    ``INBOX_PATHS`` is an OS-pathsep-separated (``:`` on POSIX) list so ops can
    watch several roots. Blank entries are dropped. The default lives under
    DATA_FOLDER so it shares the volume the uploads/sqlite folders already use.
    """
    raw = os.environ.get("INBOX_PATHS")
    if not raw or not raw.strip():
        return [os.path.join(DATA_FOLDER, "inbox")]
    return [p.strip() for p in raw.split(os.pathsep) if p.strip()]


def get_inbox_default_notebook_id() -> "str | None":
    """Notebook a GLOBAL-inbox file is attached to (``INBOX_DEFAULT_NOTEBOOK_ID``).

    None → files dropped in a global inbox are ingested without a notebook (a
    per-notebook inbox ``.../notebook:<id>/inbox/`` still routes to its own).
    """
    raw = os.environ.get("INBOX_DEFAULT_NOTEBOOK_ID")
    return raw.strip() if raw and raw.strip() else None


def get_inbox_debounce_seconds() -> int:
    """Seconds to coalesce a burst of writes to one file (default 3).

    A slow copy fires many modify events; the watcher waits this long after the
    LAST event before ingesting, so a file is enqueued once, fully written.
    """
    return _int_env("INBOX_DEBOUNCE_SECONDS", 3)
