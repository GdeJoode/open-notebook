"""Extensible operator/user alias overrides for the org-alias layer (K.2).

The built-in :data:`shared.utils.org_aliases._GOV_ORG_ALIASES` is the *baseline*.
Operators/users extend OR override it without a code change through this loader,
which merges three layers (lowest → highest precedence):

1. **Built-in** — the curated NL-ministry allow-list (the baseline).
2. **File** — a JSON object ``{alias: canonical}`` at the path given by
   ``ONB_ALIAS_OVERRIDES_PATH`` (default: ``alias_overrides.json`` in CWD, only
   loaded if present). The file-based default config source.
3. **DB overlay** — a runtime-injectable dict, the seam for K.5's per-notebook /
   global ``alias_overlay`` table. Empty until K.5 wires it.

Precedence: LAST-WINS, deliberately (K.2 rev2 decision)
=======================================================
A higher layer that re-keys a built-in alias **shadows** it — e.g. an override
``{"bzk": "…"}`` replaces the built-in ``bzk`` canonical rather than being
ignored. This is intentional: alias overrides are *operator configuration*, and
the most useful semantics let an operator correct or retarget a built-in mapping
(or a curation mistake) without waiting for a code release. The built-in is the
*baseline an operator inherits when they say nothing*, not an immutable floor.
Adding a brand-new key and overriding an existing one are both supported; the
:func:`_build_resolved` merge order (built-in → file → db) pins the resolution.

Validation on load (the user-input surface the K.2 plan flags)
==============================================================
Every entry is validated and **normalized through the same pre-alias pipeline**
(:func:`shared.utils.nl_normalization.normalize_alias_key`) so a raw config key
(``"Min AZ"``) lands in the post-K.1 space the resolver looks up:

- empty/blank key or value → rejected (logged + skipped, never raises mid-ingest);
- key and value are both normalized so ``{"Min AZ": "Algemene Zaken"}`` becomes
  ``{"min az": "algemene zaken"}`` and is honoured by ``normalize_entity_name``.

The resolved map is cached; :func:`reload_alias_overrides` rebuilds it (used by
tests and by a future config-reload endpoint), and :func:`set_db_overlay` injects
the DB layer (K.5 seam) and invalidates the cache.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from loguru import logger

from shared.utils.nl_normalization import normalize_alias_key
from shared.utils.org_aliases import _GOV_ORG_ALIASES

# Env var pointing at the file-based override source. Optional: absent or missing
# file simply means "no file overrides".
_OVERRIDE_PATH_ENV = "ONB_ALIAS_OVERRIDES_PATH"
_DEFAULT_OVERRIDE_FILENAME = "alias_overrides.json"

# DB-overlay seam for K.5 (per-notebook / global ``alias_overlay``). Raw (un-
# normalized) entries; validated/normalized like every other layer on resolve.
_db_overlay: dict[str, str] = {}

# Cached resolved map (built-in + file + db, all normalized + validated).
_resolved_cache: dict[str, str] | None = None


def _override_path() -> Path | None:
    """Resolve the file-override path from the env var, if it points at a file."""
    raw = os.environ.get(_OVERRIDE_PATH_ENV)
    if raw:
        path = Path(raw)
    else:
        path = Path.cwd() / _DEFAULT_OVERRIDE_FILENAME
    return path if path.is_file() else None


def _load_file_overrides() -> dict[str, str]:
    """Read the JSON override file. Malformed file → empty (logged, non-fatal)."""
    path = _override_path()
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"alias override file {path} unreadable, ignoring: {exc}")
        return {}
    if not isinstance(data, dict):
        logger.warning(
            f"alias override file {path} is not a JSON object, ignoring"
        )
        return {}
    return {str(k): str(v) for k, v in data.items()}


def _validated(raw: dict[str, str], *, source: str) -> dict[str, str]:
    """Normalize + validate a raw override layer.

    Rejects empty/blank keys or values (logged + skipped). Keys and values are
    pushed through the pre-alias pipeline so they match at lookup time.
    """
    out: dict[str, str] = {}
    for raw_key, raw_val in raw.items():
        key = normalize_alias_key(str(raw_key))
        val = normalize_alias_key(str(raw_val))
        if not key or not val:
            logger.warning(
                f"alias override ({source}) rejected: empty key/value "
                f"({raw_key!r} -> {raw_val!r})"
            )
            continue
        out[key] = val
    return out


def _build_resolved() -> dict[str, str]:
    """Merge the three layers into the resolved alias map (later wins).

    Last-wins is deliberate (see module docstring): a file/DB override that
    re-keys a built-in alias shadows it, letting operator config retarget a
    built-in mapping without a code change.
    """
    resolved: dict[str, str] = dict(_GOV_ORG_ALIASES)
    resolved.update(_validated(_load_file_overrides(), source="file"))
    resolved.update(_validated(_db_overlay, source="db"))
    return resolved


def get_resolved_aliases() -> dict[str, str]:
    """Return the cached built-in+override alias map (built on first use)."""
    global _resolved_cache
    if _resolved_cache is None:
        _resolved_cache = _build_resolved()
    return _resolved_cache


def reload_alias_overrides() -> dict[str, str]:
    """Rebuild and re-cache the resolved alias map. Returns the new map."""
    global _resolved_cache
    _resolved_cache = _build_resolved()
    return _resolved_cache


def set_db_overlay(overlay: dict[str, str]) -> None:
    """Inject the DB-overlay layer (K.5 seam) and invalidate the cache.

    Raw entries are accepted; they are validated/normalized on the next resolve.
    """
    global _db_overlay, _resolved_cache
    _db_overlay = dict(overlay)
    _resolved_cache = None
