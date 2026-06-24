"""Typed, validated, cached loader for ``config/triage_config.json`` (Track Q).

The triage config is the single source of truth for entity tiering, relation
predicate strength, and status-rule constants. It is version-controlled and
decoupled from pipeline logic: the pipeline reads it once per run and never
re-asks. This module turns the raw JSON into a frozen :class:`TriageConfig`
with constant-time lookups, validates it up front (so a malformed config fails
loudly rather than silently defaulting), and caches one instance per resolved
path.

Convention: this mirrors the dataclass-based config pattern used by
``entity_filtering.config.FilteringConfig`` (the plan's named reuse anchor),
not the pydantic settings pattern, because the triage config is static data
loaded from a committed JSON file rather than environment-driven settings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Literal, Mapping

from loguru import logger

# Tier of an entity, keyed on its ``primary_type``.
Tier = Literal["active", "reference", "unsure_review"]
# Base strength of a relation, keyed on its predicate.
PredicateStrength = Literal["structural", "weak"]

# Schema version this loader understands. Bumping the config schema in a
# backward-incompatible way must bump this and the file in lock-step.
SUPPORTED_VERSION = 1

# Defaults declared in the config file itself; used when a key/value is
# unmapped, matching the ``_default_for_unmapped_*`` markers in the JSON.
_DEFAULT_TIER: Tier = "unsure_review"
_DEFAULT_PREDICATE_STRENGTH: PredicateStrength = "weak"
_DEFAULT_WEAK_PROMOTION_MIN_DOCS = 2


class TriageConfigError(ValueError):
    """Raised when the triage config is missing, malformed, or inconsistent.

    Subclasses :class:`ValueError` so callers can treat it as a plain
    validation error while still catching it specifically when they care.
    """


def _default_config_path() -> Path:
    """Resolve the committed ``config/triage_config.json`` at the repo root.

    From this file (``apps/app-main/src/app_main/services/triage/``) the repo
    root is six parents up. Resolved explicitly rather than via CWD so the
    loader works regardless of where a process is launched.
    """

    return Path(__file__).resolve().parents[6] / "config" / "triage_config.json"


@dataclass(frozen=True)
class TriageConfig:
    """Immutable, validated view of the triage configuration.

    Lookups are backed by pre-built reverse maps (type/predicate -> tier/
    strength) so ``tier_for`` / ``predicate_strength`` are O(1) and never
    mutate the underlying file. Instances are frozen AND the lookup maps are
    wrapped in ``MappingProxyType`` (read-only views), so neither attribute
    rebinding nor mutating a tier/predicate map can corrupt the shared cached
    instance — a stray write raises ``TypeError`` instead of silently
    re-tiering every downstream run.
    """

    version: int
    _tier_by_type: Mapping[str, Tier] = field(repr=False)
    _strength_by_predicate: Mapping[str, PredicateStrength] = field(repr=False)
    weak_promotion_min_docs: int
    core_active_min_degree: int
    reference_isolated_max_degree: int
    well_connected_threshold: int

    def tier_for(self, primary_type: str) -> Tier:
        """Return the tier for ``primary_type``; ``unsure_review`` if unmapped.

        Defaulting (rather than raising) is intentional: a type the operator
        has not yet tiered should land in the human-review tier, never be
        silently dropped or treated as core.
        """

        return self._tier_by_type.get(primary_type, _DEFAULT_TIER)

    def predicate_strength(self, predicate: str) -> PredicateStrength:
        """Return the base strength for ``predicate``; ``weak`` if unmapped.

        Unknown predicates default to ``weak`` so they require cross-document
        recurrence to count toward connectivity, matching the config's
        ``_default_for_unmapped_predicate``.
        """

        return self._strength_by_predicate.get(predicate, _DEFAULT_PREDICATE_STRENGTH)


# Cache one validated instance per resolved config path. The loader is
# read-only, so sharing a frozen instance across callers is safe and avoids
# re-reading/re-validating the file on every pipeline step.
_CACHE: Dict[Path, TriageConfig] = {}


def _require(mapping: Dict[str, object], key: str, where: str) -> object:
    """Fetch ``key`` from ``mapping`` or raise a clear :class:`TriageConfigError`."""

    if key not in mapping:
        raise TriageConfigError(f"missing required key '{key}' in {where}")
    return mapping[key]


def _require_int(mapping: Dict[str, object], key: str, where: str) -> int:
    """Fetch ``key`` and assert it is a non-bool integer, else raise."""

    value = _require(mapping, key, where)
    # ``bool`` is a subclass of ``int``; reject it explicitly so ``true`` in
    # the JSON cannot masquerade as a degree threshold.
    if not isinstance(value, int) or isinstance(value, bool):
        raise TriageConfigError(
            f"{where}['{key}'] must be an integer, got {type(value).__name__}"
        )
    return value


def _build_reverse_map(
    tiers: Dict[str, object],
) -> Dict[str, Tier]:
    """Invert the tier -> [types] lists into a type -> tier map.

    Validates that the three tier lists are disjoint: a type appearing in more
    than one tier is ambiguous and must fail loudly rather than resolve to
    whichever list happened to be processed last.
    """

    reverse: Dict[str, Tier] = {}
    for tier in ("active", "reference", "unsure_review"):
        members = _require(tiers, tier, "entity_tiers")
        if not isinstance(members, list):
            raise TriageConfigError(
                f"entity_tiers['{tier}'] must be a list, got {type(members).__name__}"
            )
        for primary_type in members:
            if primary_type in reverse:
                raise TriageConfigError(
                    f"type '{primary_type}' appears in multiple tiers "
                    f"('{reverse[primary_type]}' and '{tier}'); tier lists must be disjoint"
                )
            reverse[primary_type] = tier  # type: ignore[assignment]
    return reverse


def _build_predicate_map(
    predicates: Dict[str, object],
) -> Dict[str, PredicateStrength]:
    """Invert the structural/weak predicate lists into a predicate -> strength map."""

    reverse: Dict[str, PredicateStrength] = {}
    for strength in ("structural", "weak"):
        members = _require(predicates, strength, "relation_predicates")
        if not isinstance(members, list):
            raise TriageConfigError(
                f"relation_predicates['{strength}'] must be a list, "
                f"got {type(members).__name__}"
            )
        for predicate in members:
            if predicate in reverse:
                raise TriageConfigError(
                    f"predicate '{predicate}' appears in multiple strength lists "
                    f"('{reverse[predicate]}' and '{strength}'); they must be disjoint"
                )
            reverse[predicate] = strength  # type: ignore[assignment]
    return reverse


def _parse(raw: Dict[str, object]) -> TriageConfig:
    """Validate a parsed JSON object and build a :class:`TriageConfig`."""

    version = _require(raw, "version", "triage config")
    if version != SUPPORTED_VERSION:
        raise TriageConfigError(
            f"unsupported triage config version {version!r}; "
            f"expected {SUPPORTED_VERSION}"
        )
    # Past the equality guard ``version`` is ``SUPPORTED_VERSION`` (an int);
    # narrow the static type for the constructor call below.
    version = SUPPORTED_VERSION

    entity_tiers = _require(raw, "entity_tiers", "triage config")
    relation_predicates = _require(raw, "relation_predicates", "triage config")
    status_rules = _require(raw, "status_rules", "triage config")
    if not isinstance(entity_tiers, dict):
        raise TriageConfigError("'entity_tiers' must be an object")
    if not isinstance(relation_predicates, dict):
        raise TriageConfigError("'relation_predicates' must be an object")
    if not isinstance(status_rules, dict):
        raise TriageConfigError("'status_rules' must be an object")

    tier_by_type = _build_reverse_map(entity_tiers)
    strength_by_predicate = _build_predicate_map(relation_predicates)

    # weak_promotion_min_docs is optional in the file; default when absent but
    # validate the type when present so a stray string fails loudly.
    raw_min_docs = relation_predicates.get(
        "weak_promotion_min_docs", _DEFAULT_WEAK_PROMOTION_MIN_DOCS
    )
    if not isinstance(raw_min_docs, int) or isinstance(raw_min_docs, bool):
        raise TriageConfigError(
            "relation_predicates['weak_promotion_min_docs'] must be an integer"
        )
    weak_promotion_min_docs: int = raw_min_docs

    return TriageConfig(
        version=version,
        _tier_by_type=MappingProxyType(tier_by_type),
        _strength_by_predicate=MappingProxyType(strength_by_predicate),
        weak_promotion_min_docs=weak_promotion_min_docs,
        core_active_min_degree=_require_int(
            status_rules, "core_active_min_degree", "status_rules"
        ),
        reference_isolated_max_degree=_require_int(
            status_rules, "reference_isolated_max_degree", "status_rules"
        ),
        well_connected_threshold=_require_int(
            status_rules, "well_connected_threshold", "status_rules"
        ),
    )


def load_triage_config(
    path: Path | str | None = None, *, reload: bool = False
) -> TriageConfig:
    """Load, validate, and cache the triage config for ``path``.

    Args:
        path: Override config location; defaults to the committed
            ``config/triage_config.json`` at the repo root.
        reload: Bypass the per-path cache and re-read from disk. Useful in
            tests; the production pipeline keeps the cached instance.

    Returns:
        A frozen, validated :class:`TriageConfig`.

    Raises:
        TriageConfigError: If the file is missing, is not valid JSON, has an
            unsupported version, is missing a required key, or has overlapping
            tier/predicate lists.

    The loader never writes to the config file.
    """

    resolved = (Path(path) if path is not None else _default_config_path()).resolve()

    if not reload and resolved in _CACHE:
        return _CACHE[resolved]

    try:
        text = resolved.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise TriageConfigError(f"triage config not found at {resolved}") from exc
    except OSError as exc:  # pragma: no cover - unexpected IO failure
        raise TriageConfigError(f"could not read triage config at {resolved}: {exc}") from exc

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TriageConfigError(
            f"triage config at {resolved} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise TriageConfigError(
            f"triage config at {resolved} must be a JSON object, got {type(raw).__name__}"
        )

    config = _parse(raw)
    _CACHE[resolved] = config
    logger.debug(
        "Loaded triage config v{} from {} ({} tiered types, {} typed predicates)",
        config.version,
        resolved,
        len(config._tier_by_type),
        len(config._strength_by_predicate),
    )
    return config
