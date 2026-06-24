"""Tests for the triage config loader/validator (Track Q.1).

Covers the committed-config happy path, the three documented defaults
(``tier_for`` unmapped, ``predicate_strength`` unmapped, ``weak_promotion_min_docs``),
the four validation-failure modes (tier overlap, missing key, bad version,
malformed JSON), and per-path caching / read-only idempotence.
"""

import json

import pytest
from app_main.services.triage.triage_config import (
    TriageConfig,
    TriageConfigError,
    load_triage_config,
)


def _valid_config() -> dict:
    """A minimal-but-valid config mirroring the committed file's shape."""

    return {
        "version": 1,
        "entity_tiers": {
            "active": ["BeleidsThema", "Ministerie"],
            "reference": ["Budget", "Indicator"],
            "unsure_review": ["Person", "Organisatie"],
        },
        "relation_predicates": {
            "structural": ["VERSTERKT", "FINANCIERT"],
            "weak": ["RELATED"],
            "weak_promotion_min_docs": 2,
        },
        "status_rules": {
            "core_active_min_degree": 1,
            "reference_isolated_max_degree": 0,
            "well_connected_threshold": 3,
        },
    }


def _write(tmp_path, payload) -> "object":
    """Write ``payload`` (dict -> JSON, str -> verbatim) to a temp file."""

    path = tmp_path / "triage_config.json"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- Acceptance 1: committed file loads with version == 1 --------------------


def test_loads_committed_config_version_1():
    config = load_triage_config()
    assert isinstance(config, TriageConfig)
    assert config.version == 1


# --- Acceptance 2: tier_for, including the unmapped default ------------------


def test_tier_for_known_and_default():
    config = load_triage_config()
    assert config.tier_for("BeleidsThema") == "active"
    assert config.tier_for("Budget") == "reference"
    assert config.tier_for("Person") == "unsure_review"
    # Default for an unmapped type is unsure_review (not active, not dropped).
    assert config.tier_for("SomethingUnknown") == "unsure_review"
    assert config.tier_for("Unknown") == "unsure_review"


# --- Acceptance 3: predicate_strength, including the unmapped default --------


def test_predicate_strength_known_and_default():
    config = load_triage_config()
    assert config.predicate_strength("VERSTERKT") == "structural"
    assert config.predicate_strength("RELATED") == "weak"
    # Default for an unmapped predicate is weak.
    assert config.predicate_strength("UNKNOWN_PRED") == "weak"
    assert config.predicate_strength("UNKNOWN") == "weak"


# --- Acceptance 4 + weak_promotion_min_docs: status-rule accessors ----------


def test_weak_promotion_min_docs_and_status_accessors():
    config = load_triage_config()
    assert config.weak_promotion_min_docs == 2
    assert config.core_active_min_degree == 1
    assert config.reference_isolated_max_degree == 0
    assert config.well_connected_threshold == 3


# --- Validation failure: overlapping tier lists -----------------------------


def test_overlapping_tier_lists_raise(tmp_path):
    payload = _valid_config()
    payload["entity_tiers"]["reference"].append("BeleidsThema")  # also in active
    path = _write(tmp_path, payload)
    with pytest.raises(TriageConfigError, match="disjoint"):
        load_triage_config(path, reload=True)


# --- Validation failure: missing required key -------------------------------


def test_missing_required_key_raises(tmp_path):
    payload = _valid_config()
    del payload["status_rules"]
    path = _write(tmp_path, payload)
    with pytest.raises(TriageConfigError, match="status_rules"):
        load_triage_config(path, reload=True)


def test_missing_nested_required_key_raises(tmp_path):
    payload = _valid_config()
    del payload["status_rules"]["well_connected_threshold"]
    path = _write(tmp_path, payload)
    with pytest.raises(TriageConfigError, match="well_connected_threshold"):
        load_triage_config(path, reload=True)


# --- Validation failure: bad version ----------------------------------------


def test_bad_version_raises(tmp_path):
    payload = _valid_config()
    payload["version"] = 99
    path = _write(tmp_path, payload)
    with pytest.raises(TriageConfigError, match="version"):
        load_triage_config(path, reload=True)


# --- Validation failure: malformed JSON -------------------------------------


def test_malformed_json_raises(tmp_path):
    path = _write(tmp_path, "{ not valid json,,, ")
    with pytest.raises(TriageConfigError, match="not valid JSON"):
        load_triage_config(path, reload=True)


def test_missing_file_raises(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(TriageConfigError, match="not found"):
        load_triage_config(missing, reload=True)


# --- Acceptance 5: caches per path, read-only, idempotent -------------------


def test_repeated_load_is_cached_and_idempotent():
    first = load_triage_config()
    second = load_triage_config()
    # Same path -> same cached instance returned.
    assert first is second


def test_cache_keyed_per_path(tmp_path):
    path = _write(tmp_path, _valid_config())
    a = load_triage_config(path)
    b = load_triage_config(path)
    assert a is b
    # The committed default config is a different cache entry.
    assert load_triage_config() is not a


def test_loader_does_not_mutate_file(tmp_path):
    path = _write(tmp_path, _valid_config())
    before = path.read_text(encoding="utf-8")
    load_triage_config(path, reload=True)
    load_triage_config(path, reload=True)
    assert path.read_text(encoding="utf-8") == before


def test_config_is_immutable():
    config = load_triage_config()
    with pytest.raises((AttributeError, Exception)):
        config.version = 2  # type: ignore[misc]


def test_lookup_maps_are_read_only_and_cache_not_poisoned():
    """Mutating the lookup maps must fail and must not corrupt the shared cache.

    ``frozen=True`` only blocks attribute rebinding; the tier/predicate maps are
    wrapped in ``MappingProxyType`` so a stray write raises ``TypeError`` instead
    of silently re-tiering every subsequent (cached) run.
    """

    config = load_triage_config()

    # Unmapped types/predicates fall to the defaults before any tampering.
    assert config.tier_for("Person") == "unsure_review"
    assert config.predicate_strength("RELATED") == "weak"

    with pytest.raises(TypeError):
        config._tier_by_type["Person"] = "active"  # type: ignore[index]
    with pytest.raises(TypeError):
        config._strength_by_predicate["RELATED"] = "structural"  # type: ignore[index]

    # The cached instance returned to every caller is uncorrupted.
    refetched = load_triage_config()
    assert refetched is config
    assert refetched.tier_for("Person") == "unsure_review"
    assert refetched.predicate_strength("RELATED") == "weak"


def test_reload_bypasses_cache(tmp_path):
    payload = _valid_config()
    path = _write(tmp_path, payload)
    load_triage_config(path)
    # Change on disk; without reload the cache wins, with reload we re-read.
    payload["status_rules"]["well_connected_threshold"] = 5
    _write(tmp_path, payload)
    assert load_triage_config(path).well_connected_threshold == 3
    assert load_triage_config(path, reload=True).well_connected_threshold == 5


def test_non_list_tier_raises(tmp_path):
    payload = _valid_config()
    payload["entity_tiers"]["active"] = "BeleidsThema"  # should be a list
    path = _write(tmp_path, payload)
    with pytest.raises(TriageConfigError, match="must be a list"):
        load_triage_config(path, reload=True)
