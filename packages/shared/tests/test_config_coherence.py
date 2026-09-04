"""Configuration that expresses one intent (Track PC.6).

The acceptance criterion is negative: the state "flag on, zero effect, one
warning" must be **unreachable**. So most of these tests assert that something
REFUSES, and the ones that assert it does not refuse are there to stop the check
becoming a blanket no.
"""

from __future__ import annotations

import pytest

from shared.config_coherence import (
    BLOCK,
    WARN,
    ConfigurationError,
    Finding,
    check_feature_dependencies,
    check_privacy_defaults,
    check_routing,
    collect_findings,
    raise_if_blocking,
)


def _routing(*, available=True, model="llama3.1:8b-instruct-q4_0", default="internal"):
    return {
        "defaults": {"default_privacy": default},
        "providers": {
            "ollama": {"base_url": "http://localhost:11434"},
            "nvidia": {"available": available, "reason": "retired"},
        },
        "routing": {
            "extraction": {
                "internal": {"provider": "ollama", "model": model},
                "public": {"provider": "nvidia", "model": "some/cloud-model"},
            }
        },
    }


# --- the flagship dependency ------------------------------------------------


def test_alignment_without_resolution_blocks() -> None:
    """The measured case: a flag whose effect needs a second, unrelated flag.

    Concept alignment classifies the entities KG resolution marks `is_new`. With
    resolution off nothing is marked, so alignment runs over an empty list and
    reports success. The workflow already logged a warning about this; the
    warning IS the failure this phase names, not the fix.
    """
    findings = check_feature_dependencies(
        concept_alignment_enabled=True,
        kg_resolution_enabled=False,
        judge_enabled=False,
        judge_model_configured=True,
    )
    assert [f.severity for f in findings] == [BLOCK]
    assert findings[0].code == "alignment-without-resolution"


def test_both_enabled_is_not_a_finding() -> None:
    """The counterweight: the check must not refuse a working configuration."""
    assert (
        check_feature_dependencies(
            concept_alignment_enabled=True,
            kg_resolution_enabled=True,
            judge_enabled=False,
            judge_model_configured=True,
        )
        == []
    )


def test_a_disabled_feature_never_blocks() -> None:
    """An unused flag's dependencies are not this check's business.

    Without this, `default_models` being empty — a legitimate "not configured
    yet", since models for summaries and transformations arrive later — would
    refuse every startup.
    """
    assert (
        check_feature_dependencies(
            concept_alignment_enabled=False,
            kg_resolution_enabled=False,
            judge_enabled=True,
            judge_model_configured=False,
        )
        == []
    )


def test_the_judge_without_a_model_blocks_only_when_alignment_runs() -> None:
    on = check_feature_dependencies(
        concept_alignment_enabled=True,
        kg_resolution_enabled=True,
        judge_enabled=True,
        judge_model_configured=False,
    )
    assert [f.code for f in on] == ["judge-without-model"]
    assert on[0].severity == BLOCK

    off = check_feature_dependencies(
        concept_alignment_enabled=True,
        kg_resolution_enabled=True,
        judge_enabled=False,
        judge_model_configured=False,
    )
    assert off == []


# --- routing ----------------------------------------------------------------


def test_a_retired_provider_blocks_only_on_the_route_that_is_taken() -> None:
    """Severity follows reachability, which is why the routes were kept.

    A `public` route to a retired provider is a record of what the cloud path
    was; with `default_privacy: internal` nothing selects it. Blocking on it
    would force deletion of exactly the history the decision chose to keep.
    """
    findings = check_routing(_routing(available=False), installed={"llama3.1:8b-instruct-q4_0"})
    by_code = {(f.code, f.severity) for f in findings}
    assert ("provider-unavailable", WARN) in by_code
    assert ("provider-unavailable", BLOCK) not in by_code

    # Flip the default so the retired route IS the one taken.
    taken = check_routing(
        _routing(available=False, default="public"),
        installed={"llama3.1:8b-instruct-q4_0"},
    )
    assert ("provider-unavailable", BLOCK) in {(f.code, f.severity) for f in taken}


def test_a_routed_local_model_that_is_not_pulled_blocks() -> None:
    findings = check_routing(_routing(model="not-pulled:9b"), installed={"llama3.1:8b"})
    blocking = [f for f in findings if f.severity == BLOCK]
    assert [f.code for f in blocking] == ["model-not-installed"]
    assert "ollama pull not-pulled:9b" in blocking[0].remedy


def test_an_implicit_latest_tag_counts_as_installed() -> None:
    """`granite3.2-vision` routed, `granite3.2-vision:latest` pulled — one model.

    A real case in this repo's config. Treating them as different produced a
    false "not installed" on a model that is present.
    """
    cfg = _routing(model="granite3.2-vision")
    installed = {"granite3.2-vision:latest", "granite3.2-vision"}
    assert [f for f in check_routing(cfg, installed=installed) if f.severity == BLOCK] == []


def test_an_unreachable_runtime_is_one_warning_not_a_cascade() -> None:
    """None and empty-set are different answers.

    A daemon that is down must not be reported as "every routed model is
    missing" — that is a false alarm of exactly the kind this module exists to
    prevent, and it would train a reader to ignore the output.
    """
    findings = check_routing(_routing(), installed=None)
    assert [f.code for f in findings] == ["ollama-unreachable"]
    assert all(f.severity == WARN for f in findings)


# --- the bridge between the two systems -------------------------------------


def test_opposite_privacy_defaults_are_reported() -> None:
    assert [f.code for f in check_privacy_defaults(
        routing_default="internal", resolver_default="CLOUD"
    )] == ["privacy-default-mismatch"]


@pytest.mark.parametrize(
    ("routing_default", "resolver_default"),
    [("internal", "PRIVATE"), ("confidential", "private"), ("public", "CLOUD")],
)
def test_agreeing_defaults_are_silent(routing_default: str, resolver_default: str) -> None:
    assert check_privacy_defaults(
        routing_default=routing_default, resolver_default=resolver_default
    ) == []


# --- the contract every finding must keep -----------------------------------


def test_every_finding_carries_a_remedy() -> None:
    """A finding without a fix is the log line this phase exists to abolish.

    Structural rather than per-case: it holds for findings added later, which is
    the point — the next check cannot quietly ship without one.
    """
    findings = collect_findings(
        routing_config=_routing(available=False, model="not-pulled:9b"),
        concept_alignment_enabled=True,
        kg_resolution_enabled=False,
        judge_enabled=True,
        judge_model_configured=False,
        resolver_default_privacy="CLOUD",
        installed_models={"llama3.1:8b"},
    )
    assert len(findings) >= 4
    for f in findings:
        assert f.remedy.strip(), f"{f.code} reports a problem with no remedy"
        assert f.severity in (BLOCK, WARN)


def test_blocking_findings_sort_first() -> None:
    findings = collect_findings(
        routing_config=_routing(available=False),
        concept_alignment_enabled=True,
        kg_resolution_enabled=False,
        judge_enabled=False,
        judge_model_configured=True,
        resolver_default_privacy="CLOUD",
        installed_models={"llama3.1:8b-instruct-q4_0"},
    )
    severities = [f.severity for f in findings]
    assert severities == sorted(severities, key=lambda s: 0 if s == BLOCK else 1)
    assert severities[0] == BLOCK


def test_the_probe_is_not_called_when_models_are_supplied() -> None:
    """`installed_models` must short-circuit the network probe.

    Otherwise every test and every startup pays a socket timeout, and the check
    becomes something people disable.
    """
    def _explode(_base_url):  # pragma: no cover - must not run
        raise AssertionError("the Ollama probe was called despite installed_models")

    collect_findings(
        routing_config=_routing(),
        concept_alignment_enabled=False,
        kg_resolution_enabled=False,
        judge_enabled=False,
        judge_model_configured=True,
        resolver_default_privacy="PRIVATE",
        installed_models={"llama3.1:8b-instruct-q4_0"},
        ollama_probe=_explode,
    )


# --- refusal ----------------------------------------------------------------


def test_raise_if_blocking_refuses_and_names_every_reason() -> None:
    findings = [
        Finding(BLOCK, "a", "first problem", "fix a"),
        Finding(WARN, "b", "cosmetic", "fix b"),
        Finding(BLOCK, "c", "second problem", "fix c"),
    ]
    with pytest.raises(ConfigurationError) as excinfo:
        raise_if_blocking(findings)
    assert {f.code for f in excinfo.value.findings} == {"a", "c"}
    text = str(excinfo.value)
    assert "first problem" in text and "second problem" in text
    assert "fix a" in text and "fix c" in text
    assert "cosmetic" not in text


def test_warnings_alone_do_not_refuse() -> None:
    raise_if_blocking([Finding(WARN, "w", "surprising", "consider x")])
