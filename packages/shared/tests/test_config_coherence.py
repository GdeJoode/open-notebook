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
    collect_findings,
    raise_if_blocking,
)


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


def test_validation_without_an_ontology_blocks() -> None:
    """Reporting success for a check that never ran.

    Measured against the real `OntologyConstraintFilter`: with `ontology=None` it
    logs one DEBUG line and returns
    ``{"total_entities": 1, "valid_entities": 1, "invalid_entities": 0, …}``. A run
    with validation "on" and no ontology is therefore indistinguishable, in its own
    output, from a run that validated everything successfully. Zero effect is the
    defect this phase targets; a success report for a check that did not happen is
    the same defect with evidence attached.
    """
    findings = check_feature_dependencies(
        concept_alignment_enabled=False,
        kg_resolution_enabled=False,
        judge_enabled=False,
        judge_model_configured=True,
        ontology_validation_enabled=True,
        ontology_supplied=False,
    )
    assert [(f.code, f.severity) for f in findings] == [
        ("validation-without-ontology", BLOCK)
    ]


@pytest.mark.parametrize(
    ("enabled", "supplied"), [(False, False), (True, True), (False, True)]
)
def test_validation_is_silent_when_it_can_work_or_is_off(
    enabled: bool, supplied: bool
) -> None:
    assert (
        check_feature_dependencies(
            concept_alignment_enabled=False,
            kg_resolution_enabled=False,
            judge_enabled=False,
            judge_model_configured=True,
            ontology_validation_enabled=enabled,
            ontology_supplied=supplied,
        )
        == []
    )


# --- routing ----------------------------------------------------------------


# --- the bridge between the two systems -------------------------------------


# --- the contract every finding must keep -----------------------------------


def test_every_finding_carries_a_remedy() -> None:
    """Every finding the module can CONSTRUCT, not every one a chosen input yields.

    The first version ran one `collect_findings` call and checked its output.
    Mutation showed the hole: blanking the remedy on an exercised path failed the
    test, and blanking it on an unexercised one — `ollama-unreachable`, which that
    input cannot produce because `installed_models` is supplied — passed. Its
    docstring claimed "the next check cannot quietly ship without one", and it
    could.

    So this enumerates the module's own `Finding(...)` constructions by AST and
    asserts each passes a non-empty remedy. That holds for findings added later,
    which is what "structural" has to mean.
    """
    import ast
    import inspect

    import shared.config_coherence as module

    tree = ast.parse(inspect.getsource(module))
    constructions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "Finding"
    ]
    assert len(constructions) >= 6, (
        f"detector control: found only {len(constructions)} Finding(...) calls"
    )
    for call in constructions:
        args = list(call.args)
        remedy = args[3] if len(args) >= 4 else None
        for keyword in call.keywords:
            if keyword.arg == "remedy":
                remedy = keyword.value
        assert remedy is not None, (
            f"a Finding at line {call.lineno} is constructed without a remedy"
        )
        # A literal empty string, or a BinOp/JoinedStr, are all acceptable shapes;
        # only an unconditionally empty literal is not.
        if isinstance(remedy, ast.Constant):
            assert str(remedy.value).strip(), (
                f"the Finding at line {call.lineno} passes an empty remedy"
            )


def test_blocking_findings_sort_first() -> None:
    findings = collect_findings(
        concept_alignment_enabled=True,
        kg_resolution_enabled=False,
        judge_enabled=False,
        judge_model_configured=True,
    )
    severities = [f.severity for f in findings]
    assert severities == sorted(severities, key=lambda s: 0 if s == BLOCK else 1)
    assert severities[0] == BLOCK


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


# --- resolution refuses a retired provider ----------------------------------


def test_resolving_a_retired_provider_raises_with_the_reason() -> None:
    """Found by mutation: the gate had no test at all.

    I verified `ProviderUnavailableError` by hand in a shell and never wrote this,
    so deleting the gate left every suite green. The error is raised at RESOLUTION
    on purpose — one step later the failure is an HTTP error from a vendor, which
    cannot say the route was retired deliberately — so the message must carry the
    step, the model and the reason.
    """
    import shared.model_routing as mr
    from shared.model_routing import ProviderUnavailableError

    original = mr._config
    mr._config = {
        "defaults": {"default_privacy": "internal"},
        "providers": {
            "ollama": {"base_url": "x", "type": "ollama"},
            "gone": {"available": False, "reason": "retired for a stated reason"},
        },
        "routing": {
            "extraction": {
                "internal": {"provider": "ollama", "model": "local:1b"},
                "public": {"provider": "gone", "model": "cloud/model"},
            }
        },
    }
    try:
        # The available route still resolves — the gate must not be a blanket no.
        assert mr.get_model_config("extraction", privacy="internal")["model"] == "local:1b"

        with pytest.raises(ProviderUnavailableError) as excinfo:
            mr.get_model_config("extraction", privacy="public")

        message = str(excinfo.value)
        for expected in ("extraction", "public", "gone", "cloud/model",
                        "retired for a stated reason", "available: true"):
            assert expected in message, f"{expected!r} missing from: {message}"
    finally:
        mr._config = original


def test_an_explicit_override_still_reaches_a_retired_provider() -> None:
    """An override is a deliberate act and is not second-guessed.

    The gate exists to stop a route SILENTLY reaching a retired provider. A caller
    who names the provider in `model_override` has already decided; refusing there
    would break the escape hatch that makes restoring a provider testable.
    """
    import shared.model_routing as mr

    original = mr._config
    mr._config = {
        "defaults": {"default_privacy": "internal"},
        "providers": {"gone": {"available": False, "reason": "retired", "base_url": "x"}},
        "routing": {},
    }
    try:
        resolved = mr.get_model_config(
            "extraction", model_override={"provider": "gone", "model": "m"}
        )
        assert resolved["provider"] == "gone"
    finally:
        mr._config = original


# --- the window a model actually gets --------------------------------------


# --- the rest of the flag-on-zero-effect class ------------------------------


def test_linking_without_a_linker_blocks() -> None:
    """`linking_provider` defaults to "none", so linking alone does nothing.

    Structurally identical to the flagship case and found by review AFTER the
    phase claimed to have closed the class — which is why the refusal now sees
    the whole config rather than the one flag someone thought of.
    """
    findings = check_feature_dependencies(
        concept_alignment_enabled=False,
        kg_resolution_enabled=False,
        judge_enabled=False,
        judge_model_configured=True,
        entity_linking_enabled=True,
        entity_linker_available=False,
    )
    assert [(f.code, f.severity) for f in findings] == [
        ("linking-without-a-linker", BLOCK)
    ]


def test_outliers_without_centrality_blocks() -> None:
    """Outlier classification is a parameter OF the graph analyser.

    With centrality off the analyser is never built, so the flag is read into an
    object that does not exist.
    """
    findings = check_feature_dependencies(
        concept_alignment_enabled=False,
        kg_resolution_enabled=False,
        judge_enabled=False,
        judge_model_configured=True,
        outlier_detection_enabled=True,
        graph_centrality_enabled=False,
    )
    assert [(f.code, f.severity) for f in findings] == [
        ("outliers-without-centrality", BLOCK)
    ]


@pytest.mark.parametrize(
    ("kwargs"),
    [
        {"entity_linking_enabled": True, "entity_linker_available": True},
        {"entity_linking_enabled": False, "entity_linker_available": False},
        {"outlier_detection_enabled": True, "graph_centrality_enabled": True},
        {"outlier_detection_enabled": False, "graph_centrality_enabled": False},
    ],
)
def test_the_new_dependencies_are_silent_when_they_can_work(kwargs) -> None:
    """The counterweight for both, in both directions.

    A check that refuses a working configuration gets disabled, and one that
    refuses a DISABLED feature refuses every default startup.
    """
    assert (
        check_feature_dependencies(
            concept_alignment_enabled=False,
            kg_resolution_enabled=False,
            judge_enabled=False,
            judge_model_configured=True,
            **kwargs,
        )
        == []
    )


# --- the surfaces a human reads ---------------------------------------------


def test_the_routing_summary_marks_a_retired_route() -> None:
    """The only human-readable routing surface must not contradict the resolver.

    `/routing` on the extraction service and the app's services proxy both serve
    this. It advertised `extraction/public → nvidia:…` exactly as it advertises a
    live route, while `get_model_config` raises on that same route. A summary that
    disagrees with the resolver is worse than no summary.
    """
    import shared.model_routing as mr

    original = mr._config
    mr._config = {
        "defaults": {"default_privacy": "internal"},
        "providers": {
            "ollama": {"base_url": "x"},
            "gone": {"available": False, "reason": "retired for a stated reason"},
        },
        "routing": {
            "extraction": {
                "internal": {"provider": "ollama", "model": "local:1b"},
                "public": {"provider": "gone", "model": "cloud/model"},
            }
        },
    }
    try:
        summary = mr.get_routing_summary()
        assert summary["routing"]["extraction"]["public"].endswith("[UNAVAILABLE]")
        assert "[UNAVAILABLE]" not in summary["routing"]["extraction"]["internal"]
        assert summary["unavailable_providers"] == {
            "gone": "retired for a stated reason"
        }
    finally:
        mr._config = original


def test_a_summary_with_no_retirements_carries_no_marker() -> None:
    """The counterweight — the marker must mean something when it appears."""
    import shared.model_routing as mr

    original = mr._config
    mr._config = {
        "defaults": {"default_privacy": "internal"},
        "providers": {"ollama": {"base_url": "x"}},
        "routing": {"extraction": {"internal": {"provider": "ollama", "model": "m"}}},
    }
    try:
        summary = mr.get_routing_summary()
        assert summary["unavailable_providers"] == {}
        assert "[UNAVAILABLE]" not in summary["routing"]["extraction"]["internal"]
    finally:
        mr._config = original
