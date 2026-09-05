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
    check_ollama_context,
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


def test_a_missing_model_blocks_only_on_a_step_every_run_exercises() -> None:
    """`extraction` refuses; `vlm` warns. Both are the same missing model.

    A partially-pulled Ollama used to hard-refuse startup for a step the operator
    may never touch, and the failure mode was worst in the COMMON case: no Ollama
    at all degrades to one WARN, while Ollama with *some* models refused. That is
    the over-reach `test_a_disabled_feature_never_blocks` refuses for feature
    flags, applied to routes.
    """
    from shared.config_coherence import REQUIRED_STEPS

    assert "extraction" in REQUIRED_STEPS and "vlm" not in REQUIRED_STEPS

    required = check_routing(_routing(model="not-pulled:9b"), installed={"llama3.1:8b"})
    blocking = [f for f in required if f.severity == BLOCK]
    assert [f.code for f in blocking] == ["model-not-installed"]
    assert "ollama pull not-pulled:9b" in blocking[0].remedy

    optional = dict(_routing())
    optional["routing"] = {
        "vlm": {"internal": {"provider": "ollama", "model": "not-pulled:9b"}}
    }
    findings = check_routing(optional, installed={"llama3.1:8b"})
    assert [(f.code, f.severity) for f in findings] == [("model-not-installed", WARN)]
    assert "not exercised by every run" in findings[0].remedy


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
            "ollama": {"base_url": "http://localhost:11434", "type": "ollama"},
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


def test_a_model_that_bakes_less_than_it_promises_is_reported() -> None:
    """Measured through the real esperanto factory before this was written:

        in   config={"num_ctx": 16384, "temperature": 0.1, "max_tokens": 8}
        out  {"options": {"num_predict": 8, "temperature": 0.1, "top_p": 0.9}}

    `num_ctx` appears zero times in the whole esperanto package, so on that path
    the Modelfile is the only thing that sets the window. A model row promising
    32768 against a model that bakes nothing means the packer sizes prompts the
    runtime truncates — from the HEAD, which is where the document content is.
    """
    findings = check_ollama_context(
        {"bare:8b": 32768}, probe=lambda _url, _model: None
    )
    assert [(f.code, f.severity) for f in findings] == [
        ("ollama-context-not-honoured", WARN)
    ]
    assert "4096" in findings[0].message
    assert "PARAMETER num_ctx 32768" in findings[0].remedy


def test_a_model_that_bakes_enough_is_silent() -> None:
    """The counterweight — and the case that proves the remedy works.

    A `-ctx16k` variant baking 16384 against a row declaring 16384 must produce
    nothing, or the check would tell an operator to do what they have already
    done.
    """
    assert check_ollama_context(
        {"m:8b-ctx16k": 16384}, probe=lambda _url, _model: 16384
    ) == []
    # And a row declaring no more than the runtime default needs no variant.
    assert check_ollama_context(
        {"small:3b": 4096}, probe=lambda _url, _model: None
    ) == []


def test_an_unreachable_runtime_reports_nothing_here() -> None:
    """`_baked_num_ctx` returns None both for "bakes nothing" and "cannot ask".

    Conflating them would turn a stopped Ollama into a warning per model row. The
    routing check already reports an unreachable runtime once; this one must not
    repeat it, so a row at or below the default stays silent either way.
    """
    assert check_ollama_context({"m:8b": 4096}, probe=lambda _url, _model: None) == []


def test_rows_without_a_declared_window_are_skipped() -> None:
    assert check_ollama_context({"m:8b": 0}, probe=lambda _u, _m: None) == []
    assert check_ollama_context({"": 32768}, probe=lambda _u, _m: None) == []


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
