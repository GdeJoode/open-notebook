"""Does this configuration express one intent? (Track PC.6)

The finding this module answers: `ENABLE_CONCEPT_ALIGNMENT=true` did nothing in a
measured run, because alignment classifies only the entities KG resolution marks
``is_new`` and KG resolution is off by default. A feature reachable only by
changing a second, unrelated default is not a feature; it is a flag that reads as
one. The phase's acceptance criterion is that **enabling a feature either works or
says why it cannot**, and that the "flag on, zero effect, one warning" state is
unreachable rather than merely logged.

**Two configuration systems, and PC.6 does not merge them.** This repository
routes models in two places, and the split is deliberate:

* ``model_routing.yaml`` — pipeline steps (extraction, classification,
  summarization, embedding, vlm, transcription, parsing), keyed by step × privacy
  level. Authoritative for **what the pipeline runs**.
* the ``default_models`` row plus ``route_resolver`` — app roles (chat, judge,
  transformations, tools), keyed by ``LLMTask`` × ``PrivacyMode``. Authoritative
  for **what the app offers**.

Merging them was considered and rejected: it touches both code paths, the
``/api/models/defaults`` surface and the J.4 telemetry, for no behaviour a user
asked for. What was missing is not one system — it is a check that the two do not
contradict each other. ``default_models`` being EMPTY is a legitimate state, not a
defect: models for summaries and transformations are added later, and an empty
row must read as "not configured yet" rather than resolve to a guess.

**Severities.** A :data:`BLOCK` finding means a configured feature cannot do its
job, and the caller is expected to refuse. A :data:`WARN` means the configuration
is surprising but functional. Nothing here writes; the caller decides.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

BLOCK = "block"
WARN = "warn"

#: Steps with a real consumer that every run exercises, so a missing model there
#: stops work outright. Everything else is reached only by a caller that asks for
#: it, and an operator who never touches the VLM path must still be able to boot:
#: refusing over a model they will never load is the over-reach
#: `check_feature_dependencies` refuses to commit, and it is worst in the COMMON
#: case, since no Ollama at all degrades to one WARN while Ollama with SOME models
#: would refuse.
#:
#: `embedding` was in this set and is REMOVED. `shared.model_routing.embed_text`
#: has no production caller — `semantic_intelligence` defines its own — so
#: `routing.embedding.any` is read by nothing, and gating startup on it was right
#: about the model by coincidence and wrong about the mechanism: re-point that
#: route and the verdict changes while nothing at runtime does; change
#: `default_embedding_model`, which is the real lever, and the check never
#: notices. For a phase arguing that configuration must express one intent, a
#: gate on an entry nothing reads is the wrong lever even when it names the right
#: model.
#:
#: Membership requires a named consumer. `extraction` has one at
#: `services/extraction/api.py`.
REQUIRED_STEPS = frozenset({"extraction"})


@dataclass(frozen=True)
class Finding:
    """One thing wrong with the configuration, and what to do about it.

    ``remedy`` is mandatory and is not decoration: a check that reports a problem
    without naming a fix produces exactly the log line PC.6 exists to abolish.
    """

    severity: str
    code: str
    message: str
    remedy: str

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.code}: {self.message} → {self.remedy}"


class ConfigurationError(RuntimeError):
    """Raised when a BLOCK finding survives to startup.

    Carries the findings so a caller can render them rather than re-deriving them
    from the message.
    """

    def __init__(self, findings: List[Finding]) -> None:
        self.findings = list(findings)
        detail = "\n  ".join(str(f) for f in self.findings)
        super().__init__(
            f"{len(self.findings)} configuration problem(s) would leave a feature "
            f"unable to do its job:\n  {detail}"
        )


def _installed_ollama_models(base_url: str, timeout: float = 5.0) -> Optional[set]:
    """Tags currently pulled, or None when Ollama could not be reached.

    None and empty-set are different answers and must not be conflated: an
    unreachable runtime is not the same as one holding no models, and reporting
    "every routed model is missing" because the daemon is down would be a false
    alarm of exactly the kind this module is meant to prevent.
    """
    try:
        with urllib.request.urlopen(
            f"{base_url.rstrip('/')}/api/tags", timeout=timeout
        ) as fh:
            payload = json.load(fh)
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    names = {m.get("name", "") for m in payload.get("models", [])}
    # `granite3.2-vision` in a route and `granite3.2-vision:latest` installed are
    # the same model; Ollama appends the implicit tag.
    return {n for n in names if n} | {n.split(":")[0] for n in names if n}


def check_routing(
    routing_config: Dict[str, Any],
    *,
    installed: Optional[set] = None,
) -> List[Finding]:
    """Every routed model resolves, and no route points at a retired provider.

    ``installed`` is the set of available local model tags, or None to skip the
    existence check (an unreachable Ollama is reported as a WARN, not as a
    cascade of missing models).
    """
    findings: List[Finding] = []
    providers = routing_config.get("providers", {})
    routing = routing_config.get("routing", {})
    default_privacy = str(
        routing_config.get("defaults", {}).get("default_privacy", "internal")
    )

    if installed is None:
        findings.append(
            Finding(
                WARN,
                "ollama-unreachable",
                "the local model runtime did not answer, so routed models could "
                "not be checked for existence",
                "start Ollama, or ignore this if no local route is used",
            )
        )

    for step, by_privacy in routing.items():
        for privacy, route in by_privacy.items():
            provider = route.get("provider", "")
            model = route.get("model", "")
            conf = providers.get(provider, {})

            if conf.get("available") is False:
                # Only a BLOCK when this is the route that will actually be
                # taken. A retired route the default privacy never selects is a
                # record, not a fault — that is why the routes were kept.
                reached = privacy in (default_privacy, "any")
                findings.append(
                    Finding(
                        BLOCK if reached else WARN,
                        "provider-unavailable",
                        f"{step}/{privacy} routes to {provider!r}, declared "
                        f"unavailable: {str(conf.get('reason', '')).strip()}",
                        f"route {step}/{privacy} elsewhere"
                        if reached
                        else f"none needed — default_privacy is {default_privacy!r}, "
                        f"so this route is not selected",
                    )
                )
                continue

            if provider == "ollama" and model and installed is not None:
                if model not in installed:
                    reached = privacy in (default_privacy, "any")
                    required = step in REQUIRED_STEPS
                    findings.append(
                        Finding(
                            BLOCK if (reached and required) else WARN,
                            "model-not-installed",
                            f"{step}/{privacy} routes to local model {model!r}, "
                            f"which is not pulled",
                            f"`ollama pull {model}`"
                            + (
                                ", or change the route"
                                if required
                                else f" before using the {step} path, or change "
                                f"the route — {step} is not exercised by every run"
                            ),
                        )
                    )
    return findings


#: What Ollama allocates when nothing asks for more. A model trained for 32k runs
#: at this unless the request or the Modelfile says otherwise.
OLLAMA_DEFAULT_NUM_CTX = 4096


def _baked_num_ctx(base_url: str, model: str, timeout: float = 5.0) -> Optional[int]:
    """The `num_ctx` a model's Modelfile bakes in, or None if it bakes none."""
    try:
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/show",
            data=json.dumps({"name": model}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as fh:
            payload = json.load(fh)
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    for line in str(payload.get("parameters", "")).splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "num_ctx" and parts[1].isdigit():
            return int(parts[1])
    return None


def check_ollama_context(
    declared: Dict[str, int],
    *,
    base_url: str = "",
    probe: Callable[[str, str], Optional[int]] = _baked_num_ctx,
) -> List[Finding]:
    """A model whose runtime window is smaller than the one its config promises.

    ``declared`` maps an Ollama model name to the context the configuration says
    it has — today, the ``context_window`` on a model row, which is what
    ``llm_call``'s packer sizes prompts against.

    Two callers reach Ollama and they are NOT equivalent:

    * ``shared/model_routing._call_ollama`` posts directly and **does** send
      ``num_ctx``, so a YAML step's declared window is what the runtime allocates.
      Those steps are not this check's business.
    * esperanto, behind ``llm_call``'s ``RoutedLLMCaller``, **cannot**:
      ``providers/llm/base.py::get_completion_kwargs`` returns only
      max_tokens/temperature/top_p/streaming, and ``num_ctx`` appears zero times
      in the package. Measured through the real factory —
      ``config={"num_ctx": 16384, …}`` in, ``{"options": {"num_predict": …,
      "temperature": …, "top_p": …}}`` out.

    So on that path the ONLY thing setting the window is a Modelfile
    ``PARAMETER num_ctx``. Where a model row declares more than the model bakes,
    the packer sizes prompts the runtime silently truncates — from the HEAD,
    which is where the document content is.

    WARN rather than BLOCK: the run still produces output, and refusing startup
    over a model row a user is midway through configuring is the over-reach
    ``REQUIRED_STEPS`` exists to avoid.
    """
    findings: List[Finding] = []
    url = base_url or os.getenv("OLLAMA_URL") or "http://localhost:11434"
    for model, window in sorted(declared.items()):
        if not model or not window:
            continue
        baked = probe(url, model)
        effective = baked if baked is not None else OLLAMA_DEFAULT_NUM_CTX
        if effective < window:
            findings.append(
                Finding(
                    WARN,
                    "ollama-context-not-honoured",
                    f"model {model!r} is configured with a context window of "
                    f"{window}, but reaches Ollama through esperanto, which "
                    f"cannot send num_ctx — the model bakes "
                    + (str(baked) if baked is not None else "nothing")
                    + f", so the runtime allocates {effective} and longer prompts "
                    f"are truncated from the HEAD",
                    f"register a variant with `PARAMETER num_ctx {window}` and "
                    f"point the model row at it, or lower the row's "
                    f"context_window to {effective} so the configuration stops "
                    f"promising a window the runtime will not give",
                )
            )
    return findings


def check_feature_dependencies(
    *,
    concept_alignment_enabled: bool,
    kg_resolution_enabled: bool,
    judge_enabled: bool,
    judge_model_configured: bool,
    ontology_validation_enabled: bool = False,
    ontology_supplied: bool = True,
    entity_linking_enabled: bool = False,
    entity_linker_available: bool = True,
    semantic_blocking_enabled: bool = False,
    llm_matcher_enabled: bool = True,
    outlier_detection_enabled: bool = False,
    graph_centrality_enabled: bool = True,
) -> List[Finding]:
    """Flags that need a second, unrelated flag to do anything.

    The flagship case, measured: concept alignment classifies the entities KG
    resolution marks ``is_new``. With KG resolution off, nothing is marked, so
    alignment runs over an empty list and reports success. The workflow already
    logs a warning about this; a warning is what the acceptance criterion calls
    the failure, not the fix.
    """
    findings: List[Finding] = []

    if concept_alignment_enabled and not kg_resolution_enabled:
        findings.append(
            Finding(
                BLOCK,
                "alignment-without-resolution",
                "concept alignment is enabled but KG resolution is not; alignment "
                "classifies the entities KG resolution marks `is_new`, so with "
                "resolution off nothing is marked and nothing is classified",
                "enable kg_resolution, or disable concept alignment "
                "(ENABLE_CONCEPT_ALIGNMENT=false)",
            )
        )

    if concept_alignment_enabled and judge_enabled and not judge_model_configured:
        findings.append(
            Finding(
                BLOCK,
                "judge-without-model",
                "the alignment judge is enabled but no chat model is configured "
                "for it, so every ambiguous RELATED/NOVEL decision falls back "
                "silently instead of being judged",
                "set default_chat_model, or disable the judge "
                "(ConceptAlignmentConfig.judge_enabled=False)",
            )
        )

    if ontology_validation_enabled and not ontology_supplied:
        # The sharpest instance of this phase's target. `OntologyConstraintFilter`
        # with `ontology=None` does not merely skip — it returns a report saying
        # every entity and relation is VALID:
        #     {"total_entities": 1, "valid_entities": 1, "invalid_entities": 0, …}
        # so a run with validation "on" and no ontology is indistinguishable in
        # its own output from a run that validated everything successfully. The
        # skip is a DEBUG line. Zero effect is bad; reporting success for a check
        # that never happened is worse.
        findings.append(
            Finding(
                BLOCK,
                "validation-without-ontology",
                "ontology validation is enabled but no ontology is supplied; the "
                "filter skips silently and still reports every entity and "
                "relation as valid, so the run cannot be told apart from one that "
                "actually validated",
                "supply an ontology to the workflow, or disable "
                "ontology_validation (OntologyValidationConfig.enabled=False)",
            )
        )

    if entity_linking_enabled and not entity_linker_available:
        # Structurally identical to the flagship case, and found by review after
        # the phase claimed to have closed the class. `linking_provider` defaults
        # to "none", so enabling entity linking without also changing that second,
        # unrelated setting leaves `_entity_linker` as None and stage 8 never runs.
        findings.append(
            Finding(
                BLOCK,
                "linking-without-a-linker",
                "entity linking is enabled but no linker can be built; "
                "`linking_provider` is not a provider that resolves and none was "
                "injected, so the linking stage never runs",
                "set semantic.linking_provider to a real provider (e.g. "
                "'dbpedia_spotlight'), inject a linker, or disable "
                "semantic.entity_linking_enabled",
            )
        )

    if semantic_blocking_enabled and not llm_matcher_enabled:
        # The ninth case, and the one that shows the survey was still an
        # enumeration of pairs someone thought of. `SemanticBlocker` is built when
        # its own flag is on, but its only use is inside `_run_llm_matching`,
        # which runs only when an LLM matcher exists — so enabling blocking alone
        # constructs a UMAP/HDBSCAN blocker and never calls it. Identical in shape
        # to `outliers-without-centrality`: a stage that is a PARAMETER of another
        # stage, gated separately.
        findings.append(
            Finding(
                BLOCK,
                "blocking-without-a-matcher",
                "semantic blocking is enabled but LLM matching is not; the "
                "blocker is only consulted while matching runs, so it would be "
                "built and never used",
                "enable llm_matcher, or disable semantic_blocking.enabled",
            )
        )

    if outlier_detection_enabled and not graph_centrality_enabled:
        # Outlier classification is a parameter OF the graph analyser, and the
        # analyser is only built when centrality is on — so the flag is read into
        # an object that is never constructed.
        findings.append(
            Finding(
                BLOCK,
                "outliers-without-centrality",
                "outlier detection is enabled but graph centrality is not; "
                "outlier classification is performed by the graph analyser, which "
                "is only built when centrality is enabled",
                "enable ontology_validation.graph_centrality_enabled, or disable "
                "outlier_detection_enabled",
            )
        )

    return findings


def check_privacy_defaults(
    *, routing_default: str, resolver_default: str
) -> List[Finding]:
    """The two systems must not disagree about where data may go.

    ``model_routing.yaml`` defaults to ``internal`` (local only) while the app's
    ``route_resolver`` defaults to ``CLOUD`` (prefer cloud, fall back to local).
    Both defaults are defensible on their own; together they mean the same
    document is treated as local by one path and cloud-eligible by the other,
    which is a privacy question rather than a routing preference.
    """
    local_routing = routing_default in ("internal", "confidential")
    # `internal`/`confidential` appear on the routing side and `private`/`local`
    # on the resolver side; both name the same intent. Accepting only the resolver
    # vocabulary reported a mismatch for `internal`/`internal`.
    local_resolver = resolver_default.lower() in (
        "private", "local", "internal", "confidential"
    )
    if local_routing == local_resolver:
        return []
    return [
        Finding(
            WARN,
            "privacy-default-mismatch",
            f"model_routing.yaml defaults to {routing_default!r} while the app "
            f"resolver defaults to {resolver_default!r}; the pipeline and the app "
            f"disagree about whether data may leave the machine by default",
            "align the two defaults, or state in model_routing.yaml which one "
            "governs which surface",
        )
    ]


def collect_findings(
    *,
    routing_config: Dict[str, Any],
    concept_alignment_enabled: bool,
    kg_resolution_enabled: bool,
    judge_enabled: bool,
    judge_model_configured: bool,
    resolver_default_privacy: str,
    ontology_validation_enabled: bool = False,
    ontology_supplied: bool = True,
    installed_models: Optional[set] = None,
    ollama_probe: Callable[[str], Optional[set]] = _installed_ollama_models,
) -> List[Finding]:
    """Run every check and return the findings, worst first."""
    if installed_models is None:
        base_url = (
            routing_config.get("providers", {})
            .get("ollama", {})
            .get("base_url")
            or os.getenv("OLLAMA_URL")
            or "http://localhost:11434"
        )
        installed_models = ollama_probe(base_url)

    findings = [
        *check_routing(routing_config, installed=installed_models),
        *check_feature_dependencies(
            concept_alignment_enabled=concept_alignment_enabled,
            kg_resolution_enabled=kg_resolution_enabled,
            judge_enabled=judge_enabled,
            judge_model_configured=judge_model_configured,
            ontology_validation_enabled=ontology_validation_enabled,
            ontology_supplied=ontology_supplied,
        ),
        *check_privacy_defaults(
            routing_default=str(
                routing_config.get("defaults", {}).get("default_privacy", "internal")
            ),
            resolver_default=resolver_default_privacy,
        ),
    ]
    return sorted(findings, key=lambda f: 0 if f.severity == BLOCK else 1)


def raise_if_blocking(findings: List[Finding]) -> None:
    """Refuse when a configured feature cannot do its job."""
    blocking = [f for f in findings if f.severity == BLOCK]
    if blocking:
        raise ConfigurationError(blocking)


__all__ = [
    "BLOCK",
    "WARN",
    "ConfigurationError",
    "Finding",
    "check_feature_dependencies",
    "check_privacy_defaults",
    "check_ollama_context",
    "check_routing",
    "collect_findings",
    "raise_if_blocking",
]
