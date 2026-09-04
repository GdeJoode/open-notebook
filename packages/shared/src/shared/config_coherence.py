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

#: Steps every run exercises, so a missing model there stops work outright.
#: Everything else — `vlm`, `classification`, `summarization` — is reached only by
#: a caller that asks for it, and an operator who never touches the VLM path must
#: still be able to boot. Refusing startup over a model they will never load is
#: the over-reach `check_feature_dependencies` already refuses to commit, and it
#: is worst in the COMMON case: no Ollama at all degrades to one WARN, while
#: Ollama with some models would refuse. Declared rather than inferred, because
#: the inference ("is this step used?") is not available at startup.
REQUIRED_STEPS = frozenset({"extraction", "embedding"})


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


def check_feature_dependencies(
    *,
    concept_alignment_enabled: bool,
    kg_resolution_enabled: bool,
    judge_enabled: bool,
    judge_model_configured: bool,
    ontology_validation_enabled: bool = False,
    ontology_supplied: bool = True,
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
    local_resolver = resolver_default.lower() in ("private", "local")
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
    "check_routing",
    "collect_findings",
    "raise_if_blocking",
]
