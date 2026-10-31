"""Does this configuration express one intent? (Track PC.6)

The finding this module answers: `ENABLE_CONCEPT_ALIGNMENT=true` did nothing in a
measured run, because alignment classifies only the entities KG resolution marks
``is_new`` and KG resolution is off by default. A feature reachable only by
changing a second, unrelated default is not a feature; it is a flag that reads as
one. The phase's acceptance criterion is that **enabling a feature either works or
says why it cannot**, and that the "flag on, zero effect, one warning" state is
unreachable rather than merely logged.

**One way to a model (F2-08, ADR 0006).** Every model call goes through
``ModelGate`` to llama-swap, with the model taken from the register
(``config/modelregister.yaml``). The checks on ``model_routing.yaml`` and on the
Ollama runtime (routed models pulled, ``num_ctx`` baked in, the two privacy
defaults agreeing) went with it: nothing reaches a model by those routes any
more. What remains are the feature dependencies.

**Severities.** A :data:`BLOCK` finding means a configured feature cannot do its
job, and the caller is expected to refuse. A :data:`WARN` means the configuration
is surprising but functional. Nothing here writes; the caller decides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

BLOCK = "block"
WARN = "warn"

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


def collect_findings(
    *,
    concept_alignment_enabled: bool,
    kg_resolution_enabled: bool,
    judge_enabled: bool,
    judge_model_configured: bool,
    ontology_validation_enabled: bool = False,
    ontology_supplied: bool = True,
) -> List[Finding]:
    """Run every check and return the findings, worst first."""
    findings = [
        *check_feature_dependencies(
            concept_alignment_enabled=concept_alignment_enabled,
            kg_resolution_enabled=kg_resolution_enabled,
            judge_enabled=judge_enabled,
            judge_model_configured=judge_model_configured,
            ontology_validation_enabled=ontology_validation_enabled,
            ontology_supplied=ontology_supplied,
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
    "collect_findings",
    "raise_if_blocking",
]
