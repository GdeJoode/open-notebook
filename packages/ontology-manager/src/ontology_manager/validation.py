"""Structural validation of a candidate ontology set (Track OE.1).

The ontologies are shared by every notebook and decide how every extracted
entity is typed. The editor validates a CANDIDATE set — each ontology's OWN
content, keyed by name, exactly what would be stored — before it takes effect.

This module holds the STRUCTURAL rules: what can be decided from the ontology
content alone. The TYPING rules — what a label will actually be typed as, over
the typing vocabulary persistence uses since PC.4 (every ontology, the
document's selection first, with the residual and passthrough fallbacks) —
live beside that typing, in ``app_main.services.ontology_validation_service``,
which combines both through :func:`classify`.

Regression semantics (:func:`classify`): a finding the ACTIVE set already has
with at least the same ``weight`` is a ``warn`` (pre-existing, not this edit's
doing); a new finding, or one whose weight grew (a label with three answers
where there were two), is a ``block``. So an imperfect vocabulary can still be
edited, and no edit can make it worse. ``unknown_key`` only ever warns.

Pure: no I/O, no registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from shared.utils.entity_type_aliases import NON_TYPE_LABELS

from ontology_manager.canonical_bridge import (
    _CANONICAL_BY_SCHEMA_ORG,
    _strip_schema_prefix,
)
from ontology_manager.schema import Ontology

BLOCK = "block"
WARN = "warn"

#: Keys the shipped YAML uses that the model has no field for (OE.0 keeps
#: them), per level. Anything else is most likely a typo and is reported.
RECOGNISED_EXTRA_KEYS: Dict[str, frozenset] = {
    "ontology": frozenset({"extraction_rules"}),
    "metadata": frozenset({"namespaces"}),
    "entity_type": frozenset({"schema_org", "dcterms", "graph_relevance", "abstract",
                              "extraction_strategy"}),
    "property": frozenset({"schema_org", "dcterms", "prism", "required", "validation_rules",
                           "enum_values", "properties"}),
    "relationship_type": frozenset({"schema_org", "dcterms", "skos"}),
    "concept": frozenset(),
}

#: Rules whose findings never block, whatever the active set holds.
ALWAYS_WARN = frozenset({"unknown_key"})

#: A relationship endpoint that stands for "any type".
WILDCARD_ENDPOINTS = frozenset({"*", "entity", "thing"})


@dataclass(frozen=True)
class Finding:
    """One problem in a candidate set.

    ``key`` identifies it across the candidate and the active set (rule,
    ontology, subject); ``weight`` measures how bad it is (e.g. the number of
    different answers a label gets), so a finding that got worse is caught.
    """

    rule: str
    ontology: Optional[str]
    subject: Optional[str]
    detail: str
    severity: str = BLOCK
    weight: int = 1

    @property
    def key(self) -> Tuple[str, Optional[str], Optional[str]]:
        return (self.rule, self.ontology, self.subject)


def _norm(text: Optional[str]) -> str:
    return (text or "").strip().lower()


def labels_of(ontology: Ontology) -> Iterable[Tuple[str, str]]:
    """(label, owning type) for every name a label can match: the type's name,
    its dict key and its aliases — the names ``canonical_bridge._find_definition``
    matches."""
    for key, definition in ontology.entity_types.items():
        name = definition.name or key
        yield name, name
        if key != name:
            yield key, name
        for alias in definition.aliases or []:
            yield alias, name


# -- inheritance within the set ------------------------------------------------


def _links(ontology: Ontology) -> List[str]:
    meta = ontology.metadata
    if not meta.extends:
        # `OntologyRegistry.get` resolves inheritance only when `extends` is
        # set (then the parent, then each include); `includes` alone is not
        # merged there, so it is not merged here.
        return []
    return [meta.extends] + list(meta.includes or [])


def _inheritance_loops(own: Dict[str, Ontology]) -> List[Set[str]]:
    """The strongly connected parts of the inheritance graph that loop — found on
    the graph itself, so one loop is one finding whichever ontology a walk would
    have entered it from."""
    index: Dict[str, int] = {}
    low: Dict[str, int] = {}
    stack: List[str] = []
    on_stack: Set[str] = set()
    loops: List[Set[str]] = []

    def visit(node: str) -> None:
        index[node] = low[node] = len(index)
        stack.append(node)
        on_stack.add(node)
        for nxt in _links(own[node]):
            if nxt not in own:
                continue
            if nxt not in index:
                visit(nxt)
                low[node] = min(low[node], low[nxt])
            elif nxt in on_stack:
                low[node] = min(low[node], index[nxt])
        if low[node] == index[node]:
            part: Set[str] = set()
            while True:
                member = stack.pop()
                on_stack.discard(member)
                part.add(member)
                if member == node:
                    break
            if len(part) > 1 or node in _links(own[node]):
                loops.append(part)

    for name in own:
        if name not in index:
            visit(name)
    return loops


def resolve_set(own: Dict[str, Ontology]) -> Tuple[Dict[str, Ontology], List[Finding]]:
    """Merge each ontology with its inheritance chain, within ``own``, the way
    ``OntologyRegistry._resolve_inheritance`` does. A missing parent or a loop is
    a finding and that link is skipped, so the rest can still be checked."""
    findings: List[Finding] = []
    loops = _inheritance_loops(own)
    loop_of = {m: frozenset(part) for part in loops for m in part}
    for part in loops:
        members = ",".join(sorted(part))
        findings.append(Finding("inheritance_cycle", None, members,
                                f"inheritance loops through {members}"))
    merged: Dict[str, Ontology] = {}

    def build(name: str) -> Ontology:
        if name in merged:
            return merged[name]
        result = own[name]
        for link in _links(own[name]):
            if link not in own:
                findings.append(Finding("unknown_parent_ontology", name, link,
                                        f"'{name}' extends or includes '{link}', which is not in the set"))
                continue
            if name in loop_of and link in loop_of[name]:
                continue  # reported above
            result = result.merge_with(build(link))
        merged[name] = result
        return result

    for name in own:
        build(name)
    return merged, findings


# -- the structural rules ---------------------------------------------------------


def _unknown_keys(name: str, level: str, subject: str, model) -> List[Finding]:
    allowed = RECOGNISED_EXTRA_KEYS[level]
    return [
        Finding("unknown_key", name, f"{subject}.{key}",
                f"{level} '{subject}' has an unrecognised key '{key}'", WARN)
        for key in (model.model_extra or {}) if key not in allowed
    ]


def _own_rules(name: str, ontology: Ontology) -> List[Finding]:
    """Rules about what THIS ontology declares (its own content, not inherited),
    so an inherited problem is reported once, where it is defined."""
    findings: List[Finding] = []
    findings += _unknown_keys(name, "ontology", name, ontology)
    findings += _unknown_keys(name, "metadata", name, ontology.metadata)
    for key, definition in ontology.entity_types.items():
        type_name = definition.name or key
        findings += _unknown_keys(name, "entity_type", type_name, definition)
        for prop in definition.properties:
            findings += _unknown_keys(name, "property", f"{type_name}.{prop.name}", prop)
        for label in [type_name, key, *(definition.aliases or [])]:
            if _norm(label) in NON_TYPE_LABELS:
                findings.append(Finding(
                    "literal_declared_as_type", name, label,
                    f"'{label}' is a literal (NON_TYPE_LABELS); declaring it as a type "
                    "pulls literals back into the type layer"))
        if definition.schema_org_type is not None:
            base = _strip_schema_prefix(definition.schema_org_type)
            if base not in _CANONICAL_BY_SCHEMA_ORG:
                # A deliberate unmapped base (`base.Concept` is
                # `schema:Intangible`) was reviewed once and is in the active
                # set; a NEW one — a typo like `schema:Persn` — blocks, because
                # the type is then typed by its name alone.
                findings.append(Finding(
                    "unmapped_schema_org_type", name, type_name,
                    f"'{type_name}' declares schema_org_type {definition.schema_org_type!r}, "
                    "which the canonical bridge does not map; the type is typed by its "
                    "name alone"))
            elif (_CANONICAL_BY_SCHEMA_ORG.get(type_name)
                  and _CANONICAL_BY_SCHEMA_ORG[base] != _CANONICAL_BY_SCHEMA_ORG[type_name]):
                # `general.Technology` -> `schema:TechArticle` is fine: both map to
                # `technology`. `schema_core.Grant` -> `schema:Event` is not.
                findings.append(Finding(
                    "base_named_type_points_elsewhere", name, type_name,
                    f"'{type_name}' is itself the schema.org base mapped to "
                    f"'{_CANONICAL_BY_SCHEMA_ORG[type_name]}' but declares schema_org_type "
                    f"{definition.schema_org_type!r}"))
    for rel in ontology.relationship_types.values():
        findings += _unknown_keys(name, "relationship_type", rel.name, rel)
        for prop in rel.properties or []:
            findings += _unknown_keys(name, "property", f"{rel.name}.{prop.name}", prop)
    for concept in ontology.concepts.values():
        findings += _unknown_keys(name, "concept", concept.name, concept)
    return findings


def _parent_cycles(merged: Dict[str, Ontology]) -> List[Finding]:
    findings, reported = [], set()
    for ontology in merged.values():
        by_name = {_norm(d.name or k): d for k, d in ontology.entity_types.items()}
        for type_name, definition in by_name.items():
            seen, current = [type_name], definition
            while current is not None and current.parent_type:
                parent = _norm(current.parent_type)
                if parent in seen:
                    loop = ",".join(sorted(seen[seen.index(parent):]))
                    if loop not in reported:
                        reported.add(loop)
                        findings.append(Finding("parent_cycle", None, loop,
                                                f"parent_type loops through {loop}"))
                    break
                seen.append(parent)
                current = by_name.get(parent)
    return findings


def _alias_collisions(merged: Dict[str, Ontology]) -> List[Finding]:
    """One name claimed by two different types within one resolved ontology."""
    found: Dict[str, Set[str]] = {}
    for ontology in merged.values():
        owners: Dict[str, Set[str]] = {}
        for label, owner in labels_of(ontology):
            owners.setdefault(_norm(label), set()).add(owner)
        for label, types in owners.items():
            if len(types) > 1:
                found.setdefault(label, set()).update(types)
    return [
        Finding("alias_collision", None, label,
                f"'{label}' names more than one type: {sorted(types)}", weight=len(types))
        for label, types in sorted(found.items())
    ]


def _relationship_endpoints(merged: Dict[str, Ontology]) -> List[Finding]:
    """A relationship's domain/range must name a type somewhere in the set."""
    known = {_norm(label) for o in merged.values() for label, _ in labels_of(o)}
    known |= {_norm(b) for b in _CANONICAL_BY_SCHEMA_ORG} | WILDCARD_ENDPOINTS
    findings, seen = [], set()
    for ontology in merged.values():
        for rel in ontology.relationship_types.values():
            for endpoint in list(rel.domain) + list(rel.range):
                if _norm(endpoint) not in known and (rel.name, endpoint) not in seen:
                    seen.add((rel.name, endpoint))
                    findings.append(Finding(
                        "relationship_endpoint_unknown", None, f"{rel.name}:{endpoint}",
                        f"relationship '{rel.name}' names '{endpoint}', which is no type "
                        "in any ontology"))
    return findings


def structural_findings(own: Dict[str, Ontology]) -> List[Finding]:
    """Every structural finding for a set (unclassified; see :func:`classify`)."""
    merged, findings = resolve_set(own)
    for name, ontology in own.items():
        findings += _own_rules(name, ontology)
    findings += _parent_cycles(merged)
    findings += _alias_collisions(merged)
    findings += _relationship_endpoints(merged)
    return findings


def classify(found: Iterable[Finding], known: Iterable[Finding]) -> List[Finding]:
    """Warn on what ``known`` (the active set) already has at the same or a
    higher weight; block on anything new or worse."""
    known_weight: Dict[Tuple, int] = {}
    for f in known:
        known_weight[f.key] = max(known_weight.get(f.key, 0), f.weight)
    out = []
    for f in found:
        pre_existing = f.key in known_weight and f.weight <= known_weight[f.key]
        severity = WARN if (f.rule in ALWAYS_WARN or pre_existing) else BLOCK
        out.append(Finding(f.rule, f.ontology, f.subject, f.detail, severity, f.weight))
    return sorted(out, key=lambda f: (f.severity != BLOCK, f.rule, f.ontology or "", f.subject or ""))


def validate(
    candidate: Dict[str, Ontology],
    active: Optional[Dict[str, Ontology]] = None,
) -> List[Finding]:
    """The structural findings for ``candidate``, classified against ``active``.

    The editor uses ``app_main.services.ontology_validation_service
    .validate_candidate``, which adds the typing rules.
    """
    return classify(structural_findings(candidate),
                    structural_findings(active) if active is not None else [])


def blocking(findings: Iterable[Finding]) -> List[Finding]:
    return [f for f in findings if f.severity == BLOCK]


__all__ = [
    "BLOCK", "WARN", "Finding", "blocking", "classify", "labels_of", "resolve_set",
    "structural_findings", "validate",
]
