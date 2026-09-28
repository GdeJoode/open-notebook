"""Validate a candidate ontology set before it takes effect (Track OE.1).

The ontologies are shared by every notebook and decide how every extracted
entity is typed. PC.4 found 15 types quietly wrong and one under the wrong
parent, caught only by a test. This module is that test turned into a gate the
editor runs before a publish:

:func:`validate` takes the candidate set — each ontology's OWN content, keyed by
name, exactly what would be stored — resolves inheritance within the set (a
change to a parent is seen by its children), and returns :class:`Finding` s.

Given the ACTIVE set as well, a finding the active set already has is a
``warn`` (pre-existing; not this edit's doing) and a finding it does not have
is a ``block`` (this edit introduces it). So an imperfect vocabulary can still
be edited, but no edit can make it worse. ``unknown_key`` is always a warning.

Pure: no I/O, no registry — the caller supplies both sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from shared.utils.entity_type_aliases import NON_TYPE_LABELS

from ontology_manager.canonical_bridge import (
    _CANONICAL_BY_SCHEMA_ORG,
    _strip_schema_prefix,
    resolve_ontology_type,
)
from ontology_manager.schema import Ontology

BLOCK = "block"
WARN = "warn"

#: Keys the shipped YAML uses that the model has no field for (OE.0 keeps them).
#: Anything else is most likely a typo (`parent_typ`) and is reported.
RECOGNISED_EXTRA_KEYS = frozenset({
    "schema_org", "dcterms", "graph_relevance", "required", "validation_rules",
    "enum_values", "namespaces", "extraction_rules", "abstract", "extraction_strategy",
})

#: Rules whose findings never block, whatever the active set holds.
ALWAYS_WARN = frozenset({"unknown_key"})


@dataclass(frozen=True)
class Finding:
    """One problem in a candidate set.

    ``key`` identifies it across the candidate and the active set (rule,
    ontology, subject), so a pre-existing finding is recognised as such.
    """

    rule: str
    ontology: Optional[str]
    subject: Optional[str]
    detail: str
    severity: str = BLOCK

    @property
    def key(self) -> Tuple[str, Optional[str], Optional[str]]:
        return (self.rule, self.ontology, self.subject)


def _norm(text: Optional[str]) -> str:
    return (text or "").strip().lower()


# -- inheritance within the set ------------------------------------------------


def _links(ontology: Ontology) -> List[str]:
    meta = ontology.metadata
    return ([meta.extends] if meta.extends else []) + list(meta.includes or [])


def _inheritance_loops(own: Dict[str, Ontology]) -> List[Set[str]]:
    """The strongly connected parts of the extends/includes graph that loop.

    Found on the graph itself, once, so one loop is one finding whichever
    ontology a walk would have entered it from.
    """
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
    """Merge each ontology with its ``extends`` / ``includes`` chain, within ``own``.

    The same order as ``OntologyRegistry._resolve_inheritance``: the parent
    first, then each include, the child winning. A missing parent or a loop is
    a finding and that link is skipped, so the rest can still be checked.
    """
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


# -- the rules -------------------------------------------------------------------


def _labels(ontology: Ontology) -> Iterable[Tuple[str, str]]:
    """(label, owning type) for every type name and alias."""
    for key, definition in ontology.entity_types.items():
        name = definition.name or key
        yield name, name
        for alias in definition.aliases or []:
            yield alias, name


def _parent_cycles(name: str, ontology: Ontology) -> List[Finding]:
    by_name = {_norm(d.name or k): d for k, d in ontology.entity_types.items()}
    findings, reported = [], set()
    for type_name, definition in by_name.items():
        seen, current = [type_name], definition
        while current is not None and current.parent_type:
            parent = _norm(current.parent_type)
            if parent in seen:
                loop = tuple(sorted(seen[seen.index(parent):]))
                if loop not in reported:
                    reported.add(loop)
                    findings.append(Finding("parent_cycle", name, loop[0],
                                            f"parent_type loops through {', '.join(loop)}"))
                break
            seen.append(parent)
            current = by_name.get(parent)
    return findings


def _declares_a_base(type_name: str, ontology: Ontology) -> bool:
    """Whether the type, or an ancestor, names a schema.org base or a parent.

    A type whose chain ends on a declared but UNMAPPED base (`base.Concept` is
    `schema:Intangible`, `general.Product` is `schema:Product`) is typed by its
    name, deliberately. An orphan is a chain that ends on nothing at all.
    """
    by_name = {_norm(d.name or k): d for k, d in ontology.entity_types.items()}
    seen, current = set(), by_name.get(_norm(type_name))
    while current is not None and _norm(current.name) not in seen:
        seen.add(_norm(current.name))
        if current.schema_org_type:
            return True
        if not current.parent_type:
            return False
        parent = by_name.get(_norm(current.parent_type))
        if parent is None:
            # Only reached when the bridge found no mapped base, so this parent
            # is neither a mapped base name nor defined: a typo or a dangling ref.
            return False
        current = parent
    return False


def _per_ontology(name: str, ontology: Ontology) -> List[Finding]:
    findings = _parent_cycles(name, ontology)
    owners: Dict[str, Set[str]] = {}
    for label, owner in _labels(ontology):
        owners.setdefault(_norm(label), set()).add(owner)
        if _norm(label) in NON_TYPE_LABELS:
            findings.append(Finding("literal_declared_as_type", name, label,
                                    f"'{label}' is a literal (NON_TYPE_LABELS); declaring it "
                                    "as a type pulls literals back into the type layer"))
    for label, types in owners.items():
        if len(types) > 1:
            findings.append(Finding("alias_collision", name, label,
                                    f"'{label}' names more than one type: {sorted(types)}"))
    for key, definition in ontology.entity_types.items():
        type_name = definition.name or key
        if definition.schema_org_type:
            base = _strip_schema_prefix(definition.schema_org_type)
            if not base:
                findings.append(Finding("invalid_schema_org_type", name, type_name,
                                        f"schema_org_type {definition.schema_org_type!r} is empty"))
        expected = _CANONICAL_BY_SCHEMA_ORG.get(type_name)
        resolved = resolve_ontology_type(type_name, [ontology])
        if expected and not definition.schema_org_type:
            got = resolved.canonical if resolved else None
            if got != expected:
                findings.append(Finding(
                    "base_named_type_misses_its_base", name, type_name,
                    f"'{type_name}' is a schema.org base mapped to '{expected}' but resolves "
                    f"to {got!r}; give it schema_org_type: \"schema:{type_name}\""))
        if resolved is None and not _declares_a_base(type_name, ontology):
            findings.append(Finding(
                "orphan", name, type_name,
                f"'{type_name}' reaches no mapped schema.org base through parent_type / "
                "schema_org_type, so its entities are typed by the alias fallback"))
        for key_name in (definition.model_extra or {}):
            if key_name not in RECOGNISED_EXTRA_KEYS:
                findings.append(Finding("unknown_key", name, f"{type_name}.{key_name}",
                                        f"'{type_name}' has an unrecognised key '{key_name}'",
                                        WARN))
    return findings


def _across_ontologies(merged: Dict[str, Ontology]) -> List[Finding]:
    answers: Dict[str, Dict[str, str]] = {}
    for name, ontology in merged.items():
        for label, _owner in _labels(ontology):
            resolved = resolve_ontology_type(label, [ontology])
            if resolved is not None:
                answers.setdefault(_norm(label), {})[name] = resolved.canonical
    findings = []
    for label, per_ontology in sorted(answers.items()):
        if len(set(per_ontology.values())) > 1:
            findings.append(Finding(
                "two_canonical_answers", None, label,
                f"'{label}' types differently depending on the ontology: "
                + ", ".join(f"{o}={c}" for o, c in sorted(per_ontology.items()))))
    return findings


def _findings(own: Dict[str, Ontology]) -> List[Finding]:
    merged, findings = resolve_set(own)
    for name, ontology in merged.items():
        findings.extend(_per_ontology(name, ontology))
    findings.extend(_across_ontologies(merged))
    return findings


def validate(
    candidate: Dict[str, Ontology],
    active: Optional[Dict[str, Ontology]] = None,
) -> List[Finding]:
    """Every finding in ``candidate``; pre-existing ones (also in ``active``) warn."""
    found = _findings(candidate)
    known = {f.key for f in _findings(active)} if active is not None else set()
    out = []
    for f in found:
        severity = WARN if (f.rule in ALWAYS_WARN or f.key in known) else f.severity
        out.append(Finding(f.rule, f.ontology, f.subject, f.detail, severity))
    return sorted(out, key=lambda f: (f.severity != BLOCK, f.rule, f.ontology or "", f.subject or ""))


def blocking(findings: Iterable[Finding]) -> List[Finding]:
    return [f for f in findings if f.severity == BLOCK]


__all__ = ["BLOCK", "WARN", "Finding", "blocking", "resolve_set", "validate"]
