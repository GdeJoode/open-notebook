"""The ontology editor's validation gate: structural + typing rules (Track OE.1).

Structural rules (``ontology_manager.validation``) decide from the ontology
content alone. The TYPING rules here decide from what a label will actually be
typed as — :func:`_resolve_entity_type` over :func:`typing_vocabulary`, the path
persistence takes since PC.4: every ontology, the document's selection first,
the bridge, then the passthrough and residual fallbacks. Modelling them on
anything else is how the first version of this gate missed the PC.4 bug class
(review OE.1 B1) and blocked 553 of 935 legitimate re-parents (M1).

What a label types as depends on which definition comes FIRST in the
vocabulary for every name on its chain — the label and each ``parent_type`` it
walks through. Only ontologies that define one of those names DIFFERENTLY can
change the answer, and only their relative order matters. So per label the
ontologies are grouped by their definitions of the chain's names, one
representative per group, and the label is typed under every ordered selection
of up to three representatives (detection's ``top_k``) plus no selection — the
exact set of outcomes a selection can produce, not a sample. (Review OE.1
round 2 M1: one leader at a time missed a label and its parent each defined
twice; a PAIR of selected ontologies typed it differently.)

Rules, each keyed so :func:`classify` can tell pre-existing from new:

* ``selection_dependent_type`` — a label (name, key or alias) whose type differs
  with the leading ontology; weight = the number of different answers;
* ``base_named_type_misses_its_base`` — a type named after a mapped schema.org
  base (and not declaring another) that does not type as that base;
* ``orphan`` — a type the bridge cannot place whose chain declares no base at all,
  so it is typed by the fallback;
  (This and the next two rules check a type with its OWN ontology leading, as
  when that ontology is selected. A type whose outcome depends on the order
  is caught by ``selection_dependent_type``, which is exact.)
* ``subtype_disagrees_with_parent`` — a type without its own base that types
  differently from the type its ``parent_type`` names (e.g. ``parent_type:
  "grant"``: the bridge's base lookup is case-sensitive, so the child walks past
  ``Grant``'s base to ``Thing``).
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Set, Tuple

from ontology_manager.canonical_bridge import (
    _CANONICAL_BY_SCHEMA_ORG,
    _find_definition,
    _strip_schema_prefix,
    resolve_ontology_type,
)
from ontology_manager.schema import Ontology
from ontology_manager.validation import (
    Finding,
    classify,
    labels_of,
    resolve_set,
    structural_findings,
)

from app_main.services.entity_persistence_service import (
    _resolve_entity_type,
    typing_vocabulary,
)


def _norm(text: Optional[str]) -> str:
    return (text or "").strip().lower()


def _orders(merged: Dict[str, Ontology]) -> Dict[Optional[str], List[Ontology]]:
    """One ontology leading, per ontology: the vocabulary a type is checked in
    by the per-type rules (its own ontology first, as when it is selected)."""
    everything = list(merged.values())
    orders: Dict[Optional[str], List[Ontology]] = {None: typing_vocabulary(None, everything)}
    for name, ontology in merged.items():
        orders[name] = typing_vocabulary([ontology], everything)
    return orders


def _chain_names(label: str, merged: Dict[str, Ontology]) -> Set[str]:
    """Every name the bridge may look up while typing ``label``: the label and,
    through ANY ontology's definition of it, each parent_type, transitively."""
    names, todo = set(), [label]
    while todo:
        name = todo.pop()
        if _norm(name) in names:
            continue
        names.add(_norm(name))
        for ontology in merged.values():
            definition = _find_definition(name, [ontology])
            if definition is not None and definition.parent_type:
                todo.append(definition.parent_type)
    return names


def _representatives(label: str, merged: Dict[str, Ontology]) -> List[str]:
    """One ontology per distinct set of definitions for the label's chain."""
    names = sorted(_chain_names(label, merged))
    groups: Dict[str, str] = {}
    for name in sorted(merged):
        signature = repr([
            (n, d.model_dump(mode="json") if (d := _find_definition(n, [merged[name]])) else None)
            for n in names
        ])
        groups.setdefault(signature, name)
    return sorted(groups.values())


def label_answers(label: str, merged: Dict[str, Ontology]) -> Dict[Tuple[str, ...], str]:
    """``label``'s type under every selection that can change it (see module doc)."""
    everything = list(merged.values())
    reps = _representatives(label, merged)
    answers = {(): _resolve_entity_type(label, typing_vocabulary(None, everything)).entity_type}
    if len(reps) < 2:
        return answers
    for size in range(1, min(3, len(reps)) + 1):
        for selection in itertools.permutations(reps, size):
            vocabulary = typing_vocabulary([merged[n] for n in selection], everything)
            answers[selection] = _resolve_entity_type(label, vocabulary).entity_type
    return answers


def _declares_a_base(type_name: str, vocabulary: List[Ontology]) -> bool:
    """Whether the type or an ancestor names a schema.org base (mapped or not).
    A chain ending on a declared UNMAPPED base (`base.Concept` is
    `schema:Intangible`) is typed by its name deliberately; an orphan is a chain
    that ends on nothing."""
    seen = set()
    current = _find_definition(type_name, vocabulary)
    while current is not None and _norm(current.name) not in seen:
        seen.add(_norm(current.name))
        if current.schema_org_type:
            return True
        if not current.parent_type:
            return False
        if current.parent_type in _CANONICAL_BY_SCHEMA_ORG:
            return True
        current = _find_definition(current.parent_type, vocabulary)
    return False


def typing_findings(own: Dict[str, Ontology]) -> List[Finding]:
    """Every typing finding for a set (unclassified)."""
    merged, _ = resolve_set(own)
    orders = _orders(merged)
    findings: List[Finding] = []

    labels = {label for ontology in merged.values() for label, _ in labels_of(ontology)}
    for label in sorted(labels, key=str.lower):
        answers = label_answers(label, merged)
        distinct = sorted(set(answers.values()))
        if len(distinct) > 1:
            by_answer = {
                answer: sorted("+".join(k) or "(none)" for k, v in answers.items() if v == answer)
                for answer in distinct
            }
            findings.append(Finding(
                "selection_dependent_type", None, _norm(label),
                f"'{label}' types as {distinct} depending on which ontology a document "
                f"selects: {by_answer}", weight=len(distinct)))

    for name, ontology in own.items():
        vocabulary = orders[name]
        for key, definition in ontology.entity_types.items():
            type_name = definition.name or key
            got = _resolve_entity_type(type_name, vocabulary).entity_type
            expected = _CANONICAL_BY_SCHEMA_ORG.get(type_name)
            own_base = (_strip_schema_prefix(definition.schema_org_type)
                        if definition.schema_org_type else None)
            if expected and own_base in (None, type_name) and got != expected:
                findings.append(Finding(
                    "base_named_type_misses_its_base", name, type_name,
                    f"'{type_name}' is a schema.org base mapped to '{expected}' but types "
                    f"as '{got}'; give it schema_org_type: \"schema:{type_name}\""))
            if (resolve_ontology_type(type_name, vocabulary) is None
                    and not _declares_a_base(type_name, vocabulary)):
                findings.append(Finding(
                    "orphan", name, type_name,
                    f"'{type_name}' reaches no schema.org base through parent_type / "
                    f"schema_org_type, so it is typed by the fallback, as '{got}'"))
            # A parent NAMED like a mapped base terminates the bridge's walk on
            # that base, whatever the parent's own definition says
            # (`BeleidsThema` -> `Concept` is `topic` though `base.Concept` is
            # `concept`) — by design. Compare only when the walk goes through
            # the parent's definition.
            if (definition.parent_type and not definition.schema_org_type
                    and definition.parent_type not in _CANONICAL_BY_SCHEMA_ORG):
                parent = _find_definition(definition.parent_type, vocabulary)
                if parent is not None:
                    parent_type = _resolve_entity_type(parent.name, vocabulary).entity_type
                    if parent_type != got:
                        findings.append(Finding(
                            "subtype_disagrees_with_parent", name, type_name,
                            f"'{type_name}' (parent_type {definition.parent_type!r}) types "
                            f"as '{got}' but its parent '{parent.name}' types as "
                            f"'{parent_type}'"))
    return findings


def all_findings(own: Dict[str, Ontology]) -> List[Finding]:
    return structural_findings(own) + typing_findings(own)


def validate_candidate(
    candidate: Dict[str, Ontology],
    active: Optional[Dict[str, Ontology]] = None,
) -> List[Finding]:
    """Every finding for ``candidate``; what ``active`` already has (at the same
    or a higher weight) warns, anything new or worse blocks."""
    return classify(all_findings(candidate), all_findings(active) if active is not None else [])


__all__ = ["all_findings", "label_answers", "typing_findings", "validate_candidate"]
