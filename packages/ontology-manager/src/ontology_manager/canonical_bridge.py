"""Language-agnostic ontology-type -> canonical-enum bridge (Track L.1).

The extraction prompt orders the LLM to emit *domain-specific* ontology types
(the rich Dutch government vocabulary). The persistence boundary, however, only
accepts a small fixed canonical enum (``_ALLOWED_ENTITY_TYPES``). Without a
bridge the rich type is lowercased, missed by an English-only alias map, and
flattened to ``other`` — the rich type is lost.

This module bridges an extracted ontology label to its coarse canonical value
by walking the ontology's *own declared* ``parent_type`` chain to a schema.org
base, then mapping that base through a small FIXED map. The map is keyed on
stable schema.org English identifiers ONLY — there are no language-specific
literals here. A localized label resolves purely because its
``EntityTypeDefinition`` declares (for example) ``parent_type:
AdministrativeArea``; adding a new language is adding an ontology, not editing
this code.

The original ontology label is preserved by the caller in ``primary_type`` /
``type_tags`` (see :class:`CanonicalResolution`). ``entity_type`` becomes the
coarse canonical projection used for dedup / filtering / the K ``(name, type)``
guard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from ontology_manager.schema import EntityTypeDefinition, Ontology

# Fixed schema.org-base -> canonical-enum map. Keys are stable schema.org
# English identifiers; values are members of the persistence
# ``_ALLOWED_ENTITY_TYPES`` enum. This is the ONLY language-specific-free
# surface of the bridge: a label resolves through the ontology's declared
# parent_type chain terminating on one of these names.
#
# L.3 will add ``programme`` / ``technology`` canonical types; until then
# ``Deal`` / ``GovernmentService`` map to ``creative_work`` as a documented
# INTERIM (a programme is a kind of creative work for dedup purposes). No
# Dutch literals — every key is an English schema.org identifier.
_CANONICAL_BY_SCHEMA_ORG: dict[str, str] = {
    "AdministrativeArea": "administrative_area",
    "GovernmentOrganization": "government_organization",
    "Person": "person",
    "Organization": "organization",
    "EducationalOrganization": "organization",
    "Place": "location",
    "Legislation": "legislation",
    "Concept": "topic",
    "DefinedTerm": "topic",
    "CreativeWork": "creative_work",
    "Article": "creative_work",
    "Report": "policy_document",
    "Event": "event",
    "Dataset": "dataset",
    "Grant": "grant",
    "Thing": "concept",
    # INTERIM (L.3 introduces ``programme``): a deal / government service is a
    # programme; until the canonical type exists it coarses to creative_work.
    "Deal": "creative_work",
    "GovernmentService": "creative_work",
}

# Bound on the parent_type walk so an ontology authoring loop (or a very deep
# chain) can never hang the persist path. Chains in practice are 1-4 deep.
_MAX_WALK_DEPTH = 16


@dataclass
class CanonicalResolution:
    """Result of bridging an ontology label to a canonical enum value.

    Attributes:
        canonical: The coarse ``_ALLOWED_ENTITY_TYPES`` member (e.g.
            ``"administrative_area"``). Becomes the persisted ``entity_type``.
        ontology_type: The ORIGINAL extracted ontology label (the rich
            domain type). Stamped into ``primary_type``.
        type_tags: ``[ontology_type, ...intermediate parents up to the
            schema.org base]`` — the rich-type trail preserved for display /
            faceting.
    """

    canonical: str
    ontology_type: str
    type_tags: List[str] = field(default_factory=list)


def _strip_schema_prefix(value: str) -> str:
    """Strip a ``schema:`` / ``schema.org`` prefix from a schema_org_type.

    ``"schema:Person"`` -> ``"Person"``; ``"Person"`` -> ``"Person"``.
    """
    v = value.strip()
    if ":" in v:
        v = v.split(":", 1)[1]
    return v.strip()


def _find_definition(
    label: str, schemas: Iterable[Ontology]
) -> Optional[EntityTypeDefinition]:
    """Find the EntityTypeDefinition matching ``label`` across applied schemas.

    Matches the type ``name`` or any of its ``aliases`` case-insensitively.
    The registry resolves inheritance, so each applied ``Ontology`` already
    carries its merged parent types in ``entity_types`` — a single pass over
    the applied schemas sees the whole layered vocabulary.
    """
    target = label.strip().lower()
    if not target:
        return None
    for ontology in schemas:
        for key, definition in ontology.entity_types.items():
            if key.lower() == target:
                return definition
            if definition.name and definition.name.lower() == target:
                return definition
            if definition.aliases:
                for alias in definition.aliases:
                    if alias.lower() == target:
                        return definition
    return None


def resolve_ontology_type(
    label: str, schemas: List[Ontology]
) -> Optional[CanonicalResolution]:
    """Bridge an ontology ``label`` to its canonical enum value.

    Resolution order for a matched ``EntityTypeDefinition``:

    1. ``schema_org_type`` (if set) — strip a ``schema:`` prefix and look the
       base up directly in :data:`_CANONICAL_BY_SCHEMA_ORG` (preferred: the
       ontology author named the schema.org base explicitly).
    2. Walk ``parent_type`` recursively. At each step, if the parent *name*
       is a key in the map, terminate — even when that base is not itself a
       loaded definition (``AdministrativeArea`` / ``GovernmentOrganization``
       live in ``policy.yaml``; ``Deal`` / ``GovernmentService`` are declared
       only as parents). This is why the bridge stays correct whether or not
       every base ontology is in the applied set.

    Args:
        label: The extracted ontology label (the rich domain type).
        schemas: The ontologies applied for this source's extraction (already
            inheritance-resolved by the registry).

    Returns:
        A :class:`CanonicalResolution` when the label is in some applied
        ontology AND its chain reaches a mapped schema.org base; ``None`` when
        the label is not in any applied ontology (deferred to the L.2 residual
        alias map) or its chain orphans (surfaced by the L.6 audit).
    """
    if not label or not label.strip():
        return None

    definition = _find_definition(label, schemas)
    if definition is None:
        return None

    original = definition.name or label

    # (1) Prefer an explicit schema_org_type when the ontology declares one.
    if definition.schema_org_type:
        base = _strip_schema_prefix(definition.schema_org_type)
        canonical = _CANONICAL_BY_SCHEMA_ORG.get(base)
        if canonical is not None:
            type_tags = [original]
            if base != original:
                type_tags.append(base)
            return CanonicalResolution(
                canonical=canonical,
                ontology_type=original,
                type_tags=type_tags,
            )

    # (2) Walk the parent_type chain, terminating on a mapped base NAME.
    type_tags: List[str] = [original]
    visited: set[str] = {original.lower()}
    current = definition
    depth = 0

    while current is not None and depth < _MAX_WALK_DEPTH:
        parent_name = current.parent_type
        if not parent_name:
            break

        canonical = _CANONICAL_BY_SCHEMA_ORG.get(parent_name)
        if canonical is not None:
            # Terminate on the schema.org base name even if no loaded
            # definition exists for it (Deal / GovernmentService / the
            # policy.yaml bases).
            type_tags.append(parent_name)
            return CanonicalResolution(
                canonical=canonical,
                ontology_type=original,
                type_tags=type_tags,
            )

        if parent_name.lower() in visited:
            # Cycle guard: an ontology authoring loop.
            break
        visited.add(parent_name.lower())
        type_tags.append(parent_name)

        # Continue the walk from the parent's own definition (if loaded).
        current = _find_definition(parent_name, schemas)
        depth += 1

    # Chain did not reach a mapped base — orphan (L.6 audit) or unmapped base.
    return None
