"""
External identifier resolution for knowledge-graph entities.

V1 stub introduced in Track D Phase D.0. Track M4 Q9 replaces this with
TOOI (Thesaurus of Open-source Ontology Identifiers) and Crossref
lookups so exported Obsidian/JSONL/NetworkX artifacts can carry stable
cross-system identifiers (DOIs, ORCIDs, Wikidata QIDs, ...).

Why a stub at this point in the timeline
========================================
Track D's three export formats (D.1 Obsidian vault, D.2 JSONL,
D.3 NetworkX) all want to emit an ``external_ids`` slot per entity, but
the upstream resolution pipeline is Track M4 work. Threading a single
function through the export call sites today means the Q9 swap is a
one-file change rather than a fan-out across three exporters.

The single import point ``shared.utils.external_ids`` is intentional --
downstream callers should depend only on the public
``resolve_external_ids`` function. This file is structurally identical
to ``shared.utils.name_normalizer`` (the B.1c stub for entity-name
normalization) so the two stub modules upgrade in parallel.
"""

from __future__ import annotations

from typing import List

from shared.models.entity import Entity


def resolve_external_ids(entity: Entity) -> List[str]:
    """Return the stable external identifiers for an entity.

    V1 stub. Always returns ``[]`` so Track D exporters can emit a
    well-formed (but empty) ``external_ids`` field without breaking when
    the real resolver lands.

    Track M4 Q9 will replace this body with TOOI + Crossref lookups
    keyed off ``entity.canonical_name`` and ``entity.entity_type`` --
    no other plumbing required, because every Track D exporter funnels
    through this single function.

    Args:
        entity: The canonical entity to resolve identifiers for. The
            ``canonical_name`` and ``entity_type`` fields are the
            inputs the Q9 implementation will key off.

    Returns:
        Empty list. The Q9 swap will return URIs like
        ``["https://doi.org/10.1145/...", "https://orcid.org/..."]``.

    Examples:
        >>> from shared.models.entity import Entity
        >>> e = Entity(canonical_name="Ada Lovelace", entity_type="Person")
        >>> resolve_external_ids(e)
        []
    """
    # Reference ``entity`` to make the V1 contract explicit -- when Q9
    # ships, the body uses entity.canonical_name + entity.entity_type as
    # the resolver inputs. Touching ``entity`` here also documents the
    # required argument shape for downstream callers.
    _ = entity
    return []


__all__ = ["resolve_external_ids"]
