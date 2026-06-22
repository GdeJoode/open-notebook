"""
External identifier resolution for knowledge-graph entities.

Introduced as a stub in Track D Phase D.0; the body landed in Track K Phase K.4
(the FEATURE_ROADMAP Q9 swap). It returns the stable cross-system identifiers
(TOOI URIs, Crossref DOIs, ...) for an entity so exported Obsidian/JSONL/NetworkX
artifacts can carry them.

How resolution works (K.4)
==========================
The async :class:`~app_main.services.entity_resolution.vocabulary_reconciler.VocabularyReconciler`
queries the external authorities (TOOI, Crossref) and, under a strict
single-high-confidence-match precision guard, writes the matched URI(s) onto
``entity.external_ids``. THIS function — the synchronous swap-point every Track D
exporter funnels through — reads that already-reconciled field. Keeping it
synchronous and side-effect-free preserves the exporter contract (no DB/network
call inside an export loop) while letting reconciled entities emit real URIs.

Contract (MUST NOT change — Track D exporters depend on it)
===========================================================
* signature ``resolve_external_ids(entity: Entity) -> List[str]`` is unchanged;
* an entity with no reconciled identifiers (e.g. ``canonical_name=""`` or an
  un-reconciled entity) still returns ``[]``;
* the single import point ``shared.utils.external_ids`` is preserved.
"""

from __future__ import annotations

from typing import List

from shared.models.entity import Entity


def resolve_external_ids(entity: Entity) -> List[str]:
    """Return the stable external identifiers for an entity.

    Reads ``entity.external_ids`` — the field the K.4 vocabulary reconciler
    populates when an entity links (under the precision guard) to a single
    high-confidence authority record. Un-reconciled entities (and entities the
    reconciler refused to link) carry an empty list, so this returns ``[]`` for
    them — preserving the Track D exporters' empty-input contract.

    The returned list is a defensive copy of deduplicated, truthy string URIs, so
    a caller cannot mutate the entity's field through the result.

    Args:
        entity: The canonical entity to resolve identifiers for. ``external_ids``
            (populated by reconciliation) is the field read here; the legacy
            ``canonical_name`` / ``entity_type`` inputs drove the now-removed
            stub and are no longer consulted directly.

    Returns:
        The entity's stable external identifier URIs, or ``[]`` when none are
        reconciled. e.g.
        ``["https://identifier.overheid.nl/tooi/id/ministerie/mnre1034"]`` or
        ``["https://doi.org/10.1145/..."]``.

    Examples:
        >>> from shared.models.entity import Entity
        >>> e = Entity(canonical_name="Ada Lovelace", entity_type="Person")
        >>> resolve_external_ids(e)
        []
        >>> e.external_ids = ["https://doi.org/10.1/x"]
        >>> resolve_external_ids(e)
        ['https://doi.org/10.1/x']
    """
    raw = getattr(entity, "external_ids", None) or []
    out: List[str] = []
    for value in raw:
        if not value:
            continue
        text = str(value)
        if text not in out:
            out.append(text)
    return out


__all__ = ["resolve_external_ids"]
