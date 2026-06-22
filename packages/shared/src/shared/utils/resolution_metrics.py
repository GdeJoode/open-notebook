"""Measurement harness for entity-resolution fragmentation and over-merge.

Two questions every matching change (K.1, K.2, K.4, K.5) must answer:

1. **Did fragmentation drop?** :func:`measure_fragmentation` groups a set of
   entity surface forms by their normalized canonical key and reports the
   distinct-canonical count, the cluster-size histogram, and — for clusters
   that actually merged ≥2 surface forms — the member surface forms. Comparing
   the report under the V1 baseline normalizer vs the new normalizer quantifies
   the win.

2. **Did anything over-merge?** :func:`count_false_merges` takes the adversarial
   ``must_not_merge`` pairs and counts how many collapsed to the same key under
   the candidate normalizer. The acceptance gate for every matching phase is
   that this stays **zero**.

   The false-merge gate is measured at the **NAME level**, ignoring
   ``entity_type``. Relations resolve their endpoints by name alone
   (``WHERE canonical_name = $name`` — no type filter), so two different-typed
   real entities that normalize to the same string corrupt the graph even
   though the ``(canonical_name, entity_type)`` dedup key keeps their entity
   rows distinct. A ``(name, type)``-level check would miss exactly that case.
   :func:`measure_fragmentation` still keys on ``(name, type)`` because the
   fragmentation count it reports mirrors the persistence dedup key.

The harness is intentionally DB-free and deterministic: it operates on plain
dicts (``{"canonical_name", "entity_type"}``) so it can run over a frozen
fixture without re-extraction, and it takes the normalizer as a callable so the
same code measures both the baseline and the candidate.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

# A normalizer is any ``str -> str`` (e.g. ``normalize_entity_name`` or the V1
# baseline). Entities are dict-like with at least ``canonical_name``; the
# optional ``entity_type`` is folded into the cluster key so two different-typed
# entities sharing a surface form (``Groningen`` the province vs the city) are
# not counted as one cluster.
Normalizer = Callable[[str], str]
EntityRecord = Mapping[str, object]


@dataclass(frozen=True)
class MergedCluster:
    """One canonical key onto which ≥2 distinct surface forms collapsed."""

    canonical_key: str
    entity_type: str
    members: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class FragmentationReport:
    """Outcome of running a normalizer over an entity set."""

    total_entities: int
    distinct_canonical: int
    cluster_size_histogram: dict[int, int] = field(default_factory=dict)
    merged_clusters: list[MergedCluster] = field(default_factory=list)

    def cluster_for(self, surface_form: str) -> MergedCluster | None:
        """Return the merged cluster a surface form landed in, if any."""
        for cluster in self.merged_clusters:
            if surface_form in cluster.members:
                return cluster
        return None


def measure_fragmentation(
    entities: Iterable[EntityRecord],
    normalizer: Normalizer,
    *,
    type_field: str = "entity_type",
    name_field: str = "canonical_name",
) -> FragmentationReport:
    """Group entities by normalized ``(name, type)`` key and report the spread.

    Args:
        entities: Iterable of dict-like records carrying at least a name field.
        normalizer: The ``str -> str`` normalizer to measure.
        type_field: Record key for the entity type (folded into the cluster key
            so different-typed homographs stay distinct).
        name_field: Record key for the surface form.

    Returns:
        A :class:`FragmentationReport`. ``distinct_canonical`` is the number of
        distinct ``(normalized_name, entity_type)`` keys — the headline
        fragmentation number. ``merged_clusters`` lists only the keys onto which
        ≥2 *distinct* surface forms collapsed (the actual dedup wins).
    """
    # key -> set of original surface forms that map onto it.
    clusters: dict[tuple[str, str], set[str]] = defaultdict(set)
    total = 0
    for record in entities:
        total += 1
        surface = str(record.get(name_field, "") or "")
        etype = str(record.get(type_field, "") or "")
        key = (normalizer(surface), etype)
        clusters[key].add(surface)

    histogram = Counter(len(forms) for forms in clusters.values())
    merged = [
        MergedCluster(
            canonical_key=key[0],
            entity_type=key[1],
            members=tuple(sorted(forms)),
        )
        for key, forms in clusters.items()
        if len(forms) >= 2
    ]
    merged.sort(key=lambda c: (-c.size, c.canonical_key))

    return FragmentationReport(
        total_entities=total,
        distinct_canonical=len(clusters),
        cluster_size_histogram=dict(sorted(histogram.items())),
        merged_clusters=merged,
    )


def _pair_keys(
    pair: Mapping[str, object] | Sequence[object],
    normalizer: Normalizer,
) -> tuple[tuple[str, str], tuple[str, str]]:
    """Extract the two normalized ``(name, type)`` keys from a corpus pair.

    Accepts two shapes so the JSONL fixtures can stay readable:

    - ``{"a": {"name": ..., "type": ...}, "b": {...}}``
    - ``{"a": "...", "b": "...", "entity_type": "..."}`` (shared type)
    """
    if "a" in pair and "b" in pair:
        a = pair["a"]
        b = pair["b"]
    else:  # positional ["nameA", "nameB", optional type]
        seq = list(pair)  # type: ignore[arg-type]
        a, b = seq[0], seq[1]

    shared_type = str(pair.get("entity_type", "")) if isinstance(pair, Mapping) else ""

    def one(side: object) -> tuple[str, str]:
        if isinstance(side, Mapping):
            name = str(side.get("name", side.get("canonical_name", "")))
            etype = str(side.get("type", side.get("entity_type", shared_type)))
        else:
            name = str(side)
            etype = shared_type
        return (normalizer(name), etype)

    return one(a), one(b)


def count_false_merges(
    must_not_merge_pairs: Iterable[Mapping[str, object] | Sequence[object]],
    normalizer: Normalizer,
) -> int:
    """Count adversarial pairs that wrongly collapse to the same NAME.

    A pair is a false merge when both sides normalize to the *same string*,
    **ignoring ``entity_type``**. This is the criterion relations actually
    depend on: endpoints resolve by name alone (``WHERE canonical_name =
    $name``), so a name-only collision between two different-typed real
    entities corrupts the graph even though their ``(name, type)`` dedup keys
    differ. The acceptance gate for every matching phase is that this returns
    ``0`` over the ``must_not_merge`` corpus.

    For a ``(name, type)``-level measurement (used in fragmentation reporting,
    not the false-merge gate), see :func:`count_false_merges_with_type`.

    Args:
        must_not_merge_pairs: Iterable of pair records (see :func:`_pair_keys`).
        normalizer: The normalizer under test.

    Returns:
        The number of pairs whose names collapsed (0 means no over-merges).
    """
    false_merges = 0
    for pair in must_not_merge_pairs:
        (name_a, _ta), (name_b, _tb) = _pair_keys(pair, normalizer)
        if name_a == name_b:
            false_merges += 1
    return false_merges


def count_false_merges_with_type(
    must_not_merge_pairs: Iterable[Mapping[str, object] | Sequence[object]],
    normalizer: Normalizer,
) -> int:
    """Count adversarial pairs that collapse to the same ``(name, type)`` key.

    The weaker, type-aware companion to :func:`count_false_merges`. Two
    different-typed entities sharing a normalized name are NOT counted here,
    because the persistence dedup key keeps their entity rows distinct. This is
    useful for fragmentation/diagnostic reporting, but it is **not** the
    false-merge gate — relations ignore type, so the name-only count is the one
    that protects the graph.

    Args:
        must_not_merge_pairs: Iterable of pair records (see :func:`_pair_keys`).
        normalizer: The normalizer under test.

    Returns:
        The number of pairs that collapsed at ``(name, type)`` granularity.
    """
    false_merges = 0
    for pair in must_not_merge_pairs:
        key_a, key_b = _pair_keys(pair, normalizer)
        if key_a == key_b:
            false_merges += 1
    return false_merges


def count_unmerged_must_merge(
    must_merge_pairs: Iterable[Mapping[str, object] | Sequence[object]],
    normalizer: Normalizer,
) -> int:
    """Count must-merge pairs that *failed* to collapse to one key.

    The dual of :func:`count_false_merges` for the ``must_merge`` corpus: a
    healthy normalizer returns ``0`` (every pair merged).
    """
    unmerged = 0
    for pair in must_merge_pairs:
        key_a, key_b = _pair_keys(pair, normalizer)
        if key_a != key_b:
            unmerged += 1
    return unmerged
