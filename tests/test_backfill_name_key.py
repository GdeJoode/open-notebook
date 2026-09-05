"""The backfill refuses rather than merges (PC.3).

Two rows whose names normalise to one key are the duplication this phase exists
to remove — and merging them picks a survivor, repoints relations and discards a
name. That is a curator's decision, for which PC.2 shipped the door: a same-type
pair equal under the fold lands in the auto-merge band of the candidate queue.

These tests drive `plan()`, the pure part, so they need no database. The
collision case is the one that matters: a backfill that merged silently would be
the exact failure PC.6 spent five review rounds making unreachable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from shared.utils.name_normalizer import normalize_entity_name

_SPEC = importlib.util.spec_from_file_location(
    "backfill_name_key",
    Path(__file__).resolve().parents[1] / "scripts" / "backfill_name_key.py",
)
backfill = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(backfill)


def _row(rid: str, name: str, etype: str = "organization") -> dict:
    return {
        "id": rid,
        "canonical_name": name,
        "entity_type": etype,
        "status": "active",
    }


def test_case_variants_are_reported_as_a_collision() -> None:
    keys, collisions = backfill.plan(
        [_row("entity:a", "Brede Welvaart"), _row("entity:b", "brede welvaart")]
    )
    assert len(collisions) == 1
    assert {r["id"] for r in collisions[0]} == {"entity:a", "entity:b"}
    assert keys["entity:a"] == keys["entity:b"]


def test_the_same_key_under_two_types_is_not_a_collision() -> None:
    """The index is on `(name_key, entity_type)`, so the plan must group by both.

    PC.2 routed byte-identical names split across `entity_type` to a curator as
    review-only, because a machine must not decide that one. Reporting them here
    would send a curator after a non-problem and, worse, train them to ignore the
    report.
    """
    _, collisions = backfill.plan(
        [
            _row("entity:a", "Regio Deal", "programme"),
            _row("entity:b", "Regio Deal", "topic"),
        ]
    )
    assert collisions == []


def test_distinct_entities_are_not_collapsed() -> None:
    """`normalize_entity_name` claims collision-safety; PC.3 tests it.

    Putting it on the write key means it decides identity, and the docstring's
    example — `Ministerie van Onderwijs` and `Onderwijs` stay distinct — is
    exactly what must hold. Inheriting that claim rather than testing it is what
    this phase's plan named as a risk.
    """
    pairs = [
        ("Ministerie van Onderwijs", "Onderwijs"),
        ("Gemeente Leudal", "Leudal"),
        ("Regio Deal Groningen", "Regio Deal Drenthe"),
        ("Provincie Groningen", "Gemeente Groningen"),
    ]
    rows = []
    for i, (left, right) in enumerate(pairs):
        rows += [_row(f"entity:l{i}", left), _row(f"entity:r{i}", right)]
    keys, collisions = backfill.plan(rows)
    assert collisions == [], (
        f"normalize_entity_name collapsed a distinct pair: "
        f"{[[r['canonical_name'] for r in g] for g in collisions]}"
    )
    for i, (left, right) in enumerate(pairs):
        assert keys[f"entity:l{i}"] != keys[f"entity:r{i}"], (left, right)


def test_a_clean_table_yields_a_key_per_row() -> None:
    rows = [_row("entity:a", "Alpha"), _row("entity:b", "Beta")]
    keys, collisions = backfill.plan(rows)
    assert collisions == []
    assert keys == {
        "entity:a": normalize_entity_name("Alpha"),
        "entity:b": normalize_entity_name("Beta"),
    }


def test_an_empty_table_is_not_a_collision() -> None:
    assert backfill.plan([]) == ({}, [])
