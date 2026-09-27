"""PC.5a: gaps grouped by LABEL, scoped to a notebook — against a real SurrealDB.

The evolution agent proposed a type per entity TEXT (an instance promoted to a
type). These tests pin the replacement: distinct entities per label, only gaps
from this notebook's sources, ignored gaps left out, and a threshold on
distinct entities rather than on raw occurrences.
"""

from __future__ import annotations

import uuid

import pytest
from app_main.services.gap_label_proposals import label_proposals_for_notebook
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]


async def _one(config, sql, params=None):
    rows = await execute_query(sql, params or {}, config=config)
    return str(rows[0])


async def _gap(config, text, label, source, status="pending", frequency=1):
    await execute_query(
        "CREATE ontology_gap SET entity_text = $t, entity_type_guess = $l, "
        "source_documents = [$s], ontology_name = 'deals', status = $st, "
        "frequency = $f, created_at = time::now();",
        {"t": text, "l": label, "s": source, "st": status, "f": frequency},
        config=config,
    )


async def test_labels_with_enough_distinct_entities_in_this_notebook(
    live_surrealdb: SurrealDBConfig,
):
    tag = uuid.uuid4().hex[:6]
    nb = await _one(live_surrealdb, "CREATE notebook SET name = $n RETURN VALUE id;",
                    {"n": f"nb-{tag}"})
    other_nb = await _one(live_surrealdb, "CREATE notebook SET name = $n RETURN VALUE id;",
                          {"n": f"other-{tag}"})
    src = await _one(live_surrealdb, "CREATE source SET title = 'a' RETURN VALUE id;")
    other_src = await _one(live_surrealdb, "CREATE source SET title = 'b' RETURN VALUE id;")
    await execute_query("RELATE $s->reference->$n;",
                        {"s": _rid(src), "n": _rid(nb)}, config=live_surrealdb)
    await execute_query("RELATE $s->reference->$n;",
                        {"s": _rid(other_src), "n": _rid(other_nb)}, config=live_surrealdb)

    label = f"Programmalijn{tag}"
    # Five distinct entities — one of them twice in different casing, and one
    # with a high frequency, which must count once, not five times.
    for text in ("Economie", "economie", "Wonen", "Zorg", "Energie", "Onderwijs"):
        await _gap(live_surrealdb, text, label, src)
    await _gap(live_surrealdb, "Mobiliteit", label, src, frequency=9)
    # Another label, four distinct entities: below the threshold.
    for text in ("A", "B", "C", "D"):
        await _gap(live_surrealdb, text, f"Opgave{tag}", src)
    # Gaps from another notebook's source and an ignored gap do not count.
    for text in ("X1", "X2", "X3", "X4", "X5"):
        await _gap(live_surrealdb, text, f"Elders{tag}", other_src)
    await _gap(live_surrealdb, "Genegeerd", f"Opgave{tag}", src, status="ignored")

    proposals = await label_proposals_for_notebook(nb, threshold=5, config=live_surrealdb)
    assert [p["type_name"] for p in proposals] == [label]
    [p] = proposals
    assert p["evidence_count"] == 6  # Economie/economie once, Mobiliteit once
    assert p["origin"] == "ontology_gap"
    # Case-insensitive order, first five; `Economie` is shown over `economie`
    # whatever order the rows come back in.
    assert p["examples"] == ["Economie", "Energie", "Mobiliteit", "Onderwijs", "Wonen"]
    assert p["rationale"].startswith(f"6 entities labelled '{label}' matched no type")


async def test_a_notebook_without_sources_gets_nothing(live_surrealdb: SurrealDBConfig):
    nb = await _one(live_surrealdb, "CREATE notebook SET name = 'empty' RETURN VALUE id;")
    assert await label_proposals_for_notebook(nb, threshold=1, config=live_surrealdb) == []


def _rid(value: str):
    from surrealdb_service.connection import ensure_record_id

    return ensure_record_id(value)
