"""Meting: duur en geheugen van een herclassificatie (F2-03, reviewpunt 4).

Alleen met ``F2_03_METING=1``; anders overgeslagen. Synthetisch: één bron met
``F2_03_FRAGMENTEN`` fragmenten (standaard 2000), evenveel embeddings van 1024
dimensies, 500 entiteiten met een vermelding in de bron (met relaties, aliassen
en triage-items) en 500 entiteiten zonder. Draai met ``-s`` om de uitkomst te
zien::

    F2_03_METING=1 uv run --project packages/surrealdb-service pytest -s \
        packages/surrealdb-service/tests/test_herclassificatie_meting.py
"""

from __future__ import annotations

import os
import random
import subprocess
import time

import pytest
from surrealdb import RecordID
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.source import SourceRepository
from surrealdb_service.testing import fixtures as fx

pytestmark = [
    pytest.mark.requires_docker,
    pytest.mark.asyncio,
    pytest.mark.skipif(os.environ.get("F2_03_METING") != "1", reason="meting: F2_03_METING=1"),
]

FRAGMENTEN = int(os.environ.get("F2_03_FRAGMENTEN", "2000"))
ENTITEITEN = 500
BATCH = 200


def _geheugen() -> str:
    """Geheugen van de SurrealDB-testcontainer volgens ``docker stats``."""
    ids = subprocess.run(
        ["docker", "ps", "-q", "--filter", f"ancestor={fx.SURREALDB_IMAGE}"],
        capture_output=True, text=True, check=False,
    ).stdout.split()
    if not ids:
        return "onbekend"
    uit = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", *ids],
        capture_output=True, text=True, check=False,
    ).stdout
    return " ".join(uit.split()) or "onbekend"


async def _q(config: SurrealDBConfig, sql: str, params: dict | None = None):
    return await execute_query(sql, params or {}, config)


async def test_herclassificatie_van_een_grote_bron(live_surrealdb: SurrealDBConfig) -> None:
    rnd = random.Random(3)
    [bron] = await _q(live_surrealdb, "CREATE source SET title = 'Meting', classification = 'extern' RETURN id")
    bron_id = str(bron["id"])
    b = RecordID("source", bron_id.split(":", 1)[1])

    for start in range(0, FRAGMENTEN, BATCH):
        n = min(BATCH, FRAGMENTEN - start)
        fragmenten = await _q(live_surrealdb, "INSERT INTO chunk $rijen RETURN id", {"rijen": [
            {"source": b, "text": f"tekst {start + i}", "order": start + i, "physical_page": 0,
             "element_type": "paragraph", "positions": [], "metadata": {}}
            for i in range(n)
        ]})
        await _q(live_surrealdb, "INSERT INTO source_embedding $rijen RETURN NONE", {"rijen": [
            {"source": b, "chunk": RecordID("chunk", str(f["id"]).split(":", 1)[1]),
             "content": "t", "order": start + i, "embedding": [rnd.random() for _ in range(1024)]}
            for i, f in enumerate(fragmenten)
        ]})
        await _q(live_surrealdb, "INSERT INTO doc_node $rijen RETURN NONE", {"rijen": [
            {"source": b, "element_type": "paragraph", "self_ref": f"#/texts/{start + i}",
             "sequence": start + i}
            for i in range(n)
        ]})

    for i in range(ENTITEITEN * 2):
        bronnen = [bron_id] if i < ENTITEITEN else ["source:anders"]
        [e] = await _q(
            live_surrealdb,
            "CREATE entity SET canonical_name = $n, name_key = $n, entity_type = 'thing',"
            " embedding = [], source_documents = $s RETURN id",
            {"n": f"meting {i}", "s": bronnen},
        )
        if i < ENTITEITEN:
            await _q(
                live_surrealdb,
                "RELATE $e->relation->$e SET relation_type = 'related_to', source_documents = $s;"
                " CREATE entity_alias SET canonical_entity = $e, alias_text = 'a', match_type = 'exact';"
                " CREATE triage_queue SET entity = $e, batch_id = 'b', name = 'x', reason = 'r';",
                {"e": RecordID("entity", str(e["id"]).split(":", 1)[1]), "s": bronnen},
            )

    repo = SourceRepository(config=live_surrealdb)
    voor = _geheugen()
    uitkomst = []
    for van, naar in (("extern", "intern"), ("intern", "extern")):
        start = time.perf_counter()
        await repo.reclassify(bron_id, from_classification=van, to_classification=naar,
                              reason="meting", agent_key="agent_keys:meting", person="Meting")
        uitkomst.append((van, naar, time.perf_counter() - start, _geheugen()))

    tellingen = await _q(
        live_surrealdb,
        "RETURN {fragmenten: count(SELECT id FROM chunk WHERE source = $b AND classification = 'extern'),"
        " entiteiten: count(SELECT id FROM entity WHERE source_documents CONTAINSANY [$s]"
        " AND classification = 'extern')}",
        {"b": b, "s": bron_id},
    )
    tellingen = tellingen[0] if isinstance(tellingen, list) else tellingen
    print(f"\nF2-03 meting: {FRAGMENTEN} fragmenten en embeddings, {ENTITEITEN} entiteiten met"
          f" vermelding, {ENTITEITEN} zonder; geheugen voor: {voor}")
    for van, naar, duur, geheugen in uitkomst:
        print(f"  {van} -> {naar}: {duur:.2f} s; geheugen na: {geheugen}")
    print(f"  na afloop extern: {tellingen}")
    assert tellingen == {"fragmenten": FRAGMENTEN, "entiteiten": ENTITEITEN}
