"""What stage 10 (KG resolution) does to a real variant class — PC.3 AC #2/#3/#4.

The phase plan assumed the measurement needed a re-extraction of the corpus.
It does not, for the question the acceptance criteria actually ask. Stage 10's
question is "given this mention, what does it match in the graph?", and that can
be asked directly, twenty times, in seconds.

WHY NOT `run_filtering_only`. That path replays filtering over a stored
`extraction_result` without re-extracting, which would have been ideal. Only 6 of
the 14 sources still have one, and they are the English-language papers — not a
single convenant. Two of 531 active entities and one of twenty
`government_organization` rows touch a replayable source, so a replay measures
nothing this phase asks about.

TWO MEASUREMENTS.

A. Against the LIVE repository. Each of the twenty active surface forms is posed
   as a mention to the real `EntityRepository`. Every one matches ITSELF at fuzzy
   1.000, which is the expected and uninteresting answer — its value is what it
   measures around that: whether the candidate cap binds, and how much of the
   candidate pool is rows triage has retired. NOTE the limit: this shows that an
   identical ACTIVE row wins the tie. It does NOT show that archived rows are
   harmless to a mention with no active twin, which this probe cannot reach.

B. Incremental build from empty. The twenty arrive one at a time in creation
   order and each asks the resolver whether it matches what is already there —
   which is what a real run does. The resolver, its thresholds, its tiers and its
   scoring are the production objects; only the repository is a stand-in, and it
   serves REAL rows with REAL embeddings through `find_by_type`'s own contract
   (id / name / embedding / weight, bounded by `limit`). It exists solely to pose
   the counterfactual "a graph that does not yet contain this row", which no live
   repository can answer.

FIDELITY OF THE EMBEDDINGS. A mention's vector is produced by
`_embed_entities` over `entity.text`; a stored entity's by
`backfill_entity_embeddings` over `canonical_name`. Both are the bare name, so
the stored vectors used here are the right KIND. (`semantic-intelligence/scripts/
test_pipeline.py` embeds `f"{etype}: {name}. {desc}"` instead — a third producer
that disagrees with the other two, recorded as a finding, not exercised here.)

Usage::

    SURREAL_DATABASE=staging uv run python scripts/pc3_resolution_measurement.py
    SURREAL_DATABASE=staging uv run python scripts/pc3_resolution_measurement.py --sweep

Read-only. `register_aliases=False`, and the stand-in repository raises if the
resolver ever tries to write.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from entity_filtering.config import KGResolutionConfig
from entity_filtering.resolution.kg_resolver import KGResolver
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.entity import EntityRepository

TYPE = "government_organization"
CFG = KGResolutionConfig(enabled=True)

#: The four groups PC.3's AC #2 names as one organisation under several surface
#: forms. A merge WITHIN a group is correct; any other merge is not.
INTENDED: Dict[str, set] = {
    "BZK": {
        "Binnenlandse Zaken en Koninkrijkrelaties (Ministerie)",
        "BZK (Binnenlandse Zaken en Koninkrijkrelaties)",
    },
    "IenW": {
        "IenW",
        "IenW (Infrastructuur en Waterstaat)",
        "Infrastructuur en Waterstaat (Ministerie)",
    },
    "VRO": {
        "Volkshuisvesting en Ruimtelijke Ordening (Ministerie)",
        "VRO (Volkshuisvesting en Ruimtelijke Ordening)",
        "VRO (ministerie van Volkshuisvesting en Ruimtelijke Ordening)",
    },
    "HG/BZK": {
        "Binnenlandse Zaken en Koninkrijkrelaties/Herstel Groningen",
        "HG/BZK (Herstel Groningen/Binnenlandse Zaken en Koninkrijkrelaties)",
    },
}


def group_of(name: str) -> Optional[str]:
    for group, members in INTENDED.items():
        if name in members:
            return group
    return None


class GrowingRepo:
    """A graph that grows as entities arrive. Mirrors `find_by_type`'s contract."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    async def find_by_alias(self, alias_text: str) -> Optional[Dict[str, Any]]:
        # `entity_alias` holds no rows of this type, so tier 1 is genuinely dead
        # here rather than stubbed away.
        return None

    async def find_by_type(self, entity_type: str, limit: int = 100):
        return [r for r in self.rows if r["_type"] == entity_type][:limit]

    async def register_alias(self, *a: Any, **k: Any) -> bool:
        raise AssertionError("this measurement must not write")


def _resolver(repo: Any, fuzzy: float, semantic: float) -> KGResolver:
    return KGResolver(
        entity_repo=repo,
        fuzzy_threshold=fuzzy,
        semantic_threshold=semantic,
        max_candidates=CFG.max_candidates,
        register_aliases=False,
        mark_new_entities=CFG.mark_new_entities,
        use_alias_table=CFG.use_alias_table,
    )


async def _load() -> List[Dict[str, Any]]:
    # SurrealDB requires an ORDER BY column to appear in the projection.
    return await execute_query(
        "SELECT id, name, embedding, weight, created_at FROM entity "
        f"WHERE entity_type = '{TYPE}' AND status = 'active' ORDER BY created_at;"
    )


async def probe_live(rows: List[Dict[str, Any]]) -> None:
    allrows = await execute_query(
        f"SELECT id, name, status FROM entity WHERE entity_type = '{TYPE}';"
    )
    status_of = {str(r["id"]): (r.get("status"), r.get("name")) for r in allrows}
    archived = sum(1 for v in status_of.values() if v[0] == "archived")

    resolver = _resolver(EntityRepository(), CFG.fuzzy_threshold, CFG.semantic_threshold)
    off_self = 0
    report: Dict[str, Any] = {}
    for row in rows:
        mention = {
            "text": row["name"],
            "label": TYPE,
            "properties": {"embedding": row.get("embedding") or []},
        }
        out, report = await resolver.resolve([mention])
        props = out[0].get("properties", {})
        matched = str(props.get("kg_entity_id", ""))
        if matched != str(row["id"]):
            off_self += 1
            st, name = status_of.get(matched, ("?", "?"))
            print(f"  {row['name'][:44]:44} -> [{st}] {name}")
    print(f"  every mention matched its own row: {off_self == 0}")
    print(
        f"  candidate pool for this type: {len(allrows)} rows, {archived} archived "
        f"— `find_by_type` applies NO status filter"
    )
    print(
        f"  cap: fetches={report.get('candidate_fetches')} "
        f"capped={report.get('capped_fetches')} cap={report.get('candidate_cap')}"
    )


async def build(
    rows: List[Dict[str, Any]], fuzzy: float, semantic: float
) -> Tuple[int, List[Tuple], List[Tuple]]:
    grow = GrowingRepo()
    resolver = _resolver(grow, fuzzy, semantic)
    merges: List[Tuple] = []
    for row in rows:
        mention = {
            "text": row["name"],
            "label": TYPE,
            "properties": {"embedding": row.get("embedding") or []},
        }
        out, _ = await resolver.resolve([mention])
        props = out[0].get("properties", {})
        if props.get("is_new"):
            grow.rows.append(
                {
                    "id": str(row["id"]),
                    "name": row["name"],
                    "embedding": row.get("embedding") or [],
                    "weight": row.get("weight") or 0,
                    "_type": TYPE,
                }
            )
        else:
            target = next(
                x for x in grow.rows if x["id"] == str(props["kg_entity_id"])
            )
            merges.append(
                (
                    row["name"],
                    target["name"],
                    props["kg_match_type"],
                    round(props["kg_similarity_score"], 3),
                )
            )
    correct = [
        m for m in merges if group_of(m[0]) and group_of(m[0]) == group_of(m[1])
    ]
    wrong = [m for m in merges if m not in correct]
    return len(grow.rows), correct, wrong


async def main(sweep: bool) -> None:
    rows = await _load()
    print(f"{len(rows)} active {TYPE} rows, in creation order\n")

    print("=" * 78)
    print("A. Against the live graph")
    print("=" * 78)
    await probe_live(rows)

    print()
    print("=" * 78)
    print("B. Incremental build from empty, at the SHIPPED thresholds "
          f"(fuzzy {CFG.fuzzy_threshold} / semantic {CFG.semantic_threshold})")
    print("=" * 78)
    n, correct, wrong = await build(rows, CFG.fuzzy_threshold, CFG.semantic_threshold)
    print(f"  {len(rows)} rows -> {n} entities")
    print(f"  CORRECT merges ({len(correct)}) — within a group AC #2 names:")
    for m in correct:
        print(f"    {m[0][:46]:46} -> {m[1][:34]:34} [{m[2]} {m[3]}]")
    print(f"  WRONG merges ({len(wrong)}):")
    for m in wrong:
        print(f"    {m[0][:46]:46} -> {m[1][:34]:34} [{m[2]} {m[3]}]")

    if not sweep:
        return

    print()
    print("=" * 78)
    print("C. Is there ANY operating point where it helps?")
    print("=" * 78)
    print(f"{'fuzzy':>6} {'sem':>6} {'entities':>9} {'correct':>8} {'wrong':>6}")
    for fuzzy, semantic in [
        (0.85, 0.90), (0.85, 0.95), (0.90, 0.90), (0.90, 0.95),
        (0.92, 0.95), (0.90, 0.97), (0.95, 0.97),
    ]:
        n, correct, wrong = await build(rows, fuzzy, semantic)
        mark = "  <- shipped" if (fuzzy, semantic) == (0.85, 0.90) else ""
        print(f"{fuzzy:>6.2f} {semantic:>6.2f} {n:>9} {len(correct):>8} "
              f"{len(wrong):>6}{mark}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true",
                        help="also sweep the thresholds (measurement C)")
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main(args.sweep))
