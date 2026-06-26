"""R.6 search-facing noise measurement — before/after, READ-ONLY (Track R.6).

Connects to the live DB (set ``SURREAL_DATABASE=staging`` explicitly), loads the
active entity → source map + active relations through the same repo loaders the
R.2 retrieval path uses, and reports the effect of the R.6 search-facing
normalization WITHOUT writing anything:

  1. entity-normalization deltas — distinct case/type-duplicate concepts unified,
     singletons (df==1) excluded, generic-bucket share in the scored set;
  2. the convenant-cluster score effect — per query source, the raw R.2 ranking
     vs the R.6-normalized ranking (does unifying the case/type duplicates raise
     the right convenant scores?);
  3. predicate canonicalization — how many active relation predicates collapse
     onto a canonical form.

This is the AC4 measurement deliverable. NO MUTATIONS — only SELECTs via the
read-only loaders + pure in-memory normalization.

Run:
    SURREAL_DATABASE=staging uv run --project apps/app-main \
        python scripts/r6_noise_measurement.py
"""

from __future__ import annotations

import asyncio
import os
from collections import Counter
from typing import Dict, List, Tuple

from shared.retrieval.kg_signal_normalizer import (
    canonical_predicate,
    normalize_entities_for_signal,
)
from shared.retrieval.kg_source_scorer import (
    EntityRecord,
    type_salience,
    score_related_sources,
)

_GENERIC_TYPES = {"topic", "concept", "other"}


def _to_records(rows: List[dict]) -> List[EntityRecord]:
    out: List[EntityRecord] = []
    for row in rows or []:
        eid = row.get("id")
        if eid is None:
            continue
        out.append(
            EntityRecord(
                entity_id=str(eid),
                canonical_name=str(row.get("canonical_name") or ""),
                entity_type=str(row.get("entity_type") or ""),
                source_documents=tuple(
                    str(s) for s in (row.get("source_documents") or [])
                ),
            )
        )
    return out


def _generic_share(entities: List[EntityRecord]) -> Tuple[int, int, float]:
    generic = sum(
        1 for e in entities if str(e.entity_type).lower() in _GENERIC_TYPES
    )
    total = len(entities)
    pct = (100.0 * generic / total) if total else 0.0
    return generic, total, pct


def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


async def main() -> None:
    from surrealdb_service.config import SurrealDBConfig
    import surrealdb_service.connection as conn
    from surrealdb_service.repositories import EntityRepository, SourceRepository

    cfg = SurrealDBConfig.from_env()
    print(f"DB={cfg.database} NS={cfg.namespace}  (READ-ONLY)")
    if cfg.database != "staging":
        print(
            "WARNING: SURREAL_DATABASE is not 'staging' — set it explicitly to "
            "measure against staging."
        )

    conn._pool = conn.ConnectionPool(config=cfg)

    entity_repo = EntityRepository(config=cfg)
    source_repo = SourceRepository(config=cfg)

    entity_rows = await entity_repo.load_active_entity_source_map()
    relation_rows = await entity_repo.load_active_relations()
    n_sources = await source_repo.count()
    raw = _to_records(entity_rows)

    # ---------------------------------------------------------------- (1) entity
    _print_header("1. ENTITY NORMALIZATION (search-facing projection)")
    # No singleton drop: isolates the case/type UNIFICATION effect.
    grouped_keep, stats_keep = normalize_entities_for_signal(
        raw, drop_singletons=False
    )
    # Default: with singleton drop (what the live signal uses).
    grouped_drop, stats_drop = normalize_entities_for_signal(
        raw, drop_singletons=True
    )

    print(f"raw active entity rows                : {stats_keep.input_entities}")
    print(f"  skipped (no name / no sources)      : {stats_keep.empty_skipped}")
    print(f"distinct concepts after case/type merge: {stats_keep.grouped_concepts}")
    print(
        f"  concepts that ABSORBED >1 row (merged): {stats_keep.merged_groups}"
        f"  -> {stats_keep.input_entities - stats_keep.empty_skipped}"
        f" rows collapsed into {stats_keep.grouped_concepts} concepts"
    )
    print(f"singletons (df==1) excluded from signal : {stats_drop.singletons_dropped}")
    print(f"concepts emitted to scorer (final)      : {stats_drop.emitted_concepts}")

    g_raw, t_raw, p_raw = _generic_share(raw)
    g_keep, t_keep, p_keep = _generic_share(grouped_keep)
    g_drop, t_drop, p_drop = _generic_share(grouped_drop)
    print("\ngeneric-bucket (topic/concept/other) share by COUNT of concepts:")
    print(f"  raw R.2                : {g_raw}/{t_raw} = {p_raw:5.1f}%")
    print(f"  +case/type merge       : {g_keep}/{t_keep} = {p_keep:5.1f}%")
    print(f"  +singleton drop (final): {g_drop}/{t_drop} = {p_drop:5.1f}%")
    print(
        "  (count-share rises post-drop because most NAMED entities are "
        "source-unique singletons; the meaningful number is the generic share "
        "of actual ranking WEIGHT, below.)"
    )

    # The number that actually matters for ranking: of the total contribution
    # weight that lands in the top-ranked results, how much comes from generic
    # buckets vs named entities. Measured across every query source's top-k.
    def _generic_weight_share(entities: List[EntityRecord]) -> float:
        gen = tot = 0.0
        seen_q = {
            s for e in entities for s in e.source_documents
        }
        for qid in seen_q:
            for sc in score_related_sources(
                qid, entities, n_sources=n_sources, limit=10
            ):
                for c in sc.contributions:
                    tot += c.weight
                    if str(c.entity_type).lower() in _GENERIC_TYPES:
                        gen += c.weight
        return (100.0 * gen / tot) if tot else 0.0

    raw_wpct = _generic_weight_share(raw)
    norm_wpct = _generic_weight_share(grouped_drop)
    print("\ngeneric-bucket share of RANKING WEIGHT in top-10 results:")
    print(f"  raw R.2                : {raw_wpct:5.1f}%")
    print(f"  R.6-normalized (final) : {norm_wpct:5.1f}%")

    # Show the concrete duplicate merges (members > 1).
    _print_header("1b. SPECIFIC CASE/TYPE DUPLICATES UNIFIED")
    # Recompute group membership to display names (re-run grouping by hand for
    # display only — same keying as the normalizer).
    from shared.retrieval.kg_signal_normalizer import _concept_key  # noqa

    members: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for e in raw:
        if not e.canonical_name.strip() or not e.source_documents:
            continue
        key = _concept_key(e.canonical_name, e.entity_type)
        members.setdefault(key, []).append((e.canonical_name, e.entity_type))
    merged = {k: v for k, v in members.items() if len(v) > 1}
    print(f"{len(merged)} concept(s) unify >1 raw row:")
    for key, vs in sorted(merged.items()):
        forms = ", ".join(f'"{n}"[{t}]' for n, t in vs)
        print(f"  {key[0]!r}: {forms}")

    # ------------------------------------------------------------- (3) predicate
    _print_header("3. PREDICATE CANONICALIZATION (active relations)")
    raw_preds = Counter(
        str(r.get("relation_type") or "") for r in (relation_rows or [])
    )
    canon_preds = Counter(canonical_predicate(p) for p in raw_preds.elements())
    print(f"active relations            : {sum(raw_preds.values())}")
    print(f"distinct raw predicates     : {len(raw_preds)}")
    print(f"distinct canon predicates   : {len(canon_preds)}")
    # Show which raw forms map to a different canonical (the actual dedup).
    changed = sorted(
        {
            (p, canonical_predicate(p))
            for p in raw_preds
            if canonical_predicate(p) != p
        }
    )
    edges_rewritten = sum(raw_preds[src] for src, _ in changed)
    print(
        f"raw predicate forms rewritten: {len(changed)} "
        f"({edges_rewritten} of {sum(raw_preds.values())} edges)"
    )
    if changed:
        print("raw -> canonical (changed forms):")
        for src, dst in changed:
            print(f"  {src!r:32} -> {dst!r}  (x{raw_preds[src]})")
    else:
        print("(no raw predicate differs from its canonical form on this data)")

    # -------------------------------------------------- (2) convenant score effect
    _print_header("2. CONVENANT-CLUSTER SCORE EFFECT (raw R.2 vs R.6-normalized)")
    titles_rows = await source_repo.get_titles_by_ids(
        list(
            {
                s
                for e in raw
                for s in e.source_documents
            }
        )
    )

    def _short(sid: str) -> str:
        return (titles_rows.get(sid) or sid)[:46]

    # Pick query sources that look like convenanten by title.
    query_ids = [
        sid
        for sid, title in titles_rows.items()
        if "convenant" in (title or "").lower()
    ] or list(titles_rows.keys())

    for qid in query_ids:
        print(f"\nQUERY: {_short(qid)}  ({qid})")
        raw_scores = score_related_sources(qid, raw, n_sources=n_sources, limit=5)
        norm_scores = score_related_sources(
            qid, grouped_drop, n_sources=n_sources, limit=5
        )
        raw_map = {s.source_id: s.score for s in raw_scores}
        norm_map = {s.source_id: s.score for s in norm_scores}
        all_ids = list(
            dict.fromkeys([*raw_map.keys(), *norm_map.keys()])
        )
        print(f"  {'related source':48} {'rawR2':>8} {'R6norm':>8} {'Δ':>8}")
        for rid in sorted(
            all_ids, key=lambda x: -(norm_map.get(x, 0.0))
        ):
            r = raw_map.get(rid, 0.0)
            nrm = norm_map.get(rid, 0.0)
            print(f"  {_short(rid):48} {r:8.3f} {nrm:8.3f} {nrm - r:+8.3f}")


if __name__ == "__main__":
    if os.getenv("SURREAL_DATABASE") != "staging":
        print(
            "Set SURREAL_DATABASE=staging to run the staging measurement "
            "(read-only). Aborting to avoid the default local DB."
        )
        raise SystemExit(0)
    asyncio.run(main())
