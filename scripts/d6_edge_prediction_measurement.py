"""D6 — what edge prediction (filtering stage 13) has written into the graph.

Question: when `EdgePredictor` writes a relation, does the source text support it?
Second question, on the way: how much of the graph did it produce?

IDENTIFICATION. Predicted edges are persisted with `relation_type = 'RELATED'` and
`properties.prediction_method = 'edge_predictor'`, and with
`extraction_method = 'llm'` like everything else. The property is what separates
them; nothing is reconstructed.

SCOPE. The snapshot's predicted edges were written 2026-06-23..09-02 under the
constructor's FALLBACK weights (cosine 0.5, adamic-adar 0.2). PC.6 (09-05) made
the workflow pass `EdgePredictionConfig` (0.6 / 0.3), so main today scores every
candidate at least as high: `census` re-scores the stored signals under both.

EVIDENCE. Entities carry no chunk link (PC.1b: `source_chunk_id` is never
persisted), and substring matching on chunks misses edges of both origins, so a
judge reads the WHOLE source document. `shared_chunks_total` is kept in the key
as a structural figure only.

Steps, each writing under ``--out``:

    census   -> census.json         every relation, classified; counts
    sample   -> sample_key.jsonl    stratified draw WITH origin (for the join)
                sample_blind.jsonl  what a judge may see
                judge_batches/      one document + its pairs per source
    (judge)  -> verdicts.jsonl      blind LLM judge, prompt in judge_prompt.md
    report   -> results.json        rates with Wilson 95% CIs, by stratum

Usage::

    SURREAL_HTTP=http://localhost:8010 python scripts/d6_edge_prediction_measurement.py census --out DIR
    ... sample --out DIR --seed 6 --n-pred 60 --n-llm 40
    ... report --out DIR

Read-only: every statement is a SELECT.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

HTTP = os.environ.get("SURREAL_HTTP", "http://localhost:8010")
NS = os.environ.get("SURREAL_NAMESPACE", "open_notebook")
DB = os.environ.get("SURREAL_DATABASE", "staging")
USER = os.environ.get("SURREAL_USER", "root")
PASS = os.environ.get("SURREAL_PASSWORD", "root")

SIGNALS = ("cosine_signal", "cooccurrence_signal", "adamic_adar_signal", "jaccard_signal")
WEIGHTS = {"cosine_signal": 0.5, "adamic_adar_signal": 0.2,
           "cooccurrence_signal": 0.1, "jaccard_signal": 0.2}


def sql(query: str) -> List[Any]:
    if not re.match(r"^\s*SELECT\b", query, re.I):
        raise ValueError("read-only script: SELECT statements only")
    req = urllib.request.Request(f"{HTTP}/sql", data=query.encode(), method="POST")
    token = base64.b64encode(f"{USER}:{PASS}".encode()).decode()
    req.add_header("Authorization", f"Basic {token}")
    req.add_header("surreal-ns", NS)
    req.add_header("surreal-db", DB)
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.load(resp)
    if out[0]["status"] != "OK":
        raise RuntimeError(out[0])
    return out[0]["result"]


def is_predicted(rel: Dict[str, Any]) -> bool:
    return (rel.get("properties") or {}).get("prediction_method") == "edge_predictor"


def signal_pattern(props: Dict[str, Any]) -> str:
    """Which signals contributed — the stratum a predicted edge is drawn from."""
    on = [s.split("_")[0] for s in SIGNALS if (props.get(s) or 0) > 0]
    return "+".join(on) or "none"


def dominant_signal(props: Dict[str, Any]) -> str:
    contrib = {s: WEIGHTS[s] * (props.get(s) or 0) for s in SIGNALS}
    best = max(contrib, key=contrib.get)
    return best.split("_")[0] if contrib[best] > 0 else "none"


def load() -> Dict[str, Any]:
    rels = sql("SELECT id, in, out, relation_type, confidence, status, properties, "
               "source_documents, extraction_method, created_at FROM relation;")
    ents = sql("SELECT id, canonical_name, name, aliases, entity_type, status, "
               "source_documents FROM entity;")
    return {"relations": rels, "entities": {e["id"]: e for e in ents}}


def cmd_census(out: Path) -> None:
    data = load()
    rels, ents = data["relations"], data["entities"]
    pred = [r for r in rels if is_predicted(r)]
    llm = [r for r in rels if not is_predicted(r)]

    pair_llm = defaultdict(list)
    for r in llm:
        pair_llm[frozenset((r["in"], r["out"]))].append(r["relation_type"])

    def endpoint_status(r):
        return "+".join(sorted((ents.get(r["in"], {}).get("status", "missing"),
                                ents.get(r["out"], {}).get("status", "missing"))))

    census = {
        "total_relations": len(rels),
        "predicted": len(pred),
        "llm": len(llm),
        "status": {"predicted": Counter(r["status"] for r in pred),
                   "llm": Counter(r["status"] for r in llm)},
        "extraction_method_on_predicted": Counter(r.get("extraction_method") for r in pred),
        "relation_type_on_predicted": Counter(r["relation_type"] for r in pred),
        "llm_edges_typed_RELATED": sum(r["relation_type"] == "RELATED" for r in llm),
        "confidence_predicted": Counter(round(r["confidence"], 2) for r in pred),
        "signal_pattern": Counter(signal_pattern(r["properties"]) for r in pred),
        "dominant_signal": Counter(dominant_signal(r["properties"]) for r in pred),
        "predicted_pair_also_has_llm_edge": sum(
            frozenset((r["in"], r["out"])) in pair_llm for r in pred),
        "predicted_extra_property_keys": Counter(
            k for r in pred for k in r["properties"] if k not in SIGNALS
            and k != "prediction_method"),
        "endpoint_status_predicted": Counter(endpoint_status(r) for r in pred),
        "endpoint_status_llm": Counter(endpoint_status(r) for r in llm),
        "endpoint_types_predicted": Counter(
            "|".join(sorted((ents.get(r["in"], {}).get("entity_type", "?"),
                             ents.get(r["out"], {}).get("entity_type", "?"))))
            for r in pred).most_common(15),
        "per_source": {
            src: {"predicted": sum(src in r["source_documents"] for r in pred),
                  "llm": sum(src in r["source_documents"] for r in llm)}
            for src in sorted({s for r in rels for s in r["source_documents"]})
        },
        "created_at_range_predicted": [min(r["created_at"] for r in pred),
                                       max(r["created_at"] for r in pred)] if pred else None,
    }

    # Rows vs pairs. Repeated runs re-predict the same pair, and the predictor's
    # kept direction depends on set iteration order while `_upsert_relation`
    # keys on (in, out, type) — so one pair can persist in both directions.
    def pair_stats(rs):
        by_pair = Counter(frozenset((r["in"], r["out"])) for r in rs)
        by_dir = Counter((r["in"], r["out"], r["relation_type"]) for r in rs)
        return {"rows": len(rs), "distinct_pairs": len(by_pair),
                "pairs_seen_twice_or_more": sum(1 for c in by_pair.values() if c > 1),
                "same_direction_duplicates": sum(c - 1 for c in by_dir.values() if c > 1)}
    census["pairs"] = {"predicted": pair_stats(pred), "llm": pair_stats(llm)}

    # "Siblings of one hub", as a number: does the predicted pair already share
    # a neighbour through LLM-extracted edges? If so the edge is a 2-hop shortcut.
    nbrs = defaultdict(set)
    for r in llm:
        nbrs[r["in"]].add(r["out"])
        nbrs[r["out"]].add(r["in"])
    hubs = Counter()
    shares = 0
    for r in pred:
        common = nbrs[r["in"]] & nbrs[r["out"]]
        if common:
            shares += 1
            hubs.update(ents.get(h, {}).get("canonical_name", h) for h in common)
    census["predicted_sharing_an_llm_neighbour"] = shares
    census["top_shared_hubs"] = hubs.most_common(5)

    census["llm_relation_types_near_RELATED"] = {
        t: sum(r["relation_type"] == t for r in llm) for t in ("RELATED", "RELATED_TO", "")}

    # The largest source's predicted subgraph: how close to a clique, over what.
    top_src = max(census["per_source"], key=lambda s: census["per_source"][s]["predicted"])
    sub = [r for r in pred if top_src in r["source_documents"]]
    nodes = {n for r in sub for n in (r["in"], r["out"])}
    sub_pairs = {frozenset((r["in"], r["out"])) for r in sub}
    deg = Counter(n for p in sub_pairs for n in p)
    census["largest_source_subgraph"] = {
        "source": top_src, "rows": len(sub), "distinct_pairs": len(sub_pairs),
        "entities": len(nodes), "possible_pairs": len(nodes) * (len(nodes) - 1) // 2,
        "degree_histogram": dict(Counter(deg.values())),
        "entity_names": sorted(ents.get(n, {}).get("canonical_name", n) for n in nodes),
        "rows_per_minute": dict(Counter(r["created_at"][:16] for r in sub)),
    }

    # Re-score the stored signals under today's config (PC.6).
    current = {"cosine_signal": 0.6, "adamic_adar_signal": 0.3,
               "cooccurrence_signal": 0.1, "jaccard_signal": 0.2}
    def score(props, w):
        return sum(w[k] * (props.get(k) or 0) for k in w)
    census["rescore"] = {
        "stored_confidence_equals_fallback_weights": sum(
            abs(min(score(r["properties"], WEIGHTS), 1.0) - r["confidence"]) < 1e-3 for r in pred),
        "graph_signals_only_max_fallback": 0.2 + 0.1 + 0.2,
        "graph_signals_only_max_current": 0.3 + 0.1 + 0.2,
        "cooccurrence_signal_is_1_on": sum(r["properties"].get("cooccurrence_signal") == 1.0 for r in pred),
        "clear_threshold_without_cosine_fallback": sum(
            score(r["properties"], {**WEIGHTS, "cosine_signal": 0}) >= 0.5 - 1e-9 for r in pred),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "census.json").write_text(json.dumps(census, indent=1, default=str))
    (out / "relations_raw.json").write_text(json.dumps(rels, default=str))
    print(json.dumps(census, indent=1, default=str))


# --------------------------------------------------------------------- sample

def names_of(ent: Dict[str, Any]) -> List[str]:
    names = {ent.get("canonical_name") or "", ent.get("name") or ""}
    names |= set(ent.get("aliases") or [])
    return sorted({n.strip() for n in names if n and len(n.strip()) >= 2}, key=len, reverse=True)


def contains(text_cf: str, names: List[str]) -> bool:
    return any(n.casefold() in text_cf for n in names)


def evidence(rel, ents, chunks_by_source, max_chunks=3, max_chars=1800):
    a, b = ents.get(rel["in"], {}), ents.get(rel["out"], {})
    na, nb = names_of(a), names_of(b)
    shared, only_a, only_b = [], 0, 0
    for src in rel["source_documents"]:
        for ch in chunks_by_source.get(src, []):
            t = ch["text_cf"]
            ha, hb = contains(t, na), contains(t, nb)
            if ha and hb:
                shared.append(ch["text"][:max_chars])
            only_a += ha and not hb
            only_b += hb and not ha
    return {
        "entity_a": a.get("canonical_name"), "type_a": a.get("entity_type"),
        "entity_b": b.get("canonical_name"), "type_b": b.get("entity_type"),
        "shared_chunks_total": len(shared),
        "chunks_a_only": only_a, "chunks_b_only": only_b,
        "passages": shared[:max_chunks],
    }


def cmd_sample(out: Path, seed: int, n_pred: int, n_llm: int) -> None:
    rels = json.loads((out / "relations_raw.json").read_text())
    ents = {e["id"]: e for e in sql(
        "SELECT id, canonical_name, name, aliases, entity_type, status FROM entity;")}
    active = [r for r in rels if r["status"] == "active"]
    pred = [r for r in active if is_predicted(r)]
    llm = [r for r in active if not is_predicted(r)]

    rng = random.Random(seed)
    # Predicted: stratified by signal pattern, proportional with a floor of 5,
    # so a rare pattern is still measured rather than averaged away.
    by_pat = defaultdict(list)
    for r in pred:
        by_pat[signal_pattern(r["properties"])].append(r)
    quota = {p: max(5, round(n_pred * len(v) / len(pred))) for p, v in by_pat.items()}
    drawn_pred = []
    for p, v in sorted(by_pat.items()):
        drawn_pred += rng.sample(v, min(quota[p], len(v)))
    drawn_llm = rng.sample(llm, min(n_llm, len(llm)))

    srcs = sorted({s for r in drawn_pred + drawn_llm for s in r["source_documents"]})
    chunks_by_source = {}
    for s in srcs:
        rows = sql(f"SELECT id, text, order FROM chunk WHERE source = {s} "
                   f"AND chunk_type = 'original' ORDER BY order;")
        chunks_by_source[s] = [{"text": c["text"] or "", "text_cf": (c["text"] or "").casefold()}
                               for c in rows]

    key_source = {r["id"]: r["source_documents"][0] for r in drawn_pred + drawn_llm}
    items = []
    for r in drawn_pred + drawn_llm:
        ev = evidence(r, ents, chunks_by_source)
        items.append({
            "rel_id": r["id"],
            "origin": "predicted" if is_predicted(r) else "llm",
            "relation_type": r["relation_type"],
            "confidence": r["confidence"],
            "signal_pattern": signal_pattern(r["properties"]) if is_predicted(r) else None,
            "dominant_signal": dominant_signal(r["properties"]) if is_predicted(r) else None,
            **ev,
        })
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["item"] = f"E{i:03d}"
    (out / "sample_key.jsonl").write_text("\n".join(json.dumps(i) for i in items))
    # What the judge sees: the pair and the text. No origin, no type, no score.
    blind = [{k: it[k] for k in ("item", "entity_a", "type_a", "entity_b", "type_b",
                                 "shared_chunks_total", "passages")} for it in items]
    (out / "sample_blind.jsonl").write_text("\n".join(json.dumps(b, ensure_ascii=False) for b in blind))

    # The judge reads the WHOLE document, not the shared-chunk passages: extraction
    # runs on packed chunks and text often names "Alkmaar" where the entity is
    # "Gemeente Alkmaar", so substring evidence misses edges of BOTH origins
    # (measured: 22 of 40 LLM edges have no shared chunk). One batch per document,
    # both origins mixed, nothing but the two names and their types.
    batches = out / "judge_batches"
    batches.mkdir(exist_ok=True)
    by_src = defaultdict(list)
    for it in items:
        by_src[key_source[it["rel_id"]]].append(it)
    for src, its in by_src.items():
        stem = src.replace("source:", "")
        text = "\n\n".join(c["text"] for c in chunks_by_source[src])
        (batches / f"{stem}.txt").write_text(text, encoding="utf-8")
        (batches / f"{stem}.pairs.jsonl").write_text("\n".join(json.dumps(
            {k: it[k] for k in ("item", "entity_a", "type_a", "entity_b", "type_b")},
            ensure_ascii=False) for it in its), encoding="utf-8")
    print(f"quota per pattern: {quota}")
    print(f"drawn: {len(drawn_pred)} predicted, {len(drawn_llm)} llm; "
          f"no shared chunk: pred={sum(1 for i in items if i['origin']=='predicted' and not i['shared_chunks_total'])}, "
          f"llm={sum(1 for i in items if i['origin']=='llm' and not i['shared_chunks_total'])}")


# --------------------------------------------------------------------- report

def cmd_report(out: Path) -> None:
    key = {j["item"]: j for j in map(json.loads, (out / "sample_key.jsonl").read_text().splitlines())}
    verdicts = {v["item"]: v for v in map(json.loads, (out / "verdicts.jsonl").read_text().splitlines())}
    missing = sorted(set(key) - set(verdicts))
    if missing:
        raise SystemExit(f"no verdict for {missing}")

    def wilson(k, n, z=1.96):
        if not n:
            return None
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
        return [round(c - h, 3), round(c + h, 3)]

    def tally(items):
        c = Counter(verdicts[i["item"]]["verdict"] for i in items)
        n = len(items)
        return {"n": n, **c,
                "direct_rate": round(c.get("direct", 0) / n, 3) if n else None,
                "direct_ci95": wilson(c.get("direct", 0), n),
                "direct_or_indirect_rate": round(
                    (c.get("direct", 0) + c.get("indirect", 0)) / n, 3) if n else None}

    rels = {r["id"]: r for r in json.loads((out / "relations_raw.json").read_text())}
    census = json.loads((out / "census.json").read_text())
    pop = census["signal_pattern"]
    rows = list(key.values())
    pred_rows = [r for r in rows if r["origin"] == "predicted"]
    by_pat = {p: tally([r for r in pred_rows if r["signal_pattern"] == p]) for p in pop}
    weighted = sum(pop[p] * by_pat[p]["direct_rate"] for p in pop) / sum(pop.values())
    main_pat = max(pop, key=pop.get)
    res = {
        "predicted_direct_rate_population_weighted": round(weighted, 3),
        "predicted_by_document": {
            doc: tally([r for r in pred_rows
                        if rels[r["rel_id"]]["source_documents"][0] == doc])
            for doc in sorted({rels[r["rel_id"]]["source_documents"][0] for r in pred_rows})},
        f"confidence_band_within_{main_pat}": {
            band: tally([r for r in pred_rows if r["signal_pattern"] == main_pat and f(r["confidence"])])
            for band, f in (("<0.6", lambda c: c < 0.6), ("0.6-0.75", lambda c: 0.6 <= c < 0.75),
                            (">=0.75", lambda c: c >= 0.75))},
        "by_origin": {o: tally([r for r in rows if r["origin"] == o]) for o in ("predicted", "llm")},
        "predicted_by_pattern": {p: tally([r for r in rows if r["signal_pattern"] == p])
                                 for p in sorted({r["signal_pattern"] for r in rows if r["signal_pattern"]})},
        "predicted_by_dominant": {d: tally([r for r in rows if r["dominant_signal"] == d])
                                  for d in sorted({r["dominant_signal"] for r in rows if r["dominant_signal"]})},
        "by_origin_with_shared_chunk": {
            o: tally([r for r in rows if r["origin"] == o and r["shared_chunks_total"]])
            for o in ("predicted", "llm")},
    }
    # Weighted rate CI: stratified bootstrap (fixed seed), each stratum resampled
    # from its own verdicts and weighted by its population size.
    rng = random.Random(2026)
    strata = {p: [verdicts[r["item"]]["verdict"] == "direct"
                  for r in pred_rows if r["signal_pattern"] == p] for p in pop}
    boots = sorted(
        sum(pop[p] * sum(rng.choice(v) for _ in v) / len(v) for p, v in strata.items())
        / sum(pop.values()) for _ in range(10000))
    res["predicted_direct_rate_population_weighted_ci95_bootstrap"] = [
        round(boots[249], 3), round(boots[9749], 3)]

    # Fisher exact, two-sided, for the >=0.75 vs <0.6 bands within the main pattern.
    def fisher(a, b, c, d):
        from math import comb
        n1, n2, k = a + b, c + d, a + c
        def p(x):
            return comb(n1, x) * comb(n2, k - x) / comb(n1 + n2, k)
        obs = p(a)
        return round(sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1)
                         if p(x) <= obs * (1 + 1e-9)), 3)
    bands = res[f"confidence_band_within_{main_pat}"]
    hi, lo = bands[">=0.75"], bands["<0.6"]
    res["fisher_p_high_vs_low_band"] = fisher(
        hi.get("direct", 0), hi["n"] - hi.get("direct", 0),
        lo.get("direct", 0), lo["n"] - lo.get("direct", 0))

    # Why the predicted `indirect` verdicts are indirect — a rule over the judge's
    # stated reason, published per item so it can be disputed item by item.
    def category(reason):
        r = reason.lower()
        if "rijk-regio cooperation" in r:
            return "rijk_regio_link"
        if any(s in r for s in ("same 'regio' list", "co-listed", "parties 1-7",
                                "same rijk-regio-overleg list", "separate partijen",
                                "named among the partijen", "listed as signatory",
                                "listed among", "grouped as signatories",
                                "same begripsbepaling list")):
            return "co_listed"
        if "convenant's date" in r or "subtitle" in r:
            return "party_to_deal_vs_deal_attribute"
        return "other"
    cats = {i: category(verdicts[i]["reason"]) for i, k in key.items()
            if k["origin"] == "predicted" and verdicts[i]["verdict"] == "indirect"}
    (out / "indirect_categories.json").write_text(json.dumps(
        [{"item": i, "category": c, "reason": verdicts[i]["reason"]} for i, c in sorted(cats.items())],
        indent=1, ensure_ascii=False))
    res["predicted_indirect_by_category"] = dict(Counter(cats.values()))
    unclassified = sorted(i for i, c in cats.items() if c == "other")
    if unclassified:
        raise SystemExit(f"indirect reasons the category rule does not cover: {unclassified}")

    # Quote fidelity: does each non-empty quote occur in the judged document?
    def norm(s):
        return re.sub(r"\s+", " ", s).strip().casefold()
    docs = {}
    fid = Counter()
    not_found = []
    for i, v in sorted(verdicts.items()):
        q = norm(v.get("quote") or "")
        if not q:
            fid["empty"] += 1
            continue
        src = rels[key[i]["rel_id"]]["source_documents"][0].replace("source:", "")
        if src not in docs:
            docs[src] = norm((out / "judge_batches" / f"{src}.txt").read_text(encoding="utf-8"))
        if q in docs[src]:
            fid["verbatim"] += 1
            continue
        frags = [f.strip(" .…\"'") for f in re.split(r"\.\.\.|…", q)]
        frags = [f for f in frags if len(f) >= 8]
        if frags and all(f in docs[src] for f in frags):
            fid["verbatim_after_ellipsis_split"] += 1
        else:
            fid["not_found"] += 1
            not_found.append({"item": i, "origin": key[i]["origin"], "verdict": v["verdict"]})
    res["quote_fidelity"] = {**fid, "not_found_items": not_found}

    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["census", "sample", "report"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--n-pred", type=int, default=60)
    ap.add_argument("--n-llm", type=int, default=40)
    a = ap.parse_args()
    if a.cmd == "census":
        cmd_census(a.out)
    elif a.cmd == "sample":
        cmd_sample(a.out, a.seed, a.n_pred, a.n_llm)
    else:
        cmd_report(a.out)
