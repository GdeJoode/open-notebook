"""D2 experiment — can GLinker resolve this corpus's ministries against TOOI?

Question: for the active `government_organization` rows PC.3 measured, which
mechanism resolves each surface form to the right TOOI organisation?

  (a) the K.4 TOOI matcher as shipped: exact, case-sensitive equality on the
      canonical name or an alias (`TOOIProvider.lookup` over
      `reference_entity.lookup_by_name/alias`), reproduced in memory;
  (b) GLinker L2 candidate retrieval (exact + fuzzy over label and aliases);
  (c) GLinker L3 disambiguation (gliner-linker, a SCORE) over (b)'s candidates.

(a) and (b)-exact answer a checkable question; (b)-fuzzy and (c) produce scores,
which by `component-decisions.md` may propose but not write.

Steps (each writes under claudedocs/d2-glinker/):

    fetch    -> tooi_records.json   TOOI via the repo's own TOOIBulkFetcher
                mentions.json       active government_organization rows + one
                                    context passage each (SELECT only)
    baseline -> baseline.json       (a), run with the project .venv
    glinker  -> glinker.json        (b) and (c), run with a separate venv that
                                    has `glinker` installed
    score    -> results.json        each mechanism against gold.json

gold.json is written by hand BEFORE any mechanism is scored.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

OUT = Path("claudedocs/d2-glinker")


# ------------------------------------------------------------------ fetch

def _sql(query: str) -> List[Any]:
    if not re.match(r"^\s*SELECT\b", query, re.I):
        raise ValueError("read-only: SELECT only")
    req = urllib.request.Request(
        os.environ.get("SURREAL_HTTP", "http://localhost:8000") + "/sql",
        data=query.encode(), method="POST")
    tok = base64.b64encode(
        f"{os.environ['SURREAL_USER']}:{os.environ['SURREAL_PASSWORD']}".encode()).decode()
    for k, v in {"Authorization": f"Basic {tok}", "surreal-ns": "open_notebook",
                 "surreal-db": "staging", "Accept": "application/json"}.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)
    if out[0]["status"] != "OK":
        raise RuntimeError(out[0])
    return out[0]["result"]


def cmd_fetch() -> None:
    import asyncio
    sys.path.insert(0, "packages/shared/src")
    from shared.vocabulary.tooi_bulk import TOOIBulkFetcher

    async def tooi():
        f = TOOIBulkFetcher()
        try:
            return await f.fetch_records()
        finally:
            await f.aclose()

    recs = asyncio.run(tooi())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tooi_records.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")

    rows = _sql("SELECT id, canonical_name, name, aliases, source_documents, created FROM entity "
                "WHERE entity_type = 'government_organization' AND status = 'active' "
                "ORDER BY created;")
    mentions = []
    for r in rows:
        name = r["canonical_name"]
        ctx, start = None, None
        for src in r["source_documents"]:
            chunks = _sql(f"SELECT text, order FROM chunk WHERE source = {src} "
                          f"AND chunk_type = 'original' ORDER BY order;")
            # Context = the first chunk that contains the surface form, or,
            # failing that, its core (text before " (") — an extracted name is
            # often a normalised form of what the text says.
            core = name.split(" (")[0]
            for probe in (name, core):
                for c in chunks:
                    i = (c["text"] or "").find(probe)
                    if i >= 0:
                        ctx, start, found = c["text"], i, probe
                        break
                if ctx:
                    break
            if ctx:
                break
        mentions.append({"entity_id": r["id"], "name": name, "aliases": r.get("aliases") or [],
                         "source": r["source_documents"][0] if r["source_documents"] else None,
                         "context": ctx, "span": [start, start + len(found)] if ctx else None,
                         "span_text": found if ctx else None})
    (OUT / "mentions.json").write_text(json.dumps(mentions, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"TOOI records: {len(recs)}; mentions: {len(mentions)}, "
          f"with context: {sum(1 for m in mentions if m['context'])}")


# --------------------------------------------------------------- baseline

def cmd_baseline() -> None:
    """(a) The K.4 matcher, with an in-memory stand-in for `reference_entity`
    that implements its two queries exactly: `canonical_name = $name` and
    `$alias IN aliases` — both exact, case-sensitive equality."""
    import asyncio
    sys.path.insert(0, "packages/shared/src")
    from shared.vocabulary.tooi_provider import TOOIProvider

    records = json.loads((OUT / "tooi_records.json").read_text(encoding="utf-8"))

    class MemRepo:
        rows: List[Dict[str, Any]] = []

        async def bulk_load(self, rows):
            self.rows = list(rows)
            return len(self.rows)

        async def lookup_by_name(self, name, source=None):
            return [r for r in self.rows if r.get("canonical_name") == name]

        async def lookup_by_alias(self, alias, source=None):
            return [r for r in self.rows if alias in (r.get("aliases") or [])]

    repo = MemRepo()
    prov = TOOIProvider(repository=repo)
    repo.rows = [prov._to_reference_row(r) for r in records]
    mentions = json.loads((OUT / "mentions.json").read_text(encoding="utf-8"))

    async def run():
        out = {}
        for m in mentions:
            matches = await prov.lookup(m["name"], "government_organization")
            out[m["name"]] = [{"uri": x.external_uri, "label": x.canonical_name,
                               "confidence": x.confidence} for x in matches]
        return out

    res = asyncio.run(run())
    (OUT / "baseline.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- glinker

def _kb_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """TOOI -> GLinker DatabaseRecord fields. The URI is built the way
    `tooi_provider._record_uri` builds it, so all mechanisms answer in one id space."""
    base = "https://identifier.overheid.nl/tooi/id"
    out = []
    for r in records:
        aliases = []
        for a in [r.get("naam_incl_soort"), r.get("afkorting"), r.get("naam_excl_soort"),
                  *(r.get("aliases") or [])]:
            if a and a not in aliases:
                aliases.append(a)
        out.append({
            "entity_id": f"{base}/{r.get('soort', 'ministerie')}/{r['organisatiecode']}",
            "label": r.get("naam_excl_soort") or r.get("afkorting") or "",
            "aliases": aliases,
            "description": r.get("naam_incl_soort") or "",
            "entity_type": r.get("soort", ""),
        })
    return out


def cmd_glinker(models: List[str], max_candidates: int) -> None:
    """(b) L2 and (c) L3, called as components so each step is measured apart.
    Needs the venv with `glinker`; imports nothing from the repo."""
    # GLinker's DictLayer prints one WARN and returns [] for every fuzzy search
    # when rapidfuzz is absent — and `pip install glinker` does not pull it in.
    # The first run of this experiment measured nothing for that reason.
    import rapidfuzz  # noqa: F401
    from glinker.l2.component import DictLayer
    from glinker.l2.models import DatabaseRecord, FuzzyConfig, LayerConfig
    from glinker.l3.component import L3Component
    from glinker.l3.models import L3Config

    kb = _kb_records(json.loads((OUT / "tooi_records.json").read_text(encoding="utf-8")))
    layer = DictLayer(LayerConfig(type="dict", priority=0, search_mode=["exact", "fuzzy"],
                                  fuzzy=FuzzyConfig(min_similarity=0.6, prefix_length=1)))
    layer.load_bulk([DatabaseRecord(**r) for r in kb])
    ministries = [r for r in kb if r["entity_type"] == "ministerie"]
    template = "{label}: {description}"          # GLinker's default schema template
    mentions = json.loads((OUT / "mentions.json").read_text(encoding="utf-8"))

    def l2(q: str) -> Dict[str, List[str]]:
        exact = [r.entity_id for r in layer.search(q)]
        fuzzy = [r.entity_id for r in layer.search_fuzzy(q) if r.entity_id not in exact]
        return {"exact": exact, "fuzzy": fuzzy[:max_candidates]}

    res: Dict[str, Any] = {"config": {"fuzzy_min_similarity": 0.6, "prefix_length": 1,
                                      "max_candidates": max_candidates, "template": template,
                                      "kb_size": len(kb), "ministries": len(ministries)},
                           "mentions": {}}
    for m in mentions:
        res["mentions"][m["name"]] = {
            "l2_stored_name": l2(m["name"]),
            "l2_text_surface": l2(m["span_text"]) if m["span_text"] else None,
            "context_from": "chunk" if m["context"] else "name_only",
        }

    by_id = {r["entity_id"]: r for r in kb}
    for model in models:
        comp = L3Component(L3Config(model_name=model, threshold=0.0, flat_ner=True,
                                    max_length=512))
        for m in mentions:
            text = m["context"] or m["name"]
            span = m["span"] or [0, len(m["name"])]
            row = res["mentions"][m["name"]]
            l2c = row["l2_stored_name"]
            for variant, cand_ids in (("l3_over_l2", (l2c["exact"] + l2c["fuzzy"])[:max_candidates]),
                                      ("l3_over_all_ministries", [r["entity_id"] for r in ministries])):
                labels, lab2id = [], {}
                for cid in cand_ids:
                    lab = template.format(**by_id[cid])
                    if lab.lower() not in {x.lower() for x in labels}:
                        labels.append(lab)
                        lab2id[lab] = cid
                if not labels:
                    row[f"{variant}@{model}"] = {"top": None, "score": None, "n_candidates": 0}
                    continue
                ents = comp.predict_entities(text, labels,
                                             input_spans=[[{"start": span[0], "end": span[1]}]])
                best = max(ents, key=lambda e: e.score) if ents else None
                row[f"{variant}@{model}"] = {
                    "top": lab2id.get(best.label) if best else None,
                    "score": round(best.score, 4) if best else None,
                    "n_candidates": len(labels),
                    "probs": {lab2id[k]: round(v, 4) for k, v in (best.class_probs or {}).items()
                              if k in lab2id} if best else {},
                }
        # Written per model: a failed download of the next one must not lose this.
        (OUT / "glinker.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "glinker.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote glinker.json")


# ------------------------------------------------------------------ score

def cmd_score() -> None:
    gold = json.loads((OUT / "gold.json").read_text(encoding="utf-8"))
    base = json.loads((OUT / "baseline.json").read_text(encoding="utf-8"))
    gl = json.loads((OUT / "glinker.json").read_text(encoding="utf-8"))

    def verdict(name: str, answer: List[str]) -> str:
        """correct / wrong / refused (no answer) / ambiguous (>1 answer).
        For a gold with no link, any answer is wrong and refusing is correct."""
        g = set(gold[name]["gold"])
        if not answer:
            return "correct" if not g else "refused"
        if len(answer) > 1:
            # gold.json's rule for a double gold (EZ): returning exactly the
            # accepted records is correct, not merely ambiguous.
            if len(g) > 1 and set(answer) <= g:
                return "correct"
            return "ambiguous_contains_gold" if g & set(answer) else "ambiguous_wrong"
        return "correct" if (answer[0] in g) else "wrong"

    # D1 SIMULATION, not shipped code: strip the decorations PC.3 measured
    # ("X (Ministerie)", "ABBR (gloss)", "Ministerie van X", "ministeries van het
    # Rijk: X") and ask the exact, case-insensitive lookup about each piece. One id
    # agreed by every piece that hits = an answer; pieces that disagree = refuse.
    kb = _kb_records(json.loads((OUT / "tooi_records.json").read_text(encoding="utf-8")))
    index: Dict[str, set] = {}
    for r in kb:
        for s in [r["label"], *r["aliases"]]:
            index.setdefault(s.casefold(), set()).add(r["entity_id"])

    def d1_pieces(name: str) -> List[str]:
        s = re.sub(r"^ministeries van het rijk:\s*", "", name, flags=re.I)
        s = re.sub(r"^(het\s+)?ministerie van\s+", "", s, flags=re.I)
        m = re.match(r"^(.*?)\s*\((.*)\)\s*$", s)
        pieces = [m.group(1), m.group(2)] if m else [s]
        return [p.strip() for p in pieces if p.strip() and p.strip().lower() != "ministerie"]

    def d1_answer(name: str) -> List[str]:
        hits = [index.get(p.casefold(), set()) for p in d1_pieces(name)]
        hits = [h for h in hits if h]
        if not hits:
            return []
        agreed = set.intersection(*hits)
        return sorted(agreed)  # pieces that disagree -> empty -> refused

    per: Dict[str, Dict[str, str]] = {}
    for name in gold:
        row = gl["mentions"][name]
        p = {"a_k4_exact": verdict(name, [x["uri"] for x in base[name]]),
             "b_l2_exact_stored": verdict(name, row["l2_stored_name"]["exact"]),
             "b_l2_exact_text": verdict(name, (row["l2_text_surface"] or {}).get("exact", [])),
             "b_l2_fuzzy_top1_stored": verdict(name, row["l2_stored_name"]["fuzzy"][:1]),
             # L2 as GLinker runs it: an exact hit first, else the fuzzy top-1.
             "b_l2_pipeline_stored": verdict(name, row["l2_stored_name"]["exact"]
                                             or row["l2_stored_name"]["fuzzy"][:1]),
             "d1_sim_then_exact": verdict(name, d1_answer(name))}
        for k, v in row.items():
            if k.startswith("l3_"):
                ans = [v["top"]] if v["top"] and v["score"] is not None else []
                p[f"c_{k}"] = verdict(name, ans)
                p[f"c_{k}@0.5"] = verdict(name, ans if (v["score"] or 0) >= 0.5 else [])
        per[name] = p

    from collections import Counter
    mechanisms = sorted({k for p in per.values() for k in p})
    summary = {mech: dict(Counter(per[n][mech] for n in per)) for mech in mechanisms}
    (OUT / "results.json").write_text(json.dumps({"summary": summary, "per_mention": per},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    for mech in mechanisms:
        print(f"{mech:60s} {summary[mech]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "baseline", "glinker", "score"])
    ap.add_argument("--models", nargs="+", default=["knowledgator/gliner-linker-base-v1.0",
                                                    "knowledgator/gliner-linker-large-v1.0"])
    ap.add_argument("--max-candidates", type=int, default=5)
    a = ap.parse_args()
    if a.cmd == "glinker":
        cmd_glinker(a.models, a.max_candidates)
    else:
        {"fetch": cmd_fetch, "baseline": cmd_baseline, "score": cmd_score}[a.cmd]()
