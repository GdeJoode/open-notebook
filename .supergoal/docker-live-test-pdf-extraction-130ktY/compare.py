#!/usr/bin/env python3
"""Phase 4 — compare app extraction vs my independent ground-truth.

Recall of MY ground-truth's salient items (the user's bar: found info ≥80%).
- entity recall: fraction of ground-truth entities fuzzy-matched in app entities.
- relation recall: fraction of ground-truth relations whose subject+object both
  fuzzy-match an app relation's endpoints (predicate wording is looser, so it is
  reported but not required for a match).
- markdown coverage: fraction of ground-truth section headings present in app md.

Defensive about schema: app JSON entities may be [{"name"|"label"|"title"}],
relations may be [{"subject","predicate","object"}] or [{"source","target","type"}]
or nested under {"entities":[...]}/{"relations":[...]}.

Run via the WSL venv:
  wsl bash -lc 'cd /mnt/e/repos/private/open-notebook && \
    .venv/bin/python .supergoal/docker-live-test-pdf-extraction-130ktY/compare.py'
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GT = ROOT / "ground-truth"
APP = ROOT / "app-output"
OUT = ROOT / "comparison"
OUT.mkdir(exist_ok=True)

SLUGS = ["economics", "jcms-cohesion", "centrifugal-state"]
THRESH = 0.80


def norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def toks(s: str) -> set[str]:
    return {t for t in norm(s).split() if len(t) > 2}


def fuzzy(a: str, b: str) -> bool:
    """Match if one normalized string contains the other, or token Jaccard>=0.5."""
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True
    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    return inter / min(len(ta), len(tb)) >= 0.6


def load_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"_error": str(e)}


def entity_names(data) -> list[str]:
    if data is None:
        return []
    if isinstance(data, dict):
        for k in ("entities", "items", "results", "data"):
            if k in data and isinstance(data[k], list):
                data = data[k]
                break
        else:
            data = list(data.values()) if not data.get("_error") else []
    out = []
    for e in data or []:
        if isinstance(e, str):
            out.append(e)
        elif isinstance(e, dict):
            for k in ("name", "label", "title", "text", "value", "canonical_name"):
                if e.get(k):
                    out.append(str(e[k]))
                    break
    return out


def entity_aliases(data) -> list[list[str]]:
    """Return [name]+aliases per ground-truth entity for recall matching."""
    out = []
    for e in data or []:
        if isinstance(e, dict):
            names = [e.get("name") or e.get("label") or ""]
            names += e.get("aliases") or []
            out.append([n for n in names if n])
    return out


def relations(data) -> list[tuple[str, str, str]]:
    if data is None:
        return []
    if isinstance(data, dict):
        for k in ("relations", "edges", "items", "results", "data"):
            if k in data and isinstance(data[k], list):
                data = data[k]
                break
        else:
            data = []
    out = []
    for r in data or []:
        if not isinstance(r, dict):
            continue
        s = r.get("subject") or r.get("source") or r.get("from") or r.get("head")
        o = r.get("object") or r.get("target") or r.get("to") or r.get("tail")
        p = r.get("predicate") or r.get("type") or r.get("relation") or r.get("label") or ""
        if s and o:
            out.append((str(s), str(p), str(o)))
    return out


def headings(md: str) -> list[str]:
    return [re.sub(r"^#+\s*", "", ln).strip()
            for ln in (md or "").splitlines() if ln.lstrip().startswith("#")]


summary = []
for slug in SLUGS:
    gt_ent = load_json(GT / f"{slug}.entities.json") or []
    gt_rel_raw = load_json(GT / f"{slug}.relations.json") or []
    app_ent_raw = load_json(APP / f"{slug}.entities.json")
    app_rel_raw = load_json(APP / f"{slug}.relations.json")
    gt_md = (GT / f"{slug}.md").read_text(encoding="utf-8") if (GT / f"{slug}.md").exists() else ""
    app_md = (APP / f"{slug}.md").read_text(encoding="utf-8") if (APP / f"{slug}.md").exists() else ""

    app_ent = entity_names(app_ent_raw)
    gt_alias = entity_aliases(gt_ent)
    gt_rel = relations(gt_rel_raw)
    app_rel = relations(app_rel_raw)

    # entity recall
    ent_matched, ent_missed = [], []
    for names in gt_alias:
        hit = any(fuzzy(n, a) for n in names for a in app_ent)
        (ent_matched if hit else ent_missed).append(names[0])
    ent_recall = len(ent_matched) / len(gt_alias) if gt_alias else 0.0

    # relation recall (subject+object both fuzzy-match some app relation's endpoints)
    rel_matched, rel_missed = [], []
    for (s, p, o) in gt_rel:
        hit = any((fuzzy(s, S) and fuzzy(o, O)) or (fuzzy(s, O) and fuzzy(o, S))
                  for (S, P, O) in app_rel)
        (rel_matched if hit else rel_missed).append(f"{s} —{p}→ {o}")
    rel_recall = len(rel_matched) / len(gt_rel) if gt_rel else 0.0

    # markdown heading coverage
    gt_h = headings(gt_md)
    md_matched = [h for h in gt_h if any(fuzzy(h, ah) for ah in headings(app_md))
                  or norm(h) in norm(app_md)]
    md_cov = len(md_matched) / len(gt_h) if gt_h else 0.0

    rep = [f"# Comparison — {slug}\n",
           f"App entities: {len(app_ent)} · ground-truth entities: {len(gt_alias)}",
           f"App relations: {len(app_rel)} · ground-truth relations: {len(gt_rel)}",
           f"App markdown: {len(app_md)} chars · ground-truth headings: {len(gt_h)}\n",
           f"## Scores (recall of ground-truth)",
           f"- Entity recall:   {ent_recall:.0%}  ({len(ent_matched)}/{len(gt_alias)})  {'PASS' if ent_recall>=THRESH else 'BELOW 80%'}",
           f"- Relation recall: {rel_recall:.0%}  ({len(rel_matched)}/{len(gt_rel)})  {'PASS' if rel_recall>=THRESH else 'BELOW 80%'}",
           f"- Markdown cover:  {md_cov:.0%}  ({len(md_matched)}/{len(gt_h)})  {'PASS' if md_cov>=THRESH else 'BELOW 80%'}\n",
           f"## Missed entities ({len(ent_missed)})", *[f"- {m}" for m in ent_missed], "",
           f"## Missed relations ({len(rel_missed)})", *[f"- {m}" for m in rel_missed], ""]
    (OUT / f"{slug}.md").write_text("\n".join(rep), encoding="utf-8")
    summary.append((slug, ent_recall, rel_recall, md_cov,
                    len(app_ent), len(app_rel), len(app_md)))
    print(f"{slug}: ent={ent_recall:.0%} rel={rel_recall:.0%} md={md_cov:.0%} "
          f"(app: {len(app_ent)} ent, {len(app_rel)} rel, {len(app_md)} md-chars)")

print("\n=== SUMMARY (≥80% target) ===")
for slug, e, r, m, ae, ar, am in summary:
    ok = "PASS" if (e >= THRESH and r >= THRESH and m >= THRESH) else "REVIEW"
    print(f"  {slug:18} ent {e:.0%} | rel {r:.0%} | md {m:.0%}  -> {ok}")
