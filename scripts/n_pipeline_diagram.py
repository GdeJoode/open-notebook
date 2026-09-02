"""Generate the extraction-pipeline diagram as an .excalidraw file (Track N review).

One picture of document -> graph: every stage in order, who decides at each
decision point, and where things fall away. Written as a generator rather than
hand-authored JSON so the layout stays consistent and the live-run numbers can be
folded in from ``claudedocs/pipeline-review-run.json``.

    uv run python scripts/n_pipeline_diagram.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Palette: what a colour MEANS, so the picture is readable without a legend hunt.
SYSTEM = "#a5d8ff"      # the system decides, no human in the loop
USER = "#b2f2bb"        # a person decides
DROP = "#ffc9c9"        # entities or relations can be discarded here
WRITE = "#ffec99"       # writes to the graph or to a queue
OFF = "#e9ecef"         # exists but is off under the shipped defaults

_seed = [1000]


def _nid(prefix: str) -> str:
    _seed[0] += 1
    return f"{prefix}{_seed[0]}"


def _base(el_id: str, x: float, y: float, w: float, h: float) -> Dict[str, Any]:
    _seed[0] += 1
    return {
        "id": el_id, "x": x, "y": y, "width": w, "height": h, "angle": 0,
        "strokeColor": "#1e1e1e", "fillStyle": "solid", "strokeWidth": 1,
        "strokeStyle": "solid", "roughness": 1, "opacity": 100, "groupIds": [],
        "frameId": None, "seed": _seed[0], "version": 1, "versionNonce": _seed[0],
        "isDeleted": False, "updated": 1, "link": None, "locked": False,
    }


def box(x: float, y: float, w: float, h: float, label: str, colour: str) -> List[Dict[str, Any]]:
    rid, tid = _nid("r"), _nid("t")
    rect = _base(rid, x, y, w, h)
    rect.update({
        "type": "rectangle", "backgroundColor": colour,
        "roundness": {"type": 3}, "boundElements": [{"type": "text", "id": tid}],
    })
    text = _base(tid, x + 6, y + 6, w - 12, h - 12)
    text.update({
        "type": "text", "backgroundColor": "transparent", "roundness": None,
        "boundElements": None, "fontSize": 14, "fontFamily": 2, "text": label,
        "textAlign": "center", "verticalAlign": "middle", "containerId": rid,
        "originalText": label, "lineHeight": 1.25, "autoResize": False,
    })
    return [rect, text]


def label(x: float, y: float, text: str, size: int = 20, colour: str = "#1e1e1e") -> Dict[str, Any]:
    el = _base(_nid("l"), x, y, 460, size + 8)
    el.update({
        "type": "text", "backgroundColor": "transparent", "roundness": None,
        "boundElements": None, "fontSize": size, "fontFamily": 2, "text": text,
        "textAlign": "left", "verticalAlign": "top", "containerId": None,
        "originalText": text, "lineHeight": 1.25, "strokeColor": colour,
        "autoResize": False,
    })
    return el


def arrow(a: Dict[str, Any], b: Dict[str, Any], dashed: bool = False) -> Dict[str, Any]:
    ax, ay = a["x"] + a["width"], a["y"] + a["height"] / 2
    bx, by = b["x"], b["y"] + b["height"] / 2
    if abs((a["y"] + a["height"] / 2) - by) > 8 and abs(a["x"] - b["x"]) < 40:
        # a vertical hop within one column
        ax, ay = a["x"] + a["width"] / 2, a["y"] + a["height"]
        bx, by = b["x"] + b["width"] / 2, b["y"]
    el = _base(_nid("a"), ax, ay, bx - ax, by - ay)
    el.update({
        "type": "arrow", "backgroundColor": "transparent", "roundness": {"type": 2},
        "boundElements": None, "points": [[0, 0], [bx - ax, by - ay]],
        "lastCommittedPoint": None, "startArrowhead": None, "endArrowhead": "arrow",
        "startBinding": {"elementId": a["id"], "focus": 0, "gap": 4},
        "endBinding": {"elementId": b["id"], "focus": 0, "gap": 4},
        "strokeStyle": "dashed" if dashed else "solid",
        "strokeColor": "#868e96" if dashed else "#1e1e1e",
    })
    return el


def build(counts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    counts = counts or {}
    els: List[Dict[str, Any]] = []
    col_w, box_h, gap = 250, 62, 26

    def n(key: str) -> str:
        v = counts.get(key)
        return f"\n[{v}]" if v is not None else ""

    columns = [
        ("1 — before the LLM", 0, [
            ("PDF / upload", SYSTEM, ""),
            ("parse\ndocling · mineru · markitdown", SYSTEM, ""),
            ("chunks\none per document element", SYSTEM, n("chunks")),
            ("applied schemas\ndetect top_k=3 + notebook base\n+ affinity bundle", SYSTEM, ""),
            ("project accepted edits\nN.4d.3 — deep copies", SYSTEM, ""),
            ("candidate anchors\nN.1 · spaCy · default ON", SYSTEM, ""),
        ]),
        ("2 — the LLM", 1, [
            ("Pass 1\nschema validation", SYSTEM, ""),
            ("pending_extensions\nawaiting a curator", USER, ""),
            ("Pass 2\ntyped extraction per chunk", SYSTEM, n("extracted")),
            ("Hearst is_a\nN.2 · default ON", SYSTEM, ""),
            ("not-a-concept gate\nN.3 · drops page furniture", DROP, n("nac_removed")),
            ("multi-schema merge\nprimary_type + type_tags", SYSTEM, ""),
        ]),
        ("3 — filtering (15 stages)", 2, [
            ("1 noise filter", DROP, n("s1")),
            ("2 normalize · 3 reclassify", SYSTEM, ""),
            ("4 string dedup\n5 fuzzy · 6 embedding · 6b LLM", DROP, n("s4")),
            ("10 KG resolution\nmatch against the existing graph", SYSTEM, ""),
            ("11 ontology filter\n12 centrality", OFF, ""),
            ("15 concept alignment\nRELATED_TO / NOVEL", OFF, n("aligned")),
        ]),
        ("4 — into the graph", 3, [
            ("canonical bridge\nrich label -> entity_type", SYSTEM, ""),
            ("entity upsert", WRITE, n("persisted_entities")),
            ("relation upsert\nmerge on (in, out, type)", WRITE, n("persisted_relations")),
            ("alias registration\nnever automatic", USER, ""),
            ("match_candidates\nprovenance, pending", USER, ""),
        ]),
        ("5 — after the graph", 4, [
            ("ontology_gap\nD-N4-6 · gated on reason code", WRITE, n("gaps")),
            ("schema_proposal\nauto at frequency 5", WRITE, ""),
            ("type placement + judge\nN.4d.1 / N.4d.2", SYSTEM, ""),
            ("curator re-parents\nPOST /schema/reparent", USER, ""),
            ("triage queue\northogonal to all of the above", USER, ""),
        ]),
    ]

    placed: List[List[Dict[str, Any]]] = []
    for title, ci, items in columns:
        x = ci * (col_w + 90)
        els.append(label(x, -70, title, size=18))
        col: List[Dict[str, Any]] = []
        for i, (text, colour, extra) in enumerate(items):
            y = i * (box_h + gap)
            parts = box(x, y, col_w, box_h, text + extra, colour)
            els.extend(parts)
            col.append(parts[0])
        placed.append(col)

    for col in placed:
        for a, b in zip(col, col[1:]):
            els.append(arrow(a, b))
    for left, right in zip(placed, placed[1:]):
        els.append(arrow(left[-1], right[0]))

    # The loop that closes: an accepted re-parent changes the vocabulary the next
    # run projects, which is what makes this a cycle rather than a funnel.
    els.append(arrow(placed[4][3], placed[0][4], dashed=True))

    legend = [
        ("the system decides", SYSTEM), ("a person decides", USER),
        ("things can be dropped here", DROP), ("writes to the graph or a queue", WRITE),
        ("off under shipped defaults", OFF),
    ]
    ly = 6 * (box_h + gap) + 40
    els.append(label(0, ly - 34, "legend", size=16))
    for i, (text, colour) in enumerate(legend):
        els.extend(box(i * 230, ly, 210, 40, text, colour))

    return {
        "type": "excalidraw", "version": 2,
        "source": "scripts/n_pipeline_diagram.py",
        "elements": els,
        "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
        "files": {},
    }


def main() -> None:
    """Fold the live-run numbers in, aggregated across every run file present.

    The corpus was measured in three passes (the six-document run, Bennett, and
    the Achterhoek re-run into the same notebook), so the picture reports the sum
    of what was extracted and the FINAL graph size — which is the pair that makes
    the linear growth visible.
    """
    counts: Dict[str, Any] = {}
    runs = [
        Path("claudedocs/pipeline-review-corpus.json"),
        Path("claudedocs/pipeline-review-bennett.json"),
        Path("claudedocs/pipeline-review-achterhoek.json"),
    ]
    chunks = extracted = 0
    graph: Dict[str, int] = {}
    for run in runs:
        if not run.exists():
            continue
        data = json.loads(run.read_text(encoding="utf-8"))
        for doc in data.get("documents", []):
            chunks += doc.get("chunks", 0)
            extracted += doc.get("result", {}).get("entity_count", 0)
        graph = data.get("graph", graph) or graph

    if chunks:
        counts["chunks"] = chunks
        counts["extracted"] = extracted
        counts["s1"] = extracted
        counts["s4"] = extracted
        counts["persisted_entities"] = graph.get("entity")
        counts["persisted_relations"] = graph.get("relation")
        counts["gaps"] = graph.get("ontology_gap")
    counts = {k: v for k, v in counts.items() if v is not None}

    diagram = build(counts)
    out = Path("claudedocs/extraction-pipeline.excalidraw")
    out.write_text(json.dumps(diagram, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(diagram['elements'])} elements) counts={counts}")


if __name__ == "__main__":
    main()
