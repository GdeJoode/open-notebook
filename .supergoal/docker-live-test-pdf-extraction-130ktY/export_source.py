"""Export one source's markdown + entities + relations + meta to app-output/.

Usage: python export_source.py <slug> <source_id> <notebook_id>
Pulls live data from the running Open Notebook API at localhost:5055.
"""
import json
import sys
import urllib.request

BASE = "http://localhost:5055"
OUT = r"E:\repos\private\open-notebook\.supergoal\docker-live-test-pdf-extraction-130ktY\app-output"


def get(path):
    req = urllib.request.Request(BASE + path)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main():
    slug, source_id, notebook_id = sys.argv[1], sys.argv[2], sys.argv[3]

    src = get(f"/api/sources/{source_id}")
    full_text = src.get("full_text") or ""

    chunks_resp = get(f"/api/sources/{source_id}/chunks")
    chunks = chunks_resp.get("chunks", chunks_resp) if isinstance(chunks_resp, dict) else chunks_resp
    chunk_count = len(chunks) if isinstance(chunks, list) else None

    ext = get(f"/api/sources/{source_id}/extraction-result")
    entities = ext.get("entities") or []
    relations = ext.get("relations") or ext.get("relationships") or []
    entity_count = ext.get("entity_count")
    relation_count = ext.get("relation_count")
    if entity_count is None:
        entity_count = len(entities)
    if relation_count is None:
        relation_count = len(relations)

    # markdown
    with open(f"{OUT}/{slug}.md", "w", encoding="utf-8") as f:
        f.write(full_text)

    # entities
    with open(f"{OUT}/{slug}.entities.json", "w", encoding="utf-8") as f:
        json.dump(entities, f, indent=2, ensure_ascii=False)

    # relations
    with open(f"{OUT}/{slug}.relations.json", "w", encoding="utf-8") as f:
        json.dump(relations, f, indent=2, ensure_ascii=False)

    # meta
    meta = {
        "slug": slug,
        "source_id": source_id,
        "notebook_id": notebook_id,
        "parse_status": src.get("status"),
        "full_text_chars": len(full_text),
        "chunk_count": chunk_count,
        "entity_count": entity_count,
        "relation_count": relation_count,
        "extraction_metadata": ext.get("metadata"),
        "endpoints": {
            "markdown": f"/api/sources/{source_id} -> full_text",
            "chunks": f"/api/sources/{source_id}/chunks",
            "entities": f"/api/sources/{source_id}/extraction-result -> entities",
            "relations": f"/api/sources/{source_id}/extraction-result -> relations",
        },
    }
    with open(f"{OUT}/{slug}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"EXPORTED {slug}: md_chars={len(full_text)} chunks={chunk_count} "
          f"entities={entity_count} relations={relation_count}")


if __name__ == "__main__":
    main()
