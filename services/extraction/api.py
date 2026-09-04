"""
Entity extraction service.

Standalone HTTP API for ontology-guided entity and relation extraction.
Accepts a file path (docling JSON, markdown, or plain text) and extracts
entities and relations using the regiodeal/policy/general ontology as guide.

Results are written to SurrealDB entity/relation tables and returned as JSON.
All output files are written to the same directory as the input file.

Configuration via environment variables for LLM and SurrealDB endpoints.
"""

import asyncio
import json
import math
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import yaml
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Validate this service's OWN copy of the routing config (PC.6).

    app-main runs the same check, and that is not enough: this service is where
    `model_routing.yaml` is consumed, it ships its own build-time COPY of both the
    module and the YAML (see the Dockerfile), and `/routing` below is the surface
    that advertises those routes to a human. Nothing enforces that the copy stays
    in step with the repository, so a stale image would pass every check run
    elsewhere.

    Only the routing half is checked here. The feature-dependency half reads
    app-layer state this service does not have, and inventing an answer for it
    would produce findings nobody can act on.
    """
    try:
        from config_coherence import BLOCK, check_routing, raise_if_blocking
        from model_routing import _load_config

        findings = check_routing(_load_config())
        for finding in findings:
            if finding.severity != BLOCK:
                logger.warning(f"config coherence: {finding}")
        raise_if_blocking(findings)
    except ImportError:
        # The shared checker is not part of this image's copy set on older
        # builds; say so rather than skipping silently.
        logger.warning(
            "config coherence: shared.config_coherence not available in this "
            "image — routing was NOT validated at startup"
        )
    yield


app = FastAPI(
    title="Entity Extraction Service",
    description="Ontology-guided entity and relation extraction via LLM",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")

# `EXTRACTION_MODEL` and `EXTRACTION_NUM_CTX` were read into constants here and
# REMOVED (PC.6). Each was assigned once and used nowhere: `_extract_from_chunk`
# resolves the model through `model_routing.call_llm(step="extraction")`, which
# reads `packages/shared/src/shared/model_routing.yaml`. Setting them in
# docker-compose looked like configuring extraction and configured nothing —
# verified live: with `EXTRACTION_MODEL=qwen2.5:14b-…` set, the resolver returned
# `ollama / llama3.1:8b-instruct-q4_0 / num_ctx 8192`. That cost a fully reverted
# branch. The lever is model_routing.yaml.

SURREALDB_URL = os.getenv("SURREALDB_URL", "ws://host.docker.internal:8000/rpc")
SURREALDB_NS = os.getenv("SURREALDB_NS", "open_notebook")
SURREALDB_DB = os.getenv("SURREALDB_DB", "open_notebook")
SURREALDB_USER = os.getenv("SURREALDB_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_PASS", "root")

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")

ONTOLOGY_DIR = Path(os.getenv("ONTOLOGY_DIR", "/app/ontologies"))
DEFAULT_ONTOLOGY = os.getenv("DEFAULT_ONTOLOGY", "regiodeal")
CHUNK_SIZE = int(os.getenv("EXTRACTION_CHUNK_SIZE", "4000"))

# ---------------------------------------------------------------------------
# Ontology loading
# ---------------------------------------------------------------------------

_ontology_cache: Dict[str, str] = {}
_ontology_raw_cache: Dict[str, Dict] = {}


def _load_ontology_yaml(name: str, _visited: Optional[set] = None) -> Dict:
    """Load an ontology YAML and resolve includes/extends recursively.

    Merges entity_types, relationship_types, extraction_rules, and
    extraction_patterns from all included ontologies. Prevents circular
    includes.
    """
    if name in _ontology_raw_cache:
        return _ontology_raw_cache[name]

    if _visited is None:
        _visited = set()
    if name in _visited:
        return {}
    _visited.add(name)

    path = ONTOLOGY_DIR / f"{name}.yaml"
    if not path.exists():
        logger.warning(f"Ontology '{name}' not found at {path}")
        return {}

    with open(path, encoding="utf-8") as f:
        ont = yaml.safe_load(f) or {}

    # Collect all entity types, relations, rules from this file
    merged_entity_types = list(ont.get("entity_types", []))
    merged_relation_types = list(ont.get("relationship_types", []))
    merged_extraction_rules = list(ont.get("extraction_rules", []))

    # Resolve 'extends' (single parent)
    extends = ont.get("metadata", {}).get("extends")
    if extends and isinstance(extends, str):
        parent = _load_ontology_yaml(extends, _visited)
        merged_entity_types = list(parent.get("entity_types", [])) + merged_entity_types
        merged_relation_types = list(parent.get("relationship_types", [])) + merged_relation_types
        merged_extraction_rules = list(parent.get("extraction_rules", [])) + merged_extraction_rules

    # Resolve 'includes' (multiple siblings)
    includes = ont.get("metadata", {}).get("includes", [])
    if isinstance(includes, list):
        for inc_name in includes:
            inc = _load_ontology_yaml(inc_name, _visited)
            merged_entity_types.extend(inc.get("entity_types", []))
            merged_relation_types.extend(inc.get("relationship_types", []))
            merged_extraction_rules.extend(inc.get("extraction_rules", []))

    # Deduplicate by name (last wins)
    seen_entities = {}
    for et in merged_entity_types:
        seen_entities[et.get("name", "")] = et
    seen_relations = {}
    for rt in merged_relation_types:
        seen_relations[rt.get("name", "")] = rt

    result = {
        "metadata": ont.get("metadata", {}),
        "entity_types": list(seen_entities.values()),
        "relationship_types": list(seen_relations.values()),
        "extraction_rules": list(dict.fromkeys(merged_extraction_rules)),
        "concepts": ont.get("concepts", []),
        "claim_types": ont.get("claim_types", []),
    }

    _ontology_raw_cache[name] = result
    return result


def _load_ontology_prompt(name: str) -> str:
    """Load an ontology and format as extraction prompt context.

    Resolves includes/extends to produce a complete prompt with
    all entity types, relation types, and extraction rules.
    """
    if name in _ontology_cache:
        return _ontology_cache[name]

    ont = _load_ontology_yaml(name)
    if not ont:
        return ""

    parts = ["ENTITY TYPES:"]
    for et in ont.get("entity_types", []):
        n = et.get("name", "")
        d = et.get("description", "")
        gr = et.get("graph_relevance", "active")
        hints = et.get("extraction_hints", [])
        if et.get("abstract"):
            continue  # Skip abstract base classes
        line = f"- {n}: {d}"
        if gr == "contextual":
            line += " [contextual — extract but not for knowledge graph]"
        if hints:
            line += f" (look for: {'; '.join(hints[:2])})"
        parts.append(line)

    parts.append("\nRELATION TYPES:")
    for rt in ont.get("relationship_types", []):
        n = rt.get("name", "")
        d = rt.get("description", "")
        parts.append(f"- {n}: {d}")

    rules = ont.get("extraction_rules", [])
    if rules:
        parts.append("\nEXTRACTION RULES:")
        for rule in rules:
            parts.append(f"- {rule}")

    prompt = "\n".join(parts)

    logger.info(
        f"Ontology '{name}' loaded: "
        f"{len(ont.get('entity_types', []))} entity types, "
        f"{len(ont.get('relationship_types', []))} relation types, "
        f"{len(rules)} rules"
    )

    _ontology_cache[name] = prompt
    return prompt


def list_ontologies() -> List[str]:
    """List available ontology names."""
    return sorted(p.stem for p in ONTOLOGY_DIR.glob("*.yaml"))


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------


def _load_and_chunk(file_path: Path) -> Tuple[List[Dict[str, Any]], str]:
    """Load a file and split into chunks for extraction.

    Returns (chunks, doc_name).
    Each chunk: {"text": str, "id": str, "section_path": list, "page": int}
    """
    suffix = file_path.suffix.lower()
    doc_name = file_path.stem

    if suffix == ".json":
        return _chunks_from_docling_json(file_path, doc_name), doc_name
    else:
        raw = file_path.read_text(encoding="utf-8")
        return _chunks_from_text(raw, doc_name), doc_name


def _chunks_from_docling_json(path: Path, doc_name: str) -> List[Dict[str, Any]]:
    """Convert docling document.json to chunks preserving structure."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)

    chunks = []
    current_texts: List[str] = []
    current_len = 0
    current_section: List[str] = []
    current_page = 1
    chunk_idx = 0

    def flush():
        nonlocal chunk_idx, current_texts, current_len
        if not current_texts:
            return
        text = "\n".join(current_texts).strip()
        if len(text) >= 20:
            chunks.append({
                "text": text,
                "id": f"{doc_name}_c{chunk_idx}",
                "section_path": list(current_section),
                "page": current_page,
            })
            chunk_idx += 1
        current_texts = []
        current_len = 0

    for page in doc.get("pages", []):
        for elem in page.get("elements", []):
            content = (elem.get("content") or "").strip()
            if not content or len(content) < 10:
                continue

            sp = elem.get("section_path", [])
            if sp != current_section:
                flush()
                current_section = sp
            current_page = page.get("page_number", 1)

            if current_len + len(content) > CHUNK_SIZE and current_texts:
                flush()
            current_texts.append(content)
            current_len += len(content)

    flush()
    return chunks


def _chunks_from_text(text: str, doc_name: str) -> List[Dict[str, Any]]:
    """Split plain text/markdown into chunks."""
    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks = []
    current_texts: List[str] = []
    current_len = 0
    idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current_len + len(para) > CHUNK_SIZE and current_texts:
            chunks.append({
                "text": "\n".join(current_texts),
                "id": f"{doc_name}_c{idx}",
                "section_path": [],
                "page": 0,
            })
            idx += 1
            current_texts = []
            current_len = 0
        current_texts.append(para)
        current_len += len(para)

    if current_texts:
        chunks.append({
            "text": "\n".join(current_texts),
            "id": f"{doc_name}_c{idx}",
            "section_path": [],
            "page": 0,
        })

    return chunks


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------


async def _extract_from_chunk(
    chunk_text: str,
    ontology_prompt: str,
    privacy: str = "internal",
    model_override: Optional[Dict] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Extract entities and relations from a single chunk via LLM.

    Uses model_routing to select provider based on privacy level.
    """
    from model_routing import call_llm

    system_msg = (
        "You are a Dutch government document entity extraction system. "
        "The input text is in Dutch. Entity names must be extracted exactly as written in the text. "
        "Return ONLY valid JSON, no other text, no markdown."
    )

    user_msg = f"""Extract named entities and relations from this Dutch text.
Use ONLY these types from the ontology:

{ontology_prompt}

Return JSON:
{{"entities": [{{"name": "exact name from text", "type": "EntityType"}}],
 "relations": [{{"source": "entity name", "target": "entity name", "type": "RelationType"}}]}}

RULES:
- The text is in Dutch (Nederlands). Preserve Dutch names exactly.
- Extract ONLY named entities, not generic terms or descriptions.
- Do not invent entities. Only extract what is explicitly mentioned.
- No descriptions needed, only name and type.
- Use Dutch entity names as they appear (e.g. "Ministerie van BZK" not "Ministry of Interior").

TEXT:
{chunk_text}"""

    try:
        response_text = await call_llm(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            step="extraction",
            privacy=privacy,
            model_override=model_override,
        )

        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(response_text[start:end])
            return data.get("entities", []), data.get("relations", [])
    except json.JSONDecodeError:
        pass
    except Exception as e:
        logger.warning(f"Chunk extraction failed: {e}")

    return [], []


# ---------------------------------------------------------------------------
# SurrealDB persistence
# ---------------------------------------------------------------------------


async def _write_to_surrealdb(
    entities: List[Dict],
    relations: List[Dict],
    doc_name: str,
) -> Tuple[int, int]:
    """Write entities and relations to SurrealDB."""
    from surrealdb import AsyncSurreal

    db = AsyncSurreal(SURREALDB_URL)
    await db.signin({"username": SURREALDB_USER, "password": SURREALDB_PASS})
    await db.use(SURREALDB_NS, SURREALDB_DB)

    entity_map: Dict[str, str] = {}
    ent_created = 0
    rel_created = 0

    try:
        for ent in entities:
            name = ent.get("name", "").strip()
            etype = ent.get("type", "Thing").strip()
            desc = ent.get("description", "")
            if not name:
                continue

            # Check existing
            existing = await db.query(
                "SELECT id FROM entity WHERE canonical_name = $name AND entity_type = $type LIMIT 1;",
                {"name": name, "type": etype},
            )
            if existing and isinstance(existing, list) and len(existing) > 0:
                if isinstance(existing[0], dict):
                    eid = str(existing[0].get("id", ""))
                elif isinstance(existing[0], list) and existing[0]:
                    eid = str(existing[0][0].get("id", ""))
                else:
                    eid = ""
                if eid:
                    entity_map[name] = eid
                    continue

            # Embed
            embedding = []
            try:
                async with httpx.AsyncClient(timeout=60) as ec:
                    resp = await ec.post(
                        f"{OLLAMA_URL}/api/embed",
                        json={"model": EMBED_MODEL, "input": f"{etype}: {name}. {desc}"},
                    )
                    if resp.status_code == 200:
                        embedding = resp.json().get("embeddings", [[]])[0]
            except Exception:
                pass

            result = await db.query(
                "CREATE entity CONTENT $data RETURN id;",
                {"data": {
                    "canonical_name": name,
                    "entity_type": etype,
                    "description": desc,
                    "source_documents": [doc_name],
                    "extraction_method": "llm",
                    "confidence": 0.85,
                    "embedding": embedding,
                    "status": "active",
                    "properties": {},
                }},
            )
            if result:
                eid = ""
                if isinstance(result, list):
                    for item in result:
                        if isinstance(item, dict) and "id" in item:
                            eid = str(item["id"])
                            break
                        elif isinstance(item, list):
                            for sub in item:
                                if isinstance(sub, dict) and "id" in sub:
                                    eid = str(sub["id"])
                                    break
                if eid:
                    entity_map[name] = eid
                    ent_created += 1

        # Relations
        for rel in relations:
            src = entity_map.get(rel.get("source", "").strip())
            tgt = entity_map.get(rel.get("target", "").strip())
            rtype = rel.get("type", "related_to").strip()
            if not src or not tgt or src == tgt:
                continue

            await db.query(
                f"RELATE {src}->relation->{tgt} SET "
                f"relation_type = $type, "
                f"source_documents = $docs, "
                f"extraction_method = 'llm', "
                f"confidence = 0.8, "
                f"status = 'active';",
                {"type": rtype, "docs": [doc_name]},
            )
            rel_created += 1

    finally:
        await db.close()

    return ent_created, rel_created


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ModelOverride(BaseModel):
    """Override the default LLM provider/model for this request."""
    provider: Optional[str] = Field(default=None, description="Provider: ollama, nvidia")
    model: Optional[str] = Field(default=None, description="Model name")
    api_key: Optional[str] = Field(default=None, description="API key (for cloud providers)")
    base_url: Optional[str] = Field(default=None, description="Base URL override")


class ExtractionOverride(BaseModel):
    ontology: Optional[str] = None
    chunk_size: Optional[int] = None


class ExtractRequest(BaseModel):
    file_path: str = Field(..., description="Path to input file (docling JSON, md, txt)")
    privacy: str = Field(default="internal", description="Privacy level: public, internal, confidential")
    write_to_db: bool = Field(default=True, description="Write entities/relations to SurrealDB")
    model_override: Optional[ModelOverride] = Field(default=None, description="Override default LLM")
    override: Optional[ExtractionOverride] = None


class EntityResult(BaseModel):
    name: str
    type: str
    description: str = ""


class RelationResult(BaseModel):
    source: str
    target: str
    type: str


class ExtractResponse(BaseModel):
    input_file: str
    output_file: Optional[str] = None
    ontology: str
    chunks_processed: int
    entities: List[EntityResult]
    relations: List[RelationResult]
    unique_entity_count: int
    unique_relation_count: int
    entities_written: int = 0
    relations_written: int = 0
    processing_seconds: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok", "service": "extraction"}


@app.get("/ontologies")
def get_ontologies():
    return {"ontologies": list_ontologies(), "default": DEFAULT_ONTOLOGY}


@app.get("/routing")
def get_routing():
    """Show current model routing configuration."""
    from model_routing import get_routing_summary
    return get_routing_summary()


@app.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest):
    """Extract entities and relations from a document."""
    file_path = Path(req.file_path)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {req.file_path}")

    t0 = time.time()

    # Resolve config
    ontology_name = (req.override.ontology if req.override and req.override.ontology else DEFAULT_ONTOLOGY)
    privacy = req.privacy or "internal"
    model_override_dict = req.model_override.model_dump(exclude_none=True) if req.model_override else None

    ontology_prompt = _load_ontology_prompt(ontology_name)
    if not ontology_prompt:
        raise HTTPException(status_code=400, detail=f"Ontology '{ontology_name}' not found")

    # Load and chunk
    chunks, doc_name = _load_and_chunk(file_path)
    if not chunks:
        raise HTTPException(status_code=400, detail="No content found in file")

    # Log routing info
    from model_routing import ProviderUnavailableError, get_model_config

    try:
        route = get_model_config("extraction", privacy, model_override_dict)
    except ProviderUnavailableError as exc:
        # PC.6: the AC is "says why it cannot". Uncaught, this reached the caller
        # as a bare 500, so the one place the reason was still known said nothing
        # at the one place a human would read it. 503 rather than 500: the route
        # is deliberately retired, not broken.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    logger.info(
        f"Extracting from {file_path.name}: {len(chunks)} chunks, "
        f"ontology={ontology_name}, privacy={privacy}, "
        f"provider={route.get('provider')}, model={route.get('model')}"
    )

    # Extract from all chunks — parallel in batches
    all_entities: List[Dict] = []
    all_relations: List[Dict] = []
    PARALLEL = 5  # concurrent LLM calls
    if route.get("provider_type") == "openai_compatible":
        PARALLEL = 3  # cloud free tiers have rate limits

    for batch_start in range(0, len(chunks), PARALLEL):
        batch = chunks[batch_start:batch_start + PARALLEL]
        tasks = [
            _extract_from_chunk(
                chunk["text"], ontology_prompt,
                privacy=privacy,
                model_override=model_override_dict,
            )
            for chunk in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, tuple):
                ents, rels = r
                all_entities.extend(ents)
                all_relations.extend(rels)
            elif isinstance(r, Exception):
                logger.warning(f"Chunk extraction failed: {r}")
        logger.info(f"  {min(batch_start + PARALLEL, len(chunks))}/{len(chunks)} chunks: {len(all_entities)} entities")

    # Deduplicate
    seen_ents: Dict[str, Dict] = {}
    for e in all_entities:
        key = f"{e.get('name','')}|{e.get('type','')}"
        if key not in seen_ents:
            seen_ents[key] = e
    unique_entities = list(seen_ents.values())

    seen_rels: set = set()
    unique_relations = []
    for r in all_relations:
        key = f"{r.get('source','')}|{r.get('target','')}|{r.get('type','')}"
        if key not in seen_rels:
            seen_rels.add(key)
            unique_relations.append(r)

    logger.info(f"Extracted: {len(all_entities)} raw → {len(unique_entities)} unique entities, "
                f"{len(all_relations)} raw → {len(unique_relations)} unique relations")

    # Write to SurrealDB
    ents_written = 0
    rels_written = 0
    if req.write_to_db:
        try:
            ents_written, rels_written = await _write_to_surrealdb(
                unique_entities, unique_relations, doc_name
            )
            logger.info(f"Written to SurrealDB: {ents_written} entities, {rels_written} relations")
        except Exception as e:
            logger.error(f"SurrealDB write failed: {e}")

    # Write output JSON
    output_dir = file_path.parent
    output_path = output_dir / f"{doc_name}_entities.json"
    output_data = {
        "document": doc_name,
        "ontology": ontology_name,
        "entities": unique_entities,
        "relations": unique_relations,
    }
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = time.time() - t0

    return ExtractResponse(
        input_file=str(file_path),
        output_file=str(output_path),
        ontology=ontology_name,
        chunks_processed=len(chunks),
        entities=[EntityResult(**e) for e in unique_entities if e.get("name") and e.get("type")],
        relations=[RelationResult(**r) for r in unique_relations if r.get("source") and r.get("target") and r.get("type")],
        unique_entity_count=len(unique_entities),
        unique_relation_count=len(unique_relations),
        entities_written=ents_written,
        relations_written=rels_written,
        processing_seconds=round(elapsed, 2),
    )


# ===========================================================================
# SurrealDB query helper
# ===========================================================================


async def _db_query(query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict]:
    """Execute a SurrealDB query and return normalized results."""
    from surrealdb import AsyncSurreal, RecordID

    def _parse(obj):
        if isinstance(obj, dict):
            return {k: _parse(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_parse(i) for i in obj]
        elif isinstance(obj, RecordID):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    db = AsyncSurreal(SURREALDB_URL)
    await db.signin({"username": SURREALDB_USER, "password": SURREALDB_PASS})
    await db.use(SURREALDB_NS, SURREALDB_DB)
    try:
        result = _parse(await db.query(query, params))
        # Normalize: always return flat list of dicts
        if isinstance(result, list):
            if result and isinstance(result[0], list):
                return result[-1]
            return result
        if isinstance(result, dict):
            return [result]
        return []
    finally:
        await db.close()


# ===========================================================================
# Entity browsing endpoints
# ===========================================================================


@app.get("/entities")
async def get_entities(
    type: Optional[str] = None,
    status: str = "active",
    limit: int = 100,
    offset: int = 0,
):
    """List entities with optional type filter."""
    where = f"WHERE status = '{status}'"
    if type:
        where += f" AND entity_type = '{type}'"

    entities = await _db_query(
        f"SELECT id, canonical_name, entity_type, description, confidence, "
        f"pagerank, community_id, array::len(source_documents) AS source_count "
        f"FROM entity {where} "
        f"ORDER BY entity_type, canonical_name "
        f"LIMIT {limit} START {offset};"
    )
    total = await _db_query(f"SELECT count() FROM entity {where} GROUP ALL;")
    total_count = total[0].get("count", 0) if total else 0

    # Entity type counts
    type_counts = await _db_query(
        f"SELECT entity_type, count() AS cnt FROM entity {where} GROUP BY entity_type;"
    )

    return {
        "entities": entities,
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "types": {r["entity_type"]: r["cnt"] for r in type_counts},
    }


@app.get("/entities/{entity_id:path}")
async def get_entity(entity_id: str):
    """Get a single entity with its relations and aliases."""
    entity = await _db_query(f"SELECT * FROM {entity_id};")
    if not entity:
        raise HTTPException(status_code=404, detail=f"Entity not found: {entity_id}")

    ent = entity[0]
    ent.pop("embedding", None)  # Don't return large embedding vector

    # Get relations (outgoing)
    outgoing = await _db_query(
        f"SELECT id, out, relation_type, confidence FROM relation "
        f"WHERE in = {entity_id} AND status = 'active';"
    )
    # Enrich with target names
    for rel in outgoing:
        target = await _db_query(f"SELECT canonical_name, entity_type FROM {rel.get('out', '')};")
        if target:
            rel["target_name"] = target[0].get("canonical_name", "")
            rel["target_type"] = target[0].get("entity_type", "")

    # Get relations (incoming)
    incoming = await _db_query(
        f"SELECT id, in, relation_type, confidence FROM relation "
        f"WHERE out = {entity_id} AND status = 'active';"
    )
    for rel in incoming:
        source = await _db_query(f"SELECT canonical_name, entity_type FROM {rel.get('in', '')};")
        if source:
            rel["source_name"] = source[0].get("canonical_name", "")
            rel["source_type"] = source[0].get("entity_type", "")

    # Get aliases
    aliases = await _db_query(
        f"SELECT alias_text, confidence FROM entity_alias "
        f"WHERE canonical_entity = {entity_id};"
    )

    return {
        "entity": ent,
        "outgoing_relations": outgoing,
        "incoming_relations": incoming,
        "aliases": aliases,
    }


@app.get("/entities/{entity_id:path}/graph")
async def get_entity_graph(entity_id: str, depth: int = 2):
    """Get the local graph around an entity (nodes + edges within depth hops)."""
    # BFS traversal
    visited_nodes: Dict[str, Dict] = {}
    edges: List[Dict] = []
    queue = [(entity_id, 0)]
    seen = {entity_id}

    while queue:
        current_id, current_depth = queue.pop(0)

        # Get node info
        node = await _db_query(
            f"SELECT id, canonical_name, entity_type, pagerank, community_id "
            f"FROM {current_id};"
        )
        if node:
            visited_nodes[current_id] = node[0]

        if current_depth >= depth:
            continue

        # Get outgoing relations
        rels = await _db_query(
            f"SELECT id, out, relation_type FROM relation "
            f"WHERE in = {current_id} AND status = 'active';"
        )
        for rel in rels:
            target = str(rel.get("out", ""))
            edges.append({
                "source": current_id,
                "target": target,
                "type": rel.get("relation_type", ""),
            })
            if target not in seen:
                seen.add(target)
                queue.append((target, current_depth + 1))

        # Get incoming relations
        rels = await _db_query(
            f"SELECT id, in, relation_type FROM relation "
            f"WHERE out = {current_id} AND status = 'active';"
        )
        for rel in rels:
            source = str(rel.get("in", ""))
            edges.append({
                "source": source,
                "target": current_id,
                "type": rel.get("relation_type", ""),
            })
            if source not in seen:
                seen.add(source)
                queue.append((source, current_depth + 1))

    return {
        "center": entity_id,
        "depth": depth,
        "nodes": list(visited_nodes.values()),
        "edges": edges,
        "node_count": len(visited_nodes),
        "edge_count": len(edges),
    }


# ===========================================================================
# Deduplication endpoints
# ===========================================================================


try:
    from rapidfuzz.distance import JaroWinkler as _JW
    def _string_sim(a: str, b: str) -> float:
        return _JW.normalized_similarity(a.lower(), b.lower())
except ImportError:
    def _string_sim(a: str, b: str) -> float:
        ta, tb = set(a.lower().split()), set(b.lower().split())
        return len(ta & tb) / max(len(ta), len(tb)) if ta and tb else 0.0


class DuplicateCandidate(BaseModel):
    entity_a_id: str
    entity_a_name: str
    entity_a_type: str
    entity_b_id: str
    entity_b_name: str
    entity_b_type: str
    string_similarity: float
    semantic_similarity: float
    combined_score: float
    auto_merge: bool = False


class DeduplicateResponse(BaseModel):
    total_entities: int
    candidates: List[DuplicateCandidate]
    auto_merge_count: int
    review_count: int
    processing_seconds: float


@app.post("/deduplicate", response_model=DeduplicateResponse)
async def deduplicate(
    entity_type: Optional[str] = None,
    merge_threshold: float = 0.85,
    review_threshold: float = 0.5,
):
    """Find duplicate entity candidates across all documents.

    Candidates above merge_threshold are marked auto_merge=True.
    Candidates between review_threshold and merge_threshold need manual review.
    """
    t0 = time.time()

    where = "WHERE status = 'active'"
    if entity_type:
        where += f" AND entity_type = '{entity_type}'"

    entities = await _db_query(f"SELECT * FROM entity {where};")

    if len(entities) < 2:
        return DeduplicateResponse(
            total_entities=len(entities), candidates=[],
            auto_merge_count=0, review_count=0, processing_seconds=0,
        )

    # Token blocking: shared tokens in canonical_name
    token_index: Dict[str, List[int]] = {}
    for i, ent in enumerate(entities):
        name = ent.get("canonical_name", "").lower()
        for token in name.split():
            if len(token) < 2:
                continue
            token_index.setdefault(token, []).append(i)

    pairs: set = set()
    for indices in token_index.values():
        if len(indices) > 50:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                pairs.add((indices[i], indices[j]))

    # Sorted neighbourhood
    sorted_idx = sorted(range(len(entities)),
                        key=lambda i: entities[i].get("canonical_name", "").lower())
    for i in range(len(sorted_idx)):
        for j in range(i + 1, min(i + 5, len(sorted_idx))):
            pairs.add((sorted_idx[i], sorted_idx[j]) if sorted_idx[i] < sorted_idx[j]
                       else (sorted_idx[j], sorted_idx[i]))

    # Score pairs
    candidates: List[DuplicateCandidate] = []
    for idx_a, idx_b in pairs:
        ea = entities[idx_a]
        eb = entities[idx_b]
        na = ea.get("canonical_name", "")
        nb = eb.get("canonical_name", "")

        str_sim = _string_sim(na, nb)

        sem_sim = 0.0
        emb_a = ea.get("embedding")
        emb_b = eb.get("embedding")
        if emb_a and emb_b and len(emb_a) == len(emb_b):
            dot = sum(a * b for a, b in zip(emb_a, emb_b))
            na_norm = math.sqrt(sum(a * a for a in emb_a))
            nb_norm = math.sqrt(sum(b * b for b in emb_b))
            if na_norm > 0 and nb_norm > 0:
                sem_sim = dot / (na_norm * nb_norm)

        combined = str_sim * 0.4 + sem_sim * 0.6

        if combined >= review_threshold:
            candidates.append(DuplicateCandidate(
                entity_a_id=str(ea.get("id", "")),
                entity_a_name=na,
                entity_a_type=ea.get("entity_type", ""),
                entity_b_id=str(eb.get("id", "")),
                entity_b_name=nb,
                entity_b_type=eb.get("entity_type", ""),
                string_similarity=round(str_sim, 3),
                semantic_similarity=round(sem_sim, 3),
                combined_score=round(combined, 3),
                auto_merge=combined >= merge_threshold,
            ))

    candidates.sort(key=lambda c: c.combined_score, reverse=True)

    auto_count = sum(1 for c in candidates if c.auto_merge)
    return DeduplicateResponse(
        total_entities=len(entities),
        candidates=candidates,
        auto_merge_count=auto_count,
        review_count=len(candidates) - auto_count,
        processing_seconds=round(time.time() - t0, 2),
    )


@app.get("/duplicates")
async def get_duplicates(
    entity_type: Optional[str] = None,
    min_score: float = 0.5,
):
    """Convenience alias: find duplicates with default thresholds."""
    return await deduplicate(
        entity_type=entity_type,
        review_threshold=min_score,
    )


class MergeRequest(BaseModel):
    canonical_id: str = Field(..., description="Entity to keep")
    duplicate_id: str = Field(..., description="Entity to merge into canonical")


@app.post("/merge")
async def merge(req: MergeRequest):
    """Merge a duplicate entity into a canonical entity.

    Transfers aliases, redirects relations, marks duplicate as merged.
    """
    from surrealdb import AsyncSurreal

    db = AsyncSurreal(SURREALDB_URL)
    await db.signin({"username": SURREALDB_USER, "password": SURREALDB_PASS})
    await db.use(SURREALDB_NS, SURREALDB_DB)

    try:
        # Verify both exist
        canonical = await db.query(f"SELECT * FROM {req.canonical_id};")
        duplicate = await db.query(f"SELECT * FROM {req.duplicate_id};")

        can_data = None
        dup_data = None
        if isinstance(canonical, list):
            for item in canonical:
                if isinstance(item, dict) and "canonical_name" in item:
                    can_data = item
                    break
                elif isinstance(item, list):
                    for sub in item:
                        if isinstance(sub, dict) and "canonical_name" in sub:
                            can_data = sub
                            break
        if isinstance(duplicate, list):
            for item in duplicate:
                if isinstance(item, dict) and "canonical_name" in item:
                    dup_data = item
                    break
                elif isinstance(item, list):
                    for sub in item:
                        if isinstance(sub, dict) and "canonical_name" in sub:
                            dup_data = sub
                            break

        if not can_data:
            raise HTTPException(status_code=404, detail=f"Canonical entity not found: {req.canonical_id}")
        if not dup_data:
            raise HTTPException(status_code=404, detail=f"Duplicate entity not found: {req.duplicate_id}")

        dup_name = dup_data.get("canonical_name", "")
        can_name = can_data.get("canonical_name", "")

        # Create alias from duplicate name
        if dup_name:
            await db.query(
                "CREATE entity_alias SET alias_text = $name, "
                f"canonical_entity = {req.canonical_id}, confidence = 1.0;",
                {"name": dup_name},
            )

        # Transfer existing aliases
        await db.query(
            f"UPDATE entity_alias SET canonical_entity = {req.canonical_id} "
            f"WHERE canonical_entity = {req.duplicate_id};"
        )

        # Merge source_documents
        can_docs = can_data.get("source_documents", []) or []
        dup_docs = dup_data.get("source_documents", []) or []
        merged_docs = list(set(can_docs + dup_docs))
        await db.query(
            f"UPDATE {req.canonical_id} SET source_documents = $docs, updated_at = time::now();",
            {"docs": merged_docs},
        )

        # Redirect relations
        await db.query(
            f"UPDATE relation SET out = {req.canonical_id} WHERE out = {req.duplicate_id};"
        )
        await db.query(
            f"UPDATE relation SET in = {req.canonical_id} WHERE in = {req.duplicate_id};"
        )

        # Mark duplicate as merged
        await db.query(
            f"UPDATE {req.duplicate_id} SET status = 'merged', "
            f"merged_into = '{req.canonical_id}', updated_at = time::now();"
        )

        return {
            "merged": True,
            "canonical": {"id": req.canonical_id, "name": can_name},
            "duplicate": {"id": req.duplicate_id, "name": dup_name},
            "alias_created": dup_name,
        }

    finally:
        await db.close()


# ===========================================================================
# Vocabulary management endpoints
# ===========================================================================


@app.get("/vocabularies")
async def get_vocabularies():
    """List loaded vocabularies with entry counts."""
    from vocabulary_manager import list_vocabs
    return {"vocabularies": list_vocabs()}


@app.post("/vocabularies/refresh")
async def refresh_vocabularies():
    """Download/refresh all external vocabularies to local YAML files."""
    from vocabulary_manager import refresh_all
    t0 = time.time()
    results = await refresh_all()
    return {
        "refreshed": results,
        "total_entries": sum(results.values()),
        "processing_seconds": round(time.time() - t0, 2),
    }


# ===========================================================================
# Entity validation endpoints
# ===========================================================================


@app.post("/validate")
async def validate_endpoint(
    entity_type: Optional[str] = None,
    use_api: bool = True,
    apply_changes: bool = True,
):
    """Validate all active entities against external vocabularies.

    Checks SurrealDB cache → local YAML → external APIs (if enabled).
    Reclassifies wrong types, flags generic terms and misclassified persons.
    """
    from entity_validator import validate_entities

    t0 = time.time()

    where = "WHERE status = 'active'"
    if entity_type:
        where += f" AND entity_type = '{entity_type}'"
    entities = await _db_query(f"SELECT * FROM entity {where};")

    if not entities:
        return {"message": "No entities to validate", "total": 0}

    results = await validate_entities(entities, use_api=use_api, apply_changes=apply_changes)

    by_status = {}
    for r in results:
        by_status.setdefault(r.status, []).append(r.to_dict())

    return {
        "total_entities": len(entities),
        "results_by_status": {k: len(v) for k, v in by_status.items()},
        "validated": by_status.get("validated", []),
        "reclassified": by_status.get("reclassified", []),
        "enriched": by_status.get("enriched", []),
        "flagged_generic": by_status.get("flagged_generic", []),
        "unvalidated": by_status.get("unvalidated", []),
        "processing_seconds": round(time.time() - t0, 2),
    }
