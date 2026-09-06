"""
Entity validator — validates extracted entities against external vocabularies.

Three-tier validation:
1. SurrealDB reference_entity cache (instant, no I/O)
2. Local YAML vocabularies (fast, filesystem only)
3. External API calls (slow, rate-limited — only for cache misses)

Results per entity:
- validated: confirmed by external source
- reclassified: entity_type was wrong, corrected
- enriched: added external_uri, canonical_name, or properties
- unvalidated: no match in any source (may still be correct)
"""

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx
from loguru import logger
from record_ids import validate_record_id
from vocabulary_manager import load_vocab

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# OLLAMA_URL was bound here and never read (PC.6, found by review): this module
# does not call Ollama. Removed rather than kept "for symmetry" with api.py — a
# constant that reads the environment and is never used is what makes an operator
# believe a knob exists.
SURREALDB_URL = os.getenv("SURREALDB_URL", "ws://host.docker.internal:8000/rpc")
SURREALDB_NS = os.getenv("SURREALDB_NS", "open_notebook")
SURREALDB_DB = os.getenv("SURREALDB_DB", "open_notebook")
SURREALDB_USER = os.getenv("SURREALDB_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_PASS", "root")


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    entity_id: str
    entity_name: str
    original_type: str
    status: str = "unvalidated"  # validated, reclassified, enriched, unvalidated
    canonical_name: Optional[str] = None
    corrected_type: Optional[str] = None
    external_uri: Optional[str] = None
    source_vocabulary: Optional[str] = None
    confidence: float = 0.0
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "original_type": self.original_type,
            "status": self.status,
            "confidence": self.confidence,
        }
        if self.canonical_name:
            d["canonical_name"] = self.canonical_name
        if self.corrected_type:
            d["corrected_type"] = self.corrected_type
        if self.external_uri:
            d["external_uri"] = self.external_uri
        if self.source_vocabulary:
            d["source_vocabulary"] = self.source_vocabulary
        if self.details:
            d["details"] = self.details
        return d


# ---------------------------------------------------------------------------
# Name matching helpers
# ---------------------------------------------------------------------------


def _normalize(name: str) -> str:
    """Normalize a name for comparison."""
    name = name.lower().strip()
    name = re.sub(r"\s+", " ", name)
    # Remove common prefixes
    for prefix in ["gemeente ", "ministerie van ", "provincie "]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def _name_matches(entity_name: str, vocab_name: str, vocab_aliases: List[str]) -> float:
    """Check if an entity name matches a vocabulary entry. Returns confidence 0-1."""
    en = _normalize(entity_name)
    vn = _normalize(vocab_name)

    # Exact match
    if en == vn:
        return 1.0

    # Entity name matches an alias
    for alias in vocab_aliases:
        if en == _normalize(alias):
            return 0.95

    # Entity name is contained in vocab name or vice versa
    if en in vn or vn in en:
        return 0.8

    # Check if normalized forms match with prefix removed
    en_stripped = en.replace("de ", "").replace("het ", "")
    vn_stripped = vn.replace("de ", "").replace("het ", "")
    if en_stripped == vn_stripped:
        return 0.85

    return 0.0


# ---------------------------------------------------------------------------
# Tier 1: SurrealDB reference_entity cache
# ---------------------------------------------------------------------------


async def _check_cache(entity_name: str, entity_type: str, db) -> Optional[Dict]:
    """Check if entity exists in the reference_entity cache."""
    norm = _normalize(entity_name)
    results = await db.query(
        "SELECT * FROM reference_entity WHERE "
        "string::lowercase(canonical_name) = $name "
        "OR $name IN aliases "
        "LIMIT 1;",
        {"name": norm},
    )
    if results and isinstance(results, list):
        for item in results:
            if isinstance(item, dict) and "canonical_name" in item:
                return item
            if isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict) and "canonical_name" in sub:
                        return sub
    return None


async def _write_cache(entry: Dict, db) -> None:
    """Write a validated entity to the reference_entity cache."""
    try:
        await db.query(
            "CREATE reference_entity CONTENT $data;",
            {"data": {
                "canonical_name": entry.get("name", ""),
                "entity_type": entry.get("type", ""),
                "source_vocabulary": entry.get("source", ""),
                "external_uri": entry.get("uri", ""),
                "aliases": [_normalize(a) for a in entry.get("aliases", [])],
                "properties": entry.get("properties", {}),
                "parent_name": entry.get("parent", ""),
            }},
        )
    except Exception as e:
        logger.debug(f"Cache write failed (may already exist): {e}")


# ---------------------------------------------------------------------------
# Tier 2: Local YAML vocabularies
# ---------------------------------------------------------------------------


# Loaded vocabularies: name → list of entries
_vocab_cache: Dict[str, List[Dict]] = {}


def _get_vocab(name: str) -> List[Dict]:
    """Load vocabulary with in-memory caching."""
    if name not in _vocab_cache:
        _vocab_cache[name] = load_vocab(name)
    return _vocab_cache[name]


def _check_local_vocabs(
    entity_name: str,
    entity_type: str,
) -> Optional[Tuple[Dict, float, str]]:
    """Check entity against all local vocabularies.

    Returns (matching_entry, confidence, vocab_name) or None.
    """
    # Map entity types to relevant vocabularies
    type_vocab_map = {
        "Gemeente": ["tooi_gemeenten"],
        "Ministerie": ["tooi_organisaties"],
        "Provincie": ["tooi_organisaties"],
        "Functionaris": [],  # No vocab — will be caught by reclassification
        "BredeWelvaartThema": ["cbs_brede_welvaart"],
        "BredeWelvaartIndicator": ["cbs_brede_welvaart"],
        "BredeWelvaartConcept": ["cbs_brede_welvaart"],
        "thematic concept": ["cbs_brede_welvaart"],
        "brede welvaart theme": ["cbs_brede_welvaart"],
    }

    # Check type-specific vocabs first
    vocabs_to_check = type_vocab_map.get(entity_type, [])

    # Then check all vocabs (for reclassification)
    all_vocabs = ["tooi_gemeenten", "tooi_organisaties", "cbs_brede_welvaart", "openalex_topics"]

    for vocab_name in vocabs_to_check + [v for v in all_vocabs if v not in vocabs_to_check]:
        vocab = _get_vocab(vocab_name)
        for entry in vocab:
            conf = _name_matches(entity_name, entry.get("name", ""), entry.get("aliases", []))
            if conf >= 0.8:
                return entry, conf, vocab_name

    return None


# ---------------------------------------------------------------------------
# Tier 3: External API calls (only for cache misses)
# ---------------------------------------------------------------------------


async def _check_ror(entity_name: str) -> Optional[Dict]:
    """Check ROR for research organizations."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://api.ror.org/v2/organizations",
                params={"query": entity_name, "filter": "locations.geonames_details.country_code:NL"},
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            items = data.get("items", [])
            if not items:
                return None
            best = items[0]
            names = best.get("names", [])
            display_name = next((n["value"] for n in names if "ror_display" in n.get("types", [])), "")
            aliases = [n["value"] for n in names if n["value"] != display_name]
            return {
                "name": display_name or entity_name,
                "type": "Kennisinstelling",
                "uri": best.get("id", ""),
                "aliases": aliases,
                "source": "ror",
                "properties": {
                    "ror_id": best.get("id", ""),
                    "country": "NL",
                    "type": best.get("types", []),
                },
            }
    except Exception as e:
        logger.debug(f"ROR check failed for '{entity_name}': {e}")
    return None


async def _check_openalex_author(entity_name: str) -> Optional[Dict]:
    """Check OpenAlex for authors (to reclassify persons wrongly typed)."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://api.openalex.org/authors",
                params={"search": entity_name, "per_page": 1},
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return None
            author = results[0]
            if author.get("works_count", 0) < 3:
                return None  # Not a notable author
            return {
                "name": author.get("display_name", entity_name),
                "type": "Persoon",
                "uri": author.get("id", ""),
                "aliases": [],
                "source": "openalex",
                "properties": {
                    "orcid": author.get("orcid", ""),
                    "works_count": author.get("works_count", 0),
                    "cited_by_count": author.get("cited_by_count", 0),
                },
            }
    except Exception as e:
        logger.debug(f"OpenAlex author check failed for '{entity_name}': {e}")
    return None


# ---------------------------------------------------------------------------
# Entity type reclassification heuristics
# ---------------------------------------------------------------------------

# Patterns that suggest an entity is a person, not a gemeente/ministerie
_PERSON_PATTERNS = [
    r"^[A-Z]\.\s",           # "K.H. Ollongren"
    r"^(dr|mr|ir|prof|drs)\.",  # Academic titles
    r"\b(van|de|den|der|het|ter)\b.*[A-Z]",  # Dutch surname patterns
    r"^[A-Z][a-z]+\s[A-Z][a-z]+$",  # "Bart Huizing" (First Last)
]


def _looks_like_person(name: str) -> bool:
    """Heuristic: does this name look like a person rather than an organization?"""
    for pattern in _PERSON_PATTERNS:
        if re.search(pattern, name):
            return True
    # Names with 2-3 words, all capitalized, no organizational keywords
    words = name.split()
    if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if len(w) > 1):
        org_words = {"gemeente", "ministerie", "provincie", "regio", "programma",
                     "deal", "fonds", "raad", "commissie", "alliantie", "board"}
        if not any(w.lower() in org_words for w in words):
            return True
    return False


# Generic terms that should not be entities
_GENERIC_TERMS = {
    "gemeente", "ministerie", "provincie", "regio", "wijk", "land",
    "nederland", "rijk", "overheden", "inwoners", "jongeren", "bedrijven",
    "bedrijfsleven", "ondernemers", "onderwijs", "zorg en welzijn",
    "gemeentes", "gemeenteraden", "provincies", "sectoren", "partijen",
    "woningcorporatie", "kennisinstelling", "onderwijsinstelling",
    "zorginstelling", "nationaalprogramma", "regiodeal", "economicboard",
    "grensregio", "functioneleRegio", "woondeal", "citydeal",
    "beroepsonderwijs", "kinderopvang",
}


def _is_generic(name: str) -> bool:
    """Check if the name is too generic to be a useful entity."""
    return _normalize(name) in _GENERIC_TERMS


# ---------------------------------------------------------------------------
# Main validation pipeline
# ---------------------------------------------------------------------------


async def validate_entities(
    entities: List[Dict[str, Any]],
    use_api: bool = True,
    apply_changes: bool = True,
) -> List[ValidationResult]:
    """Validate a list of entities through the 3-tier pipeline.

    Args:
        entities: List of entity dicts from SurrealDB (id, canonical_name, entity_type).
        use_api: Whether to make external API calls (tier 3).
        apply_changes: Whether to update entities in SurrealDB.

    Returns:
        List of ValidationResult for each entity.
    """
    from surrealdb import AsyncSurreal

    db = AsyncSurreal(SURREALDB_URL)
    await db.signin({"username": SURREALDB_USER, "password": SURREALDB_PASS})
    await db.use(SURREALDB_NS, SURREALDB_DB)

    results: List[ValidationResult] = []
    stats = {"validated": 0, "reclassified": 0, "enriched": 0,
             "flagged_generic": 0, "flagged_person": 0, "unvalidated": 0,
             "cache_hits": 0, "vocab_hits": 0, "api_hits": 0}

    try:
        for ent in entities:
            # Validated HERE, once, before any of the four UPDATEs below
            # interpolate it. `eid` comes from the database, which is precisely
            # the assumption a stored malformed id exploits — the argument
            # `api.py` already makes for its own read-back ids.
            eid = validate_record_id(str(ent.get("id", "")), field="entity id")
            name = ent.get("canonical_name", "")
            etype = ent.get("entity_type", "")

            vr = ValidationResult(entity_id=eid, entity_name=name, original_type=etype)

            # Pre-check: is it a generic term?
            if _is_generic(name):
                vr.status = "flagged_generic"
                vr.details = "Generic term, not a specific entity"
                vr.confidence = 1.0
                stats["flagged_generic"] += 1
                if apply_changes:
                    await db.query(f"UPDATE {eid} SET status = 'flagged_generic';")
                results.append(vr)
                continue

            # Pre-check: is it a person wrongly classified?
            if etype in ("Gemeente", "Ministerie", "Grensregio", "Kennisinstelling") and _looks_like_person(name):
                vr.status = "reclassified"
                vr.corrected_type = "Persoon"
                vr.details = f"Name pattern suggests person, not {etype}"
                vr.confidence = 0.85
                stats["flagged_person"] += 1
                if apply_changes:
                    await db.query(f"UPDATE {eid} SET entity_type = 'Persoon';")
                results.append(vr)
                continue

            # Tier 1: SurrealDB cache
            cached = await _check_cache(name, etype, db)
            if cached:
                vr.status = "validated"
                vr.canonical_name = cached.get("canonical_name")
                vr.external_uri = str(cached.get("external_uri", ""))
                vr.source_vocabulary = cached.get("source_vocabulary", "cache")
                vr.confidence = 0.95
                stats["validated"] += 1
                stats["cache_hits"] += 1
                results.append(vr)
                continue

            # Tier 2: Local vocabularies
            local_match = _check_local_vocabs(name, etype)
            if local_match:
                entry, conf, vocab_name = local_match
                matched_type = entry.get("type", etype)

                if matched_type != etype and matched_type not in ("", etype):
                    vr.status = "reclassified"
                    vr.corrected_type = matched_type
                    vr.details = f"Vocabulary '{vocab_name}' classifies as {matched_type}"
                    stats["reclassified"] += 1
                    if apply_changes:
                        await db.query(f"UPDATE {eid} SET entity_type = $type;", {"type": matched_type})
                else:
                    vr.status = "validated"
                    stats["validated"] += 1

                vr.canonical_name = entry.get("name")
                vr.external_uri = entry.get("uri", "")
                vr.source_vocabulary = vocab_name
                vr.confidence = conf
                stats["vocab_hits"] += 1

                # Write to cache
                entry["source"] = vocab_name
                await _write_cache(entry, db)

                # Update entity canonical_name if different
                if apply_changes and vr.canonical_name and vr.canonical_name != name:
                    await db.query(
                        f"UPDATE {eid} SET properties.validated_name = $vname;",
                        {"vname": vr.canonical_name},
                    )

                results.append(vr)
                continue

            # Tier 3: External APIs (if enabled)
            if use_api and etype in ("Onderwijsinstelling", "Kennisinstelling", "Zorginstelling"):
                ror_match = await _check_ror(name)
                if ror_match:
                    vr.status = "enriched"
                    vr.canonical_name = ror_match["name"]
                    vr.external_uri = ror_match["uri"]
                    vr.source_vocabulary = "ror"
                    vr.confidence = 0.9
                    stats["enriched"] += 1
                    stats["api_hits"] += 1
                    await _write_cache(ror_match, db)
                    results.append(vr)
                    continue

            # No match
            vr.status = "unvalidated"
            stats["unvalidated"] += 1
            results.append(vr)

    finally:
        await db.close()

    logger.info(
        f"Validation complete: {len(results)} entities — "
        f"validated={stats['validated']}, reclassified={stats['reclassified']}, "
        f"enriched={stats['enriched']}, generic={stats['flagged_generic']}, "
        f"person={stats['flagged_person']}, unvalidated={stats['unvalidated']} "
        f"(cache={stats['cache_hits']}, vocab={stats['vocab_hits']}, api={stats['api_hits']})"
    )

    return results
