"""Open Knowledge Format (OKF v0.1) import service (Phase OKF.3).

The inverse of :mod:`okf_export_service`. Reads an external OKF **Knowledge
Bundle** — a tree of markdown *concepts* with YAML frontmatter and standard
markdown cross-links (vendored spec:
``docs/tracks/OKF-open-knowledge-format/SPEC-v0.1.md``) — and reconstructs the
entities / relations / notes / sources it describes into the knowledge graph.

Two clean layers
================
* **Pure parsing** (:func:`parse_okf_bundle`) — a bundle (a directory, a zip, or
  the ``{path: text}`` mapping the exporter emits) is split into typed
  :class:`ParsedEntity` / :class:`ParsedNote` / :class:`ParsedSource` /
  :class:`ParsedRelation` records. No DB, no clock; fully offline-testable and
  the seam the OKF.1↔OKF.3 round-trip is asserted over.
* **Persistence** (:meth:`OkfImportService.import_bundle`) — writes the parsed
  records through the *existing* canonical paths: entities via
  :meth:`EntityRepository.upsert_entity` (deterministic ``(name, type)`` dedup),
  relations via the injection-safe ``_upsert_relation`` RELATE primitive, notes
  and sources via their repositories. Fuzzy / near-duplicate collapse against the
  already-persisted graph reuses the **K.5** candidate-dedup proposer + **K.3**
  recanonicalization apply — this service invents no matcher of its own.

Reuse map
=========
* Frontmatter parsing — :func:`vault_sync_service._parse_frontmatter` (the same
  ``^---\\n…\\n---`` pattern the vault sync uses), not a reinvented parser.
* Entity typing — :func:`entity_persistence_service._resolve_entity_type` (the
  L.1/L.2 ontology-bridge + alias map) so an OKF ``type: Person`` lands on the
  same canonical enum an extraction would.
* Relation write — :meth:`EntityPersistenceService._upsert_relation` (the
  ``type::thing($id)`` param-bound RELATE that dedups on
  ``(in, out, relation_type)`` and respects the SurrealDB RELATE id-injection
  guard).
* Dedup — :class:`CandidateDedupService` (propose) + :class:`RecanonicalizationService`
  (apply auto-merge clusters).

Provenance (OKF-D3)
===================
Every imported node is tagged so it is distinguishable from a natively-extracted
one: entities carry ``extraction_method="okf-import"`` plus an ``okf_import``
properties flag; notes/sources carry an ``okf_import`` metadata flag and a stable
``okf_key`` used for idempotent re-import. Re-importing the same bundle mutates
nothing new — entities/relations dedup on their natural keys and notes/sources on
``okf_key``.

Lossy-by-design
===============
The bundle is the curated-knowledge subset only (see
:data:`okf_export_service.OKF_OMITTED_FIELDS`). Import cannot recover what export
never wrote: embeddings, chunk-level provenance, verdict/similarity edges. Two
further round-trip losses are structural to OKF and documented here:

* **Relation direction** — the exporter renders a relation on *both* endpoints as
  an undirected link, so import collapses the two links back to a single edge but
  cannot recover which endpoint was ``in`` vs ``out``. Edges are written in a
  deterministic (sorted-endpoint) direction.
* **Internal ``resource`` URNs** — an entity with no real external id exports a
  synthesised ``urn:open-notebook:…`` resource; on import that is preserved as an
  ``okf_resource`` property but is NOT re-promoted to ``external_ids`` (only a
  genuine URI — DOI/TOOI/http — is), so a re-export regenerates a fresh URN.
"""

from __future__ import annotations

import hashlib
import io
import posixpath
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from loguru import logger
from shared.models.entity import Entity
from surrealdb_service.connection import execute_query

from app_main.services.entity_persistence_service import (
    EntityPersistenceService,
    _resolve_entity_type,
)
from app_main.services.okf_export_service import RESERVED_STEMS
from app_main.services.vault_sync_service import _parse_frontmatter

# ---------------------------------------------------------------------------
# Concept routing (spec §4.1 ``type``)
# ---------------------------------------------------------------------------

#: OKF ``type`` values the exporter reserves for the notebook's Note / Source
#: concepts (see :mod:`okf_export_service`). Everything else routes to an entity.
NOTE_TYPE = "note"
SOURCE_TYPE = "source"

#: Concept ``type`` values that are structural, not knowledge — skipped on import.
#: ``Index`` is the reserved root-index frontmatter type; ``Log`` mirrors the
#: reserved ``log.md``. Reserved *filenames* are also skipped up-front (§3.1).
SKIPPED_CONCEPT_TYPES = frozenset({"index", "log"})

#: Default edge label when a relation link carries no ``— <type>`` suffix.
DEFAULT_RELATION_TYPE = "related to"

# A markdown link: ``[label](url)``. ``url`` is captured for target resolution.
_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

# Text immediately following a relation link — the exporter writes
# ``[label](/entities/x.md) — RELATION_TYPE``; capture the label after an
# em/en-dash (or hyphen) separator, stopping before any further link.
_REL_SUFFIX_RE = re.compile(r"^\s*[—–-]\s+([^\[]+?)\s*$")

# A frontmatter block plus its extent, so the body (everything after the closing
# ``---``) can be sliced for cross-link extraction. Mirrors the vault-sync regex
# (reused via ``_parse_frontmatter`` for the dict) with a trailing-newline group.
_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---[ \t]*\n?", re.DOTALL)


# ---------------------------------------------------------------------------
# Parsed records (pure)
# ---------------------------------------------------------------------------


@dataclass
class ParsedEntity:
    """One concept routed to an entity (frontmatter + body)."""

    concept_id: str
    canonical_name: str
    okf_type: str
    description: Optional[str] = None
    resource: Optional[str] = None
    type_tags: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedNote:
    """A concept with ``type: Note``."""

    concept_id: str
    title: str
    content: str


@dataclass
class ParsedSource:
    """A concept with ``type: Source``."""

    concept_id: str
    title: str
    resource: Optional[str] = None
    topics: List[str] = field(default_factory=list)


@dataclass
class ParsedRelation:
    """An entity→entity edge recovered from a cross-link (direction lossy)."""

    from_id: str
    to_id: str
    relation_type: str


@dataclass
class SkippedConcept:
    """A concept that could not be imported, with a non-silent reason."""

    path: str
    reason: str


@dataclass
class ParsedBundle:
    """Pure output of :func:`parse_okf_bundle`."""

    entities: Dict[str, ParsedEntity] = field(default_factory=dict)
    notes: List[ParsedNote] = field(default_factory=list)
    sources: Dict[str, ParsedSource] = field(default_factory=dict)
    relations: List[ParsedRelation] = field(default_factory=list)
    skipped: List[SkippedConcept] = field(default_factory=list)
    dangling_links: List[Tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bundle loading (dir | zip | mapping)
# ---------------------------------------------------------------------------


def _load_bundle_files(
    source: Union[str, Path, Mapping[str, str], bytes],
) -> Dict[str, str]:
    """Normalise any accepted bundle form into a ``{concept_path: text}`` map.

    Accepts the in-memory ``{path: text}`` mapping the exporter emits, a
    directory tree, a ``.zip`` path, or raw zip ``bytes``. Only ``.md`` files
    are considered; a file that is not valid UTF-8 is recorded as unreadable by
    the caller (it never appears in the returned map).
    """
    if isinstance(source, Mapping):
        return {str(k): str(v) for k, v in source.items()}

    if isinstance(source, bytes):
        return _read_zip_bytes(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"OKF bundle not found: {path}")

    if path.is_file():
        if path.suffix.lower() == ".zip":
            return _read_zip_bytes(path.read_bytes())
        raise ValueError(
            f"OKF bundle file must be a .zip archive, got: {path.name}"
        )

    files: Dict[str, str] = {}
    for md in sorted(path.rglob("*.md")):
        rel = md.relative_to(path).as_posix()
        try:
            files[rel] = md.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            logger.warning("OKF import: unreadable file {p}: {e}", p=rel, e=exc)
    return files


def _read_zip_bytes(data: bytes) -> Dict[str, str]:
    """Extract ``{path: text}`` for every ``.md`` member of a zip archive."""
    files: Dict[str, str] = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in archive.namelist():
            if not name.endswith(".md"):
                continue
            try:
                files[name] = archive.read(name).decode("utf-8")
            except (UnicodeDecodeError, KeyError) as exc:
                logger.warning(
                    "OKF import: unreadable zip member {n}: {e}", n=name, e=exc
                )
    return files


# ---------------------------------------------------------------------------
# Concept + link parsing (pure)
# ---------------------------------------------------------------------------


def _concept_id(path: str) -> str:
    """Concept ID = bundle-relative path with the ``.md`` suffix removed (§2)."""
    return path[:-3] if path.endswith(".md") else path


def _is_reserved_file(path: str) -> bool:
    """True for a reserved filename at any level (``index.md`` / ``log.md``)."""
    return path.rsplit("/", 1)[-1].rsplit(".", 1)[0] in RESERVED_STEMS


def _split_frontmatter(content: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return ``(frontmatter_dict, body)`` reusing the vault-sync parser.

    ``_parse_frontmatter`` (vault sync) yields the dict; the body is everything
    after the closing ``---``. A missing/unparseable block yields ``(None, body)``
    where ``body`` is the raw content — the caller treats a ``None`` frontmatter
    (or one with no ``type``) as malformed and skips it non-silently.
    """
    fm = _parse_frontmatter(content)
    if not fm:
        return None, content
    m = _FRONTMATTER_RE.match(content)
    body = content[m.end():] if m else content
    return fm, body


def _resolve_link_target(url: str, current_concept_id: str) -> Optional[str]:
    """Resolve a markdown link URL to a bundle concept ID, or ``None``.

    Handles absolute bundle-relative links (``/entities/x.md`` → ``entities/x``,
    spec §5.1) and relative links (``./other.md`` resolved against the current
    concept's directory, §5.2). External URLs, fragments and anchors return
    ``None`` (they are not intra-bundle edges).
    """
    if not url or url.startswith(("http://", "https://", "mailto:", "#", "urn:")):
        return None
    clean = url.split("#", 1)[0].split("?", 1)[0].strip()
    if not clean:
        return None
    if "://" in clean:
        return None
    if clean.startswith("/"):
        rel = clean[1:]
    else:
        base = (
            current_concept_id.rsplit("/", 1)[0]
            if "/" in current_concept_id
            else ""
        )
        rel = posixpath.normpath(posixpath.join(base, clean))
    if rel.endswith(".md"):
        rel = rel[:-3]
    rel = rel.strip("/")
    return rel or None


def _extract_relation_links(
    body: str, current_concept_id: str
) -> List[Tuple[str, str]]:
    """Extract ``(target_concept_id, relation_type)`` from a concept body.

    A relation type is the ``— <label>`` suffix the exporter writes after a link
    (``[Bob](/entities/bob.md) — KNOWS``); absent, the default is used. Targets
    are returned raw (resolved but unclassified) so the caller can decide whether
    each is an entity, a citation to a source/note, or dangling.
    """
    out: List[Tuple[str, str]] = []
    for line in body.splitlines():
        for match in _LINK_RE.finditer(line):
            target = _resolve_link_target(match.group(1), current_concept_id)
            if target is None:
                continue
            suffix = _REL_SUFFIX_RE.match(line[match.end():])
            rel_type = suffix.group(1).strip() if suffix else DEFAULT_RELATION_TYPE
            out.append((target, rel_type))
    return out


def _as_str_list(value: Any) -> List[str]:
    """Coerce a frontmatter scalar/list into a clean list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def parse_okf_bundle(
    source: Union[str, Path, Mapping[str, str], bytes],
) -> ParsedBundle:
    """Parse an OKF bundle into typed records. **Pure — no DB, no clock.**

    Routing (spec §4.1): reserved filenames and ``Index``/``Log`` types are
    skipped; ``Note`` / ``Source`` types map to their records; every other type
    is an entity whose ``title`` is its canonical name and ``type`` its entity
    type. Relations are recovered from entity-body cross-links pointing at other
    entity concepts (§5); a link to a target absent from the bundle is recorded
    in ``dangling_links`` (never an error — §5.3). The two undirected links the
    exporter writes per relation collapse to one edge (direction is lossy).
    """
    files = _load_bundle_files(source)
    result = ParsedBundle()

    # Pass 1 — classify every concept so relation resolution (pass 2) can tell an
    # entity target from a note/source citation or a dangling link.
    for path in sorted(files):
        if _is_reserved_file(path):
            continue
        concept_id = _concept_id(path)
        content = files[path]
        frontmatter, body = _split_frontmatter(content)

        if frontmatter is None:
            result.skipped.append(
                SkippedConcept(path, "missing or unparseable YAML frontmatter")
            )
            continue
        raw_type = frontmatter.get("type")
        if not raw_type or not str(raw_type).strip():
            result.skipped.append(
                SkippedConcept(path, "frontmatter has no required 'type' field")
            )
            continue

        type_key = str(raw_type).strip().lower()
        if type_key in SKIPPED_CONCEPT_TYPES:
            continue

        title = frontmatter.get("title")
        title = str(title).strip() if title else ""
        if not title:
            # Consumers MAY derive a title from the filename (§4.1).
            title = concept_id.rsplit("/", 1)[-1]

        if type_key == NOTE_TYPE:
            content_body = body.strip()
            # The exporter renders the note as ``# <title>`` then the content;
            # drop the leading title heading so the stored note is just its body.
            content_body = _strip_leading_heading(content_body, title)
            if not content_body:
                result.skipped.append(
                    SkippedConcept(path, "note concept has empty content")
                )
                continue
            result.notes.append(
                ParsedNote(concept_id=concept_id, title=title, content=content_body)
            )
        elif type_key == SOURCE_TYPE:
            result.sources[concept_id] = ParsedSource(
                concept_id=concept_id,
                title=title,
                resource=_clean_scalar(frontmatter.get("resource")),
                topics=_as_str_list(frontmatter.get("tags")),
            )
        else:
            result.entities[concept_id] = ParsedEntity(
                concept_id=concept_id,
                canonical_name=title,
                okf_type=str(raw_type).strip(),
                description=_clean_scalar(frontmatter.get("description")),
                resource=_clean_scalar(frontmatter.get("resource")),
                type_tags=_as_str_list(frontmatter.get("tags")),
                properties={"okf_body_present": bool(body.strip())},
            )

    # Pass 2 — resolve entity-body cross-links into relations.
    known = set(result.entities) | set(result.sources) | {
        n.concept_id for n in result.notes
    }
    seen: set[Tuple[frozenset, str]] = set()
    for concept_id, entity in result.entities.items():
        _, body = _split_frontmatter(files[f"{concept_id}.md"])
        for target, rel_type in _extract_relation_links(body, concept_id):
            if target == concept_id:
                continue
            if target not in result.entities:
                if target not in known:
                    result.dangling_links.append((concept_id, target))
                # else: a citation to a source/note — not an entity→entity edge.
                continue
            key = (frozenset({concept_id, target}), rel_type)
            if key in seen:
                continue
            seen.add(key)
            src, dst = sorted((concept_id, target))
            result.relations.append(
                ParsedRelation(from_id=src, to_id=dst, relation_type=rel_type)
            )

    return result


def _strip_leading_heading(body: str, title: str) -> str:
    """Drop a leading ``# <title>`` heading the exporter prepends to a note."""
    lines = body.splitlines()
    if lines and lines[0].strip() in (f"# {title}", f"#{title}"):
        return "\n".join(lines[1:]).strip()
    return body.strip()


def _clean_scalar(value: Any) -> Optional[str]:
    """Return a trimmed string for a scalar frontmatter value, else ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# ---------------------------------------------------------------------------
# Import report
# ---------------------------------------------------------------------------


@dataclass
class OkfImportReport:
    """Counts + non-silent skip/dangling ledger for one import run."""

    entities_created: int = 0
    entities_matched: int = 0
    entities_failed: int = 0
    relations_created: int = 0
    notes_created: int = 0
    notes_matched: int = 0
    sources_created: int = 0
    sources_matched: int = 0
    dedup_merges: int = 0
    import_source_id: Optional[str] = None
    skipped: List[SkippedConcept] = field(default_factory=list)
    dangling_links: List[Tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities_created": self.entities_created,
            "entities_matched": self.entities_matched,
            "entities_failed": self.entities_failed,
            "relations_created": self.relations_created,
            "notes_created": self.notes_created,
            "notes_matched": self.notes_matched,
            "sources_created": self.sources_created,
            "sources_matched": self.sources_matched,
            "dedup_merges": self.dedup_merges,
            "import_source_id": self.import_source_id,
            "skipped": [{"path": s.path, "reason": s.reason} for s in self.skipped],
            "dangling_links": [
                {"from": a, "to": b} for a, b in self.dangling_links
            ],
        }


# A bundle-relative concept path used as an internal urn scope — a genuine
# external identifier looks like a URI (has a scheme + ``://`` or a known scheme
# prefix) and is NOT our own synthesised placeholder.
_REAL_URI_RE = re.compile(r"^(?:[a-z][a-z0-9+.-]*://|doi:|isbn:|hdl:)", re.IGNORECASE)
_INTERNAL_URN_PREFIX = "urn:open-notebook:"


def _is_external_id(resource: Optional[str]) -> bool:
    """True when ``resource`` is a real external URI worth pinning to external_ids."""
    if not resource:
        return False
    if resource.startswith(_INTERNAL_URN_PREFIX):
        return False
    return bool(_REAL_URI_RE.match(resource))


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class OkfImportService:
    """Import an OKF v0.1 bundle into the knowledge graph.

    Stateless aside from injected collaborators. The persistence path reuses the
    canonical write surfaces (upsert, RELATE primitive, repositories) and the
    K.5/K.3 dedup machinery; the parsing seam is pure and DB-free.
    """

    def __init__(
        self,
        entity_repository: Any,
        persistence_service: EntityPersistenceService,
        note_repository: Any,
        source_repository: Any,
        candidate_dedup_service: Optional[Any] = None,
        recanonicalization_service: Optional[Any] = None,
    ) -> None:
        self._entity_repo = entity_repository
        self._persistence = persistence_service
        self._note_repo = note_repository
        self._source_repo = source_repository
        self._dedup = candidate_dedup_service
        self._recanon = recanonicalization_service

    # -- public parse seam (pure) -------------------------------------------

    @staticmethod
    def parse(
        source: Union[str, Path, Mapping[str, str], bytes],
    ) -> ParsedBundle:
        """Pure parse of a bundle — see :func:`parse_okf_bundle`."""
        return parse_okf_bundle(source)

    # -- import --------------------------------------------------------------

    async def import_bundle(
        self,
        source: Union[str, Path, Mapping[str, str], bytes],
        *,
        notebook_id: Optional[str] = None,
        apply_dedup: bool = True,
    ) -> OkfImportReport:
        """Parse + persist a bundle, returning a counts/skip report.

        Args:
            source: a directory, ``.zip`` path/bytes, or ``{path: text}`` mapping.
            notebook_id: when given, imported notes/sources (and the provenance
                anchor source) are linked to that notebook and the K.5 dedup pass
                is scoped to it; otherwise dedup runs globally.
            apply_dedup: run the K.5 propose + K.3 apply-auto-merge pass after
                persistence. Idempotent — a no-op when nothing collides.

        Raises:
            FileNotFoundError / ValueError: for a malformed *bundle container*
                (missing dir, non-zip file). A malformed *concept* is skipped
                non-silently (recorded in the report), never fatal.
        """
        parsed = parse_okf_bundle(source)
        report = OkfImportReport(
            skipped=list(parsed.skipped),
            dangling_links=list(parsed.dangling_links),
        )

        import_source_id = await self._create_import_source(parsed, notebook_id)
        report.import_source_id = import_source_id

        # 1. Entities — canonical upsert (deterministic (name, type) dedup).
        concept_to_entity_id: Dict[str, str] = {}
        for concept_id, pe in parsed.entities.items():
            try:
                created, entity_id = await self._upsert_entity(pe, import_source_id)
            except Exception as exc:  # noqa: BLE001 - per-concept isolation
                report.entities_failed += 1
                report.skipped.append(
                    SkippedConcept(f"{concept_id}.md", f"entity upsert failed: {exc}")
                )
                logger.error("OKF import: entity '{c}' failed: {e}", c=concept_id, e=exc)
                continue
            concept_to_entity_id[concept_id] = entity_id
            if created:
                report.entities_created += 1
            else:
                report.entities_matched += 1

        # 2. Relations — injection-safe RELATE over resolved endpoint ids.
        for rel in parsed.relations:
            src_id = concept_to_entity_id.get(rel.from_id)
            tgt_id = concept_to_entity_id.get(rel.to_id)
            if not src_id or not tgt_id:
                # An endpoint entity was skipped/failed — record, never crash.
                report.dangling_links.append((rel.from_id, rel.to_id))
                continue
            try:
                created_new = await self._persistence._upsert_relation(
                    src_id=src_id,
                    tgt_id=tgt_id,
                    relation_type=rel.relation_type,
                    confidence=1.0,
                    source_id=import_source_id,
                    properties={"okf_import": True},
                )
                if created_new:
                    report.relations_created += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "OKF import: relation {a}->{b} failed: {e}",
                    a=rel.from_id, b=rel.to_id, e=exc,
                )

        # 3. Notes (idempotent on okf_key).
        for pnote in parsed.notes:
            created = await self._import_note(pnote, notebook_id)
            if created:
                report.notes_created += 1
            else:
                report.notes_matched += 1

        # 4. Sources (idempotent on okf_key).
        for psrc in parsed.sources.values():
            created = await self._import_source(psrc, notebook_id)
            if created:
                report.sources_created += 1
            else:
                report.sources_matched += 1

        # 5. Fuzzy/near-duplicate collapse against the persisted graph (K.5+K.3).
        if apply_dedup and self._dedup is not None and self._recanon is not None:
            report.dedup_merges = await self._run_dedup(notebook_id)

        logger.info(
            "OKF import: entities +{ec}/~{em} relations +{rc} notes +{nc}/~{nm} "
            "sources +{sc}/~{sm} dedup_merges={dm} skipped={sk} dangling={dl}",
            ec=report.entities_created, em=report.entities_matched,
            rc=report.relations_created,
            nc=report.notes_created, nm=report.notes_matched,
            sc=report.sources_created, sm=report.sources_matched,
            dm=report.dedup_merges, sk=len(report.skipped),
            dl=len(report.dangling_links),
        )
        return report

    # -- entity persistence --------------------------------------------------

    async def _upsert_entity(
        self, pe: ParsedEntity, import_source_id: str
    ) -> Tuple[bool, str]:
        """Upsert one parsed entity, returning ``(was_created, entity_id)``.

        Types via the shared ``_resolve_entity_type`` bridge so an OKF ``type``
        lands on the same canonical enum + rich ``primary_type`` an extraction
        would. A genuine external ``resource`` URI is pinned to ``external_ids``
        (so a re-export re-emits it); an internal urn is kept only as a property.
        """
        resolved = _resolve_entity_type(pe.okf_type)
        type_tags = list(
            dict.fromkeys([*resolved.type_tags, *pe.type_tags])
        )
        properties: Dict[str, Any] = {"okf_import": True}
        if pe.resource:
            properties["okf_resource"] = pe.resource
        external_ids = [pe.resource] if _is_external_id(pe.resource) else []

        was_created = not await self._entity_exists(
            pe.canonical_name, resolved.entity_type
        )
        entity = Entity(
            canonical_name=pe.canonical_name,
            entity_type=resolved.entity_type,
            primary_type=resolved.primary_type or pe.okf_type,
            type_tags=type_tags,
            description=pe.description,
            external_ids=external_ids,
            properties=properties,
            source_documents=[import_source_id],
            extraction_method="okf-import",
            confidence=1.0,
            embedding=[],
        )
        entity_id = await self._entity_repo.upsert_entity(entity)
        return was_created, str(entity_id)

    async def _entity_exists(self, canonical_name: str, entity_type: str) -> bool:
        """True when an entity already exists under the exact (name, type) key."""
        rows = await execute_query(
            "SELECT VALUE id FROM entity "
            "WHERE canonical_name = $name AND entity_type = $etype LIMIT 1;",
            {"name": canonical_name, "etype": entity_type},
        )
        return bool(rows)

    # -- note / source persistence ------------------------------------------

    async def _import_note(
        self, pnote: ParsedNote, notebook_id: Optional[str]
    ) -> bool:
        """Create a note idempotently. Returns whether a new note was created.

        The ``note`` table is SCHEMAFULL with no metadata bag (migration 1), so a
        re-import is de-duplicated content-addressably on ``(title, content)`` —
        an imported note is exactly its title + body, so an identical pair is the
        same note. This is the OKF-import idempotency guarantee without a schema
        change.
        """
        existing = await execute_query(
            "SELECT VALUE id FROM note "
            "WHERE title = $t AND content = $c LIMIT 1;",
            {"t": pnote.title, "c": pnote.content},
        )
        if existing:
            return False
        note = await self._note_repo.create(
            {
                "title": pnote.title,
                "content": pnote.content,
                "note_type": "human",
                # note.embedding is a strict array<float> with no DB default —
                # pass [] explicitly (never omit) so the write is accepted.
                "embedding": [],
            }
        )
        if notebook_id and getattr(note, "id", None):
            await self._note_repo.add_to_notebook(str(note.id), notebook_id)
        return True

    async def _import_source(
        self, psrc: ParsedSource, notebook_id: Optional[str]
    ) -> bool:
        """Create a source idempotently (matched on ``okf_key``). Returns created?"""
        okf_key = _okf_key("source", psrc.concept_id, psrc.resource or psrc.title)
        existing = await execute_query(
            "SELECT VALUE id FROM source WHERE metadata.okf_key = $k LIMIT 1;",
            {"k": okf_key},
        )
        if existing:
            return False
        data: Dict[str, Any] = {
            "title": psrc.title,
            "topics": psrc.topics,
            "metadata": {
                "okf_import": True,
                "okf_key": okf_key,
                "okf_concept_id": psrc.concept_id,
            },
        }
        if psrc.resource:
            data["asset"] = {"url": psrc.resource}
        source = await self._source_repo.create(data)
        if notebook_id and getattr(source, "id", None):
            await self._source_repo.add_to_notebook(str(source.id), notebook_id)
        return True

    # -- provenance anchor ---------------------------------------------------

    async def _create_import_source(
        self, parsed: ParsedBundle, notebook_id: Optional[str]
    ) -> str:
        """Create the provenance-anchor ``source`` row for this import batch.

        Every imported entity/relation records this id in ``source_documents`` so
        the whole batch is traceable to one OKF import event and distinguishable
        from natively-extracted nodes.
        """
        data = {
            "title": "OKF import",
            "topics": [],
            "metadata": {
                "okf_import": True,
                "okf_import_anchor": True,
                "entity_concepts": len(parsed.entities),
                "note_concepts": len(parsed.notes),
                "source_concepts": len(parsed.sources),
            },
        }
        source = await self._source_repo.create(data)
        source_id = str(source.id)
        if notebook_id:
            await self._source_repo.add_to_notebook(source_id, notebook_id)
        return source_id

    # -- dedup (K.5 propose → K.3 apply) ------------------------------------

    async def _run_dedup(self, notebook_id: Optional[str]) -> int:
        """Collapse auto-merge candidates against the persisted graph.

        Reuses the K.5 proposer (read-only, over-merge-guarded) and applies only
        its ``auto_merge`` band through the K.3 reviewable merge (relation
        repoint + provenance fold + soft tombstone). ``review`` candidates are
        left for a human — never auto-applied. Idempotent.
        """
        # Callers gate on both being present; assert for the type narrower.
        assert self._dedup is not None and self._recanon is not None
        try:
            candidates = await self._dedup.propose_candidates(notebook_id=notebook_id)
        except Exception as exc:  # noqa: BLE001 - dedup must not fail the import
            logger.warning("OKF import: dedup proposal failed: {e}", e=exc)
            return 0

        merges = 0
        for candidate in candidates.auto_merge:
            try:
                result = await self._recanon.apply_merge(candidate.to_merge_cluster())
                if not result.skipped:
                    merges += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "OKF import: apply_merge failed for {a}/{b}: {e}",
                    a=candidate.id_a, b=candidate.id_b, e=exc,
                )
        return merges


def _okf_key(kind: str, concept_id: str, payload: str) -> str:
    """Stable idempotency key for a note/source concept (content-addressed)."""
    basis = f"okf|{kind}|{concept_id}|{payload}".encode("utf-8")
    return hashlib.sha256(basis).hexdigest()[:32]


__all__ = [
    "OkfImportService",
    "OkfImportReport",
    "ParsedBundle",
    "ParsedEntity",
    "ParsedNote",
    "ParsedSource",
    "ParsedRelation",
    "SkippedConcept",
    "parse_okf_bundle",
]
