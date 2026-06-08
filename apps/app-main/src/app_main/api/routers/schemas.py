"""Notebook-effective-schema TTL export router (Phase B.2b).

Exposes ``GET /api/notebooks/{notebook_id}/schema.ttl`` — the canonical
download endpoint for a notebook's *effective* schema: the base ontology
plus any user-accepted extensions tracked in ``notebook_schema``.

Design notes
------------

* **Output is a file download, not JSON.** We return ``Response`` with
  ``Content-Type: text/turtle`` and ``Content-Disposition: attachment``
  so browsers offer to save the ``.ttl`` rather than render it. The
  filename uses the notebook id (sanitised) — this keeps the route
  cacheable per-notebook without needing the notebook's display name.

* **Missing notebook_schema row is *not* a 404.** B.1c populates the
  row on the first pass-1 run; before then the effective schema is
  simply the base ontology default with zero extensions. The API
  contract is "give me the effective schema", which is still defined
  in that state. Only a genuinely unknown notebook id returns 404, and
  we delegate that decision to ``NotebookService.get``.

* **Extension shape is flexible.** B.1c writes
  ``{extension_id, type_name, parent_type, properties: [...]}`` into
  ``accepted_extensions``. We tolerate missing keys defensively because
  the dict shape is intentionally schemaless on the DB side (see
  ``shared.models.notebook_schema`` for rationale).

* **type_name sanitisation for URIs.** LLM-generated extensions
  (B.1c/B.1d) may produce ``type_name`` values containing spaces or
  punctuation. rdflib's Turtle serializer rejects those because
  ``https://open-notebook.dev/ontology/My Class`` is not a valid URI.
  We CamelCase the URI fragment and preserve the original
  human-readable string in ``rdfs:label`` — the standard RDF/OWL
  convention.

* **Serialisation footprint.** Current output is buffered in memory.
  At present scale (single-notebook ontology with ~10-50 classes) this
  is ~10-30 KB and the in-memory Turtle is fine. Streaming becomes
  relevant if a notebook accumulates >100 KB of TTL (rough heuristic:
  >500 classes); revisit at that scale.

* **rdflib is a runtime dep** of ``ontology-manager`` (merged in B.2a),
  so the import is unconditional at module load time.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from loguru import logger
from pydantic import BaseModel, Field

from app_main.dependencies import get_notebook_service, get_source_repo
from app_main.services.notebook_service import NotebookService
from ontology_manager.rdf_owl_shacl import (
    ON,
    load_yaml_ontology,
)
from ontology_manager.registry import OntologyRegistry
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS
from surrealdb_service.repositories.notebook_schema import (
    NotebookSchemaRepository,
    Pass1ResultRepository,
)
from surrealdb_service.repositories.source import SourceRepository


router = APIRouter(prefix="/notebooks/{notebook_id}", tags=["schemas"])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# When a notebook has no ``notebook_schema`` row yet (B.1c hasn't run), we
# still need to pick a base ontology to export.
#
# **Divergence from ``OntologyManagerConfig.default_ontology``** (which is
# ``"general"``): scholarly is the canonical default for notebook schemas
# in the B-track corpus — it carries the classes (Article, Author,
# Cohort, PreprintServer, ...) that B.1c/B.1d's LLM pass-1 actually
# emits against. ``general.yaml`` uses a dict-of-dicts ``entity_types``
# shape that ``load_yaml_ontology`` does not currently parse, so
# switching to it would 500 every fresh-notebook download.
#
# When B.3a wires the manager config end-to-end (or when ``general.yaml``
# is normalised to the list-of-dicts shape), revisit this literal.
_DEFAULT_BASE_ONTOLOGY = "scholarly"


def _ontologies_dir() -> Path:
    """Resolve the bundled YAML ontology directory.

    Delegates to ``OntologyRegistry._find_ontology_dir`` (via the
    singleton's cached ``_ontology_dir`` attribute) so the resolution
    rules (package-relative → cwd → ``ONTOLOGY_DIR`` env → fallback)
    stay in one place. This avoids the ``parents[N]`` fragility of
    computing the repo root from this file's location.

    The ``_ontology_dir`` attribute is private by convention but stable
    across the codebase (set in ``__init__`` and never mutated). When
    ``_find_ontology_dir`` is exposed as part of a public registry
    contract in a future phase, replace this with the public helper.
    """
    return OntologyRegistry()._ontology_dir  # noqa: SLF001 — see docstring


# ---------------------------------------------------------------------------
# DI providers
# ---------------------------------------------------------------------------


def get_notebook_schema_repo() -> NotebookSchemaRepository:
    """FastAPI provider for the notebook_schema repository.

    Defined here rather than in ``app_main.dependencies`` because this is
    the only router in app-main that uses it today; will lift to the
    central dependencies module when B.3a/B.3b add the JSON browse + edit
    endpoints.
    """
    return NotebookSchemaRepository()


def get_pass1_result_repo() -> Pass1ResultRepository:
    """FastAPI provider for the pass1_results repository.

    Same locality argument as ``get_notebook_schema_repo`` — only the
    schemas router consumes pass-1 results today (B.3a's CoverageStats
    table). Lift to ``app_main.dependencies`` once a second caller
    appears.
    """
    return Pass1ResultRepository()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_CAMEL_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")


def _to_camel_case_uri_fragment(name: str) -> str:
    """Convert a human-readable type name into a CamelCase URI fragment.

    rdflib's Turtle serializer rejects URIs containing spaces or other
    URI-illegal characters. Since LLM-generated extension names
    inevitably contain such characters ("Clinical Trial Phase",
    "Author/Editor"), we transform them into CamelCase before building
    the URI. The original string is preserved separately in
    ``rdfs:label`` so human readability survives the trip.

    Examples::

        "My Class With Spaces"   → "MyClassWithSpaces"
        "preprint server"        → "PreprintServer"
        "Author/Editor"          → "AuthorEditor"
        "ARXiv paper"            → "ARXivPaper"
        "PreprintServer"         → "PreprintServer"   (no-op when valid)
        "2024 cohort"            → "_2024Cohort"      (leading-digit guard)
    """
    parts = [p for p in _CAMEL_SPLIT_RE.split(name) if p]
    if not parts:
        return ""

    # Preserve already-internally-capitalised words ("ARXiv",
    # "PreprintServer") by only upper-casing the first character if
    # it's lower-case. Avoids destroying intentional capitalisation.
    def _cap_first(s: str) -> str:
        return s[0].upper() + s[1:] if s else s

    fragment = "".join(_cap_first(p) for p in parts)

    # URI fragments cannot start with a digit if we want a valid QName
    # (Turtle's prefixed-name form uses NCName rules). Prefix with an
    # underscore in that case — still serialisable, still unique.
    if fragment and fragment[0].isdigit():
        fragment = "_" + fragment

    return fragment


def _apply_extensions(graph: Graph, extensions: List[Dict[str, Any]]) -> int:
    """Add accepted-extension classes (and their properties) to ``graph``.

    Each extension dict roughly follows::

        {
            "extension_id": "...",
            "type_name": "MyType",
            "parent_type": "ParentType",  # optional
            "properties": [               # optional
                {"name": "...", "data_type": "string"},
                ...
            ],
        }

    We add an ``owl:Class`` declaration plus an ``rdfs:subClassOf`` edge
    when ``parent_type`` is set. Properties become ``owl:DatatypeProperty``
    declarations (object-property nuance is left to B.3b's full editor).

    ``type_name`` is sanitised to a CamelCase URI fragment via
    ``_to_camel_case_uri_fragment``; the original human-readable string
    is written to ``rdfs:label``. The same transform is applied to
    ``parent_type`` so the ``rdfs:subClassOf`` URI also stays valid.

    Returns the number of new ``owl:Class`` declarations actually added.
    Defensive of missing keys: an extension dict without ``type_name`` is
    skipped with a warning rather than raising — that matches the
    FLEXIBLE storage contract on the DB side.
    """
    classes_added = 0
    for ext in extensions:
        type_name = ext.get("type_name")
        if not type_name or not isinstance(type_name, str):
            logger.warning(
                "Skipping extension without valid type_name: {!r}", ext
            )
            continue

        uri_fragment = _to_camel_case_uri_fragment(type_name)
        if not uri_fragment:
            logger.warning(
                "Skipping extension with un-sanitisable type_name: {!r}",
                type_name,
            )
            continue

        cls_uri: URIRef = ON[uri_fragment]
        graph.add((cls_uri, RDF.type, OWL.Class))
        # Preserve the original human-readable name — this is the value
        # users see in Protégé / a KG browser regardless of URI mangling.
        graph.add((cls_uri, RDFS.label, Literal(type_name)))
        classes_added += 1

        parent = ext.get("parent_type")
        if isinstance(parent, str) and parent:
            parent_fragment = _to_camel_case_uri_fragment(parent)
            if parent_fragment:
                graph.add((cls_uri, RDFS.subClassOf, ON[parent_fragment]))

        description = ext.get("description")
        if isinstance(description, str) and description:
            graph.add((cls_uri, RDFS.comment, Literal(description)))

        for prop in ext.get("properties", []) or []:
            if not isinstance(prop, dict):
                continue
            pname = prop.get("name")
            if not pname or not isinstance(pname, str):
                continue
            pfragment = _to_camel_case_uri_fragment(pname)
            if not pfragment:
                continue
            # Property URIs use lower-camel by convention; lower-case the
            # very first character.
            pfragment = pfragment[0].lower() + pfragment[1:]
            prop_uri = ON[pfragment]
            graph.add((prop_uri, RDF.type, OWL.DatatypeProperty))
            graph.add((prop_uri, RDFS.domain, cls_uri))
            graph.add((prop_uri, RDFS.label, Literal(pname)))

    return classes_added


# Characters that either break the Content-Disposition header itself
# (CR/LF, embedded quotes) or trip OS save dialogs (path separators,
# colon on older Windows clients).
_FILENAME_UNSAFE_RE = re.compile(r'[:/\\\r\n\t\x00"\']')


def _safe_filename(notebook_id: str) -> str:
    """Build a filename safe for ``Content-Disposition`` headers.

    Replaces characters that break the HTTP header (CR/LF, single and
    double quotes) or trip OS save dialogs (path separators, colon)
    with underscores, then appends ``.ttl``.

    Example: ``notebook:abc-123`` → ``notebook_abc-123.ttl``.

    Not a general-purpose filesystem sanitiser — the strip-list is
    scoped to the characters that matter for HTTP attachment downloads
    and common consumer OSes (Windows / macOS / Linux file pickers).
    """
    return _FILENAME_UNSAFE_RE.sub("_", notebook_id) + ".ttl"


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.get(
    "/schema.ttl",
    response_class=Response,
    responses={
        200: {
            "content": {"text/turtle": {}},
            "description": "Turtle serialisation of the notebook's effective schema.",
        },
        404: {"description": "Notebook not found."},
    },
)
async def export_notebook_schema_ttl(
    notebook_id: str,
    notebook_service: NotebookService = Depends(get_notebook_service),
    schema_repo: NotebookSchemaRepository = Depends(get_notebook_schema_repo),
) -> Response:
    """Export the notebook's effective schema as Turtle.

    The "effective schema" is the base ontology referenced by
    ``notebook_schema.base_ontology`` plus every entry in
    ``notebook_schema.accepted_extensions`` materialised as
    ``owl:Class`` declarations.

    Returns 404 when ``notebook_id`` doesn't resolve to a real notebook.
    Returns 200 with the bare base ontology if the notebook exists but
    hasn't been pass-1-processed yet (no ``notebook_schema`` row).
    """
    notebook = await notebook_service.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    notebook_schema = await schema_repo.get_by_notebook(notebook_id)

    if notebook_schema is None:
        base_ontology = _DEFAULT_BASE_ONTOLOGY
        accepted_extensions: List[Dict[str, Any]] = []
    else:
        base_ontology = notebook_schema.base_ontology or _DEFAULT_BASE_ONTOLOGY
        accepted_extensions = notebook_schema.accepted_extensions or []

    yaml_path = _ontologies_dir() / f"{base_ontology}.yaml"
    if not yaml_path.exists():
        logger.error(
            "Base ontology YAML missing for notebook {}: {} not found",
            notebook_id,
            yaml_path,
        )
        # 500 — this is a server-side misconfiguration, not a client error.
        raise HTTPException(
            status_code=500,
            detail=(
                f"Base ontology '{base_ontology}' YAML not found on server. "
                "Check the ONTOLOGY_DIR environment variable."
            ),
        )

    graph: Graph = load_yaml_ontology(yaml_path)
    classes_added = _apply_extensions(graph, accepted_extensions)
    logger.info(
        "Exporting schema TTL for {} (base={}, extensions_added={})",
        notebook_id,
        base_ontology,
        classes_added,
    )

    turtle = graph.serialize(format="turtle")

    filename = _safe_filename(notebook_id)
    return Response(
        content=turtle,
        media_type="text/turtle",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


# ---------------------------------------------------------------------------
# JSON browse endpoints (Phase B.3a)
# ---------------------------------------------------------------------------
#
# These two GETs back the Schema-tab UI. They are intentionally
# read-only — the four edit ops (rename/merge/split/delete) and the
# Accept/Reject mutations on pending extensions land in B.3b, which
# extends this same router.
#
# Design notes
# ------------
#
# * **Response shape is decoupled from the DB model.** The frontend
#   consumes a stable JSON contract regardless of how the FLEXIBLE
#   storage on ``notebook_schema`` evolves. New keys can be added to
#   the model without breaking the client, and obsolete keys can be
#   dropped from the response without forcing a DB migration.
#
# * **Empty-state behaviour mirrors the TTL endpoint.** A notebook
#   that has never been pass-1-processed (no ``notebook_schema`` row)
#   gets sensible defaults — the configured ``_DEFAULT_BASE_ONTOLOGY``
#   and empty extension lists. That keeps the Schema tab usable on a
#   freshly-created notebook.
#
# * **pass1_results is append-only** — we surface the full list ordered
#   newest-first. The frontend deduplicates by ``source_id`` when it
#   only wants the latest result per source (the table shows one row
#   per source). Doing the dedup server-side would couple this endpoint
#   to a presentation concern and lose the audit history.


class _PropertyDef(BaseModel):
    name: str
    description: Optional[str] = None
    data_type: str
    required: bool = False


class _EntityTypeNode(BaseModel):
    """View-model row for one base-ontology entity type.

    Mirrors the fields the SchemaBrowser tree actually renders — name,
    parent (for the tree hierarchy), tooltip text, and a flat list of
    properties for the side panel. Aliases / extraction-hints are
    omitted in B.3a; B.3b's editor surfaces them.
    """

    name: str
    description: Optional[str] = None
    parent_type: Optional[str] = None
    properties: List[_PropertyDef] = Field(default_factory=list)


class _ExtensionView(BaseModel):
    """View-model row for an accepted or pending extension.

    The DB stores extensions as FLEXIBLE dicts. This model normalises
    that bag into a predictable shape for the frontend, defending
    against missing keys at the API boundary so the React side never
    has to ``ext?.type_name ?? '(unknown)'``-style guard.
    """

    extension_id: Optional[str] = None
    type_name: str
    parent_type: Optional[str] = None
    description: Optional[str] = None
    rationale: Optional[str] = None
    properties: List[_PropertyDef] = Field(default_factory=list)


class NotebookSchemaResponse(BaseModel):
    """JSON contract for ``GET /api/notebooks/{id}/schema``.

    Carries everything the Schema tab needs in a single payload so the
    page renders without a fan-out of additional calls. The
    ``base_ontology_types`` list is sourced from the YAML registry — it
    is the immutable backbone; ``accepted_extensions`` and
    ``pending_extensions`` come from the DB and evolve over time.
    """

    notebook_id: str
    base_ontology: str
    base_ontology_types: List[_EntityTypeNode] = Field(default_factory=list)
    accepted_extensions: List[_ExtensionView] = Field(default_factory=list)
    pending_extensions: List[_ExtensionView] = Field(default_factory=list)
    coverage_pct: float = 0.0
    review_required: bool = False
    soft_nudge_dismissed: bool = False


class Pass1ResultView(BaseModel):
    """View-model row for ``GET /api/notebooks/{id}/pass1_results``.

    Trimmed projection of :class:`shared.models.Pass1Result` plus the
    parent source's title — the frontend's CoverageStatsTable shows
    ``[source_title | coverage_pct | uncovered_concept_count]``, so we
    enrich here rather than asking the client to do a second round-trip
    per row. ``source_title`` is ``None`` for orphaned rows where the
    source has been deleted; the UI displays ``"(deleted source)"`` in
    that case.
    """

    id: str
    source_id: str
    source_title: Optional[str] = None
    schema_attempted: str
    detected_schema: str
    confidence_in_choice: float = 0.0
    coverage_pct: float = 0.0
    uncovered_concept_count: int = 0
    created: Optional[str] = None


def _normalise_extension(ext: Dict[str, Any]) -> _ExtensionView:
    """Coerce a FLEXIBLE extension dict into a typed view-model row.

    Defensive of missing keys: an extension without ``type_name`` is
    surfaced as ``"(unknown)"`` rather than dropped silently — the
    frontend should still render a row so the user can reject the
    invalid extension via the (B.3b) edit ops. Properties is a list
    of dicts on the DB side; we accept either ``{name, data_type,
    description, required}`` or just ``{name}``.
    """
    props_raw = ext.get("properties") or []
    props: List[_PropertyDef] = []
    for p in props_raw:
        if not isinstance(p, dict):
            continue
        name = p.get("name")
        if not isinstance(name, str) or not name:
            continue
        props.append(
            _PropertyDef(
                name=name,
                description=p.get("description") if isinstance(p.get("description"), str) else None,
                data_type=p.get("data_type") if isinstance(p.get("data_type"), str) else "string",
                required=bool(p.get("required", False)),
            )
        )

    type_name = ext.get("type_name")
    if not isinstance(type_name, str) or not type_name:
        type_name = "(unknown)"

    return _ExtensionView(
        extension_id=ext.get("extension_id") if isinstance(ext.get("extension_id"), str) else None,
        type_name=type_name,
        parent_type=ext.get("parent_type") if isinstance(ext.get("parent_type"), str) else None,
        description=ext.get("description") if isinstance(ext.get("description"), str) else None,
        rationale=ext.get("rationale") if isinstance(ext.get("rationale"), str) else None,
        properties=props,
    )


def _load_base_ontology_types(base_ontology: str) -> List[_EntityTypeNode]:
    """Parse a YAML ontology and return its entity types as view-model rows.

    Reuses the same YAML resolution as the TTL endpoint
    (``_ontologies_dir``). The full :class:`Ontology` model carries
    relationships, concepts, etc. — B.3a only needs entity types for
    the tree view, so we project to the slimmer ``_EntityTypeNode``.

    Returns an empty list (and logs an error) if the YAML is missing
    — this matches the existing TTL endpoint's behaviour of returning
    a 500 in the same circumstance, but for the JSON endpoint we
    degrade gracefully: the Schema tab is still useful for showing
    extensions and coverage even if the base ontology is misconfigured.
    """
    yaml_path = _ontologies_dir() / f"{base_ontology}.yaml"
    if not yaml_path.exists():
        logger.error(
            "Base ontology YAML missing: {} not found (asked for {!r})",
            yaml_path,
            base_ontology,
        )
        return []

    try:
        # Local import to keep the module import-cost flat — Ontology
        # pulls in the full pydantic schema tree.
        from ontology_manager.schema import Ontology

        ontology = Ontology.from_yaml(yaml_path)
    except Exception as e:
        logger.exception(
            "Failed to parse base ontology YAML {}: {}", yaml_path, e
        )
        return []

    nodes: List[_EntityTypeNode] = []
    for name, et in ontology.entity_types.items():
        props = [
            _PropertyDef(
                name=p.name,
                description=p.description,
                data_type=str(p.data_type.value) if hasattr(p.data_type, "value") else str(p.data_type),
                required=bool(p.validation.required) if p.validation else False,
            )
            for p in et.properties
        ]
        nodes.append(
            _EntityTypeNode(
                name=et.name or name,
                description=et.description,
                parent_type=et.parent_type,
                properties=props,
            )
        )
    return nodes


@router.get(
    "/schema",
    response_model=NotebookSchemaResponse,
    responses={404: {"description": "Notebook not found."}},
)
async def get_notebook_schema_json(
    notebook_id: str,
    notebook_service: NotebookService = Depends(get_notebook_service),
    schema_repo: NotebookSchemaRepository = Depends(get_notebook_schema_repo),
) -> NotebookSchemaResponse:
    """Browse the notebook's effective schema as JSON.

    Returns the base ontology's entity types, the currently-accepted
    extensions, the pending extensions awaiting user decision, and the
    per-notebook review/soft-nudge toggles. The frontend renders this
    as the Schema-tab tree + side panels.

    Returns 404 if ``notebook_id`` does not resolve to a real notebook.
    Returns 200 with the bare base-ontology defaults if the notebook
    exists but has not been pass-1-processed yet — same contract as
    the TTL endpoint, so the Schema tab is usable on day 1.
    """
    notebook = await notebook_service.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    notebook_schema = await schema_repo.get_by_notebook(notebook_id)

    if notebook_schema is None:
        base_ontology = _DEFAULT_BASE_ONTOLOGY
        accepted_raw: List[Dict[str, Any]] = []
        pending_raw: List[Dict[str, Any]] = []
        coverage_pct = 0.0
        review_required = False
        soft_nudge_dismissed = False
    else:
        base_ontology = notebook_schema.base_ontology or _DEFAULT_BASE_ONTOLOGY
        accepted_raw = notebook_schema.accepted_extensions or []
        pending_raw = notebook_schema.pending_extensions or []
        coverage_pct = float(notebook_schema.coverage_pct)
        review_required = bool(notebook_schema.review_required)
        soft_nudge_dismissed = bool(notebook_schema.soft_nudge_dismissed)

    return NotebookSchemaResponse(
        notebook_id=notebook_id,
        base_ontology=base_ontology,
        base_ontology_types=_load_base_ontology_types(base_ontology),
        accepted_extensions=[_normalise_extension(e) for e in accepted_raw],
        pending_extensions=[_normalise_extension(e) for e in pending_raw],
        coverage_pct=coverage_pct,
        review_required=review_required,
        soft_nudge_dismissed=soft_nudge_dismissed,
    )


@router.get(
    "/pass1_results",
    response_model=List[Pass1ResultView],
    responses={404: {"description": "Notebook not found."}},
)
async def list_notebook_pass1_results(
    notebook_id: str,
    notebook_service: NotebookService = Depends(get_notebook_service),
    pass1_repo: Pass1ResultRepository = Depends(get_pass1_result_repo),
    source_repo: SourceRepository = Depends(get_source_repo),
) -> List[Pass1ResultView]:
    """List per-source pass-1 results for the notebook, newest-first.

    Powers the CoverageStatsTable on the Schema tab. Each row carries
    the source title (looked up via ``SourceRepository.get``) so the
    UI can render ``[source_title | coverage_pct | uncovered_count]``
    without a second round-trip per row.

    Returns 404 if ``notebook_id`` doesn't resolve. Returns 200 with
    an empty list if the notebook has no pass-1 history yet.
    """
    notebook = await notebook_service.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    rows = await pass1_repo.list_by_notebook(notebook_id)

    # Resolve source titles in a single pass. We cache by source_id
    # because a source can appear multiple times in the list (reprocess
    # ⇒ multiple Pass1Result rows per source).
    title_cache: Dict[str, Optional[str]] = {}

    async def _title_for(source_id: str) -> Optional[str]:
        if source_id in title_cache:
            return title_cache[source_id]
        try:
            src = await source_repo.get(source_id)
            title = src.title if src else None
        except Exception as e:
            logger.warning(
                "Failed to resolve source title for {}: {}", source_id, e
            )
            title = None
        title_cache[source_id] = title
        return title

    out: List[Pass1ResultView] = []
    for row in rows:
        src_title = await _title_for(row.source)
        out.append(
            Pass1ResultView(
                id=row.id or "",
                source_id=row.source,
                source_title=src_title,
                schema_attempted=row.schema_attempted,
                detected_schema=row.detected_schema,
                confidence_in_choice=float(row.confidence_in_choice),
                coverage_pct=float(row.coverage_pct),
                uncovered_concept_count=len(row.uncovered_concepts or []),
                created=row.created.isoformat() if row.created else None,
            )
        )
    return out
