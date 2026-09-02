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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from loguru import logger
from pydantic import BaseModel, Field

from app_main.dependencies import (
    get_notebook_schema_repo,
    get_notebook_service,
    get_pass1_result_repo,
    get_reextract_service,
    get_schema_edit_service,
    get_source_repo,
    get_type_placement_service,
)
from app_main.services.notebook_service import NotebookService
from app_main.services.reextract_service import ReextractService
from app_main.services.schema_edit_service import (
    NotebookSchemaNotFoundError,
    SchemaEditService,
    UnknownExtensionError,
)
from shared.models import NotebookSchema
from ontology_manager.rdf_owl_shacl import (
    ON,
    load_yaml_ontology,
)
from ontology_manager.registry import OntologyRegistry
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS
from shared.models.notebook_schema import NotebookSchema
from shared.types.enums import JobStatus, JobType
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
#
# ``get_notebook_schema_repo`` and ``get_pass1_result_repo`` live in
# ``app_main.dependencies`` — they were originally defined locally
# (single-consumer rule) but B.3b/B.3c will also import them, so the
# central location is now the right home. See those provider docstrings
# for the lift rationale.


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


# N.4d.3: the discriminator on an ``accepted_extensions`` entry that records a
# re-parent rather than a new type. Defined here (not imported from the service)
# to keep the router free of a service-layer import for one string.
_REPARENT_OP = "reparent"


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

    N.4d.3 — a ``reparent`` entry REPLACES the class's declared parent instead
    of adding a second one. Without that the exported graph would give a
    re-parented type two ``rdfs:subClassOf`` edges, which is a different
    statement from the one the curator made: the base's parent and the new one
    would both be true. It is not counted as a class added either — the class
    already existed. The class URI is safe to target because the base emits
    ``ON[name]`` while this function emits ``ON[camel_case(name)]``, and no
    shipped ontology has a type name where those differ (measured: zero names
    containing a space or a slash across all eleven).

    Returns the number of new ``owl:Class`` declarations actually added.
    Defensive of missing keys: an extension dict without ``type_name`` is
    skipped with a warning rather than raising — that matches the
    FLEXIBLE storage contract on the DB side.
    """
    classes_added = 0
    for ext in extensions:
        # B.3c: skip resume sentinels — they only exist to satisfy the
        # review-gate predicate (see ``_RESUME_SENTINEL_TYPE_NAME``) and
        # carry no schema content.
        if ext.get("is_resume_sentinel") is True:
            continue
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

        if ext.get("op") == _REPARENT_OP:
            if (cls_uri, RDF.type, OWL.Class) not in graph:
                # A re-parent moves an EXISTING class. Naming one the base graph
                # does not declare would otherwise mint a phantom class here —
                # an assertion the exported ontology cannot support, from a row
                # the projection would refuse for the same reason.
                logger.warning(
                    "Skipping re-parent of a class absent from the base "
                    "ontology: {!r}",
                    type_name,
                )
                continue
            # Drop the parent it was declared with so the new one REPLACES it
            # rather than joining it.
            graph.remove((cls_uri, RDFS.subClassOf, None))
        else:
            classes_added += 1

        graph.add((cls_uri, RDF.type, OWL.Class))
        # Preserve the original human-readable name — this is the value
        # users see in Protégé / a KG browser regardless of URI mangling.
        graph.add((cls_uri, RDFS.label, Literal(type_name)))

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


class _PlacementView(BaseModel):
    """Where an accepted type sits, and what a judge thinks belongs under it.

    A REPORT, never an applied change: ``selected`` names the siblings the judge
    would move under this type, and the curator applies them (or does not) via
    ``POST /schema/reparent``.

    ``candidates`` and ``selected`` are both present because an empty
    ``selected`` over five candidates reads nothing like an empty ``selected``
    over zero. ``judge_status`` says which of FOUR states produced it —
    ``not_asked`` (no model wired, or no candidates), ``unavailable`` (the model
    could not be reached), ``refused`` (a reply came back and the parser would
    not use it), ``decided`` (the judge answered; an empty ``selected`` here is
    its decision). ``judged`` is ``judge_status == "decided"``, kept because
    three of the four states carry an empty ``selected`` and a client must not
    read emptiness as a verdict.

    ``vocabulary`` names the schemas the placement was computed against: the
    notebook-level forced set. See D-N4-13 — a verdict CAN differ from the one a
    given document's applied set would give (adding a schema turns ``PLACED``
    into ``DUPLICATE``), which is tolerable only because this is advisory: it
    writes nothing, and a re-parent is applied by an explicit
    ``POST /schema/reparent``.
    """

    type_name: str
    verdict: str
    reason_code: str
    evidence: str
    parent: Optional[str] = None
    duplicate_of: Optional[str] = None
    candidates: List[str] = Field(default_factory=list)
    selected: List[str] = Field(default_factory=list)
    judged: bool = False
    judge_status: str = "not_asked"
    judge_evidence: str = ""
    vocabulary: List[str] = Field(default_factory=list)


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
    # N.4d.3 — populated only by the accept endpoint; null everywhere else.
    placement: Optional[_PlacementView] = None
    pending_extensions: List[_ExtensionView] = Field(default_factory=list)
    excluded_types: List[str] = Field(default_factory=list)
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
        # ``data_type`` falls back to ``"string"`` when the LLM-emitted
        # extension dict omits it (older Pass1 payloads predating the
        # B.1d schema, or hand-crafted fixtures). String is the safest
        # default because it round-trips losslessly to TTL
        # (``xsd:string``) and the frontend treats unknown data-types
        # as opaque labels. B.3b's editor surfaces the value so users
        # can correct it if the inference is wrong.
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


def _reparented_types(extensions: List[Dict[str, Any]]) -> Dict[str, str]:
    """``{type_name: new_parent}`` from the recorded re-parents, last one wins.

    Last-wins matches what ``ontology_manager.schema_projection`` does with the
    same rows, so the browser shows the parent extraction will actually use. A
    row missing either field is skipped rather than rendered half-applied.
    """
    moves: Dict[str, str] = {}
    for ext in extensions:
        if ext.get("op") != _REPARENT_OP:
            continue
        name = ext.get("type_name")
        parent = ext.get("new_parent")
        if isinstance(name, str) and name and isinstance(parent, str) and parent:
            moves[name] = parent
    return moves


def _apply_reparents_to_views(
    types: List[_EntityTypeNode], moves: Dict[str, str]
) -> List[_EntityTypeNode]:
    if not moves:
        return types
    return [
        node.model_copy(update={"parent_type": moves[node.name]})
        if node.name in moves
        else node
        for node in types
    ]


def _apply_reparents_to_extensions(
    extensions: List[Dict[str, Any]], moves: Dict[str, str]
) -> List[Dict[str, Any]]:
    """A re-parent can also move a type the curator accepted as an extension."""
    if not moves:
        return extensions
    out: List[Dict[str, Any]] = []
    for ext in extensions:
        name = ext.get("type_name")
        if isinstance(name, str) and name in moves:
            ext = {**ext, "parent_type": moves[name]}
        out.append(ext)
    return out


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
        excluded_types: List[str] = []
        coverage_pct = 0.0
        review_required = False
        soft_nudge_dismissed = False
    else:
        base_ontology = notebook_schema.base_ontology or _DEFAULT_BASE_ONTOLOGY
        accepted_raw = notebook_schema.accepted_extensions or []
        pending_raw = notebook_schema.pending_extensions or []
        excluded_types = list(notebook_schema.excluded_types or [])
        coverage_pct = float(notebook_schema.coverage_pct)
        review_required = bool(notebook_schema.review_required)
        soft_nudge_dismissed = bool(notebook_schema.soft_nudge_dismissed)

    # B.3c: hide resume sentinels from the SchemaBrowser — they are
    # an implementation detail of the pause/resume flow.
    #
    # N.4d.3: re-parents are hidden from this list too, and APPLIED to the row
    # of the type they move instead. They are not extensions: rendered as one, a
    # re-parented base type appears TWICE — once under base types with the parent
    # it was declared with, once as an "extension" with the new one — and two
    # moves of the same type collide on the browser's `ext:${type_name}` key,
    # since a re-parent carries no `extension_id`. This endpoint reports the
    # notebook's effective schema, so the effective parent is what belongs on the
    # row.
    visible_accepted = [
        e
        for e in accepted_raw
        if e.get("is_resume_sentinel") is not True
        and e.get("op") != _REPARENT_OP
    ]
    reparented = _reparented_types(accepted_raw)
    return NotebookSchemaResponse(
        notebook_id=notebook_id,
        base_ontology=base_ontology,
        base_ontology_types=_apply_reparents_to_views(
            _load_base_ontology_types(base_ontology), reparented
        ),
        accepted_extensions=[
            _normalise_extension(e)
            for e in _apply_reparents_to_extensions(visible_accepted, reparented)
        ],
        pending_extensions=[_normalise_extension(e) for e in pending_raw],
        excluded_types=excluded_types,
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


# ---------------------------------------------------------------------------
# Edit-ops endpoints (Phase B.3b)
# ---------------------------------------------------------------------------
#
# Six mutations that drive the Schema-tab edit UI. Each delegates to
# :class:`SchemaEditService`, which is responsible for state diffing +
# idempotency + emitting one ``notebook_event{type:"schema_changed"}``
# row per successful op.
#
# The handler layer is intentionally thin: validate inputs, translate
# service-level exceptions to HTTP status codes, serialise the updated
# ``NotebookSchema`` into the existing ``NotebookSchemaResponse`` so the
# client can refresh in place.
#
# Design notes
# ------------
#
# * **POST for state-changing ops** (except DELETE for ``delete_type``).
#   Even rename / merge / split create new "extension" entries in
#   the per-notebook schema row — they aren't strictly REST-idempotent
#   on the URL, so POST is the cleaner verb.
#
# * **Two flavours of 404**:
#   - "notebook does not exist" → caught by the notebook_service.get
#     guard at the top of every handler.
#   - "notebook exists but no schema row" / "no such extension" →
#     bubbled up by the service via NotebookSchemaNotFoundError /
#     UnknownExtensionError.
#   We deliberately surface both as 404 so the client behaviour is
#   uniform; the detail message distinguishes them.
#
# * **Response shape == GET /schema response shape**. The frontend
#   uses the same NotebookSchemaResponse for read + write so it can
#   replace the React Query cache directly. We re-load + re-render the
#   base ontology types here on each write (B.3c may switch to a
#   slimmer "delta" response if this becomes a perf concern; the YAML
#   read is cheap at current scale).


class RenameTypeRequest(BaseModel):
    """Payload for ``POST /api/notebooks/{id}/schema/rename``.

    Both names are required and non-empty; the service short-circuits
    when ``old_name == new_name`` so the client doesn't have to guard.
    """

    old_name: str = Field(min_length=1)
    new_name: str = Field(min_length=1)


class MergeTypesRequest(BaseModel):
    """Payload for ``POST /api/notebooks/{id}/schema/merge``.

    ``type_names`` must contain at least two distinct entries (service
    raises ValueError otherwise → translated to 422).
    """

    type_names: List[str] = Field(min_length=2)
    merged_name: str = Field(min_length=1)


class ReparentTypeRequest(BaseModel):
    """Payload for ``POST /api/notebooks/{id}/schema/reparent`` (N.4d.3).

    ``type_names`` are the types moving; ``new_parent`` is where they land. The
    service raises ValueError — translated to 422 — for an empty list, a blank
    parent, or a type moved under itself.
    """

    type_names: List[str] = Field(min_length=1)
    new_parent: str = Field(min_length=1)


class SplitTypeRequest(BaseModel):
    """Payload for ``POST /api/notebooks/{id}/schema/split``.

    ``criterion`` is a free-form hint passed to Pass 2 prompts so the
    LLM can disambiguate the source type (e.g. "by year published").
    """

    type_name: str = Field(min_length=1)
    into: List[str] = Field(min_length=2)
    criterion: str = Field(min_length=1)


def _schema_to_response(
    notebook_id: str, schema: NotebookSchema
) -> NotebookSchemaResponse:
    """Render an updated NotebookSchema row into the public response shape.

    The base ontology types (loaded from YAML) are re-projected here so
    the response is self-contained — the client refreshes its cached
    view from a single POST/DELETE.
    """
    base_ontology = schema.base_ontology or _DEFAULT_BASE_ONTOLOGY
    accepted_raw = schema.accepted_extensions or []
    # N.4d.3: the same effective-schema rendering the GET does. Diverging here
    # would let a client that refreshes from a POST see a re-parented type twice
    # while a client that refreshes from the GET sees it once.
    reparented = _reparented_types(accepted_raw)
    return NotebookSchemaResponse(
        notebook_id=notebook_id,
        base_ontology=base_ontology,
        base_ontology_types=_apply_reparents_to_views(
            _load_base_ontology_types(base_ontology), reparented
        ),
        accepted_extensions=[
            _normalise_extension(e)
            for e in _apply_reparents_to_extensions(
                [e for e in accepted_raw if e.get("op") != _REPARENT_OP], reparented
            )
        ],
        pending_extensions=[
            _normalise_extension(e) for e in (schema.pending_extensions or [])
        ],
        excluded_types=list(schema.excluded_types or []),
        coverage_pct=float(schema.coverage_pct),
        review_required=bool(schema.review_required),
        soft_nudge_dismissed=bool(schema.soft_nudge_dismissed),
    )


async def _placement_for_accepted(
    placement_service, schema: NotebookSchema, type_name: str
) -> Optional[_PlacementView]:
    """Report where the just-accepted type sits. Advisory — never fails the accept.

    The declared parent comes from the accepted entry itself, because
    ``place_proposed_type`` VALIDATES a proposer's claim rather than inventing
    one. An entry with no ``parent_type`` therefore yields an honest
    ``UNPARENTED`` verdict instead of a guessed placement.
    """
    entry = next(
        (
            e
            for e in (schema.accepted_extensions or [])
            if e.get("type_name") == type_name and e.get("op") is None
        ),
        None,
    )
    if entry is None:
        return None
    try:
        report = await placement_service.placement_for(
            schema,
            type_name,
            entry.get("parent_type"),
            str(entry.get("description") or entry.get("rationale") or ""),
        )
    except Exception as e:
        logger.warning(
            "placement unavailable for {!r} in {}: {}", type_name, schema.notebook, e
        )
        return None
    return _PlacementView(
        type_name=report.type_name,
        verdict=report.verdict,
        reason_code=report.reason_code,
        evidence=report.evidence,
        parent=report.parent,
        duplicate_of=report.duplicate_of,
        candidates=list(report.candidates),
        selected=list(report.selected),
        judged=report.judged,
        judge_status=report.judge_status,
        judge_evidence=report.judge_evidence,
        vocabulary=list(report.vocabulary),
    )


async def _ensure_notebook_exists(
    notebook_id: str, notebook_service: NotebookService
) -> None:
    """Common 404 guard — every edit-ops handler runs this first."""
    notebook = await notebook_service.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")


# Phase B.3c — soft-nudge UI + per-notebook pause toggle
# ---------------------------------------------------------------------------
#
# The B.1e Pass-1 orchestrator emits ``notebook_event`` rows of type
# ``extension_suggested`` (coverage 80-95%) and ``schema_mismatch``
# (coverage <80%) — both are surface-able as a dismissible banner on
# the notebook workspace.
#
# B.1f introduces a per-notebook ``review_required`` toggle. When set,
# the entity-extraction service raises :class:`SchemaReviewPendingError`
# (which subclasses :class:`JobPausedForReviewError`) and the worker
# parks the job in :class:`JobStatus.PAUSED_FOR_REVIEW`. The resume
# flow needs to lift that gate.
#
# The review-gate predicate in
# ``EntityExtractionService._run_multi_schema`` is::
#
#     review_required is True AND accepted_extensions is empty
#
# So "resume with current schema" can satisfy the gate by appending a
# sentinel entry to ``accepted_extensions``. We picked option (a) from
# the B.3c plan (sentinel) over per-source approval tracking because:
#
#  1. It re-uses the existing field — no new migration, no new repo.
#  2. The sentinel is filtered out by the TTL export defensively
#     (``type_name`` validation is already there) — see
#     ``_apply_extensions``: a sentinel with ``type_name`` starting with
#     ``_`` parses to an empty fragment after the leading-digit guard;
#     the TTL exporter additionally skips it explicitly below.
#  3. ``last_modified_by`` (``last_modified_at`` via ``upsert``) records
#     the resume action for the audit log.
#
# Documented sentinel shape::
#
#     {
#       "type_name": "_resumed_without_extensions",
#       "is_resume_sentinel": True,
#       "created_at": "<isoformat utc>",
#       "source_id": "<source record id triggering the resume, optional>",
#     }
#
# Future B.3b/B.3d edit-ops should treat ``is_resume_sentinel=True``
# entries as "do not show in the SchemaBrowser tree". As of B.3c
# attempt-2 there are now FOUR sentinel filter sites (defence-in-depth):
#
#  1. ``get_notebook_schema_json`` here (this file ~L633) — strips
#     sentinels from the JSON contract so they never reach the frontend.
#  2. ``_apply_extensions`` here (this file ~L214) — strips sentinels
#     from the TTL exporter so they never enter downloaded ontologies.
#  3. ``EntityExtractionService._run_multi_schema`` — strips sentinels
#     before forwarding ``accepted_extensions_by_schema`` to the
#     workflow, so Pass-2 never sees them.
#  4. ``ontology_extraction.prompts.pass2._format_accepted_extensions``
#     — strips sentinels at the prompt-render seam as a final guard
#     against any consumer that forgets the upstream filter.
#  4b. ``ontology_manager.schema_projection.project_accepted_edits`` (N.4d.3)
#     — strips sentinels before projecting the notebook's accepted edits onto
#     its ontologies, so the marker never becomes a defined entity type.
#  5. ``SchemaBrowser.tsx`` — frontend belt-and-braces filter using
#     ``is_resume_sentinel === true`` OR ``type_name.startsWith('_')``.
#
# If you add a new ``accepted_extensions`` consumer, add the sentinel
# filter to that consumer too — the marker is infrastructure, not
# ontology content.

_RESUME_SENTINEL_TYPE_NAME = "_resumed_without_extensions"


class ReviewRequiredRequest(BaseModel):
    """Body for ``POST /api/notebooks/{id}/schema/review_required``."""

    enabled: bool


class ReviewRequiredResponse(BaseModel):
    """Echo response confirming the new toggle state."""

    notebook_id: str
    review_required: bool


class DismissNudgeResponse(BaseModel):
    """Echo response confirming the soft-nudge has been dismissed."""

    notebook_id: str
    soft_nudge_dismissed: bool


class ResumeExtractionResponse(BaseModel):
    """Result of resuming extraction for a paused notebook.

    ``resumed_count`` is the number of paused jobs that were transitioned
    back to ``QUEUED``. ``sentinel_added`` is ``True`` when the resume
    flow appended a sentinel entry to ``accepted_extensions`` to satisfy
    the review-gate predicate (only happens when the gate was active).
    """

    notebook_id: str
    resumed_count: int
    sentinel_added: bool


class PausedExtractionStatus(BaseModel):
    """Whether any extraction job is paused for this notebook.

    Drives the ``ExtractionPausedBanner`` — the UI polls this with a
    cheap GET and only renders the banner when ``paused_count > 0``.
    """

    notebook_id: str
    paused_count: int
    paused_source_ids: List[str] = Field(default_factory=list)


async def _ensure_schema_row(
    schema_repo: NotebookSchemaRepository,
    notebook_id: str,
) -> NotebookSchema:
    """Fetch the notebook_schema row, creating a default one if absent.

    B.1c populates the row on first pass-1 run, but the soft-nudge
    toggle endpoints can fire before that — e.g. the user opens a
    fresh notebook, enables ``review_required`` proactively, and only
    then drops a source onto the upload area. We materialise the row
    eagerly so the toggle persists across restarts.
    """
    existing = await schema_repo.get_by_notebook(notebook_id)
    if existing is not None:
        return existing
    # Build a sensible default. ``base_ontology`` is whatever the TTL
    # endpoint also defaults to so the two stay in lock-step.
    return NotebookSchema(
        notebook=notebook_id,
        base_ontology=_DEFAULT_BASE_ONTOLOGY,
    )


@router.post(
    "/schema/review_required",
    response_model=ReviewRequiredResponse,
    responses={404: {"description": "Notebook not found."}},
)
async def set_review_required(
    notebook_id: str,
    body: ReviewRequiredRequest,
    notebook_service: NotebookService = Depends(get_notebook_service),
    schema_repo: NotebookSchemaRepository = Depends(get_notebook_schema_repo),
) -> ReviewRequiredResponse:
    """Toggle the per-notebook ``review_required`` flag (B.3c).

    When set to ``True``, the next entity-extraction run for any source
    in this notebook will raise :class:`SchemaReviewPendingError` at the
    Pass-1 boundary if no extensions have been accepted yet — the worker
    parks the job in ``PAUSED_FOR_REVIEW`` and the UI surfaces the
    :class:`ExtractionPausedBanner`.

    Idempotent: setting the same value twice is a no-op (but still
    bumps ``last_modified_at`` via ``upsert``).
    """
    notebook = await notebook_service.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    schema = await _ensure_schema_row(schema_repo, notebook_id)
    schema.review_required = bool(body.enabled)
    await schema_repo.upsert(schema)
    logger.info(
        "Set review_required={} for notebook {}", schema.review_required, notebook_id
    )
    return ReviewRequiredResponse(
        notebook_id=notebook_id,
        review_required=schema.review_required,
    )


@router.post(
    "/schema/extensions/{type_name}/accept",
    response_model=NotebookSchemaResponse,
    responses={
        404: {"description": "Notebook, schema row or extension not found."},
    },
)
async def accept_pending_extension(
    notebook_id: str,
    type_name: str,
    notebook_service: NotebookService = Depends(get_notebook_service),
    edit_service: SchemaEditService = Depends(get_schema_edit_service),
    placement_service=Depends(get_type_placement_service),
) -> NotebookSchemaResponse:
    """Move the matching pending extension into ``accepted_extensions``.

    Idempotent — re-accepting an already-accepted type returns the
    current state with no new event. Returns 404 if the type_name does
    not exist in either list (no-op-but-state-mismatch is distinct
    from "name unknown").

    N.4d.3 — the response carries a ``placement``: where the accepted type sits
    in the notebook's vocabulary, and which of its siblings a judge thinks belong
    under it. It is a REPORT. Nothing is re-parented here; the curator applies a
    move by posting it to ``/schema/reparent``. A placement that cannot be
    computed (no ontology, no model) comes back ``null`` rather than failing the
    accept.
    """
    await _ensure_notebook_exists(notebook_id, notebook_service)
    try:
        updated = await edit_service.accept_extension(notebook_id, type_name)
    except NotebookSchemaNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except UnknownExtensionError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    response = _schema_to_response(notebook_id, updated)
    response.placement = await _placement_for_accepted(
        placement_service, updated, type_name
    )
    return response


@router.post(
    "/schema/extensions/{type_name}/reject",
    response_model=NotebookSchemaResponse,
    responses={
        404: {"description": "Notebook or schema row not found."},
    },
)
async def reject_pending_extension(
    notebook_id: str,
    type_name: str,
    notebook_service: NotebookService = Depends(get_notebook_service),
    edit_service: SchemaEditService = Depends(get_schema_edit_service),
) -> NotebookSchemaResponse:
    """Drop the matching pending extension.

    Idempotent: rejecting an already-removed type_name returns the
    current state with no new event. We deliberately do NOT 404 here
    even when the type_name is unknown — the natural UX flow is "user
    clicks Reject after the row has disappeared from another tab", and
    the result they want (no row, no event) is the same.
    """
    await _ensure_notebook_exists(notebook_id, notebook_service)
    try:
        updated = await edit_service.reject_extension(notebook_id, type_name)
    except NotebookSchemaNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return _schema_to_response(notebook_id, updated)


@router.post(
    "/schema/rename",
    response_model=NotebookSchemaResponse,
    responses={
        404: {"description": "Notebook or schema row not found."},
    },
)
async def rename_type(
    notebook_id: str,
    payload: RenameTypeRequest,
    notebook_service: NotebookService = Depends(get_notebook_service),
    edit_service: SchemaEditService = Depends(get_schema_edit_service),
) -> NotebookSchemaResponse:
    """Record a rename as a synonym in ``accepted_extensions``.

    The base ontology YAML is not mutated. Idempotent on
    (old_name, new_name).
    """
    await _ensure_notebook_exists(notebook_id, notebook_service)
    try:
        updated = await edit_service.rename_type(
            notebook_id, payload.old_name, payload.new_name
        )
    except NotebookSchemaNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return _schema_to_response(notebook_id, updated)


@router.post(
    "/schema/merge",
    response_model=NotebookSchemaResponse,
    responses={
        404: {"description": "Notebook or schema row not found."},
        422: {"description": "merge requires at least two distinct type names."},
    },
)
async def merge_types(
    notebook_id: str,
    payload: MergeTypesRequest,
    notebook_service: NotebookService = Depends(get_notebook_service),
    edit_service: SchemaEditService = Depends(get_schema_edit_service),
) -> NotebookSchemaResponse:
    """Record a merge of N types into one in ``accepted_extensions``.

    Idempotent on (sorted source-type set, merged_name).
    """
    await _ensure_notebook_exists(notebook_id, notebook_service)
    try:
        updated = await edit_service.merge_types(
            notebook_id, payload.type_names, payload.merged_name
        )
    except NotebookSchemaNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return _schema_to_response(notebook_id, updated)


@router.post(
    "/schema/split",
    response_model=NotebookSchemaResponse,
    responses={
        404: {"description": "Notebook or schema row not found."},
        422: {"description": "split requires at least two distinct target names."},
    },
)
async def split_type(
    notebook_id: str,
    payload: SplitTypeRequest,
    notebook_service: NotebookService = Depends(get_notebook_service),
    edit_service: SchemaEditService = Depends(get_schema_edit_service),
) -> NotebookSchemaResponse:
    """Record a split of one type into N new ones.

    ``criterion`` is plumbed through to Pass 2 prompts so the LLM can
    decide which target each instance belongs to. Idempotent on
    (source, sorted-into set, criterion).
    """
    await _ensure_notebook_exists(notebook_id, notebook_service)
    try:
        updated = await edit_service.split_type(
            notebook_id, payload.type_name, payload.into, payload.criterion
        )
    except NotebookSchemaNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return _schema_to_response(notebook_id, updated)


@router.post(
    "/schema/reparent",
    response_model=NotebookSchemaResponse,
    responses={
        404: {"description": "Notebook or schema row not found."},
        422: {
            "description": (
                "reparent requires a non-empty new_parent, at least one type "
                "name, and no type moved under itself."
            )
        },
    },
)
async def reparent_type(
    notebook_id: str,
    payload: ReparentTypeRequest,
    notebook_service: NotebookService = Depends(get_notebook_service),
    edit_service: SchemaEditService = Depends(get_schema_edit_service),
) -> NotebookSchemaResponse:
    """Record that N types now hang under ``new_parent`` (Track N.4d.3).

    This is how an accepted ``BROADER_THAN`` placement is applied: as a schema
    edit, on the curator's decision. The base ontology YAML is never mutated —
    the edit is projected onto COPIES of the applied ontologies at extraction
    time, which is what lets entities of a moved type resolve through their new
    ancestor without a single per-entity write.

    Idempotent per (type, parent).
    """
    await _ensure_notebook_exists(notebook_id, notebook_service)
    try:
        updated = await edit_service.reparent_type(
            notebook_id, payload.type_names, payload.new_parent
        )
    except NotebookSchemaNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return _schema_to_response(notebook_id, updated)


@router.delete(
    "/schema/types/{type_name}",
    response_model=NotebookSchemaResponse,
    responses={
        404: {"description": "Notebook or schema row not found."},
    },
)
async def delete_type(
    notebook_id: str,
    type_name: str,
    notebook_service: NotebookService = Depends(get_notebook_service),
    edit_service: SchemaEditService = Depends(get_schema_edit_service),
) -> NotebookSchemaResponse:
    """Soft-delete a type by adding it to ``excluded_types``.

    The base ontology YAML is never mutated. Idempotent: re-deleting
    a name already in the exclusion list returns the current state
    with no new event.
    """
    await _ensure_notebook_exists(notebook_id, notebook_service)
    try:
        updated = await edit_service.delete_type(notebook_id, type_name)
    except NotebookSchemaNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return _schema_to_response(notebook_id, updated)


@router.post(
    "/schema/dismiss_nudge",
    response_model=DismissNudgeResponse,
    responses={404: {"description": "Notebook not found."}},
)
async def dismiss_soft_nudge(
    notebook_id: str,
    notebook_service: NotebookService = Depends(get_notebook_service),
    schema_repo: NotebookSchemaRepository = Depends(get_notebook_schema_repo),
) -> DismissNudgeResponse:
    """Dismiss the low-coverage soft-nudge banner for this notebook (B.3c).

    Sets ``soft_nudge_dismissed=true``. The B.1e orchestrator re-arms
    the flag when coverage drops below threshold again — that logic
    lives outside this endpoint (downstream pipeline concern).

    Idempotent: dismissing twice is a no-op.
    """
    notebook = await notebook_service.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    schema = await _ensure_schema_row(schema_repo, notebook_id)
    schema.soft_nudge_dismissed = True
    await schema_repo.upsert(schema)
    logger.info("Dismissed soft-nudge for notebook {}", notebook_id)
    return DismissNudgeResponse(
        notebook_id=notebook_id,
        soft_nudge_dismissed=True,
    )


async def _list_paused_jobs_for_notebook(
    notebook_id: str,
    source_repo: SourceRepository,
) -> List[Any]:
    """Find all ``PAUSED_FOR_REVIEW`` jobs whose source belongs to this notebook.

    Done in two steps because the job table only stores ``source_id``,
    not ``notebook_id``. We walk the ``reference`` edge from the notebook
    to its sources and then query the job table for paused entries.

    Local import of :class:`JobRepository` to keep the module-level
    import graph tight (the schemas router otherwise has no job-queue
    dependency).
    """
    from job_queue.repository import JobRepository

    job_repo = JobRepository()

    rows = await source_repo.list_with_metadata(notebook_id=notebook_id, limit=500)
    source_ids = {str(row["id"]) for row in rows if row.get("id")}
    if not source_ids:
        return []

    paused_entity_jobs = await job_repo.list_jobs(
        status=JobStatus.PAUSED_FOR_REVIEW,
        job_type=JobType.ENTITY_EXTRACT,
        limit=200,
    )
    return [j for j in paused_entity_jobs if j.source_id in source_ids]


@router.get(
    "/extraction/paused",
    response_model=PausedExtractionStatus,
    responses={404: {"description": "Notebook not found."}},
)
async def list_paused_extraction(
    notebook_id: str,
    notebook_service: NotebookService = Depends(get_notebook_service),
    source_repo: SourceRepository = Depends(get_source_repo),
) -> PausedExtractionStatus:
    """Report ``PAUSED_FOR_REVIEW`` extraction jobs for this notebook (B.3c).

    The frontend ``ExtractionPausedBanner`` polls this; render the
    banner only when ``paused_count > 0``. Returns an empty list rather
    than 404 when the notebook has no sources or no paused jobs.
    """
    notebook = await notebook_service.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    paused_jobs = await _list_paused_jobs_for_notebook(notebook_id, source_repo)
    paused_source_ids = sorted(
        {j.source_id for j in paused_jobs if j.source_id}
    )
    return PausedExtractionStatus(
        notebook_id=notebook_id,
        paused_count=len(paused_jobs),
        paused_source_ids=paused_source_ids,
    )


@router.post(
    "/extraction/resume",
    response_model=ResumeExtractionResponse,
    responses={404: {"description": "Notebook not found."}},
)
async def resume_extraction(
    notebook_id: str,
    notebook_service: NotebookService = Depends(get_notebook_service),
    schema_repo: NotebookSchemaRepository = Depends(get_notebook_schema_repo),
    source_repo: SourceRepository = Depends(get_source_repo),
) -> ResumeExtractionResponse:
    """Resume extraction for sources paused under ``review_required`` (B.3c).

    Two effects:

    1. **Sentinel marker** — if the review-gate predicate would still
       fire (``review_required=True`` AND ``accepted_extensions`` is
       empty), append a sentinel entry so the next pass-1 attempt
       advances past the gate. See the module-level comment on
       :data:`_RESUME_SENTINEL_TYPE_NAME` for the rationale and shape.
    2. **Job re-queue** — every ``ENTITY_EXTRACT`` job in
       ``PAUSED_FOR_REVIEW`` whose source belongs to this notebook is
       moved back to ``QUEUED``. The worker will pick them up on the
       next tick.

    Returns counts so the UI can render "Resumed 3 sources" instead of
    a bare 200.
    """
    notebook = await notebook_service.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    schema = await _ensure_schema_row(schema_repo, notebook_id)

    sentinel_added = False
    if schema.review_required and not schema.accepted_extensions:
        # Predicate would still raise — append a sentinel.
        # ``isoformat()`` on a tz-aware datetime renders ``+00:00`` for
        # UTC; we normalise to ``Z`` for consistency with the rest of
        # the codebase (event ``created`` timestamps + frontend parsers
        # both expect ``Z``).
        created_at = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        schema.accepted_extensions.append(
            {
                "type_name": _RESUME_SENTINEL_TYPE_NAME,
                "is_resume_sentinel": True,
                "created_at": created_at,
            }
        )
        await schema_repo.upsert(schema)
        sentinel_added = True
        logger.info(
            "Added resume sentinel to notebook {} (review_required=True, "
            "accepted_extensions was empty)",
            notebook_id,
        )

    # Move any paused jobs back to QUEUED so the worker re-picks them.
    paused_jobs = await _list_paused_jobs_for_notebook(notebook_id, source_repo)
    if paused_jobs:
        from job_queue.repository import JobRepository

        job_repo = JobRepository()
        resumed = 0
        for job in paused_jobs:
            try:
                await job_repo.update_status(
                    job.id, JobStatus.QUEUED, error_message=None
                )
                resumed += 1
            except Exception as e:
                logger.warning(
                    "Failed to re-queue paused job {}: {}", job.id, e
                )
        logger.info(
            "Resumed {} paused extraction job(s) for notebook {}",
            resumed,
            notebook_id,
        )
    else:
        resumed = 0

    return ResumeExtractionResponse(
        notebook_id=notebook_id,
        resumed_count=resumed,
        sentinel_added=sentinel_added,
    )


# ---------------------------------------------------------------------------
# Re-extract prompt endpoints (Phase B.3d)
# ---------------------------------------------------------------------------
#
# Backs the ``ReextractPromptBanner``. Two endpoints:
#
# * ``GET /schema/reextract_candidates`` — affected source ids the user
#   should consider re-extracting. V1 returns ALL sources in the
#   notebook (rationale: every schema edit op can touch any source's
#   extraction). Future revisions can narrow this to sources containing
#   entities of the modified type once we wire the entity → source
#   index.
#
# * ``POST /schema/reextract`` — enqueue one ENTITY_EXTRACT job per
#   source id. Idempotent — dedup is per-source: re-clicking the same
#   button does not stack duplicate jobs.
#
# Reconciliation note (attempt 2): the parallel B.3c branch shipped
# ``notebook_events.py`` on main with the canonical ``/events`` +
# ``/events/{event_id}/mark_read`` contract. This router previously
# carried duplicate handlers with an incompatible payload shape; they
# were removed during the rebase so the banner now consumes the B.3c
# endpoints (see ``frontend/src/lib/api/notebook-schema.ts`` for the
# matching client surface).


class ReextractCandidatesResponse(BaseModel):
    """JSON contract for ``GET /api/notebooks/{id}/schema/reextract_candidates``.

    ``source_ids`` is the affected list (V1: all notebook sources).
    ``count`` is a convenience for the banner — same as ``len(source_ids)``
    but serialised separately so the banner header can render without
    needing to count client-side.
    """

    notebook_id: str
    source_ids: List[str] = Field(default_factory=list)
    count: int = 0


class ReextractRequest(BaseModel):
    """Payload for ``POST /api/notebooks/{id}/schema/reextract``.

    Empty ``source_ids`` is accepted and treated as a no-op (returns
    200 with zero counts) — matches the "Re-extract selected" UX where
    the user hits Submit with nothing checked.
    """

    source_ids: List[str] = Field(default_factory=list)


class ReextractResponse(BaseModel):
    """JSON contract for ``POST /api/notebooks/{id}/schema/reextract``.

    The split between ``enqueued_source_ids`` and ``skipped_source_ids``
    surfaces which sources the dedup layer caught (already in-flight).
    The banner can use this to render a "3 jobs queued, 1 already
    running" status line without re-querying the job table.
    """

    notebook_id: str
    jobs_enqueued: int = 0
    source_ids: List[str] = Field(default_factory=list)
    enqueued_source_ids: List[str] = Field(default_factory=list)
    skipped_source_ids: List[str] = Field(default_factory=list)


@router.get(
    "/schema/reextract_candidates",
    response_model=ReextractCandidatesResponse,
    responses={404: {"description": "Notebook not found."}},
)
async def list_reextract_candidates(
    notebook_id: str,
    notebook_service: NotebookService = Depends(get_notebook_service),
    source_repo=Depends(get_source_repo),
) -> ReextractCandidatesResponse:
    """List affected source ids for the re-extract banner.

    V1 returns ALL sources currently linked to the notebook. The
    rationale is conservative: any rename / merge / split / delete /
    accept / reject op can in principle change which entities a source
    produces, so we treat every source as potentially affected. A
    future B.3d-r1 revision can narrow this to sources containing
    entities of the modified type once we have an entity → source
    index (Track G).

    Returns 404 if the notebook doesn't exist; 200 with an empty list
    when the notebook has no sources.
    """
    await _ensure_notebook_exists(notebook_id, notebook_service)
    try:
        rows = await source_repo.list_with_metadata(notebook_id=notebook_id)
    except Exception as e:
        # Defensive — list_with_metadata returning empty on transient
        # errors would still serve a useful (albeit zero-entry) banner.
        logger.warning(
            "Failed to list re-extract candidates for {}: {}",
            notebook_id,
            e,
        )
        rows = []
    source_ids = [str(row["id"]) for row in rows if row.get("id")]
    return ReextractCandidatesResponse(
        notebook_id=notebook_id,
        source_ids=source_ids,
        count=len(source_ids),
    )


@router.post(
    "/schema/reextract",
    response_model=ReextractResponse,
    responses={
        404: {"description": "Notebook not found."},
    },
)
async def enqueue_reextract(
    notebook_id: str,
    payload: ReextractRequest,
    notebook_service: NotebookService = Depends(get_notebook_service),
    source_repo=Depends(get_source_repo),
    reextract_service: ReextractService = Depends(get_reextract_service),
) -> ReextractResponse:
    """Enqueue one ENTITY_EXTRACT job per requested source id.

    Idempotent: a source whose latest ``entity_extract`` job is still
    in-flight (QUEUED / PROCESSING / RETRYING / PAUSED_FOR_REVIEW) is
    silently skipped, so a double-click never stacks duplicate work.
    Completed / failed jobs are NOT considered duplicates — re-running
    after a previous run finished is the normal flow under a schema
    change.

    Defence-in-depth: requested ``source_ids`` are filtered against the
    notebook's candidate list before enqueueing, so a malformed client
    payload cannot enqueue jobs for sources belonging to a different
    notebook (cross-notebook source-id injection). Out-of-list ids
    are dropped silently — the candidate-list endpoint is the source
    of truth and the banner only submits ids it received from there.

    Returns the enqueued / skipped split so the banner can render a
    precise per-source status line.
    """
    await _ensure_notebook_exists(notebook_id, notebook_service)

    # Minor 1 fix — defence-in-depth against cross-notebook source_id
    # injection. We constrain the requested ids to the notebook's own
    # candidate set; anything else is dropped before reaching the
    # service layer. A future refactor could push this guard into the
    # service, but keeping it at the router avoids loading the source
    # repo from the service.
    try:
        rows = await source_repo.list_with_metadata(notebook_id=notebook_id)
    except Exception as e:
        logger.warning(
            "Failed to list notebook sources during re-extract "
            "candidate filter for {}: {}; proceeding without filter",
            notebook_id,
            e,
        )
        rows = []
    if rows:
        candidate_ids = {str(row["id"]) for row in rows if row.get("id")}
        requested = list(payload.source_ids)
        filtered = [sid for sid in requested if sid in candidate_ids]
        dropped = [sid for sid in requested if sid not in candidate_ids]
        if dropped:
            logger.info(
                "Dropping {} out-of-notebook source id(s) from re-extract "
                "request for {}: {!r}",
                len(dropped),
                notebook_id,
                dropped,
            )
    else:
        # If we can't read the candidate list (transient DB issue),
        # fall through to the unfiltered request rather than block the
        # user — the service layer's own validity check will still
        # apply, and the worker can no-op a bogus source_id.
        filtered = list(payload.source_ids)

    result = await reextract_service.enqueue_reextract_jobs(
        notebook_id=notebook_id,
        source_ids=filtered,
    )
    return ReextractResponse(
        notebook_id=notebook_id,
        jobs_enqueued=result.jobs_enqueued,
        source_ids=result.source_ids,
        enqueued_source_ids=result.enqueued_source_ids,
        skipped_source_ids=result.skipped_source_ids,
    )
