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
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Response
from loguru import logger

from app_main.dependencies import get_notebook_service
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
)


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
