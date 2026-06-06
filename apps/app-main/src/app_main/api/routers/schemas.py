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

* **rdflib is a runtime dep** of ``ontology-manager`` (merged in B.2a),
  so the import is unconditional at module load time.
"""

from __future__ import annotations

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
# still need to pick a base ontology to export. ``scholarly`` matches the
# default in ``ontology_manager.config.OntologyManagerConfig``.
_DEFAULT_BASE_ONTOLOGY = "scholarly"


def _ontologies_dir() -> Path:
    """Resolve the bundled YAML ontology directory.

    Mirrors the search performed by ``OntologyRegistry._find_ontology_dir``
    but constrained to a single resolution since we always want the
    in-repo files for the TTL exporter (DB-stored ontologies aren't yet
    a thing the TTL exporter knows about).

    The file lives at:
        packages/ontology-manager/ontologies/<name>.yaml

    Resolution priority:
        1. ``ONTOLOGY_DIR`` env override (production / docker).
        2. Repo-relative fallback computed from this file's path.
    """
    import os

    env_dir = os.environ.get("ONTOLOGY_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.exists():
            return p

    # This file: apps/app-main/src/app_main/api/routers/schemas.py
    # parents[0..5]: routers, api, app_main, src, app-main, apps
    # parents[6]   : repo root
    repo_root = Path(__file__).resolve().parents[6]
    return repo_root / "packages" / "ontology-manager" / "ontologies"


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

        cls_uri: URIRef = ON[type_name]
        graph.add((cls_uri, RDF.type, OWL.Class))
        graph.add((cls_uri, RDFS.label, Literal(type_name)))
        classes_added += 1

        parent = ext.get("parent_type")
        if isinstance(parent, str) and parent:
            graph.add((cls_uri, RDFS.subClassOf, ON[parent]))

        description = ext.get("description")
        if isinstance(description, str) and description:
            graph.add((cls_uri, RDFS.comment, Literal(description)))

        for prop in ext.get("properties", []) or []:
            if not isinstance(prop, dict):
                continue
            pname = prop.get("name")
            if not pname or not isinstance(pname, str):
                continue
            prop_uri = ON[pname]
            graph.add((prop_uri, RDF.type, OWL.DatatypeProperty))
            graph.add((prop_uri, RDFS.domain, cls_uri))
            graph.add((prop_uri, RDFS.label, Literal(pname)))

    return classes_added


def _safe_filename(notebook_id: str) -> str:
    """Build a filesystem-safe filename from a SurrealDB record id.

    ``notebook:abc-123`` → ``notebook_abc-123.ttl``. Browsers tolerate
    most punctuation in ``Content-Disposition`` filenames, but the colon
    can trip older versions of Windows Save dialogs, so we replace it
    with an underscore.
    """
    return notebook_id.replace(":", "_").replace("/", "_") + ".ttl"


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
