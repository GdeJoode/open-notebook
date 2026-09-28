"""Ontologies router - browse ontology definitions."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from app_main.dependencies import (
    get_entity_retype_service,
    get_ontology_edit_service,
    get_ontology_service,
)
from app_main.services.entity_retype_service import EntityRetypeService
from app_main.services.ontology_edit_service import (
    OntologyEditError,
    OntologyEditService,
)
from app_main.services.ontology_service import OntologyService

router = APIRouter(prefix="/ontologies", tags=["ontologies"])


@router.get("", response_model=List[Dict[str, Any]])
async def list_ontologies(
    svc: OntologyService = Depends(get_ontology_service),
):
    """List all ontologies with summary info."""
    return await svc.list_ontologies()


@router.get("/{name}", response_model=Dict[str, Any])
async def get_ontology(
    name: str,
    svc: OntologyService = Depends(get_ontology_service),
):
    """Get full ontology detail."""
    result = await svc.get_ontology(name)
    if not result:
        raise HTTPException(status_code=404, detail=f"Ontology '{name}' not found")
    return result


@router.get("/{name}/entity-types", response_model=List[Dict[str, Any]])
async def get_entity_types(
    name: str,
    svc: OntologyService = Depends(get_ontology_service),
):
    """Get entity types for an ontology."""
    result = await svc.get_entity_types(name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Ontology '{name}' not found")
    return result


@router.get("/{name}/relationship-types", response_model=List[Dict[str, Any]])
async def get_relationship_types(
    name: str,
    svc: OntologyService = Depends(get_ontology_service),
):
    """Get relationship types for an ontology."""
    result = await svc.get_relationship_types(name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Ontology '{name}' not found")
    return result


# ---------------------------------------------------------------------------
# Editing (Track OE.2): draft -> validate -> preview -> publish; versions.
# ---------------------------------------------------------------------------


class NameIn(BaseModel):
    new_name: str = Field(..., min_length=1, max_length=200, pattern=r"^\S(.*\S)?$")


class PublishIn(BaseModel):
    note: Optional[str] = None


def _draft_out(draft: Dict[str, Any]) -> Dict[str, Any]:
    return {**{k: v for k, v in draft.items() if k != "ontology"},
            "ontology": draft["ontology"].to_storage_dict()}


async def _call(coro):
    try:
        return await coro
    except OntologyEditError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc


@router.get("/{name}/draft", response_model=Dict[str, Any])
async def get_draft(name: str, svc: OntologyEditService = Depends(get_ontology_edit_service)):
    """The draft, or the active own content when no draft exists yet."""
    return _draft_out(await _call(svc.get_draft(name)))


@router.delete("/{name}/draft", status_code=204)
async def discard_draft(name: str, svc: OntologyEditService = Depends(get_ontology_edit_service)):
    await svc.discard_draft(name)


@router.put("/{name}/draft/entity-types/{type_name}", response_model=Dict[str, Any])
async def put_entity_type(name: str, type_name: str, body: Dict[str, Any],
                          svc: OntologyEditService = Depends(get_ontology_edit_service)):
    """Create or REPLACE an entity type in the draft.

    The body is the whole type: a key it leaves out is removed from the type
    (the editor sends the stored type back with its changes, so keys it does
    not show, such as ``graph_relevance``, survive).
    """
    return _draft_out(await _call(svc.put_entity_type(name, type_name, body)))


@router.delete("/{name}/draft/entity-types/{type_name}", response_model=Dict[str, Any])
async def delete_entity_type(name: str, type_name: str,
                             svc: OntologyEditService = Depends(get_ontology_edit_service)):
    return _draft_out(await _call(svc.delete_entity_type(name, type_name)))


@router.post("/{name}/draft/entity-types/{type_name}/rename", response_model=Dict[str, Any])
async def rename_entity_type(name: str, type_name: str, body: NameIn,
                             svc: OntologyEditService = Depends(get_ontology_edit_service)):
    """Rename; the old name stays as an alias."""
    return _draft_out(await _call(svc.rename_entity_type(name, type_name, body.new_name)))


@router.post("/{name}/draft/relationship-types/{rel_name}/rename", response_model=Dict[str, Any])
async def rename_relationship_type(name: str, rel_name: str, body: NameIn,
                                   svc: OntologyEditService = Depends(get_ontology_edit_service)):
    return _draft_out(await _call(svc.rename_relationship_type(name, rel_name, body.new_name)))


@router.put("/{name}/draft/relationship-types/{rel_name}", response_model=Dict[str, Any])
async def put_relationship_type(name: str, rel_name: str, body: Dict[str, Any],
                                svc: OntologyEditService = Depends(get_ontology_edit_service)):
    return _draft_out(await _call(svc.put_relationship_type(name, rel_name, body)))


@router.delete("/{name}/draft/relationship-types/{rel_name}", response_model=Dict[str, Any])
async def delete_relationship_type(name: str, rel_name: str,
                                   svc: OntologyEditService = Depends(get_ontology_edit_service)):
    return _draft_out(await _call(svc.delete_relationship_type(name, rel_name)))


@router.post("/{name}/draft/validate", response_model=List[Dict[str, Any]])
async def validate_draft(name: str, svc: OntologyEditService = Depends(get_ontology_edit_service)):
    """The OE.1 gate on the draft: blocking and warning findings."""
    return await _call(svc.validate_draft(name))


@router.post("/{name}/draft/preview", response_model=Dict[str, Any])
async def preview_draft(name: str, include_detection: bool = True,
                        svc: OntologyEditService = Depends(get_ontology_edit_service)):
    """What publishing the draft would do (OE.1 preview). Synchronous; the
    detection replay can be left out with ``include_detection=false``."""
    return await _call(svc.preview_draft(name, include_detection))


@router.post("/{name}/draft/publish", response_model=Dict[str, Any])
async def publish_draft(name: str, body: PublishIn,
                        svc: OntologyEditService = Depends(get_ontology_edit_service)):
    """Store the draft as the new active version. 422 on a blocking finding,
    409 when another version was published since the draft was started."""
    return await _call(svc.publish(name, body.note))


@router.get("/{name}/versions", response_model=List[Dict[str, Any]])
async def list_versions(name: str, svc: OntologyEditService = Depends(get_ontology_edit_service)):
    return await svc.versions(name)


@router.post("/{name}/rollback/{version}", response_model=Dict[str, Any])
async def rollback(name: str, version: str,
                   svc: OntologyEditService = Depends(get_ontology_edit_service)):
    """Re-activate a stored version, through the same gate as a publish."""
    return await _call(svc.rollback(name, version))


@router.get("/{name}/export.yaml", response_class=Response)
async def export_yaml(name: str, svc: OntologyEditService = Depends(get_ontology_edit_service)):
    """The active own content as YAML, to commit to the repo."""
    text = await _call(svc.export_yaml(name))
    return Response(content=text, media_type="application/x-yaml",
                    headers={"Content-Disposition": f'attachment; filename="{name}.yaml"'})


# ---------------------------------------------------------------------------
# Retype (Track OE.4): the rows a publish left on their old type. Dry run, then
# an apply of exactly the plan the dry run showed (OE-D3 (a)).
# ---------------------------------------------------------------------------


class RetypeApplyIn(BaseModel):
    fingerprint: str = Field(..., min_length=1)


@router.get("/retype/dry-run", response_model=Dict[str, Any])
async def retype_dry_run(svc: EntityRetypeService = Depends(get_entity_retype_service)):
    """The retype under the ACTIVE ontologies: changes, collisions (left for the
    curator's merge), flag-only rows, counts, and the plan's ``fingerprint``."""
    return await svc.dry_run()


@router.post("/retype/apply", response_model=Dict[str, Any])
async def retype_apply(body: RetypeApplyIn,
                       svc: EntityRetypeService = Depends(get_entity_retype_service)):
    """Write the plan the dry run showed. 409 when the plan changed since (its
    fingerprint differs); nothing is written then.
    Returns the counts and the ``rollback_id`` to undo it."""
    return await _call(svc.apply(body.fingerprint))


@router.post("/retype/undo/{rollback_id}", response_model=Dict[str, Any])
async def retype_undo(rollback_id: str,
                      svc: EntityRetypeService = Depends(get_entity_retype_service)):
    """Undo an apply: the rows it wrote go back to their previous type and flag."""
    return await _call(svc.undo(rollback_id))
