"""Ontologies router - browse ontology definitions."""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from app_main.dependencies import get_ontology_service
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
