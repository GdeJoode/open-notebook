"""Vault sync router — bidirectional sync with external vault (Obsidian)."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel

from app_main.dependencies import get_settings_service
from app_main.services.settings_service import SettingsService
from app_main.services.vault_sync_service import VaultSyncService

router = APIRouter(prefix="/vault", tags=["vault"])


def _get_vault_service() -> VaultSyncService:
    return VaultSyncService()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ApplyImportsRequest(BaseModel):
    accepted_ids: List[str] = []
    rejected_ids: List[str] = []


class WriteExportsRequest(BaseModel):
    accepted_ids: List[str] = []
    rejected_ids: List[str] = []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/sync")
async def sync_vault(
    vault_svc: VaultSyncService = Depends(_get_vault_service),
    settings_svc: SettingsService = Depends(get_settings_service),
):
    """Trigger a vault scan: read entity notes and create pending imports."""
    settings = await settings_svc.get()
    vault_path = settings.vault_path
    entities_folder = settings.vault_entities_folder or "Entities"

    if not vault_path:
        raise HTTPException(
            status_code=422,
            detail="No vault path configured. Set it in Settings → Vault Integration.",
        )

    result = await vault_svc.scan_vault_entities(vault_path, entities_folder)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return result


@router.get("/imports/pending")
async def get_pending_imports(
    vault_svc: VaultSyncService = Depends(_get_vault_service),
):
    """List vault import items pending user review."""
    return await vault_svc.get_pending_imports()


@router.post("/imports/apply")
async def apply_imports(
    body: ApplyImportsRequest,
    vault_svc: VaultSyncService = Depends(_get_vault_service),
):
    """Accept or reject pending vault imports."""
    result = await vault_svc.apply_imports(
        accepted_ids=body.accepted_ids,
        rejected_ids=body.rejected_ids,
    )

    # Reload LLMMatcher dictionary with new aliases
    try:
        dictionary = await vault_svc.get_merged_dictionary()
        from entity_filtering.resolution.llm_matcher import LLMMatcher

        LLMMatcher.set_abbreviations(dictionary)
        result["dictionary_size"] = len(dictionary)
    except Exception as e:
        logger.warning(f"Failed to reload LLMMatcher dictionary: {e}")

    return result


@router.get("/exports/pending")
async def get_pending_exports(
    vault_svc: VaultSyncService = Depends(_get_vault_service),
):
    """List entities pending export to vault."""
    return await vault_svc.get_pending_exports()


@router.post("/exports/write")
async def write_exports(
    body: WriteExportsRequest,
    vault_svc: VaultSyncService = Depends(_get_vault_service),
    settings_svc: SettingsService = Depends(get_settings_service),
):
    """Accept or reject pending exports, writing approved stubs to vault."""
    settings = await settings_svc.get()
    vault_path = settings.vault_path
    entities_folder = settings.vault_entities_folder or "Entities"

    if not vault_path:
        raise HTTPException(
            status_code=422,
            detail="No vault path configured.",
        )

    return await vault_svc.write_exports(
        accepted_ids=body.accepted_ids,
        rejected_ids=body.rejected_ids,
        vault_path=vault_path,
        entities_folder=entities_folder,
    )


@router.get("/dictionary")
async def get_dictionary(
    vault_svc: VaultSyncService = Depends(_get_vault_service),
):
    """Get the full active merged dictionary (vault + entity_alias)."""
    return await vault_svc.get_merged_dictionary()
