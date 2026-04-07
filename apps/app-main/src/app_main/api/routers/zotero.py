"""Zotero integration router — connection, sync, and per-source push."""

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from surrealdb_service.connection import execute_query

router = APIRouter(prefix="/zotero", tags=["zotero"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ZoteroConnectRequest(BaseModel):
    library_id: str = Field(..., description="Zotero user or group ID")
    api_key: str = Field(..., description="Zotero API key")
    library_type: str = Field("user", description="user or group")
    default_collection: Optional[str] = Field(None, description="Default collection name")
    auto_push: bool = Field(False, description="Auto-push new sources to Zotero")
    sync_interval_minutes: int = Field(15, ge=1, le=1440)


class ZoteroPushRequest(BaseModel):
    collection: Optional[str] = Field(None, description="Collection name to add to")


class ZoteroSyncResult(BaseModel):
    pushed: int = 0
    pulled: int = 0
    conflicts: int = 0
    deleted: int = 0
    new_version: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_config() -> Optional[Dict[str, Any]]:
    """Load Zotero config from DB."""
    result = await execute_query("SELECT * FROM zotero_config LIMIT 1", {})
    if result and result[0]:
        return result[0][0] if isinstance(result[0], list) else result[0]
    return None


async def _get_service():
    """Create ZoteroService from stored config."""
    config = await _get_config()
    if not config or not config.get("enabled"):
        raise HTTPException(status_code=400, detail="Zotero not configured. Go to Settings to connect.")

    from zotero_integration import ZoteroService

    return ZoteroService(
        library_id=config["library_id"],
        api_key=config["api_key"],
        library_type=config.get("library_type", "user"),
    )


async def _get_sync_state() -> Dict[str, Any]:
    """Get current sync state."""
    result = await execute_query("SELECT * FROM zotero_sync_state LIMIT 1", {})
    if result and result[0]:
        row = result[0][0] if isinstance(result[0], list) else result[0]
        return row
    return {"library_version": 0, "status": "idle"}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/connect")
async def connect_zotero(body: ZoteroConnectRequest):
    """Validate Zotero credentials and store config."""
    from zotero_integration import ZoteroService

    service = ZoteroService(body.library_id, body.api_key, body.library_type)
    result = await asyncio.to_thread(service.test_connection)

    if result["status"] != "connected":
        raise HTTPException(status_code=400, detail=f"Connection failed: {result.get('message', 'Unknown error')}")

    # Upsert config
    await execute_query(
        """
        DELETE zotero_config;
        CREATE zotero_config SET
            library_id = $library_id,
            api_key = $api_key,
            library_type = $library_type,
            default_collection = $default_collection,
            auto_push = $auto_push,
            sync_interval_minutes = $sync_interval,
            enabled = true,
            updated = time::now()
        """,
        {
            "library_id": body.library_id,
            "api_key": body.api_key,
            "library_type": body.library_type,
            "default_collection": body.default_collection,
            "auto_push": body.auto_push,
            "sync_interval": body.sync_interval_minutes,
        },
    )

    # Initialize sync state
    await execute_query(
        """
        DELETE zotero_sync_state;
        CREATE zotero_sync_state SET
            library_version = $version,
            last_sync = time::now(),
            status = "idle"
        """,
        {"version": result["library_version"]},
    )

    return {
        "status": "connected",
        "library_version": result["library_version"],
    }


@router.delete("/disconnect")
async def disconnect_zotero():
    """Remove Zotero configuration."""
    await execute_query("DELETE zotero_config", {})
    await execute_query("DELETE zotero_sync_state", {})
    return {"status": "disconnected"}


@router.get("/status")
async def get_zotero_status():
    """Get Zotero connection and sync status."""
    config = await _get_config()
    if not config or not config.get("enabled"):
        return {"connected": False}

    sync_state = await _get_sync_state()
    return {
        "connected": True,
        "library_type": config.get("library_type", "user"),
        "auto_push": config.get("auto_push", False),
        "default_collection": config.get("default_collection"),
        "sync_interval_minutes": config.get("sync_interval_minutes", 15),
        "last_sync": sync_state.get("last_sync"),
        "library_version": sync_state.get("library_version", 0),
        "sync_status": sync_state.get("status", "idle"),
    }


@router.get("/collections")
async def list_collections():
    """List Zotero collections."""
    service = await _get_service()
    collections = await asyncio.to_thread(service.list_collections)
    return collections


@router.get("/items")
async def list_zotero_items(limit: int = 25, start: int = 0):
    """List items from the Zotero library."""
    service = await _get_service()
    items = await asyncio.to_thread(service.list_items, limit, start)
    return items


@router.post("/sources/{source_id}/push")
async def push_source_to_zotero(
    source_id: str,
    body: ZoteroPushRequest = ZoteroPushRequest(),
):
    """Push a single source to Zotero with cloud file upload."""
    service = await _get_service()
    config = await _get_config()

    # Load source from DB
    result = await execute_query(
        "SELECT * FROM $source_id",
        {"source_id": source_id},
    )
    if not result or not result[0]:
        raise HTTPException(status_code=404, detail="Source not found")

    source_data = result[0][0] if isinstance(result[0], list) else result[0]

    # Build source dict for ZoteroService
    source_dict = {
        "title": source_data.get("title", "Untitled"),
        "type": source_data.get("source_type", "document"),
        "topics": source_data.get("topics", []),
        "url": source_data.get("asset", {}).get("url") if isinstance(source_data.get("asset"), dict) else None,
    }

    # Get file path for upload
    file_path = None
    asset = source_data.get("asset")
    if isinstance(asset, dict) and asset.get("file_path"):
        file_path = asset["file_path"]

    # Determine collection
    collection_key = None
    collection_name = body.collection or config.get("default_collection")
    if collection_name:
        collection_key = await asyncio.to_thread(
            service.get_or_create_collection, collection_name
        )

    # Push
    push_result = await asyncio.to_thread(
        service.push_source, source_dict, file_path, collection_key
    )

    # Store Zotero key back in source record
    if push_result.get("zotero_key"):
        await execute_query(
            """
            UPDATE $source_id SET
                zotero_key = $zotero_key,
                zotero_version = $zotero_version,
                zotero_synced_at = time::now()
            """,
            {
                "source_id": source_id,
                "zotero_key": push_result["zotero_key"],
                "zotero_version": push_result.get("zotero_version", 0),
            },
        )

    return push_result


@router.post("/sync", response_model=ZoteroSyncResult)
async def sync_with_zotero():
    """Run bidirectional sync with Zotero."""
    service = await _get_service()
    sync_state = await _get_sync_state()

    # Mark as syncing
    await execute_query(
        "UPDATE zotero_sync_state SET status = 'syncing'", {}
    )

    try:
        # Get all sources that have been previously synced or are new
        sources_result = await execute_query(
            "SELECT * FROM source",
            {},
        )
        sources = sources_result[0] if sources_result and sources_result[0] else []

        local_sources = []
        for s in sources:
            local_sources.append({
                "id": s.get("id"),
                "title": s.get("title", ""),
                "type": s.get("source_type", "document"),
                "topics": s.get("topics", []),
                "zotero_key": s.get("zotero_key"),
                "modified_at": str(s.get("updated", "")),
                "zotero_synced_at": str(s.get("zotero_synced_at", "")),
                "file_path": s.get("asset", {}).get("file_path") if isinstance(s.get("asset"), dict) else None,
            })

        last_version = sync_state.get("library_version", 0)
        result = await asyncio.to_thread(
            service.sync, local_sources, last_version
        )

        # Update synced sources in DB
        for pushed in result.get("pushed", []):
            zkey = pushed.get("zotero_key")
            if not zkey:
                continue
            # Find the local source that was pushed
            for ls in local_sources:
                if ls.get("zotero_key") == zkey or (
                    not ls.get("zotero_key") and pushed.get("title") == ls.get("title")
                ):
                    await execute_query(
                        """
                        UPDATE $source_id SET
                            zotero_key = $zotero_key,
                            zotero_version = $zotero_version,
                            zotero_synced_at = time::now()
                        """,
                        {
                            "source_id": ls["id"],
                            "zotero_key": zkey,
                            "zotero_version": pushed.get("zotero_version", 0),
                        },
                    )
                    break

        # Update sync state
        await execute_query(
            """
            UPDATE zotero_sync_state SET
                library_version = $version,
                last_sync = time::now(),
                items_pushed = $pushed,
                items_pulled = $pulled,
                conflicts = $conflicts,
                status = 'idle'
            """,
            {
                "version": result.get("new_version", last_version),
                "pushed": len(result.get("pushed", [])),
                "pulled": len(result.get("pulled", [])),
                "conflicts": len(result.get("conflicts", [])),
            },
        )

        return ZoteroSyncResult(
            pushed=len(result.get("pushed", [])),
            pulled=len(result.get("pulled", [])),
            conflicts=len(result.get("conflicts", [])),
            deleted=len(result.get("deleted", [])),
            new_version=result.get("new_version", last_version),
        )

    except Exception as e:
        await execute_query(
            "UPDATE zotero_sync_state SET status = 'error'", {}
        )
        logger.error(f"Zotero sync failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")
