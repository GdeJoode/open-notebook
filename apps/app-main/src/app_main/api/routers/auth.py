"""Auth router."""

import os

from fastapi import APIRouter

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/status")
async def get_auth_status():
    """Return current authentication status."""
    auth_enabled = bool(os.environ.get("OPEN_NOTEBOOK_PASSWORD"))
    return {
        "auth_enabled": auth_enabled,
        "message": (
            "Authentication is required"
            if auth_enabled
            else "Authentication is disabled"
        ),
    }
