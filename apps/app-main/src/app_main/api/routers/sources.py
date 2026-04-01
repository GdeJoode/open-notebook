"""Sources router - aggregates all source sub-routers."""
from fastapi import APIRouter

from app_main.api.routers.sources_crud import router as crud_router
from app_main.api.routers.sources_upload import router as upload_router
from app_main.api.routers.sources_processing import router as processing_router
from app_main.api.routers.sources_files import router as files_router

router = APIRouter(prefix="/sources", tags=["sources"])
router.include_router(crud_router)
router.include_router(upload_router)
router.include_router(processing_router)
router.include_router(files_router)
