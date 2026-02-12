"""
File Manager API Application.

FastAPI application that provides file management, workspace organization,
and file tracking through the ingestion pipeline.
"""

import os
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from file_manager.api.routers import (
    files,
    jobs,
    knowledge_bases,
    projects,
    source_folders,
    storage,
    tracked_files,
    workspace,
)
from file_manager.config import get_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup: ensure directories exist
    config = get_config()
    config.ensure_directories()

    # Ensure workspace directories
    storage_root = os.environ.get("STORAGE_ROOT", os.path.expanduser("~/.open-notebook"))
    for subdir in ["projects", "knowledgebases", "temp", "sources"]:
        path = os.path.join(storage_root, subdir)
        os.makedirs(path, exist_ok=True)

    yield
    # Shutdown: cleanup if needed


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="File Manager API",
        description="File management, workspace organization, and pipeline tracking",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(
        workspace.router,
        prefix="/workspace",
        tags=["Workspace"],
    )
    app.include_router(
        projects.router,
        prefix="/projects",
        tags=["Projects"],
    )
    app.include_router(
        knowledge_bases.router,
        prefix="/knowledge-bases",
        tags=["Knowledge Bases"],
    )
    app.include_router(
        tracked_files.router,
        prefix="/tracked-files",
        tags=["Tracked Files"],
    )
    app.include_router(
        files.router,
        prefix="/files",
        tags=["Files"],
    )
    app.include_router(
        storage.router,
        prefix="/storage",
        tags=["Storage"],
    )
    app.include_router(
        jobs.router,
        prefix="/jobs",
        tags=["Jobs"],
    )
    app.include_router(
        source_folders.router,
        prefix="/source-folders",
        tags=["Source Folders"],
    )

    @app.get("/")
    async def root() -> Dict[str, Any]:
        """Root endpoint with API info."""
        return {
            "service": "file-manager",
            "version": "0.1.0",
            "docs": "/docs",
            "endpoints": {
                "health": "/health",
                "workspace": "/workspace",
                "projects": "/projects",
                "knowledge_bases": "/knowledge-bases",
                "tracked_files": "/tracked-files",
                "files": "/files",
                "storage": "/storage",
                "jobs": "/jobs",
                "source_folders": "/source-folders",
            },
        }

    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "file-manager",
        }

    return app


# Application instance
app = create_app()


def main():
    """Run the application with uvicorn."""
    import uvicorn

    host = os.environ.get("FILE_MANAGER_HOST", "0.0.0.0")
    port = int(os.environ.get("FILE_MANAGER_PORT", "8002"))

    uvicorn.run(
        "file_manager.api.app:app",
        host=host,
        port=port,
        reload=os.environ.get("FILE_MANAGER_RELOAD", "false").lower() == "true",
    )


if __name__ == "__main__":
    main()
