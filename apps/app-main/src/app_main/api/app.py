"""
FastAPI application factory for Open Notebook.
"""

import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from app_main.api.auth import PasswordAuthMiddleware
from app_main.exceptions import (
    AuthenticationError,
    ConfigurationError,
    DatabaseOperationError,
    ExternalServiceError,
    FileOperationError,
    InvalidInputError,
    NotFoundError,
    OpenNotebookError,
    RateLimitError,
)

from surrealdb_service.migrations import AsyncMigrationManager

# Load .env file so all env vars (API keys, HF_TOKEN, etc.) are available
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler - runs DB migrations and starts job worker."""
    logger.info("Starting API initialization...")

    try:
        migration_manager = AsyncMigrationManager()
        current_version = await migration_manager.get_current_version()
        logger.info(f"Current database version: {current_version}")

        if await migration_manager.needs_migration():
            logger.warning("Database migrations are pending. Running migrations...")
            await migration_manager.run_migration_up()
            new_version = await migration_manager.get_current_version()
            logger.success(
                f"Migrations completed. Database at version {new_version}"
            )
        else:
            logger.info("Database is at the latest version.")
    except Exception as e:
        logger.error(f"CRITICAL: Database migration failed: {e}")
        logger.exception(e)
        raise RuntimeError(f"Failed to run database migrations: {e}") from e

    # Register job handlers and start the background worker
    import app_main.handlers  # noqa: F401 — triggers @registry.register()

    from app_main.services.command_service import start_worker, stop_worker

    await start_worker()

    # Load vault dictionary into LLMMatcher (if any accepted imports exist)
    try:
        from app_main.services.vault_sync_service import VaultSyncService

        vault_svc = VaultSyncService()
        dictionary = await vault_svc.get_merged_dictionary()
        if dictionary:
            from entity_filtering.resolution.llm_matcher import LLMMatcher

            LLMMatcher.set_abbreviations(dictionary)
            logger.info(f"Loaded {len(dictionary)} vault dictionary entries")

        # Optional: auto-sync vault on startup
        from app_main.dependencies import get_settings_service

        settings_svc = get_settings_service()
        settings = await settings_svc.get()
        if settings.vault_sync_on_startup and settings.vault_path:
            stats = await vault_svc.scan_vault_entities(
                settings.vault_path,
                settings.vault_entities_folder or "Entities",
            )
            logger.info(f"Vault auto-sync at startup: {stats}")
    except Exception as e:
        logger.warning(f"Vault dictionary load skipped: {e}")

    logger.success("API initialization completed successfully")

    yield

    await stop_worker()

    from surrealdb_service.connection import close_pool

    await close_pool()
    logger.info("API shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="Noesis API",
        description="API for Noesis - Research Assistant",
        version="0.2.2",
        lifespan=lifespan,
    )

    # --- Middleware ---
    application.add_middleware(
        PasswordAuthMiddleware,
        excluded_paths=[
            "/",
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/api/auth/status",
            "/api/config",
        ],
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Routers ---
    from app_main.api.routers import (
        auth,
        chat,
        commands,
        config,
        context,
        embedding,
        embedding_rebuild,
        episode_profiles,
        insights,
        knowledge_graph,
        models,
        notebooks,
        notes,
        ontologies,
        podcasts,
        preprocessing,
        search,
        settings,
        source_chat,
        sources,
        speaker_profiles,
        summaries,
        transformations,
        vault,
    )

    application.include_router(auth.router, prefix="/api", tags=["auth"])
    application.include_router(config.router, prefix="/api", tags=["config"])
    application.include_router(notebooks.router, prefix="/api", tags=["notebooks"])
    application.include_router(search.router, prefix="/api", tags=["search"])
    application.include_router(models.router, prefix="/api", tags=["models"])
    application.include_router(
        transformations.router, prefix="/api", tags=["transformations"]
    )
    application.include_router(notes.router, prefix="/api", tags=["notes"])
    application.include_router(embedding.router, prefix="/api", tags=["embedding"])
    application.include_router(
        embedding_rebuild.router, prefix="/api/embeddings", tags=["embeddings"]
    )
    application.include_router(settings.router, prefix="/api", tags=["settings"])
    application.include_router(context.router, prefix="/api", tags=["context"])
    application.include_router(sources.router, prefix="/api", tags=["sources"])
    application.include_router(insights.router, prefix="/api", tags=["insights"])
    application.include_router(commands.router, prefix="/api", tags=["commands"])
    application.include_router(podcasts.router, prefix="/api", tags=["podcasts"])
    application.include_router(
        episode_profiles.router, prefix="/api", tags=["episode-profiles"]
    )
    application.include_router(
        speaker_profiles.router, prefix="/api", tags=["speaker-profiles"]
    )
    application.include_router(chat.router, prefix="/api", tags=["chat"])
    application.include_router(source_chat.router, prefix="/api", tags=["source-chat"])
    application.include_router(ontologies.router, prefix="/api", tags=["ontologies"])
    application.include_router(
        knowledge_graph.router, prefix="/api", tags=["knowledge-graph"]
    )
    application.include_router(summaries.router, prefix="/api", tags=["summaries"])
    application.include_router(
        preprocessing.router, prefix="/api", tags=["preprocessing"]
    )
    application.include_router(vault.router, prefix="/api", tags=["vault"])

    # --- Exception Handlers ---
    @application.exception_handler(NotFoundError)
    async def not_found_handler(request: Request, exc: NotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @application.exception_handler(InvalidInputError)
    async def invalid_input_handler(request: Request, exc: InvalidInputError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @application.exception_handler(AuthenticationError)
    async def auth_error_handler(request: Request, exc: AuthenticationError):
        return JSONResponse(status_code=401, content={"detail": str(exc)})

    @application.exception_handler(RateLimitError)
    async def rate_limit_handler(request: Request, exc: RateLimitError):
        return JSONResponse(status_code=429, content={"detail": str(exc)})

    @application.exception_handler(DatabaseOperationError)
    async def db_error_handler(request: Request, exc: DatabaseOperationError):
        logger.error(f"Database error: {exc}")
        return JSONResponse(
            status_code=500, content={"detail": "Database operation failed"}
        )

    @application.exception_handler(ExternalServiceError)
    async def external_error_handler(
        request: Request, exc: ExternalServiceError
    ):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @application.exception_handler(ConfigurationError)
    async def config_error_handler(
        request: Request, exc: ConfigurationError
    ):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @application.exception_handler(FileOperationError)
    async def file_error_handler(request: Request, exc: FileOperationError):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @application.exception_handler(OpenNotebookError)
    async def generic_app_error_handler(
        request: Request, exc: OpenNotebookError
    ):
        logger.error(f"Unhandled app error: {exc}")
        return JSONResponse(
            status_code=500, content={"detail": "Internal server error"}
        )

    # --- Request Logging Middleware ---
    class RequestLoggingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            if request.url.path in ("/health", "/"):
                return await call_next(request)
            start = time.monotonic()
            response = await call_next(request)
            duration = time.monotonic() - start
            logger.info(
                f"{request.method} {request.url.path} -> "
                f"{response.status_code} ({duration:.3f}s)"
            )
            return response

    application.add_middleware(RequestLoggingMiddleware)

    # --- Routes ---
    @application.get("/")
    async def root():
        return {"message": "Noesis API is running"}

    @application.get("/health")
    async def health():
        return {"status": "healthy"}

    return application


app = create_app()


def main():
    """Entry point for the app-main script."""
    import os

    import uvicorn

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "5055"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"

    uvicorn.run(
        "app_main.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )
