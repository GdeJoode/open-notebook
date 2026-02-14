"""
FastAPI application factory for Open Notebook.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app_main.api.auth import PasswordAuthMiddleware

# Import migration manager from monolith until migrated to surrealdb-service
from open_notebook.database.async_migrate import AsyncMigrationManager


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
    logger.success("API initialization completed successfully")

    yield

    await stop_worker()
    logger.info("API shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="Open Notebook API",
        description="API for Open Notebook - Research Assistant",
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
        search,
        settings,
        source_chat,
        sources,
        speaker_profiles,
        summaries,
        transformations,
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

    @application.get("/")
    async def root():
        return {"message": "Open Notebook API is running"}

    @application.get("/health")
    async def health():
        return {"status": "healthy"}

    return application


app = create_app()


def main():
    """Entry point for the app-main script."""
    import uvicorn

    uvicorn.run(
        "app_main.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
