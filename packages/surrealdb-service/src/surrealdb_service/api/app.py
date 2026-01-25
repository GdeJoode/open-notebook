"""
FastAPI application for SurrealDB service REST API.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from surrealdb_service.api.routers import notebooks, search, settings, sources


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="SurrealDB Service",
        description="REST API for SurrealDB database operations",
        version="0.1.0",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(notebooks.router, prefix="/notebooks", tags=["notebooks"])
    app.include_router(sources.router, prefix="/sources", tags=["sources"])
    app.include_router(settings.router, prefix="/settings", tags=["settings"])
    app.include_router(search.router, prefix="/search", tags=["search"])

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "surrealdb-service"}

    return app


def main():
    """Run the FastAPI application."""
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
