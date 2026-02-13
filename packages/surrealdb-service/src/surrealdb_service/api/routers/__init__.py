"""
API routers for SurrealDB service.
"""

from surrealdb_service.api.routers import notebooks, search, settings, sources

__all__ = ["notebooks", "sources", "settings", "search"]
