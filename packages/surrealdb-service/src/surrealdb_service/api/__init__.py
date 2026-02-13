"""
FastAPI application for SurrealDB service.
"""

from surrealdb_service.api.app import create_app

__all__ = ["create_app"]
