"""Tests for API application."""

import pytest
from fastapi.testclient import TestClient

from surrealdb_service.api.app import create_app


class TestAPIApp:
    """Tests for the FastAPI application."""

    def test_create_app(self):
        """Test that create_app returns a FastAPI instance."""
        app = create_app()
        assert app is not None
        assert app.title == "SurrealDB Service"

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        app = create_app()
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "surrealdb-service"

    def test_routers_included(self):
        """Test that all routers are included."""
        app = create_app()
        routes = [route.path for route in app.routes]

        # Check that router prefixes are present
        assert any("/notebooks" in route for route in routes)
        assert any("/sources" in route for route in routes)
        assert any("/settings" in route for route in routes)
        assert any("/search" in route for route in routes)

    def test_cors_middleware(self):
        """Test that CORS middleware is configured."""
        app = create_app()
        # Check for CORS middleware in app middleware
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes
