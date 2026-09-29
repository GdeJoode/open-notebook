"""A pool that cannot hand out a connection answers 503, not a hang (WB.0)."""

from fastapi.testclient import TestClient
from surrealdb_service.connection import DatabaseUnavailableError

from app_main.api.app import create_app


def test_database_unavailable_is_503_with_retry_after():
    app = create_app()

    @app.get("/api/_wb0_probe")
    async def probe():
        raise DatabaseUnavailableError("No SurrealDB connection free within 30.0s")

    resp = TestClient(app).get("/api/_wb0_probe")

    assert resp.status_code == 503
    assert resp.headers["Retry-After"] == "5"
    assert "No SurrealDB connection free" in resp.json()["detail"]
