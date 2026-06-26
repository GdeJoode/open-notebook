"""Unit tests for ``GET /sources/{id}/related`` (Track R.1).

Exercise the router contract with the service mocked: ordering pass-through,
k bounds/validation, the no-aggregate-embedding empty case, and 404 for a
missing source. The cosine ranking itself is covered by the DB roundtrip test
(``test_source_related_db.py``) and the live staging smoke.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.sources import router
from app_main.dependencies import get_source_service


def _make_client(svc) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_source_service] = lambda: svc
    return TestClient(app)


def _svc(*, source=SimpleNamespace(id="source:abc"), related=None):
    svc = SimpleNamespace()
    svc.get = AsyncMock(return_value=source)
    svc.find_related = AsyncMock(return_value=related or [])
    return svc


class TestRelatedSourcesEndpoint:
    def test_returns_ranked_related_excluding_self(self):
        """AC1: top-k returned in the service's (cosine-desc) order."""
        ranked = [
            {"id": "source:b", "title": "Beta", "score": 0.99},
            {"id": "source:c", "title": "Gamma", "score": 0.80},
        ]
        svc = _svc(related=ranked)
        client = _make_client(svc)

        resp = client.get("/api/sources/source:abc/related?k=5")

        assert resp.status_code == 200
        data = resp.json()
        assert [r["id"] for r in data] == ["source:b", "source:c"]
        assert data[0]["score"] >= data[1]["score"]
        # self never appears (the repo excludes it; the queried id is not here)
        assert "source:abc" not in [r["id"] for r in data]
        # default k forwarded; service called with the parsed k
        svc.find_related.assert_awaited_once_with("source:abc", 5)

    def test_default_k_is_five(self):
        svc = _svc(related=[])
        client = _make_client(svc)
        resp = client.get("/api/sources/source:abc/related")
        assert resp.status_code == 200
        svc.find_related.assert_awaited_once_with("source:abc", 5)

    def test_k_below_one_rejected(self):
        """AC2: k is validated (FastAPI 422 below the lower bound)."""
        svc = _svc(related=[])
        client = _make_client(svc)
        resp = client.get("/api/sources/source:abc/related?k=0")
        assert resp.status_code == 422

    def test_k_above_max_rejected(self):
        """AC2: k is bounded — above 50 is rejected, never an unbounded scan."""
        svc = _svc(related=[])
        client = _make_client(svc)
        resp = client.get("/api/sources/source:abc/related?k=51")
        assert resp.status_code == 422

    def test_k_at_upper_bound_accepted(self):
        svc = _svc(related=[])
        client = _make_client(svc)
        resp = client.get("/api/sources/source:abc/related?k=50")
        assert resp.status_code == 200
        svc.find_related.assert_awaited_once_with("source:abc", 50)

    def test_more_than_available_returns_all(self):
        """AC2: requesting more than exist returns however many there are."""
        ranked = [{"id": "source:b", "title": "Beta", "score": 0.9}]
        svc = _svc(related=ranked)
        client = _make_client(svc)
        resp = client.get("/api/sources/source:abc/related?k=50")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_source_without_aggregate_returns_empty(self):
        """AC3: a source that exists but has no aggregate embedding -> []."""
        # Source exists; service returns [] (repo found NONE embedding).
        svc = _svc(source=SimpleNamespace(id="source:noemb"), related=[])
        client = _make_client(svc)
        resp = client.get("/api/sources/source:noemb/related")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_missing_source_returns_404(self):
        """AC3: a genuinely missing source is a 404, not an empty 200."""
        svc = _svc(source=None)
        client = _make_client(svc)
        resp = client.get("/api/sources/source:ghost/related")
        assert resp.status_code == 404
        # find_related is never reached for a missing source
        svc.find_related.assert_not_awaited()

    def test_service_error_surfaces_as_500(self):
        svc = SimpleNamespace()
        svc.get = AsyncMock(return_value=SimpleNamespace(id="source:abc"))
        svc.find_related = AsyncMock(side_effect=RuntimeError("db down"))
        client = _make_client(svc)
        resp = client.get("/api/sources/source:abc/related")
        assert resp.status_code == 500
