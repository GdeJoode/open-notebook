"""
Unit tests for the reranker service.

The cross-encoder is mocked so these run without the ~2 GB model download:
we inject a fake model into ``api._model`` and assert the ordering logic and
response schema, not the model's quality.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# The service runs with /app on sys.path (api.py is copied to /app/api.py in the
# container). Mirror that here so ``import api`` resolves to services/reranker/api.py.
SERVICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVICE_DIR))

import api  # noqa: E402


class FakeCrossEncoder:
    """Stub cross-encoder returning a fixed score per passage.

    ``predict`` receives ``[[query, passage], ...]`` and returns one score per
    pair. We score by the length of the passage so ordering is deterministic
    and independent of any real model.
    """

    def __init__(self, scores_by_passage=None):
        self._scores_by_passage = scores_by_passage or {}

    def predict(self, pairs):
        out = []
        for _query, passage in pairs:
            if passage in self._scores_by_passage:
                out.append(self._scores_by_passage[passage])
            else:
                out.append(float(len(passage)))
        return out


@pytest.fixture
def client(monkeypatch):
    """TestClient with model warmup skipped and a fake model injected."""
    monkeypatch.setenv("RERANKER_SKIP_WARMUP", "1")
    # Reset and inject the fake so get_model() never imports sentence_transformers.
    api._model = None
    with TestClient(app=api.app) as c:
        yield c
    api._model = None


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "reranker"
    assert "model" in body


def test_rerank_orders_by_score_desc(client):
    # Highest score should come first; index maps back to the input list.
    fake = FakeCrossEncoder(
        scores_by_passage={
            "low relevance": 0.1,
            "high relevance": 0.9,
            "medium relevance": 0.5,
        }
    )
    api._model = fake

    resp = client.post(
        "/rerank",
        json={
            "query": "q",
            "passages": ["low relevance", "high relevance", "medium relevance"],
        },
    )
    assert resp.status_code == 200
    results = resp.json()["results"]

    # Schema: list of {index, score}, sorted by score desc.
    assert [r["index"] for r in results] == [1, 2, 0]
    assert [round(r["score"], 3) for r in results] == [0.9, 0.5, 0.1]
    # Monotonic non-increasing scores.
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rerank_top_k_truncates(client):
    api._model = FakeCrossEncoder(
        scores_by_passage={"a": 3.0, "b": 1.0, "c": 2.0}
    )
    resp = client.post(
        "/rerank",
        json={"query": "q", "passages": ["a", "b", "c"], "top_k": 2},
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    assert [r["index"] for r in results] == [0, 2]  # a (3.0), c (2.0)


def test_rerank_empty_passages(client):
    api._model = FakeCrossEncoder()
    resp = client.post("/rerank", json={"query": "q", "passages": []})
    assert resp.status_code == 200
    assert resp.json()["results"] == []
