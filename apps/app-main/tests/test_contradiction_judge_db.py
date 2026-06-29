"""Track Z.2 — DB-backed: the judge actually writes (and converges on) a
``source_verdict`` edge through Z.1's idempotent ``relate_verdict``.

The unit tests (``test_contradiction_judge_service``) prove the SERVICE drives
the precision gate with mocked collaborators. This file closes the persistence
ACs end-to-end against a REAL SurrealDB container: a real ``SourceService`` +
``SourceRepository`` over the live DB, the LLM still MOCKED (no live model).

Proves:
  * a confident ``contradicts`` verdict writes exactly ONE ``source_verdict``
    edge carrying the verdict/confidence/reasoning;
  * idempotency — re-judging the same pair (clear-before-relate in Z.1) leaves
    exactly one edge with the LATEST verdict, never a duplicate;
  * a neutral / below-threshold verdict writes NO edge (precision-first, on the
    real DB this time);
  * canonical ``source`` rows are untouched by judging.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest
from app_main.services.contradiction_judge_service import (
    ContradictionJudgeService,
)
from app_main.services.source_service import SourceService
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.repositories import (
    ChunkRepository,
    SourceEmbeddingRepository,
    SourceInsightRepository,
    SourceRepository,
)

pytestmark = pytest.mark.requires_docker


def _u(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


async def _make_source(config: SurrealDBConfig, *, title: str | None = None) -> str:
    rows = await execute_query(
        "CREATE source SET title = $t, full_text = $t RETURN AFTER;",
        {"t": title or _u("source")},
        config=config,
    )
    return str(rows[0]["id"])


async def _edge_rows(
    config: SurrealDBConfig, from_id: str, to_id: str
) -> list[dict]:
    return await execute_query(
        "SELECT in, out, verdict, confidence, reasoning, judge_model "
        "FROM source_verdict WHERE in = $f AND out = $t",
        {"f": ensure_record_id(from_id), "t": ensure_record_id(to_id)},
        config=config,
    )


def _service(config: SurrealDBConfig, llm_response: str) -> ContradictionJudgeService:
    """A real judge over the live DB with a fixed (mocked) LLM response.

    ``related_service`` is irrelevant for ``judge_pair`` (it takes the pair
    directly), so a bare mock suffices.
    """
    source_service = SourceService(
        source_repo=SourceRepository(config),
        chunk_repo=ChunkRepository(config),
        insight_repo=SourceInsightRepository(config),
        embedding_repo=SourceEmbeddingRepository(config),
    )
    return ContradictionJudgeService(
        source_service=source_service,
        related_service=AsyncMock(),
        source_repo=SourceRepository(config),
        llm_caller=AsyncMock(return_value=llm_response),
        judge_model="test-judge",
    )


@pytest.mark.asyncio
async def test_confident_contradicts_writes_one_edge(
    live_surrealdb: SurrealDBConfig,
) -> None:
    a = await _make_source(live_surrealdb)
    b = await _make_source(live_surrealdb)
    svc = _service(
        live_surrealdb,
        '{"verdict": "contradicts", "confidence": 0.93, "reasoning": "A vs B"}',
    )

    verdict, written = await svc.judge_pair(a, b)
    assert verdict.verdict == "contradicts"
    assert written is True

    rows = await _edge_rows(live_surrealdb, a, b)
    assert len(rows) == 1
    assert rows[0]["verdict"] == "contradicts"
    assert abs(float(rows[0]["confidence"]) - 0.93) < 1e-9
    assert rows[0]["reasoning"] == "A vs B"
    assert rows[0]["judge_model"] == "test-judge"


@pytest.mark.asyncio
async def test_rejudge_is_idempotent_latest_verdict(
    live_surrealdb: SurrealDBConfig,
) -> None:
    a = await _make_source(live_surrealdb)
    b = await _make_source(live_surrealdb)

    svc1 = _service(
        live_surrealdb,
        '{"verdict": "reinforces", "confidence": 0.8, "reasoning": "first"}',
    )
    await svc1.judge_pair(a, b)

    # Re-judge the SAME pair with a different confident verdict.
    svc2 = _service(
        live_surrealdb,
        '{"verdict": "contradicts", "confidence": 0.95, "reasoning": "second"}',
    )
    _, written = await svc2.judge_pair(a, b)
    assert written is True

    rows = await _edge_rows(live_surrealdb, a, b)
    assert len(rows) == 1, "re-judging a pair must not duplicate the edge"
    assert rows[0]["verdict"] == "contradicts"
    assert rows[0]["reasoning"] == "second"


@pytest.mark.asyncio
async def test_neutral_writes_no_edge_on_real_db(
    live_surrealdb: SurrealDBConfig,
) -> None:
    a = await _make_source(live_surrealdb)
    b = await _make_source(live_surrealdb)
    svc = _service(
        live_surrealdb,
        '{"verdict": "neutral", "confidence": 0.99, "reasoning": "same topic"}',
    )
    _, written = await svc.judge_pair(a, b)
    assert written is False
    assert await _edge_rows(live_surrealdb, a, b) == []


@pytest.mark.asyncio
async def test_below_threshold_writes_no_edge_on_real_db(
    live_surrealdb: SurrealDBConfig,
) -> None:
    a = await _make_source(live_surrealdb)
    b = await _make_source(live_surrealdb)
    svc = _service(
        live_surrealdb,
        '{"verdict": "contradicts", "confidence": 0.4, "reasoning": "weak"}',
    )
    _, written = await svc.judge_pair(a, b)
    assert written is False
    assert await _edge_rows(live_surrealdb, a, b) == []


@pytest.mark.asyncio
async def test_json_array_response_writes_no_edge_on_real_db(
    live_surrealdb: SurrealDBConfig,
) -> None:
    # The blocker, end-to-end on the real DB: a top-level array containing a
    # confident contradicts object must NOT create a source_verdict row.
    a = await _make_source(live_surrealdb)
    b = await _make_source(live_surrealdb)
    svc = _service(
        live_surrealdb,
        '[{"verdict": "contradicts", "confidence": 0.99, "reasoning": "x"}]',
    )
    verdict, written = await svc.judge_pair(a, b)
    assert verdict.verdict == "neutral"
    assert written is False
    assert await _edge_rows(live_surrealdb, a, b) == []


@pytest.mark.asyncio
async def test_canonical_source_rows_untouched(
    live_surrealdb: SurrealDBConfig,
) -> None:
    a = await _make_source(live_surrealdb, title="keep-me")
    b = await _make_source(live_surrealdb)
    svc = _service(
        live_surrealdb,
        '{"verdict": "contradicts", "confidence": 0.9, "reasoning": "x"}',
    )
    await svc.judge_pair(a, b)

    rows = await execute_query(
        "SELECT title FROM $id",
        {"id": ensure_record_id(a)},
        config=live_surrealdb,
    )
    assert rows[0]["title"] == "keep-me"
