"""Live tests for AgentKeyService (Track G.1) — mint / authenticate / revoke.

@requires_docker: hits the agent_keys table (migration 76) in a testcontainer.
Locks the security posture: the plaintext is returned once and NOT stored (only
its hash), authenticate matches by hash, and a revoked key stops authenticating.
"""

from __future__ import annotations

import pytest
from app_main.services.agents.key_service import AgentKeyService, hash_key

pytestmark = pytest.mark.asyncio


@pytest.mark.requires_docker
async def test_mint_stores_hash_not_plaintext_and_authenticates(live_surrealdb):
    from surrealdb_service.connection import ensure_record_id, execute_query

    svc = AgentKeyService(config=live_surrealdb)
    public, plaintext = await svc.mint(agent_id="agent-x", permission="write")

    assert plaintext.startswith("ak_")
    assert public["permission"] == "write"
    # The plaintext is NOT stored anywhere; only its SHA-256 hash is.
    rows = await execute_query(
        "SELECT key_hash, key_prefix FROM $id",
        {"id": ensure_record_id(public["id"])},
        live_surrealdb,
    )
    assert rows[0]["key_hash"] == hash_key(plaintext)
    assert rows[0]["key_hash"] != plaintext
    assert rows[0]["key_prefix"] == plaintext[:10]

    # authenticate matches the plaintext (by hash) and rejects a wrong key.
    matched = await svc.authenticate(plaintext)
    assert matched is not None and str(matched["agent_id"]) == "agent-x"
    assert await svc.authenticate("ak_wrong") is None


@pytest.mark.requires_docker
async def test_revoked_key_stops_authenticating(live_surrealdb):
    svc = AgentKeyService(config=live_surrealdb)
    public, plaintext = await svc.mint(agent_id="agent-y", permission="read")
    assert await svc.authenticate(plaintext) is not None

    await svc.revoke(public["id"])
    assert await svc.authenticate(plaintext) is None  # revoked → no auth


@pytest.mark.requires_docker
async def test_list_never_exposes_secrets(live_surrealdb):
    svc = AgentKeyService(config=live_surrealdb)
    _, plaintext = await svc.mint(agent_id="agent-z", permission="read")
    rows = await svc.list()
    assert rows  # at least the one we minted
    for r in rows:
        assert "key_hash" not in r
        assert "key" not in r
        assert plaintext not in r.values()


@pytest.mark.requires_docker
async def test_get_returns_public_row_and_none_for_missing(live_surrealdb):
    # get() powers the G.4 audit-log drawer's key→agent_id resolution.
    svc = AgentKeyService(config=live_surrealdb)
    public, plaintext = await svc.mint(agent_id="agent-get", permission="read")

    got = await svc.get(public["id"])
    assert got is not None
    assert got["agent_id"] == "agent-get"
    # secret-free projection (no plaintext, no hash)
    assert "key_hash" not in got and "key" not in got
    assert plaintext not in got.values()

    # A missing / malformed / cross-table id → None (never another table's row).
    assert await svc.get("agent_keys:doesnotexist") is None
    assert await svc.get("not-a-record-id") is None
    # A real, EXISTING id on another table must not leak that row (table-scoped).
    from surrealdb_service.connection import execute_query

    nb = await execute_query(
        "CREATE notebook SET name = 'not-a-key'", None, live_surrealdb
    )
    assert await svc.get(str(nb[0]["id"])) is None


@pytest.mark.requires_docker
async def test_revoke_is_table_scoped(live_surrealdb):
    # revoke() must never flip `revoked` on a non-agent_keys row (cross-table).
    from surrealdb_service.connection import execute_query

    svc = AgentKeyService(config=live_surrealdb)
    created = await execute_query(
        "CREATE notebook SET name = 'not-a-key'", None, live_surrealdb
    )
    nb_id = str(created[0]["id"])

    updated = await svc.revoke(nb_id)  # a notebook id, not an agent key
    assert updated is False  # matched nothing on agent_keys

    from surrealdb_service.connection import ensure_record_id

    rows = await execute_query(
        "SELECT revoked FROM $id",
        {"id": ensure_record_id(nb_id)},
        live_surrealdb,
    )
    # The cross-table write did NOT happen: the notebook's revoked stays unset
    # (SurrealDB projects a missing field back as None), never flipped to True.
    assert rows[0].get("revoked") is not True
