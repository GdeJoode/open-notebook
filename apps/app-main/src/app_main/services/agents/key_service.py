"""Agent API-key management (Track G.1).

Mint / list / revoke / authenticate agent keys. Security posture (G-D2):

* the plaintext key is generated once, returned ONCE from :meth:`mint`, and never
  stored — only its SHA-256 hash (``key_hash``, UNIQUE) lands in the DB;
* :meth:`authenticate` hashes the presented plaintext and looks it up by that hash
  (a single indexed hit), returning the row only when it is not revoked;
* listings expose ``key_prefix`` (a non-secret display fragment) + metadata, never
  the plaintext or the hash.

Key format: ``ak_<43 url-safe chars>`` (``secrets.token_urlsafe(32)``), so a leaked
log line is recognizable (``ak_`` prefix) and the entropy is 256 bits.
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query

_KEY_PREFIX = "ak_"
#: How many leading chars of the plaintext to keep as a non-secret display prefix.
_DISPLAY_PREFIX_LEN = 10


def hash_key(plaintext: str) -> str:
    """SHA-256 hex digest of a key's plaintext (what is stored at rest)."""
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def generate_key() -> str:
    """A fresh plaintext agent key: ``ak_`` + 256 bits of url-safe entropy."""
    return _KEY_PREFIX + secrets.token_urlsafe(32)


def _to_public(row: Dict[str, Any]) -> Dict[str, Any]:
    """Project a DB row to the public (secret-free) shape."""
    return {
        "id": str(row.get("id")),
        "agent_id": row.get("agent_id"),
        "key_prefix": row.get("key_prefix"),
        "label": row.get("label"),
        "permission": row.get("permission"),
        "rate_limit_rpm": row.get("rate_limit_rpm"),
        "revoked": bool(row.get("revoked", False)),
        "created": str(row.get("created")) if row.get("created") else None,
        "last_used_at": (
            str(row.get("last_used_at")) if row.get("last_used_at") else None
        ),
    }


class AgentKeyService:
    """Create, list, revoke, and authenticate agent keys (G.1)."""

    def __init__(self, config: Optional[SurrealDBConfig] = None) -> None:
        self.config = config

    async def mint(
        self,
        *,
        agent_id: str,
        permission: str = "read",
        label: Optional[str] = None,
        rate_limit_rpm: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """Mint a key. Returns ``(public_row, plaintext)`` — plaintext shown ONCE."""
        if permission not in ("read", "write", "admin"):
            raise ValueError(f"invalid permission: {permission!r}")
        plaintext = generate_key()
        rows = await execute_query(
            "CREATE agent_keys CONTENT $data",
            {
                "data": {
                    "agent_id": agent_id,
                    "key_hash": hash_key(plaintext),
                    "key_prefix": plaintext[:_DISPLAY_PREFIX_LEN],
                    "label": label,
                    "permission": permission,
                    "rate_limit_rpm": rate_limit_rpm,
                    "revoked": False,
                }
            },
            self.config,
        )
        row = rows[0] if rows else {}
        return _to_public(row), plaintext

    async def list(self) -> List[Dict[str, Any]]:
        """All keys as public rows (no secrets), newest first."""
        rows = await execute_query(
            "SELECT * FROM agent_keys ORDER BY created DESC", None, self.config
        )
        return [_to_public(r) for r in (rows or [])]

    async def get(self, key_id: str) -> Optional[Dict[str, Any]]:
        """One key as a public row (no secrets), or ``None``.

        Table-SCOPED select by id: a non-``agent_keys`` id (or a malformed one)
        matches nothing and returns ``None`` rather than reaching another table.
        """
        try:
            rid = ensure_record_id(key_id)
        except Exception:
            return None
        rows = await execute_query(
            "SELECT * FROM agent_keys WHERE id = $id LIMIT 1", {"id": rid}, self.config
        )
        return _to_public(rows[0]) if rows else None

    async def revoke(self, key_id: str) -> bool:
        """Revoke a key by id. Idempotent; returns whether a row was updated.

        Table-SCOPED (``UPDATE agent_keys … WHERE id = $id``) so a caller passing a
        non-agent_keys id (e.g. ``notebook:abc``) can never flip ``revoked`` on an
        unrelated table's row — it simply matches nothing.
        """
        try:
            rid = ensure_record_id(key_id)
        except Exception:
            # A malformed / colon-less id parses to nothing — honour the
            # idempotent bool contract (no match) instead of raising.
            return False
        rows = await execute_query(
            "UPDATE agent_keys SET revoked = true WHERE id = $id RETURN AFTER",
            {"id": rid},
            self.config,
        )
        return bool(rows)

    async def authenticate(self, plaintext: str) -> Optional[Dict[str, Any]]:
        """Return the non-revoked key row matching ``plaintext``, else ``None``.

        Looks up by SHA-256 hash (the UNIQUE ``ak_hash`` index) and best-effort
        stamps ``last_used_at``. Never returns a revoked row.
        """
        if not plaintext:
            return None
        # `revoked = false` (not `!= true`): a drift row with revoked = NONE would
        # satisfy `!= true` and authenticate — fail-open. Explicit `= false` is
        # drift-safe (the DEFAULT false on a clean schema makes this exact).
        rows = await execute_query(
            "SELECT * FROM agent_keys WHERE key_hash = $h AND revoked = false LIMIT 1",
            {"h": hash_key(plaintext)},
            self.config,
        )
        if not rows:
            return None
        row = rows[0]
        try:
            await execute_query(
                "UPDATE $id SET last_used_at = time::now()",
                {"id": ensure_record_id(str(row.get("id")))},
                self.config,
            )
        except Exception as exc:  # noqa: BLE001 — last_used stamp is best-effort
            logger.debug("agent key last_used_at stamp failed: {e}", e=exc)
        return row


__all__ = ["AgentKeyService", "generate_key", "hash_key"]
