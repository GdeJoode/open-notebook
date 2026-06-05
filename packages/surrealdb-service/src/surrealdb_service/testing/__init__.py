"""Test utilities for the SurrealDB service.

Exposes the ``live_surrealdb`` pytest fixture so any workspace member can boot
an isolated SurrealDB container and exercise SCHEMAFULL migrations end-to-end.

Usage in a downstream ``conftest.py``::

    from surrealdb_service.testing import live_surrealdb  # noqa: F401

The fixture is session-scoped and idempotent — a single container is reused
for the whole pytest session. Mark tests that depend on it with
``@pytest.mark.requires_docker`` so they skip cleanly when Docker is absent.
"""

from surrealdb_service.testing.fixtures import (
    SURREALDB_IMAGE,
    docker_available,
    live_surrealdb,
)

__all__ = [
    "SURREALDB_IMAGE",
    "docker_available",
    "live_surrealdb",
]
