"""Shared pytest config for surrealdb-service tests.

Re-exports the ``live_surrealdb`` fixture so test modules can request it by
name without importing from ``surrealdb_service.testing`` directly. Downstream
workspace members (e.g. ``apps/app-main/tests``) should import the fixture in
their own conftest the same way.
"""

# noqa: F401 — fixtures are picked up by pytest via re-export
from surrealdb_service.testing import live_surrealdb  # noqa: F401
