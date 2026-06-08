"""Shared service helpers callable from any workspace member.

The functions in this package are intentionally narrow:
they expose a small, stable API that can be safely imported from
``apps/*``, ``pipelines/*``, and other ``packages/*`` without pulling
the heavyweight surrealdb-service or app-main dependencies into the
caller's import graph. Each module lazy-imports its persistence
backend so circular-dep risk stays at zero.
"""

from shared.services.metrics import record_metric

__all__ = ["record_metric"]
