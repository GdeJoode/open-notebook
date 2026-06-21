"""Idempotent startup seed for the NVIDIA NIM route (Track J.4).

Seeds two things so a fresh install routes through the real cloud provider out
of the box:

  1. An ``nvidia`` ``model`` row for the validated NIM model
     (``mistralai/mistral-medium-3.5-128b``), if one does not already exist.
  2. A ``model_route`` row per LLM task whose ``provider_chain`` is
     ``[{nvidia}]`` (the resolver appends the local Ollama fallback at resolve
     time; J-Q3 ``nvidia -> local``), if no route row exists for that task.

Both steps are idempotent — re-running on a populated DB is a no-op:
  * the model row is matched by ``name`` + ``provider`` before creating;
  * ``ModelRouteRepository.upsert`` is query-by-task-then-update, but we only
    seed a task that has NO row yet, so operator-customized routes are never
    overwritten.

Best-effort + non-blocking: any failure is logged and swallowed so a seed
hiccup never blocks API startup (the resolver still has its hard-coded default
chain as a safety net).
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

_NIM_MODEL_NAME = "mistralai/mistral-medium-3.5-128b"
_NIM_PROVIDER = "nvidia"
_SEED_TASKS = ("entity_extraction", "summarization", "chat")


async def _ensure_nim_model_row() -> Optional[str]:
    """Create the NIM ``model`` row if absent; return its id (or ``None``)."""
    from surrealdb_service.connection import execute_query

    rows = await execute_query(
        "SELECT * FROM model WHERE name = $name AND provider = $provider LIMIT 1;",
        {"name": _NIM_MODEL_NAME, "provider": _NIM_PROVIDER},
    )
    if rows:
        existing = rows[0]
        model_id = existing.get("id")
        logger.debug(f"NIM model row already present: {model_id}")
        return str(model_id) if model_id is not None else None

    created = await execute_query(
        "CREATE model SET name = $name, provider = $provider, type = 'language', "
        "display_name = $display_name, is_local = false, is_enabled = true, "
        "supports_tools = true;",
        {
            "name": _NIM_MODEL_NAME,
            "provider": _NIM_PROVIDER,
            "display_name": "Mistral Medium 3.5 (NVIDIA NIM)",
        },
    )
    model_id = created[0].get("id") if created else None
    logger.info(f"Seeded NIM model row: {model_id}")
    return str(model_id) if model_id is not None else None


async def seed_nim_routes() -> None:
    """Seed the NIM model row + per-task default routes (idempotent)."""
    try:
        from shared.models.llm import ModelRoute

        from app_main.dependencies import get_model_route_repo

        model_id = await _ensure_nim_model_row()
        route_repo = get_model_route_repo()

        for task in _SEED_TASKS:
            existing = await route_repo.get_by_task(task)
            if existing is not None:
                # Operator-configured (or previously seeded) — never overwrite.
                continue
            entry = {"provider": _NIM_PROVIDER, "enabled": True}
            if model_id:
                entry["model_id"] = model_id
            await route_repo.upsert(
                ModelRoute(task=task, provider_chain=[entry], private_chain=[])
            )
            logger.info(f"Seeded default model_route for task={task} -> [nvidia]")
    except Exception as e:  # noqa: BLE001 — never block startup on the seed
        logger.warning(
            f"NIM route seed skipped ({e}); resolver falls back to its "
            "hard-coded default chain."
        )
