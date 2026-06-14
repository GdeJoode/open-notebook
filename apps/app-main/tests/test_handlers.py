"""Tests for job-queue handler registration (apps/app-main/handlers.py).

Focus: confirm that importing ``app_main.handlers`` actually wires the
handlers into the global ``HandlerRegistry``. The handlers themselves
have unit tests next to their services; this file is a structural
check that ensures the side-effect registration didn't silently break.

D.1b adds ``JobType.EXPORT_OBSIDIAN`` as the auto-pipeline entry
point for the Obsidian vault export. The test below pins that the
import-side registration takes effect.
"""

from __future__ import annotations

# Importing handlers triggers the @registry.register side-effects.
import app_main.handlers  # noqa: F401 -- import for side-effects

from app_main.handlers import registry
from shared.types.enums import JobType


def test_export_obsidian_handler_registered():
    """``JobType.EXPORT_OBSIDIAN`` resolves to a handler after import (D.1b).

    The auto-pipeline entry point relies on the worker finding a
    handler for this job type. If a handler regresses out of the
    decorator chain, this test catches it before a job hits PROD.
    """
    assert registry.has_handler(JobType.EXPORT_OBSIDIAN), (
        f"No handler registered for {JobType.EXPORT_OBSIDIAN.value!r}. "
        f"Currently registered: "
        f"{[t.value for t in registry.registered_types]}"
    )


def test_existing_handlers_still_registered():
    """Sanity: D.1b didn't break the existing handler registrations.

    Lists the handlers we expect to be present after importing
    handlers.py. If a future migration adds/removes a job type, this
    list must update too.
    """
    expected = {
        JobType.DOCUMENT_PARSE,
        JobType.BATCH_PROCESS,
        JobType.INSIGHT_EXTRACT,
        JobType.ENTITY_EXTRACT,
        JobType.EMBEDDING_GENERATE,
        JobType.EXPORT_OBSIDIAN,  # D.1b
    }
    actual = set(registry.registered_types)
    missing = expected - actual
    assert not missing, f"Handlers missing after import: {missing}"
