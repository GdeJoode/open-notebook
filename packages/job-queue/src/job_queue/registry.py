"""
Handler registry mapping JobType to async callables.

Consuming packages register their handlers at app startup:

    from job_queue.registry import HandlerRegistry

    registry = HandlerRegistry()

    @registry.register(JobType.DOCUMENT_PARSE)
    async def handle_document_parse(payload: dict) -> dict:
        ...
"""

from typing import Any, Callable, Coroutine, Dict

from loguru import logger

from shared.types.enums import JobType

# Type alias for handler functions: async (payload) -> result dict
HandlerFn = Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]]


class HandlerRegistry:
    """Maps JobType values to async handler functions."""

    def __init__(self) -> None:
        self._handlers: Dict[JobType, HandlerFn] = {}

    def register(self, job_type: JobType) -> Callable[[HandlerFn], HandlerFn]:
        """
        Decorator to register a handler for a job type.

        Usage::

            @registry.register(JobType.DOCUMENT_PARSE)
            async def handle_parse(payload: dict) -> dict:
                ...
        """

        def decorator(fn: HandlerFn) -> HandlerFn:
            if job_type in self._handlers:
                logger.warning(
                    f"Overwriting handler for {job_type.value}: "
                    f"{self._handlers[job_type].__name__} -> {fn.__name__}"
                )
            self._handlers[job_type] = fn
            return fn

        return decorator

    def register_handler(self, job_type: JobType, fn: HandlerFn) -> None:
        """Register a handler imperatively (non-decorator form)."""
        if job_type in self._handlers:
            logger.warning(
                f"Overwriting handler for {job_type.value}: "
                f"{self._handlers[job_type].__name__} -> {fn.__name__}"
            )
        self._handlers[job_type] = fn

    async def execute(self, job_type: JobType, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Look up and call the handler for this job type.

        Raises:
            KeyError: If no handler is registered for the job type.
        """
        handler = self._handlers.get(job_type)
        if handler is None:
            raise KeyError(f"No handler registered for job type: {job_type.value}")
        return await handler(payload)

    def has_handler(self, job_type: JobType) -> bool:
        """Check whether a handler is registered for the given job type."""
        return job_type in self._handlers

    @property
    def registered_types(self) -> list[JobType]:
        """List of job types that have registered handlers."""
        return list(self._handlers.keys())
