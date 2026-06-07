"""Job-queue-level exceptions that the worker treats specially.

Handlers raise these to signal "the job did not fail, it is parked"
without coupling the worker to handler-specific business logic.

Concrete example: ``app_main.services.entity_extraction_service.
SchemaReviewPendingError`` (B.1f) subclasses
:class:`JobPausedForReviewError` so that when a notebook's pending
schema extensions need user review, the worker transitions the job to
:class:`shared.types.enums.JobStatus.PAUSED_FOR_REVIEW` instead of
``FAILED`` (which would retry up to ``max_retries`` and eventually
dead-letter).
"""


class JobPausedForReviewError(Exception):
    """Raised by a job handler when execution must pause for user review.

    The worker catches this exception and updates the job's status to
    ``PAUSED_FOR_REVIEW`` rather than ``FAILED`` / ``RETRYING``.
    Subclasses should expose attributes carrying enough context for the
    UI to render a review prompt.
    """

    def __init__(self, message: str = "Job paused pending user review"):
        super().__init__(message)
