"""Provider-error -> failover-eligibility mapping (Track J.4).

:func:`is_failover_eligible` is the **whitelist** predicate the J.2 failover
executor uses to decide whether a candidate's failure should advance to the next
provider (transient/provider faults) or propagate immediately (programming bugs
that would fail on every provider). It is injected into the executor at the J.4
wiring seam (the executor left ``is_failover_eligible`` injectable).

What the predicate must classify
--------------------------------
The cloud provider is **NVIDIA NIM** (OpenAI-compatible REST). esperanto talks
to it via its ``openai-compatible`` LanguageModel, which uses ``httpx`` directly
and raises:

  * ``RuntimeError("OpenAI-compatible endpoint error: ...")`` (or
    ``"OpenAI API error: ..."`` / ``"Anthropic API error: ..."``) on any HTTP
    status >= 400. The HTTP status is only embedded in the message string when
    the body is not JSON-parseable; otherwise the message is the provider's
    ``error.message`` text.
  * ``httpx`` transport exceptions (``ConnectError``, ``ConnectTimeout``,
    ``ReadTimeout``, ``TimeoutException``, ``RemoteProtocolError``, ...) on
    connection/read failures.

We also map the ``openai`` SDK exception hierarchy defensively, because a
different esperanto version (or a future native NVIDIA provider) may surface the
SDK's structured errors instead of a bare ``RuntimeError``.

Whitelist (ELIGIBLE -> fail over to the next provider)
  * 429 rate limit
  * 5xx server error
  * connection / timeout
  * 401 / 403 auth-misconfig (drop this provider + trip its breaker; another
    provider may be correctly configured)

NOT eligible (propagate — would fail on every provider)
  * 400 / 404 / 422 bad request (our prompt/payload is wrong)
  * programming errors: ``TypeError`` / ``ValueError`` / ``KeyError`` / ...
  * an esperanto ``RuntimeError`` whose message does not look like a provider
    HTTP fault (we do not blanket-whitelist ``RuntimeError`` — that would mask
    real bugs that also raise it).
"""

from __future__ import annotations

import re

# ``httpx`` is always present (it is esperanto's transport). ``openai`` is an
# optional defensive mapping — guard the import so the predicate degrades to the
# httpx + RuntimeError paths if the SDK is ever absent.
import httpx

try:  # pragma: no cover - exercised indirectly; import-guard only
    import openai as _openai
except Exception:  # noqa: BLE001
    _openai = None  # type: ignore[assignment]


# Auth/rate/server statuses that should fail over; 400/404/422 deliberately
# absent (a bad request fails identically on every provider).
_ELIGIBLE_STATUSES = {401, 403, 429, 500, 502, 503, 504}

# esperanto wraps provider HTTP errors as RuntimeError with one of these
# prefixes. We only treat a RuntimeError as a provider fault when its message
# matches this shape — a bare RuntimeError from our own code stays non-eligible.
_ESPERANTO_HTTP_ERROR_RE = re.compile(
    r"(API error|endpoint error|HTTP\s+\d{3})", re.IGNORECASE
)
# Pull an HTTP status out of the message when esperanto embedded it (the
# non-JSON-body branch renders "HTTP 503: ...").
_STATUS_IN_MESSAGE_RE = re.compile(r"\b(\d{3})\b")
# Transient phrases NIM/OpenAI put in JSON error bodies (where no status is
# embedded in the esperanto message). These map to eligible failover.
_TRANSIENT_PHRASE_RE = re.compile(
    r"(rate limit|too many requests|over capacity|overloaded|"
    r"server error|service unavailable|temporarily|timeout|timed out|"
    r"try again|503|502|504|500|429)",
    re.IGNORECASE,
)


def _status_from_exc(exc: Exception) -> int | None:
    """Best-effort HTTP status extraction from an exception.

    Checks the common structured attributes (openai SDK ``status_code`` /
    ``response.status_code``, httpx ``response.status_code``) before falling
    back to ``None`` (the caller then inspects the message text).
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    resp_status = getattr(response, "status_code", None)
    if isinstance(resp_status, int):
        return resp_status
    return None


def _openai_sdk_eligible(exc: Exception) -> bool | None:
    """Classify ``openai`` SDK exceptions, or ``None`` if not an SDK error.

    Returns the eligibility decision for the structured SDK hierarchy and
    ``None`` when ``exc`` is not an openai SDK exception (so the caller continues
    to the httpx / RuntimeError checks).
    """
    if _openai is None:
        return None

    # Eligible: transient + auth-misconfig.
    eligible_types = tuple(
        t
        for t in (
            getattr(_openai, "RateLimitError", None),
            getattr(_openai, "APITimeoutError", None),
            getattr(_openai, "APIConnectionError", None),
            getattr(_openai, "InternalServerError", None),
            getattr(_openai, "AuthenticationError", None),
            getattr(_openai, "PermissionDeniedError", None),
        )
        if isinstance(t, type)
    )
    if eligible_types and isinstance(exc, eligible_types):
        return True

    # Non-eligible client errors: bad request, not found, unprocessable.
    not_eligible_types = tuple(
        t
        for t in (
            getattr(_openai, "BadRequestError", None),
            getattr(_openai, "NotFoundError", None),
            getattr(_openai, "UnprocessableEntityError", None),
            getattr(_openai, "ConflictError", None),
        )
        if isinstance(t, type)
    )
    if not_eligible_types and isinstance(exc, not_eligible_types):
        return False

    # A generic APIStatusError: decide by its HTTP status.
    api_status_error = getattr(_openai, "APIStatusError", None)
    if isinstance(api_status_error, type) and isinstance(exc, api_status_error):
        status = _status_from_exc(exc)
        if status is not None:
            return status in _ELIGIBLE_STATUSES
        return True  # unknown status on a transport-level API error -> fail over

    api_error = getattr(_openai, "APIError", None)
    if isinstance(api_error, type) and isinstance(exc, api_error):
        # Base APIError without a more specific subclass match — treat as a
        # transport/server fault and fail over.
        return True

    return None


def _runtime_error_eligible(exc: RuntimeError) -> bool:
    """Classify esperanto's ``RuntimeError`` HTTP wrapper.

    esperanto raises ``RuntimeError("<provider> ... error: <body or HTTP NNN>")``
    for every status >= 400. We:
      1. Require the message to look like an esperanto HTTP wrapper (so a bare
         ``RuntimeError`` from our own code is NOT whitelisted).
      2. If a 3-digit status is embedded, decide by ``_ELIGIBLE_STATUSES``.
      3. Otherwise (JSON error body, no status in the message), decide by a
         transient-phrase match — a NIM 429/5xx body says "rate limit" /
         "service unavailable" while a 400 says "invalid"/"bad request".
    """
    message = str(exc)
    if not _ESPERANTO_HTTP_ERROR_RE.search(message):
        return False

    status_match = _STATUS_IN_MESSAGE_RE.search(message)
    if status_match:
        status = int(status_match.group(1))
        if 400 <= status <= 599:
            return status in _ELIGIBLE_STATUSES

    # No embedded status: lean on transient-phrase detection. A bad-request body
    # ("invalid model", "unknown field") has none of these phrases and stays
    # non-eligible, so we don't fail a malformed-prompt over every provider.
    return bool(_TRANSIENT_PHRASE_RE.search(message))


def is_failover_eligible(exc: Exception) -> bool:
    """Return ``True`` iff ``exc`` is a transient/provider fault worth failing
    over to the next candidate.

    Decision order:
      1. ``openai`` SDK exceptions (structured) — defensive, future-proof.
      2. ``httpx`` transport exceptions (connection/timeout) -> eligible;
         ``httpx.HTTPStatusError`` -> decide by status.
      3. esperanto ``RuntimeError`` HTTP wrapper -> parse status / transient
         phrase.
      4. Everything else (``TypeError``/``ValueError``/``KeyError``/bare
         ``RuntimeError``) -> NOT eligible (propagate; never mask bugs).
    """
    sdk = _openai_sdk_eligible(exc)
    if sdk is not None:
        return sdk

    if isinstance(exc, httpx.HTTPStatusError):
        status = _status_from_exc(exc)
        return status in _ELIGIBLE_STATUSES if status is not None else True
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        # Connection refused / DNS / read timeout / protocol error — the
        # endpoint is unreachable or flaky, so fail over.
        return True

    if isinstance(exc, RuntimeError):
        return _runtime_error_eligible(exc)

    return False
