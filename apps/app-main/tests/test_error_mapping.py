"""Eligibility-whitelist tests for the J.4 error mapping.

The cloud provider is NVIDIA NIM (OpenAI-compatible). esperanto surfaces its
errors as a bare ``RuntimeError`` HTTP wrapper or as ``httpx`` transport
exceptions; the ``openai`` SDK hierarchy is mapped defensively. These tests pin
that:

  * 429 / 5xx / connection / timeout / 401 / 403 -> ELIGIBLE (fail over)
  * 400 bad-request / programming errors -> NOT eligible (propagate)
"""

import httpx
import pytest

from app_main.services.model_routing.error_mapping import is_failover_eligible


# ---------------------------------------------------------------------------
# esperanto RuntimeError HTTP wrapper (the live NIM path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "OpenAI-compatible endpoint error: HTTP 429: rate limited",
        "OpenAI-compatible endpoint error: HTTP 503: service unavailable",
        "OpenAI API error: HTTP 500: internal error",
        "Anthropic API error: HTTP 502: bad gateway",
        "OpenAI-compatible endpoint error: HTTP 504: gateway timeout",
    ],
)
def test_esperanto_runtimeerror_with_status_eligible(message):
    assert is_failover_eligible(RuntimeError(message)) is True


@pytest.mark.parametrize(
    "message",
    [
        "OpenAI-compatible endpoint error: rate limit exceeded for this org",
        "OpenAI-compatible endpoint error: the service is temporarily unavailable",
        "OpenAI API error: server error, please try again",
    ],
)
def test_esperanto_runtimeerror_transient_phrase_eligible(message):
    """NIM JSON error bodies carry no status in the esperanto message — a
    transient phrase makes them eligible."""
    assert is_failover_eligible(RuntimeError(message)) is True


@pytest.mark.parametrize(
    "message",
    [
        "OpenAI-compatible endpoint error: HTTP 400: invalid request",
        "OpenAI-compatible endpoint error: HTTP 404: model not found",
        "OpenAI-compatible endpoint error: HTTP 422: unprocessable entity",
        "OpenAI API error: invalid 'model' field",
    ],
)
def test_esperanto_runtimeerror_client_error_not_eligible(message):
    assert is_failover_eligible(RuntimeError(message)) is False


def test_bare_runtimeerror_not_eligible():
    """A RuntimeError that is not an esperanto HTTP wrapper is a real bug — it
    must not be masked as failover."""
    assert is_failover_eligible(RuntimeError("something went horribly wrong")) is False


# ---------------------------------------------------------------------------
# httpx transport exceptions
# ---------------------------------------------------------------------------


def test_httpx_connect_error_eligible():
    assert is_failover_eligible(httpx.ConnectError("connection refused")) is True


def test_httpx_read_timeout_eligible():
    assert is_failover_eligible(httpx.ReadTimeout("read timed out")) is True


def test_httpx_status_error_429_eligible():
    request = httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions")
    response = httpx.Response(429, request=request)
    exc = httpx.HTTPStatusError("429", request=request, response=response)
    assert is_failover_eligible(exc) is True


def test_httpx_status_error_400_not_eligible():
    request = httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions")
    response = httpx.Response(400, request=request)
    exc = httpx.HTTPStatusError("400", request=request, response=response)
    assert is_failover_eligible(exc) is False


# ---------------------------------------------------------------------------
# openai SDK hierarchy (defensive mapping)
# ---------------------------------------------------------------------------


def test_openai_rate_limit_eligible():
    openai = pytest.importorskip("openai")
    request = httpx.Request("POST", "https://api/x")
    response = httpx.Response(429, request=request)
    exc = openai.RateLimitError(
        "rate limited", response=response, body=None
    )
    assert is_failover_eligible(exc) is True


def test_openai_auth_error_eligible():
    openai = pytest.importorskip("openai")
    request = httpx.Request("POST", "https://api/x")
    response = httpx.Response(401, request=request)
    exc = openai.AuthenticationError("bad key", response=response, body=None)
    assert is_failover_eligible(exc) is True


def test_openai_bad_request_not_eligible():
    openai = pytest.importorskip("openai")
    request = httpx.Request("POST", "https://api/x")
    response = httpx.Response(400, request=request)
    exc = openai.BadRequestError("bad prompt", response=response, body=None)
    assert is_failover_eligible(exc) is False


# ---------------------------------------------------------------------------
# programming errors always propagate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [TypeError("not callable"), ValueError("bad value"), KeyError("missing"), AttributeError("x")],
)
def test_programming_errors_not_eligible(exc):
    assert is_failover_eligible(exc) is False
