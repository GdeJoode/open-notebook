"""Disciplined HTTP client for vocabulary providers (K.4).

The NIM / fair-use lesson: every external call a provider makes goes through
this one client, which guarantees four things so a remote authority can never
break ingest or hammer a public API:

1. **Timeout** — every request has a bounded timeout; a hung backend fails fast.
2. **Rate-limiting** — a minimum interval between requests (token-less leaky
   bucket) keeps us inside public-pool politeness (Crossref's polite pool, TOOI's
   linked-data endpoint).
3. **Caching** — identical GETs inside a TTL are served from an in-process cache,
   so a batch reconcile over many entities re-using the same lookup hits the
   network once.
4. **Graceful failure** — any transport/HTTP error returns ``None`` (logged), so
   the caller degrades to a no-match rather than raising.

The client is intentionally dependency-light (httpx only) and easy to mock:
tests patch ``httpx`` at the transport layer (respx) — no live network in CI.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import httpx
from loguru import logger


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class VocabularyHTTPClient:
    """A cached, rate-limited, timeout-bounded, fail-soft GET client.

    One instance per provider (so rate limits and caches are per-authority).
    Safe for concurrent use within one event loop: the rate-limit gate and the
    cache are guarded by an asyncio lock.
    """

    def __init__(
        self,
        *,
        user_agent: str,
        timeout: float = 10.0,
        min_interval: float = 1.0,
        cache_ttl: float = 3600.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Args:

        user_agent: Sent on every request. For Crossref's polite pool this MUST
            carry a contact (e.g. ``mailto:``) — see ``crossref_provider``.
        timeout: Per-request timeout in seconds.
        min_interval: Minimum seconds between successive requests (rate limit).
        cache_ttl: Seconds a cached GET response stays fresh.
        client: Inject a pre-built ``httpx.AsyncClient`` (tests pass a mocked
            transport here). When ``None`` a client is lazily created.
        """
        self._user_agent = user_agent
        self._timeout = timeout
        self._min_interval = min_interval
        self._cache_ttl = cache_ttl
        self._client = client
        self._owns_client = client is None
        self._last_request_at = 0.0
        self._cache: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], _CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={"User-Agent": self._user_agent},
            )
            self._owns_client = True
        return self._client

    async def get_json(
        self, url: str, params: Optional[Dict[str, str]] = None
    ) -> Optional[Any]:
        """GET ``url`` and return parsed JSON, or ``None`` on any failure.

        Cached by ``(url, sorted(params))`` for ``cache_ttl`` seconds. Rate-limited
        to ``min_interval`` between live requests. NEVER raises — a transport/HTTP/
        parse error is logged and returns ``None`` so callers treat it as a
        no-match.
        """
        cache_key = (url, tuple(sorted((params or {}).items())))
        async with self._lock:
            cached = self._cache.get(cache_key)
            now = time.monotonic()
            if cached is not None and cached.expires_at > now:
                return cached.value

            await self._respect_rate_limit()

            try:
                client = await self._ensure_client()
                response = await client.get(
                    url,
                    params=params,
                    headers={"User-Agent": self._user_agent},
                )
                self._last_request_at = time.monotonic()
                response.raise_for_status()
                value = response.json()
            except Exception as exc:  # transport, HTTP status, or JSON decode
                # Fail soft: the authority being unreachable must not break the
                # reconcile/ingest path. Log + return no-match.
                logger.warning(
                    "vocabulary HTTP GET failed (treated as no-match): {url} — {exc}",
                    url=url,
                    exc=exc,
                )
                return None

            self._cache[cache_key] = _CacheEntry(
                value=value, expires_at=time.monotonic() + self._cache_ttl
            )
            return value

    async def get_text(
        self, url: str, params: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """GET ``url`` and return the raw response body as text, or ``None``.

        The text counterpart of :meth:`get_json` for authorities that answer with
        XML rather than JSON (arXiv's Atom feed, overheid.nl's SRU endpoint). Same
        discipline — cached by ``(url, sorted(params))``, rate-limited, and
        fail-soft: any transport/HTTP error is logged and returns ``None`` so an
        unreachable authority degrades to a no-match instead of raising.
        """
        cache_key = ("text::" + url, tuple(sorted((params or {}).items())))
        async with self._lock:
            cached = self._cache.get(cache_key)
            now = time.monotonic()
            if cached is not None and cached.expires_at > now:
                return str(cached.value)

            await self._respect_rate_limit()

            try:
                client = await self._ensure_client()
                response = await client.get(
                    url,
                    params=params,
                    headers={"User-Agent": self._user_agent},
                )
                self._last_request_at = time.monotonic()
                response.raise_for_status()
                value: str = response.text
            except Exception as exc:  # transport or HTTP status
                logger.warning(
                    "vocabulary HTTP GET failed (treated as no-match): {url} — {exc}",
                    url=url,
                    exc=exc,
                )
                return None

            self._cache[cache_key] = _CacheEntry(
                value=value, expires_at=time.monotonic() + self._cache_ttl
            )
            return value

    async def _respect_rate_limit(self) -> None:
        """Sleep just enough to keep ``min_interval`` between live requests."""
        if self._min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        wait = self._min_interval - elapsed
        if wait > 0:
            await asyncio.sleep(wait)

    async def aclose(self) -> None:
        """Close the underlying client if we own it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None


__all__ = ["VocabularyHTTPClient"]
