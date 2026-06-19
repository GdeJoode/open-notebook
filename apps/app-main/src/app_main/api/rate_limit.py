"""Shared slowapi rate limiter (Track I, Phase I.H1).

Lives in its own module so both the app factory (`app_main.api.app`) and the
routers that decorate endpoints (`@limiter.limit(...)`) can import the same
`Limiter` instance without a circular import — routers must not import the
app module, which constructs the application at import time.

Key-func is `get_remote_address`, so buckets are per-IP (AC5: per-IP, not
per-process). No `default_limits`: the limit is applied selectively via
`@limiter.limit` on the upload endpoint(s) only, not globally to every route.
`headers_enabled=True` makes slowapi attach the `Retry-After` header on a
tripped limit (AC3); without it the default handler omits it.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    headers_enabled=True,
)
