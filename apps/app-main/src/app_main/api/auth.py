"""
Authentication middleware and utilities for app-main.
"""

import os
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class PasswordAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to check password authentication for all API requests.
    Only active when OPEN_NOTEBOOK_PASSWORD environment variable is set.
    """

    def __init__(self, app, excluded_paths: Optional[list] = None):
        super().__init__(app)
        self.password = os.environ.get("OPEN_NOTEBOOK_PASSWORD")
        self.excluded_paths = excluded_paths or [
            "/", "/health", "/docs", "/openapi.json", "/redoc",
        ]

    async def dispatch(self, request: Request, call_next):
        if not self.password:
            return await call_next(request)

        if request.url.path in self.excluded_paths:
            return await call_next(request)

        for excluded_path in self.excluded_paths:
            if "*" in excluded_path:
                import re

                pattern = excluded_path.replace("*", "[^/]+")
                if re.match(f"^{pattern}$", request.url.path):
                    return await call_next(request)

        # Allow PDF endpoint access for react-pdf-highlighter
        if request.url.path.startswith("/api/sources/") and request.url.path.endswith(
            "/pdf"
        ):
            return await call_next(request)

        if request.method == "OPTIONS":
            return await call_next(request)

        auth_header = request.headers.get("Authorization")

        if not auth_header:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing authorization header"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            scheme, credentials = auth_header.split(" ", 1)
            if scheme.lower() != "bearer":
                raise ValueError("Invalid authentication scheme")
        except ValueError:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid authorization header format"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        if credentials != self.password:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid password"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        response = await call_next(request)
        return response


security = HTTPBearer(auto_error=False)


def check_api_password(
    credentials: Optional[HTTPAuthorizationCredentials] = None,
) -> bool:
    """Utility function to check API password."""
    password = os.environ.get("OPEN_NOTEBOOK_PASSWORD")

    if not password:
        return True

    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing authorization",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.credentials != password:
        raise HTTPException(
            status_code=401,
            detail="Invalid password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True
