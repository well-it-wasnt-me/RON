"""API security helpers: secret masking and authentication dependency.

These are shared by the REST API and the Telegram bridge so secret
masking is consistent everywhere config is exposed.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Field names that are treated as secrets and masked in any config dump.
_SECRET_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "api_key",
        "bot_token",
        "password",
        "access_key",
        "secret",
    }
)


def mask_secret_value(value: Any) -> Any:
    """Mask a single secret string, leaving empty values as empty."""
    if not isinstance(value, str) or not value:
        return value
    if len(value) <= 4:
        return "****"
    return value[:4] + "****"


def mask_secrets_in_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively walk a config dict and mask any field whose name looks secret.

    This is the single source of truth for secret masking — used by both
    the REST ``GET /config`` endpoint and the Telegram ``/config`` command.
    """
    masked: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            masked[key] = mask_secrets_in_dict(value)
        elif key in _SECRET_FIELD_NAMES:
            masked[key] = mask_secret_value(value)
        else:
            masked[key] = value
    return masked


# ---------------------------------------------------------------------------
# Authentication dependency
# ---------------------------------------------------------------------------
_bearer_scheme = HTTPBearer(auto_error=False)


async def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
    query_key: str | None = Query(default=None, alias="api_key", include_in_schema=False),
) -> None:
    """FastAPI dependency that enforces the configured API key on control endpoints.

    When ``settings.api.api_key`` is empty, no authentication is required
    (the API is expected to be bound to ``127.0.0.1`` in that case).

    When a key is configured, callers must supply it either as an
    ``Authorization: Bearer <key>`` header or an ``?api_key=<key>`` query
    parameter.
    """
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        return  # no settings → can't enforce (test/bootstrap mode)
    configured_key: str = getattr(settings.api, "api_key", "")
    if not configured_key:
        return  # auth disabled
    provided = None
    if credentials is not None:
        provided = credentials.credentials
    elif query_key is not None:
        provided = query_key
    if provided is None or provided != configured_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


__all__ = ["mask_secret_value", "mask_secrets_in_dict", "require_api_key"]
