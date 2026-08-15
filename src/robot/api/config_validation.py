"""Configuration validation and schema endpoints.

Provides endpoints to:
- Validate a proposed config without applying it.
- Retrieve the Pydantic schema for the current configuration model.

The validate endpoint accepts a partial or full config override as JSON,
constructs a temporary :class:`AppSettings` from the proposed values merged
with the current defaults, and returns either ``{valid: true}`` or
``{valid: false, errors: [...]}`` with per-field error messages.

The schema endpoint returns the full JSON Schema for :class:`AppSettings`,
enabling dynamic form generation in the config validator UI.

Note: The ``GET /config`` endpoint lives in ``state.py`` and is not
duplicated here.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import ValidationError

from robot.api.schemas import ConfigSchemaResponse, ConfigValidateRequest, ConfigValidateResponse
from robot.config import AppSettings

router = APIRouter()


def _format_validation_error(err: Any) -> dict[str, Any]:
    """Convert a single Pydantic ``ErrorDetails`` dict to a UI-friendly dict."""
    location = ".".join(str(loc) for loc in err.get("loc", ()))
    return {
        "field": location or "(root)",
        "message": err.get("msg", "Unknown validation error"),
        "type": err.get("type", "unknown"),
        "input": err.get("input"),
    }


@router.post(
    "/config/validate",
    summary="Validate proposed configuration",
    response_model=ConfigValidateResponse,
)
async def validate_config(request: Request, body: ConfigValidateRequest) -> ConfigValidateResponse:
    """Validate a proposed configuration without applying it.

    Accepts a JSON body with partial or full config overrides. The
    proposed values are merged with the current defaults and validated
    against the :class:`AppSettings` Pydantic model.

    Returns ``{valid: true}`` on success, or ``{valid: false, errors: [...]}``
    with field-level error messages on failure.

    This endpoint **never** modifies the running configuration.
    """
    # FastAPI already parsed and validated the JSON body into `body`.
    # Convert to a plain dict for merging with the current defaults.
    body_dict = body.model_dump(exclude_none=False)
    # Start from current defaults so partial overrides are valid.
    current: AppSettings = getattr(request.app.state, "settings", None) or AppSettings()
    current_dict = current.model_dump()
    # Deep merge: proposed values override current defaults.
    merged = _deep_merge(current_dict, body_dict)
    try:
        # Use _env_file=None so the .env file does NOT override the
        # proposed values during validation.  Without this, any field
        # present in .env would silently replace what the user submitted,
        # making the endpoint useless for catching bad configs.
        AppSettings(**merged, _env_file=None)
    except ValidationError as exc:
        errors = [_format_validation_error(e) for e in exc.errors()]
        return ConfigValidateResponse(valid=False, errors=errors)
    return ConfigValidateResponse(valid=True)


@router.get(
    "/config/schema", summary="Configuration JSON Schema", response_model=ConfigSchemaResponse
)
async def get_config_schema(request: Request) -> ConfigSchemaResponse:
    """Return the full JSON Schema for :class:`AppSettings`.

    This enables the config validator UI to build dynamic form inputs,
    show field descriptions, types, defaults, and validation constraints.
    """
    return ConfigSchemaResponse.model_validate(AppSettings.model_json_schema())


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict.

    List values from *override* replace those in *base* (no merging).
    Dict values are merged recursively. Scalar values from *override*
    replace those in *base*.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


__all__ = ["router"]
