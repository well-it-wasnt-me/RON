"""REST API endpoints for user preferences."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from robot.ai.preferences import Preference, PreferenceTracker
from robot.api.schemas import (
    PreferenceDeleteResponse,
    PreferenceItem,
    PreferenceListResponse,
)
from robot.api.security import require_api_key

router = APIRouter()


def _get_tracker(request: Request) -> PreferenceTracker:
    """Get the preference tracker from the app state, or raise 404."""
    bridge = getattr(request.app.state, "bridge", None)
    if bridge is None:
        raise HTTPException(status_code=404, detail="Preference tracker not available")
    tracker: PreferenceTracker | None = getattr(bridge, "preference_tracker", None)
    if tracker is None:
        raise HTTPException(status_code=404, detail="Preference tracker not enabled")
    return tracker


def _pref_to_dict(pref: Preference) -> dict[str, object]:
    """Convert a Preference to a JSON-serializable dict."""
    return {
        "key": pref.key,
        "value": pref.value,
        "confidence": pref.confidence,
        "source": pref.source,
        "updated_at": pref.updated_at.isoformat(),
    }


@router.get("/preferences", summary="List all preferences", response_model=PreferenceListResponse)
async def list_preferences(request: Request) -> PreferenceListResponse:
    """Return all learned user preferences."""
    tracker = _get_tracker(request)
    prefs = tracker.get_all()
    return PreferenceListResponse(preferences=[_pref_to_dict(p) for p in prefs])


@router.get(
    "/preferences/{key}", summary="Get a specific preference", response_model=PreferenceItem
)
async def get_preference(key: str, request: Request) -> PreferenceItem:
    """Return a single preference by key."""
    tracker = _get_tracker(request)
    pref = tracker.get(key)
    if pref is None:
        raise HTTPException(status_code=404, detail=f"Preference '{key}' not found")
    return PreferenceItem.model_validate(_pref_to_dict(pref))


@router.delete(
    "/preferences/{key}", summary="Delete a preference", response_model=PreferenceDeleteResponse
)
async def delete_preference(
    key: str, request: Request, _: None = Depends(require_api_key)
) -> PreferenceDeleteResponse:
    """Delete a preference by key."""
    tracker = _get_tracker(request)
    store = tracker.store
    deleted = store.delete(key)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Preference '{key}' not found")
    return PreferenceDeleteResponse(status="deleted", key=key)
