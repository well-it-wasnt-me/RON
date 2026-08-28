"""State and configuration query endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from robot.api.schemas import (
    AudioStatusResponse,
    ConfigResponse,
    ConversationStatusResponse,
    PerceptionResponse,
    StateResponse,
)

router = APIRouter()


@router.get("/state", summary="Current robot state", response_model=StateResponse)
async def get_state(request: Request) -> StateResponse:
    """Return the current robot state (IDLE, CURIOUS, LISTENING, etc.).

    Returns a placeholder if the state machine hasn't been started yet.
    """
    bridge = getattr(request.app.state, "bridge", None)
    if bridge is None or not bridge.is_ready or bridge.state_machine is None:
        return StateResponse(state="unknown", detail="DeskBot app not attached")
    return StateResponse(state=bridge.state_machine.state.value)


@router.get("/config", summary="Current configuration", response_model=ConfigResponse)
async def get_config(request: Request) -> ConfigResponse:
    """Return the current configuration as a JSON dict.

    Sensitive fields (API keys) are masked.
    """
    from robot.api.security import mask_secrets_in_dict
    from robot.config import AppSettings

    settings: AppSettings = getattr(request.app.state, "settings", None) or AppSettings()
    config_dict = settings.model_dump()
    # Mask all secret fields (api_key, bot_token, password, access_key, …) centrally.
    config_dict = mask_secrets_in_dict(config_dict)
    return ConfigResponse.model_validate(config_dict)


@router.get("/perception", summary="Perception status", response_model=PerceptionResponse)
async def get_perception(request: Request) -> PerceptionResponse:
    """Return the current perception status (face detection, scan interval)."""
    bridge = getattr(request.app.state, "bridge", None)
    if bridge is None or bridge.perception is None:
        return PerceptionResponse(enabled=False)
    p = bridge.perception
    return PerceptionResponse(
        enabled=True,
        running=not p._stopped,
        scan_interval_s=p._current_interval,
        max_faces=p.max_faces,
    )


@router.get("/audio", summary="Audio output status", response_model=AudioStatusResponse)
async def get_audio(request: Request) -> AudioStatusResponse:
    """Return the current audio output status."""
    bridge = getattr(request.app.state, "bridge", None)
    if bridge is None or not bridge.is_ready:
        return AudioStatusResponse(enabled=False, detail="DeskBot app not attached")
    return AudioStatusResponse(enabled=True, talking=bridge._talking)


@router.get(
    "/conversation", summary="Conversation status", response_model=ConversationStatusResponse
)
async def get_conversation(request: Request) -> ConversationStatusResponse:
    """Return the current conversation service status."""
    bridge = getattr(request.app.state, "bridge", None)
    if bridge is None or bridge.conversation is None:
        return ConversationStatusResponse(enabled=False)
    c = bridge.conversation
    return ConversationStatusResponse(
        enabled=True,
        state=bridge.state_machine.state.value if bridge.state_machine else "unknown",
        wake_checker=type(c.wake_checker).__name__,
        stt=type(c.stt).__name__,
        tts=type(c.tts).__name__,
        llm=type(c.llm).__name__,
    )
