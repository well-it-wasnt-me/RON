"""Command endpoints - send actions to the robot."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from robot.api.security import require_api_key

from robot.api.schemas import (
    CommandResponse,
    EmotionRequest,
    SpeakRequest,
    StateRequest,
)
from robot.errors import StateTransitionError
from robot.logging import get_logger

_log = get_logger("api.commands")

router = APIRouter()


@router.post("/speak", summary="Speak text", response_model=CommandResponse)
async def speak(request: Request, body: SpeakRequest, _: None = Depends(require_api_key)) -> CommandResponse:
    """Send text through the conversation pipeline (LLM -> TTS).

    Uses :meth:`ConversationService.handle_user_text` so typed input
    enters the same canonical path as speech recognition.
    """
    bridge = getattr(request.app.state, "bridge", None)
    if bridge is None or not bridge.is_ready or bridge.conversation is None:
        return CommandResponse(status="error", detail="DeskBot app not attached")
    try:
        await bridge.conversation.handle_user_text(body.text, source="api")
    except Exception as exc:
        _log.error("speak.failed", error=str(exc))
        return CommandResponse(status="error", detail=f"Conversation failed: {exc}")
    return CommandResponse.model_validate({"status": "ok", "text": body.text})


@router.post("/speak-direct", summary="Speak text directly via TTS", response_model=CommandResponse)
async def speak_direct(request: Request, body: SpeakRequest, _: None = Depends(require_api_key)) -> CommandResponse:
    """Speak text directly through TTS without going through the LLM pipeline.

    This bypasses STT and LLM entirely - just synthesizes the given text.
    """
    bridge = getattr(request.app.state, "bridge", None)
    if bridge is None or not bridge.is_ready or bridge.tts is None:
        _log.error("speak_direct.not_ready")
        return CommandResponse(status="error", detail="DeskBot or TTS not attached")
    tts_name = getattr(bridge.tts, "name", type(bridge.tts).__name__)
    _log.info("speak_direct", text=body.text[:200], tts=tts_name)
    try:
        buffer = await bridge.tts.speak(body.text)
        audio = getattr(bridge, "audio", None)
        if audio is not None and buffer is not None and not buffer.is_empty:
            await audio.play(buffer)
    except Exception as exc:
        _log.error("speak_direct.failed", error=str(exc), tts=tts_name)
        return CommandResponse(status="error", detail=f"TTS failed: {exc}")
    return CommandResponse.model_validate({"status": "ok", "text": body.text})


@router.post("/emotion", summary="Set emotion", response_model=CommandResponse)
async def set_emotion(request: Request, body: EmotionRequest, _: None = Depends(require_api_key)) -> CommandResponse:
    """Set the robot's current emotion."""
    from robot.events.events import EmotionChanged, EmotionName

    # Validate emotion name
    try:
        emotion = EmotionName(body.emotion.lower())
    except ValueError:
        valid = [e.value for e in EmotionName]
        return CommandResponse(status="error", detail=f"Invalid emotion. Valid: {valid}")
    bridge = getattr(request.app.state, "bridge", None)
    if bridge is None or bridge.bus is None:
        return CommandResponse(status="error", detail="DeskBot app not attached")
    await bridge.bus.publish(
        EmotionChanged(previous=EmotionName.NEUTRAL, current=emotion, intensity=body.intensity)
    )
    return CommandResponse.model_validate({"status": "ok", "emotion": emotion.value})


@router.post("/state", summary="Transition state", response_model=CommandResponse)
async def set_state(request: Request, body: StateRequest, _: None = Depends(require_api_key)) -> CommandResponse:
    """Transition the robot to a new state."""
    from robot.behavior.state_machine import RobotState

    try:
        target = RobotState(body.state.lower())
    except ValueError:
        valid = [s.value for s in RobotState]
        return CommandResponse(status="error", detail=f"Invalid state. Valid: {valid}")
    bridge = getattr(request.app.state, "bridge", None)
    if bridge is None or bridge.state_machine is None:
        return CommandResponse(status="error", detail="DeskBot app not attached")
    try:
        await bridge.state_machine.transition(target)
    except StateTransitionError as exc:
        _log.warning("state.illegal_transition", detail=str(exc))
        return CommandResponse(status="error", detail=str(exc))
    return CommandResponse.model_validate({"status": "ok", "state": target.value})
