"""Tool executor - dispatches LLM tool calls to robot actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from robot.ai.tools.registry import ToolRegistry
from robot.ai.tools.schema import ToolDefinition, ToolParameter, ToolParameterType
from robot.errors import DeskBotError
from robot.events.bus import InMemoryEventBus
from robot.events.events import (
    BlinkRequested,
    EmotionChanged,
    EmotionName,
    ServoMoved,
    SoundEffectPlayed,
)
from robot.interfaces.audio import AudioOutput
from robot.interfaces.servo import ServoController
from robot.logging import get_logger
from robot.speech.tts import TextToSpeech

_log = get_logger("ai.tools.executor")


class ToolExecutionError(DeskBotError):
    """Error during tool execution."""


_BUILTIN_HANDLER_MAP: dict[str, str] = {
    "change_emotion": "_handle_change_emotion",
    "play_sound": "_handle_play_sound",
    "set_state": "_handle_set_state",
    "move_servo": "_handle_move_servo",
    "speak": "_handle_speak",
}


@dataclass(slots=True)
class ToolExecutor:
    """Dispatches LLM tool calls to robot actions."""

    registry: ToolRegistry
    bus: InMemoryEventBus
    servo_controller: ServoController | None = None
    tts: TextToSpeech | None = None
    audio: AudioOutput | None = None

    async def execute_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a single tool call."""
        _log.info("tool_executor.call", tool=tool_name, args=arguments)

        if tool_name not in self.registry:
            raise ToolExecutionError(f"unknown tool: {tool_name!r}")

        definition = self.registry.get(tool_name)
        validated = self._validate_arguments(definition, arguments)

        handler_name = _BUILTIN_HANDLER_MAP.get(tool_name)
        if handler_name is not None:
            handler = getattr(self, handler_name)
            return cast("dict[str, Any]", await handler(validated))

        return await self.registry.execute(tool_name, validated)

    def _validate_arguments(
        self, definition: ToolDefinition, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate and coerce arguments against the tool's parameters."""
        validated: dict[str, Any] = {}

        for param in definition.parameters:
            value = arguments.get(param.name, param.default)
            if param.required and value is None:
                raise ToolExecutionError(
                    f"tool {definition.name!r}: missing required parameter {param.name!r}"
                )
            if value is not None:
                value = self._coerce_type(param, value)
            validated[param.name] = value

        return validated

    @classmethod
    def _coerce_type(cls, param: ToolParameter, value: Any) -> Any:
        """Coerce a value to the expected parameter type."""
        if isinstance(value, str):
            return cls._coerce_string(param.type, value)
        if isinstance(value, (int, float)):
            return cls._coerce_number(param.type, value)
        return value

    @staticmethod
    def _coerce_string(ptype: ToolParameterType, value: str) -> Any:
        if ptype == ToolParameterType.NUMBER:
            try:
                return float(value)
            except ValueError:
                return value
        if ptype == ToolParameterType.INTEGER:
            try:
                return int(value)
            except ValueError:
                return value
        if ptype == ToolParameterType.BOOLEAN:
            lower = value.lower()
            return (
                True
                if lower in ("true", "1", "yes")
                else False
                if lower in ("false", "0", "no")
                else None
            )
        return value

    @staticmethod
    def _coerce_number(ptype: ToolParameterType, value: int | float) -> Any:
        if ptype == ToolParameterType.STRING:
            return str(value)
        if ptype == ToolParameterType.INTEGER:
            return int(value)
        return value

    async def _handle_change_emotion(self, args: dict[str, Any]) -> dict[str, Any]:
        """Dispatch an emotion change through the event bus."""
        emotion_name = str(args.get("emotion", "neutral"))
        intensity = max(0.0, min(1.0, float(args.get("intensity", 1.0))))

        try:
            emotion = EmotionName(emotion_name)
        except ValueError:
            valid = [e.value for e in EmotionName]
            return {"error": f"unknown emotion {emotion_name!r}", "valid": valid}

        await self.bus.publish(
            EmotionChanged(previous=EmotionName.NEUTRAL, current=emotion, intensity=intensity)
        )
        await self.bus.publish(BlinkRequested(speed=1.0))
        return {"status": "ok", "emotion": emotion_name, "intensity": intensity}

    async def _handle_play_sound(self, args: dict[str, Any]) -> dict[str, Any]:
        """Play a sound effect via the event bus."""
        name = str(args.get("name", ""))
        await self.bus.publish(SoundEffectPlayed(name=name, filename=f"{name}.wav"))
        return {"status": "ok", "sound": name}

    async def _handle_set_state(self, args: dict[str, Any]) -> dict[str, Any]:
        """Transition the robot to a new state via the event bus."""
        from robot.behavior.state_machine import RobotState
        from robot.events.events import StateChanged

        state_name = str(args.get("state", "idle"))
        try:
            state = RobotState(state_name)
        except ValueError:
            valid = [s.value for s in RobotState]
            return {"error": f"unknown state {state_name!r}", "valid": valid}

        await self.bus.publish(StateChanged(previous=RobotState.IDLE, current=state))
        return {"status": "ok", "state": state_name}

    async def _handle_move_servo(self, args: dict[str, Any]) -> dict[str, Any]:
        """Move a servo to a target angle."""
        if self.servo_controller is None:
            return {"error": "no servo controller available"}

        servo_name = str(args.get("servo", "pan"))
        angle = float(args.get("angle", 90.0))
        duration_s = float(args.get("duration_s", 0.4))

        try:
            servo = self.servo_controller.get(servo_name)
            await servo.move_to(angle, duration_s=duration_s)
            await self.bus.publish(ServoMoved(name=servo_name, angle=angle))
            return {"status": "ok", "servo": servo_name, "angle": angle}
        except Exception as exc:
            return {"error": str(exc), "servo": servo_name}

    async def _handle_speak(self, args: dict[str, Any]) -> dict[str, Any]:
        """Speak text through TTS and play via the audio output."""
        if self.tts is None:
            return {"error": "no TTS engine available"}
        text = str(args.get("text", ""))
        if not text:
            return {"error": "empty text"}
        buffer = await self.tts.speak(text)
        if buffer is not None and not buffer.is_empty and self.audio is not None:
            await self.audio.play(buffer)
        return {"status": "ok", "text": text[:100]}


__all__ = ["ToolExecutionError", "ToolExecutor"]
