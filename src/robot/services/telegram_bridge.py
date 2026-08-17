"""Telegram bridge — chat with DeskBot and control every aspect via Telegram.

Requires the ``httpx`` package for the Telegram Bot API HTTP calls.

The bridge runs a long-poll loop in a background task. When a message
arrives it is dispatched to the appropriate handler:

* **Chat** — plain text is forwarded to :meth:`ConversationService.handle_user_text`,
  which runs the full LLM → TTS pipeline. The reply is sent back to the
  Telegram chat.
* **Slash commands** — ``/emotion``, ``/state``, ``/servo``, ``/speak``,
  ``/sound``, ``/behavior``, ``/status``, ``/config``, ``/help`` let you
  control every aspect of the robot from Telegram.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from robot.logging import get_logger as get_log

if TYPE_CHECKING:
    from robot.events.bus import InMemoryEventBus
    from robot.events.events import BotReply

_log = get_log("services.telegram_bridge")

# Telegram Bot API long-poll timeout (seconds).
_POLL_TIMEOUT = 30
# Maximum retries on HTTP failure before backing off.
_MAX_RETRIES = 3


@dataclass(slots=True)
class TelegramConfig:
    """Telegram bot configuration (env prefix: ``DESKBOT_TELEGRAM__``)."""

    bot_token: str = ""
    """Telegram bot token from ``@BotFather``."""
    enabled: bool = False
    """Whether the Telegram bridge is active."""
    allowed_user_ids: list[int] = field(default_factory=list)
    """If non-empty, only these Telegram user IDs may interact with the bot."""
    chat_timeout_s: float = 60.0
    """How long to wait for a BotReply event before giving up."""
    api_base: str = "https://api.telegram.org"
    """Telegram Bot API base URL (override for self-hosted instances)."""


class TelegramBridge:
    """Bidirectional Telegram ↔ DeskBot bridge.

    Subscribes to :class:`BotReply` events so that when the conversation
    pipeline produces a reply (from any input source — voice, API, MQTT,
    or Telegram itself), the reply is forwarded to all allowed Telegram
    chats.
    """

    def __init__(
        self,
        config: TelegramConfig,
        bus: InMemoryEventBus,
        app: Any,
    ) -> None:
        self._config = config
        self._bus = bus
        self._app = app
        self._poll_task: asyncio.Task[None] | None = None
        self._last_update_id = 0
        self._pending_replies: dict[int, asyncio.Future[str]] = {}
        self._http: Any = None  # httpx.AsyncClient (lazy-imported)

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        """Start the bridge: verify the bot token and begin long-polling."""
        import httpx

        self._http = httpx.AsyncClient(
            base_url=self._config.api_base,
            timeout=httpx.Timeout(_POLL_TIMEOUT + 10, connect=10),
        )

        # Verify the bot token.
        resp = await self._http.get(f"/bot{self._config.bot_token}/getMe")
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram getMe failed: {data}")
        bot_name = data["result"]["username"]
        _log.info("telegram.bridge_started", bot=f"@{bot_name}")

        # Subscribe to BotReply events so replies go to Telegram.
        from robot.events.events import BotReply

        self._bus.subscribe(BotReply, self._on_bot_reply)

        # Start the long-poll loop.
        self._poll_task = asyncio.create_task(self._poll_loop(), name="TelegramBridge-poll")

    async def stop(self) -> None:
        """Stop the bridge and clean up resources."""
        from robot.events.events import BotReply

        self._bus.unsubscribe(BotReply, self._on_bot_reply)
        if self._poll_task is not None:
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task
            self._poll_task = None
        if self._http is not None:
            with contextlib.suppress(Exception):
                await self._http.aclose()
            self._http = None
        _log.info("telegram.bridge_stopped")

    # ------------------------------------------------------------------ long poll

    async def _poll_loop(self) -> None:
        """Continuously long-poll the Telegram Bot API for updates."""
        while True:
            try:
                updates = await self._get_updates()
                for update in updates:
                    self._last_update_id = update["update_id"]
                    await self._handle_update(update)
            except asyncio.CancelledError:
                raise
            except Exception:
                _log.exception("telegram.poll_error")
                await asyncio.sleep(5)

    async def _get_updates(self) -> list[dict[str, Any]]:
        """Call the Telegram Bot API getUpdates with long-polling."""
        assert self._http is not None
        params: dict[str, Any] = {
            "offset": self._last_update_id + 1,
            "timeout": _POLL_TIMEOUT,
            "allowed_updates": json.dumps(["message"]),
        }
        for attempt in range(_MAX_RETRIES):
            try:
                resp = await self._http.get(
                    f"/bot{self._config.bot_token}/getUpdates",
                    params=params,
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("ok"):
                    return list(data.get("result", []))
                _log.warning("telegram.get_updates_not_ok", data=data)
            except Exception:
                if attempt == _MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(2**attempt)
        return []

    # ------------------------------------------------------------------ dispatch

    async def _handle_update(self, update: dict[str, Any]) -> None:
        """Dispatch a single Telegram update to the right handler."""
        message = update.get("message")
        if message is None:
            return

        chat_id = message["chat"]["id"]
        user = message.get("from", {})
        user_id = user.get("id", 0)
        text = (message.get("text") or "").strip()

        # Authorise the user.
        if self._config.allowed_user_ids and user_id not in self._config.allowed_user_ids:
            await self._send_message(chat_id, "⛔ You are not authorised to control this robot.")
            return

        if not text:
            return

        _log.info("telegram.message", user_id=user_id, chat_id=chat_id, text=text[:100])

        # Slash commands.
        if text.startswith("/"):
            await self._handle_command(chat_id, text, user_id)
        else:
            await self._handle_chat(chat_id, text)

    # ------------------------------------------------------------------ chat

    async def _handle_chat(self, chat_id: int, text: str) -> None:
        """Forward user text to the conversation pipeline and wait for the reply."""
        conversation = getattr(self._app, "conversation", None)
        if conversation is None:
            await self._send_message(chat_id, "❌ Conversation service is not available.")
            return

        # Create a future to catch the BotReply.
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending_replies[chat_id] = future

        try:
            await conversation.handle_user_text(text, source="telegram")
            # Wait for the BotReply event.
            await asyncio.wait_for(future, timeout=self._config.chat_timeout_s)
            # The reply is sent via _on_bot_reply, so no need to send here.
        except TimeoutError:
            await self._send_message(chat_id, "⏳ The robot took too long to reply.")
        except Exception:
            _log.exception("telegram.chat_failed")
            await self._send_message(chat_id, "❌ Something went wrong processing your message.")
        finally:
            self._pending_replies.pop(chat_id, None)

    async def _on_bot_reply(self, event: BotReply) -> None:
        """When a BotReply is published, forward it to all pending Telegram chats."""
        reply_text = event.text
        for chat_id, future in list(self._pending_replies.items()):
            if not future.done():
                future.set_result(reply_text)
            await self._send_message(chat_id, reply_text)

    # ------------------------------------------------------------------ commands

    async def _handle_command(self, chat_id: int, text: str, user_id: int) -> None:
        """Parse and execute a slash command."""
        parts = text.split(maxsplit=1)
        command = parts[0].lower().rstrip("@")
        args_str = parts[1] if len(parts) > 1 else ""
        args = args_str.split()

        handler = _COMMAND_MAP.get(command)
        if handler is None:
            await self._send_message(chat_id, _help_text())
            return

        try:
            response = await handler(self, args_str, args)
        except Exception as exc:
            _log.exception("telegram.command_failed", command=command)
            response = f"❌ Command failed: {exc}"

        if response:
            await self._send_message(chat_id, response)

    # --- /emotion <name> [intensity] ---
    async def _cmd_emotion(self, args_str: str, args: list[str]) -> str:
        from robot.events.events import BlinkRequested, EmotionChanged, EmotionName

        if not args:
            names = [e.value for e in EmotionName]
            return f"Usage: /emotion <name> [intensity 0-1]\nEmotions: {', '.join(names)}"

        name = args[0].lower()
        try:
            emotion = EmotionName(name)
        except ValueError:
            return f"Unknown emotion '{name}'. Valid: {', '.join(e.value for e in EmotionName)}"

        intensity = 1.0
        if len(args) > 1:
            try:
                intensity = max(0.0, min(1.0, float(args[1])))
            except ValueError:
                return "Intensity must be a number between 0 and 1."

        await self._bus.publish(
            EmotionChanged(previous=EmotionName.NEUTRAL, current=emotion, intensity=intensity)
        )
        await self._bus.publish(BlinkRequested(speed=1.0))
        return f"😊 Emotion set to {emotion.value} (intensity {intensity})."

    # --- /state <name> ---
    async def _cmd_state(self, args_str: str, args: list[str]) -> str:
        from robot.behavior.state_machine import RobotState
        from robot.events.events import StateChanged

        if not args:
            current = self._app.state_machine.state.value
            states = [s.value for s in RobotState]
            return f"Current state: {current}\nUsage: /state <name>\nStates: {', '.join(states)}"

        name = args[0].lower()
        try:
            state = RobotState(name)
        except ValueError:
            return f"Unknown state '{name}'. Valid: {', '.join(s.value for s in RobotState)}"

        previous = self._app.state_machine.state
        await self._bus.publish(StateChanged(previous=previous, current=state))
        return f"🔄 State changed to {state.value}."

    # --- /speak <text> ---
    async def _cmd_speak(self, args_str: str, args: list[str]) -> str:
        if not args_str:
            return "Usage: /speak <text>"

        conversation = getattr(self._app, "conversation", None)
        if conversation is None or conversation.tts is None:
            return "❌ TTS is not available."

        tts = conversation.tts
        audio = getattr(self._app, "_audio", None)

        try:
            buffer = await tts.speak(args_str)
            if audio is not None and not buffer.is_empty:
                await audio.play(buffer)
            return f'🗣️ Spoke: "{args_str[:100]}"'
        except Exception as exc:
            return f"❌ TTS failed: {exc}"

    # --- /servo <name> <angle> [duration] ---
    async def _cmd_servo(self, args_str: str, args: list[str]) -> str:
        if len(args) < 2:
            names = list(self._app.servo_controller.servo_names)
            return f"Usage: /servo <name> <angle> [duration_s]\nServos: {', '.join(names)}"

        name = args[0]
        try:
            angle = float(args[1])
        except ValueError:
            return "Angle must be a number."

        duration = float(args[2]) if len(args) > 2 else 0.4

        try:
            servo = self._app.servo_controller.get(name)
            await servo.move_to(angle, duration_s=duration)
            from robot.events.events import ServoMoved

            await self._bus.publish(ServoMoved(name=name, angle=angle))
            return f"🎛️ Servo {name} moved to {angle}° over {duration}s."
        except Exception as exc:
            return f"❌ Servo error: {exc}"

    # --- /sound <name> ---
    async def _cmd_sound(self, args_str: str, args: list[str]) -> str:
        if not args:
            sfx = getattr(self._app, "_sound_effects", None)
            if sfx is not None:
                names = sfx.list_sounds()
                return f"Usage: /sound <name>\nAvailable: {', '.join(names)}"
            return "Usage: /sound <name>"

        name = args[0]
        sfx = getattr(self._app, "_sound_effects", None)
        if sfx is None:
            return "❌ Sound effects are not available."

        played = await sfx.play(name)
        if played:
            return f"🔊 Played sound: {name}"
        return f"❌ Sound '{name}' not found. Use /sound (no args) to list available sounds."

    # --- /behavior <name> ---
    async def _cmd_behavior(self, args_str: str, args: list[str]) -> str:
        from robot.behavior_library.behavior import (
            BehaviorRunner,
            excited,
            greeting,
            listening,
            sleeping,
            surprised,
            thinking,
        )

        behaviors = {
            "greeting": greeting,
            "thinking": thinking,
            "listening": listening,
            "sleeping": sleeping,
            "excited": excited,
            "surprised": surprised,
        }

        if not args:
            return f"Usage: /behavior <name>\nBehaviors: {', '.join(sorted(behaviors))}"

        name = args[0].lower()
        factory = behaviors.get(name)
        if factory is None:
            return f"Unknown behavior '{name}'. Available: {', '.join(sorted(behaviors))}"

        face_orch = getattr(self._app, "face_animator", None)
        body_engine = getattr(self._app, "_body_engine", None)
        if face_orch is None or body_engine is None:
            return "❌ Face animator or body engine is not available."

        from robot.utils.clock import SystemClock

        runner = BehaviorRunner(
            face=face_orch,
            body=body_engine,
            clock=SystemClock(),
        )
        await runner.run(factory())
        return f"🎭 Played behavior: {name}"

    # --- /status ---
    async def _cmd_status(self, args_str: str, args: list[str]) -> str:
        lines: list[str] = []
        lines.append("🤖 *DeskBot Status*")
        lines.append("")

        # State
        state = self._app.state_machine.state.value
        lines.append(f"State: `{state}`")

        # Degradation
        degradation = getattr(self._app, "_degradation", None)
        if degradation is not None:
            summary = degradation.summary()
            if summary:
                lines.append(f"Degradation: {summary}")

        # Hardware
        lines.append(f"Display: `{type(self._app.display).__name__}`")
        lines.append(f"Servos: `{type(self._app.servo_controller).__name__}`")
        audio = getattr(self._app, "_audio", None)
        if audio is not None:
            lines.append(f"Audio: `{type(audio).__name__}`")

        # Conversation / LLM
        conv = getattr(self._app, "conversation", None)
        if conv is not None:
            lines.append(f"LLM: `{type(conv.llm).__name__}`")
            lines.append(f"TTS: `{type(conv.tts).__name__}`")
            lines.append(f"STT: `{type(conv.stt).__name__}`")

        # Learning
        learning = getattr(self._app, "_learning_service", None)
        if learning is not None:
            status = learning.status
            lines.append(
                f"Learning: {status.total_experiences} experiences, "
                f"cycle {status.training_cycles_completed}, "
                f"{'training' if status.is_training else 'idle'}"
            )

        # Perception
        perception = getattr(self._app, "perception", None)
        lines.append(f"Perception: {'active' if perception is not None else 'disabled'}")

        # Servo positions
        try:
            for name in self._app.servo_controller.servo_names:
                servo = self._app.servo_controller.get(name)
                angle = getattr(servo, "angle", "?")
                lines.append(f"  servo {name}: {angle}°")
        except Exception:
            pass

        return "\n".join(lines)

    # --- /config ---
    async def _cmd_config(self, args_str: str, args: list[str]) -> str:
        import json as _json

        settings = self._app.settings
        d = settings.model_dump()
        # Mask sensitive values.
        for key in ("telegram",):
            if key in d and isinstance(d[key], dict) and "bot_token" in d[key]:
                d[key]["bot_token"] = "***"
        for key in ("llm",):
            if key in d and isinstance(d[key], dict) and "api_key" in d[key]:
                d[key]["api_key"] = "***" if d[key]["api_key"] else ""
        for key in ("tts",):
            if key in d and isinstance(d[key], dict) and "elevenlabs" in d[key]:
                el = d[key]["elevenlabs"]
                if isinstance(el, dict) and "api_key" in el:
                    el["api_key"] = "***" if el["api_key"] else ""

        # If a specific key is requested, show just that.""

        # If a specific key is requested, show just that.
        if args:
            parts = args[0].split(".")
            val: Any = d
            for p in parts:
                if isinstance(val, dict) and p in val:
                    val = val[p]
                else:
                    return f"Config key '{args[0]}' not found."
            return f"```\n{_json.dumps(val, indent=2, default=str)[:2000]}\n```"

        return f"```\n{_json.dumps(d, indent=2, default=str)[:3000]}\n```"

    # --- /help ---
    async def _cmd_help(self, args_str: str, args: list[str]) -> str:
        return _help_text()

    # ------------------------------------------------------------------ Telegram API

    async def _send_message(self, chat_id: int, text: str) -> None:
        """Send a text message to a Telegram chat."""
        if self._http is None:
            return
        # Telegram message limit is 4096 chars.
        for i in range(0, len(text), 4096):
            chunk = text[i : i + 4096]
            with contextlib.suppress(Exception):
                await self._http.post(
                    f"/bot{self._config.bot_token}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": chunk,
                        "parse_mode": "Markdown",
                    },
                )


# ---------------------------------------------------------------------------
# Command registry
# ---------------------------------------------------------------------------

_COMMAND_MAP: dict[str, Any] = {}


def _register(command: str) -> Any:
    """Decorator to register a command handler."""

    def decorator(method: Any) -> Any:
        _COMMAND_MAP[command] = method
        return method

    return decorator


# Register all command handlers.
_register("emotion")(TelegramBridge._cmd_emotion)
_register("state")(TelegramBridge._cmd_state)
_register("speak")(TelegramBridge._cmd_speak)
_register("servo")(TelegramBridge._cmd_servo)
_register("sound")(TelegramBridge._cmd_sound)
_register("behavior")(TelegramBridge._cmd_behavior)
_register("status")(TelegramBridge._cmd_status)
_register("config")(TelegramBridge._cmd_config)
_register("help")(TelegramBridge._cmd_help)


def _help_text() -> str:
    return """🤖 *DeskBot Telegram Bridge*

*Chat:* Send any text to talk to the robot (goes through the LLM conversation pipeline).

*Commands:*
/emotion `<name>` `[intensity]` — Set the robot's emotion
  Emotions: neutral, happy, curious, thinking, sleepy, embarrassed, excited, sad, surprised, angry
/state `<name>` — Change the robot's state
  States: boot, idle, curious, listening, thinking, speaking, sleeping, error
/speak `<text>` — Make the robot speak via TTS (bypasses LLM)
/servo `<name>` `<angle>` `[duration_s]` — Move a servo
  Servos: pan, tilt, left_arm, right_arm
/sound `<name>` — Play a sound effect
  Use `/sound` alone to list available sounds
/behavior `<name>` — Play a behavior sequence
  Behaviors: greeting, thinking, listening, sleeping, excited, surprised
/status — Show current robot status (state, hardware, LLM, learning, servos)
/config `[key]` — Show configuration (sensitive values masked)
/help — Show this help message"""


__all__ = ["TelegramBridge", "TelegramConfig"]
