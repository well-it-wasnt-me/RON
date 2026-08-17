"""Tests for the Telegram bridge service."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robot.services.telegram_bridge import TelegramBridge, TelegramConfig


@pytest.fixture
def config() -> TelegramConfig:
    return TelegramConfig(
        bot_token="test:token",
        enabled=True,
        allowed_user_ids=[123],
        chat_timeout_s=5.0,
    )


@pytest.fixture
def mock_app() -> MagicMock:
    app = MagicMock()
    app.state_machine = MagicMock()
    app.state_machine.state = MagicMock()
    app.state_machine.state.value = "idle"
    app.servo_controller = MagicMock()
    app.servo_controller.servo_names = ["pan", "tilt", "left_arm", "right_arm"]
    servo = MagicMock()
    servo.angle = 90.0
    servo.move_to = AsyncMock()
    app.servo_controller.get.return_value = servo
    app.display = MagicMock()
    app.display.__class__.__name__ = "MockDisplay"
    app.servo_controller.__class__.__name__ = "MockServoBus"
    app.conversation = MagicMock()
    app.conversation.llm = MagicMock()
    app.conversation.llm.__class__.__name__ = "MockLLM"
    app.conversation.tts = MagicMock()
    app.conversation.tts.__class__.__name__ = "MockTTS"
    app.conversation.stt = MagicMock()
    app.conversation.stt.__class__.__name__ = "MockSTT"
    app.conversation.handle_user_text = AsyncMock()
    app._audio = MagicMock()
    app._audio.__class__.__name__ = "MockAudioOutput"
    app._sound_effects = MagicMock()
    app._sound_effects.list_sounds.return_value = ["angry", "cute", "talk"]
    app._sound_effects.play = AsyncMock(return_value=True)
    app._learning_service = None
    app.perception = None
    app._degradation = MagicMock()
    app._degradation.summary.return_value = "ok"
    app.settings = MagicMock()
    app.settings.model_dump.return_value = {"telegram": {"bot_token": "secret"}}
    app.face_animator = MagicMock()
    app._body_engine = MagicMock()
    return app


@pytest.fixture
def bridge(config, mock_app):
    bus = MagicMock()
    bus.subscribe = MagicMock()
    bus.unsubscribe = MagicMock()
    bus.publish = AsyncMock()
    return TelegramBridge(config=config, bus=bus, app=mock_app)


class TestTelegramConfig:
    def test_defaults(self) -> None:
        cfg = TelegramConfig()
        assert cfg.enabled is False
        assert cfg.bot_token == ""
        assert cfg.allowed_user_ids == []
        assert cfg.chat_timeout_s == 60.0
        assert cfg.api_base == "https://api.telegram.org"


class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_calls_get_me(self, bridge: TelegramBridge) -> None:
        """start() should verify the bot token and subscribe to events."""
        mock_http = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "result": {"username": "DeskBotBot"}}
        mock_response.raise_for_status = MagicMock()
        mock_http.get.return_value = mock_response
        mock_http.aclose = AsyncMock()

        with (
            patch("httpx.AsyncClient", return_value=mock_http),
            patch.object(bridge, "_poll_loop", new_callable=AsyncMock),
        ):
            await bridge.start()

        mock_http.get.assert_called_once()
        assert bridge._http is not None
        assert bridge._poll_task is not None

    @pytest.mark.asyncio
    async def test_start_rejects_bad_token(self, bridge: TelegramBridge) -> None:
        """start() should raise if getMe fails."""
        mock_http = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "description": "bad token"}
        mock_response.raise_for_status = MagicMock()
        mock_http.get.return_value = mock_response
        mock_http.aclose = AsyncMock()

        with (
            patch("httpx.AsyncClient", return_value=mock_http),
            pytest.raises(RuntimeError, match="getMe failed"),
        ):
            await bridge.start()

    @pytest.mark.asyncio
    async def test_stop_cleans_up(self, bridge: TelegramBridge) -> None:
        """stop() should cancel the poll task and close http."""
        mock_http = AsyncMock()
        mock_http.aclose = AsyncMock()
        bridge._http = mock_http
        bridge._poll_task = asyncio.create_task(asyncio.sleep(100))

        await bridge.stop()

        # stop() sets both to None.
        poll_task = getattr(bridge, "_poll_task", "not-none")
        http = getattr(bridge, "_http", "not-none")
        assert poll_task is None
        assert http is None


class TestCommandHandling:
    @pytest.mark.asyncio
    async def test_emotion_command(self, bridge: TelegramBridge) -> None:
        result = await bridge._cmd_emotion("happy 0.8", ["happy", "0.8"])
        assert "happy" in result
        assert "0.8" in result
        bridge._bus.publish.assert_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_emotion_invalid(self, bridge: TelegramBridge) -> None:
        result = await bridge._cmd_emotion("bogus", ["bogus"])
        assert "Unknown emotion" in result

    @pytest.mark.asyncio
    async def test_emotion_no_args_lists(self, bridge: TelegramBridge) -> None:
        result = await bridge._cmd_emotion("", [])
        assert "neutral" in result
        assert "happy" in result

    @pytest.mark.asyncio
    async def test_state_command(self, bridge: TelegramBridge) -> None:
        result = await bridge._cmd_state("curious", ["curious"])
        assert "curious" in result
        bridge._bus.publish.assert_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_state_invalid(self, bridge: TelegramBridge) -> None:
        result = await bridge._cmd_state("bogus", ["bogus"])
        assert "Unknown state" in result

    @pytest.mark.asyncio
    async def test_state_no_args_shows_current(self, bridge: TelegramBridge) -> None:
        result = await bridge._cmd_state("", [])
        assert "Current state: idle" in result

    @pytest.mark.asyncio
    async def test_speak_command(self, bridge: TelegramBridge) -> None:
        bridge._app.conversation.tts.speak = AsyncMock(return_value=MagicMock(is_empty=False))
        bridge._app._audio.play = AsyncMock()
        result = await bridge._cmd_speak("hello there", ["hello", "there"])
        assert "hello there" in result

    @pytest.mark.asyncio
    async def test_speak_empty(self, bridge: TelegramBridge) -> None:
        result = await bridge._cmd_speak("", [])
        assert "Usage:" in result

    @pytest.mark.asyncio
    async def test_servo_command(self, bridge: TelegramBridge) -> None:
        result = await bridge._cmd_servo("pan 45", ["pan", "45"])
        assert "pan" in result
        assert "45" in result
        bridge._app.servo_controller.get.assert_called_with("pan")

    @pytest.mark.asyncio
    async def test_servo_missing_args(self, bridge: TelegramBridge) -> None:
        result = await bridge._cmd_servo("pan", ["pan"])
        assert "Usage:" in result

    @pytest.mark.asyncio
    async def test_sound_command(self, bridge: TelegramBridge) -> None:
        result = await bridge._cmd_sound("cute", ["cute"])
        assert "cute" in result

    @pytest.mark.asyncio
    async def test_sound_no_args_lists(self, bridge: TelegramBridge) -> None:
        result = await bridge._cmd_sound("", [])
        assert "angry" in result
        assert "Usage:" in result

    @pytest.mark.asyncio
    async def test_behavior_command(self, bridge: TelegramBridge) -> None:
        # Mock the BehaviorRunner.run
        with patch("robot.behavior_library.behavior.BehaviorRunner") as mock_runner_cls:
            mock_runner = MagicMock()
            mock_runner.run = AsyncMock()
            mock_runner_cls.return_value = mock_runner
            result = await bridge._cmd_behavior("greeting", ["greeting"])
        assert "greeting" in result

    @pytest.mark.asyncio
    async def test_behavior_invalid(self, bridge: TelegramBridge) -> None:
        result = await bridge._cmd_behavior("bogus", ["bogus"])
        assert "Unknown behavior" in result

    @pytest.mark.asyncio
    async def test_status_command(self, bridge: TelegramBridge) -> None:
        result = await bridge._cmd_status("", [])
        assert "DeskBot Status" in result
        assert "State:" in result
        assert "idle" in result

    @pytest.mark.asyncio
    async def test_config_command(self, bridge: TelegramBridge) -> None:
        result = await bridge._cmd_config("", [])
        assert "bot_token" in result
        # Token should be masked.
        assert "secret" not in result
        assert "***" in result

    @pytest.mark.asyncio
    async def test_help_command(self, bridge: TelegramBridge) -> None:
        result = await bridge._cmd_help("", [])
        assert "DeskBot Telegram Bridge" in result
        assert "/emotion" in result
        assert "/status" in result


class TestAuthorization:
    @pytest.mark.asyncio
    async def test_unauthorized_user(self, bridge: TelegramBridge) -> None:
        """Messages from users not in allowed_user_ids should be rejected."""
        bridge._send_message = AsyncMock()  # type: ignore[method-assign]
        update = {
            "update_id": 1,
            "message": {
                "chat": {"id": 999},
                "from": {"id": 999},
                "text": "hello",
            },
        }
        await bridge._handle_update(update)
        bridge._send_message.assert_called_once()
        sent_text = bridge._send_message.call_args[0][1]
        assert "not authorised" in sent_text.lower()

    @pytest.mark.asyncio
    async def test_authorized_user_passes(self, bridge: TelegramBridge) -> None:
        """Messages from authorized users should be processed."""
        bridge._send_message = AsyncMock()  # type: ignore[method-assign]
        bridge._handle_chat = AsyncMock()  # type: ignore[method-assign]
        update = {
            "update_id": 1,
            "message": {
                "chat": {"id": 123},
                "from": {"id": 123},
                "text": "hello",
            },
        }
        await bridge._handle_update(update)
        bridge._handle_chat.assert_called_once_with(123, "hello")


class TestChatHandler:
    @pytest.mark.asyncio
    async def test_chat_no_conversation(self, bridge: TelegramBridge) -> None:
        bridge._app.conversation = None
        bridge._send_message = AsyncMock()  # type: ignore[method-assign]
        await bridge._handle_chat(123, "hello")
        bridge._send_message.assert_called_once()
        assert "not available" in bridge._send_message.call_args[0][1].lower()
