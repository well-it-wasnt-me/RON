"""Tests for Telegram bridge command parsing (the /emotion bug fix).

The bug was that ``_handle_command`` kept the leading ``/`` when looking
up the command in ``_COMMAND_MAP``, so every slash command missed the
lookup and fell through to the help text.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

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


class TestCommandParsingFix:
    """The core bug: /emotion happy returned help text instead of
    triggering the emotion command."""

    @pytest.mark.asyncio
    async def test_emotion_command_via_handle_command(self, bridge: TelegramBridge) -> None:
        """Full integration: /emotion happy should NOT return help text."""
        bridge._send_message = AsyncMock()  # type: ignore[method-assign]
        await bridge._handle_command(chat_id=123, text="/emotion happy", user_id=123)

        bridge._send_message.assert_called_once()
        sent_text = bridge._send_message.call_args[0][1]
        # The reply should mention the emotion, not the help header.
        assert "happy" in sent_text.lower()
        assert "DeskBot Telegram Bridge" not in sent_text  # help header

    @pytest.mark.asyncio
    async def test_emotion_with_bot_suffix(self, bridge: TelegramBridge) -> None:
        """Telegram group commands may include @BotName suffix."""
        bridge._send_message = AsyncMock()  # type: ignore[method-assign]
        await bridge._handle_command(chat_id=123, text="/emotion@DeskBotBot happy", user_id=123)
        bridge._send_message.assert_called_once()
        sent_text = bridge._send_message.call_args[0][1]
        assert "happy" in sent_text.lower()
        assert "DeskBot Telegram Bridge" not in sent_text

    @pytest.mark.asyncio
    async def test_state_command_via_handle_command(self, bridge: TelegramBridge) -> None:
        bridge._send_message = AsyncMock()  # type: ignore[method-assign]
        await bridge._handle_command(chat_id=123, text="/state curious", user_id=123)
        bridge._send_message.assert_called_once()
        sent_text = bridge._send_message.call_args[0][1]
        assert "curious" in sent_text.lower()
        assert "DeskBot Telegram Bridge" not in sent_text

    @pytest.mark.asyncio
    async def test_help_command_still_works(self, bridge: TelegramBridge) -> None:
        bridge._send_message = AsyncMock()  # type: ignore[method-assign]
        await bridge._handle_command(chat_id=123, text="/help", user_id=123)
        bridge._send_message.assert_called_once()
        sent_text = bridge._send_message.call_args[0][1]
        assert "DeskBot Telegram Bridge" in sent_text

    @pytest.mark.asyncio
    async def test_unknown_command_returns_help(self, bridge: TelegramBridge) -> None:
        bridge._send_message = AsyncMock()  # type: ignore[method-assign]
        await bridge._handle_command(chat_id=123, text="/bogus", user_id=123)
        bridge._send_message.assert_called_once()
        sent_text = bridge._send_message.call_args[0][1]
        assert "DeskBot Telegram Bridge" in sent_text

    @pytest.mark.asyncio
    async def test_emotion_no_args_via_handle_command(self, bridge: TelegramBridge) -> None:
        bridge._send_message = AsyncMock()  # type: ignore[method-assign]
        await bridge._handle_command(chat_id=123, text="/emotion", user_id=123)
        bridge._send_message.assert_called_once()
        sent_text = bridge._send_message.call_args[0][1]
        assert "Usage" in sent_text
        assert "neutral" in sent_text
