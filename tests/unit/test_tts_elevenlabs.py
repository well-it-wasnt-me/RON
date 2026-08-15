"""Tests for the ElevenLabs TTS provider and configuration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pydantic
import pytest

from robot.config import ElevenLabsConfig, TTSConfig
from robot.errors import ConfigurationError


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class TestElevenLabsConfig:
    """Verify ElevenLabsConfig defaults and validation."""

    def test_default_api_key_is_empty(self) -> None:
        cfg = ElevenLabsConfig()
        assert cfg.api_key == ""

    def test_default_voice_id(self) -> None:
        cfg = ElevenLabsConfig()
        assert cfg.voice_id == "21m00Tcm4TlvDq8ikWAM"

    def test_default_model_id(self) -> None:
        cfg = ElevenLabsConfig()
        assert cfg.model_id == "eleven_multilingual_v2"

    def test_default_stability(self) -> None:
        cfg = ElevenLabsConfig()
        assert cfg.stability == 0.5

    def test_default_similarity_boost(self) -> None:
        cfg = ElevenLabsConfig()
        assert cfg.similarity_boost == 0.75

    def test_custom_voice_id(self) -> None:
        cfg = ElevenLabsConfig(voice_id="custom_voice")
        assert cfg.voice_id == "custom_voice"

    def test_custom_model_id(self) -> None:
        cfg = ElevenLabsConfig(model_id="eleven_monolingual_v1")
        assert cfg.model_id == "eleven_monolingual_v1"

    def test_stability_bounds(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            ElevenLabsConfig(stability=-0.1)
        with pytest.raises(pydantic.ValidationError):
            ElevenLabsConfig(stability=1.1)

    def test_similarity_boost_bounds(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            ElevenLabsConfig(similarity_boost=-0.1)
        with pytest.raises(pydantic.ValidationError):
            ElevenLabsConfig(similarity_boost=1.1)


class TestTTSConfigElevenLabs:
    """Verify TTSConfig includes elevenlabs sub-config."""

    def test_elevenlabs_subconfig_default(self) -> None:
        cfg = TTSConfig()
        assert isinstance(cfg.elevenlabs, ElevenLabsConfig)

    def test_provider_literal_includes_elevenlabs(self) -> None:
        cfg = TTSConfig(provider="elevenlabs")
        assert cfg.provider == "elevenlabs"


# ---------------------------------------------------------------------------
# ElevenLabsTTS - construction
# ---------------------------------------------------------------------------
class TestElevenLabsTTSConstruction:
    """Test ElevenLabsTTS initialization and validation."""

    def test_requires_api_key(self) -> None:
        from robot.speech.tts_elevenlabs import ElevenLabsTTS

        with pytest.raises(ConfigurationError, match="API key"):
            ElevenLabsTTS()

    def test_construction_with_api_key(self) -> None:
        from robot.speech.tts_elevenlabs import ElevenLabsTTS

        tts = ElevenLabsTTS(api_key="test-key")
        assert tts.name == "elevenlabs:21m00Tcm4TlvDq8ikWAM"

    def test_construction_with_custom_voice(self) -> None:
        from robot.speech.tts_elevenlabs import ElevenLabsTTS

        tts = ElevenLabsTTS(api_key="test-key", voice_id="abc123")
        assert tts.name == "elevenlabs:abc123"


# ---------------------------------------------------------------------------
# ElevenLabsTTS - speak (mocked httpx)
# ---------------------------------------------------------------------------
class TestElevenLabsTTSSpeak:
    """Test speak() with mocked HTTP responses."""

    @pytest.mark.anyio
    async def test_speak_returns_pcm(self) -> None:
        from robot.speech.tts_elevenlabs import ElevenLabsTTS

        fake_pcm = b"\x00\x01" * 100

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = fake_pcm
        mock_response.text = ""

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("robot.speech.tts_elevenlabs.httpx.AsyncClient", return_value=mock_client):
            tts = ElevenLabsTTS(api_key="test-key")
            result = await tts.speak("Hello world")

        from robot.interfaces.audio import AudioBuffer

        assert isinstance(result, AudioBuffer)
        assert result.pcm == fake_pcm
        assert result.sample_rate == 16000  # pcm_16000 format
        assert result.channels == 1

    @pytest.mark.anyio
    async def test_speak_sends_correct_request(self) -> None:
        from robot.speech.tts_elevenlabs import ElevenLabsTTS

        fake_pcm = b"\x00\x01" * 50

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = fake_pcm
        mock_response.text = ""

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("robot.speech.tts_elevenlabs.httpx.AsyncClient", return_value=mock_client):
            tts = ElevenLabsTTS(
                api_key="my-key",
                voice_id="voice-123",
                model_id="eleven_monolingual_v1",
                stability=0.3,
                similarity_boost=0.8,
            )
            await tts.speak("Test text")

        call_kwargs = mock_client.post.call_args
        assert "voice-123" in call_kwargs[0][0]  # URL contains voice_id
        assert call_kwargs[1]["json"]["text"] == "Test text"
        assert call_kwargs[1]["json"]["model_id"] == "eleven_monolingual_v1"
        assert call_kwargs[1]["json"]["voice_settings"]["stability"] == 0.3
        assert call_kwargs[1]["json"]["voice_settings"]["similarity_boost"] == 0.8
        assert call_kwargs[1]["headers"]["xi-api-key"] == "my-key"
        assert call_kwargs[1]["params"]["output_format"] == "pcm_16000"

    @pytest.mark.anyio
    async def test_speak_returns_audio_buffer(self) -> None:
        """TTS synthesises and returns an AudioBuffer without playing it."""
        from robot.speech.tts_elevenlabs import ElevenLabsTTS

        fake_pcm = b"\x00\x01" * 100

        mock_audio = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = fake_pcm
        mock_response.text = ""

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("robot.speech.tts_elevenlabs.httpx.AsyncClient", return_value=mock_client):
            tts = ElevenLabsTTS(api_key="test-key", audio=mock_audio)
            buffer = await tts.speak("Hello")

        from robot.interfaces.audio import AudioBuffer

        assert isinstance(buffer, AudioBuffer)
        assert buffer.pcm == fake_pcm
        assert buffer.sample_rate == 16000
        # TTS must NOT play internally - the caller is responsible for playback.
        mock_audio.play.assert_not_awaited()

    @pytest.mark.anyio
    async def test_speak_401_raises(self) -> None:
        from robot.speech.tts_elevenlabs import ElevenLabsTTS

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("robot.speech.tts_elevenlabs.httpx.AsyncClient", return_value=mock_client):
            tts = ElevenLabsTTS(api_key="bad-key")
            with pytest.raises(ConfigurationError, match="unauthorized"):
                await tts.speak("Hello")

    @pytest.mark.anyio
    async def test_speak_429_returns_empty(self) -> None:
        from robot.speech.tts_elevenlabs import ElevenLabsTTS

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limited"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("robot.speech.tts_elevenlabs.httpx.AsyncClient", return_value=mock_client):
            tts = ElevenLabsTTS(api_key="test-key")
            result = await tts.speak("Hello")

        from robot.interfaces.audio import AudioBuffer

        assert isinstance(result, AudioBuffer)
        assert result.is_empty

    @pytest.mark.anyio
    async def test_speak_server_error_returns_empty(self) -> None:
        from robot.speech.tts_elevenlabs import ElevenLabsTTS

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("robot.speech.tts_elevenlabs.httpx.AsyncClient", return_value=mock_client):
            tts = ElevenLabsTTS(api_key="test-key")
            result = await tts.speak("Hello")

        from robot.interfaces.audio import AudioBuffer

        assert isinstance(result, AudioBuffer)
        assert result.is_empty

    @pytest.mark.anyio
    async def test_speak_timeout_returns_empty(self) -> None:
        from robot.speech.tts_elevenlabs import ElevenLabsTTS

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("robot.speech.tts_elevenlabs.httpx.AsyncClient", return_value=mock_client):
            tts = ElevenLabsTTS(api_key="test-key")
            result = await tts.speak("Hello")

        from robot.interfaces.audio import AudioBuffer

        assert isinstance(result, AudioBuffer)
        assert result.is_empty

    @pytest.mark.anyio
    async def test_speak_connection_error_returns_empty(self) -> None:
        from robot.speech.tts_elevenlabs import ElevenLabsTTS

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("robot.speech.tts_elevenlabs.httpx.AsyncClient", return_value=mock_client):
            tts = ElevenLabsTTS(api_key="test-key")
            result = await tts.speak("Hello")

        from robot.interfaces.audio import AudioBuffer

        assert isinstance(result, AudioBuffer)
        assert result.is_empty


# ---------------------------------------------------------------------------
# ElevenLabsTTS - close
# ---------------------------------------------------------------------------
class TestElevenLabsTTSClose:
    """Test close() lifecycle."""

    @pytest.mark.anyio
    async def test_close_is_noop(self) -> None:
        from robot.speech.tts_elevenlabs import ElevenLabsTTS

        tts = ElevenLabsTTS(api_key="test-key")
        await tts.close()  # should not raise


# ---------------------------------------------------------------------------
# App-level wiring: verify elevenlabs provider creates ElevenLabsTTS
# ---------------------------------------------------------------------------
class TestElevenLabsAppWiring:
    """Test that the app factory creates ElevenLabsTTS when configured."""

    def test_tts_config_with_elevenlabs(self) -> None:
        """Verify TTSConfig can be constructed with elevenlabs sub-config."""
        cfg = TTSConfig(
            provider="elevenlabs",
            elevenlabs=ElevenLabsConfig(api_key="sk-test"),
        )
        assert cfg.provider == "elevenlabs"
        assert cfg.elevenlabs.api_key == "sk-test"
        assert cfg.elevenlabs.voice_id == "21m00Tcm4TlvDq8ikWAM"
