"""Tests for the eSpeak-NG TTS driver."""

from __future__ import annotations

import struct

import pytest

from robot.speech.tts_espeak import EspeakNGTTS


def test_espeak_tts_name() -> None:
    tts = EspeakNGTTS(voice="en")
    assert tts.name == "espeak-ng:en"


def test_espeak_tts_custom_voice() -> None:
    tts = EspeakNGTTS(voice="de")
    assert tts.name == "espeak-ng:de"


def test_espeak_tts_default_speed() -> None:
    tts = EspeakNGTTS()
    assert tts._speed == 175


def test_espeak_tts_default_pitch() -> None:
    tts = EspeakNGTTS()
    assert tts._pitch == 50


def test_espeak_tts_parse_wav_mono() -> None:
    """Test WAV parsing with a synthetic mono WAV - format metadata preserved."""
    import io
    import wave

    from robot.interfaces.audio import AudioBuffer

    tts = EspeakNGTTS()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(struct.pack("<10h", *range(10)))
    wav_bytes = buf.getvalue()
    audio_buf = tts._parse_wav(wav_bytes)
    assert isinstance(audio_buf, AudioBuffer)
    assert audio_buf.sample_rate == 22050
    assert audio_buf.channels == 1
    assert len(audio_buf.pcm) == 20  # 10 samples * 2 bytes each


def test_espeak_tts_parse_wav_stereo() -> None:
    """Test WAV parsing with stereo WAV - channels preserved."""
    import io
    import wave

    from robot.interfaces.audio import AudioBuffer

    tts = EspeakNGTTS()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        frames = struct.pack("<10h", 0, 0, 1, 1, 2, 2, 3, 3, 4, 4)
        wav.writeframes(frames)
    wav_bytes = buf.getvalue()
    audio_buf = tts._parse_wav(wav_bytes)
    assert isinstance(audio_buf, AudioBuffer)
    assert audio_buf.sample_rate == 22050
    assert audio_buf.channels == 2  # stereo preserved
    assert len(audio_buf.pcm) == 20  # 5 frames * 2 channels * 2 bytes


@pytest.mark.anyio
async def test_espeak_tts_close() -> None:
    tts = EspeakNGTTS()
    await tts.close()
    # close() returns None implicitly
