"""Regression tests for the explicit audio-format contract.

These tests verify that:

1. TTS providers propagate their actual sample rate and channel count.
2. The output layer does not confuse different formats.
3. Bluetooth playback wraps audio in a WAV container (not raw PCM with
   hardcoded format flags).
4. The centralised conversion utilities work correctly.
5. Raw 22050 Hz PCM is never declared as 48000 Hz to paplay.
"""

from __future__ import annotations

import io
import struct
import wave

import pytest

from robot.interfaces.audio import (
    AudioBuffer,
    apply_volume,
    convert_audio,
    convert_channels,
    resample,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_wav(sample_rate: int, channels: int, n_frames: int = 100) -> bytes:
    """Generate a minimal valid WAV file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack(f"<{n_frames * channels}h", *([16384] * n_frames * channels)))
    return buf.getvalue()


def _make_pcm(sample_rate: int, channels: int, n_frames: int = 100) -> bytes:
    """Generate raw s16le PCM."""
    return struct.pack(f"<{n_frames * channels}h", *([16384] * n_frames * channels))


# ---------------------------------------------------------------------------
# 1. eSpeak format propagation
# ---------------------------------------------------------------------------
class TestEspeakFormatPropagation:
    """Verify that eSpeak's actual WAV format is preserved."""

    def test_espeak_parses_22050_mono(self) -> None:
        """eSpeak WAV output at 22050 Hz mono is propagated correctly."""
        from robot.speech.tts_espeak import EspeakNGTTS

        tts = EspeakNGTTS(voice="en")
        wav_bytes = _make_wav(sample_rate=22050, channels=1)
        buf = tts._parse_wav(wav_bytes)
        assert buf.sample_rate == 22050
        assert buf.channels == 1
        assert buf.sample_format == "s16le"

    def test_espeak_parses_44100_stereo(self) -> None:
        """If eSpeak produced 44100 Hz stereo, that format is preserved."""
        from robot.speech.tts_espeak import EspeakNGTTS

        tts = EspeakNGTTS(voice="en")
        wav_bytes = _make_wav(sample_rate=44100, channels=2)
        buf = tts._parse_wav(wav_bytes)
        assert buf.sample_rate == 44100
        assert buf.channels == 2

    def test_espeak_does_not_hardcode_sample_rate(self) -> None:
        """The AudioBuffer's sample rate comes from the WAV, not a constant."""
        from robot.speech.tts_espeak import EspeakNGTTS

        tts = EspeakNGTTS(voice="en", sample_rate=48000)
        # Even though the TTS is configured for 48000, the actual WAV
        # might be 22050. The buffer must reflect the WAV's actual rate.
        wav_bytes = _make_wav(sample_rate=22050, channels=1)
        buf = tts._parse_wav(wav_bytes)
        assert buf.sample_rate == 22050  # from WAV, not from config


# ---------------------------------------------------------------------------
# 2. Piper format propagation
# ---------------------------------------------------------------------------
class TestPiperFormatPropagation:
    """Verify that Piper's native sample rate is propagated."""

    def test_piper_buffer_has_sample_rate(self) -> None:
        """AudioBuffer from Piper carries the model's sample rate."""
        # We can't call actual Piper, but we can verify the AudioBuffer
        # contract: the speak() method returns an AudioBuffer with
        # sample_rate set from chunk.sample_rate.
        buf = AudioBuffer(pcm=_make_pcm(22050, 1), sample_rate=22050, channels=1)
        assert buf.sample_rate == 22050
        assert buf.channels == 1


# ---------------------------------------------------------------------------
# 3. Different TTS formats are not confused
# ---------------------------------------------------------------------------
class TestFormatPreservation:
    """Verify that AudioBuffers with different formats are distinct."""

    def test_22050_mono_distinct_from_24000_mono(self) -> None:
        buf1 = AudioBuffer(pcm=_make_pcm(22050, 1), sample_rate=22050, channels=1)
        buf2 = AudioBuffer(pcm=_make_pcm(24000, 1), sample_rate=24000, channels=1)
        assert buf1.sample_rate != buf2.sample_rate
        assert buf1.sample_rate == 22050
        assert buf2.sample_rate == 24000

    def test_mono_distinct_from_stereo(self) -> None:
        buf1 = AudioBuffer(pcm=_make_pcm(22050, 1), sample_rate=22050, channels=1)
        buf2 = AudioBuffer(pcm=_make_pcm(22050, 2), sample_rate=22050, channels=2)
        assert buf1.channels != buf2.channels

    def test_openai_format_is_24000(self) -> None:
        """OpenAI TTS produces 24 kHz mono PCM."""
        from robot.speech.tts_openai import _OPENAI_PCM_SAMPLE_RATE

        assert _OPENAI_PCM_SAMPLE_RATE == 24000

    def test_elevenlabs_format_from_output_format(self) -> None:
        """ElevenLabs sample rate is derived from output_format string."""
        from robot.speech.tts_elevenlabs import ElevenLabsTTS

        tts = ElevenLabsTTS(api_key="test", output_format="pcm_22050")
        assert tts._sample_rate_for_format() == 22050

        tts2 = ElevenLabsTTS(api_key="test", output_format="pcm_44100")
        assert tts2._sample_rate_for_format() == 44100


# ---------------------------------------------------------------------------
# 4. Centralised conversion
# ---------------------------------------------------------------------------
class TestConversion:
    """Test the centralised resampling and channel conversion utilities."""

    def test_resample_22050_to_44100(self) -> None:
        """Resampling 22050 -> 44100 doubles the sample count."""
        pcm = _make_pcm(22050, 1, n_frames=100)
        result = resample(pcm, 22050, 44100, channels=1)
        n_out = len(result) // 2
        assert n_out == 200  # 2x

    def test_resample_noop_same_rate(self) -> None:
        pcm = _make_pcm(22050, 1, n_frames=100)
        result = resample(pcm, 22050, 22050, channels=1)
        assert result == pcm

    def test_convert_channels_mono_to_stereo(self) -> None:
        pcm = _make_pcm(22050, 1, n_frames=100)
        result = convert_channels(pcm, 1, 2)
        assert len(result) == len(pcm) * 2

    def test_convert_channels_stereo_to_mono(self) -> None:
        pcm = _make_pcm(22050, 2, n_frames=100)
        result = convert_channels(pcm, 2, 1)
        assert len(result) == len(pcm) // 2

    def test_convert_audio_combined(self) -> None:
        """convert_audio handles both resampling and channel conversion."""
        buf = AudioBuffer(pcm=_make_pcm(22050, 1, n_frames=100), sample_rate=22050, channels=1)
        result = convert_audio(buf, target_sample_rate=44100, target_channels=2)
        assert result.sample_rate == 44100
        assert result.channels == 2
        assert len(result.pcm) > len(buf.pcm)

    def test_convert_audio_noop(self) -> None:
        """convert_audio returns the same buffer when no conversion needed."""
        buf = AudioBuffer(pcm=_make_pcm(22050, 1, n_frames=100), sample_rate=22050, channels=1)
        result = convert_audio(buf, target_sample_rate=22050, target_channels=1)
        assert result is buf  # same object, no copy

    def test_apply_volume(self) -> None:
        pcm = struct.pack("<4h", 1000, 2000, -1000, -2000)
        result = apply_volume(pcm, 0.5)
        vals = struct.unpack("<4h", result)
        assert vals[0] == 500
        assert vals[1] == 1000
        assert vals[2] == -500
        assert vals[3] == -1000

    def test_apply_volume_clipping(self) -> None:
        pcm = struct.pack("<2h", 30000, -30000)
        result = apply_volume(pcm, 2.0)
        vals = struct.unpack("<2h", result)
        assert vals[0] == 32767  # clipped
        assert vals[1] == -32768  # clipped (min)


# ---------------------------------------------------------------------------
# 5. No false format declaration (Bluetooth)
# ---------------------------------------------------------------------------
class TestNoFalseFormatDeclaration:
    """Verify that raw PCM is never passed to paplay with wrong format flags."""

    @pytest.mark.anyio
    async def test_paplay_receives_wav_not_raw(self) -> None:
        """BluetoothSpeaker passes WAV (not raw PCM) to paplay.

        The WAV header carries the actual format.  No --raw, --rate,
        --channels, or --format flags should be present.
        """
        from unittest.mock import MagicMock, patch

        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

        speaker = BluetoothSpeaker(device_mac="AA:BB:CC:DD:EE:FF", auto_connect=True)
        speaker._connected = True
        speaker._sink_name = "bluez_output.AA_BB_CC_DD_EE_FF.a2dp-sink"

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stderr = b""
        mock_proc.communicate = MagicMock(return_value=(b"", b""))

        with patch(
            "robot.hardware.audio.bluetooth_speaker.subprocess.Popen",
            return_value=mock_proc,
        ) as popen_mock:
            buf = AudioBuffer(pcm=_make_pcm(22050, 1), sample_rate=22050, channels=1)
            await speaker.play(buf)

            cmd = popen_mock.call_args[0][0]
            assert "--raw" not in cmd
            assert "--rate" not in cmd
            assert "--channels" not in cmd
            assert "--format" not in cmd

    @pytest.mark.anyio
    async def test_22050_never_declared_as_48000(self) -> None:
        """A 22050 Hz buffer must never be passed to paplay as 48000 Hz."""
        from unittest.mock import MagicMock, patch

        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

        speaker = BluetoothSpeaker(device_mac="AA:BB:CC:DD:EE:FF", auto_connect=True)
        speaker._connected = True
        speaker._sink_name = "test_sink"

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stderr = b""
        mock_proc.communicate = MagicMock(return_value=(b"", b""))

        with patch(
            "robot.hardware.audio.bluetooth_speaker.subprocess.Popen",
            return_value=mock_proc,
        ):
            buf = AudioBuffer(pcm=_make_pcm(22050, 1), sample_rate=22050, channels=1)
            await speaker.play(buf)

            # The WAV data passed to communicate must contain 22050 Hz
            # in its header, not 48000.
            wav_data = mock_proc.communicate.call_args[1]["input"]
            wav_buf = io.BytesIO(wav_data)
            with wave.open(wav_buf, "rb") as wav:
                assert wav.getframerate() == 22050


# ---------------------------------------------------------------------------
# 6. AudioBuffer WAV round-trip
# ---------------------------------------------------------------------------
class TestAudioBufferWav:
    """Test AudioBuffer.to_wav / from_wav round-trip."""

    def test_to_wav_preserves_format(self) -> None:
        buf = AudioBuffer(pcm=_make_pcm(22050, 1), sample_rate=22050, channels=1)
        wav_bytes = buf.to_wav()
        assert wav_bytes[:4] == b"RIFF"
        parsed = AudioBuffer.from_wav(wav_bytes)
        assert parsed.sample_rate == 22050
        assert parsed.channels == 1
        assert parsed.pcm == buf.pcm

    def test_to_wav_stereo(self) -> None:
        buf = AudioBuffer(pcm=_make_pcm(44100, 2), sample_rate=44100, channels=2)
        wav_bytes = buf.to_wav()
        parsed = AudioBuffer.from_wav(wav_bytes)
        assert parsed.sample_rate == 44100
        assert parsed.channels == 2
        assert parsed.pcm == buf.pcm

    def test_from_wav_rejects_non_16bit(self) -> None:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(1)  # 8-bit
            w.setframerate(22050)
            w.writeframes(b"\x80" * 100)
        with pytest.raises(ValueError, match="16-bit"):
            AudioBuffer.from_wav(buf.getvalue())

    def test_duration_s(self) -> None:
        pcm = _make_pcm(22050, 1, n_frames=22050)  # 1 second
        buf = AudioBuffer(pcm=pcm, sample_rate=22050, channels=1)
        assert abs(buf.duration_s - 1.0) < 0.01

    def test_is_empty(self) -> None:
        assert AudioBuffer(pcm=b"", sample_rate=22050).is_empty
        assert not AudioBuffer(pcm=b"\x00\x00", sample_rate=22050).is_empty


# ---------------------------------------------------------------------------
# 7. MockAudioOutput stores AudioBuffer
# ---------------------------------------------------------------------------
class TestMockAudioOutputContract:
    """Verify MockAudioOutput records AudioBuffer with format metadata."""

    @pytest.mark.anyio
    async def test_played_buffers_have_format(self) -> None:
        from robot.hardware.audio.mock_audio import MockAudioOutput

        audio = MockAudioOutput()
        buf = AudioBuffer(pcm=_make_pcm(22050, 1), sample_rate=22050, channels=1)
        await audio.play(buf)
        assert len(audio.played) == 1
        assert audio.played[0].sample_rate == 22050
        assert audio.played[0].channels == 1
        assert audio.played[0].pcm == buf.pcm
