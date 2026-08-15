"""Tests for the SoundEffectsPlayer."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from tests.fakes.audio import FakeAudioOutput

from robot.speech.sound_effects import SoundEffectsPlayer, _wav_to_pcm


def _make_wav(sample_rate: int = 22050, channels: int = 1, duration_s: float = 0.1) -> bytes:
    """Generate a minimal valid WAV file in memory."""
    import io
    import struct
    import wave

    n_frames = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        frames = struct.pack(f"<{n_frames * channels}h", *([16384] * n_frames * channels))
        w.writeframes(frames)
    return buf.getvalue()


class TestWavToPcm:
    def test_mono_wav(self) -> None:
        wav = _make_wav(sample_rate=22050, channels=1, duration_s=0.05)
        pcm, sr = _wav_to_pcm(wav)
        assert sr == 22050
        assert len(pcm) > 0

    def test_stereo_wav_to_mono(self) -> None:
        wav = _make_wav(sample_rate=22050, channels=2, duration_s=0.05)
        pcm, sr = _wav_to_pcm(wav)
        assert sr == 22050
        # Stereo to mono should halve the data size.
        assert len(pcm) > 0

    def test_empty_wav(self) -> None:
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(22050)
            w.writeframes(b"")
        wav = buf.getvalue()
        pcm, _sr = _wav_to_pcm(wav)
        assert len(pcm) == 0

    def test_24bit_wav(self) -> None:
        """24-bit WAVs are converted to signed 16-bit PCM."""
        import struct
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(3)
            w.setframerate(22050)
            # 24-bit little-endian PCM sample.
            sample = (16384 << 8).to_bytes(3, "little", signed=False)
            w.writeframes(sample * 10)

        pcm, sr = _wav_to_pcm(buf.getvalue())

        assert sr == 22050
        assert len(pcm) == 20
        assert struct.unpack("<h", pcm[:2])[0] == 16384


class TestSoundEffectsPlayer:
    @pytest.mark.anyio
    async def test_disabled_player_returns_false(self) -> None:
        audio = FakeAudioOutput()
        player = SoundEffectsPlayer(audio=audio, enabled=False)
        assert await player.play("angry") is False

    @pytest.mark.anyio
    async def test_nonexistent_sound_returns_false(self, tmp_path: Path) -> None:
        audio = FakeAudioOutput()
        player = SoundEffectsPlayer(sounds_dir=tmp_path, audio=audio, enabled=True)
        assert await player.play("nonexistent") is False

    @pytest.mark.anyio
    async def test_play_sound_from_dir(self, tmp_path: Path) -> None:
        """Play a WAV file from the sounds directory."""
        audio = FakeAudioOutput()
        wav = _make_wav()
        (tmp_path / "test-sound.wav").write_bytes(wav)
        player = SoundEffectsPlayer(sounds_dir=tmp_path, audio=audio, enabled=True)
        result = await player.play("test-sound")
        assert result is True
        assert len(audio.played) == 1

    @pytest.mark.anyio
    async def test_random_variation(self, tmp_path: Path) -> None:
        """Multiple files with the same base name get random selection."""
        audio = FakeAudioOutput()
        wav = _make_wav()
        # openWakeWord-style naming: 826363__charonfaustinus__talk-1.wav
        (tmp_path / "826363__charonfaustinus__talk-1.wav").write_bytes(wav)
        (tmp_path / "826368__charonfaustinus__talk-2.wav").write_bytes(wav)
        player = SoundEffectsPlayer(sounds_dir=tmp_path, audio=audio, enabled=True)
        assert "talk" in player.list_sounds()
        result = await player.play("talk")
        assert result is True

    def test_list_sounds_empty(self, tmp_path: Path) -> None:
        player = SoundEffectsPlayer(sounds_dir=tmp_path, audio=None, enabled=True)
        assert player.list_sounds() == []

    def test_has_sound(self, tmp_path: Path) -> None:
        wav = _make_wav()
        (tmp_path / "826363__charonfaustinus__angry-1.wav").write_bytes(wav)
        player = SoundEffectsPlayer(sounds_dir=tmp_path, audio=None, enabled=True)
        assert player.has_sound("angry") is True
        assert player.has_sound("nonexistent") is False
