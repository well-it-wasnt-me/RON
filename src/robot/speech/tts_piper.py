"""Piper TTS - local, offline text-to-speech.

Piper is a fast, local neural TTS engine that runs entirely on-device
(no cloud API required). It produces 16-bit PCM audio that can be
played directly through the robot's speaker.

Install with::

    uv pip install piper-tts

Usage::

    DESKBOT_TTS__PROVIDER = piper
    DESKBOT_TTS__PIPER__MODEL = en_US - lessac - medium

The synthesised audio is returned as an :class:`AudioBuffer` whose
sample rate matches the Piper model's native rate (typically 22050 Hz).
The output layer is responsible for resampling if the device expects a
different rate.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from robot.interfaces.audio import AudioBuffer, AudioOutput
from robot.logging import get_logger

if TYPE_CHECKING:
    from piper import PiperVoice, SynthesisConfig

_log = get_logger("speech.tts.piper")

_DEFAULT_MODEL_DIR = Path.home() / ".local" / "share" / "piper"


class PiperTTS:
    """Local text-to-speech using `Piper <https://github.com/rhasspy/piper>`_.

    The sample rate is read from the Piper model's output chunks
    (``chunk.sample_rate``) and propagated through an
    :class:`AudioBuffer`.
    """

    def __init__(
        self,
        model: str = "en_US-lessac-medium",
        download_dir: str | Path | None = None,
        use_cuda: bool = False,
        audio: AudioOutput | None = None,
        speaker_id: int | None = None,
        noise_scale: float | None = None,
        length_scale: float | None = None,
        noise_w_scale: float | None = None,
    ) -> None:
        self._model_name = model
        self._download_dir = Path(download_dir) if download_dir else _DEFAULT_MODEL_DIR
        self._use_cuda = use_cuda
        self._audio = audio
        self._speaker_id = speaker_id
        self._noise_scale = noise_scale
        self._length_scale = length_scale
        self._noise_w_scale = noise_w_scale
        self._voice: PiperVoice | None = None

    @property
    def name(self) -> str:
        return f"piper:{self._model_name}"

    def _get_voice(self) -> PiperVoice:
        """Lazy-load the Piper voice model on first use."""
        if self._voice is not None:
            return self._voice
        try:
            from piper import PiperVoice
        except ImportError as exc:
            raise ImportError(
                "piper-tts is required for PiperTTS. Install it with: uv pip install piper-tts"
            ) from exc

        model_path = self._download_dir / f"{self._model_name}.onnx"
        if not model_path.exists():
            _log.info("piper.downloading", model=self._model_name, dir=str(self._download_dir))
            self._download_dir.mkdir(parents=True, exist_ok=True)
            from piper.download_voices import download_voice

            download_voice(self._model_name, self._download_dir)

        _log.info("piper.loading", model=self._model_name, path=str(model_path))
        self._voice = PiperVoice.load(
            str(model_path),
            download_dir=self._download_dir,
            use_cuda=self._use_cuda,
        )
        return self._voice

    def _make_config(self) -> SynthesisConfig:
        from piper import SynthesisConfig

        return SynthesisConfig(
            speaker_id=self._speaker_id,
            noise_scale=self._noise_scale,
            length_scale=self._length_scale,
            noise_w_scale=self._noise_w_scale,
        )

    async def speak(self, text: str) -> AudioBuffer:
        """Synthesise *text* and return an :class:`AudioBuffer`.

        The sample rate is read from the Piper model's output chunks.
        """
        voice = self._get_voice()
        config = self._make_config()
        _log.info("piper.synthesize", text=text[:80], model=self._model_name)

        pcm_chunks: list[bytes] = []
        sample_rate: int | None = None
        for chunk in voice.synthesize(text, syn_config=config):
            if sample_rate is None:
                sample_rate = chunk.sample_rate
            pcm_chunks.append(chunk.audio_int16_bytes)

        if sample_rate is None:
            sample_rate = 22050

        pcm = b"".join(pcm_chunks)
        buffer = AudioBuffer(pcm=pcm, sample_rate=sample_rate, channels=1)
        _log.info(
            "piper.synthesized",
            pcm_len=len(pcm),
            sample_rate=sample_rate,
            text=text[:80],
        )

        return buffer

    async def close(self) -> None:
        """Release the Piper voice model."""
        self._voice = None
        _log.info("piper.closed")


__all__ = ["PiperTTS"]
