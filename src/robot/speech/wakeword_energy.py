"""Non-semantic audio activity (VAD) based on RMS energy.

This module deliberately does **not** provide a wake-word detector.
Earlier versions of DeskBot used an RMS energy threshold as a *semantic*
wake-word trigger (loud sound == ``"hey deskbot"``). That caused the
robot to wake on arbitrary environmental noise and, critically, on its
own TTS output picked up by the microphone -- producing repeated
``"hello deskbot"`` turns and runaway conversation history. That
behaviour has been removed.

What remains is :class:`EnergyActivityDetector`, a non-semantic audio
activity detector (a simple VAD). It reports whether a chunk of audio
contains activity above an energy threshold. It never produces a
:class:`WakeWordDetected` event and intentionally does NOT implement the
:class:`WakeWordChecker` protocol.

An energy threshold may still be useful for low-level audio activity /
VAD purposes, but it must **never by itself produce a wake-word event**.
A wake event must come from an actual wake-word detector
(e.g. :class:`~robot.speech.wakeword_openwakeword.OpenWakeWordChecker`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from robot.hardware.sensors.usb_microphone import rms
from robot.logging import get_logger

_log = get_logger("speech.audio_activity.energy")


@dataclass(slots=True)
class EnergyActivityDetector:
    """Non-semantic audio activity detector (VAD) based on RMS energy.

    This is NOT a wake-word detector and intentionally does not
    implement the :class:`WakeWordChecker` protocol.
    :meth:`is_active` returns a boolean indicating whether the chunk's
    RMS energy exceeds :attr:`threshold`, subject to a warmup period and
    cooldown. It never returns a :class:`WakeWordDetected` event.

    Parameters
    ----------
    threshold:
        RMS threshold (0..1). 0.05 catches normal speech; 0.15 ignores
        quiet background noise.
    cooldown_s:
        Minimum seconds between two "active" reports. Prevents the same
        sound from being reported as active many times in a row.
    warmup_chunks:
        Number of initial chunks to skip before checking energy. Avoids
        false activity from startup transients.
    """

    threshold: float = 0.05
    cooldown_s: float = 1.5
    warmup_chunks: int = 5
    _last_active_s: float = field(default=float("-inf"), init=False)
    _chunk_count: int = field(default=0, init=False)

    def is_active(self, pcm: bytes, timestamp: float) -> bool:
        """Return True if the chunk's energy exceeds the threshold.

        Updates internal state (cooldown, chunk count) on every call.
        """
        self._chunk_count += 1
        if self._chunk_count <= self.warmup_chunks:
            return False
        energy = rms(pcm)
        if energy < self.threshold:
            return False
        if timestamp - self._last_active_s < self.cooldown_s:
            return False
        self._last_active_s = timestamp
        _log.debug(
            "audio_activity.energy_active",
            energy=energy,
            chunk=self._chunk_count,
        )
        return True

    def reset(self) -> None:
        """Reset internal state (chunk counter and cooldown)."""
        self._chunk_count = 0
        self._last_active_s = 0.0


__all__ = ["EnergyActivityDetector"]
