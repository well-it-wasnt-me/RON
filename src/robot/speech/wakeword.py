"""Wake-word detection interface and mock implementation.

The :class:`WakeWordChecker` protocol is the *semantic* wake-word
boundary: an object that returns a :class:`WakeWordDetected` event only
when an actual wake phrase is recognised. Energy / volume / RMS must
NEVER implement this protocol -- see :class:`AudioActivityDetector` for
non-semantic audio activity (VAD) detection.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from robot.events.bus import InMemoryEventBus
from robot.events.events import WakeWordDetected
from robot.interfaces.microphone import Microphone
from robot.logging import get_logger
from robot.utils.clock import Clock

_log = get_logger("speech.wakeword")


# ---------------------------------------------------------------------------
# AudioActivityDetector - NON-semantic audio activity (VAD)
# ---------------------------------------------------------------------------


@runtime_checkable
class AudioActivityDetector(Protocol):
    """Non-semantic audio activity detector (VAD).

    Implementations report whether a chunk of audio contains activity
    (e.g. voice, noise). This is intentionally NOT a wake-word detector:
    it never produces a :class:`WakeWordDetected` event and must never,
    by itself, start a conversation. It may be used for diagnostics,
    gating, or VAD -- but a wake event must always come from a real
    :class:`WakeWordChecker`.
    """

    def is_active(self, pcm: bytes, timestamp: float) -> bool:
        """Return True if the chunk contains audio activity."""


# ---------------------------------------------------------------------------
# WakeWordChecker - semantic, chunk-by-chunk wake detection
# ---------------------------------------------------------------------------


@runtime_checkable
class WakeWordChecker(Protocol):
    """Check individual audio chunks for a wake-word trigger.

    This is the companion to :class:`WakeWordDetector`: the detector owns
    the microphone stream, while the checker is called chunk-by-chunk by
    the :class:`ConversationService` audio loop.

    Implementations must only return a :class:`WakeWordDetected` event
    when an actual wake phrase is recognised. They must NOT trigger on
    audio energy, volume, or amplitude alone.
    """

    def check(self, pcm: bytes, timestamp: float) -> WakeWordDetected | None:
        """Return a :class:`WakeWordDetected` event if *pcm* triggers the
        wake word, or ``None`` otherwise.

        Implementations must be stateful (e.g. cooldown tracking) and
        must update their internal state on each call.
        """


class NullWakeWordChecker:
    """Never triggers. Used when wake-word detection is handled
    externally (e.g. via bus events) or disabled entirely."""

    def check(self, pcm: bytes, timestamp: float) -> WakeWordDetected | None:
        return None


class MockWakeWordChecker:
    """Triggers after *trigger_after_chunks* chunks have been checked.

    Useful for integration tests where you want the audio loop to
    auto-trigger a wake word without real audio energy.

    This is a test/deterministic checker, NOT a production wake-word
    detector. It does not use audio energy.
    """

    def __init__(
        self,
        phrase: str = "hey deskbot",
        trigger_after_chunks: int = 5,
    ) -> None:
        self._phrase = phrase
        self._trigger_after = trigger_after_chunks
        self._count: int = 0

    def check(self, pcm: bytes, timestamp: float) -> WakeWordDetected | None:
        self._count += 1
        if self._count >= self._trigger_after:
            self._count = 0
            return WakeWordDetected(phrase=self._phrase, confidence=1.0)
        return None


# ---------------------------------------------------------------------------
# WakeWordDetector - async generator protocol (owns the mic stream)
# ---------------------------------------------------------------------------


@runtime_checkable
class WakeWordDetector(Protocol):
    async def listen(self) -> AsyncIterator[WakeWordDetected]:
        """Yield :class:`WakeWordDetected` events as they occur."""


class MockWakeWordDetector:
    """A no-op detector that emits a single wake-word event after a short delay."""

    def __init__(
        self,
        bus: InMemoryEventBus,
        microphone: Microphone,
        clock: Clock,
        phrase: str = "hey deskbot",
        delay_s: float = 5.0,
    ) -> None:
        self._bus = bus
        self._microphone = microphone
        self._clock = clock
        self._phrase = phrase
        self._delay_s = delay_s

    async def listen(self) -> AsyncIterator[WakeWordDetected]:
        _log.info("wakeword.listening", phrase=self._phrase)
        await self._clock.sleep(self._delay_s)
        event = WakeWordDetected(phrase=self._phrase, confidence=1.0)
        await self._bus.publish(event)
        yield event

    async def close(self) -> None:
        return None


__all__ = [
    "AudioActivityDetector",
    "MockWakeWordChecker",
    "MockWakeWordDetector",
    "NullWakeWordChecker",
    "WakeWordChecker",
    "WakeWordDetected",
    "WakeWordDetector",
]
