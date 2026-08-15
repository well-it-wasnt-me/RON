"""Microphone interface."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(slots=True, frozen=True)
class AudioChunk:
    """A captured chunk of audio."""

    pcm: bytes
    sample_rate: int
    channels: int
    timestamp: float


@runtime_checkable
class Microphone(Protocol):
    """Async microphone capturing s16le mono PCM."""

    @property
    def sample_rate(self) -> int:
        """Sample rate in Hz."""

    def stream(self) -> AsyncIterator[AudioChunk]:
        """Return an async iterator of audio chunks until cancelled."""
        ...

    async def close(self) -> None:
        """Release the input device."""
