"""Camera interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(slots=True, frozen=True)
class Frame:
    """A captured image."""

    width: int
    height: int
    pixels: bytes  # RGB888
    timestamp: float


@runtime_checkable
class Camera(Protocol):
    """Camera interface returning RGB frames."""

    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...

    async def capture(self) -> Frame:
        """Capture a single frame."""

    async def close(self) -> None: ...
