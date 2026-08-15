"""Display interface.

A :class:`Display` is a single rectangular (here: circular) framebuffer that
the eye engine renders into. Implementations live in
:mod:`robot.hardware.displays`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(slots=True, frozen=True)
class EyeFrame:
    """An RGB888 image to push to the display.

    Attributes
    ----------
    width, height:
        Pixel dimensions.
    pixels:
        Row-major RGB888 buffer of length ``width * height * 3``.
    """

    width: int
    height: int
    pixels: bytes


@runtime_checkable
class Display(Protocol):
    """A single display the eye engine can render to."""

    @property
    def width(self) -> int:
        """Width of the framebuffer in pixels."""

    @property
    def height(self) -> int:
        """Height of the framebuffer in pixels."""

    async def show(self, frame: EyeFrame) -> None:
        """Push a frame to the panel."""

    async def fill(self, color: tuple[int, int, int]) -> None:
        """Fill the entire panel with a single RGB color."""

    async def clear(self) -> None:
        """Reset the panel to black."""

    async def close(self) -> None:
        """Release hardware resources."""


__all__ = ["Display", "EyeFrame"]
