"""In-memory display implementation used for tests and headless dev."""

from __future__ import annotations

from robot.interfaces.display import EyeFrame
from robot.logging import get_logger

_log = get_logger("hardware.displays.mock")


class MockDisplay:
    """A :class:`Display` that stores every frame it receives in memory."""

    def __init__(self, width: int = 240, height: int = 240) -> None:
        self._width = width
        self._height = height
        self._last: EyeFrame | None = None
        self._frames_pushed: int = 0
        self._closed = False
        self._all_frames: list[EyeFrame] = []
        self._last_logged_size: tuple[int, int] | None = None

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def last_frame(self) -> EyeFrame | None:
        return self._last

    @property
    def frames_pushed(self) -> int:
        return self._frames_pushed

    @property
    def frames(self) -> list[EyeFrame]:
        """Every frame the display has received, in order."""
        return list(self._all_frames)

    async def show(self, frame: EyeFrame) -> None:
        if self._closed:
            raise RuntimeError("display is closed")
        self._last = frame
        self._frames_pushed += 1
        self._all_frames.append(frame)
        size = (frame.width, frame.height)
        if size != self._last_logged_size:
            _log.debug(
                "display.show",
                width=frame.width,
                height=frame.height,
            )
            self._last_logged_size = size

    async def fill(self, color: tuple[int, int, int]) -> None:
        r, g, b = color
        self._last = EyeFrame(
            self._width, self._height, bytes((r, g, b)) * (self._width * self._height)
        )
        self._all_frames.append(self._last)

    async def clear(self) -> None:
        await self.fill((0, 0, 0))

    async def close(self) -> None:
        self._closed = True


__all__ = ["MockDisplay"]
