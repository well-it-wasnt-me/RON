"""Mock camera returning a single solid-colour frame."""

from __future__ import annotations

from robot.interfaces.camera import Frame


class MockCamera:
    """Returns the same frame every time :meth:`capture` is called."""

    def __init__(
        self, width: int = 320, height: int = 240, color: tuple[int, int, int] = (50, 50, 50)
    ) -> None:
        self._width = width
        self._height = height
        self._color = color
        self._closed = False
        self._captured = 0

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def captured(self) -> int:
        return self._captured

    async def capture(self) -> Frame:
        if self._closed:
            raise RuntimeError("camera is closed")
        self._captured += 1
        r, g, b = self._color
        pixels = bytes((r, g, b)) * (self._width * self._height)
        return Frame(width=self._width, height=self._height, pixels=pixels, timestamp=0.0)

    async def close(self) -> None:
        self._closed = True


__all__ = ["MockCamera"]
