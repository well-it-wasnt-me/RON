"""Fake display for tests."""

from __future__ import annotations

from robot.interfaces.display import EyeFrame


class FakeDisplay:
    def __init__(self, width: int = 240, height: int = 240) -> None:
        self._width = width
        self._height = height
        self.frames: list[EyeFrame] = []
        self.closed = False
        self.fills: list[tuple[int, int, int]] = []

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    async def show(self, frame: EyeFrame) -> None:
        self.frames.append(frame)

    async def fill(self, color: tuple[int, int, int]) -> None:
        self.fills.append(color)

    async def clear(self) -> None:
        await self.fill((0, 0, 0))

    async def close(self) -> None:
        self.closed = True
