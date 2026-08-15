"""Fake camera."""

from __future__ import annotations

from dataclasses import dataclass, field

from robot.interfaces.camera import Frame


@dataclass
class FakeCamera:
    width: int = 320
    height: int = 240
    frames: list[Frame] = field(default_factory=list)
    closed: bool = False

    async def capture(self) -> Frame:
        frame = Frame(width=self.width, height=self.height, pixels=b"", timestamp=0.0)
        self.frames.append(frame)
        return frame

    async def close(self) -> None:
        self.closed = True
