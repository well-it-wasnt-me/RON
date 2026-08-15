"""The default "Minimal" theme.

Clean, modern, minimal: white sclera, blue iris, red mouth, no extras.
"""

from __future__ import annotations

from robot.face.components import FacePalette
from robot.face.model import FaceModel


class MinimalTheme:
    """The default minimal theme."""

    name = "minimal"
    palette = FacePalette(
        background=(10, 10, 20),
        sclera=(245, 245, 235),
        iris=(40, 110, 200),
        pupil=(10, 10, 20),
        eyelid=(20, 20, 30),
        eyebrow=(40, 30, 30),
        mouth=(180, 60, 80),
        cheek=(255, 130, 130),
        blush=(255, 90, 90),
        outline=(10, 10, 10),
        highlight=(255, 255, 255),
    )

    def apply(self, model: FaceModel) -> FaceModel:
        return model.with_palette(self.palette)


__all__ = ["MinimalTheme"]
