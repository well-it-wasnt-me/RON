"""The "Retro LCD" theme - green-on-black palette."""

from __future__ import annotations

from robot.face.components import FacePalette
from robot.face.model import FaceModel


class RetroLcdTheme:
    name = "retro_lcd"
    palette = FacePalette(
        background=(0, 0, 0),
        sclera=(40, 180, 60),
        iris=(100, 255, 130),
        pupil=(0, 0, 0),
        eyelid=(20, 60, 30),
        eyebrow=(40, 180, 60),
        mouth=(40, 180, 60),
        cheek=(40, 180, 60),
        blush=(40, 180, 60),
        outline=(0, 0, 0),
        highlight=(180, 255, 200),
    )

    def apply(self, model: FaceModel) -> FaceModel:
        return model.with_palette(self.palette)


__all__ = ["RetroLcdTheme"]
