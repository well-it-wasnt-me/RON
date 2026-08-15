"""The "Wireframe" theme - outlines only, no fills."""

from __future__ import annotations

from robot.face.components import FacePalette
from robot.face.model import FaceModel


class WireframeTheme:
    name = "wireframe"
    palette = FacePalette(
        background=(0, 0, 0),
        sclera=(0, 0, 0),
        iris=(0, 0, 0),
        pupil=(255, 255, 255),
        eyelid=(0, 0, 0),
        eyebrow=(255, 255, 255),
        mouth=(255, 255, 255),
        cheek=(0, 0, 0),
        blush=(0, 0, 0),
        outline=(255, 255, 255),
        highlight=(255, 255, 255),
    )

    def apply(self, model: FaceModel) -> FaceModel:
        return model.with_palette(self.palette)


__all__ = ["WireframeTheme"]
