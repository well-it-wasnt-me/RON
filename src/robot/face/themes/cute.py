"""The "Cute" theme - softer colours, blush by default, pink cheeks."""

from __future__ import annotations

from robot.face.components import Cheeks, CheekState, FacePalette
from robot.face.model import FaceModel


class CuteTheme:
    name = "cute"
    palette = FacePalette(
        background=(255, 240, 245),
        sclera=(255, 255, 255),
        iris=(180, 100, 180),
        pupil=(40, 20, 50),
        eyelid=(230, 200, 220),
        eyebrow=(120, 70, 120),
        mouth=(255, 90, 130),
        cheek=(255, 160, 200),
        blush=(255, 120, 160),
        outline=(80, 50, 80),
        highlight=(255, 255, 255),
    )

    def apply(self, model: FaceModel) -> FaceModel:
        # Always have soft cheeks for the "cute" look
        cheeks = Cheeks(state=CheekState.SOFT, intensity=0.5)
        return model.with_palette(self.palette).with_cheeks(cheeks)


__all__ = ["CuteTheme"]
