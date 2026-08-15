"""The "Pixel" theme - quantises coordinates to a coarse grid for a chunky look."""

from __future__ import annotations

from robot.face.components import FacePalette
from robot.face.model import FaceModel


def _quantise(value: float, step: float) -> float:
    return round(value / step) * step


class PixelTheme:
    name = "pixel"
    palette = FacePalette(
        background=(15, 15, 25),
        sclera=(255, 255, 240),
        iris=(70, 140, 220),
        pupil=(15, 15, 25),
        eyelid=(30, 30, 45),
        eyebrow=(60, 40, 40),
        mouth=(220, 70, 90),
        cheek=(255, 140, 140),
        blush=(255, 90, 90),
        outline=(0, 0, 0),
        highlight=(255, 255, 255),
    )

    def __init__(self, grid: int = 8) -> None:
        self._grid = max(1, grid)

    def apply(self, model: FaceModel) -> FaceModel:
        # Quantise gaze so the iris lands on a pixel grid.
        from robot.face.components import Eye, Gaze

        def _snap_eye(eye: Eye) -> Eye:
            return Eye(
                gaze=Gaze(
                    x=_quantise(eye.gaze.x, 1.0 / self._grid),
                    y=_quantise(eye.gaze.y, 1.0 / self._grid),
                ),
                openness=_quantise(eye.openness, 1.0 / self._grid),
                pupil_dilation=_quantise(eye.pupil_dilation, 1.0 / self._grid),
                highlight=eye.highlight,
                asymmetric=eye.asymmetric,
            )

        return model.with_palette(self.palette).with_eyes(
            left_eye=_snap_eye(model.left_eye),
            right_eye=_snap_eye(model.right_eye),
        )


__all__ = ["PixelTheme"]
