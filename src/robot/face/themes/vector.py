"""The "Vector" theme - Anki Vector 2.0 minimalist face.

Two glowing green squares for eyes + a thin horizontal line for the
mouth. All other face components (eyebrows, cheeks, eyelids, iris,
pupil, sclera) are disabled.

The theme sets :attr:`FacePalette.mode` to ``"vector"`` which the
:class:`FaceRenderer` recognises and switches to a minimalist
drawing path.
"""

from __future__ import annotations

from robot.face.components import (
    PALETTE_MODE_VECTOR,
    Cheeks,
    CheekState,
    Eye,
    Eyebrow,
    EyebrowShape,
    Eyelids,
    FacePalette,
    Gaze,
    Overlay,
    OverlayKind,
)
from robot.face.model import FaceModel

# The Vector 2.0 face is monochrome green on black. The palette below
# is the only place the colours live; users can subclass to change them.
VECTOR_PALETTE = FacePalette(
    background=(0, 0, 0),
    sclera=(0, 0, 0),  # unused in vector mode
    iris=(80, 220, 120),  # bright green - the Vector eye colour
    pupil=(255, 255, 255),  # tiny white catch-light (rendered inside the eye)
    eyelid=(0, 0, 0),  # unused in vector mode
    eyebrow=(0, 0, 0),  # unused in vector mode
    mouth=(80, 220, 120),  # mouth matches the eye colour
    cheek=(0, 0, 0),  # unused in vector mode
    blush=(0, 0, 0),  # unused in vector mode
    outline=(0, 0, 0),  # unused in vector mode
    highlight=(255, 255, 255),
    mode=PALETTE_MODE_VECTOR,
)


class VectorTheme:
    """The Anki Vector 2.0 minimalist face: two glowing dots + a line.

    The renderer detects ``palette.mode == "vector"`` and switches to
    a minimalist drawing path:

    * Two small filled rounded-squares for eyes.
    * A thin horizontal bar for the mouth (curve controls the shape:
      straight = neutral, up = smile, down = frown, gap = open).
    * No eyebrows, no cheeks, no iris, no pupil, no sclera.
    """

    name = "vector"
    palette = VECTOR_PALETTE

    def apply(self, model: FaceModel) -> FaceModel:
        """Rewrite the model to draw Vector-style eyes + line mouth.

        The gaze from the animated model is preserved so that drift
        and look commands produce visible eye movement in Vector mode.
        """
        eyes = (
            Eye(
                gaze=model.left_eye.gaze,
                openness=1.0,
                pupil_dilation=1.0,
                highlight=Gaze(0.0, 0.0),
                asymmetric=False,
            ),
            Eye(
                gaze=model.right_eye.gaze,
                openness=1.0,
                pupil_dilation=1.0,
                highlight=Gaze(0.0, 0.0),
                asymmetric=False,
            ),
        )
        eyelids = Eyelids(top=0.0, bottom=0.0)
        eyebrows = (
            Eyebrow(shape=EyebrowShape.NEUTRAL, raise_amount=0.0, angle=0.0),
            Eyebrow(shape=EyebrowShape.NEUTRAL, raise_amount=0.0, angle=0.0),
        )
        cheeks = Cheeks(state=CheekState.NONE, intensity=0.0)
        overlay = Overlay(kind=OverlayKind.NONE)
        return FaceModel(
            width=model.width,
            height=model.height,
            left_eye=eyes[0],
            right_eye=eyes[1],
            eyelids=eyelids,
            left_eyebrow=eyebrows[0],
            right_eyebrow=eyebrows[1],
            mouth=model.mouth,
            cheeks=cheeks,
            overlay=overlay,
            accessory=model.accessory,
            palette=self.palette,
            bounce=model.bounce,
            squash=model.squash,
            body_hint=model.body_hint,
        )


__all__ = ["VectorTheme"]
