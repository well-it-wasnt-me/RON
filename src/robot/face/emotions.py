"""The :class:`EmotionEngine` - produces a :class:`FaceModel` from an emotion.

The engine is a **pure function** ``EmotionName -> (FaceModel, BodyLanguageHint)``.
It never touches rendering, servos, displays, or animation state. This
is the single source of truth for "what does emotion X look like?".

Emotions are built from a small set of reusable **face component
presets** (eye defaults, mouth shape, eyebrow shape, …) so that adding
a new emotion is a one-line change.
"""

from __future__ import annotations

from dataclasses import dataclass

from robot.face.components import (
    Cheeks,
    CheekState,
    Eye,
    Eyebrow,
    EyebrowShape,
    Eyelids,
    FacePalette,
    Gaze,
    Mouth,
    MouthShape,
    Overlay,
    OverlayKind,
    Point,
)
from robot.face.model import (
    ArmPose,
    BodyLanguageHint,
    FaceModel,
    HeadTilt,
)


# ---------------------------------------------------------------------------
# Emotion definitions
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class _EmotionDef:
    """A complete emotional expression: face presets + body hint."""

    eyes: tuple[Eye, Eye]  # left, right
    eyelids: Eyelids
    eyebrows: tuple[Eyebrow, Eyebrow]  # left, right
    mouth: Mouth
    cheeks: Cheeks
    overlay: Overlay
    body_hint: BodyLanguageHint


# Reusable component presets
NEUTRAL_EYES = (Eye(gaze=Gaze(0.0, 0.0)), Eye(gaze=Gaze(0.0, 0.0)))
NEUTRAL_EYEBROWS = (Eyebrow(shape=EyebrowShape.NEUTRAL), Eyebrow(shape=EyebrowShape.NEUTRAL))
NEUTRAL_MOUTH = Mouth(shape=MouthShape.NEUTRAL, openness=0.0, width=0.5, asymmetry=0.0)
NEUTRAL_CHEEKS = Cheeks(state=CheekState.NONE, intensity=0.0)
NEUTRAL_OVERLAY = Overlay(kind=OverlayKind.NONE)
NEUTRAL_HINT = BodyLanguageHint(head_tilt=HeadTilt.NEUTRAL, arm_pose=ArmPose.RELAXED, intensity=0.5)
NEUTRAL_EYELIDS = Eyelids(top=0.0, bottom=0.0)


# ---------------------------------------------------------------------------
# Catalogue of emotions
# ---------------------------------------------------------------------------
EMOTION_DEFS: dict[str, _EmotionDef] = {
    "neutral": _EmotionDef(
        eyes=NEUTRAL_EYES,
        eyelids=NEUTRAL_EYELIDS,
        eyebrows=NEUTRAL_EYEBROWS,
        mouth=NEUTRAL_MOUTH,
        cheeks=NEUTRAL_CHEEKS,
        overlay=NEUTRAL_OVERLAY,
        body_hint=NEUTRAL_HINT,
    ),
    "happy": _EmotionDef(
        eyes=(Eye(gaze=Gaze(0.0, 0.05), openness=0.85), Eye(gaze=Gaze(0.0, 0.05), openness=0.85)),
        eyelids=Eyelids(top=0.05, bottom=0.25),
        eyebrows=(
            Eyebrow(shape=EyebrowShape.RAISED, raise_amount=0.3),
            Eyebrow(shape=EyebrowShape.RAISED, raise_amount=0.3),
        ),
        mouth=Mouth(shape=MouthShape.SMILE, openness=0.3, width=0.6),
        cheeks=Cheeks(state=CheekState.SOFT, intensity=0.6),
        overlay=NEUTRAL_OVERLAY,
        body_hint=BodyLanguageHint(
            head_tilt=HeadTilt.CURIOUS, arm_pose=ArmPose.OPEN, intensity=0.7
        ),
    ),
    "curious": _EmotionDef(
        eyes=(
            Eye(gaze=Gaze(0.20, 0.10), openness=1.0, highlight=Gaze(0.4, 0.4)),
            Eye(gaze=Gaze(0.20, 0.10), openness=1.0, highlight=Gaze(0.4, 0.4)),
        ),
        eyelids=NEUTRAL_EYELIDS,
        eyebrows=(
            Eyebrow(shape=EyebrowShape.RAISED, raise_amount=0.6),
            Eyebrow(shape=EyebrowShape.RAISED, raise_amount=0.6),
        ),
        mouth=Mouth(shape=MouthShape.OPEN, openness=0.2, width=0.4),
        cheeks=NEUTRAL_CHEEKS,
        overlay=Overlay(kind=OverlayKind.QUESTION, position=Point(0.7, -0.5)),
        body_hint=BodyLanguageHint(
            head_tilt=HeadTilt.CURIOUS, arm_pose=ArmPose.POINT, intensity=0.6
        ),
    ),
    "thinking": _EmotionDef(
        eyes=(
            Eye(gaze=Gaze(0.20, 0.30), openness=0.85),
            Eye(gaze=Gaze(0.20, 0.30), openness=0.85),
        ),
        eyelids=NEUTRAL_EYELIDS,
        eyebrows=(
            Eyebrow(shape=EyebrowShape.WORRIED, raise_amount=0.2, angle=0.2),
            Eyebrow(shape=EyebrowShape.WORRIED, raise_amount=0.2, angle=-0.2),
        ),
        mouth=Mouth(shape=MouthShape.NEUTRAL, openness=0.0, width=0.4),
        cheeks=NEUTRAL_CHEEKS,
        overlay=NEUTRAL_OVERLAY,
        body_hint=BodyLanguageHint(
            head_tilt=HeadTilt.THINKING, arm_pose=ArmPose.RELAXED, intensity=0.5
        ),
    ),
    "sleepy": _EmotionDef(
        eyes=(Eye(gaze=Gaze(0.0, 0.0), openness=0.15), Eye(gaze=Gaze(0.0, 0.0), openness=0.15)),
        eyelids=Eyelids(top=0.65, bottom=0.0),
        eyebrows=(
            Eyebrow(shape=EyebrowShape.SLEEPY),
            Eyebrow(shape=EyebrowShape.SLEEPY),
        ),
        mouth=Mouth(shape=MouthShape.OPEN, openness=0.1, width=0.4),
        cheeks=NEUTRAL_CHEEKS,
        overlay=NEUTRAL_OVERLAY,
        body_hint=BodyLanguageHint(head_tilt=HeadTilt.SLEEPY, arm_pose=ArmPose.DOWN, intensity=0.7),
    ),
    "embarrassed": _EmotionDef(
        eyes=(
            Eye(gaze=Gaze(0.0, -0.30), openness=0.7, highlight=Gaze(-0.3, -0.3)),
            Eye(gaze=Gaze(0.0, -0.30), openness=0.7, highlight=Gaze(-0.3, -0.3)),
        ),
        eyelids=Eyelids(top=0.10, bottom=0.05),
        eyebrows=(
            Eyebrow(shape=EyebrowShape.WORRIED, angle=0.3),
            Eyebrow(shape=EyebrowShape.WORRIED, angle=-0.3),
        ),
        mouth=Mouth(shape=MouthShape.SMILE, openness=0.2, width=0.5, asymmetry=0.3),
        cheeks=Cheeks(state=CheekState.BRIGHT, intensity=0.9),
        overlay=NEUTRAL_OVERLAY,
        body_hint=BodyLanguageHint(head_tilt=HeadTilt.SAD, arm_pose=ArmPose.SHRUG, intensity=0.5),
    ),
    "excited": _EmotionDef(
        eyes=(
            Eye(gaze=Gaze(0.0, -0.1), openness=1.2, highlight=Gaze(0.0, 0.0)),
            Eye(gaze=Gaze(0.0, -0.1), openness=1.2, highlight=Gaze(0.0, 0.0)),
        ),
        eyelids=NEUTRAL_EYELIDS,
        eyebrows=(
            Eyebrow(shape=EyebrowShape.RAISED, raise_amount=0.8),
            Eyebrow(shape=EyebrowShape.RAISED, raise_amount=0.8),
        ),
        mouth=Mouth(shape=MouthShape.SMILE_OPEN, openness=0.6, width=0.7),
        cheeks=Cheeks(state=CheekState.SOFT, intensity=0.4),
        overlay=Overlay(kind=OverlayKind.SPARKLE, position=Point(-0.5, -0.4), size=0.3),
        body_hint=BodyLanguageHint(
            head_tilt=HeadTilt.EXCITED,
            arm_pose=ArmPose.WIDE,
            intensity=1.0,
            gestures=("bounce",),
        ),
    ),
    "sad": _EmotionDef(
        eyes=(
            Eye(gaze=Gaze(0.0, -0.30), openness=0.65, pupil_dilation=0.7),
            Eye(gaze=Gaze(0.0, -0.30), openness=0.65, pupil_dilation=0.7),
        ),
        eyelids=Eyelids(top=0.35, bottom=0.0),
        eyebrows=(
            Eyebrow(shape=EyebrowShape.SAD, angle=0.4),
            Eyebrow(shape=EyebrowShape.SAD, angle=-0.4),
        ),
        mouth=Mouth(shape=MouthShape.FROWN, openness=0.0, width=0.5),
        cheeks=Cheeks(state=CheekState.COLD, intensity=0.4),
        overlay=Overlay(kind=OverlayKind.TEAR, position=Point(-0.3, 0.1), size=0.2),
        body_hint=BodyLanguageHint(head_tilt=HeadTilt.SAD, arm_pose=ArmPose.DOWN, intensity=0.7),
    ),
    "surprised": _EmotionDef(
        eyes=(
            Eye(gaze=Gaze(0.0, 0.0), openness=1.0, pupil_dilation=0.2),
            Eye(gaze=Gaze(0.0, 0.0), openness=1.0, pupil_dilation=0.2),
        ),
        eyelids=NEUTRAL_EYELIDS,
        eyebrows=(
            Eyebrow(shape=EyebrowShape.RAISED, raise_amount=1.0),
            Eyebrow(shape=EyebrowShape.RAISED, raise_amount=1.0),
        ),
        mouth=Mouth(shape=MouthShape.WIDE_OPEN, openness=0.9, width=0.6),
        cheeks=NEUTRAL_CHEEKS,
        overlay=Overlay(kind=OverlayKind.EXCLAIM, position=Point(0.6, -0.5), size=0.3),
        body_hint=BodyLanguageHint(
            head_tilt=HeadTilt.EXCITED,
            arm_pose=ArmPose.SHRUG,
            intensity=0.8,
            gestures=("bounce",),
        ),
    ),
    "angry": _EmotionDef(
        eyes=(
            Eye(gaze=Gaze(0.0, -0.10), openness=0.7, pupil_dilation=0.2),
            Eye(gaze=Gaze(0.0, -0.10), openness=0.7, pupil_dilation=0.2),
        ),
        eyelids=Eyelids(top=0.30, bottom=0.0),
        eyebrows=(
            Eyebrow(shape=EyebrowShape.ANGRY, angle=0.7),
            Eyebrow(shape=EyebrowShape.ANGRY, angle=-0.7),
        ),
        mouth=Mouth(shape=MouthShape.FROWN, openness=0.2, width=0.5),
        cheeks=Cheeks(state=CheekState.BRIGHT, intensity=0.5),
        overlay=Overlay(kind=OverlayKind.ANGER, position=Point(0.5, -0.5), size=0.3),
        body_hint=BodyLanguageHint(
            head_tilt=HeadTilt.NEUTRAL, arm_pose=ArmPose.SHRUG, intensity=0.7
        ),
    ),
}


# ---------------------------------------------------------------------------
# EmotionEngine
# ---------------------------------------------------------------------------
class EmotionEngine:
    """Produces :class:`FaceModel` + :class:`BodyLanguageHint` from an emotion."""

    def __init__(
        self, palette: FacePalette | None = None, width: int = 240, height: int = 240
    ) -> None:
        self._palette = palette or FacePalette()
        self._width = width
        self._height = height

    def set_dimensions(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

    def available(self) -> list[str]:
        return list(EMOTION_DEFS.keys())

    def get(self, name: str) -> _EmotionDef:
        try:
            return EMOTION_DEFS[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown emotion {name!r}; available: {sorted(EMOTION_DEFS.keys())}"
            ) from exc

    def build(self, name: str) -> FaceModel:
        """Build the :class:`FaceModel` for ``name`` with default geometry."""
        d = self.get(name)
        left_eye, right_eye = d.eyes
        left_brow, right_brow = d.eyebrows
        return FaceModel(
            width=self._width,
            height=self._height,
            left_eye=left_eye,
            right_eye=right_eye,
            eyelids=d.eyelids,
            left_eyebrow=left_brow,
            right_eyebrow=right_brow,
            mouth=d.mouth,
            cheeks=d.cheeks,
            overlay=d.overlay,
            palette=self._palette,
            body_hint=d.body_hint,
        )

    def register(self, name: str, definition: _EmotionDef) -> None:
        """Register a new emotion or override an existing one."""
        EMOTION_DEFS[name] = definition


__all__ = [
    "EMOTION_DEFS",
    "EmotionEngine",
]
