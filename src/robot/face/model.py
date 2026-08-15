"""The :class:`FaceModel` - a complete, immutable snapshot of the face.

The model is *the* contract between the :class:`EmotionEngine`,
:class:`FaceAnimator`, and :class:`FaceRenderer`. Every layer reads /
writes this single value type.

Two eyes (left + right) are explicitly separate so asymmetric
expressions (winks, side-glances) are first-class.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

from robot.face.components import (
    Accessory,
    Cheeks,
    Eye,
    Eyebrow,
    Eyelids,
    FacePalette,
    Mouth,
    Overlay,
)


# ---------------------------------------------------------------------------
# Body-language hints (proposed by the EmotionEngine, consumed by the
# BodyLanguageEngine). They are *not* servo angles - they are high-level
# intents.
# ---------------------------------------------------------------------------
class HeadTilt(Enum):
    """Discrete head-tilt hints the body-language engine understands."""

    NEUTRAL = "neutral"
    CURIOUS = "curious"  # slight tilt
    THINKING = "thinking"  # small side-to-side nod
    SLEEPY = "sleepy"  # droop
    SAD = "sad"  # downward
    EXCITED = "excited"  # quick tilt


class ArmPose(Enum):
    """Discrete arm-pose hints."""

    RELAXED = "relaxed"
    OPEN = "open"  # slightly raised
    WIDE = "wide"  # celebration
    WAVING = "waving"  # one arm up
    SHRUG = "shrug"  # both up
    DOWN = "down"  # rest
    POINT = "point"  # one arm forward


@dataclass(slots=True, frozen=True)
class BodyLanguageHint:
    """A high-level body-language request produced by the EmotionEngine."""

    head_tilt: HeadTilt = HeadTilt.NEUTRAL
    arm_pose: ArmPose = ArmPose.RELAXED
    # Intensity scales how strongly the body-language engine applies the
    # pose (0.0 = ignore, 1.0 = full expression).
    intensity: float = 1.0
    # Special transient gestures the body-language engine should run once
    # (e.g. a wave). Empty tuple = "no transient gestures".
    gestures: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# The face itself
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class FaceModel:
    """The full face, captured at one instant.

    All values are immutable. Use the helper methods to build new models
    with a single field changed (this is what the renderer animates).
    """

    width: int = 240
    height: int = 240

    left_eye: Eye = field(default_factory=Eye)
    right_eye: Eye = field(default_factory=Eye)

    # Eyelids
    eyelids: Eyelids = field(default_factory=Eyelids)

    # Eyebrows (one per eye)
    left_eyebrow: Eyebrow = field(default_factory=Eyebrow)
    right_eyebrow: Eyebrow = field(default_factory=Eyebrow)

    # Mouth
    mouth: Mouth = field(default_factory=Mouth)

    # Cheeks
    cheeks: Cheeks = field(default_factory=Cheeks)

    # Overlays
    overlay: Overlay = field(default_factory=Overlay)

    # Persistent accessories
    accessory: Accessory = field(default_factory=Accessory)

    # Colour palette (themes override this)
    palette: FacePalette = field(default_factory=FacePalette)

    # Subtle bounce / squash & stretch (in normalised units)
    bounce: float = 0.0  # -1..1 vertical translation
    squash: float = 1.0  # 0..2 horizontal scale (1 = neutral)

    # Body-language hint proposed by the EmotionEngine
    body_hint: BodyLanguageHint = field(default_factory=BodyLanguageHint)

    # ------------------------------------------------------------------
    # Convenience builders (return a new model with one field changed)
    # ------------------------------------------------------------------
    def with_eyes(
        self,
        left_eye: Eye | None = None,
        right_eye: Eye | None = None,
    ) -> FaceModel:
        return replace(
            self, left_eye=left_eye or self.left_eye, right_eye=right_eye or self.right_eye
        )

    def with_eyelids(self, eyelids: Eyelids) -> FaceModel:
        return replace(self, eyelids=eyelids)

    def with_eyebrows(
        self,
        left: Eyebrow | None = None,
        right: Eyebrow | None = None,
    ) -> FaceModel:
        return replace(
            self, left_eyebrow=left or self.left_eyebrow, right_eyebrow=right or self.right_eyebrow
        )

    def with_mouth(self, mouth: Mouth) -> FaceModel:
        return replace(self, mouth=mouth)

    def with_cheeks(self, cheeks: Cheeks) -> FaceModel:
        return replace(self, cheeks=cheeks)

    def with_overlay(self, overlay: Overlay) -> FaceModel:
        return replace(self, overlay=overlay)

    def with_accessory(self, accessory: Accessory) -> FaceModel:
        return replace(self, accessory=accessory)

    def with_palette(self, palette: FacePalette) -> FaceModel:
        return replace(self, palette=palette)

    def with_transform(self, bounce: float | None = None, squash: float | None = None) -> FaceModel:
        return replace(
            self,
            bounce=self.bounce if bounce is None else bounce,
            squash=self.squash if squash is None else squash,
        )

    def with_body_hint(self, body_hint: BodyLanguageHint) -> FaceModel:
        return replace(self, body_hint=body_hint)


__all__ = [
    "ArmPose",
    "BodyLanguageHint",
    "FaceModel",
    "HeadTilt",
]
