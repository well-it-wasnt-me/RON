"""Reusable face components.

A face is a *composition* of small, independent components. Each component
is an immutable value type so the :class:`FaceModel` itself is also
immutable - that makes the renderer deterministic and unit-testable.

The components live at three levels:

* **eyes** (left/right) - gaze vector, openness, pupil dilation, etc.
* **face features** - eyebrows, mouth, cheeks, all parameterised.
* **overlays** - sparkles, hearts, tears, sweat drop, blush, accessories.
  They are off-screen by default and turned on by the
  :class:`EmotionEngine` when an emotion needs them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class Point:
    """A 2D point in the face's local frame, in pixels."""

    x: float
    y: float


@dataclass(slots=True, frozen=True)
class Gaze:
    """2D gaze target normalised in ``[-1, 1]``."""

    x: float = 0.0
    y: float = 0.0

    def clamped(self) -> Gaze:
        return Gaze(
            x=max(-1.0, min(1.0, self.x)),
            y=max(-1.0, min(1.0, self.y)),
        )


# ---------------------------------------------------------------------------
# Eye
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class Eye:
    """A single eye. The face has two of them."""

    gaze: Gaze = field(default_factory=Gaze)
    openness: float = 1.0  # 0.0 = closed, 1.0 = wide
    pupil_dilation: float = 0.5  # 0.0 = constricted, 1.0 = dilated
    highlight: Gaze = field(default_factory=lambda: Gaze(0.3, 0.3))
    asymmetric: bool = False  # for character expressions


# ---------------------------------------------------------------------------
# Eyelids
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class Eyelids:
    """Vertical eyelid positions, ``0.0`` = retracted, ``1.0`` = fully closed."""

    top: float = 0.0
    bottom: float = 0.0


# ---------------------------------------------------------------------------
# Eyebrow
# ---------------------------------------------------------------------------
class EyebrowShape(str, Enum):
    NEUTRAL = "neutral"
    RAISED = "raised"
    ANGRY = "angry"
    WORRIED = "worried"
    SLEEPY = "sleepy"
    SAD = "sad"


@dataclass(slots=True, frozen=True)
class Eyebrow:
    """A single eyebrow."""

    shape: EyebrowShape = EyebrowShape.NEUTRAL
    raise_amount: float = 0.0  # 0..1 vertical raise (additive on top of shape)
    angle: float = 0.0  # -1..1 tilt (angry < 0, worried > 0)


# ---------------------------------------------------------------------------
# Mouth
# ---------------------------------------------------------------------------
class MouthShape(str, Enum):
    CLOSED = "closed"
    NEUTRAL = "neutral"
    SMILE = "smile"
    GRIN = "grin"
    FROWN = "frown"
    OPEN = "open"  # small "oh"
    WIDE_OPEN = "wide_open"  # gasp / surprise
    TONGUE = "tongue"  # playful
    SMILE_OPEN = "smile_open"  # laughing


@dataclass(slots=True, frozen=True)
class Mouth:
    """The mouth, parameterised."""

    shape: MouthShape = MouthShape.NEUTRAL
    openness: float = 0.0  # 0.0 = closed, 1.0 = wide open
    width: float = 0.5  # 0.0 = small, 1.0 = wide
    asymmetry: float = 0.0  # -1..1 (left side higher = smirk)


# ---------------------------------------------------------------------------
# Cheeks, blush
# ---------------------------------------------------------------------------
class CheekState(str, Enum):
    NONE = "none"
    SOFT = "soft"  # neutral happy
    BRIGHT = "bright"  # very happy / embarrassed
    COLD = "cold"  # sad / cold


@dataclass(slots=True, frozen=True)
class Cheeks:
    """Cheek rendering - base + accent colour, intensity."""

    state: CheekState = CheekState.NONE
    intensity: float = 0.0  # 0..1


# ---------------------------------------------------------------------------
# Overlays
# ---------------------------------------------------------------------------
class OverlayKind(str, Enum):
    NONE = "none"
    SPARKLE = "sparkle"
    HEART = "heart"
    TEAR = "tear"
    SWEAT = "sweat"
    ANGER = "anger"  # vein symbol
    QUESTION = "question"  # "?" mark
    EXCLAIM = "exclaim"  # "!" mark
    MUSIC = "music"  # note
    STAR = "star"


@dataclass(slots=True, frozen=True)
class Overlay:
    """A floating symbol rendered on top of the face."""

    kind: OverlayKind = OverlayKind.NONE
    position: Point = field(default_factory=lambda: Point(0.0, -0.5))
    size: float = 0.3  # 0..1 fraction of the face radius
    rotation: float = 0.0  # radians
    color: tuple[int, int, int] = (255, 255, 255)


# ---------------------------------------------------------------------------
# Accessories
# ---------------------------------------------------------------------------
class AccessoryKind(str, Enum):
    NONE = "none"
    GLASSES = "glasses"
    SUNGLASSES = "sunglasses"
    MUSTACHE = "mustache"
    BOW_TIE = "bow_tie"
    HAT = "hat"
    CROWN = "crown"
    BANDAGE = "bandage"
    HEART_EYES = "heart_eyes"  # per-eye decoration (overrides pupil)


@dataclass(slots=True, frozen=True)
class Accessory:
    """A persistent decoration (vs. an :class:`Overlay` which is transient)."""

    kind: AccessoryKind = AccessoryKind.NONE
    color: tuple[int, int, int] = (255, 255, 255)
    opacity: float = 1.0  # 0..1


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
# Recognised palette rendering modes. Themes set ``FacePalette.mode``
# to switch the renderer between styles. ``"face"`` is the default
# (eyes + iris + pupil + sclera + mouth etc.). ``"vector"`` is the
# Anki Vector 2.0 minimalist mode (two glowing dots + a line).
PALETTE_MODE_FACE = "face"
PALETTE_MODE_VECTOR = "vector"


@dataclass(slots=True, frozen=True)
class FacePalette:
    """All colours the renderer can use. Themes override these.

    ``mode`` selects a renderer rendering style. The default ``"face"``
    draws the full eye stack (sclera + iris + pupil + highlight +
    eyelids + eyebrows + mouth). The ``"vector"`` mode draws two
    filled rounded-squares for eyes and a thin horizontal bar for
    the mouth - the Anki Vector 2.0 minimalist look.
    """

    background: tuple[int, int, int] = (10, 10, 20)
    sclera: tuple[int, int, int] = (245, 245, 235)
    iris: tuple[int, int, int] = (40, 110, 200)
    pupil: tuple[int, int, int] = (10, 10, 20)
    eyelid: tuple[int, int, int] = (20, 20, 30)
    eyebrow: tuple[int, int, int] = (40, 30, 30)
    mouth: tuple[int, int, int] = (180, 60, 80)
    cheek: tuple[int, int, int] = (255, 130, 130)
    blush: tuple[int, int, int] = (255, 90, 90)
    outline: tuple[int, int, int] = (10, 10, 10)
    highlight: tuple[int, int, int] = (255, 255, 255)
    mode: str = PALETTE_MODE_FACE


__all__ = [
    "PALETTE_MODE_FACE",
    "PALETTE_MODE_VECTOR",
    "Accessory",
    "AccessoryKind",
    "CheekState",
    "Cheeks",
    "Eye",
    "Eyebrow",
    "EyebrowShape",
    "Eyelids",
    "FacePalette",
    "Gaze",
    "Mouth",
    "MouthShape",
    "Overlay",
    "OverlayKind",
    "Point",
]
