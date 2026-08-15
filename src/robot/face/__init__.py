"""The Face Engine - the robot's primary expressive system.

This package groups every piece of the new face-based architecture:

* :class:`FaceModel` and the component value types in
  :mod:`robot.face.components` - the immutable data the renderer draws.
* :class:`FaceRenderer` - display-agnostic pixel art.
* :class:`EmotionEngine` - pure function ``emotion_name -> FaceModel``,
  the single source of truth for "what does emotion X look like?".
* :class:`FaceAnimator` - the 30 FPS orchestrator that combines
  rendering, theming, and the legacy per-eye state machine.
* :mod:`robot.face.themes` - pluggable visual styles (Minimal, Cute,
  Pixel, Retro LCD, Wireframe).

The architecture diagram::

    Behavior Engine
            │
            ▼
      Emotion Engine
            │
            ▼
        Face Model  ────────────────┐
            │                       │
            ▼                       ▼
     Face Renderer           Body Language Engine
            │                       │
            ▼                       ▼
       Display Driver        Servo Controller
"""

from robot.face.animator import FaceAnimator
from robot.face.emotions import EMOTION_DEFS, EmotionEngine
from robot.face.model import (
    ArmPose,
    BodyLanguageHint,
    FaceModel,
    HeadTilt,
)
from robot.face.renderer import FaceRenderer
from robot.face.themes import (
    BUILTIN_THEMES,
    CuteTheme,
    MinimalTheme,
    PixelTheme,
    RetroLcdTheme,
    Theme,
    WireframeTheme,
    get_theme,
)

__all__ = [
    "BUILTIN_THEMES",
    "EMOTION_DEFS",
    "ArmPose",
    "BodyLanguageHint",
    "CuteTheme",
    "EmotionEngine",
    "FaceAnimator",
    "FaceModel",
    "FaceRenderer",
    "HeadTilt",
    "MinimalTheme",
    "PixelTheme",
    "RetroLcdTheme",
    "Theme",
    "WireframeTheme",
    "get_theme",
]
