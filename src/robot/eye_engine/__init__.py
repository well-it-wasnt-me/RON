"""Display-agnostic eye rendering engine.

The engine is split into four layers:

* :mod:`robot.eye_engine.renderer` - pixel-level drawing (sclera, iris,
  pupil, eyelids, highlight).
* :mod:`robot.eye_engine.animation` - per-eye state machine that
  produces :class:`EyeRenderState` snapshots at 30 FPS.
* :mod:`robot.eye_engine.animator` - orchestrator that drives a single
  circular display at 30 FPS.
* :mod:`robot.eye_engine.eye_state` and
  :mod:`robot.eye_engine.render_state` - immutable value types.
"""

from robot.eye_engine.animation import EyeAnimator, EyeSide
from robot.eye_engine.animator import EyeDisplayAnimator
from robot.eye_engine.blink import BlinkController, BlinkPhase
from robot.eye_engine.emotions import Emotion, EmotionLibrary
from robot.eye_engine.eye_state import EyeState, GazeVector
from robot.eye_engine.render_state import EyeRenderState
from robot.eye_engine.renderer import EyeRenderer, RendererConfig

__all__ = [
    "BlinkController",
    "BlinkPhase",
    "Emotion",
    "EmotionLibrary",
    "EyeAnimator",
    "EyeDisplayAnimator",
    "EyeRenderState",
    "EyeRenderer",
    "EyeSide",
    "EyeState",
    "GazeVector",
    "RendererConfig",
]
