"""Animation primitives for one eye.

The :class:`EyeAnimationState` and :class:`EyeAnimator` are display-aware
in the sense that they produce :class:`~robot.eye_engine.render_state.EyeRenderState`
snapshots, but they are independent of the actual
:class:`~robot.interfaces.display.Display` driver - the renderer is the
only thing that knows about pixels.

The animator can be used in two modes:

* **Independent** (default) - every eye runs its own animator. Use this
  when the robot should wink or look in two directions at once.
* **Synchronised** - every eye produces the exact same render state on
  every frame. This is the "normal" mode and matches the most common
  desktop-companion behaviour.

Supported animations (the public API):

* :meth:`EyeAnimator.blink` - single blink.
* :meth:`EyeAnimator.double_blink` - two quick blinks in a row.
* :meth:`EyeAnimator.look` - instant gaze change.
* :meth:`EyeAnimator.look_left`, :meth:`look_right`, :meth:`look_up`,
  :meth:`look_down` - directional gazes.
* :meth:`EyeAnimator.set_emotion` - set the resting state (happy,
  sleepy, surprised, angry, …).
* :meth:`EyeAnimator.drift` - slow idle gaze drift (call repeatedly).
* :meth:`EyeAnimator.wink` - close one eye briefly.

The animator is **frame-rate independent**: the public methods are
called once per frame; the state machine consumes the frame interval
to advance internal timers.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Final

from robot.events.events import EmotionName
from robot.eye_engine.render_state import EyeRenderState
from robot.logging import get_logger

_log = get_logger("eye_engine.animation")


# ---------------------------------------------------------------------------
# Enums / data
# ---------------------------------------------------------------------------
class EyeSide(str, enum.Enum):
    """Which physical eye the animator drives."""

    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


class _AnimKind(str, enum.Enum):
    IDLE = "idle"
    BLINK = "blink"
    DOUBLE_BLINK = "double_blink"
    LOOK = "look"
    WINK = "wink"
    EMOTION = "emotion"


@dataclass(slots=True, frozen=True)
class _GazeTarget:
    x: float
    y: float
    duration_s: float = 0.25


@dataclass(slots=True, frozen=True)
class _BlinkDef:
    closing_s: float = 0.06
    closed_s: float = 0.02
    opening_s: float = 0.10


# ---------------------------------------------------------------------------
# Emotion catalogue
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class _EmotionDef:
    openness: float
    pupil_dilation: float
    gaze: tuple[float, float]
    lid_top: float = 0.0
    lid_bottom: float = 0.0
    highlight: tuple[float, float] = (0.3, 0.3)


_EMOTIONS: Final[dict[EmotionName, _EmotionDef]] = {
    EmotionName.NEUTRAL: _EmotionDef(openness=1.0, pupil_dilation=0.5, gaze=(0.0, 0.0)),
    EmotionName.HAPPY: _EmotionDef(
        openness=0.7,
        pupil_dilation=0.45,
        gaze=(0.0, 0.05),
        lid_top=0.05,
        lid_bottom=0.30,
    ),
    EmotionName.SAD: _EmotionDef(
        openness=0.7,
        pupil_dilation=0.7,
        gaze=(0.0, -0.3),
        lid_top=0.35,
        lid_bottom=0.0,
    ),
    EmotionName.ANGRY: _EmotionDef(
        openness=0.85,
        pupil_dilation=0.3,
        gaze=(0.0, -0.15),
        lid_top=0.45,
        lid_bottom=0.0,
    ),
    EmotionName.SURPRISED: _EmotionDef(
        openness=1.0,
        pupil_dilation=0.2,
        gaze=(0.0, 0.0),
        highlight=(0.0, 0.0),
    ),
    EmotionName.SLEEPY: _EmotionDef(
        openness=0.20,
        pupil_dilation=0.7,
        gaze=(0.0, 0.0),
        lid_top=0.75,
        lid_bottom=0.0,
    ),
    EmotionName.THINKING: _EmotionDef(
        openness=0.7,
        pupil_dilation=0.5,
        gaze=(0.25, 0.0),
    ),
    EmotionName.CURIOUS: _EmotionDef(
        openness=0.95,
        pupil_dilation=0.55,
        gaze=(0.10, 0.10),
        highlight=(0.4, 0.4),
    ),
}


def emotion_def(name: EmotionName) -> _EmotionDef:
    return _EMOTIONS.get(name, _EMOTIONS[EmotionName.NEUTRAL])


# ---------------------------------------------------------------------------
# Per-eye animator (the smallest unit - one eye, no global state)
# ---------------------------------------------------------------------------
class EyeAnimator:
    """State machine that produces :class:`EyeRenderState` for one eye.

    The animator exposes a small, explicit API. Every method that changes
    state is a "command" - the actual movement happens over the next
    several frames, interpolated smoothly.

    Use :meth:`step` to advance one frame; the caller (a
    :class:`DualEyeAnimator` or a test) decides the frame interval.
    """

    DEFAULT_FPS: Final[int] = 30

    def __init__(
        self,
        side: EyeSide,
        width: int = 240,
        height: int = 240,
        fps: int = DEFAULT_FPS,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self.side = side
        self._width = width
        self._height = height
        self._fps = fps
        self._frame_interval = 1.0 / fps
        self._eye_radius = float(min(width, height)) * 0.42

        # Current / target state
        self._gaze_x = 0.0
        self._gaze_y = 0.0
        self._gaze_target = _GazeTarget(0.0, 0.0, 0.0)
        self._openness = 1.0
        self._pupil_dilation = 0.5
        self._lid_top = 0.0
        self._lid_bottom = 0.0
        self._highlight_x = 0.3
        self._highlight_y = 0.3
        self._emotion: EmotionName = EmotionName.NEUTRAL
        self._intensity: float = 1.0

        # Animation state
        self._kind: _AnimKind = _AnimKind.IDLE
        self._anim_time: float = 0.0  # seconds since the animation started
        self._just_completed: bool = False  # set when an animation finishes this frame
        self._blink_def: _BlinkDef = _BlinkDef()
        self._pending_double_blink: bool = False
        self._gaze_origin: tuple[float, float] = (0.0, 0.0)
        self._gaze_duration: float = 0.0
        # Wink uses lid_top/lid_bottom driven from "wink" without affecting gaze
        self._wink_time: float = 0.0
        self._wink_duration: float = 0.0
        # Frame count + last render for diagnostics
        self._frame_count: int = 0

    # ------------------------------------------------------------------ properties
    @property
    def emotion(self) -> EmotionName:
        return self._emotion

    @property
    def openness(self) -> float:
        return self._openness

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    # ------------------------------------------------------------------ public commands
    def blink(self) -> None:
        """Trigger a single blink (interrupts any current blink)."""
        self._kind = _AnimKind.BLINK
        self._anim_time = 0.0
        self._blink_def = _BlinkDef()
        self._pending_double_blink = False

    def double_blink(self) -> None:
        """Trigger two quick blinks in succession."""
        self._kind = _AnimKind.DOUBLE_BLINK
        self._anim_time = 0.0
        self._blink_def = _BlinkDef(closing_s=0.05, closed_s=0.015, opening_s=0.08)
        self._pending_double_blink = True

    def wink(self) -> None:
        """Close this eye briefly while the other eye stays open (independent mode)."""
        self._kind = _AnimKind.WINK
        self._wink_time = 0.0
        self._wink_duration = 0.30

    def look(self, x: float, y: float, duration_s: float = 0.25) -> None:
        """Smoothly move the gaze to ``(x, y)`` over ``duration_s`` seconds.

        ``x`` and ``y`` are in ``[-1, 1]``.
        """
        x = max(-1.0, min(1.0, x))
        y = max(-1.0, min(1.0, y))
        self._gaze_origin = (self._gaze_x, self._gaze_y)
        self._gaze_duration = max(0.0, duration_s)
        self._gaze_target = _GazeTarget(x, y, duration_s)
        self._kind = _AnimKind.LOOK
        self._anim_time = 0.0

    def look_left(self, duration_s: float = 0.25) -> None:
        self.look(-1.0, 0.0, duration_s)

    def look_right(self, duration_s: float = 0.25) -> None:
        self.look(1.0, 0.0, duration_s)

    def look_up(self, duration_s: float = 0.25) -> None:
        self.look(0.0, -1.0, duration_s)

    def look_down(self, duration_s: float = 0.25) -> None:
        self.look(0.0, 1.0, duration_s)

    def look_center(self, duration_s: float = 0.25) -> None:
        self.look(0.0, 0.0, duration_s)

    def set_emotion(self, emotion: EmotionName, intensity: float = 1.0) -> None:
        """Set the resting emotion (overrides eyelid/iris defaults)."""
        self._emotion = emotion
        self._intensity = max(0.0, min(1.0, intensity))
        d = emotion_def(emotion)
        # Snap the resting values to the emotion
        self._openness = d.openness
        self._pupil_dilation = d.pupil_dilation
        self._lid_top = d.lid_top
        self._lid_bottom = d.lid_bottom
        self._highlight_x = d.highlight[0]
        self._highlight_y = d.highlight[1]
        # A non-blink/gaze animation can stay running; the next step will
        # ease gaze back to the emotion default.
        if self._kind not in (_AnimKind.BLINK, _AnimKind.DOUBLE_BLINK, _AnimKind.WINK):
            self.look(d.gaze[0], d.gaze[1], duration_s=0.4)

    def drift(self, amplitude: float = 0.15, speed: float = 0.6) -> None:
        """Apply a slow, smooth gaze drift.

        Call this once per frame from the caller; it advances an internal
        phase. The drift is sinusoidal and looks like an idle glance
        around the current gaze target.
        """
        phase = self._frame_count * speed * self._frame_interval
        # Lissajous-ish figure-eight for a "looking around" feel
        ox = math.sin(phase) * amplitude
        oy = math.sin(phase * 1.7) * amplitude * 0.6
        # Read the current gaze target from the last LOOK command
        tx, ty = self._gaze_target.x, self._gaze_target.y
        # Drift around the target but clamp
        x = max(-1.0, min(1.0, tx + ox))
        y = max(-1.0, min(1.0, ty + oy))
        # Drift is non-interrupting - we don't change _kind.
        self._gaze_x = x
        self._gaze_y = y

    def reset(self) -> None:
        """Stop any running animation and return to the neutral state."""
        self._kind = _AnimKind.IDLE
        self._anim_time = 0.0
        self._pending_double_blink = False
        self._wink_time = 0.0
        self._just_completed = False
        self._emotion = EmotionName.NEUTRAL
        self._intensity = 1.0
        d = _EMOTIONS[EmotionName.NEUTRAL]
        self._gaze_x = d.gaze[0]
        self._gaze_y = d.gaze[1]
        self._openness = d.openness
        self._pupil_dilation = d.pupil_dilation
        self._lid_top = d.lid_top
        self._lid_bottom = d.lid_bottom
        self._highlight_x = d.highlight[0]
        self._highlight_y = d.highlight[1]

    # ------------------------------------------------------------------ step / render
    def step(self) -> EyeRenderState:
        """Advance one frame and return the current render state."""
        dt = self._frame_interval
        self._frame_count += 1

        if self._kind is _AnimKind.BLINK:
            self._advance_blink(dt, double=False)
        elif self._kind is _AnimKind.DOUBLE_BLINK:
            self._advance_blink(dt, double=True)
        elif self._kind is _AnimKind.LOOK:
            self._advance_look(dt)
        elif self._kind is _AnimKind.WINK:
            self._advance_wink(dt)

        # If no gaze animation is running, ease gaze gently toward the
        # emotion default so the eye settles after a Look command.
        # If no gaze animation is running, ease gaze gently toward the
        # emotion default so the eye settles after a Look command - but
        # not on the frame the look just completed (otherwise the target
        # would be partially undone immediately).
        just_finished = self._just_completed
        self._just_completed = False
        if self._kind is not _AnimKind.LOOK and not just_finished:
            d = emotion_def(self._emotion)
            self._gaze_x = self._lerp(self._gaze_x, d.gaze[0], 0.05)
            self._gaze_y = self._lerp(self._gaze_y, d.gaze[1], 0.05)

        return self.render_state()

    def render_state(self) -> EyeRenderState:
        """Return the current :class:`EyeRenderState` without advancing."""
        return EyeRenderState(
            cx=self._width / 2.0,
            cy=self._height / 2.0,
            eye_radius=self._eye_radius,
            openness=self._openness,
            lid_top=self._lid_top,
            lid_bottom=self._lid_bottom,
            gaze_x=self._gaze_x,
            gaze_y=self._gaze_y,
            pupil_dilation=self._pupil_dilation,
            highlight_x=self._highlight_x,
            highlight_y=self._highlight_y,
        )

    # ------------------------------------------------------------------ internal
    def _advance_blink(self, dt: float, *, double: bool) -> None:
        bd = self._blink_def
        total = bd.closing_s + bd.closed_s + bd.opening_s
        if double:
            total = 2 * total + 0.05  # gap between the two blinks
        self._anim_time += dt
        t = self._anim_time

        if double:
            # Split the timeline into two blinks with a small gap.
            blink_total = bd.closing_s + bd.closed_s + bd.opening_s
            cycle = blink_total + 0.05
            phase_t = t % cycle if t < cycle * 2 else t
            if phase_t < blink_total:
                self._openness = self._blink_curve(phase_t, bd)
            else:
                self._openness = 1.0  # gap
            if t >= 2 * blink_total + 0.05:
                self._kind = _AnimKind.IDLE
                self._anim_time = 0.0
                self._openness = 1.0
                self._just_completed = True
        elif t < bd.closing_s:
            p = t / bd.closing_s
            self._openness = 1.0 - p * p  # ease-in to 0
        elif t < bd.closing_s + bd.closed_s:
            self._openness = 0.0
        elif t < total:
            p = (t - bd.closing_s - bd.closed_s) / bd.opening_s
            self._openness = p * (2.0 - p)  # ease-out to 1
        else:
            self._openness = 1.0
            self._kind = _AnimKind.IDLE
            self._anim_time = 0.0
            self._just_completed = True

    @staticmethod
    def _blink_curve(t: float, bd: _BlinkDef) -> float:
        if t < bd.closing_s:
            p = t / bd.closing_s
            return 1.0 - p * p
        if t < bd.closing_s + bd.closed_s:
            return 0.0
        p = (t - bd.closing_s - bd.closed_s) / bd.opening_s
        return p * (2.0 - p)

    def _advance_look(self, dt: float) -> None:
        self._anim_time += dt
        if self._gaze_duration <= 0.0:
            self._gaze_x, self._gaze_y = self._gaze_target.x, self._gaze_target.y
            self._kind = _AnimKind.IDLE
            self._just_completed = True
            return
        t = min(1.0, self._anim_time / self._gaze_duration)
        # Smoothstep easing
        eased = t * t * (3.0 - 2.0 * t)
        ox, oy = self._gaze_origin
        self._gaze_x = ox + (self._gaze_target.x - ox) * eased
        self._gaze_y = oy + (self._gaze_target.y - oy) * eased
        if t >= 1.0:
            self._kind = _AnimKind.IDLE
            self._anim_time = 0.0
            self._just_completed = True

    def _advance_wink(self, dt: float) -> None:
        self._wink_time += dt
        total = self._wink_duration
        half = total / 2.0
        if self._wink_time < half:
            p = self._wink_time / half
            self._lid_top = 0.6 * p
        elif self._wink_time < total:
            p = (self._wink_time - half) / half
            self._lid_top = 0.6 * (1.0 - p)
        else:
            self._lid_top = 0.0
            self._kind = _AnimKind.IDLE
            self._just_completed = True

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t


__all__ = [
    "EyeAnimator",
    "EyeRenderState",
    "EyeSide",
]
