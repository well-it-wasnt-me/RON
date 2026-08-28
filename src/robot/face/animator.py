"""The :class:`FaceAnimator` - drives the face at 30 FPS.

This is the high-level component the application boots. It owns:

* the per-eye :class:`EyeAnimator` (the existing state machine),
* a :class:`FaceRenderer`,
* a single :class:`Display` (one circular TFT),
* the active :class:`Theme`.

It runs at 30 FPS (configurable), produces one :class:`FaceModel` per
frame, applies the theme, renders to an :class:`EyeFrame`, and pushes
it to the display.

Animations are driven by **timelines** built on the existing
:mod:`robot.animation` framework (easing + parallel + queue + easing).
The animator exposes small explicit commands (blink, smile, …) and
each command internally composes a :class:`Timeline` that drives the
intermediate :class:`FaceModel` snapshots.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from robot.animation.timelines import (
    Timeline,
)
from robot.eye_engine.animation import EyeAnimator, EyeSide
from robot.face.animations import (
    SpeakingAnimation,
    ThinkingDotsAnimation,
    WakeAnimation,
    WakeFrame,
)
from robot.face.components import (
    Cheeks,
    Eye,
    Eyebrow,
    EyebrowShape,
    Eyelids,
    Gaze,
    Mouth,
    MouthShape,
)
from robot.face.emotions import EmotionEngine
from robot.face.model import (
    FaceModel,
)
from robot.face.renderer import FaceRenderer
from robot.face.themes import Theme
from robot.interfaces.display import Display, EyeFrame
from robot.logging import get_logger
from robot.utils.clock import Clock

if TYPE_CHECKING:
    from robot.events.bus import InMemoryEventBus
    from robot.performance.frame_profiler import FrameProfiler

_log = get_logger("face.animator")


# ---------------------------------------------------------------------------
# Easing shortcuts
# ---------------------------------------------------------------------------
def _ease_out(t: float) -> float:
    return 1.0 - (1.0 - t) ** 2


def _ease_in_out(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# Animator
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class FaceAnimator:
    """Drive a single circular face display at 30 FPS."""

    DEFAULT_FPS: ClassVar[int] = 30

    renderer: FaceRenderer
    display: Display
    clock: Clock
    emotions: EmotionEngine
    theme: Theme
    fps: int = DEFAULT_FPS
    bus: InMemoryEventBus | None = None
    width: int = 240
    height: int = 240

    _current: FaceModel = field(init=False)
    _target: FaceModel = field(init=False)
    _eye: EyeAnimator = field(init=False)
    _stopped: bool = field(default=True, init=False)
    _frame_count: int = field(default=0, init=False)
    _anim_time: float = field(default=0.0, init=False)
    _timeline: Timeline | None = field(default=None, init=False)
    _timeline_t: float = field(default=0.0, init=False)
    _timeline_total: float = field(default=0.0, init=False)
    _speaking_animation: SpeakingAnimation | None = field(default=None, init=False)
    _thinking_animation: ThinkingDotsAnimation | None = field(default=None, init=False)
    _wake_animation: WakeAnimation | None = field(default=None, init=False)
    _frame_profiler: FrameProfiler | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be > 0")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be > 0")
        self._eye = EyeAnimator(
            side=EyeSide.LEFT,
            width=self.width,
            height=self.height,
            fps=self.fps,
        )
        self._current = self.emotions.build("neutral")
        self._target = self._current
        if self.bus is not None:
            from robot.events.events import (
                BlinkRequested,
                EmotionChanged,
                LookRequested,
            )

            self.bus.subscribe(EmotionChanged, self._on_emotion)
            self.bus.subscribe(BlinkRequested, self._on_blink)
            self.bus.subscribe(LookRequested, self._on_look)

    # ------------------------------------------------------------------ properties
    @property
    def current(self) -> FaceModel:
        return self._current

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_running(self) -> bool:
        return not self._stopped

    # ------------------------------------------------------------------ public commands
    def set_emotion(self, name: str, intensity: float = 1.0) -> None:
        """Animate to the given emotion over ~0.3s.

        Calling this method clears any running speaking and thinking
        animations (they are interruptible).  The wake animation is
        *not* cleared -- it plays to completion.
        """
        # Clear interruptible animations on emotion change.
        self._speaking_animation = None
        self._thinking_animation = None
        target = self.emotions.build(name)
        # Apply intensity: scale the change in openness, mouth, etc.
        self._timeline = self._build_emotion_timeline(self._current, target, duration_s=0.3)
        self._target = target

    def blink(self) -> None:
        self._eye.blink()

    def double_blink(self) -> None:
        self._eye.double_blink()

    def wink(self) -> None:
        self._eye.wink()

    def look(self, x: float, y: float, duration_s: float = 0.25) -> None:
        self._eye.look(x, y, duration_s)

    def look_left(self, duration_s: float = 0.25) -> None:
        self._eye.look_left(duration_s)

    def look_right(self, duration_s: float = 0.25) -> None:
        self._eye.look_right(duration_s)

    def look_up(self, duration_s: float = 0.25) -> None:
        self._eye.look_up(duration_s)

    def look_down(self, duration_s: float = 0.25) -> None:
        self._eye.look_down(duration_s)

    def look_center(self, duration_s: float = 0.25) -> None:
        self._eye.look_center(duration_s)

    def drift(self, amplitude: float = 0.10, speed: float = 0.6) -> None:
        self._eye.drift(amplitude=amplitude, speed=speed)

    def bounce(self) -> None:
        """Run a one-shot bounce animation (squash & stretch)."""
        tl = self._build_bounce_timeline()
        self._timeline = tl

    def smile_grow(self) -> None:
        """Animate the mouth from NEUTRAL to SMILE over 0.4s."""
        target = self._current.with_mouth(Mouth(shape=MouthShape.SMILE, openness=0.4, width=0.7))
        self._timeline = self._build_mouth_timeline(self._current, target, 0.4)

    def mouth_open(self, duration_s: float = 0.3) -> None:
        target = self._current.with_mouth(Mouth(shape=MouthShape.OPEN, openness=0.6, width=0.6))
        self._start_timeline(self._build_mouth_timeline(self._current, target, duration_s))

    def eyebrow_raise(self, amount: float = 0.6) -> None:
        target = self._current.with_eyebrows(
            left=Eyebrow(shape=EyebrowShape.RAISED, raise_amount=amount),
            right=Eyebrow(shape=EyebrowShape.RAISED, raise_amount=amount),
        )
        self._timeline = self._build_eyebrow_timeline(self._current, target, 0.3)

    # ------------------------------------------------------------------ animation slots
    def set_speaking_animation(self, animation: SpeakingAnimation | None) -> None:
        """Set (or clear) the speaking animation that drives mouth shapes."""
        self._speaking_animation = animation

    def set_thinking_animation(self, animation: ThinkingDotsAnimation | None) -> None:
        """Set (or clear) the thinking animation that drives gaze shifts."""
        self._thinking_animation = animation

    def set_wake_animation(self, animation: WakeAnimation | None) -> None:
        """Set (or clear) the wake animation (highest priority override)."""
        self._wake_animation = animation

    def reset(self) -> None:
        self._current = self.emotions.build("neutral")
        self._target = self._current
        self._timeline = None
        self._speaking_animation = None
        self._thinking_animation = None
        self._wake_animation = None
        self._eye.reset()

    # ------------------------------------------------------------------ bus
    async def _on_emotion(self, event: object) -> None:
        name = getattr(event, "current", None)
        if name is None:
            return
        # The event's ``current`` can be either an ``EmotionName`` enum or
        # a plain string; normalise to a string before looking it up.
        if hasattr(name, "value"):
            name = name.value
        elif not isinstance(name, str):
            return
        self.set_emotion(name)

    async def _on_blink(self, event: object) -> None:
        self.blink()

    async def _on_look(self, event: object) -> None:
        self.look(
            float(getattr(event, "x", 0.0)),
            float(getattr(event, "y", 0.0)),
            duration_s=float(getattr(event, "duration_s", 0.25)),
        )

    # ------------------------------------------------------------------ frame loop
    def step(self, drift: bool = True) -> EyeFrame:
        """Advance one frame and push to the display.

        The rendering pipeline applies overrides in priority order:

        1. Eye drift + blink state (from :class:`EyeAnimator`).
        2. Thinking animation gaze override.
        3. Timeline-based emotion interpolation.
        4. Speaking animation mouth override.
        5. Wake animation full override (highest priority).
        """
        dt = 1.0 / self.fps
        self._frame_count += 1

        if drift:
            self._eye.drift(amplitude=0.10, speed=0.6)
        # Eye state: pull current gaze from the eye animator
        eye_state = self._eye.step()
        # Translate the eye's (gaze, openness) into left/right Eye components
        new_eyes = self._eyes_from_eye_state(eye_state)

        # Thinking animation: override gaze when active.
        if self._thinking_animation is not None:
            gaze = self._thinking_animation.step(dt)
            new_eyes = (
                Eye(
                    gaze=gaze,
                    openness=new_eyes[0].openness,
                    pupil_dilation=new_eyes[0].pupil_dilation,
                    highlight=new_eyes[0].highlight,
                ),
                Eye(
                    gaze=gaze,
                    openness=new_eyes[1].openness,
                    pupil_dilation=new_eyes[1].pupil_dilation,
                    highlight=new_eyes[1].highlight,
                ),
            )

        # Apply timeline (smile, eyebrows, bounce, …) if any
        if self._timeline is not None:
            self._timeline_t += dt
            progress = min(1.0, self._timeline_t / max(0.001, self._timeline_total))
            # Invoke the on_update callbacks directly. Each builder
            # registered a single tween whose callback writes to
            # ``self._current`` (interpolation, bounce, …).
            for animation in self._timeline._items:
                if hasattr(animation, "on_update"):
                    animation.on_update(progress)
            if progress >= 1.0:
                self._timeline = None
                self._timeline_t = 0.0
        # Merge eye state + current body
        merged = self._current.with_eyes(left_eye=new_eyes[0], right_eye=new_eyes[1])
        # Update the current model so a.current reflects the rendered state.
        self._current = merged

        # Speaking animation: override mouth when active.
        if self._speaking_animation is not None:
            viseme = self._speaking_animation.step(dt)
            mouth_shape = MouthShape.OPEN if viseme.openness > 0.1 else MouthShape.NEUTRAL
            self._current = self._current.with_mouth(
                Mouth(shape=mouth_shape, openness=viseme.openness, width=viseme.width)
            )
            if not self._speaking_animation.has_frames:
                self._speaking_animation = None

        # Wake animation: full override (highest priority).
        if self._wake_animation is not None:
            wake_frame = self._wake_animation.step(dt)
            self._current = self._apply_wake_frame(self._current, wake_frame)
            if self._wake_animation.done:
                self._wake_animation = None

        # Apply theme
        themed = self.theme.apply(self._current)
        return self.renderer.render(themed)

    async def _step_async(self, drift: bool = True) -> EyeFrame:
        frame = self.step(drift=drift)
        await self._push(frame)
        return frame

    async def _push(self, frame: EyeFrame) -> None:
        try:
            await self.display.show(frame)
        except Exception:
            _log.exception("face.animator.display_failed")
        if self.bus is not None:
            from robot.events.events import DisplayUpdated

            with contextlib.suppress(Exception):
                await self.bus.publish(DisplayUpdated(display="face"))

    async def run_forever(self) -> None:
        self._stopped = False
        frame_interval = 1.0 / self.fps
        try:
            while not self._stopped:
                start = time.monotonic()
                await self._step_async(drift=True)
                end = time.monotonic()
                if self._frame_profiler is not None and getattr(
                    self._frame_profiler, "enabled", False
                ):
                    self._frame_profiler.record_frame(start, end)
                await self.clock.sleep(frame_interval)
        finally:
            self._stopped = True

    def stop(self) -> None:
        self._stopped = True

    # ------------------------------------------------------------------ timeline helpers
    def _start_timeline(self, timeline: Timeline) -> None:
        self._timeline = timeline
        self._timeline_t = 0.0
        # _timeline_total is set by the builder (so the builder can also do its own work)

    def _build_emotion_timeline(
        self, start: FaceModel, end: FaceModel, duration_s: float
    ) -> Timeline:
        """Build a timeline that interpolates between two FaceModels.

        We don't yet have a generic FaceModel interpolator, so we use a
        sequence of tween callbacks to lerp scalar fields. The
        renderer is what makes the result look smooth.
        """
        tl = Timeline()
        # Tween the mouth shape change discretely (mid-way)
        tl.tween(
            from_value=0.0,
            to_value=1.0,
            duration_s=duration_s,
            on_update=lambda t: self._set_interp(start, end, t),
        )
        self._timeline_total = duration_s
        return tl

    def _build_mouth_timeline(
        self, start: FaceModel, end: FaceModel, duration_s: float
    ) -> Timeline:
        tl = Timeline()
        tl.tween(
            from_value=0.0,
            to_value=1.0,
            duration_s=duration_s,
            on_update=lambda t: self._set_interp(start, end, t),
        )
        self._timeline_total = duration_s
        return tl

    def _build_eyebrow_timeline(
        self, start: FaceModel, end: FaceModel, duration_s: float
    ) -> Timeline:
        tl = Timeline()
        tl.tween(
            from_value=0.0,
            to_value=1.0,
            duration_s=duration_s,
            on_update=lambda t: self._set_interp(start, end, t),
        )
        self._timeline_total = duration_s
        return tl

    def _build_bounce_timeline(self) -> Timeline:
        """One-shot bounce: bounce up, squash on landing, settle.

        Uses the model's ``bounce`` and ``squash`` fields and writes a
        per-frame value through a single tween.
        """
        # Build a sequence of keyframes as a list of (t, bounce, squash)
        keyframes: list[tuple[float, float, float]] = [
            (0.00, 0.0, 1.00),
            (0.10, 0.50, 0.92),  # anticipation squash
            (0.30, 0.85, 1.10),  # jump stretch
            (0.55, 0.00, 0.95),  # impact squash
            (0.75, 0.30, 1.04),  # small bounce
            (1.00, 0.0, 1.00),  # settle
        ]
        duration = 0.6

        class _Bounce:
            def __init__(self) -> None:
                pass

        # Use a simple Tween + linear interp through keyframes
        def _sample(t: float) -> tuple[float, float]:
            for i in range(len(keyframes) - 1):
                t0, b0, s0 = keyframes[i]
                t1, b1, s1 = keyframes[i + 1]
                if t0 <= t <= t1:
                    local = (t - t0) / max(1e-6, t1 - t0)
                    local = _ease_in_out(local)
                    return (b0 + (b1 - b0) * local, s0 + (s1 - s0) * local)
            return keyframes[-1][1], keyframes[-1][2]

        def _apply(t: float) -> None:
            bounce, squash = _sample(t)
            # Apply in-place: store the bounce/squash on the model.
            # We achieve this by replacing self._current with a new model
            # that has the transformed bounce/squash.
            self._current = self._current.with_transform(bounce=bounce, squash=squash)

        tl = Timeline()
        tl.tween(from_value=0.0, to_value=1.0, duration_s=duration, on_update=_apply)
        self._timeline_total = duration
        return tl

    def _apply_timeline_at(self, model: FaceModel, t: float) -> FaceModel:
        """Most timelines write to ``self._current`` directly via tween
        callbacks. This method is the fallback for any future timeline
        that wants to be queryable.
        """
        return model

    def _apply_wake_frame(self, model: FaceModel, frame: WakeFrame) -> FaceModel:
        """Apply a :class:`WakeFrame` override to the model.

        The wake animation has the highest priority and overrides
        eyes, eyelids, mouth, and eyebrows.
        """
        return (
            model.with_eyes(
                left_eye=Eye(
                    gaze=frame.gaze,
                    openness=frame.eye_openness,
                    pupil_dilation=model.left_eye.pupil_dilation,
                    highlight=model.left_eye.highlight,
                ),
                right_eye=Eye(
                    gaze=frame.gaze,
                    openness=frame.eye_openness,
                    pupil_dilation=model.right_eye.pupil_dilation,
                    highlight=model.right_eye.highlight,
                ),
            )
            .with_eyelids(Eyelids(top=frame.eyelid_top, bottom=0.0))
            .with_mouth(
                Mouth(
                    shape=frame.mouth_shape, openness=frame.mouth_openness, width=frame.mouth_width
                )
            )
            .with_eyebrows(
                left=Eyebrow(shape=frame.eyebrow_shape, raise_amount=frame.eyebrow_raise),
                right=Eyebrow(shape=frame.eyebrow_shape, raise_amount=frame.eyebrow_raise),
            )
        )

    def _set_interp(self, start: FaceModel, end: FaceModel, t: float) -> None:
        """Snap-interpolate scalar fields from start to end at progress t."""
        e = _ease_in_out(t)
        left_eye = _interp_eye(start.left_eye, end.left_eye, e)
        right_eye = _interp_eye(start.right_eye, end.right_eye, e)
        eyelids = _interp_eyelids(start.eyelids, end.eyelids, e)
        left_brow = _interp_eyebrow(start.left_eyebrow, end.left_eyebrow, e)
        right_brow = _interp_eyebrow(start.right_eyebrow, end.right_eyebrow, e)
        mouth = _interp_mouth(start.mouth, end.mouth, e)
        cheeks = _interp_cheeks(start.cheeks, end.cheeks, e)
        overlay = end.overlay if e > 0.5 else start.overlay
        body_hint = end.body_hint if e > 0.5 else start.body_hint
        self._current = FaceModel(
            width=start.width,
            height=start.height,
            left_eye=left_eye,
            right_eye=right_eye,
            eyelids=eyelids,
            left_eyebrow=left_brow,
            right_eyebrow=right_brow,
            mouth=mouth,
            cheeks=cheeks,
            overlay=overlay,
            accessory=start.accessory,
            palette=start.palette,
            bounce=start.bounce,
            squash=start.squash,
            body_hint=body_hint,
        )

    def _eyes_from_eye_state(self, state: object) -> tuple[Eye, Eye]:
        """Translate the eye render state into two :class:`Eye` components."""
        # The legacy EyeState has a ``gaze`` field; the new EyeRenderState
        # has separate ``gaze_x`` / ``gaze_y`` floats. Handle both.
        if hasattr(state, "gaze_x"):
            gaze = Gaze(float(getattr(state, "gaze_x", 0.0)), float(getattr(state, "gaze_y", 0.0)))
        else:
            g = getattr(state, "gaze", Gaze())
            gaze = Gaze(float(g.x), float(g.y))
        openness = float(getattr(state, "openness", 1.0))
        dilation = float(getattr(state, "pupil_dilation", 0.5))
        eye = Eye(gaze=gaze, openness=openness, pupil_dilation=dilation, highlight=Gaze(0.3, 0.3))
        return eye, eye


# ---------------------------------------------------------------------------
# Field interpolators
# ---------------------------------------------------------------------------
def _interp_eye(a: Eye, b: Eye, t: float) -> Eye:
    return Eye(
        gaze=Gaze(
            a.gaze.x + (b.gaze.x - a.gaze.x) * t,
            a.gaze.y + (b.gaze.y - a.gaze.y) * t,
        ),
        openness=a.openness + (b.openness - a.openness) * t,
        pupil_dilation=a.pupil_dilation + (b.pupil_dilation - a.pupil_dilation) * t,
        highlight=Gaze(
            a.highlight.x + (b.highlight.x - a.highlight.x) * t,
            a.highlight.y + (b.highlight.y - a.highlight.y) * t,
        ),
        asymmetric=a.asymmetric or b.asymmetric,
    )


def _interp_eyelids(a: Eyelids, b: Eyelids, t: float) -> Eyelids:
    return Eyelids(
        top=a.top + (b.top - a.top) * t,
        bottom=a.bottom + (b.bottom - a.bottom) * t,
    )


def _interp_eyebrow(a: Eyebrow, b: Eyebrow, t: float) -> Eyebrow:
    # If both shapes are equal, just lerp params. Otherwise snap at t=0.5.
    if a.shape is b.shape:
        return Eyebrow(
            shape=a.shape,
            raise_amount=a.raise_amount + (b.raise_amount - a.raise_amount) * t,
            angle=a.angle + (b.angle - a.angle) * t,
        )
    return b if t > 0.5 else a


def _interp_mouth(a: Mouth, b: Mouth, t: float) -> Mouth:
    if a.shape is b.shape:
        return Mouth(
            shape=a.shape,
            openness=a.openness + (b.openness - a.openness) * t,
            width=a.width + (b.width - a.width) * t,
            asymmetry=a.asymmetry + (b.asymmetry - a.asymmetry) * t,
        )
    return b if t > 0.5 else a


def _interp_cheeks(a: Cheeks, b: Cheeks, t: float) -> Cheeks:
    if a.state is b.state:
        return Cheeks(state=a.state, intensity=a.intensity + (b.intensity - a.intensity) * t)
    return b if t > 0.5 else a


__all__ = ["FaceAnimator"]


# Re-export the helper for the integration tests
def _interp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t
