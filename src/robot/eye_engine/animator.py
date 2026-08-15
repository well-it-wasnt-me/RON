"""Single-display eye animator.

This is the long-running component the application boots. It owns the
single :class:`EyeAnimator` (per-eye state machine), the
:class:`EyeRenderer`, and the single :class:`Display`. It runs at 30 FPS
(default) and pushes every frame to the panel.

Two modes of operation are supported:

* **Sync** (default) - the eyes track the same render state.
* **Independent** - kept as a no-op for backwards compatibility with code
  written for the original two-eye design; only one :class:`EyeAnimator`
  is hosted so both modes look identical externally.

Bus event handlers
------------------

The animator subscribes to the bus for backwards compatibility:

* :class:`EmotionChanged` - sets the resting emotion.
* :class:`BlinkRequested` - triggers a blink.
* :class:`LookRequested` - moves the gaze.

The public API on the instance is the same as on the per-eye animator
(``blink``, ``double_blink``, ``look_left``, ``set_emotion``, …) so the
behavior engine does not need to change.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from robot.events.events import (
    AnimationFinished,
    BlinkRequested,
    DisplayUpdated,
    EmotionChanged,
    EmotionName,
    LookRequested,
)
from robot.eye_engine.animation import EyeAnimator, EyeSide
from robot.eye_engine.eye_state import EyeState, GazeVector
from robot.eye_engine.render_state import EyeRenderState
from robot.eye_engine.renderer import EyeRenderer
from robot.interfaces.display import Display, EyeFrame
from robot.logging import get_logger
from robot.utils.clock import Clock

if TYPE_CHECKING:
    from robot.events.bus import InMemoryEventBus

_log = get_logger("eye_engine.animator")


@dataclass(slots=True)
class EyeDisplayAnimator:
    """Drive a single circular TFT at 30 FPS with the eye engine.

    This is the high-level component the application boots. It holds the
    per-eye :class:`EyeAnimator` (the state machine), the
    :class:`EyeRenderer`, and the :class:`Display`. The animator is
    independent of the actual display technology: the renderer never
    touches SPI/I2C, and the display driver is the only thing that knows
    how to push pixels.
    """

    DEFAULT_FPS: ClassVar[int] = 30

    renderer: EyeRenderer
    display: Display
    clock: Clock
    fps: int = DEFAULT_FPS
    bus: InMemoryEventBus | None = None
    width: int = 240
    height: int = 240
    # Kept for backwards compatibility with the original two-eye API. Has
    # no effect in the single-display build; both modes look identical.
    sync: bool = True
    _eye: EyeAnimator = field(init=False)
    _stopped: bool = field(default=True, init=False)
    _frame_count: int = field(default=0, init=False)
    _idle_phase: float = field(default=0.0, init=False)

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
        if self.bus is not None:
            self.bus.subscribe(EmotionChanged, self._on_emotion)
            self.bus.subscribe(BlinkRequested, self._on_blink)
            self.bus.subscribe(LookRequested, self._on_look)

    # ------------------------------------------------------------------ properties
    @property
    def eye(self) -> EyeAnimator:
        return self._eye

    @property
    def left(self) -> EyeAnimator:
        """Backwards-compat alias for the (only) eye animator."""
        return self._eye

    @property
    def right(self) -> EyeAnimator:
        """Backwards-compat alias - both eyes share the same animator."""
        return self._eye

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_running(self) -> bool:
        return not self._stopped

    # ------------------------------------------------------------------ direct API
    def set_emotion(self, emotion: EmotionName, intensity: float = 1.0) -> None:
        self._eye.set_emotion(emotion, intensity)

    def blink(self) -> None:
        self._eye.blink()

    def double_blink(self) -> None:
        self._eye.double_blink()

    def wink_left(self) -> None:
        self._eye.wink()

    def wink_right(self) -> None:
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

    def reset(self) -> None:
        self._eye.reset()

    def enable_sync(self) -> None:
        self.sync = True

    def enable_independent(self) -> None:
        # No-op: there is only one display. Kept for API compatibility.
        self.sync = False

    # ------------------------------------------------------------------ bus handlers
    async def _on_emotion(self, event: EmotionChanged) -> None:
        self.set_emotion(event.current, intensity=event.intensity)

    async def _on_blink(self, event: BlinkRequested) -> None:
        if event.left or event.right:
            self.blink()

    async def _on_look(self, event: LookRequested) -> None:
        self.look(event.x, event.y, event.duration_s)

    # ------------------------------------------------------------------ frame loop
    def step(self, drift: bool = True) -> EyeFrame:
        """Advance one frame, render, push to the display, and return the frame."""
        if drift:
            self._idle_phase += 1.0 / self.fps
            self._eye.drift(amplitude=0.10, speed=0.6)
        render_state = self._eye.step()
        frame = self.renderer.render(render_state)
        self._frame_count += 1
        return frame

    async def _step_async(self, drift: bool = True) -> EyeFrame:
        frame = self.step(drift=drift)
        await self._push(frame)
        return frame

    async def _push(self, frame: EyeFrame) -> None:
        try:
            await self.display.show(frame)
        except Exception:
            _log.exception("animator.display_failed")
        if self.bus is not None:
            with contextlib.suppress(Exception):
                await self.bus.publish(DisplayUpdated(display="left"))

    async def run_forever(self) -> None:
        """Run the render loop at ``fps`` until :meth:`stop` is called."""
        self._stopped = False
        frame_interval = 1.0 / self.fps
        try:
            while not self._stopped:
                await self._step_async(drift=True)
                await self.clock.sleep(frame_interval)
                if self.bus is not None and self._frame_count % self.fps == 0:
                    with contextlib.suppress(Exception):
                        await self.bus.publish(AnimationFinished(name="second_tick"))
        finally:
            self._stopped = True

    def stop(self) -> None:
        self._stopped = True

    # ------------------------------------------------------------------ introspection
    def current_eye_state(self) -> EyeState:
        """Return the legacy :class:`EyeState` view of the eye (for back-compat)."""
        state = self._eye.render_state()
        return EyeState(
            emotion=self._eye.emotion,
            gaze=GazeVector(state.gaze_x, state.gaze_y),
            openness=state.openness,
            pupil_dilation=state.pupil_dilation,
            intensity=1.0,
            asymmetric=False,
        )

    def render_state(self) -> EyeRenderState:
        """Return the current per-eye render state."""
        return self._eye.render_state()


__all__ = ["EyeDisplayAnimator"]
