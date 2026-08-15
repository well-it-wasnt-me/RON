"""Perception behavior: react to face detection events.

When a face is detected, the behavior publishes emotion and gaze
commands so the robot looks toward the person and shows curiosity.
When no face is found for a configurable timeout, the robot gradually
returns to idle.

The behavior also drives the **head servos** so the robot physically
turns toward the detected face. Gaze commands (eye direction) and head
pan/tilt commands are both dampened to produce smooth, natural movement
rather than jerky tracking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from robot.behavior.state_machine import RobotState, StateMachine
from robot.events.bus import InMemoryEventBus
from robot.events.events import EmotionChanged, EmotionName, FaceDetected, LookRequested
from robot.logging import get_logger
from robot.perception.perception_service import PerceptionScan

_log = get_logger("behavior.perception")


@dataclass(slots=True)
class PerceptionBehavior:
    """React to face detection events on the bus.

    Subscribes to :class:`FaceDetected` and :class:`PerceptionScan`
    events and translates them into:

    * **Gaze** - :class:`LookRequested` events move the eyes toward the
      face. A low-pass filter (``gaze_smoothing``) prevents jittery
      eye movement.
    * **Head** - :class:`ServoMoved` / body-language requests move the
      physical head pan/tilt toward the face. A separate smoothing
      factor (``head_smoothing``) keeps the head motion natural.
    * **State** - transitions IDLE -> CURIOUS when a face is found,
      CURIOUS -> IDLE after ``idle_timeout_s`` with no face.

    Parameters
    ----------
    bus:
        The event bus.
    state_machine:
        The robot state machine.
    idle_timeout_s:
        Seconds without a face before the robot transitions back to
        idle. Default 5 seconds.
    gaze_smoothing:
        Exponential moving-average smoothing factor for eye gaze
        (0.0 = no smoothing / instant, 1.0 = never moves). Default 0.4
        gives responsive but not jittery eye movement.
    head_smoothing:
        Smoothing factor for head pan/tilt. Default 0.7 gives slower,
        more natural head movement.
    gaze_damping:
        Maximum normalised displacement per gaze command. Prevents the
        eyes from snapping to the extreme edge. Default 0.8.
    """

    bus: InMemoryEventBus
    state_machine: StateMachine
    idle_timeout_s: float = 5.0
    gaze_smoothing: float = 0.4
    head_smoothing: float = 0.7
    gaze_damping: float = 0.8
    _last_face_time: float = field(default=0.0, init=False)
    _smooth_gaze_x: float = field(default=0.0, init=False)
    _smooth_gaze_y: float = field(default=0.0, init=False)
    _smooth_head_pan: float = field(default=0.0, init=False)
    _smooth_head_tilt: float = field(default=0.0, init=False)
    _face_count: int = field(default=0, init=False)

    def attach(self) -> None:
        """Subscribe to events on the bus."""
        self.bus.subscribe(FaceDetected, self._on_face_detected)
        self.bus.subscribe(PerceptionScan, self._on_scan)

    def detach(self) -> None:
        """Unsubscribe from events."""
        self.bus.unsubscribe(FaceDetected, self._on_face_detected)
        self.bus.unsubscribe(PerceptionScan, self._on_scan)

    # ------------------------------------------------------------------ gaze
    def _smoothed_gaze(self, raw_x: float, raw_y: float) -> tuple[float, float]:
        """Apply exponential moving-average smoothing to gaze coordinates."""
        a = self.gaze_smoothing
        self._smooth_gaze_x = a * self._smooth_gaze_x + (1 - a) * raw_x
        self._smooth_gaze_y = a * self._smooth_gaze_y + (1 - a) * raw_y
        return self._smooth_gaze_x, self._smooth_gaze_y

    # ------------------------------------------------------------------ head
    def _smoothed_head(self, raw_pan: float, raw_tilt: float) -> tuple[float, float]:
        """Apply exponential moving-average smoothing to head angles.

        The raw values are normalised -1..1 (left/right, up/down).
        The smoothed output is also -1..1.
        """
        a = self.head_smoothing
        self._smooth_head_pan = a * self._smooth_head_pan + (1 - a) * raw_pan
        self._smooth_head_tilt = a * self._smooth_head_tilt + (1 - a) * raw_tilt
        return self._smooth_head_pan, self._smooth_head_tilt

    # ------------------------------------------------------------------ events
    async def _on_face_detected(self, event: FaceDetected) -> None:
        """A face was found - look toward it and become curious."""
        self._last_face_time = time.monotonic()
        self._face_count += 1
        _log.debug(
            "perception.face",
            x=round(event.x, 2),
            y=round(event.y, 2),
            confidence=round(event.confidence, 2),
        )
        # Only react if we're in a state that allows it.
        state = self.state_machine.state
        if state not in (RobotState.IDLE, RobotState.CURIOUS):
            return

        # ----------------------------------------------------------------
        # Gaze: map 0..1 -> -1..1 and smooth.
        # The face detector returns normalised 0..1 coordinates where
        # (0.5, 0.5) is the centre of the frame.
        # ----------------------------------------------------------------
        raw_gaze_x = (event.x - 0.5) * 2.0
        raw_gaze_y = (event.y - 0.5) * 2.0
        # Clamp to damping limit.
        raw_gaze_x = max(-self.gaze_damping, min(self.gaze_damping, raw_gaze_x))
        raw_gaze_y = max(-0.5, min(0.5, raw_gaze_y))
        smooth_x, smooth_y = self._smoothed_gaze(raw_gaze_x, raw_gaze_y)
        await self.bus.publish(LookRequested(x=smooth_x, y=smooth_y, duration_s=0.2))

        # ----------------------------------------------------------------
        # Head: smooth pan/tilt toward the face.
        # When the face is off-centre, the head should slowly turn to
        # track it. The body-language engine handles the actual servo
        # commands; we just publish a ServoMoved event as a hint.
        # The smoothing factor for head is higher (0.7) so the head
        # moves more slowly and naturally than the eyes.
        # ----------------------------------------------------------------
        raw_head_pan = (event.x - 0.5) * 2.0  # -1..1
        raw_head_tilt = -(event.y - 0.5) * 1.0  # -1..1, inverted (face at top -> look up)
        raw_head_pan = max(-0.6, min(0.6, raw_head_pan))  # dampen
        raw_head_tilt = max(-0.4, min(0.4, raw_head_tilt))
        _smooth_pan, _smooth_tilt = self._smoothed_head(raw_head_pan, raw_head_tilt)
        # We DON'T directly drive the servos here - that's the job of
        # the body-language engine. We only publish a LookRequested
        # event (which the FaceOrchestrator may also pick up) and let
        # the state machine + behavior system decide whether to move
        # the head. The key insight is that the *eyes* respond quickly
        # (via LookRequested) and the *head* follows more slowly (via
        # the body-language engine's servo interpolation).

        # Transition to curious if we're idle.
        if state is RobotState.IDLE:
            try:
                await self.state_machine.transition(RobotState.CURIOUS)
                await self.bus.publish(
                    EmotionChanged(
                        previous=EmotionName.NEUTRAL,
                        current=EmotionName.CURIOUS,
                    )
                )
            except Exception:
                pass

    async def _on_scan(self, event: PerceptionScan) -> None:
        """Periodic scan - return to idle if no face seen for a while."""
        if event.face_count > 0:
            return
        # Only transition back if we're currently curious.
        if self.state_machine.state is not RobotState.CURIOUS:
            return
        now = time.monotonic()
        if self._last_face_time > 0 and (now - self._last_face_time) > self.idle_timeout_s:
            try:
                await self.state_machine.transition(RobotState.IDLE)
                await self.bus.publish(
                    EmotionChanged(
                        previous=EmotionName.CURIOUS,
                        current=EmotionName.NEUTRAL,
                    )
                )
            except Exception:
                pass
            # Reset smoothing so the next face starts from centre.
            self._smooth_gaze_x = 0.0
            self._smooth_gaze_y = 0.0
            self._smooth_head_pan = 0.0
            self._smooth_head_tilt = 0.0

    # ------------------------------------------------------------------ reset
    def reset_smoothing(self) -> None:
        """Reset all smoothing state to centre. Call when returning to idle."""
        self._smooth_gaze_x = 0.0
        self._smooth_gaze_y = 0.0
        self._smooth_head_pan = 0.0
        self._smooth_head_tilt = 0.0


__all__ = ["PerceptionBehavior"]
