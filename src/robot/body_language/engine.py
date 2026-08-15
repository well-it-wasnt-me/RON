"""The :class:`BodyLanguageEngine` - plays body-language requests on the servos.

The engine owns the :class:`ServoController` and a per-servo
:class:`ServoCalibration`. To perform a request it:

1. Asks the request for a list of :class:`ServoFrame` objects.
2. Plays each frame sequentially, interpolating every servo's angle
   over the frame's ``duration_s`` with a smoothstep easing.
3. Returns the new angles so the simulation overlay can read them.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field

from robot.body_language.requests import (
    DEFAULT_CALIBRATION,
    BodyRequest,
    ServoCalibration,
    ServoFrame,
)
from robot.face.model import ArmPose, HeadTilt
from robot.interfaces.servo import Servo, ServoController
from robot.logging import get_logger
from robot.utils.clock import Clock

_log = get_logger("body_language.engine")


# ---------------------------------------------------------------------------
# Pose (a snapshot of all 4 servo targets) - used for the simulation
# overlay and for the "propose a pose" code path that the EmotionEngine
# uses.
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class Pose:
    """A snapshot of every servo's target angle."""

    targets: dict[str, float] = field(default_factory=dict)

    def get(self, name: str, default: float = 90.0) -> float:
        return self.targets.get(name, default)


# ---------------------------------------------------------------------------
# Mapping: BodyLanguageHint -> Pose
# ---------------------------------------------------------------------------
def hint_to_pose(hint: object) -> Pose:  # noqa: PLR0912
    """Translate a :class:`BodyLanguageHint` into a static :class:`Pose`.

    The result represents a *resting* pose - the body-language engine
    applies it as a base, and transient requests (Wave, Celebrate, …)
    layer on top.
    """

    head_tilt: HeadTilt = getattr(hint, "head_tilt", HeadTilt.NEUTRAL)
    arm_pose: ArmPose = getattr(hint, "arm_pose", ArmPose.RELAXED)
    intensity: float = float(getattr(hint, "intensity", 1.0))

    head_targets: dict[str, float] = {
        "pan": 90.0,
        "tilt": 90.0,
    }
    arm_targets: dict[str, float] = {
        "left_arm": 90.0,
        "right_arm": 90.0,
    }

    # Head tilt mapping
    if head_tilt is HeadTilt.CURIOUS:
        head_targets["tilt"] = 90.0 - 15.0 * intensity
        head_targets["pan"] = 90.0 - 10.0 * intensity
    elif head_tilt is HeadTilt.THINKING:
        head_targets["tilt"] = 90.0 - 5.0 * intensity
        head_targets["pan"] = 90.0 - 5.0 * intensity
    elif head_tilt is HeadTilt.SLEEPY:
        head_targets["tilt"] = 90.0 + 25.0 * intensity
    elif head_tilt is HeadTilt.SAD:
        head_targets["tilt"] = 90.0 + 20.0 * intensity
        head_targets["pan"] = 90.0
    elif head_tilt is HeadTilt.EXCITED:
        head_targets["tilt"] = 90.0 - 10.0 * intensity
        head_targets["pan"] = 90.0
    else:
        head_targets["tilt"] = 90.0
        head_targets["pan"] = 90.0

    # Arm pose mapping
    if arm_pose is ArmPose.OPEN:
        arm_targets["left_arm"] = 90.0 - 20.0 * intensity
        arm_targets["right_arm"] = 90.0 + 20.0 * intensity
    elif arm_pose is ArmPose.WIDE:
        arm_targets["left_arm"] = 90.0 - 45.0 * intensity
        arm_targets["right_arm"] = 90.0 + 45.0 * intensity
    elif arm_pose is ArmPose.WAVING:
        arm_targets["right_arm"] = 90.0 - 30.0 * intensity
        arm_targets["left_arm"] = 90.0
    elif arm_pose is ArmPose.SHRUG:
        arm_targets["left_arm"] = 90.0 - 30.0 * intensity
        arm_targets["right_arm"] = 90.0 - 30.0 * intensity
        head_targets["tilt"] = 90.0 - 10.0 * intensity
    elif arm_pose is ArmPose.DOWN:
        arm_targets["left_arm"] = 90.0 + 20.0 * intensity
        arm_targets["right_arm"] = 90.0 + 20.0 * intensity
    elif arm_pose is ArmPose.POINT:
        arm_targets["right_arm"] = 90.0 - 30.0 * intensity
    else:  # RELAXED
        arm_targets["left_arm"] = 90.0
        arm_targets["right_arm"] = 90.0

    return Pose(targets={**head_targets, **arm_targets})


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class BodyLanguageEngine:
    """Drive the four servos with high-level body-language requests."""

    servo_controller: ServoController
    clock: Clock
    calibration: dict[str, ServoCalibration] = field(
        default_factory=lambda: dict(DEFAULT_CALIBRATION)
    )

    _current: Pose = field(default_factory=Pose, init=False)
    _busy: bool = field(default=False, init=False)

    def set_calibration(self, calibration: dict[str, ServoCalibration]) -> None:
        self.calibration = calibration

    def snapshot(self) -> Pose:
        """Current pose - used by the simulation overlay."""
        return self._current

    def apply_hint(self, hint: object) -> Pose:
        """Translate a :class:`BodyLanguageHint` into a Pose (no servo moves)."""
        pose = hint_to_pose(hint)
        self._current = pose
        return pose

    async def perform(self, request: BodyRequest) -> None:
        """Play a request on the servos."""
        if self._busy:
            # Skip: another request is already in flight. A more
            # sophisticated engine would queue or interrupt.
            return
        self._busy = True
        try:
            for frame in request.frames():
                await self._play_frame(frame)
        finally:
            self._busy = False

    def perform_sync(self, request: BodyRequest) -> None:
        """Synchronous version (for tests and the simulation CLI)."""
        for frame in request.frames():
            self._play_frame_sync(frame)

    async def play_frames(self, frames: Iterable[ServoFrame]) -> None:
        async for _ in self._aiter_frames(frames):
            pass

    async def _aiter_frames(self, frames: Iterable[ServoFrame]) -> AsyncIterator[None]:
        for frame in frames:
            await self._play_frame(frame)
            yield

    async def _play_frame(self, frame: ServoFrame) -> None:
        if not frame.targets:
            await self.clock.sleep(frame.duration_s)
            return
        # Compute the per-servo interpolation start positions
        start_targets = {name: self._current.get(name, 90.0) for name in frame.targets}
        steps = max(1, int(frame.duration_s * 30))
        step_s = frame.duration_s / steps
        for step in range(1, steps + 1):
            t = step / steps
            eased = t * t * (3.0 - 2.0 * t)
            targets: dict[str, float] = {}
            for name, target in frame.targets.items():
                start = start_targets[name]
                targets[name] = self._calibrate(name, start + (target - start) * eased)
            # Apply every servo in this slice
            await self._apply_targets(targets)
            self._current = Pose(targets={**self._current.targets, **targets})
            await self.clock.sleep(step_s)

    def _play_frame_sync(self, frame: ServoFrame) -> None:
        if not frame.targets:
            return
        # Apply every servo at this slice (the start_targets hint isn't needed
        # in the current sync implementation; kept here for future interpolation).
        for name, target in frame.targets.items():
            self._apply_targets_sync({name: self._calibrate(name, target)})
            self._current = Pose(
                targets={**self._current.targets, name: self._calibrate(name, target)}
            )
        # Hint: we don't actually interpolate in sync mode; we jump to the
        # target. The renderer/animator interpolates its own state.

    async def _apply_targets(self, targets: dict[str, float]) -> None:
        for name, angle in targets.items():
            servo = self._get_servo(name)
            if servo is not None:
                with contextlib.suppress(Exception):
                    await servo.move_to(angle, duration_s=0.0)

    def _apply_targets_sync(self, targets: dict[str, float]) -> None:
        for name, angle in targets.items():
            servo = self._get_servo(name)
            if servo is not None:
                # The sync method is used by the simulator CLI and the tests.
                # It updates the snapshot but does not call the async
                # ``servo.move_to`` (that would require a running loop).
                self._current = Pose(
                    targets={**self._current.targets, name: self._calibrate(name, angle)}
                )

    def _get_servo(self, name: str) -> Servo | None:
        try:
            return self.servo_controller.get(name)
        except Exception:
            return None

    def _calibrate(self, name: str, angle: float) -> float:
        cal = self.calibration.get(name)
        if cal is None:
            return max(0.0, min(180.0, angle))
        return cal.clamp(angle)


__all__ = ["BodyLanguageEngine", "Pose", "hint_to_pose"]
