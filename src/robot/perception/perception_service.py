"""Perception service: runs face detection on camera frames and publishes events.

The :class:`PerceptionService` is the bridge between the camera hardware
and the event bus. It owns a background task that periodically:

1. Captures a frame from the camera.
2. Runs face detection on the frame.
3. Publishes :class:`FaceDetected` events on the bus for each face found.
4. Publishes a ``PerceptionScan`` event when no faces are found (so the
   behavior engine can tell the difference between "the camera didn't
   find anyone" and "we haven't looked yet").

The scan interval is **adaptive**: when the robot is IDLE, it scans
less frequently (saving CPU) and when it is CURIOUS (tracking a face),
it scans more often. This is controlled by
:attr:`idle_scan_interval_s` and :attr:`curious_scan_interval_s`.

The service is designed to be started inside the application's task group
alongside the face animator and idle behaviour.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field

from robot.behavior.state_machine import RobotState, StateMachine
from robot.events.bus import InMemoryEventBus
from robot.events.events import FaceDetected, StateChanged
from robot.interfaces.camera import Camera
from robot.logging import get_logger
from robot.perception.face_detector import (
    FaceDetector,
    FaceDetectorResult,
    NullFaceDetector,
    create_face_detector,
)

_log = get_logger("perception.service")


@dataclass(slots=True, frozen=True)
class PerceptionScan:
    """Published once per scan cycle, even when no faces are found."""

    face_count: int = 0
    timestamp: float = 0.0


@dataclass(slots=True)
class PerceptionService:
    """Periodically capture frames and run face detection.

    Parameters
    ----------
    camera:
        The camera to capture from.
    bus:
        The event bus to publish :class:`FaceDetected` events on.
    state_machine:
        The robot state machine, used to select the adaptive scan interval.
    face_detector:
        The face detector to use. Defaults to the best available
        (YuNet on OpenCV 5, CascadeClassifier on OpenCV 4, NullFaceDetector
        if no OpenCV).
    scan_interval_s:
        Default seconds between scans (used as fallback).
    idle_scan_interval_s:
        Seconds between scans when the robot is IDLE. Higher values save
        CPU. Default 2.0 s.
    curious_scan_interval_s:
        Seconds between scans when the robot is CURIOUS (tracking a face).
        Lower values give smoother tracking. Default 0.3 s.
    max_faces:
        Maximum faces to report per scan. 0 means unlimited.
    enabled:
        Whether to start scanning immediately.
    """

    camera: Camera
    bus: InMemoryEventBus
    state_machine: StateMachine | None = None
    face_detector: FaceDetector | None = None
    scan_interval_s: float = 0.5
    idle_scan_interval_s: float = 2.0
    curious_scan_interval_s: float = 0.3
    max_faces: int = 0
    enabled: bool = True
    _task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _stopped: bool = field(default=True, init=False)
    _scan_count: int = field(default=0, init=False)
    _face_count: int = field(default=0, init=False)
    _last_faces: list[FaceDetectorResult] = field(default_factory=list, init=False)
    _current_interval: float = field(default=2.0, init=False)

    def __post_init__(self) -> None:
        if self.face_detector is None:
            try:
                self.face_detector = create_face_detector(max_faces=self.max_faces)
                _log.info("perception.face_detector", backend=type(self.face_detector).__name__)
            except RuntimeError:
                self.face_detector = NullFaceDetector()
                _log.warning("perception.face_detector.fallback", backend="null")
        self._current_interval = self.idle_scan_interval_s
        # Subscribe to state changes so we can adapt the scan interval.
        if self.state_machine is not None:
            self.bus.subscribe(StateChanged, self._on_state_changed)

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        """Start the background perception loop."""
        if self._task is not None and not self._task.done():
            return
        self._stopped = False
        self._current_interval = self._adaptive_interval()
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self._scan_loop(), name="PerceptionService-scan")
        _log.info(
            "perception.started",
            interval_s=self._current_interval,
            idle_interval_s=self.idle_scan_interval_s,
            curious_interval_s=self.curious_scan_interval_s,
        )

    async def stop(self) -> None:
        """Stop the background perception loop."""
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        _log.info("perception.stopped", scans=self._scan_count, faces=self._face_count)

    # ------------------------------------------------------------------ adaptive
    def _adaptive_interval(self) -> float:
        """Pick the scan interval based on the current robot state."""
        if self.state_machine is None:
            return self.scan_interval_s
        state = self.state_machine.state
        if state in (RobotState.CURIOUS, RobotState.LISTENING, RobotState.THINKING):
            return self.curious_scan_interval_s
        if state is RobotState.SLEEPING:
            # Very slow scan while sleeping - just enough to notice if
            # someone appears and wakes the robot up.
            return self.idle_scan_interval_s * 2.0
        # IDLE or default
        return self.idle_scan_interval_s

    async def _on_state_changed(self, event: StateChanged) -> None:
        """Adjust the scan interval when the robot state changes."""
        new_interval = self._adaptive_interval()
        if new_interval != self._current_interval:
            old = self._current_interval
            self._current_interval = new_interval
            _log.debug(
                "perception.scan_interval_changed",
                old_interval_s=old,
                new_interval_s=new_interval,
                state=event.current.value,
            )

    # ------------------------------------------------------------------ scan loop
    async def _scan_loop(self) -> None:
        """Capture frames and run face detection in a loop."""
        while not self._stopped:
            if not self.enabled:
                await asyncio.sleep(self._current_interval)
                continue
            try:
                frame = await self.camera.capture()
                faces = await self.face_detector.detect(frame)  # type: ignore[union-attr]
            except Exception:
                _log.exception("perception.scan_failed")
                await asyncio.sleep(self._current_interval)
                continue
            self._scan_count += 1
            self._face_count += len(faces)
            self._last_faces = faces
            if faces:
                for face in faces:
                    await self.bus.publish(
                        FaceDetected(
                            x=face.x,
                            y=face.y,
                            confidence=face.confidence,
                        )
                    )
            else:
                # Publish a scan event so the behavior engine knows
                # we looked but found no one.
                await self.bus.publish(
                    PerceptionScan(
                        face_count=0,
                        timestamp=frame.timestamp,
                    )
                )
            await asyncio.sleep(self._current_interval)

    # ------------------------------------------------------------------ properties
    @property
    def scan_count(self) -> int:
        return self._scan_count

    @property
    def face_count(self) -> int:
        return self._face_count

    @property
    def last_faces(self) -> list[FaceDetectorResult]:
        return list(self._last_faces)


__all__ = ["PerceptionScan", "PerceptionService"]
