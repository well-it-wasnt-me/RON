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
class _TrackedFace:
    """A face tracked across consecutive scans for the known-face heuristic.

    RON has no face-recognition model, so "known" is approximated by
    *persistence*: a face that stays in roughly the same place for several
    scans in a row is assumed to be one the robot is currently engaged with
    (it entered CURIOUS to track it). ``seen_count`` grows with each
    matching scan and decays when the face is missed, so a face that
    disappears and reappears must prove itself again.
    """

    id: int
    x: float
    y: float
    size: float
    seen_count: int = 0
    missed: int = 0


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
    known_face_scans:
        Number of consecutive scans a face must persist in roughly the same
        place before it is flagged ``known`` on the published
        :class:`FaceDetected` event. This is the tracked-face heuristic:
        RON has no face-recognition model, so a face it has been steadily
        tracking (and likely turned CURIOUS toward) is treated as
        "remembered". Default 3.
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
    known_face_scans: int = 3
    enabled: bool = True
    _task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _stopped: bool = field(default=True, init=False)
    _scan_count: int = field(default=0, init=False)
    _face_count: int = field(default=0, init=False)
    _last_faces: list[FaceDetectorResult] = field(default_factory=list, init=False)
    _current_interval: float = field(default=2.0, init=False)
    _tracked: list[_TrackedFace] = field(default_factory=list, init=False, repr=False)
    _next_id: int = field(default=0, init=False, repr=False)

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
        self._tracked.clear()
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
                for face, known in self._label_faces(faces):
                    await self.bus.publish(
                        FaceDetected(
                            x=face.x,
                            y=face.y,
                            confidence=face.confidence,
                            size=face.size,
                            known=known,
                        )
                    )
            else:
                # No faces this scan — let the tracker decay every face, then
                # publish a scan event so the behavior engine knows we looked
                # but found no one.
                self._label_faces([])
                await self.bus.publish(
                    PerceptionScan(
                        face_count=0,
                        timestamp=frame.timestamp,
                    )
                )
            await asyncio.sleep(self._current_interval)

    # ------------------------------------------------------------------ tracking
    _TRACK_DIST: float = 0.2  # normalised centre distance for a match
    _KEEP_MISSED: int = 2  # drop a track after this many missed scans

    def _label_faces(self, faces: list[FaceDetectorResult]) -> list[tuple[FaceDetectorResult, bool]]:
        """Match detected faces to existing tracks and flag persisted ones.

        Returns each detected face paired with a ``known`` flag that is True
        once the track it matched has been seen for ``known_face_scans``
        consecutive scans. Tracks that no detected face matches decay
        (``seen_count`` drops, ``missed`` climbs) and are dropped once they
        have been missed for ``_KEEP_MISSED`` scans, so a face that
        vanishes and returns must build up persistence again.
        """
        matched_ids: set[int] = set()
        labelled: list[tuple[FaceDetectorResult, bool]] = []
        for face in faces:
            best: _TrackedFace | None = None
            best_d = self._TRACK_DIST
            for t in self._tracked:
                if t.id in matched_ids:
                    continue
                d = ((t.x - face.x) ** 2 + (t.y - face.y) ** 2) ** 0.5
                if d < best_d:
                    best_d = d
                    best = t
            if best is not None:
                best.x = face.x
                best.y = face.y
                best.size = face.size
                best.seen_count += 1
                best.missed = 0
                matched_ids.add(best.id)
                labelled.append((face, best.seen_count >= self.known_face_scans))
            else:
                track = _TrackedFace(
                    id=self._next_id,
                    x=face.x,
                    y=face.y,
                    size=face.size,
                    seen_count=1,
                    missed=0,
                )
                self._next_id += 1
                self._tracked.append(track)
                matched_ids.add(track.id)  # observed this scan; don't decay it
                labelled.append((face, False))
        # Decay unmatched tracks.
        for t in self._tracked:
            if t.id not in matched_ids:
                t.missed += 1
                if t.seen_count > 0:
                    t.seen_count -= 1
        self._tracked = [t for t in self._tracked if t.missed <= self._KEEP_MISSED]
        return labelled

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
