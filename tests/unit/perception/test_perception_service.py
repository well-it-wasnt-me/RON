"""Tests for the PerceptionService."""

from __future__ import annotations

import asyncio

import pytest

from robot.behavior.state_machine import RobotState, StateMachine
from robot.events.bus import InMemoryEventBus
from robot.events.events import FaceDetected
from robot.hardware.sensors.mock_camera import MockCamera
from robot.interfaces.camera import Frame
from robot.perception.face_detector import FaceDetectorResult, NullFaceDetector
from robot.perception.perception_service import PerceptionScan, PerceptionService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _OneFaceDetector:
    """A detector that always finds exactly one face at the centre."""

    async def detect(self, frame: Frame) -> list[FaceDetectorResult]:
        return [
            FaceDetectorResult(x=0.5, y=0.4, size=0.15, confidence=0.9, timestamp=frame.timestamp)
        ]


class _NoFaceDetector:
    """A detector that never finds anything."""

    async def detect(self, frame: Frame) -> list[FaceDetectorResult]:
        return []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_service_publishes_face_events() -> None:
    bus = InMemoryEventBus()
    camera = MockCamera()
    detector = _OneFaceDetector()
    service = PerceptionService(
        camera=camera,
        bus=bus,
        face_detector=detector,
        scan_interval_s=0.05,
        max_faces=3,
    )
    received: list[FaceDetected] = []
    bus.subscribe(FaceDetected, received.append)

    await service.start()
    await asyncio.sleep(0.3)
    await service.stop()

    assert len(received) > 0
    assert received[0].x == pytest.approx(0.5, abs=0.01)
    assert received[0].confidence == pytest.approx(0.9, abs=0.01)


@pytest.mark.asyncio
async def test_service_publishes_scan_when_no_faces() -> None:
    bus = InMemoryEventBus()
    camera = MockCamera()
    detector = _NoFaceDetector()
    service = PerceptionService(
        camera=camera,
        bus=bus,
        face_detector=detector,
        scan_interval_s=0.05,
    )
    scans: list[PerceptionScan] = []
    bus.subscribe(PerceptionScan, scans.append)

    await service.start()
    await asyncio.sleep(0.3)
    await service.stop()

    assert len(scans) > 0
    assert all(s.face_count == 0 for s in scans)


@pytest.mark.asyncio
async def test_service_counts_scans_and_faces() -> None:
    bus = InMemoryEventBus()
    camera = MockCamera()
    detector = _OneFaceDetector()
    service = PerceptionService(
        camera=camera,
        bus=bus,
        face_detector=detector,
        scan_interval_s=0.05,
    )

    await service.start()
    await asyncio.sleep(0.3)
    await service.stop()

    assert service.scan_count > 0
    assert service.face_count > 0
    assert len(service.last_faces) > 0


@pytest.mark.asyncio
async def test_service_disabled_does_not_scan() -> None:
    bus = InMemoryEventBus()
    camera = MockCamera()
    detector = _OneFaceDetector()
    service = PerceptionService(
        camera=camera,
        bus=bus,
        face_detector=detector,
        scan_interval_s=0.05,
        enabled=False,
    )
    received: list[FaceDetected] = []
    bus.subscribe(FaceDetected, received.append)

    await service.start()
    await asyncio.sleep(0.2)
    await service.stop()

    assert len(received) == 0


@pytest.mark.asyncio
async def test_service_stops_cleanly() -> None:
    bus = InMemoryEventBus()
    camera = MockCamera()
    service = PerceptionService(
        camera=camera,
        bus=bus,
        face_detector=NullFaceDetector(),
        scan_interval_s=0.1,
    )
    await service.start()
    await service.stop()
    assert service._stopped is True


@pytest.mark.asyncio
async def test_adaptive_interval_idle() -> None:
    """When IDLE, the service should use idle_scan_interval_s."""
    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    await sm.transition(RobotState.IDLE)

    service = PerceptionService(
        camera=MockCamera(),
        bus=bus,
        state_machine=sm,
        face_detector=NullFaceDetector(),
        scan_interval_s=0.5,
        idle_scan_interval_s=2.0,
        curious_scan_interval_s=0.3,
    )

    interval = service._adaptive_interval()
    assert interval == 2.0  # IDLE -> idle interval


@pytest.mark.asyncio
async def test_adaptive_interval_curious() -> None:
    """When CURIOUS, the service should use curious_scan_interval_s."""
    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    await sm.transition(RobotState.IDLE)
    await sm.transition(RobotState.CURIOUS)

    service = PerceptionService(
        camera=MockCamera(),
        bus=bus,
        state_machine=sm,
        face_detector=NullFaceDetector(),
        scan_interval_s=0.5,
        idle_scan_interval_s=2.0,
        curious_scan_interval_s=0.3,
    )

    interval = service._adaptive_interval()
    assert interval == 0.3  # CURIOUS -> curious interval


@pytest.mark.asyncio
async def test_adaptive_interval_sleeping() -> None:
    """When SLEEPING, the service should use 2x the idle interval."""
    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    await sm.transition(RobotState.IDLE)
    await sm.transition(RobotState.SLEEPING)

    service = PerceptionService(
        camera=MockCamera(),
        bus=bus,
        state_machine=sm,
        face_detector=NullFaceDetector(),
        scan_interval_s=0.5,
        idle_scan_interval_s=2.0,
        curious_scan_interval_s=0.3,
    )

    interval = service._adaptive_interval()
    assert interval == 4.0  # SLEEPING -> 2 * idle interval


# ---------------------------------------------------------------------------
# Tracked-face "known" heuristic
# ---------------------------------------------------------------------------
def _service(known_face_scans: int = 3) -> PerceptionService:
    """A perception service wired only for the synchronous tracker tests."""
    return PerceptionService(
        camera=MockCamera(),
        bus=InMemoryEventBus(),
        face_detector=NullFaceDetector(),
        known_face_scans=known_face_scans,
    )


def _face(x: float, y: float) -> FaceDetectorResult:
    return FaceDetectorResult(x=x, y=y, size=0.15, confidence=0.9, timestamp=0.0)


def test_tracked_face_becomes_known_after_persistence() -> None:
    """A face in the same place becomes 'known' once it persists enough scans."""
    service = _service(known_face_scans=3)
    # Scan 1 + 2: not yet known. Scan 3: known.
    assert service._label_faces([_face(0.5, 0.4)])[0][1] is False
    assert service._label_faces([_face(0.5, 0.4)])[0][1] is False
    assert service._label_faces([_face(0.5, 0.4)])[0][1] is True


def test_new_face_is_not_known() -> None:
    """A freshly detected face is never 'known' on its first scan."""
    service = _service()
    assert service._label_faces([_face(0.3, 0.3)])[0][1] is False


def test_missed_face_decays_and_drops() -> None:
    """A face that vanishes decays; reappearing starts fresh (not instantly known)."""
    service = _service(known_face_scans=2)
    # Build up to known.
    service._label_faces([_face(0.5, 0.4)])
    assert service._label_faces([_face(0.5, 0.4)])[0][1] is True
    # Face disappears for more than _KEEP_MISSED scans -> track dropped.
    service._label_faces([])
    service._label_faces([])
    service._label_faces([])
    assert service._tracked == []
    # Reappears: must persist again before being known.
    assert service._label_faces([_face(0.5, 0.4)])[0][1] is False


def test_two_faces_tracked_separately() -> None:
    """Two distinct faces are tracked independently; only the persistent one is known."""
    service = _service(known_face_scans=3)
    left = _face(0.2, 0.5)
    right = _face(0.8, 0.5)
    # Left persists; right is new each... actually keep right stable too but
    # only scan it once.
    service._label_faces([left])
    service._label_faces([left])
    labelled = service._label_faces([left, right])
    by_pos = {round(f.x, 2): known for f, known in labelled}
    assert by_pos[0.2] is True  # left has persisted 3 scans
    assert by_pos[0.8] is False  # right is on its first scan


def test_known_flag_published_on_event() -> None:
    """The service sets FaceDetected.known once a face has persisted."""
    bus = InMemoryEventBus()
    camera = MockCamera()
    detector = _OneFaceDetector()
    service = PerceptionService(
        camera=camera,
        bus=bus,
        face_detector=detector,
        scan_interval_s=0.02,
        known_face_scans=3,
    )
    received: list[FaceDetected] = []
    bus.subscribe(FaceDetected, received.append)

    async def run() -> None:
        await service.start()
        await asyncio.sleep(0.2)
        await service.stop()

    asyncio.run(run())
    assert received, "expected FaceDetected events"
    # Early events are not known; once persisted, at least one is known.
    assert any(e.known is False for e in received), "early faces should be unknown"
    assert any(e.known is True for e in received), "a persisted face should become known"
    assert all(e.size == pytest.approx(0.15) for e in received)
