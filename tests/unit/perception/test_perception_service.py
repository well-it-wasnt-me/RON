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
