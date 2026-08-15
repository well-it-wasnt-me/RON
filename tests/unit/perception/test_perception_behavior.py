"""Tests for the PerceptionBehavior."""

from __future__ import annotations

import asyncio

import pytest

from robot.behavior.perception_behavior import PerceptionBehavior
from robot.behavior.state_machine import RobotState, StateMachine
from robot.events.bus import InMemoryEventBus
from robot.events.events import EmotionChanged, FaceDetected, LookRequested
from robot.perception.perception_service import PerceptionScan


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_face_detected_triggers_look() -> None:
    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    # Transition to IDLE so perception can react.
    await sm.transition(RobotState.IDLE)

    behavior = PerceptionBehavior(bus=bus, state_machine=sm, idle_timeout_s=5.0)
    behavior.attach()

    look_events: list[LookRequested] = []
    bus.subscribe(LookRequested, look_events.append)

    # Simulate a face detected at the right side of the frame.
    await bus.publish(FaceDetected(x=0.7, y=0.4, confidence=0.9))
    await asyncio.sleep(0.05)

    assert len(look_events) >= 1
    # The gaze should be toward the right (x > 0 since face is at 0.7).
    assert look_events[0].x > 0


@pytest.mark.asyncio
async def test_face_detected_transitions_to_curious() -> None:
    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    await sm.transition(RobotState.IDLE)

    behavior = PerceptionBehavior(bus=bus, state_machine=sm, idle_timeout_s=5.0)
    behavior.attach()

    emotion_events: list[EmotionChanged] = []
    bus.subscribe(EmotionChanged, emotion_events.append)

    await bus.publish(FaceDetected(x=0.5, y=0.5, confidence=0.8))
    await asyncio.sleep(0.05)

    # Should transition to CURIOUS
    assert sm.state == RobotState.CURIOUS
    # Should emit a CURIOUS emotion
    assert any(e.current.value == "curious" for e in emotion_events)


@pytest.mark.asyncio
async def test_gaze_smoothing() -> None:
    """Verify that successive face detections produce smoothed gaze."""
    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    await sm.transition(RobotState.IDLE)

    behavior = PerceptionBehavior(bus=bus, state_machine=sm, idle_timeout_s=5.0, gaze_smoothing=0.7)
    behavior.attach()

    look_events: list[LookRequested] = []
    bus.subscribe(LookRequested, look_events.append)

    # First face detection: raw x = 0.8, smoothed x should be less than
    # the raw value because smoothing starts from 0.
    await bus.publish(FaceDetected(x=0.8, y=0.5, confidence=0.9))
    await asyncio.sleep(0.05)

    assert len(look_events) >= 1
    first_x = look_events[0].x
    # With smoothing=0.7, smooth_x = 0.7 * 0 + 0.3 * 0.6 = 0.18
    assert 0.0 < first_x < 0.6  # smoothed is less than raw

    # Second detection at the same position: smoothed value should move
    # toward the target but not reach it yet.
    await bus.publish(FaceDetected(x=0.8, y=0.5, confidence=0.9))
    await asyncio.sleep(0.05)

    assert len(look_events) >= 2
    second_x = look_events[1].x
    # Second smoothing: 0.7 * 0.18 + 0.3 * 0.6 ≈ 0.31
    assert second_x > first_x  # moved toward target


@pytest.mark.asyncio
async def test_gaze_smoothing_resets_on_idle() -> None:
    """After returning to IDLE, smoothing should reset to centre."""
    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    await sm.transition(RobotState.IDLE)

    behavior = PerceptionBehavior(bus=bus, state_machine=sm, idle_timeout_s=5.0)
    behavior.attach()

    # Detect a face to move to CURIOUS.
    await bus.publish(FaceDetected(x=0.8, y=0.5, confidence=0.9))
    await asyncio.sleep(0.02)
    assert sm.state == RobotState.CURIOUS

    # Manually transition back to IDLE (simulating the timeout).
    await sm.transition(RobotState.IDLE)
    behavior.reset_smoothing()

    # Smoothing should have been reset.
    assert behavior._smooth_gaze_x == 0.0
    assert behavior._smooth_gaze_y == 0.0


@pytest.mark.asyncio
async def test_no_face_scan_keeps_curious() -> None:
    """If a scan reports no faces but we haven't timed out, stay CURIOUS."""
    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    await sm.transition(RobotState.IDLE)

    behavior = PerceptionBehavior(bus=bus, state_machine=sm, idle_timeout_s=100.0)
    behavior.attach()

    # First, detect a face to go to CURIOUS.
    await bus.publish(FaceDetected(x=0.5, y=0.5, confidence=0.8))
    await asyncio.sleep(0.02)
    assert sm.state == RobotState.CURIOUS

    # Immediately scan with no faces - should NOT transition back
    # because not enough time has passed.
    await bus.publish(PerceptionScan(face_count=0, timestamp=0.0))
    await asyncio.sleep(0.02)
    assert sm.state == RobotState.CURIOUS


@pytest.mark.asyncio
async def test_ignores_face_when_listening() -> None:
    """Face detection should not trigger gaze when in LISTENING state."""
    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    await sm.transition(RobotState.IDLE)
    await sm.transition(RobotState.LISTENING)

    behavior = PerceptionBehavior(bus=bus, state_machine=sm, idle_timeout_s=5.0)
    behavior.attach()

    look_events: list[LookRequested] = []
    bus.subscribe(LookRequested, look_events.append)

    await bus.publish(FaceDetected(x=0.8, y=0.5, confidence=0.9))
    await asyncio.sleep(0.05)

    # No look events should be emitted while in LISTENING state.
    assert len(look_events) == 0


@pytest.mark.asyncio
async def test_detach_stops_receiving_events() -> None:
    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    await sm.transition(RobotState.IDLE)

    behavior = PerceptionBehavior(bus=bus, state_machine=sm)
    behavior.attach()
    behavior.detach()

    # After detach, face events should not change the state.
    await bus.publish(FaceDetected(x=0.5, y=0.5, confidence=0.8))
    await asyncio.sleep(0.05)
    # State should remain IDLE since the behavior is detached.
    assert sm.state == RobotState.IDLE


@pytest.mark.asyncio
async def test_centre_face_produces_centre_gaze() -> None:
    """A face at (0.5, 0.5) - the centre - should produce a near-zero gaze."""
    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    await sm.transition(RobotState.IDLE)

    behavior = PerceptionBehavior(bus=bus, state_machine=sm, idle_timeout_s=5.0)
    behavior.attach()

    look_events: list[LookRequested] = []
    bus.subscribe(LookRequested, look_events.append)

    await bus.publish(FaceDetected(x=0.5, y=0.5, confidence=0.9))
    await asyncio.sleep(0.05)

    assert len(look_events) >= 1
    # x=0.5 maps to gaze_x=0.0 (centre), y=0.5 maps to gaze_y=0.0
    assert abs(look_events[0].x) < 0.1
    assert abs(look_events[0].y) < 0.1
