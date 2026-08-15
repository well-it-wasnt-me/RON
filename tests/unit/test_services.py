"""Tests for higher-level services."""

from __future__ import annotations

from tests.fakes.servo import FakeServo, FakeServoBus, make_servo_controller_from_fakes

from robot.behavior.actions import (
    CelebrateAction,
    LookAroundAction,
    RequestBlinkAction,
    RequestLookAction,
    RequestServoMoveAction,
)
from robot.behavior.state_machine import RobotState, StateMachine
from robot.events.bus import InMemoryEventBus
from robot.events.events import (
    EmotionChanged,
    EmotionName,
    ServoMoved,
    SpeechRecognized,
    WakeWordDetected,
)
from robot.services.conversation_service import build_default_conversation
from robot.services.executor import ActionExecutor


async def test_executor_routes_blink() -> None:
    from robot.events.events import BlinkRequested

    bus = InMemoryEventBus()
    bus_servos = make_servo_controller_from_fakes(FakeServoBus())
    seen: list[object] = []
    bus.subscribe(BlinkRequested, seen.append)
    executor = ActionExecutor(bus=bus, servo_controller=bus_servos)
    await executor.execute_one(RequestBlinkAction())
    assert executor.executed and isinstance(executor.executed[0], RequestBlinkAction)
    assert seen and isinstance(seen[0], BlinkRequested)


async def test_executor_routes_look() -> None:
    bus = InMemoryEventBus()
    bus_servos = make_servo_controller_from_fakes(FakeServoBus())
    executor = ActionExecutor(bus=bus, servo_controller=bus_servos)
    await executor.execute_one(RequestLookAction(x=0.5, y=-0.2))
    assert executor.executed and executor.executed[0].name == "look"


async def test_executor_routes_servo() -> None:
    bus = InMemoryEventBus()
    raw = FakeServoBus()
    raw.add(FakeServo("head_pan"))
    bus_servos = make_servo_controller_from_fakes(raw)
    seen: list[ServoMoved] = []
    bus.subscribe(ServoMoved, seen.append)
    executor = ActionExecutor(bus=bus, servo_controller=bus_servos)
    await executor.execute_one(RequestServoMoveAction(servo="head_pan", angle=15.0))
    assert raw.get("head_pan").angle == 15.0
    assert seen and seen[0].name == "head_pan"


async def test_executor_routes_celebrate() -> None:
    bus = InMemoryEventBus()
    bus_servos = make_servo_controller_from_fakes(FakeServoBus())
    seen: list[EmotionChanged] = []
    bus.subscribe(EmotionChanged, seen.append)
    executor = ActionExecutor(bus=bus, servo_controller=bus_servos)
    await executor.execute_one(CelebrateAction())
    assert any(e.current is EmotionName.HAPPY for e in seen)


async def test_executor_routes_look_around() -> None:
    bus = InMemoryEventBus()
    bus_servos = make_servo_controller_from_fakes(FakeServoBus())
    executor = ActionExecutor(bus=bus, servo_controller=bus_servos)
    await executor.execute_one(LookAroundAction(points=3))
    assert executor.executed and executor.executed[0].name == "look_around"


async def test_conversation_service_listens_to_wake_word() -> None:
    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    sm._state = RobotState.IDLE
    service = build_default_conversation(bus=bus, state_machine=sm)
    service.attach()
    await bus.publish(WakeWordDetected(phrase="hey deskbot"))
    assert sm.state is RobotState.LISTENING
    service.detach()


async def test_conversation_service_handles_speech() -> None:
    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    sm._state = RobotState.LISTENING
    service = build_default_conversation(bus=bus, state_machine=sm)
    service.attach()
    await bus.publish(SpeechRecognized(text="hello"))
    assert sm.state is RobotState.IDLE  # back to IDLE after reply
    service.detach()
