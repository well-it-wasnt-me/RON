"""Integration test: reaction engine + executor + bus + state machine."""

from __future__ import annotations

import anyio

from robot.app import DeskBotApp
from robot.behavior.actions import RequestServoMoveAction
from robot.events.events import ServoMoved, WakeWordDetected
from tests.integration.conftest import make_test_settings


async def test_wake_word_triggers_servo_move_via_executor() -> None:
    app = DeskBotApp.build(make_test_settings())
    seen: list[object] = []
    app.bus.subscribe(ServoMoved, seen.append)
    async with app.run():
        for _ in range(5):
            await anyio.sleep(0)
        await app.bus.publish(WakeWordDetected(phrase="hey deskbot"))
        # Allow the reaction engine to fire and the executor to be called.
        for _ in range(10):
            await anyio.sleep(0)
        if app.executor is None or app.reactions is None:
            raise RuntimeError("executor or reactions not wired")
        # Drain the reactions outbox into the executor (the production app
        # would do this in a background loop; tests do it explicitly).
        queued = app.reactions.drain()
        await app.executor.execute(queued)
        # The wake-word reaction adds a RequestServoMoveAction for head_pan.
        assert any(isinstance(a, RequestServoMoveAction) for a in app.executor.executed)
        assert any(isinstance(e, ServoMoved) and e.name == "pan" for e in seen)
