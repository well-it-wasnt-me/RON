"""Smoke test for the application bootstrap."""

from __future__ import annotations

import anyio

from robot.app import DeskBotApp
from robot.behavior.state_machine import RobotState


async def test_app_builds_and_starts() -> None:
    from robot.config import load_settings

    settings = load_settings()

    # Disable components that require external hardware/models.
    settings.memory.enabled = False
    settings.vector_memory.enabled = False
    settings.conversation.store = "memory"
    settings.preferences.store = "memory"
    settings.learning.enabled = False
    settings.api.enabled = False
    settings.perception.enabled = False
    settings.sounds.enabled = False

    app = DeskBotApp.build(settings)

    async with app.run():
        # Give the application lifecycle a few scheduling opportunities
        # to complete startup.
        for _ in range(5):
            await anyio.sleep(0)

        assert app.state_machine.state is RobotState.IDLE
        assert app.bus is not None
