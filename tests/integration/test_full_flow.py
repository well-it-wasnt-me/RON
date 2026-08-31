"""End-to-end behaviour tests that exercise the full app."""

from __future__ import annotations

import anyio

from robot.app import DeskBotApp
from robot.behavior.state_machine import RobotState
from robot.events.events import (
    EmotionChanged,
    EmotionName,
    SpeechRecognized,
    WakeWordDetected,
)
from tests.integration.conftest import make_test_settings


async def test_wake_word_drives_state_to_listening() -> None:
    app = DeskBotApp.build(make_test_settings())
    async with app.run():
        # drain any initial frames
        for _ in range(5):
            await anyio.sleep(0)
        await app.bus.publish(WakeWordDetected(phrase="hey deskbot"))
        # Let subscribers run
        for _ in range(5):
            await anyio.sleep(0)
        assert app.state_machine.state is RobotState.LISTENING


async def test_speech_reply_returns_to_idle() -> None:
    app = DeskBotApp.build(make_test_settings())
    async with app.run():
        for _ in range(5):
            await anyio.sleep(0)
        await app.bus.publish(WakeWordDetected(phrase="hey deskbot"))
        for _ in range(5):
            await anyio.sleep(0)
        await app.bus.publish(SpeechRecognized(text="hello there"))
        for _ in range(10):
            await anyio.sleep(0)
        assert app.state_machine.state is RobotState.IDLE


async def test_emotion_change_propagates() -> None:
    app = DeskBotApp.build(make_test_settings())
    async with app.run():
        for _ in range(5):
            await anyio.sleep(0)
        # The legacy eye engine was removed; FaceAnimator is the production face
        # path and subscribes to EmotionChanged. Capture the pre-event target,
        # then confirm the HAPPY event retargets the animator to the HAPPY face.
        # ``set_emotion`` sets ``_target`` synchronously (the run loop only
        # advances ``_current`` toward it), so this is a reliable signal.
        assert app.face_animator is not None
        animator = app.face_animator
        pre_target = animator._target

        await app.bus.publish(
            EmotionChanged(previous=EmotionName.NEUTRAL, current=EmotionName.HAPPY, intensity=1.0)
        )
        for _ in range(5):
            await anyio.sleep(0)

        # The emotion change reached the animator: it now targets the HAPPY
        # face. This is the new equivalent of the old
        # ``eye_animator.eye.emotion is EmotionName.HAPPY`` check.
        assert animator._target == animator.emotions.build(EmotionName.HAPPY.value)
        assert animator._target != pre_target
