"""Tests for the SoundReactor (emotion/state -> sound effect mapping)."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot.behavior.sound_reactor import EMOTION_SOUNDS, STATE_SOUNDS, SoundReactor
from robot.behavior.state_machine import RobotState
from robot.events.bus import InMemoryEventBus
from robot.events.events import EmotionChanged, EmotionName, StateChanged
from robot.speech.sound_effects import SoundEffectsPlayer


class _FakePlayer:
    """Records play() calls and pretends every sound exists."""

    def __init__(self) -> None:
        self.played: list[str] = []

    def has_sound(self, name: str) -> bool:
        return True

    def list_sounds(self) -> list[str]:
        return ["angry", "cute", "confused", "surprise", "thinking", "very-cute", "talk"]

    async def play(self, name: str) -> bool:
        self.played.append(name)
        return True


@pytest.fixture
def bus() -> InMemoryEventBus:
    return InMemoryEventBus()


@pytest.fixture
def fake_player() -> _FakePlayer:
    return _FakePlayer()


@pytest.fixture
def reactor(bus: InMemoryEventBus, fake_player: _FakePlayer) -> SoundReactor:
    return SoundReactor(bus=bus, sound_effects=fake_player, enabled=True)  # type: ignore[arg-type]


async def test_attach_subscribes_and_plays(
    bus: InMemoryEventBus, reactor: SoundReactor, fake_player: _FakePlayer
) -> None:
    reactor.attach()
    await bus.publish(EmotionChanged(previous=EmotionName.NEUTRAL, current=EmotionName.HAPPY))
    assert fake_player.played == ["cute"]
    reactor.detach()
    await bus.publish(EmotionChanged(previous=EmotionName.HAPPY, current=EmotionName.ANGRY))
    assert fake_player.played == ["cute"]


async def test_disabled_does_not_attach(bus: InMemoryEventBus, fake_player: _FakePlayer) -> None:
    r = SoundReactor(bus=bus, sound_effects=fake_player, enabled=False)  # type: ignore[arg-type]
    r.attach()
    await bus.publish(EmotionChanged(previous=EmotionName.NEUTRAL, current=EmotionName.HAPPY))
    assert fake_player.played == []


async def test_emotion_mapping_plays_expected_name(
    bus: InMemoryEventBus, reactor: SoundReactor, fake_player: _FakePlayer
) -> None:
    reactor.attach()
    await bus.publish(EmotionChanged(previous=EmotionName.NEUTRAL, current=EmotionName.HAPPY))
    assert fake_player.played == ["cute"]
    await bus.publish(EmotionChanged(previous=EmotionName.HAPPY, current=EmotionName.ANGRY))
    assert fake_player.played == ["cute", "angry"]


async def test_state_mapping_plays_thinking(
    bus: InMemoryEventBus, reactor: SoundReactor, fake_player: _FakePlayer
) -> None:
    reactor.attach()
    await bus.publish(StateChanged(previous=RobotState.LISTENING, current=RobotState.THINKING))
    assert fake_player.played == ["thinking"]


async def test_unmapped_emotion_is_silent(
    bus: InMemoryEventBus, reactor: SoundReactor, fake_player: _FakePlayer
) -> None:
    reactor.attach()
    await bus.publish(EmotionChanged(previous=EmotionName.NEUTRAL, current=EmotionName.SLEEPY))
    assert fake_player.played == []


async def test_play_skips_missing_sound() -> None:
    # A real player with a nonexistent sounds dir -> has_sound() is False, so
    # _play must be a no-op and never raise.
    empty = SoundEffectsPlayer(audio=None, enabled=True, sounds_dir=Path("/nonexistent-sounds"))
    bus = InMemoryEventBus()
    reactor = SoundReactor(bus=bus, sound_effects=empty, enabled=True)
    await reactor._play("angry")


def test_every_mapped_sound_is_available_in_assets() -> None:
    """The emotion/state -> sound names must exist in assets/sounds."""
    player = SoundEffectsPlayer(audio=None, enabled=True, sounds_dir=Path("assets/sounds"))
    for name in {*EMOTION_SOUNDS.values(), *STATE_SOUNDS.values()}:
        assert player.has_sound(name), f"missing sound {name!r} in assets/sounds"
