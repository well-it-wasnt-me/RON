"""Tests for typed observations and the reward model.

Separate Observation, Action and Reward.

Tests prove:
- every action is valid (comes from ActionSpace)
- observations contain no future reward
- events are mapped to observations
- actions come only from ActionSpace
- reward is calculated after the outcome
- serialization round-trips correctly
"""

from __future__ import annotations

import pytest

from robot.behavior.state_machine import RobotState
from robot.events.events import (
    EmotionChanged,
    EmotionName,
    FaceDetected,
    IdleTimeout,
    ServoMoved,
    StateChanged,
)
from robot.learning.action_learning import ActionSpace, deskbot_action_space
from robot.learning.observation import (
    AudioObservation,
    Observation,
    RobotObservation,
    VisionObservation,
    event_to_observation_update,
)
from robot.learning.reward import (
    RewardModel,
)
from robot.learning.state_encoder import StateEncoder

# ========================================================================
# Observation types
# ========================================================================


class TestObservation:
    """Tests for the Observation dataclass and sub-types."""

    def test_observation_creation(self) -> None:
        """An Observation can be created with defaults."""
        obs = Observation()
        assert obs.robot.state == RobotState.IDLE
        assert obs.vision.features.face_detected == 0.0
        assert obs.audio.features.rms_energy == 0.0
        assert obs.timestamp_ns > 0

    def test_from_encoder(self) -> None:
        """Observation.from_encoder captures the encoder's current context."""
        enc = StateEncoder()
        enc.update_state(RobotState.CURIOUS)
        enc.update_emotion(EmotionName.HAPPY, 0.8)
        enc.update_vision(face_detected=True, face_x=0.3, face_y=0.7, face_confidence=0.9)

        obs = Observation.from_encoder(enc)
        assert obs.robot.state == RobotState.CURIOUS
        assert obs.robot.emotions.get("happy") == 0.8
        assert obs.vision.features.face_detected == 1.0
        assert obs.vision.features.face_x == 0.3

    def test_to_vector(self) -> None:
        """Observation.to_vector produces a deterministic flat vector."""
        enc = StateEncoder()
        enc.update_state(RobotState.IDLE)
        enc.update_emotion(EmotionName.HAPPY, 0.5)

        obs = Observation.from_encoder(enc)
        vec = obs.to_vector()
        assert len(vec) == 91  # STATE_SIZE
        assert vec[1] == 0.5  # happy emotion

    def test_immutable(self) -> None:
        """Observation is frozen and cannot be mutated."""
        obs = Observation()
        with pytest.raises(AttributeError):
            obs.robot = RobotObservation()  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        """Observation serialises and deserialises correctly."""
        enc = StateEncoder()
        enc.update_state(RobotState.CURIOUS)
        enc.update_emotion(EmotionName.HAPPY, 0.7)
        enc.update_vision(face_detected=True, face_x=0.4, face_y=0.6, face_confidence=0.8)

        obs = Observation.from_encoder(enc)
        d = obs.to_dict()
        assert d["robot"]["state"] == "curious"
        assert d["robot"]["emotions"]["happy"] == 0.7
        assert d["vision"]["face_detected"] == 1.0
        assert d["vision"]["face_x"] == 0.4
        assert d["audio"]["rms_energy"] == 0.0

    def test_observations_contain_no_future_reward(self) -> None:
        """An observation never contains the current/future transition reward.

        The recent_rewards tuple only contains PAST rewards from before
        this transition.  When creating an observation from the encoder
        before an action, the reward for the action being taken is not
        yet known and cannot be in the observation.
        """
        enc = StateEncoder()
        enc.recent_rewards = [0.1, 0.2, 0.3]  # past rewards only
        obs = Observation.from_encoder(enc)
        assert obs.robot.recent_rewards == (0.1, 0.2, 0.3)
        # The reward for the transition about to happen is NOT in here

    def test_vision_observation_no_face(self) -> None:
        """VisionObservation.no_face() produces a no-face observation."""
        v = VisionObservation.no_face()
        assert v.features.face_detected == 0.0

    def test_audio_observation_no_audio(self) -> None:
        """AudioObservation.no_audio() produces a silence observation."""
        a = AudioObservation.no_audio()
        assert a.features.rms_energy == 0.0


# ========================================================================
# Event → Observation mapping
# ========================================================================


class TestEventToObservation:
    """Events are observations, never actions."""

    def test_face_detected_updates_vision(self) -> None:
        """FaceDetected updates the vision observation."""
        obs = Observation()
        event = FaceDetected(x=0.3, y=0.7, confidence=0.9)
        updated = event_to_observation_update(event, obs)
        assert updated.vision.features.face_detected == 1.0
        assert updated.vision.features.face_x == 0.3
        assert updated.vision.features.face_confidence == 0.9

    def test_emotion_changed_updates_robot(self) -> None:
        """EmotionChanged updates the robot observation."""
        obs = Observation()
        event = EmotionChanged(
            previous=EmotionName.NEUTRAL, current=EmotionName.HAPPY, intensity=0.8
        )
        updated = event_to_observation_update(event, obs)
        assert updated.robot.emotions.get("happy") == 0.8

    def test_state_changed_updates_robot(self) -> None:
        """StateChanged updates the robot state observation."""
        obs = Observation()
        event = StateChanged(previous=RobotState.IDLE, current=RobotState.CURIOUS)
        updated = event_to_observation_update(event, obs)
        assert updated.robot.state == RobotState.CURIOUS

    def test_servo_moved_updates_robot(self) -> None:
        """ServoMoved updates the servo positions."""
        obs = Observation()
        event = ServoMoved(name="pan", angle=45.0)
        updated = event_to_observation_update(event, obs)
        assert updated.robot.servos.get("pan") == 45.0

    def test_idle_timeout_updates_robot(self) -> None:
        """IdleTimeout updates the idle seconds."""
        obs = Observation()
        event = IdleTimeout(seconds_idle=30.0)
        updated = event_to_observation_update(event, obs)
        assert updated.robot.idle_seconds == 30.0

    def test_speech_recognized_is_observation_not_action(self) -> None:
        """SpeechRecognized is an observation, not an action."""
        obs = Observation()
        event = __import__("robot.events.events", fromlist=["SpeechRecognized"]).SpeechRecognized(
            text="hello", confidence=0.9
        )
        updated = event_to_observation_update(event, obs)
        # Speech recognition should not create an action — it's an observation
        # The observation should still be valid
        assert updated is not None

    def test_observation_event_names_not_in_action_space(self) -> None:
        """Observation event names must never be action names."""
        space = deskbot_action_space()
        action_names = {a.name for a in space}
        observation_event_names = {
            "FaceDetected",
            "SpeechRecognized",
            "EmotionChanged",
            "IdleTimeout",
            "StateChanged",
            "ServoMoved",
        }
        assert observation_event_names.isdisjoint(action_names)


# ========================================================================
# Reward model
# ========================================================================


class TestRewardModel:
    """Tests for the RewardModel that computes reward after the outcome."""

    @pytest.fixture
    def action_space(self) -> ActionSpace:
        return deskbot_action_space()

    def test_reward_computed_after_outcome(self, action_space: ActionSpace) -> None:
        """Reward is computed from observation + action + next_observation."""
        model = RewardModel()
        obs = Observation()
        next_obs = Observation(
            vision=VisionObservation.from_face(x=0.5, y=0.5, confidence=0.9, face_count=1)
        )
        action = action_space.get(2)  # look_center
        reward = model.compute(obs, action, next_obs)
        # Should get positive reward for engaging with a face
        assert reward > 0.0

    def test_reward_for_sleep_with_face(self, action_space: ActionSpace) -> None:
        """Sleeping when a face is present gets a penalty."""
        model = RewardModel()
        obs = Observation()
        next_obs = Observation(
            vision=VisionObservation.from_face(x=0.5, y=0.5, confidence=0.9, face_count=1)
        )
        action = action_space.get_by_name("sleep")
        reward = model.compute(obs, action, next_obs)
        assert reward < 0.0  # penalized

    def test_reward_for_sleep_without_face(self, action_space: ActionSpace) -> None:
        """Sleeping when no face and no audio is rewarded (energy saving)."""
        model = RewardModel()
        obs = Observation()
        next_obs = Observation()  # no face, no audio
        action = action_space.get_by_name("sleep")
        reward = model.compute(obs, action, next_obs)
        assert reward > 0.0  # energy saving

    def test_reward_using_action_index(self, action_space: ActionSpace) -> None:
        """Reward can be computed using an action index."""
        model = RewardModel()
        obs = Observation()
        next_obs = Observation(
            vision=VisionObservation.from_face(x=0.5, y=0.5, confidence=0.9, face_count=1)
        )
        reward = model.compute_for_action_index(
            observation=obs,
            action_index=2,  # look_center
            next_observation=next_obs,
            action_space=action_space,
        )
        assert reward > 0.0

    def test_reward_clamped(self, action_space: ActionSpace) -> None:
        """Reward is clamped to max_abs_reward."""
        model = RewardModel(max_abs_reward=0.01)
        obs = Observation()
        next_obs = Observation(
            vision=VisionObservation.from_face(x=0.5, y=0.5, confidence=0.9, face_count=1)
        )
        action = action_space.get_by_name("celebrate")
        reward = model.compute(obs, action, next_obs)
        assert reward <= 0.01

    def test_default_reward_when_no_components(self, action_space: ActionSpace) -> None:
        """When no components configured, default_reward is used."""
        model = RewardModel(components=[], default_reward=0.42)
        obs = Observation()
        action = action_space.get(0)
        reward = model.compute(obs, action, obs)
        assert reward == 0.42

    def test_custom_reward_component(self, action_space: ActionSpace) -> None:
        """Custom reward components can be added."""
        model = RewardModel(components=[lambda ctx: 1.0])
        obs = Observation()
        action = action_space.get(0)
        reward = model.compute(obs, action, obs)
        assert reward == 1.0

    def test_reward_is_not_in_observation(self, action_space: ActionSpace) -> None:
        """The reward for the current transition is not in the observation.

        This tests the 'remove leakage' requirement: the observation
        captured before the action does not contain the reward that will
        be computed after the outcome.
        """
        model = RewardModel()
        obs = Observation()  # captured before action
        next_obs = Observation(
            vision=VisionObservation.from_face(x=0.5, y=0.5, confidence=0.9, face_count=1)
        )
        action = action_space.get(2)
        reward = model.compute(obs, action, next_obs)
        # The observation captured before the action does not know the reward
        assert reward not in obs.robot.recent_rewards
        assert reward not in obs.to_vector()


# ========================================================================
# Integration: transition with typed observations and reward model
# ========================================================================


class TestTransitionWithObservations:
    """The transition lifecycle uses typed observations and reward model."""

    def test_begin_observation_transition(self) -> None:
        """begin_observation_transition captures a typed observation."""
        from robot.events.bus import InMemoryEventBus
        from robot.learning.recorder import ExperienceRecorder

        bus = InMemoryEventBus()
        recorder = ExperienceRecorder(bus=bus)
        recorder.encoder.update_state(RobotState.CURIOUS)
        recorder.encoder.update_emotion(EmotionName.HAPPY, 0.7)

        pending = recorder.begin_observation_transition(action_index=2)
        assert pending.observation is not None
        assert pending.observation.robot.state == RobotState.CURIOUS
        assert pending.observation.robot.emotions.get("happy") == 0.7

    def test_complete_with_reward_model(self) -> None:
        """complete_transition with use_reward_model computes reward."""
        from robot.events.bus import InMemoryEventBus
        from robot.learning.recorder import ExperienceRecorder

        bus = InMemoryEventBus()
        recorder = ExperienceRecorder(bus=bus)
        recorder.encoder.update_state(RobotState.IDLE)

        pending = recorder.begin_observation_transition(action_index=2)
        # Simulate outcome: face appears
        recorder.encoder.update_vision(
            face_detected=True, face_x=0.5, face_y=0.5, face_confidence=0.9, face_count=1
        )
        transition = recorder.complete_transition(pending, reward=None, use_reward_model=True)
        assert transition.observation is not None
        assert transition.next_observation is not None
        assert transition.reward > 0.0  # positive for engaging with face

    def test_transition_can_answer_four_questions(self) -> None:
        """A stored transition can answer: what did robot know/do/happen/reward."""
        from robot.events.bus import InMemoryEventBus
        from robot.learning.recorder import ExperienceRecorder

        bus = InMemoryEventBus()
        recorder = ExperienceRecorder(bus=bus)
        recorder.encoder.update_state(RobotState.IDLE)
        recorder.encoder.update_vision(
            face_detected=True, face_x=0.3, face_y=0.7, face_confidence=0.85, face_count=1
        )

        pending = recorder.begin_observation_transition(action_index=7)  # celebrate

        # Simulate outcome
        recorder.encoder.update_emotion(EmotionName.HAPPY, 0.9)
        transition = recorder.complete_transition(pending, reward=1.0)

        # 1. What did the robot know?
        assert transition.observation is not None
        assert transition.observation.vision.features.face_detected == 1.0
        assert transition.observation.robot.state == RobotState.IDLE

        # 2. What did it do?
        assert transition.action_name == "celebrate"
        assert transition.action_index == 7

        # 3. What happened?
        assert transition.next_observation is not None
        assert transition.next_observation.robot.emotions.get("happy") == 0.9

        # 4. What reward did it receive?
        assert transition.reward == 1.0
