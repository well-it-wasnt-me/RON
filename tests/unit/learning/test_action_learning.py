"""Tests for action learning: ActionSpace, ActionLearner, Reward, ActionLearningEnv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot.learning.action_learning import (
    ActionLearner,
    ActionLearningEnv,
    ActionSpace,
    DefaultValidator,
    LearningAction,
    Reward,
    deskbot_action_space,
)

# ========================================================================
# LearningAction
# ========================================================================


class TestLearningAction:
    """Tests for LearningAction dataclass."""

    def test_creation(self) -> None:
        action = LearningAction(index=0, name="look_left", description="Look left")
        assert action.index == 0
        assert action.name == "look_left"
        assert action.description == "Look left"
        assert action.action_type == ""
        assert action.params == {}

    def test_frozen(self) -> None:
        action = LearningAction(index=0, name="blink")
        with pytest.raises(AttributeError):
            action.name = "wave"  # type: ignore[misc]

    def test_with_params(self) -> None:
        action = LearningAction(
            index=3,
            name="servo_pan",
            action_type="servo",
            params={"servo": "pan", "angle": 90.0},
        )
        assert action.params["servo"] == "pan"
        assert action.params["angle"] == 90.0


# ========================================================================
# ActionSpace
# ========================================================================


class TestActionSpace:
    """Tests for ActionSpace registry."""

    def test_register_and_get(self) -> None:
        space = ActionSpace()
        space.register("look_left", description="Look left", action_type="look")
        space.register("look_right", description="Look right", action_type="look")
        assert space.size == 2
        assert space.get(0).name == "look_left"
        assert space.get(1).name == "look_right"

    def test_get_by_name(self) -> None:
        space = ActionSpace()
        space.register("blink")
        action = space.get_by_name("blink")
        assert action.name == "blink"
        assert action.index == 0

    def test_duplicate_registration(self) -> None:
        space = ActionSpace()
        space.register("blink")
        with pytest.raises(ValueError, match="already registered"):
            space.register("blink")

    def test_get_invalid_index(self) -> None:
        space = ActionSpace()
        space.register("blink")
        with pytest.raises(IndexError):
            space.get(99)

    def test_get_invalid_name(self) -> None:
        space = ActionSpace()
        space.register("blink")
        with pytest.raises(KeyError):
            space.get_by_name("nonexistent")

    def test_action_vector(self) -> None:
        space = ActionSpace()
        space.register("a")
        space.register("b")
        space.register("c")
        vec = space.action_vector(0)
        assert vec.shape == (3,)
        assert vec[0] == 1.0
        assert vec[1] == 0.0
        vec = space.action_vector(2)
        assert vec[2] == 1.0

    def test_action_vector_from_name(self) -> None:
        space = ActionSpace()
        space.register("a")
        space.register("b")
        vec = space.action_vector_from_name("b")
        assert vec[1] == 1.0

    def test_iteration(self) -> None:
        space = ActionSpace()
        space.register("a")
        space.register("b")
        actions = list(space)
        assert len(actions) == 2

    def test_len(self) -> None:
        space = ActionSpace()
        assert len(space) == 0
        space.register("a")
        assert len(space) == 1


class TestDeskbotActionSpace:
    """Tests for the default DeskBot action space."""

    def test_has_expected_actions(self) -> None:
        space = deskbot_action_space()
        assert space.size == 10
        names = [a.name for a in space]
        assert "look_left" in names
        assert "look_right" in names
        assert "look_center" in names
        assert "blink" in names
        assert "celebrate" in names
        assert "sleep" in names
        assert "look_around" in names

    def test_action_vector_size(self) -> None:
        space = deskbot_action_space()
        vec = space.action_vector(0)
        assert vec.shape == (10,)


# ========================================================================
# Reward
# ========================================================================


class TestReward:
    """Tests for Reward dataclass."""

    def test_positive(self) -> None:
        r = Reward(value=1.0, description="Good", source="face")
        assert r.is_positive
        assert not r.is_negative
        assert not r.is_neutral

    def test_negative(self) -> None:
        r = Reward(value=-0.5, description="Bad", source="idle")
        assert r.is_negative
        assert not r.is_positive

    def test_neutral(self) -> None:
        r = Reward(value=0.0)
        assert r.is_neutral

    def test_constants(self) -> None:
        from robot.learning.action_learning import REWARD_NEGATIVE, REWARD_NEUTRAL, REWARD_POSITIVE

        assert REWARD_POSITIVE.value == 1.0
        assert REWARD_NEGATIVE.value == -1.0
        assert REWARD_NEUTRAL.value == 0.0


# ========================================================================
# DefaultValidator
# ========================================================================


class TestDefaultValidator:
    """Tests for the default action validator."""

    def test_allows_valid_indices(self) -> None:
        space = deskbot_action_space()
        validator = DefaultValidator()
        state = np.zeros(4)
        for i in range(space.size):
            assert validator.is_valid(i, state, space)

    def test_rejects_invalid_indices(self) -> None:
        space = deskbot_action_space()
        validator = DefaultValidator()
        state = np.zeros(4)
        assert not validator.is_valid(-1, state, space)
        assert not validator.is_valid(999, state, space)


# ========================================================================
# ActionLearner
# ========================================================================


class TestActionLearner:
    """Tests for the Q-learning action learner."""

    def test_creation(self) -> None:
        space = deskbot_action_space()
        learner = ActionLearner(action_space=space, state_size=4, seed=42)
        assert learner.action_space.size == 10
        assert learner.param_count() > 0

    def test_q_values_shape(self) -> None:
        space = deskbot_action_space()
        learner = ActionLearner(action_space=space, state_size=4, seed=42)
        state = np.zeros(4)
        q_vals = learner.q_values(state)
        assert q_vals.shape == (space.size,)

    def test_q_value_for_single_action(self) -> None:
        space = deskbot_action_space()
        learner = ActionLearner(action_space=space, state_size=4, seed=42)
        state = np.zeros(4)
        q_val = learner.q_value(state, 0)
        assert isinstance(q_val, float)

    def test_select_action_returns_valid_index(self) -> None:
        space = deskbot_action_space()
        learner = ActionLearner(action_space=space, state_size=4, seed=42)
        state = np.zeros(4)
        for _ in range(20):
            action = learner.select_action(state)
            assert 0 <= action < space.size

    def test_greedy_action(self) -> None:
        space = deskbot_action_space()
        learner = ActionLearner(action_space=space, state_size=4, seed=42, epsilon_start=0.0)
        state = np.zeros(4)
        action = learner.select_action(state)
        assert 0 <= action < space.size

    def test_epsilon_decay(self) -> None:
        space = deskbot_action_space()
        learner = ActionLearner(
            action_space=space,
            state_size=4,
            seed=42,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.9,
        )
        initial = learner.epsilon
        learner.decay_epsilon()
        assert learner.epsilon < initial
        for _ in range(100):
            learner.decay_epsilon()
        assert abs(learner.epsilon - learner.epsilon_end) < 0.01

    def test_train_step(self) -> None:
        space = deskbot_action_space()
        learner = ActionLearner(action_space=space, state_size=4, seed=42)
        state = np.array([0.5, 0.5, 1.0, 0.0])
        next_state = np.array([0.5, 0.5, 1.0, 0.0])
        loss = learner.train_step(state, action_index=0, reward=0.5, next_state=next_state)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_train_batch(self) -> None:
        space = deskbot_action_space()
        learner = ActionLearner(action_space=space, state_size=4, seed=42)
        batch_size = 8
        states = np.random.randn(batch_size, 4)
        actions = list(np.random.randint(0, space.size, batch_size))
        rewards = np.random.randn(batch_size)
        next_states = np.random.randn(batch_size, 4)
        dones = np.zeros(batch_size, dtype=bool)
        loss = learner.train_batch(states, actions, rewards, next_states, dones)
        assert isinstance(loss, float)

    def test_reset_epsilon(self) -> None:
        space = deskbot_action_space()
        learner = ActionLearner(action_space=space, state_size=4, seed=42, epsilon_start=0.5)
        for _ in range(20):
            learner.decay_epsilon()
        assert learner.epsilon < 0.5
        learner.reset_epsilon()
        assert learner.epsilon == 0.5

    def test_save_and_load(self, tmp_path: Path) -> None:
        space = deskbot_action_space()
        learner = ActionLearner(action_space=space, state_size=4, seed=42)
        state = np.array([0.5, 0.5, 1.0, 0.0])
        for _ in range(10):
            learner.train_step(state, 0, 0.5, state)
        q_before = learner.q_values(state)
        path = tmp_path / "q_network.json"
        learner.save(str(path))
        learner2 = ActionLearner(action_space=space, state_size=4, seed=99)
        learner2.load(str(path))
        q_after = learner2.q_values(state)
        np.testing.assert_array_almost_equal(q_before, q_after, decimal=6)

    def test_validator_blocks_action(self) -> None:
        """A custom validator can block specific actions."""
        space = deskbot_action_space()
        learner = ActionLearner(action_space=space, state_size=4, seed=42, epsilon_start=0.0)

        class BlockSleepValidator:
            def is_valid(
                self, action_index: int, state: np.ndarray, action_space: ActionSpace
            ) -> bool:
                action = action_space.get(action_index)
                return action.name != "sleep"

        state = np.array([0.5, 0.5, 1.0, 0.0])
        # Train to make sleep look bad so greedy avoids it anyway
        for _ in range(50):
            learner.train_step(state, space.get_by_name("sleep").index, -1.0, state)
            learner.train_step(state, space.get_by_name("celebrate").index, 1.0, state)

        for _ in range(20):
            action_idx = learner.select_action(state, validator=BlockSleepValidator())
            assert space.get(action_idx).name != "sleep"


# ========================================================================
# ActionLearningEnv
# ========================================================================


class TestActionLearningEnv:
    """Tests for the simulation environment."""

    def test_reset(self) -> None:
        env = ActionLearningEnv(seed=42)
        state = env.reset()
        assert state.shape == (4,)
        assert state[2] == 1.0

    def test_look_center_reward(self) -> None:
        env = ActionLearningEnv(seed=42, noise_std=0.0)
        env.reset()
        _, reward, _ = env.step(env.action_space.get_by_name("look_center").index)
        assert reward > 0

    def test_celebrate_reward(self) -> None:
        env = ActionLearningEnv(seed=42, noise_std=0.0)
        env.reset()
        _, reward, _ = env.step(env.action_space.get_by_name("celebrate").index)
        assert reward == 1.0

    def test_celebrate_no_face_penalty(self) -> None:
        env = ActionLearningEnv(seed=42, noise_std=0.0)
        state = env.reset()
        state[2] = 0.0
        env._state = state.copy()
        _, reward, _ = env.step(env.action_space.get_by_name("celebrate").index)
        assert reward < 0

    def test_sleep_penalty(self) -> None:
        env = ActionLearningEnv(seed=42, noise_std=0.0)
        env.reset()
        _, reward, _ = env.step(env.action_space.get_by_name("sleep").index)
        assert reward < 0

    def test_deterministic_with_seed(self) -> None:
        env1 = ActionLearningEnv(seed=42, noise_std=0.0)
        env2 = ActionLearningEnv(seed=42, noise_std=0.0)
        env1.reset()
        env2.reset()
        for _ in range(10):
            a = env1.action_space.get_by_name("look_right").index
            s1, r1, _d1 = env1.step(a)
            s2, r2, _d2 = env2.step(a)
            np.testing.assert_array_almost_equal(s1, s2)
            assert r1 == r2

    def test_done_flag(self) -> None:
        env = ActionLearningEnv(seed=42)
        env.reset()
        for _step in range(199):
            _, _, done = env.step(0)
            assert not done
        _, _, done = env.step(0)
        assert done

    def test_boundary_clipping(self) -> None:
        env = ActionLearningEnv(seed=42, noise_std=0.0)
        env.reset()
        look_right = env.action_space.get_by_name("look_right").index
        for _ in range(20):
            state, _, _ = env.step(look_right)
            assert 0.0 <= state[0] <= 1.0


# ========================================================================
# Acceptance tests (Phase 5 spec)
# ========================================================================


class TestActionLearningAcceptance:
    """Acceptance tests matching the Phase 5 spec criteria.

    1. The robot has several possible actions.
    2. Some actions produce better rewards.
    3. The model initially chooses poorly.
    4. After training, it chooses higher-value actions more frequently.
    5. Training results are reproducible.
    """

    def test_robot_has_several_actions(self) -> None:
        """Criterion 1: The robot has several possible actions."""
        space = deskbot_action_space()
        assert space.size >= 5

    def test_some_actions_produce_better_rewards(self) -> None:
        """Criterion 2: Some actions produce better rewards than others."""
        env = ActionLearningEnv(seed=42, noise_std=0.0)
        state = env.reset()
        celebrate_reward = env.reward_for_action(
            env.action_space.get_by_name("celebrate").index, state
        )
        blink_reward = env.reward_for_action(env.action_space.get_by_name("blink").index, state)
        sleep_reward = env.reward_for_action(env.action_space.get_by_name("sleep").index, state)
        assert celebrate_reward > blink_reward
        assert celebrate_reward > sleep_reward

    def test_initially_chooses_poorly(self) -> None:
        """Criterion 3: The model initially chooses poorly."""
        space = deskbot_action_space()
        learner = ActionLearner(action_space=space, state_size=4, seed=42, epsilon_start=0.0)
        env = ActionLearningEnv(seed=42, noise_std=0.0)
        state = env.reset()
        celebrate_idx = space.get_by_name("celebrate").index
        selections = {}  # type: ignore[var-annotated]
        for _ in range(50):
            action = learner.select_action(state)
            selections[action] = selections.get(action, 0) + 1
        celebrate_pct = selections.get(celebrate_idx, 0) / 50.0
        # Untrained model should not consistently pick the best action
        assert celebrate_pct < 0.7

    def test_trained_chooses_higher_value_actions(self) -> None:
        """Criterion 4: After training, the model chooses higher-value actions."""
        space = deskbot_action_space()
        learner = ActionLearner(
            action_space=space,
            state_size=4,
            seed=42,
            hidden_sizes=[64, 32],
            learning_rate=0.01,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            gamma=0.95,
        )
        env = ActionLearningEnv(seed=42, noise_std=0.0)

        # Train for many episodes
        for episode in range(20):
            state = env.reset()
            learner._epsilon = max(0.3, 1.0 - episode * 0.04)  # Linear anneal
            for _step in range(200):
                action = learner.select_action(state)
                next_state, reward, done = env.step(action)
                learner.train_step(state, action, reward, next_state, done=done)
                state = next_state
                if done:
                    break

        # After training, the model should prefer "celebrate" and "look_center"
        # over "sleep" when face is detected
        learner._epsilon = 0.0  # Pure greedy
        state = np.array([0.5, 0.5, 1.0, 0.0])  # Face detected

        selections = {}  # type: ignore[var-annotated]
        n_trials = 100
        for _ in range(n_trials):
            # Small state perturbations
            test_state = state + np.random.normal(0, 0.02, 4)
            action = learner.greedy_action(test_state)
            selections[action] = selections.get(action, 0) + 1

        celebrate_idx = space.get_by_name("celebrate").index
        sleep_idx = space.get_by_name("sleep").index
        look_center_idx = space.get_by_name("look_center").index
        good_indices = {celebrate_idx, look_center_idx}
        bad_indices = {sleep_idx}

        good_pct = sum(selections.get(i, 0) for i in good_indices) / n_trials
        bad_pct = sum(selections.get(i, 0) for i in bad_indices) / n_trials

        assert good_pct > bad_pct, (
            f"After training, good actions ({good_pct:.0%}) should be "
            f"preferred over bad actions ({bad_pct:.0%})"
        )

    def test_training_reproducible(self) -> None:
        """Criterion 5: Training results are reproducible."""
        rewards_runs = []
        for run_seed in [42, 42]:
            space = deskbot_action_space()
            learner = ActionLearner(
                action_space=space,
                seed=run_seed,
                state_size=4,
                hidden_sizes=[64, 32],
                learning_rate=0.01,
                epsilon_start=0.3,
                epsilon_end=0.01,
                epsilon_decay=0.99,
            )
            env = ActionLearningEnv(seed=run_seed, noise_std=0.0)

            total_reward = 0.0
            state = env.reset()
            for _step in range(50):
                action = learner.select_action(state)
                next_state, reward, done = env.step(action)
                learner.train_step(state, action, reward, next_state, done=done)
                total_reward += reward
                state = next_state
                if done:
                    break
            rewards_runs.append(total_reward)

        assert rewards_runs[0] == rewards_runs[1], (
            f"Same seed should produce same results: {rewards_runs[0]} vs {rewards_runs[1]}"
        )
