"""Tests for the WorldModel and SimpleEnvironment."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from robot.learning.experience import Experience
from robot.learning.state_encoder import STATE_SIZE
from robot.learning.world_model import (
    DEFAULT_ACTION_SIZE,
    SimpleEnvironment,
    WorldModel,
)

# ========================================================================
# SimpleEnvironment
# ========================================================================


class TestSimpleEnvironment:
    """Tests for the simple simulation environment."""

    def test_reset(self) -> None:
        env = SimpleEnvironment(seed=42)
        state = env.reset()
        assert state.shape == (4,)
        assert state[0] == pytest.approx(0.5)  # face_x starts at centre
        assert state[1] == pytest.approx(0.5)  # face_y starts at centre
        assert state[2] == 1.0  # face detected
        assert state[3] == 0.0  # idle time

    def test_look_left(self) -> None:
        env = SimpleEnvironment(seed=42, step_size=0.1, noise_std=0.0)
        env.reset()
        state, _reward = env.step(0)  # look left
        assert state[0] < 0.5, "Face should move left"

    def test_look_right(self) -> None:
        env = SimpleEnvironment(seed=42, step_size=0.1, noise_std=0.0)
        env.reset()
        state, _reward = env.step(1)  # look right
        assert state[0] > 0.5, "Face should move right"

    def test_deterministic_with_seed(self) -> None:
        """Same seed should produce same trajectories."""
        env1 = SimpleEnvironment(seed=42, noise_std=0.0)
        env2 = SimpleEnvironment(seed=42, noise_std=0.0)
        s1 = env1.reset()
        s2 = env2.reset()
        for _ in range(10):
            action = 1  # always look right
            s1, _r1 = env1.step(action)
            s2, _r2 = env2.step(action)
            np.testing.assert_array_almost_equal(s1, s2)

    def test_boundary_clipping(self) -> None:
        """Face position should stay in [0, 1]."""
        env = SimpleEnvironment(seed=42, step_size=0.3, noise_std=0.0)
        env.reset()
        # Move right many times
        for _ in range(20):
            state, _ = env.step(1)
            assert 0.0 <= state[0] <= 1.0, f"Face x out of bounds: {state[0]}"
        # Move left many times
        for _ in range(40):
            state, _ = env.step(0)
            assert 0.0 <= state[0] <= 1.0, f"Face x out of bounds: {state[0]}"

    def test_collect_experiences(self) -> None:
        """collect_experiences should return valid Experience tuples."""
        env = SimpleEnvironment(seed=42)
        experiences = env.collect_experiences(n_steps=50)
        assert len(experiences) == 50
        for exp in experiences:
            assert isinstance(exp, Experience)
            assert len(exp.state) == STATE_SIZE
            assert len(exp.action) == DEFAULT_ACTION_SIZE
            assert len(exp.next_state) == STATE_SIZE

    def test_action_onehot(self) -> None:
        env = SimpleEnvironment(seed=42)
        left = env.action_onehot(0)
        right = env.action_onehot(1)
        assert left[0] == 1.0 and left[1] == 0.0
        assert right[0] == 0.0 and right[1] == 1.0

    def test_reward_structure(self) -> None:
        """Rewards should be higher near centre."""
        env = SimpleEnvironment(seed=42, step_size=0.1, noise_std=0.0)
        env.reset()
        _, reward_centre = env.step(1)  # still near centre
        # Move far from centre
        for _ in range(10):
            env.step(1)  # keep going right
        _, reward_far = env.step(1)
        # Centre reward should be higher (less negative)
        assert reward_centre >= reward_far


# ========================================================================
# WorldModel
# ========================================================================


class TestWorldModel:
    """Tests for the WorldModel."""

    def test_creation(self) -> None:
        wm = WorldModel(state_size=STATE_SIZE, seed=42)
        assert wm.state_size == STATE_SIZE
        assert wm.action_size == DEFAULT_ACTION_SIZE
        assert wm.param_count() > 0

    def test_predict_shape(self) -> None:
        wm = WorldModel(state_size=STATE_SIZE, seed=42)
        state = np.zeros(STATE_SIZE)
        action = np.zeros(DEFAULT_ACTION_SIZE)
        pred = wm.predict(state.tolist(), action.tolist())
        assert pred.shape == (STATE_SIZE,)

    def test_predict_batch_shape(self) -> None:
        wm = WorldModel(state_size=STATE_SIZE, seed=42)
        batch_size = 8
        states = np.zeros((batch_size, STATE_SIZE))
        actions = np.zeros((batch_size, DEFAULT_ACTION_SIZE))
        preds = wm.predict_batch(states, actions)
        assert preds.shape == (batch_size, STATE_SIZE)

    def test_predict_no_nan(self) -> None:
        wm = WorldModel(state_size=STATE_SIZE, seed=42)
        state = np.random.randn(STATE_SIZE)
        action = np.zeros(DEFAULT_ACTION_SIZE)
        pred = wm.predict(state.tolist(), action.tolist())
        assert not np.any(np.isnan(pred)), "Prediction contains NaN"
        assert not np.any(np.isinf(pred)), "Prediction contains inf"

    def test_train_epoch(self) -> None:
        """One training epoch should run without error."""
        wm = WorldModel(state_size=STATE_SIZE, seed=42)
        env = SimpleEnvironment(seed=42)
        experiences = env.collect_experiences(n_steps=64)
        result = wm.train(experiences, epochs=1, batch_size=32, verbose=False)
        assert result.epochs == 1
        assert len(result.metrics) == 1
        assert result.metrics[0].train_loss > 0.0

    def test_training_reduces_loss(self) -> None:
        """Training should reduce loss on the training data."""
        wm = WorldModel(state_size=STATE_SIZE, seed=42, hidden_sizes=[64, 32])
        env = SimpleEnvironment(seed=42, noise_std=0.005)
        experiences = env.collect_experiences(n_steps=200)
        result = wm.train(experiences, epochs=50, batch_size=32, verbose=False)
        assert result.improved, (
            f"Training should improve loss: initial={result.initial_loss:.6f}, "
            f"final={result.final_loss:.6f}"
        )

    def test_train_with_validation(self) -> None:
        """Training with a validation set should track val_loss."""
        wm = WorldModel(state_size=STATE_SIZE, seed=42)
        env = SimpleEnvironment(seed=42, noise_std=0.005)
        all_exp = env.collect_experiences(n_steps=200)
        # Split into train/val
        train_exp = all_exp[:150]
        val_exp = all_exp[150:]
        result = wm.train(train_exp, val_experiences=val_exp, epochs=5, verbose=False)
        assert result.metrics[0].val_loss > 0.0

    def test_evaluate(self) -> None:
        """evaluate() should return a positive loss."""
        wm = WorldModel(state_size=STATE_SIZE, seed=42)
        env = SimpleEnvironment(seed=42)
        experiences = env.collect_experiences(n_steps=50)
        loss = wm.evaluate(experiences)
        assert loss > 0.0

    def test_evaluate_empty(self) -> None:
        """evaluate() on empty data should return 0.0."""
        wm = WorldModel(state_size=STATE_SIZE, seed=42)
        loss = wm.evaluate([])
        assert loss == 0.0

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Model should be saveable and loadable."""
        wm = WorldModel(state_size=STATE_SIZE, seed=42)
        env = SimpleEnvironment(seed=42, noise_std=0.005)
        experiences = env.collect_experiences(n_steps=100)
        wm.train(experiences, epochs=10, batch_size=32, verbose=False)

        model_path = tmp_path / "world_model.json"
        wm.save(model_path)

        # Create new model and load
        wm2 = WorldModel(state_size=STATE_SIZE, seed=99)
        wm2.load(model_path)

        # Both should produce the same prediction
        state = np.zeros(STATE_SIZE)
        action = np.zeros(DEFAULT_ACTION_SIZE)
        pred1 = wm.predict(state.tolist(), action.tolist())
        pred2 = wm2.predict(state.tolist(), action.tolist())
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=6)

    def test_experiences_to_arrays_padding(self) -> None:
        """Short actions should be padded to action_size."""
        wm = WorldModel(state_size=STATE_SIZE, action_size=20, seed=42)
        # Create an experience with a short action
        exp = Experience(
            timestamp=datetime.now(tz=UTC),
            state=[0.0] * STATE_SIZE,
            action=[1.0, 0.0],  # 2-element action, should be padded to 20
            reward=0.5,
            next_state=[0.0] * STATE_SIZE,
        )
        _states, actions, _next_states = wm._experiences_to_arrays([exp])
        assert actions.shape == (1, 20)
        assert actions[0, 0] == 1.0
        assert actions[0, 1] == 0.0
        assert actions[0, 2] == 0.0  # padded


# ========================================================================
# Acceptance tests (from Phase 4 spec)
# ========================================================================


class TestWorldModelAcceptance:
    """Acceptance tests matching the Phase 4 spec criteria.

    Verify:
    1. Untrained model produces poor predictions
    2. Training decreases prediction error
    3. Trained model produces better predictions
    """

    def test_untrained_model_poor_predictions(self) -> None:
        """An untrained model should have high prediction error."""
        wm = WorldModel(state_size=STATE_SIZE, seed=42)
        env = SimpleEnvironment(seed=42, step_size=0.1, noise_std=0.005)
        experiences = env.collect_experiences(n_steps=200)

        # Evaluate untrained model
        initial_loss = wm.evaluate(experiences)
        assert initial_loss > 0.01, (
            f"Untrained model should have noticeable error, got {initial_loss:.6f}"
        )

    def test_training_decreases_prediction_error(self) -> None:
        """Training should decrease prediction error over time."""
        wm = WorldModel(state_size=STATE_SIZE, seed=42, hidden_sizes=[128, 64])
        env = SimpleEnvironment(seed=42, step_size=0.1, noise_std=0.005)
        experiences = env.collect_experiences(n_steps=300)

        # Split: train on 80%, evaluate on 20%
        n_val = max(1, int(len(experiences) * 0.2))
        train_exp = experiences[n_val:]
        val_exp = experiences[:n_val]

        # Record initial loss
        initial_val_loss = wm.evaluate(val_exp)

        # Train
        wm.train(train_exp, val_experiences=val_exp, epochs=80, batch_size=32, verbose=False)

        # Evaluate after training
        final_val_loss = wm.evaluate(val_exp)

        assert final_val_loss < initial_val_loss, (
            f"Training should decrease validation loss: "
            f"initial={initial_val_loss:.6f}, final={final_val_loss:.6f}"
        )

    def test_trained_model_better_than_untrained(self) -> None:
        """A trained model should produce better predictions than untrained."""
        env = SimpleEnvironment(seed=42, step_size=0.1, noise_std=0.005)

        # Collect training data
        train_exp = env.collect_experiences(n_steps=400)
        # Collect fresh test data (different from training)
        test_exp = env.collect_experiences(n_steps=100)

        # Untrained model
        wm_untrained = WorldModel(state_size=STATE_SIZE, seed=42, hidden_sizes=[128, 64])
        untrained_loss = wm_untrained.evaluate(test_exp)

        # Trained model
        wm_trained = WorldModel(state_size=STATE_SIZE, seed=42, hidden_sizes=[128, 64])
        wm_trained.train(train_exp, epochs=100, batch_size=32, verbose=False)
        trained_loss = wm_trained.evaluate(test_exp)

        # Trained model should have lower loss on test data
        assert trained_loss < untrained_loss, (
            f"Trained model should be better: untrained={untrained_loss:.6f}, "
            f"trained={trained_loss:.6f}"
        )

    def test_loss_curve_decreases(self) -> None:
        """Loss should generally decrease over training epochs."""
        wm = WorldModel(state_size=STATE_SIZE, seed=42, hidden_sizes=[64, 32])
        env = SimpleEnvironment(seed=42, step_size=0.1, noise_std=0.005)
        experiences = env.collect_experiences(n_steps=200)

        result = wm.train(experiences, epochs=50, batch_size=32, verbose=False)
        assert len(result.metrics) == 50

        # The average loss of the last 10 epochs should be lower than
        # the average loss of the first 10 epochs
        first_10_avg = sum(m.train_loss for m in result.metrics[:10]) / 10
        last_10_avg = sum(m.train_loss for m in result.metrics[-10:]) / 10
        assert last_10_avg < first_10_avg, (
            f"Loss should decrease: first_10_avg={first_10_avg:.6f}, last_10_avg={last_10_avg:.6f}"
        )

    def test_deterministic_training_with_seed(self) -> None:
        """Same seed should produce same training results."""
        env = SimpleEnvironment(seed=42, noise_std=0.0)
        experiences = env.collect_experiences(n_steps=100)

        wm1 = WorldModel(state_size=STATE_SIZE, seed=42)
        wm2 = WorldModel(state_size=STATE_SIZE, seed=42)

        result1 = wm1.train(experiences, epochs=10, batch_size=32, verbose=False)
        result2 = wm2.train(experiences, epochs=10, batch_size=32, verbose=False)

        # Both should have the same loss trajectory
        for m1, m2 in zip(result1.metrics, result2.metrics, strict=False):
            assert abs(m1.train_loss - m2.train_loss) < 1e-6, (
                f"Epoch {m1.epoch}: loss1={m1.train_loss:.6f}, loss2={m2.train_loss:.6f}"
            )
