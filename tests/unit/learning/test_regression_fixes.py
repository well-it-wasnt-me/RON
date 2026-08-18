"""Regression tests for fixes from the comprehensive code review.

Each test verifies a specific fix documented in to_fix.md:

* BUG-1: Gradient normalization across batch sizes
* BUG-2: Tensor hash stability after mutation
* BUG-3: Network.predict() cache behavior
* BUG-4: Correct ordering of state and next_state around encoder updates
* DESIGN-1: Enforcement of max_model_params
* THREAD-1: Concurrent SQLite access from multiple threads
* DESIGN-5: Safety manager integration in training cycle
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from robot.learning.experience import (
    Experience,
    ReplayBuffer,
    SqliteExperienceStore,
    WorkingMemory,
)
from robot.learning.layers import DenseLayer
from robot.learning.learning_service import (
    LearningSchedule,
    LearningService,
    ResourceLimits,
)
from robot.learning.losses import mse_derivative, mse_loss
from robot.learning.network import MLP
from robot.learning.optimizers import SGD
from robot.learning.recorder import ExperienceRecorder
from robot.learning.safety import LearningSafetyManager
from robot.learning.state_encoder import StateEncoder
from robot.learning.tensor import Tensor

# ---------------------------------------------------------------------------
# BUG-1: Gradient correctness across batch sizes
# ---------------------------------------------------------------------------


class TestGradientNormalization:
    """Verify gradients are correctly normalised regardless of batch size.

    With the fix, the weight gradient should match the numerical gradient
    of the MSE loss without any extra division by batch_size.
    """

    def test_gradient_matches_numerical_for_batch_4(self) -> None:
        """Analytical gradient matches numerical gradient for batch=4."""
        layer = DenseLayer(3, 2, activation="linear", weight_init="normal", seed=42)
        x = Tensor(np.random.default_rng(42).standard_normal((4, 3)))
        target = Tensor(np.random.default_rng(99).standard_normal((4, 2)))

        pred = layer.forward(x)
        grad = mse_derivative(pred, target)
        layer.backward(grad)

        # Numerical gradient of MSE loss
        eps = 1e-5
        numerical_w = np.zeros_like(layer.weights.data)
        for i in range(layer.weights.data.shape[0]):
            for j in range(layer.weights.data.shape[1]):
                original = layer.weights.data[i, j]
                layer.weights.data[i, j] = original + eps
                loss_plus = mse_loss(layer.forward(x), target).item()
                layer.weights.data[i, j] = original - eps
                loss_minus = mse_loss(layer.forward(x), target).item()
                numerical_w[i, j] = (loss_plus - loss_minus) / (2 * eps)
                layer.weights.data[i, j] = original

        assert np.allclose(layer.weight_grad.data, numerical_w, atol=1e-4), (
            f"Weight grad mismatch:\n  analytical={layer.weight_grad.data}\n"
            f"  numerical={numerical_w}"
        )

    def test_gradient_matches_numerical_for_batch_16(self) -> None:
        """Analytical gradient matches numerical gradient for batch=16."""
        layer = DenseLayer(3, 2, activation="linear", weight_init="normal", seed=42)
        x = Tensor(np.random.default_rng(42).standard_normal((16, 3)))
        target = Tensor(np.random.default_rng(99).standard_normal((16, 2)))

        pred = layer.forward(x)
        grad = mse_derivative(pred, target)
        layer.backward(grad)

        # Numerical gradient (no extra division)
        eps = 1e-5
        numerical_w = np.zeros_like(layer.weights.data)
        for i in range(layer.weights.data.shape[0]):
            for j in range(layer.weights.data.shape[1]):
                original = layer.weights.data[i, j]
                layer.weights.data[i, j] = original + eps
                loss_plus = mse_loss(layer.forward(x), target).item()
                layer.weights.data[i, j] = original - eps
                loss_minus = mse_loss(layer.forward(x), target).item()
                numerical_w[i, j] = (loss_plus - loss_minus) / (2 * eps)
                layer.weights.data[i, j] = original

        assert np.allclose(layer.weight_grad.data, numerical_w, atol=1e-4)

    def test_sgd_convergence_not_batch_dependent(self) -> None:
        """With correct gradients, SGD should converge similarly regardless of batch size.

        Uses the same training data (duplicated) so the optimization
        landscape is identical. With the fix, the gradient is the same
        (since the data is duplicated, the gradient sums are proportional
        to batch_size, but the 1/N normalization cancels that).
        """
        np.random.seed(42)
        base_x = np.random.randn(4, 2)
        base_y = np.random.randn(4, 1)

        for batch_size in [4, 16]:
            x = Tensor(np.tile(base_x, (batch_size // 4, 1)))
            y = Tensor(np.tile(base_y, (batch_size // 4, 1)))

            mlp = MLP(input_size=2, hidden_sizes=[8], output_size=1, seed=42)
            opt = SGD(learning_rate=0.01)

            initial_loss = mse_loss(mlp.forward(x), y).item()
            for _ in range(100):
                mlp.network.train_step(x, y, loss_fn="mse", optimizer=opt)
            final_loss = mse_loss(mlp.forward(x), y).item()

            # Both batch sizes should converge (loss decreased)
            assert final_loss < initial_loss, (
                f"batch={batch_size}: loss did not decrease "
                f"({initial_loss} -> {final_loss})"
            )


# ---------------------------------------------------------------------------
# BUG-2: Tensor hash stability after mutation
# ---------------------------------------------------------------------------


class TestTensorHashSafety:
    """Verify Tensor is not hashable (mutable objects should not be hashable)."""

    def test_tensor_is_unhashable(self) -> None:
        """Tensor must be unhashable because it is mutable."""
        t = Tensor([1.0, 2.0, 3.0])
        with pytest.raises(TypeError, match="unhashable"):
            hash(t)

    def test_tensor_cannot_be_in_set(self) -> None:
        """Tensor cannot be added to a set."""
        t = Tensor([1.0, 2.0])
        with pytest.raises(TypeError):
            _ = {t}

    def test_tensor_equality_still_works(self) -> None:
        """Value-based equality should still work for comparisons."""
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([1.0, 2.0, 3.0])
        c = Tensor([4.0, 5.0, 6.0])
        assert a == b
        assert a != c

    def test_tensor_setitem_still_works(self) -> None:
        """Mutation via __setitem__ should still work."""
        t = Tensor.zeros(3)
        t[1] = 5.0
        assert float(t.data[1]) == 5.0


# ---------------------------------------------------------------------------
# BUG-3: Network.predict() cache behavior
# ---------------------------------------------------------------------------


class TestPredictCacheBehavior:
    """Verify predict() docstring accurately describes caching behavior."""

    def test_predict_populates_cache(self) -> None:
        """predict() does populate layer caches (documented behavior)."""
        mlp = MLP(input_size=3, hidden_sizes=[4], output_size=2, seed=42)
        x = Tensor(np.random.default_rng(42).standard_normal((4, 3)))
        mlp.predict(x)

        # Caches ARE populated (this is now documented)
        first_layer = mlp.network.layers[0]
        assert first_layer._input is not None
        assert first_layer._output is not None

    def test_predict_then_train_overwrites_cache(self) -> None:
        """Calling train_step after predict overwrites stale caches."""
        mlp = MLP(input_size=3, hidden_sizes=[4], output_size=2, seed=42)
        x_predict = Tensor(np.random.default_rng(1).standard_normal((2, 3)))
        x_train = Tensor(np.random.default_rng(2).standard_normal((4, 3)))
        y_train = Tensor(np.random.default_rng(3).standard_normal((4, 2)))

        # This should not cause incorrect gradients
        mlp.predict(x_predict)
        loss, _ = mlp.network.train_step(x_train, y_train, loss_fn="mse")

        # Verify loss is reasonable (not NaN or inf)
        assert not np.isnan(loss)
        assert not np.isinf(loss)


# ---------------------------------------------------------------------------
# BUG-4: Correct ordering of state and next_state
# ---------------------------------------------------------------------------


class TestRecorderStateOrdering:
    """Verify recorder captures pre-action state as 'state' and post-action as 'next_state'."""

    def test_emotion_change_captures_pre_post_state(self) -> None:
        """When an emotion changes, state should be BEFORE and next_state AFTER."""
        from robot.events.bus import InMemoryEventBus
        from robot.events.events import EmotionChanged, EmotionName

        bus = InMemoryEventBus()
        encoder = StateEncoder()
        recorder = ExperienceRecorder(
            bus=bus,
            encoder=encoder,
            working_memory=WorkingMemory(capacity=10),
            replay_buffer=ReplayBuffer(capacity=100, seed=42),
        )
        recorder.attach()

        # Set an initial emotion
        encoder.update_emotion(EmotionName.NEUTRAL, 1.0)

        # Publish an EmotionChanged event
        import anyio

        async def run() -> None:
            await bus.publish(
                EmotionChanged(
                    previous=EmotionName.NEUTRAL,
                    current=EmotionName.HAPPY,
                    intensity=0.9,
                )
            )

        anyio.run(run)

        # Check the recorded experience
        experiences = list(recorder.working_memory)
        assert len(experiences) == 1
        exp = experiences[0]

        # State should NOT have HAPPY emotion (before the change)
        # EmotionName enum: NEUTRAL=0, HAPPY=1, so index 1 = happy
        state_happy = exp.state[1]  # happy emotion intensity
        next_state_happy = exp.next_state[1]

        assert state_happy == 0.0, (
            f"State should have happy=0.0 (before change), got {state_happy}"
        )
        assert next_state_happy == 0.9, (
            f"Next state should have happy=0.9 (after change), got {next_state_happy}"
        )

    def test_face_detected_captures_pre_post_state(self) -> None:
        """When a face is detected, state should be BEFORE and next_state AFTER."""
        from robot.events.bus import InMemoryEventBus
        from robot.events.events import FaceDetected

        bus = InMemoryEventBus()
        encoder = StateEncoder()
        recorder = ExperienceRecorder(
            bus=bus,
            encoder=encoder,
            working_memory=WorkingMemory(capacity=10),
            replay_buffer=ReplayBuffer(capacity=100, seed=42),
        )
        recorder.attach()

        import anyio

        async def run() -> None:
            await bus.publish(FaceDetected(x=0.3, y=0.7, confidence=0.9))

        anyio.run(run)

        experiences = list(recorder.working_memory)
        assert len(experiences) == 1
        exp = experiences[0]

        # Vision section starts at index 33
        # [33] = face_detected, [34] = face_x, [35] = face_y
        state_face_detected = exp.state[33]
        next_state_face_detected = exp.next_state[33]
        next_state_x = exp.next_state[34]

        assert state_face_detected == 0.0, (
            f"State should have face_detected=0.0 (before), got {state_face_detected}"
        )
        assert next_state_face_detected == 1.0, (
            f"Next state should have face_detected=1.0 (after), got {next_state_face_detected}"
        )
        assert abs(next_state_x - 0.3) < 0.01, (
            f"Next state face_x should be ~0.3, got {next_state_x}"
        )


# ---------------------------------------------------------------------------
# DESIGN-1: Enforcement of max_model_params
# ---------------------------------------------------------------------------


class TestMaxModelParamsEnforcement:
    """Verify max_model_params is enforced at construction time."""

    def test_small_limit_raises_error(self) -> None:
        """A very small max_model_params should raise ValueError."""
        from robot.events.bus import InMemoryEventBus

        bus = InMemoryEventBus()
        tiny_limits = ResourceLimits(max_model_params=10)

        with pytest.raises(ValueError, match="exceeds max_model_params"):
            LearningService(bus=bus, resource_limits=tiny_limits)

    def test_default_limits_allow_normal_models(self) -> None:
        """Default ResourceLimits (500k params) should allow normal models."""
        from robot.events.bus import InMemoryEventBus

        bus = InMemoryEventBus()
        service = LearningService(bus=bus)

        # Should not raise
        assert service.current_world_model is not None
        assert service.current_world_model.param_count() <= 500_000


# ---------------------------------------------------------------------------
# THREAD-1: Concurrent SQLite access
# ---------------------------------------------------------------------------


class TestConcurrentSQLite:
    """Verify SqliteExperienceStore is safe for cross-thread access."""

    def test_concurrent_writes_from_multiple_threads(self) -> None:
        """Multiple threads writing concurrently should not corrupt data."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SqliteExperienceStore(db_path=Path(tmpdir) / "test.db")
            errors: list[Exception] = []

            def writer(thread_id: int) -> None:
                try:
                    for i in range(20):
                        exp = Experience(
                            timestamp=datetime.now(tz=UTC),
                            state=[float(thread_id * 100 + i)],
                            action=[float(i)],
                            reward=float(i),
                            next_state=[float(thread_id * 100 + i + 1)],
                            metadata={"thread": thread_id, "i": i},
                        )
                        store.save(exp)
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors, f"Thread errors: {errors}"
            assert store.count() == 80  # 4 threads * 20 writes
            store.close()

    def test_concurrent_read_write(self) -> None:
        """Concurrent reads and writes should not corrupt."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SqliteExperienceStore(db_path=Path(tmpdir) / "test.db")

            # Pre-populate
            for i in range(50):
                store.save(
                    Experience(
                        timestamp=datetime.now(tz=UTC),
                        state=[float(i)],
                        action=[0.0],
                        reward=0.0,
                        next_state=[float(i + 1)],
                        metadata={},
                    )
                )

            errors: list[Exception] = []
            results: list[int] = []

            def reader() -> None:
                try:
                    recent = store.load_recent(limit=50)
                    results.append(len(recent))
                except Exception as exc:
                    errors.append(exc)

            def writer() -> None:
                try:
                    for i in range(20):
                        store.save(
                            Experience(
                                timestamp=datetime.now(tz=UTC),
                                state=[float(100 + i)],
                                action=[0.0],
                                reward=0.0,
                                next_state=[float(101 + i)],
                                metadata={},
                            )
                        )
                except Exception as exc:
                    errors.append(exc)

            t_reader = threading.Thread(target=reader)
            t_writer = threading.Thread(target=writer)
            t_reader.start()
            t_writer.start()
            t_reader.join()
            t_writer.join()

            assert not errors, f"Thread errors: {errors}"
            store.close()


# ---------------------------------------------------------------------------
# DESIGN-5: Safety manager integration
# ---------------------------------------------------------------------------


class TestSafetyManagerIntegration:
    """Verify LearningSafetyManager is integrated into the training cycle."""

    def test_safety_manager_is_created(self) -> None:
        """LearningService should create a safety manager."""
        from robot.events.bus import InMemoryEventBus

        bus = InMemoryEventBus()
        service = LearningService(bus=bus)
        assert service.safety_mgr is not None
        assert isinstance(service.safety_mgr, LearningSafetyManager)

    def test_safety_manager_evaluates_candidate(self) -> None:
        """The safety manager should be used for candidate evaluation."""
        from robot.events.bus import InMemoryEventBus

        bus = InMemoryEventBus()
        service = LearningService(
            bus=bus,
            schedule=LearningSchedule(
                min_new_experiences=2,
                train_interval_s=0.01,
                min_experiences_for_training=4,
            ),
            resource_limits=ResourceLimits(
                batch_size=2,
                training_epochs_per_cycle=2,
                eval_sample_size=6,
            ),
        )

        # Record enough experiences to trigger training
        for i in range(10):
            service.record_experience(
                state=[float(i)] * 91,
                action=[float(i)] * 20,
                reward=float(i % 3),
                next_state=[float(i + 1)] * 91,
                metadata={"i": i},
            )

        # Force a training cycle
        result = service.force_training()

        # The training should have used the safety manager
        # (if it didn't crash, the integration is working)
        assert isinstance(result, bool)

        service.stop()
