"""World model: predict the next state from the current state and action.

The :class:`WorldModel` is DeskBot's first genuine learning behaviour.
It learns to predict:

    current_state + action -> predicted_next_state

Training uses experiences collected by the Phase 2 memory system,
sampled via experience replay.  The model is an MLP built on the
Phase 1 neural network core.

The world model is **not** connected to real hardware.  It trains on
collected experiences and is evaluated against held-out data.  Only
when prediction error demonstrably decreases is the model considered
fit for use in later phases (action learning, planning).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robot.learning.experience import Experience
from robot.learning.losses import mse_loss
from robot.learning.network import MLP
from robot.learning.optimizers import Adam
from robot.learning.state_encoder import STATE_SIZE
from robot.learning.tensor import Tensor
from robot.logging import get_logger

_log = get_logger("learning.world_model")


# ---------------------------------------------------------------------------
# Training metrics
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrainingMetrics:
    """Metrics from a single training epoch.

    Attributes
    ----------
    epoch:
        Epoch number (0-indexed).
    train_loss:
        Mean training loss for this epoch.
    val_loss:
        Mean validation loss for this epoch (0.0 if no validation).
    elapsed_s:
        Wall-clock time for this epoch in seconds.
    """

    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "elapsed_s": round(self.elapsed_s, 4),
        }


@dataclass(slots=True)
class TrainingResult:
    """Result of a complete training run.

    Attributes
    ----------
    initial_loss:
        Loss before any training.
    final_loss:
        Loss after the last epoch.
    best_val_loss:
        Best validation loss seen during training.
    epochs:
        Number of epochs actually trained.
    metrics:
        Per-epoch training metrics.
    improved:
        Whether the final loss is lower than the initial loss.
    """

    initial_loss: float = 0.0
    final_loss: float = 0.0
    best_val_loss: float = float("inf")
    epochs: int = 0
    metrics: list[TrainingMetrics] = field(default_factory=list)
    improved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "best_val_loss": self.best_val_loss,
            "epochs": self.epochs,
            "improved": self.improved,
            "metrics": [m.to_dict() for m in self.metrics],
        }


# ---------------------------------------------------------------------------
# World model
# ---------------------------------------------------------------------------

# Default action size - matches the action encoding in recorder.py
# (6 one-hot event types + variable params, but we use a fixed size)
DEFAULT_ACTION_SIZE = 20


@dataclass(slots=True)
class WorldModel:
    """Predict next_state from (state, action).

    The world model is an MLP that takes a concatenated
    ``[state, action]`` vector as input and produces a predicted
    ``next_state`` vector as output.  It is trained on experience
    tuples collected by the Phase 2 memory system.

    Parameters
    ----------
    state_size:
        Dimension of the state vector.  Must match
        :data:`STATE_SIZE` from the state encoder.
    action_size:
        Dimension of the action vector.
    hidden_sizes:
        Sizes of hidden layers in the MLP.
    learning_rate:
        Learning rate for the Adam optimiser.
    seed:
        Random seed for reproducible initialisation.
    """

    state_size: int = STATE_SIZE
    action_size: int = DEFAULT_ACTION_SIZE
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 64])
    learning_rate: float = 0.001
    seed: int = 42
    _model: MLP | None = field(default=None, init=False, repr=False)
    _optimizer: Adam | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._build_model()

    def _build_model(self) -> None:
        """Construct the MLP and optimiser."""
        input_size = self.state_size + self.action_size
        self._model = MLP(
            input_size=input_size,
            hidden_sizes=self.hidden_sizes,
            output_size=self.state_size,
            activation="relu",
            output_activation="linear",
            weight_init="he",
            seed=self.seed,
        )
        self._optimizer = Adam(learning_rate=self.learning_rate)

    @property
    def model(self) -> MLP:
        """The underlying MLP model."""
        assert self._model is not None, "Model not initialised"
        return self._model

    @property
    def optimizer(self) -> Adam:
        """The Adam optimiser."""
        assert self._optimizer is not None, "Optimiser not initialised"
        return self._optimizer

    # ------------------------------------------------------------------ predict
    def predict(
        self, state: list[float] | np.ndarray, action: list[float] | np.ndarray
    ) -> np.ndarray:
        """Predict the next state given the current state and action.

        Parameters
        ----------
        state:
            Current state vector (``state_size`` elements).
        action:
            Action vector (``action_size`` elements).

        Returns
        -------
        np.ndarray
            Predicted next state (``state_size`` elements).
        """
        x = np.concatenate(
            [np.asarray(state, dtype=np.float64), np.asarray(action, dtype=np.float64)]
        )
        x = x.reshape(1, -1)  # batch dimension
        pred = self.model.predict(Tensor(x))
        return pred.data.flatten()

    def predict_batch(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Predict next states for a batch of (state, action) pairs.

        Parameters
        ----------
        states:
            Array of shape ``(batch, state_size)``.
        actions:
            Array of shape ``(batch, action_size)``.

        Returns
        -------
        np.ndarray
            Array of shape ``(batch, state_size)``.
        """
        x = np.concatenate([states, actions], axis=1)
        pred = self.model.predict(Tensor(x))
        return pred.data

    # ------------------------------------------------------------------ train
    def train_epoch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
    ) -> float:
        """Run one training epoch.

        Parameters
        ----------
        states:
            Current states, shape ``(batch, state_size)``.
        actions:
            Actions taken, shape ``(batch, action_size)``.
        next_states:
            Target next states, shape ``(batch, state_size)``.

        Returns
        -------
        float
            Mean loss for this epoch.
        """
        # Concatenate state + action as input
        x = np.concatenate([states, actions], axis=1)
        x_tensor = Tensor(x)
        y_tensor = Tensor(next_states)

        loss, _ = self.model.network.train_step(
            x_tensor, y_tensor, loss_fn="mse", optimizer=self._optimizer
        )
        return loss

    def train(
        self,
        experiences: list[Experience],
        val_experiences: list[Experience] | None = None,
        epochs: int = 100,
        batch_size: int = 32,
        val_split: float = 0.2,
        verbose: bool = True,
    ) -> TrainingResult:
        """Train the world model on experience data.

        Parameters
        ----------
        experiences:
            List of experience tuples to train on.
        val_experiences:
            Optional held-out validation set.  If ``None``, a fraction
            of ``experiences`` is used (see ``val_split``).
        epochs:
            Number of training epochs.
        batch_size:
            Mini-batch size for training.
        val_split:
            Fraction of ``experiences`` to use for validation
            (only used if ``val_experiences`` is ``None``).
        verbose:
            Whether to log progress.

        Returns
        -------
        TrainingResult
            Training metrics including loss curves.
        """
        if len(experiences) < 2:
            _log.warning("world_model.insufficient_data", count=len(experiences))
            return TrainingResult()

        # Convert experiences to arrays
        all_states, all_actions, all_next_states = self._experiences_to_arrays(experiences)

        # Split into train/val
        if val_experiences is not None:
            val_states, val_actions, val_next = self._experiences_to_arrays(val_experiences)
        else:
            n_val = max(1, int(len(all_states) * val_split))
            indices = np.arange(len(all_states))
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]
            train_states = all_states[train_idx]
            train_actions = all_actions[train_idx]
            train_next = all_next_states[train_idx]
            val_states = all_states[val_idx]
            val_actions = all_actions[val_idx]
            val_next = all_next_states[val_idx]
            all_states = train_states
            all_actions = train_actions
            all_next_states = train_next

        # Compute initial loss
        initial_loss = self._evaluate(all_states, all_actions, all_next_states)
        result = TrainingResult(initial_loss=initial_loss)
        best_val_loss = float("inf")

        rng = np.random.default_rng(self.seed)

        for epoch in range(epochs):
            t0 = time.monotonic()

            # Mini-batch training
            n_samples = len(all_states)
            indices = rng.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                batch_loss = self.train_epoch(
                    all_states[batch_idx],
                    all_actions[batch_idx],
                    all_next_states[batch_idx],
                )
                epoch_loss += batch_loss
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            val_loss = self._evaluate(val_states, val_actions, val_next)

            elapsed = time.monotonic() - t0

            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                elapsed_s=elapsed,
            )
            result.metrics.append(metrics)

            best_val_loss = min(best_val_loss, val_loss)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                _log.info(
                    "world_model.train",
                    epoch=epoch,
                    train_loss=round(avg_train_loss, 6),
                    val_loss=round(val_loss, 6),
                )

        result.final_loss = result.metrics[-1].train_loss if result.metrics else initial_loss
        result.best_val_loss = best_val_loss
        result.epochs = epochs
        result.improved = result.final_loss < initial_loss

        if result.improved:
            _log.info(
                "world_model.improved",
                initial_loss=round(initial_loss, 6),
                final_loss=round(result.final_loss, 6),
                improvement_pct=round((1.0 - result.final_loss / initial_loss) * 100, 2),
            )
        else:
            _log.warning(
                "world_model.no_improvement",
                initial_loss=round(initial_loss, 6),
                final_loss=round(result.final_loss, 6),
            )

        return result

    # ------------------------------------------------------------------ evaluate
    def evaluate(self, experiences: list[Experience]) -> float:
        """Evaluate the model on a dataset of experiences.

        Returns the mean MSE loss.
        """
        if not experiences:
            return 0.0
        states, actions, next_states = self._experiences_to_arrays(experiences)
        return self._evaluate(states, actions, next_states)

    def _evaluate(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray) -> float:
        """Compute MSE loss on a dataset."""
        if len(states) == 0:
            return 0.0
        x = np.concatenate([states, actions], axis=1)
        pred = self.model.predict(Tensor(x))
        loss = mse_loss(pred, Tensor(next_states))
        return loss.item()

    # ------------------------------------------------------------------ helpers
    def _experiences_to_arrays(
        self, experiences: list[Experience]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert a list of experiences to numpy arrays.

        Pads or truncates actions to ``action_size``.
        """
        states_list = []
        actions_list = []
        next_states_list = []

        for exp in experiences:
            state = np.array(exp.state, dtype=np.float64)
            # Pad or truncate action to fixed size
            action = np.zeros(self.action_size, dtype=np.float64)
            exp_action = np.array(exp.action, dtype=np.float64)
            n_copy = min(len(exp_action), self.action_size)
            action[:n_copy] = exp_action[:n_copy]
            next_state = np.array(exp.next_state, dtype=np.float64)

            states_list.append(state)
            actions_list.append(action)
            next_states_list.append(next_state)

        return (
            np.array(states_list),
            np.array(actions_list),
            np.array(next_states_list),
        )

    # ------------------------------------------------------------------ save/load
    def save(self, path: str | Path) -> None:
        """Save the world model to a JSON file."""
        self.model.save(path)

    def load(self, path: str | Path) -> None:
        """Load the world model from a JSON file."""
        self._model = MLP.load(path)
        # Re-create optimiser (state is not saved)
        self._optimizer = Adam(learning_rate=self.learning_rate)

    def param_count(self) -> int:
        """Return total number of trainable parameters."""
        return self.model.network.param_count()


# ---------------------------------------------------------------------------
# Simple simulation environment for world model training
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SimpleEnvironment:
    """A simple deterministic environment for testing the world model.

    The environment has a 2-D "face position" (x, y in [0, 1]) and
    responds to look-left / look-right actions by shifting the face
    position.  This creates predictable state transitions that the
    world model should learn.

    Actions are one-hot encoded:
        [1, 0] = look left (decrease x by 0.1)
        [0, 1] = look right (increase x by 0.1)

    The state is a simplified 4-element vector:
        [face_x, face_y, face_detected, idle_time]

    Parameters
    ----------
    step_size:
        How much the face position changes per action.
    noise_std:
        Standard deviation of Gaussian noise added to transitions.
    seed:
        Random seed for reproducibility.
    """

    step_size: float = 0.1
    noise_std: float = 0.01
    seed: int = 42
    _face_x: float = field(default=0.5, init=False)
    _face_y: float = field(default=0.5, init=False)
    _face_detected: float = field(default=1.0, init=False)
    _idle_time: float = field(default=0.0, init=False)
    _rng: np.random.Generator | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial state."""
        self._face_x = 0.5
        self._face_y = 0.5
        self._face_detected = 1.0
        self._idle_time = 0.0
        return self.state

    @property
    def state(self) -> np.ndarray:
        """Return the current state as a numpy array."""
        return np.array(
            [self._face_x, self._face_y, self._face_detected, self._idle_time], dtype=np.float64
        )

    @property
    def state_size(self) -> int:
        return 4

    @property
    def action_size(self) -> int:
        return 2  # look_left, look_right

    def step(self, action: int) -> tuple[np.ndarray, float]:
        """Take an action and return (next_state, reward).

        Parameters
        ----------
        action:
            0 = look left, 1 = look right.

        Returns
        -------
        tuple[np.ndarray, float]
            (next_state, reward)
        """
        assert self._rng is not None
        if action == 0:
            # look left: decrease x
            self._face_x = max(
                0.0, self._face_x - self.step_size + self._rng.normal(0, self.noise_std)
            )
        elif action == 1:
            # look right: increase x
            self._face_x = min(
                1.0, self._face_x + self.step_size + self._rng.normal(0, self.noise_std)
            )

        # Small random drift in y
        self._face_y = max(0.0, min(1.0, self._face_y + self._rng.normal(0, self.noise_std)))
        self._face_detected = 1.0
        self._idle_time = 0.0

        # Reward: staying near centre is slightly positive
        centre_dist = abs(self._face_x - 0.5) + abs(self._face_y - 0.5)
        reward = 0.1 - 0.05 * centre_dist

        return self.state.copy(), reward

    def action_onehot(self, action: int) -> np.ndarray:
        """Convert an action index to a one-hot vector."""
        vec = np.zeros(self.action_size, dtype=np.float64)
        vec[action] = 1.0
        return vec

    def collect_experiences(
        self, n_steps: int = 200, max_state_size: int = STATE_SIZE
    ) -> list[Experience]:
        assert self._rng is not None
        """Collect experiences by randomly interacting with the environment.

        Returns a list of :class:`Experience` tuples suitable for
        training the world model.  State vectors are padded to
        ``max_state_size`` (default: ``STATE_SIZE``) to be compatible
        with the full encoder.
        """
        from datetime import UTC, datetime

        experiences: list[Experience] = []
        state = self.reset()

        for _ in range(n_steps):
            action_idx = int(self._rng.integers(0, self.action_size))
            action = self.action_onehot(action_idx)
            next_state, reward = self.step(action_idx)

            # Pad state/action to full size for compatibility with Experience
            padded_state = np.zeros(max_state_size, dtype=np.float64)
            padded_state[: len(state)] = state
            padded_next = np.zeros(max_state_size, dtype=np.float64)
            padded_next[: len(next_state)] = next_state

            padded_action = np.zeros(DEFAULT_ACTION_SIZE, dtype=np.float64)
            padded_action[: len(action)] = action

            exp = Experience(
                timestamp=datetime.now(tz=UTC),
                state=padded_state.tolist(),
                action=padded_action.tolist(),
                reward=reward,
                next_state=padded_next.tolist(),
                metadata={"source": "simple_env", "action_idx": action_idx},
            )
            experiences.append(exp)
            state = next_state.copy()

        return experiences


__all__ = [
    "DEFAULT_ACTION_SIZE",
    "SimpleEnvironment",
    "TrainingMetrics",
    "TrainingResult",
    "WorldModel",
]
