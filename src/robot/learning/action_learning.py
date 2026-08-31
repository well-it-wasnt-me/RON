"""Action learning: learn which actions are valuable in a given state.

This module implements a simple action-value learning system. The
:class:`ActionLearner` estimates the value of each available action
in a given state, and uses that estimate to select actions with
a configurable exploration strategy.

The action space is constrained to a fixed set of registered DeskBot
actions.  The learner **never** directly accesses hardware.  It only
produces action indices that must be validated by a
:class:`ActionValidator` and executed by the existing
:class:`ActionExecutor`.

The learning algorithm is Q-learning with function approximation
(MLP from Phase 1), keeping it understandable and testable.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from robot.learning.network import MLP
from robot.learning.optimizers import Adam
from robot.learning.tensor import Tensor
from robot.logging import get_logger

_log = get_logger("learning.action_learning")


# ---------------------------------------------------------------------------
# Action definition
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class LearningAction:
    """A registered action that the learner can select."""

    index: int
    name: str
    description: str = ""
    action_type: str = ""
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ActionSpace:
    """Registry of actions available to the learner."""

    _actions: list[LearningAction] = field(default_factory=list, init=False)
    _name_to_index: dict[str, int] = field(default_factory=dict, init=False)

    def register(
        self,
        name: str,
        description: str = "",
        action_type: str = "",
        params: dict[str, Any] | None = None,
    ) -> LearningAction:
        if name in self._name_to_index:
            raise ValueError(f"Action {name!r} is already registered")
        index = len(self._actions)
        action = LearningAction(
            index=index,
            name=name,
            description=description,
            action_type=action_type,
            params=params or {},
        )
        self._actions.append(action)
        self._name_to_index[name] = index
        return action

    def get(self, index: int) -> LearningAction:
        if 0 <= index < len(self._actions):
            return self._actions[index]
        raise IndexError(f"Action index {index} out of range [0, {len(self._actions)})")

    def get_by_name(self, name: str) -> LearningAction:
        if name not in self._name_to_index:
            raise KeyError(f"Action {name!r} not registered")
        return self._actions[self._name_to_index[name]]

    @property
    def size(self) -> int:
        return len(self._actions)

    def action_vector(self, index: int) -> np.ndarray:
        vec = np.zeros(self.size, dtype=np.float64)
        if 0 <= index < self.size:
            vec[index] = 1.0
        return vec

    def action_vector_from_name(self, name: str) -> np.ndarray:
        return self.action_vector(self._name_to_index[name])

    def __len__(self) -> int:
        return len(self._actions)

    def __iter__(self) -> Iterator[LearningAction]:
        return iter(self._actions)


def deskbot_action_space() -> ActionSpace:
    """Create an action space with standard DeskBot actions."""
    space = ActionSpace()
    space.register(
        "look_left",
        description="Look to the left",
        action_type="look",
        params={"x": -0.5, "y": 0.0},
    )
    space.register(
        "look_right",
        description="Look to the right",
        action_type="look",
        params={"x": 0.5, "y": 0.0},
    )
    space.register(
        "look_center",
        description="Look straight ahead",
        action_type="look",
        params={"x": 0.0, "y": 0.0},
    )
    space.register(
        "look_up", description="Look upward", action_type="look", params={"x": 0.0, "y": -0.5}
    )
    space.register(
        "look_down", description="Look downward", action_type="look", params={"x": 0.0, "y": 0.5}
    )
    space.register(
        "blink",
        description="Blink both eyes",
        action_type="blink",
        params={"left": True, "right": True, "speed": 1.0},
    )
    space.register(
        "wink",
        description="Wink left eye",
        action_type="blink",
        params={"left": True, "right": False, "speed": 1.5},
    )
    space.register(
        "celebrate",
        description="Express happiness",
        action_type="celebrate",
        params={"intensity": 0.7},
    )
    space.register(
        "sleep",
        description="Enter low-power idle",
        action_type="sleep",
        params={"duration_s": 30.0},
    )
    space.register(
        "look_around",
        description="Scan the environment",
        action_type="look_around",
        params={"points": 3},
    )
    # ----- Learnable interaction actions (teaching loop) -----
    # These extend the action space from 10 -> 16. They map to real
    # BehaviourActions executed through the ActionExecutor, so experiences
    # from them carry a meaningful action identity for the Q-policy.
    space.register(
        "speak",
        description="Speak a short phrase via TTS",
        action_type="speak",
        params={"text": "hello"},
    )
    space.register(
        "change_emotion",
        description="Change the facial emotion",
        action_type="change_emotion",
        params={"emotion": "happy", "intensity": 1.0},
    )
    space.register(
        "set_state",
        description="Set the robot behaviour state directly",
        action_type="set_state",
        params={"state": "idle"},
    )
    space.register(
        "wave",
        description="Wave the right arm",
        action_type="wave",
        params={},
    )
    space.register(
        "move_left_arm",
        description="Move the left arm servo to an angle",
        action_type="move_arm",
        params={"servo": "left_arm", "angle": 90.0},
    )
    space.register(
        "move_right_arm",
        description="Move the right arm servo to an angle",
        action_type="move_arm",
        params={"servo": "right_arm", "angle": 90.0},
    )
    return space


# ---------------------------------------------------------------------------
# Reward system
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Reward:
    """A scalar reward signal."""

    value: float
    description: str = ""
    source: str = ""

    @property
    def is_positive(self) -> bool:
        return self.value > 0.0

    @property
    def is_negative(self) -> bool:
        return self.value < 0.0

    @property
    def is_neutral(self) -> bool:
        return self.value == 0.0


REWARD_POSITIVE = Reward(value=1.0, description="Positive reward", source="default")
REWARD_NEUTRAL = Reward(value=0.0, description="Neutral reward", source="default")
REWARD_NEGATIVE = Reward(value=-1.0, description="Negative reward", source="default")


@runtime_checkable
class RewardFunction(Protocol):
    def __call__(
        self,
        state: np.ndarray,
        action_index: int,
        next_state: np.ndarray,
        action_space: ActionSpace,
    ) -> Reward: ...


# ---------------------------------------------------------------------------
# Action validator (safety)
# ---------------------------------------------------------------------------


@runtime_checkable
class ActionValidator(Protocol):
    def is_valid(self, action_index: int, state: np.ndarray, action_space: ActionSpace) -> bool: ...


class DefaultValidator:
    """Default validator that allows all registered actions."""

    def is_valid(self, action_index: int, state: np.ndarray, action_space: ActionSpace) -> bool:
        return 0 <= action_index < action_space.size


# ---------------------------------------------------------------------------
# Action learner (Q-learning with function approximation)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ActionLearner:
    """Learn which actions are valuable in each state.

    Uses an MLP to estimate Q(s, a).  The network takes
    ``[state, action_onehot]`` as input and outputs a single Q-value.

    Action selection uses epsilon-greedy with configurable decay.
    Every selected action must pass through a validator before execution.

    Parameters
    ----------
    action_space:
        The set of available actions.
    state_size:
        Dimension of the state vector (default: 4 for simulation,
        STATE_SIZE for full encoder).
    hidden_sizes:
        Hidden layer sizes for the Q-network.
    learning_rate:
        Learning rate for the Adam optimiser.
    gamma:
        Discount factor for future rewards.
    epsilon_start:
        Initial exploration rate.
    epsilon_end:
        Final exploration rate after decay.
    epsilon_decay:
        Multiplicative decay per step.
    seed:
        Random seed for reproducibility.
    """

    action_space: ActionSpace
    state_size: int = 4
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 32])
    learning_rate: float = 0.001
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    seed: int = 42
    _model: MLP | None = field(default=None, init=False, repr=False)
    _optimizer: Adam | None = field(default=None, init=False, repr=False)
    _rng: np.random.Generator | None = field(default=None, init=False, repr=False)
    _epsilon: float = field(default=1.0, init=False)
    _step_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._build_model()
        self._rng = np.random.default_rng(self.seed)
        self._epsilon = self.epsilon_start

    def _build_model(self) -> None:
        input_size = self.state_size + self.action_space.size
        self._model = MLP(
            input_size=input_size,
            hidden_sizes=self.hidden_sizes,
            output_size=1,
            activation="relu",
            output_activation="linear",
            weight_init="he",
            seed=self.seed,
        )
        self._optimizer = Adam(learning_rate=self.learning_rate)

    @property
    def model(self) -> MLP:
        assert self._model is not None
        return self._model

    @property
    def epsilon(self) -> float:
        return self._epsilon

    # ------------------------------------------------------------------ select
    def select_action(self, state: np.ndarray, validator: ActionValidator | None = None) -> int:
        """Select an action using epsilon-greedy exploration."""
        valid = validator or DefaultValidator()

        assert self._rng is not None
        if self._rng.random() < self._epsilon:
            valid_indices = [
                i
                for i in range(self.action_space.size)
                if valid.is_valid(i, state, self.action_space)
            ]
            if not valid_indices:
                return 0
            return int(self._rng.choice(valid_indices))

        return self.greedy_action(state, validator=valid)

    def greedy_action(self, state: np.ndarray, validator: ActionValidator | None = None) -> int:
        """Select the best action without exploration."""
        valid = validator or DefaultValidator()
        q_vals = self.q_values(state)
        sorted_indices = np.argsort(q_vals)[::-1]
        for idx in sorted_indices:
            if valid.is_valid(int(idx), state, self.action_space):
                return int(idx)
        for i in range(self.action_space.size):
            if valid.is_valid(i, state, self.action_space):
                return i
        return 0

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Compute Q-values for all actions in the given state."""
        state = np.asarray(state, dtype=np.float64)
        if state.ndim != 1:
            raise ValueError(f"state must be a 1-D vector, got shape {state.shape}")

        if state.size != self.state_size:
            raise ValueError(
                f"state dimension mismatch: got {state.size}, expected {self.state_size}"
            )

        if not np.all(np.isfinite(state)):
            raise ValueError("state contains NaN or infinite values")

        n_actions = self.action_space.size
        inputs = np.zeros(
            (n_actions, self.state_size + n_actions),
            dtype=np.float64,
        )
        # Put the same state into every action row.
        inputs[:, : self.state_size] = state

        # my brain: "yes, you implemented that"
        # reality hititng with a brick: no you didnt.

        # Add one-hot action encoding.
        for i in range(n_actions):
            inputs[i, self.state_size + i] = 1.0

        # Evaluate Q(s, a) for every action.
        pred = self.model.predict(Tensor(inputs))

        return pred.data.flatten()

    def q_value(self, state: np.ndarray, action_index: int) -> float:
        """Compute Q(s, a) for a single state-action pair."""
        return float(self.q_values(state)[action_index])

    def q_values_batch(self, states: np.ndarray) -> np.ndarray:
        """Compute Q-values for all actions across a batch of states.

        Parameters
        ----------
        states:
            Array of shape ``(batch, state_size)``.

        Returns
        -------
        np.ndarray
            Array of shape ``(batch, n_actions)``.
        """
        states = np.asarray(states, dtype=np.float64)

        if states.ndim != 2:
            raise ValueError(f"states must be a 2-D array, got shape {states.shape}")

        if states.shape[1] != self.state_size:
            raise ValueError(
                f"state dimension mismatch: got {states.shape[1]}, expected {self.state_size}"
            )

        if not np.all(np.isfinite(states)):
            raise ValueError("states contain NaN or infinite values")

        batch_size = states.shape[0]
        n_actions = self.action_space.size
        inputs = np.zeros((batch_size * n_actions, self.state_size + n_actions), dtype=np.float64)
        # Broadcast each state to n_actions rows
        for i in range(batch_size):
            row_start = i * n_actions
            row_end = row_start + n_actions
            inputs[row_start:row_end, : self.state_size] = states[i][np.newaxis, :]
            for a in range(n_actions):
                inputs[row_start + a, self.state_size + a] = 1.0
        pred = self.model.predict(Tensor(inputs))
        return pred.data.reshape(batch_size, n_actions)

    # ------------------------------------------------------------------ train
    def train_step(
        self,
        state: np.ndarray,
        action_index: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> float:
        """Run one Q-learning update."""
        state = np.asarray(state, dtype=np.float64)
        next_state = np.asarray(next_state, dtype=np.float64)

        if state.ndim != 1 or state.size != self.state_size:
            raise ValueError(f"state must have shape ({self.state_size},), got {state.shape}")

        if next_state.ndim != 1 or next_state.size != self.state_size:
            raise ValueError(
                f"next_state must have shape ({self.state_size},), got {next_state.shape}"
            )
        if done:
            target_q = reward
        else:
            next_q_values = self.q_values(next_state)
            target_q = reward + self.gamma * float(np.max(next_q_values))

        n_actions = self.action_space.size
        x = np.zeros((1, self.state_size + n_actions), dtype=np.float64)
        x[0, : self.state_size] = state
        x[0, self.state_size + action_index] = 1.0

        y = np.array([[target_q]], dtype=np.float64)

        loss, _ = self.model.network.train_step(
            Tensor(x), Tensor(y), loss_fn="mse", optimizer=self._optimizer
        )

        self._step_count += 1
        self._epsilon = max(self.epsilon_end, self._epsilon * self.epsilon_decay)
        return loss

    def train_batch(
        self,
        states: np.ndarray,
        action_indices: list[int],
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        """Run a batch of Q-learning updates.

        Vectorised: computes Q-values for all next states in a single
        batch forward pass instead of looping per-sample.
        """
        batch_size = len(states)
        n_actions = self.action_space.size

        # Compute target Q-values in a single batch forward pass
        targets = np.zeros((batch_size, 1), dtype=np.float64)

        # Only compute next_q for non-terminal transitions
        non_terminal_mask = ~dones.astype(bool)
        if np.any(non_terminal_mask):
            # Batch Q-value computation for all non-terminal next states
            next_q_all = self.q_values_batch(next_states[non_terminal_mask])
            max_next_q = np.max(next_q_all, axis=1)
            targets[non_terminal_mask, 0] = rewards[non_terminal_mask] + self.gamma * max_next_q

        # Terminal transitions just get the reward
        terminal_mask = dones.astype(bool)
        targets[terminal_mask, 0] = rewards[terminal_mask]

        # Build input: [state, action_onehot]
        x = np.zeros((batch_size, self.state_size + n_actions), dtype=np.float64)
        x[:, : self.state_size] = states
        for i, action_idx in enumerate(action_indices):
            x[i, self.state_size + action_idx] = 1.0

        loss, _ = self.model.network.train_step(
            Tensor(x), Tensor(targets), loss_fn="mse", optimizer=self._optimizer
        )

        self._step_count += batch_size
        self._epsilon = max(self.epsilon_end, self._epsilon * self.epsilon_decay)
        return loss

    def decay_epsilon(self) -> float:
        self._epsilon = max(self.epsilon_end, self._epsilon * self.epsilon_decay)
        return self._epsilon

    def reset_epsilon(self) -> None:
        self._epsilon = self.epsilon_start
        self._step_count = 0

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self._model = MLP.load(path)
        self._optimizer = Adam(learning_rate=self.learning_rate)

    def param_count(self) -> int:
        return self.model.network.param_count()


# ---------------------------------------------------------------------------
# Action learning environment for simulation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ActionLearningEnv:
    """Simulation environment for testing action learning.

    State layout (4 elements):
        [face_x, face_y, face_detected, interaction_active]

    Actions: look_left, look_right, look_center, look_up, look_down,
             blink, wink, celebrate, sleep, look_around

    Reward structure:
        - celebrate when face detected: +1.0 (best)
        - look_center when face detected: +0.5 (good)
        - look_around when face detected: +0.2
        - look_left/right when face detected: +0.1
        - blink/wink: 0.0 (neutral)
        - celebrate when no face: -0.2 (embarrassing)
        - sleep: -0.5 (bad)
    """

    action_space: ActionSpace = field(default_factory=deskbot_action_space)
    seed: int = 42
    noise_std: float = 0.01
    _state: np.ndarray | None = field(default=None, init=False, repr=False)
    _rng: np.random.Generator | None = field(default=None, init=False, repr=False)
    _step_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._state = np.zeros(4, dtype=np.float64)
        self.reset()

    @property
    def state(self) -> np.ndarray:
        assert self._state is not None
        return self._state.copy()

    @property
    def state_size(self) -> int:
        return 4

    def reset(self) -> np.ndarray:
        self._state = np.array([0.5, 0.5, 1.0, 0.0], dtype=np.float64)
        self._step_count = 0
        assert self._state is not None
        return self._state.copy()

    def step(self, action_index: int) -> tuple[np.ndarray, float, bool]:
        assert self._rng is not None
        assert self._state is not None
        action = self.action_space.get(action_index)
        reward = 0.0
        noise = self._rng.normal(0, self.noise_std, 1)
        face_detected = self._state[2]

        if action.name == "look_left":
            self._state[0] = max(0.0, self._state[0] - 0.1)
            reward = 0.1 if face_detected > 0.5 else -0.05
        elif action.name == "look_right":
            self._state[0] = min(1.0, self._state[0] + 0.1)
            reward = 0.1 if face_detected > 0.5 else -0.05
        elif action.name == "look_center":
            self._state[0] = 0.5 + noise[0]
            reward = 0.5 if face_detected > 0.5 else -0.05
        elif action.name == "look_up":
            self._state[1] = max(0.0, self._state[1] - 0.1)
            reward = 0.05
        elif action.name == "look_down":
            self._state[1] = min(1.0, self._state[1] + 0.1)
            reward = 0.05
        elif action.name in {"blink", "wink"}:
            reward = 0.0
        elif action.name == "celebrate":
            reward = 1.0 if face_detected > 0.5 else -0.2
        elif action.name == "sleep":
            reward = -0.5
            self._state[2] = 0.0
        elif action.name == "look_around":
            reward = 0.2 if face_detected > 0.5 else 0.0
            if face_detected < 0.5:
                self._state[2] = min(1.0, self._state[2] + 0.3)

        self._state[0] = max(0.0, min(1.0, self._state[0]))
        self._state[1] = max(0.0, min(1.0, self._state[1]))
        self._state[2] = max(0.0, min(1.0, self._state[2]))

        self._step_count += 1
        done = self._step_count >= 200
        assert self._state is not None
        return self._state.copy(), reward, done

    def reward_for_action(
        self,
        action_index: int,
        state: np.ndarray | None = None,
    ) -> float:
        if state is None:
            assert self._state is not None
            state = self._state

        action = self.action_space.get(action_index)
        face_detected = state[2] > 0.5

        conditional_rewards = {
            "look_center": (0.5, -0.05),
            "celebrate": (1.0, -0.2),
            "look_around": (0.2, 0.0),
            "look_left": (0.1, -0.05),
            "look_right": (0.1, -0.05),
        }

        fixed_rewards = {
            "sleep": -0.5,
            "blink": 0.0,
            "wink": 0.0,
        }

        if action.name in conditional_rewards:
            with_face, without_face = conditional_rewards[action.name]
            return with_face if face_detected else without_face

        return fixed_rewards.get(action.name, 0.05)


__all__ = [
    "REWARD_NEGATIVE",
    "REWARD_NEUTRAL",
    "REWARD_POSITIVE",
    "ActionLearner",
    "ActionLearningEnv",
    "ActionSpace",
    "ActionValidator",
    "DefaultValidator",
    "LearningAction",
    "Reward",
    "RewardFunction",
    "deskbot_action_space",
]
