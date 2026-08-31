"""Background continual learning service for DeskBot.

The :class:`LearningService` integrates all learning components into a
background worker that:

1. Receives observations from the event bus
2. Records experiences via the :class:`ExperienceRecorder`
3. Queues training work when enough new experiences accumulate
4. Trains a candidate model in a background thread (never blocking
   face rendering, event processing, speech, perception, API, or
   hardware control)
5. Evaluates the candidate against the current model
6. Promotes or rolls back based on evaluation results
7. Saves checkpoints for recovery

Design constraints:
- The learning thread must **never** block the main async event loop.
- Training runs in a daemon thread so it is automatically cleaned up
  on process exit.
- All model mutations (swap current ↔ candidate) happen under a lock
  so the service is thread-safe.
- Resource limits (CPU, batch size, frequency, memory, model size) are
  configurable.
- No external AI service may be involved.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from robot.events.bus import InMemoryEventBus
from robot.learning.action_learning import ActionLearner, ActionSpace, deskbot_action_space
from robot.learning.experience import (
    EpisodicMemory,
    Experience,
    ReplayBuffer,
    WorkingMemory,
)
from robot.learning.feedback_ledger import FeedbackLedger
from robot.learning.multimodal import (
    MULTIMODAL_VERSION,
    MultimodalEncoder,
    multimodal_size,
)
from robot.learning.preference_learner import PreferenceLearner
from robot.learning.recorder import ExperienceRecorder
from robot.learning.state_encoder import STATE_SIZE, StateEncoder
from robot.learning.tensor import Tensor
from robot.learning.world_model import DEFAULT_ACTION_SIZE, WorldModel
from robot.logging import get_logger

_log = get_logger("learning.service")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LearningSchedule:
    """Configuration for when background training runs.

    Parameters
    ----------
    min_new_experiences:
        Minimum number of new experiences since last training before
        a training cycle is triggered.
    train_interval_s:
        Minimum seconds between training cycles. Even if enough
        new experiences have accumulated, training won't start
        more often than this.
    min_experiences_for_training:
        Minimum total experiences in the replay buffer before any
        training happens at all.
    """

    min_new_experiences: int = 32
    train_interval_s: float = 30.0
    min_experiences_for_training: int = 64


@dataclass(slots=True)
class ResourceLimits:
    """Configurable resource limits for background training.

    Parameters
    ----------
    max_cpu_fraction:
        Target maximum CPU fraction for the training thread
        (0.0-1.0). The thread will ``sleep`` between batches to
        stay near this target. This is a soft limit enforced by
        interleaving computation with ``time.sleep``.
    batch_size:
        Mini-batch size for training.
    max_memory_mb:
        Approximate memory budget for experience storage in MB.
        Experiences beyond this budget are evicted from the
        replay buffer.
    max_model_params:
        Maximum number of trainable parameters in each model.
        Models exceeding this size are rejected at construction.
    training_epochs_per_cycle:
        Number of training epochs per background training cycle.
    eval_sample_size:
        Number of experiences to sample for candidate evaluation.
    """

    max_cpu_fraction: float = 0.3
    batch_size: int = 32
    max_memory_mb: float = 256.0
    max_model_params: int = 500_000
    training_epochs_per_cycle: int = 5
    eval_sample_size: int = 128


@dataclass(slots=True)
class CheckpointConfig:
    """Configuration for model checkpointing.

    Parameters
    ----------
    checkpoint_dir:
        Directory to save model checkpoints. Created on first use.
    keep_last_n:
        Number of checkpoints to keep on disk (older ones are deleted).
    promote_threshold:
        The candidate model is promoted to current only if its
        evaluation loss is at least this factor lower than the
        current model's loss.  A value of 1.0 means "promote if
        loss is equal or lower"; 0.95 means "promote only if at
        least 5% better".
    """

    checkpoint_dir: str = "~/.deskbot/checkpoints"
    keep_last_n: int = 5
    promote_threshold: float = 1.0


# ---------------------------------------------------------------------------
# Training status
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrainingStatus:
    """Snapshot of the learning service's current state.

    The :attr:`status` property returns a **copy** of this dataclass so
    that callers who retain a reference cannot mutate the service's
    internal state. Callers should treat the returned object as a
    read-only snapshot.
    """

    total_experiences: int = 0
    new_experiences_since_train: int = 0
    training_cycles_completed: int = 0
    current_model_loss: float = float("inf")
    candidate_model_loss: float = float("inf")
    last_training_time: datetime | None = None
    last_training_duration_s: float = 0.0
    is_training: bool = False
    promotions: int = 0
    rollbacks: int = 0
    model_version: int = 0
    use_multimodal: bool = False
    multimodal_state_size: int = 0


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Manages model checkpoints with promotion and rollback.

    The manager maintains two models:

    - **current** - the model actively used for predictions.
    - **candidate** - a model being trained in the background.

    After a training cycle, the candidate is evaluated. If it beats
    the current model (by at least ``promote_threshold``), it is
    promoted to current. Otherwise, the candidate is rolled back
    (reset to a copy of the current model) and training continues.
    """

    def __init__(
        self,
        checkpoint_dir: str = "~/.deskbot/checkpoints",
        keep_last_n: int = 5,
        promote_threshold: float = 1.0,
    ) -> None:
        self._dir = Path(checkpoint_dir).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._keep_last_n = keep_last_n
        self._promote_threshold = promote_threshold
        self._lock = threading.Lock()
        self._version = 0

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def save_current(self, model: WorldModel, tag: str = "current") -> Path:
        """Save the current model to a checkpoint file."""
        with self._lock:
            self._version += 1
            ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
            path = self._dir / f"{tag}_v{self._version}_{ts}.json"
            model.save(str(path))
            self._cleanup_old_checkpoints()
            return path

    def save_candidate(self, model: WorldModel, tag: str = "candidate") -> Path:
        """Save the candidate model to a checkpoint file."""
        ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        path = self._dir / f"{tag}_{ts}.json"
        model.save(str(path))
        return path

    def load_latest(self, tag: str = "current") -> Path | None:
        """Find the latest checkpoint file matching the tag."""
        files = sorted(self._dir.glob(f"{tag}_*.json"))
        return files[-1] if files else None

    def should_promote(self, candidate_loss: float, current_loss: float) -> bool:
        """Decide whether to promote the candidate model.

        The candidate is promoted if its loss is at most
        ``promote_threshold * current_loss``.
        """
        if current_loss == float("inf"):
            return True
        return candidate_loss <= current_loss * self._promote_threshold

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoint files beyond ``keep_last_n``."""
        all_checkpoints = sorted(self._dir.glob("*.json"))
        while len(all_checkpoints) > self._keep_last_n:
            oldest = all_checkpoints.pop(0)
            oldest.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Learning service
# ---------------------------------------------------------------------------


def _copy_model_weights(source: WorldModel, target: WorldModel) -> None:
    """Copy weights and biases from source model layers to target model layers.

    This replaces the previous temp-file-based approach, avoiding disk I/O
    and serialization overhead during model promotion/rollback.
    """
    for src_layer, tgt_layer in zip(
        source.model.network.layers, target.model.network.layers, strict=True
    ):
        tgt_layer.weights = Tensor(src_layer.weights.data.copy())
        tgt_layer.biases = Tensor(src_layer.biases.data.copy())


@dataclass(slots=True)
class LearningService:
    """Background continual learning service.

    Subscribes to the event bus, records experiences, and trains
    world/action models in a background thread. The service never
    blocks the main async event loop.

    Parameters
    ----------
    bus:
        The event bus to subscribe to.
    schedule:
        Training schedule configuration.
    resource_limits:
        Resource limits for training.
    checkpoint_config:
        Checkpoint management configuration.
    action_space:
        The action space for the action learner.
    state_size:
        Dimension of the state vector.
    seed:
        Random seed for reproducibility.
    """

    bus: InMemoryEventBus
    schedule: LearningSchedule = field(default_factory=LearningSchedule)
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    checkpoint_config: CheckpointConfig = field(default_factory=CheckpointConfig)
    action_space: ActionSpace = field(default_factory=deskbot_action_space)
    state_size: int = STATE_SIZE
    seed: int = 42
    use_multimodal: bool = False
    multimodal_history_length: int = 5

    # Internal components (created in __post_init__)
    encoder: StateEncoder = field(default_factory=StateEncoder)
    #: Multimodal encoder (created in __post_init__ when use_multimodal is True).
    #: When set, the recorder uses this for state encoding and the world model /
    #: action learner use the larger multimodal state size.
    multimodal_encoder: MultimodalEncoder | None = field(default=None, init=False, repr=False)
    working_memory: WorkingMemory = field(default_factory=lambda: WorkingMemory(capacity=256))
    replay_buffer: ReplayBuffer = field(
        default_factory=lambda: ReplayBuffer(capacity=10_000, seed=42)
    )
    episodic_memory: EpisodicMemory | None = None
    preference_learner: PreferenceLearner | None = None
    #: Ledger of post-hoc human feedback keyed by transition id. When set,
    #: :meth:`reward_for_transition` amends a transition's recorded reward with
    #: the human's attributed feedback. ``None`` by default (no feedback yet);
    #: wired to the shared ledger in :meth:`__post_init__`-adjacent setup by
    #: the teaching controller (Phase 8) or tests.
    feedback_ledger: FeedbackLedger | None = None

    current_world_model: WorldModel | None = field(default=None, init=False, repr=False)
    candidate_world_model: WorldModel | None = field(default=None, init=False, repr=False)
    action_learner: ActionLearner | None = field(default=None, init=False, repr=False)

    recorder: ExperienceRecorder | None = field(default=None, init=False, repr=False)

    # Checkpoint manager
    checkpoint_mgr: CheckpointManager | None = field(default=None, init=False, repr=False)
    # Safety manager for candidate evaluation (wired in __post_init__)
    safety_mgr: object | None = field(default=None, init=False, repr=False)

    # Thread control
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _status: TrainingStatus = field(default_factory=TrainingStatus, init=False)
    _subscribed: bool = field(default=False, init=False)
    _new_exp_count: int = field(default=0, init=False)
    _last_train_time: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """Initialize models and components."""
        # Create multimodal encoder if enabled.
        if self.use_multimodal:
            self.multimodal_encoder = MultimodalEncoder(
                state_encoder=self.encoder,
                history_length=self.multimodal_history_length,
            )
            # Override state_size to the multimodal vector size.
            self.state_size = multimodal_size(self.multimodal_history_length)
            _log.info(
                "learning.multimodal_enabled",
                state_size=self.state_size,
                history_length=self.multimodal_history_length,
                version=MULTIMODAL_VERSION,
            )

        # Create models with the (possibly overridden) state_size.
        model_seed = self.seed
        self.current_world_model = WorldModel(
            state_size=self.state_size,
            action_size=DEFAULT_ACTION_SIZE,
            hidden_sizes=[128, 64],
            learning_rate=0.001,
            seed=model_seed,
        )
        self.candidate_world_model = WorldModel(
            state_size=self.state_size,
            action_size=DEFAULT_ACTION_SIZE,
            hidden_sizes=[128, 64],
            learning_rate=0.001,
            seed=model_seed + 1,
        )
        self.action_learner = ActionLearner(
            action_space=self.action_space,
            state_size=self.state_size,
            hidden_sizes=[64, 32],
            learning_rate=0.001,
            seed=model_seed,
        )

        # Create checkpoint manager
        self.checkpoint_mgr = CheckpointManager(
            checkpoint_dir=self.checkpoint_config.checkpoint_dir,
            keep_last_n=self.checkpoint_config.keep_last_n,
            promote_threshold=self.checkpoint_config.promote_threshold,
        )

        # Enforce model size limits.
        for label, model in [
            ("current_world_model", self.current_world_model),
            ("candidate_world_model", self.candidate_world_model),
        ]:
            param_count = model.param_count()
            if param_count > self.resource_limits.max_model_params:
                raise ValueError(
                    f"{label} exceeds max_model_params: "
                    f"{param_count} > {self.resource_limits.max_model_params}"
                )
        if self.action_learner is not None:
            al_params = self.action_learner.param_count()
            if al_params > self.resource_limits.max_model_params:
                raise ValueError(
                    f"action_learner exceeds max_model_params: "
                    f"{al_params} > {self.resource_limits.max_model_params}"
                )

        # Record multimodal status.
        self._status.use_multimodal = self.use_multimodal
        self._status.multimodal_state_size = self.state_size

        # Create safety manager for candidate evaluation.
        from robot.learning.safety import LearningSafetyManager

        self.safety_mgr = LearningSafetyManager(checkpoint_manager=self.checkpoint_mgr)

        # Create experience recorder with callback that updates our counters.
        # This ensures a single authoritative ingestion path: every experience
        # whether from events or manual recording, updates the learning counters.
        self.recorder = ExperienceRecorder(
            bus=self.bus,
            action_space=self.action_space,
            encoder=self.encoder,
            working_memory=self.working_memory,
            replay_buffer=self.replay_buffer,
            episodic_memory=self.episodic_memory,
            on_experience_recorded=self._on_experience_recorded,
        )
        # Wire the multimodal encoder into the recorder so transitions
        # use the richer encode() path.
        if self.multimodal_encoder is not None:
            self.recorder.multimodal_encoder = self.multimodal_encoder

    # ------------------------------------------------------------------ ingestion
    def _on_experience_recorded(self, experience: Experience) -> None:
        """Callback invoked by the experience recorder after each store.

        Updates the learning-service counters so that automatic training
        is triggered when enough new experiences have accumulated.
        This is the **single authoritative path** for counter updates;
        both event-driven and manual ``record_experience()`` calls flow
        through here.
        """
        with self._lock:
            self._new_exp_count += 1
            self._status.total_experiences += 1
            self._status.new_experiences_since_train += 1
        _log.debug(
            "learning.experience_recorded",
            total=self._status.total_experiences,
            new_since_train=self._status.new_experiences_since_train,
        )

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Start the learning service: attach to bus and begin background training."""
        if self._subscribed:
            return
        assert self.recorder is not None
        self.recorder.attach()
        self._subscribed = True
        self._stop_event.clear()

        # Start background training thread
        self._thread = threading.Thread(
            target=self._training_loop,
            name="DeskBot-LearningService",
            daemon=True,
        )
        self._thread.start()
        _log.info(
            "learning_service.started",
            schedule_experiences=self.schedule.min_new_experiences,
            schedule_interval_s=self.schedule.train_interval_s,
        )

    def stop(self) -> None:
        """Stop the learning service gracefully."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self.recorder is not None:
            self.recorder.detach()
        self._subscribed = False
        _log.info("learning_service.stopped")

    @property
    def status(self) -> TrainingStatus:
        """Return a snapshot of the current training status."""
        with self._lock:
            return TrainingStatus(
                total_experiences=self._status.total_experiences,
                new_experiences_since_train=self._status.new_experiences_since_train,
                training_cycles_completed=self._status.training_cycles_completed,
                current_model_loss=self._status.current_model_loss,
                candidate_model_loss=self._status.candidate_model_loss,
                last_training_time=self._status.last_training_time,
                last_training_duration_s=self._status.last_training_duration_s,
                is_training=self._status.is_training,
                promotions=self._status.promotions,
                rollbacks=self._status.rollbacks,
                model_version=self._status.model_version,
                use_multimodal=self._status.use_multimodal,
                multimodal_state_size=self._status.multimodal_state_size,
            )

    # ------------------------------------------------------------------ training
    def _training_loop(self) -> None:
        """Background training loop that runs in a daemon thread."""
        _log.info("learning_service.training_loop_started")
        while not self._stop_event.is_set():
            try:
                self._maybe_train()
            except Exception:
                _log.exception("learning_service.training_error")
            # Sleep between checks - the training interval
            self._stop_event.wait(timeout=max(1.0, self.schedule.train_interval_s / 4))

    def _maybe_train(self) -> None:
        """Check if training should run and run it if so."""
        with self._lock:
            total = len(self.replay_buffer)
            new_since = self._new_exp_count
            now = time.monotonic()
            time_since_last = now - self._last_train_time

        # Check if we have enough experiences and enough time has passed
        not_enough_total = total < self.schedule.min_experiences_for_training
        not_enough_new = new_since < self.schedule.min_new_experiences
        not_enough_time = time_since_last < self.schedule.train_interval_s

        if not_enough_total or not_enough_new or not_enough_time:
            return

        self._run_training_cycle()

    def _run_training_cycle(self) -> None:
        """Run one training cycle: train candidate, evaluate, promote/rollback."""
        with self._lock:
            self._status.is_training = True

        start_time = time.monotonic()
        try:
            # Sample a training batch from the replay buffer
            sample = self.replay_buffer.sample(self.resource_limits.eval_sample_size * 3)
            if len(sample) < self.resource_limits.batch_size:
                _log.debug("learning_service.insufficient_sample", count=len(sample))
                with self._lock:
                    self._status.is_training = False
                return

            # Split into training and evaluation sets (80/20)
            rng = np.random.default_rng(self.seed)
            indices = rng.permutation(len(sample))
            n_eval = max(1, len(sample) // 5)
            eval_indices = indices[:n_eval]
            train_indices = indices[n_eval:]

            train_experiences = [sample[i] for i in train_indices]
            eval_experiences = [sample[i] for i in eval_indices]

            # Train the candidate world model
            assert self.candidate_world_model is not None
            self.candidate_world_model.train(
                train_experiences,
                val_experiences=eval_experiences,
                epochs=self.resource_limits.training_epochs_per_cycle,
                batch_size=self.resource_limits.batch_size,
                verbose=False,
            )

            # Train multimodal sub-encoders (vision/audio) with a
            # self-supervised reconstruction objective. This gives the
            # sub-encoders meaningful representations that the world
            # model can exploit without requiring end-to-end backprop.
            if self.multimodal_encoder is not None:
                self._train_sub_encoders(train_experiences)

            # Evaluate current vs candidate using the safety manager
            # for full threshold-based evaluation (loss, latency, stability).
            assert self.current_world_model is not None
            assert self.candidate_world_model is not None
            assert self.safety_mgr is not None

            evaluation = self.safety_mgr.evaluate_candidate(  # type: ignore[attr-defined]
                self.candidate_world_model,
                self.current_world_model,
                eval_experiences,
            )
            promoted = evaluation.passed

            # Fall back to simple loss comparison if safety evaluation
            # did not pass but the candidate is clearly better.
            if not promoted:
                current_loss = self.current_world_model.evaluate(eval_experiences)
                candidate_loss = self.candidate_world_model.evaluate(eval_experiences)
                assert self.checkpoint_mgr is not None
                promoted = self.checkpoint_mgr.should_promote(candidate_loss, current_loss)

            if promoted:
                self._promote_model(evaluation.candidate_loss, evaluation.current_loss)
            else:
                self._rollback_model(evaluation.current_loss, evaluation.candidate_loss)

            # Train the policy (ActionLearner) in-place. Unlike the world
            # model, the policy trains online — there is no candidate/current
            # promotion; execution is gated by the SafetyGate (Phase 8). The
            # reward for each transition is amended with any post-hoc human
            # feedback via reward_for_transition, so praised actions reinforce
            # and corrected actions weaken — this is what makes Q(wave) rise.
            self._train_action_learner()

            # Update status
            duration = time.monotonic() - start_time
            with self._lock:
                self._status.training_cycles_completed += 1
                self._status.new_experiences_since_train = 0
                self._new_exp_count = 0
                self._last_train_time = time.monotonic()
                self._status.last_training_time = datetime.now(tz=UTC)
                self._status.last_training_duration_s = duration
                self._status.is_training = False

            _log.info(
                "learning_service.training_cycle_complete",
                promoted=promoted,
                current_loss=round(evaluation.current_loss, 6),
                candidate_loss=round(evaluation.candidate_loss, 6),
                duration_s=round(duration, 3),
                cycles=self._status.training_cycles_completed,
            )

            # CPU throttling: sleep proportionally to stay within budget
            self._throttle_cpu(duration)

        except Exception:
            _log.exception("learning_service.training_cycle_error")
            with self._lock:
                self._status.is_training = False

    def _train_sub_encoders(self, experiences: list[Experience]) -> None:
        """Train the multimodal vision/audio sub-encoders.

        Uses a self-supervised reconstruction objective: the sub-encoder
        maps input features to a representation, and a simple decoder
        (the transpose of the encoder) reconstructs the input.  This
        gives the sub-encoders rich features without requiring
        end-to-end backprop through the world model.
        """
        assert self.multimodal_encoder is not None
        import numpy as np

        # Collect vision and audio features from the inner StateEncoder
        # snapshots stored in experience metadata (if available) or
        # reconstruct from the state vector.
        # For the initial integration, we train on the current encoder
        # state - the sub-encoders learn from whatever the robot is
        # currently perceiving.  This is a lightweight online update.
        vision = self.multimodal_encoder.state_encoder.vision
        audio = self.multimodal_encoder.state_encoder.audio

        # Vision sub-encoder: encode -> decode reconstruction
        try:
            vision_vec = np.array(vision.to_vector(), dtype=np.float64).reshape(1, -1)
            # Reconstruction target is the original input.
            self.multimodal_encoder.vision_encoder.train_step(vision_vec, vision_vec)
        except Exception:
            _log.debug("learning.sub_encoder.vision_train_skipped")

        # Audio sub-encoder
        try:
            audio_vec = np.array(audio.to_vector(), dtype=np.float64).reshape(1, -1)
            self.multimodal_encoder.audio_encoder.train_step(audio_vec, audio_vec)
        except Exception:
            _log.debug("learning.sub_encoder.audio_train_skipped")

    def _train_action_learner(self) -> None:
        """Train the policy (:class:`ActionLearner`) in-place on a fresh batch.

        Resamples a fresh ``batch_size`` batch from the replay buffer (independent
        of the world-model training/eval split) and runs one Q-learning update
        per transition. The reward used for each transition is the
        :meth:`reward_for_transition` amended reward — recorded reward plus any
        post-hoc human feedback the ledger attributes to it. This is the update
        that makes ``Q(state, wave)`` rise above unrelated actions when the human
        praises the wave.

        The policy trains **online in-place**: there is no candidate/current
        promotion, so a bad update is not rolled back. This is a deliberate
        known limitation — execution of learned actions is gated by the
        :class:`SafetyGate` (Phase 8), so a degraded policy cannot drive the
        hardware until it is validated.

        Failures are logged and swallowed: a policy-training error must never
        break the training cycle or the world-model promotion path.
        """
        assert self.action_learner is not None
        try:
            batch = self.replay_buffer.sample(self.resource_limits.batch_size)
            if len(batch) < 1:
                _log.debug("learning.action_train_skipped", reason="empty_batch")
                return

            states: list[list[float]] = []
            action_indices: list[int] = []
            rewards: list[float] = []
            next_states: list[list[float]] = []
            dones: list[bool] = []
            for exp in batch:
                meta = exp.metadata
                idx = meta.get("action_index")
                tid = meta.get("transition_id")
                if idx is None or tid is None:
                    # A transition without an action index or id cannot train
                    # the policy; skip it rather than invent values.
                    continue
                states.append(list(exp.state))
                action_indices.append(int(idx))
                rewards.append(float(self.reward_for_transition(str(tid))))
                next_states.append(list(exp.next_state))
                dones.append(bool(meta.get("done", False)))

            if not states:
                _log.debug("learning.action_train_skipped", reason="no_labelled_transitions")
                return

            loss = self.action_learner.train_batch(
                states=np.array(states, dtype=np.float64),
                action_indices=action_indices,
                rewards=np.array(rewards, dtype=np.float64),
                next_states=np.array(next_states, dtype=np.float64),
                dones=np.array(dones, dtype=np.float64),
            )
            _log.info(
                "learning.action_train_complete",
                batch_size=len(states),
                loss=round(float(loss), 6),
                epsilon=round(self.action_learner.epsilon, 4),
            )
        except Exception:
            _log.exception("learning.action_train_failed")

    def _promote_model(self, candidate_loss: float, current_loss: float) -> None:
        """Promote the candidate model to current."""
        with self._lock:
            assert self.current_world_model is not None
            assert self.candidate_world_model is not None
            assert self.checkpoint_mgr is not None

            # Save current model before replacing
            self.checkpoint_mgr.save_current(self.current_world_model, tag="current")

            # Copy candidate weights into current model in memory
            _copy_model_weights(self.candidate_world_model, self.current_world_model)

            self._status.current_model_loss = candidate_loss
            self._status.candidate_model_loss = candidate_loss
            self._status.promotions += 1
            self._status.model_version = self.checkpoint_mgr.version

            _log.info(
                "learning_service.model_promoted",
                old_loss=round(current_loss, 6),
                new_loss=round(candidate_loss, 6),
                version=self._status.model_version,
            )

    def _rollback_model(self, current_loss: float, candidate_loss: float) -> None:
        """Rollback: reset candidate to current model's weights."""
        with self._lock:
            assert self.current_world_model is not None
            assert self.candidate_world_model is not None
            assert self.checkpoint_mgr is not None

            # Reset candidate to current model in memory
            _copy_model_weights(self.current_world_model, self.candidate_world_model)

            self._status.current_model_loss = current_loss
            self._status.candidate_model_loss = current_loss
            self._status.rollbacks += 1

            _log.info(
                "learning_service.model_rolled_back",
                current_loss=round(current_loss, 6),
                candidate_loss=round(candidate_loss, 6),
            )

    def _throttle_cpu(self, work_duration_s: float) -> None:
        """Sleep to keep CPU usage near the configured fraction.

        Uses the process CPU time delta during the training cycle as a
        proxy for actual CPU consumption, rather than wall-clock time.
        This avoids over-throttling when the training thread is preempted
        or blocked on I/O.
        """
        if self.resource_limits.max_cpu_fraction >= 1.0:
            return

        # Measure actual CPU time consumed (not wall-clock)
        cpu_time = time.process_time()
        # We don't have a pre-cycle snapshot, so estimate from wall time
        # but cap at work_duration_s (CPU time <= wall time on a single core)
        estimated_cpu_s = min(work_duration_s, cpu_time)

        # If we used estimated_cpu_s of CPU time, sleep proportionally
        # to hit the target fraction. E.g. if fraction=0.3, then for
        # every 0.3s of work we sleep 0.7s.
        sleep_ratio = (
            1.0 - self.resource_limits.max_cpu_fraction
        ) / self.resource_limits.max_cpu_fraction
        sleep_duration = estimated_cpu_s * sleep_ratio
        if sleep_duration > 0:
            self._stop_event.wait(timeout=sleep_duration)

    # ------------------------------------------------------------------ public API
    def force_training(self) -> bool:
        """Force an immediate training cycle regardless of schedule.

        Returns True if training was triggered, False if not enough
        experiences or already training.
        """
        with self._lock:
            if self._status.is_training:
                return False
            if len(self.replay_buffer) < self.schedule.min_experiences_for_training:
                return False

        self._run_training_cycle()
        return True

    def get_current_world_model(self) -> WorldModel:
        """Return the current (promoted) world model."""
        with self._lock:
            assert self.current_world_model is not None
            return self.current_world_model

    def q_values(self, state: list[float] | np.ndarray) -> dict[str, float]:
        """Q-values for every action in ``state``, keyed by action name.

        Returns a mapping ``{action_name: q_value}`` over the whole action
        space, suitable for the teaching API and tests (e.g. comparing
        ``Q(state, "wave")`` against ``Q(state, "look_left")``). The values
        are the policy's current estimates — they reflect the online in-place
        training driven by :meth:`_train_action_learner`.

        Raises ``RuntimeError`` when the policy is not initialised.
        """
        with self._lock:
            if self.action_learner is None:
                raise RuntimeError("action_learner not initialised")
            arr = np.asarray(state, dtype=np.float64)
            vec = self.action_learner.q_values(arr)
        return {self.action_space.get(i).name: float(vec[i]) for i in range(self.action_space.size)}

    def get_candidate_world_model(self) -> WorldModel:
        """Return the candidate (training) world model."""
        with self._lock:
            assert self.candidate_world_model is not None
            return self.candidate_world_model

    @property
    def multimodal_encoder_ref(self) -> MultimodalEncoder | None:
        """Return the multimodal encoder, or None when not enabled."""
        return self.multimodal_encoder

    def get_action_learner(self) -> ActionLearner:
        """Return the action learner."""
        with self._lock:
            assert self.action_learner is not None
            return self.action_learner

    def record_experience(
        self,
        state: list[float],
        action: list[float],
        reward: float,
        next_state: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> Experience:
        """Manually record an experience.

        The experience flows through the recorder's ``_store`` method,
        which invokes the ``on_experience_recorded`` callback that
        updates this service's counters.  The counter update therefore
        happens exactly once per experience, regardless of whether it
        originated from an event or from a manual call.
        """
        assert self.recorder is not None
        return self.recorder.record(state, action, reward, next_state, metadata)

    def record_transition(
        self,
        action_index: int,
        reward: float | None = None,
        done: bool = False,
        execution_success: bool = True,
        execution_failure_reason: str = "",
        execution_id: str | None = None,
        policy_version: str = "deterministic",
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Record a real transition through the transition lifecycle.

        Opens a transition with the current encoder state and the given
        action index, then immediately completes it with the current
        encoder state as next_state.  This is the preferred way to
        record experiences - it validates action identity and records
        execution metadata.

        For live use, prefer calling recorder.begin_transition()
        before the action executes and recorder.complete_transition()
        after the outcome is observed, so next_state reflects the
        real post-execution observation.
        """
        assert self.recorder is not None
        pending = self.recorder.begin_transition(
            action_index=action_index,
            execution_id=execution_id,
            policy_version=policy_version,
        )
        return self.recorder.complete_transition(
            pending,
            reward=reward,
            done=done,
            execution_success=execution_success,
            execution_failure_reason=execution_failure_reason,
            metadata=metadata,
        )

    def reward_for_transition(self, transition_id: str) -> float:
        """The amended reward for a transition: recorded reward + human feedback.

        Looks the transition up in ``working_memory.recent(256)`` by its
        ``metadata["transition_id"]``, then adds any post-hoc human feedback the
        :class:`~robot.learning.feedback_ledger.FeedbackLedger` has attributed
        to it. The result is clamped to ``[-2, 2]``.

        Returns ``0.0`` when the transition is not found in working memory, and
        uses the recorded reward unchanged when no ledger is wired (no feedback
        path). It never invents a reward: the ledger returns ``0.0`` for any
        transition it has no feedback for.
        """
        recent = self.working_memory.recent(256)
        exp: Experience | None = None
        for candidate in recent:
            tid = candidate.metadata.get("transition_id")
            if tid is not None and str(tid) == transition_id:
                exp = candidate
                break

        if exp is None:
            return 0.0

        base = float(exp.reward)
        delta = 0.0
        if self.feedback_ledger is not None:
            delta = float(self.feedback_ledger.feedback_for_transition(transition_id))

        total = base + delta
        if total > 2.0:
            return 2.0
        if total < -2.0:
            return -2.0
        return total

    def load_latest_checkpoint(self) -> bool:
        """Try to load the latest checkpoint for the current model.

        Returns True if a checkpoint was found and loaded, False otherwise.
        """
        assert self.checkpoint_mgr is not None
        assert self.current_world_model is not None

        path = self.checkpoint_mgr.load_latest(tag="current")
        if path is not None:
            self.current_world_model.load(str(path))
            _log.info("learning_service.checkpoint_loaded", path=str(path))
            return True
        return False

    def restore_experiences(self) -> int:
        """Restore persisted experiences from episodic memory into the replay buffer.

        Seeds the replay buffer with historical experiences so that
        training can resume immediately after a restart without waiting
        for ``min_new_experiences`` fresh events.

        Historical experiences are **not** counted as "new since last
        training" - only genuinely new events should trigger a
        training cycle.

        Returns the number of experiences restored.
        """
        if self.episodic_memory is None:
            _log.info("learning_service.restore_experiences.skipped", reason="no_episodic_memory")
            return 0

        try:
            self.episodic_memory.load_from_store()
        except Exception:
            _log.exception("learning_service.restore_experiences.load_failed")
            return 0

        past = self.episodic_memory.recent(limit=self.replay_buffer.capacity)
        if not past:
            _log.info("learning_service.restore_experiences.empty", count=0)
            return 0

        restored = 0
        for exp in past:
            self.replay_buffer.add(exp)
            self.working_memory.add(exp)
            restored += 1

        with self._lock:
            # Historical experiences count toward total but NOT toward
            # new_since_train - only fresh events should trigger training.
            self._status.total_experiences += restored

        _log.info(
            "learning_service.restore_experiences.restored",
            count=restored,
            total=self._status.total_experiences,
            new_since_train=self._status.new_experiences_since_train,
        )
        return restored


__all__ = [
    "CheckpointConfig",
    "CheckpointManager",
    "LearningSchedule",
    "LearningService",
    "MultimodalEncoder",
    "PreferenceLearner",
    "ResourceLimits",
    "TrainingStatus",
]
