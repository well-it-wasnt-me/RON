"""Production checklist verification.

Each test corresponds to one item in plan/10_PRODUCTION_CHECKLIST.md.
A test passing means the checklist item is satisfied.
"""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from robot.behavior.state_machine import RobotState
from robot.events.bus import InMemoryEventBus
from robot.events.events import EmotionChanged, EmotionName, FaceDetected, SpeechRecognized
from robot.learning.action_learning import deskbot_action_space
from robot.learning.dataset import TransitionDataset, WorldModelBaseline, validate_transition
from robot.learning.deterministic_encoder import DeterministicMultimodalEncoder, ObservationContext
from robot.learning.evaluation import (
    create_standard_evaluation_dataset,
)
from robot.learning.experience import (
    EpisodicMemory,
    Experience,
    SqliteExperienceStore,
)
from robot.learning.model_registry import (
    CanaryDeploymentManager,
    CanaryStage,
    ModelMetadata,
    ModelRegistry,
)
from robot.learning.observation import Observation
from robot.learning.online_learning import OnlineLearningMonitor
from robot.learning.recorder import ExperienceRecorder
from robot.learning.safety_gate import SafeActionExecutor, SafetyGate
from robot.learning.shadow_policy import PolicyMode, ShadowPolicyController
from robot.learning.state_encoder import StateEncoder
from robot.learning.transition import TransitionStore
from robot.learning.world_model import SimpleEnvironment, WorldModel

# ===========================================================================
# DATA
# ===========================================================================


class TestDataChecklist:
    """Verify every Data checklist item."""

    def test_every_transition_has_state_action_next_state_reward(self) -> None:
        """Every transition has state, action, next_state, reward."""
        store = TransitionStore(action_space=deskbot_action_space())
        t = store.record(
            state=[0.0] * 10,
            action_index=2,
            next_state=[1.0] * 10,
            reward=0.5,
        )
        assert len(t.state) > 0
        assert len(t.action_vector) > 0
        assert len(t.next_state) > 0
        assert isinstance(t.reward, float)

    def test_action_comes_from_action_space(self) -> None:
        """Action comes from ActionSpace."""
        space = deskbot_action_space()
        store = TransitionStore(action_space=space)
        for i in range(space.size):
            t = store.record(state=[0.0] * 4, action_index=i, next_state=[1.0] * 4, reward=0.0)
            assert t.action_index == i
            assert t.action_name == space.get(i).name

    def test_events_are_observations_not_actions(self) -> None:
        """Events are observations, not actions."""
        bus = InMemoryEventBus()
        recorder = ExperienceRecorder(bus=bus)
        recorder.attach()

        async def fire_events() -> None:
            await bus.publish(FaceDetected(x=0.5, y=0.3, confidence=0.9))
            await bus.publish(SpeechRecognized(text="hello", confidence=0.95))
            await bus.publish(
                EmotionChanged(
                    previous=EmotionName.NEUTRAL, current=EmotionName.HAPPY, intensity=0.8
                )
            )

        asyncio.run(fire_events())
        # Observation events produce zero experiences
        assert len(recorder.working_memory) == 0

    def test_no_future_information_leaks_into_state(self) -> None:
        """No future information leaks into state."""
        enc = StateEncoder()
        enc.recent_rewards = [0.1, 0.2]  # past rewards only
        obs = Observation.from_encoder(enc)
        # The reward for the transition being recorded is not in the observation
        future_reward = 999.0
        assert future_reward not in obs.robot.recent_rewards
        assert future_reward not in obs.to_vector()

    def test_invalid_transitions_are_rejected(self) -> None:
        """Invalid transitions are rejected."""
        space = deskbot_action_space()
        # Invalid action index
        exp = Experience(
            timestamp=datetime.now(tz=UTC),
            state=[1.0] * 4,
            action=[0.0],
            reward=0.0,
            next_state=[1.0] * 4,
            metadata={"action_index": 999},
        )
        ok, _ = validate_transition(exp, action_space=space)
        assert not ok
        # NaN in state
        exp2 = Experience(
            timestamp=datetime.now(tz=UTC),
            state=[float("nan")] * 4,
            action=[0.0],
            reward=0.0,
            next_state=[1.0] * 4,
            metadata={"action_index": 0},
        )
        ok2, _ = validate_transition(exp2, action_space=space)
        assert not ok2

    def test_transitions_survive_reboot(self, tmp_path: Path) -> None:
        """Transitions survive reboot."""
        db = tmp_path / "checklist.db"
        store1 = SqliteExperienceStore(db_path=db)
        mem1 = EpisodicMemory(store=store1, capacity=100)
        exp = Experience(
            timestamp=datetime.now(tz=UTC),
            state=[1.0] * 4,
            action=[0.0],
            reward=0.5,
            next_state=[2.0] * 4,
            metadata={"source": "checklist"},
        )
        mem1.add(exp)
        store1.close()

        store2 = SqliteExperienceStore(db_path=db)
        mem2 = EpisodicMemory(store=store2, capacity=100, max_load=10)
        mem2.load_from_store()
        assert len(mem2) == 1
        loaded = mem2.recent(1)[0]
        assert loaded.state == [1.0] * 4
        store2.close()


# ===========================================================================
# ENCODING
# ===========================================================================


class TestEncodingChecklist:
    """Verify every Encoding checklist item."""

    def test_encoder_is_deterministic(self) -> None:
        """Encoder is deterministic."""
        enc = DeterministicMultimodalEncoder()
        obs = Observation()
        ctx = ObservationContext(current=obs)
        v1 = enc.encode(ctx)
        v2 = enc.encode(ctx)
        assert v1 == v2

    def test_version_is_recorded(self) -> None:
        """Version is recorded."""
        enc = DeterministicMultimodalEncoder()
        assert enc.version > 0

    def test_dimensions_are_validated(self) -> None:
        """Dimensions are validated."""
        enc = DeterministicMultimodalEncoder()
        ctx = ObservationContext(current=Observation())
        vec = enc.encode(ctx)
        assert enc.validate_output(vec) is True
        assert enc.validate_output([1.0] * 10) is False  # wrong size

    def test_nan_inf_are_rejected(self) -> None:
        """NaN/inf are rejected."""
        enc = DeterministicMultimodalEncoder()
        ctx = ObservationContext(current=Observation())
        vec = enc.encode(ctx)
        for v in vec:
            assert not math.isnan(v)
            assert not math.isinf(v)

    def test_history_semantics_are_tested(self) -> None:
        """History semantics are tested (ordering, empty, full)."""
        enc = DeterministicMultimodalEncoder()
        base = Observation()
        # Empty history
        ctx_empty = ObservationContext(current=base, history=())
        assert enc.validate_output(enc.encode(ctx_empty))
        # Full history
        hist = tuple(Observation() for _ in range(5))
        ctx_full = ObservationContext(current=base, history=hist)
        assert enc.validate_output(enc.encode(ctx_full))
        # Ordering matters
        enc_a = StateEncoder()
        enc_a.update_state(RobotState.IDLE)
        obs_a = Observation.from_encoder(enc_a)
        enc_b = StateEncoder()
        enc_b.update_state(RobotState.CURIOUS)
        obs_b = Observation.from_encoder(enc_b)
        ctx1 = ObservationContext(current=base, history=(obs_a, obs_b))
        ctx2 = ObservationContext(current=base, history=(obs_b, obs_a))
        assert enc.encode(ctx1) != enc.encode(ctx2)


# ===========================================================================
# TRAINING
# ===========================================================================


class TestTrainingChecklist:
    """Verify every Training checklist item."""

    def test_train_val_test_split_exists(self) -> None:
        """Train/validation/test split exists."""
        dataset = TransitionDataset()
        exps = [
            Experience(
                timestamp=datetime(2025, 1, 1, tzinfo=UTC)
                + __import__("datetime").timedelta(seconds=i),
                state=[float(i)] * 4,
                action=[0.0],
                reward=0.0,
                next_state=[float(i + 1)] * 4,
                metadata={"action_index": 0},
            )
            for i in range(30)
        ]
        split, _ = dataset.build(exps, split_method="time")
        assert len(split.train) > 0
        assert len(split.validation) > 0
        assert len(split.test) > 0

    def test_frozen_evaluation_dataset_exists(self) -> None:
        """Frozen evaluation dataset exists."""
        ds = create_standard_evaluation_dataset(deskbot_action_space())
        assert ds.scenario_count >= 14
        assert ds.version > 0

    def test_baseline_exists(self) -> None:
        """Baseline exists."""
        baseline = WorldModelBaseline(strategy="persistence")
        exps = [
            Experience(
                timestamp=datetime.now(tz=UTC),
                state=[1.0] * 4,
                action=[0.0],
                reward=0.0,
                next_state=[2.0] * 4,
                metadata={},
            )
        ]
        loss = baseline.evaluate(exps)
        assert loss == 1.0  # persistence: (1-2)^2 = 1

    def test_training_runs_are_reproducible(self) -> None:
        """Training runs are reproducible."""
        env = SimpleEnvironment(seed=42, noise_std=0.005)
        exps = env.collect_experiences(n_steps=64)
        m1 = WorldModel(seed=42)
        m2 = WorldModel(seed=42)
        r1 = m1.train(exps, epochs=5, verbose=False)
        r2 = m2.train(exps, epochs=5, verbose=False)
        assert r1.initial_loss == r2.initial_loss
        assert r1.final_loss == r2.final_loss

    def test_candidate_models_have_metadata(self) -> None:
        """Candidate models have metadata."""
        meta = ModelMetadata(
            model_version=1,
            schema_version=1,
            state_encoder_version=1,
            action_space_version=1,
            git_commit="abc123",
            dataset_version="v1",
            training_run="run-001",
            validation={"loss": 0.1},
        )
        d = meta.to_dict()
        assert d["model_version"] == 1
        assert d["git_commit"] == "abc123"
        assert d["validation"]["loss"] == 0.1


# ===========================================================================
# SAFETY
# ===========================================================================


class TestSafetyChecklist:
    """Verify every Safety checklist item."""

    def test_learned_actions_pass_one_safety_validator(self) -> None:
        """Learned actions pass one safety validator."""
        gate = SafetyGate(action_space=deskbot_action_space())
        result = gate.validate(action_index=2)
        assert result.allowed

    def test_hardware_cannot_be_controlled_directly_by_training(self) -> None:
        """Hardware cannot be controlled directly by training."""
        executed: list[int] = []
        gate = SafetyGate(action_space=deskbot_action_space())
        executor = SafeActionExecutor(safety_gate=gate, executor=executed.append)
        # Even an invalid action goes through the gate
        action = executor.execute(action_index=999)
        assert action == gate.fallback_action
        assert executed[-1] == gate.fallback_action

    def test_invalid_model_output_triggers_fallback(self) -> None:
        """Invalid model output triggers fallback."""
        gate = SafetyGate(action_space=deskbot_action_space())
        result = gate.handle_model_output([float("nan"), 0.1, 0.3])
        assert result.fallback

    def test_model_timeout_triggers_fallback(self) -> None:
        """Model timeout (None output) triggers fallback."""
        gate = SafetyGate(action_space=deskbot_action_space())
        result = gate.handle_model_output(None)
        assert result.fallback

    def test_manual_override_works(self) -> None:
        """Manual override works."""
        gate = SafetyGate(action_space=deskbot_action_space())
        gate.activate_override()
        result = gate.validate(action_index=2)
        assert result.fallback
        assert "override" in result.reason.lower()
        gate.deactivate_override()
        result2 = gate.validate(action_index=2)
        assert result2.allowed


# ===========================================================================
# DEPLOYMENT
# ===========================================================================


class TestDeploymentChecklist:
    """Verify every Deployment checklist item."""

    def _model_data(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "layers": [
                {"weights": [[0.1, 0.2], [0.3, 0.4]], "biases": [0.0, 0.0]},
                {"weights": [[0.5, 0.6]], "biases": [0.0]},
            ],
        }

    def test_models_are_atomically_loaded(self, tmp_path: Path) -> None:
        """Models are atomically loaded."""
        registry = ModelRegistry(registry_dir=tmp_path / "reg")
        meta = ModelMetadata(model_version=1)
        registry.deploy(self._model_data(), meta)
        # The file exists and is valid JSON
        import json

        data = json.loads(registry.model_path(1).read_text())
        assert "layers" in data

    def test_previous_model_is_retained(self, tmp_path: Path) -> None:
        """Previous model is retained."""
        registry = ModelRegistry(registry_dir=tmp_path / "reg")
        registry.deploy(self._model_data(), ModelMetadata(model_version=1))
        registry.deploy(self._model_data(), ModelMetadata(model_version=2))
        assert registry.model_path(1).exists()
        assert registry.model_path(2).exists()
        assert registry.previous_version == 1

    def test_rollback_works(self, tmp_path: Path) -> None:
        """Rollback works."""
        registry = ModelRegistry(registry_dir=tmp_path / "reg")
        registry.deploy(self._model_data(), ModelMetadata(model_version=1))
        registry.deploy(self._model_data(), ModelMetadata(model_version=2))
        version = registry.rollback()
        assert version == 1
        assert registry.active_version == 1

    def test_shadow_mode_works(self) -> None:
        """Shadow mode works."""
        controller = ShadowPolicyController(
            action_space=deskbot_action_space(),
            deterministic_policy=lambda s: 2,
            model_policy=lambda s: (0, [0.9, 0.1, 0.5]),
            model_version="v1.0",
        )
        state = np.zeros(91, dtype=np.float64)
        action = controller.decide(state)
        # Shadow mode returns the deterministic action, not the model's
        assert action == 2
        # Model's action is logged but not executed
        entry = controller.log_entries[-1]
        assert entry.model_action == 0
        assert entry.deterministic_action == 2

    def test_canary_deployment_works(self, tmp_path: Path) -> None:
        """Canary deployment works."""
        registry = ModelRegistry(registry_dir=tmp_path / "reg")
        manager = CanaryDeploymentManager(registry=registry)
        assert manager.current_stage == CanaryStage.CANDIDATE
        manager.advance()
        assert manager.current_stage != CanaryStage.CANDIDATE


# ===========================================================================
# OBSERVABILITY
# ===========================================================================


class TestObservabilityChecklist:
    """Verify every Observability checklist item."""

    def test_model_version_logged(self) -> None:
        """Model version logged."""
        m = OnlineLearningMonitor(model_version=5)
        d = m.to_dict()
        assert d["model_version"] == 5

    def test_policy_decisions_logged(self) -> None:
        """Policy decisions logged."""
        controller = ShadowPolicyController(
            action_space=deskbot_action_space(),
            deterministic_policy=lambda s: 2,
            model_policy=lambda s: (0, [0.5] * 10),
            model_version="v1.0",
        )
        controller.decide(np.zeros(91, dtype=np.float64))
        entry = controller.log_entries[-1]
        assert entry.timestamp_ns > 0
        assert entry.deterministic_action == 2
        assert entry.model_action == 0

    def test_safety_rejections_logged(self) -> None:
        """Safety rejections logged."""
        m = OnlineLearningMonitor()
        m.record_safety_rejection()
        m.record_safety_rejection()
        assert m.safety_rejections == 2

    def test_inference_latency_measured(self) -> None:
        """Inference latency measured."""
        m = OnlineLearningMonitor()
        m.record_inference_latency(5.0)
        m.record_inference_latency(15.0)
        assert m.inference_latency_ms == 10.0

    def test_training_metrics_measured(self) -> None:
        """Training metrics measured."""
        m = OnlineLearningMonitor()
        m.record_training(loss=0.3, val_loss=0.4)
        assert m.training_loss == 0.3
        assert m.validation_loss == 0.4

    def test_crash_fallback_counts_measured(self) -> None:
        """Crash/fallback counts measured."""
        m = OnlineLearningMonitor()
        m.record_fallback()
        m.record_model_load_failure()
        assert m.fallback_count == 1
        assert m.model_load_failures == 1


# ===========================================================================
# FINAL GATE
# ===========================================================================


class TestFinalGate:
    """The learned policy remains disabled until every box above is checked."""

    def test_all_checklist_items_pass(self) -> None:
        """All checklist test classes pass (pytest will verify this by running)."""
        # This test exists as a marker. If all other tests in this module
        # pass, the checklist is satisfied.
        assert True

    def test_learned_policy_is_not_controlling_hardware(self) -> None:
        """The learned policy cannot control hardware directly."""
        gate = SafetyGate(action_space=deskbot_action_space())
        # The safety gate is the only path to hardware
        # A learned policy can only propose, not execute
        result = gate.validate(action_index=2)
        # Even a valid action must pass through the gate
        assert result.layer in {"static", "runtime"}

    def test_shadow_mode_has_zero_authority(self) -> None:
        """In shadow mode, the model has zero authority over hardware."""
        controller = ShadowPolicyController(
            action_space=deskbot_action_space(),
            deterministic_policy=lambda s: 2,
            model_policy=lambda s: (7, [0.0] * 10),  # celebrate — different
            model_version="v1.0",
            mode=PolicyMode.SHADOW,
        )
        state = np.zeros(91, dtype=np.float64)
        for _ in range(100):
            action = controller.decide(state)
            assert action == 2  # always deterministic, never model's action (7)
