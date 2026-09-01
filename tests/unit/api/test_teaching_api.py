"""Phase 9: REST API for the human teaching loop.

Covers every ``/api/v1/teaching/*`` endpoint — shapes, auth, and that the
``enabled`` flag reflects ``settings.teaching.enabled`` (not a hard-coded
value). The endpoints are exercised against a real
:class:`LearningService` + :class:`TeachingController` + :class:`FeedbackService`
so the data paths are genuine.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from robot.api.app import create_app
from robot.config import AppSettings
from robot.events.bus import InMemoryEventBus
from robot.hardware.servos.adapter import wrap_servo_controller
from robot.hardware.servos.mock_servo import MockServo, MockServoBus
from robot.learning.experience import Experience, ReplayBuffer, WorkingMemory
from robot.learning.feedback_ledger import FeedbackEntry, FeedbackLedger
from robot.learning.feedback_service import FeedbackService
from robot.learning.interaction_context import InteractionContext
from robot.learning.learning_service import (
    CheckpointConfig,
    LearningSchedule,
    LearningService,
    ResourceLimits,
)
from robot.learning.safety_gate import SafetyGate
from robot.learning.state_encoder import STATE_SIZE
from robot.learning.teaching_controller import TeachingController
from robot.services.executor import ActionExecutor


def _servo_controller() -> object:
    bus = MockServoBus(
        {
            "pan": MockServo(name="pan", min_angle=-90.0, max_angle=90.0),
            "tilt": MockServo(name="tilt", min_angle=-30.0, max_angle=30.0),
            "left_arm": MockServo(name="left_arm", min_angle=0.0, max_angle=180.0),
            "right_arm": MockServo(name="right_arm", min_angle=0.0, max_angle=180.0),
        }
    )
    return wrap_servo_controller(bus, backend_name="mock")


def _make_service(bus: InMemoryEventBus, tmp_path: Path) -> LearningService:
    schedule = LearningSchedule(
        min_new_experiences=8, train_interval_s=0.1, min_experiences_for_training=8
    )
    limits = ResourceLimits(
        batch_size=8, training_epochs_per_cycle=1, max_cpu_fraction=1.0, eval_sample_size=16
    )
    ckpt_cfg = CheckpointConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"), keep_last_n=2, promote_threshold=1.0
    )
    return LearningService(
        bus=bus,
        schedule=schedule,
        resource_limits=limits,
        checkpoint_config=ckpt_cfg,
        state_size=STATE_SIZE,
        seed=42,
        use_multimodal=False,
        working_memory=WorkingMemory(capacity=256),
        replay_buffer=ReplayBuffer(capacity=10_000, seed=42),
    )


def _wire_app(
    *,
    settings: AppSettings,
    service: LearningService,
) -> tuple[FastAPI, TeachingController, FeedbackService, FeedbackLedger]:
    """Create the API app and inject the teaching components into app.state."""
    assert service.recorder is not None
    assert service.action_learner is not None
    ledger = FeedbackLedger()
    service.feedback_ledger = ledger
    feedback_service = FeedbackService(service.recorder, ledger, feedback_window_s=30.0)
    executor = ActionExecutor(bus=service.bus, servo_controller=_servo_controller())  # type: ignore[arg-type]
    executor.experience_recorder = service.recorder
    executor.interaction_context = InteractionContext()
    safety_gate = SafetyGate(
        action_space=service.action_space,
        servo_limits={
            "pan": (-90.0, 90.0),
            "tilt": (-30.0, 30.0),
            "left_arm": (0.0, 180.0),
            "right_arm": (0.0, 180.0),
        },
        cooldown_s=0.0,
    )
    controller = TeachingController(
        action_learner=service.action_learner,
        safety_gate=safety_gate,
        action_space=service.action_space,
        interaction_context=executor.interaction_context,
        executor=executor,
        min_experiences_for_practice=64,
    )
    app = create_app(settings=settings)
    app.state.learning_service = service
    app.state.teaching_controller = controller
    app.state.feedback_service = feedback_service
    app.state.safety_gate = safety_gate
    return app, controller, feedback_service, ledger


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.fixture
def bus() -> InMemoryEventBus:
    return InMemoryEventBus()


@pytest.fixture
def settings() -> AppSettings:
    s = AppSettings(_env_file=None)
    s.teaching.enabled = True
    s.api.api_key = ""  # auth disabled by default for read tests
    return s


async def test_status_reflects_settings_enabled(
    settings: AppSettings, bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """``enabled`` tracks settings.teaching.enabled, not a hard-coded value."""
    service = _make_service(bus, tmp_path)
    app, _controller, _feedback, _ledger = _wire_app(settings=settings, service=service)
    async with _client(app) as client:
        r = await client.get("/api/v1/teaching/status")
        assert r.status_code == 200
        data = r.json()
        assert data["enabled"] is True
        assert data["in_teaching_mode"] is False
        assert data["min_experiences_for_practice"] == 64


async def test_status_disabled_when_teaching_off(
    bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """When teaching.enabled is False, status reports enabled=False."""
    s = AppSettings(_env_file=None)
    s.teaching.enabled = False
    service = _make_service(bus, tmp_path)
    app, _c, _f, _l = _wire_app(settings=s, service=service)
    async with _client(app) as client:
        r = await client.get("/api/v1/teaching/status")
        assert r.status_code == 200
        assert r.json()["enabled"] is False


async def test_transitions_empty_then_populated(
    settings: AppSettings, bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """Transitions list grows after a real action executes through the executor."""
    service = _make_service(bus, tmp_path)
    app, controller, _feedback, _ledger = _wire_app(settings=settings, service=service)
    recorder = service.recorder
    assert recorder is not None
    controller.arm_from_instruction("when i wave, wave back")
    recorder.attach()
    try:
        await controller.on_gesture_detected("wave", service.encoder.encode())
    finally:
        recorder.detach()

    async with _client(app) as client:
        r = await client.get("/api/v1/teaching/transitions?limit=10")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        item = data["transitions"][0]
        assert item["action_name"] == "wave"
        assert item["action_index"] == 13
        assert item["teaching_session_id"] is not None
        assert "teaching_context" in item["state_summary"]
        assert "gesture" in item["state_summary"]


async def test_qvalues_returns_dict(
    settings: AppSettings, bus: InMemoryEventBus, tmp_path: Path
) -> None:
    service = _make_service(bus, tmp_path)
    app, _c, _f, _l = _wire_app(settings=settings, service=service)
    async with _client(app) as client:
        r = await client.get("/api/v1/teaching/qvalues")
        assert r.status_code == 200
        q = r.json()["q_values"]
        assert "wave" in q and "look_left" in q
        assert isinstance(q["wave"], float)


async def test_feedback_attributed_to_recent_transition(
    settings: AppSettings, bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """POST /feedback attributes praise to the most-recent real transition."""
    service = _make_service(bus, tmp_path)
    app, controller, _feedback, ledger = _wire_app(settings=settings, service=service)
    recorder = service.recorder
    assert recorder is not None
    controller.arm_from_instruction("when i wave, wave back")
    recorder.attach()
    try:
        await controller.on_gesture_detected("wave", service.encoder.encode())
    finally:
        recorder.detach()

    async with _client(app) as client:
        r = await client.post(
            "/api/v1/teaching/feedback",
            json={"polarity": 1, "magnitude": 1.0, "source": "test", "text": "good"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["attributed"] is True
        assert data["delta"] == 1.0
        assert data["transition_id"] is not None

    # The ledger now carries the feedback for that transition.
    recent = service.working_memory.recent(1)[-1]
    assert ledger.get(recent.metadata["transition_id"]) is not None


async def test_feedback_requires_api_key_when_configured(
    bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """When an API key is set, feedback without it is 401."""
    s = AppSettings(_env_file=None)
    s.teaching.enabled = True
    s.api.api_key = "secret-key"
    service = _make_service(bus, tmp_path)
    app, _c, _f, _l = _wire_app(settings=s, service=service)
    async with _client(app) as client:
        r = await client.post("/api/v1/teaching/feedback", json={"polarity": 1})
        assert r.status_code == 401
        # With the key, it is accepted (no eligible transition -> attributed False).
        r2 = await client.post(
            "/api/v1/teaching/feedback",
            json={"polarity": 1},
            headers={"Authorization": "Bearer secret-key"},
        )
        assert r2.status_code == 200


async def test_demonstration_arms_and_triggers(
    settings: AppSettings, bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """POST /demonstration arms a session and triggers a gesture."""
    service = _make_service(bus, tmp_path)
    app, controller, _feedback, _ledger = _wire_app(settings=settings, service=service)
    recorder = service.recorder
    assert recorder is not None
    recorder.attach()
    try:
        async with _client(app) as client:
            r = await client.post(
                "/api/v1/teaching/demonstration",
                json={"instruction": "when i wave, wave back", "mode": "demonstrate"},
            )
            assert r.status_code == 200
            d = r.json()
            assert d["session_id"] is not None
            assert d["trigger_gesture"] == "wave"
            assert d["desired_action"] == "wave"

            # Trigger the gesture -> executes wave.
            r2 = await client.post(
                "/api/v1/teaching/demonstration",
                json={"gesture": "wave"},
            )
            assert r2.status_code == 200
            assert r2.json()["executed_action"] == "wave"
            assert r2.json()["executed_action_index"] == 13
        assert controller.in_teaching_mode is True
    finally:
        recorder.detach()


async def test_demonstration_requires_api_key(bus: InMemoryEventBus, tmp_path: Path) -> None:
    s = AppSettings(_env_file=None)
    s.teaching.enabled = True
    s.api.api_key = "k"
    service = _make_service(bus, tmp_path)
    app, _c, _f, _l = _wire_app(settings=s, service=service)
    async with _client(app) as client:
        r = await client.post("/api/v1/teaching/demonstration", json={"instruction": "when i wave, wave back"})
        assert r.status_code == 401


async def test_status_404_when_no_learning_service(
    settings: AppSettings, tmp_path: Path
) -> None:
    """Transitions/qvalues 404 when no learning service is wired."""
    app = create_app(settings=settings)
    async with _client(app) as client:
        assert (await client.get("/api/v1/teaching/transitions")).status_code == 404
        assert (await client.get("/api/v1/teaching/qvalues")).status_code == 404


# ---------------------------------------------------------------------------
# /teaching/transitions server-side filtering
# ---------------------------------------------------------------------------


def _add_exp(
    service: LearningService,
    ledger: FeedbackLedger,
    *,
    action_index: int,
    success: bool,
    interaction_id: str | None,
    transition_id: str,
    with_feedback: bool = False,
    n: int = 1,
) -> None:
    """Append crafted experiences straight into working memory."""
    zero = [0.0] * STATE_SIZE
    for k in range(n):
        exp = Experience(
            timestamp=datetime.now(tz=UTC),
            state=list(zero),
            action=[0.0],
            reward=0.0,
            next_state=list(zero),
            metadata={
                "action_index": action_index,
                "execution_success": success,
                "transition_id": f"{transition_id}-{k}",
                "interaction_id": interaction_id,
                "teaching_session_id": "sess-1",
            },
        )
        service.working_memory.add(exp)
        if with_feedback:
            ledger.record(
                FeedbackEntry(
                    transition_id=f"{transition_id}-{k}",
                    polarity=1,
                    magnitude=1.0,
                    source="test",
                )
            )


async def _transition_filter_setup(
    settings: AppSettings, bus: InMemoryEventBus, tmp_path: Path
) -> FastAPI:
    """Wire an app whose working memory holds a known mix of transitions."""
    service = _make_service(bus, tmp_path)
    app, _controller, _feedback, ledger = _wire_app(settings=settings, service=service)
    # action_index 13 = wave, 0 = look_left
    _add_exp(service, ledger, action_index=13, success=True, interaction_id="int-A",
             transition_id="t-wave", with_feedback=True, n=2)
    _add_exp(service, ledger, action_index=13, success=False, interaction_id="int-B",
             transition_id="t-wave-fail")
    _add_exp(service, ledger, action_index=0, success=True, interaction_id="int-A",
             transition_id="t-look")
    _add_exp(service, ledger, action_index=0, success=True, interaction_id="int-C",
             transition_id="t-look-nofb")
    return app


async def test_transitions_filter_by_action(
    settings: AppSettings, bus: InMemoryEventBus, tmp_path: Path
) -> None:
    app = await _transition_filter_setup(settings, bus, tmp_path)
    async with _client(app) as client:
        r = await client.get("/api/v1/teaching/transitions?action=wave&limit=100")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 3  # 2 wave-success + 1 wave-fail
        assert all(t["action_name"] == "wave" for t in data["transitions"])


async def test_transitions_filter_by_success(
    settings: AppSettings, bus: InMemoryEventBus, tmp_path: Path
) -> None:
    app = await _transition_filter_setup(settings, bus, tmp_path)
    async with _client(app) as client:
        r = await client.get("/api/v1/teaching/transitions?success=false&limit=100")
        data = r.json()
        assert data["total"] == 1
        assert data["transitions"][0]["execution_success"] is False


async def test_transitions_filter_by_feedback(
    settings: AppSettings, bus: InMemoryEventBus, tmp_path: Path
) -> None:
    app = await _transition_filter_setup(settings, bus, tmp_path)
    async with _client(app) as client:
        # Only transitions with attributed feedback.
        r = await client.get("/api/v1/teaching/transitions?feedback=true&limit=100")
        data = r.json()
        assert data["total"] == 2
        assert all(t["feedback_source"] == "test" for t in data["transitions"])

        # Only transitions without feedback.
        r2 = await client.get("/api/v1/teaching/transitions?feedback=false&limit=100")
        data2 = r2.json()
        assert data2["total"] == 3
        assert all(t["feedback_source"] is None for t in data2["transitions"])


async def test_transitions_filter_by_interaction_id(
    settings: AppSettings, bus: InMemoryEventBus, tmp_path: Path
) -> None:
    app = await _transition_filter_setup(settings, bus, tmp_path)
    async with _client(app) as client:
        r = await client.get("/api/v1/teaching/transitions?interaction_id=int-A&limit=100")
        data = r.json()
        # int-A: 2 wave + 1 look = 3
        assert data["total"] == 3
        assert all(t["interaction_id"] == "int-A" for t in data["transitions"])


async def test_transitions_limit_applied_after_filter(
    settings: AppSettings, bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """limit takes the most recent N after filtering (not before)."""
    app = await _transition_filter_setup(settings, bus, tmp_path)
    async with _client(app) as client:
        r = await client.get("/api/v1/teaching/transitions?action=wave&limit=2")
        data = r.json()
        # total reflects the full filtered set (3), but only 2 are returned.
        assert data["total"] == 3
        assert len(data["transitions"]) == 2


async def test_transitions_combined_filters(
    settings: AppSettings, bus: InMemoryEventBus, tmp_path: Path
) -> None:
    app = await _transition_filter_setup(settings, bus, tmp_path)
    async with _client(app) as client:
        r = await client.get(
            "/api/v1/teaching/transitions?action=wave&success=true&limit=100"
        )
        data = r.json()
        assert data["total"] == 2
        assert all(t["action_name"] == "wave" and t["execution_success"] for t in data["transitions"])
