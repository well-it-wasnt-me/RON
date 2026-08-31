"""REST API endpoints for the human teaching loop (Phases 8-10).

Exposes teaching status, recent transitions (with a state summary derived
from reserved state-vector slots — never raw conversation), explicit
feedback submission, demonstration arming/triggering, and current Q-values.

The synthetic gesture channel lives here: ``POST /teaching/demonstration``
arms a session from a constrained instruction and/or injects a gesture —
there is no vision gesture detector, by design.

These endpoints never bypass safety: feedback flows through the
:class:`FeedbackService` (which attributes it to a *real* transition), and
gesture triggers flow through the :class:`TeachingController` → canonical
:class:`ActionExecutor` (the single learning recording point). The LLM is
never asked to choose what action the robot should learn.
"""

from __future__ import annotations

import contextlib
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from robot.api.schemas import (
    TeachingDemonstrationRequest,
    TeachingDemonstrationResponse,
    TeachingFeedbackRequest,
    TeachingFeedbackResponse,
    TeachingQValuesResponse,
    TeachingStatusResponse,
    TeachingTransitionItem,
    TeachingTransitionsResponse,
)
from robot.api.security import require_api_key
from robot.learning.state_encoder import (
    _GESTURE_END,
    _GESTURE_NAMES,
    _GESTURE_START,
    _INTERACTION_ACTIVE,
    _PERSON_PRESENT,
    _TEACHING_CONTEXT,
)

router = APIRouter(prefix="/teaching", tags=["teaching"])

#: Slots used to derive the (conversation-free) state summary.
_TEACHING_CONTEXT_SLOT = _TEACHING_CONTEXT
_INTERACTION_ACTIVE_SLOT = _INTERACTION_ACTIVE
_PERSON_PRESENT_SLOT = _PERSON_PRESENT
_GESTURE_SLOTS = (slice(_GESTURE_START, _GESTURE_END), _GESTURE_NAMES)


def _get_teaching_controller(request: Request) -> Any:
    return getattr(request.app.state, "teaching_controller", None)


def _get_feedback_service(request: Request) -> Any:
    return getattr(request.app.state, "feedback_service", None)


def _get_learning_service(request: Request) -> Any:
    return getattr(request.app.state, "learning_service", None)


def _get_learning_state(request: Request) -> list[float]:
    """Return the state vector matching the learning service's configured state size."""
    svc = _get_learning_service(request)
    multimodal = getattr(svc, "multimodal_encoder", None)
    state = multimodal.encode() if multimodal is not None else svc.encoder.encode()

    expected_size = getattr(svc, "state_size", None)
    if expected_size is not None and len(state) != expected_size:
        raise HTTPException(
            status_code=500,
            detail=(
                "Learning state dimension mismatch: "
                f"encoder produced {len(state)}, expected {expected_size}"
            ),
        )

    return list(state)


def _state_summary(state: list[float]) -> dict[str, Any]:
    """Derive a conversation-free summary from the reserved state slots."""
    summary: dict[str, Any] = {
        "teaching_context": bool(state[_TEACHING_CONTEXT_SLOT] > 0.5)
        if len(state) > _TEACHING_CONTEXT_SLOT
        else False,
        "interaction_active": bool(state[_INTERACTION_ACTIVE_SLOT] > 0.5)
        if len(state) > _INTERACTION_ACTIVE_SLOT
        else False,
        "person_present": bool(state[_PERSON_PRESENT_SLOT] > 0.5)
        if len(state) > _PERSON_PRESENT_SLOT
        else False,
        "gesture": "none",
    }
    gest_slice, gest_names = _GESTURE_SLOTS
    segment = state[gest_slice]
    if segment:
        idx = max(range(len(segment)), key=lambda i: segment[i])
        summary["gesture"] = gest_names[idx] if 0 <= idx < len(gest_names) else "none"
    return summary


@router.get("/status", summary="Teaching loop status", response_model=TeachingStatusResponse)
async def teaching_status(request: Request) -> TeachingStatusResponse:
    """Return the current teaching-loop status."""
    settings = getattr(request.app.state, "settings", None)
    tcfg = getattr(settings, "teaching", None) if settings is not None else None
    controller = _get_teaching_controller(request)
    enabled = tcfg is not None and bool(tcfg.enabled) and controller is not None

    if not enabled or controller is None:
        return TeachingStatusResponse(
            enabled=False,
            in_teaching_mode=False,
            total_experiences=0,
            min_experiences_for_practice=int(getattr(tcfg, "min_experiences_for_practice", 0))
            if tcfg is not None
            else 0,
        )

    spec = controller.current
    svc = _get_learning_service(request)
    total = getattr(getattr(svc, "status", None), "total_experiences", 0) if svc else 0
    return TeachingStatusResponse(
        enabled=True,
        in_teaching_mode=controller.in_teaching_mode,
        session_id=controller.session_id,
        mode=controller.mode,
        trigger_gesture=spec.trigger_gesture if spec else None,
        desired_action=spec.desired_action if spec else None,
        total_experiences=int(total),
        min_experiences_for_practice=controller._min_experiences_for_practice,
    )


@router.get(
    "/transitions",
    summary="Recent teaching transitions",
    response_model=TeachingTransitionsResponse,
)
async def teaching_transitions(
    request: Request,
    limit: int = Query(default=20, ge=1, le=256),
) -> TeachingTransitionsResponse:
    """Return recent transitions with a conversation-free state summary."""
    svc = _get_learning_service(request)
    if svc is None:
        raise HTTPException(status_code=404, detail="Learning service not available")
    working = getattr(svc, "working_memory", None)
    if working is None:
        return TeachingTransitionsResponse(total=0, limit=limit, transitions=[])
    action_space = getattr(svc, "action_space", None)
    ledger = getattr(svc, "feedback_ledger", None)
    recent = working.recent(limit)
    items: list[TeachingTransitionItem] = []
    for exp in recent:
        meta = exp.metadata
        action_index = int(meta.get("action_index", -1))
        action_name = "unknown"
        if action_space is not None and 0 <= action_index < action_space.size:
            action_name = action_space.get(action_index).name
        tid = meta.get("transition_id")
        tid_str = str(tid) if tid is not None else None
        feedback_source: str | None = None
        if ledger is not None and tid_str is not None:
            entry = ledger.get(tid_str)
            if entry is not None:
                feedback_source = entry.source
        reward = float(exp.reward)
        if tid_str is not None and hasattr(svc, "reward_for_transition"):
            with contextlib.suppress(Exception):
                reward = float(svc.reward_for_transition(tid_str))
        items.append(
            TeachingTransitionItem(
                timestamp=exp.timestamp.isoformat(),
                transition_id=tid_str,
                action_name=action_name,
                action_index=action_index,
                execution_success=bool(meta.get("execution_success", True)),
                reward=reward,
                feedback_source=feedback_source,
                interaction_id=str(meta["interaction_id"]) if meta.get("interaction_id") else None,
                teaching_session_id=str(meta["teaching_session_id"])
                if meta.get("teaching_session_id")
                else None,
                state_summary=_state_summary(list(exp.state)),
            )
        )
    return TeachingTransitionsResponse(total=len(items), limit=limit, transitions=items)


@router.post(
    "/feedback",
    summary="Submit explicit human feedback",
    response_model=TeachingFeedbackResponse,
    dependencies=[Depends(require_api_key)],
)
async def teaching_feedback(
    request: Request, body: TeachingFeedbackRequest
) -> TeachingFeedbackResponse:
    """Attribute explicit feedback to the most-recent eligible real transition."""
    feedback = _get_feedback_service(request)
    if feedback is None:
        raise HTTPException(status_code=404, detail="Feedback service not available")
    entry = await feedback.handle_feedback(
        polarity=body.polarity,
        magnitude=body.magnitude,
        source=body.source,
        text=body.text,
    )
    if entry is None:
        return TeachingFeedbackResponse(attributed=False)
    return TeachingFeedbackResponse(
        attributed=True,
        transition_id=entry.transition_id,
        delta=entry.reward_delta,
    )


@router.post(
    "/demonstration",
    summary="Arm a teaching session and/or inject a gesture",
    response_model=TeachingDemonstrationResponse,
    dependencies=[Depends(require_api_key)],
)
async def teaching_demonstration(
    request: Request, body: TeachingDemonstrationRequest
) -> TeachingDemonstrationResponse:
    """Arm a session from a constrained instruction and/or trigger a gesture."""
    controller = _get_teaching_controller(request)
    if controller is None:
        raise HTTPException(status_code=404, detail="Teaching controller not available")
    if body.mode not in {"demonstrate", "practice"}:
        raise HTTPException(status_code=422, detail=f"unknown mode: {body.mode!r}")

    session_id: str | None = None
    trigger_gesture: str | None = None
    desired_action: str | None = None
    if body.instruction is not None:
        sid = controller.arm_from_instruction(body.instruction, mode=body.mode)
        if sid is None:
            raise HTTPException(
                status_code=422,
                detail="instruction did not parse as a teaching instruction",
            )
        session_id = sid
        spec = controller.current
        if spec is not None:
            trigger_gesture = spec.trigger_gesture
            desired_action = spec.desired_action

    executed_action: str | None = None
    executed_index: int | None = None
    if body.gesture is not None:
        svc = _get_learning_service(request)
        if svc is None:
            raise HTTPException(status_code=404, detail="Learning service not available")
        state = _get_learning_state(request)
        executed = await controller.on_gesture_detected(body.gesture, state)
        if executed is not None:
            executed_index = executed
            if 0 <= executed < controller._action_space.size:
                executed_action = controller._action_space.get(executed).name

    return TeachingDemonstrationResponse(
        session_id=session_id,
        trigger_gesture=trigger_gesture,
        desired_action=desired_action,
        executed_action=executed_action,
        executed_action_index=executed_index,
    )


@router.get(
    "/qvalues",
    summary="Current Q-values for the encoder state",
    response_model=TeachingQValuesResponse,
)
async def teaching_qvalues(request: Request) -> TeachingQValuesResponse:
    """Return the policy's Q-values for the current encoder state."""
    svc = _get_learning_service(request)
    if svc is None:
        raise HTTPException(status_code=404, detail="Learning service not available")
    if getattr(svc, "action_learner", None) is None:
        raise HTTPException(status_code=404, detail="Action learner not available")
    state = _get_learning_state(request)
    try:
        q = svc.q_values(state)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return TeachingQValuesResponse(q_values=q)


__all__ = ["router"]
