"""REST API endpoints for the learning system.

Exposes learning status, model version, training progress,
experience count, and current configuration.  Does NOT expose
arbitrary model execution.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from robot.api.security import require_api_key, HTTPException, Request

from robot.api.schemas import (
    ForceTrainResponse,
    LearningConfigResponse,
    LearningPreferencesResponse,
    LearningStatusResponse,
)

router = APIRouter(prefix="/learning", tags=["learning"])


def _get_learning_service(request: Request) -> Any:
    """Get the learning service from app state, or raise 404."""
    svc = getattr(request.app.state, "learning_service", None)
    if svc is None:
        raise HTTPException(status_code=404, detail="Learning service not available")
    return svc


def _get_safety_manager(request: Request) -> Any:
    """Get the safety manager from app state, or raise 404."""
    mgr = getattr(request.app.state, "safety_manager", None)
    if mgr is None:
        raise HTTPException(status_code=404, detail="Safety manager not available")
    return mgr


@router.get("/status", summary="Learning service status", response_model=LearningStatusResponse)
async def learning_status(request: Request) -> LearningStatusResponse:
    """Return the current learning service status."""
    svc = _get_learning_service(request)
    status = svc.status
    return LearningStatusResponse(
        enabled=True,
        total_experiences=status.total_experiences,
        new_experiences_since_train=status.new_experiences_since_train,
        training_cycles_completed=status.training_cycles_completed,
        current_model_loss=round(status.current_model_loss, 6)
        if status.current_model_loss != float("inf")
        else None,
        candidate_model_loss=round(status.candidate_model_loss, 6)
        if status.candidate_model_loss != float("inf")
        else None,
        last_training_time=status.last_training_time.isoformat()
        if status.last_training_time
        else None,
        last_training_duration_s=round(status.last_training_duration_s, 3),
        is_training=status.is_training,
        promotions=status.promotions,
        rollbacks=status.rollbacks,
        model_version=status.model_version,
        use_multimodal=status.use_multimodal,
        multimodal_state_size=status.multimodal_state_size,
    )


@router.get(
    "/preferences", summary="Learned preferences", response_model=LearningPreferencesResponse
)
async def learning_preferences(request: Request) -> LearningPreferencesResponse:
    """Return all learned preferences."""
    svc = _get_learning_service(request)
    learner = getattr(svc, "preference_learner", None)
    if learner is None:
        return LearningPreferencesResponse(preferences=[], total_patterns=0)
    prefs = learner.get_learned_preferences()
    return LearningPreferencesResponse(
        preferences=[
            {
                "key": p.key,
                "category": p.category,
                "value": p.value,
                "confidence": round(p.confidence, 4),
                "observation_count": p.observation_count,
                "avg_reward": round(p.avg_reward, 4),
                "source": p.source,
                "last_observed": p.last_observed.isoformat(),
            }
            for p in prefs
        ],
        total_patterns=learner.total_patterns,
    )


@router.get("/config", summary="Learning configuration", response_model=LearningConfigResponse)
async def learning_config(request: Request) -> LearningConfigResponse:
    """Return the current learning configuration."""
    svc = _get_learning_service(request)
    return LearningConfigResponse(
        schedule={
            "min_new_experiences": svc.schedule.min_new_experiences,
            "train_interval_s": svc.schedule.train_interval_s,
            "min_experiences_for_training": svc.schedule.min_experiences_for_training,
        },
        resource_limits={
            "batch_size": svc.resource_limits.batch_size,
            "training_epochs_per_cycle": svc.resource_limits.training_epochs_per_cycle,
            "max_cpu_fraction": svc.resource_limits.max_cpu_fraction,
            "max_model_params": svc.resource_limits.max_model_params,
            "eval_sample_size": svc.resource_limits.eval_sample_size,
        },
        checkpoint_config={
            "checkpoint_dir": svc.checkpoint_config.checkpoint_dir,
            "keep_last_n": svc.checkpoint_config.keep_last_n,
            "promote_threshold": svc.checkpoint_config.promote_threshold,
        },
        multimodal={
            "enabled": svc.multimodal_encoder is not None,
            "state_size": svc.state_size,
        },
    )


@router.post("/train", summary="Force a training cycle", response_model=ForceTrainResponse)
async def force_train(request: Request, _: None = Depends(require_api_key)) -> ForceTrainResponse:
    """Force an immediate training cycle regardless of schedule."""
    svc = _get_learning_service(request)
    result = svc.force_training()
    status = svc.status
    return ForceTrainResponse(
        triggered=result,
        training_cycles_completed=status.training_cycles_completed,
        is_training=status.is_training,
    )
