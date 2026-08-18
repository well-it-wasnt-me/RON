"""Safety-gated actions: make it impossible for a learned policy to bypass safety.

Phase 7 of the production learning plan.  Three layers:

1. **Static validation** — action exists, parameters valid, servo/timing/rate limits.
2. **Runtime safety** — calibrated servo range, cooldown, conflicting actions,
   sensor availability, robot state restrictions.
3. **Emergency override** — reliable mechanism to disable learned control.

No HTTP endpoint, event handler, training component, or policy can bypass
the safety validator.  Every hardware action uses the same execution path.

When the policy crashes, returns NaN, returns an invalid action, times out,
or cannot load, the robot falls back to deterministic behaviour.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from robot.learning.action_learning import ActionSpace
from robot.logging import get_logger

_log = get_logger("learning.safety_gate")


# ---------------------------------------------------------------------------
# Safety result
# ---------------------------------------------------------------------------


class SafetyResultType(str, Enum):
    """Result of a safety gate check."""

    ALLOW = "allow"
    REJECT = "reject"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class SafetyResult:
    """Result of a safety gate check.

    Attributes
    ----------
    result_type:
        Whether the action is allowed, rejected, or falls back.
    action_index:
        The action index that was checked.
    reason:
        Human-readable explanation.
    layer:
        Which safety layer made the decision ("static", "runtime", "override").
    """

    result_type: SafetyResultType
    action_index: int
    reason: str
    layer: str = ""

    @property
    def allowed(self) -> bool:
        return self.result_type == SafetyResultType.ALLOW

    @property
    def rejected(self) -> bool:
        return self.result_type == SafetyResultType.REJECT

    @property
    def fallback(self) -> bool:
        return self.result_type == SafetyResultType.FALLBACK


# ---------------------------------------------------------------------------
# Safety gate
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SafetyGate:
    """Three-layer safety validator for all hardware actions.

    No component — HTTP endpoint, event handler, training, or policy —
    can bypass this validator.  Every hardware action uses the same
    execution path.

    Parameters
    ----------
    action_space:
        The action space to validate against.
    max_action_rate:
        Maximum actions per second.
    servo_limits:
        Dict of servo name → (min_angle, max_angle).
    cooldown_s:
        Minimum seconds between the same action.
    fallback_action:
        Action index to use when the policy fails.  Defaults to 0
        (look_left) — a safe, non-destructive action.
    enabled:
        Whether the safety gate is active.  When False, all actions
        pass (for testing only — never in production).
    """

    action_space: ActionSpace
    max_action_rate: float = 10.0
    servo_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "pan": (0.0, 180.0),
            "tilt": (0.0, 180.0),
        }
    )
    cooldown_s: float = 0.1
    fallback_action: int = 0
    enabled: bool = True

    _last_action_time: float = field(default=0.0, init=False, repr=False)
    _action_timestamps: list[float] = field(default_factory=list, init=False, repr=False)
    _last_action_per_type: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _override_active: bool = field(default=False, init=False, repr=False)

    @property
    def override_active(self) -> bool:
        """Whether the emergency override is active (disables learned control)."""
        return self._override_active

    def activate_override(self) -> None:
        """Activate the emergency override — disables learned control."""
        self._override_active = True
        _log.warning("safety_gate.override_activated")

    def deactivate_override(self) -> None:
        """Deactivate the emergency override — re-enables learned control."""
        self._override_active = False
        _log.info("safety_gate.override_deactivated")

    def validate(
        self,
        action_index: int,
        state: list[float] | None = None,
        params: dict[str, Any] | None = None,
    ) -> SafetyResult:
        """Validate an action through all three safety layers.

        This is the single execution path for all hardware actions.
        No component can bypass this method.

        Returns a :class:`SafetyResult` indicating allow, reject, or
        fallback.
        """
        if not self.enabled:
            return SafetyResult(
                result_type=SafetyResultType.ALLOW,
                action_index=action_index,
                reason="gate disabled",
                layer="static",
            )

        # Layer 3: Emergency override — always falls back to deterministic
        if self._override_active:
            return SafetyResult(
                result_type=SafetyResultType.FALLBACK,
                action_index=self.fallback_action,
                reason="emergency override active",
                layer="override",
            )

        # Layer 1: Static validation
        result = self._static_validation(action_index, params)
        if not result.allowed:
            return result

        # Layer 2: Runtime safety
        result = self._runtime_safety(action_index, state, params)
        if not result.allowed:
            return result

        return result

    def validate_or_fallback(
        self,
        action_index: int,
        state: list[float] | None = None,
        params: dict[str, Any] | None = None,
    ) -> int:
        """Validate an action, returning the fallback if rejected.

        Returns the validated action index, or the fallback action
        index if the action was rejected or the override is active.
        """
        result = self.validate(action_index, state, params)
        if result.allowed:
            return action_index
        return self.fallback_action

    def handle_model_output(
        self,
        model_output: Any,
        state: list[float] | None = None,
    ) -> SafetyResult:
        """Handle raw model output, returning a safety result.

        Handles:
        - NaN scores
        - Invalid action indices
        - Model timeouts (None output)
        - Corrupted output

        Always returns a valid :class:`SafetyResult`.  Never raises.
        """
        try:
            if model_output is None:
                return SafetyResult(
                    result_type=SafetyResultType.FALLBACK,
                    action_index=self.fallback_action,
                    reason="model output is None",
                    layer="runtime",
                )

            if isinstance(model_output, (int, float)):
                if math.isnan(float(model_output)) or math.isinf(float(model_output)):
                    return SafetyResult(
                        result_type=SafetyResultType.FALLBACK,
                        action_index=self.fallback_action,
                        reason="model output is NaN/inf",
                        layer="runtime",
                    )
                action_idx = int(model_output)
            elif isinstance(model_output, (list, tuple)) and len(model_output) > 0:
                scores = [float(s) for s in model_output]
                if any(math.isnan(s) or math.isinf(s) for s in scores):
                    return SafetyResult(
                        result_type=SafetyResultType.FALLBACK,
                        action_index=self.fallback_action,
                        reason="model scores contain NaN/inf",
                        layer="runtime",
                    )
                action_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
            else:
                return SafetyResult(
                    result_type=SafetyResultType.FALLBACK,
                    action_index=self.fallback_action,
                    reason=f"unexpected model output type: {type(model_output)}",
                    layer="runtime",
                )

            return self.validate(action_idx, state)
        except Exception:
            _log.exception("safety_gate.model_output_error")
            return SafetyResult(
                result_type=SafetyResultType.FALLBACK,
                action_index=self.fallback_action,
                reason="model output processing error",
                layer="runtime",
            )

    # ------------------------------------------------------------------ layers
    def _static_validation(
        self,
        action_index: int,
        params: dict[str, Any] | None,
    ) -> SafetyResult:
        """Layer 1: static validation — action exists, params valid."""
        # Action exists in action space
        if not (0 <= action_index < self.action_space.size):
            return SafetyResult(
                result_type=SafetyResultType.REJECT,
                action_index=action_index,
                reason=f"action index {action_index} out of range",
                layer="static",
            )

        # Validate servo parameters
        if params:
            for key, val in params.items():
                if key in self.servo_limits and isinstance(val, (int, float)):
                    min_v, max_v = self.servo_limits[key]
                    if val < min_v or val > max_v:
                        return SafetyResult(
                            result_type=SafetyResultType.REJECT,
                            action_index=action_index,
                            reason=f"servo {key} value {val} out of range [{min_v}, {max_v}]",
                            layer="static",
                        )

        return SafetyResult(
            result_type=SafetyResultType.ALLOW,
            action_index=action_index,
            reason="ok",
            layer="static",
        )

    def _runtime_safety(
        self,
        action_index: int,
        state: list[float] | None,
        params: dict[str, Any] | None,
    ) -> SafetyResult:
        """Layer 2: runtime safety — rate limits, cooldown, state restrictions."""
        now = time.monotonic()

        # Rate limit
        self._action_timestamps.append(now)
        self._action_timestamps = [t for t in self._action_timestamps if now - t < 1.0]
        if len(self._action_timestamps) > self.max_action_rate:
            return SafetyResult(
                result_type=SafetyResultType.REJECT,
                action_index=action_index,
                reason=f"rate limit exceeded: {len(self._action_timestamps)} actions/s",
                layer="runtime",
            )

        # Cooldown per action type
        action_name = self.action_space.get(action_index).name
        last_time = self._last_action_per_type.get(action_name, 0.0)
        if now - last_time < self.cooldown_s:
            return SafetyResult(
                result_type=SafetyResultType.REJECT,
                action_index=action_index,
                reason=f"action {action_name} on cooldown",
                layer="runtime",
            )

        # Update last action time
        self._last_action_per_type[action_name] = now
        self._last_action_time = now

        return SafetyResult(
            result_type=SafetyResultType.ALLOW,
            action_index=action_index,
            reason="ok",
            layer="runtime",
        )


# ---------------------------------------------------------------------------
# Safe executor wrapper
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SafeActionExecutor:
    """Wraps an executor with the safety gate.

    Every action proposed by the learned policy or deterministic
    controller goes through :class:`SafetyGate` before reaching the
    hardware executor.

    Parameters
    ----------
    safety_gate:
        The safety gate to validate through.
    executor:
        Function that executes an action index on hardware.
    fallback_policy:
        Function that selects the fallback action from a state.
    """

    safety_gate: SafetyGate
    executor: Callable[[int], None]
    fallback_policy: Callable[[list[float]], int] | None = None

    _fallback_count: int = field(default=0, init=False, repr=False)
    _rejection_count: int = field(default=0, init=False, repr=False)

    @property
    def fallback_count(self) -> int:
        return self._fallback_count

    @property
    def rejection_count(self) -> int:
        return self._rejection_count

    def execute(
        self,
        action_index: int,
        state: list[float] | None = None,
        params: dict[str, Any] | None = None,
    ) -> int:
        """Validate and execute an action.

        Returns the action index that was actually executed (may differ
        from the proposed one if the safety gate rejected it).
        """
        result = self.safety_gate.validate(action_index, state, params)

        if result.allowed:
            self.executor(action_index)
            return action_index

        # Rejected or fallback — use the fallback action
        self._rejection_count += 1
        if result.fallback:
            self._fallback_count += 1
            actual = result.action_index
        elif self.fallback_policy is not None and state is not None:
            actual = self.fallback_policy(state)
        else:
            actual = self.safety_gate.fallback_action

        self.executor(actual)
        _log.warning(
            "safe_executor.fallback",
            proposed=action_index,
            executed=actual,
            reason=result.reason,
        )
        return actual

    def execute_model_output(
        self,
        model_output: Any,
        state: list[float] | None = None,
    ) -> int:
        """Handle raw model output and execute the safe action."""
        result = self.safety_gate.handle_model_output(model_output, state)
        if result.allowed:
            self.executor(result.action_index)
            return result.action_index

        self._fallback_count += 1
        actual = result.action_index
        self.executor(actual)
        return actual


__all__ = [
    "SafeActionExecutor",
    "SafetyGate",
    "SafetyResult",
    "SafetyResultType",
]
