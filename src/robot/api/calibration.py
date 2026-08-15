"""Calibration API routes for servo and display tuning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from robot.api.schemas import (
    CalibrateServoResponse,
    ClearDisplayResponse,
    DisplayConfigResponse,
    ReleaseAllResponse,
    ServoListResponse,
    ServoMoveResponse,
    ServoReleaseResponse,
    TestPatternResponse,
)

router = APIRouter(prefix="/calibration", tags=["calibration"])


@dataclass(slots=True)
class _CalibrationState:
    servo_controller: Any = None
    display: Any = None
    settings: Any = None


_state = _CalibrationState()


def set_calibration_state(
    servo_controller: Any = None,
    display: Any = None,
    settings: Any = None,
) -> None:
    """Wire the calibration routes to real hardware."""
    _state.servo_controller = servo_controller
    _state.display = display
    _state.settings = settings


def _make_gradient(w: int, h: int) -> bytes:
    buf = bytearray(w * h * 3)
    for y in range(h):
        for x in range(w):
            idx = (y * w + x) * 3
            buf[idx] = int(255 * x / max(1, w - 1))
            buf[idx + 1] = int(255 * y / max(1, h - 1))
            buf[idx + 2] = 128
    return bytes(buf)


def _make_grid(w: int, h: int) -> bytes:
    buf = bytearray(w * h * 3)
    for y in range(h):
        for x in range(w):
            idx = (y * w + x) * 3
            if x % 40 == 0 or y % 40 == 0:
                buf[idx : idx + 3] = b"\xff\xff\xff"
    return bytes(buf)


def _make_checkerboard(w: int, h: int) -> bytes:
    buf = bytearray(w * h * 3)
    for y in range(h):
        for x in range(w):
            idx = (y * w + x) * 3
            if (x // 20 + y // 20) % 2 == 0:
                buf[idx : idx + 3] = b"\xff\xff\xff"
    return bytes(buf)


def _make_solid(w: int, h: int, r: int = 255, g: int = 0, b: int = 0) -> bytes:
    return bytes([r, g, b]) * (w * h)


def _get_servo(name: str) -> Any:
    if _state.servo_controller is None:
        raise HTTPException(status_code=503, detail="No servo controller available")
    try:
        return _state.servo_controller.get(name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Servo {name!r} not found") from exc


def _servo_limits(name: str, servo: Any) -> tuple[float, float, float]:
    """Return the exact limits used by the active servo controller."""
    channel = getattr(servo, "channel", None)
    if channel is not None:
        minimum = float(channel.min_angle_deg)
        maximum = float(channel.max_angle_deg)
        centre = float(getattr(channel, "center_angle_deg", (minimum + maximum) / 2.0))
        return minimum, maximum, centre
    return 0.0, 180.0, 90.0


@router.get("/servos", summary="List all servos", response_model=ServoListResponse)
async def list_servos() -> ServoListResponse:
    if _state.servo_controller is None:
        raise HTTPException(status_code=503, detail="No servo controller available")
    try:
        result = []
        for servo in _state.servo_controller.all():
            minimum, maximum, centre = _servo_limits(servo.name, servo)
            result.append(
                {
                    "name": servo.name,
                    "angle": servo.angle,
                    "min_angle": minimum,
                    "max_angle": maximum,
                    "center_angle": centre,
                }
            )
        return ServoListResponse(servos=result)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Unable to enumerate servos: {exc}") from exc


@router.post(
    "/servos/{name}/move", summary="Move a servo to an angle", response_model=ServoMoveResponse
)
async def move_servo(name: str, angle: float, duration_s: float = 0.4) -> ServoMoveResponse:
    servo = _get_servo(name)
    minimum, maximum, _ = _servo_limits(name, servo)
    if not minimum <= angle <= maximum:
        raise HTTPException(
            status_code=422, detail=f"Angle must be between {minimum} and {maximum} degrees"
        )
    try:
        await servo.move_to(angle, duration_s=duration_s)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Unable to move servo {name!r}: {exc}"
        ) from exc
    return ServoMoveResponse(name=name, angle=angle, duration_s=duration_s)


@router.post(
    "/servos/{name}/release", summary="Release a servo", response_model=ServoReleaseResponse
)
async def release_servo(name: str) -> ServoReleaseResponse:
    servo = _get_servo(name)
    try:
        await servo.release()
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Unable to release servo {name!r}: {exc}"
        ) from exc
    return ServoReleaseResponse(name=name, released=True)


@router.post("/servos/release_all", summary="Release all servos", response_model=ReleaseAllResponse)
async def release_all_servos() -> ReleaseAllResponse:
    if _state.servo_controller is None:
        raise HTTPException(status_code=503, detail="No servo controller available")
    try:
        await _state.servo_controller.release_all()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to release servos: {exc}") from exc
    return ReleaseAllResponse(released=True)


@router.post(
    "/servos/calibrate/{name}",
    summary="Run servo calibration sequence",
    response_model=CalibrateServoResponse,
)
async def calibrate_servo(
    name: str,
    include_limits: bool = Query(False, description="Also visit configured endpoints"),
) -> CalibrateServoResponse:
    """Centre a servo safely, optionally testing configured endpoints."""
    servo = _get_servo(name)
    minimum, maximum, centre = _servo_limits(name, servo)
    positions = [("centre", centre, 0.5)]
    if include_limits:
        positions = [
            ("min", minimum, 0.8),
            ("centre", centre, 0.8),
            ("max", maximum, 0.8),
            ("centre", centre, 0.8),
        ]

    import anyio

    results = []
    try:
        for label, angle, pause in positions:
            await servo.move_to(angle, duration_s=0.4)
            await anyio.sleep(pause)
            results.append({"position": label, "angle": angle, "actual_angle": servo.angle})
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Calibration failed for servo {name!r}: {exc}"
        ) from exc
    return CalibrateServoResponse(servo=name, sequence=results)


@router.get("/display", summary="Get display configuration", response_model=DisplayConfigResponse)
async def get_display_config() -> DisplayConfigResponse:
    if _state.settings is None:
        raise HTTPException(status_code=503, detail="No settings available")
    d = _state.settings.displays
    return DisplayConfigResponse(
        backend=d.backend,
        width=d.width,
        height=d.height,
        rotation=d.rotation,
        spi_hz=d.spi_hz,
        invert=d.invert,
    )


@router.post(
    "/display/test_pattern", summary="Show a test pattern", response_model=TestPatternResponse
)
async def show_test_pattern(pattern: str = "gradient") -> TestPatternResponse:
    if _state.display is None:
        raise HTTPException(status_code=503, detail="No display available")
    w, h = _state.display.width, _state.display.height
    pattern_funcs: dict[str, Callable[[], bytes]] = {
        "gradient": lambda: _make_gradient(w, h),
        "grid": lambda: _make_grid(w, h),
        "checkerboard": lambda: _make_checkerboard(w, h),
        "solid": lambda: _make_solid(w, h),
    }
    if pattern not in pattern_funcs:
        raise HTTPException(status_code=400, detail=f"Unknown pattern: {pattern!r}")
    from robot.interfaces.display import EyeFrame

    await _state.display.show(EyeFrame(width=w, height=h, pixels=pattern_funcs[pattern]()))
    return TestPatternResponse(pattern=pattern, width=w, height=h)


@router.post("/display/clear", summary="Clear the display", response_model=ClearDisplayResponse)
async def clear_display() -> ClearDisplayResponse:
    if _state.display is None:
        raise HTTPException(status_code=503, detail="No display available")
    await _state.display.clear()
    return ClearDisplayResponse(cleared=True)


__all__ = ["router", "set_calibration_state"]
