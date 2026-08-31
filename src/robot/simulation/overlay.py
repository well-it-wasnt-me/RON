"""Render a stick-figure visualisation of the four servos.

The overlay draws a simple body diagram (head + neck + two arms) and
highlights each servo's current target angle with a coloured line. The
result is composited on top of the face framebuffer so a single
:class:`EyeFrame` per frame is enough to drive the simulation display.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from robot.body_language.engine import Pose
from robot.body_language.requests import (
    SERVO_HEAD_PAN,
    SERVO_HEAD_TILT,
    SERVO_LEFT_ARM,
    SERVO_RIGHT_ARM,
    ServoCalibration,
)


@dataclass(slots=True, frozen=True)
class OverlayConfig:
    """Configuration for the stick-figure overlay."""

    background: tuple[int, int, int] = (40, 40, 50)
    body_color: tuple[int, int, int] = (220, 220, 230)
    servo_color: tuple[int, int, int] = (255, 180, 60)
    target_color: tuple[int, int, int] = (100, 255, 100)
    pan_color: tuple[int, int, int] = (255, 100, 100)
    tilt_color: tuple[int, int, int] = (100, 100, 255)
    left_arm_color: tuple[int, int, int] = (100, 255, 100)
    right_arm_color: tuple[int, int, int] = (255, 200, 100)


class ServoOverlay:
    """Draw the four-servo stick figure into a face framebuffer."""

    def __init__(
        self,
        calibration: dict[str, ServoCalibration],
        config: OverlayConfig | None = None,
    ) -> None:
        self._calibration = calibration
        self._config = config or OverlayConfig()

    def composite(
        self,
        frame_pixels: bytearray,
        width: int,
        height: int,
        pose: Pose,
    ) -> bytearray:
        out = bytearray(frame_pixels)
        body_top = int(height * 0.70)
        self._fill_rect(out, width, height, 0, body_top, width, height, self._config.background)
        cx = width / 2.0
        head_y = body_top + height * 0.04
        neck_y = body_top + height * 0.13
        shoulder_y = body_top + height * 0.18
        waist_y = body_top + height * 0.28
        self._line(out, width, height, cx, head_y, cx, waist_y, self._config.body_color, 4)
        shoulder_half = width * 0.18
        self._line(
            out,
            width,
            height,
            cx - shoulder_half,
            shoulder_y,
            cx + shoulder_half,
            shoulder_y,
            self._config.body_color,
            4,
        )
        pan_target = pose.get(SERVO_HEAD_PAN, 90.0)
        tilt_target = pose.get(SERVO_HEAD_TILT, 90.0)
        left_arm_target = pose.get(SERVO_LEFT_ARM, 90.0)
        right_arm_target = pose.get(SERVO_RIGHT_ARM, 90.0)
        head_tilt = (tilt_target - 90.0) * 0.04
        self._line(out, width, height, cx, neck_y, cx, head_y, self._config.body_color, 4)
        self._circle(out, width, height, cx, head_y, 12, self._config.body_color)
        self._line(
            out,
            width,
            height,
            cx,
            head_y,
            cx + (pan_target - 90.0) * 0.5,
            head_y - 8,
            self._config.pan_color,
            2,
        )
        self._line(
            out,
            width,
            height,
            cx,
            head_y,
            cx,
            head_y - 12 - head_tilt,
            self._config.tilt_color,
            2,
        )
        left_wrist = self._arm_endpoint(
            cx - shoulder_half, shoulder_y, left_arm_target, "left", height * 0.10
        )
        right_wrist = self._arm_endpoint(
            cx + shoulder_half, shoulder_y, right_arm_target, "right", height * 0.10
        )
        self._line(
            out,
            width,
            height,
            cx - shoulder_half,
            shoulder_y,
            *left_wrist,
            self._config.left_arm_color,
            3,
        )
        self._line(
            out,
            width,
            height,
            cx + shoulder_half,
            shoulder_y,
            *right_wrist,
            self._config.right_arm_color,
            3,
        )
        for x, y in (
            (cx - shoulder_half, shoulder_y),
            (cx + shoulder_half, shoulder_y),
            (cx, head_y),
        ):
            self._circle(out, width, height, x, y, 4, self._config.servo_color)
        for name, x, y, color, target in (
            (SERVO_HEAD_PAN, cx, head_y, self._config.pan_color, pan_target),
            (SERVO_HEAD_TILT, cx, head_y, self._config.tilt_color, tilt_target),
            (
                SERVO_LEFT_ARM,
                left_wrist[0],
                left_wrist[1],
                self._config.left_arm_color,
                left_arm_target,
            ),
            (
                SERVO_RIGHT_ARM,
                right_wrist[0],
                right_wrist[1],
                self._config.right_arm_color,
                right_arm_target,
            ),
        ):
            self._draw_target_indicator(
                out, width, height, x, y, target, name in self._calibration, color
            )
        return out

    def _arm_endpoint(
        self, sx: float, sy: float, angle: float, side: str, length: float
    ) -> tuple[float, float]:
        # Left arm extends to the left, right arm to the right.
        # At 90° the arm hangs straight down; higher angles raise it outward.
        sign = -1.0 if side == "left" else 1.0
        rad = math.radians(angle - 90.0)
        dx = sign * length * math.sin(rad)
        dy = length * math.cos(rad)
        return (sx + dx, sy + dy)

    def _draw_target_indicator(
        self,
        out: bytearray,
        width: int,
        height: int,
        x: float,
        y: float,
        target: float,
        calibrated: bool,
        color: tuple[int, int, int],
    ) -> None:
        bar_len = 18
        bar_y = y + 12
        self._line(out, width, height, x - bar_len / 2, bar_y, x + bar_len / 2, bar_y, color, 2)
        if calibrated:
            t = (target - 0.0) / 180.0
            t = max(0.0, min(1.0, t))
            dot_x = x - bar_len / 2 + bar_len * t
            self._circle(out, width, height, dot_x, bar_y, 3, (255, 255, 255))

    def _fill_rect(
        self,
        out: bytearray,
        width: int,
        height: int,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
    ) -> None:
        r, g, b = color
        for y in range(max(0, y0), min(height, y1)):
            for x in range(max(0, x0), min(width, x1)):
                idx = (y * width + x) * 3
                out[idx] = r
                out[idx + 1] = g
                out[idx + 2] = b

    def _line(
        self,
        out: bytearray,
        width: int,
        height: int,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        steps = max(2, int(max(abs(x1 - x0), abs(y1 - y0)) * 2))
        for i in range(steps + 1):
            t = i / steps
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            self._dot(out, width, height, x, y, color, thickness)

    def _circle(
        self,
        out: bytearray,
        width: int,
        height: int,
        cx: float,
        cy: float,
        r: float,
        color: tuple[int, int, int],
    ) -> None:
        x0 = max(0, int(cx - r))
        x1 = min(width - 1, int(cx + r))
        y0 = max(0, int(cy - r))
        y1 = min(height - 1, int(cy + r))
        r2 = r * r
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                    self._set_pixel(out, width, x, y, color)

    def _dot(
        self,
        out: bytearray,
        width: int,
        height: int,
        x: float,
        y: float,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        for dx in range(-thickness // 2, thickness // 2 + 1):
            for dy in range(-thickness // 2, thickness // 2 + 1):
                xx, yy = round(x + dx), round(y + dy)
                if 0 <= xx < width and 0 <= yy < height:
                    self._set_pixel(out, width, xx, yy, color)

    def _set_pixel(
        self,
        out: bytearray,
        width: int,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        idx = (y * width + x) * 3
        out[idx] = color[0]
        out[idx + 1] = color[1]
        out[idx + 2] = color[2]


__all__ = ["OverlayConfig", "ServoOverlay"]
