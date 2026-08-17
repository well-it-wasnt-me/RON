"""Body language request types.

A *request* is a high-level intent the body-language engine knows how
to perform. Each request knows how to translate itself into a
:class:`ServoTimeline` (a list of concurrent + sequential servo
animations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

SERVO_HEAD_PAN = "pan"
SERVO_HEAD_TILT = "tilt"
SERVO_LEFT_ARM = "left_arm"
SERVO_RIGHT_ARM = "right_arm"


@dataclass(slots=True, frozen=True)
class ServoCalibration:
    """Per-servo calibration (limits, centre, sign)."""

    name: str
    min_angle: float
    max_angle: float
    center_angle: float
    inverted: bool = False

    def clamp(self, angle: float) -> float:
        return max(self.min_angle, min(self.max_angle, angle))

    def normalised(self, angle: float) -> float:
        """Return a value in ``[-1, 1]`` relative to the servo range."""
        half = (self.max_angle - self.min_angle) / 2.0
        if half <= 0:
            return 0.0
        mid = (self.max_angle + self.min_angle) / 2.0
        v = (angle - mid) / half
        return max(-1.0, min(1.0, v))

    def from_normalised(self, value: float) -> float:
        v = max(-1.0, min(1.0, value))
        half = (self.max_angle - self.min_angle) / 2.0
        mid = (self.max_angle + self.min_angle) / 2.0
        if self.inverted:
            v = -v
        return self.clamp(mid + v * half)


DEFAULT_CALIBRATION: dict[str, ServoCalibration] = {
    SERVO_HEAD_PAN: ServoCalibration(SERVO_HEAD_PAN, 30.0, 150.0, 90.0),
    SERVO_HEAD_TILT: ServoCalibration(SERVO_HEAD_TILT, 45.0, 135.0, 90.0),
    SERVO_LEFT_ARM: ServoCalibration(SERVO_LEFT_ARM, 20.0, 160.0, 90.0),
    SERVO_RIGHT_ARM: ServoCalibration(SERVO_RIGHT_ARM, 20.0, 160.0, 90.0),
}


@dataclass(slots=True, frozen=True)
class ServoFrame:
    """A target angle for every named servo at one instant."""

    targets: dict[str, float] = field(default_factory=dict)
    duration_s: float = 0.4

    def get(self, name: str, default: float) -> float:
        return self.targets.get(name, default)


@runtime_checkable
class BodyRequest(Protocol):
    """A high-level body-language request."""

    @property
    def name(self) -> str: ...

    def frames(self) -> list[ServoFrame]: ...


@dataclass(slots=True, frozen=True)
class _BaseRequest:
    name: str

    def frames(self) -> list[ServoFrame]:
        return []


def _head(center: float, pan: float, tilt: float, dur: float = 0.4) -> ServoFrame:
    return ServoFrame(
        targets={
            SERVO_HEAD_PAN: center + (pan - center) * 0.5,
            SERVO_HEAD_TILT: center + (tilt - center) * 0.5,
        },
        duration_s=dur,
    )


def _arms(center: float, left: float, right: float, dur: float = 0.4) -> ServoFrame:
    return ServoFrame(
        targets={
            SERVO_LEFT_ARM: center + (left - center) * 0.5,
            SERVO_RIGHT_ARM: center + (right - center) * 0.5,
        },
        duration_s=dur,
    )


CENTER = 90.0
MIN = 20.0
MAX = 160.0


@dataclass(slots=True, frozen=True)
class HeadTiltRequest(_BaseRequest):
    name: str = "head_tilt"
    direction: str = "left"
    amount: float = 15.0
    duration_s: float = 0.4

    def frames(self) -> list[ServoFrame]:
        sign = -1.0 if self.direction == "left" else 1.0
        return [
            ServoFrame(
                targets={SERVO_HEAD_TILT: max(MIN, min(MAX, CENTER + sign * self.amount))},
                duration_s=self.duration_s,
            )
        ]


@dataclass(slots=True, frozen=True)
class HeadNod(_BaseRequest):
    name: str = "head_nod"
    amplitude: float = 15.0
    duration_s: float = 0.5

    def frames(self) -> list[ServoFrame]:
        fwd = max(MIN, min(MAX, CENTER - self.amplitude))
        return [
            ServoFrame(targets={SERVO_HEAD_TILT: fwd}, duration_s=self.duration_s / 2),
            ServoFrame(targets={SERVO_HEAD_TILT: CENTER}, duration_s=self.duration_s / 2),
        ]


@dataclass(slots=True, frozen=True)
class LookLeft(_BaseRequest):
    name: str = "look_left"
    amount: float = 30.0
    duration_s: float = 0.3

    def frames(self) -> list[ServoFrame]:
        return [
            ServoFrame(
                targets={SERVO_HEAD_PAN: max(MIN, min(MAX, CENTER - self.amount))},
                duration_s=self.duration_s,
            )
        ]


@dataclass(slots=True, frozen=True)
class LookRight(_BaseRequest):
    name: str = "look_right"
    amount: float = 30.0
    duration_s: float = 0.3

    def frames(self) -> list[ServoFrame]:
        return [
            ServoFrame(
                targets={SERVO_HEAD_PAN: max(MIN, min(MAX, CENTER + self.amount))},
                duration_s=self.duration_s,
            )
        ]


@dataclass(slots=True, frozen=True)
class ArmsRelax(_BaseRequest):
    name: str = "arms_relax"
    duration_s: float = 0.5

    def frames(self) -> list[ServoFrame]:
        return [
            ServoFrame(
                targets={SERVO_LEFT_ARM: CENTER, SERVO_RIGHT_ARM: CENTER},
                duration_s=self.duration_s,
            )
        ]


@dataclass(slots=True, frozen=True)
class ArmsOpen(_BaseRequest):
    name: str = "arms_open"
    amount: float = 20.0
    duration_s: float = 0.5

    def frames(self) -> list[ServoFrame]:
        return [
            ServoFrame(
                targets={
                    SERVO_LEFT_ARM: max(MIN, min(MAX, CENTER - self.amount)),
                    SERVO_RIGHT_ARM: max(MIN, min(MAX, CENTER + self.amount)),
                },
                duration_s=self.duration_s,
            )
        ]


@dataclass(slots=True, frozen=True)
class Wave(_BaseRequest):
    name: str = "wave"
    amplitude: float = 30.0
    duration_s: float = 1.0

    def frames(self) -> list[ServoFrame]:
        up = max(MIN, min(MAX, CENTER - self.amplitude))
        return [
            ServoFrame(targets={SERVO_RIGHT_ARM: up}, duration_s=self.duration_s / 4),
            ServoFrame(targets={SERVO_RIGHT_ARM: CENTER}, duration_s=self.duration_s / 4),
            ServoFrame(targets={SERVO_RIGHT_ARM: up}, duration_s=self.duration_s / 4),
            ServoFrame(targets={SERVO_RIGHT_ARM: CENTER}, duration_s=self.duration_s / 4),
        ]


@dataclass(slots=True, frozen=True)
class Celebrate(_BaseRequest):
    name: str = "celebrate"
    duration_s: float = 0.6

    def frames(self) -> list[ServoFrame]:
        up = max(MIN, min(MAX, CENTER - 40.0))
        return [
            ServoFrame(
                targets={
                    SERVO_LEFT_ARM: up,
                    SERVO_RIGHT_ARM: up,
                    SERVO_HEAD_TILT: max(MIN, min(MAX, CENTER - 20.0)),
                },
                duration_s=self.duration_s / 2,
            ),
            ServoFrame(
                targets={SERVO_LEFT_ARM: CENTER, SERVO_RIGHT_ARM: CENTER, SERVO_HEAD_TILT: CENTER},
                duration_s=self.duration_s / 2,
            ),
        ]


@dataclass(slots=True, frozen=True)
class Shrug(_BaseRequest):
    name: str = "shrug"
    duration_s: float = 0.5

    def frames(self) -> list[ServoFrame]:
        up = max(MIN, min(MAX, CENTER - 30.0))
        return [
            ServoFrame(
                targets={
                    SERVO_LEFT_ARM: up,
                    SERVO_RIGHT_ARM: up,
                    SERVO_HEAD_TILT: max(MIN, min(MAX, CENTER - 10.0)),
                },
                duration_s=self.duration_s,
            )
        ]


@dataclass(slots=True, frozen=True)
class Greet(_BaseRequest):
    name: str = "greet"
    duration_s: float = 0.8

    def frames(self) -> list[ServoFrame]:
        up = max(MIN, min(MAX, CENTER - 30.0))
        return [
            ServoFrame(
                targets={SERVO_RIGHT_ARM: up, SERVO_HEAD_TILT: max(MIN, min(MAX, CENTER - 10.0))},
                duration_s=self.duration_s / 3,
            ),
            ServoFrame(
                targets={SERVO_RIGHT_ARM: CENTER, SERVO_HEAD_TILT: CENTER},
                duration_s=self.duration_s / 3,
            ),
            ServoFrame(
                targets={SERVO_RIGHT_ARM: up, SERVO_HEAD_TILT: max(MIN, min(MAX, CENTER - 10.0))},
                duration_s=self.duration_s / 3,
            ),
        ]


__all__ = [
    "DEFAULT_CALIBRATION",
    "SERVO_HEAD_PAN",
    "SERVO_HEAD_TILT",
    "SERVO_LEFT_ARM",
    "SERVO_RIGHT_ARM",
    "ArmsOpen",
    "ArmsRelax",
    "BodyRequest",
    "Celebrate",
    "Greet",
    "HeadNod",
    "HeadTiltRequest",
    "LookLeft",
    "LookRight",
    "ServoCalibration",
    "ServoFrame",
    "Shrug",
    "Wave",
]
