"""Easing functions for smooth motion.

All easings take ``t`` in [0.0, 1.0] and return a normalised progress value in
[0.0, 1.0]. Easings are pure functions and easy to unit-test.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Final, TypeAlias

Easing: TypeAlias = Callable[[float], float]


def linear(t: float) -> float:
    return float(t)


def ease_in_quad(t: float) -> float:
    return t * t


def ease_out_quad(t: float) -> float:
    return 1.0 - (1.0 - t) * (1.0 - t)


def ease_in_out_quad(t: float) -> float:
    return 2.0 * t * t if t < 0.5 else 1.0 - ((-2.0 * t + 2.0) ** 2) / 2.0


def ease_in_cubic(t: float) -> float:
    return t * t * t


def ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_cubic(t: float) -> float:
    return 4.0 * t * t * t if t < 0.5 else 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0


def ease_out_elastic(t: float) -> float:
    if t in (0.0, 1.0):
        return float(t)
    c4 = (2.0 * math.pi) / 3.0
    return float(2.0 ** (-10.0 * t) * math.sin((t * 10.0 - 0.75) * c4) + 1.0)


def ease_out_bounce(t: float) -> float:
    n1, d1 = 7.5625, 2.75
    if t < 1.0 / d1:
        return n1 * t * t
    if t < 2.0 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    if t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    t -= 2.625 / d1
    return n1 * t * t + 0.984375


builtin_easings: Final[dict[str, Easing]] = {
    "linear": linear,
    "ease_in_quad": ease_in_quad,
    "ease_out_quad": ease_out_quad,
    "ease_in_out_quad": ease_in_out_quad,
    "ease_in_cubic": ease_in_cubic,
    "ease_out_cubic": ease_out_cubic,
    "ease_in_out_cubic": ease_in_out_cubic,
    "ease_out_elastic": ease_out_elastic,
    "ease_out_bounce": ease_out_bounce,
}


__all__ = ["Easing", "builtin_easings"]
