"""Tests for the CircuitPython GC9A01 driver.

These tests skip themselves when the displayio dependencies are not
installed (i.e., when running on a dev machine without Pi 5 packages).
"""

from __future__ import annotations

import pytest

from robot.hardware.displays.circuitpython import CircuitPythonDisplay

circuitpython = pytest.importorskip(
    "adafruit_gc9a01a",
    reason="adafruit-circuitpython-gc9a01a not installed (Pi-only)",
)


async def test_circuitpython_display_implements_protocol() -> None:
    """The CircuitPython display satisfies the Display Protocol."""

    # Construct with explicit width/height that don't require hardware init.
    # We expect the constructor to fail on non-Pi platforms because it tries
    # to open the SPI bus. So just check the type / attribute access pattern.
    assert hasattr(CircuitPythonDisplay, "show")
    assert hasattr(CircuitPythonDisplay, "fill")
    assert hasattr(CircuitPythonDisplay, "clear")
    assert hasattr(CircuitPythonDisplay, "close")
    assert hasattr(CircuitPythonDisplay, "width")
    assert hasattr(CircuitPythonDisplay, "height")


def test_circuitpython_display_rejects_bad_dimensions() -> None:
    with pytest.raises(ValueError):
        CircuitPythonDisplay(width=0, height=32)
    with pytest.raises(ValueError):
        CircuitPythonDisplay(width=32, height=0)


def test_circuitpython_display_rejects_bad_rotation() -> None:
    with pytest.raises(ValueError):
        CircuitPythonDisplay(rotation=4)
