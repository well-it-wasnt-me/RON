"""Tests for the USB webcam driver."""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import numpy  # noqa: F401  (used by fake frame in type stubs)


class _FakeCap:
    """A minimal stand-in for cv2.VideoCapture."""

    def __init__(self, dev: int) -> None:
        self.dev = dev
        self.opened = True
        self.idx = 0

    def isOpened(self) -> bool:  # noqa: N802,type-ignore[override,misc]
        return self.opened

    def get(self, prop: int) -> float:
        return 320.0 if prop == 3 else 240.0 if prop == 4 else 30.0

    def set(self, prop: int, value: float) -> None:
        return None

    def read(self) -> tuple[bool, object]:
        import numpy as np

        ok = True
        w, h = 2, 2
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        return ok, frame

    def release(self) -> None:
        self.opened = False


def _install_fake_cv2(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.VideoCapture = _FakeCap  # type: ignore[attr-defined]
    fake_cv2.CAP_PROP_FRAME_WIDTH = 3  # type: ignore[attr-defined]
    fake_cv2.CAP_PROP_FRAME_HEIGHT = 4  # type: ignore[attr-defined]
    fake_cv2.CAP_PROP_FPS = 5  # type: ignore[attr-defined]
    fake_numpy = types.ModuleType("numpy")
    sys.modules.setdefault("numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)


async def test_usb_camera_initialises_with_device_and_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The camera opens with the requested size and exposes it."""
    _install_fake_cv2(monkeypatch)
    from robot.hardware.sensors.usb_camera import UsbCamera

    cam = UsbCamera(device=0, width=320, height=240, fps=30)
    try:
        assert cam.width == 320
        assert cam.height == 240
        assert cam.actual_width == 320
        assert cam.actual_height == 240
    finally:
        await cam.close()


async def test_usb_camera_capture_returns_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """capture() returns a Frame once a frame is pushed."""
    _install_fake_cv2(monkeypatch)
    from robot.hardware.sensors.usb_camera import UsbCamera
    from robot.interfaces.camera import Frame

    cam = UsbCamera(device=0, width=2, height=2, fps=30)
    # Push a frame directly (bypass the background thread for tests).
    import time as _time

    cam._latest = Frame(
        width=2,
        height=2,
        pixels=b"\xff\x00\x00" * 4,
        timestamp=_time.time(),
    )
    frame = await cam.capture()
    assert frame.width == 2
    assert frame.height == 2
    assert len(frame.pixels) == 12
    await cam.close()


def test_usb_camera_missing_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without cv2 installed, construction must raise a friendly error.

    This test is skipped when OpenCV IS installed because the module
    caches the import at load time. The test only works in a clean
    environment without cv2.
    """
    try:
        import cv2  # noqa: F401

        pytest.skip("opencv is installed; cannot test missing-opencv path")
    except ImportError:
        pass
    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.delitem(sys.modules, "cv2", raising=False)
    from robot.hardware.sensors import usb_camera

    with pytest.raises(RuntimeError, match="opencv-python is required"):
        usb_camera.UsbCamera(device=0)
