"""USB webcam driver backed by OpenCV (``cv2.VideoCapture``).

Provides the same async ``Microphone``-style API as the mock camera but
reads frames from a real USB V4L2 device (e.g. the FHD camera module on
the Pi 5).

Install with::

    uv pip install opencv-python-headless

Notes
-----

OpenCV's ``VideoCapture.read()`` is synchronous and blocks the calling
thread for ~10-30 ms per frame on a Pi 5. To avoid blocking the event
loop we run a dedicated capture thread that owns the underlying
``VideoCapture`` and pushes the latest frame into an ``asyncio.Queue``.
``capture()`` pops the latest frame without blocking.

The camera automatically drops the first ``warmup_frames`` frames after
opening (default 5) because many USB webcams deliver under-exposed or
blank frames during the first few hundred milliseconds. If the camera
is disconnected mid-stream, the driver reopens it after a short delay.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from robot.interfaces.camera import Camera, Frame
from robot.logging import get_logger

_log = get_logger("hardware.sensors.camera.usb")


@dataclass(slots=True)
class UsbCamera(Camera):
    """Real USB webcam.

    Parameters
    ----------
    device:
        V4L2 device index (e.g. ``0`` for ``/dev/video0``) or path
        (``"/dev/video0"``).
    width, height:
        Requested frame size. The driver will ask V4L2 for this size but
        the camera may negotiate a different one if the requested size
        is unsupported - the actual size is reported in :attr:`width` /
        :attr:`height` after the first :meth:`capture`.
    fps:
        Requested frame rate.
    warmup_frames:
        Number of frames to discard after opening. USB webcams often
        deliver under-exposed frames for the first few hundred
        milliseconds. Default 5.
    """

    device: int | str = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    warmup_frames: int = 5
    _frame_timeout_s: float = field(default=0.5, init=False)
    _latest: Frame | None = field(default=None, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _actual_width: int = field(default=0, init=False)
    _actual_height: int = field(default=0, init=False)
    _captured: int = field(default=0, init=False)
    _dropped: int = field(default=0, init=False)
    _closed: bool = field(default=False, init=False)
    _reconnect_attempts: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - hardware-specific
            raise RuntimeError(
                f"opencv-python is required for UsbCamera: {exc!r}. "
                "Install with `uv pip install opencv-python-headless`."
            ) from exc
        self._cv2 = cv2
        self._open_camera()

    # ------------------------------------------------------------------ open
    def _open_camera(self) -> None:
        """Open the camera device, set properties, and start the capture thread."""
        cv2 = self._cv2
        cap = cv2.VideoCapture(self.device)
        if not cap.isOpened():
            raise RuntimeError(
                f"could not open video device {self.device!r}; "
                f"check `ls /dev/video*` and that the user is in the `video` group."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap = cap
        self._actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        self._actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        # Discard warmup frames.
        for _ in range(self.warmup_frames):
            cap.read()
        self._captured = 0
        self._dropped = 0
        self._reconnect_attempts = 0
        _log.info(
            "usb_camera.opened",
            device=self.device,
            requested=f"{self.width}x{self.height}",
            actual=f"{self._actual_width}x{self._actual_height}",
            fps=self.fps,
            warmup=self.warmup_frames,
        )
        # Start the background capture thread.
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="UsbCamera-capture", daemon=True
        )
        self._thread.start()

    @property
    def actual_width(self) -> int:
        return self._actual_width

    @property
    def actual_height(self) -> int:
        return self._actual_height

    @property
    def captured(self) -> int:
        return self._captured

    @property
    def dropped(self) -> int:
        return self._dropped

    # ------------------------------------------------------------------ loop
    def _capture_loop(self) -> None:
        """Capture frames in a background thread; publish the latest."""
        cv2 = self._cv2
        cap = self._cap
        stop = self._stop
        consecutive_failures = 0
        max_consecutive_failures = 30  # ~1 s at 30 FPS before reconnect
        try:
            while not stop.is_set():
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        _log.warning(
                            "usb_camera.read_failures",
                            consecutive=consecutive_failures,
                            message="camera may be disconnected, attempting reconnect",
                        )
                        # Try to reopen the camera.
                        with contextlib.suppress(Exception):
                            cap.release()
                        time.sleep(0.5)
                        try:
                            new_cap = cv2.VideoCapture(self.device)
                            if new_cap.isOpened():
                                new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                                new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                                new_cap.set(cv2.CAP_PROP_FPS, self.fps)
                                # Discard warmup frames after reconnect.
                                for _ in range(self.warmup_frames):
                                    new_cap.read()
                                self._cap = new_cap
                                cap = new_cap
                                self._reconnect_attempts += 1
                                _log.info(
                                    "usb_camera.reconnected",
                                    device=self.device,
                                    attempts=self._reconnect_attempts,
                                )
                            else:
                                _log.error("usb_camera.reconnect_failed", device=self.device)
                        except Exception:
                            _log.exception("usb_camera.reconnect_error")
                        consecutive_failures = 0
                    time.sleep(0.033)  # ~30 FPS fallback
                    continue
                consecutive_failures = 0
                # Convert BGR (OpenCV default) -> RGB888.
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                h, w = frame_rgb.shape[:2]
                self._latest = Frame(
                    width=w,
                    height=h,
                    pixels=frame_rgb.tobytes(),
                    timestamp=time.time(),
                )
                self._captured += 1
        except Exception:  # pragma: no cover - hardware-specific
            _log.exception("usb_camera.capture_loop_crashed")

    async def capture(self) -> Frame:
        if self._closed:
            raise RuntimeError("camera is closed")
        # Wait up to frame_timeout_s for a frame.
        deadline = time.time() + self._frame_timeout_s
        while self._latest is None:
            if time.time() > deadline:
                raise RuntimeError("UsbCamera: no frame captured yet")
            await asyncio.sleep(0.01)
        frame = self._latest
        assert frame is not None
        return frame

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        with contextlib.suppress(Exception):
            self._cap.release()
        _log.info(
            "usb_camera.closed",
            device=self.device,
            captured=self._captured,
            dropped=self._dropped,
            reconnects=self._reconnect_attempts,
        )


@contextlib.asynccontextmanager
async def open_camera(device: int | str = 0, **kwargs: int) -> AsyncIterator[UsbCamera]:
    """Async context manager that opens a ``UsbCamera`` and closes it on exit."""
    cam = UsbCamera(device=device, **kwargs)
    try:
        yield cam
    finally:
        await cam.close()


__all__ = ["UsbCamera", "open_camera"]
