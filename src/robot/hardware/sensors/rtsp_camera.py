"""RTSP camera driver backed by OpenCV (``cv2.VideoCapture``).

Provides the same async :class:`~robot.interfaces.camera.Camera` API.

Candidate model loss

Install wit
Candidate model loss
h::

    uv pip install opencv-python-headless

Notes
-----

RTSP streams are inherently more latent and fragile than local USB
devices.  The driver follows the same background-thread + latest-frame
pattern as :class:`UsbCamera`, with the following differences:

* OpenCV's ``CAP_PROP_BUFFERSIZE`` is set to 1 to minimise buffering
  latency.
* Reconnection logic is more aggressive and the back-off between reconnect attempts
  starts at 1 s and grows to 5 s.
* The ``FF_OPEN_TIMEOUT`` environment variable can be used to set the
  OpenCV/FFmpeg ``timeout`` parameter (in microseconds) that controls
  how long the underlying FFmpeg library waits before declaring an
  RTSP connection failed.  Default is 5 s (5.000.000 µs).
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from robot.interfaces.camera import Camera, Frame
from robot.logging import get_logger

_log = get_logger("hardware.sensors.camera.rtsp")


def _redact_url(url: str) -> str:
    """Redact credentials from a URL for safe logging/error messages."""
    import re

    return re.sub(r"(://[^:@/]+):[^@/]+@", r"\1:****@", url)


# Environment variable override for FFmpeg RTSP open timeout (microseconds).
_OPEN_TIMEOUT_ENV = "FF_OPEN_TIMEOUT"
_DEFAULT_OPEN_TIMEOUT_US = 5_000_000  # 5 seconds


@dataclass(slots=True)
class RtspCamera(Camera):
    """RTSP stream camera.

    Parameters
    ----------
    url:
        Full RTSP URL, e.g. ``rtsp://admin:pass@192.168.1.50:554/h264``.
    width, height:
        Requested frame size.  RTSP servers negotiate the actual size
        during the DESCRIBE/SETUP phase; the actual size is reported in
        :attr:`width` / :attr:`height` after the first :meth:`capture`.
    fps:
        Requested frame rate.
    reconnect_threshold:
        Number of consecutive read failures before attempting to
        reconnect.  Default 10 (≈ 0.3 s at 30 FPS).
    reconnect_initial_delay:
        Initial delay (seconds) before the first reconnect attempt.
        Subsequent delays grow linearly up to ``reconnect_max_delay``.
    reconnect_max_delay:
        Maximum delay (seconds) between reconnect attempts.
    """

    url: str
    width: int = 640
    height: int = 480
    fps: int = 30
    reconnect_threshold: int = 10
    reconnect_initial_delay: float = 1.0
    reconnect_max_delay: float = 5.0
    _frame_timeout_s: float = field(default=5.0, init=False)
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
        if not self.url:
            raise ValueError("RtspCamera requires a non-empty 'url'")
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - hardware-specific
            raise RuntimeError(
                f"opencv-python is required for RtspCamera: {exc!r}. "
                "Install with `uv pip install opencv-python-headless`."
            ) from exc
        self._cv2 = cv2
        self._open_stream()

    # ------------------------------------------------------------------ open

    def _open_stream(self) -> None:
        """Open the RTSP stream, set properties, and start the capture thread."""
        cv2 = self._cv2

        # FF_OPEN_TIMEOUT env var (microseconds) controls how long FFmpeg
        # waits before declaring an RTSP connection failed.  It is read by
        # the underlying FFmpeg library at open time.
        # Default: 5 000 000 µs (5 s) via _DEFAULT_OPEN_TIMEOUT_US.
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError(
                f"could not open RTSP stream {_redact_url(self.url)!r}; "
                f"check the URL, credentials, and network connectivity."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Minimise buffering to reduce latency.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._cap = cap
        self._actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        self._actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        self._captured = 0
        self._dropped = 0
        self._reconnect_attempts = 0
        _log.info(
            "rtsp_camera.opened",
            url=self._safe_url(),
            requested=f"{self.width}x{self.height}",
            actual=f"{self._actual_width}x{self._actual_height}",
            fps=self.fps,
        )
        # Start the background capture thread.
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="RtspCamera-capture", daemon=True
        )
        self._thread.start()

    def _safe_url(self) -> str:
        """Return the URL with any embedded password masked for logging."""
        if "@" in self.url:
            scheme, rest = self.url.split("://", 1)
            creds, host = rest.split("@", 1)
            if ":" in creds:
                user, _ = creds.split(":", 1)
                return f"{scheme}://{user}:***@{host}"
            return f"{scheme}://***@{host}"
        return self.url

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

    @property
    def reconnect_attempts(self) -> int:
        return self._reconnect_attempts

    # ------------------------------------------------------------------ loop

    def _capture_loop(self) -> None:
        """Capture frames in a background thread; publish the latest."""
        cv2 = self._cv2
        cap = self._cap
        stop = self._stop
        consecutive_failures = 0
        reconnect_delay = self.reconnect_initial_delay
        try:
            while not stop.is_set():
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    consecutive_failures += 1
                    if consecutive_failures >= self.reconnect_threshold:
                        _log.warning(
                            "rtsp_camera.read_failures",
                            consecutive=consecutive_failures,
                            delay=reconnect_delay,
                            message="RTSP stream may be disconnected, attempting reconnect",
                        )
                        # Try to reopen the stream.
                        with contextlib.suppress(Exception):
                            cap.release()
                        stop.wait(reconnect_delay)
                        try:
                            new_cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                            if new_cap.isOpened():
                                new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                                new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                                new_cap.set(cv2.CAP_PROP_FPS, self.fps)
                                new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                self._cap = new_cap
                                cap = new_cap
                                self._reconnect_attempts += 1
                                _log.info(
                                    "rtsp_camera.reconnected",
                                    url=self._safe_url(),
                                    attempts=self._reconnect_attempts,
                                )
                                reconnect_delay = self.reconnect_initial_delay
                            else:
                                _log.error(
                                    "rtsp_camera.reconnect_failed",
                                    url=self._safe_url(),
                                )
                                # Grow the back-off for next time.
                                reconnect_delay = min(
                                    reconnect_delay + self.reconnect_initial_delay,
                                    self.reconnect_max_delay,
                                )
                        except Exception:
                            _log.exception("rtsp_camera.reconnect_error")
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
            _log.exception("rtsp_camera.capture_loop_crashed")

    async def capture(self) -> Frame:
        if self._closed:
            raise RuntimeError("camera is closed")
        # Wait up to frame_timeout_s for a frame (RTSP can be slower).
        import asyncio

        deadline = time.time() + self._frame_timeout_s
        while self._latest is None:
            if time.time() > deadline:
                raise RuntimeError(
                    f"RtspCamera: no frame captured within {self._frame_timeout_s}s "
                    f"(stream may still be connecting or unreachable)"
                )
            await asyncio.sleep(0.05)
        frame = self._latest
        assert frame is not None
        return frame

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with contextlib.suppress(Exception):
            self._cap.release()
        _log.info(
            "rtsp_camera.closed",
            url=self._safe_url(),
            captured=self._captured,
            dropped=self._dropped,
            reconnects=self._reconnect_attempts,
        )


@contextlib.asynccontextmanager
async def open_rtsp_camera(url: str, **kwargs: int) -> AsyncIterator[RtspCamera]:
    """Async context manager that opens an ``RtspCamera`` and closes it on exit."""
    cam = RtspCamera(url=url, **kwargs)
    try:
        yield cam
    finally:
        await cam.close()


__all__ = ["RtspCamera", "open_rtsp_camera"]
