"""Tests for the RTSP camera driver."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from robot.hardware.sensors.rtsp_camera import RtspCamera, open_rtsp_camera


class TestRtspCameraConstruction:
    def test_requires_url(self) -> None:
        """Empty URL should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            RtspCamera(url="")

    def test_implements_camera_protocol(self) -> None:
        """RtspCamera should satisfy the Camera protocol structurally."""
        # We can't easily construct one without mocking cv2, so check
        # that the class has the required attributes/methods.
        assert hasattr(RtspCamera, "width")
        assert hasattr(RtspCamera, "height")
        assert hasattr(RtspCamera, "capture")
        assert hasattr(RtspCamera, "close")
        # The Camera protocol is runtime_checkable.

    def test_safe_url_masks_password(self) -> None:
        """_safe_url should mask embedded passwords for logging."""
        cam = object.__new__(RtspCamera)
        cam.url = "rtsp://admin:secret@192.168.1.50:554/stream"
        assert cam._safe_url() == "rtsp://admin:***@192.168.1.50:554/stream"

    def test_safe_url_no_credentials(self) -> None:
        """_safe_url should return the URL as-is when no credentials."""
        cam = object.__new__(RtspCamera)
        cam.url = "rtsp://192.168.1.50:554/stream"
        assert cam._safe_url() == "rtsp://192.168.1.50:554/stream"

    def test_safe_url_user_only(self) -> None:
        """_safe_url should mask user-only credentials."""
        cam = object.__new__(RtspCamera)
        cam.url = "rtsp://admin@192.168.1.50:554/stream"
        assert cam._safe_url() == "rtsp://***@192.168.1.50:554/stream"


class TestRtspCameraWithMockedCv2:
    """Tests that mock cv2.VideoCapture to verify the capture loop logic."""

    def _make_mock_cap(self, frames: list[object] | None = None) -> MagicMock:
        """Create a mock cv2.VideoCapture that returns the given frames."""
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: (
            640 if prop == 3 else 480
        )  # CAP_PROP_FRAME_WIDTH=3, HEIGHT=4

        frames = frames or []
        call_count = [0]

        def read():
            if call_count[0] < len(frames):
                f = frames[call_count[0]]
                call_count[0] += 1
                return True, f
            return False, None

        cap.read = read
        return cap

    @pytest.mark.asyncio
    async def test_capture_returns_frame(self) -> None:
        """After the capture thread produces a frame, capture() should return it."""
        import numpy as np

        # Create a simple BGR frame (OpenCV default).
        frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_bgr[:, :] = (0, 255, 0)  # green in BGR

        mock_cap = self._make_mock_cap([frame_bgr])

        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam = RtspCamera(
                url="rtsp://test/stream",
                width=640,
                height=480,
            )

        # Give the capture thread a moment to produce a frame.
        import asyncio

        await asyncio.sleep(0.1)

        frame = await cam.capture()
        assert frame.width == 640
        assert frame.height == 480
        assert len(frame.pixels) == 640 * 480 * 3

        await cam.close()

    @pytest.mark.asyncio
    async def test_close_stops_thread(self) -> None:
        import numpy as np

        frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap = self._make_mock_cap([frame_bgr])

        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam = RtspCamera(url="rtsp://test/stream")

        await cam.close()
        assert cam._closed is True
        # Thread should have stopped.
        assert cam._thread is not None
        assert not cam._thread.is_alive()

    @pytest.mark.asyncio
    async def test_double_close_is_safe(self) -> None:
        import numpy as np

        frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap = self._make_mock_cap([frame_bgr])

        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam = RtspCamera(url="rtsp://test/stream")

        await cam.close()
        # Second close should not raise.
        await cam.close()

    @pytest.mark.asyncio
    async def test_capture_after_close_raises(self) -> None:
        import numpy as np

        frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap = self._make_mock_cap([frame_bgr])

        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam = RtspCamera(url="rtsp://test/stream")

        await cam.close()
        with pytest.raises(RuntimeError, match="closed"):
            await cam.capture()


class TestOpenRtspCameraContextManager:
    @pytest.mark.asyncio
    async def test_context_manager_opens_and_closes(self) -> None:
        import numpy as np

        frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 640 if prop == 3 else 480

        call_count = [0]

        def read():
            if call_count[0] < 1:
                call_count[0] += 1
                return True, frame_bgr
            return False, None

        mock_cap.read = read

        with patch("cv2.VideoCapture", return_value=mock_cap):
            async with open_rtsp_camera("rtsp://test/stream") as cam:
                assert isinstance(cam, RtspCamera)
                assert not cam._closed
            assert cam._closed
