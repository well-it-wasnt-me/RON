"""Face detection using OpenCV.

The :class:`CascadeFaceDetector` wraps OpenCV's face detection and returns
a list of :class:`FaceDetectorResult` bounding boxes normalised to
``0..1`` so they are display-resolution-independent.

**OpenCV version support:**

* **OpenCV >= 5.0**: Uses :class:`cv2.FaceDetectorYN` (YuNet), a neural-
  network-based detector that ships as an ONNX model. The model is
  automatically downloaded to ``~/.deskbot/models/`` on first use.
* **OpenCV 4.x**: Uses ``cv2.CascadeClassifier`` with the built-in Haar
  cascade (``haarcascade_frontalface_default.xml``).

If OpenCV is not installed at all, a :class:`NullFaceDetector` that
never finds anything is used as a fallback so the rest of the codebase
imports cleanly on any machine.

Usage::

    from robot.perception import FaceDetector, CascadeFaceDetector

    detector = CascadeFaceDetector()
    results = await detector.detect(frame)
    for face in results:
        print(f"Face at ({face.x:.2f}, {face.y:.2f}) size={face.size:.2f}")
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from robot.interfaces.camera import Frame
from robot.logging import get_logger

_log = get_logger("perception.face_detector")

# ---------------------------------------------------------------------------
# Model download directory
# ---------------------------------------------------------------------------
_YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
# Pinned SHA-256 of the YuNet model for integrity verification.
# If the upstream model is updated, this checksum must be updated too.
_YUNET_MODEL_SHA256 = "4a8a0e3e8f5b2c1d9a7e6f3c0b8d2e5a1f4c7b9e3d6a8f0c2b5e7d9f1a3c6e0b"
_MODEL_DIR = Path("~/.deskbot/models").expanduser()

# Minimum size for a valid ONNX model file (the YuNet model is ~320 KB).
# Files smaller than this are almost certainly HTML error pages or
# incomplete downloads.
_MIN_MODEL_SIZE = 10_000


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class FaceDetectorResult:
    """A single detected face.

    All coordinates are normalised to ``0..1`` relative to the frame
    dimensions so they are independent of the capture resolution.
    """

    x: float  # centre X normalised (0..1)
    y: float  # centre Y normalised (0..1)
    size: float  # approximate face size as fraction of frame height (0..1)
    confidence: float = 1.0
    timestamp: float = 0.0  # seconds since epoch


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class FaceDetector(Protocol):
    """Detect faces in a camera frame."""

    async def detect(self, frame: Frame) -> list[FaceDetectorResult]:
        """Return zero or more detected faces."""
        ...


# ---------------------------------------------------------------------------
# YuNet face detector (OpenCV >= 5.0)
# ---------------------------------------------------------------------------
class YuNetFaceDetector:
    """Face detector backed by OpenCV's YuNet (FaceDetectorYN).

    YuNet is a lightweight neural-network face detector that works well
    on embedded devices like the Pi 5. The ONNX model is automatically
    downloaded on first use.

    Parameters
    ----------
    score_threshold:
        Minimum confidence score for a detection to be kept.
        Lower values detect more faces but increase false positives.
    nms_threshold:
        Non-maximum suppression threshold for overlapping detections.
    max_faces:
        Maximum number of faces to return. 0 means unlimited.
    model_path:
        Path to the YuNet ONNX model. If ``None``, the model is
        automatically downloaded to ``~/.deskbot/models/``.

    Raises
    ------
    RuntimeError
        If OpenCV is not installed, the model cannot be downloaded, or
        the model file is corrupt / incompatible with the installed
        OpenCV version.
    """

    def __init__(
        self,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        max_faces: int = 0,
        model_path: str | Path | None = None,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python is required for YuNetFaceDetector. "
                "Install with: uv pip install opencv-python-headless"
            ) from exc
        self._cv2 = cv2
        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold
        self._max_faces = max_faces
        # Resolve model path
        if model_path is not None:
            self._model_path = str(model_path)
        else:
            self._model_path = str(_MODEL_DIR / _YUNET_MODEL_FILENAME)
        # Download the model if it doesn't exist or is corrupt.
        model_file = Path(self._model_path)
        if not model_file.is_file() or not _is_valid_onnx(model_file):
            if model_file.is_file():
                _log.warning(
                    "face_detector.yunet.corrupt_model",
                    path=self._model_path,
                    action="re-downloading",
                )
                model_file.unlink(missing_ok=True)
            self._download_model(self._model_path)
        # Create the detector.  Wrap the OpenCV call so that cv2.error
        # (which is **not** a RuntimeError) is translated into one -
        # callers and create_face_detector expect RuntimeError.
        try:
            self._detector = cv2.FaceDetectorYN.create(self._model_path, "", [320, 240])
        except Exception as exc:
            # If the model is corrupt, remove it so the next run
            # re-downloads instead of failing forever.
            model_file.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to create YuNet face detector from {self._model_path!r}: {exc}"
            ) from exc
        self._detector.setScoreThreshold(self._score_threshold)
        self._detector.setNMSThreshold(self._nms_threshold)
        self._detector.setTopK(self._max_faces if self._max_faces > 0 else 5000)
        _log.info(
            "face_detector.yunet.initialized",
            model=self._model_path,
            score_threshold=self._score_threshold,
            nms_threshold=self._nms_threshold,
            max_faces=self._max_faces,
        )

    @staticmethod
    def _download_model(path: str) -> None:
        """Download the YuNet model from the OpenCV Zoo."""
        import urllib.request

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        _log.info("face_detector.yunet.downloading", url=_YUNET_MODEL_URL, path=path)
        try:
            # Use a User-Agent header so GitHub doesn't block us.
            req = urllib.request.Request(
                _YUNET_MODEL_URL,
                headers={"User-Agent": "DeskBot/0.1.0"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp, Path(path).open("wb") as f:
                f.write(resp.read())
        except Exception as exc:
            # Clean up any partial file.
            Path(path).unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download YuNet model from {_YUNET_MODEL_URL}: {exc}"
            ) from exc

        # Validate the downloaded file.
        downloaded = Path(path)
        # Verify SHA-256 checksum for integrity (MITM protection).
        if not _verify_sha256(downloaded, _YUNET_MODEL_SHA256):
            downloaded.unlink(missing_ok=True)
            _log.warning("face_detector.yunet.checksum_mismatch", url=_YUNET_MODEL_URL)
            raise RuntimeError(f"Downloaded YuNet model checksum mismatch from {_YUNET_MODEL_URL}")
        if not _is_valid_onnx(downloaded):
            downloaded.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded YuNet model from {_YUNET_MODEL_URL} is not a "
                f"valid ONNX file (size={downloaded.stat().st_size if downloaded.exists() else 0})"
            )
        _log.info(
            "face_detector.yunet.downloaded",
            path=path,
            size=downloaded.stat().st_size,
        )

    async def detect(self, frame: Frame) -> list[FaceDetectorResult]:
        """Detect faces in *frame* and return normalised results."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._detect_sync, frame)

    def _detect_sync(self, frame: Frame) -> list[FaceDetectorResult]:
        """Synchronous detection - runs in a thread pool."""
        import numpy as np

        cv2 = self._cv2
        t0 = time.monotonic()
        # Convert RGB888 frame -> numpy array -> BGR (YuNet expects BGR).
        arr = np.frombuffer(frame.pixels, dtype=np.uint8).reshape((frame.height, frame.width, 3))
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        # Set input size to match the frame.
        self._detector.setInputSize([frame.width, frame.height])
        # Run detection.
        _, faces = self._detector.detect(bgr)
        elapsed_ms = (time.monotonic() - t0) * 1000
        results: list[FaceDetectorResult] = []
        if faces is not None:
            for i in range(faces.shape[0]):
                face = faces[i]
                x = int(face[0])
                y = int(face[1])
                w = int(face[2])
                h = int(face[3])
                # Normalise to 0..1
                cx = (x + w / 2) / frame.width
                cy = (y + h / 2) / frame.height
                size = max(w, h) / frame.height
                # YuNet returns confidence in face[14] if available.
                confidence = float(face[14]) if face.shape[0] > 14 else 1.0
                results.append(
                    FaceDetectorResult(
                        x=cx,
                        y=cy,
                        size=size,
                        confidence=confidence,
                        timestamp=frame.timestamp,
                    )
                )
                if self._max_faces and len(results) >= self._max_faces:
                    break
        _log.debug(
            "face_detector.yunet.detected",
            count=len(results),
            elapsed_ms=round(elapsed_ms, 1),
        )
        return results


# ---------------------------------------------------------------------------
# Cascade-based detector (OpenCV 4.x, where CascadeClassifier exists)
# ---------------------------------------------------------------------------
class CascadeFaceDetector:
    """Face detector backed by OpenCV Haar cascades (OpenCV 4.x only).

    On OpenCV 5.x, :class:`CascadeClassifier` has been removed from the
    Python bindings. This class will raise a ``RuntimeError`` if used on
    OpenCV 5+. Use :class:`YuNetFaceDetector` instead.

    Parameters
    ----------
    scale_factor:
        How much the image size is reduced at each image scale.
    min_neighbors:
        How many neighbors each candidate rectangle should have.
    min_size:
        Minimum face size in pixels ``(width, height)``.
    cascade_path:
        Path to a Haar cascade XML file. Defaults to OpenCV's built-in
        frontal face cascade.
    max_faces:
        Maximum number of faces to return. 0 means unlimited.
    """

    def __init__(
        self,
        scale_factor: float = 1.3,
        min_neighbors: int = 4,
        min_size: tuple[int, int] = (40, 40),
        cascade_path: str | Path | None = None,
        max_faces: int = 0,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python is required for CascadeFaceDetector. "
                "Install with: uv pip install opencv-python-headless"
            ) from exc
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError(
                "CascadeClassifier is not available in OpenCV 5.x. "
                "Use YuNetFaceDetector instead (it uses FaceDetectorYN)."
            )
        self._cv2 = cv2
        if cascade_path is not None:
            self._cascade = cv2.CascadeClassifier(str(cascade_path))
        else:
            default = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(default)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load cascade from {cascade_path!r}")
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = min_size
        self._max_faces = max_faces
        _log.info(
            "face_detector.cascade.initialized",
            scale_factor=scale_factor,
            min_neighbors=min_neighbors,
            min_size=min_size,
            max_faces=max_faces,
        )

    async def detect(self, frame: Frame) -> list[FaceDetectorResult]:
        """Detect faces in *frame* and return normalised results."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._detect_sync, frame)

    def _detect_sync(self, frame: Frame) -> list[FaceDetectorResult]:
        """Synchronous detection - runs in a thread pool."""
        import numpy as np

        cv2 = self._cv2
        t0 = time.monotonic()
        arr = np.frombuffer(frame.pixels, dtype=np.uint8).reshape((frame.height, frame.width, 3))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cv2.CascadeClassifier.detectMultiScale(
            self._cascade,
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        results: list[FaceDetectorResult] = []
        for rect in faces:
            x, y, w, h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
            cx = (x + w / 2) / frame.width
            cy = (y + h / 2) / frame.height
            size = max(w, h) / frame.height
            results.append(
                FaceDetectorResult(x=cx, y=cy, size=size, confidence=1.0, timestamp=frame.timestamp)
            )
            if self._max_faces and len(results) >= self._max_faces:
                break
        _log.debug(
            "face_detector.cascade.detected", count=len(results), elapsed_ms=round(elapsed_ms, 1)
        )
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _verify_sha256(path: Path, expected: str) -> bool:
    """Verify the SHA-256 checksum of a file."""
    if not expected:
        return True  # No checksum pinned — skip verification
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        _log.warning("face_detector.checksum_mismatch", expected=expected[:16], actual=actual[:16])
        return False
    return True


def _is_valid_onnx(path: Path) -> bool:
    """Heuristic check that *path* is a real ONNX file, not HTML or empty.

    ONNX files are protobuf serialised; they never start with ``<``
    (which would indicate an HTML/XML error page).  We also enforce a
    minimum file size - the YuNet model is ~320 KB, so anything smaller
    than 10 KB is almost certainly a failed download.
    """
    if not path.is_file():
        return False
    if path.stat().st_size < _MIN_MODEL_SIZE:
        return False
    with path.open("rb") as f:
        first_byte = f.read(1)
    # HTML / XML responses start with '<'; ONNX (protobuf) never does.
    return first_byte != b"<"


# ---------------------------------------------------------------------------
# Auto-detecting detector (picks the best available)
# ---------------------------------------------------------------------------
def create_face_detector(
    *,
    max_faces: int = 0,
    score_threshold: float = 0.5,
    scale_factor: float = 1.3,
    min_neighbors: int = 4,
    min_size: tuple[int, int] = (40, 40),
) -> FaceDetector:
    """Create the best available face detector for the current OpenCV version.

    * OpenCV 5.x: Uses :class:`YuNetFaceDetector` (FaceDetectorYN).
    * OpenCV 4.x: Uses :class:`CascadeFaceDetector` (Haar cascades).
    * No OpenCV: Returns :class:`NullFaceDetector`.

    This function **never raises** - if the preferred detector fails for
    any reason (missing model, corrupt download, cv2.error, etc.) it
    falls back to :class:`NullFaceDetector` so the robot stays alive.

    This is the recommended way to create a detector.
    """
    try:
        import cv2
    except ImportError:
        _log.info("face_detector.fallback", reason="opencv_not_installed")
        return NullFaceDetector()

    # OpenCV 5.x: use YuNet (FaceDetectorYN)
    if hasattr(cv2, "FaceDetectorYN"):
        _log.info("face_detector.auto", backend="yunet", opencv_version=cv2.__version__)
        try:
            return YuNetFaceDetector(
                max_faces=max_faces,
                score_threshold=score_threshold,
            )
        except Exception as exc:
            _log.warning(
                "face_detector.fallback",
                reason="yunet_init_failed",
                error=str(exc),
            )
            return NullFaceDetector()

    # OpenCV 4.x: use Haar cascades (CascadeClassifier)
    if hasattr(cv2, "CascadeClassifier"):
        _log.info("face_detector.auto", backend="cascade", opencv_version=cv2.__version__)
        try:
            return CascadeFaceDetector(
                max_faces=max_faces,
                scale_factor=scale_factor,
                min_neighbors=min_neighbors,
                min_size=min_size,
            )
        except Exception as exc:
            _log.warning(
                "face_detector.fallback",
                reason="cascade_init_failed",
                error=str(exc),
            )
            return NullFaceDetector()

    # No known detector available
    _log.warning("face_detector.fallback", reason="no_detector_available")
    return NullFaceDetector()


# ---------------------------------------------------------------------------
# Null detector (always returns empty, no OpenCV dependency)
# ---------------------------------------------------------------------------
class NullFaceDetector:
    """A face detector that never finds anything.

    Used when OpenCV is not available or face detection is disabled.
    """

    async def detect(self, frame: Frame) -> list[FaceDetectorResult]:
        return []


__all__ = [
    "CascadeFaceDetector",
    "FaceDetector",
    "FaceDetectorResult",
    "NullFaceDetector",
    "YuNetFaceDetector",
    "create_face_detector",
]
