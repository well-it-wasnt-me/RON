"""Settings & hardware test API routes.

Provides endpoints to inspect and test every hardware/perception
subsystem in real time:

* **Camera** - single-frame capture and live MJPEG preview stream.
* **Microphone** - real-time input-level meter and record-and-playback test.
* **Audio output** - test-tone playback, device listing, and runtime
  output-device switching.
* **TTS** - speak a test phrase through the configured engine.
* **LLM** - send a test prompt and inspect the response.
* **Sound effects** - trigger individual sound effects.

All endpoints degrade gracefully when the robot is running with mock
hardware or when optional dependencies (``cv2``, ``sounddevice``,
``numpy``, ``PIL``) are not installed.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import struct
import wave
from collections.abc import AsyncIterator
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse

from robot.api.schemas import (
    AudioDevicesResponse,
    AudioDeviceTestRequest,
    AudioInfoResponse,
    AudioSwitchRequest,
    AudioSwitchResponse,
    AudioTestDeviceResponse,
    CameraInfoResponse,
    LLMTestRequest,
    LLMTestResponse,
    MicLevelResponse,
    MicrophoneInfoResponse,
    MicTestRequest,
    SettingsInfoResponse,
    SoundEffectResponse,
    SoundEffectsListResponse,
    ToneRequest,
    ToneResponse,
    TTSTestRequest,
    TTSTestResponse,
)
from robot.api.security import require_api_key
from robot.interfaces.audio import AudioBuffer
from robot.logging import get_logger

_log = get_logger("api.settings")

router = APIRouter(prefix="/settings", tags=["settings"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bridge(request: Request) -> Any:
    """Return the StateBridge from app state."""
    return getattr(request.app.state, "bridge", None)


def _rms_s16le(pcm: bytes) -> float:
    """Root-mean-square energy of an s16le PCM buffer (0..1 normalised)."""
    count = len(pcm) // 2
    if count == 0:
        return 0.0
    total = 0
    for offset in range(0, count, 1024):
        n = min(1024, count - offset)
        samples = struct.unpack(f"<{n}h", pcm[offset * 2 : (offset + n) * 2])
        for s in samples:
            total += s * s
    return math.sqrt(total / count) / 32768.0


def _encode_jpeg(rgb: bytes, width: int, height: int) -> bytes | None:
    """Encode RGB888 pixels to JPEG.

    Tries ``cv2`` first (already required for UsbCamera), then ``PIL``.
    Returns ``None`` when no JPEG encoder is available.
    """
    # Try OpenCV (BGR is cv2's native format).
    try:
        import cv2
        import numpy as np

        arr = np.frombuffer(rgb, dtype=np.uint8).reshape((height, width, 3))
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            return bytes(buf.tobytes())
    except Exception:
        pass

    # Try Pillow.
    try:
        from PIL import Image

        img = Image.frombytes("RGB", (width, height), rgb)
        pil_buf = io.BytesIO()
        img.save(pil_buf, format="JPEG", quality=80)
        return pil_buf.getvalue()
    except Exception:
        pass

    return None


def _encode_bmp(rgb: bytes, width: int, height: int) -> bytes:
    """Encode RGB888 pixels as a BMP file (pure Python, no deps)."""
    # BMP rows are padded to 4-byte boundaries and stored bottom-up.
    row_stride = width * 3
    padding = (4 - row_stride % 4) % 4
    padded_stride = row_stride + padding
    pixel_data_size = padded_stride * height
    file_size = 54 + pixel_data_size

    header = struct.pack(
        "<2sIHHI",
        b"BM",
        file_size,
        0,
        0,
        54,
    )
    info_header = struct.pack(
        "<IIIHHIIIIII",
        40,
        width,
        height,
        1,
        24,
        0,
        pixel_data_size,
        2835,
        2835,
        0,
        0,
    )

    rows: list[bytes] = []
    for y in range(height - 1, -1, -1):
        start = y * row_stride
        rows.append(rgb[start : start + row_stride] + b"\x00" * padding)
    return header + info_header + b"".join(rows)


def _generate_tone_pcm(
    frequency_hz: float,
    duration_s: float,
    sample_rate: int,
    volume: float,
) -> bytes:
    """Generate a sine-wave test tone as s16le mono PCM."""
    n_samples = int(sample_rate * duration_s)
    try:
        import numpy as np

        t = np.arange(n_samples, dtype=np.float64) / sample_rate
        wave_data = (volume * 32767 * np.sin(2 * math.pi * frequency_hz * t)).astype(np.int16)
        return wave_data.tobytes()
    except ImportError:
        # Pure-Python fallback.
        samples: list[int] = []
        for i in range(n_samples):
            t_i = i / sample_rate
            val = int(volume * 32767 * math.sin(2 * math.pi * frequency_hz * t_i))
            samples.append(max(-32768, min(32767, val)))
        return struct.pack(f"<{n_samples}h", *samples)


def _pcm_to_wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw s16le PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
@router.get("/info", summary="Hardware & subsystem overview", response_model=SettingsInfoResponse)
async def settings_info(request: Request) -> SettingsInfoResponse:
    """Return a comprehensive overview of all testable subsystems."""
    bridge = _bridge(request)
    settings = getattr(request.app.state, "settings", None)
    info: dict[str, Any] = {"ready": bridge is not None and bridge.is_ready}

    # Camera
    cam = getattr(bridge, "camera", None) if bridge else None
    if cam is not None:
        info["camera"] = {
            "type": type(cam).__name__,
            "width": getattr(cam, "width", None),
            "height": getattr(cam, "height", None),
            "is_mock": type(cam).__name__ == "MockCamera",
            "captured": getattr(cam, "captured", None),
        }
    else:
        info["camera"] = None

    # Microphone
    mic = getattr(bridge, "microphone", None) if bridge else None
    if mic is not None:
        info["microphone"] = {
            "type": type(mic).__name__,
            "sample_rate": getattr(mic, "sample_rate", None),
            "is_mock": type(mic).__name__ == "MockMicrophone",
        }
    else:
        info["microphone"] = None

    # Audio output
    audio = getattr(bridge, "audio", None) if bridge else None
    if audio is not None:
        info["audio"] = {
            "type": type(audio).__name__,
            "sample_rate": getattr(audio, "sample_rate", None),
            "channels": getattr(audio, "channels", None),
            "is_mock": type(audio).__name__ == "MockAudioOutput",
            "output_device": getattr(audio, "output_device", None),
        }
    else:
        info["audio"] = None

    # TTS / STT / LLM
    conv = getattr(bridge, "conversation", None) if bridge else None
    if conv is not None:
        info["tts"] = type(conv.tts).__name__ if conv.tts else None
        info["stt"] = type(conv.stt).__name__ if conv.stt else None
        info["llm"] = type(conv.llm).__name__ if conv.llm else None
    else:
        info["tts"] = None
        info["stt"] = None
        info["llm"] = None

    # Sound effects
    sfx = getattr(bridge, "sound_effects", None) if bridge else None
    if sfx is not None:
        info["sound_effects"] = {
            "enabled": sfx.enabled,
            "available": sorted(sfx._index.keys()) if hasattr(sfx, "_index") else [],
        }
    else:
        info["sound_effects"] = None

    # Servos (via calibration state if wired)
    info["servos"] = _servo_info(bridge)

    # Display
    info["display"] = _display_info(settings)

    # Degradation
    deg = getattr(bridge, "degradation", None) if bridge else None
    if deg is not None:
        info["degradation"] = deg.summary()
    else:
        info["degradation"] = None

    return SettingsInfoResponse.model_validate(info)


def _servo_info(bridge: Any) -> dict[str, Any] | None:
    """Return servo info from the calibration module state if available."""
    try:
        from robot.api.calibration import _state as cal_state

        if cal_state.servo_controller is not None:
            servos = cal_state.servo_controller.all()
            return {
                "count": len(servos),
                "servos": [{"name": s.name, "angle": s.angle} for s in servos],
            }
    except Exception:
        pass
    return None


def _display_info(settings: Any) -> dict[str, Any] | None:
    """Return display config from settings."""
    if settings is None:
        return None
    d = settings.displays
    return {
        "backend": d.backend,
        "width": d.width,
        "height": d.height,
        "rotation": d.rotation,
        "fps": d.fps,
    }


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
@router.get("/camera/info", summary="Camera info", response_model=CameraInfoResponse)
async def camera_info(request: Request) -> CameraInfoResponse:
    """Return camera type and resolution."""
    bridge = _bridge(request)
    cam = getattr(bridge, "camera", None) if bridge else None
    if cam is None:
        raise HTTPException(status_code=503, detail="No camera available")
    return CameraInfoResponse(
        type=type(cam).__name__,
        width=getattr(cam, "width", None),
        height=getattr(cam, "height", None),
        is_mock=type(cam).__name__ == "MockCamera",
        captured=getattr(cam, "captured", None),
    )


@router.get("/camera/frame", summary="Capture a single frame")
async def camera_frame(request: Request) -> Response:
    """Capture a single frame and return it as a JPEG (or BMP fallback)."""
    bridge = _bridge(request)
    cam = getattr(bridge, "camera", None) if bridge else None
    if cam is None:
        raise HTTPException(status_code=503, detail="No camera available")
    try:
        frame = await cam.capture()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Capture failed: {exc}") from exc

    jpeg = _encode_jpeg(frame.pixels, frame.width, frame.height)
    if jpeg is not None:
        return Response(content=jpeg, media_type="image/jpeg")
    # Fallback: BMP (no external deps required).
    bmp = _encode_bmp(frame.pixels, frame.width, frame.height)
    return Response(content=bmp, media_type="image/bmp")


@router.get("/camera/stream", summary="Live MJPEG camera preview")
async def camera_stream(request: Request) -> StreamingResponse:
    """Stream a live MJPEG feed from the camera.

    The browser can display this directly in an ``<img>`` tag::

        <img src="/api/v1/settings/camera/stream" />
    """
    bridge = _bridge(request)
    cam = getattr(bridge, "camera", None) if bridge else None
    if cam is None:
        raise HTTPException(status_code=503, detail="No camera available")

    boundary = "deskbotframe"

    async def generate() -> AsyncIterator[bytes]:
        import asyncio

        while True:
            if await request.is_disconnected():
                break
            try:
                frame = await cam.capture()
            except Exception:
                await asyncio.sleep(0.1)
                continue
            jpeg = _encode_jpeg(frame.pixels, frame.width, frame.height)
            if jpeg is None:
                # No JPEG encoder - can't do MJPEG.
                await asyncio.sleep(0.5)
                continue
            yield (
                b"--" + boundary.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
            )
            await asyncio.sleep(1.0 / 15)  # ~15 FPS

    return StreamingResponse(
        generate(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
    )


# ---------------------------------------------------------------------------
# Microphone
# ---------------------------------------------------------------------------
@router.get("/mic/info", summary="Microphone info", response_model=MicrophoneInfoResponse)
async def mic_info(request: Request) -> MicrophoneInfoResponse:
    """Return microphone type and sample rate."""
    bridge = _bridge(request)
    mic = getattr(bridge, "microphone", None) if bridge else None
    if mic is None:
        raise HTTPException(status_code=503, detail="No microphone available")
    return MicrophoneInfoResponse(
        type=type(mic).__name__,
        sample_rate=getattr(mic, "sample_rate", None),
        is_mock=type(mic).__name__ == "MockMicrophone",
    )


@router.get("/mic/level", summary="Current microphone input level", response_model=MicLevelResponse)
async def mic_level(request: Request) -> MicLevelResponse:
    """Return the current microphone RMS level (0..1).

    For :class:`UsbMicrophone` this reads the continuously-updated
    ``_last_rms_value`` from the background capture thread (zero
    conflict with the conversation audio loop). For other microphones
    a best-effort value is returned.
    """
    bridge = _bridge(request)
    mic = getattr(bridge, "microphone", None) if bridge else None
    if mic is None:
        raise HTTPException(status_code=503, detail="No microphone available")

    # UsbMicrophone tracks RMS on the capture thread.
    rms_val = getattr(mic, "_last_rms_value", None)
    if rms_val is not None:
        return MicLevelResponse.model_validate(
            {"level": round(rms_val, 5), "source": "capture_thread"}
        )

    # Fallback: try to read a single chunk without disrupting the stream.
    # This is a best-effort peek for mock / custom microphones.
    return MicLevelResponse.model_validate({"level": 0.0, "source": "unavailable"})


@router.post("/mic/test", summary="Record and play back a mic test")
async def mic_test(
    request: Request, body: MicTestRequest, _: None = Depends(require_api_key)
) -> Response:
    """Record *duration* seconds from a temporary microphone and return WAV.

    A **temporary** microphone instance is created with the same settings
    as the active one so the conversation audio loop is not disturbed.
    The recorded audio is returned as a WAV file for the browser to play.
    """
    bridge = _bridge(request)
    mic = getattr(bridge, "microphone", None) if bridge else None
    if mic is None:
        raise HTTPException(status_code=503, detail="No microphone available")

    settings = getattr(request.app.state, "settings", None)
    sample_rate = getattr(mic, "sample_rate", 16000)
    channels = getattr(mic, "channels", 1) if settings is None else settings.microphone.channels
    frame_ms = getattr(mic, "frame_ms", 30) if settings is not None else 30

    # Build a temporary microphone with the same settings.
    temp_mic = _create_temp_mic(mic, settings, sample_rate, channels, frame_ms)
    if temp_mic is None:
        raise HTTPException(status_code=503, detail="Could not create test microphone")

    try:
        pcm = bytearray()
        target_samples = int(sample_rate * body.duration_s)
        collected = 0

        async for chunk in temp_mic.stream():
            pcm.extend(chunk.pcm)
            collected += len(chunk.pcm) // 2
            if collected >= target_samples:
                break
        await temp_mic.close()
    except Exception as exc:
        with contextlib.suppress(Exception):
            await temp_mic.close()
        raise HTTPException(status_code=500, detail=f"Recording failed: {exc}") from exc

    wav_bytes = _pcm_to_wav(bytes(pcm), sample_rate, channels=channels)

    # Optionally play back through the audio output.
    if body.play_back:
        audio = getattr(bridge, "audio", None)
        if audio is not None:
            try:
                await audio.play(
                    AudioBuffer(
                        pcm=bytes(pcm[: target_samples * 2 * channels]),
                        sample_rate=sample_rate,
                        channels=channels,
                    )
                )
            except Exception as exc:
                _log.warning("settings.mic_test.playback_failed", error=str(exc))

    return Response(content=wav_bytes, media_type="audio/wav")


def _create_temp_mic(
    original: Any,
    settings: Any,
    sample_rate: int,
    channels: int,
    frame_ms: int,
) -> Any:
    """Create a temporary microphone mirroring the active one's config."""
    mic_type = type(original).__name__

    if mic_type == "MockMicrophone":
        from robot.hardware.sensors.mock_microphone import MockMicrophone

        return MockMicrophone(
            sample_rate=sample_rate,
            channels=channels,
            frame_ms=frame_ms,
        )

    if mic_type == "UsbMicrophone":
        try:
            from robot.hardware.sensors.usb_microphone import UsbMicrophone

            input_device = getattr(original, "input_device", "default")
            return UsbMicrophone(
                input_device=input_device,
                _sample_rate_field=sample_rate,
                channels=channels,
                frame_ms=frame_ms,
            )
        except Exception as exc:
            _log.warning("settings.temp_mic_failed", error=str(exc))
            # Fall back to mock so the test still returns (silence).
            from robot.hardware.sensors.mock_microphone import MockMicrophone

            return MockMicrophone(
                sample_rate=sample_rate,
                channels=channels,
                frame_ms=frame_ms,
            )

    # Unknown type - try mock.
    try:
        from robot.hardware.sensors.mock_microphone import MockMicrophone

        return MockMicrophone(
            sample_rate=sample_rate,
            channels=channels,
            frame_ms=frame_ms,
        )
    except Exception:
        return None


@router.websocket("/mic/stream")
async def mic_stream_ws(websocket: WebSocket) -> None:
    """Stream live microphone audio to the browser as WebSocket binary frames.

    This backs the "hear the world as RON" Live View panel. A **temporary**
    microphone mirroring the active one's settings is created per connection
    (just like :func:`mic_test`) so the conversation audio loop keeps its own
    stream undisturbed.

    Protocol:

    * The first frame is a **text** message carrying the audio format::

        {"sample_rate": 16000, "channels": 1, "frame_ms": 30,
         "type": "UsbMicrophone", "is_mock": false}

      or ``{"error": "no_microphone"}`` if no mic is available (the socket
      then closes).
    * Every subsequent frame is **binary**: raw signed-16-bit little-endian
      PCM bytes (``chunk.pcm``), ``frame_ms`` long. The client plays them
      with the Web Audio API.
    * Send the text message ``"stop"`` to end the stream; closing the
      socket also tears it down. The temp microphone is always closed.
    """
    import asyncio

    await websocket.accept()
    app_state = websocket.app.state
    bridge = getattr(app_state, "bridge", None)
    mic = getattr(bridge, "microphone", None) if bridge else None
    if mic is None:
        await websocket.send_text(json.dumps({"error": "no_microphone"}))
        with contextlib.suppress(Exception):
            await websocket.close()
        return

    settings = getattr(app_state, "settings", None)
    sample_rate = getattr(mic, "sample_rate", 16000)
    channels = getattr(mic, "channels", 1) if settings is None else settings.microphone.channels
    frame_ms = getattr(mic, "frame_ms", 30) if settings is not None else 30
    temp_mic = _create_temp_mic(mic, settings, sample_rate, channels, frame_ms)
    if temp_mic is None:
        await websocket.send_text(json.dumps({"error": "no_microphone"}))
        with contextlib.suppress(Exception):
            await websocket.close()
        return

    await websocket.send_text(
        json.dumps(
            {
                "sample_rate": sample_rate,
                "channels": channels,
                "frame_ms": frame_ms,
                "type": type(mic).__name__,
                "is_mock": type(mic).__name__ == "MockMicrophone",
            }
        )
    )

    async def pump() -> None:
        async for chunk in temp_mic.stream():
            await websocket.send_bytes(chunk.pcm)

    async def reader() -> None:
        # Detect client disconnect / an explicit "stop" message so we tear
        # the temp microphone down promptly rather than waiting for a send
        # to fail.
        while True:
            data = await websocket.receive_text()
            if data.strip() == "stop":
                break

    pump_task = asyncio.create_task(pump())
    reader_task = asyncio.create_task(reader())
    try:
        await asyncio.wait(
            cast("set[asyncio.Future[Any]]", {pump_task, reader_task}),
            return_when=asyncio.FIRST_COMPLETED,
        )
    except WebSocketDisconnect:
        pass
    finally:
        for task in (pump_task, reader_task):
            task.cancel()
            # CancelledError is a BaseException (not Exception) on 3.8+, so
            # list it explicitly; retrieve each task's result/exception so
            # asyncio doesn't log "task exception was never retrieved".
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        with contextlib.suppress(Exception):
            await temp_mic.close()
        with contextlib.suppress(Exception):
            await websocket.close()


# ---------------------------------------------------------------------------
# Audio output
# ---------------------------------------------------------------------------
@router.get("/audio/info", summary="Audio output info", response_model=AudioInfoResponse)
async def audio_info(request: Request) -> AudioInfoResponse:
    """Return audio output type, sample rate, and channels."""
    bridge = _bridge(request)
    audio = getattr(bridge, "audio", None) if bridge else None
    if audio is None:
        raise HTTPException(status_code=503, detail="No audio output available")
    return AudioInfoResponse(
        type=type(audio).__name__,
        sample_rate=getattr(audio, "sample_rate", None),
        channels=getattr(audio, "channels", None),
        is_mock=type(audio).__name__ == "MockAudioOutput",
        output_device=getattr(audio, "output_device", None),
    )


@router.get(
    "/audio/devices",
    summary="List available audio output devices",
    response_model=AudioDevicesResponse,
)
async def audio_devices() -> AudioDevicesResponse:
    """List available audio output devices via ``sounddevice``.

    Returns an empty list when ``sounddevice`` is not installed or when
    no audio subsystem is available.  The PortAudio query runs in a
    thread with a 3-second timeout so it can never block the API.
    """
    return await _list_sd_devices("output")


@router.get(
    "/audio/input-devices",
    summary="List available audio input devices",
    response_model=AudioDevicesResponse,
)
async def audio_input_devices() -> AudioDevicesResponse:
    """List available audio input devices via ``sounddevice``."""
    return await _list_sd_devices("input")


async def _list_sd_devices(kind: str) -> AudioDevicesResponse:
    """List sounddevice devices in a daemon thread with a timeout.

    Uses ``sd.query_devices()`` and ``sd.default.device`` to determine
    the default.  Avoids ``sd.query_devices(kind=...)`` which can hang
    on headless systems.  Runs in an explicit daemon thread with a
    3-second timeout so a blocked PortAudio call can never hang the
    event loop or prevent process exit.
    """
    import asyncio
    import threading

    try:
        import sounddevice as sd
    except ImportError:
        return AudioDevicesResponse.model_validate({"devices": [], "available": False})

    channel_key = f"max_{kind}_channels"
    default_idx_key = 1 if kind == "output" else 0
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[AudioDevicesResponse] = loop.create_future()

    def _query() -> None:
        try:
            devices: list[dict[str, Any]] = []
            for i, dev in enumerate(sd.query_devices()):
                if dev[channel_key] > 0:
                    devices.append(
                        {
                            "index": i,
                            "name": dev["name"],
                            "channels": dev[channel_key],
                            "default_sample_rate": dev["default_samplerate"],
                        }
                    )
            default_index: int | None = None
            try:
                idx = sd.default.device[default_idx_key]
                default_index = idx if idx is not None and idx >= 0 else None
            except Exception:
                pass
            result = {"devices": devices, "default_index": default_index, "available": True}
            loop.call_soon_threadsafe(fut.set_result, AudioDevicesResponse.model_validate(result))
        except Exception as exc:
            with contextlib.suppress(Exception):
                loop.call_soon_threadsafe(fut.set_exception, exc)

    thread = threading.Thread(target=_query, daemon=True, name="sd-query")
    thread.start()

    try:
        return await asyncio.wait_for(fut, timeout=3.0)
    except TimeoutError:
        return AudioDevicesResponse.model_validate(
            {"devices": [], "available": False, "error": "PortAudio query timed out"}
        )
    except Exception as exc:
        return AudioDevicesResponse.model_validate(
            {"devices": [], "available": False, "error": str(exc)}
        )


@router.post("/audio/tone", summary="Play a test tone", response_model=ToneResponse)
async def audio_tone(
    request: Request, body: ToneRequest, _: None = Depends(require_api_key)
) -> ToneResponse:
    """Play a sine-wave test tone through the current audio output."""
    bridge = _bridge(request)
    audio = getattr(bridge, "audio", None) if bridge else None
    if audio is None:
        raise HTTPException(status_code=503, detail="No audio output available")

    sample_rate = getattr(audio, "sample_rate", 48000)
    pcm = _generate_tone_pcm(body.frequency_hz, body.duration_s, sample_rate, body.volume)
    buffer = AudioBuffer(pcm=pcm, sample_rate=sample_rate, channels=getattr(audio, "channels", 1))
    try:
        await audio.play(buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Playback failed: {exc}") from exc
    return ToneResponse(
        status="ok",
        frequency_hz=body.frequency_hz,
        duration_s=body.duration_s,
        sample_rate=sample_rate,
    )


@router.post(
    "/audio/test-device",
    summary="Play a test tone through a specific device",
    response_model=AudioTestDeviceResponse,
)
async def audio_test_device(
    request: Request, body: AudioDeviceTestRequest, _: None = Depends(require_api_key)
) -> AudioTestDeviceResponse:
    """Play a test tone through a specific output device (temporary).

    This creates a temporary :class:`UsbSpeaker` for the given device,
    plays the tone, and closes it. Useful for testing different outputs
    before committing to a config change.
    """
    try:
        from robot.hardware.audio.usb_speaker import UsbSpeaker

        settings = getattr(request.app.state, "settings", None)
        sample_rate = settings.audio.sample_rate if settings else 48000
        channels = settings.audio.channels if settings else 1
        speaker = UsbSpeaker(
            output_device=body.device,
            _sample_rate=sample_rate,
            channels=channels,
        )
    except ImportError:
        raise HTTPException(status_code=503, detail="sounddevice not installed")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not open device: {exc}") from exc

    pcm = _generate_tone_pcm(body.frequency_hz, body.duration_s, sample_rate, body.volume)
    buffer = AudioBuffer(pcm=pcm, sample_rate=sample_rate, channels=channels)
    try:
        await speaker.play(buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Playback failed: {exc}") from exc
    finally:
        with contextlib.suppress(Exception):
            await speaker.close()

    return AudioTestDeviceResponse(
        status="ok",
        device=body.device,
        frequency_hz=body.frequency_hz,
        duration_s=body.duration_s,
    )


@router.post(
    "/audio/switch",
    summary="Switch the active audio output device",
    response_model=AudioSwitchResponse,
)
async def audio_switch(
    request: Request, body: AudioSwitchRequest, _: None = Depends(require_api_key)
) -> AudioSwitchResponse:
    """Switch the active audio output to a different device at runtime.

    Creates a new :class:`UsbSpeaker` for the requested device and swaps
    it into the bridge, TTS engine, and sound-effects player. The old
    output is closed.
    """
    bridge = _bridge(request)
    if bridge is None:
        raise HTTPException(status_code=503, detail="DeskBot app not attached")

    settings = getattr(request.app.state, "settings", None)
    sample_rate = settings.audio.sample_rate if settings else 48000
    channels = settings.audio.channels if settings else 1

    try:
        from robot.hardware.audio.usb_speaker import UsbSpeaker

        new_audio = UsbSpeaker(
            output_device=body.device,
            _sample_rate=sample_rate,
            channels=channels,
        )
    except ImportError:
        raise HTTPException(status_code=503, detail="sounddevice not installed")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not open device: {exc}") from exc

    old_audio = bridge.audio
    bridge.audio = new_audio

    # Swap in TTS if it holds a reference.
    tts = getattr(bridge, "tts", None)
    if tts is not None and hasattr(tts, "_audio"):
        tts._audio = new_audio

    # Swap in sound effects.
    sfx = getattr(bridge, "sound_effects", None)
    if sfx is not None and hasattr(sfx, "audio"):
        sfx.audio = new_audio

    # Close old output.
    if old_audio is not None:
        with contextlib.suppress(Exception):
            await old_audio.close()

    _log.info("settings.audio_switched", device=body.device)
    return AudioSwitchResponse(
        status="ok",
        device=body.device,
        type=type(new_audio).__name__,
    )


@router.post("/audio/stop", summary="Stop audio playback", response_model=ToneResponse)
async def audio_stop(request: Request, _: None = Depends(require_api_key)) -> ToneResponse:
    """Stop whatever is currently playing on the audio output."""
    bridge = _bridge(request)
    audio = getattr(bridge, "audio", None) if bridge else None
    if audio is None:
        raise HTTPException(status_code=503, detail="No audio output available")
    try:
        await audio.stop()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Stop failed: {exc}") from exc
    return ToneResponse(status="ok")


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
@router.post("/tts/test", summary="Speak a test phrase", response_model=TTSTestResponse)
async def tts_test(
    request: Request, body: TTSTestRequest, _: None = Depends(require_api_key)
) -> TTSTestResponse:
    """Speak a test phrase through the configured TTS engine."""
    bridge = _bridge(request)
    if bridge is None or not bridge.is_ready:
        raise HTTPException(status_code=503, detail="DeskBot app not attached")

    if body.direct:
        tts = getattr(bridge, "tts", None)
        if tts is None:
            raise HTTPException(status_code=503, detail="No TTS engine available")
        try:
            buffer = await tts.speak(body.text)
            audio = getattr(bridge, "audio", None)
            if audio is not None and buffer is not None and not buffer.is_empty:
                await audio.play(buffer)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc
        return TTSTestResponse(status="ok", text=body.text, engine=type(tts).__name__)

    # Via LLM pipeline.
    from robot.events.events import SpeechRecognized

    if bridge.bus is None:
        raise HTTPException(status_code=503, detail="Event bus not available")
    await bridge.bus.publish(SpeechRecognized(text=body.text, confidence=1.0))
    return TTSTestResponse(status="ok", text=body.text, via="llm_pipeline")


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
@router.post("/llm/test", summary="Send a test prompt to the LLM", response_model=LLMTestResponse)
async def llm_test(
    request: Request, body: LLMTestRequest, _: None = Depends(require_api_key)
) -> LLMTestResponse:
    """Send a test prompt to the LLM and return the response text."""
    bridge = _bridge(request)
    conv = getattr(bridge, "conversation", None) if bridge else None
    if conv is None or conv.llm is None:
        raise HTTPException(status_code=503, detail="No LLM available")

    from robot.interfaces.llm import Message, Role

    messages = [
        Message(role=Role.SYSTEM, content="You are a helpful desktop robot. Reply concisely."),
        Message(role=Role.USER, content=body.prompt),
    ]
    try:
        result = await conv.llm.complete(messages)
        # complete() may return a plain string or an LLMResponse
        # depending on the provider; handle both.
        response_text = result.text if hasattr(result, "text") else str(result)
        return LLMTestResponse(
            status="ok",
            prompt=body.prompt,
            response=response_text,
            engine=type(conv.llm).__name__,
        )
    except Exception as exc:
        return LLMTestResponse(status="error", prompt=body.prompt, error=str(exc))


# ---------------------------------------------------------------------------
# Sound effects
# ---------------------------------------------------------------------------
@router.post(
    "/sound-effect/{name}", summary="Play a sound effect", response_model=SoundEffectResponse
)
async def play_sound_effect(
    request: Request, name: str, _: None = Depends(require_api_key)
) -> SoundEffectResponse:
    """Play a named sound effect (e.g. ``talk``, ``thinking``, ``cute``)."""
    bridge = _bridge(request)
    sfx = getattr(bridge, "sound_effects", None) if bridge else None
    if sfx is None:
        raise HTTPException(status_code=503, detail="No sound effects player available")
    try:
        await sfx.play(name)
        return SoundEffectResponse(status="ok", name=name)
    except Exception as exc:
        return SoundEffectResponse(status="error", name=name, error=str(exc))


@router.get(
    "/sound-effects",
    summary="List available sound effects",
    response_model=SoundEffectsListResponse,
)
async def list_sound_effects(request: Request) -> SoundEffectsListResponse:
    """List all available sound effect names."""
    bridge = _bridge(request)
    sfx = getattr(bridge, "sound_effects", None) if bridge else None
    if sfx is None:
        return SoundEffectsListResponse(available=False, effects=[])
    effects = sorted(sfx._index.keys()) if hasattr(sfx, "_index") else []
    return SoundEffectsListResponse(available=sfx.enabled, effects=effects)


__all__ = ["router"]
