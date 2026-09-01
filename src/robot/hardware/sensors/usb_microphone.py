"""USB microphone driver backed by ``sounddevice`` / PortAudio.

The microphone binds its asyncio loop before it starts the reader
thread. This avoids the race where PortAudio produces audio before the
consumer queue exists and chunks are silently dropped.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import struct
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from robot.interfaces.microphone import AudioChunk, Microphone
from robot.logging import get_logger

_log = get_logger("hardware.sensors.microphone.usb")

_DIAGNOSTIC_INTERVAL_S = 2.0
_STREAM_STOP_TIMEOUT_S = 2.0


def rms(pcm: bytes) -> float:
    """Root-mean-square energy of an s16le PCM buffer (0..1 normalised)."""
    count = len(pcm) // 2
    if count == 0:
        return 0.0
    total = 0
    for offset in range(0, count, 1024):
        n = min(1024, count - offset)
        samples = struct.unpack(f"<{n}h", pcm[offset * 2 : (offset + n) * 2])
        for sample in samples:
            total += sample * sample
    return math.sqrt(total / count) / 32768.0


@dataclass(slots=True)
class UsbMicrophone(Microphone):
    """Real microphone capture using a PortAudio blocking ``InputStream``."""

    input_device: str | int = "default"
    _sample_rate_field: int = 16_000
    channels: int = 1
    frame_ms: int = 30
    _queue_maxsize: int = 64
    _sd: Any = field(init=False, repr=False)
    _stream: Any | None = field(default=None, init=False, repr=False)
    _queue: asyncio.Queue[AudioChunk | None] | None = field(default=None, init=False, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)
    _stream_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _reader_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _reader_error: str | None = field(default=None, init=False)
    _reader_started_at: float | None = field(default=None, init=False)
    _configured_sample_rate: int = field(default=16_000, init=False)
    _actual_sample_rate: int = field(default=16_000, init=False)
    _frame_samples: int = field(default=480, init=False)
    _resolved_device_index: int | None = field(default=None, init=False)
    _resolved_device_info: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _chunks_received: int = field(default=0, init=False)
    _chunks_enqueued: int = field(default=0, init=False)
    _chunks_consumed: int = field(default=0, init=False)
    _chunks_dropped: int = field(default=0, init=False)
    _overflow_count: int = field(default=0, init=False)
    _silent_chunks: int = field(default=0, init=False)
    _nonzero_chunks: int = field(default=0, init=False)
    _last_rms_value: float = field(default=0.0, init=False)
    _last_min_sample: int = field(default=0, init=False)
    _last_max_sample: int = field(default=0, init=False)
    _last_diag_at: float = field(default=0.0, init=False)
    _timeline_s: float = field(default=0.0, init=False)

    @property
    def sample_rate(self) -> int:
        return (
            self._actual_sample_rate if self._stream is not None else self._configured_sample_rate
        )

    @property
    def thread_alive(self) -> bool:
        thread = self._reader_thread
        return thread is not None and thread.is_alive()

    def __post_init__(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - hardware-specific
            raise RuntimeError(
                f"sounddevice is required for UsbMicrophone: {exc!r}. "
                "Install with `uv pip install sounddevice`."
            ) from exc
        self._sd = sd
        self._configured_sample_rate = int(self._sample_rate_field)
        self._actual_sample_rate = int(self._sample_rate_field)
        self._frame_samples = max(1, round(self._configured_sample_rate * self.frame_ms / 1000))

    def describe_selection(self) -> dict[str, Any]:
        """Return the device DeskBot will select for this microphone."""
        default_input = self._default_input_index()
        resolved_index = self._resolve_device_index()
        info = self._query_input_device(resolved_index)
        return {
            "configured_input_device": self.input_device,
            "portaudio_default_input_device": default_input,
            "resolved_device_index": resolved_index,
            "resolved_device_name": info.get("name"),
            "max_input_channels": int(info.get("max_input_channels", 0)),
            "default_sample_rate": round(info.get("default_samplerate", 0) or 0),
            "requested_sample_rate": self._configured_sample_rate,
            "channels": self.channels,
            "frame_ms": self.frame_ms,
            "frame_samples": self._frame_samples,
        }

    def runtime_stats(self) -> dict[str, Any]:
        """Return capture/runtime counters for diagnostics and logging."""
        return {
            "configured_input_device": self.input_device,
            "resolved_device_index": self._resolved_device_index,
            "resolved_device_name": None
            if self._resolved_device_info is None
            else self._resolved_device_info.get("name"),
            "requested_sample_rate": self._configured_sample_rate,
            "actual_stream_sample_rate": self.sample_rate,
            "channels": self.channels,
            "frame_samples": self._frame_samples,
            "thread_alive": self.thread_alive,
            "chunks_received": self._chunks_received,
            "chunks_enqueued": self._chunks_enqueued,
            "chunks_consumed": self._chunks_consumed,
            "chunks_dropped": self._chunks_dropped,
            "nonzero_chunks": self._nonzero_chunks,
            "silent_chunks": self._silent_chunks,
            "overflows": self._overflow_count,
            "queue_size": None if self._queue is None else self._queue.qsize(),
            "queue_maxsize": self._queue_maxsize,
            "rms": round(self._last_rms_value, 5),
            "min_sample": self._last_min_sample,
            "max_sample": self._last_max_sample,
            "reader_error": self._reader_error,
        }

    @classmethod
    def list_input_devices(cls) -> dict[str, Any]:
        """Enumerate available input devices and the PortAudio default."""
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - hardware-specific
            raise RuntimeError("sounddevice is required for microphone diagnostics") from exc

        default_input: int | None = None
        with contextlib.suppress(Exception):
            raw_default = sd.default.device[0]
            if raw_default is not None and raw_default >= 0:
                default_input = int(raw_default)

        devices: list[dict[str, Any]] = []
        for index, info in enumerate(sd.query_devices()):
            max_input_channels = int(info.get("max_input_channels", 0))
            if max_input_channels <= 0:
                continue
            devices.append(
                {
                    "index": index,
                    "name": info.get("name"),
                    "max_input_channels": max_input_channels,
                    "default_sample_rate": round(info.get("default_samplerate", 0) or 0),
                    "is_default": default_input == index,
                }
            )
        return {
            "default_input_device": default_input,
            "devices": devices,
        }

    @classmethod
    def diagnose_capture(
        cls,
        *,
        input_device: str | int = "default",
        sample_rate: int = 16_000,
        channels: int = 1,
        frame_ms: int = 30,
        duration_s: float = 1.5,
    ) -> dict[str, Any]:
        """Open a microphone, capture a short sample, and report signal stats."""
        mic = cls(
            input_device=input_device,
            _sample_rate_field=sample_rate,
            channels=channels,
            frame_ms=frame_ms,
        )
        diagnostics = mic.describe_selection()
        try:
            mic._open_stream()
            stream = mic._stream
            assert stream is not None
            target_chunks = max(1, math.ceil(duration_s / max(frame_ms / 1000, 0.001)))
            collected = bytearray()
            observed_chunks = 0
            overflow_count = 0
            rms_values: list[float] = []
            min_sample = 0
            max_sample = 0
            for _ in range(target_chunks):
                data, overflowed = stream.read(mic._frame_samples)
                observed_chunks += 1
                if overflowed:
                    overflow_count += 1
                pcm = data.tobytes() if hasattr(data, "tobytes") else bytes(data)
                collected.extend(pcm)
                stats = mic._analyse_pcm(pcm)
                rms_values.append(stats["rms"])
                min_sample = min(min_sample, int(stats["min_sample"]))
                max_sample = max(max_sample, int(stats["max_sample"]))

            diagnostics.update(
                {
                    "actual_stream_sample_rate": mic.sample_rate,
                    "observed_chunks": observed_chunks,
                    "observed_bytes": len(collected),
                    "nonzero_audio": any(value > 0.0 for value in rms_values),
                    "rms_min": round(min(rms_values) if rms_values else 0.0, 5),
                    "rms_max": round(max(rms_values) if rms_values else 0.0, 5),
                    "rms_avg": round(sum(rms_values) / len(rms_values), 5) if rms_values else 0.0,
                    "min_sample": min_sample,
                    "max_sample": max_sample,
                    "overflows": overflow_count,
                }
            )
            return diagnostics
        finally:
            mic._close_stream(join_reader=False)

    async def stream(self) -> AsyncIterator[AudioChunk]:
        loop = asyncio.get_running_loop()
        await self._ensure_started(loop)
        queue = self._queue
        if queue is None:
            raise RuntimeError("microphone queue is not available")

        while True:
            item = await queue.get()
            if item is None:
                if self._reader_error is not None:
                    raise RuntimeError(self._reader_error)
                return
            self._chunks_consumed += 1
            self._log_runtime_diagnostics(force=False)
            yield item

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_stream(join_reader=True)
        self._drain_queue()
        await self._signal_queue_end()
        _log.info("microphone.closed", **self.runtime_stats())

    def _drain_queue(self) -> None:
        """Discard any buffered chunks so the stream ends immediately on close."""
        queue = self._queue
        if queue is None:
            return
        drained = 0
        while not queue.empty():
            try:
                queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        if drained:
            _log.debug("microphone.queue_drained", chunks=drained)

    async def _ensure_started(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._closed:
            raise RuntimeError("microphone is closed")
        if self._stream is not None and self.thread_alive:
            return
        with self._stream_lock:
            if self._stream is not None and self.thread_alive:
                return
            self._loop = loop
            self._queue = asyncio.Queue(maxsize=self._queue_maxsize)
            self._reader_error = None
            self._chunks_received = 0
            self._chunks_enqueued = 0
            self._chunks_consumed = 0
            self._chunks_dropped = 0
            self._overflow_count = 0
            self._silent_chunks = 0
            self._nonzero_chunks = 0
            self._timeline_s = 0.0
            self._last_diag_at = 0.0
            self._open_stream()
            self._reader_thread = threading.Thread(
                target=self._read_loop,
                name="UsbMicrophone-read",
                daemon=True,
            )
            self._reader_started_at = time.monotonic()
            self._reader_thread.start()
            self._log_runtime_diagnostics(force=True)

    def _resolve_device_index(self) -> int:
        if isinstance(self.input_device, int):
            return self._validate_input_device_index(self.input_device)

        raw = str(self.input_device).strip()
        if raw.isdigit():
            return self._validate_input_device_index(int(raw))

        if raw.lower() in {"", "default"}:
            default_input = self._default_input_index()
            if default_input is None:
                raise RuntimeError("PortAudio has no default input device configured")
            return self._validate_input_device_index(default_input)

        needle = raw.lower()
        for index, info in enumerate(self._sd.query_devices()):
            if int(info.get("max_input_channels", 0)) <= 0:
                continue
            if needle in str(info.get("name", "")).lower():
                return index
        raise RuntimeError(f"no microphone matching {self.input_device!r}")

    def _validate_input_device_index(self, index: int) -> int:
        info = self._query_input_device(index)
        max_input_channels = int(info.get("max_input_channels", 0))
        if max_input_channels <= 0:
            raise RuntimeError(f"device {index} ({info.get('name', '?')}) has no input channels")
        return index

    def _query_input_device(self, index: int) -> dict[str, Any]:
        try:
            info = self._sd.query_devices(index, kind="input")
        except Exception as exc:
            raise RuntimeError(f"could not query input device {index}: {exc}") from exc
        return dict(info)

    def _default_input_index(self) -> int | None:
        with contextlib.suppress(Exception):
            default_input = self._sd.default.device[0]
            if default_input is not None and default_input >= 0:
                return int(default_input)
        return None

    def _open_stream(self) -> None:
        resolved_index = self._resolve_device_index()
        device_info = self._query_input_device(resolved_index)
        requested_sample_rate = self._configured_sample_rate
        frame_samples = max(1, round(requested_sample_rate * self.frame_ms / 1000))
        max_input_channels = int(device_info.get("max_input_channels", 0))
        default_sample_rate = round(device_info.get("default_samplerate", 0) or 0)

        # Pre-flight check: verify the device can handle the requested
        # settings.  This is advisory - on some ALSA/PipeWire setups
        # check_input_settings rejects rates that PortAudio can actually
        # resample at open time.  If the check fails we log a warning and
        # still attempt to open the stream; the InputStream call will raise
        # a clear error if the settings are truly unsupported.
        try:
            self._sd.check_input_settings(
                device=resolved_index,
                samplerate=requested_sample_rate,
                channels=self.channels,
                dtype="int16",
            )
        except Exception as exc:
            _log.warning(
                "microphone.check_input_settings_failed",
                device=resolved_index,
                requested_sample_rate=requested_sample_rate,
                channels=self.channels,
                error=str(exc),
                message="proceeding to open stream; PortAudio may resample",
            )

        try:
            stream = self._sd.InputStream(
                samplerate=requested_sample_rate,
                channels=self.channels,
                dtype="int16",
                blocksize=frame_samples,
                device=resolved_index,
            )
            stream.start()
        except Exception as exc:
            raise RuntimeError(
                f"could not open PortAudio input stream for device {resolved_index}: {exc}"
            ) from exc

        actual_sample_rate = round(getattr(stream, "samplerate", requested_sample_rate))
        actual_channels = int(getattr(stream, "channels", self.channels))
        if actual_channels != self.channels:
            with contextlib.suppress(Exception):
                stream.stop()
                stream.close()
            raise RuntimeError(
                f"stream channel count mismatch: requested={self.channels} actual={actual_channels}"
            )

        self._stream = stream
        self._resolved_device_index = resolved_index
        self._resolved_device_info = device_info
        self._actual_sample_rate = actual_sample_rate
        self._frame_samples = max(1, round(actual_sample_rate * self.frame_ms / 1000))

        _log.info(
            "microphone.opened",
            configured_input_device=self.input_device,
            portaudio_default_input_device=self._default_input_index(),
            resolved_device_index=resolved_index,
            resolved_device_name=device_info.get("name"),
            max_input_channels=max_input_channels,
            default_sample_rate=default_sample_rate,
            requested_sample_rate=requested_sample_rate,
            actual_stream_sample_rate=actual_sample_rate,
            channels=self.channels,
            frame_size=self._frame_samples,
        )

    def _close_stream(self, *, join_reader: bool) -> None:
        # Signal the reader thread to stop before nulling the stream
        # so it doesn't write into a None reference.
        self._stop.set()
        stream = self._stream
        self._stream = None
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.stop()
            with contextlib.suppress(Exception):
                stream.close()
        thread = self._reader_thread
        if join_reader and thread is not None and thread.is_alive():
            thread.join(timeout=_STREAM_STOP_TIMEOUT_S)
            if thread.is_alive():
                _log.warning("microphone.thread_join_timeout", timeout_s=_STREAM_STOP_TIMEOUT_S)
        if join_reader:
            self._reader_thread = None

    def _read_loop(self) -> None:
        stream = self._stream
        loop = self._loop
        if stream is None or loop is None:
            self._reader_error = "microphone reader started without an open stream or loop"
            return

        while not self._closed and self._stream is not None:
            try:
                data, overflowed = stream.read(self._frame_samples)
            except Exception as exc:  # pragma: no cover - hardware-specific
                self._reader_error = f"PortAudio read failed: {exc}"
                _log.error("microphone.read_failed", error=str(exc))
                break

            pcm = data.tobytes() if hasattr(data, "tobytes") else bytes(data)
            self._chunks_received += 1
            self._timeline_s += self._frame_samples / max(self.sample_rate, 1)

            if overflowed:
                self._overflow_count += 1
                _log.warning(
                    "microphone.overflow",
                    overflows=self._overflow_count,
                    resolved_device_index=self._resolved_device_index,
                )

            stats = self._analyse_pcm(pcm)
            self._last_rms_value = stats["rms"]
            self._last_min_sample = int(stats["min_sample"])
            self._last_max_sample = int(stats["max_sample"])
            if stats["rms"] == 0.0:
                self._silent_chunks += 1
            else:
                self._nonzero_chunks += 1

            chunk = AudioChunk(
                pcm=pcm,
                sample_rate=self.sample_rate,
                channels=self.channels,
                timestamp=self._timeline_s,
            )

            try:
                loop.call_soon_threadsafe(self._enqueue, chunk)
            except RuntimeError as exc:
                self._reader_error = f"asyncio loop is unavailable: {exc}"
                _log.error("microphone.queue_failed", error=str(exc))
                break

            self._log_runtime_diagnostics(force=False)

        self._signal_queue_end_threadsafe()

    def _enqueue(self, chunk: AudioChunk) -> None:
        queue = self._queue
        if queue is None:
            self._chunks_dropped += 1
            return
        try:
            queue.put_nowait(chunk)
            self._chunks_enqueued += 1
        except asyncio.QueueFull:
            self._chunks_dropped += 1
            _log.warning(
                "microphone.queue_full",
                chunks_dropped=self._chunks_dropped,
                queue_size=queue.qsize(),
                queue_maxsize=queue.maxsize,
            )

    def _signal_queue_end_threadsafe(self) -> None:
        loop = self._loop
        if loop is None:
            return
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(self._enqueue_terminal)

    async def _signal_queue_end(self) -> None:
        loop = self._loop
        if loop is None:
            return
        future = loop.create_future()

        def _drain_and_enqueue_terminal() -> None:
            # Drain any chunks that were enqueued by the reader thread
            # but not yet consumed.  This guarantees the terminal None
            # is the *only* item left so the stream generator ends
            # immediately rather than yielding stale buffered audio.
            self._drain_queue()
            self._enqueue_terminal()
            future.set_result(None)

        loop.call_soon_threadsafe(_drain_and_enqueue_terminal)
        with contextlib.suppress(Exception):
            await future

    def _enqueue_terminal(self) -> None:
        queue = self._queue
        if queue is None or queue.full():
            return
        queue.put_nowait(None)

    def _log_runtime_diagnostics(self, *, force: bool) -> None:
        """Rate-limited INFO summary so chunk drops surface in the dashboard.

        The dashboard ring buffer captures INFO/WARNING but not DEBUG, and
        the conversation ``audio_loop.tick`` is DEBUG -- so without this
        summary a rising ``chunks_dropped`` counter is invisible until the
        mic is closed. ``force=True`` logs immediately (used at start);
        otherwise at most once per ``_DIAGNOSTIC_INTERVAL_S``.
        """
        now = time.monotonic()
        if not force and now - self._last_diag_at < _DIAGNOSTIC_INTERVAL_S:
            return
        self._last_diag_at = now
        _log.info("microphone.runtime", **self.runtime_stats())

    def _analyse_pcm(self, pcm: bytes) -> dict[str, float | int]:
        if len(pcm) < 2:
            return {"rms": 0.0, "min_sample": 0, "max_sample": 0}
        sample_count = len(pcm) // 2
        samples = struct.unpack(f"<{sample_count}h", pcm[: sample_count * 2])
        return {
            "rms": rms(pcm),
            "min_sample": int(min(samples)),
            "max_sample": int(max(samples)),
        }


__all__ = ["UsbMicrophone", "rms"]
