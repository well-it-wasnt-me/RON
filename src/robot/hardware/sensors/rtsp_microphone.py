"""RTSP audio microphone backend.

Reads the audio track from an RTSP stream and exposes it through RON's
Microphone interface.

The camera currently provides:

    pcm_alaw, 8000 Hz, mono

The backend decodes that audio with PyAV/FFmpeg, resamples it to the
configured RON microphone rate, and emits fixed-size AudioChunk objects.

The decoder is paced against the audio timeline so that RTSP audio is
presented to RON in real time, just like a physical microphone. Without
this pacing PyAV can decode buffered RTSP packets faster than real time,
which causes the asyncio queue to fill and audio chunks to be dropped.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from robot.interfaces.microphone import AudioChunk, Microphone
from robot.logging import get_logger

_log = get_logger("hardware.sensors.microphone.rtsp")

_DEFAULT_QUEUE_SIZE = 64
_DEFAULT_RTSP_TIMEOUT_US = 5_000_000
_DEFAULT_RECONNECT_INITIAL_DELAY_S = 1.0
_DEFAULT_RECONNECT_MAX_DELAY_S = 5.0
_THREAD_JOIN_TIMEOUT_S = 2.0


@dataclass(slots=True)
class RtspMicrophone(Microphone):
    """Microphone backed by the audio track of an RTSP stream."""

    url: str
    output_sample_rate: int = 16_000
    channels: int = 1
    frame_ms: int = 30
    transport: str = "tcp"
    queue_maxsize: int = _DEFAULT_QUEUE_SIZE
    reconnect_initial_delay: float = _DEFAULT_RECONNECT_INITIAL_DELAY_S
    reconnect_max_delay: float = _DEFAULT_RECONNECT_MAX_DELAY_S

    _av: Any = field(init=False, repr=False)
    _container: Any | None = field(default=None, init=False, repr=False)
    _audio_stream: Any | None = field(default=None, init=False, repr=False)
    _resampler: Any | None = field(default=None, init=False, repr=False)

    _queue: asyncio.Queue[AudioChunk | None] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _loop: asyncio.AbstractEventLoop | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _thread: threading.Thread | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _stop: threading.Event = field(
        default_factory=threading.Event,
        init=False,
        repr=False,
    )

    _closed: bool = field(default=False, init=False)
    _started: bool = field(default=False, init=False)
    _reader_error: str | None = field(default=None, init=False)

    _input_sample_rate: int = field(default=0, init=False)
    _input_channels: int = field(default=0, init=False)
    _input_codec: str = field(default="", init=False)

    _chunks_decoded: int = field(default=0, init=False)
    _chunks_emitted: int = field(default=0, init=False)
    _chunks_dropped: int = field(default=0, init=False)
    _reconnect_attempts: int = field(default=0)

    _timeline_s: float = field(default=0.0, init=False)
    _playback_start_monotonic: float | None = field(
        default=None,
        init=False,
        repr=False,
    )

    _pcm_buffer: bytearray = field(
        default_factory=bytearray,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError("RtspMicrophone requires a non-empty RTSP URL")

        if self.output_sample_rate <= 0:
            raise ValueError("output_sample_rate must be greater than zero")

        if self.channels != 1:
            raise ValueError("RtspMicrophone currently supports mono output only")

        if self.frame_ms <= 0:
            raise ValueError("frame_ms must be greater than zero")

        if self.transport not in {"tcp", "udp"}:
            raise ValueError(
                f"unsupported RTSP transport {self.transport!r}; expected 'tcp' or 'udp'"
            )

        if self.queue_maxsize <= 0:
            raise ValueError("queue_maxsize must be greater than zero")

        if self.reconnect_initial_delay <= 0:
            raise ValueError("reconnect_initial_delay must be greater than zero")

        if self.reconnect_max_delay <= 0:
            raise ValueError("reconnect_max_delay must be greater than zero")

        try:
            import av
        except ImportError as exc:
            raise RuntimeError(
                "PyAV is required for RTSP microphone support. "
                "Install the audio dependencies with `uv sync --extra audio`."
            ) from exc

        self._av = av

    @property
    def sample_rate(self) -> int:
        """Output sample rate presented to RON."""
        return self.output_sample_rate

    def stream(self) -> AsyncIterator[AudioChunk]:
        """Return an async iterator of audio chunks."""
        return self._stream()

    async def _stream(self) -> AsyncIterator[AudioChunk]:
        """Yield decoded and resampled audio chunks."""
        if self._closed:
            raise RuntimeError("RTSP microphone is closed")

        await self._ensure_started()

        queue = self._queue
        if queue is None:
            raise RuntimeError("RTSP microphone queue is unavailable")

        while True:
            item = await queue.get()

            if item is None:
                if self._reader_error is not None:
                    raise RuntimeError(self._reader_error)
                return

            yield item

    async def close(self) -> None:
        """Stop the decoder and close the RTSP stream."""
        if self._closed:
            return

        self._closed = True
        self._stop.set()

        self._close_container()

        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=_THREAD_JOIN_TIMEOUT_S)

        self._thread = None

        await self._signal_queue_end()

        _log.info(
            "rtsp_microphone.closed",
            url=self._safe_url(),
            chunks_decoded=self._chunks_decoded,
            chunks_emitted=self._chunks_emitted,
            chunks_dropped=self._chunks_dropped,
            reconnect_attempts=self._reconnect_attempts,
        )

    def diagnostics(self) -> dict[str, Any]:
        """Return RTSP microphone diagnostics."""
        queue = self._queue

        return {
            "url": self._safe_url(),
            "transport": self.transport,
            "input_codec": self._input_codec,
            "input_sample_rate": self._input_sample_rate,
            "input_channels": self._input_channels,
            "output_sample_rate": self.output_sample_rate,
            "output_channels": self.channels,
            "frame_ms": self.frame_ms,
            "frame_samples": self._frame_samples(),
            "frame_bytes": self._frame_bytes(),
            "started": self._started,
            "closed": self._closed,
            "thread_alive": (self._thread is not None and self._thread.is_alive()),
            "queue_size": None if queue is None else queue.qsize(),
            "queue_maxsize": self.queue_maxsize,
            "chunks_decoded": self._chunks_decoded,
            "chunks_emitted": self._chunks_emitted,
            "chunks_dropped": self._chunks_dropped,
            "reconnect_attempts": self._reconnect_attempts,
            "reader_error": self._reader_error,
            "timeline_s": round(self._timeline_s, 3),
        }

    async def _ensure_started(self) -> None:
        """Start the decoder thread once an asyncio loop exists."""
        if self._started:
            return

        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self.queue_maxsize)

        self._stop.clear()
        self._reader_error = None
        self._started = True

        self._thread = threading.Thread(
            target=self._decode_loop,
            name="RtspMicrophone-decode",
            daemon=True,
        )
        self._thread.start()

    def _decode_loop(self) -> None:
        """Decode RTSP audio in a background thread."""
        reconnect_delay = self.reconnect_initial_delay

        while not self._stop.is_set() and not self._closed:
            try:
                self._open_container()

                reconnect_delay = self.reconnect_initial_delay
                self._decode_current_connection()

            except Exception as exc:
                if self._stop.is_set() or self._closed:
                    break

                self._reader_error = str(exc)

                _log.warning(
                    "rtsp_microphone.connection_failed",
                    url=self._safe_url(),
                    error=str(exc),
                    reconnect_delay=reconnect_delay,
                )

                self._close_container()

                self._reconnect_attempts += 1

                self._stop.wait(reconnect_delay)

                reconnect_delay = min(
                    reconnect_delay + self.reconnect_initial_delay,
                    self.reconnect_max_delay,
                )

        self._close_container()
        self._signal_queue_end_threadsafe()

    def _open_container(self) -> None:
        """Open the RTSP stream and select its audio stream."""
        options = {
            "rtsp_transport": self.transport,
            "timeout": str(_DEFAULT_RTSP_TIMEOUT_US),
        }

        _log.info(
            "rtsp_microphone.opening",
            url=self._safe_url(),
            transport=self.transport,
        )

        container = self._av.open(
            self.url,
            mode="r",
            options=options,
        )

        audio_stream = next(
            (stream for stream in container.streams if stream.type == "audio"),
            None,
        )

        if audio_stream is None:
            container.close()
            raise RuntimeError(f"RTSP stream {self._safe_url()!r} contains no audio stream")

        input_sample_rate = int(audio_stream.codec_context.sample_rate or 0)

        if input_sample_rate <= 0:
            container.close()
            raise RuntimeError("RTSP audio stream did not report a valid sample rate")

        input_channels = int(audio_stream.codec_context.channels or 0)

        codec_name = str(
            getattr(
                audio_stream.codec_context.codec,
                "name",
                "",
            )
            or ""
        )

        self._container = container
        self._audio_stream = audio_stream
        self._input_sample_rate = input_sample_rate
        self._input_channels = input_channels
        self._input_codec = codec_name

        self._resampler = self._av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=self.output_sample_rate,
        )

        self._pcm_buffer.clear()
        self._timeline_s = 0.0
        self._playback_start_monotonic = time.monotonic()

        _log.info(
            "rtsp_microphone.opened",
            url=self._safe_url(),
            audio_codec=codec_name,
            input_sample_rate=input_sample_rate,
            input_channels=input_channels,
            output_sample_rate=self.output_sample_rate,
            output_channels=self.channels,
            frame_ms=self.frame_ms,
        )

    def _decode_current_connection(self) -> None:
        """Decode all audio frames from the current RTSP connection."""
        container = self._container
        audio_stream = self._audio_stream
        resampler = self._resampler

        if container is None:
            raise RuntimeError("RTSP container is not open")

        if audio_stream is None:
            raise RuntimeError("RTSP audio stream is not selected")

        if resampler is None:
            raise RuntimeError("RTSP audio resampler is not initialized")

        # PyAV's Stream.index is the index in the complete container
        # stream list. container.decode(audio=...) expects the index in
        # the audio-stream collection.
        #
        # Example:
        #
        #   container streams:
        #       #0 video
        #       #1 audio
        #
        #   audio streams:
        #       #0 audio
        #
        # So audio_stream.index == 1, while decode(audio=0) is required.
        audio_stream_index = next(
            (
                i
                for i, stream in enumerate(container.streams.audio)
                if stream.index == audio_stream.index
            ),
            None,
        )

        if audio_stream_index is None:
            raise RuntimeError(
                "RTSP audio stream could not be mapped to the container audio-stream index"
            )

        _log.info(
            "rtsp_microphone.decoder_started",
            codec=self._input_codec,
            input_sample_rate=self._input_sample_rate,
            input_channels=self._input_channels,
            audio_stream_index=audio_stream_index,
        )

        for frame in container.decode(
            audio=audio_stream_index,
        ):
            if self._stop.is_set() or self._closed:
                return

            self._chunks_decoded += 1

            resampled_frames = resampler.resample(frame)

            for output_frame in resampled_frames:
                if self._stop.is_set() or self._closed:
                    return

                pcm = self._frame_to_pcm(output_frame)

                if pcm:
                    self._pcm_buffer.extend(pcm)
                    self._emit_complete_chunks()

        # Flush audio buffered inside the resampler.
        flushed_frames = resampler.resample(None)

        for output_frame in flushed_frames:
            if self._stop.is_set() or self._closed:
                return

            pcm = self._frame_to_pcm(output_frame)

            if pcm:
                self._pcm_buffer.extend(pcm)

        self._emit_complete_chunks()

    def _frame_to_pcm(self, frame: Any) -> bytes:
        """Convert a resampled PyAV audio frame to packed s16 PCM."""
        array = frame.to_ndarray()

        if array.ndim == 0:
            raise RuntimeError("RTSP audio frame produced a scalar NumPy array")

        if array.ndim == 1:
            samples = array

        elif array.ndim == 2:
            if array.shape[0] == 1:
                samples = array[0]

            elif array.shape[1] == 1:
                samples = array[:, 0]

            else:
                raise RuntimeError(
                    "RTSP microphone expected mono audio after "
                    "resampling, got ndarray shape="
                    f"{array.shape!r}"
                )

        else:
            raise RuntimeError(f"unexpected RTSP audio ndarray shape: {array.shape!r}")

        return samples.astype("<i2", copy=False).tobytes()  # type: ignore[no-any-return]

    def _frame_samples(self) -> int:
        """Return the number of output samples in each chunk."""
        return max(
            1,
            round(self.output_sample_rate * self.frame_ms / 1000),
        )

    def _frame_bytes(self) -> int:
        """Return the byte size of each chunk."""
        return self._frame_samples() * 2 * self.channels

    def _emit_complete_chunks(self) -> None:
        """Emit complete fixed-size AudioChunk objects in real time."""
        frame_bytes = self._frame_bytes()

        while len(self._pcm_buffer) >= frame_bytes:
            if self._stop.is_set() or self._closed:
                return

            pcm = bytes(self._pcm_buffer[:frame_bytes])
            del self._pcm_buffer[:frame_bytes]

            chunk_duration = self._frame_samples() / self.output_sample_rate

            chunk_start_s = self._timeline_s
            self._timeline_s += chunk_duration

            # Pace the RTSP stream against wall-clock time.
            #
            # PyAV may have several seconds of RTSP data buffered and
            # can decode that data considerably faster than real time.
            # A physical microphone cannot do that, so don't let the
            # RTSP producer outrun the consumer.
            playback_start = self._playback_start_monotonic

            if playback_start is not None:
                target_time = playback_start + chunk_start_s
                delay = target_time - time.monotonic()

                if delay > 0:
                    self._stop.wait(delay)

                    if self._stop.is_set() or self._closed:
                        return

            chunk = AudioChunk(
                pcm=pcm,
                sample_rate=self.output_sample_rate,
                channels=self.channels,
                timestamp=chunk_start_s,
            )

            self._enqueue_threadsafe(chunk)

    def _enqueue_threadsafe(self, chunk: AudioChunk) -> None:
        """Schedule an audio chunk on the asyncio loop."""
        loop = self._loop

        if loop is None:
            self._chunks_dropped += 1
            return

        try:
            loop.call_soon_threadsafe(
                self._enqueue,
                chunk,
            )
        except RuntimeError:
            self._chunks_dropped += 1

    def _enqueue(self, chunk: AudioChunk) -> None:
        """Put an audio chunk into the asyncio queue."""
        queue = self._queue

        if queue is None:
            self._chunks_dropped += 1
            return

        try:
            queue.put_nowait(chunk)
            self._chunks_emitted += 1
        except asyncio.QueueFull:
            self._chunks_dropped += 1

            # Queue-full logging is intentionally rate limited. A
            # producer running faster than its consumer can otherwise
            # generate thousands of log messages per second.
            if self._chunks_dropped == 1 or self._chunks_dropped % 100 == 0:
                _log.warning(
                    "rtsp_microphone.queue_full",
                    chunks_dropped=self._chunks_dropped,
                    queue_size=queue.qsize(),
                    queue_maxsize=queue.maxsize,
                )

    async def _signal_queue_end(self) -> None:
        """Signal end of stream from asyncio context."""
        queue = self._queue

        if queue is None:
            return

        # The normal consumer should drain outstanding audio before
        # seeing the sentinel. If the queue is full during shutdown,
        # discard the oldest item to guarantee that the sentinel can
        # be delivered.
        while True:
            try:
                queue.put_nowait(None)
                return
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()

    def _signal_queue_end_threadsafe(self) -> None:
        """Signal end of stream from the decoder thread."""
        loop = self._loop

        if loop is None:
            return

        try:
            loop.call_soon_threadsafe(
                self._signal_queue_end_callback,
            )
        except RuntimeError:
            return

    def _signal_queue_end_callback(self) -> None:
        """Queue termination callback executed on asyncio's loop."""
        queue = self._queue

        if queue is None:
            return

        while True:
            try:
                queue.put_nowait(None)
                return
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()

    def _close_container(self) -> None:
        """Close the current PyAV container."""
        container = self._container

        self._container = None
        self._audio_stream = None
        self._resampler = None
        self._playback_start_monotonic = None

        if container is not None:
            with contextlib.suppress(Exception):
                container.close()

    def _safe_url(self) -> str:
        """Return the RTSP URL with its password masked."""
        if "@" not in self.url:
            return self.url

        try:
            scheme, rest = self.url.split("://", 1)
            credentials, host = rest.split("@", 1)

            if ":" in credentials:
                username, _password = credentials.split(":", 1)
                return f"{scheme}://{username}:***@{host}"

            return f"{scheme}://***@{host}"
        except ValueError:
            return "***"


__all__ = ["RtspMicrophone"]
