"""Tests for the RTSP microphone backend.

Focused on the audio-level plumbing that the dashboard relies on:
``_rms`` (used by the ``/settings/mic/level`` meter) and the
``_last_rms_value`` tracking that makes that meter work for the RTSP mic,
not just :class:`UsbMicrophone`.
"""

from __future__ import annotations

import struct
import time

import pytest

from robot.hardware.sensors.rtsp_microphone import RtspMicrophone, _rms


def test_rms_silence_is_zero() -> None:
    assert _rms(b"\x00\x00" * 480) == 0.0


def test_rms_full_scale_is_one() -> None:
    pcm = struct.pack("<480h", *([32767] * 480))
    # 32767/32768 is ~0.99997, not exactly 1.0.
    assert _rms(pcm) == pytest.approx(1.0, abs=1e-3)


def test_rms_empty_is_zero() -> None:
    assert _rms(b"") == 0.0


def test_rms_mid_level() -> None:
    # Half-scale samples produce an RMS of about 0.5.
    pcm = struct.pack("<480h", *([16384] * 480))
    assert 0.49 < _rms(pcm) < 0.51


def test_emit_complete_chunks_updates_last_rms() -> None:
    """Emitting a chunk must update _last_rms_value for the level meter."""
    mic = RtspMicrophone(
        url="rtsp://dummy:8554/live",
        output_sample_rate=16_000,
        channels=1,
        frame_ms=30,
    )
    # One frame_bytes (960B) of mid-level audio in the PCM buffer.
    mic._pcm_buffer.extend(struct.pack("<480h", *([16384] * 480)))
    mic._playback_start_monotonic = time.monotonic()

    mic._emit_complete_chunks()

    assert mic._last_rms_value > 0.4
    # The chunk was consumed from the buffer (enqueue drops it because no
    # loop is bound, but the buffer slice is removed before enqueue).
    assert len(mic._pcm_buffer) == 0


def test_diagnostics_reports_last_rms() -> None:
    mic = RtspMicrophone(url="rtsp://dummy:8554/live")
    mic._last_rms_value = 0.25
    diag = mic.diagnostics()
    assert "last_rms" in diag
    assert diag["last_rms"] == 0.25


def test_diagnostics_uses_honest_counter_names() -> None:
    """The old ``chunks_decoded`` key counted input frames and was misread
    as lost audio (``decoded=203, emitted=157`` is just resampling). The
    diagnostics must now report ``input_frames_decoded`` and
    ``chunks_produced`` instead -- never the ambiguous ``chunks_decoded``.
    """
    mic = RtspMicrophone(url="rtsp://dummy:8554/live")
    mic._input_frames_decoded = 203
    mic._chunks_produced = 157
    mic._chunks_emitted = 157
    mic._chunks_dropped = 0
    diag = mic.diagnostics()
    assert "chunks_decoded" not in diag
    assert diag["input_frames_decoded"] == 203
    assert diag["chunks_produced"] == 157
    assert diag["chunks_emitted"] == 157
    assert diag["chunks_dropped"] == 0


def test_runtime_stats_uniform_shape() -> None:
    """``runtime_stats`` mirrors UsbMicrophone so the conversation tick and
    /mic/diagnostics endpoint can treat every backend the same way."""
    mic = RtspMicrophone(url="rtsp://dummy:8554/live")
    mic._chunks_produced = 10
    mic._chunks_emitted = 8
    mic._chunks_dropped = 2
    stats = mic.runtime_stats()
    assert stats["type"] == "RtspMicrophone"
    assert stats["chunks_produced"] == 10
    assert stats["chunks_emitted"] == 8
    assert stats["chunks_dropped"] == 2
    assert "queue_size" in stats
    assert "queue_maxsize" in stats


def test_emit_complete_chunks_counts_produced() -> None:
    """Each emitted output chunk increments ``_chunks_produced`` so that
    ``chunks_produced`` is the honest counterpart to ``chunks_emitted`` +
    ``chunks_dropped``."""
    mic = RtspMicrophone(url="rtsp://dummy:8554/live", output_sample_rate=16_000, frame_ms=30)
    mic._pcm_buffer.extend(struct.pack("<480h", *([16384] * 480)))
    mic._playback_start_monotonic = time.monotonic()
    produced_before = mic._chunks_produced
    mic._emit_complete_chunks()
    assert mic._chunks_produced == produced_before + 1
