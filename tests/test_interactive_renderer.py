"""Unit tests for the interactive CLI renderer module."""

from __future__ import annotations

import math

from robot.cli.interactive.renderer import (
    config_summary,
    frame_to_braille,
    frame_to_half_blocks,
    level_bar,
    servo_bar,
    servo_dashboard,
    state_line,
)

# ---------------------------------------------------------------------------
# frame_to_braille
# ---------------------------------------------------------------------------


class TestFrameToBraille:
    """Tests for the braille-art renderer."""

    def _make_pixels(self, width: int, height: int, value: int = 0) -> bytes:
        """Create a solid-colour RGB buffer."""
        return bytes([value, value, value]) * (width * height)

    def test_all_black_returns_blank(self) -> None:
        """An all-black frame (lum < threshold) produces all braille-empty chars."""
        pixels = self._make_pixels(10, 10, value=0)
        lines = frame_to_braille(pixels, 10, 10, threshold=80)
        assert len(lines) == math.ceil(10 / 4)  # 3 rows
        for line in lines:
            # All chars should be U+2800 (braille empty = no dots)
            assert all(c == "\u2800" for c in line), f"Expected all-blank, got: {line!r}"

    def test_all_white_returns_filled(self) -> None:
        """An all-white frame (lum > threshold) produces braille dots."""
        pixels = self._make_pixels(10, 10, value=255)
        lines = frame_to_braille(pixels, 10, 10, threshold=80)
        assert len(lines) == math.ceil(10 / 4)  # 3 rows
        for line in lines:
            # All chars should have all dots set = U+28FF
            for c in line:
                assert c != "\u2800", "Expected a filled braille char, got blank"

    def test_dimensions_match(self) -> None:
        """Output dimensions are correct for non-square frames."""
        w, h = 240, 320
        pixels = self._make_pixels(w, h, value=0)
        lines = frame_to_braille(pixels, w, h, threshold=80)
        expected_cols = math.ceil(w / 2)
        expected_rows = math.ceil(h / 4)
        assert len(lines) == expected_rows
        for line in lines:
            assert len(line) == expected_cols

    def test_terminal_size_constraints(self) -> None:
        """term_cols and term_rows cap the output size."""
        pixels = self._make_pixels(240, 320, value=0)
        lines = frame_to_braille(pixels, 240, 320, threshold=80, term_cols=40, term_rows=10)
        assert len(lines) <= 10
        for line in lines:
            assert len(line) <= 40

    def test_empty_frame(self) -> None:
        """Zero-size frame returns empty list."""
        assert frame_to_braille(b"", 0, 0) == []

    def test_partial_braille(self) -> None:
        """A frame with mixed luminance produces both blank and filled chars."""
        w, h = 4, 4
        pixels = bytearray(w * h * 3)
        # Top-left pixel bright, rest dark.
        pixels[0] = 255
        pixels[1] = 255
        pixels[2] = 255
        lines = frame_to_braille(bytes(pixels), w, h, threshold=80)
        assert len(lines) == 1
        # The first char should have at least the dot for pixel (0,0)
        assert lines[0][0] != "\u2800"


# ---------------------------------------------------------------------------
# frame_to_half_blocks
# ---------------------------------------------------------------------------


class TestFrameToHalfBlocks:
    """Tests for the half-block fallback renderer."""

    def test_all_black(self) -> None:
        pixels = bytes(3 * 10)  # 10 black pixels
        lines = frame_to_half_blocks(pixels, 10, 1)
        assert len(lines) == 1
        assert lines[0] == " " * 10

    def test_all_white(self) -> None:
        pixels = bytes([255, 255, 255] * 10)  # 10 white pixels
        lines = frame_to_half_blocks(pixels, 10, 1)
        assert len(lines) == 1
        assert lines[0] == "█" * 10

    def test_empty(self) -> None:
        assert frame_to_half_blocks(b"", 0, 0) == []


# ---------------------------------------------------------------------------
# servo_bar
# ---------------------------------------------------------------------------


class TestServoBar:
    """Tests for the servo gauge renderer."""

    def test_center_position(self) -> None:
        bar = servo_bar("pan", 90.0, 0.0, 180.0, width=20)
        assert "90.0" in bar
        assert "█" in bar
        assert "░" in bar

    def test_min_position(self) -> None:
        bar = servo_bar("pan", 0.0, 0.0, 180.0, width=20)
        assert "0.0" in bar

    def test_max_position(self) -> None:
        bar = servo_bar("pan", 180.0, 0.0, 180.0, width=20)
        assert "180.0" in bar

    def test_custom_label(self) -> None:
        bar = servo_bar("tilt", 45.0, min_angle=0.0, max_angle=180.0)
        assert "Tilt" in bar

    def test_unknown_servo_name(self) -> None:
        bar = servo_bar("custom_servo", 90.0)
        # Should use first 5 chars of the name as label
        assert "custo" in bar


# ---------------------------------------------------------------------------
# servo_dashboard
# ---------------------------------------------------------------------------


class TestServoDashboard:
    """Tests for the multi-servo dashboard renderer."""

    def test_four_servos(self) -> None:
        angles = {"pan": 90.0, "tilt": 90.0, "left_arm": 90.0, "right_arm": 90.0}
        lines = servo_dashboard(angles)
        assert len(lines) == 4

    def test_with_calibration(self) -> None:
        angles = {"pan": 45.0, "tilt": 90.0, "left_arm": 120.0, "right_arm": 60.0}
        calibration = {
            "pan": (30.0, 150.0),
            "tilt": (45.0, 135.0),
            "left_arm": (20.0, 160.0),
            "right_arm": (20.0, 160.0),
        }
        lines = servo_dashboard(angles, calibration)
        assert len(lines) == 4
        for line in lines:
            assert "°" in line

    def test_missing_servo_defaults(self) -> None:
        angles = {"pan": 90.0}
        lines = servo_dashboard(angles)
        assert len(lines) == 4


# ---------------------------------------------------------------------------
# level_bar
# ---------------------------------------------------------------------------


class TestLevelBar:
    """Tests for the audio level bar renderer."""

    def test_zero_level(self) -> None:
        bar = level_bar(0.0, width=20)
        assert "0.000" in bar
        assert "░" in bar

    def test_max_level(self) -> None:
        bar = level_bar(1.0, width=20)
        assert "1.000" in bar
        assert "█" in bar

    def test_peak_marker(self) -> None:
        bar = level_bar(0.5, peak=0.8, width=20)
        assert "|" in bar

    def test_clamping(self) -> None:
        bar = level_bar(2.0, width=20)  # Over max - bar is full but value displayed as-is
        assert "█" in bar  # Bar should be fully filled
        assert "2.000" in bar  # Value is displayed as-is (not clamped)


# ---------------------------------------------------------------------------
# state_line
# ---------------------------------------------------------------------------


class TestStateLine:
    """Tests for the state + emotion summary line."""

    def test_state_only(self) -> None:
        line = state_line("idle")
        assert "idle" in line

    def test_state_with_emotion(self) -> None:
        line = state_line("curious", "happy", intensity=0.8)
        assert "curious" in line
        assert "happy" in line
        assert "0.80" in line

    def test_unknown_state(self) -> None:
        line = state_line("unknown_state")
        assert "unknown_state" in line


# ---------------------------------------------------------------------------
# config_summary
# ---------------------------------------------------------------------------


class TestConfigSummary:
    """Tests for the config summary renderer."""

    def test_masks_api_keys(self) -> None:
        d = {"llm": {"api_key": "sk-secret123456", "model": "gpt-4"}}
        lines = config_summary(d)  # type: ignore[arg-type]
        flat = " ".join(lines)
        assert "sk-secret123456" not in flat
        assert "****" in flat
        assert "gpt-4" in flat

    def test_masks_passwords(self) -> None:
        d = {"mqtt": {"password": "hunter2", "host": "localhost"}}
        lines = config_summary(d)  # type: ignore[arg-type]
        flat = " ".join(lines)
        assert "hunter2" not in flat
        assert "localhost" in flat

    def test_limits_output(self) -> None:
        d = {f"key_{i}": i for i in range(50)}
        lines = config_summary(d)  # type: ignore[arg-type]
        assert len(lines) <= 30

    def test_nested_keys(self) -> None:
        d = {"llm": {"model": "gpt-4", "api_key": "secret"}}
        lines = config_summary(d)  # type: ignore[arg-type]
        flat = " ".join(lines)
        assert "llm.model" in flat
        assert "gpt-4" in flat
