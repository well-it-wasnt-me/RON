"""ASCII-art renderers for the DeskBot interactive TUI.

This module converts pixel buffers and numeric state into terminal-friendly
ASCII/Unicode representations:

* :func:`frame_to_braille` - convert an RGB pixel buffer into braille-art.
* :func:`frame_to_half_blocks` - convert an RGB pixel buffer into
  half-block characters (simpler, coarser fallback).
* :func:`servo_bar` - render a single servo angle as a horizontal gauge.
* :func:`servo_dashboard` - render all four servos as a dashboard.
* :func:`level_bar` - render an audio level meter from RMS values.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Braille-art renderer
# ---------------------------------------------------------------------------

# Unicode braille dots: each character encodes a 2x4 grid of on/off pixels.
# Bit positions in the braille code point (U+2800 + offset):
#
#   (0,0)=0x01  (1,0)=0x08
#   (0,1)=0x02  (1,1)=0x10
#   (0,2)=0x04  (1,2)=0x20
#   (0,3)=0x40  (1,3)=0x80

_BRAILLE_MAP: list[tuple[int, int]] = [
    (0, 0),  # bit 0 -> row 0, col 0
    (0, 1),  # bit 1 -> row 1, col 0
    (0, 2),  # bit 2 -> row 2, col 0
    (0, 3),  # bit 3 -> row 3, col 0
    (1, 0),  # bit 4 -> row 0, col 1
    (1, 1),  # bit 5 -> row 1, col 1
    (1, 2),  # bit 6 -> row 2, col 1
    (1, 3),  # bit 7 -> row 3, col 1
]


def frame_to_braille(
    pixels: bytes,
    width: int,
    height: int,
    threshold: int = 80,
    term_cols: int | None = None,
    term_rows: int | None = None,
) -> list[str]:
    """Convert an RGB888 pixel buffer into braille-art lines.

    Parameters
    ----------
    pixels:
        Row-major RGB888 buffer of length ``width * height * 3``.
    width, height:
        Pixel dimensions of the source frame.
    threshold:
        Luminance value (0-255) above which a dot is considered "on".
        The luminance of each pixel is computed as
        ``0.299*R + 0.587*G + 0.114*B``.
    term_cols, term_rows:
        Maximum terminal size constraint.  If *None*, the output
        fits the native image resolution (each braille char covers
        2 horizontal x 4 vertical pixels).

    Returns
    -------
    list[str]
        One string per output row (no trailing newlines).
    """
    if width <= 0 or height <= 0:
        return []

    # Each braille char covers 2 cols x 4 rows of source pixels.
    bw, bh = 2, 4
    out_cols = math.ceil(width / bw)
    out_rows = math.ceil(height / bh)

    # Downsample if terminal constraints are given.
    if term_cols is not None and term_cols > 0:
        out_cols = min(out_cols, term_cols)
    if term_rows is not None and term_rows > 0:
        out_rows = min(out_rows, term_rows)

    lines: list[str] = []
    for br in range(out_rows):
        row_chars: list[str] = []
        for bc in range(out_cols):
            code = 0x2800
            for bit_idx, (dx, dy) in enumerate(_BRAILLE_MAP):
                px = bc * bw + dx
                py = br * bh + dy
                if px < width and py < height:
                    idx = (py * width + px) * 3
                    if idx + 2 < len(pixels):
                        r, g, b = pixels[idx], pixels[idx + 1], pixels[idx + 2]
                        lum = 0.299 * r + 0.587 * g + 0.114 * b
                        if lum > threshold:
                            code |= 1 << bit_idx
            row_chars.append(chr(code))
        lines.append("".join(row_chars))
    return lines


# ---------------------------------------------------------------------------
# Half-block fallback renderer
# ---------------------------------------------------------------------------

_HALF_BLOCK_CHARS = " ░▒▓█"


def frame_to_half_blocks(
    pixels: bytes,
    width: int,
    height: int,
    term_cols: int | None = None,
    term_rows: int | None = None,
) -> list[str]:
    """Convert an RGB888 pixel buffer into half-block characters.

    This is a coarser fallback when braille rendering is unavailable.
    Each character cell represents one source pixel; luminance maps to
    one of five density characters.

    Parameters
    ----------
    pixels:
        Row-major RGB888 buffer.
    width, height:
        Source pixel dimensions.
    term_cols, term_rows:
        Optional terminal size constraints.

    Returns
    -------
    list[str]
        One string per output row.
    """
    if width <= 0 or height <= 0:
        return []

    out_cols = min(width, term_cols or width)
    out_rows = min(height, term_rows or height)

    # Downsample coordinates.
    x_scale = width / out_cols if out_cols < width else 1.0
    y_scale = height / out_rows if out_rows < height else 1.0

    lines: list[str] = []
    for orow in range(out_rows):
        row_chars: list[str] = []
        sy = min(int(orow * y_scale), height - 1)
        for ocol in range(out_cols):
            sx = min(int(ocol * x_scale), width - 1)
            idx = (sy * width + sx) * 3
            if idx + 2 < len(pixels):
                r, g, b = pixels[idx], pixels[idx + 1], pixels[idx + 2]
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                char_idx = min(int(lum / 256 * len(_HALF_BLOCK_CHARS)), len(_HALF_BLOCK_CHARS) - 1)
                row_chars.append(_HALF_BLOCK_CHARS[char_idx])
            else:
                row_chars.append(" ")
        lines.append("".join(row_chars))
    return lines


# ---------------------------------------------------------------------------
# Servo dashboard
# ---------------------------------------------------------------------------

_SERVO_NAMES: list[str] = ["pan", "tilt", "left_arm", "right_arm"]
_SERVO_LABELS: dict[str, str] = {
    "pan": "Pan ",
    "tilt": "Tilt",
    "left_arm": "L.Arm",
    "right_arm": "R.Arm",
}

_BAR_WIDTH = 20


def servo_bar(
    name: str,
    angle: float,
    min_angle: float = 0.0,
    max_angle: float = 180.0,
    width: int = _BAR_WIDTH,
) -> str:
    """Render a single servo angle as a horizontal gauge.

    Parameters
    ----------
    name:
        Servo name (used for labelling).
    angle:
        Current angle in degrees.
    min_angle, max_angle:
        Calibrated range for the servo.
    width:
        Width of the gauge in characters.

    Returns
    -------
    str
        A single-line gauge like ``"Pan  90.0° [████████░░░░░░░░░░░░]"``.
    """
    label = _SERVO_LABELS.get(name, name[:5])
    clamped = max(min_angle, min(max_angle, angle))
    fraction = (clamped - min_angle) / (max_angle - min_angle) if max_angle > min_angle else 0.0
    filled = round(fraction * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{label} {angle:6.1f}° [{bar}]"


def servo_dashboard(
    angles: dict[str, float],
    calibration: dict[str, tuple[float, float]] | None = None,
) -> list[str]:
    """Render all four servos as a dashboard.

    Parameters
    ----------
    angles:
        Mapping of servo name -> current angle.
    calibration:
        Optional mapping of servo name -> (min_angle, max_angle).

    Returns
    -------
    list[str]
        Four lines, one per servo.
    """
    lines: list[str] = []
    for name in _SERVO_NAMES:
        angle = angles.get(name, 90.0)
        if calibration and name in calibration:
            mn, mx = calibration[name]
        else:
            mn, mx = 0.0, 180.0
        lines.append(servo_bar(name, angle, mn, mx))
    return lines


# ---------------------------------------------------------------------------
# Audio level meter
# ---------------------------------------------------------------------------

_LEVEL_BLOCKS = " ▏▎▍▌▋▊▉█"


def level_bar(
    rms: float,
    peak: float | None = None,
    width: int = 30,
    max_val: float = 1.0,
) -> str:
    """Render an audio level meter.

    Parameters
    ----------
    rms:
        Root-mean-square audio level (0.0-1.0 typically).
    peak:
        Optional peak level for a peak indicator.
    width:
        Width of the bar in characters.
    max_val:
        Value that maps to a full bar.

    Returns
    -------
    str
        A bar like ``"▊▉████████████░░░░░░░░░░░░░░░░░░░ |0.42|"``.
    """
    fraction = min(max(rms / max_val, 0.0), 1.0) if max_val > 0 else 0.0
    filled = int(fraction * width)
    remainder_frac = (fraction * width) - filled
    bar_chars: list[str] = []

    for i in range(width):
        if i < filled:
            bar_chars.append("█")
        elif i == filled:
            # Sub-character precision for the boundary.
            idx = min(int(remainder_frac * len(_LEVEL_BLOCKS)), len(_LEVEL_BLOCKS) - 1)
            bar_chars.append(_LEVEL_BLOCKS[idx])
        else:
            bar_chars.append("░")

    if peak is not None:
        peak_pos = min(int(peak / max_val * width), width - 1) if max_val > 0 else 0
        # We'll add a peak marker as a pipe character overlay.
        if 0 <= peak_pos < width:
            bar_chars[peak_pos] = "|"

    return f"{''.join(bar_chars)} {rms:.3f}"


# ---------------------------------------------------------------------------
# State / emotion display helpers
# ---------------------------------------------------------------------------

_STATE_ICONS: dict[str, str] = {
    "boot": "⏻",
    "idle": "😊",
    "curious": "🤔",
    "listening": "👂",
    "thinking": "💭",
    "speaking": "🗣",
    "sleeping": "😴",
    "error": "⚠️ ",
}

_EMOTION_ICONS: dict[str, str] = {
    "neutral": "😐",
    "happy": "😊",
    "curious": "🤔",
    "thinking": "💭",
    "sleepy": "😴",
    "embarrassed": "😳",
    "excited": "🤩",
    "sad": "😢",
    "surprised": "😲",
    "angry": "😠",
}


def state_line(state: str, emotion: str = "", intensity: float = 1.0) -> str:
    """Render a one-line state + emotion summary.

    Example::

        "😊 idle  ·  😊 happy (1.00)"
    """
    icon = _STATE_ICONS.get(state, "?")
    parts = [f"{icon} {state}"]
    if emotion:
        e_icon = _EMOTION_ICONS.get(emotion, "")
        parts.append(f"{e_icon} {emotion} ({intensity:.2f})")
    return "  ·  ".join(parts)


def config_summary(settings_dict: dict[str, object]) -> list[str]:
    """Render a compact config summary for the status panel.

    Masks any key containing ``api_key`` or ``password``.
    """
    lines: list[str] = []
    _masked_keys = {"api_key", "password"}

    def _flatten(d: dict[str, object], prefix: str = "") -> None:
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(v, key)
            else:
                display_val = "****" if any(m in k for m in _masked_keys) else v
                lines.append(f"  {key} = {display_val}")

    _flatten(settings_dict)
    return lines[:30]  # Cap at 30 lines


def frame_to_braille_cropped(
    pixels: bytes,
    width: int,
    height: int,
    threshold: int = 80,
    term_cols: int = 60,
    term_rows: int = 28,
    padding: int = 4,
) -> list[str]:
    """Convert an RGB888 pixel buffer into braille-art, auto-cropped to content.

    Unlike :func:`frame_to_braille`, this function first detects the bounding
    box of non-empty pixels and crops the image to that region before
    converting to braille.  This ensures that small faces centred on a large
    canvas are rendered at a reasonable size in the terminal rather than
    appearing as a tiny sliver of content surrounded by blank space.

    Parameters
    ----------
    pixels:
        Row-major RGB888 buffer of length ``width * height * 3``.
    width, height:
        Pixel dimensions of the source frame.
    threshold:
        Luminance value (0-255) above which a pixel is considered "on".
    term_cols, term_rows:
        Maximum terminal size constraint for the braille output.
    padding:
        Extra pixels of padding around the detected content region.

    Returns
    -------
    list[str]
        One string per output row (no trailing newlines).
    """
    if width <= 0 or height <= 0:
        return []

    # --- 1. Detect content bounding box -----------------------------------
    min_x, max_x = width, 0
    min_y, max_y = height, 0
    for y in range(height):
        row_start = y * width * 3
        for x in range(width):
            idx = row_start + x * 3
            r, g, b = pixels[idx], pixels[idx + 1], pixels[idx + 2]
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            if lum > threshold:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

    # If no content was found, fall back to rendering the full frame.
    if min_x > max_x or min_y > max_y:
        return frame_to_braille(pixels, width, height, threshold, term_cols, term_rows)

    # --- 2. Add padding and snap to braille grid --------------------------
    bw, bh = 2, 4  # braille character grid size
    min_x = max(0, ((min_x - padding) // bw) * bw)
    min_y = max(0, ((min_y - padding) // bh) * bh)
    max_x = min(width, ((max_x + padding + bw) // bw) * bw)
    max_y = min(height, ((max_y + padding + bh) // bh) * bh)

    crop_w = max_x - min_x
    crop_h = max_y - min_y

    if crop_w <= 0 or crop_h <= 0:
        return frame_to_braille(pixels, width, height, threshold, term_cols, term_rows)

    # --- 3. Extract cropped pixel buffer ----------------------------------
    cropped = bytearray(crop_w * crop_h * 3)
    for y in range(crop_h):
        src_start = ((min_y + y) * width + min_x) * 3
        dst_start = y * crop_w * 3
        src_end = src_start + crop_w * 3
        cropped[dst_start : dst_start + crop_w * 3] = pixels[src_start:src_end]

    # --- 4. Render the cropped region --------------------------------------
    return frame_to_braille(bytes(cropped), crop_w, crop_h, threshold, term_cols, term_rows)


__all__ = [
    "config_summary",
    "frame_to_braille",
    "frame_to_braille_cropped",
    "frame_to_half_blocks",
    "level_bar",
    "servo_bar",
    "servo_dashboard",
    "state_line",
]
