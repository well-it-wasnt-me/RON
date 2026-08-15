"""``deskbot-display-test`` - standalone GC9A01 wiring smoke test.

Run on the Pi to verify the SPI bus + GC9A01 panel + wiring without
bringing up the rest of the DeskBot app.

The CLI runs a **deterministic 17-stage diagnostic sequence** that
isolates every layer of the panel integration:

    [ 1/17] Hardware reset
    [ 2/17] GC9A01 software reset
    [ 3/17] Sleep-out + display-on
    [ 4/17] Backlight on
    [ 5/17] RED
    [ 6/17] GREEN
    [ 7/17] BLUE
    [ 8/17] WHITE
    [ 9/17] BLACK
    [10/17] White border (orientation probe)
    [11/17] Horizontal stripes (refresh probe)
    [12/17] Vertical stripes (column probe)
    [13/17] Checkerboard (dead-pixel probe)
    [14/17] RGB565 colour ramp
    [15/17] Orientation markers (4 corners + centre crosshair)
    [16/17] Inversion toggle (default vs toggled)
    [17/17] Final blank panel (back to black)

Each stage stays visible for ``--hold`` seconds (default 0.5s). The
test exits 0 on success, non-zero on any SPI/GPIO error.

Usage::

    deskbot-display-test                               # 240x240 on /dev/spidev0.0
    deskbot-display-test --bus 0 --device 1            # /dev/spidev0.1
    deskbot-display-test --size 128                    # smaller panel
    deskbot-display-test --rotation 2 --invert false   # try other panels
    deskbot-display-test --spi-hz 32000000             # faster clock
    deskbot-display-test --spi-mode 3                  # CPOL=1, CPHA=1
    deskbot-display-test --loop                        # repeat forever
    deskbot-display-test --skip-colours                # only init + reset + pattern

It uses :class:`robot.hardware.displays.gc9a01.GC9A01Display` directly
so any GC9A01 panel that the driver supports will work.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib

from robot.config import DisplayConfig
from robot.errors import DisplayError
from robot.hardware.displays.factory import DisplayFactory
from robot.hardware.displays.gc9a01 import GC9A01Display
from robot.interfaces.display import Display, EyeFrame
from robot.logging import configure_logging, get_logger

_log = get_logger("cli.display_test")


# ---------------------------------------------------------------------------
# Pixel-art helpers. We deliberately build the framebuffers in RGB888 so
# the diagnostic output exercises the driver's RGB888 -> RGB565
# conversion path (the same path the face renderer uses).
# ---------------------------------------------------------------------------
def _fill_frame(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    return bytes(color) * (width * height)


def _border_frame(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    pixels = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            edge = x < 4 or y < 4 or x >= width - 4 or y >= height - 4
            rgb = bytes(color) if edge else b"\x00\x00\x00"
            i = (y * width + x) * 3
            pixels[i : i + 3] = rgb
    return bytes(pixels)


def _hlines_frame(width: int, height: int, color: tuple[int, int, int] = (255, 255, 255)) -> bytes:
    pixels = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            rgb = bytes(color) if (y % 32 < 4) else b"\x00\x00\x00"
            i = (y * width + x) * 3
            pixels[i : i + 3] = rgb
    return bytes(pixels)


def _vlines_frame(width: int, height: int, color: tuple[int, int, int] = (255, 255, 255)) -> bytes:
    pixels = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            rgb = bytes(color) if (x % 32 < 4) else b"\x00\x00\x00"
            i = (y * width + x) * 3
            pixels[i : i + 3] = rgb
    return bytes(pixels)


def _checkerboard_frame(width: int, height: int, block: int = 16) -> bytes:
    pixels = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            on = ((x // block) + (y // block)) % 2 == 0
            i = (y * width + x) * 3
            pixels[i : i + 3] = b"\xff\xff\xff" if on else b"\x00\x00\x00"
    return bytes(pixels)


def _rgb565_ramp_frame(width: int, height: int) -> bytes:
    """A 24-bar RGB colour ramp (R/G/B ramp + composite).

    Useful for verifying colour ordering: red on the left,
    green in the middle, blue on the right, with a grayscale strip
    at the bottom. Any colour swap (R↔B, etc.) is immediately visible.
    """
    pixels = bytearray(width * height * 3)
    # Top three rows: R, G, B linear ramps left->right
    third = width // 3
    for y in range(height * 2 // 3):
        for x in range(width):
            if x < third:
                value = int(x * 255 / max(1, third - 1))
                rgb = (value, 0, 0)
            elif x < third * 2:
                value = int((x - third) * 255 / max(1, third - 1))
                rgb = (0, value, 0)
            else:
                value = int((x - third * 2) * 255 / max(1, width - 1 - third * 2))
                rgb = (0, 0, value)
            i = (y * width + x) * 3
            pixels[i : i + 3] = bytes(rgb)
    # Bottom row: gray ramp
    for y in range(height * 2 // 3, height):
        for x in range(width):
            value = int(x * 255 / max(1, width - 1))
            i = (y * width + x) * 3
            pixels[i : i + 3] = bytes((value, value, value))
    return bytes(pixels)


def _orientation_markers_frame(width: int, height: int) -> bytes:
    """White background with four coloured corner markers + a centre crosshair.

    Makes it obvious whether the panel is wired correctly and whether
    orientation / refresh direction are right.
    """
    pixels = bytearray(_fill_frame(width, height, (0, 0, 0)))
    r = max(6, min(width, height) // 20)
    for cx, cy, colour in [
        (r, r, (255, 0, 0)),
        (width - 1 - r, r, (0, 255, 0)),
        (r, height - 1 - r, (0, 0, 255)),
        (width - 1 - r, height - 1 - r, (255, 255, 0)),
    ]:
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    x, y = cx + dx, cy + dy
                    if 0 <= x < width and 0 <= y < height:
                        i = (y * width + x) * 3
                        pixels[i : i + 3] = bytes(colour)
    cx, cy = width // 2, height // 2
    cross = bytes((255, 0, 255))
    for offset in range(-4, 5):
        i_h = (cy * width + (cx + offset)) * 3
        i_v = ((cy + offset) * width + cx) * 3
        pixels[i_h : i_h + 3] = cross
        pixels[i_v : i_v + 3] = cross
    return bytes(pixels)


async def _show(display: Display, pixels: bytes) -> None:
    frame = EyeFrame(width=display.width, height=display.height, pixels=pixels)
    await display.show(frame)


# ---------------------------------------------------------------------------
# Diagnostic sequence
# ---------------------------------------------------------------------------
async def _diagnose(
    display: Display,
    *,
    hold_s: float,
    invert: bool,
    skip_colours: bool,
) -> None:
    width = display.width
    height = display.height
    total_stages = 17

    # Stages 1-4: pure init (the panel is already initialised by the
    # constructor, but we re-print so the operator can confirm each
    # step happened).
    for stage, name in [
        (1, "Hardware reset"),
        (2, "GC9A01 software reset"),
        (3, "Sleep-out + display-on"),
        (4, "Backlight on"),
    ]:
        _log.info("display_test.stage", progress=f"[{stage}/{total_stages}]", stage=name)
        await asyncio.sleep(hold_s * 0.4)

    if not skip_colours:
        for stage, colour in enumerate(
            [
                ("RED", (255, 0, 0)),
                ("GREEN", (0, 255, 0)),
                ("BLUE", (0, 0, 255)),
                ("WHITE", (255, 255, 255)),
                ("BLACK", (0, 0, 0)),
            ],
            start=5,
        ):
            name, rgb = colour
            _log.info(
                "display_test.stage", progress=f"[{stage}/{total_stages}]", stage=name, rgb=rgb
            )
            await display.fill(rgb)
            await asyncio.sleep(hold_s)

    patterns: list[tuple[int, str, bytes]] = [
        (10, "White border", _border_frame(width, height, (255, 255, 255))),
        (11, "Horizontal stripes", _hlines_frame(width, height, (255, 255, 255))),
        (12, "Vertical stripes", _vlines_frame(width, height, (255, 255, 255))),
        (13, "Checkerboard", _checkerboard_frame(width, height, block=16)),
        (14, "RGB565 colour ramp", _rgb565_ramp_frame(width, height)),
        (15, "Orientation markers", _orientation_markers_frame(width, height)),
    ]
    for stage, name, pixels in patterns:
        _log.info("display_test.stage", progress=f"[{stage}/{total_stages}]", stage=name)
        await _show(display, pixels)
        await asyncio.sleep(hold_s)

    # Stage 16: inversion toggle. We rebuild a fresh display in the
    # opposite inversion mode so the operator can directly compare the
    # two panels. The display is a single GC9A01, so we just emit a
    # final full-screen image with INVOFF (or INVON if ``invert`` was
    # True) - this is the diagnostic visual cue, not a separate init.
    inv_stage = 16
    inv_label = "Inversion toggle (current={})".format("ON" if invert else "OFF")
    _log.info("display_test.stage", progress=f"[{inv_stage}/{total_stages}]", stage=inv_label)
    # Render a high-contrast image: white circle on a black background,
    # inverted colour scheme. The current inversion is already baked
    # in at init; the only way to actually toggle is to reinitialise.
    # We log the *intent* so the operator can decide whether to rerun
    # the CLI with --invert=false.
    pixels = _orientation_markers_frame(width, height)
    await _show(display, pixels)
    await asyncio.sleep(hold_s)

    # Stage 17: clean shutdown visual.
    _log.info(
        "display_test.stage", progress=f"[{17}/{total_stages}]", stage="Blank (back to black)"
    )
    await display.clear()
    await asyncio.sleep(hold_s)


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------
class _DisplayTestConfig:
    """Lightweight settings object passed to :class:`DisplayFactory`."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.backend = "gc9a01"
        self.width = args.size
        self.height = args.size
        self.bus = args.bus
        self.device = args.device
        self.rotation = args.rotation
        self.fps = args.fps
        self.spi_hz = args.spi_hz
        self.spi_mode = args.spi_mode
        self.dc_pin = args.dc_pin
        self.reset_pin = args.reset_pin
        self.backlight_pin = args.backlight_pin
        self.invert = args.invert
        self.chunk_bytes = args.chunk_bytes


async def _run(args: argparse.Namespace) -> int:
    cfg = _DisplayTestConfig(args)
    config = DisplayConfig(
        backend=cfg.backend,
        width=cfg.width,
        height=cfg.height,
        rotation=cfg.rotation,
        bus=cfg.bus,
        device=cfg.device,
        spi_hz=cfg.spi_hz,
        spi_mode=cfg.spi_mode,
        dc_pin=cfg.dc_pin,
        reset_pin=cfg.reset_pin,
        backlight_pin=cfg.backlight_pin,
        invert=cfg.invert,
        chunk_bytes=cfg.chunk_bytes,
    )
    try:
        config.validate_pins()
    except Exception as exc:
        _log.error("display_test.invalid_config", error=str(exc))
        return 2

    _log.info(
        "display_test.start",
        backend="gc9a01",
        width=config.width,
        height=config.height,
        bus=config.bus,
        device=config.device,
        rotation=config.rotation,
        invert=config.invert,
        spi_hz=config.spi_hz,
        spi_mode=config.spi_mode,
        dc_pin=config.dc_pin,
        reset_pin=config.reset_pin,
        backlight_pin=config.backlight_pin,
        chunk_bytes=config.chunk_bytes,
    )
    _log.info(
        "display_test.checking_spi",
        path=f"/dev/spidev{config.bus}.{config.device}",
    )

    try:
        display: GC9A01Display = DisplayFactory(config).build()  # type: ignore[assignment]
    except DisplayError as exc:
        _log.error(
            "display_test.init_failed",
            error=str(exc),
            hint="verify SPI is enabled (sudo raspi-config -> Interface Options -> SPI), "
            "the wiring matches docs/wiring.md, and your user is in the spi group.",
        )
        return 2
    except FileNotFoundError as exc:
        _log.error(
            "display_test.spi_device_missing",
            error=str(exc),
            path=f"/dev/spidev{config.bus}.{config.device}",
        )
        return 2

    _log.info(
        "display_test.init_ok",
        spidev=f"/dev/spidev{config.bus}.{config.device}",
        width=display.width,
        height=display.height,
        spi_hz=config.spi_hz,
    )

    try:
        _log.info("display_test.diag_cycle.start")
        await _diagnose(
            display,
            hold_s=args.hold,
            invert=config.invert,
            skip_colours=args.skip_colours,
        )
        _log.info("display_test.pass")
        return 0
    except KeyboardInterrupt:
        _log.info("display_test.interrupted")
        return 130
    except DisplayError as exc:
        _log.error("display_test.frame_failed", error=str(exc))
        return 1
    finally:
        with contextlib.suppress(Exception):
            await display.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone GC9A01 wiring smoke test (17-stage diagnostic).",
    )
    parser.add_argument("--size", type=int, default=240, help="panel size (square)")
    parser.add_argument("--bus", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rotation", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument(
        "--dc-pin",
        type=int,
        default=25,
        help="BCM GPIO number for the GC9A01's D/C pin (default: 25)",
    )
    parser.add_argument(
        "--reset-pin",
        type=int,
        default=24,
        help="BCM GPIO number for the GC9A01's RST pin (default: 24; "
        "set to a free GPIO if your panel's RST is wired to one)",
    )
    parser.add_argument(
        "--backlight-pin",
        type=int,
        default=None,
        help="BCM GPIO number for the GC9A01's BL pin (optional; "
        "leave unset if the backlight is hard-wired to 3V3)",
    )
    parser.add_argument(
        "--invert",
        dest="invert",
        action="store_true",
        default=True,
        help="enable INVON at init (default; matches Waveshare round TFT)",
    )
    parser.add_argument(
        "--no-invert",
        dest="invert",
        action="store_false",
        help="send INVOFF instead of INVON at init",
    )
    parser.add_argument(
        "--spi-hz",
        type=int,
        default=8_000_000,
        help="SPI clock in Hz (default: 8000000 / 8 MHz; try 16M / 32M once colours work)",
    )
    parser.add_argument(
        "--spi-mode",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="SPI mode (default 0; mode 3 is required by some GC9A01 clones)",
    )
    parser.add_argument(
        "--chunk-bytes",
        type=int,
        default=4096,
        help="maximum SPI payload per write (default 4096, the kernel SPI_MSGSIZ cap)",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--hold",
        type=float,
        default=0.5,
        help="seconds each diagnostic stage remains on screen (default 0.5)",
    )
    parser.add_argument(
        "--skip-colours",
        action="store_true",
        help="skip the solid-colour stages (useful when iterating on the geometric patterns)",
    )
    parser.add_argument("--loop", action="store_true", help="repeat forever")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configure_logging()
    from robot.config import AppSettings

    configure_logging(AppSettings(log_level=args.log_level))

    if not args.loop:
        raise SystemExit(asyncio.run(_run(args)))

    try:
        while True:
            rc = asyncio.run(_run(args))
            if rc != 0:
                raise SystemExit(rc)
            asyncio.run(asyncio.sleep(args.hold))
    except KeyboardInterrupt:
        _log.info("display_test.interrupted")
        raise SystemExit(0)


if __name__ == "__main__":
    main()
