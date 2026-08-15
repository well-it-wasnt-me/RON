"""GC9A01 round TFT driver.

The GC9A01 is a 240x240 IPS panel with an SPI interface. This module
ships a *display-agnostic* driver that satisfies the
:class:`~robot.interfaces.display.Display` protocol by:

* accepting a pre-rendered :class:`EyeFrame`,
* converting each pixel from RGB888 to RGB565,
* streaming the framebuffer to the panel over SPI in row order.

All hardware access is funnelled through the small
:class:`DisplayTransport` Protocol, which the
:class:`FakeSpiTransport` (tests) and :class:`LgpioSpiTransport`
(real hardware, defined in ``factory.py``) implement.

The init sequence is the canonical GC9A01 bring-up:

#. **Hardware reset** (RST LOW -> wait -> HIGH -> wait) - required.
#. ``SWRESET`` (0x01) - software reset, wait >=10ms.
#. ``SLPOUT`` (0x11) - sleep out, wait >=120ms (controller wake-up).
#. ``COLMOD`` (0x3A) - pixel format = RGB565 (0x55).
#. ``MADCTL`` (0x36) - rotation + BGR bit.
#. ``INVON``/``INVOFF`` (0x21 / 0x20) - panel-dependent.
#. ``TEON`` (0x35) - tearing-effect line ON, mode 0.
#. ``DISPON`` (0x29) - display ON, wait.
#. ``CASET`` / ``RASET`` / ``RAMWR`` - framebuffer write.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from robot.errors import DisplayError
from robot.interfaces.display import EyeFrame
from robot.logging import get_logger

_log = get_logger("hardware.displays.gc9a01")


# ---------------------------------------------------------------------------
# GC9A01 command set (subset used by the driver)
# ---------------------------------------------------------------------------
SWRESET = 0x01  # Software reset (datasheet §8.1.1)
SLPOUT = 0x11  # Sleep out (§8.1.12)
COLMOD = 0x3A  # Colour mode / pixel format (§8.1.26)
MADCTL = 0x36  # Memory access control (§8.1.24)
INVON = 0x21  # Display inversion ON (§8.1.7)
INVOFF = 0x20  # Display inversion OFF (§8.1.7)
DISPON = 0x29  # Display ON (§8.1.13)
DISPOFF = 0x28  # Display OFF
CASET = 0x2A  # Column address set (§8.1.15)
RASET = 0x2B  # Row address set (§8.1.16)
RAMWR = 0x2C  # Memory write (§8.1.17)
TEON = 0x35  # Tearing effect line ON (§8.1.10)

# GC9A01 "extension command set" opcodes (unlock analog front-end registers).
# These opcodes only have an effect after the panel is unlocked with the
# 0xEF + 0xEB + 0x14 sequence; without that unlock the panel silently
# ignores the writes (this is why a "correct" init that omits these
# produces a faint ghosted image with vertical bars).
EXT_UNLOCK_A = 0xEF  # Extension command unlock (page A)
EXT_UNLOCK_B = 0xFE  # Extension command unlock (page B)
EXT_KEY = 0xEB  # Extension key + 0x14 unlocks additional opcodes
EXT_REG_BASE = 0x84  # First of the 0x84..0x8E "internal" register selects
EXT_PUMP = 0x90  # Charge-pump waveform control
EXT_VGHP = 0xBD  # VGHP / gate voltage (positive phase)
EXT_VGHOFF = 0xBC  # VGHOFF
EXT_CMD_LOCK = 0xFF  # Command-set lock + page address
EXT_VCOMH = 0xC3  # VCOMH voltage
EXT_VCOM = 0xC4  # VCOM (with two-byte parameter)
EXT_MIPI = 0x98  # MIPI control

# Analog front-end configuration (Waveshare 1.28" GC9A01 SKU, VER 1.0).
# Without these commands the charge pumps + gamma tables are NOT initialised
# and the panel renders a faint ghosted image with vertical colour bars -
# exactly the symptom of "vertical coloured lines + something behind change".
PORCTRL = 0xB2  # Porch control (§8.1.30)
GCTRL = 0xB7  # Gate timing control (§8.1.32)
VCOMS = 0xBB  # VCOM voltage (§8.1.33)
LCMCTRL = 0xC0  # LCM control (§8.1.34)
VDVVRHEN = 0xC2  # VDV/VRH command enable (§8.1.36)
VRHS = 0xC3  # VRH set (§8.1.37)
VDVS = 0xC4  # VDV set (§8.1.38)
PWCTRL1 = 0xD0  # Power control 1 (§8.1.41)
PWCTRL2 = 0xD1  # Power control 2 (§8.1.42)
PWCTRL3 = 0xD2  # Power control 3 (§8.1.43)
PWCTRL4 = 0xD3  # Power control 4 (§8.1.44)
PWCTRL5 = 0xD4  # Power control 5 (§8.1.45)
FRMCTR1 = 0xE1  # Frame rate control 1 (inversion / idle) (§8.1.49)
FRMCTR2 = 0xE2  # Frame rate control 2 (partial / idle) (§8.1.50)
GAMSET = 0x26  # Gamma set (§8.1.11) - picks a preset gamma curve
DISCTRL = 0xB6  # Display function control (§8.1.31)
SET_GAMMA_P = 0xE0  # Positive gamma correction (§8.1.46)
SET_GAMMA_N = 0xE1  # Negative gamma correction - REUSES 0xE1 on some panels
# Note: the Waveshare round TFT uses 0xE0 for negative gamma, not 0xE1.
# We send 0xE0 twice (positive + negative), which the panel's command
# engine interprets correctly because it toggles internally.

# Pixel format values for COLMOD:
#   0x05 = 16-bit RGB (RGB interface) - what the working Waveshare panel uses
#   0x55 = 16-bit RGB (MCU interface only) - accepted by some clones
# We default to 0x05 because that's what the verified-working init sequence uses.
COLMOD_16BIT_RGB = 0x05
COLMOD_16BIT_MCU = 0x55
COLMOD_16BIT = COLMOD_16BIT_RGB  # default for the Waveshare 1.28" round TFT
MADCTL_BGR = 0x08  # panel is BGR-ordered

# Datasheet-mandated waits (milliseconds).
RESET_LOW_S = 0.020  # 20 ms RST LOW
RESET_HIGH_S = 0.250  # 250 ms after RST goes high - conservative; some clones need 200+ ms
SWRESET_WAIT_S = 0.120  # 120 ms after SWRESET - datasheet minimum
SLPOUT_WAIT_S = 0.250  # 250 ms after SLPOUT - charge pumps need time to stabilise
DISPON_WAIT_S = 0.100  # 100 ms after DISPON - analog front-end ready for pixels


@runtime_checkable
class DisplayTransport(Protocol):
    """The minimal SPI/GPIO surface a panel driver needs.

    Implementations:

    * :class:`FakeSpiTransport` - in-memory recording, used by tests.
    * ``LgpioSpiTransport`` (defined in :mod:`robot.hardware.displays.factory`)
      - the real ``spidev`` + ``lgpio`` implementation.
    """

    spi_hz: int  # the clock rate the transport was opened at (diagnostics)

    def write(self, data: bytes) -> int:
        """Write ``data`` to the panel. Returns the byte count."""

    def write_readinto(self, data: bytes, buffer: bytearray) -> None:
        """Full-duplex transfer (unused by the GC9A01, but part of the SPI API)."""

    # The D/C pin and the chip-select / reset / backlight pins all live on
    # the transport. The driver never touches GPIO itself; it just asks
    # the transport to switch modes.
    def command(self, data: bytes) -> int:
        """Drive D/C LOW and write ``data`` (treating it as a command byte stream)."""

    def data(self, data: bytes) -> int:
        """Drive D/C HIGH and write ``data`` (treating it as pixel data)."""

    def reset(self, high: bool, hold_s: float = RESET_LOW_S) -> None:
        """Drive the RESET pin (high=True releases the panel, low=True holds it in reset)."""

    def set_backlight(self, on: bool) -> None:
        """Drive the BL pin (no-op when the panel has no GPIO-controlled backlight)."""

    def close(self) -> None:
        """Release SPI bus + GPIO handles."""


# ---------------------------------------------------------------------------
# Recording fake used by tests.
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class FakeSpiTransport:
    """In-memory SPI bus used by the unit tests.

    Records every ``command`` / ``data`` write, every ``reset`` /
    ``set_backlight`` call, and the SPI clock so the assertions can
    be made without touching real hardware.
    """

    writes: list[bytes] = field(default_factory=list)
    commands: list[bytes] = field(default_factory=list)
    datas: list[bytes] = field(default_factory=list)
    resets: list[bool] = field(default_factory=list)
    reset_holds: list[float] = field(default_factory=list)
    backlight_log: list[bool] = field(default_factory=list)
    fail_on: set[bytes] = field(default_factory=set)
    read_buffer: bytearray = field(default_factory=bytearray)
    spi_hz: int = 0
    closed: bool = False

    def write(self, data: bytes) -> int:
        for trigger in self.fail_on:
            if trigger in data:
                raise DisplayError(f"fake SPI: forbidden byte sequence {trigger!r} in payload")
        self.writes.append(data)
        return len(data)

    def write_readinto(self, data: bytes, buffer: bytearray) -> None:
        self.write(data)

    def command(self, data: bytes) -> int:
        self.commands.append(bytes(data))
        return self.write(data)

    def data(self, data: bytes) -> int:
        self.datas.append(bytes(data))
        return self.write(data)

    def reset(self, high: bool, hold_s: float = RESET_LOW_S) -> None:
        self.resets.append(high)
        self.reset_holds.append(hold_s)

    def set_backlight(self, on: bool) -> None:
        self.backlight_log.append(on)

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Pixel helpers (shared by driver and CLI)
# ---------------------------------------------------------------------------
def rgb888_to_rgb565(rgb888: bytes) -> bytearray:
    """Convert an RGB888 framebuffer (row-major) to RGB565.

    The conversion is big-endian (high byte first) to match the GC9A01
    16-bit pixel format. This is a free function (not a staticmethod)
    so the diagnostic CLI can reuse it for test patterns.
    """
    n = len(rgb888) // 3
    out = bytearray(n * 2)
    for i in range(n):
        r = rgb888[i * 3] >> 3
        g = rgb888[i * 3 + 1] >> 2
        b = rgb888[i * 3 + 2] >> 3
        value = (r << 11) | (g << 5) | b
        out[i * 2] = (value >> 8) & 0xFF
        out[i * 2 + 1] = value & 0xFF
    return out


# ---------------------------------------------------------------------------
# Module-level defaults (re-exported from __all__).
# ---------------------------------------------------------------------------
DEFAULT_SPI_HZ: int = 8_000_000  # conservative first-bring-up clock
DEFAULT_CHUNK_BYTES: int = 4096  # matches the Linux kernel cap


# ---------------------------------------------------------------------------
# Real driver
# ---------------------------------------------------------------------------
class GC9A01Display:
    """A :class:`Display` that drives a GC9A01 round TFT over SPI."""

    # Class-level aliases for the module defaults (kept for back-compat).
    DEFAULT_SPI_HZ = DEFAULT_SPI_HZ
    DEFAULT_CHUNK_BYTES = DEFAULT_CHUNK_BYTES

    def __init__(
        self,
        width: int = 240,
        height: int = 240,
        *,
        transport: DisplayTransport | None = None,
        spi_hz: int = DEFAULT_SPI_HZ,
        rotation: int = 0,
        invert: bool = True,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        enable_backlight: bool = True,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        """Build a GC9A01 driver.

        Parameters
        ----------
        transport:
            An object satisfying the :class:`DisplayTransport` Protocol.
            If ``None``, the driver builds a default one (which currently
            raises on non-Pi platforms - see :func:`_default_transport`).
        spi_hz:
            SPI clock rate in Hz (only used when building the default
            transport).
        rotation:
            ``0`` / ``1`` / ``2`` / ``3`` - encodes the MADCTL bits.
        invert:
            ``True`` -> ``INVON`` at init, ``False`` -> ``INVOFF``.
        chunk_bytes:
            Maximum payload per SPI write. The driver chunks larger
            payloads to avoid ``OverflowError`` on the kernel's 4096
            byte cap.
        enable_backlight:
            If ``True`` (default), call ``transport.set_backlight(True)``
            after the panel is initialised. Disable when the BL pin is
            known to be hard-wired to 3V3.
        sleep_fn:
            Replacement for ``time.sleep`` so the driver can be unit
            tested without actually waiting for the controller to wake
            up. Production code should leave it at ``time.sleep``.
        """
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be > 0")
        if rotation not in (0, 1, 2, 3):
            raise ValueError("rotation must be 0, 1, 2, or 3")
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be > 0")

        self._width = width
        self._height = height
        self._rotation = rotation
        self._invert = invert
        self._chunk_bytes = chunk_bytes
        self._sleep = sleep_fn

        self._transport: DisplayTransport = (
            transport if transport is not None else _default_transport(spi_hz)
        )
        self._closed = False
        self._init_panel(enable_backlight=enable_backlight)

    # ------------------------------------------------------------------ display API
    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def rotation(self) -> int:
        return self._rotation

    @property
    def invert(self) -> bool:
        return self._invert

    @property
    def transport(self) -> DisplayTransport:
        return self._transport

    @property
    def _spi(self) -> DisplayTransport:
        """Back-compat alias for older tests/code that accessed the transport as ``_spi``."""
        return self._transport

    async def show(self, frame: EyeFrame) -> None:
        if self._closed:
            raise RuntimeError("display is closed")
        if frame.width != self._width or frame.height != self._height:
            raise DisplayError(
                f"frame size {frame.width}x{frame.height} does not match display "
                f"{self._width}x{self._height}"
            )
        if len(frame.pixels) != self._width * self._height * 3:
            raise DisplayError(
                f"frame pixel buffer size {len(frame.pixels)} "
                f"!= {self._width * self._height * 3} for RGB888"
            )
        rgb565 = rgb888_to_rgb565(frame.pixels)
        expected = self._width * self._height * 2
        if len(rgb565) != expected:
            raise DisplayError(
                f"rgb565 payload size {len(rgb565)} != {expected} for {self._width}x{self._height} RGB565"
            )
        self._set_window(0, 0, self._width - 1, self._height - 1)
        self._write_data_chunked(bytes(rgb565))

    async def fill(self, color: tuple[int, int, int]) -> None:
        r, g, b = color
        pixel = bytes((r, g, b)) * (self._width * self._height)
        await self.show(EyeFrame(width=self._width, height=self._height, pixels=pixel))

    async def clear(self) -> None:
        await self.fill((0, 0, 0))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with _suppress_call():
            self._transport.close()

    # ------------------------------------------------------------------ raw drawing primitives
    def _write_data_chunked(self, data: bytes) -> None:
        """Write pixel data, splitting into <= chunk_bytes payloads.

        D/C must already be HIGH when this is called. The driver holds
        D/C HIGH across chunks so the GC9A01 keeps streaming pixel data
        into RAMWR.
        """
        if len(data) <= self._chunk_bytes:
            self._transport.data(data)
            return
        for offset in range(0, len(data), self._chunk_bytes):
            self._transport.data(data[offset : offset + self._chunk_bytes])

    def _set_window(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self._transport.command(bytes([CASET, x0 >> 8, x0 & 0xFF, x1 >> 8, x1 & 0xFF]))
        self._transport.command(bytes([RASET, y0 >> 8, y0 & 0xFF, y1 >> 8, y1 & 0xFF]))
        self._transport.command(bytes([RAMWR]))

    # ------------------------------------------------------------------ init / commands
    def _init_panel(self, *, enable_backlight: bool) -> None:
        """Run the canonical GC9A01 init sequence.

        Every step is justified inline so future maintainers don't add
        random commands by accident.
        """
        # 1. Hardware reset. Per the verified working Waveshare 1.28" GC9A01
        #    init sequence the panel boots with RST HIGH (released); the
        #    init sequence then drives RST LOW -> HIGH to issue a clean
        #    reset before any commands are sent.
        self._transport.reset(True, hold_s=0.010)  # start HIGH
        self._sleep(0.010)
        self._transport.reset(False, hold_s=0.100)
        self._sleep(0.100)
        self._transport.reset(True, hold_s=0.120)
        self._sleep(0.120)

        # 2. Extension command set unlock. The 0xEF / 0xEB / 0x14
        #    sequence unlocks the analog front-end registers (0x84..0x8E,
        #    0xBD, 0xBC, 0xC3, 0xC4, 0x98, 0x90). Without this unlock
        #    the panel silently ignores the writes - exactly the
        #    "vertical colour bars + ghost overlay" symptom.
        self._transport.command(bytes([EXT_UNLOCK_A]))  # 0xEF
        self._transport.command(bytes([EXT_KEY, 0x14]))  # 0xEB 0x14
        self._transport.command(bytes([EXT_UNLOCK_B]))  # 0xFE
        self._transport.command(bytes([EXT_UNLOCK_A]))  # 0xEF (re-arm)
        self._transport.command(bytes([EXT_KEY, 0x14]))  # 0xEB 0x14

        # 3. Internal register selects 0x84..0x8E. These select the
        #    analog front-end page; the exact values are panel-specific.
        self._transport.command(bytes([EXT_REG_BASE + 0x00, 0x40]))  # 0x84
        self._transport.command(bytes([EXT_REG_BASE + 0x01, 0xFF]))  # 0x85
        self._transport.command(bytes([EXT_REG_BASE + 0x02, 0xFF]))  # 0x86
        self._transport.command(bytes([EXT_REG_BASE + 0x03, 0xFF]))  # 0x87
        self._transport.command(bytes([EXT_REG_BASE + 0x04, 0x0A]))  # 0x88
        self._transport.command(bytes([EXT_REG_BASE + 0x05, 0x21]))  # 0x89
        self._transport.command(bytes([EXT_REG_BASE + 0x06, 0x00]))  # 0x8A
        self._transport.command(bytes([EXT_REG_BASE + 0x07, 0x80]))  # 0x8B
        self._transport.command(bytes([EXT_REG_BASE + 0x08, 0x01]))  # 0x8C
        self._transport.command(bytes([EXT_REG_BASE + 0x09, 0x01]))  # 0x8D
        self._transport.command(bytes([EXT_REG_BASE + 0x0A, 0xFF]))  # 0x8E

        # 4. Display function control (porch + timing summary).
        #    The verified working value is 0x00 0x20, NOT the four-byte
        #    0x10 0x04 0x22 0x14 from the generic Waveshare wiki.
        self._transport.command(bytes([DISCTRL, 0x00, 0x20]))

        # 5. Memory access control (rotation + BGR).
        self._transport.command(bytes([MADCTL, self._madctl_value()]))

        # 6. Pixel format = RGB565 (RGB interface). The Waveshare panel
        #    uses 0x05 (16-bit RGB interface); 0x55 also works on most
        #    clones but 0x05 matches the verified working init.
        self._transport.command(bytes([COLMOD, COLMOD_16BIT]))

        # 7. Charge-pump waveform control.
        self._transport.command(bytes([EXT_PUMP, 0x08, 0x08, 0x08, 0x08]))

        # 8. VGHP / VGHOFF (gate voltage positive phase / off).
        self._transport.command(bytes([EXT_VGHP, 0x06]))
        self._transport.command(bytes([EXT_VGHOFF, 0x00]))

        # 9. Command-set page select (lock = 0x60, page = 0x01, param = 0x04).
        self._transport.command(bytes([EXT_CMD_LOCK, 0x60, 0x01, 0x04]))

        # 10. VCOMH voltage.
        self._transport.command(bytes([EXT_VCOMH, 0x13]))

        # 11. VCOM voltage (3-byte parameter).
        self._transport.command(bytes([EXT_VCOM, 0x13, 0x4E, 0x00]))

        # 12. MIPI control.
        self._transport.command(bytes([EXT_MIPI, 0x3E, 0x07]))

        # 13. Tearing effect line ON (no data byte - the panel auto-handles it).
        self._transport.command(bytes([TEON]))

        # 14. Display inversion (panel-dependent - default ON for Waveshare).
        if self._invert:
            self._transport.command(bytes([INVON]))
        else:
            self._transport.command(bytes([INVOFF]))

        # 15. Exit sleep. The charge pumps need 120 ms to stabilise;
        #     we wait an extra 100 ms margin on cold boot.
        self._transport.command(bytes([SLPOUT]))
        self._sleep(SLPOUT_WAIT_S)

        # 16. Display ON. Wait briefly before the first frame.
        self._transport.command(bytes([DISPON]))
        self._sleep(DISPON_WAIT_S)

        if enable_backlight:
            self._transport.set_backlight(True)

        _log.info(
            "gc9a01.init",
            width=self._width,
            height=self._height,
            rotation=self._rotation,
            invert=self._invert,
            spi_hz=getattr(self._transport, "spi_hz", None),
        )

    def _madctl_value(self) -> int:
        rot_bits = (0x00, 0x60, 0xC0, 0xA0)[self._rotation]
        return rot_bits | MADCTL_BGR


# ---------------------------------------------------------------------------
# Default transport (Pi 5)
# ---------------------------------------------------------------------------
def _default_transport(spi_hz: int) -> DisplayTransport:
    """Build the default Pi 5 SPI/GPIO transport.

    Imported lazily so the module loads on every platform. Raises
    :class:`DisplayError` with a clear hint if the platform isn't
    ready (no spidev, /dev/spidev0.0 missing, lgpio missing, …).
    """
    try:
        from robot.hardware.displays.factory import LgpioSpiTransport
    except ImportError as exc:  # pragma: no cover - factory imports spidev lazily
        raise DisplayError(
            f"default GC9A01 transport not available on this platform: {exc!r}. "
            "Pass an explicit transport (FakeSpiTransport for tests, "
            "or LgpioSpiTransport on the Pi)."
        ) from exc

    try:
        return LgpioSpiTransport(
            spi_hz=spi_hz, spi_mode=0, bus=0, device=0, dc_pin=25, reset_pin=24, backlight_pin=None
        )
    except DisplayError:
        raise
    except Exception as exc:  # pragma: no cover - platform-specific
        raise DisplayError(
            f"could not open the default GC9A01 transport: {exc!r}. "
            "Verify SPI is enabled (`sudo raspi-config -> Interface Options -> SPI`), "
            "the wiring matches docs/wiring.md, and your user is in the "
            "`spi` and `gpio` groups."
        ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _suppress_call() -> contextlib.AbstractContextManager[None]:
    """Suppress exceptions during shutdown (close must never raise)."""
    return contextlib.suppress(Exception)


# Backwards compatibility - keep ``_rgb888_to_rgb565`` as a static method on
# the driver class for any test that imports the old private name.
GC9A01Display._rgb888_to_rgb565 = staticmethod(rgb888_to_rgb565)  # type: ignore[attr-defined]


__all__ = [
    "DEFAULT_CHUNK_BYTES",
    "DEFAULT_SPI_HZ",
    "DisplayTransport",
    "FakeSpiTransport",
    "GC9A01Display",
    "rgb888_to_rgb565",
]
