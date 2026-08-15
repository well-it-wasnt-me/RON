"""Display factory.

Selects a :class:`Display` implementation at application boot based on
``config.displays.backend``:

* ``"mock"`` - :class:`MockDisplay`, an in-memory display used by
  tests and headless dev.
* ``"gc9a01"`` - :class:`GC9A01Display`, the real driver for the
  240x240 round GC9A01 TFT over SPI.

The factory fails fast with a helpful hint if the configured backend
is unavailable in the current environment.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING, cast  # noqa: F401

if TYPE_CHECKING:
    from robot.hardware.displays.gc9a01 import DisplayTransport

from robot.config import DisplayConfig
from robot.errors import DisplayError
from robot.hardware.displays.gc9a01 import (
    DisplayTransport,
    FakeSpiTransport,
    GC9A01Display,
)
from robot.hardware.displays.mock_display import MockDisplay
from robot.interfaces.display import Display
from robot.logging import get_logger

_log = get_logger("hardware.displays.factory")


SpiTransportFactory = Callable[[DisplayConfig], object]


class DisplayFactory:
    """Build the configured :class:`Display` from a :class:`DisplayConfig`."""

    def __init__(
        self,
        config: DisplayConfig,
        *,
        spi_factory: SpiTransportFactory | None = None,
    ) -> None:
        self._config = config
        self._spi_factory = spi_factory

    def build(self) -> Display:
        backend = self._config.backend
        _log.info("display.backend_selected", backend=backend)
        match backend:
            case "mock":
                return MockDisplay(width=self._config.width, height=self._config.height)
            case "gc9a01":
                return self._build_gc9a01()
            case "circuitpython" | "cp" | "displayio":
                return self._build_circuitpython()
            case _:
                raise DisplayError(
                    f"unknown display backend {backend!r}; "
                    "expected 'mock', 'gc9a01', or 'circuitpython'"
                )

    def _build_gc9a01(self) -> Display:
        """Build a :class:`GC9A01Display` using the configured SPI bus.

        All failures (missing ``spidev``, missing /dev/spidev device,
        permission errors, missing ``dc_pin``) are wrapped in a
        :class:`DisplayError` so the app fails fast with a helpful hint.
        """
        # Validate cross-field invariants up-front. We don't want a
        # half-initialised SPI bus followed by a confusing exception.
        self._config.validate_pins()
        if self._config.dc_pin is None:
            raise DisplayError(
                "DESKBOT_DISPLAYS__DC_PIN is required for the gc9a01 backend "
                "(BCM GPIO number for the GC9A01's D/C line)."
            )

        transport: DisplayTransport
        try:
            if self._spi_factory is not None:
                transport = self._spi_factory(self._config)  # type: ignore[assignment]
            else:
                transport = LgpioSpiTransport.from_config(self._config)
            return GC9A01Display(
                width=self._config.width,
                height=self._config.height,
                rotation=self._config.rotation,
                invert=self._config.invert,
                chunk_bytes=self._config.chunk_bytes,
                transport=transport,
            )
        except DisplayError:
            raise
        except Exception as exc:
            raise DisplayError(
                f"could not initialise the GC9A01 display: {exc!r}. "
                "Verify SPI is enabled (sudo raspi-config -> Interface "
                "Options -> SPI), check the wiring in docs/wiring.md, and "
                "ensure your user is in the `spi` and `gpio` groups."
            ) from exc

    def _build_circuitpython(self) -> Display:
        """Build a Display backed by Adafruit's displayio + gc9a01a driver.

        This is the recommended path on Pi 5 - it works out of the box
        and produces a verified-working init sequence (the same one the
        user verified manually).

        Returns a :class:`CircuitPythonDisplay` that wraps a
        ``displayio.Group`` so the rest of the app can keep pushing
        :class:`EyeFrame` instances without knowing the driver changed.
        """
        try:
            from robot.hardware.displays.circuitpython import (
                CircuitPythonDisplay,
            )
        except ImportError as exc:
            raise DisplayError(
                f"CircuitPython displayio driver not available: {exc!r}. "
                "Install with `uv pip install 'deskbot[hardware]'` on a Pi."
            ) from exc
        return CircuitPythonDisplay(
            width=self._config.width,
            height=self._config.height,
            rotation=self._config.rotation,
            dc_pin=self._config.dc_pin,
            reset_pin=self._config.reset_pin,
            cs_pin=8,  # SPI0 CE0 (Pi 5 hardware default)
            baudrate=self._config.spi_hz,
        )


# ---------------------------------------------------------------------------
# Real hardware transport (spidev + lgpio).
# ---------------------------------------------------------------------------
class LgpioSpiTransport(FakeSpiTransport):
    """Real Pi 5 SPI transport that drives the GC9A01 over ``spidev`` + ``lgpio``.

    The transport handles:

    * opening ``/dev/spidev<bus>.<device>`` with the configured
      ``spi_hz`` and ``spi_mode``,
    * toggling the GC9A01's **D/C** GPIO before each SPI transfer
      (command vs. data),
    * pulsing the GC9A01's **RESET** GPIO during hardware reset,
    * optionally driving a **BL** GPIO to switch the backlight on,
    * chunking every SPI write to ``chunk_bytes`` (the Linux kernel
      caps each transaction at 4096 bytes by default; we never exceed
      whatever the user configures).

    The transport deliberately lives in ``factory.py`` rather than
    ``gc9a01.py`` so that the driver module can be imported on every
    platform (test runner, CI, dev laptop) without pulling in
    ``spidev`` or ``lgpio``.
    """

    #: Default kernel SPI cap. Bumped with ``spidev.bufsiz=``.
    DEFAULT_CHUNK_BYTES: int = 4096

    @classmethod
    def from_config(cls, config: DisplayConfig) -> LgpioSpiTransport:
        """Build a transport from a :class:`DisplayConfig`."""
        return cls(
            spi_hz=config.effective_spi_hz(),
            spi_mode=config.spi_mode,
            bus=config.bus,
            device=config.device,
            dc_pin=config.dc_pin,
            reset_pin=config.reset_pin,
            backlight_pin=config.backlight_pin,
            chunk_bytes=config.chunk_bytes,
        )

    def __init__(
        self,
        *,
        spi_hz: int,
        spi_mode: int = 0,
        bus: int = 0,
        device: int = 0,
        dc_pin: int | None,
        reset_pin: int | None,
        backlight_pin: int | None,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> None:
        super().__init__()
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be > 0")
        self.spi_hz = spi_hz
        self._spi_mode = spi_mode
        self._chunk_bytes = chunk_bytes
        self._bus = bus
        self._device = device
        self._dc_pin = dc_pin
        self._reset_pin = reset_pin
        self._backlight_pin = backlight_pin

        # Lazy imports so the rest of the codebase runs without hardware.
        try:
            import spidev
        except Exception as exc:
            raise DisplayError(
                f"spidev is not available: {exc!r}. Install it with "
                "`uv pip install spidev` or run with "
                "DESKBOT_DISPLAYS__BACKEND=mock."
            ) from exc

        self._dev = spidev.SpiDev()
        try:
            self._dev.open(bus, device)
        except (FileNotFoundError, OSError, PermissionError) as exc:
            raise DisplayError(
                f"could not open /dev/spidev{bus}.{device}: {exc!r}. "
                "Enable SPI with `sudo raspi-config -> Interface Options -> SPI` "
                "and ensure your user is in the `spi` group."
            ) from exc
        self._dev.max_speed_hz = spi_hz
        self._dev.mode = spi_mode  # CPOL/CPHA explicit - do not rely on defaults

        self._lgpio = _LgpioHandle(dc_pin=dc_pin, reset_pin=reset_pin, backlight_pin=backlight_pin)

    # ------------------------------------------------------------------ command / data / reset / backlight
    def command(self, data: bytes) -> int:
        self._lgpio.set_dc(False)
        return self._write_chunked(data)

    def data(self, data: bytes) -> int:
        self._lgpio.set_dc(True)
        return self._write_chunked(data)

    def reset(self, high: bool, hold_s: float = 0.020) -> None:
        self._lgpio.set_reset(high)
        self.resets.append(high)
        self.reset_holds.append(hold_s)
        # The caller is responsible for sleeping for the required
        # ``hold_s`` after each call. We don't sleep here so the
        # diagnostic CLI can choose the timing it wants to test.

    def set_backlight(self, on: bool) -> None:
        self._lgpio.set_backlight(on)
        self.backlight_log.append(on)

    def close(self) -> None:
        if self.closed:
            return
        try:
            self._dev.close()
        except Exception:  # pragma: no cover - hardware-specific
            _log.exception("lgpio.transport.close_error")
        try:
            self._lgpio.close()
        except Exception:  # pragma: no cover - hardware-specific
            _log.exception("lgpio.transport.gpio_close_error")
        self.closed = True

    # ------------------------------------------------------------------ internals
    def _write_chunked(self, data: bytes) -> int:
        """Write ``data`` to ``/dev/spidev`` in <= ``chunk_bytes`` slices.

        D/C must already be set to the correct level (command or data)
        by the caller. Chunks preserve D/C: a command byte sequence is
        still a command on every chunk, and pixel data stays data.

        spidev's ``writebytes`` returns ``None`` on every modern version
        - we return the total number of bytes we attempted to send.
        """
        if not data:
            return 0
        written = 0
        if len(data) <= self._chunk_bytes:
            self._dev.writebytes(list(data))
            return len(data)
        for offset in range(0, len(data), self._chunk_bytes):
            chunk = data[offset : offset + self._chunk_bytes]
            self._dev.writebytes(list(chunk))
            written += len(chunk)
        return written


class _LgpioHandle:
    """Wrapper around the optional ``lgpio`` GPIO handle for D/C / RST / BL.

    If any pin is ``None``, or lgpio isn't available, the corresponding
    method is a no-op (with a one-shot warning at construction so the
    user knows the panel may be misbehaving).
    """

    def __init__(
        self,
        *,
        dc_pin: int | None,
        reset_pin: int | None,
        backlight_pin: int | None,
    ) -> None:
        self._dc_pin = dc_pin
        self._reset_pin = reset_pin
        self._backlight_pin = backlight_pin
        self._chip = None
        self._lgpio = None
        self._warned: set[str] = set()
        self._dc_high = False
        self._reset_high = False
        self._backlight_high = False

        if not any((dc_pin, reset_pin, backlight_pin)):
            return

        try:
            import lgpio  # type: ignore[import-not-found]
        except Exception as exc:
            _log.warning("lgpio.unavailable", error=str(exc))
            return

        self._lgpio = lgpio
        try:
            self._chip = lgpio.gpiochip_open(0)
        except Exception as exc:
            _log.warning("lgpio.chip_open_failed", error=str(exc))
            return

        for pin, name in (
            (reset_pin, "RST"),
            (dc_pin, "DC"),
            (backlight_pin, "BL"),
        ):
            if pin is None:
                continue
            try:
                lgpio.gpio_claim_output(self._chip, 0, pin, 0)
                _log.info("lgpio.claim_output", pin=pin, name=name)
            except Exception as exc:
                _log.warning("lgpio.claim_failed", pin=pin, name=name, error=str(exc))

    # ------------------------------------------------------------------ pin drives
    def set_dc(self, high: bool) -> None:
        self._dc_high = high
        self._drive(self._dc_pin, "DC", high)

    def set_reset(self, high: bool) -> None:
        self._reset_high = high
        self._drive(self._reset_pin, "RST", high)

    def set_backlight(self, on: bool) -> None:
        self._backlight_high = on
        self._drive(self._backlight_pin, "BL", on)

    def close(self) -> None:
        if self._lgpio is None or self._chip is None:
            return
        for pin in (self._dc_pin, self._reset_pin, self._backlight_pin):
            if pin is None:
                continue
        with contextlib.suppress(Exception):  # pragma: no cover - hardware-specific
            for pin in (self._dc_pin, self._reset_pin, self._backlight_pin):
                if pin is None:
                    continue
                self._lgpio.gpio_free(self._chip, pin)
            self._lgpio.gpiochip_close(self._chip)

    # ------------------------------------------------------------------ helpers
    def _drive(self, pin: int | None, name: str, high: bool) -> None:
        if pin is None:
            key = f"{name}_unset"
            if key not in self._warned:
                _log.warning("lgpio.pin_unset", name=name)
                self._warned.add(key)
            return
        if self._lgpio is None or self._chip is None:
            key = f"{name}_no_chip"
            if key not in self._warned:
                _log.warning("lgpio.no_chip", name=name)
                self._warned.add(key)
            return
        try:
            self._lgpio.gpio_write(self._chip, pin, 1 if high else 0)
            # The lgpio and spidev file descriptors are independent in the
            # kernel. The SPI DMA controller can start clocking out the next
            # byte BEFORE the GPIO write has actually settled at the pin,
            # which on some GC9A01 clones corrupts the very first byte of
            # every transfer. A 10 µs delay is enough for the GPIO to latch
            # and is invisible at 30 FPS frame rates.
            import time as _time

            _time.sleep(10e-6)
        except Exception as exc:  # pragma: no cover - hardware-specific
            _log.warning("lgpio.write_failed", name=name, pin=pin, error=str(exc))


__all__ = ["DisplayFactory", "LgpioSpiTransport", "SpiTransportFactory"]
