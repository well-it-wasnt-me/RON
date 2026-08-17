"""CircuitPython GC9A01 display driver.

This driver wraps Adafruit's displayio + adafruit-circuitpython-gc9a01a
stack. The Adafruit driver ships a verified-working init sequence for the
1.28" 240x240 GC9A01 panel (the same one the user confirmed works on the
Pi 5).

Why this exists
---------------

The original :class:`GC9A01Display` driver talks directly to the panel via
``spidev`` + ``lgpio``. We spent a long debugging session trying to drive the
panel from that driver and could not get it to display anything beyond the
POR (Power-On Reset) test pattern. The Adafruit driver works on the
exact same hardware using the same ``spidev`` library, the same pins, and
the same SPI bus.

Use this driver on Pi 5 when the raw driver fails to drive the panel.

Usage
-----

.. code-block:: bash

    # In .env:
    DESKBOT_DISPLAYS__BACKEND=circuitpython

The driver requires:

* ``adafruit-circuitpython-busdevice`` (provides ``FourWire``)
* ``adafruit-circuitpython-gc9a01a`` (the panel driver)
* ``adafruit-blinka`` (provides the ``board`` module on Linux)

These come from the ``hardware`` extra of ``deskbot``.
"""

from __future__ import annotations

import contextlib

from robot.errors import DisplayError
from robot.interfaces.display import EyeFrame
from robot.logging import get_logger

_log = get_logger("hardware.displays.circuitpython")


class CircuitPythonDisplay:
    """A :class:`Display` that wraps Adafruit's displayio + gc9a01a driver.

    The driver's ``show()`` method takes an :class:`EyeFrame` (RGB888
    bytes) and pushes it to a ``displayio.Bitmap`` that the Adafruit
    driver auto-refreshes on the panel.

    Internally:

    * We hold one ``displayio.Group`` containing one 16-bit
      ``displayio.Bitmap`` (65536 colors) + one ``ColorConverter``.
    * The bitmap stores RGB565 values directly (no palette quantisation).
    * ``show()`` copies the frame's pixels into the bitmap; the next
      ``display.refresh()`` call (triggered automatically by displayio)
      pushes the bitmap to the panel.

    Rotation: handled by the Adafruit driver (``rotation=0..3``).

    SPI: 1 MHz default (matches the verified-working reference script).
    """

    def __init__(
        self,
        width: int = 240,
        height: int = 240,
        *,
        rotation: int = 0,
        dc_pin: int | None = 25,
        reset_pin: int | None = 24,
        cs_pin: int = 8,  # SPI0 CE0 on Pi 5
        baudrate: int = 1_000_000,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be > 0")
        if rotation not in (0, 1, 2, 3):
            raise ValueError("rotation must be 0, 1, 2, or 3")
        if dc_pin is None:
            raise ValueError("dc_pin is required")

        self._width = width
        self._height = height
        self._rotation = rotation
        self._dc_pin = dc_pin
        self._reset_pin = reset_pin
        self._cs_pin = cs_pin
        self._baudrate = baudrate

        # Lazy import so the rest of the codebase runs on any machine.
        try:
            import board
            import displayio
            from adafruit_gc9a01a import GC9A01A
            from fourwire import FourWire
        except ImportError as exc:  # pragma: no cover - hardware-specific
            raise DisplayError(
                f"CircuitPython displayio dependencies not available: {exc!r}. "
                "Install with `uv pip install 'deskbot[hardware]'`."
            ) from exc

        # Release any previously-attached displays (CircuitPython is
        # single-display by default). Suppress exceptions because the
        # release call can fail on platforms where no display is bound.
        with contextlib.suppress(Exception):  # pragma: no cover - hardware-specific
            displayio.release_displays()

        # The Adafruit driver handles its own SPI setup; we just tell it
        # which DC/CS/RST pins to use and what baudrate. 1 MHz is what
        # the verified-working reference script uses.
        try:
            display_bus = FourWire(
                board.SPI(),
                command=getattr(board, f"D{dc_pin}"),
                chip_select=getattr(board, f"D{cs_pin}"),
                reset=(getattr(board, f"D{reset_pin}") if reset_pin is not None else None),
                baudrate=baudrate,
            )
            self._display = GC9A01A(
                display_bus,
                width=width,
                height=height,
                rotation=rotation,
            )
        except Exception as exc:
            raise DisplayError(
                f"could not initialise the Adafruit GC9A01A driver: {exc!r}. "
                "Verify SPI is enabled, the panel wiring matches docs/wiring.md, "
                "and adafruit-blinka detected the Pi 5 correctly (`blinka-detect`)."
            ) from exc

        # Allocate the framebuffer as a 16-bit (65536-color) bitmap so we
        # can store RGB565 values directly without a palette. A
        # ColorConverter with RGB565 input colorspace tells displayio how
        # to interpret the 16-bit pixel values when pushing to the panel.
        try:
            self._bitmap = displayio.Bitmap(width, height, 65536)
            colorspace = getattr(displayio, "Colorspace", None)
            if colorspace is not None:
                self._pixel_shader = displayio.ColorConverter(input_colorspace=colorspace.RGB565)
            else:
                # Older Blinka ports: ColorConverter defaults to RGB565.
                self._pixel_shader = displayio.ColorConverter()
            self._group = displayio.Group()
            self._tile_grid = displayio.TileGrid(
                self._bitmap,
                pixel_shader=self._pixel_shader,
                x=0,
                y=0,
            )
            self._group.append(self._tile_grid)
            self._display.root_group = self._group
        except Exception as exc:
            raise DisplayError(f"could not allocate the displayio framebuffer: {exc!r}") from exc

        # Test-only counter so test_app.py can verify frames are being pushed.
        self._frames_pushed: int = 0
        self._closed = False
        _log.info(
            "circuitpython.init",
            width=width,
            height=height,
            rotation=rotation,
            baudrate=baudrate,
            dc_pin=dc_pin,
            cs_pin=cs_pin,
            reset_pin=reset_pin,
        )

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
    def frames_pushed(self) -> int:
        """Test-only counter so test_app.py can verify frames are pushed."""
        return self._frames_pushed

    async def show(self, frame: EyeFrame) -> None:
        self._frames_pushed += 1
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
        # Convert every RGB888 pixel to RGB565 (5-bit R, 6-bit G, 5-bit B)
        # and write to the 16-bit displayio.Bitmap. Storing RGB565 natively
        # avoids the RGB332 quantisation loss of a 256-entry palette.
        from robot.hardware.displays.gc9a01 import rgb888_to_rgb565

        rgb565 = rgb888_to_rgb565(frame.pixels)
        w = self._width
        h = self._height
        n = w * h

        # Fast path: if the Blinka displayio Bitmap exposes a flat
        # ``_buffer`` bytearray of the right size, write the whole frame
        # in one slice assignment (C-level memcpy) instead of 57 600
        # individual ``__setitem__`` calls. Our rgb565 is big-endian; the
        # bitmap stores 16-bit values in native (little-endian on ARM)
        # order, so we byteswap first.
        buf = getattr(self._bitmap, "_buffer", None)
        if buf is not None and len(buf) == n * 2:
            import array

            arr = array.array("H")
            arr.frombytes(bytes(rgb565))  # native-endian interpretation
            arr.byteswap()  # big-endian rgb565 -> native little-endian
            buf[:] = arr.tobytes()
        else:
            # Fallback: per-pixel write (works on any displayio version).
            for y in range(h):
                row = rgb565[y * w * 2 : (y + 1) * w * 2]
                for x in range(w):
                    hi = row[x * 2]
                    lo = row[x * 2 + 1]
                    self._bitmap[x, y] = (hi << 8) | lo

        # Explicitly refresh the display. The Adafruit GC9A01A subclass of
        # BusDisplay has auto_refresh=True by default, but the CPython
        # port's displayio sometimes defers the refresh across the event
        # loop tick, causing "first frame not visible" symptoms. Calling
        # ``refresh()`` explicitly ensures the panel is updated on every
        # show() call.
        refresh = getattr(self._display, "refresh", None)
        if refresh is not None:
            try:
                refresh()
            except Exception:  # pragma: no cover - hardware-specific
                _log.exception("circuitpython.refresh_failed")

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
        try:
            import displayio

            displayio.release_displays()
        except Exception:  # pragma: no cover - hardware-specific
            _log.exception("circuitpython.shutdown_error")


__all__ = ["CircuitPythonDisplay"]
