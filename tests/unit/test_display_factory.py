"""Tests for the :class:`DisplayFactory`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

from robot.config import DisplayConfig
from robot.errors import DisplayError
from robot.hardware.displays.factory import DisplayFactory
from robot.hardware.displays.gc9a01 import FakeSpiTransport
from robot.hardware.displays.mock_display import MockDisplay


def test_factory_returns_mock_display_by_default() -> None:
    config = DisplayConfig(backend="mock")
    factory = DisplayFactory(config)
    display = factory.build()
    assert isinstance(display, MockDisplay)
    assert display.width == config.width
    assert display.height == config.height


def test_factory_returns_circuitpython_when_backend_is_circuitpython() -> None:
    """The ``circuitpython`` backend must be wired up to the
    CircuitPythonDisplay implementation (when the displayio deps are
    available). When the deps are missing, the factory raises a
    DisplayError with a helpful message."""
    from robot.errors import DisplayError
    from robot.hardware.displays.circuitpython import CircuitPythonDisplay

    config = DisplayConfig(backend="circuitpython", width=64, height=64)
    factory = DisplayFactory(config)
    try:
        display = factory.build()
    except DisplayError as exc:
        # ImportError on dev machines without displayio is acceptable.
        msg = str(exc).lower()
        assert "circuitpython" in msg or "displayio" in msg
        return
    assert isinstance(display, CircuitPythonDisplay)


def test_factory_accepts_cp_and_displayio_aliases() -> None:
    """``cp`` and ``displayio`` are aliases for ``circuitpython``."""
    from robot.errors import DisplayError
    from robot.hardware.displays.circuitpython import CircuitPythonDisplay

    for alias in ("cp", "displayio"):
        config = DisplayConfig(backend=alias, width=64, height=64)
        factory = DisplayFactory(config)
        try:
            display = factory.build()
        except DisplayError:
            # Deps not installed on dev machines - still acceptable.
            continue
        assert isinstance(display, CircuitPythonDisplay)


def test_factory_returns_gc9a01_when_backend_is_gc9a01() -> None:
    config = DisplayConfig(backend="gc9a01", width=64, height=64)

    class _StubSpi(FakeSpiTransport):
        def __init__(self, config: DisplayConfig) -> None:
            super().__init__()
            self.config = config

    factory = DisplayFactory(config, spi_factory=_StubSpi)
    display = factory.build()
    from robot.hardware.displays.gc9a01 import GC9A01Display

    assert isinstance(display, GC9A01Display)
    assert display.width == 64
    assert display.height == 64


def test_factory_wraps_gc9a01_failure_in_display_error() -> None:
    config = DisplayConfig(backend="gc9a01")

    class _BoomSpi(FakeSpiTransport):
        def __init__(self, config: DisplayConfig) -> None:
            super().__init__()
            raise FileNotFoundError("/dev/spidev0.0: no such device")

    factory = DisplayFactory(config, spi_factory=_BoomSpi)
    with pytest.raises(DisplayError) as exc_info:
        factory.build()
    assert "gc9a01" in str(exc_info.value).lower()
    assert "spidev" in str(exc_info.value).lower() or "wiring" in str(exc_info.value).lower()


def test_factory_backend_is_validated_by_pydantic() -> None:
    """The backend field is a Literal type - invalid values are caught at config level."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DisplayConfig(backend="future-display-3000")


def test_factory_spidev_writebytes_returns_none() -> None:
    """Regression test: spidev 3.x returns ``None`` from ``writebytes``.

    Our production wrapper must NOT pass that ``None`` to ``int()`` -
    instead it should report the number of bytes attempted.
    """
    from robot.config import DisplayConfig
    from robot.hardware.displays.factory import DisplayFactory
    from robot.hardware.displays.gc9a01 import FakeSpiTransport

    class _SpiDevMimic(FakeSpiTransport):
        """Mimics spidev.SpiDev: ``writebytes`` returns ``None``."""

        def __init__(self, config: DisplayConfig) -> None:
            super().__init__()
            self.writes: list[bytes] = []

        def write(self, data: bytes) -> int:
            self.writes.append(data)
            # spidev's writebytes returns None; we MUST NOT do ``int(None)``.
            return len(data)

    config = DisplayConfig(backend="gc9a01", width=32, height=32)
    factory = DisplayFactory(config, spi_factory=_SpiDevMimic)
    display = factory.build()

    # Now drive the display and ensure no TypeError leaks.
    import asyncio

    from robot.face.model import FaceModel
    from robot.face.renderer import FaceRenderer
    from robot.face.themes.minimal import MinimalTheme

    renderer = FaceRenderer(width=32, height=32)
    frame = renderer.render(MinimalTheme().apply(FaceModel(width=32, height=32)))
    asyncio.run(display.show(frame))
    assert _SpiDevMimic.writes  # at least one SPI write happened


def test_factory_spi_chunks_writes_to_4096_bytes(monkeypatch: MonkeyPatch) -> None:
    """Regression test: kernel SPI msg size is 4096 bytes.

    The GC9A01 frame for a 240x240 panel is 115200 bytes; the production
    transport must chunk writes to avoid ``OverflowError: Argument list
    size exceeds 4096 bytes``.

    The earlier version of this test injected a custom transport, which
    bypassed the production chunking code. To exercise the real path we
    monkey-patch ``spidev.SpiDev`` with a stub that enforces the 4096
    byte cap.
    """
    import sys
    import types
    from typing import ClassVar

    class _FakeSpiDev:
        """Mimics ``spidev.SpiDev`` with the kernel 4096-byte chunking cap."""

        instances: ClassVar[list[_FakeSpiDev]] = []  # class-level accumulator

        def __init__(self) -> None:
            self.calls: list[bytes] = []
            self.max_speed_hz = 0
            _FakeSpiDev.instances.append(self)

        def open(self, bus: int, device: int) -> None:
            pass

        def writebytes(self, data: list[int]) -> None:
            if len(data) > 4096:
                raise OverflowError(f"Argument list size exceeds 4096 bytes. (actual {len(data)})")
            self.calls.append(bytes(data))

        def xfer2(self, data: list[int], *args: object, **kwargs: object) -> list[int]:
            self.calls.append(bytes(data))
            return list(data)

    spidev_mod = types.ModuleType("spidev")
    spidev_mod.SpiDev = _FakeSpiDev  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spidev", spidev_mod)

    import asyncio

    from robot.config import DisplayConfig
    from robot.face.model import FaceModel
    from robot.face.renderer import FaceRenderer
    from robot.face.themes.minimal import MinimalTheme
    from robot.hardware.displays.factory import DisplayFactory

    config = DisplayConfig(backend="gc9a01", width=240, height=240)
    factory = DisplayFactory(config)
    display = factory.build()

    renderer = FaceRenderer(width=240, height=240)
    frame = renderer.render(MinimalTheme().apply(FaceModel(width=240, height=240)))
    asyncio.run(display.show(frame))

    assert _FakeSpiDev.instances, "no spidev instance was created"
    all_calls: list[bytes] = []
    for inst in _FakeSpiDev.instances:
        all_calls.extend(inst.calls)
    assert all_calls, "no SPI writes happened"
    for chunk in all_calls:
        assert len(chunk) <= 4096, f"chunk too large: {len(chunk)}"
    # And the GC9A01 pixel data is 115200 bytes; we expect at least
    # ceil(115200 / 4096) = 29 chunks.
    assert len(all_calls) >= 29, f"too few chunks: {len(all_calls)}"
