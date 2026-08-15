"""Tests for the D/C pin handling in the GC9A01 production transport."""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

from robot.config import DisplayConfig
from robot.hardware.displays.factory import DisplayFactory


class _MockSpiDev:
    def __init__(self) -> None:
        self.calls: list[bytes] = []
        self.dc_calls: list[int] = []
        self.max_speed_hz = 0
        self.mode = 0

    def open(self, bus: int, device: int) -> None:
        pass

    def close(self) -> None:
        pass

    @property
    def max_speed_hz_set(self) -> int:
        return 0

    @max_speed_hz_set.setter
    def max_speed_hz_set(self, value: int) -> None:
        pass

    def writebytes(self, data: list[int]) -> None:
        self.calls.append(bytes(data))

    def xfer2(self, data: list[int], *args: object, **kwargs: object) -> list[int]:
        self.calls.append(bytes(data))
        return list(data)


def _install_fake_spidev(monkeypatch: MonkeyPatch) -> _MockSpiDev:
    spidev = types.ModuleType("spidev")
    mock = _MockSpiDev()
    spidev.SpiDev = lambda: mock  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spidev", spidev)
    return mock


def _install_fake_lgpio(monkeypatch: MonkeyPatch) -> list[int]:
    """Return a list that records every (pin, value) drive in order."""
    writes: list[int] = []
    fake_lgpio = types.ModuleType("lgpio")

    def _open(chip: int) -> object:
        return object()

    def _claim_output(handle: object, flags: int, pin: int, level: int) -> None:
        writes.append(pin)
        writes.append(level)

    def _gpio_write(handle: object, pin: int, value: int) -> None:
        writes.append(pin)
        writes.append(value)

    def _free(handle: object, pin: int) -> None:
        pass

    def _close(handle: object) -> None:
        pass

    fake_lgpio.gpiochip_open = _open  # type: ignore[attr-defined]
    fake_lgpio.gpio_claim_output = _claim_output  # type: ignore[attr-defined]
    fake_lgpio.gpio_write = _gpio_write  # type: ignore[attr-defined]
    fake_lgpio.gpio_free = _free  # type: ignore[attr-defined]
    fake_lgpio.gpiochip_close = _close  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lgpio", fake_lgpio)
    return writes


def test_dc_pin_required_to_toggle_real_hardware(monkeypatch: MonkeyPatch) -> None:
    """When dc_pin is None the factory must raise DisplayError immediately."""
    from robot.errors import DisplayError

    _install_fake_spidev(monkeypatch)
    config = DisplayConfig(backend="gc9a01", width=32, height=32, dc_pin=None)
    import pytest

    with pytest.raises(DisplayError):
        DisplayFactory(config).build()


def test_dc_pin_drives_gpio_when_lgpio_available(monkeypatch: MonkeyPatch) -> None:
    """With dc_pin set and lgpio available, every command/data toggles GPIO."""
    _install_fake_spidev(monkeypatch)
    writes = _install_fake_lgpio(monkeypatch)

    config = DisplayConfig(backend="gc9a01", width=32, height=32, dc_pin=25)
    display = DisplayFactory(config).build()
    transport = display._spi  # type: ignore[attr-defined]
    transport.command(b"\x01")
    transport.data(b"\x02\x03")
    transport.command(b"\x04")

    # Every command/data call writes the DC pin. We collect the values
    # following each occurrence of pin 25 in ``writes``.
    dc_values = [writes[i + 1] for i, v in enumerate(writes) if v == 25 and i + 1 < len(writes)]
    # Three calls -> at least three writes to pin 25, mix of 0 and 1.
    assert len(dc_values) >= 3
    assert 0 in dc_values
    assert 1 in dc_values


def test_reset_pin_drives_gpio(monkeypatch: MonkeyPatch) -> None:
    """The reset pin must be pulsed LOW -> HIGH at init."""
    _install_fake_spidev(monkeypatch)
    writes = _install_fake_lgpio(monkeypatch)
    config = DisplayConfig(backend="gc9a01", width=32, height=32, dc_pin=25, reset_pin=24)
    display = DisplayFactory(config).build()
    transport = display._spi  # type: ignore[attr-defined]
    assert 24 in writes  # the RST pin was claimed
    # During init the transport pulsed reset False then True.
    assert transport.resets == [True, False, True]


def test_backlight_pin_drives_gpio(monkeypatch: MonkeyPatch) -> None:
    """Backlight pin must be driven HIGH after init."""
    _install_fake_spidev(monkeypatch)
    writes = _install_fake_lgpio(monkeypatch)
    config = DisplayConfig(backend="gc9a01", width=32, height=32, dc_pin=25, backlight_pin=18)
    display = DisplayFactory(config).build()
    transport = display._spi  # type: ignore[attr-defined]
    assert 18 in writes
    assert transport.backlight_log == [True]


def test_pin_collision_raises(monkeypatch: MonkeyPatch) -> None:
    """The factory must reject overlapping GPIO pins."""
    import pytest

    from robot.errors import ConfigurationError

    _install_fake_spidev(monkeypatch)
    _install_fake_lgpio(monkeypatch)
    config = DisplayConfig(backend="gc9a01", width=32, height=32, dc_pin=25, reset_pin=25)
    with pytest.raises(ConfigurationError):
        DisplayFactory(config).build()
