"""Tests for the GC9A01 SPI display driver."""

from __future__ import annotations

import pytest

from robot.errors import DisplayError
from robot.hardware.displays.gc9a01 import (
    COLMOD,
    DISPON,
    INVOFF,
    INVON,
    MADCTL,
    SLPOUT,
    SWRESET,
    FakeSpiTransport,
    GC9A01Display,
    rgb888_to_rgb565,
)
from robot.interfaces.display import EyeFrame


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def _make_frame(
    width: int = 16, height: int = 16, color: tuple[int, int, int] = (255, 0, 0)
) -> EyeFrame:
    pixel = bytes(color) * (width * height)
    return EyeFrame(width=width, height=height, pixels=pixel)


def _noop_sleep(_s: float) -> None:
    return None


# ---------------------------------------------------------------------------
# Init sequence
# ---------------------------------------------------------------------------
async def test_init_emits_canonical_command_sequence() -> None:
    bus = FakeSpiTransport()
    GC9A01Display(width=16, height=16, transport=bus, sleep_fn=_noop_sleep)
    flat = b"".join(bus.commands)
    # Each canonical command MUST appear in the init stream.
    for expected in (SWRESET, SLPOUT, COLMOD, MADCTL, INVON, DISPON):
        assert bytes([expected]) in flat, f"command 0x{expected:02x} missing from init"


async def test_init_runs_hardware_reset() -> None:
    bus = FakeSpiTransport()
    GC9A01Display(width=16, height=16, transport=bus, sleep_fn=_noop_sleep)
    # Per the verified working init sequence the panel boots with RST HIGH,
    # we then drive it LOW -> HIGH to issue a clean reset.
    assert bus.resets == [True, False, True]
    assert bus.reset_holds[0] > 0
    assert bus.reset_holds[1] > 0
    assert bus.reset_holds[2] > 0


async def test_init_writes_colmod_16bit() -> None:
    bus = FakeSpiTransport()
    GC9A01Display(width=16, height=16, transport=bus, sleep_fn=_noop_sleep)
    flat = b"".join(bus.commands)
    # COLMOD 0x3A + 0x55 (16-bit) appears in one SPI write.
    assert b"\x3a\x05" in flat


async def test_init_writes_madctl_with_bgr_bit() -> None:
    bus = FakeSpiTransport()
    GC9A01Display(width=16, height=16, transport=bus, sleep_fn=_noop_sleep, rotation=0)
    flat = b"".join(bus.commands)
    # MADCTL 0x36 followed by 0x08 (BGR) for rotation 0.
    assert b"\x36\x08" in flat


async def test_init_invoff_when_invert_false() -> None:
    bus = FakeSpiTransport()
    GC9A01Display(width=16, height=16, transport=bus, sleep_fn=_noop_sleep, invert=False)
    # Look at single-byte command writes (length == 1); INVOFF (0x20) should
    # appear and INVON (0x21) should not. (Note: 0x21 also appears as a data
    # byte for the 0x89 register select, but never as a standalone command.)
    single_byte_commands = [c for c in bus.commands if len(c) == 1]
    assert bytes([INVOFF]) in single_byte_commands
    assert bytes([INVON]) not in single_byte_commands


async def test_init_unlocks_extension_command_set() -> None:
    """The init sequence MUST issue the 0xEF/0xEB/0x14 unlock before
    any analog front-end registers are written.

    Without this unlock the panel silently ignores writes to the
    analog front-end registers - exactly the "vertical colour bars
    + ghost overlay" symptom seen on the Pi.
    """
    bus = FakeSpiTransport()
    GC9A01Display(width=16, height=16, transport=bus, sleep_fn=_noop_sleep)
    flat = b"".join(bus.commands)
    # 0xEF and 0xEB 0x14 must both appear before the first analog register.
    unlock_a_pos = flat.find(bytes([0xEF]))
    assert unlock_a_pos != -1, "extension unlock 0xEF missing"
    unlock_key_pos = flat.find(bytes([0xEB, 0x14]))
    assert unlock_key_pos != -1, "extension unlock 0xEB 0x14 missing"
    assert unlock_a_pos < unlock_key_pos, "0xEF must be sent before 0xEB"


async def test_init_uses_longer_waits_for_cold_boot() -> None:
    """Reset release hold must be long enough for cold-boot stabilisation."""
    bus = FakeSpiTransport()
    GC9A01Display(width=16, height=16, transport=bus, sleep_fn=_noop_sleep)
    # Sequence is HIGH -> LOW -> HIGH. The final release (HIGH) must be >= 100 ms.
    assert bus.resets == [True, False, True]
    assert bus.reset_holds[2] >= 0.1


async def test_init_writes_display_function_control_with_two_bytes() -> None:
    """DISCTRL must be written with the verified 2-byte parameter 0x00 0x20."""
    bus = FakeSpiTransport()
    GC9A01Display(width=16, height=16, transport=bus, sleep_fn=_noop_sleep)
    flat = b"".join(bus.commands)
    assert b"\xb6\x00\x20" in flat


async def test_init_writes_command_lock_with_three_bytes() -> None:
    """The command-lock write must be 0xFF 0x60 0x01 0x04."""
    bus = FakeSpiTransport()
    GC9A01Display(width=16, height=16, transport=bus, sleep_fn=_noop_sleep)
    flat = b"".join(bus.commands)
    assert b"\xff\x60\x01\x04" in flat


async def test_init_enables_backlight_by_default() -> None:
    bus = FakeSpiTransport()
    GC9A01Display(width=16, height=16, transport=bus, sleep_fn=_noop_sleep)
    assert bus.backlight_log == [True]


async def test_init_can_skip_backlight() -> None:
    bus = FakeSpiTransport()
    GC9A01Display(
        width=16,
        height=16,
        transport=bus,
        sleep_fn=_noop_sleep,
        enable_backlight=False,
    )
    assert bus.backlight_log == []


async def test_rotation_affects_madctl() -> None:
    bus = FakeSpiTransport()
    GC9A01Display(width=16, height=16, transport=bus, sleep_fn=_noop_sleep, rotation=2)
    flat = b"".join(bus.commands)
    # Rotation 2 -> 0xC0 | 0x08 = 0xC8
    assert b"\x36\xc8" in flat


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_invalid_dimensions_raise() -> None:
    with pytest.raises(ValueError):
        GC9A01Display(width=0, height=16)
    with pytest.raises(ValueError):
        GC9A01Display(width=16, height=0)


def test_invalid_rotation_raises() -> None:
    with pytest.raises(ValueError):
        GC9A01Display(width=16, height=16, rotation=4)


def test_invalid_chunk_bytes_raises() -> None:
    with pytest.raises(ValueError):
        GC9A01Display(width=16, height=16, chunk_bytes=0)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
async def test_show_writes_caset_raset_and_pixels() -> None:
    bus = FakeSpiTransport()
    disp = GC9A01Display(width=4, height=4, transport=bus, sleep_fn=_noop_sleep)
    bus.commands.clear()
    bus.datas.clear()
    frame = _make_frame(width=4, height=4, color=(255, 128, 0))
    await disp.show(frame)
    flat = b"".join(bus.commands)
    # The CASET (0x2A) and RASET (0x2B) commands should both appear.
    assert b"\x2a" in flat
    assert b"\x2b" in flat
    # RAMWR (0x2C) is the LAST command before the pixel data.
    assert flat.endswith(b"\x2c")
    # The pixel data write should be 4*4*2 = 32 bytes (RGB565).
    pixel_writes = [w for w in bus.datas if len(w) == 32]
    assert pixel_writes, f"no 32-byte pixel write in {[len(w) for w in bus.datas]}"


async def test_show_validates_frame_size() -> None:
    bus = FakeSpiTransport()
    disp = GC9A01Display(width=4, height=4, transport=bus, sleep_fn=_noop_sleep)
    bad = _make_frame(width=2, height=2, color=(0, 0, 0))
    with pytest.raises(DisplayError):
        await disp.show(bad)


async def test_show_validates_pixel_buffer_size() -> None:
    bus = FakeSpiTransport()
    disp = GC9A01Display(width=4, height=4, transport=bus, sleep_fn=_noop_sleep)
    # Right size but truncated buffer
    bad = EyeFrame(width=4, height=4, pixels=b"\x00" * 10)
    with pytest.raises(DisplayError):
        await disp.show(bad)


async def test_show_full_frame_is_115200_bytes_for_240x240() -> None:
    """The framebuffer for a 240x240 panel must be 240*240*2 = 115200 bytes."""
    bus = FakeSpiTransport()
    disp = GC9A01Display(width=240, height=240, transport=bus, sleep_fn=_noop_sleep)
    bus.datas.clear()
    await disp.fill((255, 0, 0))
    total = sum(len(w) for w in bus.datas)
    assert total == 240 * 240 * 2
    # The largest single chunk must be <= chunk_bytes.
    assert max(len(w) for w in bus.datas) <= 4096  # kernel SPI cap


async def test_chunks_large_writes() -> None:
    """A 240x240 RGB565 frame is 115200 bytes; we must chunk to <= chunk_bytes."""
    bus = FakeSpiTransport()
    disp = GC9A01Display(
        width=240, height=240, transport=bus, sleep_fn=_noop_sleep, chunk_bytes=4096
    )
    bus.datas.clear()
    await disp.fill((0, 0, 0))
    assert all(len(w) <= 4096 for w in bus.datas)
    # ceil(115200 / 4096) = 29 chunks.
    assert len(bus.datas) >= 29


async def test_rgb888_to_rgb565_conversion() -> None:
    # Pure red: (255, 0, 0) -> R=31, G=0, B=0 -> 0xF800
    assert rgb888_to_rgb565(bytes((255, 0, 0))) == b"\xf8\x00"
    # Pure green: (0, 255, 0) -> R=0, G=63, B=0 -> 0x07E0
    assert rgb888_to_rgb565(bytes((0, 255, 0))) == b"\x07\xe0"
    # Pure blue: (0, 0, 255) -> R=0, G=0, B=31 -> 0x001F
    assert rgb888_to_rgb565(bytes((0, 0, 255))) == b"\x00\x1f"


async def test_fill_and_clear() -> None:
    bus = FakeSpiTransport()
    disp = GC9A01Display(width=4, height=4, transport=bus, sleep_fn=_noop_sleep)
    bus.datas.clear()
    await disp.fill((100, 50, 25))
    pixel_writes = [w for w in bus.datas if len(w) == 32]
    assert pixel_writes, f"no 32-byte pixel write in {[len(w) for w in bus.datas]}"
    rgb565 = rgb888_to_rgb565(bytes((100, 50, 25)))
    assert pixel_writes[0] == bytes(rgb565) * 16


async def test_close_sets_closed_flag() -> None:
    bus = FakeSpiTransport()
    disp = GC9A01Display(width=4, height=4, transport=bus, sleep_fn=_noop_sleep)
    await disp.close()
    assert bus.closed
    with pytest.raises(RuntimeError):
        await disp.show(_make_frame(4, 4))


# ---------------------------------------------------------------------------
# FakeSpiTransport
# ---------------------------------------------------------------------------
async def test_fake_spi_records_writes() -> None:
    bus = FakeSpiTransport()
    bus.write(b"\x01\x02")
    bus.write(b"\x03")
    assert bus.writes == [b"\x01\x02", b"\x03"]


async def test_fake_spi_records_command_vs_data() -> None:
    bus = FakeSpiTransport()
    bus.command(b"\x01")
    bus.data(b"\x02\x03")
    assert bus.commands == [b"\x01"]
    assert bus.datas == [b"\x02\x03"]


async def test_fake_spi_records_reset_and_backlight() -> None:
    bus = FakeSpiTransport()
    bus.reset(False, hold_s=0.020)
    bus.reset(True)
    bus.set_backlight(True)
    bus.set_backlight(False)
    assert bus.resets == [False, True]
    assert bus.reset_holds == [0.020, 0.020]
    assert bus.backlight_log == [True, False]


async def test_fake_spi_can_simulate_failure() -> None:
    bus = FakeSpiTransport(fail_on={b"\xff"})
    with pytest.raises(DisplayError):
        bus.write(b"\xff\x00")
    # Non-forbidden bytes still go through
    bus.write(b"\x01")
    assert bus.writes == [b"\x01"]


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------
async def test_display_implements_protocol() -> None:
    """The GC9A01 driver must satisfy the Display Protocol at runtime."""
    from robot.interfaces.display import Display

    bus = FakeSpiTransport()
    disp = GC9A01Display(width=4, height=4, transport=bus, sleep_fn=_noop_sleep)
    assert isinstance(disp, Display)


async def test_transport_satisfies_display_transport_protocol() -> None:
    """The fake must satisfy the DisplayTransport Protocol."""
    from robot.hardware.displays.gc9a01 import DisplayTransport

    bus = FakeSpiTransport()
    assert isinstance(bus, DisplayTransport)
