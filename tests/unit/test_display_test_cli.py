"""Tests for the deskbot-display-test CLI.

The CLI runs against a fake display so we don't need real hardware.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from robot.interfaces.display import EyeFrame

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.monkeypatch import MonkeyPatch


class _StubDisplayLike:
    """Type stub for the captured fake display."""

    fills: list[tuple[int, int, int]]
    shows: list[EyeFrame]
    clears: int


def _run_main(
    monkeypatch: MonkeyPatch,
    *,
    args: list[str],
) -> tuple[int, _StubDisplayLike]:
    """Invoke the CLI with ``args`` and return its exit code and fake display."""
    from robot.cli import display_test

    monkeypatch.setattr("sys.argv", ["deskbot-display-test", *args])

    # Patch configure_logging so it doesn't try to read AppSettings.
    monkeypatch.setattr(
        display_test,
        "configure_logging",
        lambda *_a, **_kw: None,
    )

    class _StubDisplay(_StubDisplayLike):
        """Captures every fill/show/clear call without touching the SPI bus."""

        def __init__(
            self,
            width: int = 32,
            height: int = 32,
            **_kwargs: object,
        ) -> None:
            self.fills = []
            self.shows = []
            self.clears = 0
            self._width = width
            self._height = height

        @property
        def width(self) -> int:
            return self._width

        @property
        def height(self) -> int:
            return self._height

        async def fill(self, color: tuple[int, int, int]) -> None:
            self.fills.append(color)

        async def show(self, frame: EyeFrame) -> None:
            self.shows.append(frame)

        async def clear(self) -> None:
            self.clears += 1

        async def close(self) -> None:
            pass

        def write(self, data: bytes) -> int:
            return len(data)

        def write_readinto(
            self,
            data: bytes,
            buffer: bytearray,
        ) -> None:
            pass

    fake = _StubDisplay()

    class _StubFactory:
        def __init__(self, _config: object) -> None:
            pass

        def build(self) -> _StubDisplay:
            return fake

    async def _noop_sleep(_s: float) -> None:
        return None

    # display_test uses asyncio.sleep(), so patch the asyncio module directly.
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

    with (
        patch.object(display_test, "DisplayFactory", _StubFactory),
        pytest.raises(SystemExit) as exc_info,
    ):
        display_test.main()

    code = exc_info.value.code
    exit_code = int(code) if isinstance(code, (int, str)) else 0

    return exit_code, fake


def test_cli_runs_full_diag_then_returns_zero(
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    code, fake = _run_main(
        monkeypatch,
        args=["--size", "32", "--fps", "10"],
    )

    assert code == 0

    # 5 solid colour fills (R/G/B/W/B), plus clear at the end.
    assert fake.fills == [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 255),
        (0, 0, 0),
    ]

    # 6 geometric patterns plus 1 inversion-toggle re-render.
    assert len(fake.shows) == 7

    # 1 clear at the very end of the sequence.
    assert fake.clears == 1

    captured = capsys.readouterr()
    assert "display_test.start" in captured.out or "display_test.start" in captured.err
    assert "display_test.pass" in captured.out or "display_test.pass" in captured.err


def test_cli_skip_colours_skips_solid_fills(
    monkeypatch: MonkeyPatch,
) -> None:
    code, fake = _run_main(
        monkeypatch,
        args=["--size", "32", "--fps", "10", "--skip-colours"],
    )

    assert code == 0
    assert fake.fills == []
    assert len(fake.shows) == 7


def test_cli_help(
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    from robot.cli import display_test

    monkeypatch.setattr(
        "sys.argv",
        ["deskbot-display-test", "--help"],
    )
    monkeypatch.setattr(
        display_test,
        "configure_logging",
        lambda *_a, **_kw: None,
    )

    with pytest.raises(SystemExit) as exc_info:
        display_test.main()

    # --help exits with 0.
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    assert "display" in captured.out.lower()


def test_cli_invalid_config_returns_2(
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    """Conflicting pins must be rejected with a non-zero exit code."""
    from robot.cli import display_test

    monkeypatch.setattr(
        "sys.argv",
        [
            "deskbot-display-test",
            "--size",
            "32",
            "--dc-pin",
            "25",
            "--reset-pin",
            "25",
        ],
    )
    monkeypatch.setattr(
        display_test,
        "configure_logging",
        lambda *_a, **_kw: None,
    )

    with pytest.raises(SystemExit) as exc_info:
        display_test.main()

    assert exc_info.value.code == 2

    captured = capsys.readouterr()
    assert (
        "display_test.invalid_config" in captured.out
        or "display_test.invalid_config" in captured.err
    )


def test_cli_loop_runs_until_interrupted(
    monkeypatch: MonkeyPatch,
) -> None:
    """With ``--loop``, the CLI should run repeatedly until KeyboardInterrupt."""
    from robot.cli import display_test

    iterations: list[int] = []
    sleep_calls = 0

    async def fake_run(_args: object) -> int:
        iterations.append(1)
        return 0

    async def fake_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1

        if sleep_calls >= 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(
        "sys.argv",
        ["deskbot-display-test", "--loop"],
    )
    monkeypatch.setattr(
        display_test,
        "configure_logging",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        display_test,
        "_run",
        fake_run,
    )
    monkeypatch.setattr(
        asyncio,
        "sleep",
        fake_sleep,
    )

    with pytest.raises(SystemExit) as exc_info:
        display_test.main()

    assert exc_info.value.code == 0
    assert len(iterations) == 3
    assert sleep_calls == 3
