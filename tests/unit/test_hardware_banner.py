"""Tests for the hardware.banner diagnostic + fallback observability."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# Import the function under test.
from robot.cli.doctor import _hardware_banner


class _PatchLog:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, object]]] = []

    def info(self, event: str, **kw: object) -> None:
        self.records.append(("info", {"event": event, **kw}))

    def error(self, event: str, **kw: object) -> None:
        self.records.append(("error", {"event": event, **kw}))


@pytest.fixture
def fake_log(monkeypatch: pytest.MonkeyPatch) -> _PatchLog:
    log = _PatchLog()
    # The robot.cli package's __init__ imports main from
    # cli.doctor into the doctor attribute name, which collides
    # with the robot.cli.doctor submodule lookup when used as
    # import robot.cli.doctor as .... Use sys.modules directly.
    import sys

    doctor_mod = sys.modules["robot.cli.doctor"]
    monkeypatch.setattr(doctor_mod, "_log", log)
    return log


def test_hardware_banner_reports_real_backends(fake_log: _PatchLog) -> None:
    settings = MagicMock(hardware="real")
    display = MagicMock()
    type(display).__name__ = "CircuitPythonDisplay"
    microphone = MagicMock()
    type(microphone).__name__ = "UsbMicrophone"
    camera = MagicMock()
    type(camera).__name__ = "UsbCamera"

    _hardware_banner(settings, display, microphone=microphone, camera=camera)

    info_events = [r for r in fake_log.records if r[0] == "info"]
    assert any(
        r[1]["event"] == "hardware.active"
        and r[1]["display_real"] is True
        and r[1]["microphone_real"] is True
        and r[1]["camera_real"] is True
        for r in info_events
    )
    assert not any(r[1]["event"] == "hardware.fallback" for r in fake_log.records)


def test_hardware_banner_logs_fallback_when_real_requested_but_mock_active(
    fake_log: _PatchLog,
) -> None:
    settings = MagicMock(hardware="real")
    display = MagicMock()
    type(display).__name__ = "MockDisplay"
    microphone = MagicMock()
    type(microphone).__name__ = "MockMicrophone"
    camera = MagicMock()
    type(camera).__name__ = "MockCamera"

    _hardware_banner(settings, display, microphone=microphone, camera=camera)

    fallback = [r for r in fake_log.records if r[1]["event"] == "hardware.fallback"]
    components = {r[1]["component"] for r in fallback}
    assert components == {"display", "microphone", "camera"}


def test_hardware_banner_no_fallback_when_hardware_is_mock(fake_log: _PatchLog) -> None:
    """If ``hardware='mock'`` the user asked for mocks; no fallback error."""
    settings = MagicMock(hardware="mock")
    display = MagicMock()
    type(display).__name__ = "MockDisplay"
    microphone = MagicMock()
    type(microphone).__name__ = "MockMicrophone"
    camera = MagicMock()
    type(camera).__name__ = "MockCamera"

    _hardware_banner(settings, display, microphone=microphone, camera=camera)

    assert not any(r[1]["event"] == "hardware.fallback" for r in fake_log.records)
