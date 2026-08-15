"""Unit tests for the interactive CLI app module."""

from __future__ import annotations

import pytest

from robot.config import AppSettings

# ---------------------------------------------------------------------------
# create_simulation_driver
# ---------------------------------------------------------------------------


class TestCreateSimulationDriver:
    """Tests for the simulation driver factory."""

    def test_default_settings_creates_driver(self) -> None:
        from robot.cli.interactive.app import create_simulation_driver

        settings = AppSettings()
        sim = create_simulation_driver(settings)
        assert sim is not None
        assert sim.width == settings.displays.width
        assert sim.fps == settings.displays.fps

    def test_driver_has_face_and_body(self) -> None:
        from robot.cli.interactive.app import create_simulation_driver

        settings = AppSettings()
        sim = create_simulation_driver(settings)
        assert sim.face is not None
        assert sim.body is not None

    def test_driver_step_returns_frame(self) -> None:
        from robot.cli.interactive.app import create_simulation_driver

        settings = AppSettings()
        sim = create_simulation_driver(settings)
        frame = sim.step()
        assert frame is not None
        assert frame.width > 0
        assert frame.height > 0
        assert len(frame.pixels) > 0


# ---------------------------------------------------------------------------
# FallbackREPL
# ---------------------------------------------------------------------------


class TestFallbackREPL:
    """Tests for the non-Textual fallback REPL."""

    def test_create_repl(self) -> None:
        from robot.cli.interactive.app import _FallbackREPL, create_simulation_driver

        settings = AppSettings()
        sim = create_simulation_driver(settings)
        repl = _FallbackREPL(settings=settings, sim=sim)
        assert repl.settings is settings
        assert repl.sim is sim
        assert repl._running is True

    def test_log_prints(self, capsys: pytest.CaptureFixture[str]) -> None:
        from robot.cli.interactive.app import _FallbackREPL, create_simulation_driver

        settings = AppSettings()
        sim = create_simulation_driver(settings)
        repl = _FallbackREPL(settings=settings, sim=sim)
        repl.write_log("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.out

    def test_exit_stops(self) -> None:
        from robot.cli.interactive.app import _FallbackREPL, create_simulation_driver

        settings = AppSettings()
        sim = create_simulation_driver(settings)
        repl = _FallbackREPL(settings=settings, sim=sim)
        repl.exit()
        assert repl._running is False

    def test_command_registry_has_commands(self) -> None:
        from robot.cli.interactive.app import _FallbackREPL, create_simulation_driver

        settings = AppSettings()
        sim = create_simulation_driver(settings)
        repl = _FallbackREPL(settings=settings, sim=sim)
        # The registry should have all the built-in commands.
        assert len(repl.command_registry.names) > 20
        assert "help" in repl.command_registry.names
        assert "quit" in repl.command_registry.names
        assert "emotion" in repl.command_registry.names


# ---------------------------------------------------------------------------
# parse_command integration
# ---------------------------------------------------------------------------


class TestParseCommandIntegration:
    """Integration tests for parse_command with the registry."""

    def test_parsed_command_found_in_registry(self) -> None:
        from robot.cli.interactive.commands import build_registry, parse_command

        reg = build_registry()
        name, args = parse_command("emotion happy 0.8")
        cmd = reg.get(name)
        assert cmd is not None
        assert cmd.name == "emotion"
        assert args == ["happy", "0.8"]

    def test_parsed_unknown_command(self) -> None:
        from robot.cli.interactive.commands import build_registry, parse_command

        reg = build_registry()
        name, _args = parse_command("nonexistent foo")
        cmd = reg.get(name)
        assert cmd is None
