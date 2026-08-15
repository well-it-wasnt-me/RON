"""Unit tests for the interactive CLI command parser and registry."""

from __future__ import annotations

from robot.cli.interactive.commands import (
    Command,
    CommandContext,
    CommandRegistry,
    ParamSpec,
    build_registry,
    parse_command,
)

# ---------------------------------------------------------------------------
# parse_command
# ---------------------------------------------------------------------------


class TestParseCommand:
    """Tests for the command line parser."""

    def test_simple_command(self) -> None:
        name, args = parse_command("emotion happy")
        assert name == "emotion"
        assert args == ["happy"]

    def test_empty_line(self) -> None:
        name, args = parse_command("")
        assert name == ""
        assert args == []

    def test_whitespace_only(self) -> None:
        name, _args = parse_command("   ")
        assert name == ""

    def test_quoted_argument(self) -> None:
        name, args = parse_command('speak "hello world"')
        assert name == "speak"
        assert args == ["hello world"]

    def test_case_insensitive_command(self) -> None:
        name, _args = parse_command("EMOTION happy")
        assert name == "emotion"

    def test_multiple_args(self) -> None:
        name, args = parse_command("servo pan 45.0")
        assert name == "servo"
        assert args == ["pan", "45.0"]


# ---------------------------------------------------------------------------
# CommandRegistry
# ---------------------------------------------------------------------------


class TestCommandRegistry:
    """Tests for the command registry."""

    def test_register_and_lookup(self) -> None:
        reg = CommandRegistry()
        cmd = Command(name="test", help="A test command")
        reg.register(cmd)
        assert reg.get("test") == cmd

    def test_missing_command_returns_none(self) -> None:
        reg = CommandRegistry()
        assert reg.get("nonexistent") is None

    def test_names_sorted(self) -> None:
        reg = CommandRegistry()
        reg.register(Command(name="zoo", help=""))
        reg.register(Command(name="alpha", help=""))
        reg.register(Command(name="mid", help=""))
        assert reg.names == ["alpha", "mid", "zoo"]

    def test_commands_sorted(self) -> None:
        reg = CommandRegistry()
        reg.register(Command(name="zoo", help="z"))
        reg.register(Command(name="alpha", help="a"))
        result = reg.commands
        assert result[0].name == "alpha"
        assert result[1].name == "zoo"


# ---------------------------------------------------------------------------
# build_registry
# ---------------------------------------------------------------------------


class TestBuildRegistry:
    """Tests for the default command registry builder."""

    def test_all_core_commands_present(self) -> None:
        reg = build_registry()
        expected = [
            "emotion",
            "blink",
            "look",
            "bounce",
            "smile",
            "theme",
            "wave",
            "nod",
            "shrug",
            "celebrate",
            "greet",
            "arms_open",
            "arms_relax",
            "servo",
            "servos",
            "release",
            "state",
            "status",
            "behavior",
            "speak",
            "say",
            "mic",
            "cam",
            "fps",
            "config",
            "events",
            "help",
            "quit",
        ]
        for name in expected:
            assert reg.get(name) is not None, f"Missing command: {name}"

    def test_quit_aliases(self) -> None:
        reg = build_registry()
        assert reg.get("exit") is not None
        assert reg.get("q") is not None

    def test_help_alias(self) -> None:
        reg = build_registry()
        assert reg.get("?") is not None

    def test_all_commands_have_handlers(self) -> None:
        reg = build_registry()
        for cmd in reg.commands:
            assert cmd.handler is not None, f"Command '{cmd.name}' has no handler"

    def test_emotion_has_params(self) -> None:
        reg = build_registry()
        cmd = reg.get("emotion")
        assert cmd is not None
        assert len(cmd.params) == 2
        assert cmd.params[0].name == "emotion"
        assert cmd.params[0].type is str
        assert cmd.params[1].name == "intensity"
        assert cmd.params[1].required is False

    def test_servo_has_params(self) -> None:
        reg = build_registry()
        cmd = reg.get("servo")
        assert cmd is not None
        assert len(cmd.params) == 2
        assert cmd.params[0].name == "name"
        assert cmd.params[1].name == "angle"
        assert cmd.params[1].type is float

    def test_state_command_params(self) -> None:
        reg = build_registry()
        cmd = reg.get("state")
        assert cmd is not None
        assert len(cmd.params) == 1
        assert cmd.params[0].name == "state"


# ---------------------------------------------------------------------------
# CommandContext
# ---------------------------------------------------------------------------


class TestCommandContext:
    """Tests for the command context dataclass."""

    def test_default_fields_are_none(self) -> None:
        ctx = CommandContext(app=None)
        assert ctx.app is None
        assert ctx.settings is None
        assert ctx.face_animator is None
        assert ctx.body_engine is None
        assert ctx.servo_controller is None
        assert ctx.state_machine is None
        assert ctx.bus is None
        assert ctx.conversation is None
        assert ctx.display is None
        assert ctx.simulation_driver is None
        assert ctx.microphone is None
        assert ctx.camera is None

    def test_app_field_required(self) -> None:
        ctx = CommandContext(app="mock_app")
        assert ctx.app == "mock_app"


# ---------------------------------------------------------------------------
# ParamSpec
# ---------------------------------------------------------------------------


class TestParamSpec:
    """Tests for the param spec dataclass."""

    def test_defaults(self) -> None:
        spec = ParamSpec(name="test")
        assert spec.type is str
        assert spec.required is True
        assert spec.default == ""
        assert spec.help == ""

    def test_custom(self) -> None:
        spec = ParamSpec(
            name="angle", type=float, required=False, default="90.0", help="Angle in degrees"
        )
        assert spec.type is float
        assert spec.required is False
        assert spec.default == "90.0"
