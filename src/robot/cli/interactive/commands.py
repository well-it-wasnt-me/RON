"""Command parser and registry for the DeskBot interactive TUI.

Each command is a small :class:`Command` dataclass with a name, short
help string, parameter spec, and a handler function. The
:class:`CommandRegistry` collects them and provides lookup, help, and
tab-completion.

The handlers receive a :class:`CommandContext` that carries live
references to the running robot components so they can drive the face,
servos, state machine, conversation, etc.
"""

from __future__ import annotations

import shlex
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from robot.behavior.state_machine import RobotState
from robot.events.events import EmotionName

# ---------------------------------------------------------------------------
# Context object passed to every command handler
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CommandContext:
    """Live references passed to command handlers at runtime.

    Not all fields are guaranteed to be non-``None``; handlers must
    check before using optional components.
    """

    app: Any  # DeskBotTUI - avoids circular import
    settings: Any = None  # AppSettings

    # Components (may be None if hardware is unavailable)
    face_animator: Any = None
    body_engine: Any = None
    servo_controller: Any = None
    state_machine: Any = None
    bus: Any = None
    conversation: Any = None
    display: Any = None
    simulation_driver: Any = None  # SimulationDriver, if in sim mode
    microphone: Any = None
    camera: Any = None


# ---------------------------------------------------------------------------
# Command definition
# ---------------------------------------------------------------------------

ParamType = type  # Alias for clarity


@dataclass(slots=True, frozen=True)
class ParamSpec:
    """Description of a single command parameter."""

    name: str
    type: ParamType = str
    required: bool = True
    default: str = ""
    help: str = ""


@dataclass(slots=True, frozen=True)
class Command:
    """A single TUI command."""

    name: str
    help: str
    params: tuple[ParamSpec, ...] = ()
    handler: Callable[..., Coroutine[Any, Any, None]] | None = None


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


async def _cmd_emotion(ctx: CommandContext, args: list[str]) -> None:
    """Set the face emotion."""
    if not args:
        valid = [e.value for e in EmotionName]
        ctx.app.write_log(f"Usage: emotion <{'|'.join(valid)}>")
        return
    try:
        emotion = EmotionName(args[0].lower())
    except ValueError:
        valid = [e.value for e in EmotionName]
        ctx.app.write_log(f"Invalid emotion. Valid: {valid}")
        return
    intensity = float(args[1]) if len(args) > 1 else 1.0
    if ctx.app.simulation_driver is not None:
        ctx.app.simulation_driver.face.set_emotion(emotion.value)
    elif ctx.face_animator is not None:
        ctx.face_animator.set_emotion(emotion.value)
    if ctx.bus is not None:
        from robot.events.events import EmotionChanged

        await ctx.bus.publish(
            EmotionChanged(previous=EmotionName.NEUTRAL, current=emotion, intensity=intensity)
        )
    ctx.app.write_log(f"Emotion -> {emotion.value} (intensity={intensity:.2f})")


async def _cmd_blink(ctx: CommandContext, args: list[str]) -> None:
    """Blink eyes."""
    if ctx.app.simulation_driver is not None:
        ctx.app.simulation_driver.face.blink()
    elif ctx.face_animator is not None:
        ctx.face_animator.blink()
    ctx.app.write_log("Blink!")


async def _cmd_look(ctx: CommandContext, args: list[str]) -> None:
    """Look in a direction: left, right, up, down, center."""
    if not args:
        ctx.app.write_log("Usage: look <left|right|up|down|center>")
        return
    direction = args[0].lower()
    animator = (
        ctx.app.simulation_driver.face
        if ctx.app.simulation_driver is not None
        else ctx.face_animator
    )
    if animator is None:
        ctx.app.write_log("No face animator available")
        return
    look_fn = {
        "left": animator.look_left,
        "right": animator.look_right,
        "up": animator.look_up,
        "down": animator.look_down,
        "center": animator.look_center,
    }.get(direction)
    if look_fn is None:
        ctx.app.write_log(f"Unknown direction '{direction}'. Valid: left, right, up, down, center")
        return
    look_fn()
    ctx.app.write_log(f"Look -> {direction}")


async def _cmd_bounce(ctx: CommandContext, args: list[str]) -> None:
    """Bounce the face animation."""
    animator = (
        ctx.app.simulation_driver.face
        if ctx.app.simulation_driver is not None
        else ctx.face_animator
    )
    if animator is None:
        ctx.app.write_log("No face animator available")
        return
    animator.bounce()
    ctx.app.write_log("Bounce!")


async def _cmd_smile(ctx: CommandContext, args: list[str]) -> None:
    """Smile animation."""
    animator = (
        ctx.app.simulation_driver.face
        if ctx.app.simulation_driver is not None
        else ctx.face_animator
    )
    if animator is None:
        ctx.app.write_log("No face animator available")
        return
    animator.smile_grow()
    ctx.app.write_log("😊 Smiling")


async def _cmd_wave(ctx: CommandContext, args: list[str]) -> None:
    """Wave the right arm."""
    body = (
        ctx.app.simulation_driver.body if ctx.app.simulation_driver is not None else ctx.body_engine
    )
    if body is None:
        ctx.app.write_log("No body engine available")
        return
    from robot.body_language.requests import Wave

    await body.perform(Wave(amplitude=30.0))
    ctx.app.write_log("👋 Wave")


async def _cmd_nod(ctx: CommandContext, args: list[str]) -> None:
    """Nod the head."""
    body = (
        ctx.app.simulation_driver.body if ctx.app.simulation_driver is not None else ctx.body_engine
    )
    if body is None:
        ctx.app.write_log("No body engine available")
        return
    from robot.body_language.requests import HeadNod

    await body.perform(HeadNod(amplitude=15.0))
    ctx.app.write_log("Nod")


async def _cmd_shrug(ctx: CommandContext, args: list[str]) -> None:
    """Shrug gesture."""
    body = (
        ctx.app.simulation_driver.body if ctx.app.simulation_driver is not None else ctx.body_engine
    )
    if body is None:
        ctx.app.write_log("No body engine available")
        return
    from robot.body_language.requests import Shrug

    await body.perform(Shrug())
    ctx.app.write_log("🤷 Shrug")


async def _cmd_celebrate(ctx: CommandContext, args: list[str]) -> None:
    """Celebrate gesture."""
    body = (
        ctx.app.simulation_driver.body if ctx.app.simulation_driver is not None else ctx.body_engine
    )
    if body is None:
        ctx.app.write_log("No body engine available")
        return
    from robot.body_language.requests import Celebrate

    await body.perform(Celebrate())
    ctx.app.write_log("🎉 Celebrate")


async def _cmd_greet(ctx: CommandContext, args: list[str]) -> None:
    """Greeting gesture."""
    body = (
        ctx.app.simulation_driver.body if ctx.app.simulation_driver is not None else ctx.body_engine
    )
    if body is None:
        ctx.app.write_log("No body engine available")
        return
    from robot.body_language.requests import Greet

    await body.perform(Greet())
    ctx.app.write_log("👋 Greet")


async def _cmd_arms_open(ctx: CommandContext, args: list[str]) -> None:
    """Open both arms."""
    body = (
        ctx.app.simulation_driver.body if ctx.app.simulation_driver is not None else ctx.body_engine
    )
    if body is None:
        ctx.app.write_log("No body engine available")
        return
    from robot.body_language.requests import ArmsOpen

    await body.perform(ArmsOpen())
    ctx.app.write_log("🙌 Arms open")


async def _cmd_arms_relax(ctx: CommandContext, args: list[str]) -> None:
    """Relax both arms to neutral."""
    body = (
        ctx.app.simulation_driver.body if ctx.app.simulation_driver is not None else ctx.body_engine
    )
    if body is None:
        ctx.app.write_log("No body engine available")
        return
    from robot.body_language.requests import ArmsRelax

    await body.perform(ArmsRelax())
    ctx.app.write_log("😌 Arms relaxed")


async def _cmd_servo(ctx: CommandContext, args: list[str]) -> None:
    """Move a servo directly: servo <name> <angle>."""
    if len(args) < 2:
        ctx.app.write_log("Usage: servo <name> <angle>")
        ctx.app.write_log("Names: pan, tilt, left_arm, right_arm")
        return
    name = args[0]
    try:
        angle = float(args[1])
    except ValueError:
        ctx.app.write_log(f"Invalid angle: {args[1]}")
        return
    controller = ctx.servo_controller
    if controller is None:
        ctx.app.write_log("No servo controller available")
        return
    try:
        servo = controller.get(name)
        await servo.move_to(angle)
        ctx.app.write_log(f"Servo {name} -> {angle:.1f}°")
    except Exception as exc:
        ctx.app.write_log(f"Error: {exc}")


async def _cmd_servos(ctx: CommandContext, args: list[str]) -> None:
    """Show current servo angles."""
    controller = ctx.servo_controller
    if controller is None:
        ctx.app.write_log("No servo controller available")
        return
    for s in controller.all():
        ctx.app.write_log(f"  {s.name}: {s.angle:.1f}°")


async def _cmd_release(ctx: CommandContext, args: list[str]) -> None:
    """Release servo(s). Optionally specify a name."""
    controller = ctx.servo_controller
    if controller is None:
        ctx.app.write_log("No servo controller available")
        return
    if args:
        try:
            servo = controller.get(args[0])
            await servo.release()
            ctx.app.write_log(f"Released {args[0]}")
        except Exception as exc:
            ctx.app.write_log(f"Error: {exc}")
    else:
        await controller.release_all()
        ctx.app.write_log("All servos released")


async def _cmd_state(ctx: CommandContext, args: list[str]) -> None:
    """Transition the robot state."""
    if not args:
        valid = [s.value for s in RobotState]
        ctx.app.write_log(f"Usage: state <{'|'.join(valid)}>")
        return
    try:
        target = RobotState(args[0].lower())
    except ValueError:
        valid = [s.value for s in RobotState]
        ctx.app.write_log(f"Invalid state. Valid: {valid}")
        return
    if ctx.state_machine is None:
        ctx.app.write_log("No state machine available")
        return
    try:
        await ctx.state_machine.transition(target)
        ctx.app.write_log(f"State -> {target.value}")
    except Exception as exc:
        ctx.app.write_log(f"Transition error: {exc}")


async def _cmd_status(ctx: CommandContext, args: list[str]) -> None:
    """Show current state, emotion, and servo positions."""
    from robot.cli.interactive.renderer import servo_dashboard, state_line

    state_val = ctx.state_machine.state.value if ctx.state_machine else "unknown"
    state_str = state_line(state_val)
    ctx.app.write_log(state_str)
    if ctx.servo_controller is not None:
        angles = {s.name: s.angle for s in ctx.servo_controller.all()}
        for line in servo_dashboard(angles):
            ctx.app.write_log(line)


async def _cmd_speak(ctx: CommandContext, args: list[str]) -> None:
    """Send text through the conversation pipeline (STT -> LLM -> TTS)."""
    if not args:
        ctx.app.write_log("Usage: speak <text>")
        return
    text = " ".join(args)
    if ctx.bus is None:
        ctx.app.write_log("No event bus available")
        return
    from robot.events.events import SpeechRecognized

    await ctx.bus.publish(SpeechRecognized(text=text, confidence=1.0))
    ctx.app.write_log(f"🗣 Speaking: {text}")


async def _cmd_say(ctx: CommandContext, args: list[str]) -> None:
    """Synthesize text directly via TTS (bypasses LLM)."""
    if not args:
        ctx.app.write_log("Usage: say <text>")
        return
    text = " ".join(args)
    if ctx.conversation is None or ctx.conversation.tts is None:
        ctx.app.write_log("No TTS available")
        return
    buffer = await ctx.conversation.tts.speak(text)
    audio = getattr(ctx.conversation, "audio", None)
    if audio is not None and buffer is not None and not buffer.is_empty:
        await audio.play(buffer)
    ctx.app.write_log(f"🗣 Said: {text}")


async def _cmd_theme(ctx: CommandContext, args: list[str]) -> None:
    """Change the face theme."""
    if not args:
        from robot.face.themes import BUILTIN_THEMES

        ctx.app.write_log(f"Themes: {', '.join(sorted(BUILTIN_THEMES.keys()))}")
        return
    theme_name = args[0].lower()
    if ctx.app.simulation_driver is not None:
        from robot.face.themes import get_theme

        try:
            theme = get_theme(theme_name)
            ctx.app.simulation_driver.theme = theme
            ctx.app.simulation_driver.face.theme = theme
            ctx.app.write_log(f"Theme -> {theme_name}")
        except KeyError as exc:
            ctx.app.write_log(f"Unknown theme: {exc}")
    elif ctx.face_animator is not None:
        ctx.app.write_log("Theme switching requires simulation mode")
    else:
        ctx.app.write_log("No face animator available")


async def _cmd_behavior(ctx: CommandContext, args: list[str]) -> None:
    """Run a pre-built behavior sequence."""
    if not args:
        ctx.app.write_log("Behaviors: greeting, thinking, listening, sleeping, excited, surprised")
        return
    from robot.behavior_library.behavior import (
        BehaviorRunner,
        excited,
        greeting,
        listening,
        sleeping,
        surprised,
        thinking,
    )
    from robot.utils.clock import SystemClock

    behaviors = {
        "greeting": greeting,
        "thinking": thinking,
        "listening": listening,
        "sleeping": sleeping,
        "excited": excited,
        "surprised": surprised,
    }
    name = args[0].lower()
    if name not in behaviors:
        ctx.app.write_log(f"Unknown behavior '{name}'. Valid: {', '.join(behaviors)}")
        return
    if ctx.app.simulation_driver is None:
        ctx.app.write_log("Behaviors require simulation mode")
        return
    runner = BehaviorRunner(
        face=ctx.app.simulation_driver.face,
        body=ctx.app.simulation_driver.body,
        clock=SystemClock(),
    )
    ctx.app.write_log(f"Running behavior: {name}")
    await runner.run(behaviors[name]())
    ctx.app.write_log(f"Behavior {name} complete")


async def _cmd_config(ctx: CommandContext, args: list[str]) -> None:
    """Show current configuration (sensitive fields masked)."""
    from robot.cli.interactive.renderer import config_summary

    if ctx.settings is None:
        ctx.app.write_log("No settings available")
        return
    d = ctx.settings.model_dump()
    for line in config_summary(d):
        ctx.app.write_log(line)


async def _cmd_fps(ctx: CommandContext, args: list[str]) -> None:
    """Show frame profiler stats (if available)."""
    profiler = getattr(ctx.app, "_frame_profiler", None)
    if profiler is None:
        ctx.app.write_log("Frame profiling not enabled")
        return
    stats = profiler.snapshot()
    if stats is None:
        ctx.app.write_log("No frame stats yet")
        return
    ctx.app.write_log(
        f"FPS: {stats.actual_fps:.1f}  "
        f"avg: {stats.avg_frame_time_ms:.1f}ms  "
        f"p50: {stats.p50_frame_time_ms:.1f}ms  "
        f"p95: {stats.p95_frame_time_ms:.1f}ms  "
        f"dropped: {stats.dropped_frames}/{stats.total_frames}"
    )


async def _cmd_help(ctx: CommandContext, args: list[str]) -> None:
    """Show command help."""
    registry = ctx.app.command_registry
    if args:
        cmd = registry.get(args[0])
        if cmd is None:
            ctx.app.write_log(f"Unknown command: {args[0]}")
            return
        ctx.app.write_log(f"  {cmd.name:16s} {cmd.help}")
        for p in cmd.params:
            req = "required" if p.required else f"default={p.default}"
            ctx.app.write_log(f"    <{p.name}> ({p.type.__name__}, {req})  {p.help}")
        return
    ctx.app.write_log("DeskBot Interactive Commands:")
    ctx.app.write_log("")
    for _name, cmd in sorted(registry._commands.items()):
        ctx.app.write_log(f"  {cmd.name:16s} {cmd.help}")


async def _cmd_quit(ctx: CommandContext, args: list[str]) -> None:
    """Shut down the robot and exit."""
    ctx.app.write_log("Shutting down…")
    ctx.app.exit()


async def _cmd_events(ctx: CommandContext, args: list[str]) -> None:
    """Toggle event log visibility."""
    if args and args[0].lower() in ("off", "0", "no"):
        ctx.app._show_events = False
        ctx.app.write_log("Event log hidden")
    else:
        ctx.app._show_events = True
        ctx.app.write_log("Event log visible")


async def _cmd_mic(ctx: CommandContext, args: list[str]) -> None:
    """Toggle microphone level meter display."""
    if args and args[0].lower() in ("off", "0", "no"):
        ctx.app._show_mic = False
        ctx.app.write_log("Mic meter hidden")
    else:
        ctx.app._show_mic = True
        ctx.app.write_log("Mic meter visible")


async def _cmd_cam(ctx: CommandContext, args: list[str]) -> None:
    """Toggle camera ASCII view."""
    if args and args[0].lower() in ("off", "0", "no"):
        ctx.app._show_cam = False
        ctx.app.write_log("Camera view hidden")
    else:
        ctx.app._show_cam = True
        ctx.app.write_log("Camera view visible")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class CommandRegistry:
    """Collects :class:`Command` objects and provides lookup / help."""

    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}

    def register(self, command: Command) -> None:
        self._commands[command.name] = command
        # Also register short aliases for convenience.
        if command.name == "quit":
            self._commands["exit"] = command
            self._commands["q"] = command
        if command.name == "help":
            self._commands["?"] = command

    def get(self, name: str) -> Command | None:
        return self._commands.get(name)

    @property
    def names(self) -> list[str]:
        return sorted(self._commands.keys())

    @property
    def commands(self) -> list[Command]:
        return sorted(self._commands.values(), key=lambda c: c.name)


def build_registry() -> CommandRegistry:
    """Build the default command registry with all built-in commands."""
    reg = CommandRegistry()

    # Face commands
    reg.register(
        Command(
            name="emotion",
            help="Set face emotion",
            params=(
                ParamSpec("emotion", str, help="Emotion name"),
                ParamSpec("intensity", float, required=False, default="1.0", help="Intensity 0-1"),
            ),
            handler=_cmd_emotion,
        )
    )
    reg.register(Command(name="blink", help="Blink eyes", handler=_cmd_blink))
    reg.register(
        Command(
            name="look",
            help="Look direction",
            params=(ParamSpec("direction", str, help="left|right|up|down|center"),),
            handler=_cmd_look,
        )
    )
    reg.register(Command(name="bounce", help="Bounce face animation", handler=_cmd_bounce))
    reg.register(Command(name="smile", help="Smile animation", handler=_cmd_smile))
    reg.register(
        Command(
            name="theme",
            help="Change face theme",
            params=(ParamSpec("name", str, help="Theme name"),),
            handler=_cmd_theme,
        )
    )

    # Body / servo commands
    reg.register(Command(name="wave", help="Wave right arm", handler=_cmd_wave))
    reg.register(Command(name="nod", help="Nod head", handler=_cmd_nod))
    reg.register(Command(name="shrug", help="Shrug gesture", handler=_cmd_shrug))
    reg.register(Command(name="celebrate", help="Celebrate gesture", handler=_cmd_celebrate))
    reg.register(Command(name="greet", help="Greeting gesture", handler=_cmd_greet))
    reg.register(Command(name="arms_open", help="Open both arms", handler=_cmd_arms_open))
    reg.register(Command(name="arms_relax", help="Relax arms to neutral", handler=_cmd_arms_relax))
    reg.register(
        Command(
            name="servo",
            help="Move a servo directly",
            params=(
                ParamSpec("name", str, help="Servo name (pan, tilt, left_arm, right_arm)"),
                ParamSpec("angle", float, help="Target angle in degrees"),
            ),
            handler=_cmd_servo,
        )
    )
    reg.register(Command(name="servos", help="Show all servo angles", handler=_cmd_servos))
    reg.register(
        Command(
            name="release",
            help="Release servo(s)",
            params=(ParamSpec("name", str, required=False, help="Servo name (all if omitted)"),),
            handler=_cmd_release,
        )
    )

    # State
    reg.register(
        Command(
            name="state",
            help="Transition robot state",
            params=(ParamSpec("state", str, help="Target state"),),
            handler=_cmd_state,
        )
    )
    reg.register(Command(name="status", help="Show current state + servos", handler=_cmd_status))

    # Behavior
    reg.register(
        Command(
            name="behavior",
            help="Run a pre-built behavior",
            params=(ParamSpec("name", str, help="Behavior name"),),
            handler=_cmd_behavior,
        )
    )

    # Conversation
    reg.register(
        Command(
            name="speak",
            help="Speak text (via LLM pipeline)",
            params=(ParamSpec("text", str, help="Text to speak"),),
            handler=_cmd_speak,
        )
    )
    reg.register(
        Command(
            name="say",
            help="TTS directly (bypass LLM)",
            params=(ParamSpec("text", str, help="Text to synthesize"),),
            handler=_cmd_say,
        )
    )

    # Sensors
    reg.register(
        Command(
            name="mic",
            help="Toggle mic level meter",
            params=(ParamSpec("on_off", str, required=False, help="on|off"),),
            handler=_cmd_mic,
        )
    )
    reg.register(
        Command(
            name="cam",
            help="Toggle camera view",
            params=(ParamSpec("on_off", str, required=False, help="on|off"),),
            handler=_cmd_cam,
        )
    )

    # Display / config
    reg.register(Command(name="fps", help="Show frame profiler stats", handler=_cmd_fps))
    reg.register(Command(name="config", help="Show current configuration", handler=_cmd_config))
    reg.register(
        Command(
            name="events",
            help="Toggle event log",
            params=(ParamSpec("on_off", str, required=False, help="on|off"),),
            handler=_cmd_events,
        )
    )

    # System
    reg.register(
        Command(
            name="help",
            help="Show command help",
            params=(ParamSpec("command", str, required=False, help="Command name"),),
            handler=_cmd_help,
        )
    )
    reg.register(Command(name="quit", help="Shut down and exit", handler=_cmd_quit))

    return reg


def parse_command(line: str) -> tuple[str, list[str]]:
    """Parse a command line into (command_name, args).

    Uses :func:`shlex.split` for proper quoting.
    """
    stripped = line.strip()
    if not stripped:
        return ("", [])
    parts = shlex.split(stripped)
    if not parts:
        return ("", [])
    return (parts[0].lower(), parts[1:])


__all__ = [
    "Command",
    "CommandContext",
    "CommandRegistry",
    "ParamSpec",
    "build_registry",
    "parse_command",
]
