"""DeskBot interactive TUI - Textual-based terminal user interface.

This is the main application module that wires together the renderer,
command parser, and the running DeskBot robot stack into a full-screen
terminal interface.

Launch with::

    deskbot - interactive

Or::

    python -m robot.cli.interactive
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from robot.behavior.state_machine import StateMachine
from robot.body_language.requests import DEFAULT_CALIBRATION
from robot.cli.interactive.commands import (
    CommandContext,
    build_registry,
    parse_command,
)
from robot.cli.interactive.renderer import (
    frame_to_braille_cropped,
    level_bar,
    servo_dashboard,
)
from robot.config import AppSettings, load_settings
from robot.events.bus import InMemoryEventBus
from robot.events.events import EmotionChanged, StateChanged
from robot.face.themes import get_theme
from robot.hardware.servos.adapter import wrap_servo_controller
from robot.logging import configure_logging, get_logger
from robot.simulation.driver import SimulationDriver
from robot.utils.clock import SystemClock

_log = get_logger("cli.interactive")

# ---------------------------------------------------------------------------
# Try to import Textual; provide a clear message if missing.
# ---------------------------------------------------------------------------
try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.reactive import reactive
    from textual.widgets import Footer, Header, Input, Static

    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False

# ---------------------------------------------------------------------------
# Fallback: minimal non-Textual interactive REPL for environments
# where Textual is not installed.
# ---------------------------------------------------------------------------


class _FallbackREPL:
    """A simple async REPL that works without Textual.

    Used when the ``textual`` package is not installed. Provides the
    same command set but without the full-screen TUI layout.
    """

    def __init__(self, settings: AppSettings, sim: SimulationDriver) -> None:
        self.settings = settings
        self.sim = sim
        self.command_registry = build_registry()
        self._running = True
        self._show_events = True
        self._show_mic = False
        self._show_cam = False

    def write_log(self, msg: str) -> None:
        """Print a message to the terminal (REPL fallback)."""
        print(f"  {msg}")

    @property
    def simulation_driver(self) -> SimulationDriver:
        return self.sim

    def exit(self) -> None:
        self._running = False

    async def run(self) -> None:
        """Run the fallback REPL."""
        print("\n  +----------------------------------+")
        print("  |     DeskBot Interactive CLI      |")
        print("  |     (fallback REPL mode)         |")
        print("  +----------------------------------+\n")
        print("  Type 'help' for commands, 'quit' to exit.\n")

        # Build command context.
        SystemClock()
        ctx = CommandContext(
            app=self,
            settings=self.settings,
            face_animator=self.sim.face,
            body_engine=self.sim.body,
            servo_controller=wrap_servo_controller(self.sim._servo_bus, backend_name="mock"),
            state_machine=StateMachine(bus=InMemoryEventBus()),
            bus=InMemoryEventBus(),
            simulation_driver=self.sim,
        )

        loop = asyncio.get_running_loop()

        while self._running:
            try:
                # Run blocking input() in a thread so it doesn't block the
                # event loop - this keeps async command handlers responsive.
                line = await loop.run_in_executor(None, lambda: input("deskbot> "))
            except (EOFError, KeyboardInterrupt):
                break

            cmd_name, args = parse_command(line)
            if not cmd_name:
                continue

            cmd = self.command_registry.get(cmd_name)
            if cmd is None:
                self.write_log(f"Unknown command: {cmd_name}. Type 'help' for commands.")
                continue
            if cmd.handler is None:
                self.write_log(f"Command '{cmd_name}' has no handler.")
                continue

            try:
                await cmd.handler(ctx, args)
            except Exception as exc:
                self.write_log(f"Error: {exc}")

        print("\n  Goodbye!")


# ---------------------------------------------------------------------------
# Textual TUI (requires textual package)
# ---------------------------------------------------------------------------

if HAS_TEXTUAL:
    from textual.widgets import RichLog as _RichLog

    class FaceDisplay(Static):
        """Widget that renders the face as braille art."""

        content: reactive[str] = reactive("")

        def render(self) -> str:
            return self.content

    class ServoPanel(Static):
        """Widget that renders servo gauges."""

        content: reactive[str] = reactive("")

        def render(self) -> str:
            return self.content

    class EventLog(_RichLog):
        """Scrollable event log."""

        pass

    class MicBar(Static):
        """Microphone level meter."""

        content: reactive[str] = reactive("")

        def render(self) -> str:
            return self.content

    class CamView(Static):
        """Camera braille-art view."""

        content: reactive[str] = reactive("")

        def render(self) -> str:
            return self.content

    class DeskBotTUI(App[Any]):
        """Full-screen interactive terminal UI for DeskBot."""

        TITLE = "DeskBot"
        CSS = """
        #face-display {
            height: auto;
            max-height: 30;
            border: round $primary;
            padding: 0 1;
        }
        #servo-panel {
            height: auto;
            max-height: 8;
            border: round $primary;
            padding: 0 1;
        }
        #event-log {
            height: 1fr;
            border: round $primary;
        }
        #mic-bar {
            display: none;
            height: auto;
            max-height: 3;
            border: round $primary;
            padding: 0 1;
        }
        #mic-bar.visible {
            display: block;
        }
        #cam-view {
            display: none;
            height: auto;
            max-height: 14;
            border: round $primary;
            padding: 0 1;
        }
        #cam-view.visible {
            display: block;
        }
        #command-input {
            height: 3;
            border: round $primary;
            padding: 0 1;
            margin: 0;
        }
        #command-input:focus {
            border: round $accent;
        }
        """

        BINDINGS = [  # noqa: RUF012
            Binding("ctrl+q", "quit", "Quit", show=True),
            Binding("ctrl+l", "clear_log", "Clear log", show=False),
        ]

        # Reactive state
        _face_text: reactive[str] = reactive("")
        _servo_text: reactive[str] = reactive("")
        _mic_text: reactive[str] = reactive("")
        _cam_text: reactive[str] = reactive("")
        _show_events: bool = True
        _show_mic: bool = False
        _show_cam: bool = False

        def __init__(
            self,
            settings: AppSettings,
            sim: SimulationDriver,
            **kwargs: object,
        ) -> None:
            super().__init__(**cast("dict[str, Any]", kwargs))
            self.settings = settings
            self.sim = sim
            self.command_registry = build_registry()
            self._last_state = "boot"
            self._last_emotion = "neutral"
            self._last_intensity = 1.0
            self._command_context = CommandContext(
                app=self,
                settings=settings,
                face_animator=sim.face,
                body_engine=sim.body,
                servo_controller=wrap_servo_controller(sim._servo_bus, backend_name="mock"),
                state_machine=StateMachine(bus=InMemoryEventBus()),
                bus=InMemoryEventBus(),
                simulation_driver=sim,
            )

        @property
        def simulation_driver(self) -> SimulationDriver:
            """Expose the simulation driver for command handlers.

            Command handlers access ``ctx.app.simulation_driver``; this
            property provides a stable interface regardless of whether the
            app is the TUI or the fallback REPL (both store it as
            ``self.sim``).
            """
            return self.sim

        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal():
                with Vertical():
                    yield FaceDisplay(id="face-display")
                    yield ServoPanel(id="servo-panel")
                    yield MicBar(id="mic-bar")
                    yield CamView(id="cam-view")
                with Vertical():
                    yield EventLog(id="event-log", highlight=True, markup=True)
                    yield Input(
                        id="command-input",
                        placeholder="Type a command (help for list)…",
                    )
            yield Footer()

        def on_mount(self) -> None:
            self._start_update_loop()
            # Focus the command input so the user can type immediately.
            self.query_one("#command-input", Input).focus()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            """Handle a command entered in the input bar."""
            if event.input.id != "command-input":
                return
            line = event.value.strip()
            # Clear the input field after submission.
            event.input.value = ""
            if not line:
                return

            cmd_name, args = parse_command(line)
            if not cmd_name:
                return

            cmd = self.command_registry.get(cmd_name)
            if cmd is None:
                self.write_log(f"Unknown command: {cmd_name}. Type 'help' for commands.")
                return
            if cmd.handler is None:
                self.write_log(f"Command '{cmd_name}' has no handler.")
                return

            async def _run_command() -> None:
                try:
                    handler = cast("Any", cmd.handler)
                    await handler(self._command_context, args)
                except Exception as exc:
                    self.write_log(f"Error: {exc}")

            self.run_worker(_run_command(), exclusive=False)

        def _start_update_loop(self) -> None:
            """Start the periodic display update."""
            self.set_interval(1.0 / 10, self._update_display)  # 10 FPS for TUI

        def _update_display(self) -> None:
            """Update face and servo displays from the simulation driver."""
            # Advance simulation one tick.  sim.step() internally calls
            # face.step(drift=True), so the face animates naturally.  We
            # extract the face portion from the composite for rendering.
            composite = self.sim.step()
            face_w = self.sim.face_size
            x_off = (self.sim.width - face_w) // 2
            face_pixels = bytearray(face_w * face_w * 3)
            for _y in range(face_w):
                src = (_y * self.sim.width + x_off) * 3
                dst = _y * face_w * 3
                face_pixels[dst : dst + face_w * 3] = composite.pixels[src : src + face_w * 3]
            face_lines = frame_to_braille_cropped(
                bytes(face_pixels),
                face_w,
                face_w,
                threshold=80,
                term_cols=60,
                term_rows=28,
            )
            self._face_text = "\n".join(face_lines)
            try:
                face_widget = self.query_one("#face-display", FaceDisplay)
                face_widget.content = self._face_text
            except Exception:
                pass  # Widget may not be mounted yet on first frame.

            # Servos
            try:
                if self.sim._servo_bus is not None:
                    angles = self.sim._servo_bus.all_angles()
                    cal = {
                        name: (cal.min_angle, cal.max_angle)
                        for name, cal in DEFAULT_CALIBRATION.items()
                    }
                    dash = servo_dashboard(angles, cal)
                    self._servo_text = "\n".join(dash)
                    servo_widget = self.query_one("#servo-panel", ServoPanel)
                    servo_widget.content = self._servo_text
            except Exception:
                pass  # Widget may not be mounted yet on first frame.

            # Mic bar (placeholder - would need real audio RMS data)
            if self._show_mic:
                self._mic_text = level_bar(0.0, width=40)
                try:
                    mic_widget = self.query_one("#mic-bar", MicBar)
                    mic_widget.content = self._mic_text
                except Exception:
                    pass

            # Camera view (placeholder)
            try:
                cam_widget = self.query_one("#cam-view", CamView)
                if self._show_cam:
                    cam_widget.add_class("visible")
                else:
                    cam_widget.remove_class("visible")
            except Exception:
                pass

        async def _on_event(self, event: object) -> None:
            """Event bus subscriber for the TUI."""
            if not self._show_events:
                return
            try:
                event_log = self.query_one("#event-log", EventLog)
                from dataclasses import fields as dc_fields

                event_type = type(event).__name__
                parts = [f"[bold]{event_type}[/bold]"]
                for f in dc_fields(cast("Any", event)):
                    val = getattr(event, f.name)
                    parts.append(f"{f.name}={val}")
                event_log.write(" ".join(parts))
            except Exception:
                pass  # Widget may not be mounted yet.

            # Track state/emotion for display.
            if isinstance(event, StateChanged):
                self._last_state = event.current.value
            if isinstance(event, EmotionChanged):
                self._last_emotion = event.current.value
                self._last_intensity = event.intensity

        def write_log(self, msg: str) -> None:
            """Write a message to the event log panel.

            Named ``write_log`` instead of ``log`` to avoid shadowing
            Textual's built-in ``App.log`` property.
            """
            try:
                event_log = self.query_one("#event-log", EventLog)
                event_log.write(f"[dim]{msg}[/dim]")
            except Exception:
                # Widget may not be mounted yet; fall back to terminal.
                _log.info("write_log (no widget): %s", msg)

        def action_clear_log(self) -> None:
            try:
                event_log = self.query_one("#event-log", EventLog)
                event_log.clear()
            except Exception:
                pass

        def exit(
            self,
            result: Any | None = None,
            return_code: int = 0,
            message: Any | None = None,
        ) -> None:
            """Shut down the TUI."""
            self.sim.stop()
            super().exit(result=result, return_code=return_code, message=message)


# ---------------------------------------------------------------------------
# Public entry point helpers
# ---------------------------------------------------------------------------


def create_simulation_driver(settings: AppSettings) -> SimulationDriver:
    """Build a SimulationDriver from settings."""
    display_cfg = settings.displays
    return SimulationDriver(
        width=display_cfg.width,
        height=int(display_cfg.width * 1.33),
        face_size=display_cfg.width,
        fps=display_cfg.fps,
    )


def run_interactive(settings: AppSettings | None = None) -> None:
    """Sync entry point for the interactive TUI.

    If ``textual`` is installed, launches the full TUI; otherwise
    falls back to a simple REPL.
    """
    import asyncio

    if settings is None:
        settings = load_settings()
    configure_logging(settings)

    sim = create_simulation_driver(settings)

    # Apply theme from config.
    theme_name = getattr(settings, "face", None)
    theme_cfg = getattr(theme_name, "theme", "vector") if theme_name else "vector"
    try:
        theme = get_theme(theme_cfg)
    except (KeyError, Exception):
        theme = get_theme("vector")
    sim.theme = theme
    sim.face.theme = theme

    if HAS_TEXTUAL:
        app = DeskBotTUI(settings=settings, sim=sim)
        app.run()
    else:
        repl = _FallbackREPL(settings=settings, sim=sim)
        asyncio.run(repl.run())


__all__ = ["DeskBotTUI", "create_simulation_driver", "run_interactive"]
