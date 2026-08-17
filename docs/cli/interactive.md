# Interactive CLI

The DeskBot interactive CLI is a full-screen terminal UI that provides
real-time visualisation of the robot's face, servos, microphone, and
camera, plus a command prompt for direct control.

## Quick start

```bash
# Install dev dependencies (includes textual)
make install-dev

# Launch the interactive TUI
deskbot-interactive
```

If `textual` is not installed, the CLI falls back to a simple REPL that
still provides the same command set without the full-screen layout.

You can also run it as a module:

```bash
python -m robot.cli.interactive
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  DeskBotTUI (Textual App)                                │
│                                                          │
│  ┌─────────────────┐  ┌──────────────────────────────┐  │
│  │  FaceDisplay    │  │  ServoPanel                   │  │
│  │  (braille art)  │  │  Pan  90.0° [████████░░░░░░]  │  │
│  │                 │  │  Tilt 90.0° [████████░░░░░░]  │  │
│  │  ⣿⣿⣿⣿⣿⣿⣿⣿⣿   │  │  L.Arm 90.0° [████████░░░░░░]  │  │
│  │  ⣿⣿⡿⡿⣿⣿⣿⣿⣿   │  │  R.Arm 90.0° [████████░░░░░░]  │  │
│  │  ⣿⣿⣿⣿⣿⣿⣿⣿⣿   │  └──────────────────────────────┘  │
│  │  ⣿⣿⣿▁▁▁⣿⣿⣿⣿   │                                    │
│  └─────────────────┘  ┌──────────────────────────────┐  │
│                        │  MicBar (audio level meter)   │  │
│  ┌─────────────────┐  └──────────────────────────────┘  │
│  │  CamView        │  ┌──────────────────────────────┐  │
│  │  (braille art)  │  │  EventLog                     │  │
│  │                 │  │  StateChanged idle -> curious   │  │
│  └─────────────────┘  │  EmotionChanged neutral ->     │  │
│                        │  ServoMoved pan -> 75.0°      │  │
│                        └──────────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────────┐│
│  │  > emotion happy                                     ││
│  └──────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────┘
```

### Key components

| Component | Module | Purpose |
|-----------|--------|---------|
| `renderer.py` | Braille-art + servo bars | Converts pixel buffers and numeric state into terminal characters |
| `commands.py` | Command parser + registry | ~30 commands for face, body, state, conversation, sensors |
| `app.py` | Textual TUI + fallback REPL | Main application: widget layout, async robot loop, event handling |

## Braille-art rendering

The face and camera are rendered using **Unicode braille characters**
(U+2800–U+283F). Each braille character represents a 2×4 grid of
on/off pixels, giving a resolution of roughly 80×40 characters from a
240×320 source image.

The luminance threshold is configurable (default: 80). Pixels above the
threshold appear as raised braille dots; pixels below are blank.

```python
from robot.cli.interactive.renderer import frame_to_braille

lines = frame_to_braille(
    pixels=frame.pixels,   # RGB888 bytes
    width=frame.width,     # 240
    height=frame.height,   # 320
    threshold=80,          # luminance cutoff
    term_cols=60,          # constrain to terminal width
    term_rows=20,          # constrain to terminal height
)
```

For terminals that don't support braille, `frame_to_half_blocks`
provides a coarser fallback using block characters (` ░▒▓█`).

## Command reference

### Face commands

| Command | Description |
|---------|-------------|
| `emotion <name> [intensity]` | Set face emotion (neutral, happy, curious, thinking, sleepy, embarrassed, excited, sad, surprised, angry) |
| `blink` | Blink both eyes |
| `look <direction>` | Look left, right, up, down, or center |
| `bounce` | Bounce the face animation |
| `smile` | Smile animation |
| `theme <name>` | Change the face theme (minimal, cute, pixel, retro_lcd, wireframe, vector) |

### Body / servo commands

| Command | Description |
|---------|-------------|
| `wave` | Wave the right arm |
| `nod` | Nod the head |
| `shrug` | Shrug gesture |
| `celebrate` | Celebrate gesture |
| `greet` | Greeting gesture |
| `arms_open` | Open both arms |
| `arms_relax` | Relax arms to neutral |
| `servo <name> <angle>` | Move a servo directly (pan, tilt, left_arm, right_arm) |
| `servos` | Show all servo angles |
| `release [name]` | Release servo(s); all if name omitted |

### State commands

| Command | Description |
|---------|-------------|
| `state <name>` | Transition robot state (boot, idle, curious, listening, thinking, speaking, sleeping) |
| `status` | Show current state, emotion, and servo positions |

### Behaviour commands

| Command | Description |
|---------|-------------|
| `behavior <name>` | Run a pre-built behaviour sequence (greeting, thinking, listening, sleeping, excited, surprised) |

### Conversation commands

| Command | Description |
|---------|-------------|
| `speak <text>` | Send text through the full conversation pipeline (STT -> LLM -> TTS) |
| `say <text>` | Synthesize text directly via TTS (bypasses LLM) |

### Sensor commands

| Command | Description |
|---------|-------------|
| `mic [on|off]` | Toggle microphone level meter |
| `cam [on|off]` | Toggle camera ASCII view |

### Display / config commands

| Command | Description |
|---------|-------------|
| `fps` | Show frame profiler stats (if enabled) |
| `config` | Show current configuration (sensitive fields masked) |
| `events [on|off]` | Toggle event log visibility |

### System commands

| Command | Description |
|---------|-------------|
| `help [command]` | Show command help; optionally for a specific command |
| `quit` / `exit` / `q` | Shut down the robot and exit |

## The renderer module

### `frame_to_braille`

```python
def frame_to_braille(
    pixels: bytes,
    width: int,
    height: int,
    threshold: int = 80,
    term_cols: int | None = None,
    term_rows: int | None = None,
) -> list[str]
```

Convert an RGB888 pixel buffer into braille-art lines. Each output
character represents a 2×4 pixel block. The `threshold` parameter
controls the luminance cutoff (0–255). Use `term_cols` and
`term_rows` to constrain output to your terminal size.

### `frame_to_half_blocks`

```python
def frame_to_half_blocks(
    pixels: bytes,
    width: int,
    height: int,
    term_cols: int | None = None,
    term_rows: int | None = None,
) -> list[str]
```

Coarser fallback using block characters. Maps luminance to five
density levels: space, ░, ▒, ▓, █.

### `servo_bar`

```python
def servo_bar(
    name: str,
    angle: float,
    min_angle: float = 0.0,
    max_angle: float = 180.0,
    width: int = 20,
) -> str
```

Render a single servo angle as a horizontal gauge string like
`"Pan  90.0° [████████░░░░░░░░░░░░]"`.

### `servo_dashboard`

```python
def servo_dashboard(
    angles: dict[str, float],
    calibration: dict[str, tuple[float, float]] | None = None,
) -> list[str]
```

Render all four servos as a four-line dashboard.

### `level_bar`

```python
def level_bar(
    rms: float,
    peak: float | None = None,
    width: int = 30,
    max_val: float = 1.0,
) -> str
```

Render an audio level meter with sub-character precision.

### `state_line`

```python
def state_line(state: str, emotion: str = "", intensity: float = 1.0) -> str
```

Render a one-line state + emotion summary with emoji icons.

### `config_summary`

```python
def config_summary(settings_dict: dict) -> list[str]
```

Render a compact config summary, masking `api_key` and `password`
fields. Output is capped at 30 lines.

## The command module

### `CommandContext`

```python
@dataclass
class CommandContext:
    app: Any               # The TUI or REPL instance
    settings: Any          # AppSettings
    face_animator: Any     # FaceAnimator
    body_engine: Any       # BodyLanguageEngine
    servo_controller: Any  # ServoController
    state_machine: Any     # StateMachine
    bus: Any               # InMemoryEventBus
    conversation: Any     # ConversationService
    display: Any           # Display
    simulation_driver: Any # SimulationDriver (if in sim mode)
    microphone: Any        # Microphone
    camera: Any            # Camera
```

Context object passed to every command handler. Not all fields are
guaranteed to be non-`None`; handlers must check before using optional
components.

### `CommandRegistry`

```python
reg = build_registry()       # Get the default command set
cmd = reg.get("emotion")     # Look up a command
names = reg.names            # Sorted list of command names
cmds = reg.commands          # Sorted list of Command objects
```

### `parse_command`

```python
name, args = parse_command('speak "hello world"')
# name = "speak", args = ["hello world"]
```

Uses `shlex.split` for proper quoting.

## Fallback REPL

When `textual` is not installed, the CLI automatically falls back to a
simple async REPL. It provides the same command set but without the
full-screen TUI layout (no braille face, no servo gauges). This
ensures the interactive CLI is always usable, even on minimal
installations.

```bash
# The fallback REPL auto-activates when textual is missing
pip uninstall textual   # Remove textual
deskbot-interactive      # Falls back to REPL mode
```

## Adding new commands

1. Write an `async def _cmd_foo(ctx: CommandContext, args: list[str]) -> None` handler.
2. Add a `Command(name="foo", help="...", handler=_cmd_foo)` to the registry in `build_registry()`.
3. The command automatically appears in `help` output and tab-completion (when textual is available).

## Configuration

The interactive CLI uses the same `AppSettings` as the main application.
It can be configured via environment variables (prefixed with
`DESKBOT_`), `.env` files, or YAML config:

```bash
# Use a specific face theme
DESKBOT_FACE__THEME=cute deskbot-interactive

# Set display size
DESKBOT_DISPLAYS__WIDTH=240 deskbot-interactive

# Enable mock hardware
DESKBOT_HARDWARE=mock deskbot-interactive
```

## Key bindings (Textual mode)

| Key | Action |
|-----|--------|
| `Ctrl+Q` | Quit |
| `Ctrl+L` | Clear event log |

## Dependencies

- **Required**: `textual>=0.79` (included in dev extras)
- **Fallback**: Works without `textual` in REPL mode
