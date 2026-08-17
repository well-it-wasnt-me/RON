# CLI tools

DeskBot ships a collection of command-line tools for running, testing, and
diagnosing the robot. All commands use mock hardware by default so they work
on any workstation without a Raspberry Pi.

> See also the [Interactive CLI](interactive.md) for the full-screen TUI.

---

## Commands

| Command | Purpose |
|---------|---------|
| `deskbot` | Run the full robot application (default subcommand) |
| `deskbot chat` | Interactive text chat (no audio hardware needed) |
| `deskbot-interactive` | Full-screen TUI with braille face, servo gauges, event log |
| `deskbot-simulate` | Headless simulation (face + body + servo overlay) |
| `deskbot-doctor` | Diagnose environment, hardware, and configuration |
| `deskbot-hardware-check` | Hardware presence and wiring diagnostics |
| `deskbot-calibration` | Servo and display calibration |
| `deskbot-display-test` | GC9A01 wiring smoke test (Pi only) |
| `deskbot-eye-demo` | Cycle through every eye animation |
| `deskbot-face-test` | Face rendering test against mock display |
| `deskbot-profile` | Run robot, collect profiling data, output JSON report |
| `deskbot-learning-status` | Show learning service status |
| `deskbot-learning-train` | Force a training cycle |
| `deskbot-learning-evaluate` | Evaluate a model checkpoint |
| `deskbot-learning-reset` | Reset learning state |
| `deskbot-learning-export` | Export learning data |

---

## Common configuration

All CLI commands read the same `AppSettings` configuration — environment
variables (prefixed `DESKBOT_`), `.env` files, or YAML config:

```bash
# Use mock hardware (default)
DESKBOT_HARDWARE=mock deskbot

# Use a specific face theme
DESKBOT_FACE__THEME=cute deskbot-simulate

# Run against real hardware
DESKBOT_HARDWARE=real deskbot
```

See [Configuration](../reference/config.md) for the full reference.

---

## Makefile shortcuts

The Makefile provides convenient wrappers:

```bash
make run           # deskbot (mock stack)
make simulate      # deskbot-simulate
make doctor        # deskbot-doctor
make interactive   # deskbot-interactive
make test          # full test suite
make check         # lint + typecheck + test
make docs          # build documentation
```

Run `make help` to see all available targets.

---

## deskbot-doctor

The doctor command diagnoses the environment and hardware:

```bash
deskbot-doctor                    # full diagnostic
deskbot-doctor --microphone       # microphone device enumeration + capture test
deskbot-doctor --audio            # speaker test tone
```

The microphone diagnostic:

1. Enumerates all available input devices
2. Shows the PortAudio default input device
3. Shows which device DeskBot would select
4. Opens the device and captures a short sample
5. Reports whether non-zero audio is being received
6. Reports RMS / min / max / overflow statistics
