# Deployment

DeskBot supports local development, Docker-based deployment, and Raspberry Pi
deployment through the included systemd installer.

## Docker

The Compose stack is intended for application/API development and mock
hardware:

```bash
docker compose up --build
```

The FastAPI service is exposed on port 8000 by the Compose configuration.

The browser dashboard is available at:

```text
http://localhost:8000/        # main web dashboard
http://localhost:8000/settings/   # hardware test page
http://localhost:8000/calibration/  # servo/display calibration
```

Conversation persistence uses the configured application storage/volume.

## Raspberry Pi

The repository includes:

```text
scripts/install.sh
deploy/systemd/deskbot.service
```

The installer provisions the system packages, audio stack, service user, the
project tree under `/opt/deskbot`, a `uv`-managed virtualenv, and the systemd
unit. Review the script before running it on a production device.

### Running the installer

```bash
sudo ./scripts/install.sh
```

It must be run as root. It is **idempotent**: re-running detects
already-installed packages, an existing service user, an existing virtualenv,
and an existing `/etc/deskbot/deskbot.env` and skips the corresponding work.
This makes it safe to re-run to apply updates or finish an interrupted install.

### Optional flags and environment variables

```bash
# First-time / idempotent install
sudo ./scripts/install.sh

# Pull the latest from the git remote (origin), then (re)install
sudo ./scripts/install.sh --update

# Force a clean rebuild of the virtualenv (otherwise the existing .venv is reused)
sudo ./scripts/install.sh --recreate-venv

# Combine: pull latest and rebuild the venv
sudo ./scripts/install.sh --update --recreate-venv

# Override the install prefix and service user
DESKBOT_INSTALL_DIR=/opt/deskbot DESKBOT_SERVICE_USER=deskbot sudo -E ./scripts/install.sh

# Disable the uv download cache (saves disk space at the cost of re-downloading)
DESKBOT_NO_CACHE=1 sudo -E ./scripts/install.sh
```

| Flag / Variable | Default | Purpose |
|------------------|---------|---------|
| `-h`, `--help` | - | Show usage and exit (works without root) |
| `--update` | off | Pull latest from the `origin` git remote before installing |
| `--recreate-venv` | off | Destroy and rebuild `.venv` instead of reusing it |
| `DESKBOT_INSTALL_DIR` | `/opt/deskbot` | Install prefix |
| `DESKBOT_SERVICE_USER` | `deskbot` | System service user |
| `DESKBOT_NO_CACHE` | `0` | Pass `--no-cache` to `uv sync` (set `1` to enable) |

`--update` runs the `git pull` **as the owner of the source checkout** (never as
root), so your working tree keeps its existing ownership. It uses
`git pull --ff-only`; if the pull cannot fast-forward (local changes or a
diverged branch) it is skipped with a warning and the current working tree is
installed. If `scripts/install.sh` itself changed upstream, the installer tells
you to re-run so the new logic takes effect. Run `sudo ./scripts/install.sh --help`
for the full reference.

### How it installs

- **System packages** are installed with `--no-install-recommends`; only
  packages not already present are installed. `pulseaudio-module-bluetooth`
  is installed separately and tolerated if it is unavailable on releases
  where PipeWire is the default.
- **uv** is installed and run **as the service user** (never as root), so the
  virtualenv is owned by `deskbot` from the start. The virtualenv is pinned to
  Python 3.12; if the system Python is older (e.g. 3.11 on Bookworm), `uv`
  fetches a managed CPython 3.12.
- **`/etc/deskbot/deskbot.env`** is seeded from `.env.example` only if it does
  not already exist - your configuration is never overwritten on re-runs.
- **SPI and I²C** are enabled on Raspberry Pi (via `raspi-config` when
  available, otherwise by editing `/boot/firmware/config.txt` or
  `/boot/config.txt`).
- **PipeWire/WirePlumber** is disabled and masked so the standalone
  PulseAudio daemon owns the audio socket.

### Install log

All installer output is tee'd to:

```text
/var/log/deskbot-install.log
```

If the installer fails, the failing command and line number are reported and
written to this log for troubleshooting.

### After installation

```bash
# Edit configuration (a default was installed from .env.example if absent)
sudo nano /etc/deskbot/deskbot.env

# On Raspberry Pi, reboot to activate SPI/I²C
sudo reboot

# Start and check the service
sudo systemctl start deskbot
systemctl status deskbot
journalctl -u deskbot -f
```

## Real hardware configuration

Set:

```env
DESKBOT_HARDWARE=real
```

and select concrete backends:

```env
DESKBOT_DISPLAYS__BACKEND=circuitpython
DESKBOT_SERVOS__BACKEND=gpio
```

Enable SPI/I²C as required by the chosen hardware.

## Audio (PulseAudio)

DeskBot uses **standalone PulseAudio** for audio output, not
PipeWire/WirePlumber. The installer handles this automatically.

See [Audio Architecture](audio-architecture.md) for the full design
and troubleshooting guide.

### Bluetooth speaker setup

```bash
# Pair a Bluetooth speaker
bluetoothctl
[bluetoothctl] power on
[bluetoothctl] agent on
[bluetoothctl] scan on
[bluetoothctl] pair <MAC>
[bluetoothctl] trust <MAC>
[bluetoothctl] connect <MAC>

# Verify the PulseAudio sink appeared
pactl list short sinks

# Set as default sink (optional)
pactl set-default-sink bluez_sink.<device>.a2dp_sink
```

Configure DeskBot to use the Bluetooth speaker:

```env
DESKBOT_AUDIO__BACKEND=bluetooth
DESKBOT_AUDIO__BLUETOOTH_MAC=<MAC>
```

## API server

The API is enabled by default and listens on:

```env
DESKBOT_API__HOST=0.0.0.0
DESKBOT_API__PORT=8000
DESKBOT_API__ENABLED=true
```

The application can start the API as part of its lifecycle.

## Web dashboard

DeskBot includes a multi-page web dashboard at the root URL:

| URL | Page |
|-----|------|
| `/` | Dashboard (system overview, robot state, audio, events) |
| `/#/logs` | Live log viewer with filtering and search |
| `/#/audio` | Audio pipeline status and test controls |
| `/#/bluetooth` | Bluetooth speaker status |
| `/#/config` | Configuration view (read-only) |
| `/#/system` | System information |
| `/#/controls` | Robot controls (emotion, state, speak) |
| `/settings/` | Hardware test page (camera, mic, audio output) |
| `/calibration/` | Servo and display calibration |

## Security

The API currently enables permissive CORS for dashboard access. Do not expose
an unauthenticated instance directly to an untrusted network.

Keep API keys and broker passwords in environment/secret configuration rather
than committing them to YAML or source control.
