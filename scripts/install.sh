#!/usr/bin/env bash
# ============================================================================
# DeskBot installation script for Raspberry Pi OS (Bookworm / Trixie)
# and Debian/Ubuntu aarch64.
#
# Run as root:
#   sudo ./scripts/install.sh
#
# Optional flags / env vars:
#   --recreate-venv          Destroy and rebuild the .venv instead of reusing it
#   DESKBOT_INSTALL_DIR=...  Override the install prefix (default /opt/deskbot)
#   DESKBOT_SERVICE_USER=... Override the service user (default deskbot)
#   DESKBOT_NO_CACHE=1       Pass --no-cache to uv sync (default: keep cache)
#
# This script:
#   1. Installs system packages (Python, audio, vision, GPIO, TTS, …)
#   2. Enables SPI and I2C on Raspberry Pi
#   3. Creates a system service user
#   4. Copies the project to /opt/deskbot
#   5. Creates a uv virtualenv and installs all Python dependencies
#   6. Installs and starts a systemd service
#
# Design notes:
#   * Idempotent: re-running detects already-installed packages, an existing
#     venv, an existing service user, and an existing /etc/deskbot/deskbot.env
#     and skips the corresponding work.
#   * uv is installed and run AS the service user (never as root), so the
#     virtualenv is owned by deskbot from the start and needs no chown.
#   * All output is tee'd to /var/log/deskbot-install.log for troubleshooting.
# ============================================================================
set -euo pipefail

# Resolve the project root from the script location so the installer does not
# depend on the caller's current working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

show_help() {
cat <<'HELP'
DeskBot installer - provisions and updates a DeskBot installation.

Usage:
  sudo ./scripts/install.sh [OPTIONS]

Description:
  Installs system packages, configures audio (PulseAudio), creates a service
  user, copies the project to the install prefix, builds a uv-managed
  virtualenv (Python 3.12), and installs/enables a systemd unit.

  Idempotent: re-running preserves already-installed packages, the existing
  service user, an existing virtualenv, and an existing /etc/deskbot/deskbot.env.

Options:
  -h, --help            Show this help and exit.
      --update          Pull the latest from the git remote (origin) before
                        installing. The pull runs as the owner of the source
                        checkout (never as root), so your working tree stays
                        owned by you. Requires running from a git checkout.
                        If the pull cannot fast-forward (local changes or a
                        diverged branch), it is skipped with a warning and the
                        current working tree is installed.
      --recreate-venv   Destroy and rebuild the .venv instead of reusing it.

Environment variables:
  DESKBOT_INSTALL_DIR    Install prefix (default: /opt/deskbot).
  DESKBOT_SERVICE_USER   System service user (default: deskbot).
  DESKBOT_NO_CACHE=1     Pass --no-cache to uv sync (default: cache enabled).

Files:
  /opt/deskbot                   Install prefix (overridable).
  /etc/deskbot/deskbot.env       Service environment file (seeded from
                                 .env.example if absent; never overwritten).
  /var/log/deskbot-install.log   Full installer log.

Examples:
  sudo ./scripts/install.sh                # first-time / idempotent install
  sudo ./scripts/install.sh --update       # pull latest, then (re)install
  sudo ./scripts/install.sh --recreate-venv --update
  DESKBOT_NO_CACHE=1 sudo -E ./scripts/install.sh --update

Run as root. See docs/deployment.md for details.
HELP
}

# Allow --help without requiring root (parsed fully again below).
for _a in "$@"; do
  case "${_a}" in
    --help|-h) show_help; exit 0 ;;
  esac
done

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [[ ${EUID} -ne 0 ]]; then
  echo "Run as root: sudo ./scripts/install.sh" >&2
  exit 1
fi

# Detect Raspberry Pi (vs generic Debian/Ubuntu).
is_pi=false
if [[ -f /proc/device-tree/model ]] && grep -qi "raspberry pi" /proc/device-tree/model 2>/dev/null; then
  is_pi=true
fi

# Detect OS / architecture for sanity messaging.
ARCH="$(uname -m)"
. /etc/os-release 2>/dev/null || true
OS_ID="${ID:-unknown}"

echo "============================================"
echo "  DeskBot installer"
echo "  Raspberry Pi detected : ${is_pi}"
echo "  OS                   : ${OS_ID} ${VERSION_ID:-}"
echo "  Architecture         : ${ARCH}"
echo "============================================"

case "${OS_ID}" in
  debian|ubuntu|raspbian)
    : # supported
    ;;
  *)
    echo "[WARN] Unsupported OS '${OS_ID}': this installer targets Debian/Ubuntu/RPi OS." >&2
    echo "       Proceeding, but some package names may not match." >&2
    ;;
esac

# System Python version (informational; uv can fetch a managed 3.12).
if command -v python3 >/dev/null 2>&1; then
  PY_VER="$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
  if [[ "${PY_VER}" < "3.12" ]]; then
    echo "[INFO] System Python is ${PY_VER}; uv will fetch a managed CPython 3.12 for the venv."
  fi
fi

# ---------------------------------------------------------------------------
# Configurable paths / user
# ---------------------------------------------------------------------------
INSTALL_DIR="${DESKBOT_INSTALL_DIR:-/opt/deskbot}"
SERVICE_USER="${DESKBOT_SERVICE_USER:-deskbot}"

# --- --update: pull latest from git (as the checkout owner, never root) ------
do_update() {
  echo ">>> --update: pulling latest from git..."
  if ! command -v git >/dev/null 2>&1; then
    echo "    [WARN] git is not installed; --update has nothing to pull."
    return 0
  fi
  if ! git -C "${SCRIPT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "    [WARN] ${SCRIPT_DIR} is not a git checkout; --update has nothing to pull."
    echo "           Run the installer from a git clone to use --update."
    return 0
  fi
  REMOTE="$(git -C "${SCRIPT_DIR}" remote get-url origin 2>/dev/null || true)"
  if [[ -z "${REMOTE}" ]]; then
    echo "    [WARN] no 'origin' remote in ${SCRIPT_DIR}; --update cannot pull."
    echo "           Add a remote or run without --update."
    return 0
  fi
  echo "    remote: ${REMOTE}"

  # Pull as the owner of the checkout so the working tree stays owned by them
  # (a root-owned pull would leave root-owned files in the user's clone).
  OWNER="$(stat -c '%U' "${SCRIPT_DIR}")"
  BEFORE="$(git -C "${SCRIPT_DIR}" rev-parse --short HEAD 2>/dev/null || true)"
  SELF_BEFORE="$(git -C "${SCRIPT_DIR}" hash-object "${SCRIPT_DIR}/scripts/install.sh" 2>/dev/null || true)"

  set +e
  sudo -u "${OWNER}" -H git -C "${SCRIPT_DIR}" fetch origin 2>&1 | sed 's/^/    fetch: /'
  sudo -u "${OWNER}" -H git -C "${SCRIPT_DIR}" pull --ff-only 2>&1 | sed 's/^/    pull:  /'
  rc=$?
  set -e
  if [[ ${rc} -ne 0 ]]; then
    echo "    [WARN] git pull --ff-only failed (local changes or diverged branch)."
    echo "           Installing the current working tree as-is."
    echo "           Resolve locally (commit/stash) and re-run --update to get new code."
    return 0
  fi

  AFTER="$(git -C "${SCRIPT_DIR}" rev-parse --short HEAD 2>/dev/null || true)"
  echo "    updated: ${BEFORE:-?} -> ${AFTER:-?}"

  # If this installer itself changed upstream, the running (old) copy cannot
  # apply the new logic; tell the user to re-run.
  SELF_AFTER="$(git -C "${SCRIPT_DIR}" hash-object "${SCRIPT_DIR}/scripts/install.sh" 2>/dev/null || true)"
  if [[ -n "${SELF_BEFORE}" && -n "${SELF_AFTER}" && "${SELF_BEFORE}" != "${SELF_AFTER}" ]]; then
    echo "    [NOTICE] scripts/install.sh changed upstream."
    echo "             Re-run: sudo ./scripts/install.sh"
  fi
}

# --- Argument parsing -------------------------------------------------------
RECREATE_VENV=0
DO_UPDATE=0
for arg in "$@"; do
  case "${arg}" in
    --recreate-venv) RECREATE_VENV=1 ;;
    --update)        DO_UPDATE=1 ;;
    --help|-h)       show_help; exit 0 ;;
    *)
      echo "Unknown argument: ${arg} (try --help)" >&2
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Logging: tee everything to a log file for troubleshooting
# ---------------------------------------------------------------------------
LOG_FILE=/var/log/deskbot-install.log
: > "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[install.sh $(date -Is)] starting (args: $*)"
trap 'rc=$?; echo "[install.sh] FAILED: command \"$BASH_COMMAND\" exited $rc (line $LINENO). Log: ${LOG_FILE}" >&2' ERR

# Run --update (if requested) before any install step so the copy below
# picks up the freshly pulled tree.
if [[ "${DO_UPDATE}" == "1" ]]; then
  do_update
fi

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo ">>> Installing system packages…"

# Core build / runtime
APT_PACKAGES=(
  curl
  git
  rsync
  ca-certificates
  python3
  python3-pip
  python3-venv
)

# Audio: PulseAudio (intentional audio backend, not PipeWire/WirePlumber)
# DeskBot uses PulseAudio for reliable Bluetooth A2DP audio playback.
# See docs/audio-architecture.md for the rationale.
# NOTE: pulseaudio-module-bluetooth is installed separately as an optional
# package because it is absent on some releases (e.g. where PipeWire is the
# default) and must not abort the whole install.
APT_PACKAGES+=(
  libportaudio2
  alsa-utils
  pulseaudio
  pulseaudio-utils
)

# Bluetooth stack (BlueZ for device pairing and A2DP)
APT_PACKAGES+=(
  bluetooth
  bluez
)

# Vision: OpenCV runtime libraries.
# opencv-python-headless bundles most of its own libs, but still needs
# GLib and the OpenMP runtime (also used by ONNX Runtime for wakeword).
APT_PACKAGES+=(
  libglib2.0-0
  libgomp1
)

# Numerical: BLAS/LAPACK for numpy / scipy
APT_PACKAGES+=(
  libatlas3-base
  libopenblas0
)

# TTS: espeak-ng (used by the espeak TTS provider)
APT_PACKAGES+=(
  espeak-ng
)

# Camera: Video4Linux utilities (optional, handy for debugging)
APT_PACKAGES+=(
  v4l-utils
)

# I2C tools (for PCA9685 servo diagnostics)
APT_PACKAGES+=(
  i2c-tools
)

# Raspberry Pi-specific: GPIO library (lgpio) and raspi-config
if [[ "${is_pi}" == true ]]; then
  APT_PACKAGES+=(
    liblgpio1
    raspi-config
  )
fi

# Install only the packages that are not already present. apt is idempotent,
# but this reports what actually changes and lets us skip the apt pass
# entirely when nothing is missing.
install_pkgs() {
  local optional="$1"; shift
  local missing=()
  local already=()
  for pkg in "$@"; do
    if dpkg -s "$pkg" >/dev/null 2>&1; then
      already+=("$pkg")
    else
      missing+=("$pkg")
    fi
  done
  if [[ ${#already[@]} -gt 0 ]]; then
    echo "    already present: ${already[*]}"
  fi
  if [[ ${#missing[@]} -eq 0 ]]; then
    echo "    nothing to install"
    return 0
  fi
  echo "    installing: ${missing[*]}"
  if [[ "$optional" == "1" ]]; then
    apt-get install -y --no-install-recommends "${missing[@]}" || \
      echo "    [WARN] some optional packages unavailable on this OS: ${missing[*]}"
  else
    apt-get install -y --no-install-recommends "${missing[@]}"
  fi
}

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
install_pkgs 0 "${APT_PACKAGES[@]}"

# Optional audio module (tolerate absence on PipeWire-default distros).
install_pkgs 1 pulseaudio-module-bluetooth

# ---------------------------------------------------------------------------
# 2. Enable SPI and I2C on Raspberry Pi
# ---------------------------------------------------------------------------
if [[ "${is_pi}" == true ]]; then
  echo ">>> Enabling SPI and I2C on Raspberry Pi…"

  # Find the firmware config file. RPi OS Bookworm+ uses /boot/firmware,
  # older releases used /boot. Fall back to /boot/firmware/config.txt for
  # first-time creation.
  find_config() {
    for f in /boot/firmware/config.txt /boot/config.txt; do
      if [[ -f "$f" ]]; then echo "$f"; return 0; fi
    done
    echo "/boot/firmware/config.txt"
  }

  # SPI
  if ! ls /dev/spidev* >/dev/null 2>&1; then
    if command -v raspi-config >/dev/null 2>&1; then
      raspi-config nonint do_spi 0  # 0 = enable
    else
      CFG="$(find_config)"
      install -d -m 0755 "$(dirname "$CFG")"
      if ! grep -q "dtparam=spi=on" "$CFG" 2>/dev/null; then
        echo "dtparam=spi=on" >> "$CFG"
      fi
    fi
    echo "    SPI enabled (reboot required for /dev/spidev* to appear)"
  else
    echo "    SPI already enabled (/dev/spidev* exists)"
  fi

  # I2C
  if ! ls /dev/i2c-* >/dev/null 2>&1; then
    if command -v raspi-config >/dev/null 2>&1; then
      raspi-config nonint do_i2c 0  # 0 = enable
    else
      CFG="$(find_config)"
      install -d -m 0755 "$(dirname "$CFG")"
      if ! grep -q "dtparam=i2c_arm=on" "$CFG" 2>/dev/null; then
        echo "dtparam=i2c_arm=on" >> "$CFG"
      fi
    fi
    echo "    I2C enabled (reboot required for /dev/i2c-* to appear)"
  else
    echo "    I2C already enabled (/dev/i2c-* exists)"
  fi

  # Camera (legacy /dev/video0 via bcm2835-v4l2 is optional; USB cams work out of the box)
  echo "    Camera: USB cameras are auto-detected. For the Pi camera module,"
  echo "            run 'raspi-config' -> Interface Options -> Camera."
fi

# ---------------------------------------------------------------------------
# 3. Create service user
# ---------------------------------------------------------------------------
# Build the supplementary-groups list dynamically: only include groups that
# actually exist on this system, and only add the Pi-specific ones (spi/i2c/
# gpio) when running on a Raspberry Pi. On generic Debian/Ubuntu those groups
# do not exist and useradd would otherwise fail.
build_groups() {
  local groups=()
  local g
  for g in audio video bluetooth; do
    getent group "$g" >/dev/null 2>&1 && groups+=("$g")
  done
  if [[ "${is_pi}" == true ]]; then
    for g in spi i2c gpio; do
      getent group "$g" >/dev/null 2>&1 && groups+=("$g")
    done
  fi
  local IFS=,
  echo "${groups[*]}"
}

if ! id -u "${SERVICE_USER}" >/dev/null 2>&1; then
  echo ">>> Creating service user '${SERVICE_USER}'…"
  USER_GROUPS="$(build_groups)"
  useradd_cmd=(useradd --system --create-home --home-dir /var/lib/deskbot)
  if [[ -n "${USER_GROUPS}" ]]; then
    useradd_cmd+=(--groups "${USER_GROUPS}")
  fi
  useradd_cmd+=("${SERVICE_USER}")
  "${useradd_cmd[@]}"
else
  # Ensure the gpio group is present for existing users (older installers
  # omitted it, which caused lgpio/gpiozero to fail with "can not open
  # gpiochip" and every backend to fall back to mock).
  if [[ "${is_pi}" == true ]] && ! id -nG "${SERVICE_USER}" | tr ' ' '\n' | grep -qx gpio; then
    if getent group gpio >/dev/null 2>&1; then
      echo ">>> Adding '${SERVICE_USER}' to the gpio group…"
      usermod -aG gpio "${SERVICE_USER}"
    fi
  fi
fi

# ---------------------------------------------------------------------------
# 3a. Disable PipeWire/WirePlumber to prevent audio conflicts
# ---------------------------------------------------------------------------
# PipeWire + WirePlumber + pipewire-pulse can intercept the PulseAudio
# socket and prevent DeskBot from using a standalone PulseAudio daemon.
# We disable the user-level services and their socket-activated units so
# only the real PulseAudio daemon serves the audio socket.
echo ">>> Disabling PipeWire/WirePlumber audio services..."

# System-level (rare but possible on minimal installs)
for unit in pipewire pipewire-pulse wireplumber; do
  systemctl disable --now "${unit}.service" 2>/dev/null || true
  systemctl disable --now "${unit}.socket" 2>/dev/null || true
  systemctl mask "${unit}.service" 2>/dev/null || true
  systemctl mask "${unit}.socket" 2>/dev/null || true
done

# User-level (per-user systemd instances - socket activation can restart
# these even after `stop`, so we mask the sockets too).
PA_UID="$(id -u "${SERVICE_USER}")"
PA_RUNTIME="/run/user/${PA_UID}"
as_user() {
  sudo -u "${SERVICE_USER}" -H XDG_RUNTIME_DIR="${PA_RUNTIME}" -- "$@"
}

for unit in pipewire pipewire-pulse wireplumber; do
  as_user systemctl --user disable --now "${unit}.service" 2>/dev/null || true
  as_user systemctl --user disable --now "${unit}.socket" 2>/dev/null || true
  as_user systemctl --user mask "${unit}.service" 2>/dev/null || true
  as_user systemctl --user mask "${unit}.socket" 2>/dev/null || true
done

# Enable lingering for the service user so the user-level systemd
# instance starts at boot (required for user-level PulseAudio).
loginctl enable-linger "${SERVICE_USER}" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 3b. Configure PulseAudio for headless Bluetooth audio
# ---------------------------------------------------------------------------
echo ">>> Configuring PulseAudio for Bluetooth audio..."

# Create the user PulseAudio configuration directory.
PA_CONF_DIR="/var/lib/deskbot/.config/pulse"
install -d -m 0755 -o "${SERVICE_USER}" -g "${SERVICE_USER}" "${PA_CONF_DIR}"

# Write the PulseAudio default configuration.  This enables Bluetooth
# discovery and the A2DP sink module.  We do NOT load
# module-bluetooth-discover here because PulseAudio's default
# configuration already loads it once at startup.  Adding it again would
# cause "should be loaded once at most" warnings.
PA_CONF="${PA_CONF_DIR}/default.pa"
if [[ ! -f "${PA_CONF}" ]]; then
  cat > "${PA_CONF}" <<'PAEOF'
#!/usr/bin/pulseaudio -nF
# DeskBot PulseAudio configuration (headless, Bluetooth A2DP)

# Load ALSA drivers (provides ALSA PCM devices)
load-module module-alsa-card

# Load the native protocol (allows paplay/pactl to connect)
load-module module-native-protocol-unix

# Load Bluetooth modules (discovery + A2DP sink)
load-module module-bluetooth-discover
load-module module-bluetooth-policy

# Auto-create sources/sinks for discovered devices
load-module module-switch-on-connect

# Default sink will be auto-selected; do NOT hardcode a Bluetooth MAC.
PAEOF
  chown "${SERVICE_USER}:${SERVICE_USER}" "${PA_CONF}"
  chmod 0644 "${PA_CONF}"
else
  echo "    PulseAudio config already exists, skipping (idempotent)"
fi

# ---------------------------------------------------------------------------
# 4. Copy project files
# ---------------------------------------------------------------------------
echo ">>> Installing DeskBot to ${INSTALL_DIR}…"
install -d -m 0755 "${INSTALL_DIR}" /etc/deskbot

# Excludes shared by rsync and the cp fallback. Keep production installs lean
# by dropping dev/build artifacts and the local .env (which may hold secrets;
# the service reads /etc/deskbot/deskbot.env instead).
EXCLUDES=(
  --exclude='.venv' --exclude='.git'
  --exclude='__pycache__' --exclude='*.pyc'
  --exclude='.pytest_cache' --exclude='.coverage'
  --exclude='.mypy_cache' --exclude='.ruff_cache'
  --exclude='site/' --exclude='coverage.json' --exclude='htmlcov/'
  --exclude='.idea/' --exclude='.vscode/'
  --exclude='to_fix/'
  --exclude='*.egg-info' --exclude='dist/' --exclude='build/'
  --exclude='.env' --exclude='.DS_Store'
  --exclude='scripts/install.sh.bak'
)

if command -v rsync >/dev/null 2>&1; then
  rsync -a "${EXCLUDES[@]}" "${SCRIPT_DIR}/" "${INSTALL_DIR}/"
else
  cp -a "${SCRIPT_DIR}/." "${INSTALL_DIR}/"
  rm -rf "${INSTALL_DIR}/.venv" "${INSTALL_DIR}/.git" \
         "${INSTALL_DIR}/site" "${INSTALL_DIR}/coverage.json" \
         "${INSTALL_DIR}/.idea" "${INSTALL_DIR}/.vscode" \
         "${INSTALL_DIR}/to_fix" "${INSTALL_DIR}/.env" \
         "${INSTALL_DIR}/scripts/install.sh.bak"
  find "${INSTALL_DIR}" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
  find "${INSTALL_DIR}" -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
fi

# Own the install tree as the service user now, so the virtualenv created
# below (as the service user) does not need a later chown.
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${INSTALL_DIR}"
chown -R "${SERVICE_USER}:${SERVICE_USER}" /var/lib/deskbot 2>/dev/null || true

# Install a default environment file only if none exists (never clobber an
# existing config).
if [[ ! -f /etc/deskbot/deskbot.env ]]; then
  if [[ -f "${INSTALL_DIR}/.env.example" ]]; then
    install -m 0640 -o root -g "${SERVICE_USER}" \
      "${INSTALL_DIR}/.env.example" /etc/deskbot/deskbot.env
    echo "    Installed default config at /etc/deskbot/deskbot.env (edit before starting)"
  else
    echo "    [WARN] .env.example not found; create /etc/deskbot/deskbot.env manually"
  fi
else
  echo "    /etc/deskbot/deskbot.env already exists, keeping it"
fi

# ---------------------------------------------------------------------------
# 5. Python environment via uv (installed as the service user, never root)
# ---------------------------------------------------------------------------
SERVICE_HOME="$(getent passwd "${SERVICE_USER}" | cut -d: -f6)"
UV_DIR="${SERVICE_HOME}/.local/bin"

# Locate an existing uv (check the known dir and the cargo fallback).
UV_BIN=""
for cand in "${UV_DIR}/uv" "${SERVICE_HOME}/.cargo/bin/uv"; do
  if [[ -x "$cand" ]]; then UV_BIN="$cand"; break; fi
done

if [[ -z "${UV_BIN}" ]]; then
  echo ">>> Installing uv (as ${SERVICE_USER}, into ${UV_DIR})…"
  sudo -u "${SERVICE_USER}" -H -- bash -c \
    "curl -LsSf https://astral.sh/uv/install.sh | sh"
    # NOTE: consider pinning a specific uv version with UV_VERSION=<x.y.z>
    # for reproducible installs. See https://docs.astral.sh/uv/
  for cand in "${UV_DIR}/uv" "${SERVICE_HOME}/.cargo/bin/uv"; do
    if [[ -x "$cand" ]]; then UV_BIN="$cand"; break; fi
  done
fi

if [[ -z "${UV_BIN}" ]]; then
  echo "ERROR: uv was not found at ${UV_DIR}/uv after install" >&2
  exit 1
fi
echo "    using uv: ${UV_BIN} ($("${UV_BIN}" --version 2>/dev/null || echo 'version unknown'))"

# Build the optional --no-cache flag (off by default to save re-download time
# and SD-card wear on the Pi).
NO_CACHE_ARG=""
if [[ "${DESKBOT_NO_CACHE:-0}" == "1" ]]; then
  NO_CACHE_ARG="--no-cache"
fi

VENV_DIR="${INSTALL_DIR}/.venv"
if [[ ! -d "${VENV_DIR}" || "${RECREATE_VENV}" == "1" ]]; then
  if [[ -d "${VENV_DIR}" ]]; then
    echo ">>> Recreating virtualenv (--recreate-venv)..."
    rm -rf "${VENV_DIR}"
  else
    echo ">>> Creating virtualenv (python 3.12)..."
  fi
  # Pin python 3.12 (satisfies requires-python>=3.12); uv fetches a managed
  # CPython if the system one is too old (e.g. 3.11 on Bookworm).
  sudo -u "${SERVICE_USER}" -H -- bash -c \
    "cd '${INSTALL_DIR}' && '${UV_BIN}' venv --python 3.12 .venv"
else
  echo ">>> Existing virtualenv found at ${VENV_DIR}, reusing (pass --recreate-venv to rebuild)"
fi

# Install with all extras so every optional dependency (hardware, audio,
# vision, TTS, API, etc.) is available.  Hardware extras are skipped
# automatically on non-aarch64 hosts via pyproject markers.
echo ">>> Syncing Python dependencies (--all-extras)..."
sudo -u "${SERVICE_USER}" -H -- bash -c \
  "cd '${INSTALL_DIR}' && '${UV_BIN}' sync --all-extras ${NO_CACHE_ARG}"

# ---------------------------------------------------------------------------
# 6. Systemd service
# ---------------------------------------------------------------------------
echo ">>> Installing systemd service…"
install -m 0644 "${SCRIPT_DIR}/deploy/systemd/deskbot.service" \
  /etc/systemd/system/deskbot.service
systemctl daemon-reload
systemctl enable deskbot

# ---------------------------------------------------------------------------
# 6a. Audio validation
# ---------------------------------------------------------------------------
echo ">>> Validating audio setup..."

# The user runtime dir may not exist yet right after enable-linger; skip the
# runtime-dependent checks in that case rather than racing against it.
if [[ ! -d "${PA_RUNTIME}" ]]; then
  echo "    [INFO] ${PA_RUNTIME} does not exist yet (user session not started)."
  echo "           PulseAudio checks are deferred until the service user logs in / lingers."
  echo "           Hint: loginctl enable-linger ${SERVICE_USER} (already attempted above)."
else
  # Start PulseAudio for the service user if not already running
  if ! as_user pactl info >/dev/null 2>&1; then
    echo "    Starting PulseAudio for ${SERVICE_USER}..."
    as_user pulseaudio --start --log-target=syslog 2>/dev/null || true
    sleep 2
  fi

  # Validate PulseAudio is reachable
  if as_user pactl info >/dev/null 2>&1; then
    echo "    [OK] PulseAudio is running and reachable"
    PA_SINKS=$(as_user pactl list short sinks 2>/dev/null | wc -l || true)
    echo "    Sinks available: ${PA_SINKS}"
    if [[ "${PA_SINKS}" -eq 0 ]]; then
      echo "    [INFO] No sinks yet. Bluetooth sinks appear after pairing a device."
      echo "           Use 'bluetoothctl' to pair a speaker, then rerun this script."
    fi
  else
    echo "    [WARNING] PulseAudio is not reachable for ${SERVICE_USER}"
    echo "             The service user may need a login session (lingering)."
    echo "             Check: loginctl enable-linger ${SERVICE_USER}"
  fi

  # Validate Bluetooth module is available
  if as_user pactl list short modules 2>/dev/null | grep -q "module-bluetooth"; then
    echo "    [OK] Bluetooth PulseAudio modules are loaded"
  else
    echo "    [INFO] Bluetooth modules not yet loaded (may load on device connect)"
  fi
fi

# Validate Bluetooth daemon
if systemctl is-active --quiet bluetooth 2>/dev/null; then
  echo "    [OK] Bluetooth daemon (bluetoothd) is active"
else
  echo "    [WARNING] Bluetooth daemon is not active. Starting..."
  systemctl enable --now bluetooth 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "  DeskBot installation complete!"
echo "============================================"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Configure the environment:"
echo "     sudo nano /etc/deskbot/deskbot.env"
if [[ -f /etc/deskbot/deskbot.env ]]; then
  echo "     (a default was installed from .env.example - edit to taste)"
else
  echo "     (or copy .env.example to /etc/deskbot/deskbot.env)"
fi
echo ""
if [[ "${is_pi}" == true ]]; then
  echo "  2. Reboot to activate SPI/I2C:"
  echo "     sudo reboot"
  echo ""
  echo "  3. After reboot, start the service:"
  echo "     sudo systemctl start deskbot"
else
  echo "  2. Start the service:"
  echo "     sudo systemctl start deskbot"
fi
echo ""
echo "  Check status:"
echo "    systemctl status deskbot"
echo "    journalctl -u deskbot -f"
echo ""
echo "  Run diagnostics:"
echo "    sudo -u ${SERVICE_USER} ${INSTALL_DIR}/.venv/bin/deskbot-doctor"
echo ""
echo "  Install log:"
echo "    ${LOG_FILE}"
echo ""
