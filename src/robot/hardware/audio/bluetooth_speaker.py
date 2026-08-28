"""Bluetooth A2DP audio output.

Uses PulseAudio/PipeWire to route audio to a paired Bluetooth A2DP
sink (e.g. wireless speakers, headphones).

The speaker accepts an :class:`AudioBuffer` with any sample rate and
channel count, wraps it in a WAV container, and passes it to ``paplay``.
PulseAudio reads the WAV header and performs the correct resampling and
channel conversion to match the Bluetooth sink's negotiated format.

For most use cases, the :class:`UsbSpeaker` is simpler and more
reliable. This module is for setups where the robot's audio must go
through a Bluetooth speaker.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from dataclasses import dataclass

from robot.interfaces.audio import AudioBuffer
from robot.logging import get_logger

_log = get_logger("hardware.audio.bluetooth_speaker")

# Timeout for pactl commands (seconds).
_PACTL_TIMEOUT = 5
# How long to poll for a sink after bluetoothctl connect (seconds).
_AUTO_CONNECT_TIMEOUT_S = 15.0
_AUTO_CONNECT_POLL_S = 2.0


# ---------------------------------------------------------------------------
# MAC / sink helpers
# ---------------------------------------------------------------------------
def _normalize_mac(mac: str) -> str:
    """Normalize a MAC address to raw lowercase hex (no separators)."""
    return mac.strip().lower().replace(":", "").replace("_", "").replace("-", "")


def _is_valid_mac(mac: str) -> bool:
    """Check if *mac* normalises to exactly 12 lowercase hex digits."""
    normalized = _normalize_mac(mac)
    return len(normalized) == 12 and all(c in "0123456789abcdef" for c in normalized)


def _mac_matches_sink(mac: str, sink_name: str) -> bool:
    """Check whether *mac* appears in *sink_name* after normalisation."""
    normalized_mac = _normalize_mac(mac)
    if not normalized_mac:
        return False
    sink_normalized = sink_name.lower().replace("_", "").replace(".", "").replace("-", "")
    return normalized_mac in sink_normalized


def _parse_pactl_sinks(output: str) -> list[tuple[str, str]]:
    """Parse ``pactl list short sinks`` output into ``(sink_name, state)`` tuples."""
    sinks: list[tuple[str, str]] = []
    for line in output.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            sink_name = parts[1]
            state = parts[-1].upper() if len(parts) >= 4 else ""
            sinks.append((sink_name, state))
    return sinks


def _run_pactl_list_sinks() -> str | None:
    """Run ``pactl list short sinks`` and return stdout, or *None* on failure."""
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sinks"],
            capture_output=True,
            text=True,
            timeout=_PACTL_TIMEOUT,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout
        _log.warning(
            "bluetooth_speaker.pactl_failed",
            returncode=result.returncode,
            stderr=result.stderr[:200],
        )
        return None
    except FileNotFoundError:
        _log.error("bluetooth_speaker.pactl_not_found")
        return None
    except subprocess.TimeoutExpired:
        _log.warning("bluetooth_speaker.pactl_timeout")
        return None


def _bluetoothctl_connect(mac: str) -> bool:
    """Attempt to connect a Bluetooth device via ``bluetoothctl connect``."""
    try:
        result = subprocess.run(
            ["bluetoothctl", "connect", mac],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            _log.info("bluetooth_speaker.bluetoothctl_connect_ok", mac=mac)
            return True
        _log.warning(
            "bluetooth_speaker.bluetoothctl_failed",
            mac=mac,
            returncode=result.returncode,
            stderr=result.stderr[:200],
        )
        return False
    except FileNotFoundError:
        _log.warning("bluetooth_speaker.bluetoothctl_not_found")
        return False
    except subprocess.TimeoutExpired:
        _log.warning("bluetooth_speaker.bluetoothctl_timeout", mac=mac)
        return False


# ---------------------------------------------------------------------------
# BluetoothSpeaker
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class BluetoothSpeaker:
    """Audio output that streams to a Bluetooth A2DP sink.

    This implementation uses PulseAudio / PipeWire to route audio to a
    paired Bluetooth device. It discovers the device by its MAC address
    (primary) or friendly name (fallback) and plays via ``paplay``.

    The ``play`` method accepts an :class:`AudioBuffer` and wraps it in
    a WAV container before passing it to ``paplay``.  PulseAudio reads
    the WAV header and handles resampling and channel conversion to
    match the sink's negotiated format (e.g. 44100 Hz stereo).

    Requirements:
    - A paired Bluetooth device (use ``bluetoothctl``).
    - ``pulseaudio`` or ``pipewire-pulse`` running.
    - ``pactl`` and ``paplay`` available on PATH.
    """

    device_mac: str = ""
    device_name: str = ""
    sample_rate: int = 48_000
    channels: int = 1
    auto_connect: bool = True
    _sink_name: str = ""
    _connected: bool = False
    _playing: bool = False

    def __post_init__(self) -> None:
        if not self.device_mac and not self.device_name:
            _log.warning(
                "bluetooth_speaker.no_device",
                msg="No device MAC or name configured; will try any Bluetooth sink",
            )
        if self.device_mac and not _is_valid_mac(self.device_mac):
            _log.warning(
                "bluetooth_speaker.invalid_mac",
                mac=self.device_mac,
                msg="Configured MAC does not look like a valid 12-hex-digit address",
            )

    # ------------------------------------------------------------------ AudioOutput API
    async def play(self, buffer: AudioBuffer) -> None:
        """Play an :class:`AudioBuffer` through the Bluetooth speaker.

        The buffer is wrapped in a WAV container and passed to ``paplay``
        via stdin.  PulseAudio reads the WAV header to determine the
        correct sample rate and channel count, then resamples to the
        sink's negotiated format automatically.
        """
        if not self._connected and self.auto_connect:
            await self._connect()

        if not self._connected:
            _log.error("bluetooth_speaker.not_connected")
            return

        if buffer.is_empty:
            _log.debug("bluetooth_speaker.play_empty")
            return

        _log.info(
            "audio.playback.started",
            sample_rate=buffer.sample_rate,
            channels=buffer.channels,
            sample_format=buffer.sample_format,
            bytes=len(buffer.pcm),
            sink=self._sink_name,
        )

        # Wrap the PCM in a WAV container so paplay can read the
        # actual format from the header rather than relying on
        # command-line arguments that might not match the data.
        wav_bytes = buffer.to_wav()

        self._playing = True
        try:
            cmd = [
                "paplay",
                "--device",
                self._sink_name,
            ]
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            _, stderr_bytes = proc.communicate(input=wav_bytes, timeout=30)
            if proc.returncode and proc.returncode != 0:
                stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
                _log.warning(
                    "audio.playback.failed",
                    returncode=proc.returncode,
                    stderr=stderr[:200],
                )
                raise RuntimeError(
                    f"Bluetooth speaker playback failed with return code {proc.returncode}"
                )
        except subprocess.TimeoutExpired:
            proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
            _log.warning("audio.playback.failed", reason="timeout")
            return
        except FileNotFoundError:
            _log.error("audio.playback.failed", reason="paplay_not_found")
            return
        finally:
            self._playing = False
        _log.info(
            "audio.playback.completed",
            duration_s=round(buffer.duration_s, 3),
            sample_rate=buffer.sample_rate,
            channels=buffer.channels,
            sink=self._sink_name,
        )

    async def stop(self) -> None:
        """Interrupt current playback."""
        self._playing = False

    async def close(self) -> None:
        """Release Bluetooth resources."""
        if self._connected:
            await self._disconnect()

    # ------------------------------------------------------------------ connection
    async def _connect(self) -> None:
        """Discover and connect to the Bluetooth A2DP sink."""
        sinks = self._list_sinks()
        if sinks is None:
            return

        # 1. MAC match (primary, most reliable).
        if self.device_mac:
            match = self._find_sink_by_mac(sinks)
            if match:
                self._select_sink(match)
                return

        # 2. Name match (secondary).
        if self.device_name:
            match = self._find_sink_by_name(sinks)
            if match:
                self._select_sink(match)
                return

        # 3. Fallback: any Bluetooth/A2DP sink.
        match = self._find_any_bluetooth_sink(sinks)
        if match:
            self._select_sink(match)
            _log.info("bluetooth_speaker.connected_fallback", sink=match[0])
            return

        # 4. Auto-connect via bluetoothctl if we have a valid MAC.
        if self.auto_connect and self.device_mac and _is_valid_mac(self.device_mac):
            _log.info("bluetooth_speaker.attempting_auto_connect", mac=self.device_mac)
            if _bluetoothctl_connect(self.device_mac) and await self._wait_for_sink(
                timeout_s=_AUTO_CONNECT_TIMEOUT_S,
                poll_interval_s=_AUTO_CONNECT_POLL_S,
            ):
                return

        _log.warning(
            "bluetooth_speaker.no_sink_found",
            mac=self.device_mac or "(none)",
            name=self.device_name or "(none)",
        )

    def _list_sinks(self) -> list[tuple[str, str]] | None:
        """Run pactl once and return parsed sinks, or *None* on failure."""
        output = _run_pactl_list_sinks()
        if output is None:
            return None
        return _parse_pactl_sinks(output)

    def _find_sink_by_mac(self, sinks: list[tuple[str, str]]) -> tuple[str, str] | None:
        for sink_name, state in sinks:
            if _mac_matches_sink(self.device_mac, sink_name):
                _log.info("bluetooth_speaker.sink_matched_by_mac", sink=sink_name, state=state)
                return (sink_name, state)
        return None

    def _find_sink_by_name(self, sinks: list[tuple[str, str]]) -> tuple[str, str] | None:
        name_lower = self.device_name.lower()
        for sink_name, state in sinks:
            if name_lower in sink_name.lower():
                _log.info("bluetooth_speaker.sink_matched_by_name", sink=sink_name, state=state)
                return (sink_name, state)
        return None

    def _find_any_bluetooth_sink(self, sinks: list[tuple[str, str]]) -> tuple[str, str] | None:
        for sink_name, state in sinks:
            lower = sink_name.lower()
            if "blue" in lower or "a2dp" in lower:
                return (sink_name, state)
        return None

    def _select_sink(self, match: tuple[str, str]) -> None:
        sink_name, state = match
        self._sink_name = sink_name
        self._connected = True
        if state == "SUSPENDED":
            _log.info(
                "bluetooth_speaker.connected_suspended",
                sink=sink_name,
                msg="Sink is suspended; PulseAudio will wake it on playback",
            )
        else:
            _log.info("bluetooth_speaker.connected", sink=sink_name, state=state)

    async def _wait_for_sink(
        self,
        timeout_s: float = _AUTO_CONNECT_TIMEOUT_S,
        poll_interval_s: float = _AUTO_CONNECT_POLL_S,
    ) -> bool:
        """Poll for the Bluetooth sink to appear after ``bluetoothctl connect``."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            await asyncio.sleep(poll_interval_s)
            sinks = self._list_sinks()
            if sinks is None:
                continue
            if self.device_mac:
                match = self._find_sink_by_mac(sinks)
                if match:
                    self._select_sink(match)
                    return True
            elif self.device_name:
                match = self._find_sink_by_name(sinks)
                if match:
                    self._select_sink(match)
                    return True
        _log.warning(
            "bluetooth_speaker.auto_connect_timeout", mac=self.device_mac, timeout_s=timeout_s
        )
        return False

    async def _disconnect(self) -> None:
        """Mark as disconnected (the Bluetooth device stays paired)."""
        self._connected = False
        self._sink_name = ""
        _log.info("bluetooth_speaker.disconnected")


__all__ = ["BluetoothSpeaker"]
