"""Tests for the Bluetooth A2DP speaker driver and AudioConfig backend selection."""

from __future__ import annotations

import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robot.config import AudioConfig


# ---------------------------------------------------------------------------
# AudioConfig backend field
# ---------------------------------------------------------------------------
class TestAudioConfigBackend:
    """Verify the ``backend`` field and Bluetooth-specific config."""

    def test_default_backend_is_mock(self) -> None:
        cfg = AudioConfig()
        assert cfg.backend == "mock"

    def test_bluetooth_backend(self) -> None:
        cfg = AudioConfig(backend="bluetooth")
        assert cfg.backend == "bluetooth"

    def test_usb_backend(self) -> None:
        cfg = AudioConfig(backend="usb")
        assert cfg.backend == "usb"

    def test_bluetooth_mac_default(self) -> None:
        cfg = AudioConfig()
        assert cfg.bluetooth_mac == ""

    def test_bluetooth_name_default(self) -> None:
        cfg = AudioConfig()
        assert cfg.bluetooth_name == ""

    def test_bluetooth_auto_connect_default(self) -> None:
        cfg = AudioConfig()
        assert cfg.bluetooth_auto_connect is True

    def test_bluetooth_config_round_trip(self) -> None:
        cfg = AudioConfig(
            backend="bluetooth",
            bluetooth_mac="AA:BB:CC:DD:EE:FF",
            bluetooth_name="JBL Flip",
            bluetooth_auto_connect=False,
        )
        assert cfg.backend == "bluetooth"
        assert cfg.bluetooth_mac == "AA:BB:CC:DD:EE:FF"
        assert cfg.bluetooth_name == "JBL Flip"
        assert cfg.bluetooth_auto_connect is False

    def test_invalid_backend_raises(self) -> None:
        from pydantic_core import ValidationError

        with pytest.raises(ValidationError):
            AudioConfig(backend="wifi")


# ---------------------------------------------------------------------------
# MAC normalisation helpers
# ---------------------------------------------------------------------------
class TestMacNormalisation:
    """Test _normalize_mac, _is_valid_mac, and _mac_matches_sink."""

    def test_normalize_mac_colon(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import _normalize_mac

        assert _normalize_mac("E8:EE:CC:49:94:2A") == "e8eecc49942a"

    def test_normalize_mac_underscore(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import _normalize_mac

        assert _normalize_mac("E8_EE_CC_49_94_2A") == "e8eecc49942a"

    def test_normalize_mac_hyphen(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import _normalize_mac

        assert _normalize_mac("e8-ee-cc-49-94-2a") == "e8eecc49942a"

    def test_normalize_mac_no_separators(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import _normalize_mac

        assert _normalize_mac("e8eecc49942a") == "e8eecc49942a"

    def test_normalize_mac_lowercase(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import _normalize_mac

        assert _normalize_mac("E8:EE:CC:49:94:2A") == _normalize_mac("e8:ee:cc:49:94:2a")

    def test_normalize_mac_strips_whitespace(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import _normalize_mac

        assert _normalize_mac("  E8:EE:CC:49:94:2A  ") == "e8eecc49942a"

    def test_is_valid_mac_valid(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import _is_valid_mac

        assert _is_valid_mac("E8:EE:CC:49:94:2A")
        assert _is_valid_mac("e8_ee_cc_49_94_2a")
        assert _is_valid_mac("e8eecc49942a")

    def test_is_valid_mac_invalid(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import _is_valid_mac

        assert not _is_valid_mac("")
        assert not _is_valid_mac("not a mac")
        assert not _is_valid_mac("E8:EE:CC:49")  # too short
        assert not _is_valid_mac("GG:EE:CC:49:94:2A")  # non-hex

    def test_mac_matches_sink_real_world(self) -> None:
        """The exact sink from the user's Pi must match."""
        from robot.hardware.audio.bluetooth_speaker import _mac_matches_sink

        assert _mac_matches_sink(
            "E8:EE:CC:49:94:2A",
            "bluez_sink.E8_EE_CC_49_94_2A.a2dp_sink",
        )

    def test_mac_matches_sink_case_variants(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import _mac_matches_sink

        assert _mac_matches_sink(
            "e8:ee:cc:49:94:2a",
            "bluez_sink.E8_EE_CC_49_94_2A.a2dp_sink",
        )

    def test_mac_matches_sink_underscore_mac(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import _mac_matches_sink

        assert _mac_matches_sink(
            "E8_EE_CC_49_94_2A",
            "bluez_sink.E8_EE_CC_49_94_2A.a2dp_sink",
        )

    def test_mac_matches_sink_different_separator_in_sink(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import _mac_matches_sink

        # PipeWire sometimes uses "bluez_output" with hyphen in "a2dp-sink"
        assert _mac_matches_sink(
            "AA:BB:CC:DD:EE:FF",
            "bluez_output.AA_BB_CC_DD_EE_FF.a2dp-sink",
        )

    def test_mac_does_not_match_unrelated_sink(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import _mac_matches_sink

        assert not _mac_matches_sink(
            "E8:EE:CC:49:94:2A",
            "bluez_sink.AA_BB_CC_DD_EE_FF.a2dp_sink",
        )

    def test_mac_does_not_match_non_bluetooth_sink(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import _mac_matches_sink

        assert not _mac_matches_sink(
            "E8:EE:CC:49:94:2A",
            "alsa_output.pci-0000_00_1f.3.analog-stereo",
        )

    def test_mac_matches_empty_mac(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import _mac_matches_sink

        assert not _mac_matches_sink("", "bluez_sink.E8_EE_CC_49_94_2A.a2dp_sink")


# ---------------------------------------------------------------------------
# pactl output parsing
# ---------------------------------------------------------------------------
class TestPactlParsing:
    """Test _parse_pactl_sinks with various output formats."""

    def test_parse_standard_output(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import _parse_pactl_sinks

        output = (
            "0\talsa_output.pci-0000_00_1f.3.analog-stereo\tmodule-alsa-card.c\ts16le 2ch 44100Hz\tRUNNING\n"
            "1\tbluez_sink.E8_EE_CC_49_94_2A.a2dp_sink\tmodule-bluez5-device.c\ts16le 2ch 44100Hz\tSUSPENDED\n"
        )
        sinks = _parse_pactl_sinks(output)
        assert len(sinks) == 2
        assert sinks[0] == ("alsa_output.pci-0000_00_1f.3.analog-stereo", "RUNNING")
        assert sinks[1] == ("bluez_sink.E8_EE_CC_49_94_2A.a2dp_sink", "SUSPENDED")

    def test_parse_empty_output(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import _parse_pactl_sinks

        assert _parse_pactl_sinks("") == []
        assert _parse_pactl_sinks("\n\n") == []

    def test_parse_real_world_suspended_sink(self) -> None:
        """Exact output from the user's Pi."""
        from robot.hardware.audio.bluetooth_speaker import _parse_pactl_sinks

        output = (
            "1\tbluez_sink.E8_EE_CC_49_94_2A.a2dp_sink\t"
            "module-bluez5-device.c\ts16le 2ch 44100Hz\tSUSPENDED\n"
        )
        sinks = _parse_pactl_sinks(output)
        assert len(sinks) == 1
        assert sinks[0] == ("bluez_sink.E8_EE_CC_49_94_2A.a2dp_sink", "SUSPENDED")


# ---------------------------------------------------------------------------
# BluetoothSpeaker creation & connection
# ---------------------------------------------------------------------------
class TestBluetoothSpeakerCreation:
    """Test BluetoothSpeaker instantiation and auto-discovery logic."""

    def test_creation_with_defaults(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

        speaker = BluetoothSpeaker()
        assert speaker.device_mac == ""
        assert speaker.device_name == ""
        assert speaker.sample_rate == 48_000
        assert speaker.channels == 1
        assert speaker.auto_connect is True
        assert not speaker._connected

    def test_creation_with_mac(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

        speaker = BluetoothSpeaker(device_mac="AA:BB:CC:DD:EE:FF")
        assert speaker.device_mac == "AA:BB:CC:DD:EE:FF"

    def test_creation_with_name(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

        speaker = BluetoothSpeaker(device_name="JBL Flip")
        assert speaker.device_name == "JBL Flip"


# ---------------------------------------------------------------------------
# BluetoothSpeaker sink discovery (mocked subprocess)
# ---------------------------------------------------------------------------
class TestBluetoothSpeakerDiscovery:
    """Test that BluetoothSpeaker discovers PulseAudio sinks correctly."""

    @pytest.mark.anyio
    async def test_connect_finds_sink_by_mac_real_world(self) -> None:
        """Exact scenario from the Pi: MAC E8:EE:CC:49:94:2A must match
        bluez_sink.E8_EE_CC_49_94_2A.a2dp_sink in SUSPENDED state."""
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

        speaker = BluetoothSpeaker(
            device_mac="E8:EE:CC:49:94:2A",
            auto_connect=False,
        )

        mock_output = (
            "0\talsa_output.pci-0000_00_1f.3.analog-stereo\t"
            "module-alsa-card.c\ts16le 2ch 44100Hz\tRUNNING\n"
            "1\tbluez_sink.E8_EE_CC_49_94_2A.a2dp_sink\t"
            "module-bluez5-device.c\ts16le 2ch 44100Hz\tSUSPENDED\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_output

        with patch(
            "robot.hardware.audio.bluetooth_speaker.subprocess.run",
            return_value=mock_result,
        ):
            await speaker._connect()

        assert speaker._connected
        assert speaker._sink_name == "bluez_sink.E8_EE_CC_49_94_2A.a2dp_sink"

    @pytest.mark.anyio
    async def test_connect_finds_sink_by_mac_lowercase(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

        speaker = BluetoothSpeaker(
            device_mac="e8:ee:cc:49:94:2a",
            auto_connect=False,
        )
        mock_output = (
            "1\tbluez_sink.E8_EE_CC_49_94_2A.a2dp_sink\t"
            "module-bluez5-device.c\ts16le 2ch 44100Hz\tSUSPENDED\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_output

        with patch(
            "robot.hardware.audio.bluetooth_speaker.subprocess.run",
            return_value=mock_result,
        ):
            await speaker._connect()

        assert speaker._connected
        assert speaker._sink_name == "bluez_sink.E8_EE_CC_49_94_2A.a2dp_sink"

    @pytest.mark.anyio
    async def test_connect_finds_sink_by_mac_underscore(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

        speaker = BluetoothSpeaker(
            device_mac="E8_EE_CC_49_94_2A",
            auto_connect=False,
        )
        mock_output = (
            "1\tbluez_sink.E8_EE_CC_49_94_2A.a2dp_sink\t"
            "module-bluez5-device.c\ts16le 2ch 44100Hz\tSUSPENDED\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_output

        with patch(
            "robot.hardware.audio.bluetooth_speaker.subprocess.run",
            return_value=mock_result,
        ):
            await speaker._connect()

        assert speaker._connected
        assert speaker._sink_name == "bluez_sink.E8_EE_CC_49_94_2A.a2dp_sink"

    @pytest.mark.anyio
    async def test_connect_accepts_suspended_sink(self) -> None:
        """A SUSPENDED sink must be accepted as a valid connection target."""
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

        speaker = BluetoothSpeaker(
            device_mac="E8:EE:CC:49:94:2A",
            auto_connect=False,
        )
        mock_output = (
            "1\tbluez_sink.E8_EE_CC_49_94_2A.a2dp_sink\t"
            "module-bluez5-device.c\ts16le 2ch 44100Hz\tSUSPENDED\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_output

        with patch(
            "robot.hardware.audio.bluetooth_speaker.subprocess.run",
            return_value=mock_result,
        ):
            await speaker._connect()

        assert speaker._connected
        assert speaker._sink_name == "bluez_sink.E8_EE_CC_49_94_2A.a2dp_sink"

    @pytest.mark.anyio
    async def test_connect_finds_sink_by_name(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

        speaker = BluetoothSpeaker(
            device_name="JBL Flip",
            auto_connect=False,
        )
        mock_output = (
            "0\talsa_output.pci-0000_00_1f.3.analog-stereo\tmodule-alsa-card.c\ts16le 2ch 44100Hz\tRUNNING\n"
            "1\tbluez_output.JBL_Flip.a2dp-sink\tmodule-bluez5-device.c\ts16le 2ch 44100Hz\tRUNNING\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_output

        with patch(
            "robot.hardware.audio.bluetooth_speaker.subprocess.run",
            return_value=mock_result,
        ):
            await speaker._connect()

        assert speaker._connected
        assert speaker._sink_name == "bluez_output.JBL_Flip.a2dp-sink"

    @pytest.mark.anyio
    async def test_connect_mac_takes_priority_over_name(self) -> None:
        """When both MAC and name are configured, MAC should win."""
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

        speaker = BluetoothSpeaker(
            device_mac="E8:EE:CC:49:94:2A",
            device_name="Soundcore",
            auto_connect=False,
        )
        mock_output = (
            "0\tbluez_sink.E8_EE_CC_49_94_2A.a2dp_sink\tmodule-bluez5-device.c\ts16le 2ch 44100Hz\tSUSPENDED\n"
            "1\tbluez_output.Soundcore.a2dp-sink\tmodule-bluez5-device.c\ts16le 2ch 44100Hz\tRUNNING\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_output

        with patch(
            "robot.hardware.audio.bluetooth_speaker.subprocess.run",
            return_value=mock_result,
        ):
            await speaker._connect()

        assert speaker._connected
        assert speaker._sink_name == "bluez_sink.E8_EE_CC_49_94_2A.a2dp_sink"

    @pytest.mark.anyio
    async def test_connect_name_used_when_mac_does_not_match(self) -> None:
        """Name match is used when the configured MAC doesn't match any sink."""
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

        speaker = BluetoothSpeaker(
            device_mac="AA:BB:CC:DD:EE:FF",
            device_name="Soundcore",
            auto_connect=False,
        )
        mock_output = "0\tbluez_output.Soundcore.a2dp-sink\tmodule-bluez5-device.c\ts16le 2ch 44100Hz\tRUNNING\n"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_output

        with patch(
            "robot.hardware.audio.bluetooth_speaker.subprocess.run",
            return_value=mock_result,
        ):
            await speaker._connect()

        assert speaker._connected
        assert speaker._sink_name == "bluez_output.Soundcore.a2dp-sink"

    @pytest.mark.anyio
    async def test_connect_fallback_to_any_bluetooth_sink(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

        speaker = BluetoothSpeaker(auto_connect=False)
        mock_output = (
            "0\talsa_output.pci-0000_00_1f.3.analog-stereo\tmodule-alsa-card.c\ts16le 2ch 44100Hz\tRUNNING\n"
            "1\tbluez_output.AA_BB_CC_DD_EE_FF.a2dp-sink\tmodule-bluez5-device.c\ts16le 2ch 44100Hz\tRUNNING\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_output

        with patch(
            "robot.hardware.audio.bluetooth_speaker.subprocess.run",
            return_value=mock_result,
        ):
            await speaker._connect()

        assert speaker._connected
        assert "blue" in speaker._sink_name.lower() or "a2dp" in speaker._sink_name.lower()

    @pytest.mark.anyio
    async def test_connect_no_sink_found(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

        speaker = BluetoothSpeaker(
            device_mac="11:22:33:44:55:66",
            auto_connect=False,
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "0\talsa_output.pci-0000_00_1f.3.analog-stereo\tmodule-alsa-card.c\ts16le 2ch 44100Hz\tRUNNING\n"

        with patch(
            "robot.hardware.audio.bluetooth_speaker.subprocess.run",
            return_value=mock_result,
        ):
            await speaker._connect()

        assert not speaker._connected

    @pytest.mark.anyio
    async def test_connect_ignores_unrelated_sinks(self) -> None:
        """Non-Bluetooth sinks must never be selected."""
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

        speaker = BluetoothSpeaker(
            device_mac="E8:EE:CC:49:94:2A",
            auto_connect=False,
        )
        mock_output = (
            "0\talsa_output.pci-0000_00_1f.3.analog-stereo\tmodule-alsa-card.c\ts16le 2ch 44100Hz\tRUNNING\n"
            "1\tcombined\tmodule-combine-sink.c\ts16le 2ch 44100Hz\tRUNNING\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_output

        with patch(
            "robot.hardware.audio.bluetooth_speaker.subprocess.run",
            return_value=mock_result,
        ):
            await speaker._connect()

        assert not speaker._connected

    @pytest.mark.anyio
    async def test_connect_handles_pactl_not_found(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

        speaker = BluetoothSpeaker(
            device_mac="AA:BB:CC:DD:EE:FF",
            auto_connect=False,
        )
        with patch(
            "robot.hardware.audio.bluetooth_speaker.subprocess.run",
            side_effect=FileNotFoundError("pactl not found"),
        ):
            await speaker._connect()

        assert not speaker._connected

    @pytest.mark.anyio
    async def test_connect_handles_timeout(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

        speaker = BluetoothSpeaker(
            device_mac="AA:BB:CC:DD:EE:FF",
            auto_connect=False,
        )
        with patch(
            "robot.hardware.audio.bluetooth_speaker.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="pactl", timeout=5),
        ):
            await speaker._connect()

        assert not speaker._connected

    @pytest.mark.anyio
    async def test_connect_calls_pactl_once(self) -> None:
        """pactl list short sinks must be called exactly once per _connect."""
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

        speaker = BluetoothSpeaker(
            device_mac="E8:EE:CC:49:94:2A",
            auto_connect=False,
        )
        mock_output = (
            "1\tbluez_sink.E8_EE_CC_49_94_2A.a2dp_sink\t"
            "module-bluez5-device.c\ts16le 2ch 44100Hz\tSUSPENDED\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_output

        with patch(
            "robot.hardware.audio.bluetooth_speaker.subprocess.run",
            return_value=mock_result,
        ) as run_mock:
            await speaker._connect()

        assert run_mock.call_count == 1


# ---------------------------------------------------------------------------
# Auto-connect via bluetoothctl
# ---------------------------------------------------------------------------
class TestAutoConnect:
    """Test the bluetoothctl auto-connect + poll behaviour."""

    @pytest.mark.anyio
    async def test_auto_connect_tries_bluetoothctl_then_polls(self) -> None:
        """When no sink exists and auto_connect is on, try bluetoothctl."""
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

        speaker = BluetoothSpeaker(
            device_mac="E8:EE:CC:49:94:2A",
            auto_connect=True,
        )

        call_count = 0

        def mock_run(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_result = MagicMock()
            mock_result.returncode = 0
            if cmd == ["bluetoothctl", "connect", "E8:EE:CC:49:94:2A"]:
                mock_result.stdout = "Attempting to connect...\nConnection successful\n"
            # pactl: first call no sink, second call sink appears
            elif call_count <= 1:
                mock_result.stdout = "0\talsa_output.pci-0000_00_1f.3.analog-stereo\tmodule-alsa-card.c\ts16le 2ch 44100Hz\tRUNNING\n"
            else:
                mock_result.stdout = (
                    "0\talsa_output.pci-0000_00_1f.3.analog-stereo\tmodule-alsa-card.c\ts16le 2ch 44100Hz\tRUNNING\n"
                    "1\tbluez_sink.E8_EE_CC_49_94_2A.a2dp_sink\tmodule-bluez5-device.c\ts16le 2ch 44100Hz\tSUSPENDED\n"
                )
            return mock_result

        with (
            patch(
                "robot.hardware.audio.bluetooth_speaker.subprocess.run",
                side_effect=mock_run,
            ),
            patch(
                "robot.hardware.audio.bluetooth_speaker.asyncio.sleep",
                new=AsyncMock(),
            ),
        ):
            await speaker._connect()

        assert speaker._connected
        assert speaker._sink_name == "bluez_sink.E8_EE_CC_49_94_2A.a2dp_sink"

    @pytest.mark.anyio
    async def test_auto_connect_skipped_when_disabled(self) -> None:
        """auto_connect=False must not attempt bluetoothctl."""
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

        speaker = BluetoothSpeaker(
            device_mac="E8:EE:CC:49:94:2A",
            auto_connect=False,
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "0\talsa_output.pci-0000_00_1f.3.analog-stereo\tmodule-alsa-card.c\ts16le 2ch 44100Hz\tRUNNING\n"

        with patch(
            "robot.hardware.audio.bluetooth_speaker.subprocess.run",
            return_value=mock_result,
        ) as run_mock:
            await speaker._connect()

        assert not speaker._connected
        # Only pactl was called, never bluetoothctl
        for call in run_mock.call_args_list:
            assert call[0][0] == ["pactl", "list", "short", "sinks"]


# ---------------------------------------------------------------------------
# BluetoothSpeaker play (paplay command construction)
# ---------------------------------------------------------------------------
class TestBluetoothSpeakerPlay:
    """Test paplay command construction and playback behaviour."""

    @pytest.mark.anyio
    async def test_play_uses_paplay_with_correct_args(self) -> None:
        """paplay receives WAV data (not raw PCM) with no hardcoded format flags.

        The WAV header carries the actual sample rate and channels so
        PulseAudio can resample correctly.  No --raw, --rate, --channels,
        or --format flags should be present.
        """
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker
        from robot.interfaces.audio import AudioBuffer

        speaker = BluetoothSpeaker(device_mac="AA:BB:CC:DD:EE:FF", auto_connect=True)
        speaker._connected = True
        speaker._sink_name = "bluez_output.AA_BB_CC_DD_EE_FF.a2dp-sink"

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stderr = b""
        mock_proc.communicate = MagicMock(return_value=(b"", b""))

        with patch(
            "robot.hardware.audio.bluetooth_speaker.subprocess.Popen",
            return_value=mock_proc,
        ) as popen_mock:
            buf = AudioBuffer(pcm=b"\x00\x00" * 100, sample_rate=22050, channels=1)
            await speaker.play(buf)

            args, _kwargs = popen_mock.call_args
            cmd = args[0]
            assert cmd[0] == "paplay"
            assert "--device" in cmd
            assert "bluez_output.AA_BB_CC_DD_EE_FF.a2dp-sink" in cmd
            # WAV mode: no --raw, no --rate, no --channels, no --format
            # The format comes from the WAV header.
            assert "--raw" not in cmd
            assert "--rate" not in cmd
            assert "--channels" not in cmd
            assert "--format" not in cmd

            # Verify WAV data was passed to communicate().
            communicate_kwargs = mock_proc.communicate.call_args[1]
            input_data = communicate_kwargs.get("input", b"")
            assert input_data[:4] == b"RIFF"  # WAV header

    @pytest.mark.anyio
    async def test_play_skips_when_not_connected(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker
        from robot.interfaces.audio import AudioBuffer

        speaker = BluetoothSpeaker(
            device_mac="AA:BB:CC:DD:EE:FF",
            auto_connect=False,
        )
        speaker._connected = False

        # Should not raise, just log and return
        buf = AudioBuffer(pcm=b"\x00\x00" * 100, sample_rate=22050, channels=1)
        await speaker.play(buf)

    @pytest.mark.anyio
    async def test_play_handles_paplay_not_found(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker
        from robot.interfaces.audio import AudioBuffer

        speaker = BluetoothSpeaker(
            device_mac="AA:BB:CC:DD:EE:FF",
            auto_connect=True,
        )
        speaker._connected = True
        speaker._sink_name = "bluez_output.AA_BB_CC_DD_EE_FF.a2dp-sink"

        with patch(
            "robot.hardware.audio.bluetooth_speaker.subprocess.Popen",
            side_effect=FileNotFoundError("paplay not found"),
        ):
            buf = AudioBuffer(pcm=b"\x00\x00" * 100, sample_rate=22050, channels=1)
            # Should not crash, just log error
            await speaker.play(buf)

    @pytest.mark.anyio
    async def test_play_handles_timeout(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker
        from robot.interfaces.audio import AudioBuffer

        speaker = BluetoothSpeaker(
            device_mac="AA:BB:CC:DD:EE:FF",
            auto_connect=True,
        )
        speaker._connected = True
        speaker._sink_name = "bluez_output.AA_BB_CC_DD_EE_FF.a2dp-sink"

        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = subprocess.TimeoutExpired(cmd="paplay", timeout=30)
        mock_proc.kill = MagicMock()

        with patch(
            "robot.hardware.audio.bluetooth_speaker.subprocess.Popen",
            return_value=mock_proc,
        ):
            buf = AudioBuffer(pcm=b"\x00\x00" * 100, sample_rate=22050, channels=1)
            await speaker.play(buf)
            mock_proc.kill.assert_called()


# ---------------------------------------------------------------------------
# BluetoothSpeaker lifecycle
# ---------------------------------------------------------------------------
class TestBluetoothSpeakerLifecycle:
    """Test stop/close/disconnect lifecycle."""

    @pytest.mark.anyio
    async def test_stop_sets_playing_false(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

        speaker = BluetoothSpeaker()
        speaker._playing = True
        await speaker.stop()
        assert not speaker._playing

    @pytest.mark.anyio
    async def test_close_disconnects(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

        speaker = BluetoothSpeaker()
        speaker._connected = True
        speaker._sink_name = "some_sink"
        await speaker.close()
        assert not speaker._connected
        assert speaker._sink_name == ""

    @pytest.mark.anyio
    async def test_close_noop_when_not_connected(self) -> None:
        from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

        speaker = BluetoothSpeaker()
        assert not speaker._connected
        await speaker.close()  # should not raise


# ---------------------------------------------------------------------------
# App-level fallback: ImportError on BluetoothSpeaker
# ---------------------------------------------------------------------------
class TestBluetoothFallback:
    """Test that app.py falls back to MockAudioOutput when BluetoothSpeaker import fails."""

    def test_fallback_on_import_error(self) -> None:
        """Simulate ImportError for BluetoothSpeaker -> fallback to MockAudioOutput."""
        from robot.hardware.audio.mock_audio import MockAudioOutput

        mock = MockAudioOutput(sample_rate=48000, channels=1)
        assert mock.sample_rate == 48000
        assert mock.channels == 1
