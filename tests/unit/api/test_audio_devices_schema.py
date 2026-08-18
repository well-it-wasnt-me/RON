"""Tests for the audio devices schema fix.

The frontend (web/settings/index.html) expects ``default_index`` on the
``AudioDevicesResponse`` and ``default_sample_rate`` on each ``AudioDevice``.
These fields were missing from the Pydantic schemas and got stripped by
``model_validate()``, causing the device list and "default" indicator to
break in the UI.
"""

from __future__ import annotations

from robot.api.schemas import AudioDevice, AudioDevicesResponse


class TestAudioDeviceSchema:
    def test_default_sample_rate_field_exists(self) -> None:
        """AudioDevice should have a default_sample_rate field."""
        dev = AudioDevice(name="test", index=0, channels=2, default_sample_rate=48000)
        assert dev.default_sample_rate == 48000

    def test_default_sample_rate_optional(self) -> None:
        """default_sample_rate should be optional."""
        dev = AudioDevice(name="test")
        assert dev.default_sample_rate is None


class TestAudioDevicesResponseSchema:
    def test_default_index_field_exists(self) -> None:
        """AudioDevicesResponse should have a default_index field."""
        resp = AudioDevicesResponse(
            devices=[AudioDevice(name="test", index=0, channels=2)],
            available=True,
            default_index=0,
        )
        assert resp.default_index == 0

    def test_default_index_optional(self) -> None:
        """default_index should be optional."""
        resp = AudioDevicesResponse(devices=[], available=False)
        assert resp.default_index is None

    def test_model_validate_preserves_default_index(self) -> None:
        """model_validate should not strip default_index."""
        data = {
            "devices": [
                {"name": "hw:1,0", "index": 2, "channels": 2, "default_sample_rate": 48000}
            ],
            "default_index": 2,
            "available": True,
        }
        resp = AudioDevicesResponse.model_validate(data)
        assert resp.default_index == 2
        assert resp.devices[0].default_sample_rate == 48000

    def test_model_validate_preserves_default_index_none(self) -> None:
        """model_validate should handle default_index=None."""
        data: dict[str, object] = {
            "devices": [],
            "default_index": None,
            "available": False,
        }
        resp = AudioDevicesResponse.model_validate(data)
        assert resp.default_index is None
        assert resp.available is False
