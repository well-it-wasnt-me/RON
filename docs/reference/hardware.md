# Hardware

DeskBot separates hardware interfaces from concrete drivers.

## Display

- `MockDisplay`
- `GC9A01Display`
- `CircuitPythonDisplay`

The production display target is a 240×240 GC9A01 circular TFT.

## Servos

- `MockServoBus`
- `RaspberryPiGPIOServoController`
- `PCA9685ServoController`

See [Servo subsystem](../modules/servos.md).

## Audio

Three audio backends are available, selected by the ``DESKBOT_AUDIO__BACKEND``
environment variable:

| Backend   | Class              | Description                                                |
|-----------|--------------------|------------------------------------------------------------|
| `mock`    | `MockAudioOutput`  | In-memory recorder, no sound (default, safe for tests)     |
| `usb`     | `UsbSpeaker`       | Plays through ``sounddevice`` / ALSA (USB / 3.5 mm jack) |
| `bluetooth` | `BluetoothSpeaker` | Routes audio to a paired Bluetooth A2DP sink via PulseAudio / PipeWire |

### Bluetooth audio setup

1. **Pair the device** on your Raspberry Pi:

```bash
# Install Bluetooth tools if missing
sudo apt install bluetooth bluez pulseaudio-module-bluetooth

# Pair and trust the speaker
bluetoothctl
[bluetoothctl] power on
[bluetoothctl] agent on
[bluetoothctl] scan on
[bluetoothctl] pair AA:BB:CC:DD:EE:FF
[bluetoothctl] trust AA:BB:CC:DD:EE:FF
[bluetoothctl] connect AA:BB:CC:DD:EE:FF
[bluetoothctl] quit
```

2. **Configure DeskBot** to use the Bluetooth backend:

```bash
export DESKBOT_AUDIO__BACKEND=bluetooth
export DESKBOT_AUDIO__BLUETOOTH_MAC=AA:BB:CC:DD:EE:FF
# Optional: use the device name instead of MAC
# export DESKBOT_AUDIO__BLUETOOTH_NAME="JBL Flip"
```

3. **Troubleshooting**

- **No audio**: Ensure ``paplay`` is available (``pulseaudio-utils`` package).
  PipeWire users should install ``pipewire-pulse`` which provides PulseAudio compatibility.
- **High latency**: Bluetooth A2DP adds ~200 ms latency. For real-time interaction,
  a USB speaker is preferred.
- **Device not found**: Set ``DESKBOT_AUDIO__BLUETOOTH_AUTO_CONNECT=false`` and manually
  connect the device before starting DeskBot.
- **Fallback**: If ``BluetoothSpeaker`` cannot import or PulseAudio/PipeWire is unavailable,
  DeskBot falls back to ``MockAudioOutput`` (no sound) instead of crashing.

## Sensors

- `MockCamera`
- `UsbCamera`
- `MockMicrophone`
- `UsbMicrophone`

## Hardware factories

Factories select concrete implementations from configuration and fail fast
when a selected real backend cannot be initialized.

The interfaces live under `robot.interfaces`; this is the dependency boundary
used by application services and tests.
