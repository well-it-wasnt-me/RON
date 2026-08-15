# Wiring

This is the hardware reference for the current DeskBot prototype.

> **Safety:** servos can draw enough current to brown out or damage a Raspberry
> Pi supply path. Use an appropriate external servo supply and connect its
> ground to Raspberry Pi ground.

## Hardware

Current target hardware:

- Raspberry Pi 5
- 240×240 circular GC9A01 SPI TFT
- two head servos
- two arm servos
- optional USB camera
- optional USB microphone
- optional USB/Bluetooth speaker

## Display

The display is SPI, despite some breakout boards labelling the clock/data
signals as `SCL`/`SDA`.

The application supports:

- `gc9a01`: raw SPI + GPIO implementation
- `circuitpython` / `cp` / `displayio`: CircuitPython displayio backend
- `mock`: development backend

Relevant settings:

```env
DESKBOT_DISPLAYS__BACKEND=circuitpython
DESKBOT_DISPLAYS__BUS=0
DESKBOT_DISPLAYS__DEVICE=0
DESKBOT_DISPLAYS__SPI_HZ=8000000
DESKBOT_DISPLAYS__SPI_MODE=0
DESKBOT_DISPLAYS__DC_PIN=25
DESKBOT_DISPLAYS__RESET_PIN=24
DESKBOT_DISPLAYS__BACKLIGHT_PIN=
DESKBOT_DISPLAYS__WIDTH=240
DESKBOT_DISPLAYS__HEIGHT=240
DESKBOT_DISPLAYS__ROTATION=0
```

Enable SPI on the Raspberry Pi before using a real SPI backend.

## Servos

Default BCM GPIO mapping:

| Function | GPIO |
|---|---:|
| Pan | 12 |
| Tilt | 13 |
| Left arm | 18 |
| Right arm | 19 |

Servo frequency defaults to 50 Hz.

The GPIO backend uses the configured pulse and angle limits. Start with
conservative limits and test one servo at a time.

## Power

Do not use the Pi 5V rail as the primary power source for four servos.

Use a separate supply sized for the actual servo stall/current requirements,
and connect the external supply ground to Pi ground.

## Camera and audio

USB camera and microphone devices are selected through:

```env
DESKBOT_CAMERA__DEVICE=0
DESKBOT_MICROPHONE__INPUT_DEVICE=default
DESKBOT_AUDIO__OUTPUT_DEVICE=default
```

The exact Linux device names depend on the hardware and OS image.

## Troubleshooting

### Nothing happens

Check:

```bash
make doctor
```

Then verify:

- `DESKBOT_HARDWARE=real`
- the display backend is not `mock`
- SPI is enabled
- `/dev/spidev*` exists
- GPIO permissions are correct
- servo power is external and ground is common
- camera/audio devices are visible to the OS

For a display-only smoke test:

```bash
make display-test
```

For hardware diagnostics:

```bash
make hardware-check
```
