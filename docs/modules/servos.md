# Servo subsystem

DeskBot exposes a `ServoController` protocol. The active implementation is
selected by `DESKBOT_SERVOS__BACKEND`.

## Backends

| Backend | Status | Purpose |
|---|---|---|
| `mock` | Available | Development and tests |
| `gpio` | Available | Raspberry Pi GPIO PWM |
| `pca9685` | Available in code | I²C controller backend; verify hardware/library support on target |

The factory fails fast if the selected backend cannot initialize.

## Default GPIO mapping

The current GPIO mapping is:

| Servo | BCM GPIO |
|---|---:|
| pan | 12 |
| tilt | 13 |
| left_arm | 18 |
| right_arm | 19 |

These are configuration defaults, not universal wiring requirements.

## Configuration

```env
DESKBOT_SERVOS__BACKEND=gpio
DESKBOT_SERVOS__GPIO__FREQUENCY=50
DESKBOT_SERVOS__GPIO__PINS__PAN=12
DESKBOT_SERVOS__GPIO__PINS__TILT=13
DESKBOT_SERVOS__GPIO__PINS__LEFT_ARM=18
DESKBOT_SERVOS__GPIO__PINS__RIGHT_ARM=19
```

PCA9685 configuration uses:

```env
DESKBOT_SERVOS__BACKEND=pca9685
DESKBOT_SERVOS__PCA9685__ADDRESS=0x40
DESKBOT_SERVOS__PCA9685__BUS=1
DESKBOT_SERVOS__PCA9685__FREQUENCY=50
```

Per-channel pulse and angle limits are represented by
`ServoChannelConfig`.

## Safety

Use an external servo power supply. Do not power four hobby servos from the
Raspberry Pi's 5V rail.

Always verify mechanical travel before running calibration or expressive
motions. Configure conservative angle limits first.
