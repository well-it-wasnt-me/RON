# Face Animations

DeskBot's face animations produce timed sequences of face-model deltas
(gaze, mouth shape, eye openness, etc.) that the
[`FaceOrchestrator`](face.md) applies to the face renderer. They are
standalone, stateless animation objects that can be driven independently
of the event bus.

The three built-in face animations are:

| Animation | Purpose | Trigger |
|-----------|---------|---------|
| [`ThinkingDotsAnimation`][robot.face.animations.thinking_dots.ThinkingDotsAnimation] | Periodic gaze-shift pattern | LLM streaming response |
| [`SpeakingAnimation`][robot.face.animations.speaking.SpeakingAnimation] | Viseme-based mouth shapes | TTS playback |
| [`WakeAnimation`][robot.face.animations.wake.WakeAnimation] | 4-phase attention-getting flash | Wake word detection |

All animations follow the same pattern: construct, then call
`step(dt)` in a loop to get the current frame.

---

## ThinkingDotsAnimation

Produces a gentle left-right-up gaze-shift pattern that gives the face a
"thinking" look while the LLM is generating a response. The animation
cycles through 7 keyframes, holding each for a configurable duration
and smoothly interpolating between positions.

```python
from robot.face.animations.thinking_dots import ThinkingDotsAnimation

anim = ThinkingDotsAnimation()

# In a render loop, advance by dt seconds each tick:
gaze = anim.step(dt=0.033)
# gaze.x, gaze.y - feed these into the face model

# Reset when LLM streaming finishes
anim.reset()
```

### Keyframe pattern

The default pattern cycles through these gaze positions:

| # | Gaze (x, y) | Hold (s) | Description |
|---|-------------|----------|-------------|
| 1 | (0.20, -0.20) | 0.40 | Up-right - classic "thinking" pose |
| 2 | (0.05, -0.15) | 0.30 | Slightly left |
| 3 | (-0.10, -0.10) | 0.35 | Look left |
| 4 | (0.00, -0.05) | 0.25 | Center-up |
| 5 | (-0.05, 0.05) | 0.30 | Slightly down-left |
| 6 | (0.15, -0.25) | 0.40 | Up-right again |
| 7 | (0.00, 0.00) | 0.50 | Return to center |

Transitions begin at 70% of each hold period and use a smoothstep
interpolation for a natural feel.

### Custom patterns

You can provide your own keyframe pattern:

```python
custom_pattern = [
    (-0.3, 0.1, 0.5),   # left, slightly down
    (0.3, -0.2, 0.5),   # right, slightly up
    (0.0, 0.0, 0.3),    # center
]
anim = ThinkingDotsAnimation(pattern=custom_pattern)
```

---

## SpeakingAnimation

Produces mouth open/close keyframes driven by a simplified **viseme model**
that maps English text to mouth shapes. Each viseme has a target openness
and width, and the animation steps through frames at a configurable speed.

### Viseme set

The animation uses 13 simplified visemes:

| Viseme | Sound | Openness | Width |
|--------|-------|----------|-------|
| `IDLE` | silence | 0.0 | 0.5 |
| `PP` | p, b, m | 0.0 | 0.4 |
| `FF` | f, v | 0.15 | 0.45 |
| `TH` | th | 0.2 | 0.5 |
| `DD` | t, d | 0.25 | 0.5 |
| `KK` | k, g | 0.3 | 0.5 |
| `CH` | ch, j, sh | 0.35 | 0.45 |
| `SS` | s, z | 0.15 | 0.5 |
| `NN` | n, ng | 0.2 | 0.5 |
| `RR` | r | 0.25 | 0.4 |
| `AA` | a | 0.7 | 0.7 |
| `EE` | e, i | 0.3 | 0.7 |
| `OO` | o, u | 0.4 | 0.35 |

### Usage

```python
from robot.face.animations.speaking import SpeakingAnimation

# From text
anim = SpeakingAnimation(text="Hello there!", speed=1.0)

while anim.has_frames:
    frame = anim.step(dt=0.033)
    # frame.openness  - mouth openness (0.0–1.0)
    # frame.width     - mouth width (0.0–1.0)
    # frame.viseme    - current Viseme enum
    # frame.duration_s - how long this frame should be held

# From explicit viseme sequence
from robot.face.animations.speaking import SpeakingAnimation, Viseme

anim2 = SpeakingAnimation.from_visemes(
    [(Viseme.AA, 0.1), (Viseme.PP, 0.06), (Viseme.SS, 0.08)],
    speed=1.2,
)
```

### Timing defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `default_phoneme_duration` | 0.08 s | Duration per phoneme |
| `pause_duration` | 0.15 s | Duration for commas/semicolons |
| `sentence_pause_duration` | 0.25 s | Duration for periods/exclamation marks |
| `speed` | 1.0 | Speed multiplier (0.5 = half speed, 2.0 = double) |

---

## WakeAnimation

A 4-phase attention-getting animation that plays when the robot detects a
wake word. The sequence creates a bright "I'm awake!" effect:

| Phase | Duration | What happens |
|-------|----------|--------------|
| **Eyes open** | 0.10 s | Eyes snap to center and open wide |
| **Double blink** | 0.30 s | Two rapid blinks |
| **Mouth surprise** | 0.20 s | Mouth opens to an "O" shape |
| **Transition** | 0.30 s | Smooth transition to a "curious" look |

### Usage

```python
from robot.face.animations.wake import WakeAnimation

anim = WakeAnimation()

while not anim.done:
    frame = anim.step(dt=0.033)
    # frame.gaze          - Gaze(x, y) target
    # frame.eye_openness  - how open the eyes are (0.0–1.2)
    # frame.eyelid_top     - eyelid position
    # frame.mouth_shape   - MouthShape enum
    # frame.mouth_openness - mouth openness
    # frame.mouth_width    - mouth width
    # frame.eyebrow_shape  - EyebrowShape enum
    # frame.eyebrow_raise  - eyebrow raise amount
    # frame.phase          - current WakePhase

# Reset for next wake
anim.reset()
```

---

## Standalone usage

All three animations are pure data objects with no dependencies on the event
bus or hardware. You can use them independently:

```python
import time
from robot.face.animations.thinking_dots import ThinkingDotsAnimation

anim = ThinkingDotsAnimation()
last = time.monotonic()

while True:
    now = time.monotonic()
    dt = now - last
    last = now

    gaze = anim.step(dt)
    # Apply gaze to your own face model or renderer
    print(f"Gaze: ({gaze.x:.2f}, {gaze.y:.2f})")
```

---

## API reference

::: robot.face.animations.thinking_dots.ThinkingDotsAnimation
    options:
      show_root_heading: true

::: robot.face.animations.speaking.SpeakingAnimation
    options:
      show_root_heading: true

::: robot.face.animations.speaking.Viseme
    options:
      show_root_heading: true

::: robot.face.animations.speaking.VisemeFrame
    options:
      show_root_heading: true

::: robot.face.animations.wake.WakeAnimation
    options:
      show_root_heading: true

::: robot.face.animations.wake.WakeFrame
    options:
      show_root_heading: true

::: robot.face.animations.wake.WakePhase
    options:
      show_root_heading: true
