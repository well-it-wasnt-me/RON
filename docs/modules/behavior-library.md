# Behavior library

The `robot.behavior_library` package provides reusable, higher-level behavior
sequences that compose face and body-language steps into named flows.

> See also the [behavior engine](behavior.md) for the lower-level state
> machine, reactions, and idle behavior.

---

## Concepts

| Concept | Description |
|---------|-------------|
| `Behavior` | A named sequence of `BehaviorStep` values |
| `BehaviorStep` | One action: a face method call, a body request, or a wait |
| `BehaviorRunner` | Executes a `Behavior` step by step with optional timing |

Steps are pure data — they describe *what* should happen, not *how*. The
runner translates them into event-bus publications and servo commands.

### Step builders

| Builder | Purpose |
|---------|---------|
| `face(method, *args)` | Call a method on the face orchestrator |
| `body(request)` | Send a `BodyRequest` to the body-language engine |
| `wait(seconds)` | Pause between steps |

---

## Built-in behaviors

| Behavior | Description |
|----------|-------------|
| `greeting()` | Welcome sequence: eyes open, smile, wave |
| `thinking()` | Thinking pose: look up, tilt head |
| `listening()` | Attentive listening: face the user, nod |
| `sleeping()` | Sleep sequence: eyes close, relax arms |
| `excited()` | Excitement: celebrate with arms up |
| `surprised()` | Surprise: eyes wide, recoil |

---

## Usage

```python
from robot.behavior_library.behavior import greeting, BehaviorRunner

runner = BehaviorRunner(face_orchestrator=face, body_engine=body)
await runner.run(greeting())
```

Behaviors are composable — you can build your own by chaining steps:

```python
from robot.behavior_library.behavior import Behavior, face, body, wait
from robot.body_language.requests import Wave

custom = Behavior(
    name="wave_and_greet",
    steps=[
        face("set_emotion", "happy"),
        body(Wave()),
        wait(0.5),
        face("set_emotion", "neutral"),
    ],
)
```

---

## API reference

::: robot.behavior_library
    options:
      show_root_heading: true
