# Interfaces & protocol contracts

The `robot.interfaces` package defines the protocol-based boundaries between
application logic and hardware/provider implementations. Every concrete backend
depends on one of these protocols; application code never imports a concrete
hardware driver directly.

> See also the [Architecture overview](../architecture/overview.md) for how these
> contracts decouple behavior from hardware.

---

## Why protocols?

DeskBot uses `typing.Protocol` (structural subtyping) rather than ABCs. This
means a class is a valid implementation if it has the right methods — no
inheritance required. Mocks, fakes, and third-party backends all satisfy the
protocol automatically.

Every protocol is `@runtime_checkable` so `isinstance()` works for validation
in factories and tests.

---

## Protocol catalogue

| Protocol | Package | Purpose |
|----------|---------|---------|
| `Display` | `robot.interfaces.display` | Push pixel frames to a screen |
| `ServoController` | `robot.interfaces.servo` | Move servos to target angles |
| `AudioOutput` | `robot.interfaces.audio` | Play audio buffers; carries the `AudioBuffer` format contract |
| `Microphone` | `robot.interfaces.microphone` | Stream audio chunks from an input device |
| `Camera` | `robot.interfaces.camera` | Capture image frames |
| `LLM` | `robot.interfaces.llm` | Generate a chat completion from message history |
| `StreamingLLM` | `robot.interfaces.streaming_llm` | Token-by-token streaming generation |
| `SpeechToText` | `robot.speech.stt` (via `LLM`-style protocol) | Transcribe audio to text |
| `EventBus` | `robot.interfaces.event_bus` | Async pub/sub event distribution |
| `Clock` | `robot.utils.clock` | Time source (injectable for deterministic tests) |
| `RandomSource` | `robot.utils.random_source` | Randomness source (injectable for deterministic tests) |

---

## Audio format contract

`AudioBuffer` is the central data contract between TTS engines (producers) and
output devices (consumers):

```python
@dataclass
class AudioBuffer:
    pcm: bytes           # raw signed 16-bit little-endian PCM
    sample_rate: int     # Hz (e.g. 22050, 24000, 44100)
    channels: int        # 1 = mono, 2 = stereo
    sample_format: str   # "s16le" (currently the only supported format)
```

Every TTS engine reports its actual output format; every output device knows
exactly what it receives. No format guessing. See
[Audio Architecture](../architecture/audio.md) for the full design.

---

## Adding a new backend

1. Implement the relevant protocol under `robot.hardware`, `robot.ai`, or
   `robot.speech`.
2. Register it in the corresponding factory (e.g. `DisplayFactory`,
   `ServoControllerFactory`).
3. Add configuration fields to the matching `*Config` settings model.
4. Add unit tests with fakes/mocks — the protocol makes this trivial.
5. Document the backend and its limitations.

Application behavior should not need to know which concrete driver is active.
