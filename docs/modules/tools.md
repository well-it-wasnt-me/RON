# Tool / Function-Calling System

DeskBot's **tool system** allows the LLM to invoke robot actions (change
emotion, move servos, play sounds, speak text) through a structured
function-calling interface compatible with OpenAI, Ollama, and other
LLM providers.

The system has three layers:

1. **Schema** - `ToolDefinition` and `ToolParameter` define the shape of a
   tool and serialise to the OpenAI function-calling JSON format.
2. **Registry** - `ToolRegistry` holds definitions and their handler
   functions, and can execute tool calls by name.
3. **Executor** - `ToolExecutor` dispatches LLM tool calls to robot actions
   via the event bus and hardware controllers.

---

## Tool definition schema

Each tool is described by a [`ToolDefinition`][robot.ai.tools.schema.ToolDefinition]
with a name, description, and a tuple of [`ToolParameter`][robot.ai.tools.schema.ToolParameter]
objects:

```python
from robot.ai.tools.schema import ToolDefinition, ToolParameter, ToolParameterType

my_tool = ToolDefinition(
    name="change_emotion",
    description="Change the robot's emotional expression.",
    parameters=(
        ToolParameter(
            name="emotion",
            type=ToolParameterType.STRING,
            description="The emotion to express.",
            enum=("neutral", "happy", "curious", "thinking", "sleepy",
                   "embarrassed", "excited", "sad", "surprised", "angry"),
        ),
        ToolParameter(
            name="intensity",
            type=ToolParameterType.NUMBER,
            description="Intensity of the emotion (0.0 to 1.0).",
            required=False,
            default=1.0,
        ),
    ),
)
```

### OpenAI-compatible schema

Every `ToolDefinition` can produce an OpenAI function-calling schema:

```python
schema = my_tool.to_openai_schema()
# {
#   "type": "function",
#   "function": {
#     "name": "change_emotion",
#     "description": "Change the robot's emotional expression.",
#     "parameters": { ... }
#   }
# }
```

This schema can be passed directly to any OpenAI-compatible LLM as a tool
definition.

---

## Tool registry

The [`ToolRegistry`][robot.ai.tools.registry.ToolRegistry] manages tool
definitions and their handler functions:

```python
from robot.ai.tools.registry import ToolRegistry, BUILTIN_TOOLS

registry = ToolRegistry()

# Register a built-in tool with its handler
registry.add(BUILTIN_TOOLS["change_emotion"], my_handler)

# Get all OpenAI schemas for an LLM prompt
schemas = registry.get_schemas()

# Execute a tool call from the LLM
result = await registry.execute("change_emotion", {"emotion": "happy"})
```

### Methods

| Method | Description |
|--------|-------------|
| `add(definition, handler)` | Register a tool and its async handler |
| `remove(name)` | Remove a tool by name |
| `get(name)` | Return a tool definition by name |
| `get_handler(name)` | Return a tool handler by name |
| `list_tools()` | Return all registered tool definitions |
| `get_schemas()` | Return OpenAI-compatible schemas for all tools |
| `execute(name, arguments)` | Execute a tool by name with arguments |

---

## Built-in tools

DeskBot ships with 5 built-in tools defined in
[`BUILTIN_TOOLS`](#built-in-tools):

| Tool | Description | Parameters |
|------|-------------|------------|
| `change_emotion` | Change the robot's emotional expression | `emotion` (enum), `intensity` (number, optional) |
| `play_sound` | Play a sound effect through the speaker | `name` (string) |
| `set_state` | Transition the robot to a new state | `state` (enum) |
| `move_servo` | Move a servo to a target angle | `servo` (enum), `angle` (number), `duration_s` (number, optional) |
| `speak` | Make the robot say something using TTS | `text` (string) |

---

## Tool executor

The [`ToolExecutor`][robot.ai.tools.executor.ToolExecutor] validates
arguments, dispatches built-in tool calls to the event bus, and delegates
custom tool calls to their registered handlers:

```python
from robot.ai.tools.executor import ToolExecutor

executor = ToolExecutor(
    registry=registry,
    bus=event_bus,
    servo_controller=servo_ctrl,
    tts=tts_engine,
    action_executor=action_executor,  # optional: route learnable tools through the executor
)

result = await executor.execute_tool_call("change_emotion", {"emotion": "happy"})
# {"status": "ok", "emotion": "happy", "intensity": 1.0}
```

### Built-in tool dispatch

When an `action_executor` is **not** wired (the legacy path), built-in tools
dispatch directly to the event bus and hardware controllers:

| Tool | Dispatch target (legacy) |
|------|----------------|
| `change_emotion` | Publishes `EmotionChanged` + `BlinkRequested` on the event bus |
| `play_sound` | Publishes `SoundEffectPlayed` on the event bus |
| `set_state` | Publishes `StateChanged` on the event bus |
| `move_servo` | Calls `ServoController.get(name).move_to()` + publishes `ServoMoved` |
| `speak` | Calls `TextToSpeech.speak(text)` |

#### Learning-aware routing

When an `action_executor` **is** wired (the default once learning is enabled in
`DeskBotApp`), the **learnable** builtins — `change_emotion`, `set_state`,
`move_servo`, and `speak` — are routed through the canonical
[`ActionExecutor`](services.md#actionexecutor) instead of going
directly to the bus/hardware. Each call is translated into the matching
`BehaviorAction` (`ChangeEmotionAction`, `SetStateAction`,
`RequestServoMoveAction`, `SpeakAction`) and dispatched via
`ActionExecutor.execute_one`, so the LLM tool call is recorded as a **real
learning transition** with the same lifecycle, safety checks, and interaction
tagging as any other action (see [Services](services.md)). `play_sound` is
intentionally **not** routed (it is not in the action space) and is logged as
`action_not_learnable`.

### Argument validation

The executor validates required parameters and coerces types. If a
required parameter is missing or an unknown tool name is provided, a
[`ToolExecutionError`][robot.ai.tools.executor.ToolExecutionError] is
raised.

---

## Registering custom tools

To add a new tool:

```python
from robot.ai.tools.schema import ToolDefinition, ToolParameter, ToolParameterType
from robot.ai.tools.registry import ToolRegistry

registry = ToolRegistry()

# Define the tool
my_tool = ToolDefinition(
    name="set_led_color",
    description="Set the LED strip to a specific colour.",
    parameters=(
        ToolParameter(
            name="color",
            type=ToolParameterType.STRING,
            description="Hex colour code (e.g. '#ff0000').",
        ),
    ),
)

# Define the handler
async def handle_set_led(color: str) -> dict:
    # ... drive hardware ...
    return {"status": "ok", "color": color}

# Register
registry.add(my_tool, handle_set_led)
```

The tool's OpenAI schema will automatically appear in
`registry.get_schemas()` for inclusion in LLM prompts.

---

## API reference

::: robot.ai.tools.schema
    options:
      show_root_heading: true

::: robot.ai.tools.registry
    options:
      show_root_heading: true

::: robot.ai.tools.executor
    options:
      show_root_heading: true
