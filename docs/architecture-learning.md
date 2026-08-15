# DeskBot Learning Architecture

## Overview

DeskBot's local learning system enables the robot to observe its environment,
record experience, learn measurable relationships from that experience, retain
what it learned, and safely use the learned knowledge to improve behaviour —
all running on-device with no external inference APIs or pretrained models.

## Data Flow

```mermaid
flowchart TD
    A["Perception (camera, mic, events)"] --> B["State Encoder (multimodal: vision + audio + robot state + history)"]
    B --> C["Experience Memory (working memory -> replay buffer -> episodic storage)"]
    C --> D["Local Neural Network (world model + action learner + preference learner)"]
    D --> E["Prediction / Policy (next state prediction, action selection, preference recall)"]
    E --> F["Safety Validator (action validation, model evaluation, rollback)"]
    F --> G["DeskBot Behaviour (events, reactions, conversation)"]
    G --> H["Robot (servos, display, speech)"]
    H --> I["Feedback (rewards, observations)"]
    I --> C
```

## Components

### Phase 1–2: Foundation
- **Tensor / Network / MLP** — Minimal neural-network framework built from scratch
- **Experience Memory** — WorkingMemory (ring buffer) -> ReplayBuffer (random access) -> EpisodicMemory (SQLite persistence)

### Phase 3: State Encoder
- **StateEncoder** — Deterministic, fixed-size vector (91 elements) from robot state, emotions, servos, vision, audio, personality, flags, rewards
- **VisionFeatures** / **AudioFeatures** — Structured feature extraction from face detection and PCM audio

### Phase 4: World Model
- **WorldModel** — MLP that predicts next_state from (state, action); trained on experience replay
- **SimpleEnvironment** — 2-D simulation for testing prediction learning

### Phase 5: Action Learning
- **ActionLearner** — Q-learning with function approximation; epsilon-greedy exploration
- **ActionSpace** — Registry of valid DeskBot actions with safety validation
- **ActionLearningEnv** — Simulation environment with reward structure

### Phase 6: Continual Learning
- **LearningService** — Background daemon thread that:
  - Subscribes to the event bus via ExperienceRecorder
  - Collects experiences in replay buffer
  - Trains candidate model in background (never blocks main loop)
  - Evaluates candidate vs current model
  - Promotes or rolls back based on evaluation
  - Saves checkpoints for recovery
- **LearningSchedule** — Configurable: train every N experiences, minimum interval
- **ResourceLimits** — Configurable: CPU fraction, batch size, epochs, model size
- **CheckpointConfig** — Configurable: directory, keep_last_n, promote_threshold

### Phase 7: Multimodal Learning
- **VisionEncoder** — Trainable MLP (6->32->16->16) encoding face detection features
- **AudioEncoder** — Trainable MLP (3->16->8->8) encoding audio signal features
- **HistoryBuffer** — Ring buffer of recent state vectors for temporal context
- **MultimodalEncoder** — Concatenates robot_state(91) + vision_encoded(16) + audio_encoded(8) + history(91×5) = 570 elements
- **MultimodalEnvironment** — Simulation where vision, audio, and both together give different reward signals

### Phase 8: Preference Learning
- **PreferenceLearner** — Observes recurring patterns with confidence scores
  - Confidence increases with repeated observations
  - Confidence decays over time without reinforcement
  - Explicit observations boost more than inferred
  - Never infers sensitive personal attributes
- **LearnedPreference** — Category/value pairs with confidence, observation count, avg reward
- **Persistence** — Via existing PreferenceStore (in-memory or SQLite)

### Phase 9: Safety, Evaluation & Rollback
- **ModelEvaluator** — Compares candidate vs current model on fixed evaluation set
- **EvaluationThresholds** — Configurable: min_improvement_ratio, max_loss, max_latency, max_prediction_std
- **ActionSafetyValidator** — Blocks invalid/unsafe actions (unknown names, out-of-range params, rate limits)
- **LearningSafetyManager** — Coordinates evaluation, validation, and checkpoint management
- **Graceful degradation** — Corrupted checkpoints, bad candidates, training exceptions, missing sensors all handled without crashing the robot

### Phase 10: Production Integration
- **Configuration** — All learning parameters configurable via environment variables (DESKBOT_LEARNING__*)
- **CLI** — `deskbot-learning-status`, `deskbot-learning-train`, `deskbot-learning-evaluate`, `deskbot-learning-reset`, `deskbot-learning-export`
- **REST API** — `/api/v1/learning/status`, `/api/v1/learning/preferences`, `/api/v1/learning/config`, `/api/v1/learning/train`
- **Event Bus Integration** — LearningService subscribes to events via ExperienceRecorder
- **Lifecycle** — Learning starts/stops with the app; failures are isolated

## Configuration

All learning parameters are configurable via `DESKBOT_LEARNING__*` environment variables:

| Variable                                      | Default                  | Description                             |
|-----------------------------------------------|--------------------------|-----------------------------------------|
| `DESKBOT_LEARNING__ENABLED`                   | `false`                  | Enable/disable the learning system      |
| `DESKBOT_LEARNING__MIN_NEW_EXPERIENCES`       | `32`                     | Min new experiences before training     |
| `DESKBOT_LEARNING__TRAIN_INTERVAL_S`          | `30.0`                   | Min seconds between training cycles     |
| `DESKBOT_LEARNING__BATCH_SIZE`                | `32`                     | Mini-batch size                         |
| `DESKBOT_LEARNING__TRAINING_EPOCHS_PER_CYCLE` | `5`                      | Epochs per training cycle               |
| `DESKBOT_LEARNING__MAX_CPU_FRACTION`          | `0.3`                    | Target CPU fraction for training thread |
| `DESKBOT_LEARNING__PROMOTE_THRESHOLD`         | `1.0`                    | Candidate must be this factor better    |
| `DESKBOT_LEARNING__CHECKPOINT_DIR`            | `~/.deskbot/checkpoints` | Model checkpoint directory              |
| `DESKBOT_LEARNING__KEEP_LAST_N_CHECKPOINTS`   | `5`                      | Checkpoints to keep on disk             |

## Design Principles

1. **Local-only** — No external AI services, pretrained models, or cloud APIs
2. **Non-blocking** — Training runs in a daemon thread, never blocks the main event loop
3. **Safe** — Candidates are evaluated before promotion; rollback is immediate
4. **Observable** — Status, configuration, and preferences accessible via API
5. **Configurable** — All parameters tunable via environment variables
6. **Graceful degradation** — Learning failures never crash the robot
