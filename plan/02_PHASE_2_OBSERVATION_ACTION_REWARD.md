# Phase 2: Separate Observation, Action and Reward

## Goal

Stop mixing things the robot observes with things the robot does.

## Target model

```text
Observation
Action
Outcome
Reward
Transition
```

## Create explicit types

Add:

```python
@dataclass(frozen=True)
class Observation:
    robot: RobotObservation
    vision: VisionObservation
    audio: AudioObservation
    timestamp_ns: int
```

```python
@dataclass(frozen=True)
class Transition:
    observation: Observation
    action: Action
    reward: float
    next_observation: Observation
    done: bool
    metadata: dict
```

Use the project's existing action-space definitions rather than inventing a second action representation.

## Important rule

These are observations:

- FaceDetected
- SpeechRecognized
- audio level
- face confidence
- sensor state
- user presence

These are actions:

- look_left
- look_right
- look_center
- blink
- wink
- celebrate
- sleep
- look_around
- other commands explicitly defined by the action space

Never encode `FaceDetected` as an action.

## Reward model

Move reward calculation out of the recorder.

Create a reward component:

```python
class RewardModel:
    def compute(
        self,
        state,
        action,
        next_state,
        events,
    ) -> float:
        ...
```

Keep reward policy configurable.

## Remove leakage

Do not put current/future reward information into the observation unless deliberately required by the learning algorithm.

If reward history is retained, document exactly why.

## Tests

Test that:

- every action is valid
- observations contain no future reward
- events are mapped to observations
- actions come only from ActionSpace
- reward is calculated after the outcome
- serialization round-trips correctly

## Definition of done

You can inspect one transition and clearly answer:

1. What did the robot know?
2. What did it do?
3. What happened?
4. What reward did it receive?

If those answers are mixed together, stop.
