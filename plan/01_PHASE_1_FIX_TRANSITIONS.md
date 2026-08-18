# Phase 1: Fix Transition Semantics

## Goal

Make every stored experience represent a real transition:

`state -> action -> physical outcome -> next_state -> reward`

Right now, do not train harder. Fix the data first.

## Problem

`ExperienceRecorder.record_with_encoder()` can encode a state and then immediately encode another state without an action being executed.

That creates fake transitions.

## Required architecture

```text
OBSERVE
  |
  v
state_t
  |
  v
SELECT ACTION
  |
  v
EXECUTE ACTION
  |
  v
WAIT FOR COMPLETION / OUTCOME WINDOW
  |
  v
state_t+1
  |
  v
COMPUTE REWARD
  |
  v
STORE TRANSITION
```

## Changes

### 1. Create an open transition

Add a transition lifecycle such as:

```python
transition = transition_store.begin(
    state=state,
    action=action,
)
```

Do not write a completed experience yet.

### 2. Close it after execution

When the executor reports completion:

```python
transition.complete(
    next_state=next_state,
    reward=reward,
    done=done,
)
```

### 3. Require action identity

Every transition must contain the real action selected from `ActionSpace`.

Do not use event type as the action.

### 4. Record execution metadata

Include:

- execution ID
- action ID
- start timestamp
- completion timestamp
- execution success/failure
- latency
- originating policy/model version

## Tests

Write tests proving:

- no action means no completed transition
- failed action execution is recorded correctly
- next_state is captured after execution
- action ID belongs to the configured action space
- timestamps are monotonic
- malformed transitions are rejected

## Definition of done

A stored transition can be reconstructed as:

```text
At T0:
observation = X

At T0:
policy selected action = Y

Robot executed Y

At T1:
observation = Z

Reward = R
```

If you cannot reconstruct that sequence from storage, Phase 1 is not done.

## Do not do this

- Do not train a new model.
- Do not increase network size.
- Do not enable exploration.
- Do not make the learned policy control hardware.

Humans love optimizing garbage before checking whether it is garbage. Resist.
