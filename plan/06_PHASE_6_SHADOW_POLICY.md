# Phase 6: Shadow-Mode Policy

## Goal

Run the learned policy against live observations without allowing it to control the robot.

## Modes

Implement:

```text
off
shadow
assist
active
```

### off

No learned inference.

### shadow

The model predicts an action but the normal controller executes the real action.

### assist

The model may suggest actions, but deterministic logic can reject them.

### active

Only explicitly approved actions can be controlled by the learned policy.

## Shadow logging

For every decision log:

```text
timestamp
observation ID
current behavior action
model action
model scores/Q-values
model version
safety result
inference latency
```

## Compare behavior

Measure:

- policy agreement
- predicted reward
- disagreement cases
- unsafe proposals
- latency
- confidence/Q-value margins

## Runtime rule

In shadow mode:

```python
learned_action = policy.predict(observation)

# DO NOT execute learned_action

actual_action = deterministic_controller.select(observation)
executor.execute(actual_action)
```

## Duration

Run continuously for a meaningful period before allowing active control.

A good first target is 24+ hours without:

- crashes
- malformed logs
- model load failures
- safety violations
- performance regressions

## Definition of done

You can prove from logs that the learned policy was active in inference but had zero authority to execute hardware actions.

This is the first genuinely production-relevant milestone.
