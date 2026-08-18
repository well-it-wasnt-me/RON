# Phase 9: Controlled Online Learning

## Goal

Only now allow production data to update the learning system.

## Preconditions

Do not start this phase until Phases 1-8 are complete.

You need:

- valid transitions
- persistent replay
- frozen evaluation dataset
- shadow results
- safety gates
- rollback
- model registry
- observability

## Process

```text
live robot
    |
    v
persistent transitions
    |
    v
replay dataset
    |
    v
background training
    |
    v
candidate
    |
    v
offline evaluation
    |
    v
shadow evaluation
    |
    v
promotion
```

## Separate processes

Prefer:

```text
robot.service
learning.service
```

The robot process handles:

- perception
- behavior
- inference
- hardware

The learning process handles:

- replay
- training
- evaluation
- candidate creation

Training must never be allowed to starve the real-time/control loop.

## Persistence

Persist:

- experiences
- training runs
- model versions
- evaluation results
- safety events
- policy decisions

Warm the replay buffer from persistent storage after reboot.

## Exploration

Do not use unrestricted epsilon-greedy exploration on the physical robot.

Explore in:

- simulation
- offline replay
- shadow mode
- constrained action subsets

## Monitoring

Track:

- model version
- replay size
- training rate
- training loss
- validation loss
- reward
- action distribution
- safety rejections
- fallback rate
- sensor dropout
- inference latency
- model load failures

## Definition of done

The robot can learn continuously while:

- the control process remains responsive
- every model is reproducible
- every promotion is auditable
- rollback remains immediate
- safety rules remain authoritative
