# Animation framework

The animation package provides reusable timing primitives independent of the
face and servo implementations.

## Easing

`robot.animation.easing` contains pure easing functions mapping normalized
progress to normalized progress.

## Timelines

The framework includes:

- `Wait`: delay without changing state.
- `Tween`: interpolate a scalar and invoke a callback.
- `Parallel`: run animations concurrently.
- `Queue`: run animations sequentially.
- `Timeline`: fluent construction of animation sequences.

## Scheduler

`AnimationScheduler` runs one-shot or periodic jobs using an async task group.
Scheduled jobs can be cancelled through `ScheduledTask`.

## Where it is used

The same animation concepts are used by face animation and body-language
orchestration. This keeps timing and interpolation logic out of individual
hardware drivers.
