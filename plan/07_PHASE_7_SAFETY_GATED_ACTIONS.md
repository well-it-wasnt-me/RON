# Phase 7: Safety-Gated Actions

## Goal

Make it impossible for a learned policy to bypass robot safety rules.

## Three layers

### 1. Static validation

Check:

- action exists
- parameters are valid
- servo limits
- timing limits
- rate limits

### 2. Runtime safety

Check current conditions before execution.

Examples:

- calibrated servo range
- action cooldown
- conflicting actions
- sensor availability
- robot state restrictions

### 3. Manual/emergency override

Provide a reliable mechanism to disable learned control immediately.

## Architecture

```text
learned policy
      |
      v
action candidate
      |
      v
safety validator
      |
   +--+--+
   |     |
 allow  reject
   |     |
   v     v
executor fallback
```

## Important rule

No HTTP endpoint, event handler, training component or policy can bypass the safety validator.

Every hardware action uses the same execution path.

## Safe fallback

When the policy:

- crashes
- returns NaN
- returns an invalid action
- times out
- cannot load
- fails validation

fall back to deterministic behavior.

Never leave the robot without a valid controller.

## Tests

Inject:

- invalid actions
- NaN scores
- missing sensors
- timeout
- corrupted model
- impossible servo positions
- rapid repeated actions

Verify the robot falls back safely.

## Definition of done

A learned policy cannot directly command hardware.

It can only propose an action.
