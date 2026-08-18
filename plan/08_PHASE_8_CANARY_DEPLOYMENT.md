# Phase 8: Canary Model Deployment

## Goal

Deploy learned models gradually and reversibly.

## Model registry

Every model needs metadata:

```json
{
  "model_version": 17,
  "schema_version": 3,
  "state_encoder_version": 2,
  "multimodal_version": 2,
  "action_space_version": 4,
  "git_commit": "...",
  "dataset_version": "...",
  "training_run": "...",
  "validation": {
    "loss": 0.0,
    "reward": 0.0,
    "safety_violations": 0,
    "latency_ms_p95": 0.0
  }
}
```

## Atomic deployment

Never overwrite the active checkpoint directly.

Write a temporary file, flush it, then atomically replace the active pointer/file.

On startup validate:

- schema
- dimensions
- checksum
- finite weights
- action-space version
- encoder version

If invalid, load the previous known-good model.

## Canary stages

```text
candidate
   |
   v
offline evaluation
   |
   v
shadow
   |
   v
small action subset
   |
   v
limited active use
   |
   v
full approved deployment
```

## Rollback

Rollback must be a single operation.

Keep the previous known-good model.

## Promotion criteria

Require:

- zero safety violations
- zero invalid actions
- no crashes
- latency within limit
- benchmark pass
- real-world metrics not worse than baseline

## Definition of done

You can deploy a candidate and return to the previous model without rebuilding software or physically accessing the robot.
