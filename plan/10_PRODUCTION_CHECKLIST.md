# Production Checklist

Use this before enabling learned control.

## Data

- [ ] Every transition has state, action, next_state, reward
- [ ] Action comes from ActionSpace
- [ ] Events are observations, not actions
- [ ] No future information leaks into state
- [ ] Invalid transitions are rejected
- [ ] Transitions survive reboot

## Encoding

- [ ] Encoder is deterministic
- [ ] Version is recorded
- [ ] Dimensions are validated
- [ ] NaN/inf are rejected
- [ ] History semantics are tested

## Training

- [ ] Train/validation/test split exists
- [ ] Frozen evaluation dataset exists
- [ ] Baseline exists
- [ ] Training runs are reproducible
- [ ] Candidate models have metadata

## Safety

- [ ] Learned actions pass one safety validator
- [ ] Hardware cannot be controlled directly by training
- [ ] Invalid model output triggers fallback
- [ ] Model timeout triggers fallback
- [ ] Manual override works

## Deployment

- [ ] Models are atomically loaded
- [ ] Previous model is retained
- [ ] Rollback works
- [ ] Shadow mode works
- [ ] Canary deployment works

## Observability

- [ ] Model version logged
- [ ] Policy decisions logged
- [ ] Safety rejections logged
- [ ] Inference latency measured
- [ ] Training metrics measured
- [ ] Crash/fallback counts measured

## Final gate

The learned policy remains disabled until every box above is checked.
