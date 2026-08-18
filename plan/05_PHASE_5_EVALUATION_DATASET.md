# Phase 5: Build a Frozen Evaluation Dataset

## Goal

Create a permanent benchmark that every candidate model must pass.

## Why

If the evaluation data changes every run, you cannot tell whether the model improved.

## Dataset structure

```text
datasets/
  evaluation/
    v1/
      metadata.json
      observations/
      transitions/
      scenarios/
```

## Include scenarios

At minimum:

1. face present + silence
2. face present + speech
3. no face + speech
4. no face + silence
5. moving face
6. multiple faces
7. low confidence face detection
8. high audio energy
9. low audio energy
10. camera dropout
11. microphone dropout
12. malformed sensor input
13. idle state
14. interaction state

## Record expected behavior

For each fixture define:

- valid actions
- preferred actions
- forbidden actions
- expected safety behavior

## Metrics

Every candidate receives:

- world model loss
- policy reward
- invalid actions
- safety violations
- inference latency
- NaN/inf count
- fallback count

## Promotion rule

A candidate must not be promoted unless:

```text
safety violations == 0
invalid actions == 0
NaN/inf == 0
latency within limit
world model >= baseline
policy performance >= baseline
```

Set exact thresholds in configuration and version them.

## Definition of done

The same candidate evaluated twice gets the same result.

The benchmark is immutable once versioned.
