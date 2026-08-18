# Phase 3: Make Multimodal Encoding Deterministic

## Goal

Turn the multimodal encoder into a reproducible representation function.

## Current issue

The encoder mutates its internal history when `encode()` is called.

That means:

```python
encoder.encode()
encoder.encode()
```

can produce different outputs even when the underlying observation did not change.

That is bad for replay, debugging and evaluation.

## Required design

History management belongs outside the encoder.

Use something like:

```python
@dataclass(frozen=True)
class ObservationContext:
    current: Observation
    history: tuple[Observation, ...]
```

Then:

```python
encoded = encoder.encode(context)
```

The same context must always produce the same vector.

## Do not train the modality encoders yet

For this phase:

- vision encoder = deterministic feature normalization
- audio encoder = deterministic feature normalization
- robot state = deterministic encoding
- temporal context = deterministic fixed-size history

Do not use arbitrary trainable MLP targets.

## Recommended representation

```text
Vision features
      |
Audio features
      |
Robot state
      |
      v
Observation encoder
      |
      v
Temporal encoder
      |
      v
64-128 dimensional latent state
```

A simple first temporal encoder can be an MLP over a fixed history window.

## Validate

Add tests for:

- same input => identical output
- no NaN
- no inf
- fixed output dimension
- history ordering
- empty history
- full history
- reset behavior
- serialization/version compatibility

## Definition of done

Given a frozen observation fixture, encoding it 10,000 times produces exactly the same result.

That is your standard.

## Do not do this

- Do not add transformers.
- Do not add pretrained models.
- Do not train vision/audio embeddings.
- Do not increase dimensions just because 570 feels scientifically impressive.

Make it correct before making it clever.
