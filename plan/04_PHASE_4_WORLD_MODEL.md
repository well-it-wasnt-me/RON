# Phase 4: Train the World Model on Real Transitions

## Goal

Train the world model using valid physical transitions.

## Target

```text
latent_state_t + action_t
            |
            v
      world model
            |
            v
predicted latent_state_t+1
```

## Dataset requirements

Only train from completed transitions.

Reject:

- missing next state
- missing action
- invalid action
- NaN/inf
- impossible timestamps
- corrupted observation vectors
- transitions created without execution

## Split the data

Create:

```text
train
validation
test
```

Do not randomly leak near-identical consecutive samples across all sets if temporal correlation makes the split meaningless.

Prefer time-based or episode-based splits.

## Metrics

Track:

- training loss
- validation loss
- test loss
- per-feature error
- prediction latency
- invalid prediction count

## Baseline

Before training, create a trivial baseline.

Examples:

- predict next state = current state
- predict feature mean
- simple persistence model

Your learned model must beat the baseline.

## Evaluation

Run the world model on fixed fixtures:

- face + silence
- face + speech
- no face + speech
- no face + silence
- moving face
- sensor dropout
- microphone dropout
- camera dropout

## Definition of done

A reproducible training run:

```text
dataset version
+
code commit
+
hyperparameters
=
same evaluation result
```

and the model beats the baseline on the frozen test set.

## Do not do this

Do not promote a model because training loss decreased.

Validation and test performance are what matter.
