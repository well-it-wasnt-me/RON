# RON Production Learning Plan

This plan turns RON's experimental learning stack into a production-safe learning system.

## Order matters

Do these phases in order. Do not skip ahead because a neural network looks more exciting than data semantics.

1. Fix transition semantics
2. Separate observation, action, reward
3. Make multimodal encoding deterministic
4. Train the world model on real transitions
5. Build a frozen evaluation dataset
6. Add shadow-mode policy inference
7. Add safety-gated actions
8. Add canary model deployment
9. Add controlled online learning

## Rule

Until Phase 6 is complete, the learned policy must not control the robot.

Each phase has:
- Goal
- Exact changes
- Files/components to touch
- Tests
- Definition of done
- Do not do this

Commit after every phase using SHORT conventional commit style message. AND NEVER PUSH.
