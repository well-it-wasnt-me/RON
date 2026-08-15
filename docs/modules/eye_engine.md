# Legacy eye engine

`robot.eye_engine` is the original eye-only rendering stack.

It remains in the repository for compatibility and for the
`deskbot-eye-demo` command, but it is **not the primary face architecture**.

New expressive behavior should normally use `robot.face`.

The legacy stack contains:

- eye state
- blink control
- eye renderer
- eye animator
- eye-specific animation helpers

Use the face engine for new work involving eyebrows, mouth, cheeks, overlays,
accessories, themes, or body-language hints.
