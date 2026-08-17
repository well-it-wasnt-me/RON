# Face engine

The face engine is DeskBot's primary expressive system. It renders a complete
face on one display and coordinates emotion, gaze, blinking, mouth animation,
overlays, accessories, and body-language hints.

## Components

- `FaceModel`: immutable snapshot of the complete face.
- `EmotionEngine`: maps an emotion name to a `FaceModel` and body-language hint.
- `FaceRenderer`: display-independent drawing.
- `FaceAnimator`: animation loop and state transitions.
- `FaceOrchestrator`: connects face animation to application events.
- `face.themes`: six visual themes.
- `face.animations`: reusable speaking, wake, and thinking-dot animations.

## Emotions

The current emotion catalogue is:

`neutral`, `happy`, `curious`, `thinking`, `sleepy`, `embarrassed`,
`excited`, `sad`, `surprised`, `angry`.

Each emotion is more than a facial label. It can specify eyes, eyelids,
eyebrows, mouth, cheeks, overlays, and a high-level body-language hint.

## Themes

The renderer currently supports:

- Vector
- Minimal
- Cute
- Pixel
- Retro LCD
- Wireframe

Select one with:

```env
DESKBOT_FACE__THEME=vector
```

## Face model

`FaceModel` contains:

- left and right eyes
- eyelids
- left and right eyebrows
- mouth
- cheeks
- overlay
- accessory
- colour palette
- bounce/squash transform
- body-language hint

The model is frozen. Animators create new snapshots rather than mutating the
current one.

## Animation

`FaceAnimator` renders at the configured display frame rate, normally 30 FPS.
The animation framework supplies easing, timelines, and scheduling.

The face also reacts to streamed LLM output. `FaceOrchestrator` consumes
`LLMTokenReceived` events and changes the face between thinking and speaking
states.

## Rendering order

The renderer composes the face from background through transforms, cheeks,
eyes, eyelids, irises, pupils, highlights, eyebrows, mouth, overlays, and
accessories.

The renderer does not require a physical display. A mock display can capture
frames during tests.

## Adding an emotion

1. Add the emotion name to `EmotionName` in `robot.events.events`.
2. Add its definition to `robot.face.emotions.EMOTION_DEFS`.
3. Define the face components and optional `BodyLanguageHint`.
4. Add/update tests.
5. Update the documentation if the public catalogue changes.
