"""Emotion parameters that the renderer knows how to draw."""

from __future__ import annotations

from dataclasses import dataclass

from robot.events.events import EmotionName
from robot.eye_engine.eye_state import EyeState, GazeVector


@dataclass(slots=True, frozen=True)
class Emotion:
    """A fully-resolved eye state for a given emotion."""

    name: EmotionName
    openness: float
    pupil_dilation: float
    gaze: GazeVector
    asymmetric: bool = False

    def to_eye_state(self, intensity: float = 1.0) -> EyeState:
        return EyeState(
            emotion=self.name,
            openness=self.openness,
            pupil_dilation=self.pupil_dilation,
            gaze=self.gaze,
            intensity=intensity,
            asymmetric=self.asymmetric,
        )


class EmotionLibrary:
    """Default catalogue of emotions. New emotions can be added at runtime."""

    def __init__(self) -> None:
        self._emotions: dict[EmotionName, Emotion] = {
            EmotionName.NEUTRAL: Emotion(
                EmotionName.NEUTRAL, openness=1.0, pupil_dilation=0.5, gaze=GazeVector(0.0, 0.0)
            ),
            EmotionName.HAPPY: Emotion(
                EmotionName.HAPPY,
                openness=0.65,
                pupil_dilation=0.45,
                gaze=GazeVector(0.0, 0.05),
                asymmetric=False,
            ),
            EmotionName.SAD: Emotion(
                EmotionName.SAD,
                openness=0.7,
                pupil_dilation=0.7,
                gaze=GazeVector(0.0, -0.3),
                asymmetric=False,
            ),
            EmotionName.ANGRY: Emotion(
                EmotionName.ANGRY,
                openness=0.85,
                pupil_dilation=0.3,
                gaze=GazeVector(0.0, -0.15),
                asymmetric=False,
            ),
            EmotionName.SURPRISED: Emotion(
                EmotionName.SURPRISED,
                openness=1.0,
                pupil_dilation=0.2,
                gaze=GazeVector(0.0, 0.0),
                asymmetric=False,
            ),
            EmotionName.SLEEPY: Emotion(
                EmotionName.SLEEPY,
                openness=0.25,
                pupil_dilation=0.7,
                gaze=GazeVector(0.0, 0.0),
                asymmetric=False,
            ),
            EmotionName.THINKING: Emotion(
                EmotionName.THINKING,
                openness=0.7,
                pupil_dilation=0.5,
                gaze=GazeVector(0.25, 0.0),
                asymmetric=False,
            ),
            EmotionName.CURIOUS: Emotion(
                EmotionName.CURIOUS,
                openness=0.95,
                pupil_dilation=0.55,
                gaze=GazeVector(0.10, 0.10),
                asymmetric=True,
            ),
            EmotionName.EXCITED: Emotion(
                EmotionName.EXCITED,
                openness=1.0,
                pupil_dilation=0.3,
                gaze=GazeVector(0.0, -0.05),
                asymmetric=False,
            ),
            EmotionName.EMBARRASSED: Emotion(
                EmotionName.EMBARRASSED,
                openness=0.75,
                pupil_dilation=0.6,
                gaze=GazeVector(0.0, -0.20),
                asymmetric=False,
            ),
        }

    def get(self, name: EmotionName) -> Emotion:
        try:
            return self._emotions[name]
        except KeyError as exc:
            raise ValueError(f"unknown emotion: {name!r}") from exc

    def register(self, emotion: Emotion) -> None:
        self._emotions[emotion.name] = emotion

    def available(self) -> list[EmotionName]:
        return list(self._emotions.keys())


__all__ = ["Emotion", "EmotionLibrary"]
