"""Pure data describing the current eye state."""

from __future__ import annotations

from dataclasses import dataclass, field

from robot.events.events import EmotionName


@dataclass(slots=True, frozen=True)
class GazeVector:
    """A 2D gaze target in normalized coordinates (-1..1)."""

    x: float = 0.0
    y: float = 0.0

    def clamped(self) -> GazeVector:
        return GazeVector(
            x=max(-1.0, min(1.0, self.x)),
            y=max(-1.0, min(1.0, self.y)),
        )


@dataclass(slots=True, frozen=True)
class EyeState:
    """A snapshot of every parameter the renderer needs."""

    emotion: EmotionName = EmotionName.NEUTRAL
    gaze: GazeVector = field(default_factory=GazeVector)
    openness: float = 1.0  # 0.0 = closed, 1.0 = wide open
    pupil_dilation: float = 0.5  # 0.0 = constricted, 1.0 = dilated
    intensity: float = 1.0  # 0.0 = subtle, 1.0 = full expression
    asymmetric: bool = False  # allow left/right divergence

    def with_emotion(self, emotion: EmotionName, intensity: float = 1.0) -> EyeState:
        return EyeState(
            emotion=emotion,
            gaze=self.gaze,
            openness=self.openness,
            pupil_dilation=self.pupil_dilation,
            intensity=intensity,
            asymmetric=self.asymmetric,
        )

    def with_gaze(self, gaze: GazeVector) -> EyeState:
        return EyeState(
            emotion=self.emotion,
            gaze=gaze.clamped(),
            openness=self.openness,
            pupil_dilation=self.pupil_dilation,
            intensity=self.intensity,
            asymmetric=self.asymmetric,
        )

    def with_openness(self, openness: float) -> EyeState:
        return EyeState(
            emotion=self.emotion,
            gaze=self.gaze,
            openness=max(0.0, min(1.0, openness)),
            pupil_dilation=self.pupil_dilation,
            intensity=self.intensity,
            asymmetric=self.asymmetric,
        )


__all__ = ["EyeState", "GazeVector"]
