"""Speaking animation with phoneme timing.

Produces mouth open/close keyframes driven by a simple viseme model.
Visemes are mouth shapes corresponding to phonemes (speech sounds).
This animation maps text or phoneme sequences to mouth openness
values, creating the illusion of lip-synced speech.

The viseme mapping is intentionally simple - a production system would
use a proper TTS engine's phoneme timing output (e.g. Piper TTS's
JSON timing data, or eSpeak-NG's phoneme output). This module provides
the animation framework and a reasonable English-language default.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Viseme definitions
# ---------------------------------------------------------------------------
class Viseme(str, Enum):
    """Simplified viseme set for English speech.

    Maps approximately to the Prestonon viseme set (13 visemes) but
    grouped for a small round display. Each viseme has a target mouth
    openness (0.0 = closed, 1.0 = wide open) and width.
    """

    IDLE = "idle"  # mouth closed, neutral
    PP = "pp"  # p, b, m - lips together
    FF = "ff"  # f, v - lower lip to upper teeth
    TH = "th"  # th - tongue between teeth
    DD = "dd"  # t, d, n - tongue to alveolar ridge
    KK = "kk"  # k, g - back of tongue to soft palate
    CH = "ch"  # ch, j, sh - lips rounded
    SS = "ss"  # s, z - teeth together
    NN = "nn"  # n, ng - mouth slightly open
    RR = "rr"  # r - lips rounded
    AA = "aa"  # a - mouth open wide
    EE = "ee"  # e - mouth stretched wide
    OO = "oo"  # o, u - mouth rounded


# Openness and width targets for each viseme.
_VISEME_TARGETS: dict[Viseme, tuple[float, float]] = {
    Viseme.IDLE: (0.0, 0.5),
    Viseme.PP: (0.0, 0.4),
    Viseme.FF: (0.15, 0.45),
    Viseme.TH: (0.2, 0.5),
    Viseme.DD: (0.25, 0.5),
    Viseme.KK: (0.3, 0.5),
    Viseme.CH: (0.35, 0.45),
    Viseme.SS: (0.15, 0.5),
    Viseme.NN: (0.2, 0.5),
    Viseme.RR: (0.25, 0.4),
    Viseme.AA: (0.7, 0.7),
    Viseme.EE: (0.3, 0.7),
    Viseme.OO: (0.4, 0.35),
}

# Simple phoneme-to-viseme mapping for English text.
# Characters map to visemes; unmapped characters use IDLE.
_CHAR_TO_VISEME: dict[str, Viseme] = {
    # Plosives
    "p": Viseme.PP,
    "b": Viseme.PP,
    "m": Viseme.PP,
    # Fricatives
    "f": Viseme.FF,
    "v": Viseme.FF,
    "t": Viseme.DD,
    "d": Viseme.DD,
    "n": Viseme.NN,
    "s": Viseme.SS,
    "z": Viseme.SS,
    "sh": Viseme.CH,
    "ch": Viseme.CH,
    "j": Viseme.CH,
    "th": Viseme.TH,
    # Velars
    "k": Viseme.KK,
    "g": Viseme.KK,
    "ng": Viseme.NN,
    # Liquids/glides
    "r": Viseme.RR,
    "l": Viseme.DD,
    "w": Viseme.OO,
    "y": Viseme.EE,
    # Vowels
    "a": Viseme.AA,
    "e": Viseme.EE,
    "i": Viseme.EE,
    "o": Viseme.OO,
    "u": Viseme.OO,
}


@dataclass(slots=True, frozen=True)
class VisemeFrame:
    """A single frame of the speaking animation.

    Attributes
    ----------
    openness:
        How open the mouth is (0.0 = closed, 1.0 = wide open).
    width:
        How wide the mouth is (0.0 = narrow, 1.0 = wide).
    duration_s:
        How long this viseme should be held, in seconds.
    viseme:
        The viseme this frame represents.
    """

    openness: float
    width: float
    duration_s: float
    viseme: Viseme = Viseme.IDLE


@dataclass(slots=True)
class SpeakingAnimation:
    """Produces a sequence of mouth shapes for lip-synced speech.

    The animation converts text into a sequence of :class:`VisemeFrame`
    objects, one per phoneme, with timing that approximates natural
    speech cadence.

    Usage::

        anim = SpeakingAnimation(text="Hello there!")
        while anim.has_frames:
            frame = anim.step(dt=0.033)
            face_model = face_model.with_mouth(
                Mouth(shape=MouthShape.NEUTRAL, openness=frame.openness, width=frame.width)
            )
    """

    text: str = ""
    # Default duration per phoneme in seconds. Real speech varies but
    # this gives a reasonable approximation for visual animation.
    default_phoneme_duration: float = 0.08
    # Duration for pauses (spaces, commas, periods).
    pause_duration: float = 0.15
    sentence_pause_duration: float = 0.25
    # Speed multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double).
    speed: float = 1.0

    _frames: list[VisemeFrame] = field(default_factory=list, init=False)
    _index: int = field(default=0, init=False)
    _elapsed: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.text:
            self._frames = self._text_to_frames(self.text)

    @classmethod
    def from_visemes(
        cls,
        visemes: Sequence[tuple[Viseme, float]],
        speed: float = 1.0,
    ) -> SpeakingAnimation:
        """Create an animation from an explicit viseme sequence.

        Each tuple is ``(viseme, duration_s)``.
        """
        anim = cls(speed=speed)
        for viseme, duration in visemes:
            openness, width = _VISEME_TARGETS.get(viseme, (0.3, 0.5))
            anim._frames.append(
                VisemeFrame(
                    openness=openness,
                    width=width,
                    duration_s=duration / speed,
                    viseme=viseme,
                )
            )
        return anim

    def step(self, dt: float) -> VisemeFrame:
        """Advance the animation by ``dt`` seconds.

        Returns the current :class:`VisemeFrame`. If the animation has
        finished, returns :class:`Viseme.IDLE` frame.
        """
        if not self._frames or self._index >= len(self._frames):
            return VisemeFrame(openness=0.0, width=0.5, duration_s=0.0)

        self._elapsed += dt

        # Advance to the next frame when the current frame's duration
        # expires.  Unlike a simple ``elapsed + duration <= dt`` check,
        # we accumulate time across calls so that frames with durations
        # longer than a single ``dt`` are held for their full duration.
        while (
            self._index < len(self._frames)
            and self._elapsed >= self._frames[self._index].duration_s
        ):
            self._elapsed -= self._frames[self._index].duration_s
            self._index += 1

        if self._index >= len(self._frames):
            return VisemeFrame(openness=0.0, width=0.5, duration_s=0.0)

        return self._frames[self._index]

    @property
    def has_frames(self) -> bool:
        """Whether there are still frames to animate."""
        return self._index < len(self._frames)

    @property
    def total_duration(self) -> float:
        """Total animation duration in seconds."""
        return sum(f.duration_s for f in self._frames)

    def reset(self) -> None:
        """Reset the animation to the beginning."""
        self._index = 0
        self._elapsed = 0.0

    # ------------------------------------------------------------------ text->viseme
    def _text_to_frames(self, text: str) -> list[VisemeFrame]:
        """Convert text into a list of viseme frames."""
        frames: list[VisemeFrame] = []
        i = 0
        while i < len(text):
            ch = text[i].lower()

            # Handle punctuation as pauses.
            if ch in ".!?":
                frames.append(
                    VisemeFrame(
                        openness=0.0,
                        width=0.5,
                        duration_s=self.sentence_pause_duration / self.speed,
                    )
                )
                i += 1
                continue
            if ch in ",;:":
                frames.append(
                    VisemeFrame(
                        openness=0.0, width=0.5, duration_s=self.pause_duration / self.speed
                    )
                )
                i += 1
                continue
            if ch in " \n\t":
                frames.append(VisemeFrame(openness=0.0, width=0.5, duration_s=0.05 / self.speed))
                i += 1
                continue

            # Try two-character digraphs first.
            if i + 1 < len(text):
                digraph = text[i : i + 2].lower()
                if digraph in _CHAR_TO_VISEME:
                    viseme = _CHAR_TO_VISEME[digraph]
                    openness, width = _VISEME_TARGETS[viseme]
                    frames.append(
                        VisemeFrame(
                            openness=openness,
                            width=width,
                            duration_s=self.default_phoneme_duration / self.speed,
                            viseme=viseme,
                        )
                    )
                    i += 2
                    continue

            # Single character.
            viseme = _CHAR_TO_VISEME.get(ch, Viseme.IDLE)
            openness, width = _VISEME_TARGETS[viseme]
            frames.append(
                VisemeFrame(
                    openness=openness,
                    width=width,
                    duration_s=self.default_phoneme_duration / self.speed,
                    viseme=viseme,
                )
            )
            i += 1

        return frames


__all__ = ["SpeakingAnimation", "Viseme", "VisemeFrame"]
