"""Theme protocol.

A :class:`Theme` is a pure function ``FaceModel -> FaceModel`` that
applies a visual style. Themes can change colours (palette swap),
overlays (e.g. a "Retro LCD" theme adds scanlines), or even slightly
deform the face (e.g. "Pixel" quantises coordinates).

Themes are *post-processors*: they run after the EmotionEngine has
produced a FaceModel and before the renderer draws it. They never
change what the face is expressing - they only change how it looks.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from robot.face.components import FacePalette
from robot.face.model import FaceModel


@runtime_checkable
class Theme(Protocol):
    """A visual style applied to a :class:`FaceModel`."""

    @property
    def name(self) -> str:
        """Human-readable theme name."""

    @property
    def palette(self) -> FacePalette:
        """The colour palette the theme uses."""

    def apply(self, model: FaceModel) -> FaceModel:
        """Return ``model`` with the theme's visual style applied."""
