"""Pure data describing the current visual state of one eye.

This is the **render-time** data: everything the renderer needs to know
to draw one frame of one eye. It is intentionally separate from
:class:`robot.eye_engine.eye_state.EyeState` (which is the semantic
"mood") so the renderer never has to interpret emotions - it just
rasterises the numeric inputs it is given.

All coordinates are in the renderer's local framebuffer:

* ``cx, cy`` - centre of the eye on the panel (defaults to the centre of
  the framebuffer, but the gaze offset shifts the iris / pupil around
  this point).
* Gaze offsets are normalised in ``[-1, 1]`` and map to a fraction of
  the framebuffer width/height (see :attr:`RendererConfig.gaze_offset_ratio`).
* ``openness`` is the eye-aperture height as a fraction of the maximum
  eye radius (0.0 = closed, 1.0 = wide open).
* ``pupil_dilation`` is normalised in ``[0, 1]`` (0 = constricted,
  1 = dilated).
* ``lid_top`` / ``lid_bottom`` are vertical fractions in ``[0, 1]`` of
  the eye-sclera radius (0.0 = fully retracted, 1.0 = fully covering
  the eye). They let the renderer draw individual eyelids instead of
  relying on a single ``openness`` value.
* ``highlight`` is the position of the catch-light on the iris,
  normalised in ``[-1, 1]`` of the eye radius.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class EyeRenderState:
    """Everything the renderer needs to draw one frame of one eye."""

    # Centre of the eye in the framebuffer
    cx: float
    cy: float
    # Maximum sclera radius (px)
    eye_radius: float
    # Iris radius (px) as a fraction of ``eye_radius``
    iris_radius_ratio: float = 0.55
    # Pupil radius (px) as a fraction of ``iris_radius``
    pupil_radius_ratio: float = 0.55
    # Eye aperture: 0.0 = closed, 1.0 = fully open (legacy single-value)
    openness: float = 1.0
    # Individual eyelid positions: 0.0 = retracted, 1.0 = fully covering
    lid_top: float = 0.0
    lid_bottom: float = 0.0
    # Gaze: normalised offset in [-1, 1] applied to iris/pupil position
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    # Pupil dilation 0..1
    pupil_dilation: float = 0.5
    # Catch-light position, normalised in [-1, 1] of eye radius
    highlight_x: float = 0.3
    highlight_y: float = 0.3
    # Per-element colours (override the renderer defaults)
    sclera_color: tuple[int, int, int] = (245, 245, 235)
    iris_color: tuple[int, int, int] = (40, 110, 200)
    pupil_color: tuple[int, int, int] = (10, 10, 20)
    lid_color: tuple[int, int, int] = (20, 20, 30)
    highlight_color: tuple[int, int, int] = (255, 255, 255)
    outline_color: tuple[int, int, int] = (10, 10, 10)
    outline_width: int = 1

    @property
    def iris_radius(self) -> float:
        return self.eye_radius * self.iris_radius_ratio

    @property
    def pupil_radius(self) -> float:
        base = self.iris_radius * self.pupil_radius_ratio
        # 0.5..1.0 of base depending on dilation
        return base * (0.6 + 0.6 * self.pupil_dilation)


__all__ = ["EyeRenderState"]
