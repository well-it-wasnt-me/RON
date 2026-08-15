"""Software rasterizer for the eye engine.

The renderer is **display-agnostic**: it produces an
:class:`~robot.interfaces.display.EyeFrame` (an RGB888 buffer) that any
:class:`~robot.interfaces.display.Display` can push to a panel.

The drawing layers (in z-order, back to front) are:

1. **Sclera** - the white of the eye, a filled circle clipped to the
   eye radius. Its size shrinks vertically as ``openness`` drops, so
   the eye appears to close.
2. **Iris** - a coloured disc on top of the sclera, shifted by the
   gaze offset.
3. **Pupil** - a dark disc on top of the iris, scaled by
   ``pupil_dilation``.
4. **Highlight** - a small bright disc (the catch-light) drawn on
   top of the iris/pupil.
5. **Eyelids** - two filled rectangles (top and bottom) that cover
   the eye from above and below. They are drawn as straight bands so
   the eye shape remains crisp at small sizes.

The renderer is split into small, individually-testable helpers
(:meth:`_fill_circle`, :meth:`_draw_circle`, :meth:`_draw_horizontal_band`,
:meth:`_draw_vertical_band`, …) so unit tests can target each layer in
isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

from robot.eye_engine.render_state import EyeRenderState
from robot.interfaces.display import EyeFrame

_WHITE: Final[tuple[int, int, int]] = (255, 255, 255)
_BLACK: Final[tuple[int, int, int]] = (0, 0, 0)


# ---------------------------------------------------------------------------
# Buffer helpers
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class RendererConfig:
    """Tunable rendering parameters shared by every frame."""

    background_color: tuple[int, int, int] = (10, 10, 20)
    outline_color: tuple[int, int, int] = (10, 10, 10)
    outline_width: int = 1
    iris_radius_ratio: float = 0.55
    pupil_radius_ratio: float = 0.55
    gaze_offset_ratio: float = 0.18


@dataclass(slots=True)
class _Raster:
    """A mutable RGB888 image buffer."""

    width: int
    height: int
    pixels: bytearray = field(default_factory=bytearray)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be > 0")
        if not self.pixels:
            self.pixels = bytearray(self.width * self.height * 3)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
class EyeRenderer:
    """Render :class:`EyeRenderState` snapshots to RGB888 frames."""

    def __init__(
        self,
        width: int = 240,
        height: int = 240,
        config: RendererConfig | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be > 0")
        self._width = width
        self._height = height
        self._config = config or RendererConfig()

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def config(self) -> RendererConfig:
        return self._config

    # ------------------------------------------------------------------ public
    def render(self, state: EyeRenderState) -> EyeFrame:
        """Render ``state`` into a new RGB888 frame."""
        raster = self._new_raster()
        self._draw_background(raster, state)
        self._draw_sclera(raster, state)
        self._draw_iris(raster, state)
        self._draw_pupil(raster, state)
        self._draw_highlight(raster, state)
        self._draw_eyelids(raster, state)
        self._draw_eye_outline(raster, state)
        return EyeFrame(width=self._width, height=self._height, pixels=bytes(raster.pixels))

    # ------------------------------------------------------------------ layers
    def _new_raster(self) -> _Raster:
        return _Raster(width=self._width, height=self._height)

    def _draw_background(self, raster: _Raster, state: EyeRenderState) -> None:
        r, g, b = self._config.background_color
        pixel = bytes((r, g, b)) * (raster.width * raster.height)
        raster.pixels[:] = pixel

    def _sclera_radius(self, state: EyeRenderState) -> tuple[float, float]:
        """Return (rx, ry) of the sclera.

        ``openness`` shrinks the vertical radius, simulating the eyelids
        closing without the renderer needing to know about eyelid positions.
        The lower bound (0.05) keeps the iris visible for tiny open values.
        """
        rx = state.eye_radius
        ry = max(rx * 0.05, rx * state.openness)
        return rx, ry

    def _draw_sclera(self, raster: _Raster, state: EyeRenderState) -> None:
        rx, ry = self._sclera_radius(state)
        self._fill_ellipse(raster, state.cx, state.cy, rx, ry, state.sclera_color)

    def _iris_offset(self, state: EyeRenderState) -> tuple[float, float]:
        r = self._config.gaze_offset_ratio
        return state.gaze_x * self._width * r, state.gaze_y * self._height * r

    def _draw_iris(self, raster: _Raster, state: EyeRenderState) -> None:
        rx, ry = self._sclera_radius(state)
        # Clip the iris to the visible sclera
        if ry <= 1.0:
            return  # eye is closed - iris is not visible
        ox, oy = self._iris_offset(state)
        cx = state.cx + ox
        cy = state.cy + oy
        # Slightly oval iris looks more natural
        iris_rx = state.iris_radius
        iris_ry = state.iris_radius * min(1.0, ry / rx)
        self._fill_ellipse(raster, cx, cy, iris_rx, iris_ry, state.iris_color)
        # Iris outline
        if state.outline_width > 0:
            self._draw_ellipse_outline(
                raster, cx, cy, iris_rx, iris_ry, state.outline_color, state.outline_width
            )

    def _draw_pupil(self, raster: _Raster, state: EyeRenderState) -> None:
        _rx, ry = self._sclera_radius(state)
        if ry <= 1.0:
            return
        ox, oy = self._iris_offset(state)
        cx = state.cx + ox
        cy = state.cy + oy
        pupil_r = state.pupil_radius
        self._fill_circle(raster, cx, cy, pupil_r, state.pupil_color)

    def _draw_highlight(self, raster: _Raster, state: EyeRenderState) -> None:
        _rx, ry = self._sclera_radius(state)
        if ry <= 1.0:
            return
        ox, oy = self._iris_offset(state)
        cx = state.cx + ox
        cy = state.cy + oy
        # Highlight is positioned on the iris, slightly inside
        hx = cx + state.highlight_x * state.iris_radius * 0.5
        hy = cy + state.highlight_y * state.iris_radius * 0.5
        self._fill_circle(raster, hx, hy, max(1.5, state.iris_radius * 0.12), state.highlight_color)

    def _draw_eyelids(self, raster: _Raster, state: EyeRenderState) -> None:
        """Draw the top and bottom eyelids as horizontal bands.

        ``lid_top`` and ``lid_bottom`` are in ``[0, 1]`` and represent the
        fraction of the eye-radius that the lid covers. A value of ``1.0``
        fully covers the eye. Both lids are drawn as flat bands so the
        shape is clean at any size.
        """
        rx = state.eye_radius
        # Top lid: covers from the top of the eye down
        if state.lid_top > 0.0:
            self._draw_horizontal_band(
                raster,
                state.cy - rx,
                state.cy - rx + 2 * rx * state.lid_top,
                state.cx,
                rx,
                state.lid_color,
            )
        # Bottom lid: covers from the bottom of the eye up
        if state.lid_bottom > 0.0:
            self._draw_horizontal_band(
                raster,
                state.cy + rx - 2 * rx * state.lid_bottom,
                state.cy + rx,
                state.cx,
                rx,
                state.lid_color,
            )

    def _draw_eye_outline(self, raster: _Raster, state: EyeRenderState) -> None:
        if state.outline_width <= 0:
            return
        rx, ry = self._sclera_radius(state)
        self._draw_ellipse_outline(
            raster, state.cx, state.cy, rx, ry, state.outline_color, state.outline_width
        )

    # ------------------------------------------------------------------ primitives
    def _fill_circle(
        self, raster: _Raster, cx: float, cy: float, r: float, color: tuple[int, int, int]
    ) -> None:
        if r <= 0.0:
            return
        self._fill_ellipse(raster, cx, cy, r, r, color)

    def _fill_ellipse(
        self,
        raster: _Raster,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        color: tuple[int, int, int],
    ) -> None:
        if rx <= 0.0 or ry <= 0.0:
            return
        pr, pg, pb = color
        w, h = raster.width, raster.height
        x0 = max(0, math.floor(cx - rx))
        x1 = min(w - 1, math.ceil(cx + rx))
        y0 = max(0, math.floor(cy - ry))
        y1 = min(h - 1, math.ceil(ry + cy))
        if rx == ry:
            r2 = rx * rx
            for y in range(y0, y1 + 1):
                dy = y - cy
                row = (y * w) * 3
                for x in range(x0, x1 + 1):
                    dx = x - cx
                    if dx * dx + dy * dy <= r2:
                        idx = row + x * 3
                        raster.pixels[idx] = pr
                        raster.pixels[idx + 1] = pg
                        raster.pixels[idx + 2] = pb
            return
        inv_rx2 = 1.0 / (rx * rx)
        inv_ry2 = 1.0 / (ry * ry)
        for y in range(y0, y1 + 1):
            dy = y - cy
            row = (y * w) * 3
            for x in range(x0, x1 + 1):
                dx = x - cx
                v = dx * dx * inv_rx2 + dy * dy * inv_ry2
                if v <= 1.0:
                    idx = row + x * 3
                    raster.pixels[idx] = pr
                    raster.pixels[idx + 1] = pg
                    raster.pixels[idx + 2] = pb

    def _draw_ellipse_outline(
        self,
        raster: _Raster,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        color: tuple[int, int, int],
        width: int,
    ) -> None:
        if rx <= 0.0 or ry <= 0.0 or width <= 0:
            return
        for w in range(width):
            self._draw_ellipse(raster, cx, cy, max(0.0, rx - w), max(0.0, ry - w), color)

    def _draw_ellipse(
        self,
        raster: _Raster,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        color: tuple[int, int, int],
    ) -> None:
        """Midpoint ellipse algorithm - draws a 1-pixel-wide outline."""
        if rx <= 0.0 or ry <= 0.0:
            return
        pr, pg, pb = color
        w, h = raster.width, raster.height

        def put(px: int, py: int) -> None:
            if 0 <= px < w and 0 <= py < h:
                idx = (py * w + px) * 3
                raster.pixels[idx] = pr
                raster.pixels[idx + 1] = pg
                raster.pixels[idx + 2] = pb

        if rx == ry:
            # Reuse the circle routine for speed
            self._draw_circle(raster, cx, cy, rx, color)
            return
        rx2 = rx * rx
        ry2 = ry * ry
        x = 0.0
        y = ry
        px = 0.0
        py = 2 * rx2 * y
        # Region 1
        d1 = ry2 - rx2 * ry + 0.25 * rx2
        while px < py:
            put(int(cx + x), int(cy + y))
            put(int(cx - x), int(cy + y))
            put(int(cx + x), int(cy - y))
            put(int(cx - x), int(cy - y))
            x += 1
            px += 2 * ry2
            if d1 < 0:
                d1 += ry2 + px
            else:
                y -= 1
                py -= 2 * rx2
                d1 += ry2 + px - py
        # Region 2
        d2 = ry2 * (x + 0.5) * (x + 0.5) + rx2 * (y - 1) * (y - 1) - rx2 * ry2
        while y > 0:
            put(int(cx + x), int(cy + y))
            put(int(cx - x), int(cy + y))
            put(int(cx + x), int(cy - y))
            put(int(cx - x), int(cy - y))
            y -= 1
            py -= 2 * rx2
            if d2 > 0:
                d2 += rx2 - py
            else:
                x += 1
                px += 2 * ry2
                d2 += rx2 - py + px

    def _draw_circle(
        self, raster: _Raster, cx: float, cy: float, r: float, color: tuple[int, int, int]
    ) -> None:
        if r <= 0.0:
            return
        pr, pg, pb = color
        w, h = raster.width, raster.height

        def put(px: int, py: int) -> None:
            if 0 <= px < w and 0 <= py < h:
                idx = (py * w + px) * 3
                raster.pixels[idx] = pr
                raster.pixels[idx + 1] = pg
                raster.pixels[idx + 2] = pb

        x = round(r)
        y = 0
        err = 0
        icx, icy = round(cx), round(cy)
        while x >= y:
            put(icx + x, icy + y)
            put(icx + y, icy + x)
            put(icx - y, icy + x)
            put(icx - x, icy + y)
            put(icx - x, icy - y)
            put(icx - y, icy - x)
            put(icx + y, icy - x)
            put(icx + x, icy - y)
            y += 1
            err += 1 + 2 * y
            if 2 * (err - x) + 1 > 0:
                x -= 1
                err += 1 - 2 * x

    def _draw_horizontal_band(
        self,
        raster: _Raster,
        y_top: float,
        y_bottom: float,
        cx: float,
        rx: float,
        color: tuple[int, int, int],
    ) -> None:
        """Fill the band ``[y_top, y_bottom]`` inside the eye-sclera circle.

        The band is clipped to the eye's circular shape so it looks like
        an eyelid rather than a rectangle.
        """
        pr, pg, pb = color
        w, h = raster.width, raster.height
        x0 = max(0, math.floor(cx - rx))
        x1 = min(w - 1, math.ceil(cx + rx))
        y0 = max(0, math.ceil(y_top))
        y1 = min(h - 1, math.floor(y_bottom))
        rx2 = rx * rx
        for y in range(y0, y1 + 1):
            band_cy = (y_top + y_bottom) / 2.0
            dy = y - band_cy
            half_height = max(1.0, (y_bottom - y_top) / 2.0)
            # ellipse half-width at this y
            if abs(dy) > rx + half_height:
                continue
            row = (y * w) * 3
            for x in range(x0, x1 + 1):
                dx = x - cx
                # point must be inside the eye-sclera circle AND inside the band
                if dx * dx + dy * dy > rx2:
                    continue
                idx = row + x * 3
                raster.pixels[idx] = pr
                raster.pixels[idx + 1] = pg
                raster.pixels[idx + 2] = pb


__all__ = ["EyeRenderer", "RendererConfig"]
