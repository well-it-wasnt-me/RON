"""The :class:`FaceRenderer` - display-agnostic pixel art.

The renderer takes a :class:`FaceModel` and produces an
:class:`EyeFrame` (an RGB888 buffer). It knows nothing about displays,
themes, or animation - it just rasterises the model it is given.

The drawing layers (z-order, back to front) are:

1. **Background** - solid colour from the palette.
2. **Bounce / squash transform** - translate / scale the entire face
   based on the model's ``bounce`` and ``squash`` fields.
3. **Cheeks** - soft circular blush patches.
4. **Sclera (eyes)** - filled circles clipped to the panel.
5. **Eyelids** - horizontal bands that cover the eyes.
6. **Iris** - coloured discs shifted by gaze offset.
7. **Pupil** - dark discs on top of the iris.
8. **Eye highlight** - small catch-light.
9. **Eyebrows** - short bars above each eye.
10. **Mouth** - the mouth shape as a small filled curve.
11. **Overlay** - sparkles, hearts, tears, sweat, etc.
12. **Accessory** - glasses, hat, etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from robot.face.components import (
    PALETTE_MODE_VECTOR,
    AccessoryKind,
    CheekState,
    EyebrowShape,
    MouthShape,
    OverlayKind,
)
from robot.face.model import FaceModel
from robot.interfaces.display import EyeFrame


# ---------------------------------------------------------------------------
# Internal buffer
# ---------------------------------------------------------------------------
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
class FaceRenderer:
    """Render a :class:`FaceModel` to an RGB888 frame."""

    def __init__(self, width: int = 240, height: int = 240) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be > 0")
        self._width = width
        self._height = height

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    # ------------------------------------------------------------------ public
    def render(self, model: FaceModel) -> EyeFrame:
        """Render ``model`` into a new RGB888 frame."""
        if model.width != self._width or model.height != self._height:
            raise ValueError(
                f"model is {model.width}x{model.height}, renderer is {self._width}x{self._height}"
            )
        raster = self._new_raster()
        # Vector mode bypasses the full face stack and renders only the
        # minimalist "two glowing squares + a horizontal line" used by the
        # Anki Vector 2.0 face. Cheeks/eyebrows/iris/sclera are skipped
        # because the Vector palette does not use them.
        if model.palette.mode == PALETTE_MODE_VECTOR:
            self._draw_background(raster, model)
            self._draw_vector_eyes(raster, model)
            self._draw_vector_mouth(raster, model)
            return EyeFrame(width=self._width, height=self._height, pixels=bytes(raster.pixels))
        self._draw_background(raster, model)
        self._draw_cheeks(raster, model)
        self._draw_sclera(raster, model)
        self._draw_eyelids(raster, model)
        self._draw_iris(raster, model)
        self._draw_pupil(raster, model)
        self._draw_highlight(raster, model)
        self._draw_eyebrows(raster, model)
        self._draw_mouth(raster, model)
        self._draw_overlay(raster, model)
        self._draw_accessory(raster, model)
        return EyeFrame(width=self._width, height=self._height, pixels=bytes(raster.pixels))

    # ------------------------------------------------------------------ vector mode
    def _draw_vector_eyes(self, raster: _Raster, model: FaceModel) -> None:
        """Draw two glowing rounded-squares for the eyes (Vector style).

        Each eye follows the model's gaze position so drift and look
        commands produce visible movement.  The mouth still animates
        independently.
        """
        cx, cy, face_radius, _, _ = self._face_geometry(model)
        # Vector eyes sit a bit above the panel centre, like the Anki robot.
        eye_radius = face_radius * 0.18
        eye_y = cy - face_radius * 0.30
        eye_offset_x = face_radius * 0.45
        # Gaze offset - both eyes share the same gaze direction.
        # A multiplier of 0.35 produces a visible (~3-5 px) shift on a
        # 240 px face so that drift and look commands are perceptible.
        gaze_shift_x = face_radius * 0.35 * model.left_eye.gaze.x
        gaze_shift_y = face_radius * 0.35 * model.left_eye.gaze.y
        for sx in (-1, 1):
            ex = cx + sx * eye_offset_x + gaze_shift_x
            ey = eye_y + gaze_shift_y
            self._fill_rounded_square(
                raster,
                ex - eye_radius,
                ey - eye_radius,
                eye_radius * 2,
                eye_radius * 2,
                eye_radius * 0.25,
                model.palette.iris,
            )

    def _draw_vector_mouth(self, raster: _Raster, model: FaceModel) -> None:
        """Draw a thin horizontal bar mouth that morphs with the mouth shape.

        Shape -> visual:
          NEUTRAL / CLOSED  -> straight horizontal bar
          SMILE / GRIN      -> bar curves up at the ends
          FROWN             -> bar curves down at the ends
          OPEN / WIDE_OPEN  -> bar with a gap in the middle (small "o")
          TONGUE            -> bar with a small dot below
        """
        cx, cy, face_radius, _, _ = self._face_geometry(model)
        bar_y = cy + face_radius * 0.45
        bar_half_w = face_radius * 0.32
        bar_thickness = max(2, int(face_radius * 0.08))
        shape = model.mouth.shape
        openness = model.mouth.openness
        color = model.palette.mouth
        if shape in (MouthShape.SMILE, MouthShape.GRIN, MouthShape.SMILE_OPEN, MouthShape.TONGUE):
            # Curve the ends upward (smile).
            curve = -face_radius * 0.18 * (1.0 + openness)
            self._draw_arc(raster, cx, bar_y, bar_half_w, curve, color, bar_thickness)
            if shape is MouthShape.TONGUE:
                # A tiny tongue dot below the mouth line.
                self._fill_circle(
                    raster,
                    cx,
                    bar_y - face_radius * 0.06,
                    face_radius * 0.06,
                    (255, 130, 180),
                )
        elif shape is MouthShape.FROWN:
            # Curve the ends downward.
            curve = face_radius * 0.18
            self._draw_arc(raster, cx, bar_y, bar_half_w, curve, color, bar_thickness)
        elif shape in (MouthShape.OPEN, MouthShape.WIDE_OPEN):
            # Two short bars with a gap = a small "o" shape.
            r = face_radius * (0.04 + 0.06 * openness) * (1.0 if shape is MouthShape.OPEN else 1.6)
            self._fill_circle(raster, cx, bar_y, r, color)
            if shape is MouthShape.WIDE_OPEN:
                self._fill_circle(raster, cx, bar_y, r * 0.6, model.palette.background)
        else:  # NEUTRAL / CLOSED
            self._draw_thick_line(
                raster,
                cx - bar_half_w,
                bar_y,
                cx + bar_half_w,
                bar_y,
                bar_thickness,
                color,
            )

    def _fill_rounded_square(
        self,
        raster: _Raster,
        x0: float,
        y0: float,
        w: float,
        h: float,
        radius: float,
        color: tuple[int, int, int],
    ) -> None:
        """Fill a rounded-square between ``(x0, y0)`` and ``(x0+w, y0+h)``.

        Used for the Vector-style eye. The corners are rounded by
        ``radius`` pixels (clamped to half the smaller side).
        """
        r = max(0.0, min(radius, min(w, h) / 2.0))
        x1 = x0 + w
        y1 = y0 + h
        col = bytes(color) * raster.width
        # Top + bottom strips + middle strips + corner arcs.
        for y in range(round(y0), round(y1)):
            if y < y0 + r or y >= y1 - r:
                # Top/bottom strip between the corners
                self._fill_rect_row(raster, round(x0 + r), y, round(x1 - r) - 1, col)
            else:
                # Full-width row in the middle
                self._fill_rect_row(raster, round(x0), y, round(x1) - 1, col)
        # Quadrant corner fills (4 circles for the rounded corners)
        cr = max(1, round(r))
        for cx_corner, cy_corner in (
            (x0 + r, y0 + r),
            (x1 - r, y0 + r),
            (x0 + r, y1 - r),
            (x1 - r, y1 - r),
        ):
            # Use the existing _fill_circle for each corner quadrant.
            # We restrict to the inside of the square so the result is a
            # smooth rounded corner rather than a full disc.
            self._fill_corner_quadrant(
                raster,
                round(cx_corner),
                round(cy_corner),
                cr,
                color,
                is_top_left=(cx_corner <= x0 + r and cy_corner <= y0 + r),
                is_top_right=(cx_corner >= x1 - r and cy_corner <= y0 + r),
                is_bottom_left=(cx_corner <= x0 + r and cy_corner >= y1 - r),
                is_bottom_right=(cx_corner >= x1 - r and cy_corner >= y1 - r),
            )

    def _fill_rect_row(self, raster: _Raster, x0: int, y: int, x1: int, color_bytes: bytes) -> None:
        """Fill a horizontal strip from ``(x0, y)`` to ``(x1, y)`` inclusive."""
        if x1 < x0 or y < 0 or y >= raster.height:
            return
        x0 = max(0, x0)
        x1 = min(raster.width - 1, x1)
        if x1 < x0:
            return
        start = (y * raster.width + x0) * 3
        end = (y * raster.width + x1 + 1) * 3
        # Write the 3-byte color (r, g, b) for every pixel in the row.
        i = start
        while i + 3 <= end:
            raster.pixels[i : i + 3] = bytes(color_bytes[:3])
            i += 3

    def _fill_corner_quadrant(
        self,
        raster: _Raster,
        cx: int,
        cy: int,
        r: int,
        color: tuple[int, int, int],
        *,
        is_top_left: bool,
        is_top_right: bool,
        is_bottom_left: bool,
        is_bottom_right: bool,
    ) -> None:
        """Fill the inside-quadrant of a circle of radius ``r`` at (cx, cy).

        Used to round the corners of :meth:`_fill_rounded_square`.
        """
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue
                px = cx + dx
                py = cy + dy
                # Only write inside the square.
                inside = True
                if is_top_left and (dx > 0 or dy > 0):
                    inside = False
                if is_top_right and (dx < 0 or dy > 0):
                    inside = False
                if is_bottom_left and (dx > 0 or dy < 0):
                    inside = False
                if is_bottom_right and (dx < 0 or dy < 0):
                    inside = False
                if not inside:
                    continue
                if 0 <= px < raster.width and 0 <= py < raster.height:
                    i = (py * raster.width + px) * 3
                    raster.pixels[i : i + 3] = bytes(color)

    # ------------------------------------------------------------------ helpers
    def _new_raster(self) -> _Raster:
        return _Raster(width=self._width, height=self._height)

    # The model is drawn around a logical centre (cx, cy). The bounce and
    # squash fields are applied here so every layer sees the transformed
    # coordinates.
    def _face_geometry(self, model: FaceModel) -> tuple[float, float, float, float, float]:
        """Return (cx, cy, face_radius, eye_radius, mouth_y) in pixels."""
        cx = self._width / 2.0
        cy = self._height / 2.0
        # Vertical bounce: -1..1 -> -face_radius..+face_radius
        face_radius = min(self._width, self._height) * 0.45
        cy = cy + model.bounce * face_radius * 0.10
        # Horizontal squash: 1.0 = normal, 0.5 = squashed, 1.5 = stretched
        # We model squash as a uniform scale of the face radius (used by
        # sub-elements that respect a "horizontal scale" hint).
        face_radius *= 1.0
        eye_radius = face_radius * 0.30
        mouth_y = cy + face_radius * 0.55
        return cx, cy, face_radius, eye_radius, mouth_y

    def _sclera_radius(self, openness: float, eye_radius: float) -> tuple[float, float]:
        rx = eye_radius
        ry = max(eye_radius * 0.05, eye_radius * openness)
        return rx, ry

    # ------------------------------------------------------------------ layers
    def _draw_background(self, raster: _Raster, model: FaceModel) -> None:
        r, g, b = model.palette.background
        pixel = bytes((r, g, b)) * (raster.width * raster.height)
        raster.pixels[:] = pixel

    def _draw_cheeks(self, raster: _Raster, model: FaceModel) -> None:
        if model.cheeks.state is CheekState.NONE or model.cheeks.intensity <= 0.0:
            return
        cx, cy, face_radius, _, _ = self._face_geometry(model)
        # Cheek positions: left/right of mouth, below eyes
        for sx in (-1, 1):
            x = cx + sx * face_radius * 0.55
            y = cy + face_radius * 0.15
            color = (
                model.palette.blush
                if model.cheeks.state is CheekState.BRIGHT
                else model.palette.cheek
            )
            self._fill_circle_blend(raster, x, y, face_radius * 0.22, color, model.cheeks.intensity)

    def _draw_sclera(self, raster: _Raster, model: FaceModel) -> None:
        cx, cy, face_radius, eye_radius, _ = self._face_geometry(model)
        # Two eyes side by side
        for sx, eye in ((-1, model.left_eye), (1, model.right_eye)):
            ex = cx + sx * face_radius * 0.40
            ey = cy - face_radius * 0.18
            rx, ry = self._sclera_radius(eye.openness, eye_radius)
            self._fill_ellipse(raster, ex, ey, rx, ry, model.palette.sclera)
            if model.palette.outline != model.palette.sclera:
                self._draw_ellipse_outline(
                    raster,
                    ex,
                    ey,
                    rx,
                    ry,
                    model.palette.outline,
                    1,
                )

    def _draw_eyelids(self, raster: _Raster, model: FaceModel) -> None:
        if model.eyelids.top <= 0.0 and model.eyelids.bottom <= 0.0:
            return
        cx, cy, face_radius, eye_radius, _ = self._face_geometry(model)
        for sx in (-1, 1):
            ex = cx + sx * face_radius * 0.40
            ey = cy - face_radius * 0.18
            rx, ry = self._sclera_radius(1.0, eye_radius)
            if model.eyelids.top > 0.0:
                self._draw_horizontal_band(
                    raster,
                    ey - ry,
                    ey - ry + 2 * ry * model.eyelids.top,
                    ex,
                    rx,
                    model.palette.eyelid,
                )
            if model.eyelids.bottom > 0.0:
                self._draw_horizontal_band(
                    raster,
                    ey + ry - 2 * ry * model.eyelids.bottom,
                    ey + ry,
                    ex,
                    rx,
                    model.palette.eyelid,
                )

    def _draw_iris(self, raster: _Raster, model: FaceModel) -> None:
        cx, cy, face_radius, eye_radius, _ = self._face_geometry(model)
        for sx, eye in ((-1, model.left_eye), (1, model.right_eye)):
            ex = cx + sx * face_radius * 0.40
            ey = cy - face_radius * 0.18
            rx, ry = self._sclera_radius(eye.openness, eye_radius)
            if ry <= 1.0:
                continue
            gaze = eye.gaze.clamped()
            ox = gaze.x * self._width * 0.06
            oy = gaze.y * self._height * 0.06
            iris_cx = ex + ox
            iris_cy = ey + oy
            iris_rx = eye_radius * 0.55
            iris_ry = eye_radius * 0.55 * min(1.0, ry / rx) if rx > 0.0 else 0.0
            self._fill_ellipse(raster, iris_cx, iris_cy, iris_rx, iris_ry, model.palette.iris)
            self._draw_ellipse_outline(
                raster,
                iris_cx,
                iris_cy,
                iris_rx,
                iris_ry,
                model.palette.outline,
                1,
            )

    def _draw_pupil(self, raster: _Raster, model: FaceModel) -> None:
        cx, cy, face_radius, eye_radius, _ = self._face_geometry(model)
        for sx, eye in ((-1, model.left_eye), (1, model.right_eye)):
            ex = cx + sx * face_radius * 0.40
            ey = cy - face_radius * 0.18
            _rx, ry = self._sclera_radius(eye.openness, eye_radius)
            if ry <= 1.0:
                continue
            gaze = eye.gaze.clamped()
            ox = gaze.x * self._width * 0.06
            oy = gaze.y * self._height * 0.06
            iris_cx = ex + ox
            iris_cy = ey + oy
            base = eye_radius * 0.55 * 0.55
            pupil_r = base * (0.6 + 0.6 * eye.pupil_dilation)
            self._fill_circle(raster, iris_cx, iris_cy, pupil_r, model.palette.pupil)

    def _draw_highlight(self, raster: _Raster, model: FaceModel) -> None:
        cx, cy, face_radius, eye_radius, _ = self._face_geometry(model)
        for sx, eye in ((-1, model.left_eye), (1, model.right_eye)):
            ex = cx + sx * face_radius * 0.40
            ey = cy - face_radius * 0.18
            _rx, ry = self._sclera_radius(eye.openness, eye_radius)
            if ry <= 1.0:
                continue
            gaze = eye.gaze.clamped()
            ox = gaze.x * self._width * 0.06
            oy = gaze.y * self._height * 0.06
            iris_cx = ex + ox
            iris_cy = ey + oy
            hx = iris_cx + eye.highlight.x * eye_radius * 0.30
            hy = iris_cy + eye.highlight.y * eye_radius * 0.30
            self._fill_circle(raster, hx, hy, max(1.0, eye_radius * 0.08), model.palette.highlight)

    def _draw_eyebrows(self, raster: _Raster, model: FaceModel) -> None:
        cx, cy, face_radius, eye_radius, _ = self._face_geometry(model)
        for sx, brow in ((-1, model.left_eyebrow), (1, model.right_eyebrow)):
            ex = cx + sx * face_radius * 0.40
            ey = cy - face_radius * 0.18
            base_y = ey - eye_radius * (1.05 + brow.raise_amount)
            # Shape -> vertical offset
            if brow.shape is EyebrowShape.ANGRY:
                outer_y = base_y - 0.05 * face_radius
                inner_y = base_y + 0.10 * face_radius
            elif brow.shape is EyebrowShape.WORRIED:
                outer_y = base_y + 0.05 * face_radius
                inner_y = base_y - 0.10 * face_radius
            elif brow.shape is EyebrowShape.SAD:
                outer_y = base_y + 0.15 * face_radius
                inner_y = base_y - 0.05 * face_radius
            elif brow.shape is EyebrowShape.SLEEPY:
                outer_y = base_y - 0.10 * face_radius
                inner_y = base_y - 0.10 * face_radius
            else:  # NEUTRAL or RAISED
                outer_y = base_y
                inner_y = base_y

            # Add the angle tilt
            tilt = brow.angle * eye_radius * 0.4
            outer_y -= tilt
            inner_y += tilt

            thickness = max(1, int(eye_radius * 0.18))
            length = eye_radius * 0.95
            self._draw_thick_line(
                raster,
                ex - length / 2,
                outer_y,
                ex + length / 2,
                inner_y,
                thickness,
                model.palette.eyebrow,
            )

    def _draw_mouth(self, raster: _Raster, model: FaceModel) -> None:
        cx, _cy, face_radius, _, mouth_y = self._face_geometry(model)
        width = face_radius * 0.45
        shape = model.mouth.shape
        openness = model.mouth.openness
        width_scale = model.mouth.width
        actual_width = width * width_scale
        # Base line at mouth_y
        if shape is MouthShape.SMILE or shape is MouthShape.GRIN:
            curve = -face_radius * 0.10 * (1.0 + openness)
            self._draw_arc(raster, cx, mouth_y, actual_width, curve, model.palette.mouth, 3)
        elif shape is MouthShape.SMILE_OPEN:
            self._draw_arc(
                raster, cx, mouth_y, actual_width, -face_radius * 0.12, model.palette.mouth, 3
            )
            self._fill_circle(
                raster,
                cx,
                mouth_y - face_radius * 0.04,
                face_radius * 0.08 * openness,
                model.palette.background,
            )
        elif shape is MouthShape.FROWN:
            self._draw_arc(
                raster, cx, mouth_y, actual_width, face_radius * 0.10, model.palette.mouth, 3
            )
        elif shape is MouthShape.OPEN or shape is MouthShape.WIDE_OPEN:
            r = face_radius * (0.05 + 0.10 * openness) * (1.0 if shape is MouthShape.OPEN else 1.6)
            self._fill_circle(raster, cx, mouth_y, r, model.palette.mouth)
            if shape is MouthShape.WIDE_OPEN:
                # Inner dark
                self._fill_circle(raster, cx, mouth_y, r * 0.6, model.palette.background)
        elif shape is MouthShape.TONGUE:
            self._draw_arc(
                raster, cx, mouth_y, actual_width, -face_radius * 0.05, model.palette.mouth, 3
            )
            # Pink tongue
            self._fill_circle(
                raster, cx, mouth_y - face_radius * 0.01, face_radius * 0.06, (255, 130, 180)
            )
        else:  # CLOSED or NEUTRAL
            self._draw_arc(raster, cx, mouth_y, actual_width, 0.0, model.palette.mouth, 2)
        if model.mouth.asymmetry != 0.0:
            skew = model.mouth.asymmetry * actual_width * 0.3
            self._draw_arc(raster, cx + skew, mouth_y, actual_width, 0.0, model.palette.mouth, 2)

    def _draw_overlay(self, raster: _Raster, model: FaceModel) -> None:
        ov = model.overlay
        if ov.kind is OverlayKind.NONE:
            return
        cx, cy, face_radius, _, _ = self._face_geometry(model)
        # Position is in normalised face coordinates (-1..1) -> pixels
        x = cx + ov.position.x * face_radius
        y = cy + ov.position.y * face_radius
        size = max(2, int(face_radius * ov.size))
        if ov.kind is OverlayKind.SPARKLE or ov.kind is OverlayKind.STAR:
            # 4-pointed star: a small filled diamond
            self._fill_diamond(raster, x, y, size, ov.color)
        elif ov.kind is OverlayKind.HEART:
            self._fill_heart(raster, x, y, size, ov.color)
        elif ov.kind is OverlayKind.TEAR:
            self._fill_teardrop(raster, x, y, size, (80, 150, 255))
        elif ov.kind is OverlayKind.SWEAT:
            self._fill_teardrop(raster, x, y, size, (130, 200, 255))
        elif ov.kind is OverlayKind.ANGER:
            self._fill_diamond(raster, x, y, size, (255, 50, 50))
        elif ov.kind is OverlayKind.QUESTION:
            self._fill_circle(raster, x, y, size * 0.6, (255, 220, 60))
        elif ov.kind is OverlayKind.EXCLAIM:
            self._fill_circle(raster, x, y, size * 0.6, (255, 80, 80))
        elif ov.kind is OverlayKind.MUSIC:
            self._fill_circle(raster, x, y, size * 0.5, (60, 200, 255))

    def _draw_accessory(self, raster: _Raster, model: FaceModel) -> None:
        acc = model.accessory
        if acc.kind is AccessoryKind.NONE:
            return
        cx, cy, face_radius, eye_radius, _ = self._face_geometry(model)
        if acc.kind is AccessoryKind.GLASSES:
            # Two round rims
            for sx in (-1, 1):
                ex = cx + sx * face_radius * 0.40
                ey = cy - face_radius * 0.18
                self._draw_ellipse_outline(
                    raster, ex, ey, eye_radius * 0.85, eye_radius * 0.65, acc.color, 2
                )
            # Bridge
            self._draw_horizontal_band(
                raster,
                cy - face_radius * 0.20,
                cy - face_radius * 0.16,
                cx,
                face_radius * 0.20,
                acc.color,
            )
        elif acc.kind is AccessoryKind.SUNGLASSES:
            for sx in (-1, 1):
                ex = cx + sx * face_radius * 0.40
                ey = cy - face_radius * 0.18
                self._fill_ellipse(raster, ex, ey, eye_radius * 0.85, eye_radius * 0.55, acc.color)
        elif acc.kind is AccessoryKind.MUSTACHE:
            self._draw_mustache(raster, cx, cy + face_radius * 0.30, face_radius * 0.5, acc.color)
        elif acc.kind is AccessoryKind.BOW_TIE:
            self._draw_bow_tie(
                raster, cx, self._height - face_radius * 0.20, face_radius * 0.4, acc.color
            )
        elif acc.kind is AccessoryKind.HAT:
            # A simple band + brim
            self._draw_horizontal_band(
                raster,
                cy - face_radius,
                cy - face_radius * 0.7,
                cx,
                face_radius,
                acc.color,
            )
            self._draw_horizontal_band(
                raster,
                cy - face_radius * 0.7,
                cy - face_radius * 0.55,
                cx,
                face_radius * 1.1,
                acc.color,
            )
        elif acc.kind is AccessoryKind.CROWN:
            self._draw_crown(raster, cx, cy - face_radius * 0.85, face_radius * 0.9, acc.color)
        elif acc.kind is AccessoryKind.BANDAGE:
            self._draw_horizontal_band(
                raster,
                cy - face_radius * 0.6,
                cy - face_radius * 0.30,
                cx,
                face_radius * 0.9,
                (200, 200, 200),
            )
        elif acc.kind is AccessoryKind.HEART_EYES:
            for sx in (-1, 1):
                ex = cx + sx * face_radius * 0.40
                ey = cy - face_radius * 0.18
                self._fill_heart(raster, ex, ey, eye_radius * 0.7, (255, 80, 120))

    # ------------------------------------------------------------------ primitive stubs
    # (Detailed pixel-art primitives live below.)

    def _fill_circle(
        self,
        raster: _Raster,
        cx: float,
        cy: float,
        r: float,
        color: tuple[int, int, int],
    ) -> None:
        if r <= 0:
            return
        self._fill_ellipse(raster, cx, cy, r, r, color)

    def _fill_circle_blend(
        self,
        raster: _Raster,
        cx: float,
        cy: float,
        r: float,
        color: tuple[int, int, int],
        alpha: float,
    ) -> None:
        """Fill a circle with alpha-blending onto the background."""
        if r <= 0 or alpha <= 0.0:
            return
        alpha = max(0.0, min(1.0, alpha))
        w, h = raster.width, raster.height
        pr, pg, pb = color
        x0 = max(0, math.floor(cx - r))
        x1 = min(w - 1, math.ceil(cx + r))
        y0 = max(0, math.floor(cy - r))
        y1 = min(h - 1, math.ceil(cy + r))
        r2 = r * r
        for y in range(y0, y1 + 1):
            dy = y - cy
            row = (y * w) * 3
            for x in range(x0, x1 + 1):
                dx = x - cx
                if dx * dx + dy * dy > r2:
                    continue
                idx = row + x * 3
                # Soft falloff at the edge
                edge = 1.0 - (dx * dx + dy * dy) / r2 if r2 > 0.0 else 0.0
                local_alpha = alpha * (0.4 + 0.6 * edge)
                br = raster.pixels[idx]
                bg = raster.pixels[idx + 1]
                bb = raster.pixels[idx + 2]
                raster.pixels[idx] = int(br * (1 - local_alpha) + pr * local_alpha)
                raster.pixels[idx + 1] = int(bg * (1 - local_alpha) + pg * local_alpha)
                raster.pixels[idx + 2] = int(bb * (1 - local_alpha) + pb * local_alpha)

    def _fill_ellipse(
        self,
        raster: _Raster,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        color: tuple[int, int, int],
    ) -> None:
        if rx <= 0 or ry <= 0:
            return
        pr, pg, pb = color
        w, h = raster.width, raster.height
        x0 = max(0, math.floor(cx - rx))
        x1 = min(w - 1, math.ceil(cx + rx))
        y0 = max(0, math.floor(cy - ry))
        y1 = min(h - 1, math.ceil(cy + ry))
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
        inv_rx2 = 1.0 / (rx * rx) if rx > 0.0 else 0.0
        inv_ry2 = 1.0 / (ry * ry) if ry > 0.0 else 0.0
        for y in range(y0, y1 + 1):
            dy = y - cy
            row = (y * w) * 3
            for x in range(x0, x1 + 1):
                dx = x - cx
                if dx * dx * inv_rx2 + dy * dy * inv_ry2 <= 1.0:
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
        if rx <= 0 or ry <= 0 or width <= 0:
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
        if rx == ry:
            self._draw_circle(raster, cx, cy, rx, color)
            return
        pr, pg, pb = color
        w, h = raster.width, raster.height

        def put(px: int, py: int) -> None:
            if 0 <= px < w and 0 <= py < h:
                idx = (py * w + px) * 3
                raster.pixels[idx] = pr
                raster.pixels[idx + 1] = pg
                raster.pixels[idx + 2] = pb

        rx2 = rx * rx
        ry2 = ry * ry
        x = 0.0
        y = ry
        px = 0.0
        py = 2 * rx2 * y
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
        self,
        raster: _Raster,
        cx: float,
        cy: float,
        r: float,
        color: tuple[int, int, int],
    ) -> None:
        if r <= 0:
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
        pr, pg, pb = color
        w, h = raster.width, raster.height
        x0 = max(0, math.floor(cx - rx))
        x1 = min(w - 1, math.ceil(cx + rx))
        y0 = max(0, math.ceil(y_top))
        y1 = min(h - 1, math.floor(y_bottom))
        rx2 = rx * rx
        band_cy = (y_top + y_bottom) / 2.0
        for y in range(y0, y1 + 1):
            dy = y - band_cy
            row = (y * w) * 3
            for x in range(x0, x1 + 1):
                dx = x - cx
                if dx * dx + dy * dy > rx2:
                    continue
                idx = row + x * 3
                raster.pixels[idx] = pr
                raster.pixels[idx + 1] = pg
                raster.pixels[idx + 2] = pb

    def _draw_arc(
        self,
        raster: _Raster,
        cx: float,
        cy: float,
        width: float,
        curve: float,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        """Draw a smile/frown arc as a thin filled strip.

        ``curve`` is positive for a frown, negative for a smile. The
        arc is sampled at many x positions and a small vertical strip
        of ``thickness`` pixels is drawn.
        """
        if abs(curve) < 0.5:
            # Treat as a horizontal line
            y = int(cy)
            for x in range(int(cx - width / 2), int(cx + width / 2) + 1):
                for t in range(thickness):
                    py = y - thickness // 2 + t
                    if 0 <= x < raster.width and 0 <= py < raster.height:
                        idx = (py * raster.width + x) * 3
                        raster.pixels[idx] = color[0]
                        raster.pixels[idx + 1] = color[1]
                        raster.pixels[idx + 2] = color[2]
            return
        # Parametric arc: y = cy + curve * (1 - ((x-cx) / (width/2)) ** 2)
        steps: int = max(8, round(width))
        half: float = width / 2.0
        if half <= 0.0:
            return
        cx_f: float = float(cx)
        cy_f: float = float(cy)
        curve_f: float = float(curve)
        for i in range(steps + 1):
            x_f: float = cx_f - half + (i / steps) * width
            t_f: float = (x_f - cx_f) / half
            y_f: float = cy_f - curve_f * (1 - t_f * t_f)
            for k in range(-thickness // 2, thickness // 2 + 1):
                px, py = round(x_f), round(y_f + k)
                if 0 <= px < raster.width and 0 <= py < raster.height:
                    idx = (py * raster.width + px) * 3
                    raster.pixels[idx] = color[0]
                    raster.pixels[idx + 1] = color[1]
                    raster.pixels[idx + 2] = color[2]

    def _draw_thick_line(
        self,
        raster: _Raster,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        thickness: int,
        color: tuple[int, int, int],
    ) -> None:
        # Bresenham with a brush
        steps = max(2, int(max(abs(x1 - x0), abs(y1 - y0)) * 2))
        for i in range(steps + 1):
            t = i / steps
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            for dx in range(-thickness // 2, thickness // 2 + 1):
                for dy in range(-thickness // 2, thickness // 2 + 1):
                    px, py = round(x + dx), round(y + dy)
                    if 0 <= px < raster.width and 0 <= py < raster.height:
                        idx = (py * raster.width + px) * 3
                        raster.pixels[idx] = color[0]
                        raster.pixels[idx + 1] = color[1]
                        raster.pixels[idx + 2] = color[2]

    def _fill_diamond(
        self,
        raster: _Raster,
        cx: float,
        cy: float,
        size: float,
        color: tuple[int, int, int],
    ) -> None:
        self._fill_ellipse(raster, cx, cy, size * 0.45, size, color)

    def _fill_heart(
        self,
        raster: _Raster,
        cx: float,
        cy: float,
        size: float,
        color: tuple[int, int, int],
    ) -> None:
        # Two small circles + a downward triangle
        r = size * 0.45
        self._fill_circle(raster, cx - r * 0.6, cy - r * 0.3, r, color)
        self._fill_circle(raster, cx + r * 0.6, cy - r * 0.3, r, color)
        # Triangle (approximated with a rotated square)
        for y in range(int(cy - r * 0.3), int(cy + size * 0.7) + 1):
            t = (y - (cy - r * 0.3)) / (size + 0.2 * r)
            t = max(0.0, min(1.0, t))
            half = size * 0.6 * (1.0 - t)
            for x in range(int(cx - half), int(cx + half) + 1):
                if 0 <= x < raster.width and 0 <= y < raster.height:
                    idx = (y * raster.width + x) * 3
                    raster.pixels[idx] = color[0]
                    raster.pixels[idx + 1] = color[1]
                    raster.pixels[idx + 2] = color[2]

    def _fill_teardrop(
        self,
        raster: _Raster,
        cx: float,
        cy: float,
        size: float,
        color: tuple[int, int, int],
    ) -> None:
        # Circle + triangle pointing down
        r = size * 0.4
        self._fill_circle(raster, cx, cy - size * 0.3, r, color)
        for y in range(int(cy - size * 0.3), int(cy + size * 0.6) + 1):
            t = (y - (cy - size * 0.3)) / (size * 0.9)
            t = max(0.0, min(1.0, t))
            half = size * 0.4 * (1.0 - t)
            for x in range(int(cx - half), int(cx + half) + 1):
                if 0 <= x < raster.width and 0 <= y < raster.height:
                    idx = (y * raster.width + x) * 3
                    raster.pixels[idx] = color[0]
                    raster.pixels[idx + 1] = color[1]
                    raster.pixels[idx + 2] = color[2]

    def _draw_mustache(
        self,
        raster: _Raster,
        cx: float,
        cy: float,
        length: float,
        color: tuple[int, int, int],
    ) -> None:
        # Two curves meeting in the middle
        for sx in (-1, 1):
            self._draw_arc(
                raster,
                cx + sx * length * 0.30,
                cy - sx * length * 0.05,
                length * 0.6,
                -length * 0.10 * sx,
                color,
                3,
            )

    def _draw_bow_tie(
        self,
        raster: _Raster,
        cx: float,
        cy: float,
        size: float,
        color: tuple[int, int, int],
    ) -> None:
        # Two triangles
        for sx in (-1, 1):
            for y in range(int(cy - size / 2), int(cy + size / 2) + 1):
                t = (y - (cy - size / 2)) / size
                half = size * 0.4 * (1.0 - abs(t - 0.5) * 2)
                x_start = cx + sx * size * 0.05
                x_end = x_start + sx * half
                for x in range(int(min(x_start, x_end)), int(max(x_start, x_end)) + 1):
                    if 0 <= x < raster.width and 0 <= y < raster.height:
                        idx = (y * raster.width + x) * 3
                        raster.pixels[idx] = color[0]
                        raster.pixels[idx + 1] = color[1]
                        raster.pixels[idx + 2] = color[2]

    def _draw_crown(
        self,
        raster: _Raster,
        cx: float,
        cy: float,
        length: float,
        color: tuple[int, int, int],
    ) -> None:
        # Three triangles
        points = [
            (cx - length / 2, cy),
            (cx - length / 4, cy - length / 3),
            (cx, cy),
            (cx + length / 4, cy - length / 3),
            (cx + length / 2, cy),
            (cx + length / 2, cy + length / 6),
            (cx - length / 2, cy + length / 6),
        ]
        for i in range(len(points) - 1):
            self._draw_thick_line(
                raster,
                points[i][0],
                points[i][1],
                points[i + 1][0],
                points[i + 1][1],
                2,
                color,
            )


__all__ = ["FaceRenderer"]
