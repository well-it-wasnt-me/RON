"""Tests for the :class:`EyeRenderer` - pixel-level drawing layers."""

from __future__ import annotations

import pytest

from robot.eye_engine.render_state import EyeRenderState
from robot.eye_engine.renderer import EyeRenderer, RendererConfig


def test_renderer_creates_correct_size_frame() -> None:
    r = EyeRenderer(width=64, height=64)
    state = EyeRenderState(cx=32, cy=32, eye_radius=24)
    frame = r.render(state)
    assert frame.width == 64
    assert frame.height == 64
    assert len(frame.pixels) == 64 * 64 * 3


def test_renderer_validates_dimensions() -> None:
    with pytest.raises(ValueError):
        EyeRenderer(width=0, height=64)
    with pytest.raises(ValueError):
        EyeRenderer(width=64, height=0)


def test_background_is_painted() -> None:
    r = EyeRenderer(width=64, height=64, config=RendererConfig(background_color=(50, 60, 70)))
    state = EyeRenderState(cx=32, cy=32, eye_radius=24)
    frame = r.render(state)
    # The top-left pixel should be exactly the background colour.
    assert tuple(frame.pixels[:3]) == (50, 60, 70)


def test_sclera_is_painted_inside_the_eye() -> None:
    r = EyeRenderer(width=64, height=64)
    state = EyeRenderState(
        cx=32,
        cy=32,
        eye_radius=20,
        sclera_color=(200, 210, 220),
    )
    frame = r.render(state)
    # A pixel *between* the iris and the sclera edge should be the sclera.
    # Iris radius = 0.55 * 20 = 11, sclera radius = 20, so x=15 is just
    # outside the iris and well inside the sclera.
    idx = (32 * 64 + (32 + 15)) * 3
    assert tuple(frame.pixels[idx : idx + 3]) == (200, 210, 220)


def test_iris_is_visible_on_top_of_sclera() -> None:
    r = EyeRenderer(width=64, height=64)
    state = EyeRenderState(
        cx=32,
        cy=32,
        eye_radius=20,
        iris_radius_ratio=0.55,
        pupil_radius_ratio=0.05,  # tiny pupil so the iris is visible
        pupil_color=(0, 0, 0),
        iris_color=(40, 110, 200),
    )
    frame = r.render(state)
    # The iris radius is 0.55 * 20 = 11. A pixel 5px to the right of centre
    # is well inside the iris and outside the tiny pupil.
    idx = (32 * 64 + (32 + 5)) * 3
    assert tuple(frame.pixels[idx : idx + 3]) == (40, 110, 200)


def test_pupil_is_dark() -> None:
    r = EyeRenderer(width=64, height=64)
    state = EyeRenderState(
        cx=32,
        cy=32,
        eye_radius=20,
        iris_radius_ratio=0.55,
        pupil_radius_ratio=0.55,
        pupil_color=(10, 10, 10),
    )
    frame = r.render(state)
    idx = (32 * 64 + 32) * 3  # exact centre
    assert tuple(frame.pixels[idx : idx + 3]) == (10, 10, 10)


def test_highlight_appears_on_iris() -> None:
    r = EyeRenderer(width=64, height=64)
    state = EyeRenderState(
        cx=32,
        cy=32,
        eye_radius=20,
        iris_radius_ratio=0.55,
        highlight_x=0.3,
        highlight_y=0.3,
        highlight_color=(255, 0, 0),
    )
    frame = r.render(state)
    # The highlight is rendered on top of the iris; look for the red pixel
    # near the centre.
    found = False
    for y in range(28, 37):
        for x in range(28, 37):
            idx = (y * 64 + x) * 3
            if tuple(frame.pixels[idx : idx + 3]) == (255, 0, 0):
                found = True
                break
        if found:
            break
    assert found, "no red highlight found near the iris"


def test_eyelid_top_covers_eye() -> None:
    r = EyeRenderer(width=64, height=64)
    state = EyeRenderState(
        cx=32,
        cy=32,
        eye_radius=20,
        lid_top=0.6,
        lid_color=(0, 0, 0),
    )
    frame = r.render(state)
    # A pixel high above the eye centre should now be the lid colour.
    idx = (15 * 64 + 32) * 3
    assert tuple(frame.pixels[idx : idx + 3]) == (0, 0, 0)


def test_eyelid_bottom_covers_eye() -> None:
    r = EyeRenderer(width=64, height=64)
    state = EyeRenderState(
        cx=32,
        cy=32,
        eye_radius=20,
        lid_bottom=0.6,
        lid_color=(0, 0, 0),
    )
    frame = r.render(state)
    idx = (49 * 64 + 32) * 3
    assert tuple(frame.pixels[idx : idx + 3]) == (0, 0, 0)


def test_closed_eye_hides_iris() -> None:
    r = EyeRenderer(width=64, height=64)
    state = EyeRenderState(
        cx=32,
        cy=32,
        eye_radius=20,
        openness=0.05,
        iris_color=(40, 110, 200),
    )
    frame = r.render(state)
    # Iris is hidden when the eye is almost closed.
    idx = (32 * 64 + 32) * 3
    assert tuple(frame.pixels[idx : idx + 3]) != (40, 110, 200)


def test_gaze_shifts_iris_position() -> None:
    """Looking right should place the iris slightly to the right of centre."""
    r = EyeRenderer(width=64, height=64)
    centre_state = EyeRenderState(
        cx=32,
        cy=32,
        eye_radius=20,
        iris_radius_ratio=0.55,
        iris_color=(40, 110, 200),
        gaze_x=0.0,
        gaze_y=0.0,
    )
    right_state = centre_state.__class__(
        cx=32,
        cy=32,
        eye_radius=20,
        iris_radius_ratio=0.55,
        iris_color=(40, 110, 200),
        gaze_x=0.5,
        gaze_y=0.0,
    )
    centre_frame = r.render(centre_state)
    right_frame = r.render(right_state)
    # The pixel just to the LEFT of centre should be the iris colour in the
    # right-looking frame, but NOT in the centre-looking frame.
    idx_left_of_centre = (32 * 64 + 30) * 3
    assert tuple(centre_frame.pixels[idx_left_of_centre : idx_left_of_centre + 3]) != (40, 110, 200)
    assert tuple(right_frame.pixels[idx_left_of_centre : idx_left_of_centre + 3]) == (40, 110, 200)


def test_pupil_dilation_changes_pupil_size() -> None:
    r = EyeRenderer(width=64, height=64)
    state_small = EyeRenderState(
        cx=32,
        cy=32,
        eye_radius=20,
        iris_radius_ratio=0.55,
        pupil_radius_ratio=0.55,
        pupil_dilation=0.0,
        pupil_color=(0, 0, 0),
    )
    state_big = EyeRenderState(
        cx=32,
        cy=32,
        eye_radius=20,
        iris_radius_ratio=0.55,
        pupil_radius_ratio=0.55,
        pupil_dilation=1.0,
        pupil_color=(0, 0, 0),
    )
    small = r.render(state_small)
    big = r.render(state_big)

    # Count dark pixels in the iris region (within 11px of the centre).
    def dark_count(frame: object) -> int:
        pixels = frame.pixels  # type: ignore[attr-defined]
        count = 0
        for y in range(32 - 11, 32 + 12):
            for x in range(32 - 11, 32 + 12):
                idx = (y * 64 + x) * 3
                if tuple(pixels[idx : idx + 3]) == (0, 0, 0):
                    count += 1
        return count

    assert dark_count(big) > dark_count(small)


def test_two_displays_produce_independent_frames() -> None:
    """The renderer is the same for both eyes; each eye can have its own state."""
    r = EyeRenderer(width=64, height=64)
    left_state = EyeRenderState(cx=32, cy=32, eye_radius=20, gaze_x=-0.5)
    right_state = EyeRenderState(cx=32, cy=32, eye_radius=20, gaze_x=0.5)
    left_frame = r.render(left_state)
    right_frame = r.render(right_state)
    assert left_frame.pixels != right_frame.pixels
