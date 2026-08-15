"""Tests for the :class:`FaceRenderer`."""

from __future__ import annotations

import pytest

from robot.face.model import FaceModel
from robot.face.renderer import FaceRenderer


def test_renderer_creates_correct_size_frame() -> None:
    r = FaceRenderer(width=64, height=64)
    frame = r.render(FaceModel(width=64, height=64))
    assert frame.width == 64
    assert frame.height == 64
    assert len(frame.pixels) == 64 * 64 * 3


def test_renderer_validates_dimensions() -> None:
    with pytest.raises(ValueError):
        FaceRenderer(width=0, height=64)


def test_renderer_validates_model_dimensions() -> None:
    r = FaceRenderer(width=64, height=64)
    with pytest.raises(ValueError):
        r.render(FaceModel(width=32, height=32))


def test_background_is_painted() -> None:
    r = FaceRenderer(width=64, height=64)
    frame = r.render(FaceModel(width=64, height=64))
    # The top-left corner is the background
    assert tuple(frame.pixels[:3]) == (10, 10, 20)


def test_sclera_is_painted() -> None:
    r = FaceRenderer(width=64, height=64)
    frame = r.render(FaceModel(width=64, height=64))
    # Some pixel near the eyes should be the sclera colour
    # The eye is at face_radius*0.40 from centre, so (32 - 24) = 8 px left of centre
    found = False
    for dy in range(-2, 3):
        for dx in range(-3, 4):
            idx = ((32 + dy) * 64 + (32 + 2 * 24 + dx)) * 3
            if tuple(frame.pixels[idx : idx + 3]) == (245, 245, 235):
                found = True
                break
        if found:
            break
    assert found, "no sclera pixel found near the eye"


def test_happy_uses_cute_palette_via_theme() -> None:
    """Themes swap colours; the renderer's output is what changes."""
    from robot.face.themes.cute import CuteTheme
    from robot.face.themes.minimal import MinimalTheme

    r = FaceRenderer(width=64, height=64)
    m = FaceModel(width=64, height=64)
    minimal = r.render(MinimalTheme().apply(m))
    cute = r.render(CuteTheme().apply(m))
    # Backgrounds differ
    assert minimal.pixels[:3] != cute.pixels[:3]
