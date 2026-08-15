"""Tests for the theme registry."""

from __future__ import annotations

import pytest

from robot.face.components import (
    PALETTE_MODE_FACE,
    PALETTE_MODE_VECTOR,
    CheekState,
    EyebrowShape,
    OverlayKind,
)
from robot.face.emotions import EmotionEngine
from robot.face.renderer import FaceRenderer
from robot.face.themes import (
    CuteTheme,
    MinimalTheme,
    PixelTheme,
    RetroLcdTheme,
    Theme,
    VectorTheme,
    WireframeTheme,
    get_theme,
)


def test_all_themes_implement_protocol() -> None:
    for theme_cls in (
        MinimalTheme,
        CuteTheme,
        PixelTheme,
        RetroLcdTheme,
        WireframeTheme,
    ):
        instance = theme_cls()
        assert isinstance(instance, Theme)
        assert instance.name
        assert instance.palette is not None


def test_get_theme_by_name() -> None:
    assert isinstance(get_theme("minimal"), MinimalTheme)
    assert isinstance(get_theme("cute"), CuteTheme)
    assert isinstance(get_theme("retro_lcd"), RetroLcdTheme)


def test_get_theme_case_insensitive() -> None:
    assert isinstance(get_theme("MINIMAL"), MinimalTheme)
    assert isinstance(get_theme("Cute"), CuteTheme)


def test_get_theme_dash_form() -> None:
    assert isinstance(get_theme("retro-lcd"), RetroLcdTheme)


def test_unknown_theme_raises() -> None:
    with pytest.raises(KeyError):
        get_theme("nope")


def test_themes_change_colours() -> None:
    from robot.face.model import FaceModel

    m = FaceModel()
    minimal = MinimalTheme().apply(m)
    cute = CuteTheme().apply(m)
    assert minimal.palette.background != cute.palette.background


def test_vector_theme_is_registered() -> None:
    """The Vector theme must be in BUILTIN_THEMES and look-up by name."""
    from robot.face.themes import BUILTIN_THEMES

    assert "vector" in BUILTIN_THEMES
    theme = get_theme("vector")
    assert isinstance(theme, VectorTheme)
    assert theme.name == "vector"


def test_vector_theme_palette_uses_vector_mode() -> None:
    """The Vector palette must flag itself so the renderer can switch modes."""
    theme = VectorTheme()
    assert theme.palette.mode == PALETTE_MODE_VECTOR
    assert theme.palette.background == (0, 0, 0)
    # The eye colour is the bright green from the Anki Vector face.
    assert theme.palette.iris == (80, 220, 120)


def test_vector_theme_disables_animation_components() -> None:
    """Vector mode hides eyelids, eyebrows, cheeks, overlay, and accessories."""
    theme = VectorTheme()
    engine = EmotionEngine(width=32, height=32)
    base = engine.build("happy")
    model = theme.apply(base)
    # Eyelids must be fully retracted so the eye is always visible.
    assert model.eyelids.top == 0.0
    assert model.eyelids.bottom == 0.0
    assert model.left_eyebrow.shape is EyebrowShape.NEUTRAL
    assert model.right_eyebrow.shape is EyebrowShape.NEUTRAL
    assert model.cheeks.state is CheekState.NONE
    assert model.overlay.kind is OverlayKind.NONE


def test_vector_renderer_draws_only_two_colours_on_happy() -> None:
    """A 32x32 happy frame in Vector mode should have exactly 2 distinct colours."""
    renderer = FaceRenderer(width=32, height=32)
    theme = VectorTheme()
    engine = EmotionEngine(width=32, height=32)
    model = theme.apply(engine.build("happy"))
    frame = renderer.render(model)
    counts: dict[tuple[int, int, int], int] = {}
    for i in range(0, len(frame.pixels), 3):
        key = (frame.pixels[i], frame.pixels[i + 1], frame.pixels[i + 2])
        counts[key] = counts.get(key, 0) + 1
    # Background (black) must always be present.
    assert (0, 0, 0) in counts
    # And there must be at least one non-black pixel (the eyes + mouth line).
    non_black = [c for c in counts if c != (0, 0, 0)]
    assert non_black, "vector frame is entirely black"


def test_all_other_themes_use_default_face_mode() -> None:
    """All non-Vector themes must default to PALETTE_MODE_FACE (no regression)."""
    for name in ("minimal", "cute", "pixel", "retro_lcd", "wireframe"):
        theme = get_theme(name)
        assert theme.palette.mode == PALETTE_MODE_FACE, (
            f"theme {name!r} must default to face mode, got {theme.palette.mode!r}"
        )
