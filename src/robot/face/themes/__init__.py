"""Built-in face themes.

The :class:`Theme` protocol is defined in :mod:`robot.face.themes.base`.
Each concrete theme lives in its own module so users can pick and choose
which to ship.
"""

from robot.face.themes.base import Theme
from robot.face.themes.cute import CuteTheme
from robot.face.themes.minimal import MinimalTheme
from robot.face.themes.pixel import PixelTheme
from robot.face.themes.retro_lcd import RetroLcdTheme
from robot.face.themes.vector import VectorTheme
from robot.face.themes.wireframe import WireframeTheme

# A registry so users can ask for a theme by name.
BUILTIN_THEMES: dict[str, type[Theme]] = {
    "minimal": MinimalTheme,
    "cute": CuteTheme,
    "pixel": PixelTheme,
    "retro_lcd": RetroLcdTheme,
    "vector": VectorTheme,
    "wireframe": WireframeTheme,
}


def get_theme(name: str) -> Theme:
    """Look up a built-in theme by name (case-insensitive)."""
    key = name.lower().replace("-", "_")
    if key not in BUILTIN_THEMES:
        raise KeyError(f"unknown theme {name!r}; available: {sorted(BUILTIN_THEMES.keys())}")
    return BUILTIN_THEMES[key]()


__all__ = [
    "BUILTIN_THEMES",
    "CuteTheme",
    "MinimalTheme",
    "PixelTheme",
    "RetroLcdTheme",
    "Theme",
    "VectorTheme",
    "WireframeTheme",
    "get_theme",
]
