"""Asset loading helpers.

The ``AssetLoader`` resolves paths relative to the project's ``assets/``
directory and gracefully reports missing files.
"""

from __future__ import annotations

from pathlib import Path

from robot.errors import DeskBotError
from robot.logging import get_logger

_log = get_logger("assets")


class AssetNotFoundError(DeskBotError):
    """The requested asset file does not exist."""


class AssetLoader:
    """Locate files under a project-relative assets directory."""

    def __init__(self, root: Path) -> None:
        if not root.exists():
            _log.warning("assets.root_missing", root=str(root))
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def path(self, *parts: str) -> Path:
        """Resolve ``parts`` under the assets root."""
        return self._root.joinpath(*parts)

    def load(self, *parts: str) -> bytes:
        """Read a binary asset. Raises :class:`AssetNotFoundError` if missing."""
        target = self.path(*parts)
        if not target.is_file():
            raise AssetNotFoundError(f"asset not found: {target}")
        return target.read_bytes()

    def load_text(self, *parts: str, encoding: str = "utf-8") -> str:
        target = self.path(*parts)
        if not target.is_file():
            raise AssetNotFoundError(f"asset not found: {target}")
        return target.read_text(encoding=encoding)

    def exists(self, *parts: str) -> bool:
        return self.path(*parts).is_file()


__all__ = ["AssetLoader", "AssetNotFoundError"]
