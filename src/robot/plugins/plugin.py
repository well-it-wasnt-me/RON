"""Plugin protocol and data types.

Every plugin must implement the :class:`Plugin` protocol. The lifecycle
is: ``load`` -> ``start`` -> ``stop`` -> ``unload``. Only ``load`` and
``start`` are required; defaults for the other hooks are no-ops.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class PluginState(str, Enum):
    """Lifecycle states a plugin can be in."""

    UNLOADED = "unloaded"
    LOADED = "loaded"
    STARTED = "started"
    STOPPED = "stopped"
    ERRORED = "errored"


@dataclass(slots=True, frozen=True)
class PluginInfo:
    """Metadata about a plugin."""

    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    depends: tuple[str, ...] = ()

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PluginInfo):
            return self.name == other.name
        return NotImplemented


@runtime_checkable
class Plugin(Protocol):
    """The plugin interface.

    Implementations must provide ``info`` and ``load``. The other hooks
    have safe defaults.
    """

    @property
    def info(self) -> PluginInfo:
        """Return plugin metadata."""
        ...

    async def load(self) -> None:
        """Called once when the plugin is first loaded.

        Use this to register event handlers, set up resources, etc.
        """
        ...

    async def start(self) -> None:
        """Called when the application starts running.

        Use this to begin background tasks.
        """
        ...

    async def stop(self) -> None:
        """Called when the application is shutting down.

        Use this to cancel background tasks and release resources.
        """
        ...

    async def unload(self) -> None:
        """Called when the plugin is being removed.

        Use this to clean up any persistent resources.
        """
        ...


__all__ = ["Plugin", "PluginInfo", "PluginState"]
