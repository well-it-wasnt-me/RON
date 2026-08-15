"""Dependency-injection container.

A simple, explicit, constructor-friendly container. No metaclass magic, no
implicit globals - the container is just a typed registry that components
read from at boot. Tests can pass their own container.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from robot.errors import DependencyResolutionError

T = TypeVar("T")

# A registered factory is stored together with a flag telling us whether the
# result should be cached. We keep both pieces in a small tuple to make the
# intent explicit.
FactoryEntry = tuple[Callable[[], Any], bool]


@dataclass(slots=True)
class Container:
    """A typed key-value container for application services."""

    _factories: dict[type, FactoryEntry] = field(default_factory=dict)
    _singletons: dict[type, Any] = field(default_factory=dict)
    _instances: dict[type, Any] = field(default_factory=dict)

    def register(
        self,
        interface: type[T],
        factory: Callable[[], T],
        *,
        singleton: bool = True,
    ) -> None:
        """Register a factory for ``interface``.

        When ``singleton`` is true (the default) the factory is called once and
        its result is cached.
        """
        self._factories[interface] = (factory, singleton)
        if not singleton:
            self._singletons.pop(interface, None)
            self._instances.pop(interface, None)

    def register_instance(self, interface: type[T], instance: T) -> None:
        """Register a pre-built instance."""
        self._instances[interface] = instance

    def resolve(self, interface: type[T]) -> T:
        """Resolve ``interface``. Cached on first call when registered as singleton."""
        if interface in self._instances:
            return cast("T", self._instances[interface])
        if interface in self._singletons:
            return cast("T", self._singletons[interface])
        if interface not in self._factories:
            raise DependencyResolutionError(f"no factory registered for {interface!r}")
        entry = self._factories[interface]
        factory: Callable[[], T] = entry[0]
        singleton: bool = entry[1]
        instance = factory()
        if singleton:
            self._singletons[interface] = instance
        return instance

    def has(self, interface: type) -> bool:
        return interface in self._factories or interface in self._instances

    def reset(self) -> None:
        self._singletons.clear()
        self._instances.clear()

    def __contains__(self, interface: object) -> bool:
        return isinstance(interface, type) and self.has(interface)


__all__ = ["Container"]
